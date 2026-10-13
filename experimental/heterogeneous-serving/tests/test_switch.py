"""The in-process production switch (hetero/switch.py): sequence, gates, segments, failure, drain timings.

Everything runs against the fakes in scripts/switch_dry_run.py: no Ray cluster, TPU, Envoy or Kubernetes.
"""

import csv
import gzip
import inspect
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from hetero import deployment, switch, transition
from scripts import switch_dry_run as dry

SCALE = 0.05  # drain 0.1 s/instance, deploy 0.25 s, gateway_up 0.05 s


def run(mode, *, scale=SCALE, **kwargs):
    tmp = tempfile.TemporaryDirectory()
    fabric, client, kw = dry.run_dry(mode, pathlib.Path(tmp.name), scale=scale, **kwargs)
    report = switch.run_switch(fabric, client, **kw)
    return tmp, fabric, client, report


def read_gz(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


class FullModeTests(unittest.TestCase):
    def test_order_and_no_dispatch_before_gate(self):  # (a)
        tmp, fabric, client, report = run("full")
        self.addCleanup(tmp.cleanup)
        self.assertEqual(fabric.order(), ["drain", "gateway_down", "deploy", "gateway_up", "start_capture"])
        st = report["stamps"]
        for earlier, later in (("controller_wake", "drain_begin"), ("drain_end", "gateway_down_begin"),
                               ("gateway_down_end", "deploy_begin"), ("deploy_end", "gateway_up_begin"),
                               ("gateway_up_end", "gate_open:target"), ("gate_open:target", "capture_arm_begin"),
                               ("capture_arm_end", "first_dispatch:target")):
            self.assertLessEqual(st[earlier], st[later], f"{earlier} must precede {later}")
        target = [d for d in client.dispatches if d["label"] == "target"]
        self.assertTrue(target)
        self.assertGreaterEqual(min(d["t"] for d in target), st["gate_open:target"])
        self.assertTrue(all(d["instance"] is not None for d in target), "a request reached an unbound fabric")
        self.assertTrue(report["completed"], report.get("error"))
        self.assertEqual(report["gates"]["target"]["via"], "event")
        self.assertTrue(client.opened and client.closed)
        self.assertTrue(report["restored"]["all_restored"])
        self.assertTrue(report["accounting"]["phase2"]["complete"] and report["accounting"]["phase1"]["complete"])
        # Artifacts of the contract exist next to the compact report.
        out = pathlib.Path(report["artifact_dir"])
        for name in ("switch_report.json", "phase1_records.json.gz", "phase2_records.json.gz",
                     "gateway_records.json.gz", "traces.json.gz", "deploy_report.json.gz",
                     "drain_report.json.gz", "accounting.json.gz", "phase2_r1_iterations.csv.gz"):
            self.assertTrue((out / name).exists(), name)
        self.assertLess((out / "switch_report.json").stat().st_size, 200_000)

    def test_segments_contiguous_and_sum_to_dispatch_gap(self):  # (b)
        for mode in ("full", "partial"):
            with self.subTest(mode=mode):
                tmp, fabric, client, report = run(mode)
                self.addCleanup(tmp.cleanup)
                segments = report["segments"]
                self.assertEqual(segments[0]["start_rel_s"], 0.0)
                for a, b in zip(segments, segments[1:]):
                    self.assertAlmostEqual(a["end_rel_s"], b["start_rel_s"], places=6)
                records = read_gz(pathlib.Path(report["artifact_dir"]) / "phase2_records.json.gz")
                kind = "target" if mode == "full" else "kept"
                first = min(r["client_started_at_s"] for r in records if r["stream"] == kind)
                gap = first - report["T0"]
                self.assertAlmostEqual(sum(s["s"] for s in segments), gap, delta=0.05)
                self.assertAlmostEqual(report["endpoints"]["gap_dispatch_s"], gap, delta=0.001)
                expected = (["controller_wake", "drain_release", "gateway_down", "deploy_controller_pre",
                             "engine_process_start", "engine_init", "endpoint_register", "gateway_up",
                             "gate_observe", "capture_arm", "dispatch"] if mode == "full" else
                            ["controller_wake", "drain_release", "adopt", "retarget", "capture_arm",
                             "gate_observe", "dispatch"])
                self.assertEqual([s["segment"] for s in segments], expected)
                self.assertEqual({s["segment"]: s["class"] for s in segments}["capture_arm"], "harness")

    def test_offline_recompute_matches_report(self):
        tmp, fabric, client, report = run("full")
        self.addCleanup(tmp.cleanup)
        out = pathlib.Path(report["artifact_dir"])
        with gzip.open(out / "phase2_r1_iterations.csv.gz", "rt") as handle:
            rows = [(r["instance_id"], float(r["completed_at_s"]), int(float(r["emitted_output_tokens"] or 0)))
                    for r in csv.DictReader(handle)]
        again = switch.build_timeline(
            mode="full", T0=report["T0"], stamps=report["stamps"], records2=read_gz(out / "phase2_records.json.gz"),
            deploy_report=read_gz(out / "deploy_report.json.gz"), drain_report=read_gz(out / "drain_report.json.gz"),
            iter_rows=rows, plan=None, static_target_s=report["static_target_s"])
        self.assertEqual([s["s"] for s in again["segments"]], [s["s"] for s in report["segments"]])
        for key in ("gap_dispatch_s", "gap_first_token_s", "C_production_s", "ramp_makespan_s"):
            self.assertEqual(again["endpoints"][key], report["endpoints"][key], key)


class PartialModeTests(unittest.TestCase):
    def test_kept_before_join_joiner_after_join_and_caps(self):  # (c)
        manifests = {"W0": dry.synthetic_manifest("W0", 8, prompt_len=32, max_tokens=16),
                     "W0p2": dry.synthetic_manifest("W0p2", 1763, prompt_len=32, max_tokens=16),
                     "W6": dry.synthetic_manifest("W6", 40, prompt_len=17000, max_tokens=8)}
        tmp, fabric, client, report = run("partial", manifests=manifests)
        self.addCleanup(tmp.cleanup)
        st = report["stamps"]
        self.assertEqual(fabric.order(), ["drain", "adopt", "retarget", "start_capture", "deploy", "join"])
        self.assertEqual({k: v["cap"] for k, v in report["streams"].items()}, {"kept": 216, "joiner": 40})
        self.assertEqual((client.max_in_flight["kept"], client.max_in_flight["joiner"]), (216, 40))
        self.assertLessEqual(client.max_total, 256)
        kept = [d["t"] for d in client.dispatches if d["label"] == "kept"]
        joiner = [d["t"] for d in client.dispatches if d["label"] == "joiner"]
        self.assertEqual((len(kept), len(joiner)), (1763, 40))
        self.assertGreaterEqual(min(kept), st["retarget_end"])
        self.assertLess(min(kept), st["join_begin"])
        self.assertGreaterEqual(min(joiner), st["join_end"] - 1e-3)
        self.assertTrue(all(d["instance"] == "large_b" for d in client.dispatches if d["label"] == "joiner"))
        self.assertTrue(all(not d.get("error") for d in client.dispatches))
        self.assertTrue(report["completed"], report.get("error"))
        ep = report["endpoints"]
        self.assertAlmostEqual(ep["join_delay_s"], ep["joiner_timeline"]["first_joiner_dispatch_s"], places=6)
        self.assertIn("kept_pair_throughput", ep)
        self.assertEqual(sum(b["tokens"] for b in ep["joiner_ramp_bins"]["bins"]), 40 * 8)
        self.assertEqual(set(ep["class_finish_s"]), {"kept", "joiner"})
        self.assertIn("C_eff_s", ep)

    def test_context_classification_uses_kept_max_model_len(self):
        manifests = {"W0": dry.synthetic_manifest("W0", 4, prompt_len=32, max_tokens=16),
                     "W0p2": dry.synthetic_manifest("W0p2", 10, prompt_len=16000, max_tokens=384),  # exactly 16384
                     "W6": dry.synthetic_manifest("W6", 3, prompt_len=16000, max_tokens=385)}  # 16385
        tmp, fabric, client, report = run("partial", manifests=manifests)
        self.addCleanup(tmp.cleanup)
        self.assertEqual({k: v["rows"] for k, v in report["streams"].items()}, {"kept": 10, "joiner": 3})


class FailureTests(unittest.TestCase):
    def _fail(self, mode, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fabric, client, kw = dry.run_dry(mode, pathlib.Path(tmp.name), scale=SCALE, fail_at="deploy", **kwargs)
        with self.assertRaisesRegex(RuntimeError, "injected failure in deploy"):
            switch.run_switch(fabric, client, **kw)
        report = json.loads((pathlib.Path(tmp.name) / kw["run_id"] / "switch_report.json").read_text())
        return fabric, client, report

    def test_full_deploy_failure_writes_report_cancels_streams_and_never_retries(self):  # (d)
        fabric, client, report = self._fail("full")
        self.assertIn("injected failure in deploy", report["error"])
        self.assertEqual(report["failed_step"], "deploy")
        self.assertFalse(report["completed"])
        self.assertEqual(report["streams_cancelled"], ["target"])
        self.assertTrue(client.closed)
        self.assertEqual([c["name"] for c in fabric.calls if c["name"] == "deploy"], ["deploy"])
        self.assertNotIn("gateway_up", fabric.order())
        self.assertIn("deploy_error", report["stamps"])
        self.assertIn("controller_error", report["stamps"])
        self.assertFalse([d for d in client.dispatches if d["label"] == "target"])

    def test_partial_deploy_failure_cancels_inflight_kept_stream(self):
        manifests = dry.default_manifests("partial")
        fabric, client, report = self._fail("partial", manifests=manifests, delays={"deploy": 5.0},
                                            service_by_label={"kept": 30.0})
        self.assertEqual(report["failed_step"], "deploy")
        self.assertIn("kept", report["streams_cancelled"])
        self.assertIn("joiner", report["streams_cancelled"])
        self.assertTrue(fabric.capturing is False or report.get("capture_disarmed_after_failure") is not None)
        self.assertNotIn("join", fabric.order())

    def test_phase1_error_never_wakes_the_controller(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fabric, client, kw = dry.run_dry("full", pathlib.Path(tmp.name), scale=SCALE)
        fabric.route = mock.Mock(side_effect=RuntimeError("boom"))  # every phase-1 request fails
        with self.assertRaisesRegex(RuntimeError, "phase 1 delivered"):
            switch.run_switch(fabric, client, **kw)
        self.assertEqual(fabric.calls, [])

    def test_preflight_refuses_a_busy_gateway(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fabric, client, kw = dry.run_dry("full", pathlib.Path(tmp.name), scale=SCALE)
        fabric.capturing = True
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            switch.run_switch(fabric, client, **kw)
        self.assertEqual(fabric.calls, [])


class Gate(unittest.TestCase):
    STATUS = {"layout_id": "Mb", "instances": ["small_a", "small_b"], "joining": ["large_b"],
              "maintenance_instances": [], "discovery_ready": True, "serving": True, "capturing": True}

    def test_gate_requires_bound_not_joining_ready_and_serving(self):
        ok = switch.gate_verified
        self.assertTrue(ok(self.STATUS, "Mb", ["small_a", "small_b"]))
        self.assertFalse(ok(self.STATUS, "Mb", ["large_b"]))  # joining, not bound
        self.assertFalse(ok(self.STATUS, "S", ["small_a"]))  # wrong layout
        self.assertFalse(ok({**self.STATUS, "maintenance_instances": ["small_c"]}, "Mb", ["small_a"]))
        self.assertFalse(ok({**self.STATUS, "discovery_ready": False}, "Mb", ["small_a"]))
        self.assertFalse(ok({**self.STATUS, "serving": False}, "Mb", ["small_a"]))
        self.assertFalse(ok(None, "Mb", ["small_a"]))
        self.assertFalse(ok({**self.STATUS, "instances": ["small_a", "large_b"], "joining": ["large_b"]},
                            "Mb", ["large_b"]))

    def test_late_event_falls_back_to_polling_the_gateway(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fabric, client, kw = dry.run_dry("full", pathlib.Path(tmp.name), scale=SCALE)
        original = switch._Switch._controller_full

        def full_without_signal(self):  # the controller "forgets" to signal; routing_status is authoritative
            self.signals["target_routes"].set = lambda: None
            original(self)

        with mock.patch.object(switch._Switch, "_controller_full", full_without_signal):
            report = switch.run_switch(fabric, client, **{**kw, "event_grace_s": 0.2})
        self.assertEqual(report["gates"]["target"]["via"], "poll_fallback")
        self.assertTrue(report["completed"])


class RoutingStatusTests(unittest.TestCase):
    def test_shape_and_values(self):  # (g)
        from types import SimpleNamespace

        from hetero.gateway.service import OmpGateway

        cls = OmpGateway.__ray_metadata__.modified_class
        gateway = cls.__new__(cls)
        core = SimpleNamespace(descriptors=lambda: {"small_b": {}, "small_a": {}})
        gateway.config = SimpleNamespace(layout_id="Mb")
        gateway.state = SimpleNamespace(
            core=core, joining={"large_b"}, maintenance={"instances": {"small_c": {}}},
            discovery_snapshot=lambda: {"ready": True, "pods": ["x" * 100]}, _task=SimpleNamespace(done=lambda: False),
            capturing=True)
        self.assertEqual(gateway.routing_status(), {
            "layout_id": "Mb", "instances": ["small_a", "small_b"], "joining": ["large_b"],
            "maintenance_instances": ["small_c"], "discovery_ready": True, "serving": True, "capturing": True})
        gateway.state.maintenance, gateway.state._task = {}, None
        status = gateway.routing_status()
        self.assertEqual((status["maintenance_instances"], status["serving"]), ([], False))
        self.assertNotIn("core", status)


class FakeRay:
    @staticmethod
    def get(value, timeout=None):
        return value

    @staticmethod
    def kill(actor, no_restart=True):
        actor.killed = True


class Remote:
    """Any attribute is a method whose ``.remote(...)`` returns the canned value."""

    def __init__(self, **results):
        self._results = results
        self.killed = False

    def __getattr__(self, name):
        if name.startswith("_") or name not in self._results:
            raise AttributeError(name)
        value = self._results[name]
        return mock.Mock(remote=lambda *a, **k: value() if callable(value) else value)


class DrainTimingTests(unittest.TestCase):
    ROW = {"instance_id": "small_a", "generation": 3, "placement_group_name": deployment.pg_name("S", "small_a"),
           "engine_actor_name": deployment.engine_actor_name("S", "small_a"),
           "ray_namespace": deployment.REGISTRY_NAMESPACE}
    KEYS = {"layout_id", "drained", "maintenance", "gateway_reconcile", "registry_snapshot",
            "owned_placement_groups"}

    def _drain(self, with_gateway):
        actor = Remote(identity={"generation": 3, "layout_id": "S", "instance_id": "small_a"},
                       drain={"drained": True, "inflight": 0}, await_idle={"idle": True},
                       idle_snapshot={"all_blocks_free": True}, stop_serving={"stopped": True}, shutdown={})
        registry = Remote(snapshot=[self.ROW], set_state=True, withdraw=True)
        gateway = Remote(status={"layout_id": "S"}, begin_maintenance={"id": "m1"}, reconcile={"ok": True},
                         drain_maintenance_instance={"generation": 3, "drain": {"drained": True}}) if with_gateway else None
        with mock.patch.object(deployment, "ray", FakeRay), \
                mock.patch("hetero.registry.find_registry", return_value=registry), \
                mock.patch("hetero.gateway.lifecycle.find_gateway", return_value=gateway), \
                mock.patch.object(deployment, "_find_engine_actor_named", return_value=actor), \
                mock.patch.object(deployment, "release_placement_group", return_value=True), \
                mock.patch.object(deployment, "list_owned_placement_groups", return_value=[]):
            return deployment.drain_deployment("S", ["small_a"], remove=True, timeout_s=5.0), actor

    def test_timings_present_and_return_keys_unchanged(self):  # (f)
        for with_gateway in (True, False):
            with self.subTest(gateway=with_gateway):
                result, actor = self._drain(with_gateway)
                self.assertEqual(set(result), self.KEYS)
                entry = result["drained"][0]
                self.assertTrue({"instance_id", "generation", "actor_found", "stop_serving", "shutdown",
                                 "actor_removed", "placement_group_removed", "withdrawn", "timings"} <= set(entry))
                steps = {"drain_ack", "registry_draining", "stop_serving", "shutdown", "kill", "pg_release",
                         "withdraw"} | ({"maintenance"} if with_gateway else set())
                self.assertEqual(set(entry["timings"]), steps)
                spans = sorted((v for k, v in entry["timings"].items() if k != "maintenance"))
                for begin, end in spans:
                    self.assertLessEqual(begin, end)
                for (_, end), (begin, _) in zip(spans, spans[1:]):
                    self.assertLessEqual(end, begin + 1e-6)  # serial steps do not overlap
                self.assertTrue(actor.killed)

    def test_keep_resources_drain_has_no_removal_timings(self):
        with mock.patch.object(deployment, "ray", FakeRay), \
                mock.patch("hetero.registry.find_registry", return_value=Remote(snapshot=[self.ROW], set_state=True)), \
                mock.patch("hetero.gateway.lifecycle.find_gateway", return_value=None), \
                mock.patch.object(deployment, "_find_engine_actor_named", return_value=Remote(
                    identity={"generation": 3, "layout_id": "S", "instance_id": "small_a"},
                    drain={"drained": True, "inflight": 0}, await_idle={"idle": True},
                    idle_snapshot={"all_blocks_free": True})), \
                mock.patch.object(deployment, "list_owned_placement_groups", return_value=[]):
            result = deployment.drain_deployment("S", ["small_a"], remove=False, timeout_s=5.0)
        self.assertEqual(set(result["drained"][0]["timings"]), {"drain_ack", "registry_draining"})


class TransitionHelperTests(unittest.TestCase):
    """(e) run_transition's extracted helpers keep its checks and step order (the full existing
    transition suites run alongside: tests/test_partial_reconfiguration.py, test_staged_transition.py)."""

    def test_run_transition_signature_and_helper_use_unchanged(self):
        self.assertEqual(list(inspect.signature(transition.run_transition).parameters),
                         ["inventory", "config", "from_layout", "arm1", "workload1", "arm2", "workload2",
                          "run_id", "pilot_policy", "repetition", "arm_timeout_s", "drain_timeout_s"])
        source = inspect.getsource(transition.run_transition)
        order = [source.index(marker) for marker in (
            "check_source_idle(", "resolve_transition(", "prepare_kept(", "run_arm(", "drain_deployment(",
            "adopt_kept(", "retarget_layout", "deploy(inventory", "join_instances", "during_run=join_when_ready")]
        self.assertEqual(order, sorted(order))

    def test_check_source_idle_checks(self):
        status = {"layout_id": "S", "maintenance": {}, "joining": [], "capturing": False,
                  "discovery": {"ready": True}, "instances": ["small_a", "small_b"]}
        layout = {"instances": [{"instance_id": "small_a", "tp_size": 8}, {"instance_id": "small_b", "tp_size": 8}]}
        gateway, registry = Remote(status=status), Remote(get_layout=layout)
        with mock.patch("ray.get", FakeRay.get):
            got = transition.check_source_idle(gateway, registry, "S")
            self.assertEqual(got[2], {"small_a": 8, "small_b": 8})
            for bad in ({"capturing": True}, {"joining": ["x"]}, {"maintenance": {"id": 1}},
                        {"discovery": {"ready": False}}, {"layout_id": "M"}):
                with self.assertRaisesRegex(RuntimeError, "not idle"):
                    transition.check_source_idle(Remote(status={**status, **bad}), registry, "S")
            with self.assertRaisesRegex(RuntimeError, "does not bind"):
                transition.check_source_idle(Remote(status={**status, "instances": ["small_a"]}), registry, "S")
            with self.assertRaisesRegex(RuntimeError, "needs the running gateway"):
                transition.check_source_idle(None, registry, "S")

    def test_policy_after_join_matches_the_inline_expression(self):
        from hetero.experiment import ARMS, BASE_POLICY

        amend = {"decode_length_blind": True}
        self.assertEqual(transition.policy_after_join("Mb0", amend),
                         {**BASE_POLICY, **amend, "migration_enabled": ARMS["Mb0"]["migration_enabled"],
                          "migration_mechanism": ARMS["Mb0"]["migration_mechanism"]})

    def test_adopt_kept_registers_each_kept_engine_under_the_target(self):
        calls = []
        actor = Remote(adopt_identity=lambda: {"generation": 7})
        registry = Remote(get={"generation": 7, "instance_id": "small_a"},
                          register=lambda: calls.append("register") or {"accepted": True, "generation": 7})
        instance = mock.Mock(to_dict=lambda: {"instance_id": "small_a"})
        with mock.patch("ray.get", FakeRay.get), \
                mock.patch("hetero.registry.descriptor_for_layout", lambda row, layout, digest: {**row, "layout_id": layout}):
            transition.adopt_kept(registry, {"small_a": actor}, {"small_a": instance}, "Mb", "digest")
            self.assertEqual(calls, ["register"])
            with self.assertRaisesRegex(RuntimeError, "generations disagree"):
                transition.adopt_kept(Remote(get={"generation": 8}, register={}), {"small_a": actor},
                                      {"small_a": instance}, "Mb", "digest")


class AnalysisTests(unittest.TestCase):
    def test_restored_flags_detect_a_fallback(self):
        def report(restored):
            return {"instances": [{"descriptor": {"instance_id": "small_a"}, "initialization": {
                "instance_id": "small_a", "worker_init_phases": [
                    {"executor_rank": 0, "weight_snapshot": {"restored": True}},
                    {"executor_rank": 1, "weight_snapshot": {"restored": restored}}]}}]}
        self.assertTrue(switch.restored_flags(report(True))["all_restored"])
        flags = switch.restored_flags(report(False))
        self.assertFalse(flags["all_restored"])
        self.assertEqual(flags["per_instance"], {"small_a": {"0": True, "1": False}})
        self.assertFalse(switch.restored_flags(None)["all_restored"])

    def test_ramp_deficit_matches_the_analyze_overlap_formula(self):
        rows = [("a", 5.0 + 10 * i, 1000 if i >= 3 else 100 * i) for i in range(20)]
        steady = 100.0
        expected = sum(max(steady - x, 0) for x in [0, 10, 20] + [100.0] * 9) * 10 / steady
        self.assertAlmostEqual(switch.ramp_deficit(rows, 0.0), expected)

    def test_policy_amendment_is_validated_like_run_arm(self):
        with self.assertRaises(ValueError):
            switch.merged_policy({"small_instances": ["x"]}, ["small_a"], "S")
        policy = switch.merged_policy({"large_fresh_traffic_share": 0.0}, ["small_a", "small_b"], "S")
        self.assertEqual(list(policy["small_instances"]), ["small_a", "small_b"])
        self.assertFalse(policy["migration_enabled"])


class SummaryTests(unittest.TestCase):
    def _campaign(self, mode):
        from scripts import switch_summary

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        artifacts, out = root / "artifacts", root / "out"
        out.mkdir()
        fabric, client, kw = dry.run_dry(mode, artifacts, scale=SCALE, run_id=f"{mode}-abc-r1")
        report = switch.run_switch(fabric, client, **kw)
        last = max(r["client_finished_at_s"] for r in read_gz(artifacts / kw["run_id"] / "phase2_records.json.gz"))
        (out / "campaign.json").write_text(json.dumps({"mode": mode, "run_base": f"{mode}-abc"}))
        (out / "runs.jsonl").write_text(json.dumps({"mode": mode, "rep": 1, "attempt": 1, "run_id": kw["run_id"], "ok": True}) + "\n")
        events = {f"{name}:ret-r1": last + offset for name, offset in (
            ("teardown_start", 1.0), ("drained", 3.0), ("gateway_down", 3.5), ("released", 5.0), ("deploy_start", 5.1),
            ("deploy_end", 9.0), ("check_start", 9.2), ("check_end", 10.0), ("pilot_start", 10.5))}
        (out / "timeline.jsonl").write_text("".join(json.dumps({"event": e, "t": t}) + "\n" for e, t in events.items()))
        pilot = lambda run_id: {"first_dispatch": last + 14.0, "ramp_deficit_s": 6.0, "all_requests_complete": True}  # noqa: E731
        return switch_summary, switch_summary.summarize(out, artifacts=artifacts, pilot_fn=pilot), report, last

    def test_full_summary_recomputes_offline_and_computes_harness_c(self):
        module, summary, report, last = self._campaign("full")
        rep = summary["reps"][0]
        self.assertLess(rep["recompute_max_segment_diff_s"], 1e-6)
        self.assertEqual(rep["endpoints"]["C_production_s"], report["endpoints"]["C_production_s"])
        harness = rep["harness"]
        self.assertAlmostEqual(harness["gap_s"], 14.0, places=2)
        self.assertAlmostEqual(harness["C_harness_s"], 20.0, places=2)
        self.assertAlmostEqual(harness["segment_sum_s"], harness["gap_s"], places=2)
        self.assertEqual([s["segment"] for s in harness["segments"]][0], "switch_tail")
        text = module.markdown(summary)
        self.assertIn("C_harness_s", text)
        self.assertIn("C_production_s", text)

    def test_partial_summary_reports_partial_endpoints(self):
        module, summary, report, last = self._campaign("partial")
        rep = summary["reps"][0]
        self.assertLess(rep["recompute_max_segment_diff_s"], 1e-6)
        self.assertNotIn("harness", rep)
        for key in ("kept_gap_s", "join_delay_s", "C_eff_s", "class_finish_s", "joiner_timeline"):
            self.assertIn(key, rep["endpoints"])
        self.assertIn("join_delay_s", module.markdown(summary))


if __name__ == "__main__":
    unittest.main()


class ClientAuthorityAccountingTest(unittest.TestCase):
    """Full mode phase 1: the HTTP segment summary has counts, not token ids (P2 attempt-2 false negative)."""

    def test_client_records_without_segment_token_ids_are_complete(self):
        from hetero.switch import _Switch

        recs = [{"request_id": f"ns:W0-{i}", "output_token_ids": [1] * 5, "useful_output_tokens": 5,
                 "segments": [{"instance": "small_a", "output_tokens": 5, "instance_id": "small_a"}]}
                for i in range(3)]
        manifest = {"requests": [{}] * 3, "useful_output_tokens": 15}
        acc = _Switch._accounting(None, "phase1", recs, recs, {r["request_id"]: 5 for r in recs}, [manifest])
        self.assertTrue(acc["flags"]["complete"])
        self.assertIn("client records", acc["flags"]["authority"])
        recs[0]["segments"][0]["output_tokens"] = 4  # a count mismatch must still fail
        acc = _Switch._accounting(None, "phase1", recs, recs, {r["request_id"]: 5 for r in recs}, [manifest])
        self.assertFalse(acc["flags"]["complete"])
