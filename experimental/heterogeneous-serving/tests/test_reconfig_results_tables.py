"""Auditor gates must reject early-stopped timers, no-op snapshots, borrowed directions."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts import reconfig_results_tables as rt


def chip(**over):
    base = {"device_id": 0, "bytes": 10, "files": 1, "reused_files": 0, "new_files": 1,
            "did_copy_arrays": 3, "strided_arrays": 0, "allocation_s": .1, "transfer_wait_s": .5,
            "copy_s": .2, "checksum_s": 0., "flush_s": .1, "transfer_end_s": .6}
    base.update(over)
    return base


def worker(**over):
    base = {"executor_rank": 0, "process_rank": 0, "bytes": 40, "snapshot_s": 2.0,
            "host_cache_bytes_before": 0, "reused_files": 0, "new_files": 4, "checksum_kind": "none",
            "chip_timings": [chip()],
            "timings": {"metadata_s": .1, "enqueue_s": .01, "transfer_s": .6, "transfer_wait_s": .5,
                        "allocation_s": .1, "copy_s": .2, "checksum_s": 0., "flush_s": .1, "publish_s": .05}}
    base.update(over)
    return base


class SnapshotWriteGates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.files = self.dump("S-files.json", {"ok": True})

    def tearDown(self):
        self.tmp.cleanup()

    def dump(self, name, measurements, **top):
        path = self.dir / name
        measurements = {"ok": True, **measurements}
        path.write_text(json.dumps({"submission_id": "x", "status": "SUCCEEDED", "measurements": measurements, **top}))
        return path

    def test_cold_write_accepted_and_reused_files_rejected(self):
        path = self.dump("S-cold.json", {"workers": {"a": [worker(), worker(process_rank=1, reused_files=2)]}})
        fp = self.dump("S-source-fingerprint.json", {"weight_equality": {"passed": True}})
        rows = rt.snapshot_write_rows(path, "cold", source_fingerprint=fp, files=self.files)["rows"]
        self.assertEqual(rows[0]["verdict"], "accepted")
        self.assertFalse(rows[0]["source_possibly_warm"])
        self.assertTrue(rows[0]["cold_snapshot_proof"])
        self.assertIn("published_files_valid", rt.snapshot_write_rows(path, "cold", source_fingerprint=fp)["rows"][0]["verdict"])
        self.assertIn("no_noop", rows[1]["verdict"])
        # without the cold-file fingerprint the bytes are unproven
        self.assertIn("cold_bytes_equal_source", rt.snapshot_write_rows(path, "cold")["rows"][0]["verdict"])

    def test_reused_write_needs_poison_evidence(self):
        path = self.dump("S-reused.json", {"workers": {"a": [worker(reused_files=4, new_files=0)]}})
        self.assertIn("no_noop", rt.snapshot_write_rows(path, "reused")["rows"][0]["verdict"])
        poison = self.dump("S-poison.json", {"hosts": [{"host": "h", "poisoned_bytes": 40}]})
        row = rt.snapshot_write_rows(path, "reused", poison=poison, files=self.files)["rows"][0]
        self.assertEqual(row["verdict"], "accepted")
        self.assertTrue(row["source_possibly_warm"])
        self.assertFalse(row["cold_snapshot_proof"])
        empty_poison = self.dump("S-poison-empty.json", {"hosts": []})
        self.assertIn("no_noop", rt.snapshot_write_rows(path, "reused", poison=empty_poison, files=self.files)["rows"][0]["verdict"])
        unpoisoned = self.dump("S-poison0.json", {"hosts": [{"host": "h", "poisoned_bytes": 0}]})
        self.assertIn("no_noop", rt.snapshot_write_rows(path, "reused", poison=unpoisoned)["rows"][0]["verdict"])

    def test_timer_stopping_before_transfer_or_flush_is_rejected(self):
        early = worker(snapshot_s=0.3)  # shorter than transfer_s: timer stopped before D2H finished
        path = self.dump("S-cold.json", {"workers": {"a": [early]}})
        self.assertIn("durable", rt.snapshot_write_rows(path, "cold")["rows"][0]["verdict"])
        warm_source = worker(host_cache_bytes_before=40, chip_timings=[chip(did_copy_arrays=0)])
        path = self.dump("S-cold2.json", {"workers": {"a": [warm_source]}})
        self.assertIn("d2h_complete", rt.snapshot_write_rows(path, "cold")["rows"][0]["verdict"])


class RestoreGates(unittest.TestCase):
    def phases(self, t0=1000.0):
        return {"init_device": {"start": t0, "end": t0 + 3},
                "load_model": {"start": t0 + 3, "end": t0 + 30},
                "snapshot_read_tmpfs": {"start": t0 + 3, "end": t0 + 4},
                "snapshot_device_placement": {"start": t0 + 20, "end": t0 + 22},
                "initialize_from_config": {"start": t0 + 30, "end": t0 + 56},
                "compile_or_warm_up_model": {"start": t0 + 56, "end": t0 + 119}}

    def restore(self, release_saved_at, restored=True, equal=True):
        with tempfile.TemporaryDirectory() as tmp:
            fp = None
            if equal is not None:
                fp = Path(tmp, "S-restored-fingerprint.json")
                fp.write_text(json.dumps({"measurements": {"weight_equality": {"passed": equal}}}))
            v = Path(tmp, "S-restored-validation.json")
            v.write_text(json.dumps({"submission_id": "v", "status": "SUCCEEDED", "measurements": {"ok": True,
                "worker_init_phases": {"a": [{"executor_rank": 0, "phases": self.phases(),
                                              "weight_snapshot": {"restored": restored, "checksums_requested": True,
                                                  "verified_checksums": False, "verified_bytes": 0, "checksum_kind": "none"}}]},
                "greedy_equality": True}}))
            r = Path(tmp, "S-normal-release.json")
            r.write_text(json.dumps({"status": "SUCCEEDED", "saved_at": release_saved_at,
                                     "measurements": {"released": [{"state": "REMOVED"}]}}))
            return rt.restore_rows(v, r, restored_fingerprint=fp)["rows"][0]

    def test_producer_must_exit_before_consumer_loads(self):
        row = self.restore(release_saved_at=900.0)
        self.assertEqual(row["verdict"], "accepted")
        self.assertAlmostEqual(row["phases_s"]["load_model"], 27.0)
        self.assertAlmostEqual(row["unattributed_load_s"], 24.0)
        self.assertIn("producer_released_first", self.restore(release_saved_at=1010.0)["verdict"])
        self.assertIn("restored_weights_equal_source", self.restore(release_saved_at=900.0, equal=False)["verdict"])
        self.assertIn("restored", self.restore(release_saved_at=900.0, restored=False)["verdict"])

    def test_missing_fingerprint_is_not_restore_proof(self):
        row = self.restore(release_saved_at=900.0, equal=None)
        self.assertIn("restored_weights_equal_source", row["verdict"])

    def test_fingerprint_does_not_imply_in_path_checksum(self):
        row = self.restore(release_saved_at=900.0)
        self.assertEqual(row["verdict"], "accepted")
        self.assertTrue(row["fingerprint_verified_outside_timer"])
        self.assertFalse(row["checksum_verified_in_path"])


class WarmupGates(unittest.TestCase):
    def deploy(self, tmp, name, cache_dir, intervals, events=()):
        d = Path(tmp, "run1")
        d.mkdir(exist_ok=True)
        p = d / name
        p.write_text(json.dumps({"submission_id": "d", "status": "SUCCEEDED", "measurements": {"ok": True, "deployment": {"instances": [{
            "descriptor": {"instance_id": "a"},
            "initialization": {"engine_settings": {"compilation_cache_dir": cache_dir},
                "resolved_engine_config": {"max_model_len": 16384, "max_num_seqs": 512,
                    "max_num_batched_tokens": 2048, "enable_prefix_caching": False},
                "worker_init_phases": [{
                "executor_rank": 0, "phases": {},
                "startup_profiles": {"compile_or_warm_up_model": {
                    "wall_s": 60.0, "exclusive_wall_s": {"cache_load": 10.0, "warmup_execute_and_sync": 50.0},
                    "intervals": intervals, "monitoring_events": list(events), "executables": [{}] * 3,
                    "compilation_cache_dir": cache_dir}}}]}}]}}}))
        return p

    def test_hits_only_from_hook_and_cold_dir_must_be_fresh(self):
        hook = [{"kind": "cache_load", "hit": True, "start": 0, "end": 1}] * 2 + \
               [{"kind": "cache_miss_lookup", "hit": False, "start": 1, "end": 2}]
        with tempfile.TemporaryDirectory() as tmp:
            p = self.deploy(tmp, "target-cold-deploy.json", "/models/jax_cache_overlap_run1_cold", hook)
            row = rt.warmup_rows(p, "run1", set(), {"qualification": "controlled"})["rows"][0]
            self.assertEqual((row["cache_hits"], row["cache_misses"]), (2, 1))
            self.assertEqual(row["verdict"], "accepted")
            # same cold dir already used by an earlier run -> not fresh
            row = rt.warmup_rows(p, "run1", {"/models/jax_cache_overlap_run1_cold"})["rows"][0]
            self.assertIn("fresh_cache_dir", row["verdict"])
            # counter claims hits the hook never saw -> not listener-proven
            p = self.deploy(tmp, "target-warm-deploy.json", "/models/jax_cache_shared", [],
                            events=[{"name": "/jax/compilation_cache/cache_hits", "metadata": {"count": 5}}])
            row = rt.warmup_rows(p, "run1", set())["rows"][0]
            self.assertEqual(row["cache_hits"], 0)
            self.assertIn("listener_hits", row["verdict"])


    def test_diagnostic_normal_and_warm_are_never_controls(self):
        # s-split-20261002c failure: success + cache hits, but tracing perturbed
        # startup and the runtime ignored the intended per-run cache directory.
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("normal-deploy.json", "warm-restore-deploy.json"):
                with self.subTest(name=name):
                    p = self.deploy(tmp, name, "/root/.cache/vllm/xla_cache", [{"hit": True}])
                    doc = json.loads(p.read_text())
                    ini = doc["measurements"]["deployment"]["instances"][0]["initialization"]
                    ini["engine_settings"]["compilation_cache_dir"] = "/models/run1_normal"
                    p.write_text(json.dumps(doc))
                    report = {"status": "diagnostic_only_not_a_cold_or_warm_control",
                              "limitations": ["Python tracing perturbed startup", "Wrong effective cache"],
                              "runs": [{"source": str(p), "run_id": "d"}]}
                    q = rt.overlap_qualification(p, doc, report)
                    for row in [*rt.warmup_rows(p, "run1", set(), q)["rows"],
                                *rt.restore_rows(p, require_restored=False, qualification=q)["rows"]]:
                        self.assertFalse(row["gates"]["evidence_qualified"])
                        self.assertFalse(row["gates"]["effective_cache_matches_requested"])
                        self.assertNotEqual(row["verdict"], "accepted")
                        self.assertEqual(row["evidence_qualification"]["qualification"], "diagnostic_only")

    def test_missing_or_stale_qualification_stays_unqualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.deploy(tmp, "warm-deploy.json", "/models/run1", [{"hit": True}])
            doc = json.loads(p.read_text())
            for report in ({}, {"runs": [{"source": str(p), "run_id": "older",
                                         "qualification": "controlled"}]}):
                q = rt.overlap_qualification(p, doc, report)
                row = rt.warmup_rows(p, "run1", set(), q)["rows"][0]
                self.assertEqual(q["qualification"], "unqualified")
                self.assertFalse(row["gates"]["evidence_qualified"])

    def test_control_qualification_cannot_override_config_or_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.deploy(tmp, "warm-deploy.json", "/models/run1", [{"hit": True}])
            doc = json.loads(p.read_text())
            report = {"runs": [{"source": str(p), "run_id": "d", "qualification": "controlled"}]}
            q = rt.overlap_qualification(p, doc, report)
            self.assertEqual(rt.warmup_rows(p, "run1", set(), q)["rows"][0]["verdict"], "accepted")
            ini = doc["measurements"]["deployment"]["instances"][0]["initialization"]
            ini["resolved_engine_config"]["max_num_seqs"] = 64
            p.write_text(json.dumps(doc))
            self.assertFalse(rt.warmup_rows(p, "run1", set(), q)["rows"][0]["gates"]["production_config"])
            q["exclusion_reasons"] = ["perturbed tracing"]
            self.assertFalse(rt.warmup_rows(p, "run1", set(), q)["rows"][0]["gates"]["evidence_qualified"])


class StartupProjectionGates(unittest.TestCase):
    def test_hash_mismatch_and_unknown_counts_never_qualify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "evidence.json"
            source.write_text("{}")
            path = root / "results/overlap/owner/startup_projection.json"
            path.parent.mkdir(parents=True)
            w = {"instance": "engine", "executor_rank": 0, "host": "host0",
                 "requested_cache_dir": "/cache", "phase_cache_dir": "/cache",
                 "resolved_engine_config": {"max_model_len": 16384, "max_num_seqs": 512,
                     "max_num_batched_tokens": 2048, "enable_prefix_caching": False},
                 "requested_worker_env": {k: "0" for k in ("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
                     "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "JAX_LOG_COMPILES", "JAX_EXPLAIN_CACHE_MISSES")},
                 "compile_cache": {"dir": "/cache", "min_compile_time_secs": 0, "min_entry_size_bytes": 0,
                     "log_compiles": False, "explain_cache_misses": False},
                 "phase_min_compile_time_secs": 0, "phase_min_entry_size_bytes": 0,
                 "cache_hits": 2, "cache_misses": 1, "keyed_cache_read_count": 3,
                 "successful_keyed_deserialization_count": 2, "cache_hit_listener_count": 2}
            doc = {"schema": "overlap-startup-table-projection-v1", "sources": [
                {"path": "evidence.json", "sha256": hashlib.sha256(b"{}").hexdigest()}],
                "runs": [{"source": "deploy.json", "run_id": "r", "phase": "initialize_from_config",
                    "qualification": "controlled", "exclusion_reasons": [], "source_qualified": False,
                    "workers": [{**w, "executor_rank": i, "host": f"host{i}"} for i in range(8)]}]}
            path.write_text(json.dumps(doc))
            runs, _ = rt.startup_projection_rows(path)
            self.assertTrue(all(r["verdict"] == "accepted" for r in runs[0]["rows"]))
            self.assertTrue(all(not r["source_qualified"] for r in runs[0]["rows"]))
            self.assertTrue(all(r["device_module_execution_wall_s"] is None for r in runs[0]["rows"]))
            doc["runs"][0]["workers"][0]["cache_hits"] = None
            path.write_text(json.dumps(doc))
            runs, _ = rt.startup_projection_rows(path)
            self.assertIsNone(runs[0]["rows"][0]["cache_hits"])
            self.assertFalse(runs[0]["rows"][0]["gates"]["keyed_cache_evidence"])
            source.write_text('{"changed":true}')
            runs, meta = rt.startup_projection_rows(path)
            self.assertFalse(meta["source_manifest_verified"])
            self.assertTrue(all(r["verdict"] != "accepted" for r in runs[0]["rows"]))


class HelperComponents(unittest.TestCase):
    def test_checksum_host_consumption_includes_readback_and_requires_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = {"repetition": 0, "kernel_ready_wall_s_including_enqueue": .27,
                   "small_output_readback_s": .46, "checksum_outputs": 4112,
                   "verified_outputs": 4112, "mismatch_indices": [], "checksum_output_sha256": "digest"}
            (root / "checksum-probe.json").write_text(json.dumps({"status": "SUCCEEDED",
                "measurements": {"probe": {"ok": True, "source_bytes": 31448474464, "runs": [run]}}}))
            audit = {"source_manifest_verified": True, "independent_digest_reconstruction": {"sha256": "digest"}}
            row = rt.helper_rows(root, audit)[0]
            self.assertAlmostEqual(row["elapsed_s"], .73)
            self.assertEqual(row["verdict"], "accepted")
            self.assertFalse(row["full_snapshot_proof"])
            self.assertFalse(rt.helper_rows(root, {})[0]["gates"]["independent_digest_verified"])
            audit["independent_digest_reconstruction"]["sha256"] = "wrong"
            self.assertFalse(rt.helper_rows(root, audit)[0]["gates"]["independent_digest_verified"])


class SwitchCostAndBreakeven(unittest.TestCase):
    def pilot(self, **changes):
        return {"run_id": "pilot", "layout": "H", "policy": "p", "all_requests_complete": True,
                "during_run": {}, "first_dispatch": 110.0, "last_finished_request": 10.0,
                "ramp_deficit_s": 5.0, "manifest_sha256": "manifest", "catalogue_sha256": "catalogue",
                "expected_catalogue_sha256": "catalogue", "unfrozen_pilot": False,
                "source_qualified": True, "control_qualified": True,
                "runtime_revision": "runtime", "backend_revision": "backend", **changes}

    def transition(self, **changes):
        return {"from": "H", "to": "S", "treatment": "warm", "last_finished_request": 10.0,
                "first_dispatch": 110.0, "dispatch_gap_s": 100.0, "ramp_deficit_s": 5.0,
                "cost_s": 105.0, "source": "source", "target": "target", **changes}

    def rates(self):
        return {"W0": {"output_tokens": 721595, "requests": 1763, "makespan_s": {
            "H": {**self.pilot(), "value": 497.2, "source": "measured-h"},
            "S": {**self.pilot(layout="S"), "value": 343.3, "source": "measured-s"}, "M": None}}}

    def cost(self, source=None, target=None, **transition):
        return rt.switch_row(self.transition(**transition), source or self.pilot(run_id="source"),
                             target or self.pilot(run_id="target", layout="S"), "c")

    def test_missing_pilot_completion_fails_even_when_cost_arithmetic_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp, "o", "run")
            d.mkdir(parents=True)
            (d / "overlap_switch.json").write_text(json.dumps({"transitions": [self.transition()]}))
            row = rt.switch_cost_rows(Path(tmp, "f"), Path(tmp, "o"))[0]
            self.assertFalse(row["observation_gates"]["target_all_requests_complete"])
            self.assertNotEqual(row["observation_verdict"], "accepted")
            target = self.pilot()
            target.pop("all_requests_complete")
            (d / "serving_impact.json").write_text(json.dumps({"pilots": {"source": self.pilot(), "target": target}}))
            row = rt.switch_cost_rows(Path(tmp, "f"), Path(tmp, "o"))[0]
            self.assertTrue(row["observation_gates"]["cost_is_gap_plus_deficit"])
            self.assertFalse(row["observation_gates"]["target_all_requests_complete"])
            self.assertFalse(row["performance_eligible"])

    def test_unfrozen_cost_is_observation_not_performance(self):
        row = self.cost(source=self.pilot(unfrozen_pilot=True, runtime_revision=None, backend_revision="UNFROZEN"))
        self.assertEqual(row["observation_verdict"], "accepted")
        self.assertEqual(row["verdict"], "OBSERVED ONLY")
        self.assertFalse(row["performance_eligible"])
        estimate = rt.breakeven_rows(self.rates(), [row])[0]
        self.assertTrue(estimate["model_compatible"])
        self.assertEqual(estimate["verdict"], "ESTIMATE ONLY")
        self.assertFalse(estimate["performance_eligible"])
        self.assertFalse(estimate["crossover_observed"])

    def test_catalogue_mismatch_or_unknown_suppresses_crossover_number(self):
        row = self.cost()
        for digest in (None, "historical-catalogue"):
            with self.subTest(digest=digest):
                table = self.rates()
                table["W0"]["makespan_s"]["H"]["catalogue_sha256"] = digest
                estimate = rt.breakeven_rows(table, [row])[0]
                self.assertFalse(estimate["model_compatible"])
                self.assertIsNone(estimate["breakeven_tokens"])
                self.assertIsNone(estimate["gain_per_phase_s"])
                self.assertFalse(estimate["crossover_observed"])
                self.assertEqual(estimate["sources"], ["measured-h", "measured-s"])

    def test_manifest_mismatch_and_missing_source_control_proof_fail(self):
        for field, value in (("manifest_sha256", "other"), ("source_qualified", None),
                             ("control_qualified", None), ("expected_catalogue_sha256", "historical")):
            with self.subTest(field=field):
                row = self.cost(target=self.pilot(**{field: value}))
                self.assertEqual(row["observation_verdict"], "accepted")
                self.assertFalse(row["performance_eligible"])

    def test_explicit_matching_evidence_supports_only_rate_model_estimate(self):
        rows = [self.cost(cost_s=228.6, dispatch_gap_s=223.6, first_dispatch=233.6,
                          target=self.pilot(first_dispatch=233.6)),
                self.cost(**{"from": "S", "to": "H"}),
                self.cost(**{"from": "M", "to": "S"})]
        be = rt.breakeven_rows(self.rates(), rows)
        self.assertEqual([(r["from"], r["to"]) for r in be], [("H", "S"), ("S", "H")])
        self.assertAlmostEqual(be[0]["breakeven_tokens"], 1071843, delta=1)
        self.assertTrue(be[0]["performance_eligible"])
        self.assertFalse(be[0]["crossover_observed"])
        self.assertIsNone(be[1]["breakeven_tokens"])
        self.assertTrue(be[1]["model_compatible"])

    def test_rate_runtime_drift_stays_provenance_labeled_estimate(self):
        table = self.rates()
        for value in table["W0"]["makespan_s"].values():
            if value:
                value["runtime_revision"] = "historical-runtime"
        row = rt.breakeven_rows(table, [self.cost()])[0]
        self.assertTrue(row["model_compatible"])
        self.assertFalse(row["performance_eligible"])
        self.assertFalse(row["performance_gates"]["source_matching_runtime_revision"])
        self.assertEqual(row["verdict"], "ESTIMATE ONLY")

    def test_compact_observation_retains_actual_catalogue_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            source, target = self.pilot(run_id="h"), self.pilot(run_id="s")
            doc = {"direction": "H_to_S", "treatment": "warm", "C_s": 105.0,
                   "dispatch_gap_s": 100.0, "ramp_deficit_s": 5.0, "source": source, "target": target,
                   "loaded_backend_provenance": {"backend_revision": "UNFROZEN", "loaded_runtime": None,
                                                  "runtime_equivalence_verified": False},
                   "equal_input_provenance": {"catalogue_sha256": "actual", "manifests": {
                       "W0": {"manifest_sha256": "manifest", "catalogue_sha256": "historical"}}}}
            (run / "warm-transition-observed.json").write_text(json.dumps(doc))
            (root / "overlap_switch.json").write_text(json.dumps({"transitions": [
                self.transition(previous_run_id="h", target_run_id="s")]}))
            (root / "serving_impact.json").write_text(json.dumps({"pilots": {"h": source, "s": target}}))
            rows = rt.switch_cost_rows(root / "absent", root)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["observation_verdict"], "accepted")
            self.assertEqual(row["verdict"], "OBSERVED ONLY")
            self.assertFalse(row["performance_gates"]["source_catalogue_matches_manifest"])
            self.assertEqual(row["comparison_provenance"]["source"]["catalogue_sha256"], "actual")

    def test_wrong_arithmetic_cannot_become_an_estimate(self):
        row = self.cost(cost_s=90.0)
        self.assertFalse(row["observation_gates"]["cost_is_gap_plus_deficit"])
        be = rt.breakeven_rows(self.rates(), [row])[0]
        self.assertFalse(be["model_compatible"])
        self.assertIsNone(be["breakeven_tokens"])

    def test_serving_comparison_requires_known_like_for_like_inputs(self):
        control = self.pilot()
        self.assertEqual(rt.verdict(rt.serving_gates(control, dict(control))), "accepted")
        for changes in ({"policy": "q"}, {"during_run": {"error": "boom"}},
                        {"catalogue_sha256": None}, {"layout": None}, {"all_requests_complete": None}):
            self.assertNotEqual(rt.verdict(rt.serving_gates(control, {**control, **changes})), "accepted")


if __name__ == "__main__":
    unittest.main()
