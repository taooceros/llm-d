"""S<->M partial reconfiguration: gateway joining semantics, additive reconcile, plan arithmetic.

No Ray cluster, TPU or Kubernetes: the gateway actor class runs in-process against
fake engine/registry actors (the same harness as test_gateway_lifecycle).
"""

import asyncio
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hetero.gateway.discovery import DiscoveryBridge, INSTANCE_LABEL, LAYOUT_LABEL
from hetero.gateway.service import OmpGateway
from hetero.registry import InstanceDescriptor, descriptor_for_layout
from hetero.transition import partition_records, plan_transition
from test_gateway_discovery import KubeFixture
from test_gateway_lifecycle import Actor, NativePeer

M = {"small_a": 8, "small_b": 8, "large": 16}
S = {"small_a": 8, "small_b": 8, "small_c": 8, "small_d": 8}
IPS = {"small_a": "10.0.0.1", "small_b": "10.0.0.2", "large": "10.0.0.3",
       "small_c": "10.0.0.4", "small_d": "10.0.0.5"}


def make_row(name, layout="M", generation=1):
    ip = IPS[name]
    return {"instance_id": name, "layout_id": layout, "generation": generation,
            "config_digest": f"{layout}-digest", "engine_actor_name": f"actor-{name}",
            "ray_namespace": "omp_hetero", "placement_group_id": f"pg-{name}",
            "endpoint_host": ip, "endpoint_port": 8000, "endpoint": f"{ip}:8000",
            "host_chip_rank_map": [{"process_rank": 0, "node_ip": ip}],
            "state": "ready", "routable": True, "health_age_s": 0,
            "tp_size": 16 if name == "large" else 8, "max_model_len": 128}


class LayoutRegistry:
    def __init__(self):
        self.rows = [make_row(n) for n in M]
        self.layouts = {"M": self.layout("M", M)}

    @staticmethod
    def layout(layout_id, members):
        return {"layout_id": layout_id, "layout_digest": f"{layout_id}-digest",
                "instances": [{"instance_id": n, "tp_size": tp} for n, tp in members.items()]}

    def snapshot(self, layout_id=None):
        return copy.deepcopy([r for r in self.rows if layout_id is None or r["layout_id"] == layout_id])

    def get_layout(self, layout_id):
        return copy.deepcopy(self.layouts.get(layout_id))


class PartialReconfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.kube = KubeFixture()
        self.kube.pods = [KubeFixture.pod("system-daemon", "10.0.0.1", worker=False)] + [
            KubeFixture.pod(f"pod-{name}", ip) for name, ip in IPS.items()]
        self.registry = LayoutRegistry()
        self.native = {r["instance_id"]: NativePeer(copy.deepcopy(r)) for r in self.registry.rows}
        self.actors = {f"actor-{n}": Actor(p) for n, p in self.native.items()}
        self.patches = [
            mock.patch("hetero.gateway.service.ray.util.get_node_ip_address", return_value="127.0.0.1"),
            mock.patch("hetero.gateway.service.find_registry", return_value=Actor(self.registry)),
            mock.patch("hetero.gateway.service.ray.get_actor", side_effect=lambda name, namespace: self.actors[name]),
        ]
        for patch in self.patches:
            patch.start()
        self.gateway = OmpGateway.__ray_metadata__.modified_class(config={
            "layout_id": "M", "ledger_path": str(Path(self.directory.name) / "ownership.json"),
            "policy": {"small_instances": ["small_a", "small_b"], "large_instances": ["large"],
                       "migration_enabled": False},
        })
        self.gateway.state.bridge = DiscoveryBridge(kube=self.kube)
        await self.gateway.reconcile()

    async def asyncTearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.directory.cleanup()

    # ------------------------------------------------------------ helpers

    async def drain_large_and_adopt_kept(self):
        maintenance = await self.gateway.begin_maintenance(["large"], reason="M->S")
        await self.gateway.drain_maintenance_instance("large", maintenance_id=maintenance["id"], timeout_s=1)
        self.registry.rows = [r for r in self.registry.rows if r["instance_id"] != "large"]
        del self.actors["actor-large"]
        self.registry.layouts["S"] = self.registry.layout("S", S)
        for row in self.registry.rows:
            row.update(descriptor_for_layout(row, "S", "S-digest"))
            self.native[row["instance_id"]].row.update(layout_id="S", config_digest="S-digest")
        return maintenance

    def register_joiners(self, generation=1):
        for name in ("small_c", "small_d"):
            row = make_row(name, "S", generation)
            self.registry.rows.append(row)
            self.native[name] = NativePeer(copy.deepcopy(row))
            self.actors[f"actor-{name}"] = Actor(self.native[name])

    async def dispatch(self, request_id):
        return await self.gateway.dispatch(
            {"request_id": request_id, "prompt_token_ids": [1], "max_tokens": 1})

    # -------------------------------------------------------------- tests

    async def test_retarget_fences_joiners_until_additive_join_under_live_traffic(self):
        kept_clients = dict(self.gateway.state.core.clients)
        await self.drain_large_and_adopt_kept()
        retargeted = await self.gateway.retarget_layout("S", ["small_c", "small_d"])
        status = self.gateway.status()
        self.assertEqual((status["layout_id"], status["joining"], status["maintenance"]),
                         ("S", ["small_c", "small_d"], {}))
        self.assertEqual(status["instances"], ["small_a", "small_b"])
        self.assertEqual(retargeted["joining"], ["small_c", "small_d"])
        self.assertTrue(status["discovery"]["ready"])
        self.assertEqual({p["metadata"]["labels"].get(LAYOUT_LABEL) for p in self.kube.pods
                          if INSTANCE_LABEL in p["metadata"]["labels"]}, {"S"})
        self.assertEqual(status["core"]["config"]["layout_id"], "S")

        # Registered-but-joining engines must neither break discovery (503) nor be bound/labelled.
        self.register_joiners()
        await self.gateway._reconcile(bind_labels=False, observe_only=True)
        await self.gateway.reconcile(bind_labels=False)
        self.assertEqual(self.gateway.status()["instances"], ["small_a", "small_b"])
        self.assertFalse(any(INSTANCE_LABEL in p["metadata"]["labels"]
                             for p in self.kube.pods if p["metadata"]["name"] in ("pod-small_c", "pod-small_d")))
        self.assertEqual((await self.dispatch("pre-join"))["initial_instance"] in ("small_a", "small_b"), True)

        # Join while two requests are genuinely in flight: not idle-gated, no rebind of live clients.
        for peer in (self.native["small_a"], self.native["small_b"]):
            peer.result_allowed.clear()
        inflight = [asyncio.create_task(self.dispatch(f"live-{i}")) for i in range(2)]
        await asyncio.wait_for(asyncio.gather(self.native["small_a"].started.wait(),
                                              self.native["small_b"].started.wait()), timeout=5)
        self.assertGreater(self.gateway.state.active_requests, 0)
        scheduler = self.gateway.state.core.scheduler
        self.assertGreater(scheduler._fresh_dispatched, 0)
        receipt = await self.gateway.join_instances(["small_c", "small_d"])

        status = self.gateway.status()
        self.assertEqual(status["instances"], ["small_a", "small_b", "small_c", "small_d"])
        self.assertEqual(status["joining"], [])
        self.assertEqual(receipt["joined"], ["small_c", "small_d"])
        self.assertIsInstance(receipt["joined_at"], float)
        for name, client in kept_clients.items():
            if name != "large":
                self.assertEqual(self.gateway.state.core.clients[name].actor, client.actor)
        self.assertEqual(receipt["policy_roles"]["small_instances"], ["small_a", "small_b", "small_c", "small_d"])
        self.assertEqual(self.gateway.state.core.config.policy.large_instances, ())
        self.assertEqual((scheduler._fresh_dispatched, scheduler._fresh_to_large), (0, 0))
        self.assertTrue(status["discovery"]["ready"])
        self.assertTrue(all(INSTANCE_LABEL in p["metadata"]["labels"]
                            for p in self.kube.pods if p["metadata"]["name"].startswith("pod-small")))

        for peer in (self.native["small_a"], self.native["small_b"]):
            peer.result_allowed.set()
        results = await asyncio.gather(*inflight)
        self.assertEqual(len(results), 2)
        later = {(await self.dispatch(f"post-join-{i}"))["initial_instance"] for i in range(8)}
        self.assertTrue(later & {"small_c", "small_d"})

    async def test_join_refuses_non_additive_change_and_keeps_serving(self):
        await self.drain_large_and_adopt_kept()
        await self.gateway.retarget_layout("S", ["small_c", "small_d"])
        self.register_joiners()
        # A kept engine silently replaced mid-join must not be swallowed by the additive path.
        row = next(r for r in self.registry.rows if r["instance_id"] == "small_a")
        row["generation"] = 2
        self.native["small_a"].row["generation"] = 2
        with self.assertRaises(RuntimeError):
            await self.gateway.join_instances(["small_c", "small_d"])
        status = self.gateway.status()
        self.assertEqual((status["instances"], status["joining"]),
                         (["small_a", "small_b"], ["small_c", "small_d"]))
        self.assertTrue(status["discovery"]["ready"])  # the verdict traffic depends on survives
        with self.assertRaises(RuntimeError):
            await self.gateway.join_instances(["small_a"])  # not a joiner

    async def test_join_with_unready_joiner_fails_without_touching_live_fabric(self):
        await self.drain_large_and_adopt_kept()
        await self.gateway.retarget_layout("S", ["small_c", "small_d"])
        self.register_joiners()
        self.registry.rows[-1]["routable"] = False
        with self.assertRaises(RuntimeError):
            await self.gateway.join_instances(["small_c", "small_d"])
        status = self.gateway.status()
        self.assertEqual(status["instances"], ["small_a", "small_b"])
        self.assertTrue(status["discovery"]["ready"])
        self.assertEqual((await self.dispatch("still-serving"))["initial_instance"][:5], "small")

    async def test_join_waits_for_joiner_pod_ready_and_never_labels_it_early(self):
        # Cluster failure mode: the engine is up but its coordinator pod is briefly Ready=False.
        await self.drain_large_and_adopt_kept()
        await self.gateway.retarget_layout("S", ["small_c", "small_d"])
        self.register_joiners()
        pod = next(p for p in self.kube.pods if p["metadata"]["name"] == "pod-small_d")
        pod["status"]["conditions"] = [{"type": "Ready", "status": "False"}]
        join = asyncio.create_task(self.gateway.join_instances(["small_c", "small_d"]))
        await asyncio.sleep(0.1)
        self.assertFalse(join.done())
        self.assertNotIn(INSTANCE_LABEL, pod["metadata"]["labels"])
        self.assertTrue(self.gateway.status()["discovery"]["ready"])  # live traffic unaffected
        pod["status"]["conditions"] = [{"type": "Ready", "status": "True"}]
        receipt = await asyncio.wait_for(join, timeout=10)
        self.assertEqual(receipt["joined"], ["small_c", "small_d"])
        self.assertEqual(pod["metadata"]["labels"].get(INSTANCE_LABEL), "small_d")

    async def test_retarget_requires_drained_and_withdrawn_release_and_registered_kept(self):
        with self.assertRaises(RuntimeError):  # no declared maintenance
            await self.gateway.retarget_layout("S", ["small_c", "small_d"])
        maintenance = await self.gateway.begin_maintenance(["large"], reason="M->S")
        self.registry.layouts["S"] = self.registry.layout("S", S)
        with self.assertRaises(RuntimeError):  # not drained
            await self.gateway.retarget_layout("S", ["small_c", "small_d"])
        await self.gateway.drain_maintenance_instance("large", maintenance_id=maintenance["id"], timeout_s=1)
        with self.assertRaises(RuntimeError):  # drained but still registered
            await self.gateway.retarget_layout("S", ["small_c", "small_d"])
        self.registry.rows = [r for r in self.registry.rows if r["instance_id"] != "large"]
        with self.assertRaises(RuntimeError):  # kept engines not yet re-registered under S
            await self.gateway.retarget_layout("S", ["small_c", "small_d"])
        with self.assertRaises(RuntimeError):  # joiners must complete the target layout
            await self.gateway.retarget_layout("S", ["small_c"])
        # Every refusal left the gateway in its declared maintenance state on layout M.
        self.assertEqual((self.gateway.status()["layout_id"], self.gateway.status()["joining"]), ("M", []))
        self.assertEqual(self.gateway.status()["maintenance"]["id"], maintenance["id"])

    async def test_failed_retarget_restores_source_layout_and_maintenance(self):
        maintenance = await self.drain_large_and_adopt_kept()
        self.native["small_b"].row.update(layout_id="M")  # actor identity disagrees with the registry
        with self.assertRaises(RuntimeError):
            await self.gateway.retarget_layout("S", ["small_c", "small_d"])
        status = self.gateway.status()
        self.assertEqual((status["layout_id"], status["joining"], status["maintenance"]["id"]),
                         ("M", [], maintenance["id"]))

    async def test_join_starts_capture_on_joiners_and_applies_arm_policy_rules(self):
        await self.drain_large_and_adopt_kept()
        await self.gateway.retarget_layout("S", ["small_c", "small_d"])
        self.register_joiners()
        captured = []
        for name in ("small_c", "small_d"):
            self.native[name].start_iteration_capture = (lambda n=name: captured.append(n))
        self.gateway.state.capturing = True
        receipt = await self.gateway.join_instances(
            ["small_c", "small_d"],
            {"large_fresh_traffic_share": 0.5, "migration_enabled": True, "migration_mechanism": "recompute"})
        self.assertEqual(sorted(captured), ["small_c", "small_d"])
        policy = self.gateway.state.core.config.policy
        # All-small after the join: the arm's homogeneous rule, not the heterogeneous share.
        self.assertEqual((policy.large_fresh_traffic_share, policy.migration_enabled), (0.0, False))
        self.assertEqual(receipt["policy_roles"]["large_instances"], [])

    async def test_no_maintenance_while_joining(self):
        await self.drain_large_and_adopt_kept()
        await self.gateway.retarget_layout("S", ["small_c", "small_d"])
        with self.assertRaises(RuntimeError):
            await self.gateway.begin_maintenance(["small_a"], reason="must refuse")

    async def test_unjoined_registered_engine_is_a_hazard_without_the_joining_fence(self):
        """Contrast: an extra registered row outside `joining` fails discovery closed."""
        await self.drain_large_and_adopt_kept()
        await self.gateway.retarget_layout("S", ["small_c", "small_d"])
        self.register_joiners()
        self.gateway.state.joining = set()
        with self.assertRaises(RuntimeError):
            await self.gateway._reconcile(bind_labels=False, observe_only=True)
        self.assertFalse(self.gateway.status()["discovery"]["ready"])


class PlanTests(unittest.TestCase):
    def test_membership_arithmetic_both_directions(self):
        down = plan_transition("M", "S", M, S)
        self.assertEqual((down.kept, down.release, down.join),
                         (("small_a", "small_b"), ("large",), ("small_c", "small_d")))
        up = plan_transition("S", "M", S, M)
        self.assertEqual((up.kept, up.release, up.join),
                         (("small_a", "small_b"), ("small_c", "small_d"), ("large",)))

    def test_refuses_unsupported_or_chip_changing_transitions(self):
        with self.assertRaises(ValueError):
            plan_transition("M", "H", M, {"large_a": 16, "large_b": 16})
        with self.assertRaises(ValueError):
            plan_transition("M", "S", M, {**S, "small_a": 16})
        with self.assertRaises(ValueError):
            plan_transition("M", "S", M, {"small_a": 8, "small_b": 8, "small_c": 8})


class PartitionTests(unittest.TestCase):
    def test_records_split_by_admitted_time_around_the_join(self):
        records = [
            {"request_id": "a", "client_started_at_s": 1.0, "client_finished_at_s": 9.0,
             "useful_output_tokens": 10, "initial_instance": "small_a"},
            {"request_id": "b", "client_started_at_s": 5.0, "client_finished_at_s": 12.0,
             "useful_output_tokens": 20, "initial_instance": "small_b"},
            {"request_id": "c", "client_started_at_s": 6.0, "client_finished_at_s": 13.0,
             "useful_output_tokens": 30, "initial_instance": "small_c"},
            {"request_id": "d", "client_started_at_s": 8.0, "client_finished_at_s": 15.0,
             "useful_output_tokens": 40, "initial_instance": "small_d"},
            {"request_id": "e", "client_started_at_s": 9.0, "client_finished_at_s": 16.0, "error": "boom"},
        ]
        part = partition_records(records, join_started_at=5.0, join_finished_at=8.0)
        self.assertEqual([part[k]["requests"] for k in ("pre_join", "during_join", "post_join")], [1, 2, 2])
        self.assertEqual(part["pre_join"]["initial_placement"], {"small_a": 1})
        self.assertEqual(part["during_join"]["useful_output_tokens"], 50)
        self.assertEqual(part["post_join"]["errors"], 1)
        self.assertEqual(part["post_join"]["last_finished_at"], 16.0)
        with self.assertRaises(ValueError):
            partition_records([{"request_id": "x"}], 0.0, 1.0)


class RegistryAndEngineIdentityTests(unittest.TestCase):
    def test_descriptor_for_layout_changes_only_layout_identity(self):
        row = {**make_row("small_a"), "endpoint": "10.0.0.1:8000", "routable": True, "health_age_s": 3.0}
        body = descriptor_for_layout(row, "S", "S-digest")
        self.assertEqual((body["layout_id"], body["config_digest"]), ("S", "S-digest"))
        for key in ("generation", "engine_actor_name", "placement_group_id", "endpoint_host", "state"):
            self.assertEqual(body[key], row[key])
        self.assertNotIn("routable", body)
        InstanceDescriptor(**{**body, "placement_group_name": "pg"})  # accepted by register()

    def test_engine_adopt_identity_validates_placement_and_keeps_generation(self):
        from hetero.engine_actor import HeteroEngine

        cls = HeteroEngine.__ray_metadata__.modified_class
        engine = object.__new__(cls)
        instance = {"instance_id": "small_a", "tp_size": 8, "host_ips": ["10.0.0.1"],
                    "hosts": [{"node_ip": "10.0.0.1", "chip_coords": [[0, 0], [1, 0]]}]}
        engine.__dict__.update(
            instance=instance, instance_id="small_a", layout_id="M", layout_digest="M-digest",
            state="ready", _draining=False, generation=7, node_ip="10.0.0.1", hostname="h",
            _init_evidence={}, settings=mock.Mock(to_dict=lambda: {}, backend_revision="r"))
        moved = {**instance, "hosts": [{"node_ip": "10.0.0.1", "chip_coords": [[2, 0], [3, 0]]}]}
        with self.assertRaises(RuntimeError):
            cls.adopt_identity(engine, "S", "S-digest", moved)
        cls.adopt_identity(engine, "S", "S-digest", copy.deepcopy(instance), dry_run=True)
        self.assertEqual(engine.layout_id, "M")
        identity = cls.adopt_identity(engine, "S", "S-digest", copy.deepcopy(instance))
        self.assertEqual((identity["layout_id"], identity["layout_digest"], identity["generation"]),
                         ("S", "S-digest", 7))
        engine._draining = True
        with self.assertRaises(RuntimeError):
            cls.adopt_identity(engine, "M", "M-digest")


if __name__ == "__main__":
    unittest.main()
