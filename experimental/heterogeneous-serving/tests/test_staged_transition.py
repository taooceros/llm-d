"""Staged S -> Mb -> H: layout Mb placement, transition planning and Mb serving policy.

Pure geometry and policy arithmetic: no Ray cluster, TPU or Kubernetes.  The
synthetic inventory copies the recorded v5e slice geometry (4x8 chips, 2x2 chips
per host, 2x4 host grid, hosts listed out of grid order as Ray reports them).
"""

import unittest

from hetero.experiment import ARMS, BASE_POLICY, _instance_roles
from hetero.gateway.discovery import instance_role
from hetero.gateway.policy import Candidate, CentralScheduler, RoutingPolicy
from hetero.layout import BUILTIN_LAYOUTS, builtin_layout, resolve_layout
from hetero.slice_inventory import HostRecord, SliceInventory
from hetero.transition import PARTIAL_TRANSITIONS, instance_placement, plan_transition

# (worker id, chip origin x, chip origin y) in the order the live inventory lists them.
HOSTS = [(1, 0, 0), (6, 2, 0), (7, 0, 2), (0, 2, 2), (5, 0, 4), (3, 2, 4), (4, 0, 6), (2, 2, 6)]
W4_RESERVE = {"decode_length_blind": True, "work_conserving_fallback": True,
              "small_prompt_token_limit": 16384, "long_reserve_tokens": 36864}
W4_BLIND = {"decode_length_blind": True}


def inventory():
    hosts = tuple(
        HostRecord(node_ip=f"10.0.0.{worker}", ray_node_id=f"node-{worker}",
                   hostname=f"host-{worker}", tpu_worker_id=worker, tpu_name="v5e",
                   tpu_count=4, chip_coords=((x, y, 0), (x + 1, y, 0), (x, y + 1, 0), (x + 1, y + 1, 0)),
                   jax_process_index=worker, jax_global_device_count=4)
        for worker, x, y in HOSTS)
    return SliceInventory(slice_name="slice", hosts=hosts, chip_dims=(4, 8, 1), host_chip_dims=(2, 2, 1),
                          host_grid_dims=(2, 4, 1), ray_session="s", observed_at=0.0,
                          ray_node_generation="g")


def resolved(layout_id):
    body = resolve_layout(inventory(), builtin_layout(layout_id)).to_dict()
    return {i["instance_id"]: i for i in body["instances"]}


def sizes(layout_id):
    return dict(BUILTIN_LAYOUTS[layout_id])


def plan(source, target, *, to_placement=None, to_instances=None):
    return plan_transition(source, target, sizes(source), to_instances or sizes(target),
                           from_placement=resolved(source),
                           to_placement=to_placement or resolved(target))


class MbPlacementTests(unittest.TestCase):
    def test_mb_is_m_with_large_b(self):
        self.assertEqual(BUILTIN_LAYOUTS["Mb"], (("small_a", 8), ("small_b", 8), ("large_b", 16)))

    def test_mb_large_b_occupies_exactly_h_large_b(self):
        mb, h, m = resolved("Mb"), resolved("H"), resolved("M")
        self.assertEqual(instance_placement(mb["large_b"]), instance_placement(h["large_b"]))
        # Same hosts in the same rank order, and the same chips on each host.
        self.assertEqual(mb["large_b"]["host_ips"], h["large_b"]["host_ips"])
        self.assertEqual([x["chip_coords"] for x in mb["large_b"]["hosts"]],
                         [x["chip_coords"] for x in h["large_b"]["hosts"]])
        self.assertEqual(mb["large_b"]["host_ips"], ["10.0.0.5", "10.0.0.3", "10.0.0.4", "10.0.0.2"])
        self.assertEqual(mb["large_b"]["chip_box"], [[0, 3], [4, 7], [0, 0]])
        # M places its TP16 identically (the documented M/H overlap); only the name differs.
        self.assertEqual(instance_placement(m["large"]), instance_placement(mb["large_b"]))

    def test_mb_smalls_occupy_exactly_s_smalls(self):
        mb, s = resolved("Mb"), resolved("S")
        for name in ("small_a", "small_b"):
            self.assertEqual(instance_placement(mb[name]), instance_placement(s[name]))
            self.assertEqual(mb[name]["chip_box"], s[name]["chip_box"])
        # The released S engines free exactly the chips large_b takes, and vice versa for H.
        chips = lambda inst: {tuple(c) for host in inst["hosts"] for c in host["chip_coords"]}
        self.assertEqual(chips(s["small_c"]) | chips(s["small_d"]), chips(mb["large_b"]))
        self.assertEqual(chips(mb["small_a"]) | chips(mb["small_b"]), chips(resolved("H")["large_a"]))


class StagedPlanTests(unittest.TestCase):
    def test_s_to_mb(self):
        p = plan("S", "Mb")
        self.assertEqual((p.kept, p.release, p.join),
                         (("small_a", "small_b"), ("small_c", "small_d"), ("large_b",)))

    def test_mb_to_h(self):
        p = plan("Mb", "H")
        self.assertEqual((p.kept, p.release, p.join), (("large_b",), ("small_a", "small_b"), ("large_a",)))

    def test_reverse_directions_and_s_m_still_planned(self):
        self.assertEqual(plan("H", "Mb").join, ("small_a", "small_b"))
        self.assertEqual(plan("Mb", "S").join, ("small_c", "small_d"))
        self.assertEqual(plan("S", "M").join, ("large",))
        self.assertEqual(plan("M", "S").release, ("large",))

    def test_refuses_undefined_pairs(self):
        for source, target in (("S", "H"), ("H", "S"), ("M", "H"), ("H", "M"), ("M", "Mb"),
                               ("Mb", "M"), ("Mb", "Mb"), ("S", "S")):
            with self.subTest(pair=(source, target)):
                with self.assertRaisesRegex(ValueError, "no partial transition"):
                    plan_transition(source, target, sizes(source), sizes(target))
        self.assertNotIn(frozenset({"M", "H"}), PARTIAL_TRANSITIONS)

    def test_refuses_kept_tp_change(self):
        with self.assertRaisesRegex(ValueError, "tensor-parallel"):
            plan_transition("Mb", "H", sizes("Mb"), {"large_a": 8, "large_b": 8, "small_c": 8, "small_d": 8})

    def test_refuses_non_chip_conserving(self):
        with self.assertRaisesRegex(ValueError, "same number of chips"):
            plan_transition("S", "Mb", sizes("S"), {"small_a": 8, "small_b": 8, "large_b": 8})
        with self.assertRaisesRegex(ValueError, "same number of chips"):
            plan_transition("Mb", "H", sizes("Mb"), {"large_a": 8, "large_b": 16})

    def test_refuses_kept_instance_on_other_hosts(self):
        h = resolved("H")
        swapped = dict(h, large_a=dict(h["large_b"], instance_id="large_a"),
                       large_b=dict(h["large_a"], instance_id="large_b"))
        with self.assertRaisesRegex(ValueError, r"change hosts or chips: \['large_b'\]"):
            plan("Mb", "H", to_placement=swapped)
        # Same hosts in another rank order also changes the instance's physical identity.
        reordered = dict(h, large_b=dict(h["large_b"], host_ips=h["large_b"]["host_ips"][::-1],
                                         hosts=h["large_b"]["hosts"][::-1]))
        with self.assertRaisesRegex(ValueError, "change hosts or chips"):
            plan("Mb", "H", to_placement=reordered)

    def test_refuses_partial_placement_arguments(self):
        with self.assertRaisesRegex(ValueError, "both source and target placements"):
            plan_transition("S", "Mb", sizes("S"), sizes("Mb"), from_placement=resolved("S"))
        with self.assertRaisesRegex(ValueError, "no placement for kept"):
            plan_transition("S", "Mb", sizes("S"), sizes("Mb"),
                            from_placement=resolved("S"), to_placement={"large_b": resolved("Mb")["large_b"]})

    def test_transition_arms_exist_for_every_staged_layout(self):
        self.assertEqual(ARMS["Mb0"], {"layout": "Mb", "migration_enabled": False,
                                       "migration_mechanism": "recompute"})
        for layout in ("S", "Mb", "H"):
            self.assertTrue(any(spec["layout"] == layout for spec in ARMS.values()))


def candidate(name, *, admissible, total=600):
    tp, max_len = (16, 36864) if name.startswith("large") else (8, 16384)
    return Candidate(instance_id=name, tp_size=tp, max_model_len=max_len, block_size=256,
                     total_blocks=total, free_blocks=admissible, admissible_blocks=admissible,
                     running=0, waiting=0, inflight_migrations=0)


class MbServingPolicyTests(unittest.TestCase):
    def policy(self, amendment):
        # As run_arm builds it: BASE_POLICY, the pilot amendment, then roles from names.
        roles = _instance_roles("Mb", [name for name, _ in BUILTIN_LAYOUTS["Mb"]])
        return RoutingPolicy.from_dict({**BASE_POLICY, **amendment, **roles})

    def test_roles_from_names(self):
        self.assertEqual(_instance_roles("Mb", ["large_b", "small_b", "small_a"]),
                         {"small_instances": ["small_a", "small_b"], "large_instances": ["large_b"]})
        self.assertEqual(instance_role("large_b"), "large")
        self.assertEqual(instance_role("small_a"), "small")

    def test_cli_gateway_policy_for_mb(self):
        import argparse
        from hetero.cli import _routing_policy
        args = argparse.Namespace(policy_file=None, policy=None, layout="Mb", migration=False)
        self.assertEqual(_routing_policy(args), {"small_instances": ["small_a", "small_b"],
                                                 "large_instances": ["large_b"], "migration_enabled": False})

    def test_w4_reserve_policy_routes_long_to_large_b_and_keeps_its_reserve(self):
        scheduler = CentralScheduler(self.policy(W4_RESERVE))
        fabric = [candidate("small_a", admissible=500), candidate("small_b", admissible=500),
                  candidate("large_b", admissible=500)]
        chosen, trace = scheduler.select_initial(fabric, prompt_tokens=20000, max_tokens=512)
        self.assertEqual(chosen, "large_b")
        self.assertEqual(trace["reason"], "prompt_does_not_fit_a_small_instance")
        # Short requests may not take large_b's 36864-token long reserve (144 blocks + headroom).
        scheduler = CentralScheduler(self.policy(W4_RESERVE))
        tight = [candidate("small_a", admissible=0), candidate("small_b", admissible=0),
                 candidate("large_b", admissible=150)]
        chosen, trace = scheduler.select_initial(tight, prompt_tokens=1000, max_tokens=512)
        self.assertIsNone(chosen)
        self.assertEqual(trace["refusal"], "no_destination_has_admissible_capacity")

    def test_w4_blind_policy_serves_both_pools_without_reading_max_tokens(self):
        scheduler = CentralScheduler(self.policy(W4_BLIND))
        fabric = [candidate("small_a", admissible=500), candidate("small_b", admissible=500),
                  candidate("large_b", admissible=500)]
        chosen_ids = {scheduler.select_initial(fabric, prompt_tokens=1000, max_tokens=10 ** 6)[0]
                      for _ in range(4)}
        self.assertIn("large_b", chosen_ids)
        self.assertTrue(chosen_ids & {"small_a", "small_b"})


if __name__ == "__main__":
    unittest.main()
