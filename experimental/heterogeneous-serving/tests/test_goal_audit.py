"""Missing cleanup evidence must never be counted as an idle cluster."""
import unittest
from unittest.mock import patch

from scripts import goal_audit


class CleanupEvidenceTest(unittest.TestCase):
    def gate(self, inspections):
        with patch.object(goal_audit, "_load", side_effect=lambda path: (
            {"status": "SUCCEEDED", "measurements": {"ok": True}} if "final_inspect_" in path else {}
        )), patch.object(
            goal_audit, "_measurements", side_effect=lambda path: (
                inspections.get(path.rsplit("_", 1)[-1].removesuffix(".json"), {})
                if "inspect_" in path else {})
        ):
            result = goal_audit.build({}, {})
        return next(row for row in result["acceptance_gates"] if row["gate"].startswith("6."))

    def idle(self):
        return {
            layout: {
                "inspection": {"layout_id": layout, "registry": [], "observed": [],
                               "owned_placement_groups": []}
            }
            for layout in ("H", "S", "M")
        }

    def test_requires_complete_inspection_for_every_layout(self):
        records = self.idle()
        self.assertEqual(self.gate(records)["status"], "passed")
        records.pop("M")
        self.assertEqual(self.gate(records)["status"], "missing_inspection_evidence")
        records = self.idle()
        del records["M"]["inspection"]["observed"]
        self.assertEqual(self.gate(records)["status"], "missing_inspection_evidence")

    def test_pending_group_prevents_cleanup_pass(self):
        records = self.idle()
        groups = records["M"]["inspection"]["owned_placement_groups"]
        groups.append({"state": "PENDING"})
        self.assertEqual(self.gate(records)["status"], "residual_state")


class SelectedCohortTest(unittest.TestCase):
    def setUp(self):
        from hetero.analysis import evaluate_criteria

        self.protocol = {"predeclaration_sha256": "selected", "source_freeze": {"status": "qualified"},
                         "arm_order": [{"workload": workload, "repetition": rep, "arms": arms}
                                       for workload, arms in (("W0", ["H", "S", "M0", "MR", "MK-H"]),
                                                              ("W1", ["H", "S", "M0", "MR"]))
                                       for rep in (1, 2, 3)]}
        runs = [{"workload": block["workload"], "arm": arm, "repetition": block["repetition"],
                 "predeclaration_sha256": "selected", "manifest_sha256": block["workload"],
                 "ingress_url": "http://gateway", "ingress_protocol": "envoy-ext-proc-v3",
                 "all_requests_complete": True, "iteration_capture_one_to_one": True,
                 "client_delivery_matches": True, "committed_prefix_preserved": True,
                 "native_output_budget_exact": True, "expected_output_tokens": 64, "useful_output_tokens": 64,
                 "continuation_qualification_passed": True,
                 "migrations_committed": 1 if arm in ("MR", "MK-H") else 0,
                 "transferred_bytes": 256 if arm == "MK-H" else 0,
                 "makespan_s": {"H": 10, "S": 11, "M0": 12, "MR": 13, "MK-H": 14}[arm]}
                for block in self.protocol["arm_order"] for arm in block["arms"]]
        self.results = {"predeclaration": self.protocol, "arms": runs,
                        "criteria": {name: evaluate_criteria([row for row in runs if row["workload"] == name],
                                                             minimum_effect=0.05) for name in ("W0", "W1")}}

    def comparison_gate(self):
        # Isolate comparison completeness from separately tested physical prerequisites.
        with patch.object(goal_audit, "_mechanism_errors", return_value=[]):
            result = goal_audit.build(self.results, {})
        return next(row for row in result["acceptance_gates"] if row["gate"].startswith("5."))

    def test_exact_selected_tuples_required_not_total_run_count(self):
        self.assertEqual(self.comparison_gate()["status"], "passed")
        self.results["arms"][-1]["predeclaration_sha256"] = "historical"
        self.results["eligible_headline_runs"] = len(self.results["arms"])
        self.assertNotEqual(self.comparison_gate()["status"], "passed")
        self.results["arms"][-1]["predeclaration_sha256"] = "selected"
        self.results["arms"].append(dict(self.results["arms"][-1]))
        self.assertNotEqual(self.comparison_gate()["status"], "passed")

    def test_partial_effect_and_missing_migration_observation_block_completion(self):
        self.results["criteria"]["W0"]["H3_transport"]["effect"]["per_repetition"].pop()
        self.assertNotEqual(self.comparison_gate()["status"], "passed")
        self.setUp()
        del next(row for row in self.results["arms"] if row["arm"] == "MR")["migrations_committed"]
        self.assertNotEqual(self.comparison_gate()["status"], "passed")

    def test_measured_zero_migration_control_is_not_a_missing_run(self):
        from hetero.analysis import evaluate_criteria

        controls = [row for row in self.results["arms"] if row["workload"] == "W1"]
        for row in controls:
            row["migrations_committed"] = 0
        self.results["criteria"]["W1"] = evaluate_criteria(controls, minimum_effect=0.05)
        self.assertEqual(self.results["criteria"]["W1"]["H2_migration"]["status"],
                         "not_exercised_no_migrations")
        self.assertEqual(self.comparison_gate()["status"], "passed")

    def test_missing_partial_and_wrong_cohort_mechanism_evidence_block(self):
        for selected in ({}, {"artifact": "unused.json", "report": {"passed": True}},
                         {"artifact": "unused.json", "report": {"passed": True,
                          "evaluation_cohort": {"predeclaration_sha256": "historical"}}}):
            self.results["mechanism_cost"] = selected
            result = goal_audit.build(self.results, {})
            gate = next(row for row in result["acceptance_gates"] if row["gate"].startswith("5."))
            self.assertNotEqual(gate["status"], "passed")
            self.assertFalse(result["goal_complete"])

    def test_hash_mismatch_and_summary_only_qualification_never_pass(self):
        import hashlib
        import json
        from hetero.engine_actor import EngineSettings
        from hetero.native_backend import BACKEND_REVISION, RUNTIME_REVISION

        raw = json.dumps({"status": "SUCCEEDED", "source_digest": {"source": "digest"},
                          "measurements": {"ok": True, "gate": {"gate_passed": {"passed": True}}}}).encode()
        self.protocol["source_freeze"]["source_sha256"] = {"source": "digest"}
        self.protocol["qualified_runtime"] = {
            "backend_revision": BACKEND_REVISION, "runtime_revision": RUNTIME_REVISION,
            "engine_settings": EngineSettings().to_dict(),
            "evidence": {"gate_H": {"path": "results/in_memory_only.json", "file_sha256": "wrong"}}}
        with patch.object(goal_audit.pathlib.Path, "read_bytes", return_value=raw):
            for digest in ("wrong", hashlib.sha256(raw).hexdigest()):
                self.protocol["qualified_runtime"]["evidence"]["gate_H"]["file_sha256"] = digest
                result = goal_audit.build(self.results, {})
                self.assertNotEqual(result["acceptance_gates"][0]["status"], "passed")
                self.assertFalse(result["goal_complete"])


if __name__ == "__main__":
    unittest.main()

