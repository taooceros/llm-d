"""Qualification decision edges, not model-output or performance simulation."""

import copy
import json
import math
import pathlib
import tempfile
import unittest

from hetero.qualify_continuation import (
    PROTOCOL_PATH,
    _compare_diagnostics,
    _matched_pair,
    _snapshot_contract,
    _suffix_comparison,
    run_continuation_qualification,
)


class ContinuationQualificationDecisions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(pathlib.Path(PROTOCOL_PATH).read_text())
        cls.criteria = cls.protocol["cross_configuration"]["criteria"]

    def diagnostic(self, tokens=range(32), shift=0.0):
        return {"token_id": 0, "top_logprobs": {token: -5.0 + shift for token in tokens}}

    def test_prospective_top32_intersection_boundary(self):
        left = self.diagnostic()
        at_boundary = self.diagnostic(list(range(29)) + [100, 101, 102])
        below = self.diagnostic(list(range(28)) + [100, 101, 102, 103])
        self.assertTrue(_compare_diagnostics(left, at_boundary, self.criteria)["passed"])
        self.assertFalse(_compare_diagnostics(left, below, self.criteria)["passed"])

    def test_logprob_tolerance_is_not_rounded_to_pass(self):
        left = self.diagnostic()
        self.assertTrue(_compare_diagnostics(left, self.diagnostic(shift=0.25), self.criteria)["passed"])
        right = self.diagnostic()
        right["top_logprobs"][0] = math.nextafter(-4.75, math.inf)
        self.assertFalse(_compare_diagnostics(left, right, self.criteria)["passed"])

    def test_nonfinite_nonshared_value_cannot_disappear_from_report(self):
        right = self.diagnostic()
        right["top_logprobs"][1000] = math.nan
        comparison = _compare_diagnostics(self.diagnostic(), right, self.criteria)
        self.assertFalse(comparison["passed"])
        self.assertFalse(comparison["all_reported_values_finite"])
        self.assertIsNone(comparison["shared_top32_max_absolute_log_probability_difference"])

    def test_same_full_history_does_not_allow_changed_prompt_boundary(self):
        case = {"prompt_token_ids": [10, 11], "generated_history_tokens": 2, "original_max_tokens": 34}
        snapshot = {"prompt_token_ids": [10, 11], "generated_token_ids": [12, 13],
                    "prompt_len": 2, "generated_len": 2, "num_computed_tokens": 3,
                    "sampling": {"max_tokens": 34, "temperature": 0, "ignore_eos": True},
                    "remaining_tokens": 32, "unsupported_continuation": [], "snapshot_at_s": 0}
        reference = [12, 13] + list(range(32))
        self.assertTrue(_snapshot_contract(snapshot, case, reference)["passed"])
        moved_boundary = copy.deepcopy(snapshot)
        moved_boundary.update(prompt_token_ids=[10, 11, 12], generated_token_ids=[13],
                              prompt_len=3, generated_len=1)
        self.assertFalse(_snapshot_contract(moved_boundary, case, reference)["passed"])
        overshot = copy.deepcopy(snapshot)
        overshot.update(generated_token_ids=[12, 13, 14], generated_len=3,
                        num_computed_tokens=4, remaining_tokens=31)
        self.assertFalse(_snapshot_contract(overshot, case, reference)["checks"]["exact_generated_checkpoint"])

    def test_suffix_report_keeps_every_divergence_and_missing_token(self):
        result = _suffix_comparison(list(range(32)), [99] + list(range(1, 30)) + [98])
        self.assertFalse(result["passed"])
        self.assertEqual([row["offset"] for row in result["divergences"]], [0, 30, 31])
        self.assertIsNone(result["divergences"][-1]["observed_token_id"])

    def test_nonidentical_source_history_is_not_a_transport_comparison(self):
        snapshot = {"prompt_token_ids": [1], "generated_token_ids": [2], "sampling": {},
                    "num_computed_tokens": 1, "remaining_tokens": 32}
        left = {"snapshot": snapshot, "passed": True}
        right = copy.deepcopy(left)
        right["snapshot"]["generated_token_ids"] = [3]
        report = _matched_pair(left, right, self.criteria)
        self.assertFalse(report["identical_histories_and_contract"])
        self.assertFalse(report["transport_attribution_resolved"])
        self.assertNotIn("prepared_first_step_logprobs", report)

    def test_actual_host_suffix_divergence_is_unresolved_even_when_logits_pass(self):
        snapshot = {"prompt_token_ids": [1], "generated_token_ids": [2], "sampling": {},
                    "num_computed_tokens": 1, "remaining_tokens": 32}
        left = {"snapshot": snapshot, "passed": True, "first_step_diagnostic": self.diagnostic(),
                "result": {"new_output_token_ids": list(range(32))}}
        right = copy.deepcopy(left)
        right["result"]["new_output_token_ids"][17] = 999
        report = _matched_pair(left, right, self.criteria)
        self.assertTrue(report["prepared_first_step_logprobs"]["passed"])
        self.assertFalse(report["passed"])
        self.assertFalse(report["transport_attribution_resolved"])
        self.assertEqual(report["host_suffix_vs_recompute"]["first_divergence_offset"], 17)

    def test_changed_protocol_fails_before_any_model_execution_and_saves_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            changed = root / "changed.json"
            changed.write_text(pathlib.Path(PROTOCOL_PATH).read_text() + "\n")
            report = run_continuation_qualification(protocol_path=str(changed), artifact_root=str(root))
            self.assertFalse(report["passed"])
            self.assertFalse(report["hardware_model_execution_started"])
            persisted = json.loads(pathlib.Path(report["report_artifact"]["path"]).read_text())
            self.assertFalse(persisted["passed_flags"]["prospective_protocol_unchanged"])
            self.assertEqual(persisted["cases"], [])


if __name__ == "__main__":
    unittest.main()
