#!/usr/bin/env python3
"""Audit selected-cohort acceptance gates against retained physical evidence."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/hetero"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hetero.analysis import eligibility
from scripts import freeze_predeclaration as freeze


def _load(rel: str) -> Dict[str, Any]:
    path = RESULTS / rel
    return json.loads(path.read_text()) if path.exists() else {}


def _measurements(rel: str) -> Dict[str, Any]:
    return _load(rel).get("measurements", {})


def _protocol(results: dict) -> dict:
    if isinstance(results.get("predeclaration"), dict):
        return results["predeclaration"]
    if results.get("protocol"):
        path = pathlib.Path(results["protocol"])
        if not path.is_absolute():
            path = ROOT / path
        return json.loads(path.read_text()) if path.exists() else {}
    return _load("evaluation_amendment_20260916_source_frozen.json") or _load("predeclaration.json")


def _artifact(ref: dict) -> dict:
    path = ROOT / ref["path"]
    freeze._require(not pathlib.Path(ref["path"]).is_absolute()
                    and path.resolve().is_relative_to(ROOT), "evidence must be repository-relative")
    raw = path.read_bytes()
    freeze._require(hashlib.sha256(raw).hexdigest() == ref["file_sha256"], "evidence file hash differs")
    return json.loads(raw)


def _lifecycle(report: dict) -> None:
    required = {"gateway_actor_preserved", "peer_actors_preserved", "peer_bindings_preserved",
                "unselected_groups_preserved", "new_target_actor", "new_target_generation",
                "fresh_gateway_binding", "fresh_discovery_binding", "no_duplicate_groups",
                "no_migration_or_owner_left", "full_static_fabric_restored"}
    freeze._require(report["passed"] is True and not report.get("error")
                    and not report.get("finalization_error") and report["layout_id"] == "M",
                    "lifecycle failed or wrong layout")
    freeze._require(required <= report["checks"].keys(), "incomplete lifecycle checks")
    freeze._all_passed(report["checks"], "lifecycle invariants failed")
    names = {row["instance_id"] for row in report["registry_before"]}
    freeze._require(names == {"small_a", "small_b", "large"} and bool(report["ingress_url"])
                    and report["restart_pending_at_probe"] is True, "missing full-fabric restart evidence")
    for stage in ("before", "withdrawn", "restarting", "ready_but_excluded", "restored"):
        probe = report["stages"][stage]
        expected = names if stage in ("before", "restored") else names - {report["instance_id"]}
        records = probe["client_records"]
        freeze._require(probe["passed"] is True and set(probe["expected_instances"]) == expected
                        and set(probe["observed_instances"]) == expected
                        and len(records) == report["requests_per_stage"] > 0
                        and {row["initial_instance"] for row in records} == expected
                        and all(not row.get("error") and row["useful_output_tokens"] == report["max_tokens"] > 0
                                and len(row["segments"]) == 1
                                and row["segments"][0]["instance_id"] in expected for row in records),
                        f"missing or failed real lifecycle probe: {stage}")


def _qualification_status(protocol: dict) -> dict:
    """Use the freeze's raw validators, not an existence-only certificate."""
    bundle = protocol.get("qualified_runtime", {})
    evidence = bundle.get("evidence", {})
    statuses = {}
    for key in freeze.EVIDENCE_KEYS:
        ref = evidence.get(key, {})
        try:
            from hetero.engine_actor import EngineSettings
            from hetero.native_backend import BACKEND_REVISION, RUNTIME_REVISION

            freeze._require(bundle["backend_revision"] == BACKEND_REVISION
                            and bundle["runtime_revision"] == RUNTIME_REVISION,
                            "qualified backend/runtime revision differs")
            settings = bundle["engine_settings"]
            freeze._require(EngineSettings.from_dict(settings).to_dict() == settings,
                            "incomplete qualified EngineSettings")
            source = protocol["source_freeze"]
            freeze._require(source["status"] == "qualified" and bool(source["source_sha256"]),
                            "source freeze not qualified")
            wrapper = _artifact(ref)
            deployed = {name: digest for name, digest in wrapper["source_digest"].items()
                        if not name.startswith("input:")}
            freeze._require(wrapper["status"] == "SUCCEEDED" and deployed == source["source_sha256"],
                            "unsuccessful wrapper or wrong frozen source")
            measurement = wrapper["measurements"]
            freeze._require(measurement["ok"] is True, "qualification command failed")
            if key.startswith("gate_"):
                freeze._gate(measurement["gate"], key[-1], settings)
            elif key == "lifecycle":
                _lifecycle(measurement["qualification"])
            elif key == "continuation":
                numerical = protocol["numerical_qualification"]
                _artifact(numerical)
                freeze._require(wrapper["source_digest"].get("input:OMP_HETERO_PROTOCOL_JSON")
                                == numerical["file_sha256"], "wrong continuation protocol")
                freeze._continuation(measurement["qualification"], numerical, numerical["file_sha256"])
            else:
                freeze._migration(measurement, key)
            statuses[key] = {"status": "passed", "artifact": ref["path"]}
        except (KeyError, TypeError, ValueError, OSError, AttributeError) as exc:
            statuses[key] = {"status": "missing_or_invalid_qualification", "artifact": ref.get("path"),
                             "reason": f"{type(exc).__name__}: {exc}"}
    return statuses


_COST_FLAGS = {"prospective_protocol_unchanged", "qualified_still_resident_native_paths",
               "held_out_evaluation_manifests_unchanged", "all_three_authentic_references",
               "all_18_declared_trials_passed", "all_nine_ab_ba_ab_pairs_matched",
               "acknowledged_cleanup_and_resident_preservation"}
_TRIAL_FLAGS = {"native_transfer_and_original_contract", "ordinary_path_without_top32",
                "all_background_admitted_and_generating", "all_background_fixed_budgets_completed",
                "background_not_exhausted_during_handoff", "unrelated_small_b_useful_progress_during_handoff",
                "observed_background_progress_consistent", "native_replay_frontier_audited",
                "all_native_iterations_strictly_retired", "useful_tokens_match_all_captured_iterations",
                "native_pages_idle_and_ownership_resolved"}


def _mechanism_errors(selected: dict, protocol: dict) -> list[str]:
    try:
        freeze._require(bool(selected.get("artifact")), "missing mechanism-cost artifact")
        report = selected["report"]
        freeze._require(bool(protocol.get("predeclaration_sha256"))
                        and report["evaluation_cohort"]["predeclaration_sha256"] == protocol["predeclaration_sha256"],
                        "mechanism-cost report belongs to a different cohort")
        freeze._require(report["passed"] is True and not report["errors"]
                        and report["hardware_model_execution_started"] is True, "mechanism-cost execution failed")
        freeze._require(_COST_FLAGS <= report["passed_flags"].keys(), "incomplete mechanism-cost flags")
        freeze._all_passed(report["passed_flags"], "mechanism-cost flag failed")
        trials = report["trials"]
        expected = {(f"p{prompt}-h{history}", repetition, mechanism)
                    for prompt, history in ((256, 64), (1024, 257), (4096, 257))
                    for repetition in (1, 2, 3) for mechanism in ("recompute", "kv_host")}
        freeze._require(len(trials) == 18 and
                        {(row["case_id"], row["repetition"], row["mechanism"]) for row in trials} == expected,
                        "missing or duplicate declared mechanism-cost trials")
        for row in trials:
            freeze._require(row["passed"] is True and row["status"] == "completed" and not row["errors"]
                            and _TRIAL_FLAGS <= row["passed_flags"].keys(), "incomplete or failed mechanism-cost trial")
            freeze._all_passed(row["passed_flags"], "mechanism-cost trial flag failed")
        pairs = report["pairs"]
        freeze._require(len(pairs) == 9 and {(row["case_id"], row["repetition"]) for row in pairs}
                        == {(case, rep) for case, rep, _ in expected}
                        and all(row["passed"] is True for row in pairs), "missing or failed mechanism pairs")
        references = report["references"]
        freeze._require(len(references) == 3 and {row["case_id"] for row in references}
                        == {case for case, _, _ in expected}
                        and all(row["passed"] is True and row["native_idle"]["passed"] is True
                                and row["capture"]["passed"] is True for row in references), "failed authentic references")
        cleanup = report["cleanup"]
        freeze._require(cleanup["passed"] is True and not cleanup["unresolved"] and not cleanup.get("error")
                        and cleanup["resident_instances_preserved"] is True
                        and cleanup["all_native_pages_reclaimed"] is True
                        and set(cleanup["idle"]) == set(cleanup["native_state"]) == {"small_a", "small_b", "large"}
                        and all(row["idle"] is True and not row["draining"] for row in cleanup["idle"].values())
                        and all(not row["quiesced"] and not row["prepared"] and not row["active_continuations"]
                                for row in cleanup["native_state"].values()), "mechanism-cost cleanup unresolved")
        return []
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return [f"{type(exc).__name__}: {exc}"]


def _comparison_evidence(results: dict, protocol: dict) -> dict:
    declared = [(row["workload"], arm, row["repetition"])
                for row in protocol.get("arm_order", []) for arm in row["arms"]]
    cohort_sha = protocol.get("predeclaration_sha256")
    cohort = [row for row in results.get("arms", [])
              if cohort_sha and row.get("predeclaration_sha256") == cohort_sha]
    admitted = Counter()
    for row in cohort:
        if not eligibility(row)["headline_eligible"] or row.get("eligibility", {}).get("headline_eligible") is False:
            continue
        migrations = row.get("migrations_committed")
        if not isinstance(migrations, int) or isinstance(migrations, bool) or migrations < 0:
            continue
        if row.get("arm") == "MK-H":
            payload = row.get("transferred_bytes")
            if not isinstance(payload, int) or isinstance(payload, bool) or payload < 0:
                continue
            if migrations > 0 and payload <= 0:
                continue
        admitted[row["workload"], row["arm"], row["repetition"]] += 1
    missing = [list(key) for key in declared if admitted[key] == 0]
    duplicate = [list(key) for key in declared if admitted[key] > 1]
    required = {}
    for workload in dict.fromkeys(key[0] for key in declared):
        arms = {arm for name, arm, _ in declared if name == workload}
        names = []
        if {"H", "S"} <= arms:
            names.append("S_vs_H")
        if "M0" in arms:
            names.append("H1_instance_mix")
        if "MR" in arms:
            names.extend(("H2_migration", "H4_system"))
        if "MK-H" in arms:
            names.append("H3_transport")
        required[workload] = names
    unresolved = []
    measured_statuses = {"supported", "not_supported", "measured_but_below_threshold",
                         "not_exercised_no_migrations", "not_exercised_no_transferred_KV"}
    for workload, names in required.items():
        for name in names:
            criterion = results.get("criteria", {}).get(workload, {}).get(name, {})
            effect = criterion.get("effect") or {}
            expected_pairs = {rep for name_, arm, rep in declared
                              if name_ == workload and arm == effect.get("baseline")} & {
                                  rep for name_, arm, rep in declared
                                  if name_ == workload and arm == effect.get("treatment")}
            if (criterion.get("status") not in measured_statuses or not expected_pairs
                    or effect.get("pairs") != len(expected_pairs)
                    or [row.get("repetition") for row in effect.get("per_repetition", [])] != sorted(expected_pairs)):
                unresolved.append([workload, name])
    cost_errors = _mechanism_errors(results.get("mechanism_cost") or {}, protocol)
    passed = (bool(declared) and len(set(declared)) == len(declared) and bool(cohort_sha)
              and protocol.get("source_freeze", {}).get("status") == "qualified"
              and not missing and not duplicate and not unresolved and not cost_errors)
    return {"passed": passed, "declared_runs": len(declared), "eligible_declared_runs": sum(admitted[key] == 1 for key in declared),
            "missing_or_ineligible": missing, "duplicate_eligible": duplicate,
            "required_criteria": required, "unresolved_criteria": unresolved, "mechanism_cost_errors": cost_errors}


def build(results: Dict[str, Any], efficiency: Dict[str, Any]) -> Dict[str, Any]:
    """Audit the selected protocol; negative measured effects are valid results."""
    protocol = _protocol(results)
    qualification = _qualification_status(protocol)
    comparisons = _comparison_evidence(results, protocol)
    final_inspections = {
        layout: _measurements(f"20260918/permanent_fp32_final_inspect_{layout}.json").get("inspection", {})
        for layout in ("H", "S", "M")
    }
    residual = {}
    for layout, inspection in final_inspections.items():
        wrapper = _load(f"20260918/permanent_fp32_final_inspect_{layout}.json")
        if (wrapper.get("status") != "SUCCEEDED" or wrapper.get("measurements", {}).get("ok") is not True
                or inspection.get("layout_id") != layout or not all(isinstance(inspection.get(key), list)
                for key in ("registry", "observed", "owned_placement_groups"))
                or any(not isinstance(row.get("actor_alive"), bool) for row in inspection["observed"])
                or any(not row.get("state") for row in inspection["owned_placement_groups"])):
            continue
        residual[layout] = {
            "registered_instances": len(inspection["registry"]),
            "live_engine_actors": sum(row["actor_alive"] for row in inspection["observed"]),
            "created_owned_placement_groups": sum(row["state"] == "CREATED" for row in inspection["owned_placement_groups"]),
            "pending_owned_placement_groups": sum(row["state"] == "PENDING" for row in inspection["owned_placement_groups"]),
        }

    def gate(title: str, keys: tuple[str, ...], note: str) -> dict:
        return {"gate": title, "status": "passed" if all(qualification[key]["status"] == "passed" for key in keys)
                else "missing_or_invalid_qualification",
                "evidence": [qualification[key]["artifact"] for key in keys if qualification[key].get("artifact")],
                "checks": {key: qualification[key] for key in keys}, "note": note}

    rows = [
        gate("1. Contiguous subslice allocation and TPU runtime bring-up", ("gate_H", "gate_S", "gate_M"),
             "All three layouts require hash-bound raw gates, native precision, concurrent serving and frozen source checks."),
        gate("2. Repeatable deployment and Ray serving-instance discovery", ("lifecycle",),
             "Full-fabric ExtProc probes, scoped restart, peer preservation and fresh-generation readmission must pass."),
        gate("3. Continuation and ownership qualified with recompute", ("continuation", "lifecycle", "recompute"),
             "Unchanged numerical criteria and actual gateway recompute handoffs; earlier diagnostics do not authorize this cohort."),
        gate("4. Host-staged KV transfer qualified; ICI optional", ("continuation", "kv_host"),
             "Actual released host-KV migrations, positive payload bytes, exact output accounting and ownership ordering; no ICI claim."),
        {"gate": "5. Predeclared research comparisons", "status": "passed" if comparisons["passed"] else "missing_or_invalid_comparisons",
         "evidence": [path for path in (results.get("protocol"), (results.get("mechanism_cost") or {}).get("artifact")) if path],
         "checks": comparisons, "note": f"{comparisons['eligible_declared_runs']}/{comparisons['declared_runs']} exact scheduled tuples eligible. "
         "Every scheduled contrast and the matched 18-trial mechanism-cost group are required; a positive speedup is not required."},
        {"gate": "6. Conclude and restore owned state", "status": "missing_inspection_evidence" if len(residual) != 3 else (
            "passed" if all(all(value == 0 for value in row.values()) for row in residual.values()) else "residual_state"),
         "evidence": [f"results/hetero/20260918/permanent_fp32_final_inspect_{layout}.json" for layout in ("H", "S", "M")],
         "note": f"Residual owned state per layout: {residual}."},
    ]
    labels = {"H1_instance_mix": "Instance-mix benefit", "H2_migration": "Incremental migration benefit",
              "H4_system": "System benefit", "H3_transport": "Transport benefit", "S_vs_H": "Equal-resource homogeneous comparison"}
    usefulness = []
    for name, label in labels.items():
        selected = {workload: results.get("criteria", {}).get(workload, {}).get(name, {})
                    for workload, names in comparisons["required_criteria"].items() if name in names}
        verdicts = {workload: criterion.get("status", "missing") for workload, criterion in selected.items()}
        usefulness.append({"criterion": label, "verdict": verdicts,
                           "measured_effect": {workload: criterion.get("effect") for workload, criterion in selected.items()},
                           "conclusion": "; ".join(f"{workload}: {verdict}" for workload, verdict in verdicts.items())
                                         or "Not scheduled in the selected protocol."})
    blocking = [row["gate"] for row in rows if row["status"] != "passed"]
    return {
        "goal_document": "HETEROGENEOUS_LLMD_GATEWAY_GOAL.md",
        "predeclaration_sha256": protocol.get("predeclaration_sha256"),
        "goal_complete": not blocking, "blocking_reason": "; ".join(blocking) if blocking else None,
        "acceptance_gates": rows, "usefulness_criteria": usefulness,
        "historical_diagnostics": {
            "original_bf16_state": {"artifact": "results/hetero/20260916/continuation_fidelity_M_dispatch_repaired.json",
                "passed": _measurements("20260916/continuation_fidelity_M_dispatch_repaired.json").get("qualification", {}).get("passed")},
            "first_passing_fp32_state": {"artifact": "results/hetero/20260918/minimal_fp32_state_continuation.json",
                "passed": _measurements("20260918/minimal_fp32_state_continuation.json").get("qualification", {}).get("passed")},
        },
        "efficiency_contrasts": efficiency.get("contrasts", {}) if efficiency.get("predeclaration_sha256")
                                == protocol.get("predeclaration_sha256") else {},
    }


def main() -> int:
    audit = build(_load("RESULTS.json"), _load("efficiency_analysis.json"))
    out = RESULTS / "goal_audit.json"
    out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"gates": [(row["gate"][:2], row["status"]) for row in audit["acceptance_gates"]],
                      "output": str(out.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
