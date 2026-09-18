#!/usr/bin/env python3
"""Render one selected research cohort without erasing retained historical runs."""
from __future__ import annotations

import argparse

import json

import pathlib
import statistics
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hetero.analysis import eligibility, evaluate_criteria, group_by_workload, load_arm_results, render_markdown, run_accounting
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_efficiency
import goal_audit

RESULTS = ROOT / "results/hetero"


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _relative(path: pathlib.Path) -> str:
    return str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path)


def _retained_rows(paths: list[pathlib.Path]) -> list[dict]:
    rows = load_arm_results(paths)
    # A wrapper and its artifact summary are copies, not two measurements.
    seen = {(r.get("predeclaration_sha256"), r.get("run_id"), r.get("workload"), r.get("arm"),
             r.get("repetition"), r.get("request_id_namespace")) for r in rows}
    for path in sorted((RESULTS / "artifacts").glob("*/*_summary.json")):
        row = _load(path)
        if "arm" not in row or "makespan_s" not in row:
            continue
        key = (row.get("predeclaration_sha256"), row.get("run_id"), row.get("workload"), row.get("arm"),
               row.get("repetition"), row.get("request_id_namespace"))
        if key not in seen:
            seen.add(key)
            rows.append(dict(row, result_artifact=_relative(path)))
    # Some retained control wrappers are themselves exact copies.
    unique = {}
    for row in rows:
        key = (row.get("predeclaration_sha256"), row.get("run_id"), row.get("workload"), row.get("arm"),
               row.get("repetition"), row.get("request_id_namespace"))
        if key not in unique:
            unique[key] = row
        else:
            unique[key].setdefault("retained_copy_artifacts", []).append(row["result_artifact"])
    return sorted(unique.values(), key=lambda r: (r["workload"], r["arm"], r["repetition"]))








def _mechanism_cost(wrappers: list[tuple[pathlib.Path, dict]], sha: str) -> tuple[dict | None, list[str]]:
    matches = [(path, wrapper["measurements"]["mechanism_cost"]) for path, wrapper in wrappers
               if wrapper.get("measurements", {}).get("mechanism_cost", {}).get("evaluation_cohort", {}).get("predeclaration_sha256") == sha]
    reports = {json.dumps(report, sort_keys=True) for _, report in matches}
    if len(reports) != 1:
        return None, [_relative(path) for path, _ in matches]
    path, report = matches[0]
    return {"artifact": _relative(path), "report": report}, [_relative(path) for path, _ in matches]


def _completion(criteria: dict, expected: list[tuple], qualified: list[dict]) -> dict:
    complete = {}
    admitted = {(row["workload"], row["arm"], row["repetition"]) for row in qualified}
    for workload, results in criteria.items():
        expected_reps = {}
        for work, arm, rep in expected:
            if work == workload:
                expected_reps.setdefault(arm, set()).add(rep)
        best = results.get("strongest_homogeneous_arm")
        fixed = {"S_vs_H": ("H", "S"), "H2_migration": ("M0", "MR"),
                 "H3_transport": ("MR", "MK-H"), "H1_instance_mix": (best, "M0"),
                 "H4_system": (best, "MR"), "M0_vs_H": ("H", "M0"),
                 "M0_vs_S": ("S", "M0"), "MR_vs_H": ("H", "MR"), "MR_vs_S": ("S", "MR")}
        for name, entry in results.items():
            if not isinstance(entry, dict) or "status" not in entry:
                continue
            effect = entry.get("effect") or {}
            baseline, treatment = fixed.get(name, (effect.get("baseline"), effect.get("treatment")))
            reference_missing = []
            if name in ("H1_instance_mix", "H4_system"):
                reference_missing = [{"arm": arm, "repetition": rep} for arm in ("H", "S")
                                     for rep in sorted(expected_reps.get(arm, set()))
                                     if (workload, arm, rep) not in admitted]
            base_reps = expected_reps.get(baseline, set())
            if baseline is None and name in ("H1_instance_mix", "H4_system"):
                base_reps = expected_reps.get("H", set()) & expected_reps.get("S", set())
            reps = base_reps & expected_reps.get(treatment, set())
            observed = {pair["repetition"] for pair in effect.get("per_repetition", [])}
            status = ("not_declared" if not reps else "complete" if observed == reps and not reference_missing
                      else "partial" if observed else "missing")
            complete[f"{workload}/{name}"] = {"expected_repetitions": sorted(reps),
                "observed_repetitions": sorted(observed), "missing_repetitions": sorted(reps - observed),
                "missing_homogeneous_reference_runs": reference_missing, "status": status}
    return complete


def _cost_markdown(cost: dict | None, candidates: list[str]) -> str:
    text = "\n## Selected-cohort matched-history mechanism costs\n\n"
    if cost is None:
        return text + (f"Missing or ambiguous matching wrapper; candidates: {candidates}. No timing is inferred from qualification or rejected submissions.\n")
    report = cost["report"]
    trials, pairs = report.get("trials", []), report.get("pairs", [])
    text += (f"Artifact: `{cost['artifact']}`. Passed: {report.get('passed')}; "
             f"{sum(t.get('passed') is True for t in trials)}/{len(trials)} retained trials passed; "
             f"{sum(p.get('passed') is True for p in pairs)}/{len(pairs)} matched pairs passed. "
             f"Errors: {report.get('errors', [])}. Scope: {report.get('evidence_scope', 'missing')}.\n\n")
    text += "| Trial | Status / passed | Proposal to source-release ACK s | First destination token clock envelope s | Native prepare s | Replayed positions | Logical payload bytes | Wire bytes |\n|---|---|---:|---|---:|---:|---:|---|\n"
    for trial in trials:
        m = trial.get("metrics", {})
        text += (f"| {trial.get('trial_id')} | {trial.get('status')} / {trial.get('passed')} | "
                 f"{m.get('proposal_to_source_release_ack_s')} | {m.get('proposal_to_first_destination_token_clock_envelope_s')} | "
                 f"{m.get('native_prepare_s')} | {m.get('actual_native_replayed_positions')} | "
                 f"{m.get('logical_payload_bytes')} | {m.get('total_wire_traffic_bytes', 'not measured')} |\n")
    text += "\n| Case / repetition | Matched history / background | Pair passed | KV-host minus recompute source-release ACK s |\n|---|---|---|---:|\n"
    by_id = {trial.get("trial_id"): trial for trial in trials}
    for pair in pairs:
        members = [by_id[key] for key in pair.get("trial_ids", []) if key in by_id]
        durations = {trial.get("mechanism"): trial.get("metrics", {}).get("proposal_to_source_release_ack_s") for trial in members}
        delta = None
        if pair.get("passed") is True and all(isinstance(durations.get(key), (int, float)) for key in ("kv_host", "recompute")):
            delta = durations["kv_host"] - durations["recompute"]
        text += (f"| {pair.get('case_id')} / {pair.get('repetition')} | "
                 f"{pair.get('matched_real_prompt_history_and_budget')} / {pair.get('byte_identical_background_vectors_and_budgets')} | "
                 f"{pair.get('passed')} | {delta} |\n")
    text += ("\nThese are direct-engine paired diagnostics, not gateway makespan or system speedups. Host clocks, "
             "RPC round trips, observer overhead and background overlap remain charged. First-token envelopes assume no "
             "unobserved clock step; raw cross-host differences are not pure transport latency. Native export includes "
             "drain/gather/host assembly; replay iterations may mix background work. Local control CPU is not remote CPU. "
             "Logical array payload excludes serialization and replicated collective/RPC wire traffic; unknown network "
             "bytes remain null. Full phase/CPU/RPC clocks, export/import receipts, replay, destination admission, "
             "background progress, trial errors and cleanup are retained verbatim in RESULTS.json mechanism_cost.report.\n")
    return text


def _gateway_overhead(rows):
    """Per-request gateway cost from retained records; never summed into makespan."""
    import gzip
    import statistics

    summary = []
    for row in sorted(rows, key=lambda item: (item.get("workload", ""), item.get("arm", ""),
                                              item.get("repetition", 0))):
        directory = row.get("artifact_dir")
        if not directory:
            continue
        # Summaries record the remote staging directory; artifacts are fetched
        # under results/hetero/artifacts/<run-id>.
        local = RESULTS / "artifacts" / pathlib.Path(directory).name
        prefix = f"{row.get('workload')}_{row.get('arm')}_r{row.get('repetition')}"
        path = local / f"{prefix}_gateway_records.json.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt") as stream:
            records = json.load(stream)
        waits, delivery, ttft = [], [], []
        for record in records:
            wait = record.get("admission_wait_s")
            if wait is not None:
                wait = float(wait)
                waits.append(wait)
            segments = record.get("segments") or []
            if segments and record.get("admitted_at_s") is not None:
                started = segments[0].get("started_at_s")
                if started is not None and wait is not None:
                    # Residual after gateway-local selection: cross-host RPC and
                    # ingress delivery up to engine API entry, plus clock skew.
                    delivery.append(float(started) - float(record["admitted_at_s"]) - wait)
                if segments[0].get("ttft_s") is not None:
                    ttft.append(float(segments[0]["ttft_s"]))
        if not delivery:
            continue
        summary.append({
            "workload": row.get("workload"), "arm": row.get("arm"),
            "repetition": row.get("repetition"), "requests": len(records),
            "admission_wait_median_s": round(statistics.median(waits), 6) if waits else None,
            "admission_wait_max_s": round(max(waits), 6) if waits else None,
            "delivery_residual_median_s": round(statistics.median(delivery), 6),
            "delivery_residual_min_s": round(min(delivery), 6),
            "delivery_residual_max_s": round(max(delivery), 6),
            "ttft_median_s": round(statistics.median(ttft), 6) if ttft else None,
            "makespan_s": row.get("makespan_s"),
        })
    return summary

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=RESULTS)
    parser.add_argument("--protocol", type=pathlib.Path,
                        default=RESULTS / "evaluation_amendment_20260916_source_frozen.json")
    args = parser.parse_args()
    protocol = _load(args.protocol)
    sha = protocol["predeclaration_sha256"]
    qualification = goal_audit._qualification_status(protocol)
    minimum = protocol["minimum_useful_effect"]["threshold_fraction"]
    paths = sorted(RESULTS.glob("*/*.json"))
    wrappers = [(path, _load(path)) for path in paths]
    initialization = analyze_efficiency.startup_evidence(wrappers)
    all_rows = _retained_rows(paths)
    expected = analyze_efficiency.declared_runs(protocol)
    counts = Counter((r.get("workload"), r.get("arm"), r.get("repetition")) for r in all_rows
                     if r.get("predeclaration_sha256") == sha)
    for row in all_rows:
        exclusions = list(row.get("protocol_exclusions", []))
        selected = row.get("predeclaration_sha256") == sha
        if selected and row.get("arm") in ("MR", "MK-H"):
            continuation = qualification.get("continuation", {})
            row["continuation_qualification_passed"] = continuation.get("status") == "passed"
            row["continuation_qualification_provenance"] = {
                "predeclaration_sha256": sha, "validation": continuation,
                "evidence": protocol.get("qualified_runtime", {}).get("evidence", {}).get("continuation")} 
        if not selected:
            exclusions.append("not_in_selected_frozen_evaluation_cohort")
        else:
            key = (row["workload"], row["arm"], row["repetition"])
            if key not in expected:
                exclusions.append("not_in_declared_run_schedule")
            if counts[key] != 1:
                exclusions.append("ambiguous_duplicate_scheduled_run")
            if protocol.get("source_freeze", {}).get("status") != "qualified":
                exclusions.append("matching_postfreeze_source_qualification_pending")
            if row.get("source_freeze_commit_id") != protocol.get("source_freeze", {}).get("commit_id"):
                exclusions.append("source_freeze_commit_mismatch")
        row["protocol_exclusions"] = sorted(set(exclusions))
        row.update(analyze_efficiency.initialization_accounting(row, initialization))
        row.update(run_accounting(row, ROOT))
        row["eligibility"] = eligibility(row)
    cohort = [row for row in all_rows if row.get("predeclaration_sha256") == sha]
    qualified = [row for row in cohort if row["eligibility"]["headline_eligible"]]
    grouped = group_by_workload(cohort)
    for work, _, _ in expected:
        grouped.setdefault(work, [])
    # Ambiguous repeats remain excluded and visible, not selected opportunistically.
    criteria = {name: evaluate_criteria(rows, minimum_effect=minimum) for name, rows in grouped.items()}
    completion = _completion(criteria, expected, qualified)
    efficiency = analyze_efficiency.build(protocol, cohort, initialization=initialization)
    cost, cost_candidates = _mechanism_cost(wrappers, sha)
    history = [row for row in all_rows if row.get("predeclaration_sha256") != sha]
    census = Counter(row.get("predeclaration_sha256") or "unfrozen" for row in history)
    observed_keys = {(row["workload"], row["arm"], row["repetition"]) for row in qualified}
    missing = [{"workload": work, "arm": arm, "repetition": rep} for work, arm, rep in expected
               if (work, arm, rep) not in observed_keys]
    context = {"selected source freeze": protocol.get("source_freeze", {}).get("status", "missing"),
               "declared scheduled runs": len(expected), "selected retained runs": len(cohort),
               "eligible scheduled runs": len(qualified), "missing or excluded scheduled runs": len(missing),
               "retained other-cohort history": f"{len(history)} runs, never pooled into selected effects",
               "minimum useful makespan reduction": minimum,
               "qualification binding": protocol.get("qualified_runtime", "not retained in selected protocol")}
    # Keep hetero.analysis verdicts intact in JSON; visibly scope partial verdicts.
    display_criteria = json.loads(json.dumps(criteria))
    for work, results in display_criteria.items():
        for name, entry in results.items():
            scope = completion.get(f"{work}/{name}")
            if scope and scope["status"] != "complete" and entry.get("effect"):
                entry["status"] = f"partial_declared_schedule; observed-pair verdict: {entry['status']}"
    markdown = render_markdown(grouped=grouped, criteria=display_criteria, predeclaration=protocol, context=context)
    # The common renderer intentionally never declares completion; replace its
    # static qualification headline with the selected observation scope.
    markdown = markdown.replace("## Status: missing measurements / qualification unresolved",
                                "## Status: selected-cohort evidence (acceptance audit below)")
    markdown += "\n## Research questions and matched effects\n\n"
    markdown += ("All conclusions below concern only the selected cohort. Partial pairs describe observations, "
                 "not completion of the prospective schedule. Effects and every paired repetition are reported above; "
                 "H4 uses the frozen MR treatment, never a retrospectively chosen winner.\n\n")
    markdown += "### 1. In which regimes are smaller instances more efficient?\n\n"
    markdown += "| Region | Pairs / expected | Status | H useful tok/s/chip | S useful tok/s/chip | S-vs-H makespan reduction median [range] |\n|---|---|---|---:|---:|---|\n"
    for work, entry in efficiency["contrasts"].items():
        effect = entry.get("effect") or {}
        markdown += (f"| {work} | {entry['paired_repetitions']} / {len(entry['expected_repetitions'])} | {entry['status']} | "
                     f"{entry['H']['tokens_per_s_per_chip_median']} | {entry['S']['tokens_per_s_per_chip_median']} | "
                     f"{effect.get('makespan_reduction_median')} {effect.get('makespan_reduction_range')} |\n")
    markdown += f"\n{efficiency['scope']}\n\n"
    markdown += "| Run | Scheduled batch median/max | Context tokens max | Free KV blocks min | Preemptions |\n|---|---|---:|---:|---:|\n"
    for row in efficiency["runs"]:
        profile = row.get("iteration_profile", {})
        markdown += (f"| {row['label']} | {profile.get('scheduled_batch_median')} / {profile.get('scheduled_batch_max')} | "
                     f"{profile.get('context_tokens_max')} | {profile.get('free_kv_blocks_min')} | {profile.get('preemptions')} |\n")
    for title, names in (("2. Does migrating growing requests beat leaving them in place?", ("H1_instance_mix", "H2_migration", "H4_system")),
                         ("3. Is rebuilding destination KV better or worse than transferring it?", ("H3_transport",))):
        markdown += f"\n### {title}\n\n| Workload | Contrast | Verdict | Declared-pair coverage |\n|---|---|---|---|\n"
        for work, results in criteria.items():
            if work in analyze_efficiency.WORKLOADS:
                continue
            for name in names:
                scope = completion.get(f"{work}/{name}", {})
                markdown += (f"| {work} | {name} | {results.get(name, {}).get('status', 'missing')} | "
                             f"{scope.get('status')}; observed {scope.get('observed_repetitions')}; missing {scope.get('missing_repetitions')} |\n")
    markdown += ("\nPositive makespan reduction favors the treatment; throughput increase is a different denominator. "
                 "Migration claims require actual eligible/committed coverage, not merely an enabled arm. "
                 "Whole-run and terminal-drain gains are not sustained deep-backlog gains without common measured boundaries. "
                 "The per-run tables and raw iteration paths above preserve this distinction. Optional ICI remains "
                 "separate: host fallback is never relabelled ICI.\n")
    markdown += _cost_markdown(cost, cost_candidates)
    overhead = _gateway_overhead(cohort)
    markdown += "\n## Gateway overhead in the selected cohort\n\nAdmission wait is a single gateway-clock selection/queue interval. Delivery residual spans gateway-to-engine API entry across host clocks (RPC, queueing and clock skew), not isolated ExtProc or native scheduling cost. First-segment TTFT is engine-observed. These overlapping distributions are not summed into makespan; migration coverage is reported separately per run. Missing observations are not zero.\n\n"
    markdown += "| Workload / arm / rep | Requests | Admission wait median/max s | Delivery residual median [min,max] s | First-segment TTFT median s |\n|---|---:|---|---|---:|\n"
    for row in overhead:
        markdown += (f"| {row['workload']}/{row['arm']}/{row['repetition']} | {row['requests']} | "
                     f"{row['admission_wait_median_s']} / {row['admission_wait_max_s']} | "
                     f"{row['delivery_residual_median_s']} [{row['delivery_residual_min_s']}, {row['delivery_residual_max_s']}] | "
                     f"{row['ttft_median_s']} |\n")
    markdown += "\n## Initialization provenance\n\n"
    markdown += ("Initialization uses only an exact layout/generation-map/settings/runtime match to an all-created-actor "
                 "deployment or gate.deployment wrapper. The coordinator engine_init_s interval already spans concurrent "
                 "initialization; worker init times are never summed. Idempotent attach clocks, partially reused deployments, "
                 "and ambiguous startups are not accepted. Per-instance generation values need not be identical. "
                 "Initialization-plus-serving is the sum of noncontiguous measured components for a reused generation, "
                 "not an actual wall interval or amortized initialization.\n\n")
    for row in cohort:
        markdown += f"- {row['workload']}/{row['arm']}/r{row['repetition']}: {json.dumps(row['initialization_provenance'], sort_keys=True)}\n"
    markdown += "\n## Missing observations, failures and exclusions\n\n"
    markdown += f"Missing or excluded scheduled tuples: `{json.dumps(missing)}`.\n\n"
    for row in cohort:
        if not row["eligibility"]["headline_eligible"]:
            markdown += f"- {row['workload']}/{row['arm']}/r{row['repetition']}: {row['eligibility']['reasons']}; `{row.get('result_artifact')}`.\n"
    markdown += "\n## Retained history (not selected-cohort conclusions)\n\n"
    markdown += "All retained run rows, including failed and superseded cohorts, remain in RESULTS.json arms. Original evidence and previous output are not overwritten when --output-dir selects a new report directory.\n\n"
    for historical_sha, count in sorted(census.items()):
        markdown += f"- `{historical_sha}`: {count} retained runs.\n"
    report = {"status": "selected_cohort_evidence_pending_acceptance_audit", "goal_complete": False,
              "protocol": _relative(args.protocol), "predeclaration": protocol, "arms": all_rows,
              "eligible_headline_runs": len(qualified), "criteria": criteria, "contrast_completion": completion,
              "missing_or_excluded_scheduled_runs": missing, "direct_ray_diagnostics": [r for r in all_rows if not r.get("ingress_url")],
              "retained_history": {"cohort_run_counts": dict(census), "runs": len(history)},
              "initialization_evidence": initialization, "mechanism_cost": cost,
              "mechanism_cost_candidates": cost_candidates, "efficiency_analysis": efficiency}
    audit = goal_audit.build(report, efficiency)
    report["goal_audit"] = audit
    report["qualification_validation"] = qualification
    report["gateway_overhead"] = overhead
    report["goal_complete"] = audit["goal_complete"]
    report["status"] = "acceptance_audit_passed" if audit["goal_complete"] else "acceptance_evidence_incomplete_or_failed"
    enabled = [row for row in cohort if row.get("policy", {}).get("migration_enabled") is True]
    observed_proposals = [row.get("counters", {}).get("proposals") for row in enabled]
    observed_commits = [row.get("migrations_committed") for row in enabled]
    coverage = {"enabled_runs": len(enabled),
                "proposals": sum(observed_proposals) if enabled and all(isinstance(n, int) for n in observed_proposals) else None,
                "committed": sum(observed_commits) if enabled and all(isinstance(n, int) for n in observed_commits) else None}
    report["observed_migration_coverage"] = coverage
    summary = ("## Research conclusion\n\n"
               f"Acceptance audit: **{report["status"]}**; {len(qualified)}/{len(expected)} declared runs eligible. "
               "Completion of the experiment does not imply a useful system speedup.\n\n"
               "**Small-instance regimes.** Paired S-versus-H makespan reductions at equal 32-chip allocation:\n\n"
               "| Workload | Median reduction | Six-run range or available-pair range | Coverage |\n|---|---:|---|---|\n")
    for work, values in criteria.items():
        effect = values.get("S_vs_H", {}).get("effect")
        if effect:
            low, high = effect["makespan_reduction_range"]
            summary += f"| {work} | {effect["makespan_reduction_median"]:+.2%} | [{low:+.2%}, {high:+.2%}] | {completion.get(work + '/S_vs_H', {}).get('status')} |\n"
    summary += ("\n**Frozen mixed-layout policy.** Positive reduction favors MR; negative means slower than the strongest homogeneous reference.\n\n"
                "| Workload | Strongest homogeneous | MR makespan reduction | Range |\n|---|---|---:|---|\n")
    for work, values in criteria.items():
        effect = values.get("H4_system", {}).get("effect")
        if effect:
            low, high = effect["makespan_reduction_range"]
            summary += f"| {work} | {effect["baseline"]} | {effect["makespan_reduction_median"]:+.2%} | [{low:+.2%}, {high:+.2%}] |\n"
    summary += f"\nEnabled-arm observations: {coverage}. Unknown eligibility counts are not interpreted as zero. "
    if coverage["enabled_runs"] and coverage["proposals"] == coverage["committed"] == 0:
        summary += ("**The frozen policy made no migration proposals.** These end-to-end contrasts do not identify "
                    "a migration or KV-transfer benefit: they measure the policy in a regime where neither ran. "
                    "This limitation also applies to the held-out growing workload; no thresholds were retuned. "
                    "Actual migrations were exercised separately by qualification and the matched mechanism group. ")
    report["mechanism_cost_summary"] = []
    if cost and not goal_audit._mechanism_errors(cost, protocol):
        summary += ("\n\n**Recompute versus host KV under fixed real background load.** Median first-destination-token clock envelopes "
                    "include the measured handoff path, not only transfer kernels. They are not end-to-end gateway makespans.\n\n"
                    "| Checkpoint | Recompute median envelope s | Host-KV median envelope s |\n|---|---|---|\n")
        trials = cost["report"]["trials"]
        for case in dict.fromkeys(row["case_id"] for row in trials):
            row = {"case_id": case}
            for mode in ("recompute", "kv_host"):
                intervals = [trial["metrics"]["proposal_to_first_destination_token_clock_envelope_s"]
                             for trial in trials if trial["case_id"] == case and trial["mechanism"] == mode]
                row[mode] = [statistics.median(interval[i] for interval in intervals) for i in (0, 1)]
            report["mechanism_cost_summary"].append(row)
            summary += f"| {case} | {row["recompute"]} | {row["kv_host"]} |\n"
        summary += "\nHost transport is not a guaranteed optimization; the full trial/pair evidence and unmeasured wire-byte limitations appear below.\n"
    for path, receipt in wrappers:
        if receipt.get("predeclaration_sha256") != sha:
            continue
        pressure = receipt.get("pressure_observations", [])
        if pressure and len(pressure) == coverage["enabled_runs"]:
            report["campaign_outcome"] = {"artifact": _relative(path), "report": receipt}
            lowest = min(free / item["qualified_tp8_usable_blocks"] for item in pressure
                         for free in item["recorded_source_free_blocks_min"].values())
            thresholds = sorted({item["threshold_fraction"] for item in pressure})
            summary += (f"\nRecorded active-source free-block fractions never fell below {lowest:.6f}; "
                        f"the configured pressure threshold(s) were {thresholds}. This is observed scheduler evidence, "
                        f"not an inferred eligibility count or proof of the admission mechanism causing the result. "
                        f"Per-run minima and artifact hashes: `{_relative(path)}`.\n")
        if "discovery_label_cleanup" in receipt:
            report["final_cleanup_receipt"] = {"artifact": _relative(path), "report": receipt}
            summary += (f"\nFinal engine/group inspections and generation-scoped discovery cleanup: `{_relative(path)}`. "
                        f"Remaining selected pool endpoints: {receipt["discovery_label_cleanup"]["observed_items"]}.\n")
    markdown = markdown.replace("This report does not declare the goal complete.",
                                "The selected-cohort acceptance audit is reported below.")
    title, rest = markdown.split("\n", 1)
    markdown = title + "\n\n" + summary + "\n" + rest
    markdown += f"\n## Goal acceptance audit\n\nGoal complete: **{audit["goal_complete"]}**. Blocking reason: {audit["blocking_reason"]}\n\n"
    markdown += "| Acceptance gate | Status | Evidence |\n|---|---|---|\n"
    for entry in audit["acceptance_gates"]:
        markdown += f"| {entry['gate']} | {entry['status']} | {', '.join(str(item) for item in entry['evidence'])} |\n"
    markdown += "\n"
    for entry in audit["acceptance_gates"]:
        markdown += f"- **{entry['gate']}**: {entry['note']}\n"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("RESULTS.json", report), ("goal_audit.json", audit), ("efficiency_analysis.json", efficiency)):
        (args.output_dir / name).write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "REPORT.md").write_text(markdown)
    print(json.dumps({"protocol": report["protocol"], "selected_runs": len(cohort), "eligible_runs": len(qualified),
                      "expected_runs": len(expected), "historical_runs": len(history), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
