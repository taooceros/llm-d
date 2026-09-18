"""Analysis and report rendering for the heterogeneous-serving evaluation.

Consumes the arm summaries produced by :mod:`hetero.experiment` and renders the
comparison tables exactly as predeclared: paired makespan differences by
repetition, makespan reduction as ``1 - T_X/T_B`` and throughput increase as
``T_B/T_X - 1``, never interchanged.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import pathlib
import statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_arm_results(paths: Iterable[pathlib.Path]) -> List[Dict[str, Any]]:
    """Collects every arm summary from saved control-plane result files."""
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        measurements = payload.get("measurements") or {}
        stack = [measurements]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if "arm" in node and "makespan_s" in node:
                    rows.append(dict(node, result_artifact=str(path)))
                    continue
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    rows.sort(key=lambda r: (r["workload"], r["arm"], r["repetition"]))
    return rows


def group_by_workload(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["workload"], []).append(row)
    return grouped


def arm_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-arm aggregates over repetitions, reporting every run."""
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    table = []
    for arm, runs in sorted(by_arm.items()):
        makespans = [r["makespan_s"] for r in runs]
        throughputs = [r["useful_tokens_per_s"] for r in runs]
        table.append(
            {
                "arm": arm,
                "layout": runs[0].get("layout_id"),
                "repetitions": len(runs),
                "complete": all(r["all_requests_complete"] for r in runs),
                "useful_output_tokens": runs[0]["useful_output_tokens"],
                "makespan_s": [round(m, 2) for m in makespans],
                "makespan_median_s": round(statistics.median(makespans), 2),
                "throughput_tok_s": [round(t, 1) for t in throughputs],
                "throughput_median_tok_s": round(statistics.median(throughputs), 1),
                "tok_s_per_chip": round(
                    statistics.median(
                        [r["useful_tokens_per_s_per_chip"] for r in runs]
                    ),
                    2,
                ),
                "migrations_committed": [r["migrations_committed"] for r in runs],
                "migrations_declined": [r["migrations_declined"] for r in runs],
                "replayed_token_fraction": [
                    r["replayed_token_fraction"] for r in runs
                ],
                "preemptions": [r["total_preemptions"] for r in runs],
            }
        )
    return table

def eligibility(row: Dict[str, Any]) -> Dict[str, Any]:
    """Missing evidence is not a pass; keep every exclusion visible."""
    reasons = []
    if not row.get("ingress_url") or row.get("ingress_protocol") != "envoy-ext-proc-v3":
        reasons.append("not_qualified_Envoy_ExtProc_ingress")
    for field in ("all_requests_complete", "iteration_capture_one_to_one",
                  "client_delivery_matches", "committed_prefix_preserved",
                  "native_output_budget_exact"):
        if row.get(field) is not True:
            reasons.append(field + "_failed_or_missing")
    expected = row.get("expected_output_tokens")
    if not expected or row.get("useful_output_tokens") != expected:
        reasons.append("useful_token_budget_not_exact")
    makespan = row.get("makespan_s")
    if not isinstance(makespan, (int, float)) or not math.isfinite(makespan) or makespan <= 0:
        reasons.append("invalid_makespan")
    if row.get("arm") in ("MR", "MK-H") and row.get("continuation_qualification_passed") is not True:
        reasons.append("continuation_qualification_failed_or_missing")
    reasons.extend(row.get("protocol_exclusions", []))
    if row.get("iteration_evidence", {}).get("status") not in (None, "available"):
        reasons.append("iteration_artifact_missing_or_invalid")
    return {"headline_eligible": not reasons, "reasons": reasons}


def _qualified_run(row: Dict[str, Any]) -> bool:
    return eligibility(row)["headline_eligible"]


def paired_effect(
    rows: List[Dict[str, Any]], baseline: str, treatment: str
) -> Optional[Dict[str, Any]]:
    """Paired per-repetition contrast; ``None`` when the pair is unavailable."""
    compared = [r for r in rows if r["arm"] in (baseline, treatment)]
    if any(not _qualified_run(r) for r in compared):
        return None
    base = {r["repetition"]: r for r in compared if r["arm"] == baseline}
    treat = {r["repetition"]: r for r in compared if r["arm"] == treatment}
    if len(base) + len(treat) != len(compared):
        raise ValueError("duplicate arm/repetition measurements; select one frozen evaluation")
    shared = sorted(set(base) & set(treat))
    if not shared:
        return None
    per_rep = []
    progress_pairs = []
    for rep in shared:
        if (base[rep]["manifest_sha256"] != treat[rep]["manifest_sha256"]
                or base[rep]["useful_output_tokens"] != treat[rep]["useful_output_tokens"]):
            raise ValueError("paired arms do not share exact manifest and useful-token budget")
        tb = base[rep]["makespan_s"]
        tx = treat[rep]["makespan_s"]
        per_rep.append(
            {
                "repetition": rep,
                "baseline_s": round(tb, 2),
                "treatment_s": round(tx, 2),
                "makespan_reduction": round(1 - tx / tb, 4),
                "throughput_increase": round(tb / tx - 1, 4),
            }
        )
        bp, tp = base[rep].get("progress", {}), treat[rep].get("progress", {})
        br, tr = bp.get("deep_backlog_native_tokens_per_s"), tp.get("deep_backlog_native_tokens_per_s")
        if (bp.get("available") and tp.get("available")
                and bp.get("backlog_boundaries_completed_requests") == tp.get("backlog_boundaries_completed_requests")
                and bp.get("boundary_definition") == tp.get("boundary_definition")
                and isinstance(br, (int, float)) and isinstance(tr, (int, float))
                and math.isfinite(br) and math.isfinite(tr) and br > 0 and tr > 0):
            progress_pairs.append({"repetition": rep, "native_throughput_increase": tr / br - 1,
                                   "baseline_terminal_drain_s": bp.get("terminal_drain_s"),
                                   "treatment_terminal_drain_s": tp.get("terminal_drain_s")})
    reductions = [p["makespan_reduction"] for p in per_rep]
    return {
        "baseline": baseline,
        "treatment": treatment,
        "pairs": len(per_rep),
        "per_repetition": per_rep,
        "progress_pairs": progress_pairs,
        "sustained_progress_status": ("measured_common_interval" if len(progress_pairs) == len(per_rep)
                                      else "missing_common_interval_no_sustained_claim"),
        "makespan_reduction_median": round(statistics.median(reductions), 4),
        "makespan_reduction_range": [round(min(reductions), 4), round(max(reductions), 4)],
        "all_same_sign": all(r > 0 for r in reductions) or all(r < 0 for r in reductions),
    }


def strongest_homogeneous(rows: List[Dict[str, Any]]) -> Optional[str]:
    """The uniform layout with the lowest median makespan, if any ran."""
    candidates = {}
    for arm in ("H", "S"):
        runs = [r for r in rows if r["arm"] == arm]
        if len(runs) >= 3 and all(_qualified_run(r) for r in runs):
            candidates[arm] = statistics.median(r["makespan_s"] for r in runs)
    if not candidates:
        return None
    return min(candidates, key=candidates.get)


def evaluate_criteria(
    rows: List[Dict[str, Any]], *, minimum_effect: float
) -> Dict[str, Any]:
    """Applies the predeclared usefulness criteria without reinterpreting them."""
    best_homogeneous = strongest_homogeneous(rows)
    out: Dict[str, Any] = {
        "strongest_homogeneous_arm": best_homogeneous,
        "minimum_useful_effect": minimum_effect,
    }
    frozen = "MR"  # Prospectively selected, never chosen from observed winners.
    out["frozen_treatment"] = frozen

    def verdict(effect: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if effect is None:
            return {"status": "not_measured_or_not_qualified"}
        if effect["pairs"] < 3:
            return {"status": "preliminary_fewer_than_three_pairs", "effect": effect}
        median = effect["makespan_reduction_median"]
        if median >= minimum_effect and effect["all_same_sign"]:
            status = "supported"
        elif median > 0:
            status = "measured_but_below_threshold"
        else:
            status = "not_supported"
        return {"status": status, "effect": effect}

    if best_homogeneous:
        out["H1_instance_mix"] = verdict(paired_effect(rows, best_homogeneous, "M0"))
    else:
        out["H1_instance_mix"] = {"status": "no_qualified_homogeneous_reference"}
    out["H2_migration"] = verdict(paired_effect(rows, "M0", "MR"))
    out["S_vs_H"] = verdict(paired_effect(rows, "H", "S"))
    for reference in ("H", "S"):
        out[f"M0_vs_{reference}"] = verdict(paired_effect(rows, reference, "M0"))
        out[f"MR_vs_{reference}"] = verdict(paired_effect(rows, reference, frozen))
    out["H3_transport"] = verdict(paired_effect(rows, "MR", "MK-H"))
    if best_homogeneous and frozen:
        out["H4_system"] = verdict(paired_effect(rows, best_homogeneous, frozen))
    else:
        out["H4_system"] = {"status": "not_measured"}

    migrated = [
        r["migrations_committed"] for r in rows if r["arm"] in ("MR", "MK-H")
    ]
    out["no_actual_migrations"] = bool(migrated) and all(m == 0 for m in migrated)
    if out["no_actual_migrations"]:
        out["H2_migration"]["status"] = "not_exercised_no_migrations"
    kv_runs = [r for r in rows if r["arm"] == "MK-H"]
    if kv_runs and not any(r.get("transferred_bytes", 0) for r in kv_runs):
        out["H3_transport"]["status"] = "not_exercised_no_transferred_KV"
    out["unqualified_runs"] = [
        {"arm": r["arm"], "repetition": r["repetition"], "reasons": eligibility(r)["reasons"]}
        for r in rows if not _qualified_run(r)
    ]
    out["incomplete_arms"] = sorted(
        {r["arm"] for r in rows if not r["all_requests_complete"]}
    )
    return out


def _artifact_path(remote: str, root: pathlib.Path) -> pathlib.Path:
    marker = "/artifacts/"
    if marker in remote:
        return root / "results/hetero/artifacts" / remote.split(marker, 1)[1]
    return root / remote


def run_accounting(row: Dict[str, Any], root: pathlib.Path) -> Dict[str, Any]:
    """Audit retained iteration rows, without reconstructing missing measurements."""
    stats = row.get("iteration_stats", {})
    evidence: Dict[str, Any] = {"status": "missing", "rows": 0, "eligible_rows": 0,
                                "control_rows": 0, "invalid_rows": 0, "per_instance_phase": {}}
    remote = stats.get("path")
    if remote:
        path = _artifact_path(remote, root)
        evidence["path"] = str(path)
        if path.exists():
            evidence["status"] = "available"
            with gzip.open(path, "rt", newline="") as handle:
                for item in csv.DictReader(handle):
                    evidence["rows"] += 1
                    phase = item.get("phase")
                    if phase == "control":
                        evidence["control_rows"] += 1
                        continue
                    try:
                        latency = float(item["scheduled_to_output_s"])
                        completed = float(item["completed_at_s"])
                        scheduled = float(item["scheduled_at_s"])
                        valid = (phase in ("prefill", "mixed", "decode")
                                 and all(math.isfinite(v) for v in (latency, completed, scheduled))
                                 and latency > 0 and completed >= scheduled)
                    except (KeyError, TypeError, ValueError):
                        valid = False
                    if not valid:
                        evidence["invalid_rows"] += 1
                        continue
                    evidence["eligible_rows"] += 1
                    key = item["instance_id"] + "/" + phase
                    group = evidence["per_instance_phase"].setdefault(key, {
                        "rows": 0, "host_latency_s_sum": 0.0, "emitted_tokens": 0,
                        "scheduled_requests_max": 0, "context_tokens_max": 0})
                    group["rows"] += 1
                    group["host_latency_s_sum"] += latency
                    group["emitted_tokens"] += int(item.get("emitted_output_tokens") or 0)
                    for field in ("scheduled_requests", "context_tokens"):
                        group[field + "_max"] = max(group[field + "_max"], int(item.get(field) or 0))
            if (evidence["invalid_rows"] or evidence["rows"] != stats.get("rows")
                    or not stats.get("one_to_one")):
                evidence["status"] = "invalid_or_unqualified_retirement"
    evidence["scope"] = "Host schedule-to-retirement rows, not device timings or independent experimental replicates. Control rows are not compute timing samples. Raw CSV retains each individual row."
    progress = row.get("progress") or {"available": False, "reason": "missing_measurement"}
    initialization = row.get("initialization_s")
    counters = row.get("counters", {})
    return {"iteration_evidence": evidence, "progress": progress,
            "initialization_s": initialization,
            "initialization_plus_serving_s": initialization + row["makespan_s"] if initialization is not None else None,
            "migration_coverage": {"eligible_requests": row.get("migration_eligible_requests"),
                "committed": row.get("migrations_committed"), "declined": row.get("migrations_declined"),
                "fallbacks": row.get("migration_fallbacks"), "gateway_counters": counters,
                "note": "Unknown eligibility/fallback counts remain null; committed counts alone are not coverage."},
            "transfer_accounting": {"actual_payload_bytes": row.get("actual_payload_bytes"),
                "network_bytes": row.get("network_bytes"), "legacy_reported_transferred_bytes": row.get("transferred_bytes"),
                "scope": "Actual payload requires native export/import evidence. Payload excludes RPC serialization, replication and network overhead; it is not measured network traffic. Legacy byte fields are retained without upgrading formula estimates to measurements."}}


def render_markdown(
    *, grouped: Dict[str, List[Dict[str, Any]]], criteria: Dict[str, Dict[str, Any]],
    predeclaration: Dict[str, Any], context: Dict[str, Any],
) -> str:
    lines = ["# Static heterogeneous bulk inference behind llm-d: empirical report", ""]
    add = lines.append
    rows = [row for group in grouped.values() for row in group]
    eligible = [row for row in rows if _qualified_run(row)]
    add("## Status: missing measurements / qualification unresolved")
    add("")
    add(f"{len(eligible)} eligible headline runs; {len(rows) - len(eligible)} retained diagnostic or unqualified runs. "
        "Engineering functionality, a working mechanism, workload-specific speedup and system advantage are distinct claims. "
        "This report does not declare the goal complete.")
    add("")
    add(f"Prospective protocol: \x60{predeclaration.get('predeclaration_sha256', 'missing')}\x60. "
        "A local source snapshot is not a qualified loaded-source freeze.")
    for key, value in context.items():
        add(f"- **{key}**: {value}")
    add("")
    add("Missing data is reported as missing, never zero. Diagnostic timings below are not headline evidence. "
        "Replay/duplicates do not count as useful output; all allocated chips are charged. "
        "Iteration clocks are host observations, not device/kernel timings or independent repetitions.")
    add("")
    for workload, runs in sorted(grouped.items()):
        add(f"## Workload {workload}")
        if not runs:
            add("No retained measurements.")
        add("| Arm | Rep | Makespan s | Useful tokens | Useful tok/s | Useful tok/s/allocated chip | Eligible | Exclusions |")
        add("|---|---:|---:|---:|---:|---:|---|---|")
        for row in sorted(runs, key=lambda r: (r["arm"], r["repetition"])):
            audit = eligibility(row)
            add(f"| {row['arm']} | {row['repetition']} | {row.get('makespan_s')} | "
                f"{row.get('useful_output_tokens')} | {row.get('useful_tokens_per_s')} | {row.get('useful_tokens_per_s_per_chip', 'missing')} | "
                f"{audit['headline_eligible']} | {', '.join(audit['reasons']) or 'none'} |")
        add("")
        add("### Per-run accounting and iteration eligibility")
        add("| Arm/rep | Capture rows / eligible / control / invalid | Retirement/artifact status | Deep-backlog tok/s | Terminal drain s / fraction | Initialization s / init+serving s | Migrations committed / declined / eligible / fallback | Actual payload / network bytes |")
        add("|---|---|---|---|---|---|---|---|")
        for row in runs:
            capture = row.get("iteration_evidence", {})
            progress = row.get("progress", {})
            migration = row.get("migration_coverage", {})
            transfer = row.get("transfer_accounting", {})
            def value(mapping, key):
                result = mapping.get(key)
                return "missing" if result is None else str(result)
            add(f"| {row['arm']}/{row['repetition']} | "
                + " / ".join(value(capture, k) for k in ("rows", "eligible_rows", "control_rows", "invalid_rows"))
                + f" | {capture.get('status', 'missing')} | {value(progress, 'deep_backlog_native_tokens_per_s')} | "
                + " / ".join(value(progress, k) for k in ("terminal_drain_s", "terminal_drain_fraction"))
                + " | " + " / ".join(value(row, k) for k in ("initialization_s", "initialization_plus_serving_s"))
                + " | " + " / ".join(value(migration, k) for k in ("committed", "declined", "eligible_requests", "fallbacks"))
                + " | " + " / ".join(value(transfer, k) for k in ("actual_payload_bytes", "network_bytes")) + " |")
        add("")
        for row in runs:
            capture = row.get("iteration_evidence", {})
            progress = row.get("progress", {})
            add(f"- {row['arm']}/{row['repetition']} source: `{row.get('result_artifact', 'missing')}`; "
                f"individual iterations: `{capture.get('path', 'missing')}`; "
                f"sustained scope: {progress.get('boundary_definition', progress.get('reason', 'missing'))}.")
        add("")
        add("Payload bytes and network bytes are distinct. No formula-priced transfer or inferred network volume is accepted. "
            "Per-instance/phase capture aggregates, raw CSV paths, gateway counters and legacy byte fields are retained in RESULTS.json. "
            "Unknown coverage is not zero coverage. Initialization is not borrowed from an unrelated deployment or summed across overlapping worker initializations.")
        add("")
        add("### Paired contrasts")
        for name, entry in criteria.get(workload, {}).items():
            if not isinstance(entry, dict) or "status" not in entry:
                continue
            effect = entry.get("effect")
            add(f"- **{name}**: {entry['status']}")
            if effect:
                add(f"  - Makespan reduction median {effect['makespan_reduction_median']:+.2%}; "
                    f"range {effect['makespan_reduction_range']}; {effect['pairs']} whole-run pairs.")
                add(f"  - Sustained progress: {effect.get('sustained_progress_status', 'missing')}; "
                    f"paired native throughput and terminal drain: {effect.get('progress_pairs', [])}.")
                for pair in effect["per_repetition"]:
                    add(f"  - Rep {pair['repetition']}: baseline {pair['baseline_s']} s; treatment {pair['treatment_s']} s; "
                        f"makespan reduction {pair['makespan_reduction']:+.2%}; throughput increase {pair['throughput_increase']:+.2%}.")
        add("")
    return "\n".join(lines) + "\n"

