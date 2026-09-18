#!/usr/bin/env python3
"""Analyze declared H/S efficiency pairs from one selected evaluation cohort."""
from __future__ import annotations

import argparse
import hashlib
import math
import csv
import gzip
import json
import pathlib
import statistics
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hetero.analysis import eligibility, paired_effect, run_accounting

WORKLOADS = ("E_SHORT", "E_LONG")
ARMS = ("H", "S")


def declared_runs(protocol: dict) -> list[tuple[str, str, int]]:
    """Use the frozen schedule, including its workload-specific repetition counts."""
    return [(item["workload"], arm, item["repetition"])
            for item in protocol.get("arm_order", []) for arm in item["arms"]]


def _iteration_profile(path: pathlib.Path) -> dict:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    batch, tokens, contexts, free = [], [], [], []
    phases: dict[str, int] = {}
    preemptions = 0
    with gzip.open(path, "rt") as stream:
        for row in csv.DictReader(stream):
            try:
                scheduled = int(row["scheduled_requests"])
                tok = int(row["scheduled_tokens"])
                context = int(row["context_tokens"])
                blocks = int(row["free_kv_blocks_after_schedule"])
                preemptions += int(row["num_preempted"])
            except (KeyError, ValueError):
                return {"status": "invalid", "path": str(path)}
            if scheduled <= 0:
                continue
            batch.append(scheduled)
            tokens.append(tok)
            contexts.append(context)
            free.append(blocks)
            phase = row.get("phase", "unknown")
            phases[phase] = phases.get(phase, 0) + 1
    if not batch:
        return {"status": "empty", "path": str(path)}
    return {"status": "ok", "path": str(path), "iterations": len(batch),
            "scheduled_batch_median": statistics.median(batch), "scheduled_batch_max": max(batch),
            "scheduled_tokens_median": statistics.median(tokens), "context_tokens_max": max(contexts),
            "free_kv_blocks_min": min(free), "phase_counts": phases, "preemptions": preemptions}


def startup_evidence(wrappers: list[tuple[pathlib.Path, dict]]) -> list[dict]:
    evidence = []
    for path, wrapper in wrappers:
        measurements = wrapper.get("measurements", {})
        deployment = measurements.get("deployment") or measurements.get("gate", {}).get("deployment")
        if not deployment:
            continue
        instances = deployment.get("instances", [])
        duration = deployment.get("stage_clocks", {}).get("engine_init_s")
        settings = deployment.get("desired_configuration", {}).get("engine_settings")
        descriptors = [item.get("descriptor", {}) for item in instances]
        generations = {item.get("instance_id"): item.get("generation") for item in descriptors}
        created = bool(instances) and all(item.get("attached_existing_actor") is False for item in instances)
        valid = (created and isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0
                 and bool(settings) and None not in generations and None not in generations.values()
                 and len(generations) == len(instances)
                 and all(item.get("initialization", {}).get("status") == "initialized" for item in instances))
        evidence.append({"artifact": str(path.resolve().relative_to(ROOT)), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "layout_id": deployment.get("layout_id"), "layout_digest": deployment.get("layout_digest"),
                         "generations": generations, "engine_settings": settings,
                         "model_builds": {item.get("instance_id"): item.get("model_build") for item in descriptors},
                         "all_actors_created": created, "usable": valid,
                         "engine_init_s": duration,
                         "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization"})
    return evidence


def initialization_accounting(row: dict, evidence: list[dict]) -> dict:
    loaded = row.get("loaded_runtime", {})
    layout = loaded.get("layout", {})
    identities = {name: value.get("identity", {}) for name, value in loaded.get("instances", {}).items()}
    matches = []
    if identities:
        generations = {name: item.get("generation") for name, item in identities.items()}
        for item in evidence:
            if (not item["usable"] or item["layout_id"] != layout.get("layout_id")
                    or item["layout_digest"] != layout.get("layout_digest") or item["generations"] != generations):
                continue
            settings = item["engine_settings"]
            if all(identity.get("instance_id") == name and identity.get("layout_id") == item["layout_id"]
                   and identity.get("layout_digest") == item["layout_digest"]
                   and identity.get("engine_settings") == settings
                   and identity.get("backend_revision") == settings.get("backend_revision")
                   and settings.get("runtime_revision")
                   and bool(identity.get("model_build")) and identity["model_build"] == item["model_builds"].get(name)
                   for name, identity in identities.items()):
                matches.append(item)
    # Multiple copies of exactly the same startup observation are harmless;
    # differing coordinator clocks for the same generation are ambiguous.
    clocks = {item["engine_init_s"] for item in matches}
    if len(clocks) != 1:
        return {"initialization_s": None, "initialization_provenance": {
            "status": "ambiguous" if matches else "missing",
            "reason": "No unique all-created-actor startup with exact layout, generation map, settings and model/runtime build",
            "matching_artifacts": [item["artifact"] for item in matches]}}
    return {"initialization_s": matches[0]["engine_init_s"], "initialization_provenance": {
        "status": "matched", "artifacts": [{"path": item["artifact"], "file_sha256": item["file_sha256"]} for item in matches],
        "layout_id": matches[0]["layout_id"], "generations": matches[0]["generations"],
        "clock": matches[0]["clock"],
        "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization."}}

def build(protocol: dict, rows: list[dict], *, initialization: list[dict]) -> dict:
    """Report missing and excluded scheduled observations, never impute them."""
    freeze_sha = protocol["predeclaration_sha256"]
    expected = [key for key in declared_runs(protocol) if key[0] in WORKLOADS and key[1] in ARMS]
    admitted, excluded = [], []
    for workload, arm, rep in expected:
        label = f"{workload}_{arm}_r{rep}"
        candidates = [row for row in rows if (row.get("workload"), row.get("arm"), row.get("repetition"))
                      == (workload, arm, rep) and row.get("predeclaration_sha256") == freeze_sha]
        if len(candidates) != 1:
            excluded.append({"label": label, "workload": workload, "arm": arm, "repetition": rep,
                             "reasons": ["summary artifact absent" if not candidates else "ambiguous duplicate scheduled run"],
                             "artifacts": [row.get("result_artifact") for row in candidates]})
            continue
        row = dict(candidates[0])
        row.update(initialization_accounting(row, initialization))
        row.update(run_accounting(row, ROOT))
        reasons = eligibility(row)["reasons"]
        if protocol.get("source_freeze", {}).get("status") != "qualified":
            reasons.append("matching_postfreeze_source_qualification_pending")
        if not row.get("source_freeze_commit_id"):
            reasons.append("run carries no source-freeze commit id")
        elif row["source_freeze_commit_id"] != protocol.get("source_freeze", {}).get("commit_id"):
            reasons.append("source-freeze commit differs from selected protocol")
        for key in ("migrations_committed", "replayed_tokens"):
            if row.get(key) != 0:
                reasons.append(f"homogeneous efficiency {key} is nonzero or missing")
        row["label"] = label
        if reasons:
            row["reasons"] = sorted(set(reasons))
            excluded.append(row)
        else:
            path = row["iteration_evidence"].get("path")
            row["iteration_profile"] = (_iteration_profile(pathlib.Path(path)) if path
                                        else {"status": "missing"})
            admitted.append(row)
    contrasts = {}
    for workload in WORKLOADS:
        region = [row for row in admitted if row["workload"] == workload]
        effect = paired_effect(region, "H", "S")
        h_reps = {rep for work, arm, rep in expected if work == workload and arm == "H"}
        s_reps = {rep for work, arm, rep in expected if work == workload and arm == "S"}
        expected_reps = sorted(h_reps & s_reps)
        paired = [entry["repetition"] for entry in (effect or {}).get("per_repetition", [])]
        entry: dict[str, Any] = {"workload": workload, "expected_repetitions": expected_reps,
                                "paired_repetitions": len(paired), "missing_repetitions": sorted(set(expected_reps) - set(paired)),
                                "status": ("complete" if expected_reps and paired == expected_reps else "partial" if paired else "missing"),
                                "effect": effect, "per_repetition": (effect or {}).get("per_repetition", [])}
        for arm in ARMS:
            vals = [row["makespan_s"] for row in region if row["arm"] == arm]
            chip = [row.get("useful_tokens_per_s_per_chip") for row in region if row["arm"] == arm]
            entry[arm] = {"runs": len(vals), "makespan_median_s": statistics.median(vals) if vals else None,
                          "makespan_min_s": min(vals) if vals else None, "makespan_max_s": max(vals) if vals else None,
                          "makespan_stdev_s": statistics.stdev(vals) if len(vals) > 1 else None,
                          "tokens_per_s_per_chip_median": (statistics.median(chip) if chip and all(v is not None for v in chip) else None)}
        if effect:
            median = effect["makespan_reduction_median"]
            entry["makespan_reduction_S_vs_H"] = {"median": median,
                "min": effect["makespan_reduction_range"][0], "max": effect["makespan_reduction_range"][1],
                "direction": "S faster" if median > 0 else "H faster" if median < 0 else "tie",
                "consistent_sign": effect["all_same_sign"]}
        contrasts[workload] = entry
    return {"predeclaration_sha256": freeze_sha,
            "source_freeze_commit_id": protocol.get("source_freeze", {}).get("commit_id"),
            "expected_runs": len(expected), "admitted_runs": len(admitted), "excluded_runs": excluded,
            "runs": admitted, "contrasts": contrasts,
            "scope": "Finite-batch prefill-plus-decode efficiency at allocated chip count; not a universal TP crossover or pure device timing. Deep-backlog claims require a measured common interval."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=pathlib.Path,
                        default=ROOT / "results/hetero/evaluation_amendment_20260916_source_frozen.json")
    parser.add_argument("--run-dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    run_dir = args.run_dir or (ROOT / protocol["run_dir"] if protocol.get("run_dir") else None)
    # No fixed run-id: the selected cohort can span multiple retained directories.
    paths = sorted(run_dir.glob("*_summary.json") if run_dir else
                   (ROOT / "results/hetero/artifacts").glob("*/*_summary.json"))
    rows = [dict(json.loads(path.read_text()), result_artifact=str(path)) for path in paths]
    wrappers = [(path, json.loads(path.read_text())) for path in sorted((ROOT / "results/hetero").glob("*/*.json"))]
    report = build(protocol, rows, initialization=startup_evidence(wrappers))
    report["run_dir"] = str(run_dir) if run_dir else None
    out = args.output or (ROOT / protocol["output"] if protocol.get("output") else
                          args.protocol.parent / f"efficiency_analysis_{protocol['predeclaration_sha256'][:12]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"admitted": report["admitted_runs"], "expected": report["expected_runs"],
                      "excluded": len(report["excluded_runs"]), "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
