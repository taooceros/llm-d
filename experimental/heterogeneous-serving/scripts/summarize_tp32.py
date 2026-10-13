#!/usr/bin/env python3
"""Builds results/tp32/e1_summary.json from the E1 (1xTP32) evidence files."""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TP32 = ROOT / "results/tp32"
H_REFERENCE = ROOT / "results/restore/ShardedRestore/H-reference.json"


def load(path: pathlib.Path):
    """Reads JSON evidence; large deploy/validate files are committed as compact .json.gz."""
    if path.is_file():
        return json.loads(path.read_text())
    packed = path.with_name(path.name + ".gz")
    return json.loads(gzip.decompress(packed.read_bytes())) if packed.is_file() else None


def deploy_summary(path: pathlib.Path):
    doc = load(path)
    if doc is None:
        return None
    m = doc["measurements"]
    init = m["deployment"]["instances"][0]["initialization"]
    config = init["resolved_engine_config"]
    phases = {}
    for name in ("init_device", "load_model", "initialize_from_config", "compile_or_warm_up_model"):
        spans = [w["phases"][name] for w in init["worker_init_phases"] if name in w.get("phases", {})]
        if spans:
            durations = [s["end"] - s["start"] for s in spans]
            phases[name] = {"min_s": round(min(durations), 1), "max_s": round(max(durations), 1),
                            "workers": len(spans)}
    return {
        "file": str(path.relative_to(ROOT)), "submission_id": doc["submission_id"],
        "ok": m.get("ok", True), "deploy_command_elapsed_s": m.get("elapsed_s"),
        "engine_init_s": init.get("init_s"),
        "engine_settings": m["deployment"]["desired_configuration"]["engine_settings"],
        "max_model_len": config["max_model_len"], "tensor_parallel_size": config["tensor_parallel_size"],
        "block_size": config["block_size"], "num_gpu_blocks": config["num_gpu_blocks"],
        "kv_pool_tokens": config["block_size"] * config["num_gpu_blocks"],
        "kv_capacity": init["kv_capacity"], "worker_phases": phases,
        "gateway_ready": m["gateway"]["bind"]["discovery"]["ready"],
    }


def arm_summary(path: pathlib.Path):
    doc = load(path)
    if doc is None:
        return None
    r = doc["measurements"]["result"]
    return {
        "file": str(path.relative_to(ROOT)), "submission_id": doc["submission_id"],
        "artifact_dir": f"results/hetero/artifacts/{r['run_id']}/", "run_id": r["run_id"],
        "workload": r["workload"], "manifest_sha256": r["manifest_sha256"],
        "num_requests": r["num_requests"], "makespan_s": r["makespan_s"],
        "useful_output_tokens": r["useful_output_tokens"],
        "expected_output_tokens": r["expected_output_tokens"],
        "useful_tokens_per_s": r["useful_tokens_per_s"],
        "useful_tokens_per_s_per_chip": r["useful_tokens_per_s_per_chip"],
        "all_requests_complete": r["all_requests_complete"],
        "failed_requests": len(r["failed_requests"]), "missing_requests": len(r["missing_requests"]),
        "short_requests": len(r["short_requests"]), "total_preemptions": r["total_preemptions"],
        "iteration_latency_p50_s": r["iteration_stats"]["per_instance"]["large"]["iteration_latency_p50_s"],
        "scheduled_requests_max": r["iteration_stats"]["per_instance"]["large"]["scheduled_requests_max"],
        "pilot_policy": r["pilot_policy"],
    }


def validate_summary(path: pathlib.Path):
    doc = load(path)
    if doc is None:
        return None
    measured = doc["measurements"]
    reference = load(H_REFERENCE)["measurements"]
    if measured["prompts"] != reference["prompts"] or measured["sampling"] != reference["sampling"]:
        raise ValueError("validation prompts/sampling differ from the TP16 reference")
    outputs = measured["outputs"]["large"]
    against = {}
    for name, ref in sorted(reference["outputs"].items()):
        against[name] = {
            "exact_sequences": sum(a == b for a, b in zip(outputs, ref)), "sequences": len(ref),
            "matching_positions": sum(sum(x == y for x, y in zip(a, b)) for a, b in zip(outputs, ref)),
            "reference_tokens": sum(len(b) for b in ref),
        }
    return {"file": str(path.relative_to(ROOT)), "submission_id": doc["submission_id"],
            "reference": str(H_REFERENCE.relative_to(ROOT)), "manifest_sha256": measured["manifest_sha256"],
            "sampling": measured["sampling"], "vs_tp16": against}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--extra", type=pathlib.Path, help="JSON merged verbatim (blockers, cleanup)")
    args = parser.parse_args()
    pilots = sorted((ROOT / "results/pilot").glob("tp32-*_T32.json"))
    summary = {
        "experiment": "E1 1xTP32 Gemma-4 31B", "layout_file": "deploy/layouts/T32.json",
        "code_commit": args.commit,
        "inventory_submission": (load(TP32 / "inventory.json") or {}).get("submission_id"),
        "preflight_submission": (load(TP32 / "preflight.json") or {}).get("submission_id"),
        "deploy_cold": deploy_summary(TP32 / "deploy-cold.json"),
        "deploy_warm": deploy_summary(TP32 / "deploy-warm.json"),
        "validate": validate_summary(TP32 / "validate.json"),
        "arms": [arm_summary(path) for path in pilots],
        "baseline_H": {"W4_plain_makespan_s": 347.1, "W4_reps": [347.2, 346.9], "W0_blind_makespan_s": 497.2,
                       "kv_pool_tokens_per_tp16": 72192},
    }
    if args.extra:
        summary.update(json.loads(args.extra.read_text()))
    (TP32 / "e1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
