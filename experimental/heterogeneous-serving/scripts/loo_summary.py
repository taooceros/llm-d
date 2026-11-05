#!/usr/bin/env python3
"""Leave-one-out (LOO) ablation of TPUFlex, cell ``tpu-jax`` (vLLM-JAX on TPU v5e, TP8), from one
``switch_campaign.sh autoscale`` partition campaign that interleaves the in-session full arm with one arm per
removed mechanism (round robin, n=3 each):

    .venv/bin/python scripts/loo_summary.py results/switch/autoscale-<rev>-psa-loo \\
        --matrix results/switch/matrix-966399c/matrix_summary.json --commit <rev> \\
        --out results/switch/loo-<rev>/loo_summary.json [--markdown results/switch/loo-<rev>/table.md]

Arm <- campaign arm (every arm overlaps delivery like gcs-ov unless the overlap itself is removed):
  pm-gcs          gcs-ov        full: snapshot + bank + XLA cache (0 s threshold) from GCS, overlapped, FastGraph
  loo-snapshot    runai-bank-ov weights via the Run:ai streamer (stock-runai-cc's loader); bank + XLA cache of a
                                Run:ai seed boot overlapped
  loo-bank        snap-xla-ov   no executable bank: compile from the shipped XLA cache
  loo-xla         snap-bank-ov  no shipped XLA cache: bank + compile of the rest from an empty cache
  loo-bank-xla    snap-ov       neither (both compile mechanisms overlap)
  loo-overlap     gcs           sequential GCS delivery, then deploy (T0 = first byte of delivery)
  loo-fast-graph  gcs-nofg-ov   FastGraph off (restore builder traces without the jax 0.10 tracer patch)

Ready, T0 and phases exactly as scripts/matrix_summary.py (this reuses its run_row).  Each run also gets a
mechanism check: what the workers report (snapshot restored, bank installed, FastGraph, state gate and the
delivered parts, load format) against what the arm must show; a run whose check fails does not count.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matrix_summary as ms  # noqa: E402

ARMS = (  # name, campaign arm, removed, fallback, expectation
    ("pm-gcs", "gcs-ov", None, None,
     dict(restored=True, bank=True, fast_graph=True, gated=True, parts={"snapshot", "bank", "xla"}, runai=False)),
    ("loo-snapshot", "runai-bank-ov", "weight snapshot",
     "Run:ai model streamer from gs:// (stock-runai-cc's loader: --tokenizer /models/gemma-4-31b, 8 GiB streamer "
     "memory limit); bank + XLA cache come from a second seed boot that loaded with Run:ai, because the bank "
     "identity names the model path",
     dict(restored=False, bank=True, fast_graph=True, gated=True, parts={"bank", "xla"}, runai=True)),
    ("loo-bank", "snap-xla-ov", "executable bank",
     "trace + lower every bucket, executables from the shipped XLA cache (0 s threshold)",
     dict(restored=True, bank=False, fast_graph=True, gated=True, parts={"snapshot", "xla"}, runai=False)),
    ("loo-xla", "snap-bank-ov", "shipped XLA cache",
     "empty scope XLA cache: what the bank does not cover (snapshot restore programs, KV/sampler setup) compiles",
     dict(restored=True, bank=True, fast_graph=True, gated=True, parts={"snapshot", "bank"}, runai=False)),
    ("loo-bank-xla", "snap-ov", "executable bank + shipped XLA cache",
     "stock compile: trace + lower + XLA compile of every program from an empty cache",
     dict(restored=True, bank=False, fast_graph=True, gated=True, parts={"snapshot"}, runai=False)),
    ("loo-overlap", "gcs", "overlapped delivery",
     "sequential: startup_state.py fetch (GCS transfer_manager) of snapshot + bank + XLA, then deploy",
     dict(restored=True, bank=True, fast_graph=True, gated=False, parts={"snapshot", "bank", "xla"}, runai=False)),
    ("loo-fast-graph", "gcs-nofg-ov", "FastGraph",
     "the snapshot builder's nnx.eval_shape without the Tracer.sharding fast-fail patch (stock jax 0.10.2)",
     dict(restored=True, bank=True, fast_graph=False, gated=True, parts={"snapshot", "bank", "xla"}, runai=False)),
)
ms.OVERLAP = tuple(a for _, a, *_ in ARMS if a.endswith("-ov"))
PHASES = ("download", "spawn", "init_device", "load_model", "kv", "compile", "tail")


def mechanism_check(directory, row, expect):
    deploy = ms.load(directory, f"{row['run_id']}-deploy.json")
    _, init = ms.target_init(deploy, row["pair"])
    workers = init.get("worker_init_phases") or []
    evidence = init.get("worker_evidence") or []
    settings = init.get("engine_settings") or {}
    gates = [w.get("state_gate") for w in workers]
    if expect["gated"]:
        parts = [set((g or {}).get("delivered", {}).get("parts") or []) for g in gates]
    else:
        fetch = ms.load(directory, f"{row['run_id']}-fetch.json") or {}
        parts = [set((fetch.get("measurements") or {}).get("parts") or [])] * max(1, len(workers))
    banks = [w.get("exec_bank") or {} for w in workers]
    snaps = [e.get("weight_snapshot") or {} for e in evidence] or [w.get("weight_snapshot") or {} for w in workers]
    observed = {
        "snapshot_restored": all(s.get("restored") for s in snaps) if snaps else False,
        "bank_installed": all(b.get("installed") for b in banks) if banks else False,
        "bank_mode": sorted({str(b.get("mode")) for b in banks}),
        "bank_refusal": sorted({str(b.get("error") or b.get("load_error") or b.get("fallback") or "")
                                for b in banks} - {""}),
        "fast_graph": all(s.get("fast_graph") for s in snaps) if snaps else None,
        "xla_backend_compiles_rank0": (row.get("evidence") or {}).get("xla_backend_compiles_rank0"),
        "gate_wait_s": row.get("gate_wait_s"),
        "gated": all(isinstance(g, dict) for g in gates) and bool(gates),
        "delivered_parts": sorted(set.union(*parts)) if parts else [],
        "load_format": settings.get("load_format"),
        "model": (init.get("resolved_engine_config") or {}).get("model"),
        "kv_blocks": (row.get("evidence") or {}).get("kv_blocks"),
        "compile_cache_dirs": sorted({(w.get("compile_cache") or {}).get("dir") for w in workers} - {None}),
    }
    failures = []
    if observed["snapshot_restored"] != expect["restored"]:
        failures.append(f"snapshot_restored={observed['snapshot_restored']}")
    if observed["bank_installed"] != expect["bank"]:
        failures.append(f"bank_installed={observed['bank_installed']} {observed['bank_refusal']}")
    if not expect["bank"] and observed["bank_mode"] != ["None"]:
        failures.append(f"bank mode {observed['bank_mode']} (expected no bank)")
    if bool(observed["fast_graph"]) != expect["fast_graph"]:
        failures.append(f"fast_graph={observed['fast_graph']}")
    if observed["gated"] != expect["gated"]:
        failures.append(f"gated={observed['gated']}")
    if any(p != expect["parts"] for p in parts):
        failures.append(f"delivered parts {[sorted(p) for p in parts]} != {sorted(expect['parts'])}")
    if (observed["load_format"] == "runai_streamer") != expect["runai"]:
        failures.append(f"load_format={observed['load_format']}")
    observed["ok"] = not failures
    observed["failures"] = failures
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--matrix", type=Path, required=True, help="the cell's matrix_summary.json")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    config = json.loads((args.directory / "autoscale.json").read_text())
    part = config["partition"]
    matrix = json.loads(args.matrix.read_text())
    runs = [json.loads(l) for l in (args.directory / "runs.jsonl").read_text().splitlines() if l.strip()]
    expect = {arm: e for _, arm, _, _, e in ARMS}
    rows, failed = [], []
    for run in runs:
        if run["kind"] not in expect:
            continue
        row = ms.run_row(args.directory, run, part)
        if row.get("phases"):
            row["phases"] = {k: (None if v is None else round(v, 2)) for k, v in row["phases"].items()}
            row["mechanism_check"] = mechanism_check(args.directory, row, expect[run["kind"]])
        good = (run.get("ok") and row.get("greedy_ok") and row.get("health_ok") and row.get("ready_s") is not None
                and (row.get("mechanism_check") or {}).get("ok"))
        row["counted"] = bool(good)
        rows.append(row)
        if not good:
            failed.append({"run_id": row["run_id"], "arm": run["kind"], "campaign_ok": run.get("ok"),
                           "greedy_ok": row.get("greedy_ok"), "error": row.get("error"),
                           "mechanism_failures": (row.get("mechanism_check") or {}).get("failures")})
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    def med(values):
        values = [v for v in values if v is not None]
        return round(statistics.median(values), 2) if values else None

    blocks = {}
    for name, arm, removed, fallback, _ in ARMS:
        good = [r for r in by_arm.get(arm, []) if r["counted"]]
        blocks[name] = {
            "campaign_arm": arm, "removed": removed, "fallback": fallback,
            "status": "ok" if len(good) >= 3 else ("partial" if good else "dnf"),
            "n": len(good), "median": med([r["ready_s"] for r in good]), "runs": [r["ready_s"] for r in good],
            "run_ids": [r["run_id"] for r in good],
            "phases": {p: med([r["phases"].get(p) for r in good]) for p in PHASES},
            "mechanism_check": [{"run_id": r["run_id"], **{k: v for k, v in r["mechanism_check"].items()
                                                          if k not in ("compile_cache_dirs", "model")}}
                                for r in good],
        }
    full = blocks["pm-gcs"]
    for name, block in blocks.items():
        if full["median"] is None or block["median"] is None:
            block["delta_vs_full_s"] = block["delta_vs_full_pct"] = None
            block["phase_deltas_vs_full"] = None
            continue
        block["delta_vs_full_s"] = round(block["median"] - full["median"], 2)
        block["delta_vs_full_pct"] = round(100.0 * (block["median"] - full["median"]) / full["median"], 1)
        block["phase_deltas_vs_full"] = {p: (None if block["phases"][p] is None or full["phases"][p] is None
                                             else round(block["phases"][p] - full["phases"][p], 2)) for p in PHASES}
    matrix_full = matrix["arms"]["pm-gcs"]["median"]
    summary = {
        "cell": "tpu-jax", "hardware": "TPU v5e (v5litepod-32 slice, 2x4 sub-slice S:" + part + ", "
                                       + config["hosts"] + ")",
        "tp": 8, "code_commit": args.commit,
        "matrix_ref": {"summary": str(args.matrix), "commit": matrix.get("code_commit"), "pm_gcs_median_s": matrix_full,
                       "pm_gcs_runs_s": matrix["arms"]["pm-gcs"]["runs"]},
        "t0": matrix["t0"],
        "ready_def": "GET /health 200 on the instance endpoint AND greedy check (32 W0 prompts x 64 tokens, token "
                     "for token against this campaign's seed reference) passed",
        "engine": matrix["engine"], "image_or_venv": matrix["image_or_venv"],
        "baseline_arm": "pm-gcs",
        "in_session_full_vs_matrix": {
            "in_session_median_s": full["median"], "matrix_median_s": matrix_full,
            "diff_pct": (round(100.0 * (full["median"] - matrix_full) / matrix_full, 1)
                         if full["median"] is not None else None)},
        "arms": blocks,
        "failed_attempts": failed,
        "campaign": {"directory": str(args.directory), **config},
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    lines = ["| arm | campaign arm | removed | n | median s | runs s | Δ vs in-session full | key phase deltas (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, b in blocks.items():
        deltas = b.get("phase_deltas_vs_full") or {}
        key = ", ".join(f"{p} {v:+.1f}" for p, v in deltas.items() if v is not None and abs(v) >= 1.0) or "–"
        delta = ("–" if name == "pm-gcs" or b["delta_vs_full_s"] is None
                 else f"{b['delta_vs_full_s']:+.1f} s ({b['delta_vs_full_pct']:+.0f} %)")
        lines.append(f"| {name} | {b['campaign_arm']} | {b['removed'] or '– (full PM)'} | {b['n']} | "
                     f"{b['median']} | {', '.join(str(v) for v in b['runs'])} | {delta} | "
                     f"{'–' if name == 'pm-gcs' else key} |")
    text = "\n".join(lines) + "\n"
    if args.markdown:
        args.markdown.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
