#!/usr/bin/env python3
"""Ablation tables for the fresh-host autoscale characterization (markdown on stdout).

    .venv/bin/python scripts/autoscale_ablation.py results/switch/autoscale-<a> results/switch/autoscale-<b> ... \\
        [--probe results/switch/autoscale-probe/ckpt-io-<tag>.json] [--json out.json]

(A) checkpoint path of the stock loader, (B) leave-one-out of TPUFlex's shipped state, (C) cold whole-
checkpoint reads without an engine (startup_state.py ckpt-io).  Per arm: segments of the median run
(autoscale_figure.collect: ok runs only; "lazy" = load_model time in excess of the same loader reading a
local tmpfs copy, inferred), and the median rank-0 count of XLA compiles that missed every cache.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscale_figure import collect  # noqa: E402
from autoscale_report import load, summarize  # noqa: E402

CKPT_ROWS = (
    ("raw", "lazy mmap of the checkpoint on GCS FUSE"),
    ("raw-pf", "+ vLLM safetensors prefetch"),
    ("raw-jax", "lazy, JAX default 1 s cache threshold"),
    ("ckpt-fuse", "copy via FUSE (32 threads) → tmpfs, then load"),
    ("ckpt-gcs", "GCS sliced download → tmpfs, then load"),
    ("ckpt-gcs-jax", "GCS → tmpfs, JAX default 1 s cache threshold"),
)
STATE_ROWS = (
    ("gcs", "snapshot + bank + XLA cache"),
    ("snap-bank", "snapshot + bank"),
    ("snap-xla", "snapshot + XLA cache"),
    ("snap", "snapshot only"),
    ("bank-ckpt", "bank + XLA cache, lazy checkpoint"),
)


def compiles(directories):
    """arm -> median rank-0 backend compiles over ok runs."""
    by_arm = {}
    for directory in directories:
        for row in summarize(directory)["arms"]:
            if row.get("ok") and row.get("xla_compiles") is not None:
                by_arm.setdefault(row["arm"], []).append(row["xla_compiles"])
    return {arm: statistics.median(v) for arm, v in by_arm.items()}


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def arm_rows(summary, counts, rows, reference=None):
    out = []
    for arm, what in rows:
        if arm not in summary:
            continue
        s, seg = summary[arm], summary[arm]["segments"]
        cells = [arm, what, s["n"], f'{seg["download"]:.1f}', f'{seg["lazy"] + seg["load_model"]:.1f}',
                 f'{seg["lazy"]:.0f}' if seg["lazy"] else "–", f'{seg["kv"]:.1f}', f'{seg["compile"]:.1f}',
                 f'{counts[arm]:.0f}' if arm in counts else "?", f'**{s["total"]:.1f}**', "/".join(f"{t:.0f}" for t in sorted(s["run_totals"]))]
        if reference:
            cells.append(f'{s["total"] - summary[reference]["total"]:+.1f}' if reference in summary else "?")
        out.append(cells)
    return out


def probe_rows(probe):
    out = []
    for variant, v in probe["measurements"]["variants"].items():
        gbps = v["per_host_gbps"]
        row = [variant, f"{min(gbps):.2f}–{max(gbps):.2f}", f'{v["max_host_s"]:.1f}',
               f'{min(r["resident_before"] for r in v["ranks"]):.3f}–{max(r["resident_before"] for r in v["ranks"]):.3f}']
        local = [r["local_touch_s"] for r in v["ranks"] if "local_touch_s" in r]
        row.append(f"{min(local):.1f}–{max(local):.1f}" if local else "–")
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    summary = collect(args.directories)
    counts = compiles(args.directories)
    head = ["arm", "what", "n", "delivery s", "load_model s", "lazy reads s", "KV init s", "compile s",
            "XLA compiles", "ready s", "runs"]
    print("## (A) checkpoint path, stock loader (empty XLA cache)\n")
    print(table(head, arm_rows(summary, counts, CKPT_ROWS)))
    print("\n## (B) TPUFlex shipped state, leave-one-out (GCS transport)\n")
    print(table(head + ["Δ vs full"], arm_rows(summary, counts, STATE_ROWS, reference="gcs")))
    probe = load(args.probe.parent, args.probe.name) if args.probe else None
    if probe:
        print("\n## (C) cold whole-checkpoint read per host, no engine (62.5 GB, 4 hosts at once)\n")
        print(table(["variant", "GB/s per host", "slowest host s", "page cache before", "local touch s"],
                    probe_rows(probe)))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "xla_compiles": counts,
                                         "probe": probe and probe["measurements"]}, indent=1) + "\n")


if __name__ == "__main__":
    main()
