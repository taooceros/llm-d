#!/usr/bin/env python3
"""Paper figure: time for a new TP16 instance on fresh hosts to start serving, per weight-delivery arm.

    .venv/bin/python scripts/autoscale_figure.py results/switch/autoscale-4712284 results/switch/autoscale-13205e2 \\
        --svg ~/tpuflex-overleaf/figures/svg/autoscale.svg [--json summary.json]

One bar per arm, drawn to scale.  Segments partition [-delivery, init_s] on large_b's critical path
(earliest worker start, latest worker end of each engine phase, relative to engine_init_started_at):
  download      startup_state.py fetch(-checkpoint) job walls (state or checkpoint onto the new hosts)
  spawn         engine start -> first init_device (Ray workers, imports)
  init_device   init_device
  load_model    load_model (+ the sub-second gap before KV init)
  kv            initialize_from_config ("KV cache init": sampler / logprob precompile, KV allocation)
  compile       exec_bank_load + compile_or_warm_up_model + engine tail, up to init_s
Odd n: the segments of the median run; even n: the mean of the two middle runs.  Either way the bar
total is the median of the per-run totals.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscale_report import instance, load  # noqa: E402

SEGMENTS = (  # key, legend label, fill, text fill
    ("download", "state / checkpoint download", "#44AA99", "#fff"),
    ("spawn", "spawn + imports", "#88CCEE", "#222"),
    ("init_device", "init_device", "#DDCC77", "#222"),
    ("load_model", "load_model", "#CC6677", "#fff"),
    ("kv", "KV cache init", "#999933", "#fff"),
    ("compile", "compile / warm-up (incl. bank load)", "#AA4499", "#fff"),
)
ROWS = (  # arm, title, subtitle
    ("raw", "vLLM default boot", "ckpt via GCS FUSE"),
    ("raw-xla", "+ shared XLA cache", "cache from a peer"),
    ("ckpt-gcs", "Checkpoint pre-pulled", "62.5 GB/host → tmpfs"),
    ("ckpt-gcs-xla", "+ shared XLA cache", "cache from a peer"),
    ("bank-ckpt", "TPUFlex, no snapshot", "ckpt via GCS FUSE"),
    ("gcs", "TPUFlex, GCS", "16.3 GB/host"),
    ("peer", "TPUFlex, peer copy", "16.3 GB/host over DCN"),
    ("fuse", "TPUFlex, GCS FUSE", "16.3 GB/host"),
)


def run_segments(directory, rid):
    deploy = load(directory, f"{rid}-deploy.json")
    init = instance(deploy, "large_b")
    t0 = init["engine_init_started_at"]

    def span(phase):
        rows = [w["phases"][phase] for w in init["worker_init_phases"] if isinstance(w["phases"].get(phase), dict)]
        return (min(r["start"] for r in rows) - t0, max(r["end"] for r in rows) - t0) if rows else None

    device, model, kv = span("init_device"), span("load_model"), span("initialize_from_config")
    compile_start = (span("exec_bank_load") or span("compile_or_warm_up_model"))[0]
    download = sum(load(directory, name)["measurements"]["wall_s"]
                   for name in (f"{rid}-ckpt-fetch.json", f"{rid}-fetch.json") if load(directory, name))
    seg = {"download": download, "spawn": device[0], "init_device": model[0] - device[0],
           "load_model": kv[0] - model[0], "kv": compile_start - kv[0], "compile": init["init_s"] - compile_start}
    if min(seg.values()) < 0:
        raise RuntimeError(f"{rid}: phases out of order {seg}")
    return seg


def collect(directories):
    by_arm = {}
    for directory in map(Path, directories):
        for line in (directory / "runs.jsonl").read_text().splitlines():
            run = json.loads(line)
            if run.get("ok"):
                by_arm.setdefault(run["kind"], []).append(
                    {"run_id": run["run_id"], **run_segments(directory, run["run_id"])})
    summary = {}
    for arm, runs in by_arm.items():
        ranked = sorted(runs, key=lambda r: sum(r[k] for k, *_ in SEGMENTS))
        middle = ranked[(len(ranked) - 1) // 2: len(ranked) // 2 + 1]
        med = {k: statistics.mean(r[k] for r in middle) for k, *_ in SEGMENTS}
        summary[arm] = {"n": len(runs), "segments": med, "total": sum(med.values()),
                        "run_totals": [sum(r[k] for k, *_ in SEGMENTS) for r in runs], "runs": runs}
    return summary


def svg(summary):
    x0, scale, top, pitch, bar = 172.0, 0.5, 26, 30, 18  # 0.5 px per second
    baseline = summary["raw"]["total"]
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 {h}" font-family="Liberation Sans, '
           'Arial, Helvetica, sans-serif" font-size="11" fill="#222">']
    bottom = top + pitch * len(ROWS) - 8
    for t in range(0, 801, 100):
        x = x0 + t * scale
        out.append(f'<line x1="{x:g}" y1="22" x2="{x:g}" y2="{bottom}" stroke="#e2e2e2" stroke-width="1"/>')
        out.append(f'<text x="{x:g}" y="18" text-anchor="middle" fill="#666">{t} s</text>')
    for i, (arm, title, subtitle) in enumerate(ROWS):
        row = summary[arm]
        y = top + pitch * i
        out.append(f'<text x="164" y="{y + 8}" text-anchor="end" font-size="11.5" font-weight="bold">{title}</text>')
        out.append(f'<text x="164" y="{y + 21}" text-anchor="end" fill="#666">{subtitle}; n = {row["n"]}</text>')
        x = x0
        for key, _, fill, ink in SEGMENTS:
            width = row["segments"][key] * scale
            if width <= 0:
                continue
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{width:.1f}" height="{bar}" fill="{fill}" '
                       f'stroke="#fff" stroke-width="0.6"/>')
            if width >= 30:
                out.append(f'<text x="{x + width / 2:.1f}" y="{y + 13}" text-anchor="middle" fill="{ink}">'
                           f'{row["segments"][key]:.0f}</text>')
            x += width
        label = f'{row["total"]:.1f} s'
        if arm != "raw":
            label += f' ({baseline / row["total"]:.1f}×)' if row["total"] < baseline / 4 else ''
        out.append(f'<text x="{x + 5:.1f}" y="{y + 13}" font-weight="bold">{label}</text>')
    legend = bottom + 14
    for j, (_, name, fill, _) in enumerate(SEGMENTS):
        lx, ly = 8 + 210 * (j % 3), legend + 15 * (j // 3)
        out.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{fill}" stroke="#555" stroke-width="0.5"/>')
        out.append(f'<text x="{lx + 14}" y="{ly + 9}">{name}</text>')
    out.append("</svg>")
    return "\n".join(out).replace("{h}", str(legend + 30)) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    summary = collect(args.directories)
    args.svg.write_text(svg(summary))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    for arm, *_ in ROWS:
        row = summary[arm]
        print(f'{arm:13s} n={row["n"]} total={row["total"]:.1f} runs={[round(t, 1) for t in row["run_totals"]]} '
              + " ".join(f'{k}={v:.1f}' for k, v in row["segments"].items()))


if __name__ == "__main__":
    main()
