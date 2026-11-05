#!/usr/bin/env python3
"""GPU counterpart of paper Fig 2: time for a new TP1 vLLM instance on a fresh H100 node to start serving.

    .venv/bin/python scripts/gpu_autoscale_figure.py results/gpu-autoscale/<campaign> [...] \\
        --svg ~/tpuflex-overleaf/figures/svg/autoscale_gpu.svg [--json summary.json]

Panel (a): each row adds one change to the row above (same style and colours as scripts/autoscale_figure.py).
Panel (b): what GKE spends before the engine process starts on that node (excluded from panel (a), as in Fig 2).
Segments come from scripts/gpu_autoscale.py parse_log (kubelet log timestamps + in-pod MARK lines):
  download     checkpoint / cache delivery onto the node (fetch.py)
  spawn        `vllm serve` start -> EngineCore "Initializing a V1 LLM engine" (API server, imports, config)
  init_device  -> "Starting to load model" (executor, worker, CUDA context, NCCL init)
  lazy         (GCS-reading arms) load_model time waiting on GCS, inferred: the arm's load_model minus the mean
               load_model of the arms that load the identical checkpoint from local tmpfs (ckpt, ckpt-cc)
  load_model   the rest of load_model ("Model loading took")
  compile      -> "torch.compile ... took" (Dynamo + Inductor, or loading the compiled graphs from cache)
  kv           -> CUDA graph capture start (rest of the profiling run, KV sizing and allocation, warm-up)
  graphs       CUDA graph capture ("Graph capturing finished in N secs")
  tail         -> /health answers (sampler warm-up, API server start)
Odd n: segments of the median run; even n: mean of the two middle runs.  Bar total = median of run totals.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

SEGMENTS = (  # key, legend label, fill, text fill
    ("download", "ckpt / cache download", "#44AA99", "#fff"),
    ("spawn", "spawn + imports", "#88CCEE", "#222"),
    ("init_device", "init_device", "#DDCC77", "#222"),
    ("lazy", "waiting on GCS (inferred)", "url(#lazy)", "#222"),
    ("load_model", "load_model compute", "#CC6677", "#fff"),
    ("compile", "torch.compile", "#AA4499", "#fff"),
    ("kv", "profile + KV cache init", "#999933", "#fff"),
    ("graphs", "CUDA graph capture", "#882255", "#fff"),
    ("tail", "API server up", "#BBBBBB", "#222"),
)
LOCAL = ("ckpt", "ckpt-cc")  # same checkpoint, same loader, from local tmpfs
GCS_READING = ("raw", "runai", "runai-cc")
STOCK = "raw"
PANEL_A = (  # arm, title, subtitle
    ("raw", "Stock vLLM", "lazy mmap via GCS FUSE"),
    ("ckpt", "+ download ckpt first", "62.6 GB → tmpfs"),
    ("ckpt-cc", "+ peer compile cache", "copied before boot"),
    ("runai", "Run:ai streamer", "GCS → GPU, no local copy"),
    ("runai-cc", "+ peer compile cache", "copied before boot"),
)
GROUP_BREAKS_A = (3,)
GKE_STAGES = (  # key, label, fill, (start event, end event) in seconds after pod submission (API: whole seconds)
    ("provision", "VM create + node boot", "#4477AA", (None, "node_created")),
    ("gpu", "GPU driver → scheduled", "#117733", ("node_created", "pod_scheduled")),
    ("pull", "image pull + start", "#332288", ("pod_scheduled", "container_started")),
)
X0, PITCH, BAR = 172.0, 25, 16


def collect(directories):
    by_arm, gke = {}, []
    for directory in map(Path, directories):
        for line in (directory / "runs.jsonl").read_text().splitlines():
            run = json.loads(line)
            if run.get("ok") and run.get("segments"):
                by_arm.setdefault(run["arm"], []).append({"run_id": run["run_id"], **run["segments"]})
                g = run.get("gke") or {}
                # GKE stages only for runs whose node was created after submission: a run can land on a spare
                # node the autoscaler had already added for an earlier pending pod (fresh, but not a scale-up)
                if g.get("container_started") is not None and (g.get("node_created") or -1) > 0:
                    gke.append(g)
    local = [r["load_model"] for a in LOCAL for r in by_arm.get(a, [])]
    reference = statistics.mean(local) if local else None
    for arm, runs in by_arm.items():
        for r in runs:
            r["lazy"] = max(0.0, r["load_model"] - reference) if arm in GCS_READING and reference else 0.0
            r["load_model"] -= r["lazy"]
    summary = {"local_load_reference_s": reference}
    for arm, runs in by_arm.items():
        ranked = sorted(runs, key=lambda r: sum(r[k] for k, *_ in SEGMENTS))
        middle = ranked[(len(ranked) - 1) // 2: len(ranked) // 2 + 1]
        med = {k: statistics.mean(r[k] for r in middle) for k, *_ in SEGMENTS}
        summary[arm] = {"n": len(runs), "segments": med, "total": sum(med.values()),
                        "run_totals": [sum(r[k] for k, *_ in SEGMENTS) for r in runs], "runs": runs}
    rows = [{key: g[end] - (g[start] if start else 0.0) for key, _, _, (start, end) in GKE_STAGES} for g in gke]
    ranked = sorted(rows, key=lambda r: sum(r.values()))  # same median-run rule as panel (a)
    middle = ranked[(len(ranked) - 1) // 2: len(ranked) // 2 + 1]
    stages = {key: statistics.mean(r[key] for r in middle) if middle else 0.0 for key, *_ in GKE_STAGES}
    summary["gke"] = {"n": len(gke), "stages": stages, "total": sum(stages.values()),
                      "container_started_s": sorted(g["container_started"] for g in gke)}
    return summary


def bar(out, y, x, segments, scale, fills):
    for key, value in segments:
        fill, ink = fills[key]
        width = value * scale
        if width <= 0:
            continue
        out.append(f'<rect x="{x:.1f}" y="{y}" width="{width:.1f}" height="{BAR}" fill="{fill}" '
                   'stroke="#fff" stroke-width="0.6"/>')
        if width >= 22:
            text = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
            if key == "lazy":
                text = "≈" + text
                out.append(f'<rect x="{x + width / 2 - 15:.1f}" y="{y + 2.5}" width="30" height="11" rx="2" '
                           'fill="#fff" fill-opacity="0.9"/>')
            out.append(f'<text x="{x + width / 2:.1f}" y="{y + 12}" text-anchor="middle" fill="{ink}">{text}</text>')
        x += width
    return x


def axis(out, top, bottom, scale, span, step):
    for t in range(0, span + 1, step):
        x = X0 + t * scale
        out.append(f'<line x1="{x:g}" y1="{top + 3}" x2="{x:g}" y2="{bottom}" stroke="#e2e2e2" stroke-width="1"/>')
        out.append(f'<text x="{x:g}" y="{top}" text-anchor="middle" fill="#666">{t} s</text>')


def nice(span):
    step = next(s for s in (10, 20, 25, 50, 100, 200, 250, 500) if span / s <= 9)
    return step, int(-(-span // step) * step)


def svg(summary):
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 {h}" font-family="Liberation Sans, '
           'Arial, Helvetica, sans-serif" font-size="11" fill="#222">',
           '<defs><pattern id="lazy" patternUnits="userSpaceOnUse" width="6" height="6" '
           'patternTransform="rotate(45)"><rect width="6" height="6" fill="#CC6677"/>'
           '<rect width="3" height="6" fill="#44AA99"/></pattern></defs>']
    rows = [r for r in PANEL_A if r[0] in summary]
    longest = max(summary[a]["total"] for a, *_ in rows)
    step, span = nice(longest * 1.12)
    scale = (632 - 70 - X0) / span
    top = 4
    out.append(f'<text x="8" y="{top + 11}" font-weight="bold" font-size="11.5">(a) vLLM on a fresh H100 node: '
               'from stock to shipped startup state</text>')
    bars = top + 32
    bottom = bars + PITCH * len(rows) - (PITCH - BAR) + 3
    axis(out, top + 25, bottom, scale, span, step)
    fills = {k: (fill, ink) for k, _, fill, ink in SEGMENTS}
    baseline = summary[STOCK]["total"] if STOCK in summary else None
    for i, (arm, name, subtitle) in enumerate(rows):
        row, y = summary[arm], bars + PITCH * i
        if i in GROUP_BREAKS_A:
            out.append(f'<line x1="8" y1="{y - 4.5}" x2="632" y2="{y - 4.5}" stroke="#bbb" stroke-width="0.6" '
                       'stroke-dasharray="3 2"/>')
        out.append(f'<text x="164" y="{y + 7}" text-anchor="end" font-size="11.5" font-weight="bold">{name}</text>')
        out.append(f'<text x="164" y="{y + 19}" text-anchor="end" fill="#666" font-size="10">{subtitle}; '
                   f'n = {row["n"]}</text>')
        x = bar(out, y, X0, [(k, row["segments"][k]) for k, *_ in SEGMENTS], scale, fills)
        label = f'{row["total"]:.1f} s'
        if baseline and arm != STOCK:
            label += f' ({baseline / row["total"]:.1f}×)'
        out.append(f'<text x="{x + 5:.1f}" y="{y + 12}" font-weight="bold">{label}</text>')
    gke = summary["gke"]
    if gke["n"]:
        top = bottom + 8
        out.append(f'<text x="8" y="{top + 11}" font-weight="bold" font-size="11.5">(b) Before the engine starts: '
                   'GKE scale-up of the H100 node (same axis)</text>')
        y = top + 32
        axis(out, top + 25, y + BAR + 3, scale, span, step)
        out.append(f'<text x="164" y="{y + 7}" text-anchor="end" font-size="11.5" font-weight="bold">'
                   'Spot a3-highgpu-1g</text>')
        out.append(f'<text x="164" y="{y + 19}" text-anchor="end" fill="#666" font-size="10">pod submit → '
                   f'container start; n = {gke["n"]}</text>')
        gfills = {k: (fill, "#fff") for k, _, fill, _ in GKE_STAGES}  # all GKE fills are dark
        x = bar(out, y, X0, [(k, gke["stages"][k]) for k, *_ in GKE_STAGES], scale, gfills)
        out.append(f'<text x="{x + 5:.1f}" y="{y + 12}" font-weight="bold">{gke["total"]:.1f} s</text>')
        bottom = y + BAR + 3
    legend = bottom + 10
    items = [(name, fill) for key, name, fill, _ in SEGMENTS] + [(name, fill) for _, name, fill, _ in GKE_STAGES]
    for j, (name, fill) in enumerate(items):
        lx, ly = 8 + 158 * (j % 4), legend + 15 * (j // 4)
        out.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{fill}" stroke="#555" stroke-width="0.5"/>')
        out.append(f'<text x="{lx + 14}" y="{ly + 9}">{name}</text>')
    out.append("</svg>")
    return "\n".join(out).replace("{h}", str(legend + 15 * ((len(items) + 3) // 4) + 6)) + "\n"


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
    print("local load_model reference:", summary["local_load_reference_s"])
    for arm, *_ in PANEL_A:
        if arm in summary:
            row = summary[arm]
            print(f'{arm:9s} n={row["n"]} total={row["total"]:.1f} runs={[round(t, 1) for t in row["run_totals"]]} '
                  + " ".join(f'{k}={v:.1f}' for k, v in row["segments"].items()))
    g = summary["gke"]
    print(f'gke n={g["n"]} total={g["total"]:.1f} ' + " ".join(f"{k}={v:.1f}" for k, v in g["stages"].items()))


if __name__ == "__main__":
    main()
