#!/usr/bin/env python3
"""Paper figure: time for a new TP16 instance on fresh hosts to start serving.

    .venv/bin/python scripts/autoscale_figure.py results/switch/autoscale-{4712284,13205e2,906d2ef,ac455cb-pa} \\
        --svg ~/tpuflex-overleaf/figures/svg/autoscale.svg [--json summary.json]

Panel (a): from stock vLLM to TPUFlex, each row adding one fix to the row above.  Panel (b), on a 3x
finer axis: TPUFlex shipping only part of its startup state (leave-one-out).
Segments partition [-delivery, init_s] on large_b's critical path (earliest worker start, latest
worker end of each engine phase, relative to engine_init_started_at):
  download      startup_state.py fetch(-checkpoint) job walls (state or checkpoint onto the new hosts)
  spawn         engine start -> first init_device (Ray workers, imports)
  init_device   init_device
  lazy          (lazy-checkpoint arms only) load_model time spent stalled on GCS FUSE page faults, inferred:
                max over ranks of normal_load read_s + host_convert_s + shard_place_s, minus the mean of
                the same quantity over the arms that load the identical checkpoint from local tmpfs
  load_model    the rest of load_model (+ the sub-second gap before KV init)
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
    ("download", "state / ckpt download", "#44AA99", "#fff"),
    ("spawn", "spawn + imports", "#88CCEE", "#222"),
    ("init_device", "init_device", "#DDCC77", "#222"),
    ("lazy", "lazy ckpt reads stalling load_model (inferred)", "url(#lazy)", "#222"),
    ("load_model", "load_model compute", "#CC6677", "#fff"),
    ("kv", "KV cache init", "#999933", "#fff"),
    ("compile", "compile / warm-up", "#AA4499", "#fff"),
)
LEGEND = ("download", "spawn", "init_device", "load_model", "lazy", None, "kv", "compile")  # 4 columns
LOCAL_CKPT_ARMS = ("ckpt-gcs", "ckpt-gcs-xla", "ckpt-fuse")  # same checkpoint, same loader, from local tmpfs
# Arms whose loader reads the checkpoint on the GCS FUSE mount -> the local-tmpfs arms with the same
# persistent-cache threshold, whose read + convert + place time is the I/O-free reference.
LAZY_REFERENCE = {"raw": LOCAL_CKPT_ARMS, "raw-xla": LOCAL_CKPT_ARMS, "bank-ckpt": LOCAL_CKPT_ARMS,
                  "raw-pf": LOCAL_CKPT_ARMS, "raw-jax": ("ckpt-gcs-jax",)}
STOCK = "raw-jax"  # speedups are against this row
PANEL_A = (  # arm, title, subtitle; each "+" row adds to the row above
    ("raw-jax", "Stock vLLM", "lazy mmap via GCS FUSE"),
    ("ckpt-gcs-jax", "+ download ckpt first", "62.5 GB/host → tmpfs"),
    ("ckpt-gcs", "+ persist every program", "zero-threshold JAX cache"),
    ("ckpt-gcs-xla", "+ XLA cache from a peer", "copied before boot"),
    ("gcs", "TPUFlex", "16.3 GB/host from GCS"),
)
GROUP_BREAKS_A = (4,)
PANEL_B = (
    ("snap", "Snapshot only", "pre-sharded weights"),
    ("snap-xla", "+ XLA cache", "no Executable Bank"),
    ("snap-bank", "+ Executable Bank", "no XLA cache"),
    ("gcs", "+ both (TPUFlex)", "all from GCS"),
)
X0, PITCH, BAR = 172.0, 25, 16


def run_segments(directory, rid, target="large_b"):
    deploy = load(directory, f"{rid}-deploy.json")
    init = instance(deploy, target)
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
    loads = [w.get("normal_load") for w in init["worker_init_phases"]]
    seg["_load_s"] = (max(n["read_s"] + n["host_convert_s"] + n["shard_place_s"] for n in loads)
                      if all(loads) else None)
    return seg


def collect(directories):
    by_arm = {}
    for directory in map(Path, directories):
        config = load(directory, "autoscale.json") or {}
        target = config.get("partition") or "large_b"  # partition campaigns boot the instance they own
        for line in (directory / "runs.jsonl").read_text().splitlines():
            run = json.loads(line)
            if run.get("ok"):
                by_arm.setdefault(run["kind"], []).append(
                    {"run_id": run["run_id"], "target": target, **run_segments(directory, run["run_id"], target)})
    references = {}
    for arm in by_arm:
        if arm in LAZY_REFERENCE:
            local = [r["_load_s"] for a in LAZY_REFERENCE[arm] for r in by_arm.get(a, [])]
            if not local:
                raise RuntimeError(f"{arm}: no local-checkpoint runs of {LAZY_REFERENCE[arm]} to compare against")
            references[arm] = statistics.mean(local)
    for arm, runs in by_arm.items():
        for r in runs:
            load_s = r.pop("_load_s")
            r["lazy"] = load_s - references[arm] if arm in LAZY_REFERENCE else 0.0
            r["load_model"] -= r["lazy"]
    summary = {"local_load_reference_s": references}
    for arm, runs in by_arm.items():
        ranked = sorted(runs, key=lambda r: sum(r[k] for k, *_ in SEGMENTS))
        middle = ranked[(len(ranked) - 1) // 2: len(ranked) // 2 + 1]
        med = {k: statistics.mean(r[k] for r in middle) for k, *_ in SEGMENTS}
        summary[arm] = {"n": len(runs), "segments": med, "total": sum(med.values()),
                        "run_totals": [sum(r[k] for k, *_ in SEGMENTS) for r in runs], "runs": runs}
    return summary


def panel(out, summary, rows, top, title, scale, span, step, breaks=()):
    """Draw one panel from y=top; return its bottom y."""
    baseline = summary[STOCK]["total"]
    axis, bars = top + 25, top + 32
    bottom = bars + PITCH * len(rows) - (PITCH - BAR) + 3
    out.append(f'<text x="8" y="{top + 11}" font-weight="bold" font-size="11.5">{title}</text>')
    for t in range(0, span + 1, step):
        x = X0 + t * scale
        out.append(f'<line x1="{x:g}" y1="{axis + 3}" x2="{x:g}" y2="{bottom}" stroke="#e2e2e2" stroke-width="1"/>')
        out.append(f'<text x="{x:g}" y="{axis}" text-anchor="middle" fill="#666">{t} s</text>')
    for i, (arm, name, subtitle) in enumerate(rows):
        row = summary[arm]
        y = bars + PITCH * i
        if i in breaks:
            out.append(f'<line x1="8" y1="{y - 4.5}" x2="632" y2="{y - 4.5}" stroke="#bbb" stroke-width="0.6" '
                       'stroke-dasharray="3 2"/>')
        out.append(f'<text x="164" y="{y + 7}" text-anchor="end" font-size="11.5" font-weight="bold">{name}</text>')
        out.append(f'<text x="164" y="{y + 19}" text-anchor="end" fill="#666" font-size="10">'
                   f'{subtitle}; n = {row["n"]}</text>')
        x = X0
        for key, _, fill, ink in SEGMENTS:
            value = row["segments"][key]
            width = value * scale
            if width <= 0:
                continue
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{width:.1f}" height="{BAR}" fill="{fill}" '
                       f'stroke="#fff" stroke-width="0.6"/>')
            if width >= 22:
                text = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
                if key == "lazy":
                    text = "≈" + text
                    out.append(f'<rect x="{x + width / 2 - 15:.1f}" y="{y + 2.5}" width="30" height="11" rx="2" '
                               'fill="#fff" fill-opacity="0.9"/>')
                out.append(f'<text x="{x + width / 2:.1f}" y="{y + 12}" text-anchor="middle" fill="{ink}">'
                           f'{text}</text>')
            x += width
        label = f'{row["total"]:.1f} s'
        if row["total"] < baseline / 4:
            label += f' ({baseline / row["total"]:.1f}×)'
        out.append(f'<text x="{x + 5:.1f}" y="{y + 12}" font-weight="bold">{label}</text>')
    return bottom


def svg(summary):
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 {h}" font-family="Liberation Sans, '
           'Arial, Helvetica, sans-serif" font-size="11" fill="#222">',
           '<defs><pattern id="lazy" patternUnits="userSpaceOnUse" width="6" height="6" '
           'patternTransform="rotate(45)"><rect width="6" height="6" fill="#CC6677"/>'
           '<rect width="3" height="6" fill="#44AA99"/></pattern></defs>']
    bottom = panel(out, summary, PANEL_A, 4, "(a) From stock vLLM to TPUFlex", 0.44, 900, 100, GROUP_BREAKS_A)
    bottom = panel(out, summary, PANEL_B, bottom + 8, "(b) TPUFlex without part of its state (3× axis)",
                   1.32, 300, 50)
    fills = {key: (name, fill) for key, name, fill, _ in SEGMENTS}
    legend = bottom + 10
    for j, key in enumerate(LEGEND):
        if key is None:
            continue
        name, fill = fills[key]
        lx, ly = 8 + 158 * (j % 4), legend + 15 * (j // 4)
        out.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{fill}" stroke="#555" stroke-width="0.5"/>')
        out.append(f'<text x="{lx + 14}" y="{ly + 9}">{name}</text>')
    out.append("</svg>")
    return "\n".join(out).replace("{h}", str(legend + 28)) + "\n"


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
    print("local load reference (max rank read+convert+place):",
          {k: round(v, 1) for k, v in summary["local_load_reference_s"].items()})
    for arm in dict.fromkeys([a for a, *_ in PANEL_A + PANEL_B] + [a for a in summary if a != "local_load_reference_s"]):
        row = summary[arm]
        print(f'{arm:13s} n={row["n"]} total={row["total"]:.1f} runs={[round(t, 1) for t in row["run_totals"]]} '
              + " ".join(f'{k}={v:.1f}' for k, v in row["segments"].items()))


if __name__ == "__main__":
    main()
