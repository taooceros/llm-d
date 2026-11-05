#!/usr/bin/env python3
"""Figures for the 2x2 cold-start matrix, its leave-one-out ablation and the TPU end-to-end runs.

Reads results/matrix-2x2/matrix.json (written by scripts/matrix_2x2.py) and writes PNG (200 dpi) and PDF:

  fig1-ready     ready time of every arm per cell, log scale, with the individual runs
  fig2-phases    phase breakdown: best stock arm vs full TPUFlex (pm-gcs) per cell
  fig3-e2e       TPU end to end from the GKE scale-up trigger: GKE milestones + the rest to ready
  fig4-loo       leave-one-out: full TPUFlex minus one mechanism, against full TPUFlex and best stock

Usage: uv run --with matplotlib python scripts/matrix_2x2_figures.py [--matrix results/matrix-2x2/matrix.json]
       [--out results/matrix-2x2/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CELLS = ["tpu-jax", "tpu-torch", "gpu-jax", "gpu-torch"]
TITLE = {"tpu-jax": "TPU v5e × vLLM-JAX (TP8)", "tpu-torch": "TPU v5e × vLLM-TorchTPU (TP8)",
         "gpu-jax": "H100 × vLLM-JAX (TP1)", "gpu-torch": "H100 × CUDA vLLM (TP1)"}
STOCK = ["stock-lazy", "stock-ckpt", "stock-ckpt-cc", "stock-runai", "stock-runai-cc"]

GRAY, GRAY_DARK, GRAY_LIGHT = "#b4b8bd", "#5f6368", "#dadce0"
PM, PM_LIGHT, PM_PALE = "#1a73e8", "#7baaf7", "#d2e3fc"

PHASES = ["download", "spawn", "init_device", "lazy", "load_model", "compile", "kv", "graphs", "tail"]
PHASE_LABEL = {"download": "download", "spawn": "process spawn", "init_device": "device init",
               "lazy": "lazy weight reads", "load_model": "load_model", "compile": "compile / warm-up",
               "kv": "KV cache init", "graphs": "CUDA graph capture", "tail": "server tail"}
PHASE_COLOR = {"download": "#80868b", "spawn": "#34a853", "init_device": "#a8dab5", "lazy": "#fbbc04",
               "load_model": "#ea4335", "compile": "#fa7b17", "kv": "#a142f4", "graphs": "#24c1e0",
               "tail": "#9aa0a6"}

LOO_LABEL = {
    "loo-snapshot": "− weight snapshot", "loo-bank": "− executable bank", "loo-xla": "− shipped XLA cache",
    "loo-bank-xla": "− bank + XLA cache", "loo-overlap": "− overlapped delivery", "loo-fast-graph": "− fast_graph",
    "loo-sampler-bank": "− sampler bank", "loo-replay-preload": "− fast replay + preload",
    "loo-engine-zygote": "− engine zygote", "loo-ray-zygote": "− Ray worker zygote",
    "loo-warmup": "− warm-up", "loo-cache0": "− 0 s JAX cache", "loo-zygote": "− zygote",
    "loo-read-threads": "− 32-thread read restore", "loo-bytecode": "− bytecode cache",
    "loo-kvpin": "− KV pin (+ bank)",
}
LOO_CLASS = {
    "loo-snapshot": "weight snapshot", "loo-bank": "compile state", "loo-xla": "compile state",
    "loo-bank-xla": "compile state", "loo-sampler-bank": "compile state", "loo-replay-preload": "compile state",
    "loo-cache0": "compile state", "loo-fast-graph": "model graph build", "loo-bytecode": "process start",
    "loo-zygote": "process start", "loo-engine-zygote": "process start", "loo-ray-zygote": "process start",
    "loo-overlap": "overlapped delivery", "loo-kvpin": "KV pin", "loo-warmup": "warm-up / tail",
    "loo-read-threads": "restore I/O",
}
CLASS_COLOR = {"weight snapshot": "#ea4335", "compile state": "#fa7b17", "model graph build": "#a142f4",
               "process start": "#34a853", "overlapped delivery": "#12b5cb", "KV pin": "#e52592",
               "warm-up / tail": "#9e7b5b", "restore I/O": "#9e9d24"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#5f6368",
    "xtick.color": "#3c4043", "ytick.color": "#3c4043", "axes.labelcolor": "#3c4043",
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def fig_ready(m: dict, out: Path) -> None:
    rows = [("stock-lazy", "stock-lazy"), ("stock-ckpt", "stock-ckpt"), ("stock-ckpt-cc", "stock-ckpt-cc"),
            ("stock-runai", "stock-runai"), ("stock-runai-cc", "stock-runai-cc"), ("pm-gcs", "TPUFlex (GCS)"),
            ("pm-peer", "TPUFlex (peer)"), None, ("warm-stock", "warm-stock"),
            ("warm-pm@launch", "warm TPUFlex, from launch"), ("warm-pm@gate", "warm TPUFlex, from gate")]
    lo, hi = 10, 1500
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)
    for ax, cell in zip(axes.flat, CELLS):
        t = m["table"][cell]
        best = t["best_stock"]
        y, labels = 0, []
        for row in rows:
            if row is None:
                y += 0.6
                continue
            arm, label = row
            labels.append((y, label))
            if arm.startswith("warm-pm@"):
                val, runs, status = t["warm_pm"][arm.split("@")[1]], [], "ok"
            else:
                a = t["arms"][arm]
                val, runs, status = a["median"], [r for r in (a.get("runs") or []) if r is not None], a["status"]
            if arm.startswith("stock"):
                color = GRAY_DARK if arm == best else GRAY
            elif arm == "warm-stock":
                color = GRAY_LIGHT
            elif arm == "pm-gcs" or arm == "warm-pm@gate":
                color = PM
            else:
                color = PM_LIGHT
            if status != "ok":
                ax.barh(y, hi - lo, left=lo, height=0.72, color="white", edgecolor=GRAY, hatch="///", lw=0.6)
                note = "dnf (900 s bound)" if cell == "tpu-torch" else "dnf (FUSE page faults)"
                ax.text(lo * 1.15, y, note, va="center", fontsize=8, color=GRAY_DARK)
            else:
                ax.barh(y, val - lo, left=lo, height=0.72, color=color, edgecolor="none")
                if runs:
                    ax.plot(runs, [y] * len(runs), "o", ms=2.2, color="black", alpha=0.55, zorder=3)
                ax.text(max([val] + runs) * 1.07, y, f"{val:.1f}" + ("  best stock" if arm == best else ""),
                        va="center", fontsize=8, color="#202124",
                        fontweight="bold" if arm in (best, "pm-gcs") else "normal")
            y += 1
        ax.set_yticks([p for p, _ in labels], [l for _, l in labels])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xticks([10, 20, 50, 100, 200, 500, 1000], ["10", "20", "50", "100", "200", "500", "1000"])
        ax.grid(axis="x", color="#e8eaed", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{TITLE[cell]}\nfresh host {t['speedup_fresh']:.2f}× · same host "
                     f"{t['speedup_warm_launch']:.2f}× (launch) / {t['speedup_warm_gate']:.2f}× (gate)", loc="left")
    for ax in axes[1]:
        ax.set_xlabel("ready time (s, log scale), median of n=3; dots = individual runs")
    fig.suptitle("Gemma-4 31B cold start: ready time per arm", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig1-ready")


def fig_phases(m: dict, out: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 6.2))
    for ax, cell in zip(axes, CELLS):
        t = m["table"][cell]
        arms = [(t["best_stock"], f"best stock\n({t['best_stock']})"), ("pm-gcs", "TPUFlex\n(pm-gcs)")]
        xmax = t["arms"][t["best_stock"]]["median"] * 1.12
        for y, (arm, _) in enumerate(arms):
            a = t["arms"][arm]
            ph = a["phases"] or {}
            left = 0.0
            for p in PHASES:
                v = ph.get(p)
                if not v or (arm == "pm-gcs" and p == "download"):
                    continue
                ax.barh(y, v, left=left, height=0.62, color=PHASE_COLOR[p], edgecolor="white", lw=0.4)
                if v >= xmax * 0.06:
                    ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white",
                            fontweight="bold")
                left += v
            if arm == "pm-gcs" and (ph.get("download") or 0) > 0.05:
                dl = ph["download"]
                ax.barh(y + 0.4, dl, left=0, height=0.12, color=PHASE_COLOR["download"])
                ax.text(dl + xmax * 0.005, y + 0.4, f"delivery {dl:.1f} s, overlapped with spawn",
                        va="center", fontsize=6.5, color=GRAY_DARK)
            ax.plot([a["median"]] * 2, [y - 0.38, y + 0.38], color="black", lw=1.2)
            ax.text(a["median"] + xmax * 0.008, y - 0.12, f"ready {a['median']:.1f} s", va="center", fontsize=8,
                    fontweight="bold")
        ax.set_yticks([0, 1], [l for _, l in arms], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_ylim(1.62, -0.45)
        ax.set_title(f"{TITLE[cell]}  ({t['speedup_fresh']:.2f}× faster on a fresh host)", loc="left")
        ax.grid(axis="x", color="#e8eaed", lw=0.6)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("seconds from T0 (phase medians, n=3; each panel has its own scale; black tick = median ready)")
    handles = [Patch(color=PHASE_COLOR[p], label=PHASE_LABEL[p]) for p in PHASES]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.2, columnspacing=1.0)
    fig.suptitle("Where the time goes: best stock arm vs TPUFlex, fresh host", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig2-phases")


def fig_e2e(m: dict, out: Path) -> None:
    cells = m["e2e"]["cells"]
    bars = [("tpu-jax-e2e", "e2e-stock-best", "vLLM-JAX, best stock\n(stock-runai-cc)"),
            ("tpu-jax-e2e", "e2e-pm", "vLLM-JAX, TPUFlex"),
            ("tpu-torch-e2e", "e2e-stock-best", "TorchTPU, best stock\n(stock-ckpt-cc)"),
            ("tpu-torch-e2e", "e2e-pm", "TorchTPU, TPUFlex")]
    segs = [("VM created → node ready", "#bdc1c6"), ("TPU allocatable", "#9aa0a6"),
            ("image pull + Ray join", "#80868b")]
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ys = [0, 1, 2.6, 3.6]
    for y, (cell, arm, _) in zip(ys, bars):
        v = cells[cell]["arms"][arm]
        g = v["gke"]
        marks = [0.0, g["provision"], g["accel_ready"], g["total"]]
        for (label, color), a, b in zip(segs, marks[:-1], marks[1:]):
            ax.barh(y, b - a, left=a, height=0.62, color=color, edgecolor="white", lw=0.4)
        rest = v["median"] - g["total"]
        ax.barh(y, rest, left=g["total"], height=0.62, color=PM if arm == "e2e-pm" else GRAY_DARK, edgecolor="white",
                lw=0.4)
        ax.text(g["total"] / 2, y, f"GKE + Ray join {g['total']:.0f} s", ha="center", va="center", fontsize=7.5,
                color="#202124", fontweight="bold")
        ax.text(g["total"] + rest / 2, y, f"{rest:.0f} s", ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")
        ax.text(v["median"] + 5, y, f"{v['median']:.1f} s  (n={v['n']})", va="center", fontsize=8.5,
                fontweight="bold")
    for cell, y in (("tpu-jax-e2e", 0.5), ("tpu-torch-e2e", 3.1)):
        a = cells[cell]["arms"]
        ax.text(690, y, f"{a['e2e-stock-best']['median'] / a['e2e-pm']['median']:.2f}× faster", va="center",
                ha="right", fontsize=10, fontweight="bold", color=PM)
    ax.set_yticks(ys, [b[2] for b in bars], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 700)
    ax.grid(axis="x", color="#e8eaed", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("seconds from the GKE scale-up trigger (medians of each milestone)\n"
                  "TorchTPU's last segment also ships the venv and, for TPUFlex, runs the per-pair chip-map probe "
                  "(median 23.7 s)", fontsize=8)
    handles = [Patch(color=c, label=l) for l, c in segs] + [
        Patch(color=GRAY_DARK, label="stock: delivery + engine start → ready"),
        Patch(color=PM, label="TPUFlex: delivery + engine start → ready")]
    ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.45, -0.32))
    ax.set_title("TPU end to end on newly provisioned 2-host 2x4 v5e slices (TP8)", loc="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig3-e2e")


def fig_loo(m: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    used = []
    for ax, cell in zip(axes.flat, CELLS):
        cl = m["loo"][cell]
        full = cl["full_in_session"]
        t = m["table"][cell]
        stock = t["arms"][t["best_stock"]]["median"]
        arms = sorted((a for a in cl["arms"] if a != "pm-gcs"), key=lambda a: -cl["arms"][a]["median"])
        y = 0
        ax.barh(y, full, height=0.66, color=PM)
        runs = cl["arms"]["pm-gcs"].get("runs") or []
        ax.text(max([full] + runs) + stock * 0.015, y, f"{full:.1f}", va="center", fontsize=8, fontweight="bold")
        ax.plot(runs, [y] * len(runs), "o", ms=2.2, color="black", alpha=0.55, zorder=3)
        ticks = [(y, "full TPUFlex (in session)")]
        for arm in arms:
            y += 1
            v = cl["arms"][arm]
            cls = LOO_CLASS[arm]
            used.append(cls)
            ax.barh(y, full, height=0.66, color=PM_PALE)
            ax.barh(y, v["median"] - full, left=full, height=0.66, color=CLASS_COLOR[cls])
            runs = v.get("runs") or []
            ax.plot(runs, [y] * len(runs), "o", ms=2.2, color="black", alpha=0.55, zorder=3)
            ax.text(max([v["median"]] + runs) + stock * 0.015, y, f"+{v['median'] - full:.1f}", va="center", fontsize=8)
            ticks.append((y, LOO_LABEL[arm]))
        y += 1
        ax.barh(y, stock, height=0.66, color=GRAY_DARK)
        ax.text(stock + stock * 0.012, y, f"{stock:.1f}", va="center", fontsize=8, fontweight="bold")
        ticks.append((y, f"best stock ({t['best_stock']}, matrix)"))
        ax.axvline(full, color=PM, lw=0.8, ls="--", zorder=0)
        ax.set_yticks([p for p, _ in ticks], [l for _, l in ticks], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, stock * 1.13)
        ax.grid(axis="x", color="#e8eaed", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(TITLE[cell], loc="left")
    for ax in axes[1]:
        ax.set_xlabel("ready time (s), median of n=3; dots = runs; label = Δ vs full")
    order = [c for c in CLASS_COLOR if c in used]
    fig.legend(handles=[Patch(color=CLASS_COLOR[c], label=c) for c in order], loc="lower center", ncol=len(order),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.07), title="mechanism removed", title_fontsize=8)
    fig.suptitle("Leave-one-out: full TPUFlex minus one mechanism (fresh host, matrix configuration)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig4-loo")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matrix", default="results/matrix-2x2/matrix.json")
    ap.add_argument("--out", default="results/matrix-2x2/figures")
    args = ap.parse_args()
    m = json.loads(Path(args.matrix).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_ready(m, out)
    fig_phases(m, out)
    fig_e2e(m, out)
    fig_loo(m, out)
    print("\n".join(sorted(str(p) for p in out.iterdir())))


if __name__ == "__main__":
    main()
