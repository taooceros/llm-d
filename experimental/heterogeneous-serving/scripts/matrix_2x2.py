#!/usr/bin/env python3
"""Assemble the 2x2 cold-start matrix (hardware x serving stack) from the committed cell summaries.

Each cell is measured on its own branch; this script reads every cell's matrix_summary.json straight from
git (no checkout needed), records the exact commit it came from, normalizes the one definition that differs
between cells (warm-pm's T0), and writes:

  results/matrix-2x2/matrix.json   the four cells + the TPU e2e cells, verbatim, plus the normalized table
  results/matrix-2x2/README.md     tables + comparability notes

Usage: python3 scripts/matrix_2x2.py [--repo ~/hetero-research] [--out results/matrix-2x2]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

# (git ref, path of the cell's summary). Bump the path when a cell regenerates its summary.
SOURCES = {
    "tpu-jax": ("research/matrix-tpu-jax", "results/switch/matrix-966399c/matrix_summary.json"),
    "tpu-torch": ("research/matrix-tpu-torch", "results/torchtpu-load/matrix-fc75e5c/matrix_summary.json"),
    "gpu-jax": ("research/matrix-gpu-jax", "results/gpu-jax/matrix-bfb02b9/matrix_summary.json"),
    "gpu-torch": ("research/matrix-gpu-torch", "results/gpu-autoscale/matrix-b2d07e03/matrix_summary.json"),
}
E2E_SOURCE = ("research/matrix-tpu-e2e", "results/tpu-e2e/matrix-6a4aef4/matrix_summary.json")
# GPU end to end (one fresh spot H100 per run, GKE trigger -> ready), one e2e_summary.json per cell
# (scripts/gpu_e2e_summary.py, same arm shape as a TPU e2e cell). None = not run yet.
GPU_E2E_SOURCES = {
    "gpu-jax": ("research/matrix-gpu-jax", "results/gpu-jax/e2e-a3b6c0e3/e2e_summary.json"),
    "gpu-torch": ("research/matrix-gpu-torch-e2e", "results/gpu-autoscale/e2e-3242634a/e2e_summary.json"),
}
# Leave-one-out at the matrix configuration: per cell, full TPUFlex (pm-gcs) rerun in-session plus
# "pm-gcs minus one mechanism" arms, n=3 each, interleaved. None = not run yet.
LOO_SOURCES = {
    "tpu-jax": ("research/matrix-tpu-jax", "results/switch/loo-8450a8cc/loo_summary.json"),
    "tpu-torch": ("research/matrix-tpu-torch", "results/torchtpu-load/loo-a74a875/loo_summary.json"),
    "gpu-jax": ("research/matrix-gpu-jax", "results/gpu-jax/loo-9bcca7c/loo_summary.json"),
    "gpu-torch": ("research/matrix-gpu-torch", "results/gpu-autoscale/loo-b2d07e03/loo_summary.json"),
}
# Factorial follow-ups (same loo_summary.json schema plus a `factorial` block), shown under the cell's LOO table.
LOO_FACTORIAL_SOURCES = {
    "gpu-jax": ("research/matrix-gpu-jax", "results/gpu-jax/loo-factorial-a3b6c0e3/loo_summary.json"),
}
# Mechanism class of every LOO arm, for the cross-cell table.
LOO_CLASS = {
    "loo-snapshot": "weight snapshot",
    "loo-bank": "compile state", "loo-xla": "compile state", "loo-bank-xla": "compile state",
    "loo-fast-graph": "model graph build", "loo-sampler-bank": "compile state", "loo-replay-preload": "compile state",
    "loo-cache0": "compile state", "loo-bytecode": "process start",
    "loo-zygote": "process start", "loo-engine-zygote": "process start", "loo-ray-zygote": "process start",
    "loo-overlap": "overlapped delivery",
    "loo-kvpin": "KV pin", "loo-kvpin-only": "KV pin",
    "loo-warmup": "warm-up / tail",
    "loo-read-threads": "restore I/O",
}
# Arms kept in the per-cell tables but left out of the cross-cell table because a later arm supersedes them.
LOO_SUPERSEDED = {
    "gpu-torch": {"loo-kvpin": "superseded by loo-kvpin-only: it also swapped the Triton JIT cache (stock cache exported "
                                "after a cold boot, so the 19 attention kernels recompiled during graph capture)"},
}
LOO_CLASSES = ["weight snapshot", "compile state", "model graph build", "process start", "overlapped delivery",
               "KV pin", "warm-up / tail", "restore I/O"]
LOO_NOTES = """\
Reading the leave-one-out tables:

- **Deltas are not additive.** Mechanisms overlap (bank and XLA cache on TPU-JAX), share a critical path with delivery
  (the GPU fetch slows the zygote by ~6 s), or depend on each other. TPU-JAX loo-bank-xla shows the overlap directly:
  either compile mechanism alone is worth 24-32 s, both together 186 s.
- **Hosts.** TPU arms ran on the shared slice after GKE's 2026-10-24 auto-upgrade moved its Ray workers to the spot
  4x8 pool: TPU-JAX on pair .116/.39, TorchTPU on pair .117/.118. GPU arms ran on spot H100 nodes (on-demand creation
  failed with GCE internal errors), one held node per cell, a new wiped pod per run.
- **tpu-jax** (code 8450a8cc; `hetero/*` unchanged from 966399c):
  - loo-snapshot needs a bank seeded under Run:ai, because the bank key includes the checkpoint path. A second seed
    boot wrote it on top of the main seed's XLA cache; that seed added no XLA entries.
  - loo-fast-graph is a seventh arm (`fast_graph` was a cheap toggle).
  - The full arm's spawn ran 2-3 s high, so totals understate each removal by about that much; phase deltas do not.
- **tpu-torch** (code a74a875):
  - loo-snapshot keeps the overlapped checkpoint download, and loses the AOT and bank preloads, which run under the restore.
  - Decode is 75.3 tok/s in every arm (one torch thread per worker with or without the Ray zygote).
- **gpu-jax** (code 9bcca7c6): in loo-snapshot the 0 s cache still serves the ~2.55k weight-conversion programs
  (3.6-4.0 s instead of 37.5-39.3 s of compile in stock-runai-cc). The snapshot x cache factorial (loo-factorial-a3b6c0e3,
  n=3 per cell) measures it: the snapshot is worth +98.7 s with the 0 s cache and +132.4 s with the stock 1 s cache,
  and with both removed TPUFlex is back at stock (240.5 s vs stock-runai-cc 239.6 s).
- **gpu-torch** (code 279604f6; loo-kvpin-only at b2d07e03, its own in-session pm-gcs baseline 46.16 s):
  - The KV pin is part of the compile key, so a pin-free arm cannot use the bank. loo-kvpin-only uses instead the warm
    stock cache of the unpinned key, which holds the same 19 Triton kernels as the bank: **the pin alone is +4.2 s**
    (kv +2.1, download +1.6, compile +0.6; capture unchanged at 8 s).
  - The earlier loo-kvpin (+14.7 s) also swapped in a stock cache exported after a cold boot, without `~/.triton/cache`, so
    about 10.5 s of it was Triton recompilation during graph capture (FULL decode round 7 s instead of 3 s). It is kept in
    the per-cell table, marked superseded.
  - The bank's content is the same kind of cache stock ships, so loo-bank's +1.7 s is only its overlapped delivery.
  - The deltas (with the pin-only value) sum to 38.4 s against a 44.2 s in-session gain over the old-cache stock-runai-cc
    (89.1 s in session); against the warm-cache stock-runai-cc (79.8 s) the gain is 36.7 s.
"""

CELLS = ["tpu-jax", "tpu-torch", "gpu-jax", "gpu-torch"]
TITLE = {"tpu-jax": "TPU × vLLM-JAX", "tpu-torch": "TPU × vLLM-TorchTPU",
         "gpu-jax": "GPU × vLLM-JAX", "gpu-torch": "GPU × CUDA vLLM"}
STOCK = ["stock-lazy", "stock-ckpt", "stock-ckpt-cc", "stock-runai", "stock-runai-cc"]
FRESH = STOCK + ["pm-gcs", "pm-peer"]
PHASES = ["download", "spawn", "init_device", "lazy", "load_model", "compile", "kv", "graphs", "tail"]


def git_show(repo: Path, ref: str, path: str) -> tuple[dict, str]:
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", ref], check=True,
                            capture_output=True, text=True).stdout.strip()
    text = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], check=True,
                          capture_output=True, text=True).stdout
    return json.loads(text), commit


def warm_pm(cell: str, d: dict) -> dict:
    """warm-pm from launch and from gate open. Cells chose different canonical T0s; the other one is
    recorded in a cell-specific field, named here explicitly rather than guessed."""
    a = d["arms"]["warm-pm"]
    if cell == "tpu-jax":  # canonical T0 = launch; gate in extras
        return {"launch": a["median"], "gate": d["extras"]["warm-pm@gate"]["median"]}
    if cell == "tpu-torch":  # canonical T0 = launch (zygotes); gated variant (no zygotes) in extra_arms
        return {"launch": a["median"], "gate": d["extra_arms"]["warm-pm gated config B (T0=gate)"]["median"]}
    if cell == "gpu-jax":  # canonical T0 = gate open
        return {"launch": a["t0_launch"]["ready_from_launch_median"], "gate": a["median"]}
    if cell == "gpu-torch":  # canonical T0 = gate open
        return {"launch": a["median_from_launch"], "gate": a["median"]}
    raise KeyError(cell)


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def build(repo: Path) -> dict:
    out = {"sources": {}, "cells": {}, "table": {}}
    for cell in CELLS:
        ref, path = SOURCES[cell]
        d, commit = git_show(repo, ref, path)
        out["sources"][cell] = {"ref": ref, "commit": commit, "path": path}
        out["cells"][cell] = d
        arms = {}
        for arm, v in d["arms"].items():
            arms[arm] = {k: v.get(k) for k in ("status", "n", "median", "runs")}
            arms[arm]["phases"] = v.get("phases")
        ok_stock = {a: arms[a]["median"] for a in STOCK if arms[a]["status"] == "ok"}
        best = min(ok_stock, key=ok_stock.get)
        wp = warm_pm(cell, d)
        out["table"][cell] = {
            "arms": arms,
            "warm_pm": wp,
            "best_stock": best,
            "speedup_fresh": ok_stock[best] / arms["pm-gcs"]["median"],
            "speedup_warm_launch": arms["warm-stock"]["median"] / wp["launch"],
            "speedup_warm_gate": arms["warm-stock"]["median"] / wp["gate"],
        }
    e2e, commit = git_show(repo, *E2E_SOURCE)
    out["sources"]["tpu-e2e"] = {"ref": E2E_SOURCE[0], "commit": commit, "path": E2E_SOURCE[1]}
    out["e2e"] = e2e
    for cell, src in GPU_E2E_SOURCES.items():
        if src is None:
            continue
        d, commit = git_show(repo, *src)
        out["sources"][f"{cell}-e2e"] = {"ref": src[0], "commit": commit, "path": src[1]}
        out["e2e"]["cells"][d["cell"]] = d
    out["loo"] = {}
    for cell in CELLS:
        if LOO_SOURCES[cell] is None:
            continue
        d, commit = git_show(repo, *LOO_SOURCES[cell])
        out["sources"][f"{cell}-loo"] = {"ref": LOO_SOURCES[cell][0], "commit": commit, "path": LOO_SOURCES[cell][1]}
        full = d["arms"][d.get("baseline_arm", "pm-gcs")]["median"]
        arms = {}
        for arm, v in d["arms"].items():
            arms[arm] = {k: v.get(k) for k in ("status", "n", "median", "runs", "removed", "fallback", "phases")}
            base = (v.get("baseline_session") or {}).get("median", full)
            arms[arm]["delta_vs_full"] = None if v.get("median") is None else v["median"] - base
            arms[arm]["baseline_median"] = base
            arms[arm]["superseded"] = LOO_SUPERSEDED.get(cell, {}).get(arm)
            arms[arm]["baseline_phases"] = (v.get("baseline_session") or {}).get("phases")
        out["loo"][cell] = {"summary": d, "full_in_session": full,
                            "full_matrix": out["table"][cell]["arms"]["pm-gcs"]["median"], "arms": arms}
        if cell in LOO_FACTORIAL_SOURCES:
            fd, fcommit = git_show(repo, *LOO_FACTORIAL_SOURCES[cell])
            out["sources"][f"{cell}-loo-factorial"] = {"ref": LOO_FACTORIAL_SOURCES[cell][0], "commit": fcommit,
                                                       "path": LOO_FACTORIAL_SOURCES[cell][1]}
            out["loo"][cell]["factorial"] = {
                "summary": fd, "effects": fd["factorial"],
                "arms": {a: {k: v.get(k) for k in ("n", "median", "runs", "removed", "fallback", "phases")}
                         for a, v in fd["arms"].items()}}
    return out


NOTES = """\
## How to read this

- **Fresh host** (stock-\\*, pm-gcs, pm-peer): the engine starts with nothing on the host except the image/venv.
  T0 is the first byte of delivery, or the engine launch when nothing is delivered. In the pm arms delivery overlaps
  the engine spawn, so their phases do not sum to ready. Hosts are existing VMs with state wiped:
  - TPU: 2-host sub-pairs of the shared 4x8 slice, with run scopes wiped.
  - GPU: a held spot node, with a wipe plus page-cache eviction and a receipt for every run.
  - GKE provisioning is excluded here. It appears only in the TPU e2e table, and is not measured on GPU (user decision during the H100 stockout).
- **Same host** (warm-\\*): restart on the same host with everything local. warm-pm is pre-spawned, so it is given
  both from the engine launch and from the gate opening (the moment the replaced engine is gone). Each cell picked one
  canonical T0; the other comes from the cell's own record (`warm_pm()` in the script names the field).
- **Ready** = `/health` OK and the greedy check passed: same prompt, temperature 0, expected
  "The primary colors are red, yellow, and blue." On TPU vLLM-JAX it is 32 prompts token by token against the seed reference.
- **Engine shape** in every cell: `--max-model-len 8192 --max-num-seqs 512 --max-num-batched-tokens 2048`, Gemma-4 31B bf16.
  TP differs by hardware: TPU TP8 (2 hosts × 4 v5e chips), GPU TP1 (one H100 80 GB). Compare
  stacks within one hardware row, and across hardware only as ratios.
- **stock** is the unmodified stack at its defaults. For the JAX stacks that includes the 1 s persistent-cache threshold
  (TPU vLLM-JAX with a 0 s threshold: stock-ckpt-cc 297.9 s, in that cell's extras).
  - "cc" = the stack's own compile cache, from a peer or GCS.
  - best stock = the fastest ok stock arm.

## Deviations and caveats per cell (details in each cell's README)

- **tpu-jax**:
  - Run:ai arms use `--tokenizer /models/gemma-4-31b` and an 8 GiB streamer limit. At defaults, vLLM copied about 80 GB of stale non-weight objects from the bucket prefix to local disk; the resulting disk-pressure eviction recreated every TPU pod (`stock-runai-default` = dnf, in extras).
  - The three pairs ran side by side and shared GCS bandwidth.
- **tpu-torch**:
  - PM = config B: sampler bank + PM_SB_PRESEED + canonical chip binding + Ray worker zygote under OMP=1.
  - The sampler bank had baked in the seed's chip order and gave wrong text on other slices: the vocab all-gather order was fixed at compile time, which produced ±65536 token shifts.
  - Canonical coordinate-ordered binding plus a rank→device-keyed guard makes it portable (verified on 3 VM pairs; the guard refuses a mismatch).
  - Config A rows (no bank) are in extra_arms.
  - KV is pinned (`--kv-cache-memory-bytes 7184619601`) in every arm.
  - stock-lazy: dnf at the 900 s bound (about 0.04 GB/s per host over gcsfuse).
- **gpu-jax**:
  - The Ray `OMP_NUM_THREADS=1` inflation is fixed (`unset`); stock load_model is about 130 s on 13 threads.
  - stock-lazy: dnf. With 13 threads every thread blocks in FUSE page faults, at about 85 s per layer (bound 1950 s).
  - Reused spot node: preempted twice; runs on two nodes overlapped from 02:29 to 03:25.
- **gpu-torch**:
  - stock-ckpt-cc and stock-runai-cc were rerun (tritonfix-b2d07e03) with stock compile caches exported after a warm
    boot, so `~/.triton/cache` holds vLLM's 19 Triton attention kernels, as TPUFlex's bank always did. The first exports
    came from a cold-compile boot whose kernels lived only in the AOT artifact's private Inductor cache; every later boot
    recompiled them during CUDA graph capture (capture 12 s instead of 8 s, KV window 8.6 s instead of 2.3 s). Those rows
    (107.2 / 87.0 s) are in extras as `*@cold-export-cache`. The other stock arms ship no compile cache and are unchanged.
  - PM pins KV (12874852680 bytes); the pin alone is worth 4.2 s (loo-kvpin-only).
  - The GPU end-to-end stock arm (e2e-3242634a) still used the cold-export Run:ai cache (`runai-98056cd.tar`), so its
    86.1 s after the container start includes the Triton recompile; the warm cache saved 7–16 s on held nodes
    (tritonfix-b2d07e03), so that e2e stock arm is pessimistic by about that much (inferred, not rerun).
  - CUDA graphs are still captured on every boot (8 s); they cannot be banked because graphs embed device pointers.
  - Three zygote CUDA-init fixes.
- **Delivery tools differ per cell:**
  - stock-ckpt download is about 55 s on tpu-jax (GCS sliced GETs) against about 17 s on tpu-torch (Run:ai libstreamer, 3.6 GB/s per host).
  - On GPU it is about 35 s.
- **Versions differ per cell** (see `versions` in matrix.json): each cell uses the newest stack version that ran this model on that hardware.

## TPU end to end (fresh GKE capacity)

- **T0** is the trigger that takes a dedicated 2-host 2x4 pool from 0 to 1 worker. The time includes VM creation, image pull and Ray join. On TorchTPU it also includes shipping the venv and the per-pair chipmap probe.
- **Pools**: own pools with a private Ray head, one pool per lane (on-demand and spot); both deleted after every run.
- **TorchTPU config B** needs a chipmap probe on every new pair. It is counted (median 23.7 s), so the engine part is 68.4 s against 159.5 s for stock.
- **Stock fairness**:
  - The stock compile cache keys on chip order. Only 3 of 30 fresh slices matched small_d's order, so stock lost about 13.5 s on 2 of its 3 runs (`compute_selected_logits`).
  - A stock arm given canonical binding plus a canonical cache (`e2e-stock-ckpt-cc-canon`, 408.8 s) is not faster, because it then pays the chipmap. If stock always hit its cache, its median would be 382.2 s (bound).
- **Extras**: the earlier config A (no sampler bank) PM median was 333.2 s; it is in extras, along with the other superseded rows.

## GPU end to end (fresh spot H100, 2026-10-25)

- **T0** is the creation of the GPU pod while no schedulable free GPU node exists; the cluster autoscaler adds one spot
  a3-highgpu-1g in zone asia-southeast1-c, and the node is deleted after the run. Counted runs triggered a scale-up within
  60 s of T0 and saw no FailedScaleUp; stockouts and delayed scale-ups are listed in each cell's `failed_attempts`.
- **GKE dominates**: the GPU joined in 200-302 s (gpu-jax: Ray worker joined; CUDA vLLM: container started after a
  12.1 GB image pull of ~110 s), against an engine start of 44.6-245.9 s after the join.
- **Totals are noisy at n=3**: node Ready alone ranged 106-182 s. Each cell's e2e_summary.json also gives the speedup after
  the join (gpu-jax 2.63x, CUDA vLLM 1.93x) and a derived total with the GKE time pooled over both arms (1.48x, 1.13x).
"""


def render(m: dict) -> str:
    t = m["table"]
    L = ["# 2x2 cold-start matrix: Gemma-4 31B, {TPU v5e, H100} x {vLLM-JAX, vLLM-torch}", "",
         "Generated by `scripts/matrix_2x2.py` from the committed cell summaries:", ""]
    for cell, s in m["sources"].items():
        L.append(f"- `{cell}`: `{s['ref']}` @ `{s['commit'][:8]}`: `{s['path']}`")
    L += ["", "Figures (`scripts/matrix_2x2_figures.py`, PNG + PDF in `results/matrix-2x2/figures/`): `fig1-ready` ready time",
          "of every arm, `fig2-phases` best stock vs TPUFlex phases, `fig3-e2e` TPU end to end, `fig4-loo` leave-one-out."]
    L += ["", "## Ready time, median of n=3 (s)", "",
          "| arm | " + " | ".join(TITLE[c] for c in CELLS) + " |",
          "|---|" + "---|" * len(CELLS)]
    for arm in FRESH + ["warm-stock"]:
        row = []
        for c in CELLS:
            a = t[c]["arms"][arm]
            row.append(fmt(a["median"]) if a["status"] == "ok" else f"**{a['status']}**")
        L.append(f"| {arm} | " + " | ".join(row) + " |")
    L.append("| warm-pm (from launch) | " + " | ".join(fmt(t[c]["warm_pm"]["launch"]) for c in CELLS) + " |")
    L.append("| warm-pm (from gate open) | " + " | ".join(fmt(t[c]["warm_pm"]["gate"]) for c in CELLS) + " |")
    L += ["", "## TPUFlex speedup", "",
          "| | " + " | ".join(TITLE[c] for c in CELLS) + " |", "|---|" + "---|" * len(CELLS),
          "| best stock arm | " + " | ".join(f"{t[c]['best_stock']} {fmt(t[c]['arms'][t[c]['best_stock']]['median'])}" for c in CELLS) + " |",
          "| fresh host: best stock / pm-gcs | " + " | ".join(f"{t[c]['speedup_fresh']:.2f}×" for c in CELLS) + " |",
          "| same host: warm-stock / warm-pm (launch) | " + " | ".join(f"{t[c]['speedup_warm_launch']:.2f}×" for c in CELLS) + " |",
          "| same host: warm-stock / warm-pm (gate) | " + " | ".join(f"{t[c]['speedup_warm_gate']:.2f}×" for c in CELLS) + " |"]
    L += ["", "## Phases, medians (s): best stock arm vs pm-gcs", "",
          "| cell | arm | " + " | ".join(PHASES) + " |", "|---|---|" + "---|" * len(PHASES)]
    for c in CELLS:
        for arm in (t[c]["best_stock"], "pm-gcs"):
            ph = t[c]["arms"][arm]["phases"] or {}
            L.append(f"| {TITLE[c]} | {arm} | " + " | ".join(fmt(ph.get(p)) for p in PHASES) + " |")
    e = m["e2e"]["cells"]
    L += ["", "## End to end: GKE scale-up trigger → ready, on newly provisioned capacity", "",
          "TPU: a new 2-host 2x4 v5e slice per run. GPU: a new spot a3-highgpu-1g (1x H100) per run, zone asia-southeast1-c.",
          "GKE milestones are seconds from the trigger; the last one is when the engine can start: TPU hosts joined Ray,",
          "GPU-JAX the Ray worker joined, CUDA vLLM the container started.", "",
          "| cell | arm | n | median (s) | runs | GKE: node Ready / accelerator allocatable / image pull (duration) / joined | engine phases: spawn / load_model / compile | PM speedup |",
          "|---|---|---|---|---|---|---|---|"]
    for cell, d in e.items():
        arms = d["arms"]
        best, pm = arms.get("e2e-stock-best", {}).get("median"), arms.get("e2e-pm", {}).get("median")
        for arm, v in arms.items():
            g, ph = v.get("gke") or {}, v.get("phases") or {}
            sp = f"{best / pm:.2f}×" if arm == "e2e-pm" and best and pm else ""
            L.append(f"| {cell} | {arm} ({v.get('source_arm', '')}) | {v['n']} | {fmt(v['median'])} | "
                     f"{' / '.join(fmt(r) for r in (v.get('runs') or []))} | "
                     f"{fmt(g.get('provision'))} / {fmt(g.get('accel_ready'))} / {fmt(g.get('pull'))} / {fmt(g.get('total'))} | "
                     f"{fmt(ph.get('spawn'))} / {fmt(ph.get('load_model'))} / {fmt(ph.get('compile'))} | {sp} |")
    L += render_loo(m)
    L += ["", NOTES]
    return "\n".join(L)


def phase_deltas(arm: dict, full: dict, top: int = 3) -> str:
    """The largest per-phase median changes of a LOO arm against the in-session full run."""
    pa, pf = arm.get("phases") or {}, full.get("phases") or {}
    deltas = [(p, (pa.get(p) or 0.0) - (pf.get(p) or 0.0)) for p in PHASES if pa.get(p) is not None or pf.get(p) is not None]
    deltas = sorted((d for d in deltas if abs(d[1]) >= 1.0), key=lambda d: -abs(d[1]))[:top]
    return ", ".join(f"{p} {d:+.1f}" for p, d in deltas) or "–"


def render_loo(m: dict) -> list[str]:
    loo = m.get("loo") or {}
    if not loo:
        return []
    L = ["", "## Leave-one-out at the matrix configuration (n=3 per arm, interleaved)", "",
         "Each `loo-*` arm is the full fresh-host TPUFlex arm (pm-gcs) with one mechanism off; the removed phase falls",
         "back to what the cell's best stock arm does. Δ is against full TPUFlex rerun in the same session.", "",
         "| | " + " | ".join(TITLE[c] for c in CELLS) + " |", "|---|" + "---|" * len(CELLS),
         "| full TPUFlex, in session (matrix) | " + " | ".join(
             f"{fmt(loo[c]['full_in_session'])} ({fmt(loo[c]['full_matrix'])})" if c in loo else "pending" for c in CELLS) + " |"]
    for cls in LOO_CLASSES:
        row = []
        for c in CELLS:
            if c not in loo:
                row.append("pending")
                continue
            hits = [f"{a.removeprefix('loo-')} {v['delta_vs_full']:+.1f}" for a, v in loo[c]["arms"].items()
                    if LOO_CLASS.get(a) == cls and v.get("delta_vs_full") is not None and not v.get("superseded")]
            row.append("; ".join(hits) or "–")
        L.append(f"| {cls} | " + " | ".join(row) + " |")
    for c in CELLS:
        if c not in loo:
            continue
        cl = loo[c]
        full = cl["arms"]["pm-gcs"]
        L += ["", f"### {TITLE[c]}", "", "| arm | removed | fallback | n | median (s) | runs | Δ vs full (s) | largest phase changes |",
              "|---|---|---|---|---|---|---|---|"]
        for arm, v in sorted(cl["arms"].items(), key=lambda kv: -(kv[1]["delta_vs_full"] or 0)):
            runs = " / ".join(fmt(r) for r in (v.get("runs") or []))
            delta = "–" if arm == "pm-gcs" else f"{v['delta_vs_full']:+.1f}"
            ref = {"phases": v["baseline_phases"]} if v.get("baseline_phases") else full
            changes = "–" if arm == "pm-gcs" else phase_deltas(v, ref)
            if v.get("baseline_phases"):
                delta += f" (vs its own session's pm-gcs, {fmt(v['baseline_median'])} s)"
            name = f"{arm} (superseded)" if v.get("superseded") else arm
            L.append(f"| {name} | {v.get('removed') or '–'} | {v.get('fallback') or '–'} | {v['n']} | {fmt(v['median'])} | "
                     f"{runs} | {delta} | {changes} |")
        fx = cl.get("factorial")
        if fx:
            e, fa = fx["effects"], fx["arms"]
            L += ["", f"Factorial follow-up (`{m['sources'][c + '-loo-factorial']['path']}`), separate session, n=3 per cell:", "",
                  "| | 0 s cache (TPUFlex) | stock 1 s cache |", "|---|---|---|",
                  f"| snapshot on | pm-gcs {fmt(fa['pm-gcs']['median'])} | loo-cache0 {fmt(fa['loo-cache0']['median'])} |",
                  f"| snapshot off | loo-snapshot {fmt(fa['loo-snapshot']['median'])} | loo-snapshot-cache0 {fmt(fa['loo-snapshot-cache0']['median'])} |",
                  "", f"Snapshot worth +{e['snapshot_worth_with_pm_cache']} s with the 0 s cache and +{e['snapshot_worth_with_stock_cache']} s "
                  f"with the stock cache; the 0 s cache worth +{e['cache_worth_with_snapshot']} / +{e['cache_worth_without_snapshot']} s; "
                  f"interaction {e['interaction']} s."]
    return L + ["", LOO_NOTES]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=os.path.expanduser("~/hetero-research"))
    ap.add_argument("--out", default="results/matrix-2x2")
    args = ap.parse_args()
    m = build(Path(args.repo))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix.json").write_text(json.dumps(m, indent=1, sort_keys=False) + "\n")
    (out / "README.md").write_text(render(m) + "\n")
    print(render(m))


if __name__ == "__main__":
    main()
