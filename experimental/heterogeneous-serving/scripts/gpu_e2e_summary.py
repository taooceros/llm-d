#!/usr/bin/env python3
"""GPU end-to-end summary: GKE scale-up trigger -> ready on a freshly provisioned spot H100 (a3-highgpu-1g), shaped
like one cell of the TPU e2e matrix_summary.json (results/tpu-e2e/matrix-*/) so that matrix_2x2.py renders both.

Input: the runs.jsonl (+ <pod>-k8s.json records) a cell's driver wrote with fresh nodes (no --node):
  gpu-torch  ~/hetero-wt/GpuAutoscale: scripts/gpu_autoscale.py run --arms runai-cc,pm-gcs --pool gpu-pool-spot --zone Z
  gpu-jax    ~/hetero-wt/GpuJax:       scripts/gpu_jax_autoscale.py run --units stock-runai-cc,pm-gcs --pool ... --zone Z

T0 = the driver's submission of the GPU pod while no schedulable free GPU node exists (pending pod -> cluster
autoscaler). A run counts only if it passed the cell's ready check (greedy reply) and its node was created after T0.
Milestones are seconds from T0 (API timestamps are whole seconds); `gke` keeps the TPU e2e keys:
  provision = node Ready, accel_ready = nvidia.com/gpu allocatable first seen by the driver, pull = kubelet image pull
  duration (Pulling -> Pulled), total = container started (gpu-torch) or Ray worker joined (gpu-jax: MARK setup_done;
  the engine then runs as a Ray job on that worker, as on TPU).
Run total = T0 -> greedy reply = gke.total + control (-> the cell's in-pod T0) + the engine phases.

usage: python3 scripts/gpu_e2e_summary.py --cell gpu-torch|gpu-jax --dir RUNS_DIR --commit REV
                                          [--other OTHER_CELL_RUNS_DIR] [--note TEXT] --out RUNS_DIR/e2e_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path

REPO = Path.home() / "hetero-research"
CELLS = {
    "gpu-torch": {"matrix": ("research/matrix-gpu-torch", "results/gpu-autoscale/matrix-f6149be/matrix_summary.json"),
                  "arms": {"e2e-stock-best": "runai-cc", "e2e-pm": "pm-gcs"},
                  "source": {"e2e-stock-best": "stock-runai-cc", "e2e-pm": "pm-gcs"}},
    "gpu-jax": {"matrix": ("research/matrix-gpu-jax", "results/gpu-jax/matrix-bfb02b9/matrix_summary.json"),
                "arms": {"e2e-stock-best": "stock-runai-cc", "e2e-pm": "pm-gcs"},
                "source": {"e2e-stock-best": "stock-runai-cc", "e2e-pm": "pm-gcs"}},
}
PHASES = ("download", "spawn", "init_device", "lazy", "load_model", "compile", "kv", "graphs", "tail")
MILESTONES = ("scale_up_triggered", "node_created", "node_ready", "gpu_allocatable_seen", "pod_scheduled",
              "pull_start", "pulled", "container_started")
EXPECTED = "The primary colors are red, yellow, and blue."
TRIGGER_MAX_S = 60  # a clean scale-up: TriggeredScaleUp within a minute of T0 (the TPU e2e saw up to ~12 s)


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def git_json(ref, path):
    return json.loads(subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"], check=True,
                                     capture_output=True, text=True).stdout)


def k8s_record(d: Path, r: dict) -> dict:
    pod = r.get("pod") or f"as-{r['run_id']}"
    for name in (f"{pod}-k8s.json", f"{r['run_id']}-k8s.json"):
        if (d / name).exists():
            return json.loads((d / name).read_text())
    return {}


def image_bytes(rec: dict):
    for e in (rec.get("events") or {}).get("items", []):
        m = re.search(r"Image size: (\d+) bytes", e.get("message", ""))
        if e.get("reason") == "Pulled" and m:
            return int(m.group(1))
    return None


def scale_up_failures(rec: dict) -> list[str]:
    """FailedScaleUp events of the pod (e.g. 'GCE out of resources'): the autoscaler backed off and retried, so the
    provisioning time includes a stockout wait and the run is not a clean scale-up measurement."""
    return [e.get("message", "")[:160] for e in (rec.get("events") or {}).get("items", [])
            if e.get("reason") == "FailedScaleUp"]


def run_row(cell: str, d: Path, r: dict) -> dict:
    t_submit = r.get("t_submit") or r.get("t_pod_submit")
    marks, gke = r.get("marks") or {}, r.get("gke") or {}
    end = marks.get("check_done")
    joined = (marks["setup_done"] - t_submit) if cell == "gpu-jax" and "setup_done" in marks and t_submit \
        else gke.get("container_started")
    t0 = marks.get("download_start")
    pull = (gke["pulled"] - gke["pull_start"]) if gke.get("pulled") is not None and gke.get("pull_start") is not None \
        else None
    text = r.get("check_text")
    fresh = gke.get("node_created") is not None and gke["node_created"] > 0
    rec = k8s_record(d, r)
    stockout = scale_up_failures(rec)
    trig = gke.get("scale_up_triggered")
    if trig is None or trig > TRIGGER_MAX_S:  # the autoscaler sat in backoff after another pod's failed scale-up
        stockout = stockout + [f"TriggeredScaleUp {'never' if trig is None else f'{trig:.0f} s'} after T0 "
                               f"(> {TRIGGER_MAX_S} s: the autoscaler did not scale up at once)"]
    row = {"run_id": r["run_id"], "arm": r["arm"], "node": r.get("node"), "zone": r.get("zone"),
           "t_submit": t_submit,
           "ok": bool(r.get("ok")) and text == EXPECTED and fresh and end is not None and not stockout,
           "fresh_node": fresh, "check_text": text, "scale_up_failures": stockout,
           "ready_s": round(end - t_submit, 2) if end and t_submit else None,
           "joined_s": round(joined, 2) if joined is not None else None,
           "control_s": round(t0 - t_submit - joined, 2) if t0 and t_submit and joined is not None else None,
           "engine_ready_s": r.get("ready_s"),  # the cell's own fresh-host ready (from its in-pod T0)
           "gke_detail": {k: (round(gke[k], 2) if isinstance(gke.get(k), (int, float)) else gke.get(k))
                          for k in (*MILESTONES, "pull_reported")},
           "pull_s": round(pull, 2) if pull is not None else None,
           "image_bytes": image_bytes(rec),
           "phases": r.get("segments") or {}, "error": r.get("error")}
    if not row["ok"]:
        row["cause"] = (r.get("error") or ("node created before T0" if not fresh else None)
                        or (("GCE stockout (FailedScaleUp: GCE out of resources)" if any("out of resources" in m for m in stockout)
                         else "scale-up delayed, no GCE error on this pod [likely: the previous run's retired node still carried "
                              "the gpu-autoscale/used label, so the autoscaler's node-group template did not match]")
                        + ": " + "; ".join(stockout) if stockout else None)
                        or ("greedy check failed" if text != EXPECTED else "no ready"))
    return row


def overlaps(rows, others):
    """Other runs (either cell) whose provisioning window [T0, node Ready] intersects this run's."""
    def window(x):
        nr = x["gke_detail"].get("node_ready")
        return (x["t_submit"], x["t_submit"] + nr) if x["t_submit"] and nr is not None else None
    for x in rows:
        wx = window(x)
        x["provisioning_overlapped_with"] = sorted(
            y["run_id"] for y in rows + others if y is not x and wx and window(y)
            and window(y)[0] < wx[1] and wx[0] < window(y)[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--other", help="the other GPU cell's runs dir (provisioning-overlap check)")
    ap.add_argument("--other-cell", choices=CELLS)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec, d = CELLS[a.cell], Path(a.dir)
    wanted = set(spec["arms"].values())
    rows = [run_row(a.cell, d, r) for r in map(json.loads, (d / "runs.jsonl").read_text().splitlines())
            if r.get("arm") in wanted and "run_id" in r]
    others = []
    if a.other:
        od = Path(a.other)
        oc = a.other_cell or next(c for c in CELLS if c != a.cell)
        ow = set(CELLS[oc]["arms"].values())
        others = [run_row(oc, od, r) for r in map(json.loads, (od / "runs.jsonl").read_text().splitlines())
                  if r.get("arm") in ow and "run_id" in r]
    overlaps(rows, others)
    matrix = git_json(*spec["matrix"])
    arms = {}
    for name, campaign_arm in spec["arms"].items():
        good = [x for x in rows if x["arm"] == campaign_arm and x["ok"]]
        g = lambda k: med([x["gke_detail"].get(k) for x in good])  # noqa: E731
        arms[name] = {
            "source_arm": spec["source"][name], "status": "ok" if len(good) >= 3 else ("partial" if good else "dnf"),
            "n": len(good), "median": med([x["ready_s"] for x in good]),
            "runs": [x["ready_s"] for x in good], "run_ids": [x["run_id"] for x in good],
            "gke": {"provision": g("node_ready"), "accel_ready": g("gpu_allocatable_seen"),
                    "pull": med([x["pull_s"] for x in good]), "total": med([x["joined_s"] for x in good])},
            "gke_detail": {k: g(k) for k in MILESTONES},
            "control": med([x["control_s"] for x in good]),
            "engine_ready_from_cell_t0": med([x["engine_ready_s"] for x in good]),
            "phases": {p: med([x["phases"].get(p) for x in good]) for p in PHASES},
            "image_bytes": med([x["image_bytes"] for x in good]),
            "nodes": [x["node"] for x in good],
            "after_join": med([x["ready_s"] - x["joined_s"] for x in good if x["joined_s"] is not None]),
            "after_join_runs": [round(x["ready_s"] - x["joined_s"], 2) for x in good if x["joined_s"] is not None],
            "matrix_fresh_host_median": matrix["arms"][spec["source"][name]]["median"],
        }
    best, pm = arms["e2e-stock-best"], arms["e2e-pm"]
    # GKE provisioning does not depend on the arm (same pool, zone, image); pooling both arms' counted runs gives
    # a less noisy GKE estimate. ready_pooled = pooled GKE median + the arm's after-join median (derived, not a run).
    pooled = med([x["joined_s"] for x in rows if x["ok"]])
    for v in arms.values():
        v["ready_pooled_gke"] = round(pooled + v["after_join"], 2) if pooled is not None and v["after_join"] else None
    best, pm = arms["e2e-stock-best"], arms["e2e-pm"]
    out = {
        "cell": f"{a.cell}-e2e", "hardware": "H100 80GB: a NEW spot a3-highgpu-1g node per run (pool gpu-pool-spot, "
                                             "zone asia-southeast1-c), provisioned by the GKE cluster autoscaler",
        "tp": 1, "hosts": 1, "engine": matrix.get("engine"), "versions": matrix.get("versions"),
        "image_or_venv": matrix.get("image_or_venv"), "code_commit": a.commit,
        "t0": "Scale-up trigger: the GPU pod is created while no schedulable free GPU node exists (pending pod -> "
              "cluster autoscaler). Ready = the cell's ready check (greedy reply). Every counted run is on a node "
              "created after T0; the node is deleted from its MIG after the run.",
        "arms": arms,
        "speedup_total": round(best["median"] / pm["median"], 2) if best["median"] and pm["median"] else None,
        "speedup_after_join": (round(best["after_join"] / pm["after_join"], 2)
                               if best["after_join"] and pm["after_join"] else None),
        "gke_total_pooled": {"median": pooled, "n": sum(1 for x in rows if x["ok"]),
                             "runs": sorted(x["joined_s"] for x in rows if x["ok"])},
        "speedup_pooled_gke": (round(best["ready_pooled_gke"] / pm["ready_pooled_gke"], 2)
                               if best["ready_pooled_gke"] and pm["ready_pooled_gke"] else None),
        "runs": rows, "failed_attempts": [x for x in rows if not x["ok"]], "note": a.note,
    }
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for name, v in arms.items():
        print(name, v["status"], v["n"], v["median"], v["runs"], "gke", v["gke"], "control", v["control"])
    print("speedup total", out["speedup_total"], "after join", out["speedup_after_join"], "pooled gke",
          out["gke_total_pooled"], out["speedup_pooled_gke"],
          "failed", [(x["run_id"], x.get("cause")) for x in out["failed_attempts"]])


if __name__ == "__main__":
    main()
