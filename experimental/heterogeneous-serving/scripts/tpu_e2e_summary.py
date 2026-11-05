#!/usr/bin/env python3
"""Cross-stack startup matrix, cells ``tpu-jax-e2e`` / ``tpu-torch-e2e``: a TP8 instance on NEW TPU capacity.

    .venv/bin/python scripts/tpu_e2e_summary.py --commit <c> --out results/tpu-e2e/matrix-<c>/matrix_summary.json

Samples are scripts/tpu_e2e.py run directories (results/tpu-e2e/{jax,torch}/<rid>/sample.json + evidence).
T0 = the scale-up trigger (the patch of the private RayCluster's TPU worker group replicas 0 -> 1); ready =
GET /health 200 on the new instance's endpoint AND the greedy check passed.  Each run's total splits as

    total = gke.total (T0 -> both hosts' Ray workers joined the head)
          + control  (Ray joined -> first engine action: inventory from the template, RCM/Ray job submission,
                      and for sequential arms the job plumbing between the delivery and the deploy job)
          + sum(phases)  (the sibling cells' engine phases, same definitions: scripts/matrix_summary.py of
                          research/matrix-tpu-jax and the TorchTPU cell)

GKE stages are seconds from T0, critical path = the last of the 2 hosts (or pods) to reach the stage.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("download", "spawn", "init_device", "lazy", "load_model", "compile", "kv", "graphs", "tail")
GKE_STAGES = ("triggered", "vm_created", "node_registered", "node_ready", "tpu_allocatable", "pod_scheduled",
              "image_pulled", "container_started", "ray_joined")
JAX_CANON = {"e2e-stock-best": ("runai-jax-xla", "ckpt-gcs-jax-xla"), "e2e-pm": ("gcs-ov",)}
OVERLAP = ("gcs-ov",)


def load(directory: Path, name: str):
    for path in (directory / name, directory / (name + ".gz")):
        if path.exists():
            with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as handle:
                return json.load(handle)
    return None


def med(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 2) if values else None


def span_max(workers, phase, edge):
    values = [w["phases"][phase][edge] for w in workers if isinstance((w.get("phases") or {}).get(phase), dict)]
    return max(values) if values else None


def pull_seconds(msgs: list[str]) -> float | None:
    """Longest vllm-tpu image pull of the 2 hosts, from the kubelet's `Successfully pulled image ... in Xs`."""
    out = []
    for msg in msgs:
        if "vllm/vllm-tpu" not in msg:
            continue
        m = re.search(r" in (?:(\d+)m)?([\d.]+)s", msg)
        if m:
            out.append(int(m.group(1) or 0) * 60 + float(m.group(2)))
    return round(max(out), 1) if out else None


def gke_row(sample: dict) -> dict:
    g = sample["gke"]
    row = {k: g.get(k) for k in GKE_STAGES}
    row.update(provision=g.get("node_ready"), accel_ready=g.get("tpu_allocatable"),
               pull=pull_seconds(g.get("pull_msgs") or []), total=g.get("ray_joined"))
    return row


def jax_row(directory: Path) -> dict:
    sample = load(directory, "sample.json")
    eng = sample.get("engine") or {}
    arm = eng.get("arm")
    row = {"run_id": sample["rid"], "lane": sample["lane"], "arm": arm, "status": sample["status"],
           "code_commit": sample.get("code_commit")}
    if sample.get("gke"):
        row["gke"] = gke_row(sample)
    deploy = load(directory, "deploy.json")
    if deploy is None or sample["status"] != "ok":
        row["error"] = sample.get("error") or "no deploy evidence / failed run"
        return row
    t0 = sample["t0"]
    dep = deploy["measurements"]["deployment"]
    inst = dep["instances"][0]
    init = inst["initialization"]
    workers = init.get("worker_init_phases") or []
    t_deploy = dep["started_at"]
    health = (inst.get("serving_endpoint") or {}).get("health") or {}
    t_health = health["ok_at"]
    walls = [(load(directory, n) or {}).get("measurements", {}).get("wall_s") for n in
             ("ckpt-fetch.json", "xla-fetch.json", "state-fetch.json")]
    seq = sum(w for w in walls if w)
    gate = max([w["phases"]["state_gate"]["end"] - w["phases"]["state_gate"]["start"] for w in workers
                if isinstance((w.get("phases") or {}).get("state_gate"), dict)] or [0.0])
    touch = span_max(workers, "init_device", "start")
    cuts = {k: span_max(workers, p, "end") for k, p in (("dev", "init_device"), ("load", "load_model"),
                                                        ("kv", "initialize_from_config"),
                                                        ("compile", "compile_or_warm_up_model"))}
    ov = arm in OVERLAP
    phases = {"download": gate if ov else seq, "spawn": touch - t_deploy - (gate if ov else 0.0),
              "init_device": cuts["dev"] - touch, "lazy": None, "load_model": cuts["load"] - cuts["dev"],
              "kv": cuts["kv"] - cuts["load"], "compile": cuts["compile"] - cuts["kv"], "graphs": None,
              "tail": t_health - cuts["compile"]}
    total = t_health - t0
    engine_s = sum(v for v in phases.values() if v is not None)
    row.update({
        "ready_s": round(total, 2),
        "engine_s": round(engine_s, 2),
        "control_s": round(total - row["gke"]["total"] - engine_s, 2),
        "phases": {k: (None if v is None else round(v, 2)) for k, v in phases.items()},
        "greedy": eng.get("greedy"),
        "greedy_ok": bool((eng.get("greedy") or {}).get("passed")),
        "health_ok": health.get("status") == "ok",
        "overlap_fetch_wall_s": ((dep.get("state_delivery") or {}).get("fetch") or {}).get("wall_s"),
        "delivery_walls_s": dict(zip(("ckpt", "xla", "state"), walls)),
        "stage_clocks": dep.get("stage_clocks"),
        # spawn on a fresh host: deploy start -> engine actor's initialize() (Ray runtime env of the job built on
        # the new nodes, backend, placement group, actor) | -> first device touch minus gate wait (vLLM engine
        # construction, Ray executor workers, cold imports from the just-pulled image)
        "spawn_split": {"to_engine_init": round(init["engine_init_started_at"] - t_deploy, 2),
                        "engine_init_to_touch": round(touch - init["engine_init_started_at"] - (gate if ov else 0.0), 2)}
        if init.get("engine_init_started_at") else None,
        # control: Ray joined -> deploy start (geometry probe wait, RCM daemon + Ray job submission, and for
        # sequential arms the delivery jobs' plumbing beyond their remote wall)
        "control_split": {m["event"]: round(m["t"] - t0, 2) for m in sample.get("marks", [])
                          if m["event"] in ("ray_joined", "inventory_derived", "deploy_start", "deploy_end")},
        "evidence": {
            "snapshot_restored": all((w.get("weight_snapshot") or {}).get("restored") for w in workers),
            "bank_installed": all((w.get("exec_bank") or {}).get("installed") for w in workers),
            "kv_blocks": (init.get("kv_capacity") or {}).get("usable_blocks"),
            "model": (init.get("resolved_engine_config") or {}).get("model"),
            "load_format": (init.get("engine_settings") or {}).get("load_format"),
            "hosts": [n.get("internal_ip") for n in sample["gke_raw"]["nodes"]],
            "nodes": [n.get("name") for n in sample["gke_raw"]["nodes"]],
        },
    })
    return row


def arm_block(rows: list[dict], note: str, min_n: int = 3) -> dict:
    good = [r for r in rows if r.get("greedy_ok") and r.get("health_ok") and r.get("ready_s") is not None]
    block = {"status": "ok" if len(good) >= min_n else "dnf", "n": len(good),
             "runs": [r["ready_s"] for r in good], "median": med([r["ready_s"] for r in good]),
             "phases": {p: (None if p == "graphs" else med([r["phases"][p] for r in good])) for p in PHASES},
             "gke": {k: med([r["gke"][k] for r in good]) for k in ("provision", "accel_ready", "pull", "total")}
             if good else None,
             "gke_stages": {k: med([r["gke"][k] for r in good]) for k in GKE_STAGES} if good else None,
             "control": med([r["control_s"] for r in good]),
             "venv": med([r.get("venv_s") for r in good]),
             "engine": med([r["engine_s"] for r in good]),
             "by_lane": {lane: {"n": len(rs), "median": med([r["ready_s"] for r in rs]),
                                "gke_total": med([r["gke"]["total"] for r in rs])}
                         for lane in ("od", "spot") if (rs := [r for r in good if r["lane"] == lane])},
             "note": note, "run_ids": [r["run_id"] for r in good]}
    # torch rows only (the JAX cell's blocks keep their keys): the counted chipmap step, d128 decode speed
    if any("chipmap_s" in r for r in good):
        block["chipmap"] = med([r.get("chipmap_s") for r in good])
    if any("decode_d128_tok_s" in r for r in good):
        block["decode_d128_tok_s"] = med([r.get("decode_d128_tok_s") for r in good])
    failed = [r["run_id"] for r in rows if r not in good]
    if failed:
        block["failed_run_ids"] = failed
    return block


HARDWARE = ("TPU v5e: a NEW 2-host 2x4 slice (node pool e2e-v5e-2x4-{od,spot}: ct5lp-hightpu-4t x2, "
            "--tpu-topology 2x4, COMPACT placement), provisioned by the GKE cluster autoscaler per run")
T0 = ("Scale-up trigger: patch of the private RayCluster's TPU worker group replicas 0 -> 1 on an empty pool "
      "(pending pods -> cluster autoscaler). Ready = GET /health 200 on the new instance AND the greedy check "
      "passed. Every run is on 2 freshly provisioned VMs (pool deleted and recreated between runs).")


# canonical variant name of each source arm (for the extras rows)
VARIANT = {"runai-jax-xla": "stock-runai-cc", "ckpt-gcs-jax-xla": "stock-ckpt-cc", "gcs-ov": "pm-gcs",
           "stock-runai-cc": "stock-runai-cc", "stock-ckpt-cc": "stock-ckpt-cc", "pm-gcs": "pm-gcs",
           "stock-ckpt-cc-canon": "stock-ckpt-cc-canon"}


def cell(name: str, rows: list[dict], canon: dict, notes: dict, meta: dict) -> dict:
    """Each canonical arm takes the fastest (median) variant with status ok among its choices; the other
    variants are kept as extras rows (`e2e-<variant>`), dnf ones with their failed run ids."""
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    arms, extras = {}, {}
    for canon_name, choices in canon.items():
        blocks = {c: arm_block(by_arm.get(c, []), notes.get(c, "")) for c in choices}
        for c, b in blocks.items():
            b["source_arm"] = c
        ok = [c for c in choices if blocks[c]["status"] == "ok"]
        chosen = min(ok, key=lambda c: blocks[c]["median"]) if ok else choices[0]
        arms[canon_name] = blocks[chosen]
        for c in choices:
            if c != chosen and (by_arm.get(c) or blocks[c]["status"] == "ok"):
                extras[f"e2e-{VARIANT.get(c, c)}"] = blocks[c]
    # runs of a superseded engine config (torch_row: `<arm>@<config digest>`) are extras rows of their own;
    # any other arm (the canonical stock cache seed, n=1) is an extras row with its own n
    canon_arms = {c for choices in canon.values() for c in choices}
    for a, rs in sorted(by_arm.items(), key=lambda kv: str(kv[0])):
        if a in canon_arms or a is None:
            continue
        if "@" in str(a):
            base, _, digest = a.partition("@")
            block = arm_block(rs, notes.get(a) or f"superseded config {digest}: " + notes.get(base, ""))
            block["source_arm"] = a
            extras[f"e2e-{VARIANT.get(base, base)}@{digest}"] = block
        else:
            block = arm_block(rs, notes.get(a, ""), min_n=1)
            block["source_arm"] = a
            extras[f"e2e-{a}"] = block
    return {"cell": name, "hardware": HARDWARE, "tp": 8, "hosts": 2,
            "engine": {"max_model_len": 8192, "max_num_seqs": 512, "max_num_batched_tokens": 2048,
                       "extra": meta.get("engine_extra", "")},
            "versions": meta.get("versions", {}), "image_or_venv": meta.get("image_or_venv"),
            "code_commit": meta.get("code_commit"), "t0": T0, "arms": arms, "extras": extras, "runs": rows,
            "note": meta.get("note", "")}


TORCH_CANON = {"e2e-stock-best": ("stock-runai-cc", "stock-ckpt-cc", "stock-ckpt-cc-canon"), "e2e-pm": ("pm-gcs",)}
TORCH_CFG = ROOT / "results/tpu-e2e/torch/cfg.json"


def torch_env(cfg: dict, arm: str) -> list[str]:
    return [*cfg.get("host_env_common", []), *(cfg.get("host_env") or {}).get(arm, [])]


def torch_arm(sample: dict, arm: str | None) -> str | None:
    """The arm, or `<arm>@<digest>` when the run's host env differs from the current cfg.json (a superseded
    config: only runs with the final config count for the arm; the others are an extras row)."""
    if not arm or not TORCH_CFG.exists():
        return arm
    ran = torch_env(sample.get("config") or {}, arm)
    if ran == torch_env(json.loads(TORCH_CFG.read_text()), arm):
        return arm
    return f"{arm}@{hashlib.sha256(json.dumps(ran).encode()).hexdigest()[:8]}"


def zygote_report(directory: Path, sample: dict, rep: dict) -> dict:
    """TTL_RAY_ZYGOTE=1 runs: the workers are forked from the host's Ray zygote; a worker's `Worker_TPn pid=` tag
    appears only in its .err file (warnings), not in the .out file that holds the INFO milestones, so the cell's
    run_report maps rank 0 to the .err alone and spawn/init_device/load_model come out None (folded into kv).
    Re-run the same run_report with each worker's .out and .err grouped by file stem and the rank taken from the
    stem's `Worker_TPn pid=` tag, else its `PM_INIT_DEVICE rank=N` line; every milestone regex is unchanged."""
    import sys  # noqa: PLC0415
    # the run's own checkout, else the current cfg.json's (same milestone regexes; only rank detection changed)
    repos = [(sample.get("config") or {}).get("torch_repo", "")]
    if TORCH_CFG.exists():
        repos.append(json.loads(TORCH_CFG.read_text()).get("torch_repo", ""))
    repo = next((Path(r).expanduser() for r in repos
                 if r and (Path(r).expanduser() / "scripts/torchtpu_matrix_report.py").exists()), Path("/nonexistent"))
    logroot = directory / "pod_logs"
    if not (repo / "scripts/torchtpu_matrix_report.py").exists() or not (logroot / sample["rid"]).is_dir():
        return rep
    sys.path.insert(0, str(repo / "scripts"))
    import torchtpu_matrix_report as tmr  # noqa: PLC0415
    orig = tmr.worker_logs

    def worker_logs(run_dir, vlog=""):
        out: dict[int, str] = {}
        for d in (run_dir / "ray_logs", run_dir / "host1" / "ray_logs"):
            stems: dict[str, str] = {}
            for f in sorted(d.glob("worker-*.gz")) if d.is_dir() else []:
                stem = f.name.split(".")[0]
                stems[stem] = stems.get(stem, "") + tmr.read(f) + "\n"
            for text in stems.values():
                m = re.search(r"Worker_TP(\d+) pid=", text) or re.search(r"PM_INIT_DEVICE rank=(\d+) ", text)
                if m:
                    out[int(m.group(1))] = out.get(int(m.group(1)), "") + text
        return out or orig(run_dir, vlog)

    tmr.worker_logs = worker_logs
    try:
        new = tmr.run_report(logroot, sample["rid"])
    finally:
        tmr.worker_logs = orig
    if new.get("ready_s") != rep.get("ready_s"):  # same T0 and READY, or keep the engine's own report
        return rep
    new["worker_logs_grouped_by_stem"] = True
    return new


def torch_evidence(directory: Path, sample: dict) -> dict:
    """The run's config checks (tpu_e2e.torch_checks: rank -> device_id, sampler bank, chip binding, worker threads,
    per-program compile seconds), recomputed from the pod logs for runs made before the checks existed (evidence
    only: those runs were not gated on them, so their `ok` is dropped)."""
    eng = sample.get("engine") or {}
    if eng.get("checks"):
        return eng["checks"]
    import tpu_e2e  # noqa: PLC0415
    env = torch_env(sample.get("config") or {}, eng.get("arm") or "")
    checks = tpu_e2e.torch_checks(directory / "pod_logs", sample["rid"], env)
    checks.pop("ok", None)
    return {**checks, "recomputed_from_logs": True}


def decode_d128(directory: Path, rid: str) -> float | None:
    """Steady-state decode of the cell's req probe (`d128`: 128 tokens after the first, tok/s)."""
    probe = load(directory / "pod_logs" / rid, "req_probe.json") or {}
    return next((r.get("decode_tok_per_s_after_first") for r in probe.get("requests", []) if r.get("name") == "d128"), None)


def torch_row(directory: Path) -> dict:
    """One tpu_e2e.py torch sample: GKE stages + venv delivery + the counted chipmap step (TTL_CHIP_BIND arms) +
    control + the TorchTPU cell's own phases (torchtpu_matrix_report.run_report, T0 = its fresh-host T0, placed on
    the epoch by engine_t0)."""
    sample = load(directory, "sample.json")
    eng = sample.get("engine") or {}
    row = {"run_id": sample["rid"], "lane": sample["lane"], "arm": torch_arm(sample, eng.get("arm")),
           "status": sample["status"], "code_commit": sample.get("code_commit"),
           "torch_repo": (sample.get("config") or {}).get("torch_repo")}
    if sample.get("gke"):
        row["gke"] = gke_row(sample)
    rep = eng.get("report") or {}
    if rep.get("ready_s") is not None and (rep.get("milestones") or {}).get("workers_up") is None:
        rep = zygote_report(directory, sample, rep)
    checks = torch_evidence(directory, sample) if (directory / "pod_logs" / sample["rid"]).is_dir() else {}
    row["rank_device"] = checks.get("rank_device")
    if sample["status"] != "ok" or rep.get("ready_s") is None or not eng.get("engine_t0"):
        greedy = eng.get("greedy") or {}
        if sample.get("error"):
            row["error"] = sample["error"]
        elif rep.get("ready_s") is not None and greedy.get("passed") is False:
            row["error"] = f"greedy check failed (ready {rep['ready_s']} s from the host T0)"
        elif eng.get("checks") and not eng["checks"].get("ok"):
            row["error"] = (f"config checks failed (ready {rep.get('ready_s')} s from the host T0): " + json.dumps(
                {k: checks.get(k) for k in ("canonical_ok", "bank_ok", "threads_ok") if k in checks}))
        else:
            row["error"] = "failed run (not ready)"
        row["greedy"] = greedy or None
        row["report"] = rep
        row["checks"] = checks or None
        return row
    t0 = sample["t0"]
    marks = {m["event"]: m["t"] for m in sample.get("marks", [])}
    venv = marks["venv_end"] - marks["venv_start"]
    # the chip map of the fresh pair (TTL_CHIP_BIND arms): its own counted step between the venv and the serve
    chipmap = marks["chipmap_end"] - marks["chipmap_start"] if "chipmap_start" in marks else None
    total = eng["engine_t0"] + rep["ready_s"] - t0
    phases = rep["phases"]
    engine_s = sum(v for v in phases.values() if v is not None)
    row.update({"ready_s": round(total, 2), "engine_s": round(engine_s, 2), "venv_s": round(venv, 2),
                "control_s": round(total - row["gke"]["total"] - venv - (chipmap or 0.0) - engine_s, 2),
                "phases": phases, "greedy": eng.get("greedy"), "greedy_ok": bool((eng.get("greedy") or {}).get("passed")),
                "health_ok": rep.get("ready_s") is not None, "report": rep, "venv": eng.get("venv")})
    if chipmap is not None:
        row["chipmap_s"] = round(chipmap, 2)
        row["chipmap"] = eng.get("chipmap")
    row["decode_d128_tok_s"] = decode_d128(directory, sample["rid"])
    row["compile_programs"] = checks.get("compile_programs")
    row["checks"] = {k: checks.get(k) for k in ("ok", "canonical_ok", "bank_ok", "threads_ok", "worker_files",
                                                 "chip_bind_errors", "stock_path", "placement_mismatch", "preseed",
                                                 "sampler_bank", "torch_threads", "mem_min_available_kb",
                                                 "disk_min_available_kb", "recomputed_from_logs") if k in checks}
    return row


# The cell's stockcc / runaicc compile caches were built on small_d (.45/.51) under its stock rank -> device_id map;
# the canonical stock cache (stock-ckpt-cc-canon) under the canonical binding (rank r -> device_id r).
SMALL_D_STOCK_MAP = [1, 0, 2, 3, 7, 5, 4, 6]
STOCK_ARMS = ("stock-ckpt-cc", "stock-runai-cc", "stock-ckpt-cc-canon")


def torch_stock_fairness(rows: list[dict]) -> dict:
    """Did stock lose compile time to device-order misses? Per ready stock run: its rank -> device_id map, whether it
    equals the map its compile cache was seeded under, the compile phase and its per-program parts."""
    runs = []
    for r in rows:
        base = str(r["arm"]).split("@")[0]
        if base not in STOCK_ARMS or r.get("ready_s") is None:
            continue
        seed = list(range(8)) if base == "stock-ckpt-cc-canon" else SMALL_D_STOCK_MAP
        cp = r.get("compile_programs") or {}
        runs.append({"run_id": r["run_id"], "arm": r["arm"], "lane": r["lane"], "rank_device": r.get("rank_device"),
                     "cache_seed_map": seed, "matches_cache_seed_map": r.get("rank_device") == seed,
                     "ready_s": r["ready_s"], "compile_s": r["phases"].get("compile"),
                     "compute_selected_logits_s": cp.get("compute_selected_logits"),
                     "model_backbone_s": cp.get("model backbone"), "compile_programs": cp})
    groups = {}
    for x in runs:
        g = groups.setdefault(f"{x['arm']} | {'match' if x['matches_cache_seed_map'] else 'other map'}", [])
        g.append(x)
    return {"small_d_stock_map": SMALL_D_STOCK_MAP, "runs": runs,
            "groups": {k: {"n": len(v), "run_ids": [x["run_id"] for x in v],
                           "ready_s": med([x["ready_s"] for x in v]), "compile_s": med([x["compile_s"] for x in v]),
                           "compute_selected_logits_s": med([x["compute_selected_logits_s"] for x in v]),
                           "model_backbone_s": med([x["model_backbone_s"] for x in v])} for k, v in groups.items()}}


JAX_NOTES = {
    "runai-jax-xla": "stock-runai-cc: vLLM --load-format runai_streamer from gs:// + the stock XLA cache (JAX 1 s "
                     "threshold) fetched from GCS before the deploy (TpuJaxTP8 settings).",
    "ckpt-gcs-jax-xla": "stock-ckpt-cc: full HF checkpoint to tmpfs on both hosts (startup_state.py fetch-checkpoint, "
                        "sliced GCS GETs) + the stock XLA cache (JAX 1 s threshold) from GCS, then the default loader.",
    "gcs-ov": "pm-gcs: TPUFlex snapshot + executable bank + XLA cache from TpuJaxTP8's TP8 seed in GCS, fetched "
              "as a child of the deploy while the engine boots (state gate before the first device touch).",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", help="required unless --inspect")
    ap.add_argument("--out", type=Path, help="required unless --inspect")
    ap.add_argument("--jax-dir", type=Path, default=ROOT / "results/tpu-e2e/jax")
    ap.add_argument("--jax-meta", type=Path, default=ROOT / "results/tpu-e2e/jax/meta.json")
    ap.add_argument("--torch-dir", type=Path, default=ROOT / "results/tpu-e2e/torch")
    ap.add_argument("--torch-meta", type=Path, default=ROOT / "results/tpu-e2e/torch/meta.json")
    ap.add_argument("--inspect", type=Path, nargs="+", help="print one JAX run's checks and breakdown, then exit")
    args = ap.parse_args()
    if args.inspect:
        for d in args.inspect:
            r = torch_row(d) if (load(d, "sample.json").get("engine") or {}).get("report") is not None else jax_row(d)
            s = load(d, "sample.json")
            print(f"== {r['run_id']} ({r['lane']}, {r['arm']}) status={r['status']} error={r.get('error')}")
            print(f"   ready={r.get('ready_s')} gke_total={(r.get('gke') or {}).get('total')} control={r.get('control_s')} "
                  f"engine={r.get('engine_s')} greedy={(r.get('greedy') or {}).get('prompts_equal')}/32 ok={r.get('greedy_ok')}")
            print(f"   gke={r.get('gke')}")
            print(f"   phases={r.get('phases')} overlap_fetch_wall={r.get('overlap_fetch_wall_s')} walls={r.get('delivery_walls_s')}")
            print(f"   evidence={r.get('evidence')}")
            print(f"   cleanup={s.get('cleanup')} cotenants={[c.get('rid') for c in s.get('cotenants_start', []) + s.get('cotenants_end', [])]}")
            print(f"   steps={(s.get('engine') or {}).get('steps')} failure_reasons={(s.get('gke') or {}).get('failure_reasons')}")
            if "rank_device" in r:  # torch
                print(f"   venv={r.get('venv_s')} chipmap={r.get('chipmap_s')} {json.dumps(r.get('chipmap'))[:600]}")
                print(f"   rank_device={r.get('rank_device')} decode_d128={r.get('decode_d128_tok_s')} "
                      f"compile_programs={r.get('compile_programs')}")
                print(f"   checks={json.dumps(r.get('checks'))[:1500]}")
                print(f"   greedy_text={(r.get('greedy') or {}).get('text')!r}")
        return
    if not (args.commit and args.out):
        ap.error("--commit and --out are required")
    rows = [jax_row(d) for d in sorted(p.parent for p in args.jax_dir.glob("*/sample.json"))]
    meta = json.loads(args.jax_meta.read_text()) if args.jax_meta.exists() else {}
    cells = {"tpu-jax-e2e": cell("tpu-jax-e2e", rows, JAX_CANON, JAX_NOTES, meta)}
    trows = [torch_row(d) for d in sorted(p.parent for p in args.torch_dir.glob("*/sample.json"))]
    if trows:
        tmeta = json.loads(args.torch_meta.read_text()) if args.torch_meta.exists() else {}
        cells["tpu-torch-e2e"] = cell("tpu-torch-e2e", trows, TORCH_CANON, tmeta.get("notes", {}), tmeta)
        cells["tpu-torch-e2e"]["stock_fairness"] = torch_stock_fairness(trows)
    out = {"summary_commit": args.commit, "cells": cells}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for c in cells.values():
        print(f"## {c['cell']}")
        print("| arm | src | status | n | median | runs | gke total | control | " + " | ".join(PHASES) + " |")
        for name, b in [*c["arms"].items(), *c["extras"].items()]:
            print(f"| {name} | {b['source_arm']} | {b['status']} | {b['n']} | {b['median']} | {b['runs']} | "
                  f"{(b['gke'] or {}).get('total')} | {b['control']} | "
                  + " | ".join(str(b["phases"][p]) for p in PHASES) + " |")


if __name__ == "__main__":
    main()
