#!/usr/bin/env python3
"""Cross-stack startup matrix, cell ``tpu-jax``: vLLM-JAX (vLLM + tpu-inference) on v5e at TP8, from
``switch_campaign.sh autoscale`` partition runs on the S pairs (AS_PART=small_*).

    .venv/bin/python scripts/matrix_summary.py results/switch/autoscale-<commit>-ps{a,b,c} \\
        --out results/switch/matrix-<commit>/matrix_summary.json

Canonical arm <- campaign arm:
  stock-lazy      raw-jax           lazy checkpoint reads through GCS FUSE, empty XLA cache, JAX default 1 s threshold
  stock-ckpt      ckpt-gcs-jax      checkpoint first copied to tmpfs (sliced GCS GETs), then stock-lazy's loader
  stock-ckpt-cc   ckpt-gcs-jax-xla  stock-ckpt + the XLA cache a stock-ckpt boot wrote (1 s threshold), via GCS
  stock-runai     runai-jax         vLLM --load-format runai_streamer from gs://, empty XLA cache
  stock-runai-cc  runai-jax-xla     stock-runai + the stock XLA cache
  pm-gcs          gcs-ov            TPUFlex state (snapshot + bank + XLA) from GCS, fetch overlapped with the boot
  pm-peer         peer-ov           the same state from another pair's live seed scope over TCP, overlapped
  warm-stock      warm-ckpt-jax     same-host restart: the stock boot's tmpfs checkpoint + its XLA cache
  warm-pm         warm-pm           same-host restart: the seed's snapshot + bank + XLA, pre-spawned
Extras: stock-ckpt-cc-0s <- ckpt-gcs-xla (ckpt + the TPUFlex seed's XLA cache, 0 s threshold);
        warm-pm@gate (warm-pm with T0 = gate open instead of launch).

Ready = GET /health 200 on the instance endpoint (deployment serving_endpoint.health.ok_at) AND the
greedy check (32 W0 prompts x 64 tokens, token for token against the pair's seed reference) passed.
T0: sequential arms = the first byte of delivery (ready = delivery remote-driver wall + deploy start ->
health; the Ray job plumbing between the fetch and the deploy job is excluded); -ov arms and arms with
nothing to deliver = deploy start (the delivery child starts inside the deploy).

Phases are cuts on the critical path (max over the 2 workers of each boundary), so they sum to ready:
  download     delivery wall (sequential) | longest worker wait at the state gate (-ov, exposed part) | 0
  spawn        deploy start -> first device touch (control, process spawn, imports, vLLM construction)
  init_device  TPU runtime init
  lazy         (GCS-reading arms: stock-lazy, stock-runai*) load_model minus the mean load_model of the
               local-checkpoint arms (stock-ckpt, stock-ckpt-cc); inferred, as in the GPU cells
  load_model   the rest of load_model (graph build, checkpoint -> device, or snapshot restore)
  kv           initialize_from_config (memory profile, KV allocation, sampler/logprob precompiles)
  compile      compile_or_warm_up_model (XLA precompile of every padding bucket + warm-up; bank install)
  graphs       null: no CUDA graph capture on TPU
  tail         compile end -> /health 200 (engine core start, endpoint bind, registration)
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path

CANONICAL = (  # canonical arm, campaign arm
    ("stock-lazy", "raw-jax"), ("stock-ckpt", "ckpt-gcs-jax"), ("stock-ckpt-cc", "ckpt-gcs-jax-xla"),
    ("stock-runai", "runai-jax"), ("stock-runai-cc", "runai-jax-xla"), ("pm-gcs", "gcs-ov"),
    ("pm-peer", "peer-ov"), ("warm-stock", "warm-ckpt-jax"), ("warm-pm", "warm-pm"))
EXTRA = (("stock-ckpt-cc-0s", "ckpt-gcs-xla"),)
PHASES = ("download", "spawn", "init_device", "lazy", "load_model", "compile", "kv", "graphs", "tail")
LOCAL = ("ckpt-gcs-jax", "ckpt-gcs-jax-xla")
GCS_READING = ("raw-jax", "runai-jax", "runai-jax-xla")
OVERLAP = ("gcs-ov", "peer-ov")
NOTES = {
    "stock-lazy": "vLLM default loader, safetensors lazy reads through the GCS FUSE /models mount; empty XLA cache "
                  "with JAX's default 1 s persistence threshold.",
    "stock-ckpt": "startup_state.py fetch-checkpoint: the full 62.6 GB HF checkpoint to tmpfs on both hosts "
                  "(32 processes, 64 MiB sliced GCS GETs), then the default loader from tmpfs; empty XLA cache, 1 s threshold.",
    "stock-ckpt-cc": "stock-ckpt + the XLA cache that this pair's own stock-ckpt boot persisted (1 s threshold), "
                     "shipped via GCS before boot (download = checkpoint + cache). Programs under 1 s compile again.",
    "stock-runai": "vLLM --load-format runai_streamer, model gs://hongtaozhang-tpu-model-cache-us-west1/gemma-4-31b "
                   "(Run:ai model streamer 0.15.9); tpu-inference streams into Gemma4ForCausalLM's LoadableWithIterator "
                   "path. Empty XLA cache, 1 s threshold. Deviations: --tokenizer /models/gemma-4-31b and "
                   "model_loader_extra_config memory_limit=8 GiB (defaults evicted a host: extras['stock-runai-default']); "
                   "run last, under the exclusive cluster lock, with a host memory/disk watchdog.",
    "stock-runai-cc": "stock-runai + the stock XLA cache (1 s threshold) via GCS before boot.",
    "pm-gcs": "gcs-ov: snapshot + executable bank + XLA cache (seed upload of this pair) fetched with GCS "
              "transfer_manager as a child of the deploy; workers wait on their rank's delivery marker before the "
              "first device touch. gcs-ov is the current best GCS arm (TP16: 51.4 s vs 62.7 s sequential).",
    "pm-peer": "peer-ov: the same state pulled over TCP (8 streams per rank pair) from small_c's kept seed scope; "
               "the source pair was idle (its campaign holds until the peer arms are done).",
    "warm-stock": "Restart on the same pair with the stock-ckpt boot's tmpfs checkpoint and the XLA cache it wrote "
                  "(1 s threshold); nothing delivered.",
    "warm-pm": "Restart on the same pair on the seed's own snapshot + bank + XLA cache, pre-spawned: CPU-only "
               "workers park before the first device touch, the gate opens as soon as all are parked. "
               "T0 = launch here; T0 = gate open in extras['warm-pm@gate'].",
    "stock-ckpt-cc-0s": "stock-ckpt with JAX's 0 s threshold (every program persisted) and the XLA cache of the "
                        "TPUFlex seed boot (0 s threshold), via GCS.",
}


def load(directory, name):
    for path in (directory / name, directory / (name + ".gz")):
        if path.exists():
            with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as handle:
                return json.load(handle)
    return None


def target_init(deploy, name):
    for row in deploy["measurements"]["deployment"]["instances"]:
        init = row.get("initialization") or {}
        if (row.get("descriptor") or {}).get("instance_id", init.get("instance_id")) == name:
            return row, init
    raise KeyError(f"no instance {name} in deploy evidence")


def span_max(workers, phase, edge):
    values = [w["phases"][phase][edge] for w in workers if isinstance((w.get("phases") or {}).get(phase), dict)]
    return max(values) if values else None


def wall(evidence):
    return (evidence or {}).get("measurements", {}).get("wall_s")


def run_row(directory, run, part):
    rid, arm = run["run_id"], run["kind"]
    row = {"run_id": rid, "arm": arm, "pair": part, "campaign_ok": run.get("ok")}
    if run.get("skipped"):
        row["skipped"] = run["skipped"]
        return row
    deploy = load(directory, f"{rid}-deploy.json")
    if deploy is None:
        row["error"] = "no deploy evidence"
        return row
    dep = deploy["measurements"]["deployment"]
    record, init = target_init(deploy, part)
    workers = init.get("worker_init_phases") or []
    t_deploy = dep["started_at"]
    health = (record.get("serving_endpoint") or {}).get("health") or {}
    t_health = health.get("ok_at")
    fetch, ckpt = load(directory, f"{rid}-fetch.json"), load(directory, f"{rid}-ckpt-fetch.json")
    delivery = dep.get("state_delivery")
    seq_delivery = (wall(fetch) or 0.0) + (wall(ckpt) or 0.0)
    gate_waits = [w["phases"]["state_gate"]["end"] - w["phases"]["state_gate"]["start"] for w in workers
                  if isinstance((w.get("phases") or {}).get("state_gate"), dict)]
    gate_wait = max(gate_waits) if gate_waits else 0.0
    touch = span_max(workers, "init_device", "start")
    cuts = {k: span_max(workers, p, "end") for k, p in (("dev", "init_device"), ("load", "load_model"),
                                                        ("kv", "initialize_from_config"),
                                                        ("compile", "compile_or_warm_up_model"))}
    validation = load(directory, f"{rid}-validation.json")
    check = load(directory, f"{rid}-check.json")
    vm = (validation or {}).get("measurements") or {}
    row.update({
        "greedy_ok": bool(vm.get("ok") and (vm.get("greedy_equality") or {}).get("passed")),
        "check_ok": bool(((check or {}).get("measurements") or {}).get("ok")),
        "health_ok": health.get("status") == "ok" and t_health is not None,
        "delivery_s": round(seq_delivery, 2) if arm not in OVERLAP else None,
        "overlap_fetch_wall_s": round(delivery["fetch"]["wall_s"], 2)
        if delivery and (delivery.get("fetch") or {}).get("wall_s") else None,
        "gate_wait_s": round(gate_wait, 2),
        "stage_clocks": dep.get("stage_clocks"),
    })
    if None in (t_health, touch, *cuts.values()):
        row["error"] = "incomplete phase evidence"
        return row
    download = gate_wait if arm in OVERLAP else seq_delivery
    phases = {"download": download, "spawn": touch - t_deploy - (gate_wait if arm in OVERLAP else 0.0),
              "init_device": cuts["dev"] - touch, "lazy": None, "load_model": cuts["load"] - cuts["dev"],
              "kv": cuts["kv"] - cuts["load"], "compile": cuts["compile"] - cuts["kv"], "graphs": None,
              "tail": t_health - cuts["compile"]}
    row["ready_s"] = round(seq_delivery + (t_health - t_deploy) if arm not in OVERLAP else t_health - t_deploy, 2)
    row["phases"] = phases
    pre = dep.get("prespawn")
    if pre and pre.get("gate_open_at"):
        gate = pre["gate_open_at"]
        row["gate_open_after_launch_s"] = round(gate - t_deploy, 2)
        row["ready_from_gate_s"] = round(t_health - gate, 2)
        row["phases_from_gate"] = {**phases, "spawn": touch - gate}
    rank0 = next((w for w in workers if w.get("executor_rank") == 0), workers[0] if workers else {})
    compiles = 0
    for profile in (rank0.get("startup_profiles") or {}).values():
        for interval in (profile.get("intervals") or []) if isinstance(profile, dict) else []:
            compiles += interval.get("kind") == "backend_compile_and_load"
    row["evidence"] = {
        "snapshot_restored": all((w.get("weight_snapshot") or {}).get("restored") for w in workers),
        "bank_installed": all((w.get("exec_bank") or {}).get("installed") for w in workers),
        "xla_backend_compiles_rank0": compiles,
        "weight_load_s": round(max((w["phases"]["weight_load"]["end"] - w["phases"]["weight_load"]["start"])
                                   for w in workers if "weight_load" in (w.get("phases") or {})), 2)
        if any("weight_load" in (w.get("phases") or {}) for w in workers) else None,
        "model": init.get("resolved_engine_config", {}).get("model"),
        "load_format": (init.get("engine_settings") or {}).get("load_format"),
        "kv_blocks": (init.get("kv_capacity") or {}).get("usable_blocks"),
    }
    return row


def med(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 2) if values else None


def arm_block(name, rows, key="ready_s", phase_key="phases"):
    good = [r for r in rows if r.get("greedy_ok") and r.get("health_ok") and r.get(key) is not None]
    tried = [r for r in rows if not r.get("skipped")]
    status = "ok" if len(good) >= 3 else ("na" if any(r.get("skipped") for r in rows) and not good else "dnf")
    block = {"status": status,
             "n": len(good), "runs": [r[key] for r in good], "median": med([r[key] for r in good]),
             "phases": {p: (None if p == "graphs" else med([r[phase_key][p] for r in good])) for p in PHASES}
             if good else {p: None for p in PHASES},
             "gke": None, "note": NOTES.get(name, ""),
             "run_ids": [r["run_id"] for r in good]}
    failed = [r["run_id"] for r in tried if r not in good]
    if failed:
        block["failed_run_ids"] = failed
    return block


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--runai-default", type=Path,
                        help="campaign dir of the stock-runai boot at streamer defaults (recorded as dnf)")
    args = parser.parse_args()
    rows, campaigns = [], []
    for directory in args.directories:
        config = json.loads((directory / "autoscale.json").read_text())
        part = config["partition"]
        campaigns.append({"directory": str(directory), **config})
        for line in (directory / "runs.jsonl").read_text().splitlines():
            if line.strip():
                rows.append(run_row(directory, json.loads(line), part))
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    local = [r["phases"]["load_model"] for a in LOCAL for r in by_arm.get(a, [])
             if r.get("phases") and r.get("greedy_ok")]
    reference = statistics.mean(local) if local else None
    for arm in GCS_READING:
        for r in by_arm.get(arm, []):
            if r.get("phases") and reference is not None:
                r["phases"]["lazy"] = max(0.0, r["phases"]["load_model"] - reference)
                r["phases"]["load_model"] -= r["phases"]["lazy"]
    for r in rows:
        for key in ("phases", "phases_from_gate"):
            if r.get(key):
                r[key] = {k: (None if v is None else round(v, 2)) for k, v in r[key].items()}
    arms = {name: arm_block(name, by_arm.get(arm, [])) for name, arm in CANONICAL}
    extras = {name: arm_block(name, by_arm.get(arm, [])) for name, arm in EXTRA}
    extras["warm-pm@gate"] = arm_block("warm-pm", by_arm.get("warm-pm", []), "ready_from_gate_s", "phases_from_gate")
    extras["warm-pm@gate"]["note"] = ("warm-pm with T0 = gate open (all workers parked, TPUs released to them); "
                                      "spawn = gate open -> first device touch.")
    if args.runai_default:
        rid = next(json.loads(l)["run_id"] for l in (args.runai_default / "runs.jsonl").read_text().splitlines()
                   if l.strip() and json.loads(l).get("kind") == "runai-jax")
        extras["stock-runai-default"] = {
            "status": "dnf", "n": 0, "runs": [], "median": None, "phases": {p: None for p in PHASES}, "gke": None,
            "note": ("stock-runai at Run:ai/vLLM defaults (tokenizer = the gs:// model, 40 GB streamer buffer): "
                     "~5 min into the boot node .198 got the node.kubernetes.io/disk-pressure taint (23:44:16Z) and "
                     "kubelet evicted the worker pod (Ray node SIGTERM 23:46:59Z, deploy failed at 500 s); KubeRay "
                     "then recreated all 8 TPU worker pods. Cause: with the tokenizer at gs://, vLLM pulls every "
                     "non-weight object under the prefix (~80 GB of stale .cache/huggingface/download/*.incomplete; "
                     "the prefix holds 142.4 GB, 62.5 GB of it safetensors) to local disk (95 GB root). The ok "
                     "stock-runai arms use tokenizer=/models/gemma-4-31b and memory_limit=8 GiB (deviations)."),
            "run_ids": [rid], "evidence": [str(args.runai_default / f"{rid}-deploy-joblog.json.gz"),
                                           str(args.runai_default / "runai-incident-ray-nodes.json")]}
    summary = {
        "cell": "tpu-jax", "hardware": "TPU v5e (v5litepod, 2x4 sub-slice of the 4x8 slice)", "tp": 8, "hosts": 2,
        "engine": {"max_model_len": 8192, "max_num_seqs": 512, "max_num_batched_tokens": 2048,
                   "extra": "bf16 weights + KV, gpu_memory_utilization 0.85, prefix caching off, Ray distributed "
                            "executor, one engine driver with tensor_parallel_size=8 over the pair's 2 hosts "
                            "(hetero/engine_actor.py); startup_profile on"},
        "versions": {"vllm": "0.23.1rc1.dev1474+g0ba2aa35a.tpu", "tpu_inference": "8d31f0866147 (2026-07-28)",
                     "jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.43", "flax": "0.12.4", "ray": "2.56.1",
                     "runai_model_streamer": "0.15.9", "python_torch_cpu": "2.10.0+cpu"},
        "image_or_venv": "docker.io/vllm/vllm-tpu:v0.26.0@sha256:186f1940c353d2703bf80f80fff101b38fa8e26a9562a767b37e103f212d3d79",
        "code_commit": args.commit,
        "t0": ("Fresh scope on hosts that already run the image (no host provisioning, no image pull): T0 = first "
               "byte of delivery for sequential arms, deploy start (engine launch) when delivery overlaps or "
               "nothing is delivered; warm-* = restart on the same pair; warm-pm also at T0 = gate open (extras). "
               "Ready = GET /health 200 and the greedy check passed."),
        "arms": arms,
        "extras": extras,
        "lazy_reference_load_model_s": round(reference, 2) if reference is not None else None,
        "campaigns": campaigns,
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print("| arm | status | n | median s | runs s | " + " | ".join(PHASES) + " |")
    print("|---|---|---|---|---|" + "---|" * len(PHASES))
    for name, block in [*arms.items(), *extras.items()]:
        print(f"| {name} | {block['status']} | {block['n']} | {block['median']} | "
              f"{', '.join(str(v) for v in block['runs'])} | "
              + " | ".join(str(block["phases"][p]) for p in PHASES) + " |")


if __name__ == "__main__":
    main()
