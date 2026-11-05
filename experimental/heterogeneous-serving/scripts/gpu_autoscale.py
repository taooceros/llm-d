#!/usr/bin/env python3
"""GPU analog of the TPU autoscale experiment (paper Fig 2) on brent-llm-d-asia-southeast1-h100.

Every run boots one TP1 vLLM instance (Gemma-4 31B, 62.6 GB bf16) on a *fresh* a3-highgpu-1g node (1x H100 80GB):
the pod's node affinity excludes nodes labelled gpu-autoscale/used, so each run triggers a GKE scale-up from
zero; after the run the node is labelled used and its VM is deleted from the MIG.  A fresh node has an empty
page cache and no checkpoint, compile cache or CUDA/Triton JIT cache.  Runs use the Spot pool (gpu-pool-spot, max
5 nodes): on-demand a3-highgpu-1g (gpu-pool) fails VM creation with INTERNAL_ERROR in both zones (2026-10-22).

    scripts/gpu_autoscale.py configmap
    scripts/gpu_autoscale.py run --arms ckpt --samples 1 --export gs://.../compile-cache/local.tar --out DIR
    scripts/gpu_autoscale.py run --arms raw,ckpt,ckpt-cc --samples 3 --parallel 3 --cc-local URI --out DIR
    scripts/gpu_autoscale.py report DIR [DIR ...]

Arms (scripts/gpu_autoscale/pod_run.sh):
  raw       stock vLLM: lazy safetensors mmap through gcsfuse (TPU cluster's build, implicit-dirs, no file cache)
  ckpt      checkpoint first copied to tmpfs (transfer_manager sliced GETs, 32 process workers, 64 MiB chunks)
  ckpt-cc   ckpt + the compile/JIT caches ($HOME: vLLM torch.compile, Inductor, Triton, FlashInfer) from a peer
  runai     Run:ai model streamer straight from gs:// to the GPU (vLLM --load-format runai_streamer)
  runai-cc  runai + the compile/JIT caches from a peer
Outputs in DIR: runs.jsonl (one record per run), <run_id>.log (kubectl logs --timestamps), <run_id>-k8s.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import statistics
import string
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KUBECONFIG = str(Path.home() / ".kube/brent-h100.yaml")
NAMESPACE = "gpu-autoscale"
CLUSTER, LOCATION = "brent-llm-d-asia-southeast1-h100", "asia-southeast1-c"
# Docker Hub vllm/vllm-openai:v0.23.0-x86_64-cu129 (12.1 GB compressed) through an Artifact Registry remote
# (pull-through) repository in the cluster's region; pulls straight from Docker Hub stalled past containerd's
# 5 min progress timeout.
IMAGE = ("asia-southeast1-docker.pkg.dev/sysres-disagg-ml/dockerhub-remote/vllm/vllm-openai:v0.23.0-x86_64-cu129"
         "@sha256:871762282db5bc464b5a3f0a59e41207ef25c2d95edf5f701e57a6bfc27b9496")
BUCKET, PREFIX = "hongtaozhang-gpu-model-cache-asia-southeast1", "gemma-4-31b"
USED_LABEL = "gpu-autoscale/used"
HERE = Path(__file__).resolve().parent / "gpu_autoscale"
ARMS = ("raw", "ckpt", "ckpt-cc", "runai", "runai-cc")
_print_lock = threading.Lock()


def log(*parts):
    with _print_lock:
        print(time.strftime("%H:%M:%S"), *parts, flush=True)


def kubectl(*args, inp=None, check=True):
    r = subprocess.run(["kubectl", "--kubeconfig", KUBECONFIG, "-n", NAMESPACE, *args], input=inp,
                       capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def kjson(*args):
    return json.loads(kubectl(*args, "-o", "json"))


def ts(text):
    """RFC3339 (any fractional precision) -> epoch seconds."""
    if text is None:
        return None
    main, _, frac = text.rstrip("Z").partition(".")
    base = dt.datetime.strptime(main, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
    return base + (float("0." + frac) if frac else 0.0)


def configmap():
    files = {p.name: p.read_text() for p in (HERE / "pod_run.sh", HERE / "fetch.py")}
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "gpu-autoscale-scripts", "namespace": NAMESPACE},
          "data": files}
    kubectl("apply", "-f", "-", inp=json.dumps(cm))


def pod_spec(name, arm, pool, cc_uri):
    env = {"ARM": arm, "BUCKET": BUCKET, "PREFIX": PREFIX, "CC_URI": cc_uri or "", "HOME": "/cache/home",
           "PYTHONUNBUFFERED": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
    mounts = [{"name": "scripts", "mountPath": "/scripts"}, {"name": "cache", "mountPath": "/cache"},
              {"name": "dshm", "mountPath": "/dev/shm"}]
    volumes = [{"name": "scripts", "configMap": {"name": "gpu-autoscale-scripts", "defaultMode": 0o755}},
               {"name": "cache", "emptyDir": {}},
               {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}]
    if arm.startswith("ckpt"):  # the TPU arms also stage the checkpoint in tmpfs
        mounts.append({"name": "model", "mountPath": "/model"})
        volumes.append({"name": "model", "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"}})
    return {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": name, "namespace": NAMESPACE, "labels": {"app": "gpu-autoscale", "arm": arm}},
        "spec": {
            "restartPolicy": "Never", "terminationGracePeriodSeconds": 5,
            "nodeSelector": {"cloud.google.com/gke-nodepool": pool},
            "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
                {"matchExpressions": [{"key": USED_LABEL, "operator": "DoesNotExist"}]}]}}},
            "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
            "containers": [{
                "name": "engine", "image": IMAGE, "command": ["/scripts/pod_run.sh"],
                "env": [{"name": k, "value": v} for k, v in env.items()],
                "securityContext": {"privileged": True},  # /dev/fuse for the gcsfuse arm
                "resources": {"limits": {"nvidia.com/gpu": 1}},
                "volumeMounts": mounts}],
            "volumes": volumes}}


def retire_node(node):
    """Label the node used (no later run lands on it) and delete its VM from the managed instance group."""
    subprocess.run(["kubectl", "--kubeconfig", KUBECONFIG, "label", "node", node, f"{USED_LABEL}=true", "--overwrite"],
                   capture_output=True, text=True)
    zone = json.loads(subprocess.run(["kubectl", "--kubeconfig", KUBECONFIG, "get", "node", node, "-o", "json"],
                                     capture_output=True, text=True).stdout or "{}")
    zone = zone.get("metadata", {}).get("labels", {}).get("topology.kubernetes.io/zone")
    if not zone:
        return f"{node}: no zone label; left to the autoscaler"
    created_by = subprocess.run(["gcloud", "compute", "instances", "describe", node, "--zone", zone,
                                 "--format=value(metadata.items.created-by)"], capture_output=True, text=True).stdout
    mig = created_by.strip().split("/")[-1]
    if not mig:
        return f"{node}: no created-by MIG; left to the autoscaler"
    r = subprocess.run(["gcloud", "compute", "instance-groups", "managed", "delete-instances", mig, "--instances", node,
                        "--zone", zone, "--quiet"], capture_output=True, text=True)
    return f"{node}: delete from {mig} rc={r.returncode} {r.stderr.strip()[-200:]}"


def export_cache(pod, uri, attempts=3):
    """Stream the pod's $HOME caches (minus pip/HF downloads) into a GCS tarball, via this host.  The upload
    must match the tar size measured in the pod; a `kubectl exec` stream reset (seen 2026-10-22) left a
    truncated object, so failed attempts delete the object and retry."""
    tar_cmd = "tar -C /cache/home -cf - --exclude=./.cache/pip --exclude=./.cache/huggingface ."
    listing = kubectl("exec", pod, "--", "bash", "-c",
                      "cd /cache/home && du -sb .cache/* .triton .nv .config 2>/dev/null; true")
    expected = int(kubectl("exec", pod, "--", "bash", "-c", f"{tar_cmd} | wc -c").strip())
    for attempt in range(1, attempts + 1):
        tar = subprocess.Popen(["kubectl", "--kubeconfig", KUBECONFIG, "-n", NAMESPACE, "exec", pod, "--", "bash",
                                "-c", tar_cmd], stdout=subprocess.PIPE)
        up = subprocess.run(["gcloud", "storage", "cp", "-", uri], stdin=tar.stdout, capture_output=True, text=True)
        tar.stdout.close()
        tar_rc = tar.wait()
        listed = subprocess.run(["gcloud", "storage", "ls", "-l", uri], capture_output=True, text=True).stdout.split()
        size = int(listed[0]) if listed and listed[0].isdigit() else None
        if not tar_rc and not up.returncode and size == expected:
            return {"uri": uri, "bytes": size, "attempts": attempt, "du": listing}
        subprocess.run(["gcloud", "storage", "rm", uri], capture_output=True)
        log(pod, f"export attempt {attempt} failed: tar rc={tar_rc} cp rc={up.returncode} size={size}/{expected}")
    raise RuntimeError(f"export {pod} -> {uri} failed after {attempts} attempts")


def poll_logs(name, logfile, deadline, interval=5.0):
    """Snapshot `kubectl logs --timestamps` into logfile every few seconds until MARK check_done / engine_died,
    the container exits, or the deadline.  Polling, not `logs -f`: on 2026-10-22 followed streams both ended
    early (mid-compile) and hung for an hour on five concurrent pods while the kubelet log stayed complete."""
    while True:
        text = kubectl("logs", "--timestamps", name, check=False)
        logfile.write_text(text)
        if re.search(r"^\S+ MARK (check_done|engine_died) ", text, re.M) or time.time() > deadline:
            return
        state = (kjson("get", "pod", name)["status"].get("containerStatuses") or [{}])[0].get("state", {})
        if "terminated" in state:
            logfile.write_text(kubectl("logs", "--timestamps", name, check=False))
            return
        time.sleep(interval)


def run_one(arm, sample, pool, cc_uri, out, export=None, keep=False, timeout=3600, adopt=None):
    """Boot one arm on a fresh node; with adopt=<pod name>, finish an already-submitted run instead (its
    submission time is the pod's creationTimestamp and the GPU-allocatable time is unknown)."""
    if adopt:
        name, rid = adopt, adopt.removeprefix("as-")
        pod = kjson("get", "pod", name)
        env = {e["name"]: e.get("value", "") for e in pod["spec"]["containers"][0]["env"]}  # k8s drops "" values
        arm, cc_uri, sample = env["ARM"], env["CC_URI"] or None, int(rid.split("-")[-2])
        pool = pod["spec"]["nodeSelector"]["cloud.google.com/gke-nodepool"]
    else:
        rid = f"{arm}-{sample}-{''.join(random.choices(string.ascii_lowercase + string.digits, k=4))}"
        name = f"as-{rid}"
    record = {"run_id": rid, "arm": arm, "sample": sample, "pool": pool, "cc_uri": cc_uri, "image": IMAGE, "ok": False}
    if adopt:
        record["t_submit"], record["adopted"] = ts(pod["metadata"]["creationTimestamp"]), True
    else:
        record["t_submit"] = time.time()
        kubectl("apply", "-f", "-", inp=json.dumps(pod_spec(name, arm, pool, cc_uri)))
        log(rid, "submitted")
    node, gpu_seen, deadline = None, None, time.time() + timeout
    logfile = out / f"{rid}.log"
    try:
        while True:  # until the container runs; note when the node first advertises its GPU
            pod = kjson("get", "pod", name)
            node = pod["spec"].get("nodeName") or node
            if node and gpu_seen is None and not adopt:
                n = json.loads(subprocess.run(["kubectl", "--kubeconfig", KUBECONFIG, "get", "node", node, "-o", "json"],
                                              capture_output=True, text=True).stdout or "{}")
                if n.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu") not in (None, "0"):
                    gpu_seen = time.time()
            state = (pod["status"].get("containerStatuses") or [{}])[0].get("state", {})
            if "running" in state or "terminated" in state:
                break
            if time.time() > deadline:
                raise TimeoutError("container never started")
            time.sleep(2)
        if node and gpu_seen is None and not adopt:
            gpu_seen = time.time()
        log(rid, "running on", node)
        poll_logs(name, logfile, deadline)
        parsed = parse_log(logfile.read_text())
        record.update(parsed)
        record["ok"] = parsed.get("ready") is not None and parsed.get("check_text") is not None
        if export and record["ok"]:
            record["export"] = export_cache(name, export)
            log(rid, "exported", record["export"]["bytes"], "B ->", export)
    except Exception as exc:  # recorded, never fatal for sibling runs
        record["error"] = repr(exc)
        log(rid, "ERROR", exc)
    finally:
        pod = kjson("get", "pod", name)
        events = kjson("get", "events", "--field-selector", f"involvedObject.name={name}")
        node_obj = json.loads(subprocess.run(["kubectl", "--kubeconfig", KUBECONFIG, "get", "node", node, "-o", "json"],
                                             capture_output=True, text=True).stdout or "{}") if node else {}
        (out / f"{rid}-k8s.json").write_text(json.dumps({"pod": pod, "events": events, "node": node_obj}))
        record["node"] = node
        record["gke"] = gke_stages(record["t_submit"], pod, events, node_obj, gpu_seen)
        if not keep:
            kubectl("delete", "pod", name, "--wait=false", check=False)
            if node:
                record["retire"] = retire_node(node)
        with _print_lock, (out / "runs.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
    log(rid, "ok" if record["ok"] else "FAILED", f"ready {record.get('ready_s')}",
        " ".join(f"{k}={v:.1f}" for k, v in (record.get("segments") or {}).items()))
    return record


LOG_ANCHORS = {  # first log line containing the text -> event (kubelet timestamp)
    "engine_init": "Initializing a V1 LLM engine",
    "load_start": "Starting to load model",
    "load_end": "Model loading took",
    "compile_end": ("torch.compile took", "torch.compile and initial profiling/warmup run together took"),
    "kv_size": "GPU KV cache size",
    "capture_end": "Graph capturing finished in",
    "init_engine_end": "init engine (profile, create kv cache, warmup model) took",
}


def parse_log(text):
    events, marks, values, fetches, check = {}, {}, {}, [], None
    for line in text.splitlines():
        stamp, _, body = line.partition(" ")
        if body.startswith("MARK "):
            event = body.split()[1]
            marks.setdefault(event, ts(stamp))  # one clock (kubelet) for MARK lines and vLLM log lines
            continue
        if body.startswith("FETCH "):
            fetches.append(json.loads(body[6:]))
        if body.startswith("CHECK "):
            choice = json.loads(body[6:])["choices"][0]
            check = choice["message"]["content"] if "message" in choice else choice["text"]
        for key, needles in LOG_ANCHORS.items():
            if key not in events and any(n in body for n in ((needles,) if isinstance(needles, str) else needles)):
                events[key] = ts(stamp)
        for key, pattern in (("weights_s", r"Loading weights took ([\d.]+) seconds"),
                             ("model_loading_s", r"Model loading took [\d.]+ GiB(?: memory)? and ([\d.]+) seconds"),
                             ("compile_s", r"torch\.compile took ([\d.]+) s in total"),
                             ("compile_profile_s", r"torch\.compile and initial profiling/warmup run together took ([\d.]+) s"),
                             ("dynamo_s", r"Dynamo bytecode transform time: ([\d.]+) s"),
                             ("capture_s", r"Graph capturing finished in (\d+) secs"),
                             ("capture_gib", r"Graph capturing finished in \d+ secs, took ([\d.]+) GiB"),
                             ("kv_tokens", r"GPU KV cache size: ([\d,]+) tokens"),
                             ("kv_gib", r"Available KV cache memory: ([\d.]+) GiB"),
                             ("init_engine_s", r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s")):
            m = re.search(pattern, body)
            if m and key not in values:
                values[key] = float(m.group(1).replace(",", ""))
        if "Directly load the compiled graph" in body:
            values["compile_cache_hits"] = values.get("compile_cache_hits", 0) + 1
        if "Compiling a graph for compile range" in body:
            values["compile_cache_misses"] = values.get("compile_cache_misses", 0) + 1
    out = {"marks": marks, "events": events, "values": values, "fetch": fetches, "check_text": check,
           "ready": marks.get("ready")}
    need = ("download_start", "download_end", "engine_start", "ready")
    if all(k in marks for k in need) and all(k in events for k in ("engine_init", "load_start", "load_end",
                                                                       "compile_end", "capture_end")):
        e, m = events, marks
        capture_start = e["capture_end"] - values.get("capture_s", 0.0)
        seg = {"download": m["download_end"] - m["download_start"],
               "spawn": e["engine_init"] - m["engine_start"],
               "init_device": e["load_start"] - e["engine_init"],
               "load_model": e["load_end"] - e["load_start"],
               "compile": e["compile_end"] - e["load_end"],
               "kv": capture_start - e["compile_end"],
               "graphs": e["capture_end"] - capture_start,
               "tail": m["ready"] - e["capture_end"]}
        out["segments"] = seg
        out["ready_s"] = m["ready"] - m["download_start"]
        out["engine_s"] = m["ready"] - m["engine_start"]
    return out


def gke_stages(t_submit, pod, events, node, gpu_seen):
    """Seconds after pod submission (API timestamps are whole seconds)."""
    def cond(obj, kind):
        return next((ts(c.get("lastTransitionTime")) for c in obj.get("status", {}).get("conditions", [])
                     if c["type"] == kind), None)
    ev = {}
    for e in events.get("items", []):
        when = ts(e.get("eventTime") or e.get("firstTimestamp"))
        ev.setdefault(e["reason"], (when, e.get("message", "")))
    state = (pod.get("status", {}).get("containerStatuses") or [{}])[0].get("state", {})
    started = ts((state.get("running") or state.get("terminated") or {}).get("startedAt"))
    pulled_msg = ev.get("Pulled", (None, ""))[1]
    m = re.search(r"in ([\dhms.]+) \(([\dhms.]+) including waiting\)", pulled_msg)
    stages = {"scale_up_triggered": ev.get("TriggeredScaleUp", (None,))[0],
              "node_created": ts(node.get("metadata", {}).get("creationTimestamp")),
              "node_ready": cond(node, "Ready"), "gpu_allocatable_seen": gpu_seen,
              "pod_scheduled": cond(pod, "PodScheduled"), "pull_start": ev.get("Pulling", (None,))[0],
              "pulled": ev.get("Pulled", (None,))[0], "container_started": started}
    rel = {k: (v - t_submit if v else None) for k, v in stages.items()}
    rel["pull_reported"] = m.group(1) if m else pulled_msg[-120:] or None
    return rel


def cmd_run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configmap()
    arms = args.arms.split(",")
    for arm in arms:
        if arm not in ARMS:
            sys.exit(f"unknown arm {arm}")
        if arm == "ckpt-cc" and not args.cc_local or arm == "runai-cc" and not args.cc_runai:
            sys.exit(f"{arm} needs --cc-local / --cc-runai")
    (out / "campaign.json").open("a").write(json.dumps({"argv": sys.argv, "at": time.time(), "image": IMAGE,
                                                        "bucket": BUCKET, "prefix": PREFIX}) + "\n")
    jobs = [(arm, s) for s in range(args.samples) for arm in arms]  # interleave arms across samples
    cc = {"ckpt-cc": args.cc_local, "runai-cc": args.cc_runai}
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = []
        for i, (arm, s) in enumerate(jobs):
            futures.append(pool.submit(run_one, arm, s, args.pool, cc.get(arm), out,
                                       args.export if i == 0 else None, args.keep))
            time.sleep(args.stagger)
        records = [f.result() for f in futures]
    return 0 if all(r["ok"] for r in records) else 1


def summarize(dirs):
    by_arm = {}
    for d in map(Path, dirs):
        for line in (d / "runs.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("ok") and r.get("segments"):
                by_arm.setdefault(r["arm"], []).append(r)
    return by_arm


def cmd_report(args):
    by_arm = summarize(args.dirs)
    for arm in ARMS:
        runs = by_arm.get(arm, [])
        if not runs:
            continue
        totals = sorted(r["ready_s"] for r in runs)
        print(f"{arm:9s} n={len(runs)} ready median {statistics.median(totals):7.1f} s  runs {totals}")
        keys = runs[0]["segments"].keys()
        print("          " + " ".join(f"{k}={statistics.median(r['segments'][k] for r in runs):.1f}" for k in keys))
        g = [r["gke"] for r in runs if r.get("gke")]
        if g:
            print("          gke(s after submit): " + " ".join(
                f"{k}={statistics.median(x[k] for x in g if x.get(k) is not None):.0f}"
                for k in ("node_created", "node_ready", "gpu_allocatable_seen", "pull_start", "pulled",
                          "container_started") if any(x.get(k) is not None for x in g)))
        checks = {r["check_text"] for r in runs}
        print(f"          distinct greedy outputs: {len(checks)}  {next(iter(checks))[:60]!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("configmap")
    run = sub.add_parser("run")
    run.add_argument("--arms", required=True)
    run.add_argument("--samples", type=int, default=1)
    run.add_argument("--parallel", type=int, default=1)
    run.add_argument("--stagger", type=float, default=5.0, help="seconds between submissions")
    run.add_argument("--pool", default="gpu-pool-spot")
    run.add_argument("--cc-local", help="gs:// tarball of a peer's caches for ckpt-cc")
    run.add_argument("--cc-runai", help="gs:// tarball of a peer's caches for runai-cc")
    run.add_argument("--export", help="export the first job's caches to this gs:// tarball")
    run.add_argument("--keep", action="store_true", help="leave pod and node up")
    run.add_argument("--out", required=True)
    adopt = sub.add_parser("adopt", help="finish runs whose driver died: logs, export, k8s record, retire node")
    adopt.add_argument("pods", nargs="+", help="pod names (as-<run_id>); POD=gs://... also exports its caches")
    adopt.add_argument("--out", required=True)
    report = sub.add_parser("report")
    report.add_argument("dirs", nargs="+")
    args = parser.parse_args()
    if args.cmd == "configmap":
        configmap()
    elif args.cmd == "run":
        sys.exit(cmd_run(args))
    elif args.cmd == "adopt":
        out = Path(args.out)
        with ThreadPoolExecutor(max_workers=len(args.pods)) as pool:
            jobs = [pool.submit(run_one, None, None, None, None, out, export=(spec.partition("=")[2] or None),
                                adopt=spec.partition("=")[0]) for spec in args.pods]
            sys.exit(0 if all(j.result()["ok"] for j in jobs) else 1)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
