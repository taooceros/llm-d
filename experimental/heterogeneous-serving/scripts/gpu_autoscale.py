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
  warm-stock  ckpt-cc boot ("warm-stock-prime" record), then a stock restart on the same node (warm-stock record)
  pm-gcs    TPUFlex (pm_cuda.py plugin + pm_zygote.py + pm_state.py) with its startup state streamed from GCS
            (--state gs://...), then a pre-spawned TPUFlex restart on the same node (warm-pm record)
  pm-peer   TPUFlex with the state streamed from a live peer pod on another node (--peer POD)
  pm-seed   setup: builds the TPUFlex state (--state-dst gs://...) and then stays up as the live peer
  loo-<m>   leave-one-out ablations of pm-gcs (one TPUFlex mechanism off, PM_LOO=<m>, no warm-pm restart):
            loo-snapshot, loo-bank (--cc-bank), loo-bytecode, loo-kvpin (--cc-kvpin), loo-zygote, loo-overlap,
            loo-warmup; see pod_run.sh.  `loo DIR --matrix M` writes DIR/loo_summary.json
  loo-kvpin-only  loo-kvpin (PM_LOO=kvpin) whose --cc-kvpin is a warm stock cache of the unpinned key (with
            the Triton JIT cache), so that only the pin differs from pm-gcs in what the boot has to compile
  sllm      ServerlessLLM (OSDI'24, sllm-store 0.8.0 + its vLLM loader, scripts/gpu_autoscale/sllm_vllm.py): the
            ServerlessLLM-format checkpoint downloaded to tmpfs as ckpt does, then `--load-format serverless_llm`
            through the node's sllm-store daemon (started before T0, like a DaemonSet on a booted node)
  sllm-cc   sllm + the compile/JIT caches from a peer (--cc-sllm; same compile key as ckpt-cc: model=/model)
  warm-sllm   sllm-cc boot ("warm-sllm-prime" record), then a restart in the same pod with the checkpoint in the
            store's pinned pool (ServerlessLLM's DRAM tier) and every cache warm (warm-sllm record)
  sllm-seed setup: builds sllm-store in the image + a wheelhouse, converts and verifies the checkpoint, times
            load_dict per I/O thread count, uploads both under --sllm-dst (`sllm DIR` writes DIR/summary.json)

    scripts/gpu_autoscale.py run --arms sllm-seed --sllm-dst gs://.../sllm/<c> --node N --out DIR
    scripts/gpu_autoscale.py run --arms sllm-cc,warm-sllm --sllm gs://.../sllm/<c>/gemma-4-31b \
        --sllm-wheels gs://.../sllm/<c>/wheels --cc-sllm <tar> --node N --samples 3 --out DIR
    scripts/gpu_autoscale.py run --arms pm-seed --state-dst gs://.../pm-state/<c> --keep --out DIR
    scripts/gpu_autoscale.py run --arms pm-gcs --state gs://.../pm-state/<c> --out DIR
    scripts/gpu_autoscale.py run --arms pm-peer --peer as-pm-seed-0-xxxx --out DIR
    scripts/gpu_autoscale.py matrix --stock DIR549 --new DIR --commit C --out results/gpu-autoscale/matrix-C
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
CONFIGMAP = "gpu-autoscale-scripts"  # --configmap: a campaign's own copy, so concurrent drivers do not swap scripts
LOO_ARMS = ("loo-snapshot", "loo-bank", "loo-bytecode", "loo-kvpin", "loo-zygote", "loo-overlap", "loo-warmup",
            "loo-kvpin-only")
LOO_POD = {"loo-kvpin-only": "kvpin"}  # arm -> PM_LOO (default: the arm name without "loo-")
SLLM_ARMS = ("sllm", "sllm-cc", "warm-sllm")
ARMS = ("raw", "ckpt", "ckpt-cc", "runai", "runai-cc", "warm-stock", "pm-gcs", "pm-peer", "pm-seed") + LOO_ARMS \
    + SLLM_ARMS + ("sllm-seed",)
END_MARK = {"pm-gcs": "warm_check_done", "warm-stock": "warm_check_done", "pm-seed": "seed_ready",
            "warm-sllm": "warm_check_done", "sllm-seed": "sllm_seed_done"}
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
    files = {p.name: p.read_text() for p in (HERE / "pod_run.sh", HERE / "fetch.py", HERE / "pm_cuda.py",
                                            HERE / "pm_zygote.py", HERE / "pm_state.py", HERE / "wipe.py",
                                            HERE / "sllm_vllm.py", HERE / "sllm_loader.py",
                                            HERE / "sllm_store_fixes.patch")}
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": CONFIGMAP, "namespace": NAMESPACE},
          "data": files}
    kubectl("apply", "-f", "-", inp=json.dumps(cm))


def pod_spec(name, arm, pool, cc_uri, extra_env=None, node=None, zone=None):
    pod_arm = "pm-gcs" if arm in LOO_ARMS else arm  # a loo-* pod is a pm-gcs pod with PM_LOO set
    env = {"ARM": pod_arm, "BUCKET": BUCKET, "PREFIX": PREFIX, "CC_URI": cc_uri or "", "HOME": "/cache/home",
           "PYTHONUNBUFFERED": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", **(extra_env or {})}
    if arm in LOO_ARMS:
        env["PM_LOO"] = LOO_POD.get(arm, arm.removeprefix("loo-"))
    if node:
        env["WIPE"] = "1"
    mounts = [{"name": "scripts", "mountPath": "/scripts"}, {"name": "cache", "mountPath": "/cache"},
              {"name": "dshm", "mountPath": "/dev/shm"}]
    # pm-*: the 58 GB terminal weight snapshot lands in /dev/shm (tmpfs)
    volumes = [{"name": "scripts", "configMap": {"name": CONFIGMAP, "defaultMode": 0o755}},
               {"name": "cache", "emptyDir": {}},
               {"name": "dshm", "emptyDir": {"medium": "Memory",
                                             "sizeLimit": "80Gi" if pod_arm.startswith("pm-") else "16Gi"}}]
    if arm.startswith(("ckpt", "sllm")) or arm in ("warm-stock", "warm-sllm", "pm-seed"):
        # the TPU arms also stage the checkpoint in tmpfs; the sllm arms keep theirs there too
        mounts.append({"name": "model", "mountPath": "/model"})
        volumes.append({"name": "model", "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"}})
    annotations = {}
    if pool == QUEUED_POOL:  # flex-start with queued provisioning: the pod consumes its ProvisioningRequest
        annotations = {"autoscaling.x-k8s.io/consume-provisioning-request": name,
                       "autoscaling.x-k8s.io/provisioning-class-name": "queued-provisioning.gke.io"}
    return {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": name, "namespace": NAMESPACE, "labels": {"app": "gpu-autoscale", "arm": arm},
                     "annotations": annotations},
        "spec": {
            "restartPolicy": "Never", "terminationGracePeriodSeconds": 5,
            "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
                {"matchExpressions": [{"key": "kubernetes.io/hostname", "operator": "In", "values": [node]}]}
                if node else  # a held, reused node; else a fresh one from pool ("a,b" = either), optionally in one zone
                {"matchExpressions": [{"key": USED_LABEL, "operator": "DoesNotExist"},
                                      {"key": "cloud.google.com/gke-nodepool", "operator": "In",
                                       "values": pool.split(",")}]
                 + ([{"key": "topology.kubernetes.io/zone", "operator": "In", "values": [zone]}] if zone else [])}]}}},
            "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
                            {"key": "cloud.google.com/gke-queued", "operator": "Exists", "effect": "NoSchedule"}],
            "containers": [{
                "name": "engine", "image": IMAGE, "command": ["/scripts/pod_run.sh"],
                "env": [{"name": k, "value": v} for k, v in env.items()] + (
                    # the seeds upload: the node SA is read-only, so a host-minted token (see token_secret)
                    # arrives through a Secret and never appears in the pod spec or the run files
                    [{"name": "GCS_TOKEN", "valueFrom": {"secretKeyRef": {"name": TOKEN_SECRET, "key": "token"}}}]
                    if arm == "pm-seed" and env.get("STATE_DST") or arm == "sllm-seed" else []),
                "securityContext": {"privileged": True},  # /dev/fuse for the gcsfuse arm
                "resources": {"limits": {"nvidia.com/gpu": 1}},
                "volumeMounts": mounts}],
            "volumes": volumes}}


QUEUED_POOL = "gpu-pool-flexq"  # flex-start + queued provisioning (spot and plain flex-start were STOCKOUT 2026-10-24)


def provisioning_request(spec, max_run_s=10800):
    """PodTemplate + ProvisioningRequest (queued-provisioning.gke.io) for one pod on QUEUED_POOL; the node comes
    up once GCE can supply it, and is removed when the request is deleted or after max_run_s."""
    name = spec["metadata"]["name"]
    tmpl = {"apiVersion": "v1", "kind": "PodTemplate", "metadata": {"name": name, "namespace": NAMESPACE},
            "template": {"metadata": {"labels": spec["metadata"]["labels"]}, "spec": spec["spec"]}}
    req = {"apiVersion": "autoscaling.x-k8s.io/v1", "kind": "ProvisioningRequest",
           "metadata": {"name": name, "namespace": NAMESPACE},
           "spec": {"provisioningClassName": "queued-provisioning.gke.io",
                    "parameters": {"maxRunDurationSeconds": str(max_run_s)},
                    "podSets": [{"count": 1, "podTemplateRef": {"name": name}}]}}
    kubectl("apply", "-f", "-", inp=json.dumps(tmpl))
    kubectl("apply", "-f", "-", inp=json.dumps(req))


TOKEN_SECRET = "gpu-autoscale-gcs-token"


def token_secret(create):
    """create: a Secret holding a fresh access token of this host's gcloud account (valid ~1 h; the seed uploads
    within ~15 min of its start); otherwise delete it."""
    if not create:
        kubectl("delete", "secret", TOKEN_SECRET, "--ignore-not-found", check=False)
        return
    token = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True,
                           check=True).stdout.strip()
    kubectl("apply", "-f", "-", inp=json.dumps({"apiVersion": "v1", "kind": "Secret", "type": "Opaque",
                                                 "metadata": {"name": TOKEN_SECRET, "namespace": NAMESPACE},
                                                 "stringData": {"token": token}}))


def apply_pod(spec):
    if spec["metadata"].get("annotations"):
        provisioning_request(spec)
    kubectl("apply", "-f", "-", inp=json.dumps(spec))


def delete_pod(name):
    kubectl("delete", "pod", name, "--wait=false", check=False)
    kubectl("delete", "provisioningrequest", name, "--wait=false", check=False)
    kubectl("delete", "podtemplate", name, "--wait=false", check=False)


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


def export_cache(pod, uri, attempts=3, chunk=32 << 20):
    """The pod's $HOME caches (minus pip/HF downloads) as a GCS tarball, via this host.  The tarball is written
    inside the pod (/cache/export.tar, outside $HOME), then pulled in size-checked chunks and checked against the
    pod's sha256 before the upload: a streamed `kubectl exec` left truncated objects (2026-10-22: resets at ~118 MB;
    2026-10-25: the last 3-6 MB of a 133 MB stream missing in 3 of 3 attempts, exit codes 0)."""
    tar_cmd = "tar -C /cache/home -cf /cache/export.tar --exclude=./.cache/pip --exclude=./.cache/huggingface ."
    listing = kubectl("exec", pod, "--", "bash", "-c",
                      "cd /cache/home && du -sb .cache/* .triton .nv .config 2>/dev/null; "
                      "echo triton_cubins $(find .triton/cache -name '*.cubin' 2>/dev/null | wc -l); true")
    size, digest = kubectl("exec", pod, "--", "bash", "-c",
                           f"{tar_cmd} && stat -c %s /cache/export.tar && sha256sum /cache/export.tar").split()[:2]
    size, retries = int(size), 0
    local = Path("/tmp") / f"export-{pod}.tar"
    try:
        with local.open("wb") as f:
            for i in range((size + chunk - 1) // chunk):
                want = min(chunk, size - i * chunk)
                for _ in range(attempts):
                    data = subprocess.run(["kubectl", "--kubeconfig", KUBECONFIG, "-n", NAMESPACE, "exec", pod, "--",
                                           "dd", "if=/cache/export.tar", f"bs={chunk}", f"skip={i}", "count=1",
                                           "iflag=fullblock", "status=none"], capture_output=True).stdout
                    if len(data) == want:
                        break
                    retries += 1
                else:
                    raise RuntimeError(f"export {pod}: chunk {i} short after {attempts} attempts")
                f.write(data)
        got = subprocess.run(["sha256sum", str(local)], capture_output=True, text=True, check=True).stdout.split()[0]
        if got != digest:
            raise RuntimeError(f"export {pod}: sha256 {got} != pod's {digest}")
        up = subprocess.run(["gcloud", "storage", "cp", str(local), uri], capture_output=True, text=True)
        listed = subprocess.run(["gcloud", "storage", "ls", "-l", uri], capture_output=True, text=True).stdout.split()
        if up.returncode or not listed or listed[0] != str(size):
            raise RuntimeError(f"export {pod} -> {uri}: cp rc={up.returncode} listed={listed[:1]} size={size}")
    finally:
        local.unlink(missing_ok=True)
        kubectl("exec", pod, "--", "rm", "-f", "/cache/export.tar", check=False)
    return {"uri": uri, "bytes": size, "sha256": digest, "chunk_retries": retries, "du": listing}


def poll_logs(name, logfile, deadline, end_mark="check_done", interval=5.0):
    """Snapshot `kubectl logs --timestamps` into logfile every few seconds until MARK <end_mark> / *engine_died,
    the container exits, or the deadline.  Polling, not `logs -f`: on 2026-10-22 followed streams both ended
    early (mid-compile) and hung for an hour on five concurrent pods while the kubelet log stayed complete."""
    while True:
        text = kubectl("logs", "--timestamps", name, check=False)
        logfile.write_text(text)
        if re.search(rf"^\S+ MARK ({end_mark}|\w*engine_died) ", text, re.M) or time.time() > deadline:
            return
        state = (kjson("get", "pod", name)["status"].get("containerStatuses") or [{}])[0].get("state", {})
        if "terminated" in state:
            logfile.write_text(kubectl("logs", "--timestamps", name, check=False))
            return
        time.sleep(interval)


def run_one(arm, sample, pool, cc_uri, out, export=None, keep=False, timeout=3600, adopt=None, extra_env=None,
            held=None, zone=None):
    """Boot one arm on a fresh node; with adopt=<pod name>, finish an already-submitted run instead (its
    submission time is the pod's creationTimestamp and the GPU-allocatable time is unknown)."""
    if adopt:
        name, rid = adopt, adopt.removeprefix("as-")
        pod = kjson("get", "pod", name)
        env = {e["name"]: e.get("value", "") for e in pod["spec"]["containers"][0]["env"]}  # k8s drops "" values
        arm, cc_uri, sample = env["ARM"], env["CC_URI"] or None, int(rid.split("-")[-2])
        pool = ",".join(next(e["values"] for e in pod["spec"]["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
            if e["key"] == "cloud.google.com/gke-nodepool"))
    else:
        rid = f"{arm}-{sample}-{''.join(random.choices(string.ascii_lowercase + string.digits, k=4))}"
        name = f"as-{rid}"
    record = {"run_id": rid, "arm": arm, "sample": sample, "pool": pool, "zone": zone, "cc_uri": cc_uri, "image": IMAGE,
              "ok": False}
    if adopt:
        record["t_submit"], record["adopted"] = ts(pod["metadata"]["creationTimestamp"]), True
    else:
        record["t_submit"] = time.time()
        apply_pod(pod_spec(name, arm, pool, cc_uri, extra_env, held, zone))
        log(rid, "submitted")
    node, gpu_seen, deadline, subs = None, None, time.time() + timeout, []
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
        poll_logs(name, logfile, deadline, END_MARK.get(arm, "check_done"))
        text = logfile.read_text()
        parsed = parse_log(text, kind="pm" if arm.startswith("pm-") or arm in LOO_ARMS else "stock")
        record.update(parsed)
        record["ok"] = parsed.get("ready_s") is not None and parsed.get("check_text") is not None
        if arm == "pm-seed":  # setup, not a timed arm: built, uploaded (if asked) and serving
            record["ok"] = "seed_ready" in parsed["marks"] and "seed_upload_failed" not in parsed["marks"]
        if arm == "sllm-seed":  # setup: wheelhouse + checkpoint uploaded, both verification boots answered
            record["ok"] = "sllm_seed_done" in parsed["marks"]
            record["sllm_seed"] = sllm_seed_report(text)
        for sub, prefix, kind in WARM_STREAMS.get(arm, ()):
            w = parse_log(text, prefix=prefix, kind=kind)
            subs.append({**{k: record[k] for k in ("arm", "sample", "pool", "cc_uri", "image")},
                         "run_id": f"{rid}-w", "arm": sub, "parent": rid, **w,
                         "ok": w.get("ready_s") is not None and w.get("check_text") is not None})
        if arm in WARM_STREAMS:  # the first boot of a warm-stock pod is the prime boot
            record["arm"] = {"warm-stock": "warm-stock-prime", "warm-sllm": "warm-sllm-prime"}.get(arm, arm)
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
        record["node_pool"] = node_obj.get("metadata", {}).get("labels", {}).get("cloud.google.com/gke-nodepool")
        record["gke"] = gke_stages(record["t_submit"], pod, events, node_obj, gpu_seen)
        if held:  # reused node: never retired; GKE stages not measured on GPU (user decision, H100 stockout)
            record["held_node"], record["gke"] = held, None
            record["note"] = "reused-node"
            record["wipe"] = next((json.loads(line.split("WIPE ", 1)[1]) for line in text.splitlines()
                                   if " WIPE " in line), None) if "text" in locals() else None
            t0_receipt = next((json.loads(line.split("WIPE_T0 ", 1)[1]) for line in text.splitlines()
                               if " WIPE_T0 " in line), None) if "text" in locals() else None
            if t0_receipt:  # sllm arms with SLLM_STORE_ISOLATE=1: the second drop, right before T0
                record["wipe_t0"] = t0_receipt
            for sub in subs:
                sub["note"], sub["held_node"], sub["wipe"] = "reused-node", held, record["wipe"]
            keep = True
            if arm != "pm-seed":  # gone before the next run's pod can take the GPU
                delete_pod(name)
                t_del = time.time()
                while kubectl("get", "pod", name, "--ignore-not-found", "-o", "name", check=False).strip():
                    if time.time() - t_del > 300:
                        break
                    time.sleep(1)
                else:
                    record["pod_gone_s"] = round(time.time() - t_del, 1)
        if arm == "pm-seed":
            keep = True  # the live peer of pm-peer; deleted (and its node retired) by `retire`
        if not keep:
            delete_pod(name)
            if node:
                record["retire"] = retire_node(node)
        with _print_lock, (out / "runs.jsonl").open("a") as f:
            for r in [record] + subs:
                r["node"], r["node_pool"] = node, record["node_pool"]
                f.write(json.dumps(r) + "\n")
    for r in [record] + subs:
        log(r["run_id"], r["arm"], "ok" if r["ok"] else "FAILED", f"ready {r.get('ready_s')}",
            " ".join(f"{k}={v:.1f}" for k, v in (r.get("segments") or {}).items()))
    return record


# warm restarts in the same pod: (record arm, log-line prefix of the second engine, segment kind)
WARM_STREAMS = {"pm-gcs": (("warm-pm", "W2| ", "warm-pm"),), "warm-stock": (("warm-stock", "W2| ", "warm-stock"),),
                "warm-sllm": (("warm-sllm", "W2| ", "warm-stock"),)}


LOG_ANCHORS = {  # first log line containing the text -> event (kubelet timestamp)
    "engine_init": "Initializing a V1 LLM engine",
    "load_start": "Starting to load model",
    "load_end": "Model loading took",
    "compile_end": ("torch.compile took", "torch.compile and initial profiling/warmup run together took"),
    "kv_size": "GPU KV cache size",
    "warmup_end": "Initial profiling/warmup run took",
    "cudagraph_profile": "Profiling CUDA graph memory",
    "capture_end": "Graph capturing finished in",
    "init_engine_end": "init engine (profile, create kv cache, warmup model) took",
}


PM_PATTERNS = (  # TPUFlex plugin / zygote / fetch lines -> pm values (first match each)
    ("bank_wait_s", r"PM_BANK_WAIT waited=([\d.]+)s"),
    ("manifest_wait_s", r"PM_SNAP_RESTORE started: manifest waited ([\d.]+)s"),
    ("zero_fill_s", r"parameters zero-filled in ([\d.]+)s"),
    ("join_wait_s", r"PM_SNAP_JOIN at=\S+ waited=([\d.]+)s"),
    ("restore_wall_s", r"PM_SNAP_JOIN at=\S+ waited=[\d.]+s restore_wall=([\d.]+)s"),
    ("restore_GBps", r"restore_wall=[\d.]+s \(([\d.]+) GB/s\)"),
    ("restore_copy_s", r"PM_SNAP_JOIN .* copy=([\d.]+)s"),
    ("restore_shard_wait_s", r"PM_SNAP_JOIN .* shard_wait=([\d.]+)s"),
    ("zygote_ready_s", r"PM_ZYGOTE_READY took=([\d.]+)s"),
    ("kv_bytes", r"PM_KV_BYTES bytes=(\d+)"),
    ("init_device_s", r"PM_INIT_DEVICE took=([\d.]+)s"),
    ("warm_s", r"PM_WARM api_server took=([\d.]+)s"),
    ("snap_save_s", r"PM_SNAP_SAVE .* took=([\d.]+)s"),
    ("copy_done_s", r"PM_SNAP_COPY_DONE after=([\d.]+)s"),
)
PM_EVENTS = {"parked": "PM_PARKED", "gate_open": "PM_GATE_OPEN", "zygote_ready": "PM_ZYGOTE_READY",
             "bank_wait_end": "PM_BANK_WAIT", "join_end": "PM_SNAP_JOIN", "restore_start": "PM_SNAP_RESTORE started",
             "copy_done": "PM_SNAP_COPY_DONE"}
STREAM_PREFIXES = ("W2| ", "S0| ", "S1| ", "S2| ", "S3| ", "S4| ", "SLLM| ")
# tqdm's final frame of each CUDA graph capture round; these lines carry neither the kubelet timestamp nor the
# stream prefix, so a round belongs to the engine whose "Graph capturing finished" line comes next
CAPTURE_ROUND = re.compile(r"Capturing CUDA graphs \((?:mixed prefill-decode, )?(?:decode, )?(PIECEWISE|FULL)\): +100%"
                           r"\|[^|]*\| *(\d+)/\2 \[(\d+):(\d+)<")


def parse_log(text, prefix="", kind="stock"):
    """One engine's lines of a pod log: with prefix, only the lines carrying it (stripped); without, the lines
    of no prefixed stream. MARK and FETCH lines belong to every stream; a stream's own MARKs are warm_*."""
    events, marks, values, fetches, check, pm, info = {}, {}, {}, [], None, {}, {}
    rounds, pending = None, {}  # capture rounds: {PIECEWISE|FULL: tqdm elapsed s} of this stream's first capture
    for line in text.splitlines():
        m = CAPTURE_ROUND.search(line)
        if m:
            pending[m.group(1)] = int(m.group(3)) * 60 + int(m.group(4))
            continue
        stamp, _, body = line.partition(" ")
        if "Graph capturing finished in" in body:
            mine = body.startswith(prefix) if prefix else not body.startswith(STREAM_PREFIXES)
            if mine and rounds is None:
                rounds = dict(pending)
            pending = {}
        if body.startswith("MARK "):
            marks.setdefault(body.split()[1], ts(stamp))  # one clock (kubelet) for MARK lines and vLLM log lines
            continue
        if body.startswith("FETCH "):
            fetches.append(json.loads(body[6:]))
            continue
        if prefix:
            if not body.startswith(prefix):
                continue
            body = body[len(prefix):]
        elif body.startswith(STREAM_PREFIXES):
            continue
        if body.startswith("CHECK "):
            try:
                choice = json.loads(body[6:])["choices"][0]
                check = choice["message"]["content"] if "message" in choice else choice["text"]
            except (ValueError, KeyError, IndexError):
                check = None
        for key, needles in LOG_ANCHORS.items():
            if key not in events and any(n in body for n in ((needles,) if isinstance(needles, str) else needles)):
                events[key] = ts(stamp)
        for key, needle in PM_EVENTS.items():
            if key not in events and needle in body:
                events[key] = ts(stamp)
        for key, pattern in PM_PATTERNS:
            m = re.search(pattern, body)
            if m and key not in pm:
                pm[key] = float(m.group(1))
        m = re.search(r"PM_SNAP_JOIN at=(\S+)", body)
        if m and "join_at" not in pm:
            pm["join_at"] = m.group(1)
        m = re.search(r"^TRITON_CACHE at=(\w+) cubins=(\d+)", body)
        if m:
            info.setdefault("triton_cubins", {}).setdefault(m.group(1), int(m.group(2)))
        for key, pattern in (("weights_s", r"Loading weights took ([\d.]+) seconds"),
                             ("model_loading_s", r"Model loading took [\d.]+ GiB(?: memory)? and ([\d.]+) seconds"),
                             ("compile_s", r"torch\.compile took ([\d.]+) s in total"),
                             ("compile_profile_s", r"torch\.compile and initial profiling/warmup run together took ([\d.]+) s"),
                             ("dynamo_s", r"Dynamo bytecode transform time: ([\d.]+) s"),
                             ("capture_s", r"Graph capturing finished in (\d+) secs"),
                             ("capture_gib", r"Graph capturing finished in \d+ secs, took ([\d.]+) GiB"),
                             ("kv_tokens", r"GPU KV cache size: ([\d,]+) tokens"),
                             ("kv_gib", r"Available KV cache memory: ([\d.]+) GiB"),
                             ("init_engine_s", r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"),
                             ("sllm_load_dict_s", r"SLLM_LOAD_DICT took=([\d.]+)s"),
                             ("sllm_load_bytes", r"SLLM_LOAD_DICT .* bytes=(\d+)"),
                             ("sllm_load_GBps", r"SLLM_LOAD_DICT .* GBps=([\d.]+)"),
                             ("sllm_load_model_s", r"SLLM_LOAD_MODEL took=([\d.]+)s")):
            m = re.search(pattern, body)
            if m and key not in values:
                values[key] = float(m.group(1).replace(",", ""))
        for key, pattern in (("compile_cache_dir", r"Using cache directory: (\S+) for vLLM's torch\.compile"),
                             ("sllm_store", r"^SLLM_STORE (.*)$"), ("sllm_wheels", r"^SLLM_WHEELS (.*)$"),
                             ("kv_pinned", r"PM_KV_BYTES bytes=\d+ took=[\d.]+s pinned=(\w+)"),
                             ("weights_uri", r"PM_WEIGHTS_URI \S+ -> (\S+)"),
                             ("pm_loo", r"^PM_LOO=(.*)$")):
            m = re.search(pattern, body)
            if m and key not in info:
                info[key] = m.group(1)
        if "SLLM_PLUGIN registered" in body:
            info["sllm_plugin"] = info.get("sllm_plugin", 0) + 1
        if "PM_ZYGOTE client" in body:
            info["zygote_clients"] = info.get("zygote_clients", 0) + 1
        if "Loading safetensors using Runai Model Streamer" in line:  # tqdm lines carry no kubelet timestamp
            info["runai_streamer"] = True
        if "Directly load the compiled graph" in body:
            values["compile_cache_hits"] = values.get("compile_cache_hits", 0) + 1
        if "Compiling a graph for compile range" in body:
            values["compile_cache_misses"] = values.get("compile_cache_misses", 0) + 1
    if rounds:
        values.update({f"capture_{k.lower()}_s": float(x) for k, x in rounds.items()})
    if "warmup_end" in events and "kv_size" in events:  # the KV window: warm-up run -> KV sized (profiling, if any)
        values["warmup_to_kv_s"] = events["kv_size"] - events["warmup_end"]
    info["cudagraph_mem_profiled"] = "cudagraph_profile" in events
    out = {"marks": marks, "events": events, "values": values, "pm": pm, "fetch": fetches, "check_text": check,
           "kind": kind, "info": info}
    out.update(derive(out, kind))
    return out


SEGMENT_KEYS = ("download", "spawn", "init_device", "lazy", "load_model", "compile", "kv", "graphs", "tail")


def derive(r, kind):
    """Critical-path segments (s) of one boot, from T0 to the greedy check's reply (Ready = /health OK and the
    check answered).  kind: stock / pm (fresh host: T0 = first byte of delivery, i.e. download_start), warm-stock
    (T0 = relaunch), warm-pm (T0 = gate open; the engine was pre-spawned and parked before init_device).
    lazy = time blocked on state that had not arrived: PM bank/manifest waits and the snapshot join (taken out of
    the segment they fall in); for the stock GCS-reading arms the matrix infers it from load_model."""
    m, e, v, pm = r["marks"], r["events"], r["values"], r.get("pm") or {}
    if kind in ("stock", "pm"):
        t0, launch, end, health = m.get("download_start"), m.get("engine_start"), m.get("check_done"), m.get("ready")
    elif kind == "warm-stock":
        t0 = launch = m.get("warm_launch")
        end, health = m.get("warm_check_done"), m.get("warm_ready")
    else:  # warm-pm
        t0 = launch = m.get("warm_gate")
        end, health = m.get("warm_check_done"), m.get("warm_ready")
    need = ("engine_init", "load_start", "load_end", "compile_end", "capture_end")
    if None in (t0, launch, end) or not all(k in e for k in need):
        return {}
    capture_start = e["capture_end"] - v.get("capture_s", 0.0)
    init = max(e["engine_init"], launch)
    seg = {"download": launch - t0, "spawn": init - launch, "init_device": e["load_start"] - init, "lazy": 0.0,
           "load_model": e["load_end"] - e["load_start"], "compile": e["compile_end"] - e["load_end"],
           "kv": capture_start - e["compile_end"], "graphs": e["capture_end"] - capture_start,
           "tail": end - e["capture_end"]}
    for wait, where in (("bank_wait_s", "compile"), ("manifest_wait_s", "compile"),
                        ("join_wait_s", "tail" if pm.get("join_at") == "after_warmup" else "kv")):
        w = min(pm.get(wait, 0.0), seg[where])
        seg[where] -= w
        seg["lazy"] += w
    out = {"segments": seg, "ready_s": end - t0, "health_s": health - t0 if health else None,
           "engine_s": end - launch}
    if "copy_done" in e:  # the snapshot copy against CUDA graph capture: < 0 done before it, 0..graphs during it
        out["copy_done_after_capture_start_s"] = e["copy_done"] - capture_start
    if kind == "warm-pm":
        out["from_launch_s"] = end - m["warm_launch"] if "warm_launch" in m else None
        out["downtime_s"] = end - m["warm_stop"] if "warm_stop" in m else None
        out["prespawn_s"] = m.get("warm_parked", 0) - m.get("warm_launch", 0) if "warm_parked" in m else None
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
        if (arm == "ckpt-cc" and not args.cc_local or arm == "runai-cc" and not args.cc_runai
                or arm == "loo-bank" and not args.cc_bank or arm in ("loo-kvpin", "loo-kvpin-only") and not args.cc_kvpin):
            sys.exit(f"{arm} needs --cc-local / --cc-runai / --cc-bank / --cc-kvpin")
        if arm in LOO_ARMS and not args.state:
            sys.exit(f"{arm} needs --state")
        if (arm == "warm-stock" and not args.cc_local or arm == "pm-gcs" and not args.state
                or arm == "pm-peer" and not args.peer or arm == "pm-seed" and not (args.state_dst or args.state)):
            sys.exit(f"{arm} needs --cc-local / --state / --peer / --state-dst (pm-seed: or --state)")
        if (arm in SLLM_ARMS and not (args.sllm and args.sllm_wheels) or arm in ("sllm-cc", "warm-sllm")
                and not args.cc_sllm or arm == "sllm-seed" and not args.sllm_dst):
            sys.exit(f"{arm} needs --sllm + --sllm-wheels (+ --cc-sllm for sllm-cc / warm-sllm); sllm-seed --sllm-dst")
    env = dict(kv.split("=", 1) for kv in args.env)
    if "pm-peer" in arms:
        ip = kjson("get", "pod", args.peer)["status"]["podIP"]
        env_peer = {"STATE_SRC": f"http://{ip}:8080"}
    (out / "campaign.json").open("a").write(json.dumps({"argv": sys.argv, "at": time.time(), "image": IMAGE,
                                                        "bucket": BUCKET, "prefix": PREFIX, "env": env}) + "\n")
    jobs = [(arm, s) for s in range(args.samples) for arm in arms]  # interleave arms across samples
    cc = {"ckpt-cc": args.cc_local, "runai-cc": args.cc_runai, "warm-stock": args.cc_local,
          "loo-bank": args.cc_bank, "loo-kvpin": args.cc_kvpin, "loo-kvpin-only": args.cc_kvpin,
          "sllm-cc": args.cc_sllm, "warm-sllm": args.cc_sllm}
    # pm-seed with --state-dst builds and publishes the state; with --state it boots from a published one (a peer)
    arm_env = {"pm-gcs": {"STATE_SRC": args.state or ""}, "pm-peer": env_peer if "pm-peer" in arms else {},
               "pm-seed": {"STATE_DST": args.state_dst} if args.state_dst else {"STATE_SRC": args.state or ""},
               **{a: {"STATE_SRC": args.state or ""} for a in LOO_ARMS},
               **{a: {"SLLM_SRC": args.sllm or "", "SLLM_WHEELS": args.sllm_wheels or ""} for a in SLLM_ARMS},
               "sllm-seed": {"SLLM_DST": args.sllm_dst or ""}}
    upload = "pm-seed" in arms and bool(args.state_dst) or "sllm-seed" in arms
    if upload:
        token_secret(True)
    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = []
            for i, (arm, s) in enumerate(jobs):
                futures.append(pool.submit(run_one, arm, s + args.sample_base, args.pool, cc.get(arm), out,
                                           args.export if i == 0 else None, args.keep,
                                           extra_env={**arm_env.get(arm, {}), **env}, held=args.node, zone=args.zone))
                time.sleep(args.stagger)
            records = [f.result() for f in futures]
    finally:
        if upload:  # the seed has uploaded by seed_ready (run_one's end mark)
            token_secret(False)
    return 0 if all(r["ok"] for r in records) else 1


def summarize(dirs):
    by_arm = {}
    for d in map(Path, dirs):
        for line in (d / "runs.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("ok") and r.get("segments"):
                by_arm.setdefault(r["arm"], []).append(r)
    return by_arm

# canonical arm -> (campaign arm, segment kind, phases that do not exist in that arm's pipeline)
CANONICAL = {
    "stock-lazy": ("raw", "stock", ("download",)),
    "stock-ckpt": ("ckpt", "stock", ("lazy",)),
    "stock-ckpt-cc": ("ckpt-cc", "stock", ("lazy",)),
    "stock-runai": ("runai", "stock", ("download",)),
    "stock-runai-cc": ("runai-cc", "stock", ()),
    "pm-gcs": ("pm-gcs", "pm", ()),
    "pm-peer": ("pm-peer", "pm", ()),
    "warm-stock": ("warm-stock", "warm-stock", ("download", "lazy")),
    "warm-pm": ("warm-pm", "warm-pm", ("download", "spawn")),
}
LOCAL_LOAD = ("ckpt", "ckpt-cc")  # same checkpoint, same stock loader, from tmpfs: the lazy-inference reference
GCS_READING = ("raw", "runai", "runai-cc")


def matrix_runs(stock_dir, new_dirs, exclude=()):
    """ok runs per canonical arm, re-derived with derive() (ready = greedy-check reply).  The stock arms are the
    runs autoscale_gpu_summary.json used; each re-derived phase must equal that summary's (tail aside: it now
    ends at the check reply instead of /health), and lazy is inferred the same way (load_model minus the mean
    load_model of the tmpfs arms)."""
    stock_dir = Path(stock_dir)
    summary = json.loads((stock_dir / "autoscale_gpu_summary.json").read_text())
    used = {r["run_id"]: r for a in LOCAL_LOAD + GCS_READING for r in summary[a]["runs"]}
    runs = {}
    for line in (stock_dir / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["run_id"] in used:
            r.update(derive(r, "stock"))
            runs.setdefault(r["arm"], []).append(r)
    ref = statistics.mean(r["segments"]["load_model"] for a in LOCAL_LOAD for r in runs[a])
    assert abs(ref - summary["local_load_reference_s"]) < 1e-6, (ref, summary["local_load_reference_s"])
    for arm in GCS_READING:
        for r in runs[arm]:
            s = r["segments"]
            s["lazy"] = max(0.0, s["load_model"] - ref)
            s["load_model"] -= s["lazy"]
    for arm, rs in runs.items():
        for r in rs:
            old = used[r["run_id"]]
            bad = {k: (r["segments"][k], old[k]) for k in SEGMENT_KEYS if k not in ("tail", "download")
                   and abs(r["segments"][k] - old[k]) > 1e-3}
            assert not bad, (r["run_id"], bad)
    for d in map(Path, new_dirs):
        for line in (d / "runs.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["run_id"].removesuffix("-w") in exclude:  # a pod's warm record goes with it
                continue
            if r.get("ok") and r["arm"] in ("pm-gcs", "pm-peer", "warm-pm", "warm-stock"):
                r.update(derive(r, r.get("kind") or CANONICAL[r["arm"]][1]))
                runs.setdefault(r["arm"], []).append(r)
    return runs, ref


def gke_row(g):
    if not g or g.get("container_started") is None or (g.get("node_created") or -1) <= 0:
        return None  # landed on a node that existed before submission: not a scale-up
    return {"provision": g["node_created"], "accel_ready": g["pod_scheduled"] - g["node_created"],
            "pull": g["container_started"] - g["pod_scheduled"], "total": g["container_started"]}


GKE_NOTE = "GKE stages not measured on GPU (user decision; H100 stockout)"


def wipe_summary(w):
    """The per-run fresh-host receipt (wipe.py), condensed."""
    if not w:
        return None
    return {"leftovers": w["empty_dirs_leftovers"], "drop_caches": w["drop_caches"],
            "gpu_free_frac": round(w["gpu_free_bytes"] / w["gpu_total_bytes"], 4) if w["gpu_total_bytes"] else None,
            "page_cache_GB_before_after": [w["before"]["meminfo"]["Cached_GB"], w["after"]["meminfo"]["Cached_GB"]],
            "image_resident_bytes_before_after": [w["before"]["image"]["resident_bytes"],
                                                  w["after"]["image"]["resident_bytes"]],
            "image_bytes": w["after"]["image"]["bytes"]}


def cmd_matrix(args):
    runs, ref = matrix_runs(args.stock, args.new, set(args.exclude))
    expected = args.expect_text
    meta = json.loads(Path(args.meta).read_text()) if args.meta else {}
    cell = {"cell": "gpu-torch", "hardware": "1x H100 80GB HBM3 (GKE spot a3-highgpu-1g, asia-southeast1)", "tp": 1,
            "hosts": 1, "engine": {"max_model_len": 8192, "max_num_seqs": 512, "max_num_batched_tokens": 2048,
                                   "extra": meta.get("engine_extra", "")},
            "versions": meta.get("versions", {}), "image_or_venv": IMAGE, "code_commit": args.commit,
            "t0": meta.get("t0", ""), "arms": {},
            "sources": {"stock": args.stock, "new": args.new, "excluded_runs": sorted(args.exclude)}}
    for name, (arm, kind, absent) in CANONICAL.items():
        rs = sorted(runs.get(arm, []), key=lambda r: r["run_id"])
        notes = "; ".join(x for x in (GKE_NOTE, meta.get("notes", {}).get(name, "")) if x)
        if not rs:
            cell["arms"][name] = {"status": meta.get("status", {}).get(name, "dnf"), "n": 0, "runs": [],
                                  "median": None, "phases": {k: None for k in SEGMENT_KEYS}, "gke": None,
                                  "note": notes}
            continue
        texts = {r["check_text"] for r in rs}
        if texts != {expected}:
            raise SystemExit(f"{name}: greedy outputs {texts} != {expected!r}")
        phases = {k: None if k in absent else statistics.median(r["segments"][k] for r in rs) for k in SEGMENT_KEYS}
        g = [x for x in (gke_row(r.get("gke")) for r in rs) if x] if not kind.startswith("warm") else []
        row = {"status": "ok", "n": len(rs), "runs": [round(r["ready_s"], 2) for r in rs],
               "median": statistics.median(r["ready_s"] for r in rs),
               "phases": phases, "gke": None, "note": notes,
               "run_ids": [r["run_id"] for r in rs], "health_s": [round(r["health_s"], 2) for r in rs],
               "greedy_check": {"passed": len(rs), "text": expected}}
        if g:  # campaign-549bd16 ran each stock boot on a node it scaled up: kept for reference only
            row["gke_reference_549bd16"] = {
                **{k: statistics.median(x[k] for x in g) for k in ("provision", "accel_ready", "pull", "total")},
                "n": len(g), "note": "scale-up stages of these runs (s after pod submission); not part of the matrix"}
        if any(r.get("wipe") for r in rs):
            row["wipe_receipts"] = {r["run_id"]: wipe_summary(r.get("wipe")) for r in rs}
        if any(r.get("note") == "reused-node" for r in rs):
            row["held_node"] = sorted({r.get("held_node") for r in rs if r.get("held_node")})
        if kind == "warm-pm":
            row["from_launch_s"] = [round(r["from_launch_s"], 2) for r in rs]
            row["downtime_s"] = [round(r["downtime_s"], 2) for r in rs]
            row["median_from_launch"] = statistics.median(r["from_launch_s"] for r in rs)
            row["median_downtime"] = statistics.median(r["downtime_s"] for r in rs)
        if kind in ("pm", "warm-pm"):
            keys = sorted({k for r in rs for k in (r.get("pm") or {}) if k != "join_at"})
            row["pm"] = {k: statistics.median(r["pm"][k] for r in rs if k in r["pm"]) for k in keys}
            if all("copy_done_after_capture_start_s" in r for r in rs):
                row["copy_done_after_capture_start_s"] = [round(r["copy_done_after_capture_start_s"], 2) for r in rs]
            fetch = [f for r in rs for f in r.get("fetch", []) if f.get("kind") == "pm"]
            if fetch and kind == "pm":
                row["delivery"] = {k: statistics.median(f[k] for f in fetch) for k in ("seconds", "GBps", "first_shard_s")
                                   if all(f.get(k) is not None for f in fetch)}
        cell["arms"][name] = row
    cell["stock_lazy_reference_load_model_s"] = ref
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix_summary.json").write_text(json.dumps(cell, indent=1) + "\n")
    for name, row in cell["arms"].items():
        ph = " ".join(f"{k}={v:.1f}" for k, v in row["phases"].items() if v is not None)
        print(f"{name:15s} {row['status']} n={row['n']} median={row['median'] and round(row['median'], 1)} "
              f"runs={row['runs']} {ph} gke={row['gke'] and {k: round(v) for k, v in row['gke'].items()}}")



LOO_REMOVED = {  # arm -> (mechanism removed, what the phase falls back to)
    "pm-gcs": (None, None),
    "loo-snapshot": ("terminal weight snapshot (zero-fill + 2-stream restore joined at serve)",
                     "stock Run:ai streamer loader, safetensors streamed from gs:// inside load_model (as stock-runai-cc)"),
    "loo-bank": ("compile bank in the PM delivery (bank.tar, overlapped with engine start, PM_BANK_DONE gate)",
                 "the stack's own compile cache for the same (pinned) key, downloaded into $HOME before the engine "
                 "launches (as stock-runai-cc)"),
    "loo-bytecode": ("bytecode in the PM delivery (pyc.tar + PYTHONPYCACHEPREFIX)",
                     "the image's own bytecode handling (no prefix; as every stock arm)"),
    "loo-kvpin": ("KV pin (--kv-cache-memory-bytes) and, because the pin is part of the compile key, the bank",
                  "vLLM profiles KV and CUDA-graph memory; stock compile cache of the unpinned key "
                  "(local-549bd16.tar, model=/model) downloaded before launch (as stock-runai-cc)"),
    "loo-zygote": ("zygote forkserver (PM_ZYGOTE_SOCK)", "vLLM spawns the EngineCore (fresh interpreter)"),
    "loo-overlap": ("delivery overlapped with engine start",
                    "the PM delivery (all 32 shards) completes before the engine launches"),
    "loo-warmup": ("tokenizer/chat-template warm-up thread (PM_WARM=1)", "PM_WARM=0: loaded on the first request"),
    "loo-kvpin-only": ("KV pin (--kv-cache-memory-bytes) only; the bank's key includes the pin, so the compile cache "
                       "comes from the stock path instead, with the same content (warm stock export incl. .triton)",
                       "vLLM profiles KV and CUDA-graph memory; WARM stock compile cache of the unpinned key "
                       "(local-warm-<rev>.tar, model=/model, exported after a warm boot: 19 Triton kernels) "
                       "downloaded before launch"),
}


def mechanism_check(r):
    """What each PM mechanism actually did in one boot (from its log), and whether that matches the arm:
    the removed mechanism off, every other one on, and a compile-cache hit in every arm."""
    info, pm, v, m = r.get("info") or {}, r.get("pm") or {}, r.get("values") or {}, r.get("marks") or {}
    pmf = next((f for f in r.get("fetch", []) if f.get("kind") == "pm"), {})
    ev = pmf.get("events", {})
    cc = next((f for f in r.get("fetch", []) if f.get("kind") == "tar"), None)
    fetch_end = m.get("fetch_done") or (  # loo-overlap marks the end of delivery; else T0 + the fetch's own time
        m["download_start"] + pmf["seconds"] if pmf.get("seconds") is not None and "download_start" in m else None)
    shards = sum(1 for k in ev if k.startswith("snap/shard"))
    c = {
        "snapshot_restore": "join_wait_s" in pm, "restore_join_at": pm.get("join_at"),
        "restore_join_waited_s": pm.get("join_wait_s"), "restore_wall_s": pm.get("restore_wall_s"),
        "shards_delivered": shards, "runai_weights": bool(info.get("runai_streamer") and info.get("weights_uri")),
        "bank_delivered": "bank.tar" in ev, "bank_wait_s": pm.get("bank_wait_s"),
        "stock_cc_tar": cc and {"uri": cc["uri"], "bytes": cc["bytes"], "seconds": cc["seconds"]},
        "compile_cache_dir": (info.get("compile_cache_dir") or "").split("torch_compile_cache/")[-1].split("/")[0],
        "compile_cache_hits": int(v.get("compile_cache_hits", 0)), "compile_cache_misses": int(v.get("compile_cache_misses", 0)),
        "compile_s": v.get("compile_s"),
        "kv_pinned": info.get("kv_pinned") == "True", "kv_tokens": v.get("kv_tokens"), "kv_bytes": pm.get("kv_bytes"),
        "zygote": bool(info.get("zygote_clients")) and "zygote_ready_s" in pm,
        "pyc_delivered": "pyc.tar" in ev,
        "pycache_prefix": "PYTHONPYCACHEPREFIX=" in (info.get("pm_loo") or ""),
        "fetch_end_after_launch_s": round(fetch_end - m["engine_start"], 2) if fetch_end and "engine_start" in m else None,
        "warm_thread": "warm_s" in pm,
        **capture_check(r),
    }
    c["compile_cache_hit"] = c["compile_cache_hits"] > 0 and c["compile_cache_misses"] == 0
    c["overlapped"] = c["fetch_end_after_launch_s"] is not None and c["fetch_end_after_launch_s"] > 0
    on = {"snapshot": c["snapshot_restore"] and c["shards_delivered"] == 32, "bank": c["bank_delivered"],
          "bytecode": c["pyc_delivered"] and c["pycache_prefix"], "kvpin": c["kv_pinned"], "zygote": c["zygote"],
          "overlap": c["overlapped"], "warmup": c["warm_thread"]}
    removed = LOO_POD.get(r["arm"], r["arm"].removeprefix("loo-")) if r["arm"].startswith("loo-") else None
    expect = {k: k != removed for k in on}
    if removed == "kvpin":
        expect["bank"] = False  # the pin is part of the compile key: the bank goes too
    if removed == "snapshot":
        expect["overlap"] = None  # nothing large left to overlap: the delivery is model + pyc + bank
    bad = [k for k in on if expect[k] is not None and on[k] != expect[k]]
    if not c["compile_cache_hit"]:
        bad.append("compile_cache_hit")
    if removed == "snapshot" and not c["runai_weights"]:
        bad.append("runai_weights")
    if removed in ("bank", "kvpin") and not c["stock_cc_tar"]:
        bad.append("stock_cc_tar")
    c["mechanisms_on"] = on
    c["consistent"] = not bad
    c["contradictions"] = bad
    return c


def capture_check(r):
    """The KV window and CUDA graph capture of one boot, and the Triton JIT cache around it (pod_run.sh counts the
    .cubin files of $HOME/.triton/cache right before the engine launches and right after /health answers)."""
    info, v = r.get("info") or {}, r.get("values") or {}
    cubins = info.get("triton_cubins") or {}
    return {"triton_cubins_pre_launch": cubins.get("pre_launch"), "triton_cubins_ready": cubins.get("ready"),
            "triton_cubins_check_done": cubins.get("check_done"),
            "capture_s": v.get("capture_s"), "capture_piecewise_s": v.get("capture_piecewise_s"),
            "capture_full_s": v.get("capture_full_s"),
            "warmup_to_kv_s": round(v["warmup_to_kv_s"], 2) if v.get("warmup_to_kv_s") is not None else None,
            "cudagraph_mem_profiled": info.get("cudagraph_mem_profiled"),
            "kv_tokens": v.get("kv_tokens")}



def cmd_loo(args):
    d = Path(args.dir)
    recs = [json.loads(line) for line in (d / "runs.jsonl").read_text().splitlines()]
    mx = json.loads(Path(args.matrix).read_text())
    ref = mx["arms"]["pm-gcs"]
    by_arm, failed, controls = {}, [], []
    for r in recs:
        if r["arm"] == "runai-cc" and r.get("ok") and r.get("check_text") == args.expect_text:
            controls.append(r)  # in-session stock-runai-cc: the best stock arm on the same node, same session
        if r["arm"] not in LOO_REMOVED:
            continue  # warm-pm records of the pm-gcs pods, stock controls: not part of the ablation
        r["mechanism_check"] = mechanism_check(r) if r.get("ok") else None
        good = (r.get("ok") and r.get("check_text") == args.expect_text and r["mechanism_check"]["consistent"]
                and r["run_id"] not in args.exclude)
        if good:
            by_arm.setdefault(r["arm"], []).append(r)
        else:
            failed.append({"run_id": r["run_id"], "arm": r["arm"], "ok": r.get("ok"), "error": r.get("error"),
                           "ready_s": r.get("ready_s"), "check_text": r.get("check_text"),
                           "contradictions": (r.get("mechanism_check") or {}).get("contradictions"),
                           "excluded": r["run_id"] in args.exclude,
                           "cause": args.cause.get(r["run_id"], "")})
    full = by_arm.get("pm-gcs", [])
    full_med = statistics.median(r["ready_s"] for r in full) if full else None
    full_ph = {k: statistics.median(r["segments"][k] for r in full) for k in SEGMENT_KEYS} if full else None
    arms = {}
    for arm, (removed, fallback) in LOO_REMOVED.items():
        rs = sorted(by_arm.get(arm, []), key=lambda r: r["sample"])
        if not rs:
            arms[arm] = {"removed": removed, "fallback": fallback, "status": "dnf", "n": 0}
            continue
        med = statistics.median(r["ready_s"] for r in rs)
        ph = {k: statistics.median(r["segments"][k] for r in rs) for k in SEGMENT_KEYS}
        arms[arm] = {
            "removed": removed, "fallback": fallback, "status": "ok" if len(rs) >= 3 else "partial", "n": len(rs),
            "median": round(med, 2), "runs": [round(r["ready_s"], 2) for r in rs],
            "run_ids": [r["run_id"] for r in rs], "health_s": [round(r["health_s"], 2) for r in rs],
            "phases": {k: round(x, 2) for k, x in ph.items()},
            "delta_vs_full": round(med - full_med, 2) if full_med is not None else None,
            "phase_deltas_vs_full": {k: round(ph[k] - full_ph[k], 2) for k in SEGMENT_KEYS} if full_ph else None,
            "greedy_check": {"passed": len(rs), "text": args.expect_text},
            "mechanism_check": {r["run_id"]: r["mechanism_check"] for r in rs},
            "wipe_receipts": {r["run_id"]: wipe_summary(r.get("wipe")) for r in rs},
            "held_node": sorted({r.get("held_node") for r in rs if r.get("held_node")}),
            "pm": {k: statistics.median(r["pm"][k] for r in rs if k in r["pm"])
                   for k in sorted({k for r in rs for k in r["pm"] if k != "join_at"})},
            "gke": None, "note": "reused-node; " + GKE_NOTE,
        }
    control = None
    if controls:
        cph = {k: statistics.median(r["segments"][k] for r in controls) for k in SEGMENT_KEYS}
        control = {"arm": "stock-runai-cc (in-session control, not an ablation arm)", "n": len(controls),
                   "median": round(statistics.median(r["ready_s"] for r in controls), 2),
                   "runs": [round(r["ready_s"], 2) for r in controls], "run_ids": [r["run_id"] for r in controls],
                   "phases": {k: round(x, 2) for k, x in cph.items()}, "cc_uri": controls[0].get("cc_uri"),
                   "matrix_median": mx["arms"]["stock-runai-cc"]["median"],
                   "note": "phases as derive() gives them: Run:ai streaming is inside load_model (the matrix splits "
                           "it into lazy + load_model)"}
    meta = json.loads(Path(args.meta).read_text()) if args.meta else {}
    out = {"cell": "gpu-torch", "hardware": mx["hardware"], "tp": 1, "code_commit": args.commit,
           "matrix_ref": {"summary": args.matrix, "code_commit": mx["code_commit"], "pm-gcs_median": ref["median"],
                          "pm-gcs_runs": ref["runs"], "stock-runai-cc_median": mx["arms"]["stock-runai-cc"]["median"]},
           "in_session_full_vs_matrix": {"in_session": full_med, "matrix": ref["median"],
                                          "diff_frac": round(full_med / ref["median"] - 1, 4) if full_med else None},
           "t0": mx["t0"], "ready_def": "/health OK and the greedy /v1/chat/completions reply (temperature 0, 32 "
                                       "tokens) received; matrix definition",
           "engine": mx["engine"], "image": IMAGE, "baseline_arm": "pm-gcs", "state": args.state,
           "arms": arms, "stock_control": control, "failed_attempts": failed, **meta}
    (d / "loo_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for arm, a in arms.items():
        print(f"{arm:13s} {a['status']} n={a['n']} median={a.get('median')} runs={a.get('runs')} "
              f"delta={a.get('delta_vs_full')} "
              + " ".join(f"{k}={x:+.1f}" for k, x in (a.get("phase_deltas_vs_full") or {}).items() if abs(x) >= 0.3))
    print("failed/excluded:", [(f["run_id"], f["contradictions"], f["error"]) for f in failed])


TRITONFIX = {  # campaign arm -> (summary arm, matrix arm it re-measures, segments absent in that pipeline)
    "pm-gcs": ("pm-gcs", "pm-gcs", ()), "runai-cc": ("stock-runai-cc", "stock-runai-cc", ()),
    "ckpt-cc": ("stock-ckpt-cc", "stock-ckpt-cc", ("lazy",)), "loo-kvpin-only": ("loo-kvpin-only", None, ()),
}


def tritonfix_check(r, delivered):
    """Per-run mechanism check of the tritonfix campaign: compile-cache key and hit, Triton JIT cache before launch
    and at ready, JIT during startup, capture rounds, KV window and pin, restore join, greedy text."""
    info, pm, v = r.get("info") or {}, r.get("pm") or {}, r.get("values") or {}
    c = capture_check(r)
    jit = None if c["triton_cubins_ready"] is None or delivered is None else c["triton_cubins_ready"] - delivered
    return {"cc_uri": r.get("cc_uri"),
            "compile_cache_key": (info.get("compile_cache_dir") or "").split("torch_compile_cache/")[-1].split("/")[0],
            "compile_cache_hits": int(v.get("compile_cache_hits", 0)),
            "compile_cache_misses": int(v.get("compile_cache_misses", 0)), "compile_s": v.get("compile_s"),
            "triton_cubins_delivered": delivered, **c, "triton_jit_kernels_startup": jit,
            # JIT during startup: new kernels by ready, or the FULL decode round's stalls (3 s warm, 6-7 s with JIT)
            "triton_jit_during_capture": bool(jit) or (c["capture_full_s"] or 0) >= 5,
            "kv_pinned": info.get("kv_pinned") == "True", "kv_profiled": bool(info.get("cudagraph_mem_profiled")),
            "restore_join_at": pm.get("join_at"), "restore_join_waited_s": pm.get("join_wait_s"),
            "greedy_text": r.get("check_text")}


def cmd_tritonfix(args):
    """tritonfix-<rev>/summary.json (warm stock caches, n=3 per arm, interleaved on one held node), plus the
    matrix and LOO summaries with the re-measured arms."""
    d = Path(args.dir)
    mx, lo = json.loads(Path(args.matrix).read_text()), json.loads(Path(args.loo).read_text())
    ref = mx["stock_lazy_reference_load_model_s"]
    meta = json.loads(Path(args.meta).read_text()) if args.meta else {}
    by_arm, failed = {}, []
    for line in (d / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["arm"] not in TRITONFIX:
            continue  # warm-pm records of the pm-gcs pods
        name, _, absent = TRITONFIX[r["arm"]]
        if r.get("cc_uri") in args.old_tars:
            name += "@cold-export-cache"
        delivered = args.cubins.get(r.get("cc_uri") or args.bank)
        r["check"] = tritonfix_check(r, delivered) if r.get("ok") else None
        if r["arm"] == "loo-kvpin-only" and r.get("ok"):
            r["check"]["mechanism_check"] = mechanism_check(r)
        good = (r.get("ok") and r.get("check_text") == args.expect_text and r["check"]["compile_cache_hits"] > 0
                and not r["check"]["compile_cache_misses"] and r["run_id"] not in args.exclude
                and (r["arm"] != "loo-kvpin-only" or r["check"]["mechanism_check"]["consistent"]))
        if not good:
            failed.append({"run_id": r["run_id"], "arm": name, "ok": r.get("ok"), "error": r.get("error"),
                           "ready_s": r.get("ready_s"), "check_text": r.get("check_text"),
                           "cause": args.cause.get(r["run_id"], "")})
            continue
        seg = r["segments"]
        if r["arm"] == "runai-cc":  # the matrix's split: Run:ai streaming beyond the tmpfs load is lazy
            seg["lazy"] = max(0.0, seg["load_model"] - ref)
            seg["load_model"] -= seg["lazy"]
        for k in absent:
            seg[k] = None
        by_arm.setdefault(name, []).append(r)

    def med(rs, key):
        xs = [key(r) for r in rs]
        return None if any(x is None for x in xs) else round(statistics.median(xs), 2)

    arms = {}
    for name, rs in by_arm.items():
        rs.sort(key=lambda r: (r["sample"], r["run_id"]))
        base = name.split("@")[0]
        mrow = mx["arms"].get(base) if base != "loo-kvpin-only" else None
        ph = {k: med(rs, lambda r: r["segments"][k]) for k in SEGMENT_KEYS}
        row = {"n": len(rs), "status": "ok" if len(rs) >= 3 else "partial", "median": med(rs, lambda r: r["ready_s"]),
               "runs": [round(r["ready_s"], 2) for r in rs], "run_ids": [r["run_id"] for r in rs],
               "health_s": [round(r["health_s"], 2) for r in rs], "phases": ph, "cc_uri": rs[0].get("cc_uri"),
               "greedy_check": {"passed": len(rs), "text": args.expect_text},
               "capture": {k: med(rs, lambda r: r["check"][k]) for k in
                           ("capture_s", "capture_piecewise_s", "capture_full_s", "warmup_to_kv_s", "kv_tokens")},
               "mechanism_check": {r["run_id"]: r["check"] for r in rs},
               "wipe_receipts": {r["run_id"]: wipe_summary(r.get("wipe")) for r in rs},
               "held_node": sorted({r.get("held_node") for r in rs if r.get("held_node")}),
               "pm": {k: med(rs, lambda r: r["pm"].get(k)) for k in sorted({k for r in rs for k in r["pm"]
                                                                            if k != "join_at"})} if rs[0]["pm"] else None}
        if mrow:
            row["matrix"] = {"median": round(mrow["median"], 2), "runs": mrow["runs"],
                             "phases": {k: None if v is None else round(v, 2) for k, v in mrow["phases"].items()}}
            row["delta_vs_matrix"] = round(row["median"] - mrow["median"], 2)
            row["phase_deltas_vs_matrix"] = {k: None if ph[k] is None or mrow["phases"][k] is None
                                             else round(ph[k] - mrow["phases"][k], 2) for k in SEGMENT_KEYS}
        arms[name] = row
    full = arms.get("pm-gcs")
    if "loo-kvpin-only" in arms and full:
        a, old = arms["loo-kvpin-only"], lo["arms"]["loo-kvpin"]
        a["delta_vs_session_pm_gcs"] = round(a["median"] - full["median"], 2)
        a["phase_deltas_vs_session_pm_gcs"] = {k: round(a["phases"][k] - full["phases"][k], 2) for k in SEGMENT_KEYS}
        a["loo_279604f6_loo_kvpin"] = {"median": old["median"], "runs": old["runs"], "delta_vs_full": old["delta_vs_full"]}
    out = {"cell": "gpu-torch", "hardware": mx["hardware"], "tp": 1, "code_commit": args.commit, "image": IMAGE,
           "engine": mx["engine"], "state": args.state, "t0": mx["t0"],
           "ready_def": "/health OK and the greedy /v1/chat/completions reply received; matrix definition",
           "matrix_ref": args.matrix, "loo_ref": args.loo, "triton_cubins_listed": args.cubins, "arms": arms,
           "failed_attempts": failed, **meta}
    if full and "loo-kvpin-only" in arms:
        out["kv_pin_effect"] = {"delta_s": arms["loo-kvpin-only"]["delta_vs_session_pm_gcs"],
                                "phase_deltas": arms["loo-kvpin-only"]["phase_deltas_vs_session_pm_gcs"],
                                "def": "loo-kvpin-only - pm-gcs, same session and node"}
    stock = {k: v for k, v in arms.items() if k in ("stock-runai-cc", "stock-ckpt-cc")}
    if stock and full:
        best = min(stock, key=lambda k: stock[k]["median"])
        out["best_stock"] = {"arm": best, "median": stock[best]["median"], "pm_gcs_session_median": full["median"],
                             "speedup_session": round(stock[best]["median"] / full["median"], 3),
                             "speedup_vs_matrix_pm_gcs": round(stock[best]["median"] / mx["arms"]["pm-gcs"]["median"], 3)}
    (d / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, a in arms.items():
        print(f"{name:32s} n={a['n']} median={a['median']} runs={a['runs']} matrix={a.get('matrix', {}).get('median')} "
              f"delta={a.get('delta_vs_matrix')} capture={a['capture']}")
    print("failed:", [(f["run_id"], f["error"]) for f in failed])
    if args.matrix_out:  # the matrix with the warm-cache stock -cc arms; the old ones under extras
        new = json.loads(json.dumps(mx))
        new["code_commit"] = f"{mx['code_commit']}; stock-ckpt-cc, stock-runai-cc: {args.commit}"
        new["sources"]["tritonfix"] = str(d)
        new.setdefault("extras", {})
        for name in ("stock-ckpt-cc", "stock-runai-cc"):
            if name not in arms:
                continue
            a, oldrow = arms[name], mx["arms"][name]
            new["extras"][f"{name}@cold-export-cache"] = {**oldrow, "note": oldrow["note"] + (
                "; REPLACED: its compile cache was exported right after a cold-compile boot, so the tarball's "
                ".triton/cache held only cuda_utils.so and every boot re-JITted vLLM's 19 Triton kernel variants "
                "during KV profiling and graph capture (see " + str(d) + ")")}
            new["arms"][name] = {
                "status": a["status"], "n": a["n"], "runs": a["runs"], "median": a["median"],
                "phases": a["phases"], "gke": None,
                "note": f"{GKE_NOTE}; {meta.get('matrix_notes', {}).get(name, '')}".rstrip("; "),
                "run_ids": a["run_ids"], "health_s": a["health_s"], "greedy_check": a["greedy_check"],
                "wipe_receipts": a["wipe_receipts"], "held_node": a["held_node"]}
        Path(args.matrix_out).mkdir(parents=True, exist_ok=True)
        (Path(args.matrix_out) / "matrix_summary.json").write_text(json.dumps(new, indent=1) + "\n")
    if args.loo_out and "loo-kvpin-only" in arms:
        new = json.loads(json.dumps(lo))
        a = arms["loo-kvpin-only"]
        new["code_commit"] = f"{lo['code_commit']}; loo-kvpin-only: {args.commit}"
        new["arms"]["loo-kvpin"]["note"] += meta.get("loo_kvpin_note", "")
        removed, fallback = LOO_REMOVED["loo-kvpin-only"]
        new["arms"]["loo-kvpin-only"] = {
            "removed": removed, "fallback": fallback.replace("<rev>", args.commit), "status": a["status"],
            "n": a["n"], "median": a["median"], "runs": a["runs"], "run_ids": a["run_ids"], "health_s": a["health_s"],
            "phases": a["phases"], "delta_vs_full": a.get("delta_vs_session_pm_gcs"),
            "phase_deltas_vs_full": a.get("phase_deltas_vs_session_pm_gcs"), "greedy_check": a["greedy_check"],
            "mechanism_check": {k: v["mechanism_check"] for k, v in a["mechanism_check"].items()},
            "wipe_receipts": a["wipe_receipts"], "held_node": a["held_node"], "pm": a["pm"], "gke": None,
            "note": "reused-node; " + GKE_NOTE + meta.get("loo_kvpin_only_note", ""),
            "baseline_session": {"arm": "pm-gcs", "dir": str(d), "code_commit": args.commit, "n": full["n"],
                                 "median": full["median"], "runs": full["runs"], "run_ids": full["run_ids"],
                                 "phases": full["phases"],
                                 "note": "delta_vs_full of this arm is against this in-session pm-gcs, not the "
                                         "279604f6 in-session full (in_session_full_vs_matrix)"}}
        new["failed_attempts"] = lo["failed_attempts"] + [f for f in failed if f["arm"] == "loo-kvpin-only"]
        Path(args.loo_out).mkdir(parents=True, exist_ok=True)
        (Path(args.loo_out) / "loo_summary.json").write_text(json.dumps(new, indent=1) + "\n")



# ------------------------------------------------------------------------------------------- ServerlessLLM

SLLM_STORE_EVENTS = (("disk_load_start", re.compile(r"Loading model (\S+) to host with (\d+) threads")),
                     ("disk_load_end", re.compile(r"Finished loading model (\S+) from disk")),
                     ("no_odirect", re.compile(r"O_DIRECT not supported on (\S+)")),
                     ("pool_created", re.compile(r"Memory pool created with (\d+)GB")))


def store_events(text):
    """The sllm-store daemon's own lines (the "SLLM| " stream): reads into its pinned pool, O_DIRECT fallbacks."""
    out = []
    for line in text.splitlines():
        stamp, _, body = line.partition(" ")
        if not body.startswith("SLLM| "):
            continue
        for kind, pattern in SLLM_STORE_EVENTS:
            m = pattern.search(body)
            if m:
                try:
                    when = ts(stamp)
                except ValueError:
                    when = None
                out.append({"t": when, "event": kind, "arg": m.group(1)})
    return out


def sllm_seed_report(text):
    """sllm-seed: build provenance, conversion, the two verification boots (S1 cold, S2 restart from the store's
    pool) and the bare load_dict bench."""
    rep = {"uploads": [], "bench": [], "apt": [], "deps_from_image": [], "save": None}
    for line in text.splitlines():
        body = line.partition(" ")[2].removeprefix("S0| ")
        for tag, key in (("SLLM_SRC_COMMIT ", "commit"), ("SLLM_WHEELS ", "wheels"), ("SLLM_SAVE_TO ", "save_to"),
                         ("SLLM_IMPORT ", "import")):
            if body.startswith(tag):
                rep[key] = body[len(tag):].strip()
        if body.startswith("SLLM_SAVE "):
            rep["save"] = json.loads(body[len("SLLM_SAVE "):])
        if body.startswith("UPLOAD "):
            rep["uploads"].append(json.loads(body[len("UPLOAD "):]))
        if body.startswith("SLLM_BENCH "):
            rep["bench"].append(json.loads(body[len("SLLM_BENCH "):]))
        if body.startswith("apt: "):
            rep["apt"].append(body[5:].strip())
        if body.startswith("dep: "):
            rep["deps_from_image"].append(body[5:].strip())
    rep["loader_verbatim"] = "SLLM_LOADER verbatim" in text
    marks = parse_log(text)["marks"]
    for s, k in (("S1", "seed1"), ("S2", "seed2")):
        p = parse_log(text, prefix=f"{s}| ")
        e, v, info = p["events"], p["values"], p["info"]
        rep[s] = {"ready_s": marks[f"{k}_ready"] - marks[f"{k}_start"] if {f"{k}_ready", f"{k}_start"} <= set(marks)
                  else None,
                  "load_model_s": e["load_end"] - e["load_start"] if {"load_end", "load_start"} <= set(e) else None,
                  "check_text": p["check_text"], "compile_cache_dir": info.get("compile_cache_dir"),
                  "sllm_plugin": info.get("sllm_plugin"),
                  **{x: v.get(x) for x in ("sllm_load_dict_s", "sllm_load_GBps", "sllm_load_bytes", "sllm_load_model_s",
                                           "compile_s", "kv_tokens", "capture_s", "compile_cache_hits",
                                           "compile_cache_misses")}}
    rep["store_events"] = store_events(text)
    rep["marks"] = {k: v for k, v in marks.items() if k.startswith(("sllm_", "seed", "setup"))}
    return rep


# campaign arm -> (summary arm, segment kind, segments absent in that pipeline); the matrix's arm has the same name
SLLM_SESSION = {
    "sllm": ("sllm", "stock", ("lazy",)), "sllm-cc": ("sllm-cc", "stock", ("lazy",)),
    "warm-sllm": ("warm-sllm", "warm-stock", ("download", "lazy")),
    "ckpt-cc": ("stock-ckpt-cc", "stock", ("lazy",)), "runai-cc": ("stock-runai-cc", "stock", ()),
    "pm-gcs": ("pm-gcs", "pm", ()), "warm-pm": ("warm-pm", "warm-pm", ("download", "spawn")),
    "warm-stock": ("warm-stock", "warm-stock", ("download", "lazy")),
}
SLLM_PRIMES = {"warm-sllm-prime": "sllm-cc", "warm-stock-prime": "stock-ckpt-cc"}  # informational only


def sllm_check(r, text, delivered):
    """Per-run check of the sllm campaign: everything tritonfix_check reports, plus what the ServerlessLLM loader and
    the store did (load_dict time and rate, CPU-side model build, the store's reads into its pool)."""
    c = tritonfix_check(r, delivered)
    v, info, m = r.get("values") or {}, r.get("info") or {}, r.get("marks") or {}
    ev = store_events(text) if text else []
    relaunch = m.get("warm_launch") if r["arm"] in ("warm-sllm", "warm-stock") else None
    loads = [e for e in ev if e["event"] == "disk_load_start"]
    both = v.get("sllm_load_model_s") is not None and v.get("sllm_load_dict_s") is not None
    c.update({
        "loader": "serverless_llm" if info.get("sllm_plugin") and v.get("sllm_load_dict_s") is not None else "stock",
        "load_dict_s": v.get("sllm_load_dict_s"), "load_dict_GBps": v.get("sllm_load_GBps"),
        "load_dict_bytes": v.get("sllm_load_bytes"), "sllm_load_model_s": v.get("sllm_load_model_s"),
        "cpu_side_s": round(v["sllm_load_model_s"] - v["sllm_load_dict_s"], 3) if both else None,
        "store_start_s": round(m["sllm_store_ready"] - m["sllm_store_start"], 2)
        if {"sllm_store_ready", "sllm_store_start"} <= set(m) else None,
        "store_ready_before_t0_s": round(m["download_start"] - m["sllm_store_ready"], 2)
        if {"sllm_store_ready", "download_start"} <= set(m) else None,
        "store": info.get("sllm_store"),
        "store_disk_loads": len(loads),
        "store_disk_loads_after_relaunch": len([e for e in loads if relaunch and e["t"] and e["t"] > relaunch])
        if relaunch else None,
        "o_direct_fallbacks": len([e for e in ev if e["event"] == "no_odirect"]),
    })
    return c


SHARED_LIBS = "@store-shares-engine-libs"  # sllm / sllm-cc runs whose store daemon used the engine's own library files


def cmd_sllm(args):
    """summary.json (in the first dir) for the ServerlessLLM campaign dirs (sllm arms, one held node,
    interleaved), and --matrix-out: the matrix with the ServerlessLLM rows added. sllm / sllm-cc runs whose store
    ran from the engine's own site-packages (no SLLM_STORE_ISOLATE) are kept as <arm>@store-shares-engine-libs."""
    dirs = [Path(x) for x in args.dirs]
    d = dirs[0]
    mx = json.loads(Path(args.matrix).read_text())
    ref = mx["stock_lazy_reference_load_model_s"]
    meta = json.loads(Path(args.meta).read_text()) if args.meta else {}
    logs = {}
    by_arm, primes, failed = {}, {}, []
    for r, rd in ((json.loads(line), rd) for rd in dirs for line in (rd / "runs.jsonl").read_text().splitlines()):
        parent = r.get("parent") or r["run_id"]
        if parent not in logs:
            p = rd / f"{parent}.log"
            logs[parent] = p.read_text() if p.exists() else ""
        text = logs[parent]
        r["dir"] = str(rd)
        if r["arm"] in SLLM_PRIMES:
            if r.get("ok"):
                primes.setdefault(r["arm"], []).append(r)
            continue
        if r["arm"] not in SLLM_SESSION:
            continue
        name, kind, absent = SLLM_SESSION[r["arm"]]
        if r["arm"] in ("sllm", "sllm-cc") and "isolate=1" not in ((r.get("info") or {}).get("sllm_store") or ""):
            name += SHARED_LIBS
        delivered = args.cubins.get(r.get("cc_uri") or args.bank)
        r["check"] = sllm_check(r, text, delivered) if r.get("ok") else None
        c = r["check"]
        reasons = []
        if not r.get("ok"):
            reasons.append("not ok")
        elif r.get("check_text") != args.expect_text:
            reasons.append("greedy text")
        else:
            if r["arm"] != "sllm" and (not c["compile_cache_hits"] or c["compile_cache_misses"]):
                reasons.append("compile cache")
            if r["arm"] != "sllm" and c["triton_jit_during_capture"]:
                reasons.append("triton JIT")
            if r["arm"] in SLLM_ARMS and c["loader"] != "serverless_llm":
                reasons.append("loader")
            if r["arm"] in ("sllm", "sllm-cc") and c["store_disk_loads"] != 1:
                reasons.append("store disk loads")
            if r["arm"] == "warm-sllm" and c["store_disk_loads_after_relaunch"] != 0:
                reasons.append("restart read the disk")
        if r["run_id"] in args.exclude:
            reasons.append("excluded")
        if reasons:
            failed.append({"run_id": r["run_id"], "arm": name, "ok": r.get("ok"), "error": r.get("error"),
                           "ready_s": r.get("ready_s"), "check_text": r.get("check_text"), "reasons": reasons,
                           "cause": args.cause.get(r["run_id"], "")})
            continue
        seg = r["segments"]
        if r["arm"] == "runai-cc":  # the matrix's split: Run:ai streaming beyond the tmpfs load is lazy
            seg["lazy"] = max(0.0, seg["load_model"] - ref)
            seg["load_model"] -= seg["lazy"]
        for k in absent:
            seg[k] = None
        r["kind"] = kind
        by_arm.setdefault(name, []).append(r)

    def med(rs, key):
        xs = [key(r) for r in rs]
        return None if not xs or any(x is None for x in xs) else round(statistics.median(xs), 2)

    arms = {}
    for name, rs in by_arm.items():
        rs.sort(key=lambda r: (r["sample"], r["run_id"]))
        mrow = mx["arms"].get(name)
        ph = {k: med(rs, lambda r: r["segments"][k]) for k in SEGMENT_KEYS}
        row = {"n": len(rs), "status": "ok" if len(rs) >= 3 else "partial", "median": med(rs, lambda r: r["ready_s"]),
               "runs": [round(r["ready_s"], 2) for r in rs], "run_ids": [r["run_id"] for r in rs],
               "health_s": [round(r["health_s"], 2) for r in rs], "phases": ph, "cc_uri": rs[0].get("cc_uri"),
               "greedy_check": {"passed": len(rs), "text": args.expect_text},
               "capture": {k: med(rs, lambda r: r["check"][k]) for k in
                           ("capture_s", "capture_piecewise_s", "capture_full_s", "warmup_to_kv_s", "kv_tokens")},
               "mechanism_check": {r["run_id"]: r["check"] for r in rs},
               "wipe_receipts": {r["run_id"]: wipe_summary(r.get("wipe")) for r in rs},
               "held_node": sorted({r.get("held_node") for r in rs if r.get("held_node")})}
        if rs[0]["arm"] in SLLM_ARMS:
            row["sllm"] = {k: med(rs, lambda r: r["check"][k]) for k in
                           ("load_dict_s", "load_dict_GBps", "sllm_load_model_s", "cpu_side_s", "store_start_s",
                            "store_ready_before_t0_s")}
            row["dirs"] = sorted({r["dir"] for r in rs})
            if any(r.get("wipe_t0") for r in rs):  # the second drop (isolated store), right before T0
                row["t0_receipts"] = {r["run_id"]: {
                    "image_resident_bytes_before_after": [r["wipe_t0"]["before"]["image"]["resident_bytes"],
                                                          r["wipe_t0"]["after"]["image"]["resident_bytes"]],
                    "image_bytes": r["wipe_t0"]["after"]["image"]["bytes"],
                    "page_cache_GB_before_after": [r["wipe_t0"]["before"]["meminfo"]["Cached_GB"],
                                                   r["wipe_t0"]["after"]["meminfo"]["Cached_GB"]]}
                    for r in rs if r.get("wipe_t0")}
        if rs[0]["kind"] == "warm-stock":
            row["prime"] = {"arm": f"{rs[0]['arm']}-prime", "runs": [round(p["ready_s"], 2) for p in
                                                                       primes.get(f"{rs[0]['arm']}-prime", [])]}
        if mrow and mrow.get("median") is not None:
            row["matrix"] = {"median": round(mrow["median"], 2), "runs": mrow["runs"],
                             "phases": {k: None if v is None else round(v, 2) for k, v in mrow["phases"].items()}}
            row["delta_vs_matrix"] = round(row["median"] - mrow["median"], 2)
        arms[name] = row

    def ratio(a, b):
        return round(arms[a]["median"] / arms[b]["median"], 3) if a in arms and b in arms else None

    def delta(a, b):
        return round(arms[a]["median"] - arms[b]["median"], 2) if a in arms and b in arms else None

    def vs_matrix(a, b):
        m = (mx["arms"].get(b) or {}).get("median")
        return None if a not in arms or m is None else {"matrix_median": round(m, 2), "delta": round(arms[a]["median"] - m, 2),
                                                        "ratio": round(arms[a]["median"] / m, 3)}

    shared = "sllm-cc" + SHARED_LIBS
    session = {
        "sllm-cc_minus_stock-ckpt-cc": delta("sllm-cc", "stock-ckpt-cc"),
        "sllm-cc_minus_stock-runai-cc": delta("sllm-cc", "stock-runai-cc"),
        "sllm-cc_over_pm-gcs": ratio("sllm-cc", "pm-gcs"),
        "warm-sllm_minus_warm-stock": delta("warm-sllm", "warm-stock"),
        "warm-sllm_over_warm-pm": ratio("warm-sllm", "warm-pm"),
        "shared_libs_effect": {a: delta(a + SHARED_LIBS, a) for a in ("sllm", "sllm-cc")},
        "phase_deltas_shared_libs_sllm-cc": {
            k: None if arms[shared]["phases"][k] is None else round(arms[shared]["phases"][k]
                                                                     - arms["sllm-cc"]["phases"][k], 2)
            for k in SEGMENT_KEYS} if {"sllm-cc", shared} <= set(arms) else None}
    matrix_cmp = {f"{a}_vs_matrix_{b}": vs_matrix(a, b) for a, b in (
        ("sllm-cc", "stock-ckpt-cc"), ("sllm-cc", "stock-runai-cc"), ("sllm-cc", "pm-gcs"), ("sllm", "stock-ckpt"),
        ("sllm", "stock-runai"), ("warm-sllm", "warm-stock"), ("warm-sllm", "warm-pm"))}
    matrix_cmp["phase_deltas_sllm-cc_vs_matrix_stock-ckpt-cc"] = {
        k: None if "sllm-cc" not in arms or arms["sllm-cc"]["phases"][k] is None
        or mx["arms"]["stock-ckpt-cc"]["phases"][k] is None
        else round(arms["sllm-cc"]["phases"][k] - mx["arms"]["stock-ckpt-cc"]["phases"][k], 2) for k in SEGMENT_KEYS}
    session["vs_matrix"] = matrix_cmp
    out = {"cell": "gpu-torch", "hardware": mx["hardware"], "tp": 1, "code_commit": args.commit, "image": IMAGE,
           "engine": mx["engine"], "t0": mx["t0"],
           "ready_def": "/health OK and the greedy /v1/chat/completions reply received; matrix definition",
           "matrix_ref": args.matrix, "sllm_src": args.sllm, "triton_cubins_listed": args.cubins,
           "arms": arms, "session": session, "failed_attempts": failed, **meta}
    if args.seed:
        seeds = [json.loads(x) for x in (Path(args.seed) / "runs.jsonl").read_text().splitlines()]
        seed = next((s for s in reversed(seeds) if s["arm"] == "sllm-seed" and s.get("ok")), None)
        out["seed"] = {"run_id": seed["run_id"], "dir": args.seed, **seed["sllm_seed"]} if seed else None
    (d / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, a in arms.items():
        print(f"{name:16s} n={a['n']} median={a['median']} runs={a['runs']} matrix={a.get('matrix', {}).get('median')} "
              f"sllm={a.get('sllm')} capture={a['capture']}")
    print("session:", json.dumps(session))
    print("failed:", [(f["run_id"], f["reasons"], f["error"]) for f in failed])
    if args.matrix_out:  # the matrix + the ServerlessLLM rows (this session's runs, same row schema)
        new = json.loads(json.dumps(mx))
        added = [a for a in ("sllm", "sllm-cc", "warm-sllm") if a in arms]
        if added:
            new["code_commit"] = f"{mx['code_commit']}; {', '.join(added)}: {args.commit}"
        new["sources"]["sllm"] = [str(x) for x in dirs]
        new.setdefault("extras", {})
        for name in added + [a for a in arms if a.endswith(SHARED_LIBS)]:
            a = arms[name]
            note = meta.get("matrix_notes", {}).get(name, "")
            (new["arms"] if name in added else new["extras"])[name] = {
                "status": a["status"], "n": a["n"], "runs": a["runs"], "median": a["median"], "phases": a["phases"],
                "gke": None, "note": f"{GKE_NOTE}; {note}".rstrip("; "),
                "run_ids": a["run_ids"], "health_s": a["health_s"], "greedy_check": a["greedy_check"],
                "wipe_receipts": a["wipe_receipts"], "held_node": a["held_node"], "sllm": a["sllm"],
                **({"t0_receipts": a["t0_receipts"]} if "t0_receipts" in a else {})}
        Path(args.matrix_out).mkdir(parents=True, exist_ok=True)
        (Path(args.matrix_out) / "matrix_summary.json").write_text(json.dumps(new, indent=1) + "\n")


def cmd_report(args):
    by_arm = summarize(args.dirs)
    for arm in ARMS + ("warm-stock-prime", "warm-pm", "warm-sllm-prime"):
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
    run.add_argument("--pool", default="gpu-pool-spot,gpu-pool-flex", help="node pool(s), comma-separated")
    run.add_argument("--zone", help="fresh nodes only in this zone (e.g. asia-southeast1-c; zone b was STOCKOUT)")
    run.add_argument("--cc-local", help="gs:// tarball of a peer's caches for ckpt-cc")
    run.add_argument("--cc-runai", help="gs:// tarball of a peer's caches for runai-cc")
    run.add_argument("--cc-bank", help="loo-bank: gs:// tarball of the stack's compile cache for the pinned key")
    run.add_argument("--cc-kvpin", help="loo-kvpin: gs:// tarball of the stock compile cache (unpinned key)")
    run.add_argument("--export", help="export the first job's caches to this gs:// tarball")
    run.add_argument("--keep", action="store_true", help="leave pod and node up")
    run.add_argument("--state", help="pm-gcs: gs:// prefix of the TPUFlex state (pm_state.py upload)")
    run.add_argument("--state-dst", help="pm-seed: gs:// prefix to publish the TPUFlex state to")
    run.add_argument("--peer", help="pm-peer: the live peer pod (pm-seed) serving the state on :8080")
    run.add_argument("--sllm", help="sllm arms: gs:// prefix of the ServerlessLLM-format checkpoint (sllm-seed)")
    run.add_argument("--sllm-wheels", help="sllm arms: gs:// prefix of the sllm-store wheelhouse (sllm-seed)")
    run.add_argument("--cc-sllm", help="sllm-cc / warm-sllm: gs:// tarball of the compile caches (model=/model)")
    run.add_argument("--sllm-dst", help="sllm-seed: gs:// prefix to publish the wheelhouse and checkpoint under")
    run.add_argument("--env", action="append", default=[], help="K=V passed to the pod (PM_* / SLLM_* tunables)")
    run.add_argument("--sample-base", type=int, default=0, help="first sample index (run ids)")
    run.add_argument("--node", help="reuse this held node (wipe + receipt in the pod; no retire, no GKE stages)")
    run.add_argument("--configmap", help="ConfigMap for the pod scripts (default gpu-autoscale-scripts); a "
                                         "campaign's own name keeps concurrent drivers from swapping each other's scripts")
    run.add_argument("--out", required=True)
    adopt = sub.add_parser("adopt", help="finish runs whose driver died: logs, export, k8s record, retire node")
    adopt.add_argument("pods", nargs="+", help="pod names (as-<run_id>); POD=gs://... also exports its caches")
    adopt.add_argument("--out", required=True)
    adopt.add_argument("--node", help="the pods run on this held node: keep it (no retire), as `run --node` does")
    report = sub.add_parser("report")
    report.add_argument("dirs", nargs="+")
    mx = sub.add_parser("matrix", help="matrix_summary.json: stock arms from --stock, new arms from --new")
    mx.add_argument("--stock", required=True, help="campaign dir of the stock arms (with autoscale_gpu_summary.json)")
    mx.add_argument("--new", nargs="+", required=True, help="campaign dirs of the pm-* / warm-* runs")
    mx.add_argument("--commit", required=True)
    mx.add_argument("--meta", help="JSON file: versions, t0, engine_extra, notes{arm: str}")
    mx.add_argument("--expect-text", default="The primary colors are red, yellow, and blue.")
    mx.add_argument("--exclude", nargs="*", default=[], help="run ids (pods) of --new dirs left out, see README")
    mx.add_argument("--out", required=True)
    lo = sub.add_parser("loo", help="loo_summary.json for a leave-one-out results dir")
    lo.add_argument("dir")
    lo.add_argument("--matrix", required=True, help="matrix_summary.json of the cell")
    lo.add_argument("--commit", required=True)
    lo.add_argument("--state", required=True)
    lo.add_argument("--meta", help="JSON merged into the summary (notes)")
    lo.add_argument("--expect-text", default="The primary colors are red, yellow, and blue.")
    lo.add_argument("--exclude", nargs="*", default=[], help="run ids left out (listed with cause)")
    lo.add_argument("--cause", type=json.loads, default={}, help='JSON {run_id: cause} for failed/excluded runs')
    tf = sub.add_parser("tritonfix", help="summary.json of a tritonfix dir (+ matrix / LOO summaries with its arms)")
    tf.add_argument("dir")
    tf.add_argument("--matrix", required=True)
    tf.add_argument("--loo", required=True)
    tf.add_argument("--commit", required=True)
    tf.add_argument("--state", required=True)
    tf.add_argument("--bank", required=True, help="the state's bank.tar uri (the PM boots' Triton kernels)")
    tf.add_argument("--cubins", type=json.loads, required=True, help="JSON {tar uri: .cubin files it lists}")
    tf.add_argument("--old-tars", nargs="*", default=[], help="the matrix's cold-export caches (control arms)")
    tf.add_argument("--meta", help="JSON merged into the summary; matrix_notes, loo_kvpin_note, loo_kvpin_only_note")
    tf.add_argument("--matrix-out")
    tf.add_argument("--loo-out")
    tf.add_argument("--expect-text", default="The primary colors are red, yellow, and blue.")
    tf.add_argument("--exclude", nargs="*", default=[])
    tf.add_argument("--cause", type=json.loads, default={})
    sl = sub.add_parser("sllm", help="summary.json of ServerlessLLM campaign dirs (+ the matrix with its rows)")
    sl.add_argument("dirs", nargs="+", help="campaign dirs; summary.json goes to the first")
    sl.add_argument("--matrix", required=True, help="matrix_summary.json the rows are added to")
    sl.add_argument("--commit", required=True)
    sl.add_argument("--sllm", required=True, help="gs:// prefix of the ServerlessLLM checkpoint the runs used")
    sl.add_argument("--seed", help="the sllm-seed run dir (provenance, conversion, bench)")
    sl.add_argument("--bank", required=True, help="the PM state's bank.tar uri (the PM boots' Triton kernels)")
    sl.add_argument("--cubins", type=json.loads, required=True, help="JSON {tar uri: .cubin files it lists}")
    sl.add_argument("--meta", help="JSON merged into the summary; matrix_notes{arm: str}")
    sl.add_argument("--matrix-out")
    sl.add_argument("--expect-text", default="The primary colors are red, yellow, and blue.")
    sl.add_argument("--exclude", nargs="*", default=[])
    sl.add_argument("--cause", type=json.loads, default={})
    args = parser.parse_args()
    if getattr(args, "configmap", None):
        global CONFIGMAP
        CONFIGMAP = args.configmap
    if args.cmd == "configmap":
        configmap()
    elif args.cmd == "run":
        sys.exit(cmd_run(args))
    elif args.cmd == "adopt":
        out = Path(args.out)
        with ThreadPoolExecutor(max_workers=len(args.pods)) as pool:
            jobs = [pool.submit(run_one, None, None, None, None, out, export=(spec.partition("=")[2] or None),
                                adopt=spec.partition("=")[0], held=args.node) for spec in args.pods]
            sys.exit(0 if all(j.result()["ok"] for j in jobs) else 1)
    elif args.cmd == "loo":
        cmd_loo(args)
    elif args.cmd == "matrix":
        cmd_matrix(args)
    elif args.cmd == "tritonfix":
        cmd_tritonfix(args)
    elif args.cmd == "sllm":
        cmd_sllm(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
