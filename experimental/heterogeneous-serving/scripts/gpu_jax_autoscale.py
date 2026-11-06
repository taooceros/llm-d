#!/usr/bin/env python3
"""vLLM-JAX (vLLM + tpu-inference JAX backend, vllm-tpu 0.32.0 on JAX 0.11.2) booting Gemma-4 31B on a fresh host,
a3-highgpu-1g (GPU_JAX_GOAL.md): the stock delivery arms of the CUDA vLLM campaign plus TPUFlex. Node
handling, cache export, GKE stage timing come from scripts/gpu_autoscale.py.

Fresh host = node reuse (--node, user decision 2026-10-24; H100 spot stockout): every pod is pinned to one held node
(nodeName, no `used` anti-affinity), the node is never retired between runs, and each fresh-host arm (stock-*,
pm-gcs, the pm-peer target) gets a NEW pod that runs gpu_autoscale/wipe.py before its setup (page-cache eviction,
receipt <pod>-wipe.json). GKE stages are not measured (summary: gke null). Without --node: a fresh node per pod.

    scripts/gpu_jax_autoscale.py run --node N --units stock-ckpt,stock-lazy,stock-runai --samples 3 --export gs://.../compile-cache/jax-ckpt-X.tar --out DIR
    scripts/gpu_jax_autoscale.py run --node N --units stock-ckpt-cc,stock-runai-cc --samples 3 --cc URI --out DIR
    scripts/gpu_jax_autoscale.py seed --node N --dest gs://.../pm-state/<id> --out DIR    (TPUFlex state, untimed)
    scripts/gpu_jax_autoscale.py run --node N [--peer-node M] --units pm --samples 3 --pm-state URI --kv-bytes K --out DIR
    scripts/gpu_jax_autoscale.py run ... --dry-run          (plan + server-side validation of every pod spec)
    scripts/gpu_jax_autoscale.py run --node N --units pm-gcs,loo-snapshot,loo-cache0,... --samples 3 --pm-state URI
                                     --kv-bytes K --cc STOCK_CC_TAR --out DIR    (leave-one-out, interleaved per sample)
    scripts/gpu_jax_autoscale.py loo-summary DIR --out results/gpu-jax/loo-<rev>/loo_summary.json
    scripts/gpu_jax_autoscale.py report DIR [DIR ...]
    scripts/gpu_jax_autoscale.py summary DIR [DIR ...] --out results/gpu-jax/matrix-<commit>/matrix_summary.json

A pod (image gpu_jax/image/IMAGE, no fork code) lands on its node, does the setup (wipe; gcsfuse mount for
stock-lazy; Ray worker join with resource run_<id>) and the driver then submits the timed parts,
gpu_jax/hetero_jax_gpu/run_arm.sh, as Ray jobs (working_dir = gpu_jax/) pinned to that pod through the Ray head in
namespace gpu-jax (dashboard port-forwarded to RAY_DASHBOARD). The patches that make tpu-inference run on CUDA,
and the TPUFlex engine mechanisms (hetero_jax_gpu/pm.py, only with PM_* set), load in every engine process
through gpu_jax/sitecustomize.py.

Units (each starts on fresh nodes; arms are the canonical matrix names):
  stock-lazy      lazy safetensors reads through gcsfuse
  stock-ckpt      checkpoint copied to tmpfs first (fetch.py ckpt), then warm-stock on the same pod: a restart
                  with the checkpoint in tmpfs and the stack's own compile cache warm
  stock-ckpt-cc   stock-ckpt + a peer's $HOME caches (JAX persistent cache at JAX's default 1 s threshold + XLA
                  autotune results)
  stock-runai     Run:ai model streamer from gs:// (vLLM --load-format runai_streamer)
  stock-runai-cc  stock-runai + the same peer caches
  pm              node A: pm-gcs (TPUFlex state from GCS), then warm-pm (same pod: pre-spawned engine, the old
                  engine exits, gate opens); A then serves its state as a live peer for node B: pm-peer (node
                  reuse: only with --peer-node, a second held node)
  pm-gcs          pm-gcs alone (no warm-pm, no peer): the leave-one-out baseline
  loo-<mech>      pm-gcs with one mechanism replaced by stock-runai-cc's equivalent (LOO below; run_arm.sh)
Outputs in DIR: runs.jsonl, <run_id>.log (Ray job log, epoch-prefixed), <pod>-pod.log, <pod>-k8s.json,
<pod>-wipe.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import string
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_autoscale as ga  # noqa: E402

ga.NAMESPACE = "gpu-jax"
ROOT = Path(__file__).resolve().parent.parent
GPU_JAX = ROOT / "gpu_jax"
IMAGE = (GPU_JAX / "image" / "IMAGE").read_text().strip()
LOCK_SHA = hashlib.sha256((GPU_JAX / "image" / "requirements.lock").read_bytes()).hexdigest()
RAY_DASHBOARD = os.environ.get("RAY_DASHBOARD", "http://127.0.0.1:18265")
RAY_GCS = "ray-head.gpu-jax.svc.cluster.local:6379"
ARMS = ("stock-lazy", "stock-ckpt", "stock-ckpt-cc", "stock-runai", "stock-runai-cc", "pm-gcs", "pm-peer",
        "warm-stock", "warm-pm")
# Leave-one-out arms: removed mechanism -> what stands in for it (stock-runai-cc's way of doing that phase).
LOO = {"loo-snapshot": ("weight snapshot (restore of the post-conversion device arrays)",
                        "stock tpu-inference loader, weights via vLLM's runai_streamer from gs:// (as stock-runai-cc)"),
       "loo-cache0": ("persistent JAX cache at a 0 s threshold (the state's cc.tar, cache env from process start)",
                      "stock-runai-cc's $HOME cache tar (JAX default 1 s threshold; tpu-inference enables its cache)"),
       "loo-zygote": ("zygote forkserver (PM_ZYGOTE=1)", "vLLM's stock spawn of the engine core (PM_ZYGOTE=0)"),
       "loo-overlap": ("overlapped delivery (engine launched at meta_done, download overlaps spawn)",
                       "sequential: whole state fetched, then the engine launched"),
       "loo-read-threads": ("restore by read() into memory with 32 threads",
                            "np.memmap views with 4 threads (pre-bfb02b9 restore, stock-like mmap path)"),
       # factorial cell (results/gpu-jax/loo-factorial-*): both loo-snapshot's and loo-cache0's stand-ins at once
       "loo-snapshot-cache0": ("weight snapshot AND the 0 s-threshold cache",
                               "stock loader on vLLM's runai_streamer + stock-runai-cc's 1 s-threshold cache tar")}
LOO_ARMS = ("pm-gcs", *LOO)
UNITS = {"stock-lazy": ["stock-lazy"], "stock-ckpt": ["stock-ckpt", "warm-stock"], "stock-ckpt-cc": ["stock-ckpt-cc"],
         "stock-runai": ["stock-runai"], "stock-runai-cc": ["stock-runai-cc"], "pm": ["pm-gcs", "warm-pm", "pm-peer"],
         **{a: [a] for a in LOO_ARMS}}
OLD_NAMES = {"jax-raw": "stock-lazy", "jax-ckpt": "stock-ckpt", "jax-ckpt-cc": "stock-ckpt-cc",
             "jax-runai": "stock-runai", "jax-runai-cc": "stock-runai-cc"}
JOB_ENV = {"HETERO_JAX_GPU": "1", "HETERO_JAX_GPU_VERBOSE": "1", "JAX_PLATFORMS": "cuda,cpu",
           # JAX's GPU allocator pool (default 75% of HBM) must hold 58 GiB of weights plus KV cache;
           # tpu-inference then sizes the KV cache to --gpu-memory-utilization of this pool.
           "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
           "PYTHONUNBUFFERED": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
CONFIGMAP = "gpu-jax-scripts"
PEER_PORT = 7070
EXPECTED_CHECK = "The primary colors are red, yellow, and blue."
GKE_NOTE = "GKE stages not measured on GPU (user decision; H100 stockout)"
FRESH_HOST_ARMS = ("stock-lazy", "stock-ckpt", "stock-ckpt-cc", "stock-runai", "stock-runai-cc", "pm-gcs", "pm-peer",
                   *LOO)


def configmap(dry_run=False):
    files = {p.name: p.read_text() for p in (ga.HERE / "pod_run_jax.sh", ga.HERE / "fetch.py", ga.HERE / "pm_state.py",
                                             ga.HERE / "wipe.py")}
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": CONFIGMAP, "namespace": ga.NAMESPACE},
          "data": files}
    ga.kubectl("apply", *(["--dry-run=server"] if dry_run else []), "-f", "-", inp=json.dumps(cm))


def pod_spec(name, rid, arm, pool, node=None, wipe=False, zone=None):
    """pool: comma-separated node pools (spot, and flex-start since the 2026-10-23 spot stockout). node: pin to this
    held node (nodeName: no scheduler, no pool or `used` affinity). wipe: wipe.py before the pod's setup. zone: fresh
    nodes only in this zone (GPU e2e on spot: zone b was STOCKOUT)."""
    spec = ga.pod_spec(name, "ckpt" if arm in ("stock-ckpt", "stock-ckpt-cc", "pm-seed") else "raw", pool, None)
    spec["metadata"]["labels"] = {"app": "gpu-jax", "arm": arm, "run": rid}
    del spec["spec"]["nodeSelector"]
    if node:
        spec["spec"]["nodeName"] = node
        del spec["spec"]["affinity"]
    else:
        terms = spec["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
            "nodeSelectorTerms"][0]["matchExpressions"]
        terms.append({"key": "cloud.google.com/gke-nodepool", "operator": "In", "values": pool.split(",")})
        if zone:
            terms.append({"key": "topology.kubernetes.io/zone", "operator": "In", "values": [zone]})
    c = spec["spec"]["containers"][0]
    c["image"], c["command"] = IMAGE, ["/scripts/pod_run_jax.sh"]
    if arm.startswith(("pm", "loo")):  # TPUFlex state: model meta + snapshot (~58 GiB) + live cache tar on a peer
        c["volumeMounts"].append({"name": "pmstate", "mountPath": "/pmstate"})
        spec["spec"]["volumes"].append({"name": "pmstate", "emptyDir": {"medium": "Memory", "sizeLimit": "80Gi"}})
    env = {"ARM": arm, "BUCKET": ga.BUCKET, "PREFIX": ga.PREFIX, "RUN_ID": rid, "RAY_ADDRESS": RAY_GCS,
           "HOME": "/cache/home", "PYTHONUNBUFFERED": "1",
           # GKE mounts the driver here; the python base image has no loader entry for it
           "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64",
           "PATH": "/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
    if wipe:
        env["WIPE"] = "1"
    c["env"] = [{"name": k, "value": v} for k, v in env.items()]
    c["env"].append({"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}})
    spec["spec"]["volumes"][0]["configMap"]["name"] = CONFIGMAP
    return spec


def provenance():
    """Image, lock hash and the jj commit holding gpu_jax/ (the parent when the working copy is empty;
    otherwise the working-copy commit, marked dirty)."""
    state = subprocess.run(["jj", "log", "-r", "@", "--no-graph", "-T",
                            'if(empty, "clean", "dirty") ++ " " ++ commit_id ++ " " ++ parents.map(|p| p.commit_id()).join(",")'],
                           cwd=ROOT, capture_output=True, text=True).stdout.split()
    commit = state[2] if state and state[0] == "clean" else (state[1] + "+dirty" if state else "unknown")
    return {"image": IMAGE, "lock_sha256": LOCK_SHA, "gpu_jax_commit": commit}


def job_client():
    from ray.job_submission import JobSubmissionClient
    return JobSubmissionClient(RAY_DASHBOARD)


def retrying(call, *args, attempts=40, delay=3.0, **kwargs):
    """The dashboard is reached through a kubectl port-forward that drops now and then (restarted by a
    loop); retry connection failures for about two minutes instead of failing the run."""
    for attempt in range(attempts):
        try:
            return call(*args, **kwargs)
        except Exception as exc:  # requests ConnectionError / RuntimeError from the SDK
            if attempt == attempts - 1 or "onnect" not in repr(exc):
                raise
            time.sleep(delay)


def rand(k=4):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=k))


# ------------------------------------------------------------------------------------------------ nodes

class Nodes:
    """At most `limit` GPU pods (= fresh nodes) of this driver at once; a pm unit takes two."""

    def __init__(self, limit):
        self.free, self.cv = limit, threading.Condition()

    def take(self, n):
        with self.cv:
            self.cv.wait_for(lambda: self.free >= n)
            self.free -= n

    def give(self, n=1):
        with self.cv:
            self.free += n
            self.cv.notify_all()


class Pod:
    """One pod, created and waited to MARK setup_done; finish() saves its logs/k8s record and deletes it. Fresh node
    (node=None): scheduled onto a node no run has used, which finish() retires. Node reuse (node=<held node>):
    pinned with nodeName once no other GPU pod is on the node, never retired; with wipe, wipe.py runs before the
    setup and its receipt is saved as <pod>-wipe.json (self.receipt)."""

    def __init__(self, arm, sample, pool, out, timeout, node=None, wipe=False, zone=None):
        self.rid = f"{arm}-{sample}-{rand()}"
        self.name, self.out, self.arm = f"jx-{self.rid}", out, arm
        self.node, self.gpu_seen, self.deadline = node, None, time.time() + timeout
        self.reuse, self.wipe, self.receipt, self.zone = bool(node), wipe, None, zone
        self.node_wait_s = wait_node_free(node, self.deadline) if node else None
        self.t_submit = time.time()
        ga.kubectl("apply", "-f", "-", inp=json.dumps(pod_spec(self.name, self.rid, arm, pool, node=node, wipe=wipe,
                                                               zone=zone)))
        ga.log(self.rid, "submitted", f"(pinned to {node}, wipe={int(wipe)})" if node else "")
        self.gke = None
        self.recorded = False
        self.finished = False

    def wait_setup(self):
        while True:  # until the container runs; note when the node first advertises its GPU
            pod = ga.kjson("get", "pod", self.name)
            self.node = pod["spec"].get("nodeName") or self.node
            if self.node and self.gpu_seen is None:
                n = node_json(self.node)
                if n.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu") not in (None, "0"):
                    self.gpu_seen = time.time()
            state = (pod["status"].get("containerStatuses") or [{}])[0].get("state", {})
            if "running" in state or "terminated" in state:
                break
            if pod["status"].get("phase") == "Failed":  # e.g. kubelet admission (OutOfnvidia.com/gpu) of a pinned pod
                raise RuntimeError(f"pod failed: {pod['status'].get('reason')} {pod['status'].get('message')}")
            if time.time() > self.deadline:
                raise TimeoutError("container never started")
            time.sleep(2)
        ga.log(self.rid, "running on", self.node)
        text = wait_pod_mark(self.name, "setup_done", self.deadline)
        if self.wipe:
            found = [line.split(" WIPE ", 1)[1] for line in text.splitlines() if " WIPE {" in line]
            if not found:
                raise RuntimeError("no wipe receipt in the pod log")
            self.receipt = json.loads(found[0])
            (self.out / f"{self.name}-wipe.json").write_text(json.dumps(self.receipt, indent=1))
            ga.log(self.rid, "wipe", json.dumps(wipe_summary(self.receipt, self.name)))
        self.ip = ga.kjson("get", "pod", self.name)["status"]["podIP"]
        self.node_pool = node_json(self.node).get("metadata", {}).get("labels", {}).get("cloud.google.com/gke-nodepool")

    def stages(self):
        """Saves the pod/events/node record once; GKE stages only on a fresh node (None with node reuse)."""
        if not self.recorded:
            pod = ga.kjson("get", "pod", self.name)
            events = ga.kjson("get", "events", "--field-selector", f"involvedObject.name={self.name}")
            node_obj = node_json(self.node) if self.node else {}
            (self.out / f"{self.name}-k8s.json").write_text(json.dumps({"pod": pod, "events": events, "node": node_obj}))
            self.recorded = True
            if not self.reuse:
                self.gke = ga.gke_stages(self.t_submit, pod, events, node_obj, self.gpu_seen)
        return self.gke

    def finish(self, keep=False):
        if self.finished:
            return None
        self.finished = True
        (self.out / f"{self.name}-pod.log").write_text(ga.kubectl("logs", "--timestamps", self.name, check=False))
        try:
            self.stages()
        except Exception as exc:  # noqa: BLE001
            ga.log(self.rid, "k8s record failed", exc)
        if keep:
            return None
        ga.kubectl("delete", "pod", self.name, "--wait=false", check=False)
        if self.reuse:
            return None
        return ga.retire_node(self.node) if self.node else None


def wipe_summary(receipt, pod_name):
    """The run record's view of a wipe receipt (full receipt: <pod>-wipe.json)."""
    before, after = receipt["residency_before"], receipt["residency_after"]
    return {"file": f"{pod_name}-wipe.json", "drop_caches_3": receipt["evicted"]["drop_caches_3"],
            "fadvise_files": receipt["evicted"]["fadvise_dontneed_files"],
            "resident_before": {d: v["resident_bytes"] for d, v in before.items()},
            "resident_after": {d: v["resident_bytes"] for d, v in after.items()},
            "pod_state_files_before": sum((v or {}).get("files", 0) for v in receipt["pod_paths_before"].values()),
            "shmem_kB_before": receipt["meminfo_before"].get("Shmem_kB"),
            "cached_kB_before": receipt["meminfo_before"].get("Cached_kB"),
            "cached_kB_after": receipt["meminfo_after"].get("Cached_kB"), "seconds": receipt["seconds"]}


def wait_node_free(node, deadline, poll=3.0):
    """Node reuse: wait until the held node is Ready and no pod (any namespace) holding a GPU is on it, else kubelet
    rejects the pinned pod (OutOfnvidia.com/gpu). Returns the seconds waited."""
    t0 = time.time()
    while True:
        busy, ready = gpu_pods_on(node)
        if ready and not busy:
            return time.time() - t0
        if time.time() > deadline:
            raise TimeoutError(f"held node {node}: ready={ready}, GPU pods {busy}")
        time.sleep(poll)


def gpu_pods_on(node):
    """(non-terminated pods with a GPU limit on node, node Ready). Raises if the node is gone."""
    n = node_json(node)
    if not n:
        raise RuntimeError(f"held node {node} is gone (spot preemption?)")
    ready = any(c["type"] == "Ready" and c["status"] == "True" for c in n["status"].get("conditions", []))
    pods = json.loads(subprocess.run(["kubectl", "--kubeconfig", ga.KUBECONFIG, "get", "pods", "-A", "--field-selector",
                                      f"spec.nodeName={node}", "-o", "json"], capture_output=True, text=True,
                                     check=True).stdout)["items"]
    busy = [f"{p['metadata']['namespace']}/{p['metadata']['name']}" for p in pods
            if p["status"].get("phase") not in ("Succeeded", "Failed")
            and any(c.get("resources", {}).get("limits", {}).get("nvidia.com/gpu") for c in p["spec"]["containers"])]
    return busy, ready


def verify_cleanup(pod, retire, wait=180):
    """After finish(): the pod is gone, its Ray worker left the cluster and the node's VM delete was accepted (node
    reuse: the node stays)."""
    deadline, gone = time.time() + wait, False
    while time.time() < deadline and not gone:
        gone = not ga.kubectl("get", "pod", pod.name, "--ignore-not-found", "-o", "name", check=False).strip()
        if not gone:
            time.sleep(3)
    ip, alive, deadline = getattr(pod, "ip", None), [], time.time() + wait
    while ip:  # the raylet is marked DEAD after its heartbeats time out
        try:
            from ray.util.state import list_nodes
            alive = [n.node_id for n in retrying(lambda: list_nodes(address=RAY_DASHBOARD, limit=1000))
                     if n.node_ip == ip and n.state == "ALIVE"]
        except Exception as exc:  # noqa: BLE001
            alive = [f"unknown: {exc!r}"[:160]]
        if not alive or time.time() > deadline:
            break
        time.sleep(5)
    retire_ok = bool(retire) and "rc=0" in retire
    return {"cleanup": pod.name, "node": pod.node, "pod_deleted": gone, "ray_worker_alive": alive,
            "node_delete": retire, "node_kept": pod.reuse,
            "ok": gone and not alive and (pod.reuse or retire_ok or pod.node is None), "at": time.time()}


def node_json(node):
    return json.loads(subprocess.run(["kubectl", "--kubeconfig", ga.KUBECONFIG, "get", "node", node, "-o", "json"],
                                     capture_output=True, text=True).stdout or "{}")


def wait_pod_mark(name, mark, deadline):
    while time.time() < deadline:
        text = ga.kubectl("logs", "--timestamps", name, check=False)
        if re.search(rf"^\S+ MARK {mark} ", text, re.M):
            return text
        state = (ga.kjson("get", "pod", name)["status"].get("containerStatuses") or [{}])[0].get("state", {})
        if "terminated" in state:
            raise RuntimeError(f"pod ended before MARK {mark}")
        time.sleep(2)
    raise TimeoutError(f"no MARK {mark}")


# ------------------------------------------------------------------------------------------------- jobs

def submit_job(pod, jid, arm, env):
    env = dict(JOB_ENV, ARM=arm, BUCKET=ga.BUCKET, PREFIX=ga.PREFIX, **{k: str(v) for k, v in env.items()})
    return retrying(
        lambda **kw: job_client().submit_job(**kw),
        entrypoint="bash hetero_jax_gpu/run_arm.sh", submission_id=f"run-{jid}",
        entrypoint_resources={f"run_{pod.rid}": 1},
        runtime_env={"working_dir": str(GPU_JAX), "excludes": ["__pycache__", "tests", "image/*.sh"],
                     "env_vars": env})


def wait_job(job_id, logfile, deadline, until_check=False, interval=5.0):
    """Snapshot the Ray job log every few seconds until the job ends (or, with until_check, until MARK check_done:
    a kept engine keeps serving), MARK engine_died, or the deadline; returns the job status."""
    while True:
        status = str(retrying(lambda: job_client().get_job_status(job_id))).split(".")[-1]
        text = retrying(lambda: job_client().get_job_logs(job_id))
        logfile.write_text(text)
        if status in ("SUCCEEDED", "FAILED", "STOPPED") or time.time() > deadline:
            return status
        if until_check and re.search(r"^MARK check_done ", text, re.M):
            return status
        if re.search(r"^MARK (check_done|engine_died) ", text, re.M):
            time.sleep(interval)  # let the engine shut down and the job finish
        time.sleep(interval)


def run_job(pod, arm, sample, out, prov, env=None, keep=False, fresh=True, extra=None):
    """One timed boot (Ray job) on pod; appends its record to runs.jsonl. fresh: the first job on a new pod (a fresh
    node, or the held node after the pod's wipe)."""
    jid = f"{arm}-{sample}-{rand()}" if arm != pod.arm else pod.rid
    env = dict(env or {}, KEEP=int(keep))
    reuse = getattr(pod, "reuse", False)
    record = {"run_id": jid, "arm": arm, "sample": sample, "pod": pod.name, "node_fresh": fresh and not reuse,
              "node_reused": reuse, "stack": "jax", "job_env": {k: v for k, v in env.items()}, **prov, "ok": False,
              "t_pod_submit": pod.t_submit, "zone": getattr(pod, "zone", None), **(extra or {})}
    if fresh and reuse:
        receipt = getattr(pod, "receipt", None)
        record["wipe"] = wipe_summary(receipt, pod.name) if receipt else None
    try:
        record["t_job_submit"] = time.time()
        record["job_id"] = submit_job(pod, jid, arm, env)
        record["job_status"] = wait_job(record["job_id"], out / f"{jid}.log", pod.deadline, until_check=keep)
        parsed = parse_logs(ga.kubectl("logs", "--timestamps", pod.name, check=False) if fresh else "",
                            (out / f"{jid}.log").read_text())
        record.update(parsed)
        record["check_ok"] = parsed.get("check_text") == EXPECTED_CHECK
        record["ok"] = ((keep or record["job_status"] == "SUCCEEDED") and parsed.get("ready") is not None
                        and record["check_ok"] and "segments" in parsed)
    except Exception as exc:  # recorded, never fatal for sibling runs
        record["error"] = repr(exc)
        ga.log(jid, "ERROR", exc)
    record["node"] = pod.node
    record["node_pool"] = getattr(pod, "node_pool", None)
    if fresh:
        try:
            record["gke"] = pod.stages()  # None with node reuse
        except Exception as exc:  # noqa: BLE001
            record["gke_error"] = repr(exc)
    with ga._print_lock, (out / "runs.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    ga.log(jid, "ok" if record["ok"] else "FAILED", f"ready {fmt(record.get('ready_s'))}",
           " ".join(f"{k}={fmt(v)}" for k, v in (record.get("segments") or {}).items()))
    return record


def fmt(v):
    return "-" if v is None else f"{v:.1f}"


def start_peer(pod):
    """Make pod a live peer: its $HOME caches as /pmstate/cc.tar (the engine keeps serving), then pm_state.py serve
    detached in the container."""
    t0 = time.time()
    size = ga.kubectl("exec", pod.name, "--", "python3", "/scripts/pm_state.py", "home-tar", "--home", "/cache/home",
                      "--out", "/pmstate/cc.tar").strip()
    ga.kubectl("exec", pod.name, "--", "bash", "-c",
               f"setsid nohup python3 /scripts/pm_state.py serve --root /pmstate --port {PEER_PORT} "
               f">/pmstate/serve.log 2>&1 < /dev/null &")
    time.sleep(1)
    ga.log(pod.rid, f"peer serving on {pod.ip}:{PEER_PORT} (cc.tar {size} B, {time.time() - t0:.1f} s)")
    return {"peer": f"{pod.ip}:{PEER_PORT}", "peer_cc_tar_bytes": int(size), "peer_prep_s": time.time() - t0}


def run_unit(unit, sample, args, out, prov, nodes):
    """One unit on fresh nodes, or on the held node (args.node) after a wipe; returns its records. Never raises
    (failures are records)."""
    arms, records = UNITS[unit], []
    reuse = bool(args.node)
    need = 2 if unit == "pm" else 1
    nodes.take(need)
    pods, held = [], need
    try:
        a = Pod(arms[0], sample, args.pool, out, args.timeout, node=args.node, wipe=reuse, zone=args.zone)
        pods.append(a)
        a.wait_setup()
        if unit == "pm":
            pm_env = {"PM_STATE_URI": args.pm_state, "PM_KV_BYTES": args.kv_bytes}
            records.append(run_job(a, "pm-gcs", sample, out, prov, pm_env, keep=True))
            if records[-1]["ok"]:
                b = None
                if not reuse or args.peer_node:  # node reuse: pm-peer needs a second held node at the same time
                    b = Pod("pm-peer", sample, args.pool, out, args.timeout, node=args.peer_node, wipe=reuse,
                            zone=args.zone)
                    pods.append(b)  # sets up while warm-pm runs on A
                records.append(run_job(a, "warm-pm", sample, out, prov, pm_env, keep=True, fresh=False))
                if b is not None:
                    peer = start_peer(a)
                    b.wait_setup()
                    records.append(run_job(b, "pm-peer", sample, out, prov, {**pm_env, "PM_PEER": peer["peer"]},
                                           extra={"peer_pod": a.name, **peer}))
        elif unit in LOO_ARMS:  # one fresh-host PM boot; the engine stops after the check
            env = {"PM_STATE_URI": args.pm_state, "PM_KV_BYTES": args.kv_bytes,
                   **({"CC_URI": args.cc} if unit in ("loo-cache0", "loo-snapshot-cache0") else {})}
            records.append(run_job(a, unit, sample, out, prov, env))
        else:
            env = {"CC_URI": args.cc} if unit.endswith("-cc") else {}
            records.append(run_job(a, arms[0], sample, out, prov, env))
            if unit == "stock-ckpt" and records[-1]["ok"]:
                if args.export and sample == 0:
                    records[-1]["export"] = ga.export_cache(a.name, args.export)
                    ga.log(a.rid, "exported", records[-1]["export"]["bytes"], "B ->", args.export)
                records.append(run_job(a, "warm-stock", sample, out, prov, {}, fresh=False))
    except Exception as exc:  # noqa: BLE001
        ga.log(unit, sample, "unit ERROR", exc)
        records.append({"unit": unit, "sample": sample, "error": repr(exc), "ok": False})
    finally:
        for p in pods:
            retire = p.finish(keep=args.keep)
            if retire:
                ga.log(p.rid, retire)
            if not args.keep:
                cleanup = verify_cleanup(p, retire)
                with ga._print_lock, (out / "runs.jsonl").open("a") as f:
                    f.write(json.dumps(cleanup) + "\n")
                ga.log(p.rid, "cleanup", "OK" if cleanup["ok"] else "NOT CLEAN", json.dumps(cleanup))
        nodes.give(held)
    return records


def cmd_run(args):
    out = Path(args.out)
    units = args.units.split(",")
    for u in units:
        if u not in UNITS:
            sys.exit(f"unknown unit {u}")
        if (u.endswith("-cc") or u in ("loo-cache0", "loo-snapshot-cache0")) and not args.cc:
            sys.exit(f"{u} needs --cc")
        if (u == "pm" or u in LOO_ARMS) and not (args.pm_state and args.kv_bytes):
            sys.exit(f"{u} needs --pm-state and --kv-bytes")
    if args.peer_node and not args.node:
        sys.exit("--peer-node needs --node")
    if args.dry_run:
        return dry_run(args, units)
    out.mkdir(parents=True, exist_ok=True)
    configmap()
    job_client().list_jobs()  # fail fast without the dashboard port-forward
    prov = provenance()
    with (out / "campaign.json").open("a") as f:
        f.write(json.dumps({"argv": sys.argv, "at": time.time(), **prov, "bucket": ga.BUCKET, "prefix": ga.PREFIX})
                + "\n")
    workers = 1 if args.node else args.nodes  # node reuse: one held node, units one after another
    nodes = Nodes(2 if args.node else args.nodes)
    jobs = [(u, s) for s in range(args.first_sample, args.first_sample + args.samples) for u in units]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for u, s in jobs:
            futures.append(pool.submit(run_unit, u, s, args, out, prov, nodes))
            time.sleep(args.stagger)
        records = [r for f in futures for r in f.result()]
    return 0 if all(r["ok"] for r in records) else 1


def dry_run(args, units):
    """The plan, plus server-side validation (kubectl apply --dry-run=server) of the configmap and of every pod
    spec; creates nothing."""
    reuse = bool(args.node)
    peer = not reuse or bool(args.peer_node)
    print(json.dumps({"mode": "node reuse" if reuse else "fresh nodes", "node": args.node, "peer_node": args.peer_node,
                      "workers": 1 if reuse else args.nodes, **provenance()}))
    for n in filter(None, (args.node, args.peer_node)):
        busy, ready = gpu_pods_on(n)
        print(f"held node {n}: ready={ready}, GPU pods on it now: {busy or 'none'}")
    configmap(dry_run=True)
    print(f"configmap {CONFIGMAP}: validated (pod_run_jax.sh, fetch.py, pm_state.py, wipe.py)")
    specs = {}
    for s in range(args.first_sample, args.first_sample + args.samples):
        for u in units:
            pods = [(UNITS[u][0], args.node)] + ([("pm-peer", args.peer_node)] if u == "pm" and peer else [])
            jobs = UNITS[u] if u != "pm" or peer else UNITS[u][:2]
            export = " (+ export of $HOME caches to " + args.export + ")" if u == "stock-ckpt" and s == 0 and args.export else ""
            print(f"unit {u} sample {s}: new pods {[a for a, _ in pods]} -> jobs {jobs}{export}; "
                  f"wipe before setup: {reuse}; after: pod deleted, " + ("node kept" if reuse else "node retired"))
            for arm, node in pods:
                specs.setdefault(arm, pod_spec(f"jx-dryrun-{arm}", "dryrun", arm, args.pool, node=node, wipe=reuse,
                                               zone=args.zone))
    for arm, spec in specs.items():
        ga.kubectl("apply", "--dry-run=server", "-f", "-", inp=json.dumps(spec))
        s, c = spec["spec"], spec["spec"]["containers"][0]
        env = {e["name"]: e.get("value", e.get("valueFrom")) for e in c["env"]}
        print(f"pod spec {arm}: server dry-run OK; nodeName={s.get('nodeName')} affinity={s.get('affinity')} "
              f"WIPE={env.get('WIPE')} NODE_NAME={env.get('NODE_NAME')} "
              f"volumes={[(v['name'], v.get('emptyDir')) for v in s['volumes']]}")
    return 0


def cmd_seed(args):
    """TPUFlex state: stock-ckpt boot with the snapshot written after tpu-inference's conversion and every program
    persisted, then the state uploaded to args.dest. Untimed; on a fresh node like any run (or on the held node after
    a wipe). With --pod, only the upload step for a kept seed pod (its job log in --out)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    if args.pod:
        rid = ga.kjson("get", "pod", args.pod)["metadata"]["labels"]["run"]
        return upload_seed(args.pod, rid, out, args.dest, prov)
    configmap()
    job_client().list_jobs()
    pod = Pod("pm-seed", 0, args.pool, out, args.timeout, node=args.node, wipe=bool(args.node))
    try:
        pod.wait_setup()
        rec = run_job(pod, "pm-seed", 0, out, prov, {})
        if not rec["ok"]:
            raise RuntimeError("seed boot failed")
        upload_seed(pod.name, pod.rid, out, args.dest, prov)
    finally:
        retire = pod.finish(keep=args.keep)
        if retire:
            ga.log(pod.rid, retire)
        if not args.keep:
            cleanup = verify_cleanup(pod, retire)
            with (out / "runs.jsonl").open("a") as f:
                f.write(json.dumps(cleanup) + "\n")
            ga.log(pod.rid, "cleanup", "OK" if cleanup["ok"] else "NOT CLEAN", json.dumps(cleanup))


def upload_seed(pod_name, rid, out, dest, prov):
    kv = re.findall(r"PM_KV_BYTES measured=(\d+)", (out / f"{rid}.log").read_text())
    if not kv:
        raise RuntimeError("seed log has no PM_KV_BYTES line")
    token = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True,
                           check=True).stdout.strip()
    t0 = time.time()
    text = ga.kubectl("exec", "-i", pod_name, "--", "python3", "/scripts/pm_state.py", "upload", "--root", "/pmstate",
                      "--home", "/cache/home", "--ckpt", f"gs://{ga.BUCKET}/{ga.PREFIX}", "--dest", dest,
                      inp=token + "\n")
    upload = json.loads(text.strip().splitlines()[-1].removeprefix("FETCH "))
    seed = {"dest": dest, "kv_bytes": int(kv[0]), "seed_run": rid, "upload": upload,
            "upload_wall_s": time.time() - t0, **prov}
    (out / "seed.json").write_text(json.dumps(seed, indent=1))
    ga.log("seed", json.dumps(seed))
    return seed


def cmd_job(args):
    """Dev: one more job on a kept pod."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pod_obj = ga.kjson("get", "pod", args.pod)

    class P:  # minimal Pod stand-in
        rid, name, arm = pod_obj["metadata"]["labels"]["run"], args.pod, None
        node, deadline = pod_obj["spec"].get("nodeName"), time.time() + args.timeout
    env = dict(kv.split("=", 1) for kv in args.env)
    run_job(P, args.arm, args.sample, out, provenance(), env, keep=args.keep_engine, fresh=False)


# ---------------------------------------------------------------------------------------------- parsing

# First job-log line containing the text -> event (epoch prefix written by hetero_jax_gpu.serve).
# tpu-inference order: engine core up -> [PM gate] -> load weights (flax_nnx loader | PM restore) -> "Init model"
# -> memory statistics + KV sizing -> precompile (sampler, logprobs, then KV allocation, backbone for every
# token padding, logits) with warm-up passes -> "init engine ... took" -> API server up.
LOG_ANCHORS = {
    "engine_init": "Initializing a V1 LLM engine",
    "pm_parked": "[pm] PM_PARKED",
    "pm_gate_open": "[pm] PM_GATE_OPEN",
    "load_start": "Loading model with MODEL_IMPL_TYPE",
    "load_end": "Init model |",
    "kv_start": "Memory statistics |",
    "precompile_start": "Precompile worker0",
    "backbone_start": "Precompile all the subgraphs",
    "precompile_end": "init engine (profile, create kv cache, warmup model) took",
    "reducing_torch_threads": "Reducing Torch threads",
}


def parse_logs(pod_text, job_text):
    marks, events, values, fetches, patches, check = {}, {}, {}, [], set(), None
    mech = {"jax_events": {}, "zygote_started": False, "zygote_child": False, "restore_source": None,
            "restore_threads": None, "runai_streamer": False, "pm_config": None}
    for line in pod_text.splitlines():  # kubelet RFC3339 stamp, then the MARK line (epoch also in the line)
        _, _, body = line.partition(" ")
        if body.startswith("MARK "):
            _, event, epoch = body.split()[:3]
            marks.setdefault(event, float(epoch))
    for line in job_text.splitlines():
        if line.startswith("MARK "):
            _, event, epoch = line.split()[:3]
            marks.setdefault(event, float(epoch))
            continue
        if line.startswith("PM_CONFIG "):  # run_arm.sh: the PM knobs the engine is launched with
            mech["pm_config"] = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
        if line.startswith("RAY_OMP_NUM_THREADS="):
            values["ray_omp_num_threads"] = line.split("=", 1)[1]
        if line.startswith("FETCH "):
            fetches.append(json.loads(line[6:]))
        if line.startswith("CHECK "):
            body = json.loads(line[6:])
            if "choices" in body:
                choice = body["choices"][0]
                check = choice["message"]["content"] if "message" in choice else choice["text"]
            continue
        stamp, _, body = line.partition(" ")
        try:
            t = float(stamp)
        except ValueError:
            continue
        m = re.search(r"\[hetero_jax_gpu\] patched (.+) \(pid \d+\)", body)
        if m:
            patches.add(m.group(1))
        for key, needle in LOG_ANCHORS.items():
            if key not in events and needle in body:
                events[key] = t
        for key, pattern in (("compiles", r"Compilation of .* finished in ([\d.]+) \[secs\]"),):
            for v in re.findall(pattern, body):
                values.setdefault(key, []).append(float(v))
        for key, pattern in (("warmup_s", r"Warm-up call pass finished in ([\d.]+) \[secs\]"),
                             ("weights_s", r"Loading weights took ([\d.]+) seconds"),
                             ("kv_tokens", r"KV cache size: ([\d,]+) tokens"),
                             ("init_engine_s", r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+)"),
                             ("pm_kv_measured", r"PM_KV_BYTES measured=(\d+)"),
                             ("pm_kv_returned", r"PM_KV_BYTES measured=\d+ returned=(\d+)"),
                             ("pm_restore_total_s", r"PM_SNAP_RESTORED .* total=([\d.]+)"),
                             ("pm_restore_wait_s", r"PM_SNAP_RESTORED .* wait=([\d.]+)"),
                             ("pm_restore_place_s", r"PM_SNAP_RESTORED .* place=([\d.]+)"),
                             ("pm_restore_graph_s", r"PM_SNAP_RESTORED .* graph=([\d.]+)"),
                             ("pm_snap_saved_s", r"PM_SNAP_SAVED .* took=([\d.]+)"),
                             ("torch_threads_reduced_from", r"Reducing Torch threads from (\d+)"),
                             ("load_torch_threads", r"\[hetero_jax_gpu\] load_model torch_threads=(\d+)")):
            mm = re.search(pattern, body)
            if mm and key not in values:
                values[key] = float(mm.group(1).replace(",", ""))
        if "Persistent compilation cache hit" in body or "cache hit for" in body.lower():
            values["jax_cache_hits"] = values.get("jax_cache_hits", 0) + 1
        mj = re.search(r"\[hetero_jax_gpu\] JAX_EVENTS (\S+) (\{.*\}) \(pid", body)
        if mj:
            ev = json.loads(mj.group(2))
            mech["jax_events"][mj.group(1)] = {
                "hits": ev.get("/jax/compilation_cache/cache_hits", [0])[0],
                "requests": ev.get("/jax/compilation_cache/compile_requests_use_cache", [0])[0],
                "compile_time_saved_s": ev.get("/jax/compilation_cache/compile_time_saved_sec", [0, 0.0])[1],
                "backend_compile_s": ev.get("/jax/core/compile/backend_compile_duration", [0, 0.0])[1]}
        if "[pm] PM_ZYGOTE started" in body:
            mech["zygote_started"] = True
        if "[pm] PM_ZYGOTE child running" in body:
            mech["zygote_child"] = True
        mr = re.search(r"PM_SNAP_RESTORED .* source=(\S+) threads=(\d+)", body)
        if mr:
            mech["restore_source"], mech["restore_threads"] = mr.group(1), int(mr.group(2))
        if "Loading safetensors using Runai Model Streamer" in body:
            mech["runai_streamer"] = True
    ready = marks.get("check_done") if check is not None else None
    out = {"marks": marks, "events": events, "values": values, "fetch": fetches, "check_text": check,
           "ready": ready, "health": marks.get("ready"), "patches": sorted(patches),
           "reducing_torch_threads": "reducing_torch_threads" in events, "mechanism": mech}
    jev = mech["jax_events"].values()
    mech["cache_hits"] = sum(p["hits"] for p in jev)
    mech["cache_requests"] = sum(p["requests"] for p in jev)
    mech["cache_misses"] = mech["cache_requests"] - mech["cache_hits"]
    fetch = next((f for f in fetches if f.get("kind") == "fetch-gcs"), None)
    if fetch:
        mech.update(fetch_snap=fetch.get("snap", True), fetch_snap_bytes=fetch.get("snap_bytes"),
                    fetch_cc_source=fetch.get("cc_source"), fetch_cc_bytes=fetch.get("cc_bytes"))
    if "engine_start" in marks and "download_end" in marks:  # overlap: the engine starts before delivery ends
        mech["engine_start_minus_download_end_s"] = marks["engine_start"] - marks["download_end"]
    need_m = ("download_start", "download_end", "engine_start", "ready", "check_done")
    need_e = ("engine_init", "load_start", "load_end", "precompile_start", "precompile_end")
    if all(k in marks for k in need_m) and all(k in events for k in need_e):
        e, m = events, marks
        t0 = m["download_start"]
        # Pre-spawned (warm-pm): the engine started before T0 and parked; spawn happened before T0. PM gate: device
        # init starts when the gate opens (fresh hosts: the compile cache landed, usually before the engine got there).
        device_from = max(e["engine_init"], e.get("pm_gate_open", 0.0))
        out["segments"] = {"download": m["download_end"] - m["download_start"],
                           "spawn": max(0.0, e["engine_init"] - max(m["engine_start"], t0)),
                           "init_device": e["load_start"] - device_from,
                           "lazy": None,
                           "load_model": e["load_end"] - e["load_start"],
                           "kv": e["precompile_start"] - e["load_end"],  # memory stats + KV sizing
                           # XLA precompile + warm-up of every padding (KV tensors are allocated in here)
                           "compile": e["precompile_end"] - e["precompile_start"],
                           "graphs": None,
                           "tail": m["check_done"] - e["precompile_end"]}  # API server start + greedy check
        out["ready_s"] = m["check_done"] - t0
        out["health_s"] = m["ready"] - t0
        out["engine_s"] = m["check_done"] - m["engine_start"]
        if "pm_parked" in e and "pm_gate_open" in e:
            out["gate_wait_s"] = e["pm_gate_open"] - e["pm_parked"]
        if m["engine_start"] < t0:
            out["prespawn"] = {"launch_to_parked": e.get("pm_parked", e["engine_init"]) - m["engine_start"],
                               "ready_from_launch_s": m["check_done"] - m["engine_start"],
                               "old_engine_exit_s": (m["old_engine_exited"] - m["prespawn_parked"]
                                                     if "old_engine_exited" in m and "prespawn_parked" in m else None)}
        out["setup"] = {"pod_to_setup_done": m["setup_done"] - m["pod_start"]
                        if "pod_start" in m and "setup_done" in m else None,
                        "job_dispatch": m["download_start"] - m["setup_done"] if "setup_done" in m else None}
    return out


# ---------------------------------------------------------------------------------------------- reports

def load_runs(dirs):
    by_arm = {}
    for d in map(Path, dirs):
        for line in (d / "runs.jsonl").read_text().splitlines():
            r = json.loads(line)
            if "arm" not in r:
                continue
            r["arm"] = OLD_NAMES.get(r["arm"], r["arm"])
            r["_dir"] = os.path.relpath(d.resolve(), ROOT)
            by_arm.setdefault(r["arm"], []).append(r)
    return by_arm


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def cmd_report(args):
    by_arm = load_runs(args.dirs)
    for arm in ARMS:
        runs = [r for r in by_arm.get(arm, []) if r.get("ok")]
        if not runs:
            continue
        totals = sorted(round(r["ready_s"], 1) for r in runs)
        print(f"{arm:15s} n={len(runs)} ready median {statistics.median(totals):7.1f} s  runs {totals}")
        keys = runs[0]["segments"].keys()
        print("                " + " ".join(f"{k}={fmt(med(r['segments'][k] for r in runs))}" for k in keys))
        v = lambda k: med(r["values"].get(k) for r in runs)  # noqa: E731
        print(f"                weights_s={v('weights_s')} kv_tokens={v('kv_tokens')} restore={v('pm_restore_total_s')}"
              f" load_torch_threads={sorted({r['values'].get('load_torch_threads') for r in runs}, key=str)}")
        print(f"                greedy: {sorted({r['check_text'] for r in runs})}")


def arm_note(arm, runs, notes):
    """User decision 2026-10-24: every GPU arm carries the GKE note, plus `reused-node` when it ran on a held node."""
    parts = [GKE_NOTE] + (["reused-node"] if any(r.get("node_reused") for r in runs) else [])
    return "; ".join(parts + ([notes[arm]] if notes.get(arm) else []))


def cmd_summary(args):
    by_arm = load_runs(args.dirs)
    commit = args.commit or provenance()["gpu_jax_commit"][:12]
    arms = {}
    notes = json.loads(Path(args.notes).read_text()) if args.notes else {}
    for arm in ARMS:
        runs = by_arm.get(arm, [])
        good = [r for r in runs if r.get("ok")]
        if not good:
            arms[arm] = {"status": "na" if arm in notes.get("_na", []) else "dnf", "n": 0, "runs": [], "median": None,
                         "phases": None, "gke": None, "note": arm_note(arm, runs, notes),
                         "failed_attempts": [{"run_id": r["run_id"], "node": r.get("node"), "dir": r["_dir"],
                                              "job_status": r.get("job_status"), "error": r.get("error")} for r in runs]}
            continue
        phases = {k: med(r["segments"][k] for r in good) for k in
                  ("download", "spawn", "init_device", "lazy", "load_model", "compile", "kv", "graphs", "tail")}
        arms[arm] = {"status": "ok", "n": len(good), "runs": [round(r["ready_s"], 1) for r in good],
                     "median": med(r["ready_s"] for r in good), "phases": phases,
                     "gke": None,  # not measured on GPU (user decision): node reuse, no fresh-node provisioning
                     "note": arm_note(arm, runs, notes),
                     "nodes": sorted({r.get("node") for r in good}, key=str),
                     "node_pools": dict(sorted(Counter(r.get("node_pool") or "?" for r in good).items())),
                     "failed_attempts": len(runs) - len(good),
                     "health_median": med(r.get("health_s") for r in good),
                     "greedy_checks_passed": all(r.get("check_text") == EXPECTED_CHECK for r in good),
                     "weights_s_median": med(r["values"].get("weights_s") for r in good),
                     "kv_tokens": sorted({r["values"].get("kv_tokens") for r in good}, key=str),
                     "load_torch_threads": sorted({r["values"].get("load_torch_threads") for r in good}, key=str),
                     "run_ids": [r["run_id"] for r in good], "dirs": sorted({r["_dir"] for r in good})}
        pm = {k: med(r["values"].get(k) for r in good) for k in
              ("pm_restore_total_s", "pm_restore_graph_s", "pm_restore_wait_s", "pm_restore_place_s")}
        if any(v is not None for v in pm.values()):
            arms[arm]["pm_restore"] = pm
        if arm in FRESH_HOST_ARMS:
            wiped = [r["wipe"] for r in good if r.get("wipe")]
            arms[arm]["wipe_receipts"] = {
                "n": len(wiped), "files": [w["file"] for w in wiped],
                "venv_resident_bytes_before": [w["resident_before"].get("/opt/venv") for w in wiped],
                "venv_resident_bytes_after": [w["resident_after"].get("/opt/venv") for w in wiped],
                "drop_caches_3": bool(wiped) and all(w["drop_caches_3"] is True for w in wiped),
                "pod_state_files_before": [w["pod_state_files_before"] for w in wiped]}
        pre = [r["prespawn"] for r in good if r.get("prespawn")]
        if pre:
            arms[arm]["t0_launch"] = {"ready_from_launch_median": med(p["ready_from_launch_s"] for p in pre),
                                      "old_engine_exit_median": med(p["old_engine_exit_s"] for p in pre)}
    prov = provenance()
    summary = {"cell": "gpu-jax", "hardware": "H100 80GB (a3-highgpu-1g, spot)", "tp": 1, "hosts": 1,
               "engine": {"max_model_len": 8192, "max_num_seqs": 512, "max_num_batched_tokens": 2048,
                          "extra": notes.get("_engine_extra", "")},
               "versions": {"vllm-tpu": "0.32.0", "tpu-inference": "0.32.0", "jax": "0.11.2",
                            "jax-cuda13-plugin": "0.11.2", "jax-triton": "0.4.1", "triton": "3.8.0",
                            "torch": "2.10.0+cpu", "ray": "2.56.1", "flax": "0.12.10"},
               "image_or_venv": prov["image"], "lock_sha256": prov["lock_sha256"], "code_commit": commit,
               "t0": notes.get("_t0", ""), "arms": arms, "note": notes.get("_note", "")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({a: (v["status"], v["n"], v["median"]) for a, v in arms.items()}))


def loo_check(arm, rec):
    """Did the run's mechanisms match its arm? Every PM mechanism must be on except the arm's removed one, which must
    be off (its stand-in observed). From the job log (parse_logs: mechanism)."""
    m = rec.get("mechanism") or {}
    cfg = m.get("pm_config") or {}
    env = rec.get("job_env") or {}
    gap = m.get("engine_start_minus_download_end_s")
    state_cc = (env.get("PM_STATE_URI") or "").rstrip("/") + "/cc.tar"
    restore = (m.get("restore_source"), m.get("restore_threads"))
    no_snap = arm in ("loo-snapshot", "loo-snapshot-cache0")
    stock_cache = arm in ("loo-cache0", "loo-snapshot-cache0")
    checks = {
        "zygote": bool(m.get("zygote_started") and m.get("zygote_child")) == (arm != "loo-zygote"),
        "snapshot": (restore == (None, None) and bool(m.get("runai_streamer")) and m.get("fetch_snap") is False
                     if no_snap else restore[0] is not None and not m.get("runai_streamer")
                     and bool(m.get("fetch_snap_bytes"))),
        "read_threads": (restore == (None, None) if no_snap
                         else restore == (("mmap", 4) if arm == "loo-read-threads" else ("read", 32))),
        "cache0": (cfg.get("cache_min_compile_s") == "unset" and cfg.get("cache_dir") == "unset"
                   and m.get("fetch_cc_source") == env.get("CC_URI") if stock_cache
                   else cfg.get("cache_min_compile_s") == "0" and m.get("fetch_cc_source") == state_cc),
        "overlap": gap is not None and ((gap >= 0) if arm == "loo-overlap" else (gap < 0)),
        "kv_bytes_pinned": cfg.get("kv_bytes") == str(env.get("PM_KV_BYTES"))
                           and (rec.get("values") or {}).get("kv_tokens") == 12800,
    }
    evidence = {"restore": None if restore == (None, None) else f"{restore[0]}+{restore[1]}",
                "runai_streamer": m.get("runai_streamer"), "zygote": bool(m.get("zygote_child")),
                "cache_hits": m.get("cache_hits"), "cache_misses": m.get("cache_misses"),
                "cache_requests": m.get("cache_requests"),
                "cache_by_phase": {k: [v["hits"], v["requests"]] for k, v in (m.get("jax_events") or {}).items()},
                "cc_source": m.get("fetch_cc_source"), "cc_bytes": m.get("fetch_cc_bytes"),
                "snap_bytes": m.get("fetch_snap_bytes"),
                "engine_start_minus_download_end_s": None if gap is None else round(gap, 2),
                "cache_min_compile_s": cfg.get("cache_min_compile_s"),
                "kv_tokens": (rec.get("values") or {}).get("kv_tokens"),
                "load_torch_threads": (rec.get("values") or {}).get("load_torch_threads"),
                "pm_restore_total_s": (rec.get("values") or {}).get("pm_restore_total_s"),
                "weights_s": (rec.get("values") or {}).get("weights_s")}
    return {"ok": all(checks.values()), "checks": checks, **evidence}


def cmd_loo_summary(args):
    """loo_summary.json: per arm the counted runs (ok, greedy pass, mechanism check pass), medians, phases, deltas vs
    the in-session full PM arm (pm-gcs), and every failed or contradicted attempt."""
    by_arm = load_runs(args.dirs)
    matrix = json.loads((ROOT / args.matrix).read_text())
    notes = json.loads(Path(args.notes).read_text()) if args.notes else {}
    phase_keys = ("download", "spawn", "init_device", "load_model", "kv", "compile", "tail")
    arms, failed, commits = {}, [], set()
    for arm in LOO_ARMS:
        good = []
        for r in by_arm.get(arm, []):
            check = loo_check(arm, r) if "mechanism" in r else {"ok": False, "checks": {}}
            r["_check"] = check
            commits.add(r.get("gpu_jax_commit"))
            if r.get("ok") and check["ok"]:
                good.append(r)
            else:
                failed.append({"arm": arm, "run_id": r.get("run_id"), "dir": r["_dir"],
                               "job_status": r.get("job_status"), "error": r.get("error"),
                               "check_ok": r.get("check_ok"), "ready_s": r.get("ready_s"),
                               "mechanism_check": check, "cause": notes.get("_failed", {}).get(r.get("run_id"))})
        phases = {k: med(r["segments"][k] for r in good) for k in phase_keys} if good else None
        pm = {k: med(r["values"].get(k) for r in good) for k in
              ("pm_restore_total_s", "pm_restore_wait_s", "pm_restore_place_s", "weights_s")} if good else None
        removed, fallback = LOO.get(arm, (None, None))
        arms[arm] = {"removed": removed, "fallback": fallback, "status": "ok" if len(good) >= 3 else
                     ("partial" if good else "dnf"), "n": len(good),
                     "median": med(r["ready_s"] for r in good), "runs": [round(r["ready_s"], 1) for r in good],
                     "run_ids": [r["run_id"] for r in good], "samples": [r["sample"] for r in good],
                     "phases": phases, "load": pm, "greedy_checks_passed": all(r["check_ok"] for r in good),
                     "mechanism_check": [{"run_id": r["run_id"], **r["_check"]} for r in good],
                     "wipe_receipts": [r["wipe"]["file"] for r in good if r.get("wipe")],
                     "nodes": sorted({r.get("node") for r in good}, key=str), "note": notes.get(arm, "")}
    full = arms["pm-gcs"]
    for arm, a in arms.items():
        a["delta_vs_full_s"] = (round(a["median"] - full["median"], 1)
                                if a["median"] is not None and full["median"] is not None else None)
        a["phase_deltas_vs_full_s"] = ({k: round(a["phases"][k] - full["phases"][k], 1) for k in phase_keys
                                        if a["phases"][k] is not None and full["phases"][k] is not None}
                                       if a["phases"] and full["phases"] else None)
    mref = matrix["arms"]["pm-gcs"]["median"]
    summary = {"cell": "gpu-jax", "hardware": notes.get("_hardware", "H100 80GB (a3-highgpu-1g)"), "tp": 1, "hosts": 1,
               "code_commit": sorted(c for c in commits if c), "matrix_ref": {
                   "summary": args.matrix, "code_commit": matrix["code_commit"], "pm-gcs": mref,
                   "stock-runai-cc": matrix["arms"]["stock-runai-cc"]["median"]},
               "t0": "MARK download_start (run_arm.sh): first byte of delivery",
               "ready_def": "MARK check_done: /health OK and the greedy check (chat, temperature 0, 32 tokens) returned "
                            f"\"{EXPECTED_CHECK}\"",
               "baseline_arm": "pm-gcs",
               "in_session_full_vs_matrix": {"in_session": full["median"], "matrix": mref,
                                             "diff_pct": round(100 * (full["median"] - mref) / mref, 1)
                                             if full["median"] else None},
               "engine": {"max_model_len": 8192, "max_num_seqs": 512, "max_num_batched_tokens": 2048,
                          "extra": notes.get("_engine_extra", "")},
               "image_or_venv": IMAGE, "lock_sha256": LOCK_SHA, "dirs": sorted({os.path.relpath(Path(d).resolve(), ROOT)
                                                                                for d in args.dirs}),
               "arms": arms, "failed_attempts": failed + notes.get("_excluded", []), "note": notes.get("_note", "")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")
    print("| arm | removed | fallback | n | median (s) | runs (s) | Δ vs in-session full (s) | key phase deltas (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for arm, a in arms.items():
        deltas = ", ".join(f"{k} {v:+.1f}" for k, v in (a["phase_deltas_vs_full_s"] or {}).items() if abs(v) >= 1.0)
        print(f"| {arm} | {a['removed'] or '-'} | {a['fallback'] or '-'} | {a['n']} | {fmt(a['median'])} | "
              f"{', '.join(f'{x:.1f}' for x in a['runs'])} | "
              f"{'-' if arm == 'pm-gcs' else fmt(a['delta_vs_full_s'])} | {deltas or '-'} |")
    print(json.dumps(summary["in_session_full_vs_matrix"]), f"failed_attempts={len(failed)}")



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("configmap")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pool", default="gpu-pool-spot,gpu-pool-flex")
    common.add_argument("--zone", help="fresh nodes only in this zone (e.g. asia-southeast1-c)")
    common.add_argument("--keep", action="store_true", help="leave pods and nodes up")
    common.add_argument("--node", help="node reuse: pin every pod to this held node (never retired), wipe before "
                                       "each fresh-host pod, units one after another")
    common.add_argument("--timeout", type=float, default=3600)
    common.add_argument("--out", required=True)
    run = sub.add_parser("run", parents=[common])
    run.add_argument("--units", required=True, help=",".join(UNITS))
    run.add_argument("--samples", type=int, default=1)
    run.add_argument("--first-sample", type=int, default=0)
    run.add_argument("--nodes", type=int, default=3, help="max concurrent GPU nodes of this driver (fresh nodes)")
    run.add_argument("--peer-node", help="node reuse: second held node for the pm-peer target (else no pm-peer)")
    run.add_argument("--dry-run", action="store_true", help="print the plan, validate the specs server-side")
    run.add_argument("--stagger", type=float, default=10.0, help="seconds between unit starts")
    run.add_argument("--cc", help="gs:// tarball of a peer's $HOME caches for the stock *-cc arms")
    run.add_argument("--export", help="export stock-ckpt sample 0's caches to this gs:// tarball")
    run.add_argument("--pm-state", help="gs://.../pm-state/<seed> (scripts/gpu_jax_autoscale.py seed)")
    run.add_argument("--kv-bytes", help="KV bytes the seed boot measured (seed.json kv_bytes)")
    seed = sub.add_parser("seed", parents=[common])
    seed.add_argument("--dest", required=True)
    seed.add_argument("--pod", help="only upload the state of this kept seed pod")
    job = sub.add_parser("job")
    job.add_argument("--pod", required=True)
    job.add_argument("--arm", required=True)
    job.add_argument("--sample", type=int, default=0)
    job.add_argument("--env", nargs="*", default=[])
    job.add_argument("--keep-engine", action="store_true")
    job.add_argument("--timeout", type=float, default=3600)
    job.add_argument("--out", required=True)
    report = sub.add_parser("report")
    report.add_argument("dirs", nargs="+")
    summ = sub.add_parser("summary")
    summ.add_argument("dirs", nargs="+")
    summ.add_argument("--out", required=True)
    summ.add_argument("--notes", help="JSON: per-arm notes, _na, _t0, _note, _engine_extra")
    summ.add_argument("--commit")
    loo = sub.add_parser("loo-summary")
    loo.add_argument("dirs", nargs="+")
    loo.add_argument("--out", required=True)
    loo.add_argument("--matrix", default="results/gpu-jax/matrix-bfb02b9/matrix_summary.json")
    loo.add_argument("--notes", help="JSON: per-arm notes, _note, _hardware, _engine_extra, _failed {run_id: cause}, "
                                     "_excluded [attempts outside DIRS, with cause]")
    args = parser.parse_args()
    if args.cmd == "configmap":
        configmap()
    elif args.cmd == "run":
        sys.exit(cmd_run(args))
    elif args.cmd == "seed":
        cmd_seed(args)
    elif args.cmd == "job":
        cmd_job(args)
    elif args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "loo-summary":
        cmd_loo_summary(args)
    else:
        cmd_summary(args)


if __name__ == "__main__":
    main()
