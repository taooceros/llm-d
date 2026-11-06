#!/usr/bin/env python3
"""TPU e2e autoscale on NEW capacity: scale-up trigger -> ready, greedy-checked TP8 instance.

One lane = one 2-host ct5lp-hightpu-4t 2x4 node pool (min 0, max 2: one slice) + one private KubeRay
RayCluster in namespace tpu-e2e whose CPU head lives on the default pool and whose single TPU worker group
(numOfHosts 2) is pinned to that pool with replicas 0.  The scale-up trigger is the patch replicas 0 -> 1:
KubeRay creates the 2 worker pods, they stay Pending, the cluster autoscaler provisions both VMs of the
slice, the pods pull the serving image and `ray start` joins the private head.  The engine then boots on
those 2 fresh hosts with the same code the sibling TP8 cells use on the shared slice.

    python3 scripts/tpu_e2e.py up --lane od            # namespace + RayCluster (no TPU pods)
    python3 scripts/tpu_e2e.py boot-test --lane od      # trigger, GKE stages, Ray join, release, reset
    python3 scripts/tpu_e2e.py down --lane od           # delete the RayCluster

The GKE stage bookkeeping (VM created, node registered/Ready, TPU allocatable, pod scheduled, image pulled,
container started) follows scripts/gke_autoscale_probe.py @ research/autoscale (campaign-5962c75).
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

CLUSTER = "hongtaozhang-llm-gke-us-west1"
ZONE = "us-west1-c"
NS = "tpu-e2e"
IMAGE = "vllm/vllm-tpu:v0.26.0"  # the shared RayCluster's image (worker + head)
RAY_VERSION = "2.56.0"           # the shared RayCluster's spec.rayVersion
BUCKET = "hongtaozhang-tpu-model-cache-us-west1"
KUBECONFIG = os.path.expanduser("~/.kube/config")  # user creds: the hetero-agent SA is namespaced elsewhere
LANES = {
    "od": {"pool": "e2e-v5e-2x4-od", "port": 8266, "spot": False},
    "spot": {"pool": "e2e-v5e-2x4-spot", "port": 8267, "spot": True},
}
HOSTS, CHIPS = 2, 4
TOPOLOGY = "2x4"
MACHINE = "ct5lp-hightpu-4t"
POLL_S = 3.0


def rc_name(lane: str) -> str:
    return f"e2e-{lane}"


def log(msg: str) -> None:
    print(f"{dt.datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def sh(cmd: list[str], check: bool = True, input_text: str | None = None, timeout: float | None = None) -> str:
    env = {**os.environ, "KUBECONFIG": KUBECONFIG}
    proc = subprocess.run(cmd, capture_output=True, text=True, input=input_text, env=env, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")
    return proc.stdout


def kjson(*args: str, ns: str | None = NS) -> dict:
    pre = ["kubectl"] + (["-n", ns] if ns else [])
    return json.loads(sh([*pre, *args, "-o", "json"]))


def ts(value: str | None) -> float | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


# ---------------------------------------------------------------- manifests

GCS_VOLUME = {"name": "gcs-model-weights",
              "csi": {"driver": "gcsfuse.csi.storage.gke.io",
                      "volumeAttributes": {"bucketName": BUCKET, "mountOptions": "implicit-dirs"}}}
MOUNTS = [{"name": "gcs-model-weights", "mountPath": "/models"}]
HOST_IP = {"valueFrom": {"fieldRef": {"fieldPath": "status.hostIP"}}}

# Physical slice geometry at pod start.  The hetero runtime resolves layouts from OBSERVED chip coordinates
# (hetero/slice_inventory.py observe_slice) and refuses a host whose runtime chips differ from the inventory.
# On a fresh slice the physical position of each VM is not the webhook's TPU_WORKER_ID (which pod lands on
# which VM is up to the scheduler), and it is not in any node label/annotation or GCE metadata.  So each
# worker pod runs observe_slice's own JAX probe once, beside `ray start`: jax.distributed.initialize() over
# both hosts (GKE/KubeRay TPU env), local chip coords -> /tmp/e2e_geometry.json, then exits (chips free).
GEOMETRY_PROBE = r"""
import json, os, socket, time
t = time.time()
import jax
t_import = time.time()
jax.distributed.initialize()
local = jax.local_devices()
row = {"hostname": socket.gethostname(), "node_ip": os.environ.get("NODE_IP"), "t_start": t, "t_import": t_import,
       "t_end": time.time(), "process_index": int(jax.process_index()), "process_count": int(jax.process_count()),
       "global_device_count": int(jax.device_count()), "local_device_count": int(jax.local_device_count()),
       "chip_coords": [list(int(v) for v in d.coords) for d in local], "device_ids": [int(d.id) for d in local],
       "tpu_worker_id": os.environ.get("TPU_WORKER_ID"), "tpu_name": os.environ.get("TPU_NAME"),
       "jax_version": jax.__version__}
tmp = "/tmp/e2e_geometry.json.tmp"
open(tmp, "w").write(json.dumps(row))
os.replace(tmp, "/tmp/e2e_geometry.json")
print(json.dumps(row), flush=True)
jax.distributed.shutdown()
"""

def raycluster(lane: str) -> dict:
    spec = LANES[lane]
    tolerations = [{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}]
    if spec["spot"]:
        tolerations.append({"key": "cloud.google.com/gke-spot", "operator": "Exists", "effect": "NoSchedule"})
    head = {
        "rayStartParams": {"dashboard-host": "0.0.0.0", "num-gpus": "0"},
        "template": {
            "metadata": {"annotations": {"gke-gcsfuse/volumes": "true"},
                         "labels": {"app.kubernetes.io/name": "kuberay", "tpu-e2e/lane": lane}},
            "spec": {
                "nodeSelector": {"cloud.google.com/gke-nodepool": "default-pool"},
                "containers": [{
                    "name": "ray-head", "image": IMAGE, "imagePullPolicy": "IfNotPresent",
                    "command": ["/bin/bash", "-lc", "--"],
                    "args": ['ulimit -n 65536; eval "$KUBERAY_GEN_RAY_START_CMD"'],
                    "env": [{"name": "VLLM_HOST_IP", **HOST_IP}, {"name": "HF_HOME", "value": "/.cache/huggingface"}],
                    "ports": [{"containerPort": 6379, "name": "gcs-server"}, {"containerPort": 8265, "name": "dashboard"},
                              {"containerPort": 10001, "name": "client"}],
                    "resources": {"requests": {"cpu": "3", "memory": "12Gi"}, "limits": {"cpu": "8", "memory": "16Gi"}},
                    "volumeMounts": MOUNTS,
                }],
                "volumes": [GCS_VOLUME],
            },
        },
    }
    worker = {
        "groupName": "tpu", "replicas": 0, "minReplicas": 0, "maxReplicas": 1, "numOfHosts": HOSTS,
        "rayStartParams": {},
        "template": {
            "metadata": {"annotations": {"gke-gcsfuse/volumes": "true"},
                         "labels": {"app.kubernetes.io/name": "kuberay", "tpu-e2e/lane": lane}},
            "spec": {
                # As the shared worker group: host networking (multi-host libtpu needs it), same env.
                "hostNetwork": True, "dnsPolicy": "ClusterFirstWithHostNet",
                "nodeSelector": {"cloud.google.com/gke-nodepool": spec["pool"],
                                 "cloud.google.com/gke-tpu-accelerator": "tpu-v5-lite-podslice",
                                 "cloud.google.com/gke-tpu-topology": TOPOLOGY},
                "tolerations": tolerations,
                "containers": [{
                    "name": "ray-worker", "image": IMAGE, "imagePullPolicy": "IfNotPresent",
                    # The pod-start geometry probe runs beside `ray start` (see GEOMETRY_PROBE).
                    "command": ["/bin/bash", "-lc", "--"],
                    # e2e_geometry.rc appears only once the probe process has exited (TPU chips free again).
                    "args": ['(python3 -u -c "$E2E_GEOMETRY_PROBE" > /tmp/e2e_geometry.log 2>&1; '
                             'echo "$? $(date +%s.%N)" > /tmp/e2e_geometry.rc) & '
                             'ulimit -n 65536; eval "$KUBERAY_GEN_RAY_START_CMD"'],
                    "env": [{"name": "E2E_GEOMETRY_PROBE", "value": GEOMETRY_PROBE},
                            {"name": "TPU_MULTIHOST_BACKEND", "value": "ray"},
                            {"name": "VLLM_HOST_IP", **HOST_IP}, {"name": "NODE_IP", **HOST_IP},
                            {"name": "VBAR_CONTROL_SERVICE_URL", "value": "$(NODE_IP):8353"},
                            {"name": "JAX_PLATFORMS", "value": "tpu,cpu"},
                            {"name": "ENABLE_PJRT_COMPATIBILITY", "value": "true"},
                            {"name": "HF_HOME", "value": "/.cache/huggingface"}],
                    "ports": [{"containerPort": 8080, "name": "metrics"}, {"containerPort": 8000, "name": "http"}],
                    "resources": {"requests": {"google.com/tpu": str(CHIPS)}, "limits": {"google.com/tpu": str(CHIPS)}},
                    "volumeMounts": MOUNTS,
                }],
                "volumes": [GCS_VOLUME],
            },
        },
    }
    return {"apiVersion": "ray.io/v1", "kind": "RayCluster",
            "metadata": {"name": rc_name(lane), "namespace": NS, "labels": {"tpu-e2e/lane": lane},
                         # Containers run their own command; it must exec KubeRay's generated `ray start`.
                         "annotations": {"ray.io/overwrite-container-cmd": "true"}},
            "spec": {"rayVersion": RAY_VERSION, "headGroupSpec": head, "workerGroupSpecs": [worker]}}


def head_pod(lane: str) -> str:
    pods = kjson("get", "pods", "-l", f"ray.io/cluster={rc_name(lane)},ray.io/node-type=head")["items"]
    running = [p["metadata"]["name"] for p in pods if p["status"].get("phase") == "Running"]
    if not running:
        raise RuntimeError(f"no running head pod for {rc_name(lane)}")
    return running[0]


def up(lane: str) -> None:
    sh(["kubectl", "create", "namespace", NS], check=False)
    sh(["kubectl", "apply", "-f", "-"], input_text=json.dumps(raycluster(lane)))
    for _ in range(120):
        try:
            log(f"head pod {head_pod(lane)} running")
            return
        except RuntimeError:
            time.sleep(5)
    raise RuntimeError("head pod not running after 600 s")


def down(lane: str) -> None:
    sh(["kubectl", "-n", NS, "delete", "raycluster", rc_name(lane), "--wait=true"], check=False)


# ---------------------------------------------------------------- pools

def pool_cmd(lane: str) -> list[str]:
    cmd = ["gcloud", "container", "node-pools", "create", LANES[lane]["pool"], "--cluster", CLUSTER, "--zone", ZONE,
           "--node-locations", ZONE, "--machine-type", MACHINE, "--tpu-topology", TOPOLOGY,
           "--num-nodes", "0", "--enable-autoscaling", "--min-nodes", "0", "--max-nodes", str(HOSTS),
           "--disk-type", "pd-balanced", "--disk-size", "100", "--workload-metadata", "GKE_METADATA",
           "--node-labels", "hetero-probe=tpu-e2e", "--quiet"]
    if LANES[lane]["spot"]:
        cmd.append("--spot")
    return cmd


def cluster_op(cmd: list[str], attempts: int = 60) -> None:
    """Node-pool operation, retried while GKE reports another operation in progress."""
    for attempt in range(attempts):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        err = proc.stderr.lower()
        if ("operation" in err or "try again" in err) and attempt + 1 < attempts:
            time.sleep(15)
            continue
        raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr.strip()[-600:]}")


def pool_exists(lane: str) -> bool:
    names = sh(["gcloud", "container", "node-pools", "list", "--cluster", CLUSTER, "--zone", ZONE,
                "--format=value(name)"]).split()
    return LANES[lane]["pool"] in names


def pool_nodes(lane: str) -> list[dict]:
    return kjson("get", "nodes", "-l", f"cloud.google.com/gke-nodepool={LANES[lane]['pool']}", ns=None)["items"]


def reset_pool(lane: str) -> float:
    """Multi-host TPU pools reject manual resize (501), so a fresh slice = delete + recreate the pool."""
    t0 = time.time()
    if pool_exists(lane):
        cluster_op(["gcloud", "container", "node-pools", "delete", LANES[lane]["pool"], "--cluster", CLUSTER,
                    "--zone", ZONE, "--quiet"])
    cluster_op(pool_cmd(lane))
    return round(time.time() - t0, 1)


# ---------------------------------------------------------------- trigger, GKE stages, Ray join

def set_replicas(lane: str, n: int) -> None:
    patch = [{"op": "test", "path": "/spec/workerGroupSpecs/0/groupName", "value": "tpu"},
             {"op": "replace", "path": "/spec/workerGroupSpecs/0/replicas", "value": n}]
    sh(["kubectl", "-n", NS, "patch", "raycluster", rc_name(lane), "--type=json", "-p", json.dumps(patch)])


def worker_pods(lane: str) -> list[dict]:
    return kjson("get", "pods", "-l", f"ray.io/cluster={rc_name(lane)},ray.io/node-type=worker")["items"]


def ray_tpu_nodes(lane: str) -> list[dict]:
    """Alive Ray nodes with TPU resources, via the private head's dashboard (port-forwarded)."""
    url = f"http://127.0.0.1:{LANES[lane]['port']}/api/v0/nodes?limit=100&detail=1"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.load(resp)
    rows = data["data"]["result"]["result"]
    return [r for r in rows if r.get("state") == "ALIVE" and float((r.get("resources_total") or {}).get("TPU", 0)) > 0]


class GkeWatch:
    """Polls the pool's nodes and the lane's worker pods from T0 on (first-seen times that the API objects
    do not keep: TPU allocatable, Ray node alive)."""

    def __init__(self, lane: str, t0: float):
        self.lane, self.t0 = lane, t0
        self.seen_nodes: dict[str, float] = {}
        self.tpu_alloc: dict[str, float] = {}
        self.ray_alive: dict[str, float] = {}
        self.stop = threading.Event()
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                for node in pool_nodes(self.lane):
                    name = node["metadata"]["name"]
                    self.seen_nodes.setdefault(name, time.time())
                    if int(node["status"].get("allocatable", {}).get("google.com/tpu", "0")) > 0:
                        self.tpu_alloc.setdefault(name, time.time())
            except Exception as exc:  # noqa: BLE001 - keep polling, record
                self.errors.append(f"nodes: {exc!r}"[:300])
            try:
                for r in ray_tpu_nodes(self.lane):
                    self.ray_alive.setdefault(r["node_ip"], time.time())
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"ray: {exc!r}"[:300])
            self.stop.wait(POLL_S)

    def wait_ray(self, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if len(self.ray_alive) >= HOSTS:
                return True
            time.sleep(1.0)
        return False

    def finish(self) -> None:
        self.stop.set()
        self.thread.join(timeout=30)


def collect_gke(lane: str, watch: GkeWatch) -> dict:
    """Absolute timestamps of every GKE stage, per node and per pod (critical path = last host)."""
    rec: dict = {"pods": [], "nodes": [], "events": []}
    pods = worker_pods(lane)
    for pod in pods:
        p = {"name": pod["metadata"]["name"], "node": pod["spec"].get("nodeName"),
             "host_ip": pod["status"].get("hostIP"), "created": ts(pod["metadata"]["creationTimestamp"])}
        for c in pod["status"].get("conditions", []):
            if c["status"] == "True":
                p[f"cond_{c['type']}"] = ts(c.get("lastTransitionTime"))
        for cs in pod["status"].get("containerStatuses", []):
            if cs["name"] != "ray-worker":
                continue
            state = cs.get("state", {})
            p["container_started"] = ts((state.get("running") or state.get("terminated") or {}).get("startedAt"))
            p["image_id"] = cs.get("imageID")
        rec["pods"].append(p)
    names = {p["name"] for p in rec["pods"]}
    for e in kjson("get", "events")["items"]:
        if e["involvedObject"].get("name") in names:
            rec["events"].append({
                "t": ts(e.get("eventTime")) or ts(e.get("firstTimestamp")) or ts(e["metadata"].get("creationTimestamp")),
                "t_last": ts(e.get("lastTimestamp")), "obj": e["involvedObject"].get("name"),
                "reason": e.get("reason"), "msg": (e.get("message") or "")[:400], "count": e.get("count")})
    rec["events"].sort(key=lambda e: e["t"] or 0)
    for node in pool_nodes(lane):
        name = node["metadata"]["name"]
        n = {"name": name, "registered": ts(node["metadata"]["creationTimestamp"]),
             "first_seen": watch.seen_nodes.get(name), "tpu_allocatable_seen": watch.tpu_alloc.get(name),
             "internal_ip": next((a["address"] for a in node["status"].get("addresses", [])
                                  if a["type"] == "InternalIP"), None),
             "labels": {k: v for k, v in node["metadata"]["labels"].items()
                        if k.startswith("cloud.google.com/gke-tpu") or k in ("cloud.google.com/gke-spot",
                                                                               "node.kubernetes.io/instance-type")}}
        for c in node["status"].get("conditions", []):
            if c["type"] == "Ready" and c["status"] == "True":
                n["ready"] = ts(c.get("lastTransitionTime"))
        vm = sh(["gcloud", "compute", "instances", "describe", name, "--zone", ZONE,
                 "--format=value(creationTimestamp)"], check=False).strip()
        n["vm_created"] = ts(vm) if vm else None
        rec["nodes"].append(n)
    rec["ray_alive"] = dict(watch.ray_alive)
    rec["watch_errors"] = watch.errors[-5:]
    return rec


def derive_gke(rec: dict, t0: float) -> dict:
    """Seconds from T0 to each stage, critical path = the last host of the slice."""
    def last(values):
        values = [v for v in values if v is not None]
        return round(max(values) - t0, 1) if values else None

    def first(values):
        values = [v for v in values if v is not None]
        return round(min(values) - t0, 1) if values else None

    pulled = [e for e in rec["events"] if e["reason"] == "Pulled"]
    return {
        "triggered": first(e["t"] for e in rec["events"] if e["reason"] == "TriggeredScaleUp"),
        "pods_created": last(p.get("created") for p in rec["pods"]),
        "vm_created": last(n.get("vm_created") for n in rec["nodes"]),
        "node_registered": last(n.get("registered") for n in rec["nodes"]),
        "node_ready": last(n.get("ready") for n in rec["nodes"]),
        "tpu_allocatable": last(n.get("tpu_allocatable_seen") for n in rec["nodes"]),
        "pod_scheduled": last(p.get("cond_PodScheduled") for p in rec["pods"]),
        # The worker image pull (vllm-tpu); the gcsfuse sidecar's own pull is a separate, smaller image.
        "image_pulled": last(e["t_last"] or e["t"] for e in pulled if IMAGE.split(":")[0] in e["msg"]),
        "pull_msgs": [e["msg"] for e in pulled],
        "container_started": last(p.get("container_started") for p in rec["pods"]),
        "ray_joined": last(rec["ray_alive"].values()) if len(rec["ray_alive"]) >= HOSTS else None,
        "failure_reasons": sorted({e["reason"] for e in rec["events"] if e["reason"] in (
            "FailedScaleUp", "NotTriggerScaleUp", "FailedScheduling", "Preempted", "Evicted", "FailedMount")}),
    }


def wait_autoscaler_knows_pool(lane: str, timeout_s: float = 600) -> float:
    """A recreated pool has a new MIG; until the cluster autoscaler lists it as a node group, a pending pod gets
    NotTriggerScaleUp ('max node group size reached') and the trigger is late by up to a scan or two (seen:
    13 s on jax-pm-spot-2).  A long-lived autoscaled pool is always known, so wait for that state before T0."""
    t = time.time()
    url = sh(["gcloud", "container", "node-pools", "describe", LANES[lane]["pool"], "--cluster", CLUSTER, "--zone", ZONE,
              "--format=value(instanceGroupUrls)"]).strip()
    grp = url.rsplit("/", 1)[-1]
    while time.time() - t < timeout_s:
        status = sh(["kubectl", "-n", "kube-system", "get", "configmap", "cluster-autoscaler-status",
                     "-o", "jsonpath={.data.status}"], check=False)
        if f"instanceGroups/{grp}" in status:
            return round(time.time() - t, 1)
        time.sleep(5)
    raise RuntimeError(f"cluster autoscaler does not list node group {grp} after {timeout_s} s")


def trigger(lane: str) -> tuple[float, GkeWatch]:
    if pool_nodes(lane):
        raise RuntimeError(f"pool {LANES[lane]['pool']} is not at 0 nodes; reset it first")
    if worker_pods(lane):
        raise RuntimeError(f"{rc_name(lane)} still has worker pods")
    wait_autoscaler_knows_pool(lane)
    t0 = time.time()
    set_replicas(lane, 1)
    return t0, GkeWatch(lane, t0)


def release(lane: str, timeout_s: float = 600) -> float:
    t = time.time()
    set_replicas(lane, 0)
    while worker_pods(lane) and time.time() - t < timeout_s:
        time.sleep(3)
    return round(time.time() - t, 1)


def boot_test(lane: str, out: Path, ray_timeout_s: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, watch = trigger(lane)
    log(f"[{lane}] T0 trigger; waiting for {HOSTS} Ray TPU nodes")
    ok = watch.wait_ray(ray_timeout_s)
    log(f"[{lane}] ray joined: {ok}")
    rec = {"lane": lane, "t0": t0, "ok": ok}
    try:
        gke = collect_gke(lane, watch)
        rec["gke_raw"] = gke
        rec["gke"] = derive_gke(gke, t0)
        rec["pod_env"] = [sh(["kubectl", "-n", NS, "exec", p["name"], "-c", "ray-worker", "--", "sh", "-c",
                              "env | grep -E '^TPU_|^TORCH_TPU|^RAY_ADDRESS' | sort"], check=False)
                          for p in gke["pods"]]
        rec["ray_nodes"] = ray_tpu_nodes(lane)
    finally:
        watch.finish()
        rec["release_s"] = release(lane)
        out.write_text(json.dumps(rec, indent=1, default=str))
        log(f"[{lane}] gke stages {json.dumps(rec.get('gke'))}")
        rec["reset_s"] = reset_pool(lane)
        out.write_text(json.dumps(rec, indent=1, default=str))
        log(f"[{lane}] pool reset in {rec['reset_s']} s")


# ---------------------------------------------------------------- RCM jobs on the private head

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")


def ray_url(lane: str) -> str:
    return f"http://127.0.0.1:{LANES[lane]['port']}"


def rcm_env(lane: str) -> dict:
    # Own RCM daemon per lane; RAY_DASHBOARD_URL is omp_rcm's default URL, so nothing falls back to the
    # shared cluster's :8265 even where a script has no --ray-url.
    return {**os.environ, "KUBECONFIG": KUBECONFIG, "OMP_RCM_DIR": f"/tmp/omp_rcm-e2e-{lane}",
            "RAY_DASHBOARD_URL": ray_url(lane)}


def rcm(lane: str, script_args: list[str], log_path: Path, cwd: Path = ROOT) -> int:
    """`rcm_exec run <script> ...` on this lane's head; a failed run is aborted so the next can start.
    cwd: the checkout whose rcm_exec/scripts run (the TorchTPU cell's commit for torch arms)."""
    env = rcm_env(lane)
    py = str(Path(cwd) / ".venv/bin/python")
    with log_path.open("a") as fh:
        fh.write(f"\n### {time.time():.3f} [{cwd}] rcm_exec run {' '.join(script_args)}\n")
        fh.flush()
        rc = subprocess.run([py, "rcm_exec", "run", *script_args], cwd=cwd, env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
        if rc != 0:
            subprocess.run([py, "rcm_exec", "resume", "--abort"], cwd=cwd, env=env, stdout=fh,
                           stderr=subprocess.STDOUT)
    if rc != 0:  # keep the failed Ray jobs' driver logs: the private head may be gone by the time anyone looks
        for sid in sorted(set(re.findall(r"(raysubmit_\w+) \[FAILED\]", log_path.read_text()))):
            try:
                with urllib.request.urlopen(f"{ray_url(lane)}/api/jobs/{sid}/logs", timeout=30) as resp:
                    text = json.load(resp).get("logs", "")
                (log_path.parent / f"job-{sid}.log").write_text(text[-400_000:])
            except Exception as exc:  # noqa: BLE001
                log(f"cannot fetch {sid} logs: {exc!r}")
    time.sleep(3)  # rcm_exec: consecutive runs need the daemon to exit (hetero-autoscale-fresh-host skill)
    return rc


def pod_geometry(lane: str, timeout_s: float = 300) -> dict[str, dict]:
    """host IP -> the pod-start geometry probe's row (GEOMETRY_PROBE), waiting until every pod wrote it."""
    deadline = time.time() + timeout_s
    while True:
        rows = {}
        for pod in worker_pods(lane):
            text = sh(["kubectl", "-n", NS, "exec", pod["metadata"]["name"], "-c", "ray-worker", "--", "sh", "-c",
                       "cat /tmp/e2e_geometry.rc && cat /tmp/e2e_geometry.json"], check=False, timeout=60).strip()
            if text:
                rc_line, _, body = text.partition("\n")
                rc, t_exit = rc_line.split()
                if rc != "0":
                    raise RuntimeError(f"geometry probe failed on {pod['metadata']['name']}: rc={rc}")
                rows[pod["status"]["hostIP"]] = {**json.loads(body), "t_exit": float(t_exit)}
        if len(rows) >= HOSTS:
            return rows
        if time.time() > deadline:
            raise RuntimeError(f"geometry probe incomplete after {timeout_s} s: {sorted(rows)}")
        time.sleep(1.0)


def live_inventory(lane: str, template: dict) -> dict:
    """The slice inventory of the fresh hosts, as observe_slice builds it: chip coordinates OBSERVED by the
    pod-start JAX probe (one jax.distributed init over the slice, the same calls as _SliceProbe), Ray node
    ids / IPs from the head.  The slice-level dims must equal the 2x4 template's (one `hetero_ctl inventory`
    on an earlier fresh slice of this pool shape), else this refuses."""
    sys.path.insert(0, str(ROOT))
    from hetero.slice_inventory import _node_generation  # noqa: PLC0415
    rows = ray_tpu_nodes(lane)
    geo = pod_geometry(lane)
    names = {next(a["address"] for a in node["status"]["addresses"] if a["type"] == "InternalIP"):
             node["metadata"]["name"] for node in pool_nodes(lane)}
    hosts = []
    for r in rows:
        g = geo[r["node_ip"]]
        coords = [list(c) for c in g["chip_coords"]]
        box = [[min(c[a] for c in coords), max(c[a] for c in coords)] for a in range(3)]
        hosts.append({"node_ip": r["node_ip"], "ray_node_id": r["node_id"], "hostname": names[r["node_ip"]],
                      "tpu_worker_id": int(g["tpu_worker_id"]), "tpu_name": g.get("tpu_name") or "",
                      "tpu_count": int(float(r["resources_total"]["TPU"])), "chip_coords": coords,
                      "jax_process_index": g["process_index"], "jax_global_device_count": g["global_device_count"],
                      "chip_box": box, "host_grid_position": [box[a][0] // template["host_chip_dims"][a] for a in range(3)]})
    hosts.sort(key=lambda h: tuple(b[0] for b in h["chip_box"])[::-1])
    want = sorted(tuple(c) for h in template["hosts"] for c in h["chip_coords"])
    got = sorted(tuple(c) for h in hosts for c in h["chip_coords"])
    if want != got or {h["jax_global_device_count"] for h in hosts} != {template["total_chips"]}:
        raise RuntimeError(f"observed slice geometry {got} differs from the 2x4 template {want}")
    nodes = [{"ray_node_id": r["node_id"], "node_ip": r["node_ip"], "tpu_count": int(float(r["resources_total"]["TPU"]))}
             for r in rows]
    inv = {k: template[k] for k in ("slice_name", "chip_dims", "host_chip_dims", "host_grid_dims", "total_chips")}
    inv.update(hosts=hosts, ray_node_generation=_node_generation(nodes), observed_at=time.time(),
               ray_session="tpu-e2e-pod-probe", observed_by="pod-start JAX probe (tpu_e2e GEOMETRY_PROBE)",
               probe_rows=geo)
    return inv


class Marks:
    def __init__(self, path: Path):
        self.path, self.rows = path, []

    def __call__(self, event: str, **kw) -> float:
        t = time.time()
        self.rows.append({"event": event, "t": t, **kw})
        with self.path.open("a") as fh:
            fh.write(json.dumps(self.rows[-1]) + "\n")
        log(f"{event} {kw if kw else ''}")
        return t


def gzip_evidence(out: Path, lane: str) -> None:
    """Local disk is ~94% full: gzip every JSON over 100 KB (deploy JSONs are 150-250 MB) and rotate this
    lane's RCM telemetry once it passes 200 MB."""
    for f in out.glob("*.json"):
        if f.stat().st_size > 100_000:
            subprocess.run(["gzip", "-f", str(f)], check=False)
    tele = Path(f"/tmp/omp_rcm-e2e-{lane}/telemetry.jsonl")
    if tele.exists() and tele.stat().st_size > 200_000_000:
        rotated = tele.with_name(f"telemetry.jsonl.{int(time.time())}")
        tele.rename(rotated)
        subprocess.run(["gzip", str(rotated)], check=False)


ACTIVE_DIR = Path("/tmp/tpu-e2e-active")


def active_samples(lane: str) -> list[dict]:
    """The other lane's sample in flight (both lanes read the same GCS bucket), recorded per run."""
    out = []
    for f in ACTIVE_DIR.glob("*.json") if ACTIVE_DIR.exists() else []:
        if f.stem != lane:
            try:
                out.append(json.loads(f.read_text()))
            except (OSError, ValueError):
                pass
    return out


def verify_clean(lane: str) -> dict:
    """After a sample: the pool has 0 nodes, the lane has no worker pods, the private head holds no live placement
    group or running job, and no hetero registry actor is left."""
    res = {"pool_nodes": len(pool_nodes(lane)), "worker_pods": len(worker_pods(lane))}
    try:
        with urllib.request.urlopen(f"{ray_url(lane)}/api/jobs/", timeout=30) as resp:
            jobs = json.load(resp)
        res["running_jobs"] = [j.get("submission_id") for j in jobs if j.get("status") in ("RUNNING", "PENDING")]
        code = ("import ray, json; ray.init(address='auto', logging_level='ERROR');"
                "from ray.util import placement_group_table as t;"
                "print(json.dumps([v['name'] for v in t().values() if v['state'] != 'REMOVED']))")
        res["live_placement_groups"] = json.loads(sh(["kubectl", "-n", NS, "exec", head_pod(lane), "-c", "ray-head",
                                                      "--", "python3", "-c", code], timeout=120).strip().splitlines()[-1])
        res["registry"] = kill_registry(lane)
    except Exception as exc:  # noqa: BLE001
        res["error"] = repr(exc)[:300]
    res["clean"] = (res["pool_nodes"] == 0 and res["worker_pods"] == 0 and not res.get("running_jobs")
                    and not res.get("live_placement_groups") and "error" not in res
                    and res.get("registry") in ("absent", "killed"))
    return res


def collect_node_logs(lane: str, out: Path) -> None:
    """Before the slice goes away: each worker pod's Ray runtime-env setup logs (the per-node cost of building a
    job's runtime env on a new node is inside `spawn`) and the geometry probe's log."""
    for pod in worker_pods(lane):
        name = pod["metadata"]["name"]
        text = sh(["kubectl", "-n", NS, "exec", name, "-c", "ray-worker", "--", "sh", "-c",
                   "cd /tmp/ray/session_latest/logs 2>/dev/null && for f in runtime_env_setup-*.log "
                   "runtime_env_agent.log; do [ -f \"$f\" ] && echo \"##### $f\" && tail -c 200000 \"$f\"; done; "
                   "echo '##### e2e_geometry.log'; cat /tmp/e2e_geometry.log 2>/dev/null"],
                  check=False, timeout=120)
        if text:
            path = out / f"node-logs-{pod['status'].get('hostIP')}.txt"
            path.write_text(text)
            subprocess.run(["gzip", "-f", str(path)], check=False)


class HostWatchdog:
    """Risky boots (Run:ai): every 3 s, MemAvailable and root-fs free space of each worker pod of the lane. Below
    25% of either, stop every running Ray job on the lane's head (the deploy) and record the breach; the engine's
    drain/release and the pool reset then clean up.  Same thresholds as scripts/host_watchdog.sh @ 966399c."""

    CMD = 'awk "/^MemTotal:|^MemAvailable:/{print \\$2}" /proc/meminfo; df -Pk / | awk "NR==2{print \\$2, \\$4}"'

    def __init__(self, lane: str, path: Path):
        self.lane, self.path, self.breach = lane, path, None
        self.min = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        pods = [(p["metadata"]["name"], p["status"].get("hostIP")) for p in worker_pods(self.lane)]
        while not self.stop.is_set() and self.breach is None:
            for name, ip in pods:
                raw = sh(["kubectl", "-n", NS, "exec", name, "-c", "ray-worker", "--", "sh", "-c", self.CMD],
                         check=False, timeout=30).split()
                if len(raw) < 4:
                    continue
                mt, ma, ft, ff = (int(v) for v in raw[:4])
                row = {"t": time.time(), "ip": ip, "mem_total_kb": mt, "mem_available_kb": ma,
                       "root_total_kb": ft, "root_free_kb": ff}
                with self.path.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                m = self.min.setdefault(ip, {"mem_available_kb": ma, "root_free_kb": ff})
                m["mem_available_kb"], m["root_free_kb"] = min(m["mem_available_kb"], ma), min(m["root_free_kb"], ff)
                if ma < 0.25 * mt or ff < 0.25 * ft:
                    self.breach = row
                    try:
                        with urllib.request.urlopen(f"{ray_url(self.lane)}/api/jobs/", timeout=30) as resp:
                            for j in json.load(resp):
                                if j.get("status") in ("RUNNING", "PENDING") and j.get("submission_id"):
                                    req = urllib.request.Request(
                                        f"{ray_url(self.lane)}/api/jobs/{j['submission_id']}/stop", method="POST")
                                    urllib.request.urlopen(req, timeout=30).read()
                                    self.breach.setdefault("stopped", []).append(j["submission_id"])
                    except Exception as exc:  # noqa: BLE001
                        self.breach["stop_error"] = repr(exc)[:300]
                    log(f"WATCHDOG breach {self.breach}")
                    break
            self.stop.wait(3.0)

    def finish(self) -> dict:
        self.stop.set()
        self.thread.join(timeout=60)
        return {"breach": self.breach, "minimum": self.min, "thresholds": "MemAvailable >= 25% and root free >= 25%"}


def run_sample(lane: str, rid: str, out_dir: Path, engine, ray_timeout_s: float = 1800, cfg: dict | None = None) -> dict:
    """One fresh-capacity sample: trigger, GKE stages, Ray join, engine(ctx), release, pool reset."""
    out = out_dir / rid
    out.mkdir(parents=True, exist_ok=True)
    mark = Marks(out / "timeline.jsonl")
    rec: dict = {"rid": rid, "lane": lane, "pool": LANES[lane]["pool"], "status": "error", "config": cfg or {},
                 "code_commit": sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).strip(),
                 "dirty": sh(["git", "-C", str(ROOT), "status", "--porcelain", "--", "hetero", "scripts"]).strip()}
    head = head_pod(lane)
    rec["head_pod"] = head
    rec["cotenants_start"] = active_samples(lane)
    ACTIVE_DIR.mkdir(exist_ok=True)
    (ACTIVE_DIR / f"{lane}.json").write_text(json.dumps({"rid": rid, "lane": lane, "since": time.time()}))
    t0, watch = trigger(lane)
    rec["t0"] = t0
    mark("trigger", lane=lane)
    try:
        if not watch.wait_ray(ray_timeout_s):
            rec["status"] = "dnf"
            rec["error"] = f"Ray TPU nodes not joined within {ray_timeout_s} s"
            return rec
        mark("ray_joined", ips=sorted(watch.ray_alive))
        ctx = {"lane": lane, "rid": rid, "out": out, "mark": mark, "head": head, "t0": t0,
               "log": out / "rcm.log", "cfg": cfg or {}}
        rec["engine"] = engine(ctx)
        rec["status"] = "ok" if rec["engine"].get("ok") else "failed"
    except Exception as exc:  # noqa: BLE001 - record and clean up
        rec["error"] = repr(exc)[:2000]
    finally:
        try:
            gke = collect_gke(lane, watch)
            rec["gke_raw"], rec["gke"] = gke, derive_gke(gke, t0)
        except Exception as exc:  # noqa: BLE001
            rec["gke_error"] = repr(exc)[:500]
        watch.finish()
        collect_node_logs(lane, out)
        rec["marks"] = mark.rows
        (out / "sample.json").write_text(json.dumps(rec, indent=1, default=str))
        rec["release_s"] = release(lane)
        mark("released")
        rec["reset_s"] = reset_pool(lane)
        mark("pool_reset", seconds=rec["reset_s"])
        rec["cleanup"] = verify_clean(lane)
        mark("cleanup_verified", clean=rec["cleanup"]["clean"])
        rec["cotenants_end"] = active_samples(lane)
        (ACTIVE_DIR / f"{lane}.json").unlink(missing_ok=True)
        rec["marks"] = mark.rows
        (out / "sample.json").write_text(json.dumps(rec, indent=1, default=str))
        gzip_evidence(out, lane)
    return rec


def engine_inventory(ctx: dict) -> dict:
    """Calibration: the real JAX slice probe (`hetero_ctl inventory`) on fresh hosts -> template + cost."""
    lane, out, mark = ctx["lane"], ctx["out"], ctx["mark"]
    mark("inventory_start")
    rc = rcm(lane, ["scripts/hetero_ctl.py", "inventory", "--ray-url", ray_url(lane), "--namespace", NS,
                    "--head-pod", ctx["head"], "--output", str(out / "inventory.json"), "--timeout", "900"], ctx["log"])
    mark("inventory_end", rc=rc)
    if rc != 0:
        return {"ok": False, "rc": rc}
    inv = json.loads((out / "inventory.json").read_text())["measurements"]["inventory"]
    (out / "inventory-only.json").write_text(json.dumps(inv, indent=1))
    return {"ok": True, "inventory": inv}


TEMPLATE = ROOT / "results/tpu-e2e/inventory-template-2x4.json"


def write_live_inventory(ctx: dict) -> Path:
    inv = live_inventory(ctx["lane"], json.loads(TEMPLATE.read_text()))
    path = ctx["out"] / "inventory-live.json"
    path.write_text(json.dumps(inv, indent=1))
    ctx["mark"]("inventory_derived")
    return path


def engine_derive_check(ctx: dict) -> dict:
    """Calibration: the template-derived inventory must equal what the real JAX probe observes."""
    derived = json.loads(write_live_inventory(ctx).read_text())
    real = engine_inventory(ctx)
    if not real["ok"]:
        return real
    keys = ("node_ip", "ray_node_id", "hostname", "tpu_worker_id", "chip_coords", "tpu_count")
    obs = real["inventory"]
    same = ([{k: h[k] for k in keys} for h in derived["hosts"]] == [{k: h[k] for k in keys} for h in obs["hosts"]]
            and derived["ray_node_generation"] == obs["ray_node_generation"]
            and all(derived[k] == obs[k] for k in ("chip_dims", "host_chip_dims", "host_grid_dims", "slice_name")))
    return {"ok": same, "derived_equals_observed": same}


# ---------------------------------------------------------------- vLLM-JAX: the hetero runtime (layout S1)

JAX_LAYOUT, JAX_INSTANCE = "S1", "small_a"  # 60271da+: one TP8 instance on a whole 2-host 2x4 slice
JAX_MODEL = "/models/gemma-4-31b"
JAX_SHAPE = ["--max-model-len", "8192", "--max-num-seqs", "512", "--max-num-batched-tokens", "2048"]
JAX_BASE = {"startup_profile": True}
# As TpuJaxTP8's autoscale_settings @ 966399c (partition mode, layout S) and its runai-jax-xla message.
JAX_ARMS = ("ckpt-gcs-jax-xla", "runai-jax-xla", "gcs", "gcs-ov")


def jax_settings(arm: str, scope: str, cfg: dict) -> dict:
    sys.path.insert(0, str(ROOT))
    from hetero.run_scope import scope_checkpoint_dir, scope_roots, scope_xla_dir  # noqa: PLC0415
    r = scope_roots(scope)
    roots = {"weight_snapshot_root": r["weight_snapshot_root"], "exec_bank_dir": r["exec_bank_dir"],
             "compilation_cache_dir": scope_xla_dir(scope)}
    raw = {**JAX_BASE, "weight_snapshot": "off"}
    restore = {**JAX_BASE, "weight_snapshot": "read", "weight_snapshot_verify": False, "fast_graph": True,
               "exec_bank": "read"}
    jax_env = {"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1.0"}  # stock: jax 0.10.2's default threshold
    if arm == "ckpt-gcs-jax-xla":
        return {**raw, "compilation_cache_dir": scope_xla_dir(scope), "worker_env": jax_env,
                "model": scope_checkpoint_dir(scope) + "/" + Path(JAX_MODEL).name}
    if arm == "runai-jax-xla":
        # tokenizer from /models: with tokenizer = the gs:// model vLLM pulls every non-weight object under the
        # prefix (~80 GB of stale .cache objects) to local disk; memory_limit bounds the streamer's buffer.
        return {**raw, "compilation_cache_dir": scope_xla_dir(scope), "model": cfg["ckpt_gcs"],
                "load_format": "runai_streamer", "worker_env": jax_env, "tokenizer": JAX_MODEL,
                "model_loader_extra_config": {"memory_limit": 8589934592}}
    if arm == "gcs":
        return {**restore, **roots}
    if arm == "gcs-ov":
        return {**restore, **roots, "state_gate_timeout_s": 900.0}
    raise ValueError(arm)


def kill_registry(lane: str) -> str:
    """The head's detached hetero registry keeps the last layout digest (host IPs); a fresh slice has new
    IPs, so drop it after a sample (it is recreated by the next deploy, as on a fresh deployment)."""
    code = ("import ray; ray.init(address='auto', logging_level='ERROR');\n"
            "try:\n a=ray.get_actor('omp-hetero-registry', namespace='omp_hetero'); ray.kill(a, no_restart=True); print('killed')\n"
            "except ValueError: print('absent')")
    return sh(["kubectl", "-n", NS, "exec", head_pod(lane), "-c", "ray-head", "--", "python3", "-c", code],
              check=False, timeout=120).strip()[-200:]


def engine_jax(ctx: dict, cfg: dict) -> dict:
    lane, rid, out, mark, logp = ctx["lane"], ctx["rid"], ctx["out"], ctx["mark"], ctx["log"]
    arm = cfg["arm"]
    scope = rid
    target = f"{JAX_LAYOUT}:{JAX_INSTANCE}"
    common = ["--ray-url", ray_url(lane)]
    inv = write_live_inventory(ctx)
    ips = ",".join(h["node_ip"] for h in json.loads(inv.read_text())["hosts"])
    part = ["--hosts", ips, "--instances", JAX_INSTANCE]
    res: dict = {"arm": arm, "scope": scope, "steps": {}}

    def step(name: str, args: list[str]) -> bool:
        mark(f"{name}_start")
        rc = rcm(lane, args, logp)
        mark(f"{name}_end", rc=rc)
        res["steps"][name] = rc
        return rc == 0

    def ok_json(path: Path) -> bool:
        try:
            return json.loads(path.read_text())["measurements"].get("ok") is not False
        except Exception:  # noqa: BLE001
            return False

    try:
        if arm == "ckpt-gcs-jax-xla" and not step("ckpt_fetch", [
                "scripts/startup_state.py", "fetch-checkpoint", *common, "--inventory", str(inv),
                "--gcs", cfg["ckpt_gcs"], "--method", "gcs", "--target", target, "--target-scope", scope,
                "--output", str(out / "ckpt-fetch.json"), "--timeout", "1800"]):
            return {**res, "ok": False}
        # cfg stock_gcs null: development boot without the stock XLA cache (cold compile), never a matrix run.
        if arm in ("ckpt-gcs-jax-xla", "runai-jax-xla") and cfg.get("stock_gcs") and not step("xla_fetch", [
                "scripts/startup_state.py", "fetch", *common, "--inventory", str(inv), "--method", "gcs",
                "--parts", "xla", "--gcs", cfg["stock_gcs"], "--target", target, "--target-scope", scope,
                "--output", str(out / "xla-fetch.json"), "--timeout", "1800"]):
            return {**res, "ok": False}
        if arm == "gcs" and not step("state_fetch", [
                "scripts/startup_state.py", "fetch", *common, "--inventory", str(inv), "--method", "gcs",
                "--parts", "snapshot,bank,xla", "--gcs", cfg["pm_gcs"], "--target", target, "--target-scope", scope,
                "--output", str(out / "state-fetch.json"), "--timeout", "1800"]):
            return {**res, "ok": False}
        delivery = []
        if arm == "gcs-ov":
            argv = subprocess.run(
                [PY, "scripts/startup_state.py", "fetch", "--print-remote-argv", *common, "--inventory", str(inv),
                 "--method", "gcs", "--parts", "snapshot,bank,xla", "--gcs", cfg["pm_gcs"], "--target", target,
                 "--target-scope", scope, "--timeout", "1800"],
                cwd=ROOT, env=rcm_env(lane), capture_output=True, text=True, check=True).stdout.strip()
            delivery = ["--state-delivery", json.dumps({"argv": json.loads(argv), "timeout_s": 1800})]
        settings = {JAX_INSTANCE: jax_settings(arm, scope, cfg)}
        res["engine_settings"] = settings
        dog = HostWatchdog(lane, out / "watchdog.jsonl") if arm.startswith("runai") else None
        try:
            deploy_ok = step("deploy", [
                "scripts/hetero_ctl.py", "deploy", "--layout", JAX_LAYOUT, *common, "--namespace", NS,
                "--head-pod", ctx["head"], "--inventory", str(inv), "--output", str(out / "deploy.json"),
                "--timeout", "5400", "--", "--model", JAX_MODEL, *JAX_SHAPE,
                "--engine-settings", json.dumps(JAX_BASE), "--instance-engine-settings", json.dumps(settings),
                *delivery, "--partition", "--instances", JAX_INSTANCE]) and ok_json(out / "deploy.json")
        finally:
            if dog:
                res["watchdog"] = dog.finish()
        if not deploy_ok:
            return {**res, "ok": False}
        restored = ["--require-restored"] if arm in ("gcs", "gcs-ov") else []
        check_ok = step("check", ["scripts/snapshot_validate.py", "check", "--layout", JAX_LAYOUT, *common, *part,
                                  *restored, "--output", str(out / "check.json")]) and ok_json(out / "check.json")
        val_ok = step("validate", ["scripts/snapshot_validate.py", "validate", "--layout", JAX_LAYOUT, *common, *part,
                                   *restored, "--output", str(out / "validation.json")]) and ok_json(out / "validation.json")
        res["greedy"] = greedy_compare(out / "validation.json", cfg.get("reference")) if val_ok else {"passed": False}
        res["ok"] = bool(check_ok and val_ok and res["greedy"]["passed"])
        return res
    finally:
        # The slice is deleted right after, but the private head's registry and placement groups are not.
        step("drain", ["scripts/hetero_ctl.py", "drain", "--layout", JAX_LAYOUT, *common, "--namespace", NS,
                       "--head-pod", ctx["head"], "--output", str(out / "drain.json"), "--timeout", "900",
                       "--", "--force", "--instances", JAX_INSTANCE])
        step("release", ["scripts/hetero_ctl.py", "release", "--layout", JAX_LAYOUT, *common, "--namespace", NS,
                         "--head-pod", ctx["head"], "--output", str(out / "release.json"), "--timeout", "600",
                         "--", "--instances", JAX_INSTANCE])
        res["registry"] = kill_registry(lane)


def greedy_compare(path: Path, reference: str | None) -> dict:
    """32 W0 prompts x 64 greedy tokens of small_a, token for token vs the reference run's small_a outputs
    (snapshot_validate's own --reference also compares layout id and model_build, which differ by arm)."""
    m = json.loads(path.read_text())["measurements"]
    outs = m["outputs"][JAX_INSTANCE]
    row = {"prompts": len(outs), "tokens": sum(len(o) for o in outs), "manifest_sha256": m.get("manifest_sha256")}
    if not reference:
        return {**row, "passed": False, "error": "no reference"}
    ref = json.loads(Path(reference).read_text())
    ref = ref.get("measurements", ref)
    ref_outs = ref["outputs"][JAX_INSTANCE] if JAX_INSTANCE in ref["outputs"] else next(iter(ref["outputs"].values()))
    same = [a == b for a, b in zip(outs, ref_outs)]
    return {**row, "reference": str(reference), "passed": len(outs) == len(ref_outs) == 32 and all(same)
            and m.get("manifest_sha256") == ref.get("manifest_sha256"), "prompts_equal": sum(same)}


# ---------------------------------------------------------------- vLLM-TorchTPU: the TorchTPU cell's probe scripts
#
# cfg (results/tpu-e2e/torch/cfg.json): torch_repo (checkout of TpuTorchTP8's commit, .venv symlinked),
# venv_gcs (gs:// tarball of /root/torchtpu_load/venv packed after `setup`), host_env {arm: [K=V, ...]} as sent by
# TpuTorchTP8, reference_text (TP8 greedy text), timeout; post_phases {arm: [{phase, host_env, timeout}]}: probe
# phases run on the same fresh slice after the serve (the canonical stock compile-cache seed: pmupload).
# stock-ckpt-cc-canon = stock-ckpt-cc + the canonical chip binding + a stock compile cache seeded under it
# (stock-ckpt-canon-seed: a cold stock boot with the canonical binding, then pmupload of its compile cache).
TORCH_ARMS = ("stock-runai-cc", "stock-ckpt-cc", "pm-gcs", "stock-ckpt-cc-canon", "stock-ckpt-canon-seed")
TORCH_BASE = "/root/torchtpu_load"
VENV_FETCH = r"""
import os, subprocess, sys, time
from google.cloud.storage import Client, transfer_manager
t = time.time()
bucket, _, name = sys.argv[1][len("gs://"):].partition("/")
blob = Client().bucket(bucket).blob(name)
blob.reload()
tmp = "/root/e2e_venv.tar"
transfer_manager.download_chunks_concurrently(blob, tmp, chunk_size=64 << 20, max_workers=32)
t_dl = time.time()
subprocess.run(["tar", "-xf", tmp, "-C", "/"], check=True)
os.remove(tmp)
print("VENV_FETCH", blob.size, round(t_dl - t, 2), round(time.time() - t_dl, 2), flush=True)
"""


def torch_abs(sample_t0: float, hms: float | None) -> float | None:
    """host.log / vLLM time of day (UTC seconds since midnight) -> epoch, on the day closest to the sample's T0."""
    if hms is None:
        return None
    midnight = sample_t0 - sample_t0 % 86400
    return min((midnight + d * 86400 + hms for d in (-1, 0, 1)), key=lambda v: abs(v - sample_t0))


def torch_chip_bind(host_env: list[str]) -> str:
    """TTL_CHIP_BIND of the arm's host env ('' = the stock per-VM binding)."""
    return next((kv.split("=", 1)[1] for kv in reversed(host_env) if kv.startswith("TTL_CHIP_BIND=")), "")


def torch_copy_logs(lane: str, pods: dict, hosts: list[str], logroot: Path, run_id: str) -> dict:
    """The host script's log dir of `run_id` from both pods (host rank r>0 under host<r>/, the cell's layout)."""
    rcs = {}
    for r, ip in enumerate(hosts):
        dst = logroot / run_id if r == 0 else logroot / run_id / f"host{r}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run(["kubectl", "-n", NS, "cp", "-c", "ray-worker", f"{pods[ip]}:{TORCH_BASE}/logs/{run_id}", str(dst)],
                            capture_output=True, text=True, env={**os.environ, "KUBECONFIG": KUBECONFIG})
        rcs[ip] = cp.returncode
    for f in (logroot / run_id).rglob("*"):
        if f.is_file() and f.suffix != ".gz" and f.stat().st_size > 1_000_000:
            subprocess.run(["gzip", "-f", str(f)], check=False)
    return rcs


def torch_chipmap_summary(logroot: Path, cm_rid: str, probe_dir: Path) -> dict:
    """Each host's chipmap.json (torchtpu_chipmap.py) + the probe job's per-host task seconds (TOPOLOGY_RESULT_JSON
    in the job log the probe saved under <probe_dir>/<cm_rid>/)."""
    out: dict = {"hosts": {}}
    for r, d in enumerate((logroot / cm_rid, logroot / cm_rid / "host1")):
        try:
            c = json.loads((d / "chipmap.json").read_text())
        except (OSError, ValueError) as exc:
            out["hosts"][r] = {"error": repr(exc)[:200]}
            continue
        out["hosts"][r] = {k: c.get(k) for k in ("host", "context", "stock_device_id", "canonical", "canonical_device_id",
                                                 "ok", "error")}
        out["hosts"][r]["slice_round_s"] = (c.get("round_slice") or {}).get("seconds")
    rows = []
    for f in sorted((probe_dir / cm_rid).glob("rayjob_chipmap_*.log")):
        rows += [json.loads(m) for m in re.findall(r"TOPOLOGY_RESULT_JSON=(\{[^\n]*\})", f.read_text(errors="replace"))]
    out["task_seconds"] = [r.get("seconds") for r in rows] or None
    return out


def torch_checks(logroot: Path, rid: str, host_env: list[str]) -> dict:
    """Per-run checks of the cell's config B (matrix-CONFIG-B.json what_to_check_per_run), from the workers' logs:
    rank -> device_id (TTL_RANK_DEVICE; the stock map from `Init mesh ... device_id=` when the line is absent), no
    TTL_CHIP_BIND error, sampler bank loaded/built/stock/placement per rank, `stock path` and PLACEMENT MISMATCH
    counts, TTL_WORKER_CPU torch threads, host minima, and the per-program compile seconds (vLLM-TorchTPU's
    `Compiling <program> ...` / `Compilation finished in S`, summed per rank, max over ranks).  `ok` covers only what
    the arm's env turns on."""
    run = logroot / rid
    texts, nfiles = [], {}
    for h, d in ((0, run / "ray_logs"), (1, run / "host1" / "ray_logs")):
        files = sorted(d.glob("worker-*")) if d.is_dir() else []
        nfiles[h] = len(files)
        for f in files:
            texts.append((gzip.open(f, "rt", errors="replace") if f.suffix == ".gz" else f.open(errors="replace")).read())
    t = "\n".join(texts)
    ranks: dict[int, dict] = {}
    programs: dict[str, float] = {}
    for block in texts:  # stock map: vLLM's `parallel_state ... rank=R` + `Init mesh ... device_id=D` of one worker file
        m, n = re.search(r"world_size=\d+ rank=(\d+) local_rank=", block), re.search(r"Init mesh \|.*?device_id=(\d+)", block)
        if m and n:
            ranks.setdefault(int(m.group(1)), {})["init_mesh_device_id"] = int(n.group(1))
        if m:
            per, cur = {}, None
            for name, secs in re.findall(r"\] (?:Compiling ([^\n]+?) with different input shapes|"
                                         r"Compilation finished in ([\d.]+) \[secs\])", block):
                if name:
                    cur = name
                elif cur:
                    per[cur] = per.get(cur, 0.0) + float(secs)
            for k, v in per.items():
                programs[k] = round(max(programs.get(k, 0.0), v), 2)
    for r, chip, dev, coords, mode in re.findall(
            r"TTL_RANK_DEVICE rank=(\d+) local_rank=\d+ chip=(\S+) device_id=(\S+) coords=(\[[^\]]*\]) mode=(\S+)", t):
        ranks.setdefault(int(r), {}).update(chip=chip, device_id=int(dev) if dev.isdigit() else dev, coords=coords, mode=mode)
    bank = {int(r): {"loaded": int(a), "built": int(b), "stock": int(s), "placement": p}
            for r, a, b, s, p in re.findall(r"PM_SAMPLER_BANK rank=(\d+) loaded=(\d+) \([\d.]+s\) built=(\d+) \([\d.]+s\) "
                                            r"stock=(\d+) salt=\S+ placement=(\S+)", t)}
    cpu = {int(r): int(n) for r, n in re.findall(r"TTL_WORKER_CPU rank=(\d+) torch_threads=(\d+)", t)}
    res = {"worker_files": nfiles, "ranks": {r: ranks[r] for r in sorted(ranks)},
           "rank_device": [ranks.get(r, {}).get("device_id", ranks.get(r, {}).get("init_mesh_device_id")) for r in range(8)],
           "chip_bind_errors": len(re.findall(r"ERROR[^\n]*TTL_CHIP_BIND rank=", t)),
           "sampler_bank": bank, "stock_path": t.count("stock path"), "placement_mismatch": t.count("PLACEMENT MISMATCH"),
           "torch_threads": cpu, "preseed": len(set(re.findall(r"PM_SB_PRESEED rank=(\d+) took=", t))),
           "compile_programs": programs}
    for name in ("mem_min_available_kb", "disk_min_available_kb"):
        vals = []
        for d in (run, run / "host1"):
            try:
                vals.append(int((d / name).read_text().split()[0]))
            except (OSError, ValueError, IndexError):
                vals.append(None)
        res[name] = vals
    ok = len(ranks) == 8
    if torch_chip_bind(host_env) == "canonical":
        res["canonical_ok"] = (all(ranks.get(r, {}).get("device_id") == r and ranks[r].get("mode") == "canonical"
                                   for r in range(8)) and res["chip_bind_errors"] == 0)
        ok = ok and res["canonical_ok"]
    extra = " ".join(kv.split("=", 1)[1] for kv in host_env if kv.startswith("TTL_EXTRA_ENV="))
    if "PM_SAMPLER_BANK=1" in extra.split():
        res["bank_ok"] = (len(bank) == 8 and all(b["loaded"] == 118 and b["built"] == 0 and b["stock"] == 0
                                                 and b["placement"] == "match" for b in bank.values())
                          and res["stock_path"] == 0 and res["placement_mismatch"] == 0)
        ok = ok and res["bank_ok"]
    if "TTL_RAY_ZYGOTE=1" in host_env:  # the zygote OMP fix (fc75e5c): workers run with Ray's 1 intra-op thread
        # None = not logged (runs before fc75e5c have no TTL_WORKER_CPU line); a config B run must log 8 ranks
        res["threads_ok"] = (len(cpu) == 8 and set(cpu.values()) == {1}) if cpu else None
        ok = ok and res["threads_ok"] is True
    res["ok"] = bool(ok)
    return res


def engine_torch(ctx: dict, cfg: dict) -> dict:
    lane, rid, out, mark, logp = ctx["lane"], ctx["rid"], ctx["out"], ctx["mark"], ctx["log"]
    arm, repo = cfg["arm"], Path(cfg["torch_repo"]).expanduser()
    inv = json.loads(write_live_inventory(ctx).read_text())
    # Host rank 0 = the host whose chips are y=0..1 (TP rank order of the seed); hosts are sorted by chip origin.
    hosts = [h["node_ip"] for h in inv["hosts"]]
    pods = {p["status"]["hostIP"]: p["metadata"]["name"] for p in worker_pods(lane)}
    res: dict = {"arm": arm, "hosts": hosts, "steps": {}}
    host_env = [*cfg.get("host_env_common", []), *cfg["host_env"][arm]]
    logroot = out / "pod_logs"
    probe_out = str((out / "probe").resolve())  # absolute: the probe runs with cwd = the TorchTPU checkout
    # 1. The venv (stands in for the image layer TorchTPU needs): GCS tarball -> both hosts, in parallel.
    mark("venv_start")
    procs = {ip: subprocess.Popen(["kubectl", "-n", NS, "exec", pods[ip], "-c", "ray-worker", "--", "python3", "-c",
                                   VENV_FETCH, cfg["venv_gcs"]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, env={**os.environ, "KUBECONFIG": KUBECONFIG}) for ip in hosts}
    res["venv"] = {ip: {"rc": p.wait(timeout=1800), "out": p.stdout.read()[-600:]} for ip, p in procs.items()}
    mark("venv_end")
    if any(v["rc"] for v in res["venv"].values()):
        return {**res, "ok": False}
    # 2. The chip map of this fresh pair (TTL_CHIP_BIND arms): real work every new slice must do before the serve, on
    #    the T0 critical path and counted (step wall chipmap_start -> chipmap_end). The cell's two-host Ray-mode probe
    #    (`--phase chipmap`: torch_tpu world over both hosts, slice round only, driver log_to_driver=False).
    cm_rid = f"{rid}-cm" if torch_chip_bind(host_env) else None
    if cm_rid:
        mark("chipmap_start")
        rc = rcm(lane, ["scripts/torchtpu_load_probe.py", "--phase", "chipmap", "--run-id", cm_rid, "--hosts",
                        ",".join(hosts), "--ray-url", ray_url(lane), "--timeout", "900", "--out-dir", probe_out],
                 logp, cwd=repo)
        mark("chipmap_end", rc=rc)
        res["steps"]["chipmap"] = rc
        if rc != 0:
            res["log_copy_chipmap"] = torch_copy_logs(lane, pods, hosts, logroot, cm_rid)
            res["chipmap"] = torch_chipmap_summary(logroot, cm_rid, out / "probe")
            return {**res, "ok": False}
    # 3. The serve phase of the TorchTPU probe on this lane's head (Ray mode over both hosts).
    # TTL_SINCE bounds which /dev/shm caches the host script may treat as this run's (TpuTorchTP8: "<your start>").
    host_env.append("TTL_SINCE=" + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ctx["t0"] - 60)))
    mark("serve_start")
    # Run:ai streams the weights straight into host memory: same host watchdog as the JAX runai arms.
    dog = HostWatchdog(lane, out / "watchdog.jsonl") if arm == "stock-runai-cc" else None
    try:
        rc = rcm(lane, ["scripts/torchtpu_load_probe.py", "--phase", "serve", "--run-id", rid, "--hosts", ",".join(hosts),
                        "--ray-url", ray_url(lane), "--timeout", str(cfg.get("timeout", 3600)),
                        "--out-dir", probe_out, *[a for kv in host_env for a in ("--host-env", kv)]],
                 logp, cwd=repo)
    finally:
        if dog:
            res["watchdog"] = dog.finish()
    mark("serve_end", rc=rc)
    res["steps"]["serve"] = rc
    # 4. Pod logs of both hosts (after the measured part).
    res["log_copy"] = torch_copy_logs(lane, pods, hosts, logroot, rid)
    if cm_rid:
        res["log_copy_chipmap"] = torch_copy_logs(lane, pods, hosts, logroot, cm_rid)
        res["chipmap"] = torch_chipmap_summary(logroot, cm_rid, out / "probe")
    # 5. Phases and greedy text with the cell's own report (scripts/torchtpu_matrix_report.py run_report).
    sys.path.insert(0, str(repo / "scripts"))
    import torchtpu_matrix_report as tmr  # noqa: PLC0415
    rep = tmr.run_report(logroot, rid)
    res["report"] = rep
    host = tmr.read(logroot / rid / "host.log")
    t0_line = next((ln for ln in host.splitlines() if "T0 critical path starts (fresh host" in ln), None)
    m = tmr.TS_HOST.match(t0_line) if t0_line else None
    res["engine_t0"] = torch_abs(ctx["t0"], tmr.secs(m.group(1))) if m else None
    text = rep.get("greedy_text")
    res["greedy"] = {"text": text, "reference": cfg["reference_text"],
                     "passed": text is not None and text.strip() == cfg["reference_text"].strip()}
    res["checks"] = torch_checks(logroot, rid, host_env)
    res["ok"] = bool(rc == 0 and rep.get("ready_s") is not None and res["greedy"]["passed"] and res["checks"]["ok"]
                     and res["engine_t0"] is not None and not (res.get("watchdog") or {}).get("breach"))
    # 6. Phases on the same slice after the measured boot (the canonical stock compile-cache seed). `pmupload-relay`:
    #    the e2e pools' nodes have the default read-only storage scope (hostNetwork pods use the node's credentials;
    #    the cell's pmupload got 403 "Provided scope(s) are not authorized"), so the cell's pack runs on each pod and
    #    the pack is streamed through this workstation (kubectl exec cat | gcloud storage cp -) to <dst>/host<r>/, the
    #    layout pm_artifacts.py upload writes (model.tar, state.tar, files.json).
    for post in (cfg.get("post_phases") or {}).get(arm, []):
        if not res["ok"]:
            break
        prid = f"{rid}-{post['phase']}"
        mark(f"post_{post['phase']}_start")
        if post["phase"] == "pmupload-relay":
            pub = torch_relay_upload(pods, hosts, post["dst"], repo, out / f"{prid}.log")
            prc = 0 if pub["ok"] else 1
            res.setdefault("published", {})[prid] = pub
        else:
            prc = rcm(lane, ["scripts/torchtpu_load_probe.py", "--phase", post["phase"], "--run-id", prid, "--hosts",
                             ",".join(hosts), "--ray-url", ray_url(lane), "--timeout", str(post.get("timeout", 1800)),
                             "--out-dir", probe_out, *[a for kv in post.get("host_env", []) for a in ("--host-env", kv)]],
                      logp, cwd=repo)
            res.setdefault("log_copy_post", {})[prid] = torch_copy_logs(lane, pods, hosts, logroot, prid)
        mark(f"post_{post['phase']}_end", rc=prc)
        res["steps"][f"post_{post['phase']}"] = prc
        res["ok"] = res["ok"] and prc == 0
    return res


def torch_relay_upload(pods: dict, hosts: list[str], dst: str, repo: Path, log_path: Path) -> dict:
    """Pack each host's stock compile cache with the cell's pm_artifacts.py (the pmupload phase's exact pack
    arguments) and publish it to <dst>/host<r>/ through this workstation's gcloud credentials.  A kubectl exec stream
    can end early without an error (torch-stockseed-canon-od-2: host0 state.tar 1.86 MB short), so every file is
    verified against its md5 on the pod (GCS md5_hash) and re-streamed up to 4 times."""
    env = {**os.environ, "KUBECONFIG": KUBECONFIG}
    pack = "/dev/shm/pm_pack"
    res: dict = {"dst": dst, "hosts": {}}

    def one(r: int, ip: str) -> None:
        pod, row = pods[ip], {}
        res["hosts"][r] = row
        t = time.time()
        with log_path.open("a") as fh:
            def run(cmd: list[str], **kw) -> int:
                fh.write(f"### host{r} {' '.join(cmd)[:300]}\n")
                fh.flush()
                return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, **kw).returncode
            rc = run(["kubectl", "-n", NS, "cp", "-c", "ray-worker", str(repo / "scripts/pm_artifacts.py"),
                      f"{pod}:/tmp/e2e_pm_artifacts.py"])
            rc = rc or run(["kubectl", "-n", NS, "exec", pod, "-c", "ray-worker", "--", "sh", "-c",
                            f"rm -rf {pack} && {TORCH_BASE}/venv/bin/python /tmp/e2e_pm_artifacts.py pack "
                            f"--model-dir /dev/shm/torchtpu_load_gemma4_31b --bank-dir {TORCH_BASE}/vllm_cache "
                            f"--xla-dir /dev/shm/torch_tpu_cache --snap-dir /dev/shm/pm_torchtpu/snap --out {pack}"])
            man = json.loads(sh(["kubectl", "-n", NS, "exec", pod, "-c", "ray-worker", "--", "cat",
                                 f"{pack}/files.json"], check=False, timeout=60) or '{"files": []}')
            row["files"] = [{k: v for k, v in f.items() if k != "path"} for f in man["files"]]
            row["pack_s"] = round(time.time() - t, 1)
            for f in man["files"] if rc == 0 else []:
                want = sh(["kubectl", "-n", NS, "exec", pod, "-c", "ray-worker", "--", "md5sum", f["path"]],
                          check=False, timeout=300).split()[:1]
                f["md5"] = want[0] if want else None
                url = f"{dst}/host{r}/{f['name']}"
                for attempt in range(4):
                    cat = subprocess.Popen(["kubectl", "-n", NS, "exec", pod, "-c", "ray-worker", "--", "cat", f["path"]],
                                           stdout=subprocess.PIPE, stderr=fh, env=env)
                    rc = run(["gcloud", "storage", "cp", "-", url], stdin=cat.stdout)
                    cat.stdout.close()
                    rc = rc or cat.wait()
                    got = sh(["gcloud", "storage", "objects", "describe", url, "--format=value(md5_hash,size)"],
                             check=False).split()
                    md5 = base64.b64decode(got[0]).hex() if got else None
                    f.setdefault("attempts", []).append({"rc": rc, "md5": md5, "size": int(got[1]) if len(got) > 1 else None})
                    fh.write(f"### host{r} {f['name']} attempt {attempt}: rc={rc} md5={md5} want={f['md5']}\n")
                    if rc == 0 and f["md5"] and md5 == f["md5"]:
                        break
                    rc = rc or 1
                row.setdefault("verified", {})[f["name"]] = {"md5": f["md5"], "attempts": f["attempts"]}
                if rc:
                    break
            if rc == 0:
                proc = subprocess.run(["gcloud", "storage", "cp", "-", f"{dst}/host{r}/files.json"], env=env,
                                      input=json.dumps({"files": row["files"]}), text=True, capture_output=True)
                rc = proc.returncode
            listed = sh(["gcloud", "storage", "ls", "-l", f"{dst}/host{r}/"], check=False)
            got = {ln.split()[-1].rsplit("/", 1)[-1]: int(ln.split()[0]) for ln in listed.splitlines()
                   if ln.strip() and ln.split()[0].isdigit()}
            row["uploaded"] = got
            row["ok"] = rc == 0 and all(got.get(f["name"]) == f["size"] for f in row["files"]) and bool(row["files"])
            run(["kubectl", "-n", NS, "exec", pod, "-c", "ray-worker", "--", "rm", "-rf", pack, "/tmp/e2e_pm_artifacts.py"])
            row["seconds"] = round(time.time() - t, 1)

    threads = [threading.Thread(target=one, args=(r, ip)) for r, ip in enumerate(hosts)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    res["ok"] = len(res["hosts"]) == len(hosts) and all(h.get("ok") for h in res["hosts"].values())
    return res


ENGINES = {"inventory": engine_inventory, "derive-check": engine_derive_check}
for _arm in JAX_ARMS:
    ENGINES[f"jax:{_arm}"] = (lambda a: lambda ctx: engine_jax(ctx, {**ctx["cfg"], "arm": a}))(_arm)
for _arm in TORCH_ARMS:
    ENGINES[f"torch:{_arm}"] = (lambda a: lambda ctx: engine_torch(ctx, {**ctx["cfg"], "arm": a}))(_arm)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("up", "down", "reset-pool", "boot-test", "release", "manifest", "sample"):
        p = sub.add_parser(name)
        p.add_argument("--lane", choices=list(LANES), required=True)
        if name == "boot-test":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--ray-timeout", type=float, default=1800)
        if name == "sample":
            p.add_argument("--engine", choices=list(ENGINES), required=True)
            p.add_argument("--rid", required=True)
            p.add_argument("--out-dir", type=Path, required=True)
            p.add_argument("--ray-timeout", type=float, default=1800)
            p.add_argument("--config", type=Path, help="JSON: ckpt_gcs, stock_gcs, pm_gcs, reference, ...")
    args = ap.parse_args()
    if args.cmd == "up":
        up(args.lane)
    elif args.cmd == "down":
        down(args.lane)
    elif args.cmd == "reset-pool":
        print(reset_pool(args.lane))
    elif args.cmd == "release":
        print(release(args.lane))
    elif args.cmd == "manifest":
        print(json.dumps(raycluster(args.lane), indent=1))
    elif args.cmd == "boot-test":
        boot_test(args.lane, args.out, args.ray_timeout)
    elif args.cmd == "sample":
        cfg = json.loads(args.config.read_text()) if args.config else {}
        rec = run_sample(args.lane, args.rid, args.out_dir, ENGINES[args.engine], args.ray_timeout, cfg)
        log(f"{args.rid}: {rec['status']} gke={json.dumps({k: v for k, v in (rec.get('gke') or {}).items() if k != 'pull_msgs'})}")
        return 0 if rec["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
