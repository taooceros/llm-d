#!/usr/bin/env python3
"""Autoscale-burst (case study P0): trace-driven burst autoscaling of TP8 Gemma-4 31B instances on new TPU v5e
capacity, stock vLLM-JAX vs TPUFlex scale-up under one reactive policy.

Infrastructure (namespace `asb`, cluster hongtaozhang-llm-gke-us-west1):
  * every instance = one TP8 engine (hetero runtime, layout S1:small_a) on its own 2-host ct5lp-hightpu-4t 2x4
    slice (a multi-host slice fails and is recreated as a unit, so never two instances on one slice), under its
    OWN private KubeRay head (the hetero registry/layout is one per Ray cluster) - one "lane" per slice;
  * bed x: x0 (base, fixed pool) + x1..x3 (standby slices, fixed pools, wiped between runs by recreating the
    worker pods); bed y: y0 (base, fixed pool) + y1..y3 (pools autoscaled 0..2, deleted+recreated between runs:
    new capacity, or provisioned before a static-peak run);
  * Ray heads and the per-bed load generator (router + controller + open-loop client, scripts/asb_loadgen.py)
    run on a dedicated CPU pool `asb-cpu`; the data path client -> router -> engines is in-cluster;
  * this workstation runs the actuators (rcm_exec jobs on the lanes' private heads through dashboard
    port-forwards, as scripts/tpu_e2e.py does) and the run orchestration.

    .venv/bin/python scripts/asb.py up --lanes x0,x1          # namespace, RayClusters, loadgen pods
    .venv/bin/python scripts/asb.py pf                        # port-forward commands (run as long-lived services)
    .venv/bin/python scripts/asb.py seed --lane x0            # TPUFlex seed -> GCS (pm prefix)
    .venv/bin/python scripts/asb.py stock-seed --lane x1      # stock XLA cache -> GCS (stock-xla prefix)
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_e2e as e2e  # noqa: E402  (the e2e harness: manifests, GKE stages, geometry probe, rcm, settings)

CLUSTER, ZONE = e2e.CLUSTER, e2e.ZONE
NS = "asb"
IMAGE = e2e.IMAGE
BUCKET = e2e.BUCKET
CPU_POOL = "asb-cpu"
GCS_ROOT = f"gs://{BUCKET}/omp_asb_20261025"
PM_GCS = f"{GCS_ROOT}/pm"            # TPUFlex seed: snapshot + bank + XLA cache of S1:small_a
STOCK_GCS = f"{GCS_ROOT}/stock-xla"  # stock XLA cache (JAX 1 s threshold) of a ckpt-gcs-jax boot
CKPT_GCS = f"gs://{BUCKET}/gemma-4-31b"
OUT_ROOT = ROOT / "results/autoscale-burst"
REFERENCE = OUT_ROOT / "ref/jax-S-small_a-reference-966399c.json"
TEMPLATE = OUT_ROOT / "inventory-template-2x4.json"
KUBECONFIG = os.path.expanduser("~/.kube/config")

LANES = {
    "x0": {"pool": "asb-x0", "port": 8270, "bed": "x", "kind": "fixed"},
    "x1": {"pool": "asb-x1", "port": 8271, "bed": "x", "kind": "fixed"},
    "x2": {"pool": "asb-x2", "port": 8272, "bed": "x", "kind": "fixed"},
    "x3": {"pool": "asb-x3", "port": 8273, "bed": "x", "kind": "fixed"},
    "y0": {"pool": "asb-y0", "port": 8274, "bed": "y", "kind": "fixed"},
    "y1": {"pool": "asb-y1", "port": 8275, "bed": "y", "kind": "auto"},
    "y2": {"pool": "asb-y2", "port": 8276, "bed": "y", "kind": "auto"},
    "y3": {"pool": "asb-y3", "port": 8277, "bed": "y", "kind": "auto"},
}
for _spec in LANES.values():
    _spec["spot"] = False
BEDS = {"x": {"base": "x0", "lanes": ["x1", "x2", "x3"], "router_port": 9100},
        "y": {"base": "y0", "lanes": ["y1", "y2", "y3"], "router_port": 9101}}

# The e2e harness's helpers read these module globals at call time.
e2e.NS = NS
e2e.LANES = LANES
e2e.KUBECONFIG = KUBECONFIG
e2e.TEMPLATE = TEMPLATE
e2e.rc_name = lambda lane: f"asb-{lane}"
e2e.ACTIVE_DIR = Path("/tmp/asb-active")
_e2e_kjson = e2e.kjson


def _kjson(*args: str, ns: str | None = NS) -> dict:  # tpu_e2e.kjson binds its namespace default at import
    return _e2e_kjson(*args, ns=ns)


e2e.kjson = _kjson


def _rcm_env(lane: str) -> dict:
    return {**os.environ, "KUBECONFIG": KUBECONFIG, "OMP_RCM_DIR": f"/tmp/omp_rcm-asb-{lane}",
            "RAY_DASHBOARD_URL": e2e.ray_url(lane)}


e2e.rcm_env = _rcm_env
log, sh, kjson = e2e.log, e2e.sh, e2e.kjson
PY = str(ROOT / ".venv/bin/python")
JAX_LAYOUT, JAX_INSTANCE, JAX_MODEL, JAX_SHAPE, JAX_BASE = (e2e.JAX_LAYOUT, e2e.JAX_INSTANCE, e2e.JAX_MODEL,
                                                             e2e.JAX_SHAPE, e2e.JAX_BASE)
TARGET = f"{JAX_LAYOUT}:{JAX_INSTANCE}"
# hetero_ctl.py's runtime env, verbatim: the warm-up job must build the SAME working_dir package and pip env on
# a standby host that the deploy job will use (both are cached per node by content hash).
HETERO_CTL_EXCLUDES = [".venv*", ".git", ".jj", "formalization", "skaffold", "__pycache__", "*.pyc", "docs", "results",
                       "experiments", "reports", "**/results/**", "**/reports/**",
                       "data/workloads/sharegpt_large.json", "data/workloads/longdoc_govreport.json",
                       "data/workloads/longctx_govreport.json", "data/workloads/midctx_govreport.json",
                       "data/workloads/translation_ted.json"]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_rev() -> str:
    return sh(["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"]).strip()


def git_dirty() -> str:
    return sh(["git", "-C", str(ROOT), "status", "--porcelain", "--", "hetero", "scripts"]).strip()


# ---------------------------------------------------------------- manifests

TOLERATE_CPU = [{"key": "asb", "operator": "Equal", "value": "only", "effect": "NoSchedule"}]


def raycluster(lane: str) -> dict:
    """The e2e lane's RayCluster (same image, worker env, hostNetwork, pod-start geometry probe, /models fuse),
    with the head on the asb-cpu pool and the TPU worker group pinned to this lane's pool."""
    base = e2e.raycluster(lane)
    head = base["spec"]["headGroupSpec"]["template"]["spec"]
    head["nodeSelector"] = {"cloud.google.com/gke-nodepool": CPU_POOL}
    head["tolerations"] = TOLERATE_CPU
    head["containers"][0]["resources"] = {"requests": {"cpu": "2", "memory": "8Gi"},
                                          "limits": {"cpu": "6", "memory": "14Gi"}}
    for meta in (base["metadata"], base["spec"]["headGroupSpec"]["template"]["metadata"],
                 base["spec"]["workerGroupSpecs"][0]["template"]["metadata"]):
        meta.setdefault("labels", {}).pop("tpu-e2e/lane", None)
        meta["labels"]["asb/lane"] = lane
    worker = base["spec"]["workerGroupSpecs"][0]
    worker["replicas"] = 1 if LANES[lane]["kind"] == "fixed" else 0
    return base


def loadgen_manifests(bed: str) -> list[dict]:
    name = f"asb-loadgen-{bed}"
    labels = {"app": name, "asb/bed": bed}
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": name, "namespace": NS, "labels": labels},
           "spec": {"nodeSelector": {"cloud.google.com/gke-nodepool": CPU_POOL}, "tolerations": TOLERATE_CPU,
                    "restartPolicy": "Always", "terminationGracePeriodSeconds": 5,
                    "containers": [{"name": "loadgen", "image": IMAGE, "imagePullPolicy": "IfNotPresent",
                                    "command": ["/bin/bash", "-c", "mkdir -p /opt/asb /data && sleep infinity"],
                                    "env": [{"name": "PYTHONUNBUFFERED", "value": "1"}],
                                    "ports": [{"containerPort": 9000, "name": "router"}],
                                    "resources": {"requests": {"cpu": "3", "memory": "6Gi"},
                                                  "limits": {"cpu": "6", "memory": "12Gi"}},
                                    "volumeMounts": [{"name": "data", "mountPath": "/data"}]}],
                    "volumes": [{"name": "data", "emptyDir": {}}]}}
    svc = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": f"asb-router-{bed}", "namespace": NS},
           "spec": {"selector": labels, "ports": [{"name": "router", "port": 9000, "targetPort": 9000}]}}
    return [pod, svc]


def router_url_incluster(bed: str) -> str:
    return f"http://asb-router-{bed}.{NS}.svc.cluster.local:9000"


def router_url_local(bed: str) -> str:
    return f"http://127.0.0.1:{BEDS[bed]['router_port']}"


def pool_cmd(lane: str) -> list[str]:
    spec = LANES[lane]
    cmd = ["gcloud", "container", "node-pools", "create", spec["pool"], "--cluster", CLUSTER, "--zone", ZONE,
           "--node-locations", ZONE, "--machine-type", e2e.MACHINE, "--tpu-topology", e2e.TOPOLOGY,
           "--disk-type", "pd-balanced", "--disk-size", "100", "--workload-metadata", "GKE_METADATA",
           "--node-labels", "hetero-probe=asb", "--quiet"]
    if spec["kind"] == "fixed":
        return cmd + ["--num-nodes", str(e2e.HOSTS)]
    return cmd + ["--num-nodes", "0", "--enable-autoscaling", "--min-nodes", "0", "--max-nodes", str(e2e.HOSTS)]


e2e.pool_cmd = pool_cmd


def pool_exists(lane: str) -> bool:
    names = sh(["gcloud", "container", "node-pools", "list", "--cluster", CLUSTER, "--zone", ZONE,
                "--format=value(name)"]).split()
    return LANES[lane]["pool"] in names


# ---------------------------------------------------------------- kubectl helpers

def kexec(pod: str, script: str, container: str | None = None, timeout: float = 120, check: bool = False,
          stdin: str | None = None) -> str:
    cmd = ["kubectl", "-n", NS, "exec", *(["-i"] if stdin is not None else []), pod]
    if container:
        cmd += ["-c", container]
    return sh([*cmd, "--", "sh", "-c", script], check=check, timeout=timeout, input_text=stdin)


def head_pod(lane: str) -> str:
    return e2e.head_pod(lane)


def apply(docs: list[dict]) -> None:
    sh(["kubectl", "apply", "-f", "-"], input_text="\n---\n".join(json.dumps(d) for d in docs))


def up(lanes: list[str], beds: list[str]) -> None:
    sh(["kubectl", "create", "namespace", NS], check=False)
    for lane in lanes:
        apply([raycluster(lane)])
    for bed in beds:
        apply(loadgen_manifests(bed))
    for lane in lanes:
        for _ in range(120):
            try:
                log(f"{lane}: head {head_pod(lane)} running")
                break
            except RuntimeError:
                time.sleep(5)


def pf_commands() -> list[str]:
    rows = []
    for lane, spec in LANES.items():
        rows.append(f"while true; do KUBECONFIG={KUBECONFIG} kubectl port-forward -n {NS} --address 127.0.0.1 "
                    f"service/asb-{lane}-head-svc {spec['port']}:8265; sleep 2; done")
    for bed, spec in BEDS.items():
        rows.append(f"while true; do KUBECONFIG={KUBECONFIG} kubectl port-forward -n {NS} --address 127.0.0.1 "
                    f"service/asb-router-{bed} {spec['router_port']}:9000; sleep 2; done")
    return rows


def stage_token(lane: str) -> dict:
    """GCS writes from the TPU hosts (their node scopes are read-only): stage this workstation's access token on
    the lane's head pod for startup_state.py's driver_token (TOKEN_FILE), valid ~55 min."""
    token = sh(["gcloud", "auth", "print-access-token"]).strip()
    account = sh(["gcloud", "config", "get-value", "account"]).strip()
    project = sh(["gcloud", "config", "get-value", "project"]).strip()
    body = json.dumps({"access_token": token, "project": project, "account": account,
                       "expires_at": time.time() + 3300})
    kexec(head_pod(lane), "umask 077 && cat > /tmp/asb_gcs_token.json", container="ray-head", stdin=body, check=True)
    return {"account": account, "project": project, "staged_at": time.time()}


def unstage_token(lane: str) -> None:
    kexec(head_pod(lane), "rm -f /tmp/asb_gcs_token.json", container="ray-head")


# ---------------------------------------------------------------- lanes

def ray_nodes_alive(lane: str) -> list[dict]:
    try:
        return e2e.ray_tpu_nodes(lane)
    except Exception:  # noqa: BLE001 - the dashboard port-forward may be restarting
        return []


def wait_lane_joined(lane: str, timeout_s: float = 1200) -> float:
    t = time.time()
    while time.time() - t < timeout_s:
        pods = e2e.worker_pods(lane)
        if len(pods) == e2e.HOSTS and len(ray_nodes_alive(lane)) >= e2e.HOSTS:
            return round(time.time() - t, 1)
        time.sleep(3)
    raise RuntimeError(f"{lane}: Ray TPU nodes not joined within {timeout_s} s")


def restart_workers(lane: str, timeout_s: float = 900) -> dict:
    """Fresh-host wipe of a fixed standby slice: delete the worker group's pods (replicas 1 -> 0) and recreate
    them on the same VMs (0 -> 1).  New containers: no /dev/shm state, no /root/.cache (bank, XLA), no Ray
    runtime-env caches, a new gcsfuse mount (cold page cache for /models); the VMs and the pulled image stay."""
    t = time.time()
    e2e.set_replicas(lane, 0)
    while e2e.worker_pods(lane):
        if time.time() - t > timeout_s:
            raise RuntimeError(f"{lane}: worker pods not gone after {timeout_s} s")
        time.sleep(2)
    while len(ray_nodes_alive(lane)) > 0 and time.time() - t < timeout_s:
        time.sleep(2)
    t_gone = time.time()
    e2e.set_replicas(lane, 1)
    joined = wait_lane_joined(lane, timeout_s)
    return {"gone_s": round(t_gone - t, 1), "joined_s": joined, "total_s": round(time.time() - t, 1)}


def lane_inventory(lane: str, out: Path) -> Path:
    """inventory-live.json from the pod-start geometry probe (tpu_e2e.live_inventory), once both hosts joined."""
    wait_lane_joined(lane)
    inv = e2e.live_inventory(lane, json.loads(TEMPLATE.read_text()))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"inventory-{lane}.json"
    path.write_text(json.dumps(inv, indent=1))
    return path


def inv_hosts(inv_path: Path) -> list[str]:
    return [h["node_ip"] for h in json.loads(inv_path.read_text())["hosts"]]


WARM_REMOTE = r"""
import json, socket, sys, time
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
ray.init(address="auto")
def warm():
    t = time.time()
    import hetero.engine_actor, hetero.tpu_worker  # noqa: F401  (the runtime env + hetero imports)
    t1 = time.time()
    import vllm.engine.arg_utils, vllm.v1.engine.llm_engine, tpu_inference  # noqa: F401  (page cache of imports)
    return {"host": socket.gethostname(), "hetero_s": round(t1 - t, 2), "vllm_s": round(time.time() - t1, 2)}
nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("TPU")]
refs = [ray.remote(num_cpus=0, max_retries=0)(warm).options(
    scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)).remote() for n in nodes]
rows = ray.get(refs, timeout=900)
print("ASB_WARM " + json.dumps(rows), flush=True)
"""


def warm_lane(lane: str, log_path: Path) -> dict:
    """Standby = joined and ready to boot: build the deploy job's Ray runtime env (hetero_ctl's working_dir
    package + pip env) on both hosts and read the engine's imports once, without touching the TPUs.  The matrix
    cells' long-lived hosts had both; a just-recreated pod has neither."""
    from ray.job_submission import JobSubmissionClient  # noqa: PLC0415
    client = JobSubmissionClient(e2e.ray_url(lane))
    t = time.time()
    sid = client.submit_job(entrypoint="python3 -u -c " + shlex.quote(WARM_REMOTE),
                            runtime_env={"working_dir": str(ROOT), "excludes": HETERO_CTL_EXCLUDES,
                                         "pip": ["xds-protos==1.84.0"],
                                         "env_vars": {"PYTHONUNBUFFERED": "1", "OMP_HETERO_DATA_DIR": "/tmp/omp_hetero/data"}},
                            metadata={"purpose": "asb-warm"})
    while True:
        status = client.get_job_status(sid)
        if status.is_terminal():
            break
        if time.time() - t > 1200:
            client.stop_job(sid)
            raise RuntimeError(f"{lane}: warm-up job {sid} did not finish")
        time.sleep(2)
    logs = client.get_job_logs(sid)
    with log_path.open("a") as fh:
        fh.write(f"\n### warm {lane} {sid} {status}\n{logs[-4000:]}\n")
    line = next((ln for ln in logs.splitlines() if ln.startswith("ASB_WARM ")), None)
    if str(status) != "SUCCEEDED" or line is None:
        raise RuntimeError(f"{lane}: warm-up job {sid} {status}")
    return {"submission_id": sid, "seconds": round(time.time() - t, 1), "hosts": json.loads(line[len("ASB_WARM "):])}


# ---------------------------------------------------------------- engine steps (hetero runtime, layout S1)

def settings_for(arm: str, scope: str) -> dict:
    """Engine settings of a boot: the e2e/matrix arms (tpu_e2e.jax_settings) plus the seeds and hot standby."""
    from hetero.run_scope import scope_checkpoint_dir, scope_roots, scope_xla_dir  # noqa: PLC0415
    r = scope_roots(scope)
    roots = {"weight_snapshot_root": r["weight_snapshot_root"], "exec_bank_dir": r["exec_bank_dir"],
             "compilation_cache_dir": scope_xla_dir(scope)}
    restore = {**JAX_BASE, "weight_snapshot": "read", "weight_snapshot_verify": False, "fast_graph": True,
               "exec_bank": "read"}
    if arm == "seed":  # switch_campaign.sh autoscale_settings seed (the seeding instance a)
        return {**JAX_BASE, "weight_snapshot": "write", "fast_graph": True, "exec_bank": "write", **roots}
    if arm == "stock-seed":  # ckpt-gcs-jax: the stock boot whose 1 s-threshold XLA cache becomes stock-xla
        return {**JAX_BASE, "weight_snapshot": "off", "compilation_cache_dir": scope_xla_dir(scope),
                "worker_env": {"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1.0"},
                "model": scope_checkpoint_dir(scope) + "/" + Path(JAX_MODEL).name}
    if arm == "hot":  # warm-pm: restore + pre-spawn (workers parked at the gate)
        return {**restore, **roots, "prespawn": True}
    if arm == "xpeer-ov":
        return {**restore, **roots, "state_gate_timeout_s": 900.0}
    return e2e.jax_settings(arm, scope, {"ckpt_gcs": CKPT_GCS})


class Lane:
    """One slice + its private head: rcm steps with marks, an evidence directory per boot."""

    def __init__(self, lane: str, out: Path, mark=None):
        self.lane, self.out = lane, out
        out.mkdir(parents=True, exist_ok=True)
        self.log = out / "rcm.log"
        self.mark = mark or (lambda event, **kw: time.time())
        self.common = ["--ray-url", e2e.ray_url(lane)]

    def step(self, name: str, args: list[str], cwd: Path = ROOT) -> int:
        self.mark(f"{self.lane}:{name}_start")
        rc = e2e.rcm(self.lane, args, self.log, cwd=cwd)
        self.mark(f"{self.lane}:{name}_end", rc=rc)
        return rc

    def part(self, inv: Path) -> list[str]:
        return ["--hosts", ",".join(inv_hosts(inv)), "--instances", JAX_INSTANCE]

    def fetch_state(self, inv: Path, scope: str, gcs: str, parts: str = "snapshot,bank,xla", name: str = "state_fetch") -> bool:
        return self.step(name, ["scripts/startup_state.py", "fetch", *self.common, "--inventory", str(inv),
                                "--method", "gcs", "--parts", parts, "--gcs", gcs, "--target", TARGET,
                                "--target-scope", scope, "--output", str(self.out / f"{name}.json"),
                                "--timeout", "1800"]) == 0

    def fetch_checkpoint(self, inv: Path, scope: str) -> bool:
        return self.step("ckpt_fetch", ["scripts/startup_state.py", "fetch-checkpoint", *self.common, "--inventory",
                                        str(inv), "--gcs", CKPT_GCS, "--method", "gcs", "--target", TARGET,
                                        "--target-scope", scope, "--output", str(self.out / "ckpt-fetch.json"),
                                        "--timeout", "1800"]) == 0

    def delivery_argv(self, inv: Path, scope: str, method: str, *, xpeer_agents: dict | None = None,
                      source_scope: str | None = None) -> list[str]:
        args = [PY, "scripts/startup_state.py", "fetch", "--print-remote-argv", *self.common, "--inventory", str(inv),
                "--method", method, "--parts", "snapshot,bank,xla", "--target", TARGET, "--target-scope", scope,
                "--timeout", "1800"]
        if method == "gcs":
            args += ["--gcs", PM_GCS]
        else:
            args += ["--source", TARGET, "--source-scope", source_scope, "--xpeer-agents", json.dumps(xpeer_agents)]
        return json.loads(subprocess.run(args, cwd=ROOT, env=_rcm_env(self.lane), capture_output=True, text=True,
                                         check=True).stdout.strip())

    def deploy(self, inv: Path, settings: dict, *, delivery: list[str] | None = None, name: str = "deploy") -> bool:
        extra = ["--state-delivery", json.dumps({"argv": delivery, "timeout_s": 1800})] if delivery else []
        rc = self.step(name, ["scripts/hetero_ctl.py", "deploy", "--layout", JAX_LAYOUT, *self.common, "--namespace",
                              NS, "--head-pod", head_pod(self.lane), "--inventory", str(inv),
                              "--output", str(self.out / f"{name}.json"), "--timeout", "5400", "--",
                              "--model", JAX_MODEL, *JAX_SHAPE, "--engine-settings", json.dumps(JAX_BASE),
                              "--instance-engine-settings", json.dumps({JAX_INSTANCE: settings}), *extra,
                              "--partition", "--instances", JAX_INSTANCE])
        return rc == 0 and ok_json(self.out / f"{name}.json")

    def check(self, inv: Path, restored: bool) -> dict:
        req = ["--require-restored"] if restored else []
        ok_c = self.step("check", ["scripts/snapshot_validate.py", "check", "--layout", JAX_LAYOUT, *self.common,
                                   *self.part(inv), *req, "--output", str(self.out / "check.json")]) == 0 \
            and ok_json(self.out / "check.json")
        ok_v = self.step("validate", ["scripts/snapshot_validate.py", "validate", "--layout", JAX_LAYOUT,
                                      *self.common, *self.part(inv), *req,
                                      "--output", str(self.out / "validation.json")]) == 0 \
            and ok_json(self.out / "validation.json")
        greedy = e2e.greedy_compare(self.out / "validation.json", str(REFERENCE)) if ok_v else {"passed": False}
        return {"check_ok": ok_c, "validate_ok": ok_v, "greedy": greedy,
                "ok": bool(ok_c and ok_v and greedy.get("passed"))}

    def teardown(self) -> dict:
        res = {"drain": self.step("drain", ["scripts/hetero_ctl.py", "drain", "--layout", JAX_LAYOUT, *self.common,
                                            "--namespace", NS, "--head-pod", head_pod(self.lane), "--output",
                                            str(self.out / "drain.json"), "--timeout", "900", "--", "--force",
                                            "--instances", JAX_INSTANCE])}
        res["release"] = self.step("release", ["scripts/hetero_ctl.py", "release", "--layout", JAX_LAYOUT,
                                               *self.common, "--namespace", NS, "--head-pod", head_pod(self.lane),
                                               "--output", str(self.out / "release.json"), "--timeout", "600", "--",
                                               "--instances", JAX_INSTANCE])
        res["registry"] = e2e.kill_registry(self.lane)
        return res

    def scope_cleanup(self, inv: Path, scope: str) -> bool:
        return self.step("scope_cleanup", ["scripts/snapshot_validate.py", "cleanup", "--layout", JAX_LAYOUT,
                                           *self.common, *self.part(inv), "--run-scope", scope,
                                           "--output", str(self.out / f"cleanup-{scope}.json")]) == 0

    def upload(self, inv: Path, scope: str, gcs: str, parts: str, name: str) -> bool:
        stage_token(self.lane)
        try:
            return self.step(name, ["scripts/startup_state.py", "upload", *self.common, "--inventory", str(inv),
                                    "--source", TARGET, "--source-scope", scope, "--gcs", gcs, "--parts", parts,
                                    "--output", str(self.out / f"{name}.json"), "--timeout", "1800"]) == 0
        finally:
            unstage_token(self.lane)


def ok_json(path: Path) -> bool:
    try:
        return json.loads(path.read_text())["measurements"].get("ok") is not False
    except Exception:  # noqa: BLE001
        return False


def gz(path: Path) -> None:
    for f in ([path] if path.is_file() else path.rglob("*.json")):
        if f.suffix == ".json" and f.exists() and f.stat().st_size > 100_000:
            subprocess.run(["gzip", "-f", str(f)], check=False)


def load_json(path: Path):
    for p in (path, path.with_suffix(path.suffix + ".gz")):
        if p.exists():
            return json.loads((gzip.open(p, "rt") if p.suffix == ".gz" else p.open()).read())
    return None


# ---------------------------------------------------------------- seeds

def seed(lane: str, out: Path) -> dict:
    """The TPUFlex seed (switch_campaign.sh autoscale, partition mode, at 966399c's hetero/*): a cold S1:small_a
    boot that writes the snapshot, the bank and the XLA cache into its scope; greedy check against the TP8 seed
    reference of the matrix cell; teardown; upload snapshot+bank+xla to PM_GCS."""
    mark = e2e.Marks(out / "timeline.jsonl")
    ln = Lane(lane, out, mark)
    scope = f"asb-seed-{lane}"
    inv = lane_inventory(lane, out)
    res: dict = {"lane": lane, "scope": scope, "pm_gcs": PM_GCS, "code_commit": git_rev(), "dirty": git_dirty()}
    res["deploy_ok"] = ln.deploy(inv, settings_for("seed", scope))
    if res["deploy_ok"]:
        res["check"] = ln.check(inv, restored=False)
    res["teardown"] = ln.teardown()
    if res["deploy_ok"] and res["check"]["ok"]:
        res["upload_ok"] = ln.upload(inv, scope, PM_GCS, "snapshot,bank,xla", "upload")
    res["ok"] = bool(res.get("upload_ok"))
    (out / "seed.json").write_text(json.dumps(res, indent=1))
    gz(out)
    return res


def stock_seed(lane: str, out: Path) -> dict:
    """The stock XLA cache: a ckpt-gcs-jax boot (checkpoint to tmpfs, normal loader, JAX's default 1 s
    persistence threshold) writing its cache into the scope; greedy check; teardown; upload parts=xla."""
    mark = e2e.Marks(out / "timeline.jsonl")
    ln = Lane(lane, out, mark)
    scope = f"asb-stockseed-{lane}"
    inv = lane_inventory(lane, out)
    res: dict = {"lane": lane, "scope": scope, "stock_gcs": STOCK_GCS, "code_commit": git_rev(), "dirty": git_dirty()}
    res["ckpt_ok"] = ln.fetch_checkpoint(inv, scope)
    res["deploy_ok"] = res["ckpt_ok"] and ln.deploy(inv, settings_for("stock-seed", scope))
    if res["deploy_ok"]:
        res["check"] = ln.check(inv, restored=False)
    res["teardown"] = ln.teardown()
    if res["deploy_ok"] and res["check"]["ok"]:
        res["upload_ok"] = ln.upload(inv, scope, STOCK_GCS, "xla", "upload")
    res["ok"] = bool(res.get("upload_ok"))
    (out / "seed.json").write_text(json.dumps(res, indent=1))
    gz(out)
    return res


def boot_pm(lane: str, out: Path, scope: str, *, inv: Path | None = None, check: bool = True) -> dict:
    """A TPUFlex instance from GCS (gcs-ov: state delivery overlapped with spawn), e.g. a bed's base."""
    mark = e2e.Marks(out / "timeline.jsonl")
    ln = Lane(lane, out, mark)
    inv = inv or lane_inventory(lane, out)
    res: dict = {"lane": lane, "scope": scope, "code_commit": git_rev(), "dirty": git_dirty(), "inventory": str(inv)}
    res["deploy_ok"] = ln.deploy(inv, settings_for("gcs-ov", scope), delivery=ln.delivery_argv(inv, scope, "gcs"))
    if res["deploy_ok"] and check:
        res["check"] = ln.check(inv, restored=True)
    res["ok"] = bool(res["deploy_ok"] and (not check or res["check"]["ok"]))
    res["endpoint"] = endpoint_of(out / "deploy.json")
    (out / "boot.json").write_text(json.dumps(res, indent=1))
    gz(out)
    return res


def endpoint_of(deploy_json: Path) -> str | None:
    d = load_json(deploy_json)
    try:
        desc = d["measurements"]["deployment"]["instances"][0]["descriptor"]
        return f"http://{desc['endpoint_host']}:{desc['endpoint_port']}"
    except Exception:  # noqa: BLE001
        return None


def xpeer_serve(lane: str, inv: Path, out: Path) -> dict:
    """Leave the cross-cluster peer agents (startup_state.py xpeer-serve) on the lane's hosts: the source side of
    the pm-peer arm's state delivery, serving the base instance's own scope (snapshot + bank + XLA cache)."""
    ln = Lane(lane, out)
    rc = ln.step("xpeer_serve", ["scripts/startup_state.py", "xpeer-serve", *ln.common, "--inventory", str(inv),
                                 "--source", TARGET, "--output", str(out / "xpeer-serve.json"), "--timeout", "300"])
    if rc != 0:
        raise RuntimeError(f"{lane}: xpeer-serve failed")
    return json.loads((out / "xpeer-serve.json").read_text())["measurements"]["xpeer_agents"]


def base_up(lane: str, *, xpeer: bool) -> dict:
    """A bed's base instance: TPUFlex from GCS into scope asb-base-<lane> (kept on the hosts: the pm-peer
    source), greedy-checked; recorded in OUT_ROOT/base/<lane>.json for the runs."""
    out = OUT_ROOT / "base" / f"{lane}-boot-{int(time.time())}"
    scope = f"asb-base-{lane}"
    res = boot_pm(lane, out, scope)
    if not res["ok"]:
        raise RuntimeError(f"{lane}: base boot failed: {res}")
    row = {"lane": lane, "scope": scope, "endpoint": res["endpoint"], "booted_at": time.time(), "boot_dir": str(out),
           "inventory": res["inventory"], "greedy": res["check"]["greedy"], "code_commit": res["code_commit"]}
    if xpeer:
        row["xpeer_agents"] = xpeer_serve(lane, Path(res["inventory"]), out)
    (OUT_ROOT / "base" / f"{lane}.json").write_text(json.dumps(row, indent=1))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("up")
    p.add_argument("--lanes", default="")
    p.add_argument("--beds", default="")
    sub.add_parser("pf")
    for name in ("seed", "stock-seed", "inventory", "warm", "restart", "base", "teardown", "stage-token"):
        p = sub.add_parser(name)
        p.add_argument("--lane", required=True)
        p.add_argument("--out", type=Path)
        p.add_argument("--scope")
        p.add_argument("--xpeer", action="store_true")
    p = sub.add_parser("manifest")
    p.add_argument("--lane")
    p.add_argument("--bed")
    args = ap.parse_args()
    if args.cmd == "up":
        up([x for x in args.lanes.split(",") if x], [x for x in args.beds.split(",") if x])
    elif args.cmd == "pf":
        print("\n".join(pf_commands()))
    elif args.cmd == "manifest":
        print(json.dumps(raycluster(args.lane) if args.lane else loadgen_manifests(args.bed), indent=1))
    else:
        out = args.out or OUT_ROOT / "dev" / f"{args.cmd}-{args.lane}-{int(time.time())}"
        out.mkdir(parents=True, exist_ok=True)
        if args.cmd == "seed":
            res = seed(args.lane, out)
        elif args.cmd == "stock-seed":
            res = stock_seed(args.lane, out)
        elif args.cmd == "inventory":
            res = {"inventory": str(lane_inventory(args.lane, out))}
        elif args.cmd == "warm":
            res = warm_lane(args.lane, out / "warm.log")
        elif args.cmd == "restart":
            res = restart_workers(args.lane)
        elif args.cmd == "base":
            res = base_up(args.lane, xpeer=args.xpeer)
        elif args.cmd == "teardown":
            res = Lane(args.lane, out).teardown()
        elif args.cmd == "stage-token":
            res = stage_token(args.lane)
        log(json.dumps(res, default=str)[:3000])
        return 0 if res.get("ok", True) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
