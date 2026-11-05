#!/usr/bin/env python3
"""GKE TPU scale-from-zero latency probe, per TPU v5e topology x provisioning model (on-demand / spot).

Each arm is its own node pool (0..N nodes, cluster autoscaler).  One sample: wait until the pool has
0 nodes, submit an Indexed Job that needs every host of the slice, and record the critical path

    submit -> TriggeredScaleUp -> GCE VM created -> k8s Node registered -> Node Ready
           -> google.com/tpu allocatable -> pod scheduled -> image pulled -> container started
           -> `import jax` -> jax.devices() sees every chip of the slice

then delete the Job and record how long the autoscaler takes to remove the nodes again.

    python3 scripts/gke_autoscale_probe.py create-pools [--arms ...]
    python3 scripts/gke_autoscale_probe.py run --samples 5 --out results/gke-autoscale/<tag> [--arms ...]
    python3 scripts/gke_autoscale_probe.py summarize results/gke-autoscale/<tag>
    python3 scripts/gke_autoscale_probe.py delete-pools [--arms ...]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

CLUSTER = "hongtaozhang-llm-gke-us-west1"
ZONE = "us-west1-c"
NAMESPACE = "gke-autoscale-probe"
IMAGE = "vllm/vllm-tpu:v0.26.0"  # the serving image, so the image pull is the real one
ACCELERATOR = "tpu-v5-lite-podslice"

# topology -> (machine type, hosts, chips per host)
TOPOLOGIES = {
    "1x1": ("ct5lp-hightpu-1t", 1, 1),
    "2x2": ("ct5lp-hightpu-4t", 1, 4),
    "2x4": ("ct5lp-hightpu-8t", 1, 8),
    "4x4": ("ct5lp-hightpu-4t", 4, 4),
    "4x8": ("ct5lp-hightpu-4t", 8, 4),
}
MODES = ("od", "spot")
ALL_ARMS = [f"{topo}-{mode}" for topo in TOPOLOGIES for mode in MODES]

POLL_S = 5.0
SAMPLE_TIMEOUT_S = 45 * 60  # scale-up + boot; longer means no capacity
DRAIN_TIMEOUT_S = 60 * 60   # waiting for the pool to return to 0 nodes

PROBE = r"""
import json, socket, time
def emit(stage, **kw):
    print("PROBE " + json.dumps({"stage": stage, "t": time.time(), "host": socket.gethostname(), **kw}), flush=True)
emit("container_start")
import jax, jax.numpy as jnp
emit("jax_imported", version=jax.__version__)
devices = jax.devices()
emit("jax_devices", device_count=len(devices), local_device_count=jax.local_device_count())
x = jnp.ones((1024, 1024), jnp.bfloat16)
float((x @ x).sum())
emit("first_compute")
time.sleep(20)
"""


def pool_name(arm: str) -> str:
    return f"as-v5e-{arm}"


def job_labels(arm: str) -> str:
    return f"probe-{arm}"


def sh(cmd: list[str], check: bool = True, input_text: str | None = None) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, input=input_text)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


def kjson(*args: str) -> dict:
    return json.loads(sh(["kubectl", "-n", NAMESPACE, *args, "-o", "json"]))


def ts(value: str | None) -> float | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def log(arm: str, msg: str) -> None:
    print(f"{dt.datetime.now().strftime('%H:%M:%S')} [{arm}] {msg}", flush=True)


# ---------------------------------------------------------------- pools

def create_pool_cmd(arm: str) -> list[str]:
    topo, mode = arm.rsplit("-", 1)
    machine, hosts, _ = TOPOLOGIES[topo]
    cmd = ["gcloud", "container", "node-pools", "create", pool_name(arm), "--cluster", CLUSTER, "--zone", ZONE,
           "--node-locations", ZONE, "--machine-type", machine,
           "--num-nodes", "0", "--enable-autoscaling", "--min-nodes", "0", "--max-nodes", str(hosts),
           "--disk-type", "pd-balanced", "--disk-size", "100", "--workload-metadata", "GKE_METADATA",
           "--node-labels", "hetero-probe=gke-autoscale", "--quiet"]
    if hosts > 1:  # single-host slice pools reject an explicit topology; GKE still labels the nodes
        cmd += ["--tpu-topology", topo]
    if mode == "spot":
        cmd.append("--spot")
    return cmd


def cluster_op(cmd: list[str], attempts: int = 40) -> None:
    """Run a node-pool operation, retrying while GKE reports a conflicting operation in progress
    (operations on different pools normally run concurrently)."""
    for attempt in range(attempts):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        if "operation" in proc.stderr.lower() and attempt + 1 < attempts:
            time.sleep(15)
            continue
        raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr.strip()[-600:]}")


def reset_pool(arm: str) -> None:
    """Force the pool back to 0 nodes instead of waiting ~600 s for the autoscaler's unneeded-node timer.
    Single-host pools resize; multi-host TPU pools reject manual resize (501 UNIMPLEMENTED), so they are
    deleted and recreated with the same spec."""
    _, hosts, _ = TOPOLOGIES[arm.rsplit("-", 1)[0]]
    if hosts == 1:
        cluster_op(["gcloud", "container", "clusters", "resize", CLUSTER, "--zone", ZONE, "--node-pool",
                    pool_name(arm), "--num-nodes", "0", "--quiet"])
    else:
        cluster_op(["gcloud", "container", "node-pools", "delete", pool_name(arm), "--cluster", CLUSTER,
                    "--zone", ZONE, "--quiet"])
        cluster_op(create_pool_cmd(arm))


def create_pools(arms: list[str]) -> None:
    existing = set(sh(["gcloud", "container", "node-pools", "list", "--cluster", CLUSTER, "--zone", ZONE,
                       "--format=value(name)"]).split())
    for arm in arms:
        name = pool_name(arm)
        if name in existing:
            print(f"{name}: exists")
            continue
        print(f"{name}: creating", flush=True)
        t0 = time.time()
        sh(create_pool_cmd(arm))
        print(f"{name}: created in {time.time() - t0:.0f} s", flush=True)
    sh(["kubectl", "create", "namespace", NAMESPACE], check=False)


def delete_pools(arms: list[str]) -> None:
    existing = set(sh(["gcloud", "container", "node-pools", "list", "--cluster", CLUSTER, "--zone", ZONE,
                       "--format=value(name)"]).split())
    for arm in arms:
        name = pool_name(arm)
        if name in existing:
            print(f"{name}: deleting", flush=True)
            sh(["gcloud", "container", "node-pools", "delete", name, "--cluster", CLUSTER, "--zone", ZONE, "--quiet"])


# ---------------------------------------------------------------- one sample

def manifest(arm: str, job: str) -> str:
    topo, mode = arm.rsplit("-", 1)
    _, hosts, chips = TOPOLOGIES[topo]
    svc = job_labels(arm)
    tolerations = [{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}]
    if mode == "spot":
        tolerations.append({"key": "cloud.google.com/gke-spot", "operator": "Exists", "effect": "NoSchedule"})
    service = {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": svc, "namespace": NAMESPACE},
        "spec": {"clusterIP": "None", "selector": {"probe-arm": arm}},
    }
    job_obj = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": job, "namespace": NAMESPACE, "labels": {"probe-arm": arm}},
        "spec": {
            "completionMode": "Indexed", "completions": hosts, "parallelism": hosts, "backoffLimit": 0,
            "template": {
                "metadata": {"labels": {"probe-arm": arm, "probe-job": job}},
                "spec": {
                    # Host networking as in the serving Ray workers: with pod networking, the multi-host
                    # libtpu SliceBuilder never finishes (all 8471 ports reachable, backend init hangs).
                    "hostNetwork": True, "dnsPolicy": "ClusterFirstWithHostNet",
                    "subdomain": svc, "restartPolicy": "Never",
                    "nodeSelector": {"cloud.google.com/gke-nodepool": pool_name(arm),
                                     "cloud.google.com/gke-tpu-accelerator": ACCELERATOR,
                                     "cloud.google.com/gke-tpu-topology": topo},
                    "tolerations": tolerations,
                    "containers": [{
                        "name": "probe", "image": IMAGE, "imagePullPolicy": "Always",
                        "command": ["python3", "-c", PROBE],
                        "ports": [{"containerPort": 8471}, {"containerPort": 8431}],
                        "resources": {"requests": {"google.com/tpu": str(chips)},
                                      "limits": {"google.com/tpu": str(chips)}},
                    }],
                },
            },
        },
    }
    return json.dumps({"apiVersion": "v1", "kind": "List", "items": [service, job_obj]})


def pool_nodes(arm: str) -> list[dict]:
    out = json.loads(sh(["kubectl", "get", "nodes", "-l", f"cloud.google.com/gke-nodepool={pool_name(arm)}",
                         "-o", "json"]))
    return out["items"]


def wait_pool_empty(arm: str, deadline_s: float) -> float | None:
    """Seconds until the pool has 0 nodes (None on timeout)."""
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if not pool_nodes(arm):
            return time.time() - t0
        time.sleep(POLL_S * 2)
    return None


def event_time(ev: dict) -> float | None:
    return ts(ev.get("eventTime")) or ts(ev.get("firstTimestamp")) or ts(ev.get("metadata", {}).get("creationTimestamp"))


def run_sample(arm: str, index: int, tag: str) -> dict:
    topo, mode = arm.rsplit("-", 1)
    _, hosts, chips = TOPOLOGIES[topo]
    job = f"probe-{arm}-{tag}-{index}"
    rec: dict = {"arm": arm, "topology": topo, "mode": mode, "hosts": hosts, "chips": hosts * chips,
                 "sample": index, "job": job, "image": IMAGE}

    sh(["kubectl", "apply", "-f", "-"], input_text=manifest(arm, job))
    submitted = time.time()
    rec["t_submit"] = submitted
    seen_tpu_alloc: dict[str, float] = {}
    seen_nodes: dict[str, float] = {}
    status = "timeout"
    while time.time() - submitted < SAMPLE_TIMEOUT_S:
        for node in pool_nodes(arm):
            name = node["metadata"]["name"]
            seen_nodes.setdefault(name, time.time())
            if int(node["status"].get("allocatable", {}).get("google.com/tpu", "0")) > 0:
                seen_tpu_alloc.setdefault(name, time.time())
        jobj = kjson("get", "job", job)
        conds = {c["type"]: c for c in jobj.get("status", {}).get("conditions", []) if c["status"] == "True"}
        if "Complete" in conds:
            status = "complete"
            break
        if "Failed" in conds:
            status = "failed"
            rec["failure"] = conds["Failed"].get("message")
            break
        time.sleep(POLL_S)
    rec["status"] = status
    rec["t_poll_end"] = time.time()

    # --- pods: scheduling, image pull, container lifetime, probe log lines
    pods = kjson("get", "pods", "-l", f"probe-job={job}")["items"]
    rec["pods"] = []
    for pod in pods:
        p = {"name": pod["metadata"]["name"], "node": pod["spec"].get("nodeName"),
             "phase": pod["status"].get("phase")}
        for c in pod["status"].get("conditions", []):
            if c["status"] == "True":
                p[f"cond_{c['type']}"] = ts(c.get("lastTransitionTime"))
        for cs in pod["status"].get("containerStatuses", []):
            state = cs.get("state", {})
            term = state.get("terminated") or {}
            p["container_started"] = ts((state.get("running") or term).get("startedAt"))
            p["container_finished"] = ts(term.get("finishedAt"))
            p["exit_code"] = term.get("exitCode")
        logs = sh(["kubectl", "-n", NAMESPACE, "logs", pod["metadata"]["name"]], check=False)
        p["probe"] = [json.loads(line[6:]) for line in logs.splitlines() if line.startswith("PROBE ")]
        rec["pods"].append(p)

    # --- events: autoscaler decisions, pulls, preemptions
    events = kjson("get", "events")["items"]
    pod_names = {p["name"] for p in rec["pods"]}
    rec["events"] = sorted(
        ({"t": event_time(e), "obj": e["involvedObject"].get("name"), "kind": e["involvedObject"].get("kind"),
          "reason": e.get("reason"), "msg": (e.get("message") or "")[:400], "count": e.get("count")}
         for e in events if e["involvedObject"].get("name") in pod_names | {job}),
        key=lambda e: e["t"] or 0)

    # --- nodes: GCE VM creation, k8s registration, Ready, TPU allocatable
    rec["nodes"] = []
    for node in pool_nodes(arm):
        name = node["metadata"]["name"]
        n = {"name": name, "registered": ts(node["metadata"]["creationTimestamp"]),
             "first_seen": seen_nodes.get(name), "tpu_allocatable_seen": seen_tpu_alloc.get(name)}
        for c in node["status"].get("conditions", []):
            if c["type"] == "Ready" and c["status"] == "True":
                n["ready"] = ts(c.get("lastTransitionTime"))
        vm = sh(["gcloud", "compute", "instances", "describe", name, "--zone", ZONE,
                 "--format=value(creationTimestamp)"], check=False).strip()
        n["vm_created"] = ts(vm) if vm else None
        rec["nodes"].append(n)

    # --- release; the caller measures (or forces) the scale-down
    sh(["kubectl", "-n", NAMESPACE, "delete", "job", job, "--cascade=foreground", "--wait=true"], check=False)
    rec["t_released"] = time.time()
    rec["derived"] = derive(rec)
    return rec


def derive(rec: dict) -> dict:
    t0 = rec["t_submit"]

    def rel(values):
        values = [v for v in values if v is not None]
        return round(max(values) - t0, 1) if values else None

    def first(values):
        values = [v for v in values if v is not None]
        return round(min(values) - t0, 1) if values else None

    probe = [ev for p in rec["pods"] for ev in p["probe"]]
    by_stage = {}
    for ev in probe:
        by_stage.setdefault(ev["stage"], []).append(ev["t"])
    pulled = [e for e in rec["events"] if e["reason"] == "Pulled"]
    scale_up = [e["t"] for e in rec["events"] if e["reason"] == "TriggeredScaleUp"]
    devices = [ev.get("device_count") for ev in probe if ev["stage"] == "jax_devices"]
    # Critical path = the last host of the slice to reach each stage.
    return {
        "triggered_scale_up": first(scale_up),
        "vm_created": rel(n.get("vm_created") for n in rec["nodes"]),
        "node_registered": rel(n.get("registered") for n in rec["nodes"]),
        "node_ready": rel(n.get("ready") for n in rec["nodes"]),
        "tpu_allocatable": rel(n.get("tpu_allocatable_seen") for n in rec["nodes"]),
        "pod_scheduled": rel(p.get("cond_PodScheduled") for p in rec["pods"]),
        "image_pulled": rel(e["t"] for e in pulled),
        "pull_msgs": [e["msg"] for e in pulled],
        "container_started": rel(by_stage.get("container_start", [])),
        "jax_imported": rel(by_stage.get("jax_imported", [])),
        "jax_devices": rel(by_stage.get("jax_devices", [])),
        "device_count": max(devices) if devices else None,
        "first_compute": rel(by_stage.get("first_compute", [])),
        "scale_down_s": None,  # filled in by run_arm on natural-scale-down samples
        "failure_reasons": sorted({e["reason"] for e in rec["events"]
                                   if e["reason"] in ("FailedScaleUp", "NotTriggerScaleUp", "FailedScheduling",
                                                      "Preempted", "Evicted", "BackoffLimitExceeded")}),
    }


# ---------------------------------------------------------------- driver

def run_arm(arm: str, samples: int, natural: int, out_dir: Path, tag: str, lock: threading.Lock) -> None:
    out = out_dir / "samples.jsonl"
    for index in range(1, samples + 1):
        log(arm, f"sample {index}: waiting for 0 nodes")
        if wait_pool_empty(arm, DRAIN_TIMEOUT_S) is None:
            log(arm, "pool never drained; stopping arm")
            return
        try:
            rec = run_sample(arm, index, tag)
        except Exception as exc:  # keep the other arms going, record the error
            rec = {"arm": arm, "sample": index, "status": "error", "error": repr(exc), "t": time.time()}
        d = rec.get("derived", {})
        log(arm, f"sample {index}: {rec['status']} ready={d.get('jax_devices')} s "
                 f"(node_ready={d.get('node_ready')} pulled={d.get('image_pulled')} devices={d.get('device_count')}) "
                 f"{d.get('failure_reasons') or ''}")
        # Natural autoscaler scale-down on the first `natural` samples, forced reset afterwards.
        t0 = time.time()
        if index <= natural:
            rec["scale_down"] = "autoscaler"
            rec["scale_down_s"] = wait_pool_empty(arm, DRAIN_TIMEOUT_S)
            if d:
                d["scale_down_s"] = None if rec["scale_down_s"] is None else round(rec["scale_down_s"], 1)
            log(arm, f"sample {index}: autoscaler scale-down {d.get('scale_down_s')} s")
        else:
            rec["scale_down"] = "forced"
            try:
                reset_pool(arm)
            except Exception as exc:
                log(arm, f"reset failed ({exc!r}); falling back to the autoscaler")
            rec["reset_s"] = round(time.time() - t0, 1)
        with lock, out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")


def run(arms: list[str], samples: int, natural: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = dt.datetime.now().strftime("%m%d%H%M")
    (out_dir / "config.json").write_text(json.dumps(
        {"cluster": CLUSTER, "zone": ZONE, "image": IMAGE, "arms": arms, "samples": samples, "tag": tag,
         "natural_scale_down_samples": natural, "topologies": TOPOLOGIES, "started": time.time()}, indent=1))
    lock = threading.Lock()
    threads = [threading.Thread(target=run_arm, args=(arm, samples, natural, out_dir, tag, lock), daemon=False)
               for arm in arms]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    summarize(out_dir)


STAGES = ["triggered_scale_up", "vm_created", "node_registered", "node_ready", "tpu_allocatable", "pod_scheduled",
          "image_pulled", "container_started", "jax_imported", "jax_devices", "scale_down_s"]


def summarize(out_dir: Path) -> None:
    rows: dict[str, list[dict]] = {}
    for line in (out_dir / "samples.jsonl").read_text().splitlines():
        rec = json.loads(line)
        rows.setdefault(rec["arm"], []).append(rec)
    print("| arm | ok/n | " + " | ".join(STAGES) + " |")
    print("|---|---|" + "---:|" * len(STAGES))
    for arm in sorted(rows, key=lambda a: (list(TOPOLOGIES).index(a.rsplit("-", 1)[0]), a)):
        recs = rows[arm]
        ok = [r for r in recs if r.get("status") == "complete"]
        cells = []
        for stage in STAGES:
            vals = [r["derived"][stage] for r in ok if r["derived"].get(stage) is not None]
            cells.append(f"{statistics.median(vals):.0f} [{min(vals):.0f}-{max(vals):.0f}]" if vals else "-")
        print(f"| {arm} | {len(ok)}/{len(recs)} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("create-pools", "delete-pools", "run"):
        p = sub.add_parser(name)
        p.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
        if name == "run":
            p.add_argument("--samples", type=int, default=5)
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--natural-scale-down", type=int, default=1,
                           help="samples per arm whose scale-down is left to the autoscaler (rest are forced)")
    p = sub.add_parser("summarize")
    p.add_argument("out", type=Path)
    args = ap.parse_args()
    if args.cmd == "create-pools":
        create_pools(args.arms)
    elif args.cmd == "delete-pools":
        delete_pools(args.arms)
    elif args.cmd == "run":
        run(args.arms, args.samples, args.natural_scale_down, args.out)
    else:
        summarize(args.out)


if __name__ == "__main__":
    sys.exit(main())
