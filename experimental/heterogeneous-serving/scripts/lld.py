#!/usr/bin/env python3
"""llm-d autoscaling of the TP8 Gemma-4 31B vLLM-JAX engine (hetero S1:small_a, stock vs TPUFlex boot) on a
dedicated GKE cluster: LeaderWorkerSet replicas (1 group = 1 2x4 v5e slice) behind the llm-d router (EPP +
Envoy, standalone mode, InferencePool, flow control) scaled by the HPA on the EPP flow-control queue
(llm_d_epp_flow_control_queue_size -> Prometheus -> prometheus-adapter -> external metric epp_queue_size).

Beds (one namespace + router + LWS + HPA each, run concurrently):
    s  standby capacity: lld-s0 (base) + lld-s1..s3 on-demand slices held by low-priority balloon pods
    n  new capacity:     lld-n0 (base, on-demand) + lld-n1..n3 spot pools autoscaled 0..2 nodes

    lld.py pools  --bed s|n            # create the bed's node pools (one at a time)
    lld.py setup  --bed s|n            # namespace, code ConfigMap, router (helm), balloons (s)
    lld.py adapter                     # prometheus-adapter + external rule (once per cluster)
    lld.py lws    --bed s --arm pm --replicas 1 [--wait]
    lld.py hpa    --bed s --arm pm --target 4 [--up-stab 0] [--down-stab 300]
    lld.py run    --bed s --arm pm|stock|static-base|static-peak --rep 1 [--trace a] [--hpa ...]
    lld.py teardown --bed s|n          # delete LWS/HPA (checks run in preStop), router, namespace, pools
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLUSTER, ZONE, REGION = "lld-autoscale", "us-west1-c", "us-west1"
KCFG = str(Path.home() / ".kube/lld-usw1.yaml")
IMAGE = "vllm/vllm-tpu@sha256:186f1940c353d2703bf80f80fff101b38fa8e26a9562a767b37e103f212d3d79"  # v0.26.0
BUCKET = "hongtaozhang-tpu-model-cache-us-west1"
GCS_ROOT = f"gs://{BUCKET}"
EVIDENCE = f"{GCS_ROOT}/lld-evidence"
ROUTER_CHART = "oci://ghcr.io/llm-d/charts/llm-d-router-standalone"
ROUTER_VERSION = "v0.9.0"
MON_NS = "llm-d-monitoring"
PROM_URL = f"http://llmd-kube-prometheus-stack-prometheus.{MON_NS}.svc"
RESULTS = ROOT / "results/llmd-autoscale"
# the v0.9.0 standalone chart's Envoy preset + one response header (x-lld-upstream = the EPP-picked endpoint)
ENVOY_TPL = ROOT / "deploy/lld/envoy-v0.9.0-standalone.yaml.tpl"
CAP = 8  # per-endpoint in-flight cap (AutoscaleBurst router cap 8) = EPP concurrency-detector maxConcurrency
TTL_S = 600  # AutoscaleBurst admission timeout 600 s = EPP flowControl defaultRequestTTL
CODE_PATHS = ["hetero", "scripts/lld_pod.py", "scripts/startup_state.py", "scripts/snapshot_validate.py",
              "scripts/bank_relocate.py", "scripts/lld"]
# code that determines a run's behaviour (the bundle + the workstation orchestrator + the loadgen pod's scripts)
RUN_CODE = CODE_PATHS + ["scripts/lld.py", "scripts/asb_loadgen.py", "scripts/lld_sampler.py", "deploy/lld"]

BEDS = {
    "s": {"ns": "lld-s", "pools": {"lld-s0": "base", "lld-s1": "standby", "lld-s2": "standby", "lld-s3": "standby"}},
    "n": {"ns": "lld-n", "pools": {"lld-n0": "base", "lld-n1": "new", "lld-n2": "new", "lld-n3": "new"}},
}


def log(msg: str) -> None:
    print(f"[lld {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kenv() -> dict:
    return {**os.environ, "KUBECONFIG": KCFG}


def sh(cmd: list[str], check: bool = True, input_text: str | None = None, timeout: float | None = None,
       quiet: bool = False) -> str:
    p = subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=timeout, env=kenv())
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:6])} rc={p.returncode}: {p.stderr[-1500:]}")
    if not quiet and p.stderr.strip() and p.returncode != 0 and "exit code 1" not in p.stderr:
        log(p.stderr.strip()[-400:])
    return p.stdout


def kubectl(*args: str, check: bool = True, input_text: str | None = None, timeout: float | None = 120) -> str:
    return sh(["kubectl", *args], check=check, input_text=input_text, timeout=timeout)


def kjson(*args: str) -> dict:
    return json.loads(kubectl(*args, "-o", "json"))


def apply(docs: list[dict], server_side: bool = False) -> str:
    extra = ["--server-side", "--force-conflicts"] if server_side else []
    return kubectl("apply", *extra, "-f", "-", input_text="\n---\n".join(json.dumps(d) for d in docs))


def git_rev() -> str:
    """The code revision: $LLD_REV (a campaign's frozen code commit; HEAD may add evidence or report-only commits
    on top, but the run code (RUN_CODE) must be identical) or HEAD."""
    head = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    rev = os.environ.get("LLD_REV")
    if not rev:
        return head
    if subprocess.run(["git", "diff", "--quiet", rev, "HEAD", "--", *RUN_CODE], cwd=ROOT).returncode != 0:
        raise RuntimeError(f"LLD_REV={rev}: run code differs from HEAD {head} ({RUN_CODE})")
    return rev


def git_dirty(paths: list[str] | None = None) -> str:
    return subprocess.run(["git", "status", "--porcelain", "--", *(paths or ["scripts", "hetero"])], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- node pools

def pool_cmd(name: str, bed: str, role: str) -> list[str]:
    cmd = ["gcloud", "container", "node-pools", "create", name, "--cluster", CLUSTER, "--zone", ZONE,
           "--node-locations", ZONE, "--machine-type", "ct5lp-hightpu-4t", "--tpu-topology", "2x4",
           "--disk-type", "pd-balanced", "--disk-size", "100", "--workload-metadata", "GKE_METADATA",
           "--scopes", "cloud-platform", "--node-labels", f"lld/bed={bed},lld/role={role},lld/pool={name}", "--quiet"]
    if role == "new":
        cmd += ["--spot", "--num-nodes", "0", "--enable-autoscaling", "--min-nodes", "0", "--max-nodes", "2"]
    else:
        cmd += ["--num-nodes", "2"]
    return cmd


def pools(bed: str, only: list[str] | None = None) -> None:
    existing = set(sh(["gcloud", "container", "node-pools", "list", "--cluster", CLUSTER, "--zone", ZONE,
                       "--format=value(name)"]).split())
    for name, role in BEDS[bed]["pools"].items():
        if (only and name not in only) or name in existing:
            continue
        t = time.time()
        log(f"creating pool {name} ({role})")
        p = subprocess.run(pool_cmd(name, bed, role), capture_output=True, text=True, env=kenv())
        log(f"pool {name}: rc={p.returncode} in {time.time() - t:.0f} s {p.stderr.strip()[-300:]}")
        if p.returncode != 0:
            raise RuntimeError(f"pool {name} failed: {p.stderr[-600:]}")


# ---------------------------------------------------------------- code bundle / router / balloons

def code_bundle(rev: str) -> bytes:
    """git archive of CODE_PATHS at the code rev (committed code only) as a tar.gz."""
    if git_dirty(CODE_PATHS):
        raise RuntimeError(f"uncommitted changes under {CODE_PATHS}: commit first (the bundle is git HEAD)")
    tar = subprocess.run(["git", "archive", "--format=tar", rev, *CODE_PATHS], cwd=ROOT, capture_output=True,
                         check=True).stdout
    return gzip.compress(tar, 9, mtime=0)


def code_configmap(ns: str, rev: str) -> dict:
    data = code_bundle(rev)
    return {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": f"lld-code-{rev}", "namespace": ns,
                                                                  "labels": {"lld/rev": rev}},
            "binaryData": {"code.tar.gz": base64.b64encode(data).decode()}}


def router_values(bed: str) -> dict:
    """llm-d guides: recipes/router/base.values.yaml + flow-control router values, adapted: one priority band,
    concurrency-detector maxConcurrency = CAP (AutoscaleBurst's per-instance cap), request TTL = 600 s
    (AutoscaleBurst's admission timeout), the optimized-baseline scorers; monitoring.values.yaml (10 s, no auth).
    Requests trimmed to fit the CPU pool (2 beds)."""
    plugins = f"""apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: prefix-cache-scorer
- type: concurrency-detector
  parameters:
    maxConcurrency: {CAP}
    concurrencyMode: requests
    headroom: 0.0
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
    weight: 2
  - pluginRef: kv-cache-utilization-scorer
    weight: 2
  - pluginRef: prefix-cache-scorer
    weight: 3
flowControl:
  maxBytes: "10Gi"
  maxRequests: "10k"
  defaultRequestTTL: "{TTL_S}s"
  saturationDetector:
    pluginRef: concurrency-detector
"""
    return {"router": {
        # (the flow-control guide also lists metrics 9090; the v0.9.0 chart's Service already has it)
        "extraServicePorts": [{"name": "http", "port": 80, "protocol": "TCP", "targetPort": 8081}],
        "epp": {"replicas": 1, "flags": {"v": 2}, "pluginsConfigFile": "lld-plugins.yaml",
                "pluginsCustomConfig": {"lld-plugins.yaml": plugins},
                "resources": {"requests": {"cpu": "2", "memory": "4Gi"}, "limits": {"memory": "16Gi"}}},
        "inferencePool": {"failureMode": "FailOpen"},
        "modelServers": {"protocol": "http", "matchLabels": {"lld/serving": "true", "lld/bed": bed,
                                                             "lld/role": "leader"},
                         "targetPorts": [{"number": 8000}]},
        "proxy": {"presets": {"envoy": {"configMap": {"data": {"envoy.yaml": ENVOY_TPL.read_text()}}}},
                  "args": ["--service-node", "envoy-sidecar", "--log-level", "warn", "--concurrency", "4",
                           "--drain-strategy", "immediate", "--drain-time-s", "60", "-c", "/etc/envoy/envoy.yaml"],
                  "resources": {"requests": {"cpu": "2", "memory": "4Gi"}, "limits": {"memory": "16Gi"}}},
        "monitoring": {"interval": "10s", "prometheus": {"enabled": True, "auth": {"enabled": False}}},
    }}


def router_install(bed: str) -> None:
    ns = BEDS[bed]["ns"]
    vals = RESULTS / f"manifests/router-values-{bed}.json"
    vals.parent.mkdir(parents=True, exist_ok=True)
    vals.write_text(json.dumps(router_values(bed), indent=1))
    sh(["helm", "upgrade", "--install", f"lld-{bed}", ROUTER_CHART, "--version", ROUTER_VERSION, "-n", ns,
        "-f", str(vals), "--wait", "--timeout", "10m"], timeout=900)


BALLOON_PRIORITY = {"apiVersion": "scheduling.k8s.io/v1", "kind": "PriorityClass",
                    "metadata": {"name": "lld-balloon"}, "value": -10, "preemptionPolicy": "Never",
                    "globalDefault": False,
                    "description": "lld standby balloons: placeholders the serving replicas preempt"}
SERVING_PRIORITY = {"apiVersion": "scheduling.k8s.io/v1", "kind": "PriorityClass",
                    "metadata": {"name": "lld-serving"}, "value": 1000, "globalDefault": False,
                    "description": "lld serving replicas (preempt balloons)"}


def balloons(bed: str, n_slices: int) -> dict:
    """GKE spare-capacity pattern: low-priority placeholders holding the standby slices' chips, running the
    serving image (pre-pulled).  A new LWS group (priority lld-serving) preempts one slice's two balloons."""
    ns = BEDS[bed]["ns"]
    return {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": "lld-balloon", "namespace": ns},
            "spec": {"replicas": 2 * n_slices, "selector": {"matchLabels": {"app": "lld-balloon"}},
                     "template": {"metadata": {"labels": {"app": "lld-balloon", "lld/bed": bed}},
                                  "spec": {"priorityClassName": "lld-balloon", "terminationGracePeriodSeconds": 0,
                                           "nodeSelector": {"lld/bed": bed, "lld/role": "standby",
                                                            "cloud.google.com/gke-tpu-accelerator": "tpu-v5-lite-podslice",
                                                            "cloud.google.com/gke-tpu-topology": "2x4"},
                                           "tolerations": [{"key": "google.com/tpu", "operator": "Exists",
                                                            "effect": "NoSchedule"}],
                                           "containers": [{"name": "balloon", "image": IMAGE,
                                                           "command": ["sleep", "infinity"],
                                                           "resources": {"requests": {"google.com/tpu": "4"},
                                                                         "limits": {"google.com/tpu": "4"}}}]}}}}


def loadgen_pod(bed: str) -> dict:
    ns = BEDS[bed]["ns"]
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "lld-loadgen", "namespace": ns,
                                                            "labels": {"app": "lld-loadgen"}},
            "spec": {"nodeSelector": {"cloud.google.com/gke-nodepool": "default-pool"},
                     "restartPolicy": "Always", "terminationGracePeriodSeconds": 5,
                     "containers": [{"name": "loadgen", "image": IMAGE, "command": ["/bin/bash", "-c",
                                                                                      "mkdir -p /opt/lld /data && sleep infinity"],
                                     "env": [{"name": "PYTHONUNBUFFERED", "value": "1"}],
                                     "resources": {"requests": {"cpu": "2", "memory": "4Gi"},
                                                   "limits": {"cpu": "6", "memory": "12Gi"}},
                                     "volumeMounts": [{"name": "data", "mountPath": "/data"}]}],
                     "volumes": [{"name": "data", "emptyDir": {}}]}}


def setup(bed: str) -> None:
    ns = BEDS[bed]["ns"]
    kubectl("create", "namespace", ns, check=False)
    apply([BALLOON_PRIORITY, SERVING_PRIORITY])
    rev = git_rev()
    apply([code_configmap(ns, rev)], server_side=True)
    router_install(bed)
    apply([loadgen_pod(bed)])
    if bed == "s":
        apply([balloons(bed, 3)])


# ---------------------------------------------------------------- LWS / HPA

def pod_spec(bed: str, arm: str, role: str, rev: str) -> dict:
    ns = BEDS[bed]["ns"]
    field = lambda p: {"valueFrom": {"fieldRef": {"fieldPath": p}}}  # noqa: E731
    env = [{"name": "POD_NAME", **field("metadata.name")}, {"name": "POD_UID", **field("metadata.uid")},
           {"name": "NODE_NAME", **field("spec.nodeName")}, {"name": "NODE_IP", **field("status.hostIP")},
           {"name": "VLLM_HOST_IP", **field("status.hostIP")},
           {"name": "LLD_WORKER_INDEX", **field("metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']")},
           {"name": "LLD_ROLE", "value": role}, {"name": "LLD_ARM", "value": arm},
           {"name": "LLD_GCS_ROOT", "value": GCS_ROOT}, {"name": "LLD_EVIDENCE", "value": f"{EVIDENCE}/{ns}"},
           # as the AutoscaleBurst / tpu_e2e worker group
           {"name": "TPU_MULTIHOST_BACKEND", "value": "ray"},
           {"name": "VBAR_CONTROL_SERVICE_URL", "value": "$(NODE_IP):8353"},
           {"name": "JAX_PLATFORMS", "value": "tpu,cpu"}, {"name": "ENABLE_PJRT_COMPATIBILITY", "value": "true"},
           {"name": "HF_HOME", "value": "/.cache/huggingface"}]
    start = ("export LLD_T_START=$(date +%s.%N); ulimit -n 65536; mkdir -p /opt/lld/code && "
             f"tar xzf /opt/lld-src/code.tar.gz -C /opt/lld/code && exec python3 -u /opt/lld/code/scripts/lld_pod.py {role}")
    tolerations = [{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"},
                   {"key": "cloud.google.com/gke-spot", "operator": "Exists", "effect": "NoSchedule"}]
    return {
        "hostNetwork": True, "dnsPolicy": "ClusterFirstWithHostNet", "priorityClassName": "lld-serving",
        "terminationGracePeriodSeconds": 1800,
        "nodeSelector": {"lld/bed": bed, "cloud.google.com/gke-tpu-accelerator": "tpu-v5-lite-podslice",
                         "cloud.google.com/gke-tpu-topology": "2x4"},
        "tolerations": tolerations,
        "containers": [{
            "name": "engine", "image": IMAGE, "imagePullPolicy": "IfNotPresent",
            "command": ["/bin/bash", "-c", start], "env": env,
            "ports": [{"containerPort": 8000, "name": "http"}, {"containerPort": 8101, "name": "lld-agent"}],
            "resources": {"requests": {"google.com/tpu": "4"}, "limits": {"google.com/tpu": "4"}},
            # leader: /health of the replica's engine on its own host (direct, or the pass-through when the
            # instance's coordinator is the other host) = the InferencePool endpoint + the HPA's ready count;
            # worker: mirrors the leader's serving state (agent :8101/lld/ready)
            "readinessProbe": {"httpGet": {"path": "/health" if role == "leader" else "/lld/ready",
                                           "port": 8000 if role == "leader" else 8101},
                               "periodSeconds": 1, "timeoutSeconds": 1, "failureThreshold": 2,
                               "successThreshold": 1},
            "lifecycle": {"preStop": {"exec": {"command": ["python3", "/opt/lld/code/scripts/lld_pod.py",
                                                           "prestop"]}}},
            "volumeMounts": [{"name": "gcs-model-weights", "mountPath": "/models"},
                             {"name": "dshm", "mountPath": "/dev/shm"},
                             {"name": "code", "mountPath": "/opt/lld-src"}],
        }],
        "volumes": [{"name": "gcs-model-weights", "csi": {"driver": "gcsfuse.csi.storage.gke.io",
                                                          "volumeAttributes": {"bucketName": BUCKET,
                                                                               "mountOptions": "implicit-dirs"}}},
                    {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "170Gi"}},
                    {"name": "code", "configMap": {"name": f"lld-code-{rev}"}}],
    }


def lws_manifest(bed: str, arm: str, replicas: int, rev: str) -> dict:
    ns = BEDS[bed]["ns"]
    labels = {"lld/serving": "true", "lld/bed": bed, "lld/arm": arm, "lld/rev": rev}
    tmpl = lambda role: {"metadata": {"labels": {**labels, "lld/role": role},  # noqa: E731
                                      "annotations": {"gke-gcsfuse/volumes": "true"}},
                         "spec": pod_spec(bed, arm, role, rev)}
    return {"apiVersion": "leaderworkerset.x-k8s.io/v1", "kind": "LeaderWorkerSet",
            "metadata": {"name": "lld", "namespace": ns, "labels": labels,
                         "annotations": {"leaderworkerset.sigs.k8s.io/exclusive-topology": "cloud.google.com/gke-nodepool"}},
            "spec": {"replicas": replicas, "startupPolicy": "LeaderCreated",
                     "leaderWorkerTemplate": {"size": 2, "restartPolicy": "RecreateGroupOnPodRestart",
                                              "leaderTemplate": tmpl("leader"), "workerTemplate": tmpl("worker")}}}


RUNNING_TARGET = 4.0  # in-engine requests per replica at C (calibration: mean 4.05 at 0.7 req/s, cap 8)


def hpa_manifest(bed: str, target: float, up_stab: int = 0, down_stab: int = 300, max_replicas: int = 4,
                 running_target: float | None = RUNNING_TARGET) -> dict:
    """llm-d guides/workload-autoscaling/README.hpa-epp.md: the dual-metric HPA (external epp_queue_size, target
    type Value; external epp_running_requests, AverageValue), scale-up stabilization 0 + 100 %/15 s, scale-down
    300 s (EPP example) + 100 %/15 s.  Targets from the calibration (README)."""
    ns = BEDS[bed]["ns"]
    return {"apiVersion": "autoscaling/v2", "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": "lld", "namespace": ns},
            "spec": {"scaleTargetRef": {"apiVersion": "leaderworkerset.x-k8s.io/v1", "kind": "LeaderWorkerSet",
                                        "name": "lld"},
                     "minReplicas": 1, "maxReplicas": max_replicas,
                     "metrics": [{"type": "External", "external": {
                         "metric": {"name": "epp_queue_size", "selector": {"matchLabels": {"inference_pool": f"lld-{bed}"}}},
                         "target": {"type": "Value", "value": str(target)}}}] + ([{"type": "External", "external": {
                         "metric": {"name": "epp_running_requests", "selector": {"matchLabels": {"job": f"lld-{bed}-epp"}}},
                         "target": {"type": "AverageValue", "averageValue": str(running_target)}}}] if running_target else []),
                     "behavior": {"scaleUp": {"stabilizationWindowSeconds": up_stab,
                                              "policies": [{"type": "Percent", "value": 100, "periodSeconds": 15}]},
                                  "scaleDown": {"stabilizationWindowSeconds": down_stab,
                                                "policies": [{"type": "Percent", "value": 100, "periodSeconds": 15}]}}}}


def adapter_values() -> dict:
    """README.hpa-epp.md step 2 rule, with the pool picked by the HPA's metric selector (<<.LabelMatchers>>)."""
    return {"prometheus": {"url": PROM_URL, "port": 9090},
            "rules": {"default": False, "external": [{
                "seriesQuery": "llm_d_epp_flow_control_queue_size",
                "resources": {"overrides": {"namespace": {"resource": "namespace"}}, "namespaced": False},
                "name": {"as": "epp_queue_size"},
                "metricsQuery": "sum(<<.Series>>{<<.LabelMatchers>>})"}, {
                # the guide filters top_level_controller_name=<pool>-epp; v0.9.0's series carry job=<release>-epp
                "seriesQuery": "inference_objective_running_requests",
                "resources": {"overrides": {"namespace": {"resource": "namespace"}}, "namespaced": False},
                "name": {"as": "epp_running_requests"},
                "metricsQuery": "sum(<<.Series>>{<<.LabelMatchers>>})"}]}}


def adapter() -> None:
    vals = RESULTS / "manifests/prometheus-adapter-values.json"
    vals.parent.mkdir(parents=True, exist_ok=True)
    vals.write_text(json.dumps(adapter_values(), indent=1))
    sh(["helm", "upgrade", "--install", "prometheus-adapter", "prometheus-community/prometheus-adapter", "-n", MON_NS,
        "-f", str(vals), "--wait", "--timeout", "5m"], timeout=600)


def lws_pods(bed: str) -> list[dict]:
    return kjson("-n", BEDS[bed]["ns"], "get", "pods", "-l", "leaderworkerset.sigs.k8s.io/name=lld")["items"]


def wait_ready_groups(bed: str, n: int, timeout_s: float = 3600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ready = [p for p in lws_pods(bed) if any(c["type"] == "Ready" and c["status"] == "True"
                                                 for c in p["status"].get("conditions", []))
                 and not p["metadata"].get("deletionTimestamp")
                 and p["metadata"]["labels"].get("leaderworkerset.sigs.k8s.io/worker-index") == "0"]
        if len(ready) >= n:
            log(f"{bed}: {len(ready)} serving pod(s) ready after {time.time() - t0:.0f} s")
            return
        time.sleep(5)
    raise RuntimeError(f"{bed}: {n} ready serving pods not reached in {timeout_s} s")


def lws(bed: str, arm: str, replicas: int, wait: bool) -> None:
    rev = git_rev()
    ns = BEDS[bed]["ns"]
    if not kubectl("-n", ns, "get", "configmap", f"lld-code-{rev}", check=False, timeout=60).strip():
        apply([code_configmap(ns, rev)], server_side=True)
    apply([lws_manifest(bed, arm, replicas, rev)])
    if wait:
        wait_ready_groups(bed, replicas)


# ---------------------------------------------------------------- observation

def _cond(obj: dict) -> dict:
    return {c["type"]: [c["status"], c.get("lastTransitionTime")] for c in obj.get("status", {}).get("conditions", [])}


def pod_row(p: dict) -> dict:
    md, st = p["metadata"], p.get("status", {})
    cs = (st.get("containerStatuses") or [{}])
    eng = next((c for c in cs if c.get("name") == "engine"), cs[0])
    lab = md.get("labels", {})
    return {"name": md["name"], "uid": md["uid"], "created": md["creationTimestamp"],
            "deleted": md.get("deletionTimestamp"), "node": p["spec"].get("nodeName"), "phase": st.get("phase"),
            "group": lab.get("leaderworkerset.sigs.k8s.io/group-index"),
            "worker": lab.get("leaderworkerset.sigs.k8s.io/worker-index"), "app": lab.get("app"),
            "cond": _cond(p), "nominated": st.get("nominatedNodeName"),
            "started": ((eng.get("state") or {}).get("running") or {}).get("startedAt"),
            "ready": eng.get("ready"), "restarts": eng.get("restartCount")}


class Watcher:
    """Polls the bed's k8s state into <run>/k8s/*.jsonl (receipt time t): pods/LWS/HPA every 2 s, the external
    metric every 2 s, the bed's nodes every 5 s, events (namespace + kube-system autoscaler) every 30 s."""

    def __init__(self, bed: str, out: Path):
        self.bed, self.ns, self.out = bed, BEDS[bed]["ns"], out / "k8s"
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.seen_events: set = set()
        self.threads = [threading.Thread(target=self._loop, args=(f, s), daemon=True)
                        for f, s in ((self.state, 2.0), (self.metric, 2.0), (self.nodes, 5.0), (self.events, 30.0))]

    def start(self) -> "Watcher":
        for t in self.threads:
            t.start()
        return self

    def close(self) -> None:
        self.events()
        self.stop.set()
        for t in self.threads:
            t.join(timeout=60)

    def _loop(self, fn, period: float) -> None:
        while not self.stop.is_set():
            t = time.time()
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                self._write("errors", {"t": t, "fn": fn.__name__, "error": repr(exc)[:300]})
            self.stop.wait(max(0.0, period - (time.time() - t)))

    def _write(self, name: str, row: dict) -> None:
        with (self.out / f"{name}.jsonl").open("a") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    def state(self) -> None:
        t = time.time()
        d = json.loads(kubectl("-n", self.ns, "get", "pods,leaderworkersets,hpa", "-o", "json", timeout=30))
        row = {"t": t, "pods": [], "lws": None, "hpa": None}
        for o in d["items"]:
            if o["kind"] == "Pod":
                row["pods"].append(pod_row(o))
            elif o["kind"] == "LeaderWorkerSet":
                row["lws"] = {"spec_replicas": o["spec"].get("replicas"), "status": o.get("status", {}).get("replicas"),
                              "ready": o.get("status", {}).get("readyReplicas"), "rv": o["metadata"]["resourceVersion"]}
            elif o["kind"] == "HorizontalPodAutoscaler":
                s = o.get("status", {})
                row["hpa"] = {"current": s.get("currentReplicas"), "desired": s.get("desiredReplicas"),
                              "last_scale": s.get("lastScaleTime"), "metrics": s.get("currentMetrics"),
                              "cond": _cond(o)}
        self._write("state", row)

    def metric(self) -> None:
        t = time.time()
        raw = kubectl("get", "--raw", f"/apis/external.metrics.k8s.io/v1beta1/namespaces/{self.ns}/epp_queue_size"
                      f"?labelSelector=inference_pool%3Dlld-{self.bed}", check=False, timeout=20)
        try:
            items = json.loads(raw).get("items", [])
            row = {"t": t, "value": items[0]["value"] if items else None,
                   "ts": items[0].get("timestamp") if items else None}
        except Exception:  # noqa: BLE001
            row = {"t": t, "error": raw[:200]}
        self._write("metric", row)

    def nodes(self) -> None:
        t = time.time()
        d = kjson("get", "nodes", "-l", f"lld/bed={self.bed}")
        self._write("nodes", {"t": t, "nodes": [
            {"name": n["metadata"]["name"], "pool": n["metadata"]["labels"].get("cloud.google.com/gke-nodepool"),
             "created": n["metadata"]["creationTimestamp"], "deleted": n["metadata"].get("deletionTimestamp"),
             "ready": _cond(n).get("Ready"), "unschedulable": n["spec"].get("unschedulable"),
             "taints": [x.get("key") for x in n["spec"].get("taints", [])]} for n in d["items"]]})

    def events(self) -> None:
        for ns in (self.ns, "kube-system"):
            d = kjson("-n", ns, "get", "events")
            for e in d["items"]:
                key = (e["metadata"]["uid"], e.get("count"), e.get("series", {}).get("count"))
                if key in self.seen_events:
                    continue
                if ns == "kube-system" and e.get("source", {}).get("component") != "cluster-autoscaler" \
                        and "autoscaler" not in (e.get("reportingComponent") or ""):
                    continue
                self.seen_events.add(key)
                self._write("events", {"ns": ns, "first": e.get("firstTimestamp"), "last": e.get("lastTimestamp"),
                                       "event_time": e.get("eventTime"), "count": e.get("count"),
                                       "reason": e.get("reason"), "type": e.get("type"),
                                       "obj": f"{e['involvedObject'].get('kind')}/{e['involvedObject'].get('name')}",
                                       "msg": (e.get("message") or "")[:500],
                                       "src": e.get("source", {}).get("component") or e.get("reportingComponent")})


def prom_query_range(query: str, t0: float, t1: float, step: float = 1.0) -> dict:
    from urllib.parse import urlencode  # noqa: PLC0415
    q = urlencode({"query": query, "start": f"{t0:.3f}", "end": f"{t1:.3f}", "step": f"{step}"})
    return json.loads(kubectl("get", "--raw", f"/api/v1/namespaces/{MON_NS}/services/http:llmd-kube-prometheus-stack"
                              f"-prometheus:9090/proxy/api/v1/query_range?{q}", timeout=120))


def prom_raw(series: str, t_end: float, window_s: int) -> dict:
    """Raw scraped samples (exact scrape timestamps) of a series over [t_end - window_s, t_end]."""
    from urllib.parse import urlencode  # noqa: PLC0415
    q = urlencode({"query": f"{series}[{int(window_s)}s]", "time": f"{t_end:.3f}"})
    return json.loads(kubectl("get", "--raw", f"/api/v1/namespaces/{MON_NS}/services/http:llmd-kube-prometheus-stack"
                              f"-prometheus:9090/proxy/api/v1/query?{q}", timeout=120))


# ---------------------------------------------------------------- runs

TRACES = {"a": Path.home() / "hetero-wt/AutoscaleBurst/results/autoscale-burst/4daec1af/traces/trace-a.json",
          "b": Path.home() / "hetero-wt/AutoscaleBurst/results/autoscale-burst/4daec1af/traces/trace-b.json"}
SEED, CLIENT_TIMEOUT_S = 20261025, 2700  # AutoscaleBurst 4daec1af/config.json
# arm -> (bed, engine boot, initial replicas, hpa)
ARMS = {"s-pm": ("s", "pm", 1, True), "s-stock": ("s", "stock", 1, True),
        "n-pm": ("n", "pm", 1, True), "n-stock": ("n", "stock", 1, True),
        "static-base": ("s", "pm", 1, False), "static-peak": ("s", "pm", 4, False),
        "s-pm-stab120": ("s", "pm", 1, True), "s-stock-stab120": ("s", "stock", 1, True)}


def kexec(bed: str, script: str, check: bool = False, timeout: float = 120) -> str:
    cmd = ["kubectl", "-n", BEDS[bed]["ns"], "exec", "lld-loadgen", "--", "bash", "-c", script]
    return sh(cmd, check=check, timeout=timeout, quiet=True)


def kcp(bed: str, src: str, dst: str) -> None:
    kubectl("-n", BEDS[bed]["ns"], "cp", src, f"lld-loadgen:{dst}", timeout=300)


def delete_lws(bed: str, timeout_s: float = 2400) -> None:
    """Delete HPA + LWS; every group's preStop runs its greedy check; wait until all LWS pods are gone."""
    ns = BEDS[bed]["ns"]
    kubectl("-n", ns, "delete", "hpa", "lld", "--ignore-not-found", "--wait=false")
    kubectl("-n", ns, "delete", "leaderworkerset", "lld", "--ignore-not-found", "--wait=false")
    t = time.time()
    while lws_pods(bed):
        if time.time() - t > timeout_s:
            raise RuntimeError(f"{bed}: LWS pods still present after {timeout_s} s")
        time.sleep(5)
    log(f"{bed}: LWS pods gone after {time.time() - t:.0f} s")


def bed_quiet(bed: str, timeout_s: float = 3600) -> dict:
    """Bed back to its idle state: s = 6 balloon pods running; n = new-capacity pools at 0 nodes."""
    t = time.time()
    while True:
        if bed == "s":  # one balloon per standby node
            pods = kjson("-n", BEDS[bed]["ns"], "get", "pods", "-l", "app=lld-balloon")["items"]
            want = len(kjson("get", "nodes", "-l", f"lld/bed={bed},lld/role=standby")["items"])
            ok = sum(1 for p in pods if p["status"].get("phase") == "Running"
                     and not p["metadata"].get("deletionTimestamp")) >= want
        else:
            nodes = kjson("get", "nodes", "-l", f"lld/bed={bed},lld/role=new")["items"]
            ok = not nodes
        if ok:
            return {"waited_s": round(time.time() - t, 1)}
        if time.time() - t > timeout_s:
            raise RuntimeError(f"bed {bed} not idle after {timeout_s} s")
        time.sleep(10)


def wait_hpa_active(bed: str, timeout_s: float = 300) -> dict:
    t = time.time()
    while time.time() - t < timeout_s:
        h = kjson("-n", BEDS[bed]["ns"], "get", "hpa", "lld")
        c = _cond(h)
        if c.get("ScalingActive", [""])[0] == "True" and h.get("status", {}).get("currentMetrics"):
            return {"waited_s": round(time.time() - t, 1), "metrics": h["status"]["currentMetrics"]}
        time.sleep(3)
    raise RuntimeError(f"HPA of bed {bed} not active after {timeout_s} s: {c}")


def collect_evidence(bed: str, out: Path, pod_names: set[str]) -> list[str]:
    ns = BEDS[bed]["ns"]
    prefix = f"{EVIDENCE}/{ns}"
    names = sh(["gcloud", "storage", "ls", f"{prefix}/"], check=False).split()
    got = []
    for url in names:
        leaf = url.rstrip("/").rsplit("/", 1)[-1]
        if leaf in pod_names:  # <pod>-<uid8>
            dst = out / "pods" / leaf
            dst.mkdir(parents=True, exist_ok=True)
            sh(["gcloud", "storage", "cp", "-r", url.rstrip("/") + "/*", str(dst)], check=False, timeout=600)
            got.append(leaf)
    return got


ACTIVE = Path("/tmp/lld-active")  # one file per bed with its running rid (co-tenant record)


def cotenants(bed: str) -> dict:
    return {f.name: f.read_text() for f in sorted(ACTIVE.glob("*")) if f.name != bed} if ACTIVE.exists() else {}


def run(arm: str, rep: int, trace: str, target: float, tail_s: float, limit: int | None = None,
        time_scale: float = 1.0, tag: str = "", trace_file: str | None = None, max_replicas: int = 4,
        bed_override: str | None = None) -> dict:
    bed, boot, replicas0, use_hpa = ARMS[arm]
    if bed_override:  # static arms only: no scale-up, so the bed only changes which pools hold the slices
        if use_hpa:
            raise ValueError(f"{arm}: the bed of an autoscaled arm is fixed")
        bed = bed_override
    ns = BEDS[bed]["ns"]
    rev = git_rev()
    rid = f"{trace}-{arm}-r{rep}{tag}"
    out = (RESULTS / "smoke" / f"{rid}-{rev}") if "smoke" in tag else (RESULTS / rev / "runs" / rid)
    if (out / "run.json").exists():
        log(f"{rid}: exists, skipping")
        return json.loads((out / "run.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    up_stab = 120 if arm.endswith("stab120") else 0
    tpath = Path(trace_file) if trace_file else TRACES[trace]
    rec: dict = {"rid": rid, "arm": arm, "bed": bed, "boot": boot, "trace": trace, "rep": rep, "rev": rev,
                 "trace_file": str(tpath), "max_replicas": max_replicas,
                 "dirty": git_dirty(), "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                                     capture_output=True, text=True).stdout.strip(),
                 "replicas0": replicas0, "hpa": None, "status": "running", "marks": {}}
    mark = lambda k: rec["marks"].__setitem__(k, time.time())  # noqa: E731
    try:
        mark("prep_start")
        delete_lws(bed)
        rec["bed_quiet"] = bed_quiet(bed)
        mark("bed_quiet")
        lws(bed, boot, replicas0, wait=True)
        mark("base_ready")
        if use_hpa:
            h = hpa_manifest(bed, target, up_stab, 300, max_replicas)
            apply([h])
            rec["hpa"] = h["spec"]
            rec["hpa_active"] = wait_hpa_active(bed)
        mark("hpa_active")
        ACTIVE.mkdir(exist_ok=True)
        (ACTIVE / bed).write_text(rid)
        rec["cotenants_start"] = cotenants(bed)
        # loadgen: client + EPP sampler (same pod, same clock)
        kexec(bed, f"mkdir -p /opt/lld /data/{rid}; pkill -f asb_loadgen.py; pkill -f lld_sampler.py; true")
        kcp(bed, str(ROOT / "scripts/asb_loadgen.py"), "/opt/lld/asb_loadgen.py")
        kcp(bed, str(ROOT / "scripts/lld_sampler.py"), "/opt/lld/lld_sampler.py")
        kcp(bed, str(tpath), f"/data/{rid}/trace.json")
        tr = json.loads(tpath.read_text())
        reqs = tr["requests"][:limit] if limit else tr["requests"]
        dur = float(reqs[-1]["t"]) * time_scale
        start_at = time.time() + 15.0
        rec["client_start_at"] = start_at
        epp = f"http://lld-{bed}-epp.{ns}.svc.cluster.local"
        watcher = Watcher(bed, out).start()
        samp = ["python3", "-u", "/opt/lld/lld_sampler.py", "--url", f"{epp}:9090/metrics",
                "--out", f"/data/{rid}/samples.jsonl", "--until", f"{start_at + dur + tail_s + 1200:.0f}"]
        kexec(bed, f"nohup {shlex.join(samp)} > /data/{rid}/sampler.log 2>&1 &", check=True)
        cmd = ["python3", "-u", "/opt/lld/asb_loadgen.py", "client", "--router", f"{epp}:80",
               "--trace", f"/data/{rid}/trace.json", "--out", f"/data/{rid}/client.jsonl", "--run-id", rid,
               "--start-at", f"{start_at:.3f}", "--seed", str(SEED), "--timeout-s", str(CLIENT_TIMEOUT_S),
               "--model", "/models/gemma-4-31b", "--time-scale", str(time_scale)]
        if limit:
            cmd += ["--limit", str(limit)]
        kexec(bed, f"nohup {shlex.join(cmd)} > /data/{rid}/client.log 2>&1 &", check=True)
        log(f"{rid}: client starts at {start_at:.0f} ({dur:.0f} s of trace), HPA target {target if use_hpa else '-'}")
        while kexec(bed, f"test -f /data/{rid}/client.jsonl.summary.json && echo yes").strip() != "yes":
            if time.time() > start_at + dur + CLIENT_TIMEOUT_S + 600:
                raise RuntimeError("client did not finish")
            time.sleep(15)
        mark("client_done")
        rec["cotenants_end"] = cotenants(bed)
        # tail: until the HPA is back at the initial size (scale-down cooldown visible) or tail_s
        t_tail = time.time()
        while use_hpa and time.time() - t_tail < tail_s:
            groups = {p["metadata"]["labels"].get("leaderworkerset.sigs.k8s.io/group-index") for p in lws_pods(bed)}
            spec = kjson("-n", ns, "get", "leaderworkerset", "lld")["spec"]["replicas"]
            if spec <= replicas0 and len(groups) <= replicas0:
                break
            time.sleep(10)
        mark("tail_done")
        rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["status"], rec["error"] = "failed", repr(exc)[:2000]
        log(f"{rid}: FAILED {exc!r}")
    finally:
        pods_seen: set[str] = set()
        (ACTIVE / bed).unlink(missing_ok=True)
        st = out / "k8s/state.jsonl"
        try:
            mark("teardown_start")
            delete_lws(bed)  # base (+ any remaining replica) greedy-checked in preStop
            mark("teardown_done")
        except Exception as exc:  # noqa: BLE001
            rec["teardown_error"] = repr(exc)[:500]
        if "watcher" in locals():
            if bed == "n":
                try:  # new capacity: keep watching until the cluster autoscaler removed the new slices
                    rec["nodes_released"] = bed_quiet(bed, timeout_s=2400)
                    mark("nodes_released")
                except Exception as exc:  # noqa: BLE001
                    rec["nodes_release_error"] = repr(exc)[:300]
            watcher.close()
        if st.exists():
            for line in st.read_text().splitlines():
                pods_seen |= {f'{p["name"]}-{p["uid"][:8]}' for p in json.loads(line)["pods"] if p.get("worker") is not None}
        rec["pods"] = sorted(pods_seen)
        try:
            lg = out / "loadgen"
            lg.mkdir(exist_ok=True)
            tar = subprocess.Popen(["kubectl", "-n", ns, "exec", "lld-loadgen", "--", "tar", "czf", "-", "-C",
                                    f"/data/{rid}", "."], stdout=subprocess.PIPE, env=kenv())
            subprocess.run(["tar", "xzf", "-", "-C", str(lg)], stdin=tar.stdout, check=True)
            tar.wait(timeout=600)
        except Exception as exc:  # noqa: BLE001
            rec["loadgen_fetch_error"] = repr(exc)[:300]
        rec["evidence"] = collect_evidence(bed, out, pods_seen)
        try:
            t0, t1 = rec["marks"].get("hpa_active", rec["marks"]["prep_start"]), time.time()
            win = int(t1 - t0) + 60
            (out / "prom").mkdir(exist_ok=True)
            for name, series in (("epp_queue", f'llm_d_epp_flow_control_queue_size{{inference_pool="lld-{bed}"}}'),
                                 ("ready_pods", f'inference_pool_ready_pods{{name="lld-{bed}"}}')):
                (out / "prom" / f"{name}.json").write_text(json.dumps(prom_raw(series, t1, win)))
        except Exception as exc:  # noqa: BLE001
            rec["prom_error"] = repr(exc)[:300]
        mark("end")
        (out / "run.json").write_text(json.dumps(rec, indent=1, default=str))
        for f in out.rglob("*.jsonl"):
            if f.stat().st_size > 200_000:
                subprocess.run(["gzip", "-f", str(f)], check=False)
        log(f"{rid}: {rec['status']} -> {out}")
    return rec


def loadtest(bed: str, name: str, trace: Path, time_scale: float = 1.0, limit: int | None = None,
             sequential: bool = False) -> Path:
    """Calibration through the llm-d chain (client -> Envoy/EPP -> serving pod), as asb_run loadtest: one client
    run + the 1-s EPP sampler; results/llmd-autoscale/calibration/<name>/."""
    ns = BEDS[bed]["ns"]
    out = RESULTS / "calibration" / name
    out.mkdir(parents=True, exist_ok=True)
    rid = f"cal-{name}"
    kexec(bed, f"mkdir -p /opt/lld /data/{rid}; pkill -f asb_loadgen.py; pkill -f lld_sampler.py; true")
    kcp(bed, str(ROOT / "scripts/asb_loadgen.py"), "/opt/lld/asb_loadgen.py")
    kcp(bed, str(ROOT / "scripts/lld_sampler.py"), "/opt/lld/lld_sampler.py")
    kcp(bed, str(trace), f"/data/{rid}/trace.json")
    epp = f"http://lld-{bed}-epp.{ns}.svc.cluster.local"
    samp = ["python3", "-u", "/opt/lld/lld_sampler.py", "--url", f"{epp}:9090/metrics", "--out",
            f"/data/{rid}/samples.jsonl", "--until", f"{time.time() + 7200:.0f}"]
    kexec(bed, f"nohup {shlex.join(samp)} > /data/{rid}/sampler.log 2>&1 &", check=True)
    cmd = ["python3", "-u", "/opt/lld/asb_loadgen.py", "client", "--router", f"{epp}:80", "--trace",
           f"/data/{rid}/trace.json", "--out", f"/data/{rid}/client.jsonl", "--run-id", rid, "--seed", str(SEED),
           "--timeout-s", str(CLIENT_TIMEOUT_S), "--model", "/models/gemma-4-31b", "--time-scale", str(time_scale)]
    if limit:
        cmd += ["--limit", str(limit)]
    if sequential:
        cmd += ["--sequential"]
    t = time.time()
    kexec(bed, f"nohup {shlex.join(cmd)} > /data/{rid}/client.log 2>&1 &", check=True)
    while kexec(bed, f"test -f /data/{rid}/client.jsonl.summary.json && echo yes").strip() != "yes":
        if time.time() - t > CLIENT_TIMEOUT_S + 3600:
            raise RuntimeError("loadtest client did not finish")
        time.sleep(10)
    kexec(bed, "pkill -f lld_sampler.py; true")
    tar = subprocess.Popen(["kubectl", "-n", ns, "exec", "lld-loadgen", "--", "tar", "czf", "-", "-C", f"/data/{rid}",
                            "."], stdout=subprocess.PIPE, env=kenv())
    subprocess.run(["tar", "xzf", "-", "-C", str(out)], stdin=tar.stdout, check=True)
    tar.wait(timeout=600)
    s = json.loads((out / "client.jsonl.summary.json").read_text())
    log(f"loadtest {name}: n={s['n']} ok={s['ok']} ttft p50/p99 {s['ttft_s']['p50']}/{s['ttft_s']['p99']} "
        f"tpot p50/p99 {s['tpot_s']['p50']}/{s['tpot_s']['p99']} errors {s['errors_by_status']}")
    return out


def commit(paths: list[Path], msg: str) -> None:
    """git add --sparse -f + commit, serialized across concurrent campaign drivers (one per bed)."""
    import fcntl  # noqa: PLC0415
    with open("/tmp/lld-git.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        subprocess.run(["git", "add", "--sparse", "-f", *[str(p) for p in paths]], cwd=ROOT, check=True)
        p = subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, capture_output=True, text=True)
        log(f"commit rc={p.returncode}: {msg[:120]} {p.stderr.strip()[-200:]}")


def campaign(bed: str, plan_path: Path) -> None:
    """Run a bed's plan in order (re-read before every run, so it can be edited live): each entry
    {"arm", "rep", "trace"="a", "target"=2, "max_replicas"=4, "tail_s"=900, "bed"=None (static arms)}; the first
    entry without a run.json is run, analysed (lld_report.py run) and committed.  A failed run moves to
    <rev>/runs-failed/<rid>-<unix> (committed) and its entry is retried at most `retries` (default 1) times.
    A file <plan>.stop ends the loop between runs."""
    rev = git_rev()
    while True:
        if plan_path.with_suffix(".stop").exists():
            log(f"campaign {bed}: stop file, exiting")
            return
        plan = json.loads(plan_path.read_text())
        todo = None
        for e in plan:
            if (e.get("bed") or ARMS[e["arm"]][0]) != bed or e.get("skip"):
                continue
            rid = f"{e.get('trace', 'a')}-{e['arm']}-r{e['rep']}"
            if (RESULTS / rev / "runs" / rid / "run.json").exists():
                continue
            failed = list((RESULTS / rev / "runs-failed").glob(f"{rid}-*"))
            if len(failed) > int(e.get("retries", 1)):
                continue
            todo = (e, rid)
            break
        if todo is None:
            log(f"campaign {bed}: plan done")
            return
        e, rid = todo
        log(f"campaign {bed}: {rid}")
        rec = run(e["arm"], int(e["rep"]), e.get("trace", "a"), float(e.get("target", 2.0)),
                  float(e.get("tail_s", 900.0)), max_replicas=int(e.get("max_replicas", 4)),
                  bed_override=e.get("bed"))
        out = RESULTS / rev / "runs" / rid
        if rec.get("status") != "ok":
            dst = RESULTS / rev / "runs-failed" / f"{rid}-{int(time.time())}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            out.rename(dst)
            subprocess.run([sys.executable, str(ROOT / "scripts/lld_report.py"), "run", str(dst)], cwd=ROOT,
                           check=False)
            commit([dst], f"llmd-autoscale {rid} FAILED (bed {bed}, code {rev}): {rec.get('error', '')[:200]}")
            continue
        p = subprocess.run([sys.executable, str(ROOT / "scripts/lld_report.py"), "run", str(out)], cwd=ROOT,
                           capture_output=True, text=True)
        brief = ""
        try:
            b = json.loads(p.stdout)
            brief = (f"burst TTFT P99 {b['burst']['ttft_p99']:.2f} s, burst SLO viol {b['burst']['slo_violation_pct']:.1f}%, "
                     f"max queue {b['queue_max']}, chip-s {b['cost']['chip_s']:.0f}, "
                     f"trigger->served {(b.get('pipeline') or {}).get('total_trigger_to_first_served')}, "
                     f"greedy {b['all_greedy_ok']} mechanism {b['all_mechanism_ok']}")
        except Exception as exc:  # noqa: BLE001
            brief = f"report failed: {exc!r} {p.stderr[-300:]}"
        log(f"{rid}: {brief}")
        commit([out], f"llmd-autoscale {rid} (bed {bed}, code {rev}): {brief}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("pools"); q.add_argument("--bed", required=True); q.add_argument("--only", default=None)
    q = sp.add_parser("setup"); q.add_argument("--bed", required=True)
    sp.add_parser("adapter")
    q = sp.add_parser("lws"); q.add_argument("--bed", required=True); q.add_argument("--arm", required=True)
    q.add_argument("--replicas", type=int, default=1); q.add_argument("--wait", action="store_true")
    q = sp.add_parser("hpa"); q.add_argument("--bed", required=True); q.add_argument("--target", type=float, required=True)
    q.add_argument("--up-stab", type=int, default=0); q.add_argument("--down-stab", type=int, default=300)
    q = sp.add_parser("manifests"); q.add_argument("--bed", required=True); q.add_argument("--arm", default="pm")
    q = sp.add_parser("run"); q.add_argument("--arm", required=True, choices=sorted(ARMS))
    q.add_argument("--rep", type=int, required=True); q.add_argument("--trace", default="a")
    q.add_argument("--target", type=float, default=4.0); q.add_argument("--tail-s", type=float, default=900.0)
    q.add_argument("--limit", type=int, default=None); q.add_argument("--time-scale", type=float, default=1.0)
    q.add_argument("--tag", default=""); q.add_argument("--trace-file", default=None)
    q.add_argument("--max-replicas", type=int, default=4); q.add_argument("--bed", default=None)
    q = sp.add_parser("campaign"); q.add_argument("--bed", required=True); q.add_argument("--plan", required=True)
    q = sp.add_parser("delete-lws"); q.add_argument("--bed", required=True)
    q = sp.add_parser("loadtest"); q.add_argument("--bed", required=True); q.add_argument("--name", required=True)
    q.add_argument("--trace", required=True); q.add_argument("--time-scale", type=float, default=1.0)
    q.add_argument("--limit", type=int, default=None); q.add_argument("--sequential", action="store_true")
    a = p.parse_args()
    if a.cmd == "pools":
        pools(a.bed, a.only.split(",") if a.only else None)
    elif a.cmd == "setup":
        setup(a.bed)
    elif a.cmd == "adapter":
        adapter()
    elif a.cmd == "lws":
        lws(a.bed, a.arm, a.replicas, a.wait)
    elif a.cmd == "hpa":
        apply([hpa_manifest(a.bed, a.target, a.up_stab, a.down_stab)])
    elif a.cmd == "run":
        rec = run(a.arm, a.rep, a.trace, a.target, a.tail_s, a.limit, a.time_scale, a.tag, a.trace_file,
                  a.max_replicas, a.bed)
        return 0 if rec.get("status") == "ok" else 1
    elif a.cmd == "campaign":
        campaign(a.bed, Path(a.plan))
    elif a.cmd == "loadtest":
        loadtest(a.bed, a.name, Path(a.trace), a.time_scale, a.limit, a.sequential)
    elif a.cmd == "delete-lws":
        delete_lws(a.bed)
    elif a.cmd == "manifests":
        print(json.dumps([lws_manifest(a.bed, a.arm, 1, git_rev()), hpa_manifest(a.bed, 4)], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
