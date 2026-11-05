#!/usr/bin/env python3
"""lld_pod: the in-pod side of one llm-d LeaderWorkerSet replica = one TP8 Gemma-4 31B engine (hetero layout
S1:small_a) on one 2x4 v5e slice (2 hosts x 4 chips, one pod per host, hostNetwork).

    lld_pod.py leader    # LWS worker index 0: geometry probe + Ray head + status agent (:8101) + boot + idle
    lld_pod.py worker    # LWS worker index 1: geometry probe + Ray worker joined to the leader's head
    lld_pod.py prestop   # preStop hook. leader: drain, check + validate (greedy W0 32 x 64), upload evidence;
                         #               worker: wait until the leader's check is done (its host holds rank 1)

The boot is the AutoscaleBurst lane boot (scripts/asb.py Lane + tpu_e2e.jax_settings), run as Ray jobs on the
group's own head instead of from the workstation through rcm_exec:
    stock: startup_state fetch --method gcs --parts xla (stock XLA seed) -> hetero.cli deploy (runai-jax-xla)
    pm:    hetero.cli deploy (gcs-ov) --state-delivery <startup_state fetch --print-remote-argv snapshot,bank,xla>
Each job gets the runtime env of the script that submits it in the workstation flow (working_dir = this code
bundle, same env vars).  The one difference: hetero_ctl adds `pip: [xds-protos]` for its gateway, which a
--partition deploy never starts (AutoscaleBurst pre-built that env on standby slices before the trigger).
Inventory = tpu_e2e.live_inventory without kubectl: chip coords from the pod-start JAX geometry probe of both
hosts, Ray node ids from the head, the 2x4 template's slice dims.

Marks go to stdout (`LLD_MARK {json}`) and to /lld/status on :8101; evidence (inventory, job payloads, marks,
check/validate) to $LLD_EVIDENCE/<pod>-<uid8>/ on GCS.
"""
from __future__ import annotations

import gzip
import http.server
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LAYOUT, INSTANCE = "S1", "small_a"
TARGET = f"{LAYOUT}:{INSTANCE}"
MODEL = "/models/gemma-4-31b"
SHAPE = ["--max-model-len", "8192", "--max-num-seqs", "512", "--max-num-batched-tokens", "2048"]
JAX_BASE = {"startup_profile": True}
AGENT_PORT = int(os.environ.get("LLD_AGENT_PORT", "8101"))
ENGINE_PORT = 8000
GEOMETRY = Path("/tmp/e2e_geometry.json")
GEOMETRY_RC = Path("/tmp/e2e_geometry.rc")
WORK = Path("/tmp/lld")
TEMPLATE = ROOT / "scripts/lld/inventory-template-2x4.json"
PROMPTS = "scripts/lld/w0-validation-prompts.json"  # snapshot_validate.submit's shipped W0[:32] file
# startup_state.submit / snapshot_validate.submit runtime env (their excludes; the bundle has no data/ results/)
SCRIPT_EXCLUDES = [".venv*", ".git", ".jj", "__pycache__", "*.pyc", "results", "**/results/**", "reports", "data"]
HETERO_EXCLUDES = [".venv*", ".git", ".jj", "formalization", "skaffold", "__pycache__", "*.pyc", "docs", "results",
                   "experiments", "reports", "**/results/**", "**/reports/**"]

# tpu_e2e.GEOMETRY_PROBE + node_name (the k8s node; tpu_e2e took it from the node list)
GEOMETRY_PROBE = r"""
import json, os, socket, time
t = time.time()
import jax
t_import = time.time()
jax.distributed.initialize()
local = jax.local_devices()
row = {"hostname": socket.gethostname(), "node_ip": os.environ.get("NODE_IP"), "node_name": os.environ.get("NODE_NAME"),
       "t_start": t, "t_import": t_import,
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

T_START = float(os.environ.get("LLD_T_START") or time.time())
POD = os.environ.get("POD_NAME", socket.gethostname())
UID8 = os.environ.get("POD_UID", "nouid")[:8]
NODE_IP = os.environ.get("NODE_IP", "127.0.0.1")
ARM = os.environ.get("LLD_ARM", "pm")
GCS_ROOT = os.environ.get("LLD_GCS_ROOT", "gs://hongtaozhang-tpu-model-cache-us-west1").rstrip("/")
EVIDENCE = os.environ.get("LLD_EVIDENCE", f"{GCS_ROOT}/lld-evidence/unknown").rstrip("/") + f"/{POD}-{UID8}"

STATE: dict = {"role": None, "pod": POD, "uid8": UID8, "node_ip": NODE_IP, "node": os.environ.get("NODE_NAME"),
               "arm": ARM, "state": "starting", "marks": [], "endpoint": None, "instance_id": None,
               "check_state": "none", "check": None, "error": None, "evidence": EVIDENCE}
LOCK = threading.Lock()


def log(msg: str) -> None:
    print(f"[lld {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mark(event: str, **kw) -> float:
    t = kw.pop("t", None) or time.time()
    row = {"event": event, "t": t, **kw}
    with LOCK:
        STATE["marks"].append(row)
    print("LLD_MARK " + json.dumps(row, default=str), flush=True)
    return t


def set_state(**kw) -> None:
    with LOCK:
        STATE.update(kw)
    if "state" in kw or "check_state" in kw:
        print("LLD_STATE " + json.dumps({k: v for k, v in kw.items()}, default=str), flush=True)


# ---------------------------------------------------------------- status agent

class Agent(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/lld/status"):
            with LOCK:
                body = json.loads(json.dumps(STATE, default=str))
            self._send(200, body)
        elif self.path.startswith("/lld/ready"):
            with LOCK:
                ok = STATE["state"] == "serving"
            self._send(200 if ok else 503, {"ready": ok})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/lld/prestop"):
            run_check_once()
            with LOCK:
                self._send(200, {"check_state": STATE["check_state"], "check": STATE["check"]})
        else:
            self._send(404, {"error": "not found"})


def start_agent() -> None:
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", AGENT_PORT), Agent)
    threading.Thread(target=srv.serve_forever, daemon=True, name="lld-agent").start()


# ---------------------------------------------------------------- helpers

def sh(cmd: list[str], check: bool = True, timeout: float | None = None, env: dict | None = None) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(ROOT))
    if check and p.returncode != 0:
        raise RuntimeError(f"{cmd[:3]} rc={p.returncode}: {p.stderr[-800:]}")
    return p.stdout


def start_geometry_probe() -> None:
    """tpu_e2e's pod-start probe, beside `ray start`; e2e_geometry.rc appears once it exited (chips free)."""
    env = {**os.environ, "E2E_GEOMETRY_PROBE": GEOMETRY_PROBE}
    subprocess.Popen(["/bin/bash", "-c", '(python3 -u -c "$E2E_GEOMETRY_PROBE" > /tmp/e2e_geometry.log 2>&1; '
                      'echo "$? $(date +%s.%N)" > /tmp/e2e_geometry.rc)'], env=env)


XDS_WHEEL = "/models/lld-wheels/xds_protos-1.84.0-py3-none-any.whl"
XDS_OK = Path("/tmp/lld-xds.ok")


def start_xds_install() -> None:
    """hetero_ctl's runtime env has pip [xds-protos==1.84.0] (hetero.gateway imports it, also in a partition
    deploy's _check_partition); AutoscaleBurst built that env before the trigger.  Here: the same wheel
    (staged in GCS) into the container's python at pod start, beside `ray start` + the geometry probe."""
    subprocess.Popen(["/bin/bash", "-c", f"pip install -q --no-deps --no-index --no-warn-script-location {XDS_WHEEL} "
                      f"> /tmp/lld-xds.log 2>&1 && touch {XDS_OK}"])


def wait_xds(timeout_s: float = 300) -> None:
    deadline = time.time() + timeout_s
    while not XDS_OK.exists():
        if time.time() > deadline:
            raise RuntimeError("xds-protos install did not finish: " + Path("/tmp/lld-xds.log").read_text()[-500:])
        time.sleep(0.1)


def gcs_put(name: str, data: bytes) -> None:
    try:
        from google.cloud import storage  # noqa: PLC0415
        bucket, _, prefix = EVIDENCE.removeprefix("gs://").partition("/")
        storage.Client().bucket(bucket).blob(f"{prefix}/{name}").upload_from_string(data, timeout=600)
    except Exception as exc:  # noqa: BLE001 - evidence is best effort, the run goes on
        log(f"evidence upload {name} failed: {exc!r}")


def put_json(name: str, obj, gz: bool = False) -> None:
    raw = json.dumps(obj, indent=1, default=str).encode()
    if gz:
        gcs_put(name + ".gz", gzip.compress(raw, 6))
    else:
        gcs_put(name, raw)


def extract_measurements(text: str) -> dict:
    """omp_rcm.ray_job.RayJobSupervisor.extract_json_measurements (copied: omp_rcm's __init__ is heavy)."""
    for m in re.finditer(r"EMPIRICAL TPU HARDWARE MEASUREMENTS \(JSON OUTPUT\)\n=+\n(\{[^\n]*\})\n", text):
        try:
            return json.loads(m.group(1))
        except Exception:  # noqa: BLE001
            pass
    for pat in (r"EMPIRICAL TPU HARDWARE MEASUREMENTS \(JSON OUTPUT\)\n=+\n(\{.*?\})\n\n\[OMP_EVENT",
                r"EMPIRICAL TPU HARDWARE MEASUREMENTS \(JSON OUTPUT\)\n=+\n(\{.*?\})\n=+"):
        m = re.search(pat, text, re.DOTALL | re.MULTILINE)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:  # noqa: BLE001
                pass
    return {}


def ray_job(name: str, argv: list[str], runtime_env: dict, timeout_s: float, driver_env: str = "") -> dict:
    """Submit one Ray job to this group's head, wait, return {status, submission_id, measurements, seconds}.
    driver_env prefixes only the driver process (the head is a TPU host here; in the AutoscaleBurst lanes the
    driver ran on a CPU head, so it must not open the TPU)."""
    from ray.job_submission import JobSubmissionClient  # noqa: PLC0415
    client = JobSubmissionClient(f"http://127.0.0.1:8265")
    t0 = mark(f"{name}_start")
    entry = (driver_env + " " if driver_env else "") + shlex.join(argv)
    sid = client.submit_job(entrypoint=entry, runtime_env=runtime_env,
                            metadata={"purpose": f"lld-{name}", "pod": POD})
    status = None
    while True:
        status = client.get_job_status(sid)
        if status.is_terminal():
            break
        if time.time() - t0 > timeout_s:
            client.stop_job(sid)
            status = "TIMEOUT"
            break
        time.sleep(0.25)
    logs = client.get_job_logs(sid)
    measured = extract_measurements(logs)
    t1 = mark(f"{name}_end", status=str(status), submission_id=sid)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f"{name}.log").write_text(logs[-2_000_000:])
    return {"name": name, "status": str(status), "submission_id": sid, "seconds": round(t1 - t0, 3),
            "measurements": measured, "log_tail": logs[-3000:]}


# ---------------------------------------------------------------- inventory (tpu_e2e.live_inventory, no kubectl)

READ_GEOMETRY = r"""
import json, os, time
deadline = time.time() + 600
while not os.path.exists("/tmp/e2e_geometry.rc"):
    if time.time() > deadline:
        raise RuntimeError("geometry probe did not finish in 600 s")
    time.sleep(0.2)
rc, t_exit = open("/tmp/e2e_geometry.rc").read().split()
if rc != "0":
    raise RuntimeError("geometry probe failed rc=" + rc + ": " + open("/tmp/e2e_geometry.log").read()[-2000:])
print(json.dumps({**json.load(open("/tmp/e2e_geometry.json")), "t_exit": float(t_exit)}))
"""


def tpu_nodes() -> list[dict]:
    import ray  # noqa: PLC0415
    return [n for n in ray.nodes() if n["Alive"] and float(n["Resources"].get("TPU", 0)) > 0]


def geometry_rows(nodes: list[dict]) -> dict[str, dict]:
    import ray  # noqa: PLC0415
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy  # noqa: PLC0415

    def read():
        import subprocess as sp  # noqa: PLC0415
        return json.loads(sp.run(["python3", "-c", READ_GEOMETRY], capture_output=True, text=True,
                                 check=True).stdout.strip().splitlines()[-1])
    fn = ray.remote(num_cpus=0, max_retries=0)(read)
    refs = {n["NodeManagerAddress"]: fn.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        n["NodeID"], soft=False)).remote() for n in nodes}
    return {ip: ray.get(ref, timeout=900) for ip, ref in refs.items()}


def build_inventory(nodes: list[dict], geo: dict[str, dict]) -> dict:
    from hetero.slice_inventory import _node_generation  # noqa: PLC0415
    template = json.loads(TEMPLATE.read_text())
    hosts = []
    for n in nodes:
        ip = n["NodeManagerAddress"]
        g = geo[ip]
        coords = [list(c) for c in g["chip_coords"]]
        box = [[min(c[a] for c in coords), max(c[a] for c in coords)] for a in range(3)]
        hosts.append({"node_ip": ip, "ray_node_id": n["NodeID"], "hostname": g.get("node_name") or g["hostname"],
                      "tpu_worker_id": int(g["tpu_worker_id"]), "tpu_name": g.get("tpu_name") or "",
                      "tpu_count": int(float(n["Resources"]["TPU"])), "chip_coords": coords,
                      "jax_process_index": g["process_index"], "jax_global_device_count": g["global_device_count"],
                      "chip_box": box,
                      "host_grid_position": [box[a][0] // template["host_chip_dims"][a] for a in range(3)]})
    hosts.sort(key=lambda h: tuple(b[0] for b in h["chip_box"])[::-1])
    want = sorted(tuple(c) for h in template["hosts"] for c in h["chip_coords"])
    got = sorted(tuple(c) for h in hosts for c in h["chip_coords"])
    if want != got or {h["jax_global_device_count"] for h in hosts} != {template["total_chips"]}:
        raise RuntimeError(f"observed slice geometry {got} differs from the 2x4 template {want}")
    rows = [{"ray_node_id": n["NodeID"], "node_ip": n["NodeManagerAddress"],
             "tpu_count": int(float(n["Resources"]["TPU"]))} for n in nodes]
    inv = {k: template[k] for k in ("slice_name", "chip_dims", "host_chip_dims", "host_grid_dims", "total_chips")}
    inv.update(hosts=hosts, ray_node_generation=_node_generation(rows), observed_at=time.time(),
               ray_session="lld-pod-probe", observed_by="pod-start JAX probe (lld_pod GEOMETRY_PROBE)",
               probe_rows=geo)
    return inv


# ---------------------------------------------------------------- boot (asb.settings_for / Lane.deploy)

def settings_for(arm: str, scope: str) -> dict:
    """asb.settings_for(arm) for arm in {runai-jax-xla, gcs-ov} with this cluster's checkpoint prefix."""
    from hetero.run_scope import scope_roots, scope_xla_dir  # noqa: PLC0415
    r = scope_roots(scope)
    roots = {"weight_snapshot_root": r["weight_snapshot_root"], "exec_bank_dir": r["exec_bank_dir"],
             "compilation_cache_dir": scope_xla_dir(scope)}
    raw = {**JAX_BASE, "weight_snapshot": "off"}
    restore = {**JAX_BASE, "weight_snapshot": "read", "weight_snapshot_verify": False, "fast_graph": True,
               "exec_bank": "read"}
    if arm == "runai-jax-xla":
        return {**raw, "compilation_cache_dir": scope_xla_dir(scope), "model": f"{GCS_ROOT}/gemma-4-31b",
                "load_format": "runai_streamer", "worker_env": {"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1.0"},
                "tokenizer": MODEL, "model_loader_extra_config": {"memory_limit": 8589934592}}
    if arm == "gcs-ov":
        return {**restore, **roots, "state_gate_timeout_s": 900.0}
    raise ValueError(arm)


def fetch_argv(inv_path: Path, scope: str, parts: str, gcs: str) -> list[str]:
    """startup_state.py fetch --print-remote-argv (the remote driver's argv, ranks resolved from the inventory)."""
    out = sh([sys.executable, "scripts/startup_state.py", "fetch", "--print-remote-argv", "--inventory",
              str(inv_path), "--method", "gcs", "--parts", parts, "--target", TARGET, "--target-scope", scope,
              "--timeout", "1800", "--gcs", gcs], timeout=120)
    return json.loads(out.strip().splitlines()[-1])


def scope_name() -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", f"lld-{POD}-{UID8}")[:120]


def boot(inv_path: Path, inv: dict) -> dict:
    scope = scope_name()
    set_state(scope=scope)
    inv_raw = inv_path.read_text()
    script_env = {"working_dir": str(ROOT), "excludes": SCRIPT_EXCLUDES, "env_vars": {"PYTHONUNBUFFERED": "1"}}
    res: dict = {"arm": ARM, "scope": scope, "jobs": {}}
    delivery: list[str] | None = None
    if ARM == "stock":
        argv = fetch_argv(inv_path, scope, "xla", f"{GCS_ROOT}/omp_asb_20261025/stock-xla")
        job = ray_job("xla_fetch", argv, script_env, 1800)
        res["jobs"]["xla_fetch"] = job
        if job["status"] != "SUCCEEDED" or job["measurements"].get("ok") is not True:
            raise RuntimeError(f"stock XLA fetch failed: {job['status']} {job['log_tail'][-600:]}")
        settings = settings_for("runai-jax-xla", scope)
    elif ARM == "pm":
        delivery = fetch_argv(inv_path, scope, "snapshot,bank,xla", f"{GCS_ROOT}/omp_asb_20261025/pm")
        settings = settings_for("gcs-ov", scope)
    else:
        raise ValueError(ARM)
    extra = ["--state-delivery", json.dumps({"argv": delivery, "timeout_s": 1800})] if delivery else []
    argv = ["python3", "-u", "-m", "hetero.cli", "deploy", "--layout", LAYOUT, "--inventory-file", str(inv_path),
            "--model", MODEL, *SHAPE, "--engine-settings", json.dumps(JAX_BASE),
            "--instance-engine-settings", json.dumps({INSTANCE: settings}), *extra,
            "--partition", "--instances", INSTANCE]
    env = {"working_dir": str(ROOT), "excludes": HETERO_EXCLUDES,
           "env_vars": {"PYTHONUNBUFFERED": "1", "OMP_HETERO_DATA_DIR": "/tmp/omp_hetero/data",
                        "OMP_HETERO_INVENTORY_JSON": inv_raw}}
    job = ray_job("deploy", argv, env, 5400, driver_env="env JAX_PLATFORMS=cpu")
    res["jobs"]["deploy"] = job
    res["settings"] = settings
    m = job["measurements"]
    if job["status"] != "SUCCEEDED" or m.get("ok") is False or not m.get("deployment"):
        raise RuntimeError(f"deploy failed: {job['status']} {job['log_tail'][-800:]}")
    desc = m["deployment"]["instances"][0]["descriptor"]
    res["endpoint"] = f"http://{desc['endpoint_host']}:{desc['endpoint_port']}"
    return res


def http_json(url: str, timeout: float = 5.0, data: bytes | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:  # noqa: BLE001
            return e.code, {}


def wait_health(endpoint: str, timeout_s: float = 900) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            code, body = http_json(endpoint + "/health", timeout=2)
            if code == 200:
                return body
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{endpoint}/health not OK after {timeout_s} s")


HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length"}


def start_proxy(upstream: str) -> None:
    """aiohttp pass-through 0.0.0.0:8000 -> the coordinator host's engine endpoint (same slice, ~0.1 ms hop)."""
    import asyncio  # noqa: PLC0415

    import aiohttp  # noqa: PLC0415
    from aiohttp import web  # noqa: PLC0415
    ready = threading.Event()

    def serve():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def main():
            session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                            timeout=aiohttp.ClientTimeout(total=None))

            async def handle(request):
                body = await request.read()
                headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
                try:
                    async with session.request(request.method, upstream + request.rel_url.raw_path_qs, data=body,
                                               headers=headers) as r:
                        data = await r.read()
                        out = {k: v for k, v in r.headers.items() if k.lower() not in HOP}
                        return web.Response(status=r.status, body=data, headers=out)
                except aiohttp.ClientError as exc:
                    return web.Response(status=502, text=f"lld proxy: {exc!r}")
            app = web.Application(client_max_size=64 << 20)
            app.router.add_route("*", "/{tail:.*}", handle)
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            await web.TCPSite(runner, "0.0.0.0", ENGINE_PORT, backlog=1024).start()
            ready.set()
            await asyncio.Event().wait()
        loop.run_until_complete(main())
    threading.Thread(target=serve, daemon=True, name="lld-proxy").start()
    if not ready.wait(timeout=60):
        raise RuntimeError("proxy did not start")


LOAD: list = []  # [t, inflight, num_running, num_waiting, served] every ~1 s while serving


def watch_engine(endpoint: str) -> None:
    """first_request_seen (the EPP routed the first request here) at 0.1 s resolution, then a 1-s load series."""
    first, last = False, 0.0
    while True:
        try:
            _, st = http_json(endpoint + "/omp/status", timeout=2)
        except Exception:  # noqa: BLE001
            st = None
        t = time.time()
        if st:
            if not first and (int(st.get("served") or 0) > 0 or int(st.get("inflight") or 0) > 0):
                mark("first_request_seen", served=st.get("served"), inflight=st.get("inflight"))
                first = True
            if t - last >= 1.0:
                LOAD.append([round(t, 2), st.get("inflight"), st.get("num_running"), st.get("num_waiting"),
                             st.get("served")])
                last = t
        if STATE["check_state"] not in ("none",):
            return
        time.sleep(0.1 if not first else 0.5)


# ---------------------------------------------------------------- check (asb Lane.check, at preStop)

CHECK_DONE = threading.Event()
CHECK_LOCK = threading.Lock()


def run_check_once() -> None:
    with CHECK_LOCK:
        if STATE["check_state"] in ("done", "failed", "skipped"):
            return
        if STATE["state"] != "serving" or not STATE.get("endpoint"):
            set_state(check_state="skipped")
            CHECK_DONE.set()
            return
        set_state(check_state="draining")
        mark("prestop_start")
        endpoint = STATE["endpoint"]
        quiet_since = None
        deadline = time.time() + 240
        while time.time() < deadline:
            try:
                _, st = http_json(endpoint + "/omp/status", timeout=3)
                busy = int(st.get("inflight", 0) or 0) + int(st.get("num_running", 0) or 0) \
                    + int(st.get("num_waiting", 0) or 0)
            except Exception:  # noqa: BLE001
                busy = 0
            if busy == 0:
                quiet_since = quiet_since or time.time()
                if time.time() - quiet_since >= 2.0:
                    break
            else:
                quiet_since = None
            time.sleep(0.5)
        mark("drained")
        set_state(check_state="checking")
        hosts = ",".join(h["node_ip"] for h in STATE["inventory_hosts"])
        out = {}
        try:
            base = ["--layout", LAYOUT, "--reserve-bytes", "0", "--instances", INSTANCE, "--hosts", hosts]
            env = {"working_dir": str(ROOT), "excludes": SCRIPT_EXCLUDES,
                   "env_vars": {"PYTHONUNBUFFERED": "1", "OMP_SNAPSHOT_MODEL": MODEL}}
            out["check"] = ray_job("check", ["python3", "-u", "scripts/snapshot_validate.py", "check", "--remote",
                                             *base], env, 900, driver_env="env JAX_PLATFORMS=cpu")
            env_v = {**env, "env_vars": {**env["env_vars"], "OMP_SNAPSHOT_MANIFEST_FILE": PROMPTS}}
            out["validate"] = ray_job("validate", ["python3", "-u", "scripts/snapshot_validate.py", "validate",
                                                   "--remote", *base], env_v, 1800, driver_env="env JAX_PLATFORMS=cpu")
            for k in ("check", "validate"):
                put_json(f"{k}.json", {"submission_id": out[k]["submission_id"], "status": out[k]["status"],
                                       "measurements": out[k]["measurements"]})
            ok = all(out[k]["status"] == "SUCCEEDED" and out[k]["measurements"].get("ok") is True
                     for k in ("check", "validate"))
            summary = {k: {"status": out[k]["status"], "ok": out[k]["measurements"].get("ok"),
                           "seconds": out[k]["seconds"]} for k in out}
            set_state(check=summary, check_state="done" if ok else "failed")
        except Exception as exc:  # noqa: BLE001
            set_state(check={"error": repr(exc)}, check_state="failed")
        mark("prestop_check_end", check_state=STATE["check_state"])
        put_json("marks.json", STATE["marks"])
        put_json("load.json", LOAD)
        CHECK_DONE.set()


# ---------------------------------------------------------------- roles

def ray_start_head() -> None:
    sh(["ray", "start", "--head", f"--node-ip-address={NODE_IP}", "--port=6379", "--dashboard-host=0.0.0.0",
        "--dashboard-port=8265", "--num-gpus=0", "--disable-usage-stats"], timeout=300)


def on_sigterm(*_):
    # preStop (check + evidence) has already returned when the kubelet sends SIGTERM; PID 1 needs a handler
    CHECK_DONE.wait(timeout=5)
    os._exit(0)


def leader() -> None:
    signal.signal(signal.SIGTERM, on_sigterm)
    set_state(role="leader")
    mark("container_start", t=T_START)
    start_agent()
    start_geometry_probe()
    start_xds_install()
    ray_start_head()
    mark("ray_head_up")
    import ray  # noqa: PLC0415
    ray.init(address="auto", logging_level="ERROR")
    while len(tpu_nodes()) < 2:
        time.sleep(0.2)
    nodes = sorted(tpu_nodes(), key=lambda n: n["NodeManagerAddress"])
    mark("ray_joined", nodes=[n["NodeManagerAddress"] for n in nodes])
    geo = geometry_rows(nodes)
    mark("geometry_ready", t_exit=max(g["t_exit"] for g in geo.values()))
    inv = build_inventory(nodes, geo)
    WORK.mkdir(parents=True, exist_ok=True)
    inv_path = WORK / "inventory.json"
    inv_path.write_text(json.dumps(inv, indent=1))
    set_state(inventory_hosts=[{"node_ip": h["node_ip"], "hostname": h["hostname"]} for h in inv["hosts"]])
    mark("inventory")
    wait_xds()
    mark("xds_ready")
    set_state(state="booting")
    try:
        res = boot(inv_path, inv)
        mark("deploy_returned", endpoint=res["endpoint"])
        health = wait_health(res["endpoint"])
        mark("engine_health_ok", endpoint=res["endpoint"])
        host = res["endpoint"].split("//")[-1].split(":")[0]
        if host != NODE_IP:
            # The instance's coordinator (its HTTP endpoint) is the other host of the slice.  The replica's
            # serving pod is the LWS leader (HPA counts Ready leaders; the InferencePool selects leaders), so
            # the leader serves :8000 on its own host as a pass-through to the coordinator.
            start_proxy(res["endpoint"])
            mark("proxy_up", upstream=res["endpoint"])
            health = wait_health(f"http://{NODE_IP}:{ENGINE_PORT}")
        set_state(endpoint=res["endpoint"], instance_id=health.get("instance_id"), state="serving",
                  serving_url=f"http://{NODE_IP}:{ENGINE_PORT}")
        mark("engine_ready", endpoint=res["endpoint"], instance_id=health.get("instance_id"),
             serving_url=f"http://{NODE_IP}:{ENGINE_PORT}", proxied=host != NODE_IP)
        threading.Thread(target=watch_engine, args=(res["endpoint"],), daemon=True).start()
    except Exception as exc:  # noqa: BLE001
        set_state(state="failed", error=repr(exc))
        mark("boot_failed", error=repr(exc)[:2000])
        put_json("marks.json", STATE["marks"])
        # stay up (NotReady, state=failed on :8101): a crash would make LWS recreate the group in a loop
        while True:
            time.sleep(3600)

    def evidence():
        put_json("inventory.json", inv)
        for name, job in res["jobs"].items():
            put_json(f"{name}.json", {"submission_id": job["submission_id"], "status": job["status"],
                                      "seconds": job["seconds"], "measurements": job["measurements"]},
                     gz=True)
        put_json("boot.json", {k: v for k, v in res.items() if k != "jobs"} | {"state": STATE["state"]})
        put_json("marks.json", STATE["marks"])
    threading.Thread(target=evidence, daemon=True).start()
    while True:
        time.sleep(3600)


def worker() -> None:
    signal.signal(signal.SIGTERM, on_sigterm)
    set_state(role="worker")
    mark("container_start", t=T_START)
    start_agent()
    start_geometry_probe()
    start_xds_install()
    leader_addr = os.environ["LWS_LEADER_ADDRESS"]
    while True:
        p = subprocess.run(["ray", "start", f"--address={leader_addr}:6379", f"--node-ip-address={NODE_IP}",
                            "--num-gpus=0", "--disable-usage-stats"], capture_output=True, text=True)
        if p.returncode == 0:
            break
        time.sleep(1.0)
    mark("ray_worker_joined", leader=leader_addr)
    set_state(state="joined")
    misses = 0
    while True:  # readiness of the worker pod = the group's (leader's) serving state, for the LWS group status
        try:
            _, st = http_json(f"http://{leader_addr}:{AGENT_PORT}/lld/status", timeout=3)
            misses = 0
            if st.get("state") in ("serving", "failed") and STATE["state"] != st["state"]:
                set_state(state=st["state"])
                mark(f"leader_{st['state']}")
        except Exception:  # noqa: BLE001
            misses += 1
        time.sleep(1.0)
    while True:
        time.sleep(3600)


def prestop() -> None:
    role = os.environ.get("LLD_ROLE") or ("leader" if os.environ.get("LLD_WORKER_INDEX", "0") == "0" else "worker")
    if role == "leader":
        http_json(f"http://127.0.0.1:{AGENT_PORT}/lld/prestop", timeout=2400, data=b"{}")
        return
    # worker: keep this host's ranks alive until the leader's check finished (or the leader is gone)
    leader_addr = os.environ.get("LWS_LEADER_ADDRESS", "")
    deadline = time.time() + 2400
    misses = 0
    while time.time() < deadline:
        try:
            _, st = http_json(f"http://{leader_addr}:{AGENT_PORT}/lld/status", timeout=3)
            misses = 0
            if st.get("check_state") in ("done", "failed", "skipped") or st.get("state") in ("failed", "starting"):
                return
        except Exception:  # noqa: BLE001
            misses += 1
            if misses >= 10:
                return
        time.sleep(1.0)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "leader"
    {"leader": leader, "worker": worker, "prestop": prestop}[cmd]()
