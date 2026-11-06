#!/usr/bin/env python3
"""One autoscale-burst run (or a campaign of runs) on a bed: lanes prepared per arm, a fresh router +
controller in the bed's loadgen pod, the open-loop trace client, the arm's actuator on every controller
trigger, then evidence, per-instance mechanism + greedy checks, and teardown.

Arms (the same reactive policy everywhere; only what a trigger does differs):
  static-base   1 instance, controller dry (decisions logged, nothing actuated)
  static-peak   base + every lane of the bed serving from the start (TPUFlex boots before the run), dry
  stock         a trigger boots stock vLLM-JAX (Run:ai streamer + stock XLA cache, = stock-runai-cc) on the next
                standby slice of the bed (joined, idle, wiped: fresh containers)
  pm-gcs        a trigger boots TPUFlex from GCS (gcs-ov) on the next standby slice
  pm-peer       a trigger boots TPUFlex from the serving base instance (xpeer-ov: rank r -> rank r TCP streams)
  pm-hot        hot standby: every standby slice holds the state in /dev/shm with workers parked at the gate; the
                trigger opens the gate (the parked driver long-polls the controller in-cluster)
  new-stock     the trigger scales an empty 2x4 pool from 0 (GKE), then the stock boot (pools recreated per run)
  new-pm        same, then the TPUFlex gcs-ov boot

    .venv/bin/python scripts/asb_run.py run --bed x --arm pm-gcs --trace a --rep 1
    .venv/bin/python scripts/asb_run.py campaign --bed x --plan results/autoscale-burst/<rev>/plan-x.json
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asb  # noqa: E402
from asb import (BEDS, LANES, NS, OUT_ROOT, PM_GCS, ROOT, STOCK_GCS, Lane, e2e, endpoint_of, gz, inv_hosts,  # noqa: E402
                 kexec, lane_inventory, load_json, log, restart_workers, router_url_incluster, router_url_local,
                 settings_for, sh, warm_lane)

ARMS = {
    "static-base": {"controller": "dry", "lanes": "none", "boot": None},
    "static-peak": {"controller": "dry", "lanes": "serving", "boot": None},
    "stock": {"controller": "live", "lanes": "standby", "boot": "stock"},
    "pm-gcs": {"controller": "live", "lanes": "standby", "boot": "pm-gcs"},
    "pm-peer": {"controller": "live", "lanes": "standby", "boot": "pm-peer"},
    "pm-hot": {"controller": "live", "lanes": "hot", "boot": "hot"},
    "new-stock": {"controller": "live", "lanes": "new", "boot": "stock"},
    "new-pm": {"controller": "live", "lanes": "new", "boot": "pm-gcs"},
}
ACTIVE = Path("/tmp/asb-active")


def cfg(rev: str) -> dict:
    """<rev>/config.json, written once from the calibration: cap, threshold, sustain_s, slo, timeouts, seed."""
    return json.loads((OUT_ROOT / rev / "config.json").read_text())


# ---------------------------------------------------------------- router (via the bed's port-forward)

class RouterAPI:
    def __init__(self, bed: str):
        self.bed, self.base = bed, router_url_local(bed)

    def req(self, method: str, path: str, body: dict | None = None, timeout: float = 15.0):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)

    def retry(self, method: str, path: str, body: dict | None = None, timeout: float = 15.0, attempts: int = 8):
        for i in range(attempts):
            try:
                return self.req(method, path, body, timeout)
            except urllib.error.HTTPError:
                raise
            except Exception:  # noqa: BLE001 - the port-forward loop restarts within seconds
                if i + 1 == attempts:
                    raise
                time.sleep(1.5)

    def state(self) -> dict:
        return self.retry("GET", "/admin/state")[1]

    def event(self, kind: str, **kw) -> None:
        try:
            self.retry("POST", "/admin/event", {"kind": kind, "t_ws": time.time(), **kw}, attempts=4)
        except Exception as exc:  # noqa: BLE001 - also kept in the run's marks
            log(f"router event {kind} not recorded: {exc!r}")

    def register(self, name: str, url: str, mode: str, **meta) -> None:
        self.retry("POST", "/admin/instances", {"name": name, "url": url, "mode": mode, "meta": meta})

    def triggers(self, after: int, wait: float = 10.0) -> list[dict]:
        status, body = self.retry("GET", f"/controller/triggers?after={after}&wait={wait}", timeout=wait + 15)
        return body if isinstance(body, list) else (body or {}).get("triggers", [])

    def close(self) -> None:
        self.retry("POST", "/controller/close", {})


# ---------------------------------------------------------------- loadgen pod

def lg_pod(bed: str) -> str:
    return f"asb-loadgen-{bed}"


def kcp(src: str, dst: str) -> None:
    sh(["kubectl", "-n", NS, "cp", src, dst], timeout=300)


def ship_loadgen(bed: str) -> None:
    kexec(lg_pod(bed), "mkdir -p /opt/asb /data", check=True)
    kcp(str(ROOT / "scripts/asb_loadgen.py"), f"{lg_pod(bed)}:/opt/asb/asb_loadgen.py")


def start_router(bed: str, rid: str, controller: dict, conf: dict) -> RouterAPI:
    pod = lg_pod(bed)
    kexec(pod, "pkill -f 'asb_loadgen.py router' ; pkill -f 'asb_loadgen.py client' ; true")
    time.sleep(1.0)
    cmd = ["python3", "-u", "/opt/asb/asb_loadgen.py", "router", "--port", "9000", "--out-dir", f"/data/{rid}/router",
           "--cap", str(conf["cap"]), "--admission-timeout-s", str(conf["admission_timeout_s"]),
           "--upstream-timeout-s", str(conf["upstream_timeout_s"]), "--controller", json.dumps(controller),
           "--run-id", rid]
    kexec(pod, f"mkdir -p /data/{rid}/router && nohup {shlex.join(cmd)} > /data/{rid}/router.log 2>&1 &",
          check=True)
    api = RouterAPI(bed)
    t = time.time()
    while True:
        try:
            api.req("GET", "/admin/state", timeout=5)
            return api
        except Exception:  # noqa: BLE001
            if time.time() - t > 120:
                raise RuntimeError(f"router of bed {bed} not reachable: "
                                   + kexec(pod, f"tail -20 /data/{rid}/router.log"))
            time.sleep(1.0)


def start_client(bed: str, rid: str, trace: Path, start_at: float, conf: dict, extra: list[str] | None = None) -> None:
    pod = lg_pod(bed)
    kcp(str(trace), f"{pod}:/data/{rid}/trace.json")
    cmd = ["python3", "-u", "/opt/asb/asb_loadgen.py", "client", "--router", "http://127.0.0.1:9000",
           "--trace", f"/data/{rid}/trace.json", "--out", f"/data/{rid}/client.jsonl", "--run-id", rid,
           "--start-at", f"{start_at:.3f}", "--seed", str(conf["seed"]), "--timeout-s", str(conf["client_timeout_s"]),
           *(extra or [])]
    kexec(pod, f"nohup {shlex.join(cmd)} > /data/{rid}/client.log 2>&1 &", check=True)


def client_done(bed: str, rid: str) -> bool:
    return kexec(lg_pod(bed), f"test -f /data/{rid}/client.jsonl.summary.json && echo yes").strip() == "yes"


def fetch_run_dir(bed: str, rid: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tar = subprocess.Popen(["kubectl", "-n", NS, "exec", lg_pod(bed), "--", "tar", "czf", "-", "-C", f"/data/{rid}",
                            "."], stdout=subprocess.PIPE, env={**asb.os.environ, "KUBECONFIG": asb.KUBECONFIG})
    subprocess.run(["tar", "xzf", "-", "-C", str(out)], stdin=tar.stdout, check=True)
    tar.wait(timeout=600)


def endpoint_health(bed: str, url: str) -> int | None:
    code = ("import urllib.request,sys\ntry:\n print(urllib.request.urlopen(sys.argv[1]+'/health',timeout=5).status)\n"
            "except Exception as e:\n print(getattr(e,'code','none'))")
    raw = kexec(lg_pod(bed), f"python3 -c {shlex.quote(code)} {shlex.quote(url)}", timeout=60).strip()
    return int(raw) if raw.isdigit() else None


# ---------------------------------------------------------------- hot standby jobs

def hot_parked(lane: str) -> dict | None:
    """The lane's running asb-hot-standby Ray job, once its log has ASB_HOT_STANDBY_PARKED."""
    from ray.job_submission import JobSubmissionClient  # noqa: PLC0415
    client = JobSubmissionClient(e2e.ray_url(lane))
    for job in client.list_jobs():
        if (job.metadata or {}).get("purpose") == "asb-hot-standby" and str(job.status) == "RUNNING":
            logs = client.get_job_logs(job.submission_id)
            if "ASB_HOT_STANDBY_PARKED" in logs:
                return {"submission_id": job.submission_id, "seen_at": time.time()}
    return None


# ---------------------------------------------------------------- one run

class Run:
    def __init__(self, bed: str, arm: str, trace: str, rep: int, rev: str):
        self.bed, self.arm, self.trace, self.rep, self.rev = bed, arm, trace, rep, rev
        self.spec = ARMS[arm]
        self.rid = f"{trace}-{arm}-r{rep}"
        self.out = OUT_ROOT / rev / "runs" / self.rid
        if (self.out / "run.json").exists():
            raise SystemExit(f"{self.rid} already has a run.json at {self.out}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.mark = e2e.Marks(self.out / "timeline.jsonl")
        self.conf = cfg(rev)
        self.base = BEDS[bed]["base"]
        self.lanes = BEDS[bed]["lanes"][:int(self.conf.get("max_lanes", 3))]
        self.lane: dict[str, dict] = {ln: {"scope": f"{self.rid}-{ln}"} for ln in self.lanes}
        self.actuations: list[dict] = []
        self.threads: list[threading.Thread] = []
        self.hot_threads: dict[str, threading.Thread] = {}
        self.router: RouterAPI | None = None
        self.rec: dict = {"rid": self.rid, "bed": bed, "arm": arm, "arm_spec": self.spec, "trace": trace, "rep": rep,
                          "rev": rev, "code_commit": asb.git_rev(), "dirty": asb.git_dirty(), "config": self.conf,
                          "base_lane": self.base, "lanes_order": self.lanes}

    def lane_dir(self, lane: str) -> Path:
        return self.out / "lanes" / lane

    # -- preparation
    def check_base(self) -> dict:
        base = json.loads((OUT_ROOT / "base" / f"{self.base}.json").read_text())
        status = endpoint_health(self.bed, base["endpoint"])
        if status != 200:
            raise RuntimeError(f"base {self.base} at {base['endpoint']} health {status}")
        return base

    def prep_lane(self, lane: str) -> None:
        L, d = self.lane[lane], self.lane_dir(lane)
        d.mkdir(parents=True, exist_ok=True)
        kind, mode = LANES[lane]["kind"], self.spec["lanes"]
        try:
            if mode == "new":
                if kind != "auto":
                    raise RuntimeError(f"{lane} is not an autoscaled lane")
                if e2e.pool_nodes(lane) or e2e.worker_pods(lane):
                    raise RuntimeError(f"{lane}: pool not at 0 nodes / worker pods exist; reset it first")
                L["autoscaler_known_s"] = e2e.wait_autoscaler_knows_pool(lane)
                L["ready"] = True
                return
            if kind == "fixed":
                L["restart"] = restart_workers(lane)
            else:  # an auto lane provisioned for a static-peak run
                t0, watch = e2e.trigger(lane)
                L["provision_t0"] = t0
                try:
                    joined = watch.wait_ray(1800)
                finally:  # also on the failure path: stop the watcher thread
                    watch.finish()
                if not joined:
                    raise RuntimeError(f"{lane}: not joined")
                L["provision_s"] = round(time.time() - t0, 1)
            L["inv"] = str(lane_inventory(lane, d))
            L["warm"] = warm_lane(lane, d / "warm.log")
            inv = Path(L["inv"])
            ln = Lane(lane, d, self.mark)
            if mode == "serving":
                ok = ln.deploy(inv, settings_for("gcs-ov", L["scope"]), delivery=ln.delivery_argv(inv, L["scope"], "gcs"))
                L["boot_ok"] = ok
                L["endpoint"] = endpoint_of(d / "deploy.json")
                if not ok:
                    raise RuntimeError(f"{lane}: static-peak boot failed")
            elif mode == "hot":
                if not ln.fetch_state(inv, L["scope"], PM_GCS):
                    raise RuntimeError(f"{lane}: hot-standby state fetch failed")
                k = self.lanes.index(lane) + 1
                args = ["scripts/asb_hot_standby.py", "--ray-url", e2e.ray_url(lane), "--inventory", str(inv),
                        "--output", str(d / "hot.json"),
                        "--gate-url", f"{router_url_incluster(self.bed)}/controller/trigger/{k}",
                        "--event-url", f"{router_url_incluster(self.bed)}/admin/event", "--lane", lane, "--",
                        "--layout", asb.JAX_LAYOUT, "--model", asb.JAX_MODEL, *asb.JAX_SHAPE,
                        "--engine-settings", json.dumps(asb.JAX_BASE),
                        "--instance-engine-settings", json.dumps({asb.JAX_INSTANCE: settings_for("hot", L["scope"])}),
                        "--partition", "--instances", asb.JAX_INSTANCE]
                th = threading.Thread(target=lambda: L.__setitem__("hot_rc", ln.step("hot_standby", args)), daemon=True)
                th.start()
                self.hot_threads[lane] = th
                t = time.time()
                while not (parked := hot_parked(lane)):
                    if not th.is_alive() or time.time() - t > 900:
                        raise RuntimeError(f"{lane}: hot standby not parked (rc={L.get('hot_rc')})")
                    time.sleep(3)
                L["parked"] = parked
                L["parked_s"] = round(time.time() - t, 1)
            L["ready"] = True
        except Exception as exc:  # noqa: BLE001
            L["error"] = repr(exc)[:1500]
            L["ready"] = False

    # -- actuation
    def actuate(self, k: int, decision: dict) -> None:
        lane = self.lanes[k - 1]
        L, d = self.lane[lane], self.lane_dir(lane)
        boot = self.spec["boot"]
        rec = {"k": k, "lane": lane, "boot": boot, "trigger_t": decision.get("t"), "seen_at": time.time(),
               "decision": decision}
        self.actuations.append(rec)
        self.mark("actuation_start", lane=lane, k=k)
        self.router.event("actuation_start", lane=lane, k=k, boot=boot)
        ln = Lane(lane, d, self.mark)
        try:
            if self.spec["lanes"] == "new":
                t0, watch = e2e.trigger(lane)
                rec["gke_t0"] = t0
                L["watch"] = watch
                self.router.event("gke_trigger", lane=lane, k=k, t=t0)
                if not watch.wait_ray(1800):
                    raise RuntimeError(f"{lane}: Ray TPU nodes not joined")
                rec["ray_joined_at"] = time.time()
                self.mark("ray_joined", lane=lane)
                L["inv"] = str(lane_inventory(lane, d))
                rec["inventory_at"] = time.time()
            inv = Path(L["inv"])
            rec["endpoint"] = f"http://{inv_hosts(inv)[0]}:8000"
            if boot != "hot":
                self.router.register(lane, rec["endpoint"], "pending", k=k, boot=boot)
            scope = L["scope"]
            if boot == "stock":
                dog = e2e.HostWatchdog(lane, d / "watchdog.jsonl")
                try:
                    ok = (ln.fetch_state(inv, scope, STOCK_GCS, parts="xla", name="xla_fetch")
                          and ln.deploy(inv, settings_for("runai-jax-xla", scope)))
                finally:
                    rec["watchdog"] = dog.finish()
            elif boot == "pm-gcs":
                ok = ln.deploy(inv, settings_for("gcs-ov", scope), delivery=ln.delivery_argv(inv, scope, "gcs"))
            elif boot == "pm-peer":
                base = json.loads((OUT_ROOT / "base" / f"{self.base}.json").read_text())
                ok = ln.deploy(inv, settings_for("xpeer-ov", scope), delivery=ln.delivery_argv(
                    inv, scope, "xpeer", xpeer_agents=base["xpeer_agents"], source_scope=base["scope"]))
            elif boot == "hot":
                self.hot_threads[lane].join(timeout=1800)
                ok = L.get("hot_rc") == 0 and asb.ok_json(d / "hot.json") and \
                    (load_json(d / "hot.json") or {}).get("measurements", {}).get("opened") is not None
            else:
                raise ValueError(boot)
            rec["deploy_ok"] = bool(ok)
            got = endpoint_of(d / ("hot.json" if boot == "hot" else "deploy.json"))
            rec["endpoint_deployed"] = got
            if got and got != rec["endpoint"] and boot != "hot":
                self.router.register(f"{lane}-ep", got, "pending", k=k, boot=boot, note="endpoint differs")
        except Exception as exc:  # noqa: BLE001
            rec["error"] = repr(exc)[:1500]
            rec["deploy_ok"] = False
        rec["done_at"] = time.time()
        self.mark("actuation_done", lane=lane, k=k, ok=rec["deploy_ok"])
        self.router.event("actuation_done", lane=lane, k=k, ok=rec["deploy_ok"])

    def actuator_loop(self, stop: threading.Event) -> None:
        seen = 0
        while not stop.is_set():
            try:
                for decision in self.router.triggers(seen, wait=5.0):
                    k = int(decision["k"])
                    if k <= seen:
                        continue
                    seen = k
                    if decision.get("mode") != "live" or decision.get("actuate") is False:
                        continue
                    if k > len(self.lanes) or not self.lane[self.lanes[k - 1]].get("ready"):
                        self.mark("trigger_without_lane", k=k)
                        continue
                    th = threading.Thread(target=self.actuate, args=(k, decision), daemon=True)
                    th.start()
                    self.threads.append(th)
            except Exception as exc:  # noqa: BLE001 - port-forward hiccup; keep polling
                self.mark("actuator_poll_error", error=repr(exc)[:200])
                time.sleep(1.0)

    # -- the run
    def execute(self) -> dict:
        conf, rec = self.conf, self.rec
        ACTIVE.mkdir(exist_ok=True)
        rec["cotenants_start"] = [json.loads(f.read_text()) for f in ACTIVE.glob("*.json") if f.stem != self.bed]
        (ACTIVE / f"{self.bed}.json").write_text(json.dumps({"rid": self.rid, "bed": self.bed, "since": time.time()}))
        self.mark("run_start", rid=self.rid)
        try:
            base = self.check_base()
            rec["base"] = {k: base.get(k) for k in ("lane", "endpoint", "scope", "booted_at")}
            ship_loadgen(self.bed)
            controller = {"mode": self.spec["controller"], "threshold": conf["threshold"], "sustain_s": conf["sustain_s"],
                          "period_s": 1.0, "max_triggers": len(self.lanes), "lanes": self.lanes}
            rec["controller"] = controller
            self.router = start_router(self.bed, self.rid, controller, conf)
            self.mark("router_started")
            self.router.register(self.base, base["endpoint"], "join", role="base")
            if self.spec["lanes"] != "none":
                preps = [threading.Thread(target=self.prep_lane, args=(ln,), daemon=True) for ln in self.lanes]
                for th in preps:
                    th.start()
                for th in preps:
                    th.join()
                self.mark("lanes_prepared", ready={ln: self.lane[ln].get("ready") for ln in self.lanes})
                bad = [ln for ln in self.lanes if not self.lane[ln].get("ready")]
                if bad:
                    raise RuntimeError(f"lanes not ready: {bad}: " + "; ".join(self.lane[b].get("error", "") for b in bad))
                if self.spec["lanes"] == "serving":
                    for ln in self.lanes:
                        self.router.register(ln, self.lane[ln]["endpoint"], "join", role="static")
                if self.spec["lanes"] == "hot":
                    for k, ln in enumerate(self.lanes, 1):
                        ep = f"http://{inv_hosts(Path(self.lane[ln]['inv']))[0]}:8000"
                        self.lane[ln]["endpoint"] = ep
                        self.router.register(ln, ep, "pending", k=k, boot="hot")
            stop = threading.Event()
            actuator = threading.Thread(target=self.actuator_loop, args=(stop,), daemon=True)
            actuator.start()
            trace = OUT_ROOT / self.rev / "traces" / f"trace-{self.trace}.json"
            start_at = time.time() + 8.0
            start_client(self.bed, self.rid, trace, start_at, conf)
            rec["client_start_at"] = start_at
            self.mark("client_started", start_at=start_at)
            t = time.time()
            while not client_done(self.bed, self.rid):
                if time.time() - t > conf["client_timeout_s"] + 1200:
                    raise RuntimeError("client did not finish")
                time.sleep(15)
            self.mark("client_done")
            self.router.close()
            self.mark("controller_closed")
            stop.set()
            actuator.join(timeout=60)
            for th in self.threads:
                th.join(timeout=2400)
            for ln, th in self.hot_threads.items():
                th.join(timeout=900)
            self.mark("actuations_done")
            time.sleep(3)
            fetch_run_dir(self.bed, self.rid, self.out / "loadgen")
            self.mark("evidence_fetched")
            rec["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "error"
            rec["error"] = repr(exc)[:3000]
            self.mark("run_error", error=repr(exc)[:300])
            try:
                if self.router:
                    self.router.close()
                fetch_run_dir(self.bed, self.rid, self.out / "loadgen")
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                kexec(lg_pod(self.bed), "pkill -f 'asb_loadgen.py' ; true")
            except Exception:  # noqa: BLE001
                pass
            rec["actuations"] = self.actuations
            self.verify_and_teardown()
            rec["lanes"] = {ln: {k: v for k, v in L.items() if k != "watch"} for ln, L in self.lane.items()}
            rec["cotenants_end"] = [json.loads(f.read_text()) for f in ACTIVE.glob("*.json") if f.stem != self.bed]
            (ACTIVE / f"{self.bed}.json").unlink(missing_ok=True)
            rec["marks"] = self.mark.rows
            (self.out / "run.json").write_text(json.dumps(rec, indent=1, default=str))
            gz(self.out)
        return rec

    def booted_lanes(self) -> list[str]:
        mode = self.spec["lanes"]
        if mode == "serving":
            return [ln for ln in self.lanes if self.lane[ln].get("boot_ok")]
        return [a["lane"] for a in self.actuations if a.get("deploy_ok")]

    def verify_and_teardown(self) -> None:
        mode, boot = self.spec["lanes"], self.spec["boot"]
        checks: dict[str, dict] = {}

        def verify(lane: str) -> None:
            d = self.lane_dir(lane)
            res = Lane(lane, d, self.mark).check(Path(self.lane[lane]["inv"]), restored=boot != "stock")
            res["mechanism"] = mechanism(d / ("hot.json" if boot == "hot" else "deploy.json"),
                                         "pm-gcs" if mode == "serving" else boot)
            checks[lane] = res

        ths = [threading.Thread(target=verify, args=(ln,), daemon=True) for ln in self.booted_lanes()]
        for th in ths:
            th.start()
        for th in ths:
            th.join(timeout=1800)
        self.rec["verification"] = checks
        # a lane that failed in prep_lane (e.g. an auto lane that never joined) still holds pending pods/nodes:
        # `provision_t0` marks it as used so it is released and its pool reset
        used = [ln for ln in self.lanes if self.lane[ln].get("ready") or self.lane[ln].get("inv")
                or self.lane[ln].get("provision_t0") or any(a["lane"] == ln for a in self.actuations)]
        teardown: dict[str, dict] = {}

        def down(lane: str) -> None:
            d = self.lane_dir(lane)
            row: dict = {}
            if self.lane[lane].get("inv") or self.lane[lane].get("watch"):
                try:
                    row = Lane(lane, d, self.mark).teardown()
                except Exception as exc:  # noqa: BLE001
                    row["error"] = repr(exc)[:300]
            if LANES[lane]["kind"] == "auto":
                watch = self.lane[lane].get("watch")
                if watch is not None:
                    try:
                        gke = e2e.collect_gke(lane, watch)
                        self.lane[lane]["gke"] = e2e.derive_gke(gke, self.lane[lane].get("gke_t0") or watch.t0)
                        self.lane[lane]["gke_raw"] = gke
                    except Exception as exc:  # noqa: BLE001
                        self.lane[lane]["gke_error"] = repr(exc)[:300]
                    watch.finish()
                try:
                    e2e.collect_node_logs(lane, d)
                except Exception:  # noqa: BLE001
                    pass
                row["release_s"] = e2e.release(lane)
                teardown[lane] = row
            else:
                teardown[lane] = row

        ths = [threading.Thread(target=down, args=(ln,), daemon=True) for ln in used]
        for th in ths:
            th.start()
        for th in ths:
            th.join(timeout=1800)
        for ln in used:  # GKE serializes node-pool operations: one reset after the other
            if LANES[ln]["kind"] == "auto" and (self.lane[ln].get("watch") or self.lane[ln].get("provision_t0")):
                teardown[ln]["reset_s"] = e2e.reset_pool(ln)
                teardown[ln]["cleanup"] = e2e.verify_clean(ln)
        self.rec["teardown"] = teardown


def mechanism(deploy_json: Path, boot: str) -> dict:
    """What actually happened inside the boot: snapshot restored / bank installed / XLA backend compiles /
    gate wait / load format (stock arms must show none of the TPUFlex state)."""
    d = load_json(deploy_json)
    if not d:
        return {"error": f"no {deploy_json.name}"}
    try:
        dep = d["measurements"]["deployment"]
        inst = dep["instances"][0]
        init = inst["initialization"]
        workers = init.get("worker_init_phases") or []
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)[:200]}
    compile_s = 0.0
    for w in workers:
        for prof in (w.get("startup_profiles") or {}).values():
            compile_s += float((prof.get("exclusive_wall_s") or {}).get("backend_compile_and_load") or 0.0)
    snap = [(w.get("weight_snapshot") or {}) for w in workers]
    bank = [(w.get("exec_bank") or {}) for w in workers]
    gate = [(w.get("state_gate") or {}).get("waited_s") for w in workers]
    settings = init.get("engine_settings") or {}
    row = {"workers": len(workers), "snapshot_restored": [s.get("restored") for s in snap],
           "snapshot_mode": [s.get("mode") for s in snap], "bank_installed": [b.get("installed") for b in bank],
           "bank_mode": [b.get("mode") for b in bank], "xla_backend_compile_s": round(compile_s, 3),
           "gate_wait_s": gate, "load_format": settings.get("load_format"), "model": settings.get("model"),
           "kv_blocks": (init.get("kv_capacity") or {}).get("usable_blocks"),
           "provenance": [((w.get("state_gate") or {}).get("delivered") or {}).get("provenance") for w in workers],
           "stage_clocks": dep.get("stage_clocks"), "prespawn": bool(dep.get("prespawn"))}
    if boot == "stock":
        row["ok"] = (len(workers) == 2 and not any(row["snapshot_restored"]) and not any(row["bank_installed"])
                     and row["load_format"] == "runai_streamer")
    else:
        row["ok"] = (len(workers) == 2 and all(row["snapshot_restored"]) and all(row["bank_installed"])
                     and row["xla_backend_compile_s"] < 1.0)
    return row


def run_one(bed: str, arm: str, trace: str, rep: int, rev: str) -> dict:
    rec = Run(bed, arm, trace, rep, rev).execute()
    log(f"{rec['rid']}: {rec.get('status')} {rec.get('error', '')[:300]}")
    return rec


def campaign(bed: str, plan_path: Path, rev: str) -> None:
    """Run the plan's (arm, trace, rep) entries in order, skipping those with a run.json; stop on a file flag."""
    plan = json.loads(plan_path.read_text())
    for item in plan["runs"]:
        rid = f"{item['trace']}-{item['arm']}-r{item['rep']}"
        if (OUT_ROOT / rev / "runs" / rid / "run.json").exists() or \
                (OUT_ROOT / rev / "runs" / rid / "run.json.gz").exists():
            continue
        if Path(f"/tmp/asb-stop-{bed}").exists():
            log(f"bed {bed}: stop flag present; stopping before {rid}")
            return
        run_one(bed, item["arm"], item["trace"], int(item["rep"]), rev)


# ---------------------------------------------------------------- calibration (one instance: the bed's base)

def loadtest(bed: str, name: str, trace: Path, cap: int, out_root: Path, client_extra: list[str] | None = None,
             admission_timeout_s: float = 600.0, endpoints: list[tuple[str, str]] | None = None) -> dict:
    """The bed's base (or `endpoints`) behind a fresh router (controller off), one client run of `trace`."""
    import asb_metrics as am  # noqa: PLC0415
    rid = f"cal-{name}"
    out = out_root / name
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    conf = {"cap": cap, "admission_timeout_s": admission_timeout_s, "upstream_timeout_s": 1800, "seed": 20261025,
            "client_timeout_s": 3600}
    if endpoints is None:
        base = json.loads((OUT_ROOT / "base" / f"{BEDS[bed]['base']}.json").read_text())
        endpoints = [(BEDS[bed]["base"], base["endpoint"])]
    ship_loadgen(bed)
    router = start_router(bed, rid, {"mode": "off"}, conf)
    for lane, url in endpoints:
        router.register(lane, url, "join", role="calibration")
    start_at = time.time() + 5.0
    start_client(bed, rid, trace, start_at, conf, client_extra)
    t = time.time()
    while not client_done(bed, rid):
        if time.time() - t > 7200:
            raise RuntimeError(f"{name}: client did not finish")
        time.sleep(10)
    time.sleep(2)
    fetch_run_dir(bed, rid, out / "loadgen")
    kexec(lg_pod(bed), f"pkill -f 'asb_loadgen.py' ; rm -rf /data/{rid} ; true")
    recs = am.read_jsonl(out / "loadgen/client.jsonl")
    meta = json.loads(trace.read_text())["meta"]
    res = {"name": name, "cap": cap, "trace": str(trace), "trace_meta": meta, "client_extra": client_extra or [],
           "endpoints": endpoints, "summary": am.summarize(recs), "saturation": am.throughput_window(recs),
           "code_commit": asb.git_rev(), "at": time.time()}
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    gz(out)
    log(f"{name}: {json.dumps(res['summary'])[:600]} sat={res['saturation']}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--bed", required=True, choices=list(BEDS))
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--trace", required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--rev", required=True)
    p = sub.add_parser("campaign")
    p.add_argument("--bed", required=True, choices=list(BEDS))
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--rev", required=True)
    p = sub.add_parser("loadtest")
    p.add_argument("--bed", required=True, choices=list(BEDS))
    p.add_argument("--name", required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--cap", type=int, required=True)
    p.add_argument("--out-root", type=Path, default=OUT_ROOT / "calibration")
    p.add_argument("--sequential", action="store_true")
    p.add_argument("--admission-timeout-s", type=float, default=600.0)
    p.add_argument("--time-scale", type=float)
    p.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.cmd == "run":
        rec = run_one(args.bed, args.arm, args.trace, args.rep, args.rev)
        return 0 if rec.get("status") == "ok" else 1
    if args.cmd == "loadtest":
        extra = (["--sequential"] if args.sequential else []) + \
            (["--time-scale", str(args.time_scale)] if args.time_scale else []) + \
            (["--limit", str(args.limit)] if args.limit else [])
        loadtest(args.bed, args.name, args.trace, args.cap, args.out_root, extra or None, args.admission_timeout_s)
        return 0
    campaign(args.bed, args.plan, args.rev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
