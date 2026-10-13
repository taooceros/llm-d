#!/usr/bin/env python3
"""Run ``hetero.switch.run_switch`` end to end against fakes (no Ray, no cluster, no Envoy).

The fake fabric keeps a small gateway state machine (bound instances / joining / maintenance /
capture) with configurable delays; the fake client "serves" requests with small sleeps, routes
each one against the *currently bound* instances (a request dispatched before its gate is a
recorded error), and returns records with segments and ttft.  tests/test_switch.py imports
this module for the same fakes.

    scripts/switch_dry_run.py [full|partial|both] [--scale 1.0] [--overlap]   # scale 1.0: drain 2 s/instance,
                                                                              # deploy 5 s, gateway_up 1 s

``--overlap`` runs the pre-spawn path (hetero/prespawn.py): the fake target engines are spawned
during phase 1, park, are released after the drain and committed instead of deployed.
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hetero.switch import run_switch  # noqa: E402

SMALL = ["small_a", "small_b", "small_c", "small_d"]
MAX_MODEL_LEN = {"small": 16384, "large": 131072}
PLANS = {
    "full": {"source": "S", "target": "H", "source_instances": SMALL,
             "target_instances": ["large_a", "large_b"], "plan": None, "kept_max_model_len": None},
    "partial": {"source": "S", "target": "Mb", "source_instances": SMALL,
                "target_instances": ["large_b", "small_a", "small_b"],
                "plan": {"from_layout": "S", "to_layout": "Mb", "kept": ["small_a", "small_b"],
                         "release": ["small_c", "small_d"], "join": ["large_b"]},
                "kept_max_model_len": 16384},
}
DEFAULT_DELAYS = {"drain": 2.0, "gateway_down": 0.3, "deploy": 5.0, "gateway_up": 1.0, "adopt": 0.2,
                  "retarget": 0.3, "capture": 0.2, "join": 1.0,
                  # pre-spawn: spawn (imports, config, actors) and park during phase 1; commit = engine init
                  "prespawn": 2.0, "park": 1.0, "release": 0.05, "commit": 3.0}


def synthetic_manifest(name: str, n: int, *, prompt_len: int, max_tokens: int) -> Dict[str, Any]:
    prompt = [1] * prompt_len  # one shared list: cheap even for 17k-token "W6" prompts
    return {"name": name, "model": "fake", "num_requests": n, "useful_output_tokens": n * max_tokens,
            "manifest_sha256": f"fake-{name}",
            "requests": [{"request_id": f"{name}-{i:05d}", "prompt_token_ids": prompt,
                          "max_tokens": max_tokens, "arrival_index": i} for i in range(n)]}


def make_loader(manifests: Dict[str, Dict[str, Any]]):
    def load(name: str) -> Dict[str, Any]:
        return manifests[name]
    return load


class FakeFabric:
    def __init__(self, mode: str, *, delays: Optional[Dict[str, float]] = None, scale: float = 1.0,
                 fail_at: Optional[str] = None, clock=time.time, overlap: bool = False) -> None:
        self.overlap = overlap
        self.prespawn_started: Optional[float] = None
        self.parked_at: Optional[float] = None
        self.released_at: Optional[float] = None
        self.prespawn_ids: List[str] = []
        self.aborted = False
        self.mode, self.clock = mode, clock
        self.cfg = PLANS[mode]
        self.delays = {k: v * scale for k, v in {**DEFAULT_DELAYS, **(delays or {})}.items()}
        self.fail_at = fail_at
        self.lock = threading.Lock()
        self.calls: List[Dict[str, Any]] = []  # {name, begin, end}
        self.exists, self.layout_id = True, self.cfg["source"]
        self.bound, self.joining, self.maintenance = set(SMALL), set(), set()
        self.serving, self.discovery_ready, self.capturing = True, True, False
        self.capture_started_at: Optional[float] = None
        self.client: Optional["FakeClient"] = None
        self.rr = 0
        self.policy: Dict[str, Any] = {}

    # -- instrumentation
    def _call(self, name: str, delay: float) -> None:
        begin = self.clock()
        if self.fail_at == name:
            time.sleep(delay / 2)
            self.calls.append({"name": name, "begin": begin, "end": None})
            raise RuntimeError(f"injected failure in {name}")
        time.sleep(delay)
        self.calls.append({"name": name, "begin": begin, "end": self.clock()})

    def order(self) -> List[str]:
        return [c["name"] for c in sorted(self.calls, key=lambda c: c["begin"])]

    # -- fabric interface
    def routing_status(self):
        with self.lock:
            if not self.exists:
                return None
            return {"layout_id": self.layout_id, "instances": sorted(self.bound),
                    "joining": sorted(self.joining), "maintenance_instances": sorted(self.maintenance),
                    "discovery_ready": self.discovery_ready, "serving": self.serving,
                    "capturing": self.capturing}

    def preflight(self, mode, source, target):
        return {k: self.cfg[k] for k in ("source_instances", "target_instances", "plan", "kept_max_model_len")}

    def quiesce(self) -> None:
        return None

    def set_policy(self, policy):
        self.policy = dict(policy)
        return policy

    def drain(self, instance_ids):
        ids = list(self.cfg["source_instances"] if instance_ids is None else instance_ids)
        began = self.clock()
        with self.lock:
            self.maintenance = set(ids)
            self.bound -= set(ids)
        maintenance_span = [began, self.clock()]
        entries = []
        fractions = (("drain_ack", .4), ("registry_draining", .02), ("stop_serving", .1), ("shutdown", .3),
                     ("kill", .05), ("pg_release", .1), ("withdraw", .03))
        for name in ids:
            timings: Dict[str, List[float]] = {"maintenance": list(maintenance_span)}
            for step, frac in fractions:
                start = self.clock()
                time.sleep(self.delays["drain"] * frac)
                timings[step] = [start, self.clock()]
            entries.append({"instance_id": name, "generation": 1, "actor_removed": True, "timings": timings})
        self.calls.append({"name": "drain", "begin": began, "end": self.clock()})
        if self.fail_at == "drain":
            raise RuntimeError("injected failure in drain")
        return {"layout_id": self.cfg["source"], "drained": entries, "maintenance": {"id": "m1"}}

    def gateway_down(self):
        self._call("gateway_down", self.delays["gateway_down"])
        with self.lock:
            self.exists, self.serving, self.bound = False, False, set()
        return {"found": True, "removed": True}

    def deploy(self, instance_ids=None):
        if self.overlap:
            raise AssertionError("a pre-spawn run must commit, never deploy")
        if self.fail_at == "deploy":
            self._call("deploy", self.delays["deploy"])
        ids = list(self.cfg["target_instances"] if instance_ids is None else instance_ids)
        total = self.delays["deploy"]
        began = self.clock()
        time.sleep(total * .15)  # backend / placement / actor
        stage = {"backend_s": round(total * .03, 2), "placement_s": round(total * .05, 2),
                 "actor_s": round(total * .07, 2)}
        e0 = self.clock()
        time.sleep(total * .05)
        phases: Dict[str, Dict[str, float]] = {}
        for name, frac in (("init_device", .05), ("load_model", .4), ("determine_available_memory", .02),
                           ("initialize_from_config", .08), ("compile_or_warm_up_model", .15)):
            start = self.clock()
            time.sleep(total * frac)
            phases[name] = {"start": start, "end": self.clock()}
        lm = phases["load_model"]
        span = lm["end"] - lm["start"]
        for i, sub in enumerate(("snapshot_manifest_verify", "snapshot_read_tmpfs", "snapshot_verify",
                                 "snapshot_device_placement")):
            phases[sub] = {"start": lm["start"] + span * i / 5, "end": lm["start"] + span * (i + 1) / 5}
        time.sleep(total * .03)  # serving endpoint + registry
        instances = [{"descriptor": {"instance_id": n},
                      "initialization": {"instance_id": n, "engine_init_started_at": e0,
                                         "worker_init_phases": [
                                             {"executor_rank": r, "phases": phases,
                                              "weight_snapshot": {"restored": True}} for r in range(2)]}}
                     for n in ids]
        self.calls.append({"name": "deploy", "begin": began, "end": self.clock()})
        return {"layout_id": self.cfg["target"], "stage_clocks": stage, "instances": instances}

    def gateway_up(self, policy):
        self._call("gateway_up", self.delays["gateway_up"])
        with self.lock:
            self.exists, self.serving, self.discovery_ready = True, True, True
            self.layout_id = self.cfg["target"]
            self.bound = set(self.cfg["target_instances"])
            self.joining, self.maintenance = set(), set()
        self.policy = dict(policy)
        return {"created": True}

    def adopt(self):
        self._call("adopt", self.delays["adopt"])

    def retarget(self, joiners):
        self._call("retarget", self.delays["retarget"])
        with self.lock:
            self.layout_id = self.cfg["target"]
            self.bound = set(self.cfg["plan"]["kept"])
            self.joining, self.maintenance = set(joiners), set()
        return {"layout_id": self.layout_id, "joining": sorted(joiners), "retargeted_at": self.clock()}

    def join(self, joiners, policy_after):
        self._call("join", self.delays["join"])
        with self.lock:
            self.bound |= set(joiners)
            self.joining -= set(joiners)
        return {"joined": sorted(joiners), "joined_at": self.clock(), "policy_roles": {},
                "remaining_joining": []}

    def start_capture(self):
        if self.capturing:
            raise RuntimeError("already capturing")
        self._call("start_capture", self.delays["capture"])
        with self.lock:
            self.capturing, self.capture_started_at = True, self.clock()
        return {"capturing": sorted(self.bound)}

    def stop_capture(self):
        with self.lock:
            self.capturing = False
        served = list(self.client.served.values()) if self.client else []
        rows: Dict[str, List[Dict[str, Any]]] = {}
        for rec in served:
            for seg in rec["segments"]:
                if self.capture_started_at is not None and seg["finished_at_s"] >= self.capture_started_at:
                    rows.setdefault(seg["instance_id"], []).append({
                        "completed_at_s": seg["finished_at_s"], "emitted_output_tokens": seg["output_tokens"],
                        "phase": "decode", "scheduled_requests": 1, "scheduled_to_output_s": 0.01,
                        "preempted_request_ids": [], "finished_request_ids": [rec["request_id"]]})
        return {inst: {"rows": r, "audit": {"unmatched_retirements": 0, "rows": len(r), "retired_rows": len(r)}}
                for inst, r in rows.items()}

    def records(self):
        return [{k: v for k, v in rec.items() if not k.startswith("client_")}
                for rec in (self.client.served.values() if self.client else [])]

    def traces(self, limit=0):
        return []

    # -- pre-spawn
    def prespawn_requested(self, instance_ids):
        return self.overlap

    def prespawn(self, instance_ids=None):
        self.prespawn_ids = list(self.cfg["target_instances"] if instance_ids is None else instance_ids)
        self.prespawn_started = self.clock()
        self._call("prespawn", self.delays["prespawn"])
        return {"gate_name": "fake-gate", "clocks": {"prespawn_begin": self.prespawn_started}}

    def await_parked(self, timeout_s, stop=None):
        began = self.clock()
        if self.fail_at == "park":
            self.calls.append({"name": "await_parked", "begin": began, "end": None})
            raise RuntimeError("injected failure in park")
        deadline = began + self.delays["park"]
        while self.clock() < deadline:
            if stop is not None and stop():
                raise RuntimeError("pre-spawn park wait aborted")
            time.sleep(0.005)
        self.parked_at = self.clock()
        self.calls.append({"name": "await_parked", "begin": began, "end": self.parked_at})
        return {"parked": {f"10.0.0.{i}": self.parked_at for i in range(2)}, "all_parked_at": self.parked_at}

    def release(self, instance_ids=None):
        self._call("release", self.delays["release"])
        self.released_at = self.clock()
        ids = self.prespawn_ids if instance_ids is None else list(instance_ids)
        return {"released_at": {f"10.0.0.{i}": self.released_at for i in range(2)},
                "tpu_hold_pgs": [f"omp-hetero--{self.cfg['target']}--{n}-tpuhold" for n in ids]}

    def commit(self, instance_ids=None):
        if self.fail_at == "commit":
            self._call("commit", self.delays["commit"])
        ids = self.prespawn_ids if instance_ids is None else list(instance_ids)
        assert sorted(ids) == sorted(self.prespawn_ids), "commit set differs from the pre-spawn"
        total = self.delays["commit"]
        began = self.clock()
        time.sleep(total * .05)  # gate wake-up + libtpu lock / vfio free
        free = self.clock()
        phases: Dict[str, Dict[str, float]] = {"prespawn_gate": {"start": self.parked_at, "end": free}}
        for name, frac in (("init_device", .05), ("load_model", .4), ("determine_available_memory", .02),
                           ("initialize_from_config", .08), ("compile_or_warm_up_model", .3)):
            start = self.clock()
            time.sleep(total * frac)
            phases[name] = {"start": start, "end": self.clock()}
        lm = phases["load_model"]
        for i, sub in enumerate(("snapshot_manifest_verify", "snapshot_read_tmpfs", "snapshot_verify",
                                 "snapshot_device_placement")):
            phases[sub] = {"start": lm["start"] + (lm["end"] - lm["start"]) * i / 5,
                           "end": lm["start"] + (lm["end"] - lm["start"]) * (i + 1) / 5}
        time.sleep(total * .05)  # serving endpoint + registry
        gate = {"key": "10.0.0.0", "parked_at": self.parked_at, "released_at": self.released_at,
                "gate_returned_at": self.released_at, "lock_free_at": free, "vfio_free_at": free,
                "devices_ready_at": phases["init_device"]["end"], "device_free_wait_s": free - self.released_at,
                "holders_seen": {}, "self_check_parked": {"clean": True}, "self_check_released": {"clean": True}}
        instances = [{"descriptor": {"instance_id": n},
                      "initialization": {"instance_id": n, "engine_init_started_at": self.prespawn_started,
                                         "worker_init_phases": [
                                             {"executor_rank": r, "phases": phases, "prespawn_gate": gate,
                                              "weight_snapshot": {"restored": True}} for r in range(2)]}}
                     for n in ids]
        self.calls.append({"name": "commit", "begin": began, "end": self.clock()})
        return {"layout_id": self.cfg["target"], "stage_clocks": {"backend_s": 0.1, "placement_s": 0.1},
                "instances": instances,
                "prespawn": {"clocks": {"prespawn_begin": self.prespawn_started, "all_parked": self.parked_at},
                             "tpu_hold": {n: {"created_at": self.released_at, "ready_at": self.released_at + 0.01}
                                          for n in ids}}}

    def abort_prespawn(self):
        now = self.clock()
        self.calls.append({"name": "abort_prespawn", "begin": now, "end": now})
        self.aborted = True
        return {"aborted": True, "actors": list(self.prespawn_ids)}

    # -- used by the fake client
    def route(self, item) -> str:
        context = len(item["prompt_token_ids"]) + int(item["max_tokens"])
        with self.lock:
            if not (self.exists and self.serving and self.discovery_ready):
                raise RuntimeError("gateway is not routing")
            eligible = sorted(n for n in self.bound
                              if MAX_MODEL_LEN["large" if n.startswith("large") else "small"] >= context)
            if not eligible:
                raise RuntimeError(f"no bound instance fits context {context}: bound={sorted(self.bound)}")
            self.rr += 1
            return eligible[self.rr % len(eligible)]


class FakeClient:
    def __init__(self, fabric: FakeFabric, *, service_s: float = 0.02, service_by_label=None,
                 clock=time.time) -> None:
        self.fabric, self.service_s, self.clock = fabric, service_s, clock
        self.service_by_label = dict(service_by_label or {})
        fabric.client = self
        self.served: Dict[str, Dict[str, Any]] = {}
        self.opened = self.closed = False
        self.cancelled_labels: List[str] = []
        self.in_flight: Dict[str, int] = {}
        self.max_in_flight: Dict[str, int] = {}
        self.max_total = 0
        self.dispatches: List[Dict[str, Any]] = []  # {label, request_id, t, instance}

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.closed = True

    async def dispatch(self, items, *, concurrency, timeout_s, label=None):
        semaphore = asyncio.Semaphore(concurrency)

        async def one(item):
            async with semaphore:
                started = self.clock()
                try:
                    instance = self.fabric.route(item)
                except Exception as exc:  # noqa: BLE001
                    self.dispatches.append({"label": label, "request_id": item["request_id"], "t": started,
                                            "instance": None, "error": str(exc)})
                    return {"request_id": item["request_id"], "client_started_at_s": started,
                            "client_finished_at_s": self.clock(), "error": f"RuntimeError: {exc}"}
                self.dispatches.append({"label": label, "request_id": item["request_id"], "t": started,
                                        "instance": instance})
                self.in_flight[label] = self.in_flight.get(label, 0) + 1
                self.max_in_flight[label] = max(self.max_in_flight.get(label, 0), self.in_flight[label])
                self.max_total = max(self.max_total, sum(self.in_flight.values()))
                delay = self.service_by_label.get(label, self.service_s)
                try:
                    await asyncio.sleep(delay)
                finally:
                    self.in_flight[label] -= 1
                n = int(item["max_tokens"])
                tokens = list(range(n)) if n <= 64 else [7] * n
                finished = self.clock()
                record = {"request_id": item["request_id"], "ingress_protocol": "fake",
                          "output_token_ids": tokens, "useful_output_tokens": n,
                          "prompt_tokens": len(item["prompt_token_ids"]), "initial_instance": instance,
                          "segments": [{"instance_id": instance, "output_tokens": n, "output_token_ids": tokens,
                                        "ttft_s": delay / 2, "started_at_s": started,
                                        "finished_at_s": finished}]}
                self.served[item["request_id"]] = dict(record)
                return {**record, "client_started_at_s": started, "client_finished_at_s": finished}

        tasks = [asyncio.create_task(one(item)) for item in items]
        try:
            return list(await asyncio.gather(*tasks))
        except asyncio.CancelledError:
            self.cancelled_labels.append(label)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


def default_manifests(mode: str, *, phase1_rows=64, w0_rows=120, w6_rows=20) -> Dict[str, Dict[str, Any]]:
    return {"W0": synthetic_manifest("W0", phase1_rows if mode == "full" else w0_rows, prompt_len=32, max_tokens=16),
            "W6": synthetic_manifest("W6", w6_rows, prompt_len=17000, max_tokens=8),
            "W0p2": synthetic_manifest("W0p2", w0_rows, prompt_len=32, max_tokens=16)}


def run_dry(mode: str, out_dir: pathlib.Path, *, scale: float = 1.0, delays=None, fail_at=None,
            service_s: float = 0.02, service_by_label=None, manifests=None, run_id: Optional[str] = None,
            overlap: bool = False, prespawn_at_s: float = 0.0):
    fabric = FakeFabric(mode, scale=scale, delays=delays, fail_at=fail_at, overlap=overlap)
    if overlap and service_by_label is None:
        # Phase 1 outlasts spawn + park, so the engines are parked before T0 (as intended in production).
        service_by_label = {"phase1": prespawn_at_s + 1.5 * (fabric.delays["prespawn"] + fabric.delays["park"])}
    client = FakeClient(fabric, service_s=service_s, service_by_label=service_by_label)
    manifests = manifests or default_manifests(mode)
    cfg = PLANS[mode]
    kwargs = dict(mode=mode, source_layout=cfg["source"], target_layout=cfg["target"], phase1="W0",
                  phase2="W6" if mode == "full" else "W0p2,W6", run_id=run_id or f"dry-{mode}",
                  policy={"decode_length_blind": True, "large_fresh_traffic_share": 0.0,
                          "reserve_large_headroom_blocks": 0},
                  static_target_s=0.5 if mode == "full" else 1.0, artifact_root=out_dir,
                  manifest_loader=make_loader(manifests), gate_poll_s=0.05, event_grace_s=1.0,
                  prespawn_at_s=prespawn_at_s, prespawn_park_timeout_s=60.0)
    return fabric, client, kwargs


def print_report(report: Dict[str, Any], fabric: FakeFabric, client: FakeClient) -> None:
    T0 = report["T0"]
    print(f"\n=== {report['mode']} {report['source_layout']}->{report['target_layout']} "
          f"completed={report['completed']} ===")
    print("-- stamp timeline (s after T0)")
    for event in sorted(report["stamp_events"], key=lambda e: e["t"]):
        print(f"  {event['t'] - T0:9.3f}  {event['name']}")
    print("-- segments")
    print(f"  {'segment':24} {'start':>9} {'end':>9} {'s':>9}  class")
    for s in report["segments"]:
        print(f"  {s['segment']:24} {s['start_rel_s']:9.3f} {s['end_rel_s']:9.3f} {s['s']:9.3f}  {s['class']}")
    tl = report["timeline"]
    print(f"  sum={tl['segment_sum_s']:.3f}  gap_dispatch_s={tl['gap_dispatch_s']:.3f}  reconcile={tl['segments_reconcile']}")
    print("-- endpoints")
    for key, value in report["endpoints"].items():
        print(f"  {key}: {value}")
    print("-- streams", report["phase2_streams"])
    print("-- caps", {k: report["streams"][k]["cap"] for k in report["streams"]},
          "max in flight", client.max_in_flight, "total", client.max_total)
    print("-- restored", report["restored"]["all_restored"], " accounting", report["accounting"]["phase2"]["complete"])


def check_order(report: Dict[str, Any], fabric: FakeFabric, client: FakeClient) -> None:
    st, mode = report["stamps"], report["mode"]
    seq = [("T0", "controller_wake"), ("controller_wake", "drain_begin"), ("drain_end", "adopt_begin" if mode == "partial" else "gateway_down_begin")]
    if mode == "full":
        seq += [("gateway_down_end", "deploy_begin"), ("deploy_end", "gateway_up_begin"),
                ("gateway_up_end", "gate_open:target"), ("gate_open:target", "capture_arm_begin")]
        first = min(d["t"] for d in client.dispatches if d["label"] == "target")
        assert first >= st["capture_arm_end"] >= st["gate_open:target"], "target dispatched before capture armed/gate open"
    else:
        seq += [("adopt_end", "retarget_begin"), ("retarget_end", "capture_arm_begin"), ("capture_arm_end", "gate_open:kept"),
                ("deploy_begin", "deploy_end"), ("deploy_end", "join_begin"), ("join_begin", "gate_open:joiner")]
        assert min(d["t"] for d in client.dispatches if d["label"] == "kept") >= st["gate_open:kept"]
        assert st["gate_open:kept"] < st["deploy_end"], "kept stream must not wait for the joiners' deploy"
        assert min(d["t"] for d in client.dispatches if d["label"] == "joiner") >= st["gate_open:joiner"] >= st["join_begin"]
    if report.get("prespawn"):
        pre = [("phase1_begin", "prespawn_begin"), ("prespawn_end", "prespawn_park_begin"),
               ("prespawn_parked", "T0"), ("controller_wake", "prespawn_wait_begin"),
               ("prespawn_wait_end", "drain_begin"), ("drain_end", "prespawn_release_begin"),
               ("prespawn_release_end", "deploy_begin")]
        if mode == "full":
            pre.append(("gateway_down_end", "prespawn_release_begin"))
        else:
            pre.append(("prespawn_release_end", "adopt_begin"))
        seq += pre
        assert report["prespawn"]["parked_before_T0"], "pre-spawned engines were not parked before T0"
        assert "deploy" not in fabric.order() and "commit" in fabric.order()
        assert st["prespawn_begin"] - st["phase1_begin"] >= report["prespawn"]["at_s"] - 1e-3
    for a, b in seq:
        assert st[a] <= st[b] + 1e-6, f"{a} after {b}"
    assert report["timeline"]["segments_reconcile"], "segments do not sum to the dispatch gap"
    assert report["completed"], "dry run incomplete"
    assert client.max_total <= 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", default="both", choices=("full", "partial", "both"))
    parser.add_argument("--scale", type=float, default=1.0, help="multiply every fake delay")
    parser.add_argument("--overlap", action="store_true", help="pre-spawn the target engines during phase 1")
    parser.add_argument("--prespawn-at-s", type=float, default=0.0, help="pre-spawn start after phase 1 begins")
    args = parser.parse_args()
    for mode in ("full", "partial") if args.mode == "both" else (args.mode,):
        with tempfile.TemporaryDirectory() as tmp:
            fabric, client, kwargs = run_dry(mode, pathlib.Path(tmp), scale=args.scale, overlap=args.overlap,
                                             prespawn_at_s=args.prespawn_at_s)
            report = run_switch(fabric, client, **kwargs)
            print_report(report, fabric, client)
            if report.get("prespawn"):
                print("-- prespawn window", report["prespawn"]["window"])
                print("-- prespawn anatomy", report["timeline"]["prespawn"])
            check_order(report, fabric, client)
            print(f"-- call order {fabric.order()}\n-- ordering asserts OK")


if __name__ == "__main__":
    main()
