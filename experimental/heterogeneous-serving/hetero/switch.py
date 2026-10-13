"""Production in-process layout switch: one Ray job, no benchmark warm-up, no per-step job.

``run_switch`` drives phase 1 on the source layout through Envoy, and the instant its last
request finishes (T0) a controller thread tears the source down and brings the target up while
phase-2 request streams wait behind *gates* that open only when the gateway verifiably routes
to the required instances.  Every call is stamped on the head's wall clock, so the interval
T0 -> first target dispatch is split into contiguous segments that sum to it exactly.

    full     S -> H   drain(all, remove) -> gateway_down -> deploy -> gateway_up -> [gate] -> capture -> dispatch
    partial  S -> Mb  drain(released) -> adopt kept -> retarget -> capture -> [kept gate] -> dispatch
                      deploy(joiners) -> join -> [joiner gate] -> dispatch

With pre-spawn (engine setting ``prespawn=true`` on every deployed instance; hetero/prespawn.py)
the target engines start ``prespawn_at_s`` after phase 1 begins, next to the TPU-owning source,
and park before their first device touch; the controller then only releases and commits them:

    full     [pre-spawn during phase 1] T0 -> prespawn_wait -> drain -> gateway_down -> release -> commit
             -> gateway_up -> [gate] -> capture -> dispatch
    partial  [pre-spawn joiners] T0 -> prespawn_wait -> drain(released) -> release -> adopt -> retarget
             -> capture -> [kept gate] -> dispatch;  commit(joiners) -> join -> [joiner gate] -> dispatch

The real operations sit behind a *fabric* (``RayFabric``) and the HTTP dispatcher behind a
*client* (``HttpClient``); tests and ``scripts/switch_dry_run.py`` inject fakes.

Fabric interface (all blocking; called from the controller thread or a worker thread):
    routing_status() -> dict | None          gateway ``routing_status`` or None when no gateway
    preflight(mode, source, target) -> dict  {source_instances, target_instances, plan, kept_max_model_len}
    quiesce()                                reset journal, resume admission, idle barrier (not timed)
    set_policy(policy) -> dict               source gateway, between workloads
    drain(instance_ids | None) -> dict       ``drain_deployment(..., remove=True)`` report
    gateway_down() / gateway_up(policy)      full mode
    deploy(instance_ids | None) -> dict      ``deploy`` report
    adopt() / retarget(joiners) / join(joiners, policy_after)   partial mode
    start_capture() / stop_capture() / records() / traces(limit)
    prespawn_requested(instance_ids) -> bool     every deployed instance has prespawn=true
    prespawn(instance_ids | None) -> dict         ``prespawn_deployment``: CPU-only groups, parked workers
    await_parked(timeout_s, stop) -> dict         every pre-spawned worker parked on the gate
    release(instance_ids | None) -> dict          open the gate for those hosts + TPU-hold groups
    commit(instance_ids | None) -> dict           ``commit_prespawned`` (``deploy`` report shape)
    abort_prespawn() -> dict | None               remove only the pre-spawn's own actors/groups/gate

Client interface: ``async open()``, ``async close()``,
    ``async dispatch(items, *, concurrency, timeout_s, label=None) -> list[record]``.

Deviation from the P1 text, forced by the gateway: ``begin_maintenance`` / ``retarget_layout``
require ``not capturing`` (``OmpGateway._require_idle``), so partial mode cannot arm capture
before phase 1.  It arms it once after the retarget, before the first kept dispatch (the same
point ``run_transition``'s phase-2 ``run_arm`` arms it); joiners start capturing at join.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import gzip
import json
import pathlib
import statistics
import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .evidence import emit_event
from .experiment import (
    ADMITTED_CAP, ARMS, ARTIFACT_ROOT, BASE_POLICY, INGRESS_URL,
    _extproc_record, _failure_signature, _instance_roles, _write_gz, _write_iterations,
)
from .manifest import load_manifest

MODES = ("full", "partial")
BIN_S = 10.0
CLOCK_NOTE = ("T0, controller stamps and client dispatch/finish are head-process time.time(); "
              "engine_init_started_at, worker phase clocks and segment started_at_s/ttft_s are "
              "engine-host time.time() (hosts NTP-synchronised, not verified here)")
LOAD_SUB = ["snapshot_manifest_verify", "snapshot_read_tmpfs", "snapshot_verify", "snapshot_device_placement"]
PHASES = ["init_device", "load_model", "determine_available_memory", "initialize_from_config",
          "compile_or_warm_up_model"]


# --------------------------------------------------------------------------- policy

def arm_for_layout(layout_id: str) -> str:
    """The non-migrating arm that defines a layout's policy semantics (S, H, Mb0, M0)."""
    for name, spec in ARMS.items():
        if spec["layout"] == layout_id and not spec["migration_enabled"]:
            return name
    raise KeyError(f"no non-migrating arm for layout {layout_id}")


def merged_policy(amend: Optional[Mapping[str, Any]], instances: Iterable[str], layout_id: str) -> Dict[str, Any]:
    """The policy ``run_arm`` would declare for this layout/instance set (pilot amendment)."""
    from .gateway.policy import RoutingPolicy

    amendable = {f.name for f in dataclasses.fields(RoutingPolicy)} - {
        "small_instances", "large_instances", "migration_enabled", "migration_mechanism"}
    unknown = set(amend or {}) - amendable
    if unknown:
        raise ValueError(f"switch policy may only amend routing policy fields: {sorted(unknown)}")
    spec = ARMS[arm_for_layout(layout_id)]
    roles = _instance_roles(layout_id, instances)
    policy = dict(BASE_POLICY)
    policy.update(amend or {})
    policy.update(roles)
    policy["migration_enabled"] = spec["migration_enabled"]
    policy["migration_mechanism"] = spec["migration_mechanism"]
    if not roles["large_instances"] or not roles["small_instances"]:
        policy["large_fresh_traffic_share"] = 0.0
        policy["migration_enabled"] = False
    return RoutingPolicy.from_dict(policy).to_dict()


# --------------------------------------------------------------------------- stamps and signals

class StampLog:
    """Thread-safe wall-clock stamps: ``mark`` is a point, ``span`` records ``<name>_begin/_end``."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        self.events: List[Dict[str, Any]] = []
        self.stamps: Dict[str, float] = {}

    def mark(self, name: str, t: Optional[float] = None) -> float:
        t = self.clock() if t is None else t
        with self._lock:
            self.events.append({"name": name, "t": t, "thread": threading.current_thread().name})
            self.stamps[name] = t
        emit_event("SWITCH_STEP", step=name)
        return t

    @contextlib.contextmanager
    def span(self, name: str):
        self.mark(f"{name}_begin")
        try:
            yield
        except BaseException:
            self.mark(f"{name}_error")
            raise
        self.mark(f"{name}_end")


class Signal:
    """A threading.Event the controller sets and the client's event loop can await promptly."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._aio: Optional[asyncio.Event] = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._aio = asyncio.Event()
        if self._event.is_set():
            self._aio.set()

    def set(self) -> None:
        self._event.set()
        if self._loop is not None and self._aio is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._aio.set)

    def is_set(self) -> bool:
        return self._event.is_set()

    async def wait(self, timeout: float) -> bool:
        assert self._aio is not None, "Signal.bind() before wait()"
        try:
            await asyncio.wait_for(self._aio.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        return self._event.is_set()


# --------------------------------------------------------------------------- real client

class HttpClient:
    """``_dispatch_http`` through Envoy with one httpx client kept open for the whole run."""

    def __init__(self, url: str = INGRESS_URL, *, timeout_s: float = 7200.0) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self._client: Any = None
        # label -> records completed so far; survives a cancelled stream (failure evidence).
        self.sinks: Dict[str, List[Dict[str, Any]]] = {}

    async def open(self) -> None:
        import httpx

        limits = httpx.Limits(max_connections=ADMITTED_CAP, max_keepalive_connections=ADMITTED_CAP)
        self._client = httpx.AsyncClient(base_url=self.url, limits=limits, timeout=self.timeout_s,
                                         trust_env=False)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def dispatch(self, items: List[Dict[str, Any]], *, concurrency: int, timeout_s: float,
                       label: Optional[str] = None) -> List[Dict[str, Any]]:
        client = self._client
        semaphore = asyncio.Semaphore(concurrency)
        sink = self.sinks.setdefault(label or "unlabelled", [])

        async def one(item: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                started = time.time()
                response = None
                try:
                    response = await client.post("/v1/completions", json=item)
                    response.raise_for_status()
                    record = _extproc_record(item, response)
                    record.update(client_started_at_s=started, client_finished_at_s=time.time())
                    sink.append(record)
                    return record
                except Exception as exc:  # noqa: BLE001 - delivered as a failed record, never retried
                    failure = {"request_id": item["request_id"], "client_started_at_s": started,
                               "client_finished_at_s": time.time(),
                               "error": f"{type(exc).__name__}: {exc}"}
                    if response is not None:
                        failure.update(_failure_signature(response))
                    sink.append(failure)
                    return failure

        tasks = [asyncio.create_task(one(item)) for item in items]
        try:
            _, pending = await asyncio.wait(tasks, timeout=timeout_s)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        return [
            ({"request_id": item["request_id"], "client_finished_at_s": time.time(),
              "error": "whole-batch deadline exceeded before final client delivery"}
             if task in pending else task.result())
            for item, task in zip(items, tasks)
        ]


# --------------------------------------------------------------------------- real fabric

class RayFabric:
    """The existing Ray/gateway operations, unchanged, behind the fabric interface."""

    def __init__(self, *, inventory: Any, config: Any, mode: str, source_layout: str,
                 drain_timeout_s: float = 600.0, head_node_ip: Optional[str] = None,
                 prespawn_gate_timeout_s: float = 3600.0) -> None:
        self.inventory, self.config, self.mode = inventory, config, mode
        self.source_layout = source_layout
        self.target_layout = config.layout.layout_id
        self.drain_timeout_s = drain_timeout_s
        self.head_node_ip = head_node_ip
        self.prespawn_gate_timeout_s = prespawn_gate_timeout_s
        self._registry = None
        self._plan = None
        self._actors: Dict[str, Any] = {}
        self._by_id: Dict[str, Any] = {}
        self._digest: Optional[str] = None
        self._prepared: Any = None

    @staticmethod
    def _gateway() -> Any:
        from .gateway.lifecycle import find_gateway

        return find_gateway()

    def routing_status(self) -> Optional[Dict[str, Any]]:
        import ray

        gateway = self._gateway()
        return None if gateway is None else ray.get(gateway.routing_status.remote(), timeout=30)

    def preflight(self, mode: str, source_layout: str, target_layout: str) -> Dict[str, Any]:
        from . import transition as tr
        from .layout import resolve_layout
        from .registry import find_registry

        self._registry = find_registry()
        _, source, from_instances = tr.check_source_idle(self._gateway(), self._registry, source_layout)
        out: Dict[str, Any] = {"source_instances": sorted(from_instances), "plan": None,
                               "kept_max_model_len": None}
        if mode == "full":
            resolved = resolve_layout(self.inventory, self.config.layout)
            out["target_instances"] = sorted(i.instance_id for i in resolved.instances)
            return out
        resolved, target, digest, plan, by_id, rows = tr.resolve_transition(
            self.inventory, self.config, source_layout, self._registry, source, from_instances)
        self._actors = tr.prepare_kept(self._registry, self.inventory, self.config, plan, by_id, rows, target)
        self._plan, self._by_id, self._digest = plan, by_id, digest
        out["target_instances"] = sorted(by_id)
        out["plan"] = dataclasses.asdict(plan)
        out["kept_max_model_len"] = min(int(self.config.settings_for(n)["max_model_len"]) for n in plan.kept)
        return out

    def quiesce(self) -> None:
        import ray

        gateway = self._gateway()
        ray.get(gateway.reset_records.remote(), timeout=120)
        ray.get(gateway.resume_admission.remote(), timeout=120)
        ray.get(gateway.await_idle_instances.remote(300.0), timeout=360)

    def set_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        import ray

        return ray.get(self._gateway().set_policy.remote(policy), timeout=60)

    def drain(self, instance_ids: Optional[Sequence[str]]) -> Dict[str, Any]:
        from .deployment import drain_deployment

        return drain_deployment(self.source_layout, None if instance_ids is None else list(instance_ids),
                                remove=True, timeout_s=self.drain_timeout_s)

    def gateway_down(self) -> Dict[str, Any]:
        from .gateway.lifecycle import gateway_down

        return gateway_down(remove=True)

    def gateway_up(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        from .gateway.lifecycle import gateway_up

        return gateway_up(layout_id=self.target_layout, policy=policy, head_node_ip=self.head_node_ip)

    def deploy(self, instance_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        from .deployment import deploy

        return deploy(self.inventory, self.config,
                      instance_ids=None if instance_ids is None else list(instance_ids))

    def adopt(self) -> None:
        from . import transition as tr

        tr.adopt_kept(self._registry, self._actors, self._by_id, self.target_layout, self._digest)

    def retarget(self, joiners: Sequence[str]) -> Dict[str, Any]:
        import ray

        return ray.get(self._gateway().retarget_layout.remote(self.target_layout, list(joiners)), timeout=600)

    def join(self, joiners: Sequence[str], policy_after: Dict[str, Any]) -> Dict[str, Any]:
        import ray

        return ray.get(self._gateway().join_instances.remote(list(joiners), policy_after), timeout=600)

    def start_capture(self) -> Dict[str, Any]:
        import ray

        return ray.get(self._gateway().start_capture.remote(), timeout=300)

    def stop_capture(self) -> Dict[str, Any]:
        import ray

        return ray.get(self._gateway().stop_capture.remote(), timeout=600)

    def records(self) -> List[Dict[str, Any]]:
        import ray

        return ray.get(self._gateway().records.remote(), timeout=300)

    def traces(self, limit: int = 200000) -> List[Dict[str, Any]]:
        import ray

        return ray.get(self._gateway().traces.remote(limit), timeout=300)

    # ----- pre-spawn (hetero/deployment.py prespawn_deployment / release_prepared / commit_prespawned)
    def prespawn_requested(self, instance_ids: Sequence[str]) -> bool:
        flags = {bool(self.config.settings_for(i)["prespawn"]) for i in instance_ids}
        if len(flags) > 1:
            raise ValueError(f"engine setting prespawn must agree across the deployed instances {sorted(instance_ids)}")
        return flags == {True}

    def prespawn(self, instance_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        from .deployment import prespawn_deployment

        self._prepared = prespawn_deployment(
            self.inventory, self.config, instance_ids=None if instance_ids is None else list(instance_ids),
            gate_timeout_s=self.prespawn_gate_timeout_s)
        return self._prepared.evidence()

    def await_parked(self, timeout_s: float, stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        from .deployment import await_parked

        return await_parked(self._prepared, timeout_s=timeout_s, stop=stop)

    def release(self, instance_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        from .deployment import release_prepared

        return release_prepared(self._prepared, instance_ids)

    def commit(self, instance_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        from .deployment import commit_prespawned

        wanted = sorted(i.instance_id for i in self._prepared.instances)
        if instance_ids is not None and sorted(instance_ids) != wanted:
            raise ValueError(f"commit of {sorted(instance_ids)} does not match the pre-spawn {wanted}")
        return commit_prespawned(self._prepared)

    def abort_prespawn(self) -> Optional[Dict[str, Any]]:
        from .deployment import abort_prespawned

        return None if self._prepared is None else abort_prespawned(self._prepared)


# --------------------------------------------------------------------------- pure analysis

def _med(values: Sequence[float]) -> float:
    return round(statistics.median(values), 2)


def deploy_anatomy(deploy_report: Mapping[str, Any]) -> Dict[str, Any]:
    """Engine start-up clocks of one ``deploy()`` report (the production_c_recut.deploy_anatomy logic)."""
    per_worker: List[Dict[str, Any]] = []
    ready: Dict[str, float] = {}
    started: List[float] = []
    for inst in deploy_report["instances"]:
        init = inst["initialization"]
        name = init.get("instance_id") or inst["descriptor"]["instance_id"]
        e0 = init["engine_init_started_at"]
        started.append(e0)
        ends: List[float] = []
        for w in init.get("worker_init_phases", []):
            ph = w["phases"]
            row: Dict[str, Any] = {"instance": name, "rank": w["executor_rank"], "engine_init_started_at": e0,
                                   "init_device_start": ph["init_device"]["start"],
                                   "end": ph["compile_or_warm_up_model"]["end"]}
            for p in PHASES + LOAD_SUB:
                if p in ph:
                    row[p] = ph[p]["end"] - ph[p]["start"]
            lm = ph["load_model"]
            row["load_model_unattributed"] = (lm["end"] - lm["start"]) - sum(row.get(p, 0) for p in LOAD_SUB)
            per_worker.append(row)
            ends.append(row["end"])
        if ends:
            ready[name] = max(ends)
    out: Dict[str, Any] = {"stage_clocks": dict(deploy_report.get("stage_clocks") or {}),
                           "engine_init_started_at_min": min(started) if started else None,
                           "engine_init_started_at_max": max(started) if started else None,
                           "workers": len(per_worker), "complete": bool(per_worker)}
    if per_worker:
        keys = PHASES + LOAD_SUB + ["load_model_unattributed"]
        out.update(
            init_device_start_min=min(w["init_device_start"] for w in per_worker),
            init_device_start_max=max(w["init_device_start"] for w in per_worker),
            last_engine_ready=max(ready.values()),
            instance_ready=dict(sorted(ready.items())),
            phase_median_s={p: _med([w[p] for w in per_worker if p in w]) for p in keys
                            if any(p in w for w in per_worker)},
            phase_max_s={p: round(max(w[p] for w in per_worker if p in w), 2) for p in keys
                         if any(p in w for w in per_worker)})
    return out


def restored_flags(deploy_report: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per instance and worker rank: did the weights come from the snapshot?

    The field ``scripts/snapshot_validate.py check --require-restored`` requires is
    ``worker_init_phases[*]['weight_snapshot']['restored'] is True``.
    """
    flags: Dict[str, Dict[str, Optional[bool]]] = {}
    for inst in (deploy_report or {}).get("instances", []):
        init = inst["initialization"]
        name = init.get("instance_id") or inst["descriptor"]["instance_id"]
        flags[name] = {str(w.get("executor_rank")): (w.get("weight_snapshot") or {}).get("restored")
                       for w in init.get("worker_init_phases", [])}
    return {"per_instance": flags,
            "all_restored": bool(flags) and all(v is True for ranks in flags.values() for v in ranks.values())
            and all(ranks for ranks in flags.values())}


def drain_anatomy(drain_report: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-instance step durations from the ``timings`` ``drain_deployment`` adds to each entry."""
    out: Dict[str, Any] = {}
    for entry in (drain_report or {}).get("drained", []):
        steps = {k: round(v[1] - v[0], 3) for k, v in (entry.get("timings") or {}).items()}
        out[entry["instance_id"]] = steps
    return out


GATE_KEYS = ("parked_at", "released_at", "gate_returned_at", "lock_free_at", "vfio_free_at", "devices_ready_at")


def prespawn_anatomy(deploy_report: Optional[Mapping[str, Any]], T0: float) -> Dict[str, Any]:
    """Per-worker release-gate timeline (engine-host clocks, s after T0) of a committed pre-spawn."""
    workers: List[Dict[str, Any]] = []
    drivers: Dict[str, Any] = {}
    for inst in (deploy_report or {}).get("instances", []):
        init = inst["initialization"]
        name = init.get("instance_id") or inst["descriptor"]["instance_id"]
        drivers[name] = round(init["engine_init_started_at"] - T0, 3)
        for w in init.get("worker_init_phases", []):
            gate = w.get("prespawn_gate") or {}
            row: Dict[str, Any] = {"instance": name, "rank": w.get("executor_rank"), "host": gate.get("key")}
            row.update({k: round(gate[k] - T0, 4) for k in GATE_KEYS if gate.get(k) is not None})
            row["device_free_wait_s"] = gate.get("device_free_wait_s")
            row["holders_seen"] = gate.get("holders_seen")
            row["self_check_clean"] = bool((gate.get("self_check_parked") or {}).get("clean")
                                           and (gate.get("self_check_released") or {}).get("clean"))
            workers.append(row)
    block = dict((deploy_report or {}).get("prespawn") or {})
    hold = {iid: {k: round(v - T0, 4) for k, v in h.items() if k.endswith("_at") and v is not None}
            for iid, h in (block.get("tpu_hold") or {}).items()}
    clocks = {k: round(v - T0, 3) for k, v in (block.get("clocks") or {}).items()}
    return {"clock_note": "s after T0; workers on engine-host clocks, controller clocks on the head",
            "driver_engine_init_started_s": drivers, "controller_clocks_s": clocks, "tpu_hold_s": hold,
            "workers": workers, "all_parked_before_release": bool(workers) and all(
                r.get("parked_at") is not None and r.get("released_at") is not None
                and r["parked_at"] <= r["released_at"] for r in workers),
            "all_self_checks_clean": bool(workers) and all(r["self_check_clean"] for r in workers)}


def prespawn_window(stamps: Mapping[str, float], records1: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Phase-1 (source) delivery before / during / after the pre-spawn window, from client records.

    Window = [prespawn_begin, prespawn_parked]; useful output tokens of requests finishing in
    each interval over its length, so a source slowdown while the target imports shows up.
    """
    begin, parked = stamps.get("prespawn_begin"), stamps.get("prespawn_parked")
    p1, t0 = stamps.get("phase1_begin"), stamps.get("T0")
    out: Dict[str, Any] = {"phase1_begin": p1, "prespawn_begin": begin, "prespawn_parked": parked, "T0": t0}
    if None in (begin, p1, t0):
        return out
    end = min(parked, t0) if parked is not None else t0
    spans = {"before": (p1, begin), "during": (begin, end), "after": (end, t0)}
    done = [(r["client_finished_at_s"], int(r.get("useful_output_tokens") or 0)) for r in records1
            if r.get("client_finished_at_s") is not None and not r.get("error")]
    for name, (lo, hi) in spans.items():
        tokens = sum(n for t, n in done if lo < t <= hi)
        requests = sum(1 for t, _ in done if lo < t <= hi)
        out[name] = {"s": round(hi - lo, 3), "requests": requests, "useful_output_tokens": tokens,
                     "tokens_per_s": round(tokens / (hi - lo), 2) if hi > lo else None}
    out["parked_before_T0"] = parked is not None and parked <= t0
    return out


def rate_series(rows: Sequence[Tuple[str, float, int]], start: float, bucket: float = BIN_S) -> List[float]:
    acc: Dict[int, int] = {}
    for _, completed, emitted in rows:
        index = int((completed - start) // bucket)
        acc[index] = acc.get(index, 0) + emitted
    return [acc.get(i, 0) / bucket for i in range(max(acc) + 1)] if acc else []


def ramp_deficit(rows: Sequence[Tuple[str, float, int]], first: float) -> Optional[float]:
    """scripts/analyze_overlap.py:pilot's deficit: 10-s bins, steady = median of the middle 60 %."""
    series = rate_series(rows, first)
    n = len(series)
    if not n:
        return None
    middle = series[int(.2 * n):max(int(.8 * n), int(.2 * n) + 1)]
    steady = statistics.median(middle)
    return sum(max(steady - x, 0) for x in series[:12]) * 10 / steady if steady else None


def iteration_rows(iterations: Mapping[str, Mapping[str, Any]]) -> List[Tuple[str, float, int]]:
    return [(instance, float(row["completed_at_s"]), int(row.get("emitted_output_tokens") or 0))
            for instance, capture in iterations.items() for row in capture["rows"]
            if row.get("completed_at_s") is not None]


def _bins(rows: Sequence[Tuple[str, float, int]], instances: Iterable[str], origin: float) -> List[Dict[str, Any]]:
    wanted = set(instances)
    chosen = [r for r in rows if r[0] in wanted and r[1] >= origin]
    series = rate_series(chosen, origin)
    return [{"bin_start_s": round(i * BIN_S, 1), "tokens": round(rate * BIN_S), "tokens_per_s": round(rate, 2)}
            for i, rate in enumerate(series)]


def _tokens_between(rows: Sequence[Tuple[str, float, int]], instances: Iterable[str], lo: float, hi: float) -> int:
    wanted = set(instances)
    return sum(e for inst, t, e in rows if inst in wanted and lo < t <= hi)


def _first_tokens(records: Iterable[Mapping[str, Any]], instances: Optional[Iterable[str]] = None) -> Optional[float]:
    """Earliest engine-host first-token time (segment started_at_s + ttft_s)."""
    wanted = None if instances is None else set(instances)
    times = []
    for rec in records:
        for seg in rec.get("segments") or []:
            inst = seg.get("instance_id") or seg.get("instance")
            if seg.get("ttft_s") is None or seg.get("started_at_s") is None:
                continue
            if wanted is None or inst in wanted:
                times.append(seg["started_at_s"] + seg["ttft_s"])
    return min(times) if times else None


def _segments(T0: float, bounds: Sequence[Tuple[str, float, str, str]]) -> List[Dict[str, Any]]:
    out, prev = [], T0
    for name, end, cls, note in bounds:
        out.append({"segment": name, "start_rel_s": round(prev - T0, 4), "end_rel_s": round(end - T0, 4),
                    "s": round(end - prev, 4), "class": cls, "note": note})
        prev = end
    return out


def build_timeline(
    *, mode: str, T0: float, stamps: Mapping[str, float], records2: Sequence[Mapping[str, Any]],
    deploy_report: Optional[Mapping[str, Any]], drain_report: Optional[Mapping[str, Any]],
    iter_rows: Sequence[Tuple[str, float, int]], plan: Optional[Mapping[str, Any]],
    static_target_s: Optional[float],
) -> Dict[str, Any]:
    """Segments and endpoints from stamps + phase-2 records + iterations (online and offline)."""
    ok = [r for r in records2 if r.get("client_started_at_s") is not None]
    first_dispatch = min(r["client_started_at_s"] for r in ok)
    last_finish = max(r["client_finished_at_s"] for r in ok if r.get("client_finished_at_s") is not None)
    st = stamps
    anatomy = deploy_anatomy(deploy_report) if deploy_report else None
    drain_detail = drain_anatomy(drain_report)
    out: Dict[str, Any] = {"clock_note": CLOCK_NOTE, "drain_detail_s": drain_detail}
    ep: Dict[str, Any] = {"T0": T0, "first_phase2_dispatch": first_dispatch,
                          "last_phase2_completion": last_finish,
                          "target_makespan_s": round(last_finish - first_dispatch, 3)}
    capture_s = (st["capture_arm_end"] - st["capture_arm_begin"]) if "capture_arm_end" in st else None
    prespawned = "prespawn_wait_end" in st
    wake = [("controller_wake", st["controller_wake"], "controller", "T0 -> controller thread running")]
    if prespawned:
        wake.append(("prespawn_wait", st["prespawn_wait_end"], "engine",
                     "controller wake -> every pre-spawned worker parked (~0 when parked before T0)"))
    if mode == "full":
        bounds = wake + [("drain_release", st["drain_end"], "engine",
                          "per instance: maintenance/drain ack/stop_serving/shutdown/kill/PG release"),
                         ("gateway_down", st["gateway_down_end"], "controller", "stop + kill source gateway")]
        complete = bool(anatomy and anatomy["complete"] and anatomy.get("last_engine_ready") is not None)
        if prespawned:
            bounds.append(("prespawn_release", st["prespawn_release_end"], "controller",
                           "open the release gate for the target hosts + create TPU-hold groups"))
            if complete:
                bounds += [
                    ("device_handoff", anatomy["init_device_start_max"], "engine",
                     "release -> last worker past the gate, libtpu lock and /dev/vfio free (engine-host clocks)"),
                    ("engine_init", anatomy["last_engine_ready"], "engine",
                     "last init_device start -> last compile_or_warm_up end"),
                    ("endpoint_register", st["deploy_end"], "engine", "last engine ready -> commit() return")]
            else:
                bounds.append(("deploy_unsplit", st["deploy_end"], "engine",
                               "commit() report has no worker phase clocks"))
        elif complete:
            bounds += [
                ("deploy_controller_pre", anatomy["engine_init_started_at_min"], "controller",
                 "deploy() call -> first engine_init_started_at (backend/placement/actor)"),
                ("engine_process_start", anatomy["init_device_start_min"], "engine",
                 "engine_init_started_at -> first init_device"),
                ("engine_init", anatomy["last_engine_ready"], "engine",
                 "first init_device -> last compile_or_warm_up end"),
                ("endpoint_register", st["deploy_end"], "engine", "last engine ready -> deploy() return")]
        else:
            bounds.append(("deploy_unsplit", st["deploy_end"], "engine",
                           "deploy() report has no worker phase clocks"))
        bounds += [("gateway_up", st["gateway_up_end"], "controller", "create/bind/start the target gateway"),
                   ("gate_observe", st["gate_open:target"], "controller", "gateway_up return -> verified gate open"),
                   ("capture_arm", st["capture_arm_end"], "harness", "start_capture on the new gateway"),
                   ("dispatch", first_dispatch, "controller", "capture armed -> first target dispatch")]
        segments = _segments(T0, bounds)
        gap = first_dispatch - T0
        ft = _first_tokens(ok)
        ep.update(gap_dispatch_s=round(gap, 3),
                  gap_first_token_s=None if ft is None else round(ft - T0, 3),
                  capture_arm_s=None if capture_s is None else round(capture_s, 3))
        deficit = ramp_deficit(iter_rows, first_dispatch) if iter_rows else None
        ep["ramp_deficit_iter_s"] = None if deficit is None else round(deficit, 3)
        if static_target_s is not None:
            ramp = (last_finish - first_dispatch) - static_target_s
            ep.update(static_target_s=static_target_s, ramp_makespan_s=round(ramp, 3),
                      C_production_s=round(gap + ramp, 3),
                      C_production_ex_capture_s=None if capture_s is None else round(gap + ramp - capture_s, 3))
        if deficit is not None:
            ep["C_production_deficit_s"] = round(gap + deficit, 3)
            if capture_s is not None:
                ep["C_production_deficit_ex_capture_s"] = round(gap + deficit - capture_s, 3)
        if anatomy:
            out["deploy_anatomy"] = anatomy
    else:
        kept_recs = [r for r in ok if r.get("stream") == "kept"]
        join_recs = [r for r in ok if r.get("stream") == "joiner"]
        kept_ids, join_ids = list((plan or {}).get("kept", [])), list((plan or {}).get("join", []))
        first_kept = min((r["client_started_at_s"] for r in kept_recs), default=None)
        first_join = min((r["client_started_at_s"] for r in join_recs), default=None)
        if first_kept is None:
            raise ValueError("partial switch has no kept-class dispatch")
        bounds = wake + [("drain_release", st["drain_end"], "engine", "released engines only")]
        if prespawned:
            bounds.append(("prespawn_release", st["prespawn_release_end"], "controller",
                           "open the release gate for the joiners' hosts + create TPU-hold groups"))
        bounds += [("adopt", st["adopt_end"], "engine", "kept engines adopt the target layout"),
                  ("retarget", st["retarget_end"], "controller", "gateway retarget_layout"),
                  ("capture_arm", st["capture_arm_end"], "harness", "start_capture after retarget"),
                  ("gate_observe", st["gate_open:kept"], "controller", "capture armed -> verified kept gate"),
                  ("dispatch", first_kept, "controller", "gate open -> first kept dispatch")]
        segments = _segments(T0, bounds)
        kept_gap = first_kept - T0
        join_called, join_returned = st.get("join_begin"), st.get("join_end")
        ep.update(kept_gap_s=round(kept_gap, 3), gap_dispatch_s=round(kept_gap, 3),
                  join_delay_s=None if first_join is None else round(first_join - T0, 3),
                  joiner_timeline={
                      "deploy_begin_s": round(st["deploy_begin"] - T0, 3) if "deploy_begin" in st else None,
                      "deploy_ready_s": round(st["deploy_end"] - T0, 3) if "deploy_end" in st else None,
                      "join_called_s": None if join_called is None else round(join_called - T0, 3),
                      "join_returned_s": None if join_returned is None else round(join_returned - T0, 3),
                      "joiner_gate_open_s": round(st["gate_open:joiner"] - T0, 3) if "gate_open:joiner" in st else None,
                      "first_joiner_dispatch_s": None if first_join is None else round(first_join - T0, 3),
                      "first_joiner_token_s": None if not join_ids or _first_tokens(join_recs, join_ids) is None
                      else round(_first_tokens(join_recs, join_ids) - T0, 3),
                      **({"release_s": round(st["prespawn_release_end"] - T0, 3)} if prespawned else {})})
        last_kept = max((r["client_finished_at_s"] for r in kept_recs if r.get("client_finished_at_s")), default=None)
        last_join = max((r["client_finished_at_s"] for r in join_recs if r.get("client_finished_at_s")), default=None)
        if join_returned is not None and iter_rows:
            windows = {"during_join": (first_kept, join_returned)}
            if last_kept is not None and last_kept > join_returned:
                windows["after_join"] = (join_returned, last_kept)
            ep["kept_pair_throughput"] = {
                name: {"window_s": round(hi - lo, 3), "tokens": _tokens_between(iter_rows, kept_ids, lo, hi),
                       "tokens_per_s": round(_tokens_between(iter_rows, kept_ids, lo, hi) / (hi - lo), 2)
                       if hi > lo else None} for name, (lo, hi) in windows.items()}
        if first_join is not None and iter_rows:
            ep["joiner_ramp_bins"] = {"origin": "first joiner dispatch", "bin_s": BIN_S,
                                      "bins": _bins(iter_rows, join_ids, first_join)}
        ep["class_finish_s"] = {"kept": None if last_kept is None else round(last_kept - T0, 3),
                                "joiner": None if last_join is None else round(last_join - T0, 3)}
        if static_target_s is not None:
            ep.update(static_target_s=static_target_s,
                      C_eff_s=round((last_finish - T0) - static_target_s, 3))
        gap = kept_gap
    if prespawned:
        out["prespawn"] = prespawn_anatomy(deploy_report, T0)
    total = sum(s["s"] for s in segments)
    out.update(segments=segments, segment_sum_s=round(total, 4), gap_dispatch_s=round(gap, 4),
               segments_reconcile=abs(total - gap) <= 0.05, endpoints=ep)
    return out


# --------------------------------------------------------------------------- orchestration

@dataclasses.dataclass
class StreamSpec:
    name: str
    items: List[Dict[str, Any]]
    cap: int
    signal: Signal
    required: List[str]
    capture_before_dispatch: bool = False
    # Partial kept stream: the gateway must already be capturing (start_capture is idle-gated, so a
    # poll-fallback gate opened before the controller armed it would make start_capture refuse).
    require_capturing: bool = False


class SwitchAborted(RuntimeError):
    pass


def _names(value: Any) -> List[str]:
    return [n.strip() for n in (value.split(",") if isinstance(value, str) else value) if n.strip()]


def _request_bodies(manifests: Sequence[Mapping[str, Any]], namespace: str) -> List[Dict[str, Any]]:
    # Exactly run_arm's body: {request_id: "<ns>:<manifest id>", prompt_token_ids, max_tokens, ignore_eos}
    return [{"request_id": f"{namespace}:{row['request_id']}", "prompt_token_ids": row["prompt_token_ids"],
             "max_tokens": row["max_tokens"], "ignore_eos": True}
            for m in manifests for row in m["requests"]]


def _stream_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    starts = [r["client_started_at_s"] for r in records if r.get("client_started_at_s") is not None]
    ends = [r["client_finished_at_s"] for r in records if r.get("client_finished_at_s") is not None]
    return {"requests": len(records), "errors": sum(1 for r in records if r.get("error")),
            "useful_output_tokens": sum(int(r.get("useful_output_tokens") or 0) for r in records),
            "first_dispatch": min(starts, default=None), "last_finish": max(ends, default=None)}


def gate_verified(status: Optional[Mapping[str, Any]], layout_id: str, required: Iterable[str]) -> bool:
    """Required instances bound under the target layout: not joining, no maintenance, discovery ready, serving."""
    if not status:
        return False
    need = set(required)
    return bool(status["layout_id"] == layout_id and status["discovery_ready"] and status["serving"]
                and not status["maintenance_instances"] and need <= set(status["instances"])
                and not (need & set(status["joining"])))


class _Switch:
    def __init__(self, fabric: Any, client: Any, *, mode: str, source_layout: str, target_layout: str,
                 phase1: str, phase2: Any, run_id: str, policy: Optional[Mapping[str, Any]],
                 static_target_s: Optional[float], artifact_root: pathlib.Path, arm_timeout_s: float,
                 manifest_loader: Callable[[str], Mapping[str, Any]], gate_poll_s: float,
                 event_grace_s: float, clock: Callable[[], float], prespawn_at_s: float = 0.0,
                 prespawn_park_timeout_s: float = 900.0) -> None:
        self.fabric, self.client = fabric, client
        self.mode, self.source, self.target = mode, source_layout, target_layout
        self.phase1_name, self.phase2_names = phase1, _names(phase2)
        self.run_id, self.amend, self.static_target_s = run_id, dict(policy or {}), static_target_s
        self.timeout_s, self.loader = arm_timeout_s, manifest_loader
        self.gate_poll_s, self.event_grace_s = gate_poll_s, event_grace_s
        self.out_dir = artifact_root / run_id
        self.log = StampLog(clock)
        self.t0_event = threading.Event()
        self.abort = threading.Event()
        self.signals = {n: Signal(n) for n in ("target_routes", "kept_routes", "joiners_route")}
        self.report: Dict[str, Any] = {
            "run_id": run_id, "mode": mode, "source_layout": source_layout, "target_layout": target_layout,
            "phase1": phase1, "phase2": self.phase2_names, "policy_amendment": self.amend,
            "static_target_s": static_target_s, "clock_note": CLOCK_NOTE, "completed": False,
            "error": None, "failed_step": None, "artifact_dir": str(self.out_dir)}
        self.step = "preflight"
        self.controller_exc: Optional[BaseException] = None
        self.drain_report: Optional[Dict[str, Any]] = None
        self.deploy_report: Optional[Dict[str, Any]] = None
        self.gateway_up_report: Optional[Dict[str, Any]] = None
        self.retarget_receipt: Optional[Dict[str, Any]] = None
        self.join_receipt: Optional[Dict[str, Any]] = None
        self.capture_armed = False
        self.records1: List[Dict[str, Any]] = []
        self.records2: List[Dict[str, Any]] = []
        self.T0: Optional[float] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._tasks: Dict[str, asyncio.Task] = {}
        self.cancelled_streams: List[str] = []
        self.gate_info: Dict[str, Any] = {}
        # Pre-spawn (decided in preflight from the target engine settings).
        self.prespawn = False
        self.prespawn_ids: Optional[List[str]] = None
        self.prespawn_at_s, self.prespawn_park_timeout_s = float(prespawn_at_s), float(prespawn_park_timeout_s)
        self.prespawn_thread: Optional[threading.Thread] = None
        self.prespawn_exc: Optional[BaseException] = None
        self.committed = False
        self._phase1_task: Optional[asyncio.Task] = None

    # ----- pre-flight (not timed)
    def preflight(self) -> None:
        log = self.log
        log.mark("preflight_begin")
        rs = self.fabric.routing_status()
        if (not rs or rs["layout_id"] != self.source or rs["joining"] or rs["maintenance_instances"]
                or rs["capturing"] or not rs["discovery_ready"] or not rs["serving"] or not rs["instances"]):
            raise RuntimeError(f"gateway is not idle, serving and discovery-ready on source layout "
                               f"{self.source}: {rs}")
        pf = self.fabric.preflight(self.mode, self.source, self.target)
        if set(rs["instances"]) != set(pf["source_instances"]):
            raise RuntimeError("gateway does not bind every source instance")
        self.pf = pf
        self.plan = pf.get("plan")
        self.manifest1 = self.loader(self.phase1_name)
        self.manifests2 = [self.loader(n) for n in self.phase2_names]
        self.ns1, self.ns2 = uuid.uuid4().hex, uuid.uuid4().hex
        self.items1 = _request_bodies([self.manifest1], self.ns1)
        self.items2 = _request_bodies(self.manifests2, self.ns2)
        self.max_tokens1 = {i["request_id"]: i["max_tokens"] for i in self.items1}
        self.max_tokens2 = {i["request_id"]: i["max_tokens"] for i in self.items2}
        if len(self.max_tokens2) != len(self.items2) or len(self.max_tokens1) != len(self.items1):
            raise ValueError("duplicate request ids in a phase")
        self.source_policy = merged_policy(self.amend, pf["source_instances"], self.source)
        self.target_policy = merged_policy(self.amend, pf["target_instances"], self.target)
        if self.mode == "partial":
            if not self.plan or not self.plan.get("kept") or not self.plan.get("join") or not pf.get("kept_max_model_len"):
                raise RuntimeError("partial switch needs a kept/join plan and the kept max_model_len")
            from . import transition as tr

            self.policy_after = tr.policy_after_join(arm_for_layout(self.target), self.amend)
        deployed = list(pf["target_instances"]) if self.mode == "full" else list(self.plan["join"])
        self.prespawn = bool(self.fabric.prespawn_requested(deployed))
        if self.prespawn:
            if self.prespawn_at_s < 0:
                raise ValueError("prespawn_at_s must be >= 0")
            self.prespawn_ids = None if self.mode == "full" else deployed
            self.report["prespawn"] = {"enabled": True, "at_s": self.prespawn_at_s, "instances": deployed,
                                       "park_timeout_s": self.prespawn_park_timeout_s}
        self.fabric.quiesce()
        applied = self.fabric.set_policy(self.source_policy)
        from .gateway.policy import RoutingPolicy

        if RoutingPolicy.from_dict(applied).to_dict() != self.source_policy:
            raise RuntimeError("gateway applied policy differs from the declared source policy")
        self.streams = self._build_streams()
        self.report["streams"] = {s.name: {"rows": len(s.items), "cap": s.cap, "required": s.required}
                                  for s in self.streams}
        self.report["preflight"] = {k: v for k, v in pf.items()}
        log.mark("preflight_end")

    def _build_streams(self) -> List[StreamSpec]:
        if self.mode == "full":
            return [StreamSpec("target", self.items2, ADMITTED_CAP, self.signals["target_routes"],
                               list(self.pf["target_instances"]), capture_before_dispatch=True)]
        limit = int(self.pf["kept_max_model_len"])
        kept, joiner = [], []
        for item in self.items2:
            context = len(item["prompt_token_ids"]) + int(item["max_tokens"])
            (kept if context <= limit else joiner).append(item)
        joiner_cap = min(len(joiner), ADMITTED_CAP)
        streams = []
        if kept:
            streams.append(StreamSpec("kept", kept, ADMITTED_CAP - joiner_cap, self.signals["kept_routes"],
                                      list(self.plan["kept"]), require_capturing=True))
        if joiner:
            streams.append(StreamSpec("joiner", joiner, joiner_cap, self.signals["joiners_route"],
                                      list(self.plan["join"])))
        if not kept:
            raise RuntimeError("phase 2 has no kept-class rows")
        return streams

    # ----- pre-spawn thread (started with phase 1)
    def _prespawn_main(self) -> None:
        """Spawn the target engines ``prespawn_at_s`` after phase 1 begins (at T0 at the latest)
        and wait until every worker is parked; a failure before T0 cancels phase 1."""
        try:
            delay = self.log.stamps["phase1_begin"] + self.prespawn_at_s - self.log.clock()
            if delay > 0:
                self.t0_event.wait(delay)
            if self.abort.is_set():
                return
            with self.log.span("prespawn"):
                self.report["prespawn"]["spawn"] = self.fabric.prespawn(self.prespawn_ids)
            with self.log.span("prespawn_park"):
                parked = self.fabric.await_parked(self.prespawn_park_timeout_s, stop=self.abort.is_set)
            self.log.mark("prespawn_parked")
            self.report["prespawn"]["parked"] = parked
        except BaseException as exc:  # noqa: BLE001 - surfaced by the controller or phase 1
            self.log.mark("prespawn_error")
            self.prespawn_exc = exc
            self.report["prespawn"]["error"] = f"{type(exc).__name__}: {str(exc)[-2000:]}"
            if self.abort.is_set():
                return  # the run already failed elsewhere; this is its consequence
            if self.T0 is None:
                self.report["failed_step"] = "prespawn"
                self.report["error"] = self.report["prespawn"]["error"]
                loop, task = self._loop, self._phase1_task
                if loop is not None and task is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(task.cancel)  # fail closed before T0: stop phase 1

    def _await_prespawn(self) -> None:
        thread = self.prespawn_thread
        if thread is None:
            raise RuntimeError("pre-spawn was never started")
        thread.join(timeout=self.prespawn_park_timeout_s + 1800.0)
        if thread.is_alive():
            raise TimeoutError("pre-spawn thread did not finish")
        if self.prespawn_exc is not None:
            raise RuntimeError(f"pre-spawn failed: {type(self.prespawn_exc).__name__}: "
                               f"{str(self.prespawn_exc)[-1500:]}") from self.prespawn_exc
        self.report["prespawn"]["parked_before_T0"] = self.log.stamps["prespawn_parked"] <= self.T0

    def abort_prespawn(self) -> None:
        """Failure path: remove only the uncommitted pre-spawn's own actors, groups and gate."""
        if not self.prespawn or self.committed:
            return
        if self.prespawn_thread is not None:
            self.prespawn_thread.join(timeout=900.0)
        try:
            self.report["prespawn_abort"] = self.fabric.abort_prespawn()
        except Exception as exc:  # noqa: BLE001 - the error itself is evidence
            self.report["prespawn_abort_error"] = f"{type(exc).__name__}: {str(exc)[-600:]}"

    # ----- controller thread
    def _controller(self) -> None:
        try:
            self.t0_event.wait()
            if self.abort.is_set():
                return
            self.log.mark("controller_wake")
            if self.prespawn:
                with self._step("prespawn_wait"):
                    self._await_prespawn()
            (self._controller_full if self.mode == "full" else self._controller_partial)()
        except BaseException as exc:  # noqa: BLE001 - recorded, streams cancelled, re-raised by run_switch
            self.log.mark("controller_error")
            self.controller_exc = exc
            self.report.setdefault("failed_step", self.step)
            self.report["error"] = f"{type(exc).__name__}: {str(exc)[-2000:]}"
            self.abort.set()
            self._cancel_streams()

    def _step(self, name: str):
        self.step = name
        return self.log.span(name)

    def _controller_full(self) -> None:
        f = self.fabric
        with self._step("drain"):
            self.drain_report = f.drain(None)
        with self._step("gateway_down"):
            f.gateway_down()
        if self.prespawn:
            with self._step("prespawn_release"):
                self.report["prespawn"]["release"] = f.release(None)
            with self._step("deploy"):
                self.deploy_report = f.commit(None)
                self.committed = True
        else:
            with self._step("deploy"):
                self.deploy_report = f.deploy(None)
        with self._step("gateway_up"):
            self.gateway_up_report = f.gateway_up(self.target_policy)
        self.log.mark("signal:target_routes")
        self.signals["target_routes"].set()

    def _controller_partial(self) -> None:
        f, joiners = self.fabric, list(self.plan["join"])
        with self._step("drain"):
            self.drain_report = f.drain(list(self.plan["release"]))
        if self.prespawn:
            with self._step("prespawn_release"):
                self.report["prespawn"]["release"] = f.release(joiners)
        with self._step("adopt"):
            f.adopt()
        with self._step("retarget"):
            self.retarget_receipt = f.retarget(joiners)
        with self._step("capture_arm"):
            f.start_capture()
            self.capture_armed = True
        self.log.mark("signal:kept_routes")
        self.signals["kept_routes"].set()
        with self._step("deploy"):
            if self.prespawn:
                self.deploy_report = f.commit(joiners)
                self.committed = True
            else:
                self.deploy_report = f.deploy(joiners)
        with self._step("join"):
            self.join_receipt = f.join(joiners, self.policy_after)
        self.log.mark("signal:joiners_route")
        self.signals["joiners_route"].set()

    def _cancel_streams(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def cancel() -> None:
            for name, task in self._tasks.items():
                if not task.done():
                    self.cancelled_streams.append(name)
                    task.cancel()

        loop.call_soon_threadsafe(cancel)

    # ----- client side
    async def _wait_gate(self, spec: StreamSpec) -> None:
        started = time.monotonic()
        verified_since: Optional[float] = None
        errors = 0
        while True:
            if self.abort.is_set():
                raise SwitchAborted(f"gate {spec.name} aborted")
            if time.monotonic() - started > self.timeout_s:
                raise TimeoutError(f"gate {spec.name} did not open within {self.timeout_s:.0f} s")
            try:
                status = await asyncio.to_thread(self.fabric.routing_status)
            except Exception:  # noqa: BLE001 - gateway may be mid-restart; keep polling
                status, errors = None, errors + 1
            if gate_verified(status, self.target, spec.required) and (
                    not spec.require_capturing or bool((status or {}).get("capturing"))):
                now = time.monotonic()
                verified_since = verified_since if verified_since is not None else now
                if spec.signal.is_set():
                    via = "event"
                elif now - verified_since >= self.event_grace_s:
                    via = "poll_fallback"  # the controller event is late; routing_status is authoritative
                else:
                    via = None
                if via:
                    self.log.mark(f"gate_open:{spec.name}")
                    self.gate_info[spec.name] = {"via": via, "status_errors": errors}
                    return
            if spec.signal.is_set():
                await asyncio.sleep(0.05)  # event is up but the gateway view is not yet: do not spin
            else:
                await spec.signal.wait(self.gate_poll_s)

    async def _stream(self, spec: StreamSpec) -> List[Dict[str, Any]]:
        await self._wait_gate(spec)
        if spec.capture_before_dispatch:
            # Full mode: the new gateway's capture is armed after it verifiably routes and before
            # the first target dispatch (partial mode armed it inside the controller).
            with self.log.span("capture_arm"):
                await asyncio.to_thread(self.fabric.start_capture)
            self.capture_armed = True
        records = await self.client.dispatch(spec.items, concurrency=spec.cap, timeout_s=self.timeout_s,
                                             label=spec.name)
        for record in records:
            record["stream"] = spec.name
        return records

    async def _client_main(self) -> None:
        self._loop = asyncio.get_running_loop()
        for signal in self.signals.values():
            signal.bind(self._loop)
        await self.client.open()
        try:
            self.log.mark("phase1_begin")
            self._phase1_task = asyncio.ensure_future(self.client.dispatch(
                self.items1, concurrency=ADMITTED_CAP, timeout_s=self.timeout_s, label="phase1"))
            if self.prespawn:
                self.prespawn_thread = threading.Thread(target=self._prespawn_main, name="omp-switch-prespawn",
                                                        daemon=True)
                self.prespawn_thread.start()
            try:
                self.records1 = await self._phase1_task
            except asyncio.CancelledError:
                if self.prespawn_exc is None:
                    raise
                raise RuntimeError(f"pre-spawn failed before T0, phase 1 cancelled: "
                                   f"{type(self.prespawn_exc).__name__}: {str(self.prespawn_exc)[-1500:]}"
                                   ) from self.prespawn_exc
            failed = [r for r in self.records1 if r.get("error")]
            finished = [r["client_finished_at_s"] for r in self.records1 if r.get("client_finished_at_s")]
            if failed or not finished:
                raise RuntimeError(f"phase 1 delivered {len(failed)} errors; first: "
                                   f"{failed[0]['error'] if failed else 'no records'}")
            self.T0 = max(finished)
            self.log.mark("T0", self.T0)
            self.log.mark("phase1_end")
            self.t0_event.set()  # the controller starts here
            self.log.mark("t0_event_set")
            self._tasks = {s.name: asyncio.create_task(self._stream(s)) for s in self.streams}
            results = await asyncio.gather(*self._tasks.values(), return_exceptions=True)
            for name, result in zip(self._tasks, results):
                if isinstance(result, (asyncio.CancelledError, SwitchAborted)):
                    self.cancelled_streams.append(name)
            errors = [r for r in results if isinstance(r, BaseException)
                      and not isinstance(r, (asyncio.CancelledError, SwitchAborted))]
            if errors and self.controller_exc is None:
                raise errors[0]
            for result in results:
                if isinstance(result, list):
                    self.records2.extend(result)
        finally:
            if self.controller_exc is not None or self.T0 is None:
                self.abort.set()
            await self.client.close()

    # ----- finalize
    def _accounting(self, tag: str, authoritative: List[Dict[str, Any]], records: List[Dict[str, Any]],
                    max_tokens: Dict[str, int], manifests: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        from .qualify_migration import _check_accounting

        accounting = _check_accounting(authoritative, max_tokens)
        by_id = {r["request_id"]: r for r in authoritative}
        delivery = (len(authoritative) == len(by_id) == len(records) == len(max_tokens)
                    and all(not r.get("error") and r["output_token_ids"] ==
                            by_id.get(r["request_id"], {}).get("output_token_ids") for r in records))
        useful = sum(int(r.get("useful_output_tokens") or 0) for r in records if not r.get("error"))
        expected = sum(int(m["useful_output_tokens"]) for m in manifests)
        flags = {"requests": len(records), "expected_requests": sum(len(m["requests"]) for m in manifests),
                 "errors": sum(1 for r in records if r.get("error")),
                 "useful_output_tokens": useful, "expected_useful_output_tokens": expected,
                 "all_requests_delivered": len(records) == sum(len(m["requests"]) for m in manifests),
                 "exact_tokens_per_manifest": useful == expected and accounting["all_budgets_exact"],
                 "no_errors": not any(r.get("error") for r in records),
                 "all_budgets_exact": accounting["all_budgets_exact"],
                 "no_duplicate_delivery": accounting["no_duplicate_delivery"],
                 "client_delivery_matches": delivery}
        if authoritative is records:
            # Client authority (full mode, phase 1: the source journal went with the source gateway).
            # The HTTP response's per-segment summary carries counts, not token ids, so the committed-
            # prefix check reduces to: one segment, and delivered = accounted = segment tokens.
            flags["authority"] = "client records (source gateway journal discarded with the source gateway)"
            flags["no_duplicate_delivery"] = bool(records) and all(
                not r.get("error") and len(r.get("segments") or []) == 1
                and len(r["output_token_ids"]) == r["useful_output_tokens"]
                == sum(int(s.get("output_tokens") or 0) for s in r["segments"]) for r in records)
        flags["complete"] = bool(flags["all_requests_delivered"] and flags["exact_tokens_per_manifest"]
                                 and flags["no_errors"] and flags["no_duplicate_delivery"] and delivery)
        return {"flags": flags, "accounting": accounting}

    def dump_evidence(self) -> None:
        """Failure path: client records so far and the gateway journal/traces, before any stop_capture
        (in E2 stop_capture itself raised, which would otherwise lose everything)."""
        out = self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        sinks = getattr(self.client, "sinks", {}) or {}
        try:
            _write_gz(out / "phase1_records.json.gz", self.records1 or sinks.get("phase1", []))
            partial2 = self.records2 or [dict(r, stream=n) for n, rows in sinks.items() if n != "phase1"
                                         for r in rows]
            _write_gz(out / "phase2_records_partial.json.gz", partial2)
            self.report["failure_records"] = {n: len(rows) for n, rows in sinks.items()}
        except Exception as exc:  # noqa: BLE001
            self.report["failure_records_error"] = f"{type(exc).__name__}: {exc}"[:300]
        for name, fetch in (("gateway_records", self.fabric.records),
                            ("traces", lambda: self.fabric.traces(200000)),
                            ("routing_status", self.fabric.routing_status)):
            try:
                _write_gz(out / f"failure_{name}.json.gz", fetch())
            except Exception as exc:  # noqa: BLE001 - best effort; the error itself is evidence
                self.report[f"failure_{name}_error"] = f"{type(exc).__name__}: {str(exc)[-600:]}"

    def finalize(self) -> None:
        f, out = self.fabric, self.out_dir
        self.log.mark("finalize_begin")
        for r in self.records2:
            r["canonical_request_id"] = r["request_id"].removeprefix(f"{self.ns2}:")
        for r in self.records1:
            r["canonical_request_id"] = r["request_id"].removeprefix(f"{self.ns1}:")
        # Evidence first: records, journal and traces are written before stop_capture, whose failure
        # (as in E2) is recorded instead of losing them.
        _write_gz(out / "phase1_records.json.gz", self.records1)
        _write_gz(out / "phase2_records.json.gz", self.records2)
        gateway_records = f.records()
        traces = f.traces(200000)
        _write_gz(out / "gateway_records.json.gz", gateway_records)
        _write_gz(out / "traces.json.gz", traces)
        try:
            iterations = f.stop_capture()
        except Exception as exc:  # noqa: BLE001 - run is marked incomplete below
            iterations = {}
            self.report["stop_capture_error"] = f"{type(exc).__name__}: {str(exc)[-1000:]}"
        self.capture_armed = False
        self.log.mark("finalize_fetched")
        for spec in self.streams:  # client-clock dispatch/finish instants, stamped into the timeline
            summary = _stream_summary([r for r in self.records2 if r.get("stream") == spec.name])
            if summary["first_dispatch"] is not None:
                self.log.mark(f"first_dispatch:{spec.name}", summary["first_dispatch"])
                self.log.mark(f"last_finish:{spec.name}", summary["last_finish"])
        auth2 = [r for r in gateway_records if r["request_id"].startswith(f"{self.ns2}:")]
        # Engine-host first-token clocks live in the journal's segments, not the HTTP summary: copy them
        # onto the client records (same request, same segment order) so first-token endpoints exist.
        journal = {r["request_id"]: r for r in auth2}
        for r in self.records2:
            segs = (journal.get(r["request_id"]) or {}).get("segments") or []
            for mine, theirs in zip(r.get("segments") or [], segs):
                for key in ("started_at_s", "ttft_s", "finished_at_s"):
                    if theirs.get(key) is not None:
                        mine.setdefault(key, theirs[key])
        _write_gz(out / "phase2_records.json.gz", self.records2)
        auth1 = [r for r in gateway_records if r["request_id"].startswith(f"{self.ns1}:")]
        if self.mode == "partial" and len(auth1) == len(self.records1):
            acc1 = self._accounting("phase1", auth1, self.records1, self.max_tokens1, [self.manifest1])
        else:
            acc1 = self._accounting("phase1", self.records1, self.records1, self.max_tokens1, [self.manifest1])
        acc2 = self._accounting("phase2", auth2, self.records2, self.max_tokens2, self.manifests2)
        _write_gz(out / "accounting.json.gz", {"phase1": acc1["accounting"], "phase2": acc2["accounting"]})
        stats = _write_iterations(out, "phase2", 1, iterations)
        rows = iteration_rows(iterations)
        timeline = build_timeline(mode=self.mode, T0=self.T0, stamps=self.log.stamps, records2=self.records2,
                                  deploy_report=self.deploy_report, drain_report=self.drain_report,
                                  iter_rows=rows, plan=self.plan, static_target_s=self.static_target_s)
        restored = restored_flags(self.deploy_report)
        complete = bool(acc1["flags"]["complete"] and acc2["flags"]["complete"] and stats["one_to_one"]
                        and timeline["segments_reconcile"])
        self.report.update(
            T0=self.T0, timeline=timeline, segments=timeline["segments"], endpoints=timeline["endpoints"],
            restored=restored, iteration_capture_one_to_one=stats["one_to_one"],
            iteration_rows=stats["rows"], accounting={"phase1": acc1["flags"], "phase2": acc2["flags"]},
            phase1_summary=_stream_summary(self.records1),
            phase2_streams={n: _stream_summary([r for r in self.records2 if r.get("stream") == n])
                            for n in self.report["streams"]},
            gates=self.gate_info, completed=complete)
        if self.prespawn:
            self.report["prespawn"]["window"] = prespawn_window(self.log.stamps, self.records1)

    def write_report(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        r = self.report
        r["stamps"] = dict(self.log.stamps)
        r["stamp_events"] = list(self.log.events)
        r["streams_cancelled"] = sorted(set(self.cancelled_streams))
        if self.T0 is not None:
            r.setdefault("T0", self.T0)
        # Heavy evidence (worker profiles, drain receipts) goes to gz artifacts, not the compact report.
        if self.drain_report is not None:
            _write_gz(self.out_dir / "drain_report.json.gz", self.drain_report)
        if self.deploy_report is not None:
            _write_gz(self.out_dir / "deploy_report.json.gz", self.deploy_report)
            r.setdefault("restored", restored_flags(self.deploy_report))
            r["deploy_stage_clocks"] = self.deploy_report.get("stage_clocks")
        if self.retarget_receipt is not None:
            r["retarget"] = {k: self.retarget_receipt.get(k) for k in ("layout_id", "joining", "retargeted_at")}
        if self.join_receipt is not None:
            r["join"] = {k: self.join_receipt.get(k) for k in ("joined", "joined_at", "policy_roles", "remaining_joining")}
        (self.out_dir / "switch_report.json").write_text(json.dumps(r, indent=2, default=str) + "\n")
        emit_event("SWITCH_REPORT", path=str(self.out_dir / "switch_report.json"), completed=r["completed"])


def run_switch(
    fabric: Any, client: Any, *, mode: str, source_layout: str, target_layout: str,
    phase1: str, phase2: Any, run_id: str, policy: Optional[Mapping[str, Any]] = None,
    static_target_s: Optional[float] = None, artifact_root: Optional[pathlib.Path] = None,
    arm_timeout_s: float = 7200.0, manifest_loader: Callable[[str], Mapping[str, Any]] = load_manifest,
    gate_poll_s: float = 0.25, event_grace_s: float = 2.0, clock: Callable[[], float] = time.time,
    prespawn_at_s: float = 0.0, prespawn_park_timeout_s: float = 900.0,
) -> Dict[str, Any]:
    """Phase 1 -> T0 -> in-process controller -> gated phase 2; returns the compact report.

    Any failure is stamped and recorded in ``switch_report.json``, the client streams are
    cancelled, and the exception is re-raised after the report is written; nothing is retried.
    """
    if mode not in MODES:
        raise ValueError(f"switch mode must be one of {MODES}")
    root = pathlib.Path(artifact_root) if artifact_root is not None else ARTIFACT_ROOT
    sw = _Switch(fabric, client, mode=mode, source_layout=source_layout, target_layout=target_layout,
                 phase1=phase1, phase2=phase2, run_id=run_id, policy=policy, static_target_s=static_target_s,
                 artifact_root=root, arm_timeout_s=arm_timeout_s, manifest_loader=manifest_loader,
                 gate_poll_s=gate_poll_s, event_grace_s=event_grace_s, clock=clock,
                 prespawn_at_s=prespawn_at_s, prespawn_park_timeout_s=prespawn_park_timeout_s)
    if (sw.out_dir / "switch_report.json").exists():
        raise FileExistsError(f"refusing to overwrite retained evidence in {sw.out_dir}")
    sw.out_dir.mkdir(parents=True, exist_ok=True)
    controller = threading.Thread(target=sw._controller, name="omp-switch-controller", daemon=True)
    failure: Optional[BaseException] = None
    try:
        sw.preflight()
        controller.start()
        asyncio.run(sw._client_main())
        controller.join()
        if sw.controller_exc is not None:
            raise sw.controller_exc
        sw.step = "finalize"
        sw.finalize()
    except BaseException as exc:  # noqa: BLE001 - report first, then re-raise
        failure = exc
        sw.log.mark("run_error")
        sw.abort.set()
        sw.t0_event.set()
        sw.report["error"] = sw.report.get("error") or f"{type(exc).__name__}: {str(exc)[-2000:]}"
        sw.report["failed_step"] = sw.report.get("failed_step") or sw.step
        if controller.is_alive():
            controller.join(timeout=600.0)
            sw.report["controller_still_running"] = controller.is_alive()
        sw.abort_prespawn()
        if sw.step != "finalize":  # finalize already wrote its evidence before failing
            sw.dump_evidence()
        if sw.capture_armed:
            try:  # never leave the fabric armed for a run that will not be recorded
                fabric.stop_capture()
                sw.report["capture_disarmed_after_failure"] = True
            except Exception as stop_exc:  # noqa: BLE001
                sw.report["capture_disarm_error"] = f"{type(stop_exc).__name__}: {stop_exc}"[:300]
        raise
    finally:
        if failure is not None:
            sw.report["completed"] = False
        sw.write_report()
    return sw.report
