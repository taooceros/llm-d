"""Partial reconfiguration between built-in layouts on a live fabric.

Defined pairs (``PARTIAL_TRANSITIONS``), each matched by instance name:

    S  <-> M    kept small_a, small_b      released/joined large       <-> small_c, small_d
    S  <-> Mb   kept small_a, small_b      released/joined large_b     <-> small_c, small_d
    Mb <-> H    kept large_b               released/joined small_a, small_b <-> large_a

S = small_a..small_d (TP8); M = small_a, small_b (TP8) + large (TP16); Mb = M with the
TP16 instance named large_b; H = large_a, large_b (TP16).  First-fit placement puts
each kept instance on the same chips and hosts in both layouts, so kept engines are
never restarted: they adopt the target layout's identity in place while only the
released engines are drained and only the new engines are deployed (~355 s engine
start-up without snapshot restore).  Phase 2 starts serving on the kept engines
immediately and the new engines join the live gateway additively when ready.
S -> Mb -> H is the staged route from 4xTP8 to 2xTP16 that never stops serving.

    phase 1  run_arm on the source layout (optional)
    prepare  verify kept engines occupy identical chips; register the target layout
    drain    scoped drain of the released engines (gateway maintenance, as qualified)
    adopt    kept engines report + register under the target layout (same generation)
    retarget gateway moves to the target layout; new instance ids are 'joining'
    deploy   new engines, in a thread, concurrent with phase 2
    phase 2  run_arm (unfrozen pilot) with a hook that joins the new engines when ready
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Unordered layout pairs with a defined live partial transition.  plan_transition still
# checks chip conservation, TP and (when given) placement for every pair.
PARTIAL_TRANSITIONS = frozenset({
    frozenset({"S", "M"}),
    frozenset({"S", "Mb"}),
    frozenset({"Mb", "H"}),
})
SUPPORTED_LAYOUTS = frozenset().union(*PARTIAL_TRANSITIONS)


@dataclasses.dataclass(frozen=True)
class TransitionPlan:
    from_layout: str
    to_layout: str
    kept: Tuple[str, ...]
    release: Tuple[str, ...]
    join: Tuple[str, ...]


def instance_placement(instance: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Physical identity of a resolved instance: rank-ordered hosts and their chips."""
    return (instance["tp_size"], tuple(instance["host_ips"]),
            tuple((h["node_ip"], tuple(tuple(c) for c in h["chip_coords"]))
                  for h in instance["hosts"]))


def plan_transition(
    from_layout: str, to_layout: str,
    from_instances: Mapping[str, int], to_instances: Mapping[str, int],
    *,
    from_placement: Optional[Mapping[str, Mapping[str, Any]]] = None,
    to_placement: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> TransitionPlan:
    """Membership arithmetic for a defined pair; refuses anything not chip-conserving.

    ``*_placement`` map instance ids to resolved instance dicts (``host_ips``,
    ``hosts[].chip_coords``); when given, every kept instance must occupy exactly
    the same hosts, in the same rank order, and the same chips in both layouts.
    """
    pair = frozenset({from_layout, to_layout})
    if from_layout == to_layout or pair not in PARTIAL_TRANSITIONS:
        supported = sorted("<->".join(sorted(p)) for p in PARTIAL_TRANSITIONS)
        raise ValueError(f"no partial transition is defined for {from_layout}->{to_layout}; "
                         f"supported: {supported}")
    kept = sorted(set(from_instances) & set(to_instances))
    release = sorted(set(from_instances) - set(to_instances))
    join = sorted(set(to_instances) - set(from_instances))
    if not kept or not release or not join:
        raise ValueError("a partial transition needs kept, released and joining instances")
    if any(from_instances[n] != to_instances[n] for n in kept):
        raise ValueError("a kept instance may not change tensor-parallel size")
    if sum(from_instances.values()) != sum(to_instances.values()):
        raise ValueError("source and target layouts must allocate the same number of chips")
    if (from_placement is None) != (to_placement is None):
        raise ValueError("pass both source and target placements, or neither")
    if from_placement is not None:
        missing = sorted(n for n in kept if n not in from_placement or n not in to_placement)
        if missing:
            raise ValueError(f"no placement for kept instances {missing}")
        moved = sorted(n for n in kept
                       if instance_placement(from_placement[n]) != instance_placement(to_placement[n]))
        if moved:
            raise ValueError(f"kept instances would change hosts or chips: {moved}")
    return TransitionPlan(from_layout, to_layout, tuple(kept), tuple(release), tuple(join))


def partition_records(
    records: Sequence[Mapping[str, Any]], join_started_at: float, join_finished_at: float,
) -> Dict[str, Any]:
    """Split delivered records by *admitted* time relative to the join call.

    pre  : admitted before join_instances() was called (new engines absent)
    mid  : admitted while the join call was in flight
    post : admitted after the join returned (new engines serving)
    """
    groups: Dict[str, List[Mapping[str, Any]]] = {"pre_join": [], "during_join": [], "post_join": []}
    for row in records:
        admitted = row.get("client_started_at_s")
        if admitted is None:
            raise ValueError(f"record {row.get('request_id')} has no admitted time")
        key = ("pre_join" if admitted < join_started_at
               else "post_join" if admitted >= join_finished_at else "during_join")
        groups[key].append(row)
    out: Dict[str, Any] = {"join_started_at": join_started_at, "join_finished_at": join_finished_at}
    for key, rows in groups.items():
        placed: Dict[str, int] = {}
        for row in rows:
            if row.get("initial_instance"):
                placed[row["initial_instance"]] = placed.get(row["initial_instance"], 0) + 1
        out[key] = {
            "requests": len(rows),
            "errors": sum(1 for r in rows if r.get("error")),
            "useful_output_tokens": sum(int(r.get("useful_output_tokens") or 0) for r in rows),
            "initial_placement": dict(sorted(placed.items())),
            "first_admitted_at": min((r["client_started_at_s"] for r in rows), default=None),
            "last_finished_at": max((r["client_finished_at_s"] for r in rows
                                     if r.get("client_finished_at_s") is not None), default=None),
        }
    return out


_SUMMARY_KEYS = ("makespan_s", "useful_tokens_per_s", "useful_output_tokens", "all_requests_complete",
                 "started_wall", "finished_wall", "artifact_dir", "instances", "layout_id")


def _digest(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: summary.get(k) for k in _SUMMARY_KEYS}


def run_transition(
    *,
    inventory: Any,
    config: Any,
    from_layout: str,
    arm1: str,
    workload1: Optional[str],
    arm2: str,
    workload2: str,
    run_id: str,
    pilot_policy: Optional[Dict[str, Any]] = None,
    repetition: int = 1,
    arm_timeout_s: float = 7200.0,
    drain_timeout_s: float = 600.0,
) -> Dict[str, Any]:
    """Execute one partial transition of a ``PARTIAL_TRANSITIONS`` pair.

    ``config`` is the TARGET layout's DeploymentConfig: kept engines must match its
    settings exactly, and the joiners are deployed with ``config.settings_for(name)``
    (e.g. weight_snapshot=read + fast_graph for a snapshot-backed join).

    Fails closed and leaves the fabric as it is on any error after the drain
    begins; the report (also written under the run's artifact directory) says
    exactly which step was reached.
    """
    import ray

    from .deployment import _find_engine_actor_named, deploy, drain_deployment
    from .evidence import emit_event
    from .experiment import ARMS, ARTIFACT_ROOT, BASE_POLICY, run_arm
    from .gateway.lifecycle import find_gateway
    from .layout import resolve_layout
    from .registry import descriptor_for_layout, find_registry

    to_layout = config.layout.layout_id
    for arm, layout in ((arm1, from_layout), (arm2, to_layout)):
        if arm not in ARMS or ARMS[arm]["layout"] != layout:
            raise ValueError(f"arm {arm} does not run on layout {layout}")
    gateway = find_gateway()
    registry = find_registry()
    if gateway is None or registry is None:
        raise RuntimeError("a transition needs the running gateway and registry")
    status = ray.get(gateway.status.remote(), timeout=120)
    source = ray.get(registry.get_layout.remote(from_layout), timeout=60)
    if (status["layout_id"] != from_layout or status.get("maintenance") or status.get("joining")
            or status.get("capturing") or not status["discovery"].get("ready") or not source):
        raise RuntimeError("gateway is not idle on the complete source layout")
    from_instances = {i["instance_id"]: i["tp_size"] for i in source["instances"]}
    if set(status["instances"]) != set(from_instances):
        raise RuntimeError("gateway does not bind every source instance")
    resolved = resolve_layout(inventory, config.layout)
    target = resolved.to_dict()
    digest = target["layout_digest"]
    plan = plan_transition(from_layout, to_layout, from_instances,
                           {i.instance_id: i.tp_size for i in resolved.instances},
                           from_placement={i["instance_id"]: i for i in source["instances"]},
                           to_placement={i["instance_id"]: i for i in target["instances"]})
    by_id = {i.instance_id: i for i in resolved.instances}
    rows = {r["instance_id"]: r for r in ray.get(registry.snapshot.remote(from_layout), timeout=60)}

    report: Dict[str, Any] = {"plan": dataclasses.asdict(plan), "run_id": run_id, "steps": [],
                              "timestamps": {}, "completed": False}
    stamps = report["timestamps"]

    def mark(step: str) -> None:
        stamps[step] = time.time()
        report["steps"].append(step)
        emit_event("TRANSITION_STEP", step=step)

    out_dir = ARTIFACT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"transition_{from_layout}_to_{to_layout}.json"
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omp-transition-deploy")
    try:
        # ---- prepare: nothing destructive until every kept engine is proven adoptable
        actors = {n: _find_engine_actor_named(rows[n]["engine_actor_name"]) for n in plan.kept}
        if any(a is None for a in actors.values()):
            raise RuntimeError("a kept engine actor is missing")
        for name, actor in actors.items():
            instance = by_id[name].to_dict()
            verdict = ray.get(actor.matches.remote(instance, config.settings_for(name)), timeout=60)
            if not verdict["matches"]:
                raise RuntimeError(f"kept engine {name} conflicts with the target config: "
                                   f"{verdict['differences']}")
            ray.get(actor.adopt_identity.remote(to_layout, digest, instance, dry_run=True), timeout=60)
        ray.get(registry.put_inventory.remote(inventory.to_dict()), timeout=60)
        ray.get(registry.put_layout.remote(target), timeout=60)
        mark("prepared")

        # ---- phase 1: workload on the source layout (optional)
        if workload1 is not None:
            mark("phase1_begin")
            report["phase1"] = _digest(run_arm(
                arm=arm1, workload=workload1, repetition=repetition, run_id=run_id,
                timeout_s=arm_timeout_s, pilot_policy=dict(pilot_policy or {})))
            mark("phase1_end")

        # ---- drain: released engines only (the gateway keeps serving the kept ones)
        mark("drain_begin")
        report["drain"] = drain_deployment(from_layout, list(plan.release), timeout_s=drain_timeout_s)
        mark("drain_end")

        # ---- adopt: kept engines move to the target layout in place
        for name, actor in actors.items():
            identity = ray.get(actor.adopt_identity.remote(to_layout, digest, by_id[name].to_dict()),
                               timeout=60)
            row = ray.get(registry.get.remote(name), timeout=60)
            if identity["generation"] != row["generation"]:
                raise RuntimeError(f"{name}: actor and registry generations disagree")
            accepted = ray.get(registry.register.remote(
                descriptor_for_layout(row, to_layout, digest)), timeout=60)
            if not accepted.get("accepted") or accepted["generation"] != row["generation"]:
                raise RuntimeError(f"registry refused the adopted generation of {name}: {accepted}")
        mark("adopted")

        # ---- retarget: joiners are fenced out of discovery BEFORE they can be registered
        report["retarget"] = ray.get(gateway.retarget_layout.remote(to_layout, list(plan.join)),
                                     timeout=600)
        mark("retargeted")

        # ---- deploy in a thread; phase 2 serves on the kept engines meanwhile
        def deploy_joiners() -> Dict[str, Any]:
            result = deploy(inventory, config, instance_ids=list(plan.join))
            stamps["deploy_ready"] = time.time()
            return result

        stamps["deploy_begin"] = time.time()
        deployed = pool.submit(deploy_joiners)
        join_box: Dict[str, Any] = {}

        # The arm's declared non-role policy fields: run_arm zeroes the large share and migration
        # while the large instance does not exist yet, so the join must restore them.
        policy_after = {**BASE_POLICY, **(pilot_policy or {}),
                        "migration_enabled": ARMS[arm2]["migration_enabled"],
                        "migration_mechanism": ARMS[arm2]["migration_mechanism"]}

        def join_when_ready(cancel: Any) -> Dict[str, Any]:
            while not deployed.done():  # a deploy failure surfaces as the hook's error
                if cancel.wait(1.0):
                    raise RuntimeError("phase 2 aborted before the new engines were ready")
            deployed.result()
            if cancel.is_set():
                raise RuntimeError("phase 2 aborted before the join")
            join_box["join_called_at"] = time.time()
            receipt = ray.get(gateway.join_instances.remote(list(plan.join), policy_after),
                              timeout=600)
            join_box["join_returned_at"] = time.time()
            join_box["receipt"] = receipt
            return dict(join_box)

        mark("phase2_begin")
        summary = run_arm(
            arm=arm2, workload=workload2, repetition=repetition, run_id=run_id,
            timeout_s=arm_timeout_s, pilot_policy=dict(pilot_policy or {}), during_run=join_when_ready)
        mark("phase2_end")
        report["phase2"] = _digest(summary)
        during = summary.get("during_run") or {}
        report["during_run"] = during
        if during.get("error"):
            raise RuntimeError(f"new engines did not join: {during['error']}")
        result = during["result"]
        label = f"{workload2}_{arm2}_r{repetition}"
        with gzip.open(pathlib.Path(summary["artifact_dir"]) / f"{label}_records.json.gz", "rt") as fh:
            records = json.load(fh)
        report["join_partition"] = partition_records(
            records, result["join_called_at"], result["join_returned_at"])
        report["timeline"] = {
            "phase1_drained_at": stamps.get("phase1_end"),
            "drain_s": round(stamps["drain_end"] - stamps["drain_begin"], 2),
            "deploy_s": round(stamps["deploy_ready"] - stamps["deploy_begin"], 2),
            "join_s": round(result["join_returned_at"] - result["join_called_at"], 2),
            "phase2_first_dispatch": summary["started_wall"],
            "phase2_last_finish": summary["finished_wall"],
            "join_after_phase2_start_s": round(result["join_returned_at"] - summary["started_wall"], 2),
        }
        report["completed"] = True
        return report
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
        raise
    finally:
        pool.shutdown(wait=False)
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        emit_event("TRANSITION_REPORT", path=str(report_path), completed=report["completed"])
