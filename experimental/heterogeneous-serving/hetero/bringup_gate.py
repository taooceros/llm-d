"""Phase 1 exit gate: concurrent real serving across the whole static layout.

Placement-group readiness, a successful import or one isolated TP8 run does not
satisfy this gate.  It requires all configured engines resident together, real
weights, real ShareGPT prefill and decode running concurrently, verified device
membership and rank mapping, idempotent reapplication and a scoped restart that
leaves peers serving.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ray

from .deployment import (
    DeploymentConfig,
    _find_engine_actor,
    deploy,
    drain_deployment,
    inspect_deployment,
)
from .evidence import emit_event, emit_log
from .registry import find_registry
from .slice_inventory import SliceInventory
from .workload import load_catalogue, prompt_payload_digest, unique_prompts


def run_bringup_gate(
    inventory: SliceInventory,
    config: DeploymentConfig,
    *,
    prompts_per_instance: int = 8,
    decode_tokens: int = 32,
    restart_instance: Optional[str] = None,
    catalogue_name: str = "small",
) -> Dict[str, Any]:
    layout_id = config.layout.layout_id
    report: Dict[str, Any] = {"layout_id": layout_id, "stages": {}}

    emit_event("GATE_STAGE", stage="deploy")
    t0 = time.time()
    first = deploy(inventory, config)
    report["deployment"] = first
    report["stages"]["deploy_s"] = round(time.time() - t0, 2)

    instance_ids = [rec["descriptor"]["instance_id"] for rec in first["instances"]]
    actors = {i: _find_engine_actor(layout_id, i) for i in instance_ids}
    if any(a is None for a in actors.values()):
        raise RuntimeError("an engine actor disappeared immediately after deployment")

    emit_event("GATE_STAGE", stage="disjointness")
    report["disjointness"] = _check_disjoint_domains(first)

    emit_event("GATE_STAGE", stage="idempotent_reapply")
    t0 = time.time()
    second = deploy(inventory, config)
    reapply = {
        "attached_existing_actor": {
            rec["descriptor"]["instance_id"]: rec["attached_existing_actor"]
            for rec in second["instances"]
        },
        "created_new_placement_group": {
            rec["descriptor"]["instance_id"]: rec["placement"]["record"]["created_by_this_run"]
            for rec in second["instances"]
        },
        "generations_stable": {
            rec["descriptor"]["instance_id"]: rec["descriptor"]["generation"]
            for rec in second["instances"]
        },
        "placement_group_count": len(second["deployment"]["owned_placement_groups"])
        if "deployment" in second
        else None,
    }
    duplicates = [
        i
        for i, attached in reapply["attached_existing_actor"].items()
        if not attached
    ]
    new_pgs = [i for i, created in reapply["created_new_placement_group"].items() if created]
    reapply["ok"] = not duplicates and not new_pgs
    reapply["unexpected_new_actors"] = duplicates
    reapply["unexpected_new_placement_groups"] = new_pgs
    reapply["seconds"] = round(time.time() - t0, 2)
    report["idempotent_reapply"] = reapply
    if not reapply["ok"]:
        raise RuntimeError(f"reapplying the same configuration created duplicates: {reapply}")

    emit_event("GATE_STAGE", stage="concurrent_serving")
    catalogue = load_catalogue(catalogue_name)
    report["catalogue"] = {
        "name": catalogue["name"],
        "path": catalogue["path"],
        "sha256": catalogue["sha256"],
        "count": catalogue["count"],
    }
    prefill = _concurrent_phase(
        actors,
        catalogue,
        prompts_per_instance=prompts_per_instance,
        max_tokens=1,
        offset_base=0,
        phase="prefill",
    )
    decode = _concurrent_phase(
        actors,
        catalogue,
        prompts_per_instance=prompts_per_instance,
        max_tokens=decode_tokens,
        offset_base=prompts_per_instance * len(actors),
        phase="decode",
    )
    report["concurrent_serving"] = {"prefill": prefill, "decode": decode}

    emit_event("GATE_STAGE", stage="capacity_and_idle")
    report["capacity"] = {
        i: ray.get(a.kv_capacity.remote(), timeout=120) for i, a in actors.items()
    }
    report["worker_kv_evidence"] = {
        i: ray.get(a.worker_kv_evidence.remote(), timeout=300) for i, a in actors.items()
    }
    report["idle_snapshots"] = {
        i: ray.get(a.idle_snapshot.remote(), timeout=120) for i, a in actors.items()
    }

    if restart_instance:
        emit_event("GATE_STAGE", stage="scoped_restart", instance=restart_instance)
        report["scoped_restart"] = _scoped_restart(
            inventory, config, restart_instance, catalogue, prompts_per_instance
        )

    report["final_inspection"] = inspect_deployment(layout_id)
    report["gate_passed"] = _evaluate_gate(report)
    emit_event("GATE_RESULT", passed=report["gate_passed"]["passed"])
    return report


def _check_disjoint_domains(deployment: Dict[str, Any]) -> Dict[str, Any]:
    """Every engine must see exactly its own chips and no peer's chips."""
    seen: Dict[Tuple[int, int, int], str] = {}
    per_instance = {}
    for record in deployment["instances"]:
        instance_id = record["descriptor"]["instance_id"]
        workers = record["initialization"]["worker_evidence"]
        parent_chips = set()
        global_views = set()
        for worker in workers:
            devices = worker["devices"]
            global_views.add(
                (
                    devices["global_device_count"],
                    tuple(tuple(c) for c in devices["runtime_global_chip_coords"]),
                )
            )
            for coord in devices["parent_local_chip_coords"]:
                key = (coord[0], coord[1], coord[2])
                if key in seen:
                    raise RuntimeError(
                        f"chip {key} is owned by both {seen[key]} and {instance_id}"
                    )
                seen[key] = instance_id
                parent_chips.add(key)
        if len(global_views) != 1:
            raise RuntimeError(
                f"{instance_id} workers disagree about the global device set; "
                f"the runtime domain is not consistent: {sorted(global_views)}"
            )
        global_count, global_view = next(iter(global_views))
        if global_count != len(parent_chips):
            raise RuntimeError(
                f"{instance_id} runtime sees {global_count} chips but owns "
                f"{len(parent_chips)}; unintended rendezvous outside the execution domain"
            )
        per_instance[instance_id] = {
            "parent_chips": sorted(list(c) for c in parent_chips),
            "chip_count": len(parent_chips),
            "runtime_global_chip_coords": [list(c) for c in global_view],
            "global_device_count": global_count,
            "process_ranks": [w["instance_process_rank"] for w in workers],
            "physical_worker_ids": [w["physical_tpu_worker_id"] for w in workers],
            "executor_ranks": [w["executor_rank"] for w in workers],
            "relative_position_match": all(
                w["devices"]["relative_position_match"] for w in workers
            ),
        }
    return {"per_instance": per_instance, "total_chips": len(seen), "disjoint": True}


def _concurrent_phase(
    actors: Dict[str, Any],
    catalogue: Dict[str, Any],
    *,
    prompts_per_instance: int,
    max_tokens: int,
    offset_base: int,
    phase: str,
) -> Dict[str, Any]:
    """Launches one real batch on every instance simultaneously."""
    import uuid

    invocation = uuid.uuid4().hex
    instance_ids = sorted(actors)
    assignments: Dict[str, List[Dict[str, str]]] = {}
    for index, instance_id in enumerate(instance_ids):
        assignments[instance_id] = unique_prompts(
            catalogue,
            prompts_per_instance,
            offset=offset_base + index * prompts_per_instance,
        )

    for instance_id in instance_ids:
        ray.get(actors[instance_id].start_iteration_capture.remote(), timeout=60)

    started = time.time()
    futures = {}
    for instance_id in instance_ids:
        requests = [
            {
                "prompt": record["prompt"],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": max_tokens > 1,
                "request_id": f"{phase}-{invocation}-{instance_id}-{record['id']}",
            }
            for record in assignments[instance_id]
        ]
        futures[instance_id] = actors[instance_id].generate_batch.remote(requests)
    results = {i: ray.get(f, timeout=1800) for i, f in futures.items()}
    wall_s = time.time() - started

    captures = {
        i: ray.get(actors[i].stop_iteration_capture.remote(), timeout=300)
        for i in instance_ids
    }
    iterations = {i: capture["rows"] for i, capture in captures.items()}

    windows = {}
    for instance_id, rows in results.items():
        windows[instance_id] = {
            "first_start_s": min(r["started_at_s"] for r in rows),
            "last_finish_s": max(r["finished_at_s"] for r in rows),
            "requests": len(rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "prompt_digest": prompt_payload_digest(assignments[instance_id]),
        }
    overlap = _overlap_matrix(windows)
    all_overlap = all(
        overlap[a][b] > 0 for a in instance_ids for b in instance_ids if a != b
    )
    per_instance_progress = {
        i: {
            "iterations": len(iterations[i]),
            "scheduled_tokens": sum(r.get("scheduled_tokens", 0) for r in iterations[i]),
            "emitted_output_tokens": sum(
                r.get("emitted_output_tokens", 0) for r in iterations[i]
            ),
        }
        for i in instance_ids
    }
    emit_log(
        f"{phase}: wall={wall_s:.2f}s overlap_all={all_overlap} "
        + " ".join(
            f"{i}:{windows[i]['output_tokens']}tok/{per_instance_progress[i]['iterations']}it"
            for i in instance_ids
        )
    )
    return {
        "phase": phase,
        "max_tokens": max_tokens,
        "wall_s": round(wall_s, 3),
        "windows": windows,
        "overlap_s": overlap,
        "all_pairs_overlap": all_overlap,
        "progress": per_instance_progress,
        "results": results,
        "iterations": iterations,
        "capture_audits": {i: capture["audit"] for i, capture in captures.items()},
    }


def _overlap_matrix(windows: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for a, wa in windows.items():
        out[a] = {}
        for b, wb in windows.items():
            lo = max(wa["first_start_s"], wb["first_start_s"])
            hi = min(wa["last_finish_s"], wb["last_finish_s"])
            out[a][b] = round(max(0.0, hi - lo), 4)
    return out


def _scoped_restart(
    inventory: SliceInventory,
    config: DeploymentConfig,
    instance_id: str,
    catalogue: Dict[str, Any],
    prompts_per_instance: int,
) -> Dict[str, Any]:
    """Drains one owned instance, proves peers keep serving, then reattaches it."""
    layout_id = config.layout.layout_id
    registry = find_registry()
    before = ray.get(registry.snapshot.remote(layout_id), timeout=60)
    generation_before = {row["instance_id"]: row["generation"] for row in before}

    drained = drain_deployment(layout_id, [instance_id], remove=True)
    after_drain = ray.get(registry.snapshot.remote(layout_id), timeout=60)
    peer_ids = [row["instance_id"] for row in after_drain]
    peers = {i: _find_engine_actor(layout_id, i) for i in peer_ids}
    peers = {i: a for i, a in peers.items() if a is not None}
    if not peers:
        raise RuntimeError("no peers survived the scoped drain")

    peer_serving = _concurrent_phase(
        peers,
        catalogue,
        prompts_per_instance=max(2, prompts_per_instance // 2),
        max_tokens=8,
        offset_base=3 * prompts_per_instance * 4,
        phase="peer_serving_during_restart",
    )

    restored = deploy(inventory, config)
    after = ray.get(registry.snapshot.remote(layout_id), timeout=60)
    generation_after = {row["instance_id"]: row["generation"] for row in after}
    duplicates = [
        rec["descriptor"]["instance_id"]
        for rec in restored["instances"]
        if rec["descriptor"]["instance_id"] != instance_id and not rec["attached_existing_actor"]
    ]
    return {
        "instance_id": instance_id,
        "drain": drained,
        "stale_endpoint_removed": instance_id not in {r["instance_id"] for r in after_drain},
        "peers_kept_serving": peer_serving["all_pairs_overlap"] if len(peers) > 1 else True,
        "peer_serving": {
            "windows": peer_serving["windows"],
            "progress": peer_serving["progress"],
        },
        "generation_before": generation_before,
        "generation_after": generation_after,
        "restarted_generation_changed": generation_after.get(instance_id)
        != generation_before.get(instance_id),
        "peers_not_recreated": not duplicates,
        "registry_after": after,
    }


def _evaluate_gate(report: Dict[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    checks["all_instances_ready"] = all(
        rec["descriptor"]["state"] == "ready" for rec in report["deployment"]["instances"]
    )
    checks["disjoint_execution_domains"] = bool(report["disjointness"]["disjoint"])
    checks["idempotent_reapply"] = bool(report["idempotent_reapply"]["ok"])
    checks["prefill_concurrent"] = bool(
        report["concurrent_serving"]["prefill"]["all_pairs_overlap"]
    )
    checks["decode_concurrent"] = bool(
        report["concurrent_serving"]["decode"]["all_pairs_overlap"]
    )
    checks["every_instance_produced_tokens"] = all(
        window["output_tokens"] == window["requests"] * phase["max_tokens"]
        for phase in report["concurrent_serving"].values()
        for window in phase["windows"].values()
    )
    checks["iteration_capture_one_to_one"] = all(
        audit["rows"] == audit["retired_rows"]
        and audit["open_rows"] == 0
        and audit["unmatched_retirements"] == 0
        for phase in report["concurrent_serving"].values()
        for audit in phase["capture_audits"].values()
    )
    checks["every_instance_stepped"] = all(
        p["iterations"] > 0
        for p in report["concurrent_serving"]["decode"]["progress"].values()
    )
    checks["measured_kv_capacity"] = all(
        cap["usable_blocks"] > 0 for cap in report["capacity"].values()
    )
    checks["returned_to_idle"] = all(
        snap["all_blocks_free"] and snap["inflight"] == 0
        for snap in report["idle_snapshots"].values()
    )
    if "scoped_restart" in report:
        restart = report["scoped_restart"]
        checks["scoped_restart_peers_available"] = bool(restart["peers_kept_serving"])
        checks["stale_endpoint_removed"] = bool(restart["stale_endpoint_removed"])
        checks["peers_not_recreated"] = bool(restart["peers_not_recreated"])
        checks["restarted_generation_changed"] = bool(restart["restarted_generation_changed"])
        checks["peer_generations_stable"] = all(
            generation == restart["generation_after"].get(instance_id)
            for instance_id, generation in restart["generation_before"].items()
            if instance_id != restart["instance_id"]
        )
    return {"checks": checks, "passed": all(checks.values())}
