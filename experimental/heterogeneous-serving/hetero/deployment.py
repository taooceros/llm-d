"""Configuration-driven deploy / reconcile / inspect / drain for the fabric.

One desired configuration (layout + engine settings) is reconciled against the
existing Ray cluster.  Reapplying the same configuration attaches to the same
placement groups, actors and registrations instead of creating duplicates; a
conflicting live configuration is an actionable refusal, never a silent
reconfiguration of running work.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .engine_actor import EngineSettings, HeteroEngine
from .evidence import emit_event, emit_log
from .layout import LayoutRequest, ResolvedInstance, ResolvedLayout, resolve_layout
from .placement import (
    ensure_placement_group,
    list_owned_placement_groups,
    pg_name,
    release_placement_group,
    verify_bundle_placement,
)
from .registry import REGISTRY_NAMESPACE, get_or_create_registry
from .slice_inventory import SliceInventory
from .native_backend import BACKEND_REVISION, prepare_backend

ENGINE_ACTOR_PREFIX = "omp-hetero-engine"
DEFAULT_CPU_PER_HOST = 8.0
DEFAULT_TPU_PER_HOST = 4.0
SERVING_PORT = 8000


def engine_actor_name(layout_id: str, instance_id: str) -> str:
    return f"{ENGINE_ACTOR_PREFIX}--{layout_id}--{instance_id}"


@dataclasses.dataclass
class DeploymentConfig:
    """The single desired configuration for one static arm."""

    layout: LayoutRequest
    engine_settings: Dict[str, Any] = dataclasses.field(default_factory=dict)
    cpu_per_host: float = DEFAULT_CPU_PER_HOST
    tpu_per_host: float = DEFAULT_TPU_PER_HOST
    pg_ready_timeout_s: float = 600.0
    engine_init_timeout_s: float = 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layout": self.layout.to_dict(),
            "engine_settings": EngineSettings.from_dict(self.engine_settings).to_dict(),
            "cpu_per_host": self.cpu_per_host,
            "tpu_per_host": self.tpu_per_host,
            "pg_ready_timeout_s": self.pg_ready_timeout_s,
            "engine_init_timeout_s": self.engine_init_timeout_s,
        }


def _find_engine_actor(layout_id: str, instance_id: str) -> Optional[Any]:
    try:
        return ray.get_actor(engine_actor_name(layout_id, instance_id), namespace=REGISTRY_NAMESPACE)
    except ValueError:
        return None


def plan_deployment(
    inventory: SliceInventory, config: DeploymentConfig
) -> Tuple[ResolvedLayout, Dict[str, Any]]:
    """Non-mutating preflight: what would be created, attached or refused."""
    resolved = resolve_layout(inventory, config.layout)
    body = resolved.to_dict()
    owned = {row["name"]: row for row in list_owned_placement_groups(config.layout.layout_id)}
    actions = []
    for instance in resolved.instances:
        name = pg_name(config.layout.layout_id, instance.instance_id)
        actor = _find_engine_actor(config.layout.layout_id, instance.instance_id)
        actions.append(
            {
                "instance_id": instance.instance_id,
                "placement_group": name,
                "placement_group_action": "attach" if name in owned else "create",
                "placement_group_state": owned.get(name, {}).get("state"),
                "engine_actor": engine_actor_name(config.layout.layout_id, instance.instance_id),
                "engine_actor_action": "attach" if actor is not None else "create",
                "hosts": list(instance.host_ips),
                "coordinator_ip": instance.coordinator_ip,
                "chips": instance.tp_size,
                "required_resources": {
                    "TPU": config.tpu_per_host * len(instance.hosts),
                    "CPU": config.cpu_per_host * len(instance.hosts),
                },
            }
        )
    preflight = {
        "layout": body,
        "desired_configuration": config.to_dict(),
        "actions": actions,
        "owned_placement_groups": list(owned.values()),
        "total_chips_allocated": resolved.total_chips_allocated,
        "total_chips_in_slice": resolved.total_chips_in_slice,
        "native_backend": prepare_backend(
            [ip for instance in resolved.instances for ip in instance.host_ips], apply=False
        ),
    }
    return resolved, preflight


def deploy(
    inventory: SliceInventory,
    config: DeploymentConfig,
    *,
    smoke_tokens: int = 0,
    instance_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Reconcile all or an explicit maintenance subset of the same static layout."""
    resolved = resolve_layout(inventory, config.layout)
    wanted = set(instance_ids) if instance_ids is not None else {i.instance_id for i in resolved.instances}
    instances = [instance for instance in resolved.instances if instance.instance_id in wanted]
    if not wanted or wanted != {instance.instance_id for instance in instances}:
        raise ValueError("deployment subset is empty or contains an unknown instance")
    layout_body = resolved.to_dict()
    layout_id = config.layout.layout_id
    digest = layout_body["layout_digest"]
    settings = EngineSettings.from_dict(config.engine_settings).to_dict()

    registry = get_or_create_registry()
    if instance_ids is not None:
        existing = ray.get(registry.get_layout.remote(layout_id), timeout=60)
        if not existing or existing["layout_digest"] != digest:
            raise RuntimeError("partial restart cannot change the registered static layout")
        from .gateway.lifecycle import find_gateway

        gateway = find_gateway()
        if gateway is not None:
            status = ray.get(gateway.status.remote(), timeout=60)
            if (status["layout_id"] != layout_id
                    or set(status.get("maintenance", {}).get("instances", {})) != wanted):
                raise RuntimeError("partial restart requires matching declared gateway maintenance")
    existing_actors = {}
    for instance in resolved.instances:
        actor = _find_engine_actor(layout_id, instance.instance_id)
        if actor is None:
            continue
        try:
            verdict = ray.get(actor.matches.remote(instance.to_dict(), settings), timeout=60)
        except Exception as exc:
            raise RuntimeError(
                f"live engine actor {engine_actor_name(layout_id, instance.instance_id)} "
                f"did not answer a configuration check ({type(exc).__name__}: "
                f"{str(exc)[:160]}). Drain or release it before redeploying."
            ) from exc
        if not verdict["matches"]:
            raise RuntimeError(
                f"live engine actor for {instance.instance_id} conflicts with the "
                f"requested configuration: {verdict["differences"]}; drain it first"
            )
        existing_actors[instance.instance_id] = actor

    backend_started = time.time()
    native_backend = prepare_backend(
        [ip for instance in instances for ip in instance.host_ips], apply=True
    )
    backend_elapsed = round(time.time() - backend_started, 2)
    emit_event("NATIVE_BACKEND_READY", revision=BACKEND_REVISION, seconds=backend_elapsed)
    ray.get(registry.put_inventory.remote(inventory.to_dict()))
    ray.get(registry.put_layout.remote(layout_body))

    created_pgs: List[str] = []
    created_actors: List[str] = []
    records: List[Dict[str, Any]] = []
    stage_clocks: Dict[str, float] = {"backend_s": backend_elapsed}

    try:
        t0 = time.time()
        handles: Dict[str, Any] = {}
        placement_records: Dict[str, Any] = {}
        for instance in instances:
            pg, record = ensure_placement_group(
                instance,
                layout_id,
                cpu_per_host=config.cpu_per_host,
                tpu_per_host=config.tpu_per_host,
                ready_timeout_s=config.pg_ready_timeout_s,
                created_sink=created_pgs,
            )
            placement_records[instance.instance_id] = {
                "record": record.to_dict(),
                "bundles": verify_bundle_placement(pg, instance),
            }
            handles[instance.instance_id] = pg
        stage_clocks["placement_s"] = round(time.time() - t0, 2)
        emit_event("PLACEMENT_READY", layout_id=layout_id, seconds=stage_clocks["placement_s"])

        t0 = time.time()
        actors: Dict[str, Any] = {}
        attached: Dict[str, bool] = {}
        for instance in instances:
            actor = existing_actors.get(instance.instance_id)
            if actor is not None:
                attached[instance.instance_id] = True
            else:
                actor = HeteroEngine.options(
                    name=engine_actor_name(layout_id, instance.instance_id),
                    namespace=REGISTRY_NAMESPACE,
                    lifetime="detached",
                    num_cpus=2,
                    max_concurrency=1024,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=handles[instance.instance_id],
                        placement_group_bundle_index=0,
                        placement_group_capture_child_tasks=True,
                    ),
                ).remote(
                    instance=instance.to_dict(),
                    layout_id=layout_id,
                    layout_digest=digest,
                    engine_settings=settings,
                )
                created_actors.append(engine_actor_name(layout_id, instance.instance_id))
                attached[instance.instance_id] = False
            actors[instance.instance_id] = actor

        env_reports = ray.get(
            [actors[i.instance_id].prepare_environment.remote() for i in instances],
            timeout=300,
        )
        stage_clocks["actor_s"] = round(time.time() - t0, 2)
        emit_event("ENGINE_ACTORS_READY", layout_id=layout_id, seconds=stage_clocks["actor_s"])

        t0 = time.time()
        futures = {
            instance.instance_id: actors[instance.instance_id].initialize.remote(
                pg_name=pg_name(layout_id, instance.instance_id),
                timeout_s=config.engine_init_timeout_s,
            )
            for instance in instances
        }
        init_reports = {
            instance_id: ray.get(future, timeout=config.engine_init_timeout_s)
            for instance_id, future in futures.items()
        }
        stage_clocks["engine_init_s"] = round(time.time() - t0, 2)
        emit_event(
            "ENGINES_INITIALIZED", layout_id=layout_id, seconds=stage_clocks["engine_init_s"]
        )

        t0 = time.time()
        serving_reports = {
            instance.instance_id: ray.get(
                actors[instance.instance_id].start_serving.remote(port=SERVING_PORT),
                timeout=300,
            )
            for instance in instances
        }
        stage_clocks["serving_endpoint_s"] = round(time.time() - t0, 2)
        emit_event(
            "SERVING_ENDPOINTS_READY",
            layout_id=layout_id,
            endpoints={k: f"{v['endpoint_host']}:{v['endpoint_port']}" for k, v in serving_reports.items()},
        )

        for instance in instances:
            report = init_reports[instance.instance_id]
            actor = actors[instance.instance_id]
            identity = ray.get(actor.identity.remote(), timeout=60)
            capacity = report["kv_capacity"]
            resolved_cfg = report["resolved_engine_config"]
            descriptor = {
                "instance_id": instance.instance_id,
                "layout_id": layout_id,
                "generation": identity["generation"],
                "config_digest": digest,
                "ray_namespace": REGISTRY_NAMESPACE,
                "engine_actor_name": engine_actor_name(layout_id, instance.instance_id),
                "placement_group_name": pg_name(layout_id, instance.instance_id),
                "placement_group_id": placement_records[instance.instance_id]["record"]["pg_id"],
                "host_chip_rank_map": [h.to_dict() for h in instance.hosts],
                "endpoint_host": instance.coordinator_ip,
                "endpoint_port": SERVING_PORT,
                "model": resolved_cfg["model"],
                "model_build": resolved_cfg["model_build"],
                "tp_size": instance.tp_size,
                "max_model_len": resolved_cfg["max_model_len"],
                "kv_block_size": capacity["block_size"],
                "kv_usable_blocks": capacity["usable_blocks"],
                "kv_usable_tokens": capacity["usable_tokens"],
                "max_num_seqs": resolved_cfg["max_num_seqs"],
                "continuation_modes": ["recompute"],
                "transport_modes": ["gateway_replay"],
                "state": "ready",
            }
            accepted = ray.get(registry.register.remote(descriptor), timeout=60)
            if not accepted.get("accepted"):
                raise RuntimeError(f"registry refused the initialized generation: {accepted}")
            records.append(
                {
                    "descriptor": descriptor,
                    "registry": accepted,
                    "attached_existing_actor": attached[instance.instance_id],
                    "placement": placement_records[instance.instance_id],
                    "initialization": report,
                    "serving_endpoint": serving_reports[instance.instance_id],
                }
            )
            emit_event(
                "INSTANCE_READY",
                instance_id=instance.instance_id,
                tp=instance.tp_size,
                hosts=list(instance.host_ips),
                usable_kv_blocks=capacity["usable_blocks"],
                usable_kv_tokens=capacity["usable_tokens"],
            )
    except Exception as exc:
        emit_event("DEPLOY_FAILED", layout_id=layout_id, error=str(exc)[:400])
        _rollback(created_actors, created_pgs)
        raise

    return {
        "layout_id": layout_id,
        "layout_digest": digest,
        "desired_configuration": config.to_dict(),
        "resolved_layout": layout_body,
        "native_backend": native_backend,
        "stage_clocks": stage_clocks,
        "driver_environment": env_reports,
        "instances": records,
        "registry_snapshot": ray.get(registry.snapshot.remote(layout_id), timeout=60),
    }


def _rollback(created_actors: Sequence[str], created_pgs: Sequence[str]) -> None:
    """Removes only the new resources this failed attempt created."""
    for name in created_actors:
        try:
            actor = ray.get_actor(name, namespace=REGISTRY_NAMESPACE)
            ray.kill(actor, no_restart=True)
            emit_log(f"rolled back engine actor {name}")
        except Exception:  # pragma: no cover - best effort
            pass
    for name in created_pgs:
        try:
            release_placement_group(name)
            emit_log(f"rolled back placement group {name}")
        except Exception:  # pragma: no cover - best effort
            pass


def inspect_deployment(layout_id: Optional[str] = None) -> Dict[str, Any]:
    """Desired versus observed instance state without mutating anything."""
    from .registry import find_registry

    registry = find_registry()
    snapshot = ray.get(registry.snapshot.remote(layout_id), timeout=60) if registry else []
    layout = (
        ray.get(registry.get_layout.remote(layout_id), timeout=60)
        if (registry and layout_id)
        else None
    )
    observed = []
    for row in snapshot:
        actor = _find_engine_actor(row["layout_id"], row["instance_id"])
        if actor is None:
            observed.append({**row, "actor_alive": False, "engine_status": None})
            continue
        try:
            status = ray.get(actor.status.remote(), timeout=60)
            alive = True
        except Exception as exc:
            status = {"error": str(exc)[:300]}
            alive = False
        observed.append({**row, "actor_alive": alive, "engine_status": status})
    return {
        "layout_id": layout_id,
        "registry": snapshot,
        "observed": observed,
        "resolved_layout": layout,
        "owned_placement_groups": list_owned_placement_groups(layout_id),
    }


def drain_deployment(
    layout_id: str,
    instance_ids: Optional[Sequence[str]] = None,
    *,
    remove: bool = True,
    timeout_s: float = 600.0,
) -> Dict[str, Any]:
    """Drain acknowledged native work before deleting only selected owned resources.

    A live gateway declares maintenance before the first mutation. Failure
    leaves the target excluded and its actor/placement group intact; it never
    converts a timeout or missing actor into permission to destroy native state.
    """
    from .registry import find_registry
    from .gateway.lifecycle import find_gateway

    registry = find_registry()
    rows = ray.get(registry.snapshot.remote(layout_id), timeout=60) if registry else []
    wanted = set(instance_ids) if instance_ids is not None else {row["instance_id"] for row in rows}
    targets = [row for row in rows if row["instance_id"] in wanted]
    if wanted != {row["instance_id"] for row in targets}:
        raise ValueError("drain target is absent from this layout's registry")
    gateway = find_gateway()
    maintenance = None
    if gateway is not None:
        status = ray.get(gateway.status.remote(), timeout=60)
        if status["layout_id"] != layout_id:
            gateway = None
        elif targets:
            maintenance = ray.get(gateway.begin_maintenance.remote(
                sorted(wanted), reason="explicit deployment drain"), timeout=300)
    results = []
    for row in targets:
        instance_id, generation = row["instance_id"], row["generation"]
        if (row["engine_actor_name"] != engine_actor_name(layout_id, instance_id)
                or row["placement_group_name"] != pg_name(layout_id, instance_id)
                or row["ray_namespace"] != REGISTRY_NAMESPACE):
            raise RuntimeError("refusing to drain resources not owned by this deployment")
        actor = _find_engine_actor(layout_id, instance_id)
        if actor is None:
            raise RuntimeError(f"cannot acknowledge native drain for missing actor {instance_id}")
        identity = ray.get(actor.identity.remote(), timeout=60)
        if (identity["generation"] != generation or identity["layout_id"] != layout_id
                or identity["instance_id"] != instance_id):
            raise RuntimeError("actor/registry identity changed before drain")
        entry: Dict[str, Any] = {"instance_id": instance_id, "generation": generation, "actor_found": True}
        if gateway is not None:
            entry.update(ray.get(gateway.drain_maintenance_instance.remote(
                instance_id, maintenance_id=maintenance["id"], timeout_s=timeout_s),
                timeout=2 * timeout_s + 120))
        else:
            entry["drain"] = ray.get(actor.drain.remote(timeout_s=timeout_s), timeout=timeout_s + 60)
            if not entry["drain"].get("drained") or entry["drain"].get("inflight") != 0:
                raise RuntimeError(f"native admission drain was not acknowledged: {entry}")
            entry["native_idle"] = ray.get(actor.await_idle.remote(timeout_s=timeout_s), timeout=timeout_s + 60)
            entry["idle_snapshot"] = ray.get(actor.idle_snapshot.remote(), timeout=120)
            if not entry["native_idle"].get("idle") or not entry["idle_snapshot"].get("all_blocks_free"):
                raise RuntimeError(f"native scheduler/KV is not idle: {entry}")
        if not ray.get(registry.set_state.remote(instance_id, "draining", generation=generation), timeout=60):
            raise RuntimeError("registry generation changed after native drain")
        if remove:
            entry["stop_serving"] = ray.get(actor.stop_serving.remote(), timeout=120)
            entry["shutdown"] = ray.get(actor.shutdown.remote(), timeout=600)
            if any(key.endswith("_error") for key in entry["shutdown"]):
                raise RuntimeError(f"native shutdown was not acknowledged: {entry['shutdown']}")
            ray.kill(actor, no_restart=True)
            entry["actor_removed"] = True
            entry["placement_group_removed"] = release_placement_group(pg_name(layout_id, instance_id))
            entry["withdrawn"] = ray.get(registry.withdraw.remote(instance_id, generation=generation), timeout=60)
            if not entry["withdrawn"]:
                raise RuntimeError("registry generation changed before withdrawal")
        results.append(entry)
        emit_event("INSTANCE_DRAINED", instance_id=instance_id, generation=generation, removed=remove)
    reconciliation = (ray.get(gateway.reconcile.remote(), timeout=300)
                      if gateway is not None and targets else None)
    return {
        "layout_id": layout_id, "drained": results, "maintenance": maintenance,
        "gateway_reconcile": reconciliation,
        "registry_snapshot": ray.get(registry.snapshot.remote(layout_id), timeout=60) if registry else [],
        "owned_placement_groups": list_owned_placement_groups(layout_id),
    }
