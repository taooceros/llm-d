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
from .placement import parse_pg_name
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
    # Per-instance overrides merged over engine_settings, e.g. a longer
    # max_model_len on the large instance of a mixed layout.
    instance_engine_settings: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    cpu_per_host: float = DEFAULT_CPU_PER_HOST
    tpu_per_host: float = DEFAULT_TPU_PER_HOST
    pg_ready_timeout_s: float = 600.0
    engine_init_timeout_s: float = 3600.0

    def settings_for(self, instance_id: str) -> Dict[str, Any]:
        merged = {**self.engine_settings, **self.instance_engine_settings.get(instance_id, {})}
        return EngineSettings.from_dict(merged).to_dict()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layout": self.layout.to_dict(),
            "engine_settings": EngineSettings.from_dict(self.engine_settings).to_dict(),
            "instance_engine_settings": self.instance_engine_settings,
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


def _find_engine_actor_named(name: str) -> Optional[Any]:
    """Actor by its physical name (a kept engine keeps its original layout's name)."""
    try:
        return ray.get_actor(name, namespace=REGISTRY_NAMESPACE)
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


def _select_instances(
    inventory: SliceInventory, config: DeploymentConfig, instance_ids: Optional[Sequence[str]]
) -> Tuple[ResolvedLayout, List[ResolvedInstance], Dict[str, Any], str, str]:
    resolved = resolve_layout(inventory, config.layout)
    wanted = set(instance_ids) if instance_ids is not None else {i.instance_id for i in resolved.instances}
    instances = [instance for instance in resolved.instances if instance.instance_id in wanted]
    if not wanted or wanted != {instance.instance_id for instance in instances}:
        raise ValueError("deployment subset is empty or contains an unknown instance")
    layout_body = resolved.to_dict()
    unknown = set(config.instance_engine_settings) - {i.instance_id for i in resolved.instances}
    if unknown:
        raise ValueError(f"instance engine settings name unknown instances: {sorted(unknown)}")
    return resolved, instances, layout_body, config.layout.layout_id, layout_body["layout_digest"]


def _check_partial_gateway(registry: Any, layout_id: str, digest: str, wanted: set) -> None:
    existing = ray.get(registry.get_layout.remote(layout_id), timeout=60)
    if not existing or existing["layout_digest"] != digest:
        raise RuntimeError("partial restart cannot change the registered static layout")
    from .gateway.lifecycle import find_gateway

    gateway = find_gateway()
    if gateway is not None:
        status = ray.get(gateway.status.remote(), timeout=60)
        joining = set(status.get("joining", ()))
        if (status["layout_id"] != layout_id
                or (set(status.get("maintenance", {}).get("instances", {})) != wanted
                    and joining != wanted)):
            raise RuntimeError(
                "partial deploy requires matching declared gateway maintenance or joining set")


def _existing_engine_actors(
    instances: Sequence[ResolvedInstance], layout_id: str, config: DeploymentConfig
) -> Dict[str, Any]:
    existing_actors = {}
    for instance in instances:
        actor = _find_engine_actor(layout_id, instance.instance_id)
        if actor is None:
            continue
        try:
            verdict = ray.get(actor.matches.remote(
                instance.to_dict(), config.settings_for(instance.instance_id)), timeout=60)
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
    return existing_actors


def _prepare_backend(instances: Sequence[ResolvedInstance]) -> Tuple[Dict[str, Any], float]:
    backend_started = time.time()
    native_backend = prepare_backend(
        [ip for instance in instances for ip in instance.host_ips], apply=True
    )
    backend_elapsed = round(time.time() - backend_started, 2)
    emit_event("NATIVE_BACKEND_READY", revision=BACKEND_REVISION, seconds=backend_elapsed)
    return native_backend, backend_elapsed


def _place(
    instances: Sequence[ResolvedInstance], layout_id: str, config: DeploymentConfig,
    tpu_per_host: float, created_pgs: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    t0 = time.time()
    handles: Dict[str, Any] = {}
    placement_records: Dict[str, Any] = {}
    for instance in instances:
        pg, record = ensure_placement_group(
            instance,
            layout_id,
            cpu_per_host=config.cpu_per_host,
            tpu_per_host=tpu_per_host,
            ready_timeout_s=config.pg_ready_timeout_s,
            created_sink=created_pgs,
        )
        placement_records[instance.instance_id] = {
            "record": record.to_dict(),
            "bundles": verify_bundle_placement(pg, instance),
        }
        handles[instance.instance_id] = pg
    placement_s = round(time.time() - t0, 2)
    emit_event("PLACEMENT_READY", layout_id=layout_id, seconds=placement_s)
    return handles, placement_records, placement_s


def _spawn_engine_actors(
    instances: Sequence[ResolvedInstance], layout_id: str, digest: str, config: DeploymentConfig,
    handles: Dict[str, Any], existing_actors: Dict[str, Any], created_actors: List[str],
) -> Tuple[Dict[str, Any], Dict[str, bool], List[Dict[str, Any]], float]:
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
                engine_settings=config.settings_for(instance.instance_id),
            )
            created_actors.append(engine_actor_name(layout_id, instance.instance_id))
            attached[instance.instance_id] = False
        actors[instance.instance_id] = actor

    env_reports = ray.get(
        [actors[i.instance_id].prepare_environment.remote() for i in instances],
        timeout=300,
    )
    actor_s = round(time.time() - t0, 2)
    emit_event("ENGINE_ACTORS_READY", layout_id=layout_id, seconds=actor_s)
    return actors, attached, env_reports, actor_s


def _initialize(
    actors: Dict[str, Any], instances: Sequence[ResolvedInstance], layout_id: str,
    config: DeploymentConfig, prespawn_gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    extra = {} if prespawn_gate is None else {"prespawn_gate": prespawn_gate}
    return {
        instance.instance_id: actors[instance.instance_id].initialize.remote(
            pg_name=pg_name(layout_id, instance.instance_id),
            timeout_s=config.engine_init_timeout_s,
            **extra,
        )
        for instance in instances
    }


def _serve_and_register(
    registry: Any, instances: Sequence[ResolvedInstance], layout_id: str, digest: str,
    actors: Dict[str, Any], attached: Dict[str, bool], placement_records: Dict[str, Any],
    init_reports: Dict[str, Any], stage_clocks: Dict[str, float],
) -> List[Dict[str, Any]]:
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

    records: List[Dict[str, Any]] = []
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
    return records


def _prespawn_flags(instances: Sequence[ResolvedInstance], config: DeploymentConfig) -> set:
    return {bool(config.settings_for(i.instance_id)["prespawn"]) for i in instances}


def deploy(
    inventory: SliceInventory,
    config: DeploymentConfig,
    *,
    smoke_tokens: int = 0,
    instance_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Reconcile all or an explicit maintenance subset of the same static layout."""
    resolved, instances, layout_body, layout_id, digest = _select_instances(inventory, config, instance_ids)
    if True in _prespawn_flags(instances, config):
        raise ValueError("engine setting prespawn=true is deployed only by prespawn_deployment/commit_prespawned")

    registry = get_or_create_registry()
    if instance_ids is not None:
        _check_partial_gateway(registry, layout_id, digest, {i.instance_id for i in instances})
    existing_actors = _existing_engine_actors(instances, layout_id, config)

    native_backend, backend_elapsed = _prepare_backend(instances)
    ray.get(registry.put_inventory.remote(inventory.to_dict()))
    ray.get(registry.put_layout.remote(layout_body))

    created_pgs: List[str] = []
    created_actors: List[str] = []
    stage_clocks: Dict[str, float] = {"backend_s": backend_elapsed}

    try:
        handles, placement_records, stage_clocks["placement_s"] = _place(
            instances, layout_id, config, config.tpu_per_host, created_pgs)
        actors, attached, env_reports, stage_clocks["actor_s"] = _spawn_engine_actors(
            instances, layout_id, digest, config, handles, existing_actors, created_actors)

        t0 = time.time()
        futures = _initialize(actors, instances, layout_id, config)
        init_reports = {
            instance_id: ray.get(future, timeout=config.engine_init_timeout_s)
            for instance_id, future in futures.items()
        }
        stage_clocks["engine_init_s"] = round(time.time() - t0, 2)
        emit_event(
            "ENGINES_INITIALIZED", layout_id=layout_id, seconds=stage_clocks["engine_init_s"]
        )

        records = _serve_and_register(registry, instances, layout_id, digest, actors, attached,
                                      placement_records, init_reports, stage_clocks)
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


# --------------------------------------------------------------------------- pre-spawn / commit

@dataclasses.dataclass
class PreparedDeployment:
    """A pre-spawned, uncommitted deployment: CPU-only groups, workers parked on a release gate.

    Nothing of it is registered with the registry or a gateway before ``commit_prespawned``.
    """

    inventory: SliceInventory
    config: DeploymentConfig
    instance_ids: Optional[List[str]]
    instances: List[ResolvedInstance]
    layout_id: str
    digest: str
    layout_body: Dict[str, Any]
    gate_name: str
    gate: Any = None
    native_backend: Dict[str, Any] = dataclasses.field(default_factory=dict)
    stage_clocks: Dict[str, float] = dataclasses.field(default_factory=dict)
    env_reports: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    actors: Dict[str, Any] = dataclasses.field(default_factory=dict)
    placement_records: Dict[str, Any] = dataclasses.field(default_factory=dict)
    futures: Dict[str, Any] = dataclasses.field(default_factory=dict)
    created_actors: List[str] = dataclasses.field(default_factory=list)
    created_pgs: List[str] = dataclasses.field(default_factory=list)
    clocks: Dict[str, float] = dataclasses.field(default_factory=dict)
    released_at: Dict[str, float] = dataclasses.field(default_factory=dict)
    hold: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    committed: bool = False
    aborted: bool = False

    def host_keys(self, instance_ids: Optional[Sequence[str]] = None) -> List[str]:
        chosen = [i for i in self.instances if instance_ids is None or i.instance_id in set(instance_ids)]
        return [ip for instance in chosen for ip in instance.host_ips]

    def evidence(self) -> Dict[str, Any]:
        return {"gate_name": self.gate_name, "clocks": dict(self.clocks), "released_at": dict(self.released_at),
                "tpu_hold": {iid: {k: v for k, v in h.items() if k not in ("pg", "thread")}
                             for iid, h in self.hold.items()},
                "created_actors": list(self.created_actors), "created_pgs": list(self.created_pgs),
                "committed": self.committed, "aborted": self.aborted}


def prespawn_deployment(
    inventory: SliceInventory,
    config: DeploymentConfig,
    *,
    instance_ids: Optional[Sequence[str]] = None,
    gate_timeout_s: float = 3600.0,
    device_timeout_s: float = 120.0,
) -> PreparedDeployment:
    """Start the target engines next to the TPU-owning source, up to the first device touch.

    CPU-only placement groups (``tpu_per_host=0``), fresh detached engine actors and a named
    release gate; ``initialize`` is submitted, not awaited: every worker parks in
    ``HeteroTPUWorker.init_device`` until ``release_prepared``.  Nothing is registered (registry
    or gateway) before ``commit_prespawned``.  Any failure removes only what this call created.
    """
    from .prespawn import create_gate, gate_name

    _, instances, layout_body, layout_id, digest = _select_instances(inventory, config, instance_ids)
    if _prespawn_flags(instances, config) != {True}:
        raise ValueError("pre-spawn requires engine setting prespawn=true on every deployed instance")
    live = [i.instance_id for i in instances if _find_engine_actor(layout_id, i.instance_id) is not None]
    if live:
        raise RuntimeError(f"pre-spawn needs fresh engine actors; live actors exist for {live}: release them first")
    prepared = PreparedDeployment(
        inventory=inventory, config=config, instance_ids=None if instance_ids is None else list(instance_ids),
        instances=list(instances), layout_id=layout_id, digest=digest, layout_body=layout_body,
        gate_name=gate_name(layout_id))
    prepared.clocks["prespawn_begin"] = time.time()
    try:
        prepared.native_backend, prepared.stage_clocks["backend_s"] = _prepare_backend(instances)
        prepared.gate = create_gate(prepared.gate_name, REGISTRY_NAMESPACE, prepared.host_keys())
        handles, prepared.placement_records, prepared.stage_clocks["placement_s"] = _place(
            instances, layout_id, config, 0.0, prepared.created_pgs)
        prepared.actors, _, prepared.env_reports, prepared.stage_clocks["actor_s"] = _spawn_engine_actors(
            instances, layout_id, digest, config, handles, {}, prepared.created_actors)
        prepared.clocks["actors_ready"] = time.time()
        prepared.futures = _initialize(prepared.actors, instances, layout_id, config, prespawn_gate={
            "name": prepared.gate_name, "namespace": REGISTRY_NAMESPACE,
            "timeout_s": float(gate_timeout_s), "device_timeout_s": float(device_timeout_s)})
        prepared.clocks["initialize_submitted"] = time.time()
    except Exception as exc:
        emit_event("PRESPAWN_FAILED", layout_id=layout_id, error=str(exc)[:400])
        abort_prespawned(prepared)
        raise
    emit_event("PRESPAWNED", layout_id=layout_id, instances=[i.instance_id for i in instances],
               gate=prepared.gate_name)
    return prepared


def await_parked(prepared: PreparedDeployment, *, timeout_s: float, poll_s: float = 0.25,
                 stop: Optional[Any] = None) -> Dict[str, Any]:
    """Blocks until every pre-spawned worker is parked on the gate; raises on any early exit."""
    expected = set(prepared.host_keys())
    deadline = time.time() + timeout_s
    futures = list(prepared.futures.values())
    while True:
        done, _ = ray.wait(futures, num_returns=len(futures), timeout=0)
        if done:
            ray.get(done)  # an initialization error surfaces here
            raise RuntimeError("a pre-spawned engine finished initializing before its hosts were released")
        status = ray.get(prepared.gate.status.remote(), timeout=60)
        if expected <= set(status["parked"]):
            prepared.clocks["all_parked"] = time.time()
            emit_event("PRESPAWN_PARKED", layout_id=prepared.layout_id, hosts=sorted(expected))
            return {"parked": status["parked"], "all_parked_at": prepared.clocks["all_parked"]}
        if stop is not None and stop():
            raise RuntimeError("pre-spawn park wait aborted")
        if time.time() > deadline:
            raise TimeoutError(f"pre-spawned workers not parked within {timeout_s:.0f} s: "
                               f"parked {sorted(status['parked'])} of {sorted(expected)}")
        time.sleep(poll_s)


def release_prepared(prepared: PreparedDeployment,
                     instance_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Open the gate for these instances' hosts (their source engines are gone), then create the
    TPU-hold placement groups that restore Ray's TPU accounting, without waiting for them."""
    import threading

    from ray.util.placement_group import placement_group

    from .placement import tpu_hold_bundles, tpu_hold_pg_name

    chosen = [i for i in prepared.instances if instance_ids is None or i.instance_id in set(instance_ids)]
    began = time.time()
    released = ray.get(prepared.gate.release.remote(prepared.host_keys([i.instance_id for i in chosen])),
                       timeout=60)
    prepared.released_at.update(released)
    gate_s = time.time() - began
    timeout_s = prepared.config.pg_ready_timeout_s
    for instance in chosen:
        name = tpu_hold_pg_name(prepared.layout_id, instance.instance_id)
        entry: Dict[str, Any] = {"name": name, "created_at": time.time()}
        prepared.created_pgs.append(name)
        entry["pg"] = placement_group(tpu_hold_bundles(instance, prepared.config.tpu_per_host),
                                      strategy="STRICT_SPREAD", name=name, lifetime="detached")

        def watch(entry: Dict[str, Any] = entry) -> None:
            try:
                ray.get(entry["pg"].ready(), timeout=timeout_s)
                entry["ready_at"] = time.time()
            except Exception as exc:  # noqa: BLE001 - surfaced by commit_prespawned
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"

        entry["thread"] = threading.Thread(target=watch, name=f"tpuhold-{instance.instance_id}", daemon=True)
        entry["thread"].start()
        prepared.hold[instance.instance_id] = entry
    return {"released_at": released, "gate_release_s": round(gate_s, 4),
            "tpu_hold_pgs": [prepared.hold[i.instance_id]["name"] for i in chosen]}


def commit_prespawned(prepared: PreparedDeployment) -> Dict[str, Any]:
    """Wait for the released engines, check Ray TPU accounting, then serve and register.

    Same report shape as ``deploy`` plus a ``prespawn`` block.  Any failure aborts the
    pre-spawn (own actors, own groups, gate) and re-raises.
    """
    config, layout_id = prepared.config, prepared.layout_id
    stage_clocks = dict(prepared.stage_clocks)
    try:
        unreleased = [i.instance_id for i in prepared.instances if i.instance_id not in prepared.hold]
        if unreleased:
            raise RuntimeError(f"commit before release of {unreleased}")
        registry = get_or_create_registry()
        if prepared.instance_ids is not None:
            _check_partial_gateway(registry, layout_id, prepared.digest,
                                   {i.instance_id for i in prepared.instances})
        ray.get(registry.put_inventory.remote(prepared.inventory.to_dict()))
        ray.get(registry.put_layout.remote(prepared.layout_body))

        t0 = time.time()
        init_reports = {
            instance_id: ray.get(future, timeout=config.engine_init_timeout_s)
            for instance_id, future in prepared.futures.items()
        }
        stage_clocks["engine_init_s"] = round(time.time() - t0, 2)
        emit_event("ENGINES_INITIALIZED", layout_id=layout_id, seconds=stage_clocks["engine_init_s"])

        t0 = time.time()
        for entry in prepared.hold.values():
            entry["thread"].join(timeout=config.pg_ready_timeout_s + 30.0)
            if "ready_at" not in entry:
                raise RuntimeError(f"TPU-hold placement group {entry['name']} not ready: "
                                   f"{entry.get('error', 'still pending')}")
        stage_clocks["tpu_hold_wait_s"] = round(time.time() - t0, 2)

        records = _serve_and_register(registry, prepared.instances, layout_id, prepared.digest,
                                      prepared.actors, {i.instance_id: False for i in prepared.instances},
                                      prepared.placement_records, init_reports, stage_clocks)
        snapshot = ray.get(registry.snapshot.remote(layout_id), timeout=60)
    except Exception as exc:
        emit_event("DEPLOY_FAILED", layout_id=layout_id, error=str(exc)[:400])
        abort_prespawned(prepared)
        raise
    prepared.committed = True
    prepared.clocks["committed"] = time.time()
    _kill_gate(prepared)
    return {
        "layout_id": layout_id,
        "layout_digest": prepared.digest,
        "desired_configuration": config.to_dict(),
        "resolved_layout": prepared.layout_body,
        "native_backend": prepared.native_backend,
        "stage_clocks": stage_clocks,
        "driver_environment": prepared.env_reports,
        "instances": records,
        "registry_snapshot": snapshot,
        "prespawn": prepared.evidence(),
    }


def _kill_gate(prepared: PreparedDeployment) -> None:
    if prepared.gate is not None:
        try:
            ray.kill(prepared.gate, no_restart=True)
        except Exception:  # pragma: no cover - best effort
            pass


def abort_prespawned(prepared: PreparedDeployment) -> Dict[str, Any]:
    """Fail closed: remove only this pre-spawn's own actors, groups (incl. TPU-hold) and gate."""
    if prepared.committed:
        return {"aborted": False, "reason": "already committed"}
    if prepared.aborted:
        return {"aborted": True, "reason": "already aborted"}
    _rollback(prepared.created_actors, prepared.created_pgs)
    _kill_gate(prepared)
    prepared.aborted = True
    prepared.clocks["aborted"] = time.time()
    emit_event("PRESPAWN_ABORTED", layout_id=prepared.layout_id, actors=list(prepared.created_actors),
               placement_groups=list(prepared.created_pgs))
    return {"aborted": True, "actors": list(prepared.created_actors), "placement_groups": list(prepared.created_pgs)}


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
        actor = _find_engine_actor_named(row["engine_actor_name"])
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
    maintenance_span = None
    if gateway is not None:
        status = ray.get(gateway.status.remote(), timeout=60)
        if status["layout_id"] != layout_id:
            gateway = None
        elif targets:
            began = time.time()
            maintenance = ray.get(gateway.begin_maintenance.remote(
                sorted(wanted), reason="explicit deployment drain"), timeout=300)
            maintenance_span = [began, time.time()]
    results = []
    for row in targets:
        instance_id, generation = row["instance_id"], row["generation"]
        parsed_pg = parse_pg_name(row["placement_group_name"])
        if (not row["engine_actor_name"].startswith(f"{ENGINE_ACTOR_PREFIX}--")
                or not row["engine_actor_name"].endswith(f"--{instance_id}")
                or parsed_pg is None or parsed_pg[1] != instance_id
                or row["ray_namespace"] != REGISTRY_NAMESPACE):
            raise RuntimeError("refusing to drain resources not owned by this deployment")
        # A kept engine adopted by another layout keeps its original physical names.
        actor = _find_engine_actor_named(row["engine_actor_name"])
        if actor is None:
            raise RuntimeError(f"cannot acknowledge native drain for missing actor {instance_id}")
        identity = ray.get(actor.identity.remote(), timeout=60)
        if (identity["generation"] != generation or identity["layout_id"] != layout_id
                or identity["instance_id"] != instance_id):
            raise RuntimeError("actor/registry identity changed before drain")
        entry: Dict[str, Any] = {"instance_id": instance_id, "generation": generation, "actor_found": True}
        # Wall-clock [start, end] of every step, for the switch harness's segment accounting.
        timings: Dict[str, List[float]] = {}
        entry["timings"] = timings
        if maintenance_span is not None:
            timings["maintenance"] = list(maintenance_span)
        step_began = time.time()
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
        timings["drain_ack"] = [step_began, time.time()]
        step_began = time.time()
        if not ray.get(registry.set_state.remote(instance_id, "draining", generation=generation), timeout=60):
            raise RuntimeError("registry generation changed after native drain")
        timings["registry_draining"] = [step_began, time.time()]
        if remove:
            step_began = time.time()
            entry["stop_serving"] = ray.get(actor.stop_serving.remote(), timeout=120)
            timings["stop_serving"] = [step_began, time.time()]
            step_began = time.time()
            entry["shutdown"] = ray.get(actor.shutdown.remote(), timeout=600)
            timings["shutdown"] = [step_began, time.time()]
            if any(key.endswith("_error") for key in entry["shutdown"]):
                raise RuntimeError(f"native shutdown was not acknowledged: {entry['shutdown']}")
            step_began = time.time()
            ray.kill(actor, no_restart=True)
            entry["actor_removed"] = True
            timings["kill"] = [step_began, time.time()]
            step_began = time.time()
            entry["placement_group_removed"] = release_placement_group(row["placement_group_name"])
            timings["pg_release"] = [step_began, time.time()]
            step_began = time.time()
            entry["withdrawn"] = ray.get(registry.withdraw.remote(instance_id, generation=generation), timeout=60)
            timings["withdraw"] = [step_began, time.time()]
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
