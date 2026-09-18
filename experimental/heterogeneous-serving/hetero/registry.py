"""Serving-instance registry for Ray-managed logical engines.

A multihost TP16 engine is *one* schedulable serving endpoint, not four
backends, and a live Ray worker node does not imply a ready model endpoint.  The
registry is the authoritative mapping from stable instance IDs to the current
generation, endpoint binding, capability and lifecycle state.  It describes
endpoints; it makes no routing decision.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional

import ray

REGISTRY_NAMESPACE = "omp_hetero"
REGISTRY_ACTOR_NAME = "omp-hetero-registry"

READY_STATES = ("ready",)
LIFECYCLE_STATES = ("starting", "ready", "draining", "unavailable", "stopped")


@dataclasses.dataclass
class InstanceDescriptor:
    """One registered logical serving instance."""

    # Logical instance identity
    instance_id: str
    layout_id: str
    generation: int
    config_digest: str
    # Ray ownership
    ray_namespace: str
    engine_actor_name: str
    placement_group_name: str
    placement_group_id: str
    host_chip_rank_map: List[Dict[str, Any]]
    # Reachable serving endpoint
    endpoint_host: Optional[str] = None
    endpoint_port: Optional[int] = None
    completions_path: str = "/v1/completions"
    metrics_path: str = "/metrics"
    control_path: str = "/control"
    # Serving capability
    model: str = ""
    model_build: str = ""
    tp_size: int = 0
    max_model_len: int = 0
    kv_block_size: int = 0
    kv_usable_blocks: int = 0
    kv_usable_tokens: int = 0
    max_num_seqs: int = 0
    continuation_modes: List[str] = dataclasses.field(default_factory=list)
    transport_modes: List[str] = dataclasses.field(default_factory=list)
    # Lifecycle
    state: str = "starting"
    registered_at: float = 0.0
    health_at: float = 0.0
    last_error: Optional[str] = None

    @property
    def endpoint(self) -> Optional[str]:
        if self.endpoint_host and self.endpoint_port:
            return f"{self.endpoint_host}:{self.endpoint_port}"
        return None

    def routable(self, lease_s: float, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return (
            self.state in READY_STATES
            and self.endpoint is not None
            and (now - self.health_at) <= lease_s
        )

    def to_dict(self) -> Dict[str, Any]:
        body = dataclasses.asdict(self)
        body["endpoint"] = self.endpoint
        return body


@ray.remote(num_cpus=0)
class ServingInstanceRegistry:
    """Detached named actor holding the live serving-instance descriptors."""

    def __init__(self, health_lease_s: float = 45.0) -> None:
        self._records: Dict[str, InstanceDescriptor] = {}
        self._health_lease_s = health_lease_s
        self._created_at = time.time()
        self._inventory: Optional[Dict[str, Any]] = None
        self._layouts: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------ layout

    def put_inventory(self, inventory: Dict[str, Any]) -> None:
        self._inventory = inventory

    def get_inventory(self) -> Optional[Dict[str, Any]]:
        return self._inventory

    def put_layout(self, layout: Dict[str, Any]) -> None:
        self._layouts[layout["layout_id"]] = layout

    def get_layout(self, layout_id: str) -> Optional[Dict[str, Any]]:
        return self._layouts.get(layout_id)

    def layouts(self) -> List[str]:
        return sorted(self._layouts)

    # ---------------------------------------------------------- registry

    def register(self, descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """Registers or replaces a descriptor. Stale generations are rejected."""
        record = InstanceDescriptor(**descriptor)
        if record.state not in LIFECYCLE_STATES:
            raise ValueError(f"invalid lifecycle state {record.state}")
        existing = self._records.get(record.instance_id)
        if existing is not None and record.generation < existing.generation:
            return {
                "accepted": False,
                "reason": "stale_generation",
                "current_generation": existing.generation,
            }
        record.registered_at = time.time()
        record.health_at = time.time()
        self._records[record.instance_id] = record
        return {"accepted": True, "generation": record.generation}

    def heartbeat(
        self,
        instance_id: str,
        generation: int,
        *,
        state: Optional[str] = None,
        kv_usable_blocks: Optional[int] = None,
        last_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = self._records.get(instance_id)
        if record is None:
            return {"accepted": False, "reason": "unknown_instance"}
        if generation != record.generation:
            return {
                "accepted": False,
                "reason": "stale_generation",
                "current_generation": record.generation,
            }
        record.health_at = time.time()
        if state is not None:
            if state not in LIFECYCLE_STATES:
                raise ValueError(f"invalid lifecycle state {state}")
            record.state = state
        if kv_usable_blocks is not None:
            record.kv_usable_blocks = int(kv_usable_blocks)
        if last_error is not None:
            record.last_error = last_error
        return {"accepted": True}

    def set_state(self, instance_id: str, state: str, generation: Optional[int] = None) -> bool:
        record = self._records.get(instance_id)
        if record is None:
            return False
        if generation is not None and generation != record.generation:
            return False
        if state not in LIFECYCLE_STATES:
            raise ValueError(f"invalid lifecycle state {state}")
        record.state = state
        record.health_at = time.time()
        return True

    def withdraw(self, instance_id: str, generation: Optional[int] = None) -> bool:
        record = self._records.get(instance_id)
        if record is None:
            return False
        if generation is not None and generation != record.generation:
            return False
        del self._records[instance_id]
        return True

    def get(self, instance_id: str) -> Optional[Dict[str, Any]]:
        record = self._records.get(instance_id)
        return record.to_dict() if record else None

    def snapshot(self, layout_id: Optional[str] = None) -> List[Dict[str, Any]]:
        now = time.time()
        rows = []
        for record in self._records.values():
            if layout_id is not None and record.layout_id != layout_id:
                continue
            body = record.to_dict()
            body["routable"] = record.routable(self._health_lease_s, now)
            body["health_age_s"] = round(now - record.health_at, 3)
            rows.append(body)
        rows.sort(key=lambda r: r["instance_id"])
        return rows

    def routable(self, layout_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return [row for row in self.snapshot(layout_id) if row["routable"]]

    def health_lease_s(self) -> float:
        return self._health_lease_s


def get_or_create_registry(health_lease_s: float = 45.0) -> Any:
    """Idempotently attaches to the detached registry actor."""
    try:
        return ray.get_actor(REGISTRY_ACTOR_NAME, namespace=REGISTRY_NAMESPACE)
    except ValueError:
        pass
    try:
        return ServingInstanceRegistry.options(
            name=REGISTRY_ACTOR_NAME,
            namespace=REGISTRY_NAMESPACE,
            lifetime="detached",
            get_if_exists=True,
        ).remote(health_lease_s=health_lease_s)
    except ValueError:  # pragma: no cover - lost a create race
        return ray.get_actor(REGISTRY_ACTOR_NAME, namespace=REGISTRY_NAMESPACE)


def find_registry() -> Optional[Any]:
    try:
        return ray.get_actor(REGISTRY_ACTOR_NAME, namespace=REGISTRY_NAMESPACE)
    except ValueError:
        return None
