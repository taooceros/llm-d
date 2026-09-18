"""Scoped, owned placement-group lifecycle for concurrent TPU serving instances.

``admission_control/tpu/placement.py`` unconditionally removes every
``CREATED``/``PENDING`` placement group in the cluster before creating one, so
bringing up a second instance destroys the first.  This module never performs
cluster-wide cleanup: it only ever touches placement groups whose name is inside
this deployment's namespace, and only ever removes ones it created or was
explicitly asked to drain.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ray
from ray.util.placement_group import (
    PlacementGroup,
    get_placement_group,
    placement_group,
    placement_group_table,
    remove_placement_group,
)

from .evidence import emit_event, emit_log
from .layout import ResolvedInstance

OWNER_PREFIX = "omp-hetero"


def pg_name(layout_id: str, instance_id: str) -> str:
    return f"{OWNER_PREFIX}--{layout_id}--{instance_id}"


def parse_pg_name(name: str) -> Optional[Tuple[str, str]]:
    if not name.startswith(OWNER_PREFIX + "--"):
        return None
    parts = name.split("--")
    if len(parts) != 3:
        return None
    return parts[1], parts[2]


@dataclasses.dataclass(frozen=True)
class PlacementRecord:
    """Ownership record for one instance placement group."""

    name: str
    pg_id: str
    layout_id: str
    instance_id: str
    state: str
    bundles: Tuple[Dict[str, float], ...]
    node_ips: Tuple[str, ...]
    created_by_this_run: bool

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["bundles"] = [dict(b) for b in self.bundles]
        out["node_ips"] = list(self.node_ips)
        return out


def intended_bundles(
    instance: ResolvedInstance, cpu_per_host: float, tpu_per_host: float
) -> List[Dict[str, float]]:
    """Node-pinned bundles in instance-local process-rank order."""
    return [
        {
            "TPU": float(tpu_per_host),
            "CPU": float(cpu_per_host),
            f"node:{host.node_ip}": 0.001,
        }
        for host in instance.hosts
    ]


def _table_entry(pg_id: str) -> Dict[str, Any]:
    return placement_group_table().get(pg_id, {})


def _bundles_match(actual: Sequence[Dict[str, Any]], intended: Sequence[Dict[str, float]]) -> bool:
    if len(actual) != len(intended):
        return False
    for got, want in zip(actual, intended):
        got_keys = {k: float(v) for k, v in got.items() if float(v) > 0}
        want_keys = {k: float(v) for k, v in want.items() if float(v) > 0}
        if set(got_keys) != set(want_keys):
            return False
        for key, value in want_keys.items():
            if abs(got_keys[key] - value) > 1e-6:
                return False
    return True

def _owned_group_ids(name: str) -> List[str]:
    """All placement-group ids currently registered under an owned name."""
    return [
        pg_id
        for pg_id, info in placement_group_table().items()
        if (info.get("name") or "") == name and info.get("state") != "REMOVED"
    ]


def _handle(pg_id: str) -> PlacementGroup:
    from ray._raylet import PlacementGroupID

    return PlacementGroup(PlacementGroupID.from_hex(pg_id))


def _live_owned_group(name: str) -> Optional[PlacementGroup]:
    ids = _owned_group_ids(name)
    if not ids:
        return None
    if len(ids) > 1:
        raise RuntimeError(
            f"placement group name {name} resolves to {len(ids)} live groups {ids}; "
            "release the duplicates before deploying"
        )
    return _handle(ids[0])



def ensure_placement_group(
    instance: ResolvedInstance,
    layout_id: str,
    *,
    cpu_per_host: float,
    tpu_per_host: float,
    ready_timeout_s: float = 300.0,
    detached: bool = True,
    created_sink: Optional[List[str]] = None,
) -> Tuple[PlacementGroup, PlacementRecord]:
    """Attaches to this instance's owned placement group or creates it.

    Reapplying the same layout must reuse the same group rather than creating a
    duplicate, and a mismatching live group is an actionable refusal rather than
    a silent reconfiguration.
    """
    name = pg_name(layout_id, instance.instance_id)
    bundles = intended_bundles(instance, cpu_per_host, tpu_per_host)

    existing = _live_owned_group(name)

    if existing is not None:
        entry = _table_entry(existing.id.hex())
        state = entry.get("state", "UNKNOWN")
        if state in ("REMOVED", "RESCHEDULING"):
            raise RuntimeError(
                f"placement group {name} is in state {state}; drain it before redeploying"
            )
        if not _bundles_match(entry.get("bundles", {}).values(), bundles):
            raise RuntimeError(
                f"live placement group {name} has bundles {entry.get('bundles')} which "
                f"conflict with the requested layout {bundles}; refusing to reconfigure "
                f"an active allocation"
            )
        ray.get(existing.ready(), timeout=ready_timeout_s)
        record = PlacementRecord(
            name=name,
            pg_id=existing.id.hex(),
            layout_id=layout_id,
            instance_id=instance.instance_id,
            state=state,
            bundles=tuple(bundles),
            node_ips=instance.host_ips,
            created_by_this_run=False,
        )
        emit_event("PG_ATTACHED", name=name, pg_id=record.pg_id, state=state)
        return existing, record

    started = time.time()
    pg = placement_group(
        bundles,
        strategy="STRICT_SPREAD",
        name=name,
        lifetime="detached" if detached else None,
    )
    if created_sink is not None:
        created_sink.append(name)
    try:
        ray.get(pg.ready(), timeout=ready_timeout_s)
    except Exception:
        # A failed partial initialization cleans up only its own new resource.
        try:
            remove_placement_group(pg)
        except Exception:  # pragma: no cover - best effort
            pass
        emit_event("PG_CREATE_FAILED", name=name, waited_s=round(time.time() - started, 2))
        raise
    record = PlacementRecord(
        name=name,
        pg_id=pg.id.hex(),
        layout_id=layout_id,
        instance_id=instance.instance_id,
        state=_table_entry(pg.id.hex()).get("state", "CREATED"),
        bundles=tuple(bundles),
        node_ips=instance.host_ips,
        created_by_this_run=True,
    )
    emit_event(
        "PG_CREATED",
        name=name,
        pg_id=record.pg_id,
        hosts=list(instance.host_ips),
        ready_s=round(time.time() - started, 2),
    )
    return pg, record


def list_owned_placement_groups(layout_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists only placement groups inside this deployment's naming namespace."""
    rows: List[Dict[str, Any]] = []
    for pg_id, info in placement_group_table().items():
        name = info.get("name") or ""
        parsed = parse_pg_name(name)
        if parsed is None:
            continue
        if layout_id is not None and parsed[0] != layout_id:
            continue
        rows.append(
            {
                "pg_id": pg_id,
                "name": name,
                "layout_id": parsed[0],
                "instance_id": parsed[1],
                "state": info.get("state"),
                "bundles": info.get("bundles"),
                "bundles_to_node_id": info.get("bundles_to_node_id"),
            }
        )
    rows.sort(key=lambda r: r["name"])
    return rows


def release_placement_group(name: str) -> bool:
    """Removes every live owned group under this name. Never touches foreign ones."""
    if parse_pg_name(name) is None:
        raise ValueError(f"refusing to remove placement group outside {OWNER_PREFIX}: {name}")
    ids = _owned_group_ids(name)
    removed = 0
    for pg_id in ids:
        try:
            remove_placement_group(_handle(pg_id))
            removed += 1
        except Exception as exc:  # pragma: no cover - already gone
            emit_event("PG_RELEASE_FAILED", name=name, pg_id=pg_id, error=str(exc)[:200])
    if removed:
        emit_event("PG_RELEASED", name=name, groups=removed)
    return removed > 0


def verify_bundle_placement(
    pg: PlacementGroup, instance: ResolvedInstance
) -> List[Dict[str, Any]]:
    """Checks that bundle *i* really landed on this instance's rank-*i* host."""
    entry = _table_entry(pg.id.hex())
    mapping = entry.get("bundles_to_node_id", {}) or {}
    by_node_id = {h.ray_node_id: h for h in instance.hosts}
    rows: List[Dict[str, Any]] = []
    for host in instance.hosts:
        actual_node = mapping.get(host.bundle_index)
        matched = actual_node == host.ray_node_id
        rows.append(
            {
                "bundle_index": host.bundle_index,
                "expected_node_ip": host.node_ip,
                "expected_ray_node_id": host.ray_node_id,
                "actual_ray_node_id": actual_node,
                "matched": matched,
            }
        )
        if not matched:
            observed = by_node_id.get(actual_node)
            raise RuntimeError(
                f"bundle {host.bundle_index} of {instance.instance_id} landed on "
                f"{actual_node} ({observed.node_ip if observed else 'unknown node'}) "
                f"instead of {host.ray_node_id} ({host.node_ip})"
            )
    emit_log(f"bundle placement verified for {instance.instance_id}")
    return rows
