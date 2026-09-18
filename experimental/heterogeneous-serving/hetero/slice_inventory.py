"""Physical TPU slice inventory backed by observed hardware evidence.

Chip coordinates are *observed* from an initialized JAX runtime on the real
hosts, never derived from ``TPU_WORKER_ID`` arithmetic or sorted IP order.  A
prior full-slice proof (``results/measurement_audit_20260911/physical_topology.json``)
shows worker id 0 sitting at chip origin ``(2, 2)``, so id-derived geometry is
demonstrably wrong on this hardware.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .evidence import emit_event, emit_log

Coord = Tuple[int, int, int]

PROBE_PG_NAME = "omp-hetero-slice-probe"


@dataclasses.dataclass(frozen=True)
class HostRecord:
    """One physical TPU host with observed chip coordinates."""

    node_ip: str
    ray_node_id: str
    hostname: str
    tpu_worker_id: int
    tpu_name: str
    tpu_count: int
    chip_coords: Tuple[Coord, ...]
    jax_process_index: int
    jax_global_device_count: int

    @property
    def chip_box(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        xs = [c[0] for c in self.chip_coords]
        ys = [c[1] for c in self.chip_coords]
        zs = [c[2] for c in self.chip_coords]
        return ((min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs)))

    @property
    def chip_origin(self) -> Coord:
        box = self.chip_box
        return (box[0][0], box[1][0], box[2][0])

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["chip_coords"] = [list(c) for c in self.chip_coords]
        out["chip_box"] = [list(a) for a in self.chip_box]
        return out


@dataclasses.dataclass(frozen=True)
class SliceInventory:
    """Observed geometry of the whole parent TPU slice."""

    slice_name: str
    hosts: Tuple[HostRecord, ...]
    chip_dims: Coord
    host_chip_dims: Coord
    host_grid_dims: Coord
    ray_session: str
    observed_at: float
    ray_node_generation: str

    @property
    def total_chips(self) -> int:
        return sum(h.tpu_count for h in self.hosts)

    def host_grid_position(self, host: HostRecord) -> Tuple[int, int, int]:
        ox, oy, oz = host.chip_origin
        cx, cy, cz = self.host_chip_dims
        return (ox // cx, oy // cy, oz // cz)

    def by_ip(self, node_ip: str) -> HostRecord:
        for host in self.hosts:
            if host.node_ip == node_ip:
                return host
        raise KeyError(f"host {node_ip} is not part of slice {self.slice_name}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "chip_dims": list(self.chip_dims),
            "host_chip_dims": list(self.host_chip_dims),
            "host_grid_dims": list(self.host_grid_dims),
            "total_chips": self.total_chips,
            "ray_session": self.ray_session,
            "ray_node_generation": self.ray_node_generation,
            "observed_at": self.observed_at,
            "hosts": [
                dict(h.to_dict(), host_grid_position=list(self.host_grid_position(h)))
                for h in self.hosts
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SliceInventory":
        hosts = tuple(
            HostRecord(
                node_ip=h["node_ip"],
                ray_node_id=h["ray_node_id"],
                hostname=h["hostname"],
                tpu_worker_id=int(h["tpu_worker_id"]),
                tpu_name=h["tpu_name"],
                tpu_count=int(h["tpu_count"]),
                chip_coords=tuple(tuple(int(v) for v in c) for c in h["chip_coords"]),
                jax_process_index=int(h["jax_process_index"]),
                jax_global_device_count=int(h["jax_global_device_count"]),
            )
            for h in data["hosts"]
        )
        return cls(
            slice_name=data["slice_name"],
            hosts=hosts,
            chip_dims=tuple(data["chip_dims"]),
            host_chip_dims=tuple(data["host_chip_dims"]),
            host_grid_dims=tuple(data["host_grid_dims"]),
            ray_session=data.get("ray_session", ""),
            observed_at=float(data.get("observed_at", 0.0)),
            ray_node_generation=data.get("ray_node_generation", ""),
        )


def discover_ray_tpu_nodes() -> List[Dict[str, Any]]:
    """Returns alive Ray nodes that own TPU chips, with their resource facts."""
    rows: List[Dict[str, Any]] = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        resources = node.get("Resources", {}) or {}
        chips = int(resources.get("TPU", 0))
        if chips <= 0:
            continue
        rows.append(
            {
                "node_ip": node["NodeManagerAddress"],
                "ray_node_id": node["NodeID"],
                "tpu_count": chips,
                "resources": {k: v for k, v in resources.items() if not k.startswith("node:")},
                "labels": dict(node.get("Labels", {}) or {}),
            }
        )
    rows.sort(key=lambda r: r["node_ip"])
    return rows


@ray.remote(num_cpus=1)
class _SliceProbe:
    """Per-host probe that initializes JAX once and reports observed geometry."""

    def facts(self) -> Dict[str, Any]:
        env = os.environ
        return {
            "node_ip": ray.util.get_node_ip_address(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "tpu_worker_id": env.get("TPU_WORKER_ID"),
            "tpu_worker_hostnames": env.get("TPU_WORKER_HOSTNAMES"),
            "tpu_name": env.get("TPU_NAME"),
            "tpu_multihost_backend": env.get("TPU_MULTIHOST_BACKEND"),
            "tpu_visible_chips": env.get("TPU_VISIBLE_CHIPS"),
            "jax_platforms": env.get("JAX_PLATFORMS"),
            "ray_node_id": ray.get_runtime_context().get_node_id(),
        }

    def observe_devices(self) -> Dict[str, Any]:
        """Initializes the full-slice JAX runtime and reports local chip coords."""
        import jax  # imported here so no TPU runtime exists until asked

        started = time.time()
        jax.distributed.initialize()
        local = jax.local_devices()
        payload = {
            "node_ip": ray.util.get_node_ip_address(),
            "hostname": socket.gethostname(),
            "process_index": int(jax.process_index()),
            "process_count": int(jax.process_count()),
            "global_device_count": int(jax.device_count()),
            "local_device_count": int(jax.local_device_count()),
            "chip_coords": [list(int(v) for v in d.coords) for d in local],
            "device_kind": local[0].device_kind if local else None,
            "device_ids": [int(d.id) for d in local],
            "slice_index": [int(getattr(d, "slice_index", 0)) for d in local],
            "init_s": round(time.time() - started, 3),
            "jax_version": jax.__version__,
        }
        return payload


def _bundles(node_ips: Sequence[str], tpu_per_node: float, cpu_per_node: float) -> List[Dict[str, float]]:
    return [
        {"TPU": float(tpu_per_node), "CPU": float(cpu_per_node), f"node:{ip}": 0.001}
        for ip in node_ips
    ]


def observe_slice(
    *,
    probe_cpu: float = 1.0,
    ready_timeout_s: float = 180.0,
    probe_timeout_s: float = 600.0,
) -> SliceInventory:
    """Observes the parent slice geometry with a short-lived owned probe runtime.

    Creates one owned placement group spanning every discovered TPU host, runs
    one probe actor per host, initializes JAX across the full slice, records the
    observed chip coordinates and then destroys everything it created.
    """
    nodes = discover_ray_tpu_nodes()
    if not nodes:
        raise RuntimeError("No alive Ray nodes expose TPU resources")
    node_ips = [n["node_ip"] for n in nodes]
    chips_per_host = {n["node_ip"]: n["tpu_count"] for n in nodes}
    if len(set(chips_per_host.values())) != 1:
        raise RuntimeError(f"Heterogeneous chips per host is unsupported: {chips_per_host}")
    per_host = float(next(iter(chips_per_host.values())))

    emit_event("SLICE_PROBE_BEGIN", hosts=len(node_ips), chips_per_host=per_host)
    pg: Optional[PlacementGroup] = None
    actors: List[Any] = []
    try:
        pg = placement_group(
            _bundles(node_ips, per_host, probe_cpu),
            strategy="STRICT_SPREAD",
            name=PROBE_PG_NAME,
        )
        ray.get(pg.ready(), timeout=ready_timeout_s)
        emit_event("SLICE_PROBE_PG_READY", pg_id=pg.id.hex())

        for index, ip in enumerate(node_ips):
            actors.append(
                _SliceProbe.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=index,
                        placement_group_capture_child_tasks=False,
                    ),
                    resources={f"node:{ip}": 0.001},
                ).remote()
            )

        placement = ray.get([a.facts.remote() for a in actors], timeout=probe_timeout_s)
        for index, fact in enumerate(placement):
            if fact["node_ip"] != node_ips[index]:
                raise RuntimeError(
                    f"probe {index} landed on {fact['node_ip']}, expected {node_ips[index]}"
                )
        emit_event("SLICE_PROBE_PINNED", hosts=len(placement))

        observed = ray.get([a.observe_devices.remote() for a in actors], timeout=probe_timeout_s)
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:  # pragma: no cover - best effort teardown
                pass
        if pg is not None:
            try:
                ray.util.remove_placement_group(pg)
            except Exception:  # pragma: no cover - best effort teardown
                pass
        emit_event("SLICE_PROBE_RELEASED")
        # Let the probe processes actually exit before anything else claims chips.
        time.sleep(15.0)

    facts_by_ip = {f["node_ip"]: f for f in placement}
    observed_by_ip = {o["node_ip"]: o for o in observed}
    global_counts = {o["global_device_count"] for o in observed}
    if len(global_counts) != 1:
        raise RuntimeError(f"Inconsistent global device counts across hosts: {global_counts}")
    global_count = next(iter(global_counts))
    if global_count != sum(chips_per_host.values()):
        raise RuntimeError(
            f"JAX reports {global_count} devices but Ray reports {sum(chips_per_host.values())} chips"
        )

    hosts: List[HostRecord] = []
    for ip in node_ips:
        fact = facts_by_ip[ip]
        obs = observed_by_ip[ip]
        worker_id = fact.get("tpu_worker_id")
        if worker_id is None:
            raise RuntimeError(f"host {ip} does not expose TPU_WORKER_ID; geometry is unqualified")
        coords = tuple(tuple(int(v) for v in c) for c in obs["chip_coords"])
        if len(coords) != int(chips_per_host[ip]):
            raise RuntimeError(
                f"host {ip} reports {len(coords)} JAX chips but Ray advertises {chips_per_host[ip]}"
            )
        hosts.append(
            HostRecord(
                node_ip=ip,
                ray_node_id=fact["ray_node_id"],
                hostname=fact["hostname"],
                tpu_worker_id=int(worker_id),
                tpu_name=fact.get("tpu_name") or "",
                tpu_count=int(chips_per_host[ip]),
                chip_coords=coords,
                jax_process_index=int(obs["process_index"]),
                jax_global_device_count=int(obs["global_device_count"]),
            )
        )

    all_coords = [c for h in hosts for c in h.chip_coords]
    if len(set(all_coords)) != len(all_coords):
        raise RuntimeError("Duplicate chip coordinates observed across hosts")
    chip_dims = tuple(max(c[axis] for c in all_coords) + 1 for axis in range(3))
    if chip_dims[0] * chip_dims[1] * chip_dims[2] != len(all_coords):
        raise RuntimeError(
            f"Observed chips {len(all_coords)} do not fill the bounding box {chip_dims}"
        )

    host_dims = {h.node_ip: _box_dims(h) for h in hosts}
    if len(set(host_dims.values())) != 1:
        raise RuntimeError(f"Hosts do not share one chip sub-box shape: {host_dims}")
    host_chip_dims = next(iter(host_dims.values()))
    for host in hosts:
        origin = host.chip_origin
        for axis in range(3):
            if origin[axis] % host_chip_dims[axis]:
                raise RuntimeError(
                    f"host {host.node_ip} chip origin {origin} is not aligned to {host_chip_dims}"
                )
    host_grid_dims = tuple(chip_dims[axis] // host_chip_dims[axis] for axis in range(3))

    slice_names = {h.tpu_name for h in hosts if h.tpu_name}
    slice_name = slice_names.pop() if len(slice_names) == 1 else "unknown-slice"

    inventory = SliceInventory(
        slice_name=slice_name,
        hosts=tuple(sorted(hosts, key=lambda h: h.chip_origin[::-1])),
        chip_dims=chip_dims,
        host_chip_dims=host_chip_dims,
        host_grid_dims=host_grid_dims,
        ray_session=ray.get_runtime_context().get_job_id(),
        observed_at=time.time(),
        ray_node_generation=_node_generation(nodes),
    )
    emit_log(
        "observed slice "
        f"{inventory.slice_name} chips={inventory.chip_dims} "
        f"host_chips={inventory.host_chip_dims} host_grid={inventory.host_grid_dims}"
    )
    emit_event(
        "SLICE_PROBE_COMPLETE",
        slice_name=inventory.slice_name,
        chip_dims=list(inventory.chip_dims),
        host_grid_dims=list(inventory.host_grid_dims),
    )
    return inventory


def _box_dims(host: HostRecord) -> Coord:
    box = host.chip_box
    return tuple(box[axis][1] - box[axis][0] + 1 for axis in range(3))


def _node_generation(nodes: Sequence[Dict[str, Any]]) -> str:
    import hashlib

    payload = json.dumps(
        sorted((n["ray_node_id"], n["node_ip"], n["tpu_count"]) for n in nodes),
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
