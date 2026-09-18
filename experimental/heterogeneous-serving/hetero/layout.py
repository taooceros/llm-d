"""Static heterogeneous instance layout resolution and TPU runtime contract.

A layout maps every selected physical chip to exactly one logical serving
instance and to a deterministic instance-local process rank.  Resolution is
driven by *observed* chip coordinates from :mod:`hetero.slice_inventory`; the
resolver refuses to guess geometry from worker ids or sorted IP order.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .slice_inventory import Coord, HostRecord, SliceInventory

TPU_PROCESS_PORT = "8471"
JAX_COMPILATION_CACHE_DIR = "/models/jax_cache"


@dataclasses.dataclass(frozen=True)
class InstanceRequest:
    """Requested logical serving instance."""

    instance_id: str
    tp_size: int
    host_ips: Optional[Tuple[str, ...]] = None  # explicit pin; else auto-packed

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"instance_id": self.instance_id, "tp_size": self.tp_size}
        if self.host_ips:
            out["host_ips"] = list(self.host_ips)
        return out


@dataclasses.dataclass(frozen=True)
class LayoutRequest:
    """A named static layout: the full set of instances for one workload."""

    layout_id: str
    instances: Tuple[InstanceRequest, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layout_id": self.layout_id,
            "instances": [i.to_dict() for i in self.instances],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayoutRequest":
        return cls(
            layout_id=data["layout_id"],
            instances=tuple(
                InstanceRequest(
                    instance_id=i["instance_id"],
                    tp_size=int(i["tp_size"]),
                    host_ips=tuple(i["host_ips"]) if i.get("host_ips") else None,
                )
                for i in data["instances"]
            ),
        )


@dataclasses.dataclass(frozen=True)
class ResolvedHost:
    """One host of one instance, with physical identity kept separate from rank."""

    node_ip: str
    ray_node_id: str
    hostname: str
    physical_tpu_worker_id: int
    process_rank: int
    bundle_index: int
    chip_coords: Tuple[Coord, ...]
    host_grid_position: Coord

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["chip_coords"] = [list(c) for c in self.chip_coords]
        out["host_grid_position"] = list(self.host_grid_position)
        return out


@dataclasses.dataclass(frozen=True)
class ResolvedInstance:
    """A validated, contiguous, disjoint execution domain."""

    instance_id: str
    tp_size: int
    hosts: Tuple[ResolvedHost, ...]
    chip_box: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    chip_dims: Coord
    host_grid_dims: Coord
    host_chip_dims: Coord

    @property
    def coordinator_ip(self) -> str:
        return self.hosts[0].node_ip

    @property
    def host_ips(self) -> Tuple[str, ...]:
        return tuple(h.node_ip for h in self.hosts)

    @property
    def subslice_topology(self) -> str:
        return f"{self.chip_dims[0]}x{self.chip_dims[1]}"

    @property
    def host_bounds(self) -> str:
        return ",".join(str(v) for v in self.host_grid_dims)

    @property
    def chips_per_host_bounds(self) -> str:
        return ",".join(str(v) for v in self.host_chip_dims)

    @property
    def process_addresses(self) -> str:
        return ",".join(f"{ip}:{TPU_PROCESS_PORT}" for ip in self.host_ips)

    def runtime_env(self) -> Dict[str, str]:
        """Intended per-process TPU runtime environment for this instance.

        ``CLOUD_TPU_TASK_ID``/``TPU_WORKER_ID`` are deliberately absent: they are
        per-process and are assigned inside the worker from this instance's
        ordered ``TPU_PROCESS_ADDRESSES`` (see
        :mod:`hetero.tpu_worker`), which keeps the physical worker id intact for
        evidence while giving libtpu an instance-local rank.
        """
        return {
            "TPU_MULTIHOST_BACKEND": "ray",
            "TPU_TOPOLOGY": self.subslice_topology,
            "TPU_HOST_BOUNDS": self.host_bounds,
            "TPU_PROCESS_BOUNDS": self.host_bounds,
            "TPU_CHIPS_PER_HOST_BOUNDS": self.chips_per_host_bounds,
            "TPU_CHIPS_PER_PROCESS_BOUNDS": self.chips_per_host_bounds,
            "TPU_PROCESS_ADDRESSES": self.process_addresses,
            "TPU_PROCESS_PORT": TPU_PROCESS_PORT,
            "JAX_PLATFORMS": "tpu,cpu",
            "JAX_COMPILATION_CACHE_DIR": JAX_COMPILATION_CACHE_DIR,
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "OMP_HETERO_INSTANCE_ID": self.instance_id,
        }

    def process_env(self, process_rank: int) -> Dict[str, str]:
        """Intended environment for one specific process of this instance."""
        env = self.runtime_env()
        env["CLOUD_TPU_TASK_ID"] = str(process_rank)
        env["TPU_WORKER_ID"] = str(process_rank)
        return env

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "tp_size": self.tp_size,
            "coordinator_ip": self.coordinator_ip,
            "host_ips": list(self.host_ips),
            "chip_box": [list(a) for a in self.chip_box],
            "chip_dims": list(self.chip_dims),
            "host_grid_dims": list(self.host_grid_dims),
            "host_chip_dims": list(self.host_chip_dims),
            "subslice_topology": self.subslice_topology,
            "host_bounds": self.host_bounds,
            "chips_per_host_bounds": self.chips_per_host_bounds,
            "process_addresses": self.process_addresses,
            "intended_runtime_env": self.runtime_env(),
            "intended_process_env": {
                h.node_ip: self.process_env(h.process_rank) for h in self.hosts
            },
            "hosts": [h.to_dict() for h in self.hosts],
        }


@dataclasses.dataclass(frozen=True)
class ResolvedLayout:
    """The complete validated mapping for one static configuration."""

    layout_id: str
    slice_name: str
    instances: Tuple[ResolvedInstance, ...]
    slice_chip_dims: Coord
    host_chip_dims: Coord
    host_grid_dims: Coord
    total_chips_allocated: int
    total_chips_in_slice: int

    def instance(self, instance_id: str) -> ResolvedInstance:
        for inst in self.instances:
            if inst.instance_id == instance_id:
                return inst
        raise KeyError(instance_id)

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "layout_id": self.layout_id,
            "slice_name": self.slice_name,
            "slice_chip_dims": list(self.slice_chip_dims),
            "host_chip_dims": list(self.host_chip_dims),
            "host_grid_dims": list(self.host_grid_dims),
            "total_chips_allocated": self.total_chips_allocated,
            "total_chips_in_slice": self.total_chips_in_slice,
            "instances": [i.to_dict() for i in self.instances],
        }
        body["layout_digest"] = layout_digest(body)
        return body


def layout_digest(body: Dict[str, Any]) -> str:
    """Stable configuration digest over the resolved physical assignment."""
    material = {
        "layout_id": body["layout_id"],
        "slice_name": body["slice_name"],
        "instances": [
            {
                "instance_id": i["instance_id"],
                "tp_size": i["tp_size"],
                "host_ips": i["host_ips"],
                "chip_box": i["chip_box"],
                "intended_runtime_env": i["intended_runtime_env"],
            }
            for i in body["instances"]
        ],
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class LayoutError(RuntimeError):
    """Raised when a requested layout cannot be physically satisfied."""


def _rectangle_shapes(n_hosts: int, grid: Coord) -> List[Tuple[int, int]]:
    """Candidate (x, y) host-grid rectangles for ``n_hosts``, most square first."""
    shapes = []
    for ax in range(1, n_hosts + 1):
        if n_hosts % ax:
            continue
        ay = n_hosts // ax
        if ax > grid[0] or ay > grid[1]:
            continue
        shapes.append((ax, ay))
    shapes.sort(key=lambda s: (abs(s[0] - s[1]), s[0]))
    return shapes


def _place_rectangle(
    occupied: Dict[Tuple[int, int], str],
    grid: Coord,
    n_hosts: int,
) -> Optional[List[Tuple[int, int]]]:
    """First-fit, most-square rectangle in host-grid space, scanning y then x."""
    for ax, ay in _rectangle_shapes(n_hosts, grid):
        for oy in range(grid[1] - ay + 1):
            for ox in range(grid[0] - ax + 1):
                cells = [(ox + dx, oy + dy) for dy in range(ay) for dx in range(ax)]
                if any(cell in occupied for cell in cells):
                    continue
                return cells
    return None


def resolve_layout(inventory: SliceInventory, request: LayoutRequest) -> ResolvedLayout:
    """Resolves and fully validates a static layout against observed geometry."""
    if inventory.chip_dims[2] != 1 or inventory.host_chip_dims[2] != 1:
        raise LayoutError(
            f"Only 2D slices are supported here; observed chip dims {inventory.chip_dims}"
        )
    chips_per_host = (
        inventory.host_chip_dims[0] * inventory.host_chip_dims[1] * inventory.host_chip_dims[2]
    )
    grid = inventory.host_grid_dims
    by_grid: Dict[Tuple[int, int], HostRecord] = {}
    for host in inventory.hosts:
        pos = inventory.host_grid_position(host)
        by_grid[(pos[0], pos[1])] = host

    occupied: Dict[Tuple[int, int], str] = {}
    # Honour explicit pins first so auto-packing cannot steal their hosts.
    pinned: Dict[str, List[Tuple[int, int]]] = {}
    for req in request.instances:
        if not req.host_ips:
            continue
        cells = []
        for ip in req.host_ips:
            host = inventory.by_ip(ip)
            pos = inventory.host_grid_position(host)
            cell = (pos[0], pos[1])
            if cell in occupied:
                raise LayoutError(
                    f"instance {req.instance_id} pins host {ip} already owned by "
                    f"{occupied[cell]}"
                )
            occupied[cell] = req.instance_id
            cells.append(cell)
        pinned[req.instance_id] = cells

    assignment: Dict[str, List[Tuple[int, int]]] = {}
    for req in request.instances:
        if req.tp_size % chips_per_host:
            raise LayoutError(
                f"instance {req.instance_id} tp_size={req.tp_size} is not a multiple of "
                f"{chips_per_host} chips per host"
            )
        n_hosts = req.tp_size // chips_per_host
        if req.instance_id in pinned:
            cells = pinned[req.instance_id]
            if len(cells) != n_hosts:
                raise LayoutError(
                    f"instance {req.instance_id} pins {len(cells)} hosts but tp_size="
                    f"{req.tp_size} needs {n_hosts}"
                )
        else:
            cells = _place_rectangle(occupied, grid, n_hosts)
            if cells is None:
                raise LayoutError(
                    f"no free contiguous {n_hosts}-host rectangle remains for "
                    f"{req.instance_id} in host grid {grid}"
                )
            for cell in cells:
                occupied[cell] = req.instance_id
        assignment[req.instance_id] = cells

    instances: List[ResolvedInstance] = []
    for req in request.instances:
        cells = assignment[req.instance_id]
        hosts = [by_grid[cell] for cell in cells]
        # Deterministic instance-local rank order: row-major with x fastest.
        hosts.sort(key=lambda h: (h.chip_origin[1], h.chip_origin[0]))
        resolved_hosts = tuple(
            ResolvedHost(
                node_ip=h.node_ip,
                ray_node_id=h.ray_node_id,
                hostname=h.hostname,
                physical_tpu_worker_id=h.tpu_worker_id,
                process_rank=rank,
                bundle_index=rank,
                chip_coords=h.chip_coords,
                host_grid_position=inventory.host_grid_position(h),
            )
            for rank, h in enumerate(hosts)
        )
        instance = _validate_instance(
            req, resolved_hosts, inventory.host_chip_dims, chips_per_host
        )
        instances.append(instance)

    _validate_disjoint(instances)

    total = sum(i.tp_size for i in instances)
    layout = ResolvedLayout(
        layout_id=request.layout_id,
        slice_name=inventory.slice_name,
        instances=tuple(instances),
        slice_chip_dims=inventory.chip_dims,
        host_chip_dims=inventory.host_chip_dims,
        host_grid_dims=inventory.host_grid_dims,
        total_chips_allocated=total,
        total_chips_in_slice=inventory.total_chips,
    )
    return layout


def _validate_instance(
    req: InstanceRequest,
    hosts: Sequence[ResolvedHost],
    host_chip_dims: Coord,
    chips_per_host: int,
) -> ResolvedInstance:
    coords = [c for h in hosts for c in h.chip_coords]
    if len(coords) != req.tp_size:
        raise LayoutError(
            f"instance {req.instance_id} selects {len(coords)} chips for tp_size={req.tp_size}"
        )
    if len(set(coords)) != len(coords):
        raise LayoutError(f"instance {req.instance_id} selects duplicate chips")
    box = tuple(
        (min(c[axis] for c in coords), max(c[axis] for c in coords)) for axis in range(3)
    )
    dims = tuple(box[axis][1] - box[axis][0] + 1 for axis in range(3))
    if dims[0] * dims[1] * dims[2] != len(coords):
        raise LayoutError(
            f"instance {req.instance_id} chips {sorted(coords)} do not fill the "
            f"contiguous box {box}"
        )
    ranks = [h.process_rank for h in hosts]
    if ranks != list(range(len(hosts))):
        raise LayoutError(f"instance {req.instance_id} has non-contiguous process ranks {ranks}")
    host_grid_dims = tuple(dims[axis] // host_chip_dims[axis] for axis in range(3))
    if host_grid_dims[0] * host_grid_dims[1] * host_grid_dims[2] != len(hosts):
        raise LayoutError(
            f"instance {req.instance_id} host grid {host_grid_dims} does not match "
            f"{len(hosts)} hosts"
        )
    if len({h.node_ip for h in hosts}) != len(hosts):
        raise LayoutError(f"instance {req.instance_id} repeats a host")
    if any(len(h.chip_coords) != chips_per_host for h in hosts):
        raise LayoutError(f"instance {req.instance_id} has a host with unexpected chip count")
    return ResolvedInstance(
        instance_id=req.instance_id,
        tp_size=req.tp_size,
        hosts=tuple(hosts),
        chip_box=box,
        chip_dims=dims,
        host_grid_dims=host_grid_dims,
        host_chip_dims=host_chip_dims,
    )


def _validate_disjoint(instances: Sequence[ResolvedInstance]) -> None:
    seen_chips: Dict[Coord, str] = {}
    seen_hosts: Dict[str, str] = {}
    for inst in instances:
        for host in inst.hosts:
            if host.node_ip in seen_hosts:
                raise LayoutError(
                    f"host {host.node_ip} is claimed by both {seen_hosts[host.node_ip]} "
                    f"and {inst.instance_id}"
                )
            seen_hosts[host.node_ip] = inst.instance_id
            for coord in host.chip_coords:
                if coord in seen_chips:
                    raise LayoutError(
                        f"chip {coord} is claimed by both {seen_chips[coord]} and "
                        f"{inst.instance_id}"
                    )
                seen_chips[coord] = inst.instance_id


BUILTIN_LAYOUTS: Dict[str, Tuple[Tuple[str, int], ...]] = {
    # Heterogeneous mixed layout: two dense-batch small instances plus one large.
    "M": (("small_a", 8), ("small_b", 8), ("large", 16)),
    # Homogeneous large baseline.
    "H": (("large_a", 16), ("large_b", 16)),
    # Homogeneous small baseline.
    "S": (("small_a", 8), ("small_b", 8), ("small_c", 8), ("small_d", 8)),
}


def builtin_layout(layout_id: str) -> LayoutRequest:
    if layout_id not in BUILTIN_LAYOUTS:
        raise KeyError(f"unknown layout {layout_id}; known: {sorted(BUILTIN_LAYOUTS)}")
    return LayoutRequest(
        layout_id=layout_id,
        instances=tuple(
            InstanceRequest(instance_id=name, tp_size=tp)
            for name, tp in BUILTIN_LAYOUTS[layout_id]
        ),
    )
