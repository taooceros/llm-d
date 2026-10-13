"""CPU proof of replica-aware TP8 <-> TP16 resharding; no cluster access.

XLA_FLAGS=--xla_force_host_platform_device_count=16 JAX_PLATFORMS=cpu \
  .venv/bin/python scripts/reshard_prototype.py --inventory /tmp/hetero-inventory-only.json

The byte ledger counts minimum destination payload absent on the same chip,
not measured ICI traffic (routing/collective implementation may move more).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from hetero.layout import builtin_layout, resolve_layout
from hetero.slice_inventory import SliceInventory


@dataclasses.dataclass(frozen=True)
class TopologyDevice:
    coords: tuple[int, int, int]
    parent_coords: tuple[int, int, int]
    id: int
    device_kind: str = "TPU v5 lite"
    platform: str = "tpu"
    core_on_chip: int = 0


def resolve_orders(inventory):
    """Replay native v5e mesh construction with instance-relative coordinates."""
    layouts = {}
    for name in ("H", "S", "M"):
        layout = resolve_layout(SliceInventory.from_dict(inventory), builtin_layout(name))
        instances = []
        for inst in layout.instances:
            origin = tuple(lo for lo, hi in inst.chip_box)
            devices = [TopologyDevice(tuple(c[a] - origin[a] for a in range(3)),
                                      tuple(c), idx)
                       for idx, c in enumerate(c for h in inst.hosts for c in h.chip_coords)]
            ordered = mesh_utils.create_device_mesh((1, inst.tp_size), devices).flat
            instances.append({"id": inst.instance_id, "tp": inst.tp_size,
                              "box": inst.chip_box,
                              "hosts": [h.node_ip for h in inst.hosts],
                              "order": [d.parent_coords for d in ordered]})
        layouts[name] = instances
    return layouts


def gemma_tree(width=32, layers=2, kv_heads=4):
    """Dense Gemma4 schemas: gemma4.py:75-90,313-340,393-398,798-802.

    Include fused local QKV, global k_eq_v head_dim fallback, norms, MLP,
    embedding, untied head, and optional replicated PLE. All bf16, not a
    parameter-count-faithful reduction of the 31B checkpoint.
    """
    import jax.numpy as jnp
    shapes = {"embed_tokens": (width * 2, width), "lm_head": (width, width * 2)}
    specs = {"embed_tokens": P("model", None), "lm_head": P(None, None)}
    for layer in range(layers):
        prefix = f"layers.{layer}."
        local = {
            "mlp.gate_up": ((width, width * 4), P(None, "model")),
            "mlp.down": ((width * 2, width), P("model", None)),
            "attn.qkv": ((width, width * 3), P(None, "model")),
            "attn.q": ((width, 16, width), P(None, "model", None)),
            "attn.k": ((width, kv_heads, width), P(None, None, "model")),
            "attn.o": ((16, width, width), P("model", None, None)),
            "norm": ((width,), P(None)),
            "q_norm": ((width,), P(None)),
            "ple_gate": ((width, 16), P(None, None)),
        }
        for key, (shape, spec) in local.items():
            shapes[prefix + key] = shape
            specs[prefix + key] = spec
    tree = {key: jnp.asarray((np.arange(np.prod(shape)).reshape(shape) % 127).astype(np.float32),
                            dtype=jnp.bfloat16) for key, shape in shapes.items()}
    return tree, specs


def normalized_index(index, shape):
    return tuple((sl.start or 0, sl.stop if sl.stop is not None else size)
                 for sl, size in zip(index, shape))


def overlap(a, b):
    region = tuple((max(x[0], y[0]), min(x[1], y[1])) for x, y in zip(a, b))
    return region if all(lo < hi for lo, hi in region) else None


def region_size(region):
    return int(np.prod([hi - lo for lo, hi in region]))


def replica_aware_merge(sources, destination):
    """Use both identical replicas, retaining on-chip slices where possible.

    This performs device-to-device JAX transfers; no np.asarray/device_get of
    parameters is used until the separate verification step. Construction from
    per-device arrays models a common-runtime cross-process implementation.
    """
    import jax.numpy as jnp
    exemplar = sources[0]
    shape = exemplar.shape
    outputs = []
    for device, index in destination.devices_indices_map(shape).items():
        target = normalized_index(index, shape)
        output = jax.device_put(jnp.zeros(tuple(hi - lo for lo, hi in target), exemplar.dtype), device)
        # Each source replica's shards partition the whole parameter. Choose
        # the replica owning this destination chip to maximize retained bytes.
        owner = next((arr for arr in sources if device in arr.sharding.device_set), sources[0])
        for shard in owner.addressable_shards:
            source = normalized_index(shard.index, shape)
            part = overlap(source, target)
            if part is None:
                continue
            src_slice = tuple(slice(lo - source[a][0], hi - source[a][0])
                              for a, (lo, hi) in enumerate(part))
            dst_slice = tuple(slice(lo - target[a][0], hi - target[a][0])
                              for a, (lo, hi) in enumerate(part))
            piece = jax.jit(lambda x, s=src_slice: x[s])(shard.data)
            piece = jax.device_put(piece, device)
            output = jax.jit(lambda x, y, s=dst_slice: x.at[s].set(y))(output, piece)
        outputs.append(output)
    return jax.make_array_from_single_device_arrays(shape, destination, outputs)


def movement(tree, source_shardings, destination_shardings, chip_for_device):
    """Exact logical-byte lower bound from JAX's actual shard indices."""
    rows = {}
    for key, array in tree.items():
        local_regions = {}
        for sharding in source_shardings[key]:
            for device, index in sharding.devices_indices_map(array.shape).items():
                local_regions.setdefault(device, []).append(normalized_index(index, array.shape))
        for sharding in destination_shardings[key]:
            for device, index in sharding.devices_indices_map(array.shape).items():
                target = normalized_index(index, array.shape)
                # Source regions on a chip are disjoint or identical replicas;
                # this prototype has at most one resident source engine/chip.
                retained = sum(region_size(part) for source in local_regions.get(device, [])
                               if (part := overlap(source, target)) is not None)
                total = region_size(target)
                row = rows.setdefault(str(chip_for_device[device]), {"received_bytes": 0,
                                                                     "retained_bytes": 0})
                row["received_bytes"] += (total - retained) * array.dtype.itemsize
                row["retained_bytes"] += retained * array.dtype.itemsize
    return {"per_device": rows, "received_bytes": sum(r["received_bytes"] for r in rows.values()),
            "max_received_bytes": max(r["received_bytes"] for r in rows.values())}


def interval_ledger(source_instances, destination_instances, weight_bytes):
    """31B approximation: treat all payload as one TP-sharded parameter.

    Sixteen equally sized units avoid float arithmetic. Replicated norms and
    TP-dependent spec changes are NOT covered by this extrapolation.
    """
    resident = {}
    for inst in source_instances:
        units = 16 // inst["tp"]
        for rank, coord in enumerate(inst["order"]):
            resident[tuple(coord)] = set(range(rank * units, (rank + 1) * units))
    rows = []
    for inst in destination_instances:
        units = 16 // inst["tp"]
        for rank, coord in enumerate(inst["order"]):
            desired = set(range(rank * units, (rank + 1) * units))
            have = resident.get(tuple(coord), set())
            missing = len(desired - have)
            rows.append({"chip": coord, "received_bytes_estimate": missing * weight_bytes / 16,
                         "retained_bytes_estimate": len(desired & have) * weight_bytes / 16})
    return {"per_device": rows, "received_bytes_estimate": sum(r["received_bytes_estimate"] for r in rows),
            "max_received_bytes_estimate": max(r["received_bytes_estimate"] for r in rows),
            "status": "ESTIMATE: 62.5GB checkpoint approximated as entirely TP-sharded bf16"}


def run(inventory, width=32, layers=2, weight_bytes=62_500_000_000):
    if jax.default_backend() != "cpu" or len(jax.devices()) != 16:
        raise RuntimeError("Run on CPU with XLA_FLAGS=--xla_force_host_platform_device_count=16")
    layouts = resolve_orders(inventory)
    small = layouts["S"][:2]
    large = layouts["H"][0]
    # S-style halves use CPU devices 0..7 and 8..15, in native TP8 order.
    chip_order = small[0]["order"] + small[1]["order"]
    devices = jax.devices()
    by_chip = dict(zip(chip_order, devices))
    chip_for_device = dict(zip(devices, chip_order))
    meshes = [Mesh(np.array([by_chip[c] for c in inst["order"]]).reshape(1, inst["tp"]),
                   ("data", "model")) for inst in [*small, large]]
    tree, specs = gemma_tree(width, layers)
    shardings = [{k: NamedSharding(mesh, spec) for k, spec in specs.items()} for mesh in meshes]
    replicas = [jax.device_put(tree, sh) for sh in shardings[:2]]
    jax.block_until_ready(replicas)
    started = time.perf_counter()
    merged = {k: replica_aware_merge([rep[k] for rep in replicas], shardings[2][k]) for k in tree}
    jax.block_until_ready(merged)
    merge_s = time.perf_counter() - started
    # JIT resharding additionally requires identical device assignment ORDER.
    # Probe the native orders, preserving its observed error as evidence.
    replica_mesh = Mesh(np.array(devices).reshape(2, 8), ("data", "model"))
    combined = {k: jax.make_array_from_single_device_arrays(
        tree[k].shape, NamedSharding(replica_mesh, specs[k]),
        [s.data for rep in replicas for s in rep[k].addressable_shards]) for k in tree}
    started = time.perf_counter()
    direct_jit_error = None
    try:
        merged_jit = jax.jit(lambda x: x, out_shardings=shardings[2])(combined)
    except ValueError as exc:
        if "incompatible devices" not in str(exc):
            raise
        direct_jit_error = str(exc)
        # Explicitly transfer to a DP2xTP8 assignment in H's order, then
        # convert replication to TP16 shards using jit(out_shardings).
        reordered_mesh = Mesh(meshes[2].devices.reshape(2, 8), ("data", "model"))
        reordered = {k: NamedSharding(reordered_mesh, spec) for k, spec in specs.items()}
        combined = jax.device_put(combined, reordered)
        merged_jit = jax.jit(lambda x: x, out_shardings=shardings[2])(combined)
    jax.block_until_ready(merged_jit)
    jit_merge_s = time.perf_counter() - started
    started = time.perf_counter()
    back = [jax.jit(lambda x: x, out_shardings=sh)(jax.device_put(merged, sh)) for sh in shardings[:2]]
    jax.block_until_ready(back)
    split_s = time.perf_counter() - started
    for restored in [merged, merged_jit, *back]:
        for key in tree:
            np.testing.assert_array_equal(np.asarray(restored[key]), np.asarray(tree[key]))
    merge_ledger = movement(tree, {k: [sh[k] for sh in shardings[:2]] for k in tree},
                            {k: [shardings[2][k]] for k in tree}, chip_for_device)
    split_ledger = movement(tree, {k: [shardings[2][k]] for k in tree},
                            {k: [sh[k] for sh in shardings[:2]] for k in tree}, chip_for_device)
    transitions = {f"{a}->{b}": interval_ledger(layouts[a], layouts[b], weight_bytes)
                   for a in layouts for b in layouts if a != b}
    for ledger in transitions.values():
        # 400 GB/s bidirectional/chip -> 200 GB/s one direction. Efficiency
        # sensitivity is an assumption, not a measured confidence interval.
        seconds = ledger["max_received_bytes_estimate"] / 200_000_000_000
        ledger["ideal_transfer_seconds_estimate"] = seconds
        ledger["transfer_seconds_at_10_to_100_percent_efficiency_estimate"] = [seconds, seconds * 10]
        ledger["C_seconds_component_estimate"] = [90 + seconds, 100 + seconds * 10]
    return {"jax_version": jax.__version__, "jaxlib_version": jax.lib.__version__,
            "platform": "cpu", "device_count": 16, "width": width, "layers": layers,
            "logical_parameter_bytes": sum(a.size * a.dtype.itemsize for a in tree.values()),
            "tp_sharded_parameter_bytes": sum(a.size * a.dtype.itemsize for k, a in tree.items()
                                               if "model" in specs[k]),
            "partition_specs": {k: list(spec) for k, spec in specs.items()},
            "equality": {"S_to_H": True, "H_to_S_both_replicas": True},
            "cpu_compile_and_execution_s_not_ICI": {"merge": merge_s, "jit_merge": jit_merge_s,
                                                    "split": split_s},
            "prototype_movement_lower_bound": {"S_to_H": merge_ledger, "H_to_S": split_ledger},
            "direct_jit_native_order_error": direct_jit_error,
            "layout_order_status": "INFERENCE: replayed native relative-coordinate JAX mesh algorithm; not TPU-traced",
            "layouts": layouts, "transitions": transitions,
            "bandwidth_source": "https://docs.cloud.google.com/tpu/docs/v5e (400 GB/s bidirectional per chip)",
            "C_excludes": ["drain", "runner rebuild", "coordination", "HBM copies", "cold compiler cache", "errors"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    raw = args.inventory.read_bytes()
    result = run(json.loads(raw), args.width, args.layers)
    result["inventory_sha256"] = hashlib.sha256(raw).hexdigest()
    result["inventory_source"] = str(args.inventory)
    result["inventory"] = json.loads(raw)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(json.dumps({"output": str(args.output), "equality": result["equality"],
                      "prototype_movement": {k: v["received_bytes"] for k, v in
                                              result["prototype_movement_lower_bound"].items()},
                      "transitions_GB_estimate": {k: v["received_bytes_estimate"] / 1e9 for k, v in
                                                   result["transitions"].items()},
                      "cpu_seconds": result["cpu_compile_and_execution_s_not_ICI"]}, indent=2))


if __name__ == "__main__":
    main()
