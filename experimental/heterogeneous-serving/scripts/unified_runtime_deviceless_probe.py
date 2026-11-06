#!/usr/bin/env python3
"""Deviceless (compile-only) libtpu probe of the unified-runtime programs. No TPU needed.

Needs a venv with the worker stack (jax/jaxlib 0.10.2 + libtpu 0.0.43); libtpu
builds a v5e 4x4 compile-only topology and runs the real TPU compiler:
    JAX_PLATFORMS=cpu python scripts/unified_runtime_deviceless_probe.py [--model] --output OUT.json

U1 attempt 1 failed on TPU with ``RET_CHECK ... net_util.cc:2304 device_id <
kMaxDeviceCount Unexpected device_id 8 [0:4] in replica group (0..7)`` while
compiling the TP8 program for the x-lower half; this reproduces it without a
cluster, and records which halves, orders and programs compile.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true", help="also time the U2 model-size programs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import importlib.metadata
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax._src import dispatch
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from hetero.unified_runtime import Shape, UnifiedConfig, forward_fn, split_halves, target_devices

    topology = topologies.get_topology_desc(topology_name="v5e:4x4", platform="tpu")
    devices = sorted(topology.devices, key=lambda d: d.id)
    by_id = {d.id: d for d in devices}
    config = UnifiedConfig()
    results = {"versions": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")},
               "topology": "v5e:4x4", "device_coords": {d.id: list(d.coords) for d in devices},
               "programs": []}

    def specs(mesh, shape):
        x = NamedSharding(mesh, P())
        wu = NamedSharding(mesh, P(None, None, "model"))
        wd = NamedSharding(mesh, P(None, "model", None))
        up, down = shape.weight_shapes()
        return (x, wu, wd), (jax.ShapeDtypeStruct((shape.batch, shape.d_model), jnp.float32, sharding=x),
                             jax.ShapeDtypeStruct(up, jnp.bfloat16, sharding=wu),
                             jax.ShapeDtypeStruct(down, jnp.bfloat16, sharding=wd))

    def record(label, ids, thunk):
        start = time.monotonic()
        entry = {"label": label, "device_ids": ids}
        try:
            thunk()
            entry["ok"] = True
        except Exception as exc:  # compile errors are Python exceptions, not aborts
            message = str(exc)
            at = message.find("RET_CHECK")
            entry.update(ok=False, error=(message[at:at + 300] if at >= 0 else message[:300]))
        entry["compile_s"] = time.monotonic() - start
        results["programs"].append(entry)
        print(f"{label:52} {entry['compile_s']:6.2f} s {'ok' if entry['ok'] else 'FAIL ' + entry['error'][:110]}",
              flush=True)

    def forward(devs, shape):
        mesh = Mesh(np.array(devs), ("model",))
        shardings, args = specs(mesh, shape)
        return lambda: jax.jit(forward_fn(shape.layers), in_shardings=shardings,
                               out_shardings=shardings[0]).lower(*args).compile()

    shapes = {"small": config.small()}
    if args.model:
        shapes.update(c=config.compile_shape(), model=config.model())
    for split in ("x", "y"):
        lower, upper = split_halves(devices, split)
        for half, devs in (("lower", lower), ("upper", upper)):
            for name, shape in shapes.items():
                record(f"{split}-{half} TP8 {name}", [d.id for d in devs], forward(devs, shape))
        for order in ("local", "natural"):
            target = target_devices(lower, upper, order)
            record(f"{split} target TP16 small {order}", [d.id for d in target],
                   forward(target, config.small()))
    for label, ids in (("origin 2x2 host0 TP4", [0, 1, 4, 5]), ("pair 0,1 TP2", [0, 1]),
                       ("pair 4,5 TP2", [4, 5])):
        record(label, ids, forward([by_id[i] for i in ids], config.small()))
    if args.model:
        lower, upper = split_halves(devices, "y")
        target = target_devices(lower, upper, "local")
        for bucket in config.target_buckets:
            shape = Shape(config.model_layers, config.model_d_model, config.model_d_ff, bucket)
            record(f"y target TP16 model local b{bucket}", [d.id for d in target], forward(target, shape))
        pair = Mesh(np.array(list(zip(lower, upper))), ("model", "replica"))
        tmesh = Mesh(np.array(target), ("model",))
        (_, pu, pd), (_, su, sd) = specs(pair, config.model())
        (_, tu, td), (_, stu, std) = specs(tmesh, config.model())
        for label, source, sharding in (("pair->target up", su, tu), ("pair->target down", sd, td),
                                        ("target->pair up", stu, pu), ("target->pair down", std, pd)):
            record(f"y warm device_put {label}", [], lambda source=source, sharding=sharding: jax.jit(
                dispatch._device_put_reshard, out_shardings=sharding).lower(source).compile())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
