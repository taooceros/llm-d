#!/usr/bin/env python3
"""Deviceless (compile-only) probe for an in-engine TP16 <-> DP2xTP8 switch on a v5e 4x4 block.

Builds the (data, model) meshes the way tpu-inference does (devices sorted by
``coords``, then ``jax.make_mesh``; tpu_inference/utils.py:244-322), reports which
hosts land in each data group, and compiles a DP2xTP8 forward (batch over ``data``,
all-reduce over ``model``) with libtpu, without a TPU, for three DP2xTP8 orders:
tpu-inference's own, host rows, and the S-block TP8 rings (each data group in the
order tpu-inference gives a TP8 engine on those two hosts). Weight reshards between
meshes with different device orders are not jit programs (jit rejects mismatched
device orders), so they are left to ``jax.device_put`` on hardware.

  JAX_PLATFORMS=cpu PY scripts/dp_switch_deviceless_probe.py OUT.json
"""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

D, F = 5376, 21504  # Gemma-4 31B hidden and MLP sizes; programs are compiled, never run


def host_of(dev, chips_per_host=4):
    return int(dev.id) // chips_per_host if dev.process_index is None else int(dev.process_index)


def tpu_inference_mesh(devices, shape):
    ordered = sorted(devices, key=lambda d: d.coords)
    try:
        return jax.make_mesh(shape, ("data", "model"), (jax.sharding.AxisType.Auto,) * 2, devices=ordered), "make_mesh"
    except (AssertionError, ValueError, RuntimeError):
        return Mesh(np.array(ordered).reshape(shape), ("data", "model")), "reshape"


def compile_ok(fn, *args):
    try:
        fn.lower(*args).compile()
        return {"result": "ok"}
    except Exception as exc:  # libtpu compile errors surface as XlaRuntimeError
        return {"result": "fail", "error": str(exc).splitlines()[0][:300]}


def forward(mesh):
    x = jax.ShapeDtypeStruct((64, D), jnp.bfloat16, sharding=NamedSharding(mesh, P("data", None)))
    w1 = jax.ShapeDtypeStruct((D, F), jnp.bfloat16, sharding=NamedSharding(mesh, P(None, "model")))
    w2 = jax.ShapeDtypeStruct((F, D), jnp.bfloat16, sharding=NamedSharding(mesh, P("model", None)))
    fn = jax.jit(lambda x, a, b: jax.nn.gelu(x @ a) @ b, out_shardings=NamedSharding(mesh, P("data", None)))
    return compile_ok(fn, x, w1, w2)


def tp8_ring(by_id, ids):
    devs = sorted([by_id[i] for i in ids], key=lambda d: d.coords)
    mesh, _ = tpu_inference_mesh(devs, (1, 8))
    return [int(d.id) for d in mesh.devices.flat]


def main():
    out_path = sys.argv[1]
    topo = topologies.get_topology_desc(topology_name="v5e:4x4", platform="tpu")
    devices = sorted(topo.devices, key=lambda d: d.id)
    by_id = {int(d.id): d for d in devices}
    tp16, how16 = tpu_inference_mesh(devices, (1, 16))
    dp2, how2 = tpu_inference_mesh(devices, (2, 8))
    rows = Mesh(np.array([by_id[i] for i in (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15)]).reshape(2, 8),
                ("data", "model"))
    s_rings = [tp8_ring(by_id, range(0, 8)), tp8_ring(by_id, range(8, 16))]
    s_order = Mesh(np.array([[by_id[i] for i in ring] for ring in s_rings]), ("data", "model"))
    results = {"versions": {"jax": jax.__version__}, "meshes": {}, "programs": {}}
    meshes = (("tp16", tp16, how16), ("dp2tp8", dp2, how2), ("dp2tp8_host_rows", rows, "manual"),
              ("dp2tp8_s_rings", s_order, "manual"))
    for name, mesh, how in meshes:
        grid = [[int(d.id) for d in row] for row in mesh.devices]
        results["meshes"][name] = {"built_by": how, "ids": grid,
                                   "hosts_per_data_group": [sorted({host_of(d) for d in row}) for row in mesh.devices]}
        results["programs"][f"{name} forward"] = forward(mesh)
    print(json.dumps({k: (v["ids"], v["hosts_per_data_group"]) for k, v in results["meshes"].items()}))
    print(json.dumps(results["programs"]))
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
