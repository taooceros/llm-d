#!/usr/bin/env python3
"""Deviceless sweep: which TPU slice sizes/generations reject chip-0 proper subsets (limited ICI routing mode).

    JAX_PLATFORMS=cpu python scripts/full_mesh_size_sweep.py OUT.json v5e:4x4 v5e:4x8 ...
"""
import json, sys
import numpy as np, jax, jax.numpy as jnp
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
def tp(devs):
    mesh = Mesh(np.array(devs), ("model",))
    x = jax.ShapeDtypeStruct((8, 1024), jnp.float32, sharding=NamedSharding(mesh, P()))
    w = jax.ShapeDtypeStruct((1024, 4096), jnp.bfloat16, sharding=NamedSharding(mesh, P(None, "model")))
    v = jax.ShapeDtypeStruct((4096, 1024), jnp.bfloat16, sharding=NamedSharding(mesh, P("model", None)))
    f = lambda x, w, v: (jnp.maximum(x.astype(jnp.bfloat16) @ w, 0) @ v).astype(jnp.float32)
    try:
        jax.jit(f, out_shardings=x.sharding).lower(x, w, v).compile(); return "ok"
    except Exception as e:
        s = str(e)
        return "chip0-check" if "limited ICI routing" in s else "net_util" if "net_util" in s else "other:" + s[:60]
out_path, names = sys.argv[1], sys.argv[2:]
out = {}
for name in names:
    try:
        t = topologies.get_topology_desc(topology_name=name, platform="tpu")
    except Exception as e:
        print(f"{name:10} topology error {str(e)[:80]}"); continue
    ds = sorted(t.devices, key=lambda d: d.id); n = len(ds)
    X = max(d.coords[0] for d in ds) + 1
    first_rows = [d for d in ds if d.coords[1] < max(1, 8 // X)][:8]       # origin-anchored, dense ids
    r = {"chips": n, "pair(0,1)": tp(ds[:2]), "pair(2,3)": tp(ds[2:4]),
         "origin row-block 8": tp(first_rows) if n > 8 else "n/a",
         "half (origin)": tp(ds[: n // 2]) if n > 2 else "n/a", "half (other)": tp(ds[n // 2:]) if n > 2 else "n/a",
         "full": tp(ds)}
    out[name] = r
    print(f"{name:10} " + "  ".join(f"{k}={v}" for k, v in r.items()), flush=True)
json.dump(out, open(out_path, "w"), indent=1)
