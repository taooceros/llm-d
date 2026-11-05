#!/usr/bin/env python3
"""Deviceless: which device subsets of a v5e 4x8 compile in limited-ICI-routing mode (rectangle / chip-0 rule).

    JAX_PLATFORMS=cpu python scripts/full_mesh_subset_rule.py OUT.json
"""
import sys
import json, numpy as np, jax, jax.numpy as jnp
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
t = topologies.get_topology_desc(topology_name="v5e:4x8", platform="tpu")
by = {d.id: d for d in t.devices}
def tp(ids):
    n = len(ids); mesh = Mesh(np.array([by[i] for i in ids]), ("model",))
    x = jax.ShapeDtypeStruct((8, 1024), jnp.float32, sharding=NamedSharding(mesh, P()))
    w = jax.ShapeDtypeStruct((1024, 256 * n), jnp.bfloat16, sharding=NamedSharding(mesh, P(None, "model")))
    v = jax.ShapeDtypeStruct((256 * n, 1024), jnp.bfloat16, sharding=NamedSharding(mesh, P("model", None)))
    f = lambda x, w, v: (jnp.maximum(x.astype(jnp.bfloat16) @ w, 0) @ v).astype(jnp.float32)
    try:
        jax.jit(f, out_shardings=x.sharding).lower(x, w, v).compile(); return "ok"
    except Exception as e:
        s = str(e)
        return "all-slice-check" if "limited ICI routing" in s else "net_util" if "net_util" in s else "other: " + s[:90].replace("\n", " ")
def box(x0, x1, y0, y1): return [x + 4 * y for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
cases = {
 # rectangles not containing chip 0
 "box x0-3 y2-7 (24)": box(0, 3, 2, 7), "box x0-3 y3-7 (20)": box(0, 3, 3, 7), "box x0-3 y1-7 (28)": box(0, 3, 1, 7),
 "box x1-2 y1-2 (4)": box(1, 2, 1, 2), "box x1-3 y0-0 (3)": box(1, 3, 0, 0), "box x1-1 y0-7 (8)": box(1, 1, 0, 7),
 "box x1-3 y0-7 (24)": box(1, 3, 0, 7), "box x2-3 y0-1 (4)": box(2, 3, 0, 1), "box x1-3 y1-1 (3)": box(1, 3, 1, 1),
 "box x0-0 y1-7 (7)": box(0, 0, 1, 7), "box x3-3 y5-5 +x3 y6 (2)": box(3, 3, 5, 6),
 # rectangles containing chip 0
 "box x0-0 y0-1 (2)": box(0, 0, 0, 1), "box x0-3 y0-6 (28)": box(0, 3, 0, 6), "box x0-2 y0-7 (24)": box(0, 2, 0, 7),
 # not rectangles, no chip 0
 "L-shape 9,10,13 (3)": [9, 10, 13], "gap pair (8,10)": [8, 10], "gap pair (4,12)": [4, 12],
 "two boxes R1+R3 (16)": box(0, 3, 2, 3) + box(0, 3, 6, 7), "all but 0 (31)": list(range(1, 32)),
 "pair (5,30)": [5, 30],
}
out = {}
for k, ids in cases.items():
    out[k] = {"ids": ids, "result": tp(ids)}; print(f"{k:28} {out[k]['result']}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
