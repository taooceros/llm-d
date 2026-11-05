#!/usr/bin/env python3
"""Deviceless host-granular compile matrix: every set of whole hosts (4-host topologies), sizes 1/2/4/all (8-host).

    JAX_PLATFORMS=cpu python scripts/vproc_host_sets.py OUT.json v5e:4x4 v6e:4x4 v5e:4x8 v6e:4x8
"""
import itertools, json, sys
import numpy as np, jax, jax.numpy as jnp
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def tp(devs):
    n = len(devs); mesh = Mesh(np.array(devs), ("model",))
    x = jax.ShapeDtypeStruct((8, 1024), jnp.float32, sharding=NamedSharding(mesh, P()))
    w = jax.ShapeDtypeStruct((1024, 256 * n), jnp.bfloat16, sharding=NamedSharding(mesh, P(None, "model")))
    v = jax.ShapeDtypeStruct((256 * n, 1024), jnp.bfloat16, sharding=NamedSharding(mesh, P("model", None)))
    f = lambda x, w, v: (jnp.maximum(x.astype(jnp.bfloat16) @ w, 0) @ v).astype(jnp.float32)
    try:
        jax.jit(f, out_shardings=x.sharding).lower(x, w, v).compile(); return "ok"
    except Exception as e:
        s = str(e)
        return "all-slice" if "limited ICI routing" in s else "net_util" if "net_util" in s else "other:" + s[:80].replace("\n", " ")

out = {}
for name in sys.argv[2:]:
    t = topologies.get_topology_desc(topology_name=name, platform="tpu")
    ds = sorted(t.devices, key=lambda d: d.id)
    hosts = sorted({d.process_index for d in ds})
    by_host = {h: [d for d in ds if d.process_index == h] for h in hosts}
    host_xy = {h: (min(d.coords[0] for d in by_host[h]) // 2, min(d.coords[1] for d in by_host[h]) // 2) for h in hosts}
    rows = []
    for k in range(1, len(hosts) + 1):
        for hs in itertools.combinations(hosts, k):
            devs = sorted((d for h in hs for d in by_host[h]), key=lambda d: d.id)
            if len(hosts) > 4 and k not in (1, 2, 4, len(hosts)):
                continue
            rows.append({"hosts": list(hs), "host_xy": [host_xy[h] for h in hs], "ids": [d.id for d in devs],
                         "result": tp(devs)})
    out[name] = {"hosts": {h: {"xy": host_xy[h], "ids": [d.id for d in by_host[h]]} for h in hosts}, "sets": rows}
    print(f"== {name}  hosts(xy): {[(h, host_xy[h]) for h in hosts]}", flush=True)
    for r in rows:
        print(f"   hosts {str(r['hosts']):26} {r['result']}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
