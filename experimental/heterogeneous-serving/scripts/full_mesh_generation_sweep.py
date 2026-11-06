#!/usr/bin/env python3
"""Deviceless sweep: does the limited-ICI-routing subset rule hold on other TPU generations? No TPU needed.

For each topology, compiles the U-series TP MLP (all-reduce) over chip subsets
of every shape that distinguished the rule on v5e 4x8: chip-0 boxes, non-origin
boxes, non-rectangles, scattered sets, all-but-chip-0, and the full slice.
Subsets are chosen in chips; on multi-core-per-chip parts (TPU7x) every core of
a chosen chip is included.

    JAX_PLATFORMS=cpu python scripts/full_mesh_generation_sweep.py OUT.json v5e:4x8 v6e:8x8 tpu7x:2x2x4 ...
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def main():
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    out_path, names = sys.argv[1], sys.argv[2:]
    results = {"versions": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")},
               "topologies": {}}

    def tp(devs):
        n = len(devs)
        mesh = Mesh(np.array(devs), ("model",))
        x = jax.ShapeDtypeStruct((8, 1024), jnp.float32, sharding=NamedSharding(mesh, P()))
        w = jax.ShapeDtypeStruct((1024, 256 * n), jnp.bfloat16, sharding=NamedSharding(mesh, P(None, "model")))
        v = jax.ShapeDtypeStruct((256 * n, 1024), jnp.bfloat16, sharding=NamedSharding(mesh, P("model", None)))
        f = lambda x, w, v: (jnp.maximum(x.astype(jnp.bfloat16) @ w, 0) @ v).astype(jnp.float32)
        try:
            jax.jit(f, out_shardings=x.sharding).lower(x, w, v).compile()
            return "ok"
        except Exception as exc:
            s = str(exc)
            if "limited ICI routing" in s:
                return "all-slice-check"
            at = s.find("RET_CHECK")
            return "FAIL " + (s[at:at + 140] if at >= 0 else s[:140]).replace("\n", " ")

    for name in names:
        topology = topologies.get_topology_desc(topology_name=name, platform="tpu")
        devices = sorted(topology.devices, key=lambda d: d.id)
        chips = {}
        for d in devices:
            chips.setdefault(tuple(d.coords), []).append(d)
        dims = [max(c[i] for c in chips) + 1 for i in range(3)]
        axis = max(range(3), key=lambda i: dims[i])           # longest axis
        far = tuple(n - 1 for n in dims)                       # chip farthest from the origin

        def shift(c, i, k):
            c = list(c); c[i] += k
            return tuple(c)

        def devs(cs):
            return [d for c in cs for d in chips[c]]

        origin = (0, 0, 0)
        x_axis = 0 if dims[0] > 1 else axis
        other = next((i for i in range(3) if i != x_axis and dims[i] > 1), None)
        half = dims[axis] // 2
        cases = {
            "origin pair": [origin, shift(origin, x_axis, 1)],
            "non-origin pair": [far, shift(far, x_axis, -1)],
            "origin half": [c for c in chips if c[axis] < half],
            "non-origin half": [c for c in chips if c[axis] >= half],
            "all but chip 0": [c for c in chips if c != origin],
        }
        if dims[axis] >= 3:
            cases["non-origin gapped pair"] = [far, shift(far, axis, -2)]
        if other is not None:
            cases["non-origin L-shape"] = [far, shift(far, x_axis, -1), shift(far, other, -1)]
            cases["origin 2x2"] = [c for c in chips if c[x_axis] < 2 and c[other] < 2 and
                                   all(c[i] == 0 for i in range(3) if i not in (x_axis, other))]
        if len(chips) >= 16:
            cases["scattered 8 (no chip 0)"] = sorted(c for c in chips if c != origin)[3::max(1, (len(chips) - 1) // 8)][:8]
        cases["full"] = list(chips)
        if len(chips[origin]) > 1:                               # multi-core chips: one chip, both cores
            cases["chip 0 only (its cores)"] = [origin]
        row = {"devices": len(devices), "chips": len(chips), "dims": dims,
               "cores_per_chip": len(chips[origin]), "cases": {}}
        for label, cs in cases.items():
            row["cases"][label] = {"chips": [list(c) for c in cs], "result": tp(devs(cs))}
        results["topologies"][name] = row
        summary = "  ".join(f"{k}={'ok' if v['result'] == 'ok' else ('ALL' if v['result'] == 'all-slice-check' else 'FAIL')}"
                            for k, v in row["cases"].items())
        print(f"{name:12} chips={len(chips):3} {summary}", flush=True)
        for k, v in row["cases"].items():
            if v["result"] not in ("ok", "all-slice-check"):
                print(f"    {k}: {v['result']}", flush=True)

    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
