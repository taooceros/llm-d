#!/usr/bin/env python3
"""Deviceless (compile-only) libtpu probe for a full-mesh 32-chip runtime. No TPU needed.

Question: if one JAX runtime spans the whole v5e 4x8 slice (8 hosts, 32 chips),
which component submeshes can compile ICI collectives, and do the in-HBM
reshard programs between the H / S / M layouts compile?  libtpu 0.0.43 failed
on origin-anchored, non-dense 16-chip sub-boxes (U1 attempt 1, net_util.cc:2304),
so every box the layouts need is compiled here, for both TP8 packings:

* row packing (host rows, 4x2 chips), the only packing U1/U2 could use;
* column packing (host columns, 2x4 chips), today's ``resolve_layout``.

Two views of the same 32 chips:

* ``--view slice``: one v5e 4x8 slice (what a plain 8-process runtime sees);
* ``--view multislice``: two v5e 4x4 slices (one per 4-host block) joined as a
  multislice runtime, i.e. per-block ICI plus DCN between blocks.

Devices are addressed by grid position x + 4y on the 4x8 grid (block B = y 4-7),
which equals the device id in the slice view.

Needs a venv with the worker stack (jax/jaxlib 0.10.2 + libtpu 0.0.43):
    JAX_PLATFORMS=cpu python scripts/full_mesh_deviceless_probe.py --view slice --output OUT.json
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

VIEWS = {"slice": ("v5e:4x8", {}), "multislice": ("v5e:4x4", {"num_slices": 2})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true",
                        help="also compile the U2 model-size TP8/TP16 programs (timing only)")
    parser.add_argument("--view", choices=sorted(VIEWS), default="slice")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import importlib.metadata
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax._src import dispatch
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from hetero.unified_runtime import UnifiedConfig, forward_fn

    topology_name, topology_kwargs = VIEWS[args.view]
    topology = topologies.get_topology_desc(topology_name=topology_name, platform="tpu", **topology_kwargs)

    def slice_of(d):
        return getattr(d, "slice_index", 0) or 0

    def grid(d):  # (x, y) on the 4x8 grid; a 4x4 slice k covers y in [4k, 4k + 4)
        return d.coords[0], d.coords[1] + 4 * slice_of(d)

    devices = sorted(topology.devices, key=lambda d: grid(d)[::-1])
    if len(devices) != 32 or [grid(d) for d in devices] != [(x, y) for y in range(8) for x in range(4)]:
        raise SystemExit(f"{args.view}: unexpected device grid {[grid(d) for d in devices]}")
    position = {d.id: i for i, d in enumerate(devices)}
    config = UnifiedConfig()
    results = {"versions": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")},
               "view": args.view, "topology": topology_name, "topology_kwargs": topology_kwargs,
               "devices": [{"position": i, "id": d.id, "slice": slice_of(d), "coords": list(d.coords),
                            "process_index": d.process_index} for i, d in enumerate(devices)],
               "components": [], "reshards": []}

    def box(xs, ys):
        return [d for d in devices if grid(d)[0] in xs and grid(d)[1] in ys]

    def interleave(lower, upper):
        return [d for pair in zip(lower, upper) for d in pair]

    rows = {f"R{k}": box(range(4), (2 * k, 2 * k + 1)) for k in range(4)}
    cols = {f"{blk}-{half}": box(xs, ys)
            for blk, ys in (("A", range(0, 4)), ("B", range(4, 8)))
            for half, xs in (("lo", (0, 1)), ("hi", (2, 3)))}
    tp16 = {"A rows0-3": box(range(4), range(0, 4)), "B rows4-7": box(range(4), range(4, 8)),
            "mid rows2-5": box(range(4), range(2, 6)),
            "col x0-1": box((0, 1), range(8)), "col x2-3": box((2, 3), range(8))}
    hosts = {f"host{p}": [d for d in devices if d.process_index == p] for p in range(8)}

    def specs(mesh, shape):
        x = NamedSharding(mesh, P())
        wu = NamedSharding(mesh, P(None, None, "model"))
        wd = NamedSharding(mesh, P(None, "model", None))
        up, down = shape.weight_shapes()
        return (x, wu, wd), (jax.ShapeDtypeStruct((shape.batch, shape.d_model), jnp.float32, sharding=x),
                             jax.ShapeDtypeStruct(up, jnp.bfloat16, sharding=wu),
                             jax.ShapeDtypeStruct(down, jnp.bfloat16, sharding=wd))

    def record(kind, label, devs, thunk, **extra):
        start = time.monotonic()
        entry = {"label": label, "positions": [position[d.id] for d in devs],
                 "device_ids": [d.id for d in devs], "slices": sorted({slice_of(d) for d in devs}),
                 "processes": sorted({d.process_index for d in devs}), **extra}
        local = {(slice_of(d), tuple(d.coords[:2])) for d in devs}
        entry["contains_origin"] = 0 in entry["positions"]
        entry["contains_slice_origin"] = any(xy == (0, 0) for _, xy in local)
        try:
            thunk()
            entry["ok"] = True
        except Exception as exc:  # compile errors surface as Python exceptions
            message = str(exc)
            at = message.find("RET_CHECK")
            entry.update(ok=False, error=(message[at:at + 300] if at >= 0 else message[:300]))
        entry["compile_s"] = time.monotonic() - start
        results[kind].append(entry)
        status = "ok" if entry["ok"] else "FAIL " + entry["error"][:100]
        print(f"{kind[:9]:9} {label:44} {entry['compile_s']:6.2f} s {status}", flush=True)

    def forward(devs, shape):
        mesh = Mesh(np.array(devs), ("model",))
        shardings, avals = specs(mesh, shape)
        return lambda: jax.jit(forward_fn(shape.layers), in_shardings=shardings,
                               out_shardings=shardings[0]).lower(*avals).compile()

    small = config.small()
    groups = [("TP8 row", rows), ("TP8 col", cols), ("TP16", tp16), ("TP4", hosts)]
    for kind, members in groups:
        for name, devs in members.items():
            record("components", f"{kind} {name}", devs, forward(devs, small))
    record("components", "TP32 all", devices, forward(devices, small))
    # Today's resolve_layout puts the TP8 on host columns; the U1 failure was
    # order-independent on the 4x4, re-check two orders on the 4x8 origin column.
    a_lo = cols["A-lo"]
    column_major = sorted(a_lo, key=grid)
    record("components", "TP8 col A-lo column-major", column_major, forward(column_major, small))
    # Controls: is it the collective, or any multi-chip assignment containing chip 0?
    by_position = dict(enumerate(devices))

    def elementwise(devs):
        mesh = Mesh(np.array(devs), ("model",))
        aval = jax.ShapeDtypeStruct((len(devs) * 128, 1024), jnp.float32,
                                    sharding=NamedSharding(mesh, P("model")))
        return lambda: jax.jit(lambda a: a * 2 + 1, out_shardings=aval.sharding).lower(aval).compile()

    for name in ("R0", "R1"):
        record("components", f"TP8 row {name} no-collective", rows[name], elementwise(rows[name]))
    for ids in ((0,), (8,), (0, 1), (0, 4), (4, 5), (8, 9), (16, 17), (16, 20)):
        devs = [by_position[i] for i in ids]
        thunk = elementwise(devs) if len(ids) == 1 else forward(devs, small)
        record("components", f"pos {ids} {'no-collective' if len(ids) == 1 else 'TP' + str(len(ids))}",
               devs, thunk)
    # Order controls: does it matter where chip 0 sits in the device assignment?
    orders = {"R0 rotate1": list(range(1, 8)) + [0], "R0 reversed": list(range(7, -1, -1)),
              "R0 chip0 mid": [1, 2, 3, 0, 4, 5, 6, 7], "pair (1, 0)": [1, 0], "pair (4, 0)": [4, 0],
              "TP16 A reversed": list(range(15, -1, -1)), "col A-lo rotate1": [1, 4, 5, 8, 9, 12, 13, 0]}
    for name, ids in orders.items():
        devs = [by_position[i] for i in ids]
        record("components", f"order {name}", devs, forward(devs, small))
    if args.model:
        model = config.model()
        for name in ("R0", "R3"):
            record("components", f"TP8 row {name} model", rows[name], forward(rows[name], model))
        for name in ("A rows0-3", "B rows4-7"):
            target = interleave(*(rows[f"R{k}"] for k in ((0, 1) if name.startswith("A") else (2, 3))))
            record("components", f"TP16 {name} model local", target, forward(target, model))

    def reshard(label, source_mesh, target_mesh, shape, direction):
        (_, pu, pd), (_, su, sd) = specs(source_mesh, shape)
        (_, tu, td), (_, stu, std) = specs(target_mesh, shape)
        devs = sorted(set(source_mesh.devices.flat), key=lambda d: d.id)
        pairs = ((su, tu), (sd, td)) if direction == "forward" else ((stu, pu), (std, pd))

        def thunk():
            for source, sharding in pairs:
                jax.jit(dispatch._device_put_reshard, out_shardings=sharding).lower(source).compile()
        record("reshards", label, devs, thunk, direction=direction)

    # Block-local 2xTP8 <-> TP16, as U2: the two TP8 replicas form one (8, 2)
    # array, the TP16 target is one 16-device mesh over the same chips.
    block_pairs = {"A rows": (rows["R0"], rows["R1"]), "B rows": (rows["R2"], rows["R3"]),
                   "mid rows": (rows["R1"], rows["R2"]),
                   "A cols": (cols["A-lo"], cols["A-hi"]), "B cols": (cols["B-lo"], cols["B-hi"])}
    for name, (lower, upper) in block_pairs.items():
        pair = Mesh(np.array(list(zip(lower, upper))), ("model", "replica"))
        # jit only accepts one device order for source and target, so the reshard
        # is compiled in the local order; the natural (id) order TP16 is compiled
        # as a component (jax.device_put compiles its own reorder at switch time).
        target = interleave(lower, upper)
        tmesh = Mesh(np.array(target), ("model",))
        for direction in ("forward", "reverse"):
            arrow = "2xTP8->TP16" if direction == "forward" else "TP16->2xTP8"
            reshard(f"{name} {arrow} local", pair, tmesh, small, direction)
        natural = sorted(lower + upper, key=lambda d: d.id)
        record("components", f"TP16 {name} natural order", natural, forward(natural, small))
    # Slice-wide S <-> H in one 32-chip program (4 TP8 replicas <-> 2 TP16 replicas).
    # Replica order R0, R2, R1, R3 makes the flattened device order equal H's
    # (hA[2k], hB[2k], hA[2k+1], hB[2k+1]) = (R0[k], R2[k], R1[k], R3[k]).
    s_mesh = Mesh(np.array(list(zip(*(rows[f"R{k}"] for k in (0, 2, 1, 3))))), ("model", "replica"))
    h_mesh = Mesh(np.array(list(zip(interleave(rows["R0"], rows["R1"]), interleave(rows["R2"], rows["R3"])))),
                  ("model", "replica"))
    reshard("slice S->H one program local", s_mesh, h_mesh, small, "forward")
    reshard("slice H->S one program local", s_mesh, h_mesh, small, "reverse")
    # H -> TP32 (full-mesh only), and back.
    t32 = Mesh(np.array(interleave(interleave(rows["R0"], rows["R1"]), interleave(rows["R2"], rows["R3"]))),
               ("model",))
    reshard("slice H->TP32 local", h_mesh, t32, small, "forward")
    reshard("slice TP32->H local", h_mesh, t32, small, "reverse")
    # Lockstep alternative: each layout is ONE slice-wide SPMD program, the TP
    # replicas are a data axis (S = DP4 x TP8, H = DP2 x TP16), so every program
    # spans all 32 chips and every process launches every step.

    def lockstep(mesh, shape):
        x = NamedSharding(mesh, P("replica", None))
        wu = NamedSharding(mesh, P(None, None, "model"))
        wd = NamedSharding(mesh, P(None, "model", None))
        up, down = shape.weight_shapes()
        avals = (jax.ShapeDtypeStruct((shape.batch, shape.d_model), jnp.float32, sharding=x),
                 jax.ShapeDtypeStruct(up, jnp.bfloat16, sharding=wu),
                 jax.ShapeDtypeStruct(down, jnp.bfloat16, sharding=wd))
        return lambda: jax.jit(forward_fn(shape.layers), in_shardings=(x, wu, wd),
                               out_shardings=x).lower(*avals).compile()

    for name, mesh in (("S DP4xTP8", s_mesh), ("H DP2xTP16", h_mesh)):
        record("components", f"lockstep {name}", devices, lockstep(mesh, small))
        if args.model:
            record("components", f"lockstep {name} model", devices, lockstep(mesh, config.model()))

    failed = [e["label"] for kind in ("components", "reshards") for e in results[kind] if not e["ok"]]
    results["libtpu_init_args"] = os.environ.get("LIBTPU_INIT_ARGS", "")
    results["summary"] = {"programs": sum(len(results[k]) for k in ("components", "reshards")),
                          "failed": failed}
    print(json.dumps(results["summary"], indent=1))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
