#!/usr/bin/env python3
"""Deviceless compile of the reshard metadata gather (``hetero/vproc/resident.py:_union_gather``).

Compares the two meshes the gather has used on a v5e 4x4 block: one chip per participating host
(the version that aborted host kernel 2 in G2 run vproc-G2-23efaf8 with net_util.cc:2304) and
every chip of the participating hosts (the fix). Each case compiles in its own subprocess, since
libtpu may abort the process instead of raising. No TPU needed:

    JAX_PLATFORMS=cpu PY=<venv with jax 0.10.2 + libtpu 0.0.43> \
        $PY scripts/vproc_union_gather_probe.py results/vproc/deviceless-union-gather-libtpu0.0.43.json
"""
import json
import subprocess
import sys

TOPOLOGY = "v5e:4x4"
ROW_BYTES = 1 << 17  # resident.EXCHANGE_BYTES
HOST_SETS = [(0, 1, 2, 3), (0, 1), (2, 3), (0, 3), (1, 2)]


def compile_case(mesh_kind: str, hosts: tuple) -> dict:
    import numpy as np
    import jax
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    topology = topologies.get_topology_desc(topology_name=TOPOLOGY, platform="tpu")
    physical = sorted(topology.devices, key=lambda d: (d.process_index, d.id))
    if mesh_kind == "one_chip_per_host":
        devices = [min((d for d in physical if d.process_index == h), key=lambda d: d.id) for h in hosts]
    else:
        devices = [d for d in physical if d.process_index in hosts]
    mesh = Mesh(np.array(devices, dtype=object).reshape(1, -1), ("replica", "hosts"))
    arg = jax.ShapeDtypeStruct((len(devices), ROW_BYTES), np.uint8, sharding=NamedSharding(mesh, P("hosts")))
    row = {"mesh": mesh_kind, "hosts": list(hosts), "device_ids": [d.id for d in devices]}
    try:
        jax.jit(lambda x: x, out_shardings=NamedSharding(mesh, P())).lower(arg).compile()
        row["result"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["result"] = "error"
        row["error"] = str(exc).splitlines()[0][:400]
    return row


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--case":
        print(json.dumps(compile_case(sys.argv[2], tuple(json.loads(sys.argv[3])))))
        return
    out_path = sys.argv[1]
    rows = []
    for mesh_kind in ("one_chip_per_host", "whole_hosts"):
        for hosts in HOST_SETS:
            proc = subprocess.run([sys.executable, __file__, "--case", mesh_kind, json.dumps(hosts)],
                                  capture_output=True, text=True, timeout=600)
            line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith("{")), None)
            if line is not None:
                row = json.loads(line)
            else:
                lines = proc.stderr.strip().splitlines()
                fatal = [l for l in lines if "Check failed" in l or "RET_CHECK" in l]
                row = {"mesh": mesh_kind, "hosts": list(hosts), "result": f"process exit {proc.returncode}",
                       "error": (fatal[0] if fatal else (lines[-1] if lines else ""))[:400]}
            rows.append(row)
            print(f"{mesh_kind:18s} hosts={hosts} ids={row.get('device_ids')} -> {row['result']} "
                  f"{row.get('error', '')[:160]}", flush=True)
    import importlib.metadata as md
    with open(out_path, "w") as handle:
        json.dump({"topology": TOPOLOGY, "row_bytes": ROW_BYTES,
                   "versions": {p: md.version(p) for p in ("jax", "jaxlib", "libtpu")}, "cases": rows},
                  handle, indent=1)


if __name__ == "__main__":
    main()
