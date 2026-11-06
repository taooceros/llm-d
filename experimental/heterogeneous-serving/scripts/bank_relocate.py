#!/usr/bin/env python3
"""Copy one instance's executable bank onto the hosts of a congruent placement (RCM-supervised).

    flock /tmp/hetero-cluster.lock .venv/bin/python rcm_exec run scripts/bank_relocate.py -- \\
        --inventory /tmp/hetero-inventory-only.json --run-scope <scope> \\
        --source H:large_a --target Mb:large_b --output results/.../relocate.json

Both placements are resolved locally from the inventory and must be congruent: same tp size,
chip box and host grid, and every rank at the same position relative to its placement's origin
(``resolve_layout`` orders ranks row-major). Rank r's bank directory
``<scope bank root>/<identity digest>/rank<r>/`` is read on source rank r's host (every entry
sha256-checked against its manifest), moved through the Ray object store, and written on target
rank r's host byte for byte by ``exec_bank.install_rank_bank`` (atomic; an existing directory is
refused; ``relocation.json`` records the source). Only the scope's bank root is touched, so
``snapshot_validate.py cleanup --run-scope`` removes everything this writes.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def placement(inventory, spec):
    """[(rank, node_ip, relative host grid position)], tp, chip dims of LAYOUT:INSTANCE."""
    from hetero.layout import builtin_layout, resolve_layout
    layout_id, _, instance_id = spec.partition(":")
    layout = resolve_layout(inventory, builtin_layout(layout_id))
    instance = next((i for i in layout.instances if i.instance_id == instance_id), None)
    if instance is None:
        raise SystemExit(f"{instance_id} is not an instance of {layout_id}")
    origin = [min(h.host_grid_position[a] for h in instance.hosts) for a in range(3)]
    hosts = [(h.process_rank, h.node_ip,
              tuple(h.host_grid_position[a] - origin[a] for a in range(3))) for h in instance.hosts]
    return sorted(hosts), instance.tp_size, tuple(instance.chip_dims)


def congruent_pairs(inventory, source, target):
    src, src_tp, src_dims = placement(inventory, source)
    dst, dst_tp, dst_dims = placement(inventory, target)
    if (src_tp, src_dims) != (dst_tp, dst_dims):
        raise SystemExit(f"not congruent: {source} tp{src_tp} {src_dims} vs {target} tp{dst_tp} {dst_dims}")
    pairs = []
    for (rank, src_ip, src_pos), (dst_rank, dst_ip, dst_pos) in zip(src, dst):
        if rank != dst_rank or src_pos != dst_pos:
            raise SystemExit(f"rank {rank}: relative positions differ ({src_pos} vs {dst_pos})")
        if src_ip == dst_ip:
            raise SystemExit(f"rank {rank}: source and target are the same host {src_ip}")
        pairs.append({"rank": rank, "source_host": src_ip, "target_host": dst_ip,
                      "relative_host_position": list(src_pos)})
    return pairs


def read_bundle(bank_root, rank, identity_digest):
    from hetero import exec_bank
    root = Path(bank_root)
    if identity_digest is None:
        found = sorted(p.parent.parent.name for p in root.glob(f"*/rank{rank}/{exec_bank.MANIFEST}"))
        if len(found) != 1:
            raise RuntimeError(f"{root}: expected exactly one bank for rank {rank}, found {found}")
        identity_digest = found[0]
    started = time.time()
    bundle = exec_bank.read_rank_bank(str(root), identity_digest, rank)
    bundle["read_s"] = time.time() - started
    return bundle


def install_bundle(bank_root, bundle, provenance):
    from hetero import exec_bank
    report = exec_bank.install_rank_bank(bank_root, bundle, provenance)
    # Read back what landed on this host (same check a loading worker makes).
    exec_bank.read_rank_bank(bank_root, bundle["identity_digest"], bundle["rank"])
    report["verified_after_install"] = True
    return report


def remote(args):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from hetero.evidence import emit_measurements
    from hetero.run_scope import scope_roots
    ray.init(address="auto", ignore_reinit_error=True)
    pairs = json.loads(args.pairs)
    bank_root = scope_roots(args.run_scope)["exec_bank_dir"]
    nodes = {n["NodeManagerAddress"]: n["NodeID"] for n in ray.nodes() if n["Alive"]}

    def on(ip):
        if ip not in nodes:
            raise RuntimeError(f"host {ip} is not a live Ray node")
        return NodeAffinitySchedulingStrategy(nodes[ip], soft=False)

    reader = ray.remote(num_cpus=0)(read_bundle)
    writer = ray.remote(num_cpus=0)(install_bundle)
    started = time.time()
    rows = []
    for pair in pairs:
        bundle = reader.options(scheduling_strategy=on(pair["source_host"])).remote(
            bank_root, pair["rank"], args.identity_digest)
        provenance = {"source_host": pair["source_host"], "source_rank": pair["rank"],
                      "source": args.source, "target": args.target, "run_scope": args.run_scope}
        report = ray.get(writer.options(scheduling_strategy=on(pair["target_host"])).remote(
            bank_root, bundle, provenance), timeout=args.timeout)
        rows.append({**pair, **report})
    emit_measurements({"ok": True, "action": "bank-relocate", "source": args.source, "target": args.target,
                       "run_scope": args.run_scope, "bank_root": bank_root, "ranks": rows,
                       "identity_digests": sorted({Path(r["path"]).parent.name for r in rows}),
                       "bytes": sum(r["bytes"] for r in rows), "wall_s": time.time() - started})


def submit(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    from hetero.slice_inventory import SliceInventory
    data = json.loads(Path(args.inventory).read_text())
    inventory = SliceInventory.from_dict(data.get("measurements", {}).get("inventory", data))
    pairs = congruent_pairs(inventory, args.source, args.target)
    argv = ["python3", "-u", "scripts/bank_relocate.py", "--remote", "--run-scope", args.run_scope,
            "--source", args.source, "--target", args.target, "--pairs", json.dumps(pairs),
            "--timeout", str(args.timeout)]
    if args.identity_digest:
        argv += ["--identity-digest", args.identity_digest]

    @stage(name="bank_relocate", order_idx=1, expected_s=args.timeout)
    def run(state, timing):
        supervisor = state.get("supervisor")
        if supervisor is None:
            supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
            state.set("supervisor", supervisor)
        job = state.get("submission_id")
        if job is None:
            job = supervisor.submit_job(entrypoint=shlex.join(argv), runtime_env={
                "working_dir": str(ROOT), "excludes": [".venv*", ".git", ".jj", "__pycache__",
                "*.pyc", "results", "**/results/**", "reports", "data"],
                "env_vars": {"PYTHONUNBUFFERED": "1"}},
                metadata={"purpose": "bank-relocate"}, name="bank-relocate")
            state.set("submission_id", job)
        result = supervisor.supervise_until_completion(submission_id=job, timeout_s=args.timeout,
                                                       stage_name="bank_relocate", state_store=state)
        measured = result["measurements"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"submission_id": job, "status": result["status"],
                                           "pairs": pairs, "measurements": measured}, indent=2) + "\n")
        if not measured or measured.get("ok") is not True:
            raise RuntimeError("bank relocation missing successful evidence: "
                               + str((measured or {}).get("error")))

    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(run)
    if not runner.run():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-scope", required=True, help="hetero/run_scope.py scope whose bank root is used")
    parser.add_argument("--source", required=True, help="LAYOUT:INSTANCE that wrote the bank, e.g. H:large_a")
    parser.add_argument("--target", required=True, help="LAYOUT:INSTANCE that will read it, e.g. Mb:large_b")
    parser.add_argument("--identity-digest", help="bank to copy when the scope holds several")
    parser.add_argument("--inventory", help="slice inventory JSON (hetero_ctl inventory output or its inventory object)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--pairs", help=argparse.SUPPRESS)
    parser.add_argument("--ray-url", default="http://127.0.0.1:8265")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    from hetero.run_scope import check_scope
    check_scope(args.run_scope)
    if args.remote:
        remote(args)
    else:
        if not args.inventory or args.output is None:
            parser.error("--inventory and --output are required for supervised submission")
        submit(args)


if __name__ == "__main__":
    main()
