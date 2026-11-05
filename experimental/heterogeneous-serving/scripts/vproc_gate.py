#!/usr/bin/env python3
"""G1/G2 cluster gates for the virtual-process block runtime (results/vproc/plan.md). RCM-only.

G1: host kernels on block A (4 hosts x 4 chips, one JAX runtime) running real vLLM +
    tpu-inference engines under views (C2), no tpu-inference edits. S-block (TP8 on
    hosts {0,1} and {2,3}) must be token-equal to the S reference; a cold switch to the
    H-block (TP16 on all four hosts) must be token-equal to the H reference.
G2: S-block -> H-block -> S-block with resident weights (C4) through the controller
    (C5) and in-memory endpoints (C6), ``--reps`` cycles, token checks after every
    switch, C segments against InPlaceEngine (100.6 s warm) and O3 (27.5 s).

    rcm_exec run scripts/vproc_gate.py -- --gate G1 --inventory /tmp/hetero-inventory-only.json
    python scripts/vproc_gate.py --local --gate G2     # dry run: same body, local Ray, CPU block, synthetic engines

On the cluster a failure never rebuilds the block: it stops the gate and becomes evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLING = {"temperature": 0, "max_tokens": 64, "ignore_eos": False}  # ShardedRestore references
LOCAL_REQUEST = {"prompt": [[1, 2, 3], [4, 5], [9]], "steps": 6}
KERNEL_PREFIX = "vproc-kernel-e0"


def refuse_respawn():
    raise RuntimeError("gates never rebuild the block; the failure is the evidence")


def serve_layout(ctl, request):
    """Serve ``request`` on every engine of the current layout concurrently; engine_id -> result."""
    from concurrent.futures import ThreadPoolExecutor
    ids = sorted(ctl.engines)
    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        results = list(pool.map(lambda engine_id: ctl.serve(engine_id, request, retry=False), ids))
    return dict(zip(ids, results))


def check_phase(ctl, evidence, label, request, reference_for):
    started = time.time()
    served = serve_layout(ctl, request)
    rows = []
    for engine_id, result in served.items():
        spec = ctl.engines[engine_id]
        rows.append({"phase": label, **ctl.measure.check_tokens(f"{label}/{engine_id}", result["tokens"],
                                                                  reference_for(spec)),
                     "engine_id": engine_id, "tp": spec.tp_size, "hosts": list(spec.hosts),
                     "generation": spec.generation, "serve_s": result["serve"]["elapsed_s"]})
    evidence["checks"].extend(rows)
    evidence["phases"].append({"label": label, "layout": ctl.layout_name, "serve_wall_s": time.time() - started,
                               "runtime_ids": ctl.runtime_ids()})
    return all(row["equal"] for row in rows)


def run_gate(gate, reps, ctl, request, reference_for, evidence):
    runtime = set(ctl.runtime_ids())
    evidence["deploy"] = ctl.deploy("S-block")
    check_phase(ctl, evidence, "deploy_S", request, reference_for)
    if gate == "G1":
        evidence["switches"].append(ctl.switch("H-block", resident=False))
        check_phase(ctl, evidence, "cold_H", request, reference_for)
    else:
        for rep in range(reps):
            for target, label in (("H-block", "H"), ("S-block", "S")):
                evidence["switches"].append(ctl.switch(target, resident=True))
                check_phase(ctl, evidence, f"rep{rep}_{label}", request, reference_for)
    evidence["runtime_unchanged"] = set(ctl.runtime_ids()) == runtime and len(runtime) == 1


def summarize(evidence):
    from hetero.vproc import measure
    rows = []
    for report in evidence["switches"]:
        total = report.get("total_s")
        rows.append({"target": report.get("target"), "mode": report.get("mode"), "ok": report.get("ok"),
                     "total_s": total, "unaccounted_s": report.get("unaccounted_s"),
                     "segments": report.get("segments"), "compiles": report.get("compiles"),
                     "vs": measure.compare(total) if isinstance(total, (int, float)) else None})
    evidence["summary"] = {"switches": rows,
                           "tokens_equal": bool(evidence["checks"]) and all(c["equal"] for c in evidence["checks"]),
                           "runtime_unchanged": evidence.get("runtime_unchanged")}
    evidence["ok"] = (evidence["failure"] is None and evidence["summary"]["tokens_equal"]
                      and bool(evidence.get("runtime_unchanged")) and all(row["ok"] for row in rows))


def new_evidence(args, platform):
    return {"ok": False, "gate": args.gate, "reps": args.reps, "run_id": args.run_id, "platform": platform,
            "deploy": None, "switches": [], "checks": [], "phases": [], "failure": None, "cleanup": None}


def local(args):
    """Dry run of the remote body: local Ray, 4 x 4 CPU block, synthetic engines, oracle references."""
    import ray
    from hetero.vproc.controller import BlockController
    from hetero.vproc.endpoints import InMemoryEndpoints
    from hetero.vproc.host_kernel import kill_block, spawn_block
    from hetero.vproc.synthetic_engine import expected_tokens
    evidence = new_evidence(args, "cpu")
    ray.init(address="local", num_cpus=16, include_dashboard=False, logging_level="ERROR")
    actors, ctl = [], None
    try:
        actors = spawn_block(4, 4, name_prefix=KERNEL_PREFIX, actor_options={"namespace": "omp_hetero"},
                             preinit=None)
        model = {"seed": 7}
        layout_models = {"H-block": {"device_order": args.local_h_order}}
        ctl = BlockController(actors, InMemoryEndpoints(), respawn=refuse_respawn, engine_kind="synthetic",
                              engine_model=model, layout_models=layout_models, checksums=True)
        run_gate(args.gate, args.reps, ctl, LOCAL_REQUEST,
                 lambda spec: expected_tokens(spec, LOCAL_REQUEST), evidence)
    except Exception:
        evidence["failure"] = traceback.format_exc()
    finally:
        if ctl is not None:
            evidence["report"] = ctl.report()
            ctl.close()
        else:
            kill_block(actors)
        ray.shutdown()
    summarize(evidence)
    write(args.output, evidence)
    print(json.dumps(evidence["summary"], indent=1, default=str))
    if not evidence["ok"]:
        raise SystemExit(1)


def remote(args):
    """Runs as the Ray job on the cluster head: block A of the H layout, real engines."""
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from hetero.evidence import emit_measurements
    from hetero.layout import builtin_layout, resolve_layout
    from hetero.native_backend import prepare_backend
    from hetero.slice_inventory import SliceInventory
    from hetero.vproc.controller import BlockController
    from hetero.vproc.endpoints import InMemoryEndpoints
    from hetero.vproc.host_kernel import kill_block, spawn_block
    ray.init(address="auto", ignore_reinit_error=True)
    evidence = new_evidence(args, "tpu")
    actors, ctl, pg = [], None, None
    try:
        payload = json.loads((ROOT / os.environ["VPROC_PAYLOAD"]).read_text())
        if payload["sampling"] != SAMPLING:
            raise RuntimeError(f"reference sampling {payload['sampling']} != {SAMPLING}")
        inventory = SliceInventory.from_dict(payload["inventory"])
        box = resolve_layout(inventory, builtin_layout("H")).instances[0]
        evidence["box"] = box.to_dict()
        if any(row["state"] != "REMOVED" for row in ray.util.placement_group_table().values()):
            raise RuntimeError("refusing: placement groups owned by another experiment exist")
        evidence["backend"] = prepare_backend(box.host_ips, apply=False)
        if not evidence["backend"]["ready"]:
            raise RuntimeError("native backend must already be prepared; the gate does not mutate packages")
        pg = placement_group([{"TPU": 4, f"node:{host.node_ip}": 0.001} for host in box.hosts],
                             strategy="STRICT_SPREAD", name=f"vproc-gate-{args.run_id}")
        ray.get(pg.ready(), timeout=120)
        cache = f"/models/jax_cache/vproc-{args.run_id}"  # fresh per run: rep0 S->H cold, later switches warm
        rank_env = [{**box.process_env(rank), "JAX_COMPILATION_CACHE_DIR": cache, "VLLM_XLA_CACHE_PATH": cache,
                     "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1"} for rank in range(len(box.hosts))]
        actor_options = [{"namespace": "omp_hetero", "num_cpus": 0, "resources": {"TPU": 4}, "max_restarts": 0,
                          "scheduling_strategy": PlacementGroupSchedulingStrategy(
                              placement_group=pg, placement_group_bundle_index=rank,
                              placement_group_capture_child_tasks=True)}
                         for rank in range(len(box.hosts))]
        actors = spawn_block(len(box.hosts), 4, platform="tpu", rank_env=rank_env, actor_options=actor_options,
                             name_prefix=KERNEL_PREFIX, init_timeout_s=600)
        ctl = BlockController(actors, InMemoryEndpoints(), respawn=refuse_respawn, engine_kind="vllm",
                              engine_model={}, checksums=args.checksums)
        request = {"prompt": payload["prompts"], "steps": SAMPLING["max_tokens"],
                   "ignore_eos": SAMPLING["ignore_eos"]}
        references = payload["references"]
        run_gate(args.gate, args.reps, ctl, request, lambda spec: references[str(spec.tp_size)]["outputs"],
                 evidence)
    except Exception:
        evidence["failure"] = traceback.format_exc()
    finally:
        if ctl is not None:
            try:
                evidence["report"] = ctl.report()
            except Exception:
                evidence["report_failure"] = traceback.format_exc()
            ctl.close()
        else:
            kill_block(actors)
        if pg is not None:
            remove_placement_group(pg)
        evidence["cleanup"] = {"kernels": len(actors), "placement_group_removed": pg is not None,
                               "time": time.time()}
        summarize(evidence)
        time.sleep(15)
        emit_measurements(evidence)


def build_payload(args):
    manifest = json.loads((ROOT / "data/manifests/W0.json").read_text())
    references = {}
    for tp, layout in ((16, "H"), (8, "S")):
        path = ROOT / f"results/restore/ShardedRestore/{layout}-reference.json"
        reference = json.loads(path.read_text())["measurements"]
        references[str(tp)] = {"file": str(path.relative_to(ROOT)), "prompts": reference["prompts"],
                               "sampling": reference["sampling"],
                               "outputs": next(iter(reference["outputs"].values()))}
    prompts = [row["prompt_token_ids"] for row in manifest["requests"][:32]]
    for tp, reference in references.items():
        if reference["prompts"] != prompts or reference["sampling"] != SAMPLING:
            raise SystemExit(f"TP{tp} reference prompts/sampling differ from the gate contract")
    return {"inventory": json.loads(args.inventory.read_text()), "prompts": prompts,
            "references": references, "sampling": SAMPLING}


def submit(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    payload = build_payload(args)
    payload_path = ROOT / f"vproc-input-{args.run_id}.json"
    argv = ["python3", "-u", "scripts/vproc_gate.py", "--remote", "--gate", args.gate, "--reps", str(args.reps),
            "--run-id", args.run_id] + (["--checksums"] if args.checksums else [])
    env = {"PYTHONUNBUFFERED": "1", "VPROC_PAYLOAD": payload_path.name}
    if args.dry_run:
        print(json.dumps({"entrypoint": " ".join(argv), "env": env, "prompts": len(payload["prompts"]),
                          "references": {k: v["file"] for k, v in payload["references"].items()}}, indent=1))
        return

    @stage(name=f"vproc_{args.gate}", order_idx=1, expected_s=args.timeout)
    def run(state, timing):
        supervisor = state.get("supervisor")
        if supervisor is None:
            supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
            state.set("supervisor", supervisor)
        job = state.get("submission_id")
        if job is None:
            payload_path.write_text(json.dumps(payload))
            try:
                job = supervisor.submit_job(entrypoint=" ".join(argv), runtime_env={
                    "working_dir": str(ROOT), "env_vars": env,
                    "excludes": [".venv*", ".git", ".jj", "__pycache__", "*.pyc", "results", "**/results/**",
                                 "reports", "data"]},
                    metadata={"purpose": f"vproc-{args.gate}"}, name=f"vproc-{args.gate}-{args.run_id}")
            finally:
                payload_path.unlink(missing_ok=True)
            state.set("submission_id", job)
        result = supervisor.supervise_until_completion(submission_id=job, timeout_s=args.timeout,
                                                       stage_name=f"vproc_{args.gate}", state_store=state)
        measured = result["measurements"]
        write(args.output, {"submission_id": job, "status": result["status"], "measurements": measured})
        # A failed gate is evidence, not an RCM trap holding the block.
        if not measured:
            raise RuntimeError("gate returned no machine-readable evidence")

    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(run)
    if not runner.run():
        raise SystemExit(1)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=str) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate", choices=("G1", "G2"), required=True)
    parser.add_argument("--reps", type=int, default=3, help="G2 S->H->S cycles")
    parser.add_argument("--checksums", action="store_true", help="verify logical weight checksums in every reshard")
    parser.add_argument("--remote", action="store_true", help="cluster body (Ray job entrypoint)")
    parser.add_argument("--local", action="store_true", help="dry run on local Ray with a CPU block")
    parser.add_argument("--local-h-order", default="halves_interleaved", choices=("natural", "halves_interleaved"))
    parser.add_argument("--dry-run", action="store_true", help="build the payload, print the job, submit nothing")
    parser.add_argument("--inventory", type=Path, default=Path("/tmp/hetero-inventory-only.json"))
    parser.add_argument("--ray-url", default="http://127.0.0.1:8265")
    parser.add_argument("--run-id", default="vproc-" + str(int(time.time())))
    parser.add_argument("--timeout", type=int, default=5400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / f"results/vproc/{args.gate}-{'local' if args.local else args.run_id}.json"
    if args.remote:
        remote(args)
    elif args.local:
        local(args)
    else:
        submit(args)
