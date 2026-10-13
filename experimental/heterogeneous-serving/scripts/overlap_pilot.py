#!/usr/bin/env python3
"""Run the existing blind W0 pilot with opt-in, compile-only CPU background work."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def _input_provenance():
    from hetero.manifest import load_manifest
    from hetero.workload import load_catalogue
    catalogue = load_catalogue("large")
    expected = os.environ.get("OMP_OVERLAP_CATALOGUE_SHA256")
    if expected is not None and catalogue["sha256"] != expected:
        raise ValueError("staged catalogue differs from the predeclared pilot input hash")
    source_ids = {row["id"] for row in catalogue["records"]}
    manifests = {}
    for name in ("W0", "W1", "W2", "E_SHORT", "E_LONG"):
        manifest = load_manifest(name)
        requests = manifest["requests"]
        missing = sorted({row["source_id"] for row in requests} - source_ids)
        if missing:
            raise ValueError(f"{name} source IDs absent from staged catalogue: {missing[:5]}")
        if (any(len(row["prompt_token_ids"]) != row["prompt_tokens"] or row["max_tokens"] <= 0 for row in requests)
                or sum(row["prompt_tokens"] for row in requests) != manifest["prompt_tokens_total"]
                or sum(row["max_tokens"] for row in requests) != manifest["useful_output_tokens"]):
            raise ValueError(f"{name} token payloads differ from the manifest budget contract")
        manifests[name] = {"manifest_sha256": manifest["manifest_sha256"],
                           "catalogue_sha256": manifest["catalogue_sha256"],
                           "num_requests": manifest["num_requests"],
                           "prompt_tokens_total": manifest["prompt_tokens_total"],
                           "useful_output_tokens": manifest["useful_output_tokens"],
                           "source_ids_resolved": True, "token_budgets_verified": True}
    return {"catalogue_path": catalogue["path"], "catalogue_sha256": catalogue["sha256"],
            "catalogue_bytes": Path(catalogue["path"]).stat().st_size, "manifests": manifests}

def compile_task(bundle_root, cache_dir, topology):
    from scripts.overlap_replay import replay
    import shutil
    import socket
    import tempfile
    started, cpu = time.time(), time.process_time()
    bundles = [p for p in sorted(Path(bundle_root).glob("*/*"))
               if p.is_dir() and any(p.glob("*.json"))]
    # Replicas export identical executable keys. Compile each key once even
    # if a compile-only backend cannot deserialize its own cached executable.
    # This is JAX's actual key identity, not a rewritten key or shape heuristic.
    entries = {}
    source_descriptors = 0
    for bundle in bundles:
        for path in sorted(bundle.glob("*.json")):
            descriptor = json.loads(path.read_text())
            source_descriptors += 1
            key = descriptor["key"]
            if key not in entries:
                entries[key] = (bundle, descriptor, [])
            entries[key][2].append(str(bundle))
    with tempfile.TemporaryDirectory(prefix="omp-overlap-replay-") as temp:
        destination = Path(temp)
        for key, (bundle, descriptor, origins) in entries.items():
            for suffix in (".mlir", ".options"):
                shutil.copy2(bundle / (key + suffix), destination / (key + suffix))
            descriptor["source_bundles"] = origins
            (destination / (key + ".json")).write_text(json.dumps(descriptor))
        result = replay(destination, cache_dir, topology, enabled=True)
    return {"host": socket.gethostname(), "pid": os.getpid(), "start": started,
            "end": time.time(), "process_cpu_s": time.process_time() - cpu,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "nice": os.getpriority(os.PRIO_PROCESS, 0), "ray_cpu_reservation": 1,
            "source_bundle_count": len(bundles), "source_descriptor_count": source_descriptors,
            "distinct_executable_keys": len(entries),
            "bundles": [{"bundle": bundle_root, **result}], "ok": result["all_compiled"]}


def remote(args):
    input_provenance = _input_provenance()
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from hetero.evidence import emit_measurements
    from hetero.experiment import run_arm
    ray.init(address='auto', ignore_reinit_error=True)
    task_ref = None
    def during(cancel):
        nonlocal task_ref
        nodes = sorted((n for n in ray.nodes() if n['Alive'] and n['Resources'].get('TPU', 0)),
                       key=lambda n: n['NodeManagerAddress'])
        if not nodes:
            raise RuntimeError('no live TPU worker hosts')
        task = ray.remote(num_cpus=1)(compile_task)
        task_ref = task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[0]['NodeID'], soft=False),
                    runtime_env={'env_vars': {'JAX_PLATFORMS': 'cpu', 'TPU_SKIP_MDS_QUERY': '1'}}).remote(
                    args.bundle_root, args.cache_dir, args.topology)
        deadline = time.monotonic() + args.timeout
        while True:
            done, _ = ray.wait([task_ref], timeout=1)
            if done:
                return ray.get(task_ref)
            if cancel.is_set() or time.monotonic() > deadline:
                ray.cancel(task_ref, force=True)
                raise RuntimeError('own precompile task cancelled or timed out')
    try:
        summary = run_arm(arm=args.layout, workload='W0', repetition=1, run_id=args.run_id,
                          timeout_s=args.timeout, pilot_policy={'decode_length_blind': True},
                          during_run=during if args.background_precompile else None)
        if summary["warmup"]["catalogue_sha256"] != input_provenance["catalogue_sha256"]:
            raise RuntimeError("staged catalogue changed between preflight and pilot warmup")
        emit_measurements({'ok': True, 'pilot': summary, 'inputs': input_provenance})
    finally:
        if task_ref is not None:
            # get/wait guarantees compile has completed before this job returns.
            ray.cancel(task_ref, force=True)


def submit(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    argv = ['python3', '-u', 'scripts/overlap_pilot.py', '--remote', '--layout', args.layout,
            '--run-id', args.run_id, '--bundle-root', args.bundle_root, '--cache-dir', args.cache_dir,
            '--topology', args.topology, '--timeout', str(args.timeout)]
    if args.background_precompile:
        argv.append('--background-precompile')
    @stage(name='overlap_pilot', order_idx=1, expected_s=args.timeout)
    def run(state, timing):
        input_provenance = _input_provenance()
        supervisor = state.get('supervisor')
        if supervisor is None:
            supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
            state.set('supervisor', supervisor)
        job = state.get('submission_id')
        if job is None:
            job = supervisor.submit_job(entrypoint=shlex.join(argv), runtime_env={
                'working_dir': str(ROOT),
                'excludes': ['.venv*', '.git', '.jj', '__pycache__', '*.pyc', 'results', '**/results/**',
                             "reports", "inputs", "data/workloads"],
                'pip': ['xds-protos==1.84.0'],
                "env_vars": {"PYTHONUNBUFFERED": "1", "OMP_HETERO_DATA_DIR": args.data_dir,
                             "OMP_OVERLAP_CATALOGUE_SHA256": input_provenance["catalogue_sha256"]}},
                metadata={'purpose': 'overlap-precompile-pilot'}, name='overlap-precompile-pilot')
            state.set('submission_id', job)
        result = supervisor.supervise_until_completion(submission_id=job, timeout_s=args.timeout,
                    stage_name='overlap_pilot', state_store=state)
        evidence = {'submission_id': job, 'status': result['status'], 'measurements': result['measurements']}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + '\n')
        measured = result['measurements']
        if not measured or not measured.get('ok'):
            raise RuntimeError('missing successful pilot evidence')
        if any(measured["inputs"][key] != input_provenance[key]
               for key in ("catalogue_sha256", "catalogue_bytes", "manifests")):
            raise RuntimeError("remote staged inputs differ from pre-submission inputs")
        remote_dir = Path(measured['pilot']['artifact_dir'])
        destination = ROOT / 'results/hetero/artifacts'
        destination.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(['kubectl', '-n', 'llm-d-optimized-baseline', 'exec', args.head_pod,
            '-c', 'ray-head', '--', 'tar', 'czf', '-', '-C', str(remote_dir.parent), remote_dir.name],
            capture_output=True, check=True).stdout
        subprocess.run(['tar', 'xzf', '-', '-C', str(destination)], input=archive, check=True)
    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(run)
    if not runner.run():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--background-precompile', action='store_true')
    parser.add_argument('--layout', choices=('S', 'H'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--bundle-root', default='')
    parser.add_argument('--cache-dir', default='')
    parser.add_argument('--topology', default='v5e:4x4')
    parser.add_argument('--timeout', type=float, default=2400)
    parser.add_argument('--head-pod', default='tpu-ray-cluster-vllm-tpu-head-r5tfg')
    parser.add_argument('--ray-url', default='http://127.0.0.1:8265')
    parser.add_argument("--data-dir", default="/tmp/omp_hetero/data")
    parser.add_argument("--check-inputs", action="store_true", help="Validate real catalogue and frozen manifests without submitting a job")
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.background_precompile and (not args.bundle_root or not args.cache_dir):
        parser.error('background precompile requires bundle-root and cache-dir')
    if args.check_inputs:
        evidence = {"ok": True, "inputs": _input_provenance()}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence, indent=2))
        return
    (remote if args.remote else submit)(args)


if __name__ == '__main__':
    main()
