#!/usr/bin/env python3
"""Compare concurrent shared-parent creation with serial precreation on /models."""
import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def operate(root, index, mode):
    import hashlib
    import os
    import shutil
    import socket
    import time
    directory = Path(root)
    if mode == "cleanup-cache":
        if directory.parent != Path("/models") or not directory.name.startswith("jax_cache_overlap_"):
            raise ValueError("cleanup refuses a non-experiment cache directory")
        if directory.is_symlink():
            raise ValueError("cleanup refuses a symlink cache directory")
        existed = directory.exists()
        if existed:
            shutil.rmtree(directory)
        return {"root": str(directory), "existed": existed, "removed": not directory.exists()}
    if mode == 'prepare':
        for i in range(8):
            (directory / str(i) / 'phase').mkdir(parents=True, exist_ok=True)
        return {'prepared': True}
    if mode == 'cleanup':
        shutil.rmtree(directory, ignore_errors=False)
        return {'removed': True}
    report = {'host': socket.gethostname(), 'index': index, 'mode': mode,
              'start': time.time(), 'operations': [],
              'models_mounts': [line for line in Path('/proc/self/mountinfo').read_text().splitlines()
                               if ' /models ' in line]}
    try:
        target = directory / str(0 if mode == "shared" else index) / "phase"
        if mode == 'create':
            target.mkdir(parents=True, exist_ok=True)
        else:
            report['directory_stat'] = list(target.stat())
        payload = (str(0 if mode == "shared" else index) * 4096).encode()
        for n in range(20):
            row = {'n': n}
            try:
                file = target / f'{n}.mlir'
                file.write_bytes(payload)
                actual = file.read_bytes()
                row.update(ok=actual == payload, sha256=hashlib.sha256(actual).hexdigest())
            except OSError as exc:
                row.update(ok=False, errno=exc.errno, error=str(exc))
            report['operations'].append(row)
    except OSError as exc:
        report.update(error=str(exc), errno=exc.errno)
    report['end'] = time.time()
    report['ok'] = not report.get('error') and all(r['ok'] for r in report['operations'])
    return report


def remote(args):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from hetero.evidence import emit_measurements
    ray.init(address='auto', ignore_reinit_error=True)
    nodes = sorted((n for n in ray.nodes() if n['Alive'] and n['Resources'].get('TPU', 0)),
                   key=lambda n: n['NodeManagerAddress'])
    task = ray.remote(num_cpus=0)(operate)
    def launch(index, root, mode):
        return task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[index]['NodeID'], soft=False)).remote(root, index, mode)
    if args.cleanup_cache:
        cleanup = ray.get(launch(0, args.root, "cleanup-cache"))
        emit_measurements({"ok": cleanup["removed"], "cleanup": cleanup})
        return
    reports = []
    roots = [args.root + "_concurrent", args.root + "_prepared", args.root + "_shared"]
    try:
        for root, mode in zip(roots, ("create", "existing", "shared")):
            if mode != "create":
                ray.get(launch(0, root, 'prepare'))
            reports.append({'root': root, 'mode': mode,
                            'hosts': ray.get([launch(i, root, mode) for i in range(len(nodes))])})
    finally:
        cleanup = []
        for root in roots:
            try:
                cleanup.append(ray.get(launch(0, root, 'cleanup')))
            except Exception as exc:
                cleanup.append({'error': str(exc)})
    emit_measurements({'ok': True, 'cases': reports, 'cleanup': cleanup})


def submit(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    @stage(name='overlap_storage', order_idx=1, expected_s=300)
    def run(state, timing):
        supervisor = state.get('supervisor')
        if supervisor is None:
            supervisor = RayJobSupervisor(ray_url='http://127.0.0.1:8265', timing_engine=timing)
            state.set('supervisor', supervisor)
        job = state.get('submission_id')
        if job is None:
            argv = ["python3", "-u", "scripts/overlap_storage_probe.py", "--remote", "--root", args.root]
            if args.cleanup_cache:
                argv.append("--cleanup-cache")
            job = supervisor.submit_job(entrypoint=shlex.join(argv),
                runtime_env={'working_dir': str(ROOT), 'excludes': ['.venv*','.git','.jj','data','results','reports','inputs','__pycache__','*.pyc']},
                name='overlap-storage-probe')
            state.set('submission_id', job)
        result = supervisor.supervise_until_completion(submission_id=job, timeout_s=300, stage_name='overlap_storage', state_store=state)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'submission_id':job,'status':result['status'],'measurements':result['measurements']},indent=2)+'\n')
        if not result['measurements']:
            raise RuntimeError('storage probe produced no evidence')
    runner = StageRunner(silence_timeout_s=300)
    runner.register(run)
    if not runner.run():
        raise SystemExit(1)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--remote',action='store_true')
    ap.add_argument('--root',required=True)
    ap.add_argument("--cleanup-cache", action="store_true")
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    (remote if args.remote else submit)(args)
