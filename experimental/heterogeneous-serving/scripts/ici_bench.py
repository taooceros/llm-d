#!/usr/bin/env python3
"""RCM-only launcher for persistent native-JAX ICI experiments.

Launch through: flock /tmp/hetero-cluster.lock scripts/ici_sequence.sh MODE SCOPE OUTPUT
Remote reservations always die with the job; no detached actors or PGs.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
from pathlib import Path
import shlex
import sys
import time
import traceback
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def remote(args):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from hetero.layout import resolve_layout, builtin_layout, LayoutRequest, InstanceRequest
    from hetero.slice_inventory import SliceInventory
    from hetero.evidence import emit_event, emit_measurements
    inventory = SliceInventory.from_dict(json.loads(os.environ['ICI_INVENTORY']))
    layout = resolve_layout(inventory, builtin_layout('H'))
    instance = layout.instances[0] if args.scope == 'box' else resolve_layout(
        inventory, LayoutRequest('ICI32', (InstanceRequest('slice', 32),))).instances[0]
    # Keep distributed and libtpu physical process identities identical.
    controller_id = {h.node_ip: h.process_rank for h in instance.hosts}
    coordinator = f'{instance.coordinator_ip}:12355'

    @ray.remote(num_cpus=1, resources={'TPU': 4})
    class Controller:
        def __init__(self, env, rank, count):
            os.environ.update(env)
            os.environ['TPU_VISIBLE_CHIPS'] = '0,1,2,3'
            for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                os.environ.pop(name, None)
            self.rank, self.count = rank, count
        def initialize(self):
            import jax
            jax.distributed.initialize(coordinator_address=coordinator,
                num_processes=self.count, process_id=self.rank,
                initialization_timeout=180)
            from hetero.ici_benchmark import orders
            return orders()
        def run(self, mode, sizes, repetitions, profile):
            from hetero.ici_benchmark import run
            return run(mode, sizes, repetitions, profile)

    ray.init(address='auto', namespace='ici-benchmark')
    pg, actors = None, []
    result = {'scope': args.scope, 'mode': args.mode, 'instance': instance.to_dict(),
              'coordinator': coordinator, 'controller_ids': controller_id, 'ok': False}
    try:
        pg = placement_group([{'CPU': 1, 'TPU': 4, f'node:{h.node_ip}': .001}
                              for h in instance.hosts], strategy='STRICT_SPREAD', name=f'ici-{args.scope}-{int(time.time())}')
        ray.get(pg.ready(), timeout=120)
        emit_event('ICI_RESERVED', placement_group=pg.id.hex())
        for i, host in enumerate(instance.hosts):
            env = instance.process_env(host.process_rank)
            actors.append(Controller.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=i),
                resources={'TPU': 4, f'node:{host.node_ip}': .001}).remote(
                    env, controller_id[host.node_ip], len(instance.hosts)))
        result['orders'] = ray.get([a.initialize.remote() for a in actors], timeout=300)
        emit_event('ICI_INITIALIZED', hosts=len(actors))
        if args.mode != 'orders':
            initial_mode = 'reshard' if args.mode == 'campaign' else args.mode
            result['processes'] = ray.get([a.run.remote(initial_mode, args.sizes.split(','), args.repetitions, args.profile)
                                          for a in actors], timeout=args.timeout - 360)
            if initial_mode == 'reshard':
                result['reshard_ok'] = True
                emit_event('ICI_RESHARD_COMPLETE')
            if args.mode == 'campaign':
                mixed = ray.get([a.run.remote('mixed', [], args.repetitions, False) for a in actors], timeout=args.timeout - 360)
                for process, probe in zip(result['processes'], mixed):
                    process['mixed'] = probe
                result['mixed_ok'] = all(p['correct'] for p in mixed)
        result['ok'] = result.get('mixed_ok', True)
    except Exception as exc:
        result['error'] = str(exc)
        result['traceback'] = traceback.format_exc()
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception as exc:
                emit_event('ICI_KILL_ERROR', error=str(exc))
        if pg:
            remove_placement_group(pg)
        time.sleep(15)
        result['cleanup'] = {'actors_killed': len(actors), 'placement_group_removed': pg is not None}
        emit_event('ICI_RELEASED')
        emit_measurements(result)


def local(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    @stage(name='ici_native_jax', order_idx=0, expected_s=args.timeout)
    def submit(state, timing):
        supervisor = RayJobSupervisor(ray_url='http://127.0.0.1:8265', timing_engine=timing)
        argv = ['python3', '-u', 'scripts/ici_bench.py', '--remote', '--scope', args.scope,
                '--mode', args.mode, '--sizes', args.sizes, '--repetitions', str(args.repetitions), '--timeout', str(args.timeout)]
        if args.profile:
            argv.append('--profile')
        submission = supervisor.submit_job(entrypoint=shlex.join(argv),
            runtime_env={'working_dir': str(ROOT),
                         'excludes': ['.venv*', '.git', '.jj', 'data', 'results', '**/results/**', 'formalization', 'reports', '__pycache__', '*.pyc'],
                         'env_vars': {'ICI_INVENTORY': Path(args.inventory).read_text(), 'PYTHONUNBUFFERED': '1',
                                      'ICI_PARAMETER_CATALOGUE': base64.b64encode(zlib.compress(args.catalogue.read_bytes())).decode()}},
            name=f'ici-{args.scope}-{args.mode}', metadata={'purpose': 'synthetic-ICI-reshard'})
        state.set('submission_id', submission)
        result = supervisor.supervise_until_completion(submission_id=submission,
            timeout_s=args.timeout, stage_name='ici_native_jax', state_store=state)
        payload = {'submission_id': submission, 'saved_at': time.time(),
                   'status': result['status'], 'measurements': result['measurements'],
                   'entrypoint': shlex.join(argv)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + '\n')
        print(f'[OMP_EVENT: ICI_SAVED path={args.output}]', flush=True)
    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(submit)
    if not runner.run():
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--scope', choices=('box', 'slice'), default='box')
    parser.add_argument('--mode', choices=('orders', 'subgroup', 'reshard', 'mixed', 'campaign'), default='orders')
    parser.add_argument('--sizes', default='256MiB,1GiB,model')
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--inventory', default='/tmp/hetero-inventory-only.json')
    parser.add_argument('--catalogue', type=Path, default=ROOT / 'results/ici/ICIBench/parameter_catalogue.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ici/ICIBench/run.json')
    parser.add_argument('--timeout', type=float, default=3000)
    args = parser.parse_args()
    remote(args) if args.remote else local(args)
