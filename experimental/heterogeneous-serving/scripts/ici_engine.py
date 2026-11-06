#!/usr/bin/env python3
"""RCM-only real-engine resident reconfiguration experiment (one TPU box)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def serve(engine, prompts, prefix):
    from vllm import SamplingParams
    start = time.time()
    outputs = {}
    first = None
    for index, prompt in enumerate(prompts):
        engine.add_request(f'{prefix}-{index}', {'prompt_token_ids': prompt},
            SamplingParams(temperature=0, max_tokens=64, ignore_eos=False))
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.outputs and output.outputs[0].token_ids and first is None:
                first = time.time()
            if output.finished:
                outputs[int(output.request_id.rsplit('-', 1)[1])] = list(output.outputs[0].token_ids)
    end = time.time()
    ordered = [outputs[i] for i in range(len(prompts))]
    return {'outputs': ordered, 'start': start, 'first_token': first, 'end': end,
            'elapsed_s': end - start, 'output_tokens': sum(map(len, ordered)),
            'tokens_per_s': sum(map(len, ordered)) / (end - start)}


def remote(args):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from hetero.slice_inventory import SliceInventory
    from hetero.layout import builtin_layout, resolve_layout
    from hetero.native_backend import prepare_backend
    from hetero.evidence import emit_measurements
    ray.init(address='auto', ignore_reinit_error=True)
    evidence = {'ok': False, 'run_id': os.environ['ICI_RUN_ID'], 'transitions': [],
                'serving': {}, 'workers': [], 'failure': None, 'cleanup': None}
    actors = []
    pg = None
    engines = []
    try:
        payload = json.loads((ROOT / os.environ['ICI_PAYLOAD']).read_text())
        inventory = SliceInventory.from_dict(payload['inventory'])
        box = resolve_layout(inventory, builtin_layout('H')).instances[0]
        standalone = payload['standalone_tp8']
        hosts = box.hosts[:2] if standalone else box.hosts
        initial_tp = 8 if standalone else 16
        evidence['runtime_process_count'] = len(hosts)
        evidence['mode'] = 'standalone_TP8_control' if standalone else 'resident_transitions'
        evidence['runtime_host_ips'] = [host.node_ip for host in hosts]
        evidence['runtime_host_bounds'] = '2,1,1' if standalone else box.host_bounds
        evidence['runtime_chip_topology'] = '4x2' if standalone else box.subslice_topology
        evidence['submesh_cache_writer'] = payload['submesh_cache_writer']
        if any(row['state'] != 'REMOVED' for row in ray.util.placement_group_table().values()):
            raise RuntimeError('refusing placement groups owned by another experiment')
        evidence['box'] = box.to_dict()
        evidence['backend'] = prepare_backend(box.host_ips, apply=False)
        if not evidence['backend']['ready']:
            raise RuntimeError('native backend must already be prepared; prototype does not mutate packages')
        from hetero.ici_worker import ResidentWorker, engine_config
        from hetero.ici_executor import ResidentExecutor
        from vllm.v1.engine.llm_engine import LLMEngine
        pg = placement_group([{'TPU': 4, f'node:{host.node_ip}': 0.001} for host in hosts],
                             strategy='STRICT_SPREAD', name='ici-in-place-' + os.environ['ICI_RUN_ID'])
        ray.get(pg.ready(), timeout=120)
        cls = ray.remote(num_cpus=0, resources={'TPU': 4}, max_restarts=0)(ResidentWorker)
        for rank, host in enumerate(hosts):
            environment = box.process_env(rank)
            if standalone:
                environment.update({'TPU_PROCESS_BOUNDS': '2,1,1', 'TPU_HOST_BOUNDS': '2,1,1',
                    'TPU_TOPOLOGY': '4x2',
                    'TPU_PROCESS_ADDRESSES': ','.join(f'{h.node_ip}:8471' for h in hosts)})
            environment.update({'TPU_VISIBLE_CHIPS': '0,1,2,3', 'PYTHONUNBUFFERED': '1',
                'VLLM_DISABLE_REQUEST_ID_RANDOMIZATION': '1', 'RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS': '1',
                'ICI_COORDINATOR': f'{box.coordinator_ip}:12355',
                'ICI_PROCESS_COUNT': str(len(hosts)), 'ICI_INITIAL_TP': str(initial_tp),
                'ICI_SUBMESH_CACHE_WRITER': '1' if payload['submesh_cache_writer'] else '0',
                'JAX_COMPILATION_CACHE_DIR': '/models/jax_cache/ici-' + os.environ['ICI_RUN_ID']})
            actors.append(cls.options(name=f'ici-{os.environ["ICI_RUN_ID"]}-{rank}', namespace='ici-resident-prototype', runtime_env={'env_vars': environment},
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=rank,
                    placement_group_capture_child_tasks=True)).remote(rank, environment))
        evidence['workers'] = ray.get([a.start.remote() for a in actors], timeout=1200)
        prompts = payload['prompts']
        references = payload['references']
        evidence['prompts'] = prompts
        evidence['sampling'] = {'temperature': 0, 'max_tokens': 64, 'ignore_eos': False}
        evidence['greedy_checks'] = []
        def check(tp, name, rows):
            reference = references[str(tp)]
            if prompts != reference['prompts'] or evidence['sampling'] != reference['sampling']:
                raise RuntimeError('reference prompt/sampling contract differs')
            for index, row in enumerate(rows):
                evidence['greedy_checks'].append({'stage': name, 'engine': index,
                    'token_identical': row['outputs'] == reference['outputs'], 'reference_file': reference['file']})
        def build(tp, group, defer_warmup=False):
            names = [f'ici-{os.environ["ICI_RUN_ID"]}-{actors.index(actor)}' for actor in group]
            config = engine_config(tp, names)
            config.additional_config['ici_defer_warmup'] = defer_warmup
            return LLMEngine(vllm_config=config, executor_class=ResidentExecutor,
                             log_stats=False, multiprocess_mode=False)
        engines = [build(initial_tp, actors)]
        name = f'initial_TP{initial_tp}'
        evidence['serving'][name] = [serve(engines[0], prompts, 'initial')]
        check(initial_tp, name, evidence['serving'][name])
        for switch_index, layout in enumerate(() if standalone else ('2TP8', 'TP16', '2TP8', 'TP16')):
            transition = {'index': switch_index, 'target': layout, 'start': time.time(),
                          'phases_complete': False, 'C_s': None}
            evidence['transitions'].append(transition)
            for engine in engines:
                core = engine.engine_core.engine_core
                while core.scheduler.has_requests():
                    scheduled = core.scheduler.schedule(False)
                    if scheduled.total_num_scheduled_tokens:
                        raise RuntimeError('drain found an unfinished request')
                    output = core.model_executor.collective_rpc('drain_execute_model', args=(scheduled,))[0]
                    core.scheduler.update_from_output(scheduled, output)
            ray.get([a.retire.remote() for a in actors], timeout=180)
            for engine in engines:
                engine.engine_core.shutdown()
            engines = []
            transition['workers_after_reshard'] = ray.get([a.switch.remote(layout) for a in actors], timeout=900)
            engines = [build(16, actors)] if layout == 'TP16' else [build(8, actors[:2], True), build(8, actors[2:], True)]
            if layout == '2TP8':
                for engine in engines:
                    engine.engine_core.engine_core.model_executor.start_warmup()
            for engine in engines:
                engine.engine_core.engine_core.model_executor.wait_for_warmup()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(engines)) as pool:
                rows = list(pool.map(lambda pair: serve(pair[1], prompts, f'switch{switch_index}-{pair[0]}'), enumerate(engines)))
            transition['first_token'] = max(row['first_token'] for row in rows)
            transition['C_s'] = transition['first_token'] - transition['start']
            transition['workers'] = ray.get([a.report.remote() for a in actors], timeout=30)
            transition['phases_complete'] = True
            evidence['serving'][f'switch{switch_index}_{layout}'] = rows
            check(16 if layout == 'TP16' else 8, f'switch{switch_index}_{layout}', rows)
        evidence['workers'] = ray.get([a.report.remote() for a in actors], timeout=30)
        evidence['ok'] = all(row['token_identical'] for row in evidence['greedy_checks'])
    except Exception:
        evidence['failure'] = traceback.format_exc()
        if actors:
            try:
                pending = [a.report.remote() for a in actors]
                ready, waiting = ray.wait(pending, num_returns=len(pending), timeout=60)
                evidence['workers'] = ray.get(ready)
                evidence['missing_worker_reports'] = len(waiting)
            except Exception:
                evidence['worker_report_failure'] = traceback.format_exc()
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        if pg is not None:
            remove_placement_group(pg)
        evidence['cleanup'] = {'actor_kill_issued': len(actors), 'placement_group_remove_issued': pg is not None,
                               'time': time.time()}
        time.sleep(15)
        emit_measurements(evidence)


def submit(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    manifest = json.loads((ROOT / 'data/manifests/W0.json').read_text())
    references = {}
    for tp, layout in ((16, 'H'), (8, 'S')):
        path = ROOT / f'results/restore/ShardedRestore/{layout}-reference.json'
        reference = json.loads(path.read_text())['measurements']
        references[str(tp)] = {'file': str(path.relative_to(ROOT)), 'prompts': reference['prompts'],
            'sampling': reference['sampling'], 'outputs': next(iter(reference['outputs'].values()))}
    payload_path = ROOT / f'ici-input-{args.run_id}.json'
    payload = {'inventory': json.loads(args.inventory.read_text()),
               'prompts': [p['prompt_token_ids'] for p in manifest['requests'][:32]],
               'references': references, 'standalone_tp8': args.standalone_tp8, 'submesh_cache_writer': args.submesh_cache_writer}
    env = {'PYTHONUNBUFFERED': '1', 'ICI_RUN_ID': args.run_id, 'ICI_PAYLOAD': payload_path.name}
    @stage(name='ici_engine', order_idx=1, expected_s=args.timeout)
    def run(state, timing):
        supervisor = state.get('supervisor')
        if supervisor is None:
            supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
            state.set('supervisor', supervisor)
        job = state.get('submission_id')
        if job is None:
            payload_path.write_text(json.dumps(payload))
            try:
                job = supervisor.submit_job(entrypoint='python3 -u scripts/ici_engine.py --remote',
                    runtime_env={'working_dir': str(ROOT), 'excludes': ['.venv*', '.git', '.jj', '__pycache__',
                        '*.pyc', 'results', '**/results/**', 'reports', 'data'], 'env_vars': env},
                    metadata={'purpose': 'ici-resident-real-engine'}, name='ici-real-engine-' + args.run_id)
            finally:
                payload_path.unlink(missing_ok=True)
            state.set('submission_id', job)
        result = supervisor.supervise_until_completion(submission_id=job, timeout_s=args.timeout,
                         stage_name='ici_engine', state_store=state)
        measured = result['measurements']
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'submission_id': job, 'status': result['status'],
                                          'measurements': measured}, indent=2) + '\n')
        # Experiment failures are evidence, not RCM traps holding cluster resources.
        if not measured:
            raise RuntimeError('experiment returned no machine-readable evidence')
    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(run)
    if not runner.run():
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--standalone-tp8', action='store_true', help='matched two-controller TP8 throughput control, no transitions')
    parser.add_argument('--submesh-cache-writer', action='store_true', help='opt-in prototype-local JAX 0.10.2 cache writer patch')
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--inventory', type=Path, default=Path('/tmp/hetero-inventory-only.json'))
    parser.add_argument('--ray-url', default='http://127.0.0.1:8265')
    parser.add_argument('--run-id', default='inplace-' + str(int(time.time())))
    parser.add_argument('--timeout', type=int, default=3000)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ici/InPlaceEngine/run.json')
    args = parser.parse_args()
    remote(args) if args.remote else submit(args)
