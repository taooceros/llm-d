#!/usr/bin/env python3
"""RCM-supervised snapshot validation and host-local tmpfs housekeeping."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hetero import agreement  # noqa: E402
from hetero.layout import BUILTIN_LAYOUTS  # noqa: E402


def _host_measurements(action, layout, rows):
    return {'ok': True, 'action': action, 'layout': layout, 'hosts': rows}


def _verify_fingerprint_reference(reference, measured):
    if not reference.get('layout') or not measured.get('layout'):
        raise RuntimeError('weight fingerprint reference is missing layout identity')

    def hashes(data):
        if 'hosts' in data:
            return {(worker['instance'], worker['process_rank']):
                    (worker['bytes'], worker['sha256'], worker['shards'])
                    for host in data['hosts'] for worker in host['fingerprints']}
        return {(name, worker['process_rank']):
                (worker['bytes'], worker['sha256'], worker['shards'])
                for name, workers in data['workers'].items() for worker in workers}

    if reference['layout'] != measured['layout'] or hashes(reference) != hashes(measured):
        raise RuntimeError('snapshot/source/restored weight byte digest mismatch')


def host_action(action, reserve_bytes, capture_config=False, run_scope=None):
    import shutil
    import socket
    root = Path('/dev/shm/omp_weights')
    if run_scope is not None:
        # A first-switch run scope (hetero/run_scope.py): only its two directories are read or
        # removed; the default snapshot root and every warm cache stay untouched.
        from hetero import run_scope as scopes
        if action not in ('cleanup', 'files', 'memory', 'file-fingerprint'):
            raise RuntimeError(f'--run-scope is not supported for {action}')
    if action == 'cleanup' and run_scope is None:
        shutil.rmtree(root, ignore_errors=False) if root.exists() else None
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0]) * 1024
    free = shutil.disk_usage('/dev/shm').free
    floor = 40_000_000_000
    row = {'host': socket.gethostname(), 'mem_available_bytes': available,
           'tmpfs_free_bytes': free, 'reserve_bytes': reserve_bytes,
           'headroom_bytes': floor, 'cleaned': action == 'cleanup' and run_scope is None}
    row['snapshot_directory_bytes'] = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if run_scope is not None:
        row['run_scope'] = run_scope
        if action == 'cleanup':
            row['scope_removed'] = scopes.remove_scope(run_scope)
        else:
            row['scope_files'] = scopes.scope_files(run_scope)
            if action == 'files' and not (row['scope_files']['snapshot_manifests']
                                          and row['scope_files']['bank_manifests']):
                raise RuntimeError(f'run scope {run_scope}: no snapshot or bank manifest on '
                                   f'{socket.gethostname()}: ' + json.dumps(row['scope_files']))
    if action == 'poison':
        import fcntl
        import mmap
        import time
        import numpy as np
        begin = time.perf_counter()
        poisoned = 0
        for manifest_path in root.glob('*/*/rank*/manifest.json'):
            with (manifest_path.parent / '.snapshot.lock').open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                manifest = json.loads(manifest_path.read_text())
                # Invalidate before modifying bytes; the next snapshot must
                # rewrite every page, not republish an identical destination.
                manifest_path.unlink()
                for array in manifest['arrays']:
                    for shard in array['shards']:
                        with (manifest_path.parent / shard['file']).open('r+b') as file:
                            if shard['bytes']:
                                with mmap.mmap(file.fileno(), 0) as mapped:
                                    raw = np.frombuffer(mapped, dtype=np.uint8)
                                    raw.fill(0xA5)
                                    poisoned += raw.size
                                    del raw
        row.update(poisoned_bytes=poisoned, poison_s=time.perf_counter() - begin)
        if not poisoned:
            raise RuntimeError('no published snapshot to poison')
    if action == 'files' and run_scope is None:
        manifests = list(root.glob('*/*/rank*/manifest.json'))
        if not manifests:
            raise RuntimeError(f'no snapshot manifests on {socket.gethostname()}')
        verified = []
        for path in manifests:
            manifest = json.loads(path.read_text())
            if manifest.get('version') not in (1, 2) or not manifest.get('arrays'):
                raise RuntimeError(f'incomplete snapshot manifest: {path}')
            for array in manifest['arrays']:
                for shard in array['shards']:
                    file = path.parent / shard['file']
                    if file.parent != path.parent or not file.is_file() or file.stat().st_size != shard['bytes']:
                        raise RuntimeError(f'missing/wrong-sized snapshot shard: {file}')
            verified.append(str(path))
        row['verified_manifest_and_sizes'] = verified
    if action == 'file-fingerprint':
        import time
        from hetero.weight_snapshot import file_fingerprint
        fingerprint_start = time.perf_counter()
        # A run scope fingerprints only that scope's snapshot root (same row format).
        fingerprint_root = (Path(scopes.scope_roots(run_scope)['weight_snapshot_root'])
                            if run_scope is not None else root)
        fingerprints = [file_fingerprint(path) for path in sorted(fingerprint_root.glob('*/*/rank*/manifest.json'))]
        if not fingerprints:
            raise RuntimeError('no published snapshot to fingerprint')
        row['fingerprints'] = fingerprints
        row['validation_s'] = time.perf_counter() - fingerprint_start
        row['outside_snapshot_and_restore_timer'] = True
    if action == 'pagecache':
        row['checkpoint_page_cache'] = checkpoint_residency(
            Path(os.environ.get('OMP_SNAPSHOT_MODEL', '/models/gemma-4-31b')))
        row['meminfo_kb'] = {k: int(memory[k].split()[0]) for k in (
            'MemAvailable', 'Cached', 'Shmem', 'Dirty', 'Writeback') if k in memory}
    if capture_config:
        config = Path(os.environ.get('OMP_SNAPSHOT_MODEL', '/models/gemma-4-31b')) / 'config.json'
        row['model_config'] = json.loads(config.read_text())
    if action not in ('cleanup', 'pagecache') and min(available, free) - reserve_bytes < floor:
        raise RuntimeError('insufficient snapshot host headroom: ' + json.dumps(row))
    return row


def checkpoint_residency(model_dir):
    """Page-cache residency of the checkpoint's safetensors files (mincore; reads no file bytes).

    A private, never-written mapping faults nothing; mincore reports the file pages the kernel
    holds in its page cache, i.e. how warm the next normal-loader read of the checkpoint is."""
    import ctypes
    import mmap
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    page = os.sysconf('SC_PAGE_SIZE')
    files, total, resident = [], 0, 0
    for path in sorted(Path(model_dir).glob('*.safetensors')):
        size = path.stat().st_size
        if not size:
            continue
        pages = (size + page - 1) // page
        vector = (ctypes.c_ubyte * pages)()
        with path.open('rb') as handle, mmap.mmap(handle.fileno(), size, flags=mmap.MAP_PRIVATE,
                                                   prot=mmap.PROT_READ | mmap.PROT_WRITE) as mapped:
            anchor = ctypes.c_char.from_buffer(mapped)
            try:
                if libc.mincore(ctypes.addressof(anchor), size, vector) != 0:
                    raise OSError(ctypes.get_errno(), f'mincore failed for {path}')
            finally:
                del anchor
        import numpy as np
        hot = int((np.frombuffer(vector, dtype=np.uint8) & 1).sum()) * page
        files.append({'file': path.name, 'bytes': size, 'resident_bytes': min(hot, size)})
        total += size
        resident += min(hot, size)
    return {'model_dir': str(model_dir), 'files': len(files), 'bytes': total, 'resident_bytes': resident,
            'resident_fraction': round(resident / total, 4) if total else None, 'per_file': files}


def remote(args):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from hetero.evidence import emit_measurements
    ray.init(address='auto', ignore_reinit_error=True)
    if args.action == 'preflight':
        from hetero.registry import find_registry
        registry = find_registry()
        if registry is not None and ray.get(registry.snapshot.remote(), timeout=60):
            raise RuntimeError('preflight refuses pre-existing registered engines')
        from ray.util import placement_group_table
        if any(row['state'] != 'REMOVED' for row in placement_group_table().values()):
            raise RuntimeError('preflight refuses pre-existing placement groups')
    if args.action in ('preflight', 'memory', 'cleanup', 'files', 'file-fingerprint', 'poison', 'pagecache'):
        nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('TPU', 0) > 0]
        if not nodes:
            raise RuntimeError('no live TPU hosts')
        task = ray.remote(num_cpus=0)(host_action)
        rows = ray.get([task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            n['NodeID'], soft=False)).remote(args.action, args.reserve_bytes,
                args.action == 'preflight' and index == 0, args.run_scope)
                for index, n in enumerate(nodes)], timeout=120)
        result = _host_measurements(args.action, args.layout, rows)
    else:
        from hetero.registry import find_registry
        registry = find_registry()
        if registry is None:
            raise RuntimeError('no registry')
        rows = ray.get(registry.snapshot.remote(args.layout), timeout=60)
        expected = len(BUILTIN_LAYOUTS[args.layout])
        if len(rows) != expected or len({r['instance_id'] for r in rows}) != expected:
            raise RuntimeError(f'incomplete {args.layout} registry: {rows}')
        actors = {r['instance_id']: ray.get_actor(r['engine_actor_name'], namespace=r['ray_namespace']) for r in rows}
        if args.action in ('snapshot', 'fingerprint'):
            names = sorted(actors)
            if args.instances:
                # Publish only the selected engines, e.g. Mb's large_b for a staged join
                # whose kept engines never restart (snapshots are keyed by layout/instance).
                selected = sorted(set(args.instances.split(',')))
                if not set(selected) <= set(names):
                    raise RuntimeError(f'{selected} not all registered in {args.layout}: {names}')
                names = selected
            method = 'snapshot_weights' if args.action == 'snapshot' else 'weight_fingerprint'
            reports = dict(zip(names, ray.get([getattr(actors[name], method).remote()
                                              for name in names], timeout=1800)))
            if any(not report for report in reports.values()):
                raise RuntimeError('missing worker snapshot reports')
            result = {'ok': True, 'action': args.action, 'layout': args.layout, 'workers': reports}
        else:
            phases = {name: ray.get(actor.worker_init_phases.remote(), timeout=120)
                      for name, actor in sorted(actors.items())}
            if args.require_restored:
                bad = {name: reports for name, reports in phases.items()
                       if not reports or any(r.get('weight_snapshot', {}).get('restored') is not True for r in reports)}
                if bad:
                    emit_measurements({'ok': False, 'error': 'restore refused/fell back/missing evidence',
                                       'layout': args.layout, 'worker_init_phases': phases})
                    return
            if args.action == 'check':
                emit_measurements({'ok': True, 'layout': args.layout, 'worker_init_phases': phases})
                return
            if args.action == 'probe':
                manifest = json.loads((ROOT / os.environ['OMP_SNAPSHOT_MANIFEST_FILE']).read_text())
                prompts = manifest['requests'][:32]
                points = manifest['probe_points']
                probe_rows = []
                for name in sorted(actors):
                    for k, point in enumerate(points):
                        i, prefix = point['prompt_index'], point['prefix']
                        row = ray.get(actors[name].next_token_diagnostic.remote(
                            token_ids=prompts[i]['prompt_token_ids'] + prefix,
                            request_id=f'probe-{args.layout}-{name}-{i}-{len(prefix)}-{k}'), timeout=600)
                        probe_rows.append({'engine': name, 'prompt_index': i, 'prefix_len': len(prefix),
                                           'token_id': row['token_id'], 'top_logprobs': row['top_logprobs']})
                emit_measurements({'ok': True, 'action': 'probe', 'layout': args.layout,
                    'manifest_sha256': manifest['manifest_sha256'],
                    'engine_identity': {r['instance_id']: r['model_build'] for r in rows},
                    'tp_sizes': {name: int(ray.get(actors[name].identity.remote(), timeout=120)['tp_size'])
                                 for name in sorted(actors)},
                    'probe_points': points, 'rows': probe_rows})
                return
            manifest = json.loads((ROOT / os.environ['OMP_SNAPSHOT_MANIFEST_FILE']).read_text())
            prompts = manifest['requests'][:32]
            if len(prompts) != 32:
                raise RuntimeError('W0 must provide 32 validation prompts')
            outputs = {}
            for name, actor in sorted(actors.items()):
                outputs[name] = []
                for index, prompt in enumerate(prompts):
                    row = ray.get(actor.generate.remote(prompt_token_ids=prompt['prompt_token_ids'],
                        max_tokens=64, temperature=0.0, ignore_eos=False,
                        request_id=f'snapshot-{args.layout}-{name}-{index}'), timeout=600)
                    outputs[name].append(row['output_token_ids'])
            engine_identity = {r['instance_id']: r['model_build'] for r in rows}
            tp_sizes = {name: int(ray.get(actors[name].identity.remote(), timeout=120)['tp_size'])
                        for name in sorted(actors)}
            replica = agreement.replica_equality(outputs, tp_sizes)
            cross_layout = diagnostics = None
            if 'other_outputs' in manifest:
                canonical = agreement.canonical_engine(outputs)
                diagnostics = []
                for point in agreement.divergence_points(outputs[canonical], manifest['other_outputs']):
                    i, d = point['prompt_index'], point['d']
                    row = ray.get(actors[canonical].next_token_diagnostic.remote(
                        token_ids=prompts[i]['prompt_token_ids'] + outputs[canonical][i][:d],
                        request_id=f'agree-{args.layout}-{canonical}-{i}-{d}'), timeout=600)
                    diagnostics.append({'prompt_index': i, 'd': d, 'token_id': row['token_id'],
                                        'top_logprobs': row['top_logprobs']})
                cross = agreement.evaluate_side(outputs[canonical], manifest['other_outputs'], diagnostics)
                cross_layout = {'canonical_engine': canonical, **cross}
            result = {'ok': True, 'action': args.action, 'layout': args.layout,
                      'manifest_sha256': manifest['manifest_sha256'],
                      'prompts': [p['prompt_token_ids'] for p in prompts],
                      'sampling': {'temperature': 0, 'ignore_eos': False, 'max_tokens': 64},
                      'outputs': outputs, 'worker_init_phases': phases,
                      'require_restored': args.require_restored,
                      'engine_identity': engine_identity, 'tp_sizes': tp_sizes,
                      'replica_equality': replica}
            if not replica['passed']:
                result['ok'] = False
                result['error'] = 'replica outputs differ within layout'
            if cross_layout is not None:
                result['cross_layout'] = cross_layout
                result['diagnostics'] = diagnostics
    emit_measurements(result)


def _verify_greedy_reference(reference, measured):
    if 'engine_identity' not in reference:
        raise RuntimeError('reference has no engine_identity; regenerate it under the current engine settings')
    keys = ('layout', 'manifest_sha256', 'prompts', 'sampling', 'engine_identity', 'outputs')
    mismatches = [key for key in keys if measured.get(key) != reference[key]]
    if mismatches:
        raise RuntimeError('greedy restored output mismatch: ' + ', '.join(mismatches))
    return {'passed': True, 'prompts_per_engine': 32, 'engines': len(measured['outputs']),
            'comparison': 'output_token_ids, token for token'}


def _verify_agreement(reference, measured, path):
    mismatches = [key for key in ('manifest_sha256', 'prompts', 'sampling')
                  if measured.get(key) != reference.get(key)]
    if mismatches:
        raise RuntimeError('agreement reference mismatch: ' + ', '.join(mismatches))
    cross = measured.get('cross_layout')
    if not cross:
        raise RuntimeError('measurement has no cross_layout verdict')
    # Full bf16 state: cross-layout drift is recorded, not gated (plan decision log,
    # 2026-10-05).  The withdrawn tie-gate verdict is kept as a drift descriptor only.
    gaps = [row.get('gap_nats') for row in cross['divergences']]
    return {'reference': str(path), 'other_layout': reference['layout'], 'gated': False,
            'tie_gate_passed': cross['passed'],
            'summary': {'exact_prompts': cross['exact_prompts'],
                        'first_divergence_min': cross['first_divergence_min'],
                        'divergences': len(cross['divergences']),
                        'divergence_indices': [row.get('d') for row in cross['divergences']],
                        'tie_gaps_nats': gaps,
                        'max_tie_gap_nats': max((g for g in gaps if g is not None), default=None)}}


def submit(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    env = {'PYTHONUNBUFFERED': '1', 'OMP_SNAPSHOT_MODEL': os.environ.get('MODEL', '/models/gemma-4-31b')}
    input_directory = None
    if args.action in ('validate', 'probe'):
        import tempfile
        input_directory = tempfile.TemporaryDirectory(prefix='_snapshot_validation_inputs_', dir=ROOT)
        input_file = Path(input_directory.name) / 'prompts.json'
        manifest = json.loads((ROOT / 'data/manifests/W0.json').read_text())
        shipped = {'manifest_sha256': manifest['manifest_sha256'], 'requests': manifest['requests'][:32]}
        if args.agreement_reference:
            ref = json.loads(args.agreement_reference.read_text())['measurements']
            if ref.get('ok') is not True:
                raise SystemExit('agreement reference is not a successful measurement')
            if ref['layout'] == args.layout:
                raise SystemExit('agreement reference must come from a different layout')
            shipped['other_outputs'] = ref['outputs'][agreement.canonical_engine(ref['outputs'])]
            shipped['other_layout'] = ref['layout']
        if args.action == 'probe':
            shipped['probe_points'] = json.loads(args.probe_points.read_text())
        input_file.write_text(json.dumps(shipped))
        env['OMP_SNAPSHOT_MANIFEST_FILE'] = str(input_file.relative_to(ROOT))
    argv = ['python3', '-u', 'scripts/snapshot_validate.py', args.action, '--remote',
            '--layout', args.layout, '--reserve-bytes', str(args.reserve_bytes)]
    if args.instances:
        argv += ['--instances', args.instances]
    if args.run_scope:
        argv += ['--run-scope', args.run_scope]

    @stage(name='snapshot_validation', order_idx=1, expected_s=args.timeout)
    def run(state, timing):
        supervisor = state.get('supervisor')
        if supervisor is None:
            supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
            state.set('supervisor', supervisor)
        job = state.get('submission_id')
        if job is None:
            job = supervisor.submit_job(entrypoint=shlex.join(argv), runtime_env={
                'working_dir': str(ROOT), 'excludes': ['.venv*', '.git', '.jj', '__pycache__',
                '*.pyc', 'results', '**/results/**', 'reports', 'data'], 'env_vars': env},
                metadata={'purpose': 'sharded-snapshot-' + args.action}, name='sharded-snapshot-' + args.action)
            state.set('submission_id', job)
        result = supervisor.supervise_until_completion(submission_id=job, timeout_s=args.timeout,
                    stage_name='snapshot_validation', state_store=state)
        measured = result['measurements']
        evidence = {'submission_id': job, 'status': result['status'], 'measurements': measured}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + '\n')
        if not measured or measured.get('ok') is not True:
            raise RuntimeError('snapshot job missing successful evidence: '
                               + str((measured or {}).get('error')))
        if args.reference:
            reference = json.loads(args.reference.read_text())['measurements']
            if args.action == 'fingerprint':
                _verify_fingerprint_reference(reference, measured)
                measured['weight_equality'] = {'passed': True, 'reference': str(args.reference),
                    'comparison': 'SHA-256 and identity of every shard; cold file vs live source' if 'hosts' in reference else 'SHA-256 and identity of every live device shard',
                    'outside_snapshot_and_restore_timer': True}
            else:
                measured['greedy_equality'] = _verify_greedy_reference(reference, measured)
                measured['greedy_equality']['reference'] = str(args.reference)
            args.output.write_text(json.dumps(evidence, indent=2) + '\n')
        if args.agreement_reference:
            ref = json.loads(args.agreement_reference.read_text())['measurements']
            measured['agreement'] = _verify_agreement(ref, measured, args.agreement_reference)
            args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(run)
    try:
        if not runner.run():
            raise SystemExit(1)
    finally:
        if input_directory is not None:
            input_directory.cleanup()


def analyze(args):
    """Adapt two distinct transition prefixes to the existing timeline analyzer."""
    import shutil
    import subprocess
    events = [json.loads(line) for line in (args.output / 'timeline.jsonl').read_text().splitlines()]
    reconfig = ROOT / 'results/reconfig'
    artifacts = ROOT / 'results/hetero/artifacts'
    reconfig.mkdir(parents=True, exist_ok=True)
    transitions = []
    for source_index, source, target_index, target in ((1, 'S', 2, 'H'), (2, 'H', 3, 'S')):
        prefix = f'snapshot-{source}-to-{target}'
        source_label = f'transition-{source_index}-{source}'
        target_label = f'transition-{target_index}-{target}'
        begin = next(e['t'] for e in events if e['event'] == f'deploy_start:{source_label}')
        end = next(e['t'] for e in events if e['event'] == f'run_end:W0:{target_label}')
        selected = []
        for event in events:
            if begin <= event['t'] <= end:
                event = dict(event)
                if event['event'].startswith(('deploy_start:', 'deploy_end:')):
                    event['event'] = event['event'].split(':', 1)[0]
                selected.append(event)
        timeline = reconfig / f'timeline-{prefix}.jsonl'
        timeline.write_text(''.join(json.dumps(event) + '\n' for event in selected))
        for layout, label in ((source, source_label), (target, target_label)):
            shutil.copyfile(args.output / f'{label}-deploy.json', reconfig / f'deploy-{prefix}-{layout}.json')
            original = artifacts / f'rc-snapshot-{label}-{layout}-s1'
            destination = artifacts / f'rc-{prefix}-{layout}-s1'
            if not original.is_dir():
                raise RuntimeError(f'missing physical pilot artifacts: {original}')
            shutil.copytree(original, destination, dirs_exist_ok=True)
        destination = args.output / f'cost-{source}-to-{target}.json'
        subprocess.run([sys.executable, 'scripts/reconfig_timeline.py', prefix,
                        '--json', str(destination.resolve())], cwd=ROOT, check=True)
        measured = json.loads(destination.read_text())
        if len(measured) != 2 or measured[1]['cost_s'] is None:
            raise RuntimeError(f'incomplete transition cost: {destination}')
        transitions.append(measured[1])
    (args.output / 'transitions.json').write_text(json.dumps(transitions, indent=2) + '\n')
    validations = {layout: json.loads((args.output / f'{layout}-restored-validation.json').read_text())['measurements']
                   for layout in ('S', 'H')}
    for layout, measured in validations.items():
        reference = json.loads((args.output / f'{layout}-reference.json').read_text())['measurements']
        if measured['outputs'] != reference['outputs']:
            raise RuntimeError(f'{layout} output mismatch during consolidation')
    unchecked = {}
    for layout in ('S', 'H'):
        path = args.output / f'{layout}-unchecked-validation.json'
        if path.exists():
            measured = json.loads(path.read_text())['measurements']
            if measured['outputs'] != validations[layout]['outputs']:
                raise RuntimeError(f'{layout} unchecked output mismatch during consolidation')
            unchecked[layout] = {'label': 'SHA verification disabled; not used for transition costs',
                                 'measurement': measured}
    (args.output / 'validation.json').write_text(json.dumps(
        {'ok': True, 'layouts': validations, 'unchecked_variants': unchecked}, indent=2) + '\n')
    table_path = Path(os.environ.get('THROUGHPUT_TABLE', str(Path.home() / 'hetero-research/results/reconfig/throughput_table_measured.json')))
    mix_path = Path(os.environ.get('MIX_ESTIMATES', str(Path.home() / 'hetero-research/results/reconfig/mixA_model.json')))
    table = json.loads(table_path.read_text())
    mix = json.loads(mix_path.read_text())
    from reconfig_breakeven import breakeven
    thresholds = []
    for transition in transitions:
        source, target, cost = transition['previous'], transition['layout'], transition['cost_s']
        for workload, entry in table.items():
            source_row, target_row = entry['makespan_s'].get(source), entry['makespan_s'].get(target)
            if source_row is None or target_row is None:
                continue
            tokens = entry['output_tokens']
            threshold = breakeven(tokens / source_row['value'], tokens / target_row['value'], cost)
            thresholds.append({'workload': workload, 'from': source, 'to': target, 'cost_s': cost,
                               'breakeven_tokens': threshold,
                               'breakeven_phase_multiple': threshold / tokens if threshold is not None else None,
                               'sources': [source_row['source'], target_row['source']]})
    target_s_cost = next(row['cost_s'] for row in transitions if row['layout'] == 'S')
    mix_copies = (target_s_cost + mix['static_overlap_forfeited_s']) / mix['gain_per_W0_copy_s']
    report = {'thresholds': thresholds, 'throughput_table_source': str(table_path),
              'mix_estimate_source': str(mix_path), 'mix_projection': {
                  'status': 'estimate, not measured M-to-S transition',
                  'assumption': 'substitute measured restored H-to-S cost for M-to-S cost; preserve measured mix rates and overlap',
                  'breakeven_W0_copies': mix_copies,
                  'breakeven_tokens': mix_copies * table['W0']['output_tokens'],
                  'predicted_k4_full_s': mix['T_M_W4_s'] + mix['T_S_W0x4_s'] + target_s_cost}}
    (args.output / 'breakeven.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'memory', 'cleanup', 'files', 'file-fingerprint', 'poison', 'pagecache', 'snapshot', 'fingerprint', 'validate', 'check', 'probe', 'analyze'))
    parser.add_argument('--layout', choices=sorted(BUILTIN_LAYOUTS), default='S')
    parser.add_argument('--instances', help='snapshot/fingerprint only these comma-separated instances')
    parser.add_argument('--run-scope', help='cleanup/files/memory/file-fingerprint only: act on this first-switch run scope '
                        '(hetero/run_scope.py) instead of /dev/shm/omp_weights')
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--agreement-reference', type=Path)
    parser.add_argument('--probe-points', type=Path, help='probe only: JSON list of {prompt_index, prefix}')
    parser.add_argument('--require-restored', action='store_true')
    parser.add_argument('--reserve-bytes', type=int, default=0)
    parser.add_argument('--ray-url', default='http://127.0.0.1:8265')
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()
    if args.agreement_reference and args.action != 'validate':
        raise SystemExit('--agreement-reference is only valid with validate')
    if args.probe_points and args.action != 'probe':
        raise SystemExit('--probe-points is only valid with probe')
    if args.action == 'probe' and not args.probe_points and not args.remote:
        raise SystemExit('probe requires --probe-points')
    if not args.remote and args.output is None:
        parser.error('--output is required for supervised submission')
    (analyze if args.action == 'analyze' else remote if args.remote else submit)(args)


if __name__ == '__main__':
    main()
