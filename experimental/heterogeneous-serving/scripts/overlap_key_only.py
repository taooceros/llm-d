#!/usr/bin/env python3
"""Compare captured persistent keys with an abstract TPU backend; never compile."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUNDLE = '/models/omp_overlap_s-split-20261002c_bundle'
ENV_KEYS = ('XLA_FLAGS', 'LIBTPU_INIT_ARGS')


def index_descriptors(root):
    groups = {}
    phases = Counter()
    environments = set()
    paths = sorted(Path(root).glob('*/*/*.json'))
    for path in paths:
        row = json.loads(path.read_text())
        identity = {k: row[k] for k in ('key', 'module_sha256', 'options_sha256',
                    'device_ids', 'device_shape', 'jax_version', 'compiler_env')}
        environments.add(json.dumps(row['compiler_env'], sort_keys=True))
        signature = json.dumps(identity, sort_keys=True)
        group = groups.setdefault(signature, {'descriptor': row, 'paths': []})
        group['paths'].append(str(path))
        phases[str(path.parent.relative_to(root))] += 1
    if not paths:
        raise ValueError('no finalized capture descriptors')
    if len(environments) != 1:
        raise ValueError('captured compiler environments differ; refusing mixed initialization')
    return list(groups.values()), dict(phases), json.loads(environments.pop())


def read_verified(path, sha256):
    payload = Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != sha256:
        raise ValueError(f'captured payload hash mismatch: {path}: {actual} != {sha256}')
    return payload


def compare_one(group, available):
    import jax
    import numpy as np
    from jax._src import compiler
    from jax._src.interpreters import mlir
    from jax._src.lib import xla_client as xc
    from jaxlib.mlir import ir

    d = group['descriptor']
    path = Path(group['paths'][0])
    row = {'expected_key': d['key'], 'descriptor': str(path),
           'descriptor_copies': len(group['paths']), 'device_ids': d['device_ids'],
           'device_shape': d['device_shape'], 'module_sha256': d['module_sha256'],
           'options_sha256': d['options_sha256']}
    try:
        if d['jax_version'] != jax.__version__:
            raise ValueError(f"captured JAX {d['jax_version']} != installed {jax.__version__}")
        devices = np.asarray([available[i] for i in d['device_ids']], dtype=object).reshape(d['device_shape'])
        backend = devices.flat[0].client
        options = xc.CompileOptions.ParseFromString(read_verified(path.with_suffix('.options'), d['options_sha256']))
        payload = read_verified(path.with_suffix('.mlir'), d['module_sha256'])
        try:
            row['topology_fingerprint'] = str(xc.get_topology_for_devices(list(devices.flat)).fingerprint())
        except Exception as exc:
            # The real key function decides whether a supported fallback applies.
            row['topology_fingerprint_error'] = f'{type(exc).__name__}: {exc}'
        row['virtual_device_kinds'] = [v.device_kind for v in devices.flat]
        row['virtual_process_indices'] = [v.process_index for v in devices.flat]
        with mlir.make_ir_context():
            module = ir.Module.parse(payload)
            row['module'] = ir.StringAttr(module.operation.attributes['sym_name']).value
            row['actual_key'] = compiler._get_cache_key(options, backend, module, devices)
        if row['actual_key'] is None:
            raise ValueError('persistent key generation returned None')
        row['matches'] = row['actual_key'] == row['expected_key']
        row['status'] = 'match' if row['matches'] else 'mismatch'
    except Exception as exc:
        if isinstance(exc, TimeoutError):
            raise
        row.update(status='error', matches=False, error=f'{type(exc).__name__}: {exc}')
    return row


def source_evidence():
    import inspect
    from jax._src import cache_key, compilation_cache, compiler, xla_bridge
    from jax.experimental import topologies
    result = {}
    for fn in (compiler._get_cache_key, compilation_cache.is_cache_used,
               compilation_cache.get_cache_key, cache_key.get,
               cache_key._hash_accelerator_config, cache_key._hash_platform,
               cache_key._hash_xla_flags, cache_key._hash_serialized_compile_options,
               topologies.get_topology_desc, xla_bridge.make_pjrt_tpu_topology):
        lines, start = inspect.getsourcelines(fn)
        result[f'{fn.__module__}.{fn.__name__}'] = {
            'file': inspect.getsourcefile(fn), 'start': start, 'end': start + len(lines) - 1,
            'source': [f'{start+i}:{line.rstrip()}' for i, line in enumerate(lines)]}
    return result


def audit(root, topology_name, seconds):
    started, cpu = time.time(), time.process_time()
    report = {'bundle_root': root, 'topology_name': topology_name,
              'host': socket.gethostname(), 'pid': os.getpid(), 'started': started,
              'compile_calls': 0, 'executable_invocations': 0,
              'rows': [], 'errors': [], 'ok': False}
    def deadline(*_):
        raise TimeoutError(f'key-only work exceeded {seconds}s')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(seconds)
    try:
        if os.environ.get('JAX_PLATFORMS') != 'cpu' or os.environ.get('TPU_SKIP_MDS_QUERY') != '1':
            raise ValueError('requires JAX_PLATFORMS=cpu and TPU_SKIP_MDS_QUERY=1')
        groups, phases, compiler_env = index_descriptors(root)
        for key in ENV_KEYS:
            value = compiler_env.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # Restore compiler environment before importing JAX/plugin code.
        import jax
        import jaxlib
        from jax.experimental import topologies
        from jax._src import cache_key, compilation_cache, distributed, xla_bridge
        from jax._src.lib import xla_client as xc
        report.update(jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
                      compiler_env=compiler_env, descriptor_count=sum(phases.values()),
                      unique_descriptor_count=len(groups), descriptors_by_phase=phases,
                      backends_before=list(xla_bridge._backends))
        if 'tpu' in xla_bridge._backends:
            raise RuntimeError('real TPU backend was already initialized')
        topology = topologies.get_topology_desc(platform='tpu', topology_name=topology_name)
        available = {v.id: v for v in topology.devices}
        backend = topology.devices[0].client
        report['virtual_backend'] = {
            'platform': backend.platform, 'platform_version': backend.platform_version,
            'client_type': f'{type(backend).__module__}.{type(backend).__name__}',
            'device_ids': list(available),
            'fingerprint': str(xc.get_topology_for_devices(list(topology.devices)).fingerprint())}
        report['backends_after_topology'] = list(xla_bridge._backends)
        if 'tpu' in xla_bridge._backends:
            raise RuntimeError('topology creation initialized real TPU backend')
        report['python_distributed_process_id'] = distributed.global_state.process_id
        # _get_cache_key's standard enabled-cache gate needs a configured path.
        # is_cache_used only checks configuration; no cache get/put is invoked.
        with tempfile.TemporaryDirectory(prefix='overlap-key-only-') as cache_dir:
            jax.config.update('jax_compilation_cache_dir', cache_dir)
            names = ('jax_enable_compilation_cache', 'jax_compilation_cache_dir',
                     'jax_compilation_cache_include_metadata_in_key',
                     'jax_remove_custom_partitioning_ptr_from_cache_key',
                     'jax_enable_pgle', 'jax_pgle_profiling_runs',
                     'jax_compilation_cache_expect_pgle', 'jax_use_shardy_partitioner')
            report['config_key_inputs'] = {name: jax.config.values.get(name) for name in names}
            report['compression'] = 'zstandard' if compilation_cache.zstandard is not None else 'zlib'
            report['custom_hook'] = cache_key.custom_hook()
            report['flag_prefixes'] = cache_key.get_flag_prefixes()
            report['argv_key_flags'] = [v for v in sys.argv if v.startswith('--xla') or
                                       any(v.startswith(p) for p in cache_key.get_flag_prefixes())]
            for group in groups:
                report['rows'].append(compare_one(group, available))
                if 'tpu' in xla_bridge._backends:
                    raise RuntimeError('key calculation initialized real TPU backend')
            report['temporary_cache_entries'] = [p.name for p in Path(cache_dir).iterdir()]
            if report['temporary_cache_entries']:
                raise RuntimeError('key-only probe unexpectedly wrote cache files')
        report['temporary_cache_removed'] = not Path(cache_dir).exists()
        report['source'] = source_evidence()
        report['backends_after'] = list(xla_bridge._backends)
        report['counts'] = dict(Counter(r['status'] for r in report['rows']))
        report['coverage_complete'] = len(report['rows']) == len(groups) == 261 and sum(phases.values()) == 1044
        report['all_keys_match'] = bool(report['rows']) and all(r['matches'] for r in report['rows'])
        report['ok'] = report['coverage_complete'] and report['all_keys_match'] and not report['backends_after']
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        signal.alarm(0)
        if 'jax._src.xla_bridge' in sys.modules:
            report['backends_after'] = list(sys.modules['jax._src.xla_bridge']._backends)
        report.update(ended=time.time(), wall_s=time.time()-started, process_cpu_s=time.process_time()-cpu)
    return report


def worker(root, topology):
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'TPU_SKIP_MDS_QUERY': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    command = [sys.executable, str(Path(__file__).resolve()), 'audit', '--bundle-root', root, '--topology', topology]
    try:
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=155)
        if result.returncode:
            return {'ok': False, 'error': f'key-only subprocess exit {result.returncode}',
                    'stderr': result.stderr[-8000:], 'stdout': result.stdout[-2000:]}
        report = json.loads(result.stdout)
        report['subprocess_stderr_tail'] = result.stderr[-8000:]
        return report
    except subprocess.TimeoutExpired as exc:
        return {'ok': False, 'error': str(exc), 'subprocess_killed': True}


def remote(args):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from hetero.evidence import emit_measurements
    ray.init(address='auto', ignore_reinit_error=True)
    nodes = sorted((n for n in ray.nodes() if n['Alive'] and n['Resources'].get('TPU', 0)),
                   key=lambda n: n['NodeManagerAddress'])
    if not nodes:
        raise RuntimeError('no live TPU worker host')
    task = ray.remote(num_cpus=1, max_calls=1, max_retries=0)(worker)
    ref = task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[0]['NodeID'], soft=False)).remote(args.bundle_root, args.topology)
    try:
        result = ray.get(ref, timeout=165)
        result['ray_node_id'] = nodes[0]['NodeID']
        emit_measurements(result)
    finally:
        ray.cancel(ref, force=True)
        ray.shutdown()


def submit(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    @stage(name='overlap_key_only', order_idx=1, expected_s=180)
    def run(state, timing):
        supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
        argv = ['python3', '-u', 'scripts/overlap_key_only.py', 'remote', '--bundle-root', args.bundle_root, '--topology', args.topology]
        job = supervisor.submit_job(entrypoint=shlex.join(argv), runtime_env={
            'working_dir': str(ROOT),
            'excludes': ['.venv*', '.git', '.jj', 'data', 'results', '**/results/**', 'reports', 'inputs', '__pycache__', '*.pyc'],
            'env_vars': {'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'JAX_PLATFORMS': 'cpu', 'TPU_SKIP_MDS_QUERY': '1'}},
            metadata={'purpose': 'overlap-key-only-no-compile'}, name='overlap-key-only')
        state.set('submission_id', job)
        try:
            result = supervisor.supervise_until_completion(submission_id=job, timeout_s=180,
                        stage_name='overlap_key_only', state_store=state)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'submission_id': job, 'status': result['status'],
                                  'measurements': result['measurements']}, indent=2)+'\n')
            # A key mismatch is a measured negative result, not a reason to retry.
            if not result['measurements']:
                raise RuntimeError('key-only job produced no measurements')
        finally:
            if not supervisor.get_job(job).terminal:
                supervisor.stop_job(job)
    runner = StageRunner(silence_timeout_s=180)
    runner.register(run)
    if not runner.run():
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('audit', 'remote', 'submit'))
    ap.add_argument('--bundle-root', default=BUNDLE, choices=(BUNDLE,))
    ap.add_argument('--topology', default='v5e:2x4', choices=('v5e:2x4',))
    ap.add_argument('--ray-url', default='http://127.0.0.1:8265')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.mode == 'audit':
        print(json.dumps(audit(args.bundle_root, args.topology, 145)))
    elif args.mode == 'remote':
        remote(args)
    else:
        if args.output is None:
            ap.error('submit requires --output')
        submit(args)


if __name__ == '__main__':
    main()
