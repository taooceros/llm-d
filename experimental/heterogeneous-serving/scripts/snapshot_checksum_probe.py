"""Bounded checksum-only real TP8 shard-shape probe, launched only via RCM."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def catalogue(raw):
    rows = []
    for item in json.loads(raw)['parameters']:
        if item['dtype'] != 'bfloat16':
            raise ValueError('only BF16 catalogue supported')
        node = ast.parse(item['partition_spec'], mode='eval').body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'P' or node.keywords:
            raise ValueError('unexpected PartitionSpec')
        shape = list(item['shape'])
        for axis, part in enumerate(node.args):
            value = ast.literal_eval(part)
            if value == 'model':
                if shape[axis] % 8:
                    raise ValueError('non-divisible TP8 dimension')
                shape[axis] //= 8
            elif value is not None:
                raise ValueError('unsupported partition axis')
        rows.append({'path': item['path'], 'shape': tuple(shape), 'elements': math.prod(shape)})
    return rows


def expected_checksum(elements, salt):
    import numpy as np
    import ml_dtypes
    period = ((np.arange(251, dtype=np.uint32) + salt) % 251).astype(ml_dtypes.bfloat16).view(np.uint16)
    cycles, remain = divmod(elements, 251)
    total = (int(period.sum(dtype=np.uint64)) * cycles + int(period[:remain].sum(dtype=np.uint64))) & 0xffffffff
    xor = int(np.bitwise_xor.reduce(period, initial=np.uint16(0))) if cycles % 2 else 0
    xor ^= int(np.bitwise_xor.reduce(period[:remain], initial=np.uint16(0)))
    return [total, xor]


def checksum(array):
    import jax
    import jax.numpy as jnp
    bits = jax.lax.bitcast_convert_type(array, jnp.uint16).reshape(-1).astype(jnp.uint32)
    return jnp.stack((jnp.sum(bits, dtype=jnp.uint32),
                      jax.lax.reduce(bits, jnp.uint32(0), jax.lax.bitwise_xor, (0,))))


def probe(raw, cpu=False):
    import jax
    import jax.numpy as jnp
    import jaxlib
    import numpy as np
    from jax.sharding import SingleDeviceSharding
    rows = catalogue(raw)
    full_bytes = sum(r['elements'] * 2 for r in rows)
    if len(rows) != 1028 or full_bytes != 7862118616:
        raise ValueError(f'catalogue mismatch: {len(rows)} shards / {full_bytes} bytes per chip')
    real_unique = len({r['shape'] for r in rows})
    if cpu:
        # Exercise each real rank/shape family at bounded CPU sizes, plus wrap/empty cases separately.
        rows = [{'path': r['path'], 'shape': tuple(min(d, 17) for d in r['shape']),
                 'elements': math.prod(min(d, 17) for d in r['shape'])} for r in rows]
    devices = jax.local_devices()[:1 if cpu else 4]
    if not cpu and len(devices) != 4:
        raise ValueError('expected exactly four local chips')
    result = {'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'cpu_smoke': cpu,
              'catalogue_sha256': hashlib.sha256(raw).hexdigest(), 'catalogue_shards_per_chip': 1028,
              'catalogue_bytes_per_chip': full_bytes, 'catalogue_unique_shapes': real_unique,
              'devices': list(map(str, devices)), 'algorithm': 'sum of uint16 BF16 bit patterns modulo 2^32, plus xor of the same bits; output uint32[2]',
              'limits': 'Noncryptographic and position-insensitive: all single-bit flips change sum, but permutations and some compensating changes can collide. No physical-bandwidth claim.',
              'runs': []}
    sources, jobs, expected = [], [], []
    generators, reducers = {}, {}
    try:
        start = time.perf_counter()
        for chip, device in enumerate(devices):
            sharding = SingleDeviceSharding(device)
            for shape in dict.fromkeys(r['shape'] for r in rows):
                def generate(salt, shape=shape):
                    values = jnp.arange(math.prod(shape), dtype=jnp.uint32).reshape(shape)
                    return ((values + salt) % jnp.uint32(251)).astype(jnp.bfloat16)
                scalar = jax.ShapeDtypeStruct((), np.uint32)
                abstract = jax.ShapeDtypeStruct(shape, jnp.bfloat16, sharding=sharding)
                generators[chip, shape] = jax.jit(generate, in_shardings=sharding, out_shardings=sharding).lower(scalar).compile()
                reducers[chip, shape] = jax.jit(checksum, in_shardings=sharding, out_shardings=sharding).lower(abstract).compile()
        result['compile_s'] = time.perf_counter() - start
        start = time.perf_counter()
        for chip, device in enumerate(devices):
            salts = [jax.device_put(np.uint32((index * 31 + chip * 17 + 13) % 251), device) for index in range(len(rows))]
            for index, row in enumerate(rows):
                source = generators[chip, row['shape']](salts[index])
                sources.append(source)
                jobs.append(reducers[chip, row['shape']])
                expected.append(expected_checksum(row['elements'], (index * 31 + chip * 17 + 13) % 251))
        jax.block_until_ready(sources)
        result['source_generation_ready_s'] = time.perf_counter() - start
        result['source_arrays'] = len(sources)
        result['source_bytes'] = sum(r['elements'] * 2 for r in rows) * len(devices)
        result['source_python_cache_none_before'] = sum(s._npy_value is None for s in sources)
        # Two repetitions of the same checksum-only case; no matrix or extra one-chip native run.
        for repetition in range(2):
            start = time.perf_counter()
            outputs = [fn(source) for fn, source in zip(jobs, sources)]
            submitted = time.perf_counter()
            jax.block_until_ready(outputs)
            ready = time.perf_counter()
            host = [np.asarray(output).tolist() for output in outputs]
            readback = time.perf_counter()
            mismatches = [i for i, (got, want) in enumerate(zip(host, expected)) if got != want]
            result['runs'].append({'repetition': repetition, 'enqueue_s': submitted - start,
                                   'kernel_ready_wall_s_including_enqueue': ready - start,
                                   'wait_after_enqueue_s': ready - submitted,
                                   'small_output_readback_s': readback - ready,
                                   'checksum_outputs': len(host), 'checksum_output_bytes': len(host) * 8,
                                   'verified_outputs': len(host) - len(mismatches), 'mismatch_indices': mismatches,
                                   'checksum_output_sha256': hashlib.sha256(np.asarray(host, dtype=np.uint32).tobytes()).hexdigest()})
            for output in outputs:
                output.delete()
            if mismatches:
                raise AssertionError('analytic checksum verification failed')
        result['source_python_cache_none_after'] = sum(s._npy_value is None for s in sources)
        result['ok'] = True
    except Exception as exc:
        result['error'] = repr(exc)
        result['ok'] = False
    finally:
        for source in sources:
            source.delete()
    return result


def cpu_checks(raw):
    import jax
    import numpy as np
    import ml_dtypes
    fn = jax.jit(checksum)
    for size in (0, 1, 250, 251, 252, 502, 100003):
        for salt in (0, 1, 250):
            a = ((np.arange(size, dtype=np.uint32) + salt) % 251).astype(ml_dtypes.bfloat16)
            assert np.asarray(fn(a)).tolist() == expected_checksum(size, salt)
    bits = np.arange(512, dtype=np.uint16)
    a = bits.view(ml_dtypes.bfloat16)
    original = np.asarray(fn(a))
    changed = bits.copy()
    changed[17] ^= 1
    assert not np.array_equal(original, np.asarray(fn(changed.view(ml_dtypes.bfloat16))))
    assert np.array_equal(original, np.asarray(fn(a[::-1].copy())))
    result = probe(raw, cpu=True)
    result['cpu_edge_checks'] = '21 finite-period cases, a single-bit mutation detected, reversal collision documented'
    return result


def remote(args):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from hetero.layout import resolve_layout, builtin_layout, LayoutRequest, InstanceRequest
    from hetero.slice_inventory import SliceInventory
    from hetero.evidence import emit_measurements
    inventory = SliceInventory.from_dict(json.loads(Path(args.inventory).read_text()))
    raw = Path(args.catalogue).read_bytes()
    source = Path(__file__).read_text()
    ip = resolve_layout(inventory, builtin_layout('S')).instances[0].hosts[0].node_ip
    instance = resolve_layout(inventory, LayoutRequest('CHK4', (InstanceRequest('host4', 4, host_ips=(ip,)),))).instances[0]
    ray.init(address='auto', namespace='snapshot-helper-checksum')
    result = {'source_sha256': hashlib.sha256(source.encode()).hexdigest()}
    pg = actor = None
    @ray.remote(num_cpus=1, resources={'TPU': 4})
    class Controller:
        def run(self, env, source, raw):
            os.environ.update(env)
            os.environ['TPU_VISIBLE_CHIPS'] = '0,1,2,3'
            os.environ.pop('JAX_COMPILATION_CACHE_DIR', None)
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                os.environ.pop(key, None)
            scope = {'__name__': 'checksum_worker', '__file__': '/tmp/snapshot_checksum_probe.py'}
            exec(compile(source, scope['__file__'], 'exec'), scope)
            return scope['probe'](raw)
    try:
        pg = placement_group([{'CPU': 1, 'TPU': 4, f'node:{ip}': .001}], name='snapshot-helper-checksum')
        ray.get(pg.ready(), timeout=15)
        actor = Controller.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=0), resources={'TPU': 4, f'node:{ip}': .001}).remote()
        result['probe'] = ray.get(actor.run.remote(instance.process_env(0), source, raw), timeout=145)
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        try:
            if actor is not None:
                ray.kill(actor, no_restart=True)
        finally:
            if pg is not None:
                remove_placement_group(pg)
        result['cleanup'] = {'actor_kill_requested': actor is not None, 'pg_remove_requested': pg is not None, 'tmpfs_files_created': False}
        emit_measurements(result)


def local(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    inputs = ROOT / '_snapshot_checksum_inputs'
    @stage(name='snapshot_checksum_probe', order_idx=0, expected_s=170)
    def submit(state, timing):
        inputs.mkdir(exist_ok=False)
        try:
            (inputs / 'inventory.json').write_bytes(Path(args.inventory).read_bytes())
            (inputs / 'catalogue.json').write_bytes(Path(args.catalogue).read_bytes())
            supervisor = RayJobSupervisor(ray_url='http://127.0.0.1:8265', timing_engine=timing)
            command = shlex.join(['python3', '-u', 'scripts/snapshot_checksum_probe.py', '--remote',
                                 '--inventory', str(inputs.relative_to(ROOT) / 'inventory.json'),
                                 '--catalogue', str(inputs.relative_to(ROOT) / 'catalogue.json')])
            submission = supervisor.submit_job(entrypoint=command, runtime_env={
                'working_dir': str(ROOT), 'excludes': ['.venv*', '.git', '.jj', 'data', 'results', '**/results/**', 'formalization', 'reports', '__pycache__', '*.pyc'],
                'env_vars': {'PYTHONUNBUFFERED': '1'}}, name='snapshot-helper-checksum')
            state.set('submission_id', submission)
            result = supervisor.supervise_until_completion(submission_id=submission, timeout_s=170,
                stage_name='snapshot_checksum_probe', state_store=state)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'submission_id': submission, **result}, indent=2, default=str) + '\n')
        finally:
            shutil.rmtree(inputs)
    runner = StageRunner(silence_timeout_s=175)
    runner.register(submit)
    if not runner.run():
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--remote', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--inventory', default='/tmp/hetero-inventory-only.json')
    p.add_argument('--catalogue', required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'results/restore/SnapshotHelper/checksum-probe.json')
    args = p.parse_args()
    if args.cpu:
        result = cpu_checks(Path(args.catalogue).read_bytes())
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['ok'] else 1)
    elif args.remote:
        remote(args)
    else:
        local(args)
