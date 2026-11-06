"""Synthetic fresh-D2H versus preallocated pinned-host donation; RCM-only cluster run."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def probe(cpu=False):
    import jax
    import jax.numpy as jnp
    import jaxlib
    import numpy as np
    import ml_dtypes
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    devices = jax.local_devices()[:4]
    result = {'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'devices': list(map(str, devices)),
              'cases': [], 'timing': 'source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included',
              'scope': 'runtime-owned pinned host, NOT caller mmap; synthetic BF16, no physical ceiling claim'}
    sizes = (2**20, 4 * 2**20) if cpu else (256 * 2**20, 2**30)
    counts = (1,) if cpu else (1, 4)
    began = time.monotonic()
    salt_id = 0
    for size in sizes:
        for count in counts:
            chips = devices[:count]
            shape = (size // 2048, 1024)
            generators = []
            shardings = []
            for dev in chips:
                mesh = Mesh(np.array([dev]), ('x',))
                ds = NamedSharding(mesh, PartitionSpec(), memory_kind='device')
                hs = NamedSharding(mesh, PartitionSpec(), memory_kind='pinned_host')
                def generate(salt):
                    values = jnp.arange(size // 2, dtype=jnp.uint32).reshape(shape)
                    return ((values + salt) % jnp.uint32(251)).astype(jnp.bfloat16)
                generators.append(jax.jit(generate, out_shardings=ds))
                shardings.append((ds, hs))
            for mechanism in ('async_native_collector', 'preallocated_pinned_donation'):
                destinations, compiled = [], []
                try:
                    if mechanism == 'preallocated_pinned_donation':
                        # Nonzero initialization writes/faults every destination page before timing.
                        poison = np.full(shape, -17, dtype=ml_dtypes.bfloat16)
                        for ds, hs in shardings:
                            destinations.append(jax.device_put(poison, hs, may_alias=False))
                        jax.block_until_ready(destinations)
                        del poison
                    for repetition in range(2):
                        if time.monotonic() - began > 125:
                            result['budget_stop'] = True
                            return result
                        salt_id += 1
                        salts = [salt_id * 11 + c * 19 for c in range(count)]
                        sources = [fn(np.uint32(s)) for fn, s in zip(generators, salts)]
                        jax.block_until_ready(sources)
                        row = {'mechanism': mechanism, 'bytes_per_chip': size, 'chips': count,
                               'bytes': size * count, 'repetition': repetition, 'salts': salts,
                               'source_cache_none_before': [s._npy_value is None for s in sources],
                               'source_pointers': [int(s.unsafe_buffer_pointer()) for s in sources]}
                        result['cases'].append(row)
                        outputs, hosts = [], []
                        if mechanism == 'preallocated_pinned_donation' and not compiled:
                            start = time.perf_counter()
                            for dest, src, (ds, hs) in zip(destinations, sources, shardings):
                                fn = jax.jit(lambda dst, src: src, in_shardings=(hs, ds), out_shardings=hs,
                                             keep_unused=True, donate_argnums=(0,))
                                compiled.append(fn.lower(dest, src).compile())
                            row['compile_s_outside_timer'] = time.perf_counter() - start
                            row['hlo_alias_headers'] = [fn.as_text().splitlines()[0] for fn in compiled]
                        if mechanism == 'preallocated_pinned_donation':
                            pointers = [int(d.unsafe_buffer_pointer()) for d in destinations]
                            start = time.perf_counter()
                            outputs = [fn(dst, src) for fn, dst, src in zip(compiled, destinations, sources)]
                            jax.block_until_ready(outputs)
                            elapsed = time.perf_counter() - start
                            row['destination_pointers_before'] = pointers
                            row['output_pointers'] = [int(a.unsafe_buffer_pointer()) for a in outputs]
                            row['destination_reused'] = [a == b for a, b in zip(pointers, row['output_pointers'])]
                            destinations = outputs
                            for output in outputs:
                                host, did_copy = output._single_device_array_to_np_array_did_copy()
                                hosts.append(host)
                                row.setdefault('posttimer_host_view_did_copy', []).append(bool(did_copy))
                            row['host_view_aliases_output'] = [int(h.ctypes.data) == p for h, p in zip(hosts, row['output_pointers'])]
                        else:
                            start = time.perf_counter()
                            for source in sources:
                                source.copy_to_host_async()
                            flags = []
                            for source in sources:
                                host, did_copy = source._single_device_array_to_np_array_did_copy()
                                hosts.append(host)
                                flags.append(bool(did_copy))
                            elapsed = time.perf_counter() - start
                            row['collector_did_copy'] = flags
                        row['seconds'] = elapsed
                        row['GBps'] = size * count / elapsed / 1e9
                        row['source_cache_none_after'] = [s._npy_value is None for s in sources]
                        row['host_strides'] = [list(h.strides) for h in hosts]
                        verify_start = time.perf_counter()
                        verified = 0
                        for host, salt in zip(hosts, salts):
                            for start_row in range(0, shape[0], 512):
                                stop = min(start_row + 512, shape[0])
                                expected = ((np.arange(start_row * 1024, stop * 1024, dtype=np.uint32) + np.uint32(salt)) % np.uint32(251)).astype(ml_dtypes.bfloat16).reshape(stop - start_row, 1024)
                                if not np.array_equal(host[start_row:stop].view(np.uint16), expected.view(np.uint16)):
                                    raise AssertionError('fresh source logical BF16 bytes mismatch')
                                verified += expected.nbytes
                        row['verified_bytes'] = verified
                        row['verification_s_outside_timer'] = time.perf_counter() - verify_start
                        row['ok'] = verified == row['bytes']
                        # Do not leave host views alive across the next donation.
                        del host, expected
                        hosts.clear()
                        outputs = []
                        for source in sources:
                            source.delete()
                        sources.clear()
                        gc.collect()
                except Exception as exc:
                    result.setdefault('errors', []).append({'mechanism': mechanism, 'size': size, 'chips': count, 'error': repr(exc)})
                finally:
                    for dest in destinations:
                        try:
                            dest.delete()
                        except Exception:
                            pass
                    destinations.clear()
                    compiled.clear()
                    gc.collect()
    result['elapsed_s'] = time.monotonic() - began
    return result


def remote(args):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from hetero.layout import resolve_layout, builtin_layout, LayoutRequest, InstanceRequest
    from hetero.slice_inventory import SliceInventory
    from hetero.evidence import emit_measurements
    inventory = SliceInventory.from_dict(json.loads(Path(args.inventory).read_text()))
    ip = resolve_layout(inventory, builtin_layout('S')).instances[0].hosts[0].node_ip
    instance = resolve_layout(inventory, LayoutRequest('PIN4', (InstanceRequest('host4', 4, host_ips=(ip,)),))).instances[0]
    ray.init(address='auto', namespace='snapshot-helper-pinned')
    pg = actor = None
    result = {}
    source = Path(__file__).read_text()
    result['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    @ray.remote(num_cpus=1, resources={'TPU': 4})
    class Controller:
        def run(self, env, text):
            os.environ.update(env)
            os.environ['TPU_VISIBLE_CHIPS'] = '0,1,2,3'
            os.environ.pop('JAX_COMPILATION_CACHE_DIR', None)
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                os.environ.pop(key, None)
            scope = {'__name__': 'snapshot_pinned_worker', '__file__': '/tmp/snapshot_pinned_bandwidth.py'}
            exec(compile(text, scope['__file__'], 'exec'), scope)
            return scope['probe']()
    try:
        pg = placement_group([{'CPU': 1, 'TPU': 4, f'node:{ip}': .001}], name='snapshot-helper-pinned')
        ray.get(pg.ready(), timeout=15)
        actor = Controller.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=0), resources={'TPU': 4, f'node:{ip}': .001}).remote()
        result['probe'] = ray.get(actor.run.remote(instance.process_env(0), source), timeout=145)
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
    inputs = ROOT / '_snapshot_pinned_inventory.json'
    @stage(name='snapshot_pinned_bandwidth', order_idx=0, expected_s=175)
    def submit(state, timing):
        inputs.write_bytes(Path(args.inventory).read_bytes())
        try:
            supervisor = RayJobSupervisor(ray_url='http://127.0.0.1:8265', timing_engine=timing)
            command = shlex.join(['python3', '-u', 'scripts/snapshot_pinned_bandwidth.py', '--remote', '--inventory', inputs.name])
            submission = supervisor.submit_job(entrypoint=command, runtime_env={
                'working_dir': str(ROOT), 'excludes': ['.venv*', '.git', '.jj', 'data', 'results', '**/results/**', 'formalization', 'reports', '__pycache__', '*.pyc'],
                'env_vars': {'PYTHONUNBUFFERED': '1'}}, name='snapshot-helper-pinned')
            state.set('submission_id', submission)
            result = supervisor.supervise_until_completion(submission_id=submission, timeout_s=175,
                stage_name='snapshot_pinned_bandwidth', state_store=state)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'submission_id': submission, **result}, indent=2, default=str) + '\n')
        finally:
            inputs.unlink(missing_ok=True)
    runner = StageRunner(silence_timeout_s=180)
    runner.register(submit)
    if not runner.run():
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--remote', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--inventory', default='/tmp/hetero-inventory-only.json')
    p.add_argument('--output', type=Path, default=ROOT / 'results/restore/SnapshotHelper/pinned-bandwidth.json')
    args = p.parse_args()
    if args.cpu:
        print(json.dumps(probe(cpu=True), indent=2))
    elif args.remote:
        remote(args)
    else:
        local(args)
