"""Bounded synthetic caller-mmap donation probe; launch cluster mode through RCM."""
import argparse
import gc
import inspect
import json
import mmap
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time
import uuid
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def probe(root='/dev/shm', prefix='snapshot-helper-'):
    import jax
    import jaxlib
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    result = {'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'cases': [], 'source': {}}
    for obj in (jax.device_put, jax.jit):
        lines, start = inspect.getsourcelines(obj)
        result['source'][obj.__name__] = {'file': inspect.getsourcefile(obj), 'start': start,
                                        'text': ''.join(lines)}
    devices = jax.local_devices()
    result['devices'] = [str(d) for d in devices]
    client = devices[0].client
    result['client_transfer_names'] = [n for n in dir(client) if any(k in n.lower() for k in ('dma', 'buffer', 'copy', 'host'))]
    result['array_transfer_names'] = [n for n in dir(jax.device_put(np.zeros(1), devices[0])) if any(k in n.lower() for k in ('raw', 'buffer', 'copy', 'host'))]
    with tempfile.TemporaryDirectory(prefix=prefix, dir=root) as directory:
        for device in devices[:4]:
            for shape in ((1024,), (128, 128)):
                for register in (False, True):
                    row = {'device': device.id, 'shape': shape, 'dma_register': register}
                    result['cases'].append(row)
                    path = Path(directory) / 'destination.raw'
                    size = int(np.prod(shape)) * 4
                    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
                    os.ftruncate(fd, size)
                    os.posix_fallocate(fd, 0, size)
                    mapped = mmap.mmap(fd, size, flags=mmap.MAP_SHARED | mmap.MAP_POPULATE)
                    host = np.ndarray(shape, dtype=np.float32, buffer=mapped)
                    host.fill(-123)
                    pointer = int(host.ctypes.data)
                    registered = False
                    destination = output = source = None
                    try:
                        if register:
                            try:
                                client.dma_map(pointer, size)
                                registered = True
                                row['dma_map'] = 'ok'
                            except Exception as exc:
                                row['dma_map'] = repr(exc)
                                continue
                        mesh = Mesh(np.array([device]), ('x',))
                        hs = NamedSharding(mesh, PartitionSpec(), memory_kind='pinned_host')
                        ds = NamedSharding(mesh, PartitionSpec(), memory_kind='device')
                        destination = jax.device_put(host, hs, may_alias=True)
                        destination.block_until_ready()
                        row['mmap_pointer'] = pointer
                        try:
                            row['destination_pointer'] = int(destination.unsafe_buffer_pointer())
                            row['input_aliases_mmap'] = row['destination_pointer'] == pointer
                        except Exception as exc:
                            row['destination_pointer_error'] = repr(exc)
                        expected = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) + device.id * 65536
                        source = jax.device_put(expected, ds, may_alias=False)
                        source.block_until_ready()
                        fn = jax.jit(lambda dest, src: src, in_shardings=(hs, ds), out_shardings=hs,
                                     donate_argnums=(0,), keep_unused=True)
                        start = time.perf_counter()
                        compiled = fn.lower(destination, source).compile()
                        row['compile_s'] = time.perf_counter() - start
                        row['hlo'] = compiled.as_text()
                        start = time.perf_counter()
                        output = compiled(destination, source)
                        output.block_until_ready()
                        row['execute_ready_s'] = time.perf_counter() - start
                        try:
                            row['output_pointer'] = int(output.unsafe_buffer_pointer())
                            row['output_aliases_mmap'] = row['output_pointer'] == pointer
                        except Exception as exc:
                            row['output_pointer_error'] = repr(exc)
                        row['mmap_bytes_match_source'] = bool(np.array_equal(host, expected))
                        row['mmap_still_poison'] = bool(np.all(host == -123))
                        row['output_bytes_match_source'] = bool(np.array_equal(np.asarray(output), expected))
                        mapped.flush()
                        # Fresh mapping proves file bytes, not merely a returned runtime-owned view.
                        disk = np.fromfile(path, dtype=np.float32).reshape(shape)
                        row['reopened_file_matches_source'] = bool(np.array_equal(disk, expected))
                    except Exception as exc:
                        row['error'] = repr(exc)
                    finally:
                        for a in (output, destination, source):
                            if a is not None:
                                try:
                                    a.delete()
                                except Exception:
                                    pass
                        output = destination = source = None
                        gc.collect()
                        if registered:
                            client.dma_unmap(pointer)
                        del host
                        mapped.close()
                        os.close(fd)
    result['tmpfs_cleaned'] = not Path(directory).exists()
    return result


def remote(args):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
    from hetero.layout import resolve_layout, builtin_layout, LayoutRequest, InstanceRequest
    from hetero.slice_inventory import SliceInventory
    from hetero.evidence import emit_measurements
    inventory = SliceInventory.from_dict(json.loads(Path(args.inventory).read_text()))
    ip = resolve_layout(inventory, builtin_layout('S')).instances[0].hosts[0].node_ip
    instance = resolve_layout(inventory, LayoutRequest('PROBE4', (InstanceRequest('host4', 4, host_ips=(ip,)),))).instances[0]
    ray.init(address='auto', namespace='snapshot-helper')
    pg = actor = None
    result = {}
    prefix = 'snapshot-helper-' + uuid.uuid4().hex + '-'
    source = Path(__file__).read_text()
    @ray.remote(num_cpus=1, resources={'TPU': 4})
    class Controller:
        def run(self, env, text, prefix):
            os.environ.update(env)
            os.environ['TPU_VISIBLE_CHIPS'] = '0,1,2,3'
            os.environ.pop('JAX_COMPILATION_CACHE_DIR', None)
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                os.environ.pop(key, None)
            scope = {'__name__': 'snapshot_mmap_worker', '__file__': '/tmp/snapshot_mmap_probe.py'}
            exec(compile(text, scope['__file__'], 'exec'), scope)
            return scope['probe'](prefix=prefix)
    try:
        pg = placement_group([{'CPU': 1, 'TPU': 4, f'node:{ip}': .001}], name='snapshot-helper-mmap')
        ray.get(pg.ready(), timeout=30)
        actor = Controller.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=0), resources={'TPU': 4, f'node:{ip}': .001}).remote()
        result['probe'] = ray.get(actor.run.remote(instance.process_env(0), source, prefix), timeout=210)
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        if actor is not None:
            ray.kill(actor, no_restart=True)
        @ray.remote(num_cpus=0)
        def sweep(prefix):
            paths = list(Path('/dev/shm').glob(prefix + '*'))
            for path in paths:
                shutil.rmtree(path)
            return {'removed': [str(p) for p in paths], 'remaining': list(map(str, Path('/dev/shm').glob(prefix + '*')))}
        try:
            result['tmpfs_cleanup'] = ray.get(sweep.options(scheduling_strategy=NodeAffinitySchedulingStrategy(instance.hosts[0].ray_node_id, soft=False)).remote(prefix), timeout=15)
        except Exception as exc:
            result['tmpfs_cleanup_error'] = repr(exc)
        if pg is not None:
            remove_placement_group(pg)
        result['cleanup'] = {'actor_kill_requested': actor is not None, 'pg_remove_requested': pg is not None}
        emit_measurements(result)


def local(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    inputs = ROOT / '_snapshot_helper_inventory.json'
    @stage(name='snapshot_mmap_probe', order_idx=0, expected_s=280)
    def submit(state, timing):
        inputs.write_bytes(Path(args.inventory).read_bytes())
        try:
            supervisor = RayJobSupervisor(ray_url='http://127.0.0.1:8265', timing_engine=timing)
            command = shlex.join(['python3', '-u', 'scripts/snapshot_mmap_probe.py', '--remote', '--inventory', inputs.name])
            submission = supervisor.submit_job(entrypoint=command, runtime_env={
                'working_dir': str(ROOT), 'excludes': ['.venv*', '.git', '.jj', 'data', 'results', '**/results/**', 'formalization', 'reports', '__pycache__', '*.pyc'],
                'env_vars': {'PYTHONUNBUFFERED': '1'}}, name='snapshot-helper-mmap')
            state.set('submission_id', submission)
            result = supervisor.supervise_until_completion(submission_id=submission, timeout_s=280,
                stage_name='snapshot_mmap_probe', state_store=state)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'submission_id': submission, **result}, indent=2, default=str) + '\n')
        finally:
            inputs.unlink(missing_ok=True)
    runner = StageRunner(silence_timeout_s=290)
    runner.register(submit)
    if not runner.run():
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--remote', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--inventory', default='/tmp/hetero-inventory-only.json')
    p.add_argument('--output', type=Path, default=ROOT / 'results/restore/SnapshotHelper/mmap-probe.json')
    args = p.parse_args()
    if args.cpu:
        print(json.dumps(probe('/tmp'), indent=2))
    elif args.remote:
        remote(args)
    else:
        local(args)
