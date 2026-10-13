#!/usr/bin/env python3
"""Read-only installed-JAX source + abstract-topology AOT compile feasibility probe.

Establishes, from the *installed* JAX (worker image JAX 0.10.2), whether a next-layout
Gemma4 program can be compiled ahead of time against a ``TopologyDescription`` without
opening the TPU backend, and what the persistent-cache key depends on.

Never initialises a TPU client, never reserves a TPU resource, never writes to shared
paths.  Output is one machine-readable JSON document.

    python3 scripts/overlap_source_probe.py local --output /tmp/probe.json   # this host only
    python3 scripts/overlap_source_probe.py submit --output results/overlap/source_probe.json
        # RCM-supervised Ray CPU task on a worker node (operator runs this, in sequence)
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# file (relative to the jax package) -> symbols whose source is cited with numbered lines.
SOURCE_SYMBOLS = {
    '_src/compiler.py': ['get_compile_options', 'backend_compile_and_load', 'compile_or_get_cached',
                         '_resolve_compilation_strategy', '_get_cache_key', '_cache_read', '_cache_write'],
    '_src/cache_key.py': ['get', '_hash_devices', '_hash_accelerator_config', '_hash_platform',
                          '_hash_serialized_compile_options'],
    '_src/compilation_cache.py': ['is_cache_used', 'get_cache_key', 'put_executable_and_time',
                                  'get_executable_and_time'],
    'experimental/topologies.py': ['TopologyDescription', 'get_topology_desc', 'make_mesh'],
    '_src/xla_bridge.py': ['make_pjrt_tpu_topology', 'make_pjrt_topology'],
    '_src/stages.py': ['Lowered'],
    "_src/dispatch.py": ["LogElapsedTimeContextManager"],
    "_src/distributed.py": ["State"],
}
# line-level facts: (file, regex) -> numbered matches (process-0 / device-assignment restrictions)
SOURCE_PATTERNS = {
    '_src/compiler.py': r'process_id|process_index|share_binary|compile_only|cross_compile',
    '_src/cache_key.py': r'device_assignment|_hash_devices|_hash_accelerator_config|fingerprint|process_index',
    '_src/compilation_cache.py': r'process_index|process_id|supports_executable_serialization|compile_only',
}


def jax_root():
    spec = importlib.util.find_spec('jax')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('jax is not installed')
    return Path(list(spec.submodule_search_locations)[0])


def numbered(lines, start, end):
    return [f'{n}:{lines[n - 1]}' for n in range(start, end + 1)]


def inspect_source(root):
    """Static (ast + regex) inspection; imports nothing from jax."""
    import re
    out = {'files': {}, 'symbols': {}, 'patterns': {}}
    targets = [(rel, root / rel, names) for rel, names in SOURCE_SYMBOLS.items()]
    from importlib.util import find_spec
    spec = find_spec("tpu_inference")
    if spec is not None and spec.origin:
        package = Path(spec.origin).parent
        for rel, names in {
            "runner/compilation_manager.py": ["_flush_compilations", "capture_model", "_run_compilation"],
            "runner/kv_cache_manager.py": ["initialize_kv_cache", "create_kv_caches"],
            "worker/tpu_worker.py": ["compile_or_warm_up_model", "initialize_from_config", "init_device"],
        }.items():
            targets.append(("tpu_inference/" + rel, package / rel, names))
    for rel, path, symbols in targets:
        if not path.exists():
            out['files'][rel] = {'missing': True}
            continue
        text = path.read_text()
        lines = text.splitlines()
        out['files'][rel] = {'sha256': hashlib.sha256(text.encode()).hexdigest(), 'lines': len(lines)}
        tree = ast.parse(text)
        defs = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs.setdefault(node.name, node)
        for name in symbols:
            node = defs.get(name)
            key = f'{rel}:{name}'
            if node is None:
                out['symbols'][key] = {'missing': True}
                continue
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            out['symbols'][key] = {'start': start, 'end': node.end_lineno,
                                   'source': numbered(lines, start, node.end_lineno)}
    for rel, pattern in SOURCE_PATTERNS.items():
        path = root / rel
        if path.exists():
            rx = re.compile(pattern)
            out['patterns'][rel] = [f'{n}:{line}' for n, line in enumerate(path.read_text().splitlines(), 1)
                                    if rx.search(line)]
    return out


def versions():
    info = {'python': sys.version, 'platform': platform.platform(), 'pid': os.getpid()}
    for dist in ('jax', 'jaxlib', 'libtpu', 'libtpu-nightly', 'ray'):
        try:
            from importlib import metadata
            info[dist] = metadata.version(dist)
        except Exception:
            info[dist] = None
    return info


def step(report, name, fn):
    try:
        report[name] = {'ok': True, 'value': fn()}
    except BaseException as error:  # evidence collection: record every failure mode
        report[name] = {'ok': False, 'error': f'{type(error).__name__}: {error}',
                        'traceback': traceback.format_exc().splitlines()[-12:]}
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise


def toy_program(jax, jnp, mesh, P):
    """Tensor-parallel MLP stand-in: sharded matmul + all-reduce, enough to exercise SPMD partitioning."""
    from jax.sharding import NamedSharding
    axis = mesh.axis_names[-1]
    x = jax.ShapeDtypeStruct((16, 1024), jnp.bfloat16, sharding=NamedSharding(mesh, P(None, None)))
    w1 = jax.ShapeDtypeStruct((1024, 4096), jnp.bfloat16, sharding=NamedSharding(mesh, P(None, axis)))
    w2 = jax.ShapeDtypeStruct((4096, 1024), jnp.bfloat16, sharding=NamedSharding(mesh, P(axis, None)))

    @jax.jit
    def fn(x, w1, w2):
        return jax.nn.gelu(x @ w1) @ w2
    return fn, (x, w1, w2)


def aot_probe(topology_name, mesh_shape, cache_probe, platform_name='tpu'):
    """Build abstract topology devices and lower+compile for several device assignments."""
    # CPU/offline only: no TPU client may be created.  Must precede first jax use.
    prior_platforms = os.environ.get('JAX_PLATFORMS')
    os.environ['JAX_PLATFORMS'] = 'cpu'  # unconditional: a caller's tpu setting would let cross-compile open the TPU
    if platform_name == 'attached-cpu':
        os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=8')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, PartitionSpec as P
    from jax.experimental import topologies
    from jax._src import compilation_cache, xla_bridge

    report = {'jax_platforms_overridden_from': prior_platforms, 'topology_name': topology_name, 'platform': platform_name, 'mesh_shape': list(mesh_shape)}
    state = {}

    def make_topology():
        if platform_name == 'attached-cpu':  # machinery smoke only: real CPU devices, not an abstract topology
            topo = topologies.get_attached_topology('cpu')
        else:
            topo = topologies.get_topology_desc(platform=platform_name, topology_name=topology_name)
        state['topo'] = topo
        devices = list(topo.devices)
        return {'device_count': len(devices),
                'devices': [{'id': d.id, 'kind': d.device_kind, 'process_index': d.process_index,
                             'platform': d.platform, 'coords': list(getattr(d, 'coords', []) or [])}
                            for d in devices],
                'process_indices': sorted({d.process_index for d in devices}),
                'serialized_topology_sha256': _safe(lambda: hashlib.sha256(topo.serialize()).hexdigest())}

    step(report, 'topology', make_topology)
    report['tpu_backend_initialized_after_topology'] = ('cpu' if platform_name == 'attached-cpu' else platform_name) in _initialized_backends(xla_bridge)
    if 'topo' not in state:
        report['feasible'] = False
        return report

    devices = list(state['topo'].devices)
    count = int(np.prod(mesh_shape))
    variants = {
        'full': devices[:count],
        'first_half': devices[:count // 2],
        'second_half': devices[count // 2:count],
        'reversed': devices[:count][::-1],
    }
    captured = []
    original = compilation_cache.get_cache_key

    def recording(module, devs, options, backend, *args, **kwargs):
        key = original(module, devs, options, backend, *args, **kwargs)
        captured.append({'key': key, 'device_ids': [d.id for d in np.asarray(devs).flat],
                         'process_indices': sorted({d.process_index for d in np.asarray(devs).flat}),
                         'backend_platform': backend.platform,
                         'backend_platform_version': str(getattr(backend, 'platform_version', None))})
        return key

    compilation_cache.get_cache_key = recording
    tmp = None
    if cache_probe:
        import tempfile
        tmp = tempfile.mkdtemp(prefix='overlap_probe_cache_')
        jax.config.update('jax_compilation_cache_dir', tmp)
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    report['persistent_cache_dir_private_tmp'] = tmp
    results = {}
    try:
        for name, devs in variants.items():
            shape = (len(devs) // mesh_shape[-1], mesh_shape[-1]) if len(devs) % mesh_shape[-1] == 0 \
                else (1, len(devs))
            row = {}
            results[name] = row

            def run(devs=devs, shape=shape):
                row = {}
                mesh = Mesh(np.asarray(devs, dtype=object).reshape(shape), ('data', 'model'))
                fn, args = toy_program(jax, jnp, mesh, P)
                before = len(captured)
                lowered = fn.lower(*args)
                row['stablehlo_sha256'] = hashlib.sha256(lowered.as_text().encode()).hexdigest()
                row['lowering_device_ids'] = [d.id for d in mesh.devices.flat]
                compiled = lowered.compile()
                row['compiled'] = True
                row['cache_key_calls'] = captured[before:]
                row['hlo_text_sha256'] = _safe(lambda: hashlib.sha256(compiled.as_text().encode()).hexdigest())
                row['memory_analysis'] = _safe(lambda: str(compiled.memory_analysis()))
                row['serialize_executable'] = _safe(lambda: _serialized_size(jax, compiled))
                return row

            step(row, 'compile', run)
    finally:
        compilation_cache.get_cache_key = original
    report['variants'] = results
    keys = {n: [c['key'] for c in r.get('compile', {}).get('value', {}).get('cache_key_calls', [])]
            for n, r in results.items()}
    report['cache_keys'] = keys
    flat = {n: k[0] for n, k in keys.items() if k and k[0]}
    report['cache_key_equalities'] = {
        'first_half_vs_second_half': _eq(flat, 'first_half', 'second_half'),
        'full_vs_reversed': _eq(flat, 'full', 'reversed')}
    if tmp:
        report['persistent_cache_entries'] = sorted(os.listdir(tmp))
    report['feasible'] = all(r.get('compile', {}).get('ok') for r in results.values())
    report['tpu_backend_initialized_after_compile'] = ('cpu' if platform_name == 'attached-cpu' else platform_name) in _initialized_backends(xla_bridge)
    return report


def _serialized_size(jax, compiled):
    from jax.experimental import serialize_executable
    payload = serialize_executable.serialize(compiled)[0]
    return len(payload)


def _eq(flat, a, b):
    return None if a not in flat or b not in flat else flat[a] == flat[b]


def _safe(fn):
    try:
        return fn()
    except Exception as error:
        return f'unavailable: {type(error).__name__}: {error}'


def _initialized_backends(xla_bridge):
    try:
        return sorted(getattr(xla_bridge, '_backends', {}).keys())
    except Exception:
        return None


def probe(args):
    report = {'versions': versions(), 'host': platform.node()}
    step(report, 'jax_root', lambda: str(jax_root()))
    if report['jax_root']['ok']:
        step(report, 'source', lambda: inspect_source(Path(report['jax_root']['value'])))
    if not args.skip_aot:
        step(report, 'aot', lambda: aot_probe(args.topology, tuple(args.mesh), args.cache_probe, args.platform))
    report['ok'] = all(v.get('ok') for k, v in report.items() if isinstance(v, dict) and 'ok' in v) and (
        args.skip_aot or report['aot'].get('value', {}).get('feasible') is True)
    return report


def remote(args):
    """Runs inside the Ray job: one CPU-only task pinned to a live TPU worker node (no TPU resource)."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from hetero.evidence import emit_measurements
    ray.init(address='auto', ignore_reinit_error=True)
    nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('TPU', 0) > 0]
    if not nodes:
        raise RuntimeError('no live TPU worker node')
    node = sorted(nodes, key=lambda n: n['NodeID'])[0]
    task = ray.remote(num_cpus=0, resources={})(probe)
    result = ray.get(task.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node['NodeID'], soft=False),
        runtime_env={'env_vars': {'JAX_PLATFORMS': 'cpu', 'TPU_SKIP_MDS_QUERY': '1'}}).remote(args),
        timeout=args.timeout)
    result['node_id'] = node['NodeID']
    emit_measurements(result)


def submit(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    argv = ['python3', '-u', 'scripts/overlap_source_probe.py', 'remote', '--topology', args.topology,
            '--platform', args.platform, '--mesh', *map(str, args.mesh), '--timeout', str(args.timeout)]
    if args.cache_probe:
        argv.append('--cache-probe')
    if args.skip_aot:
        argv.append('--skip-aot')

    @stage(name='overlap_source_probe', order_idx=1, expected_s=args.timeout)
    def run(state, timing):
        supervisor = state.get('supervisor')
        if supervisor is None:
            supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
            state.set('supervisor', supervisor)
        job = state.get('submission_id')
        if job is None:
            job = supervisor.submit_job(entrypoint=shlex.join(argv), runtime_env={
                'working_dir': str(ROOT), 'excludes': ['.venv*', '.git', '.jj', '__pycache__',
                '*.pyc', 'results', '**/results/**', 'reports', 'data'],
                'env_vars': {'PYTHONUNBUFFERED': '1', 'JAX_PLATFORMS': 'cpu'}},
                metadata={'purpose': 'overlap-source-probe'}, name='overlap-source-probe')
            state.set('submission_id', job)
        result = supervisor.supervise_until_completion(submission_id=job, timeout_s=args.timeout,
                    stage_name='overlap_source_probe', state_store=state)
        evidence = {'submission_id': job, 'status': result['status'], 'measurements': result['measurements']}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + '\n')
        if not result['measurements']:
            raise RuntimeError('probe job produced no evidence')
    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(run)
    if not runner.run():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('local', 'submit', 'remote'))
    parser.add_argument('--topology', default='v5e:2x4')
    parser.add_argument('--platform', default='tpu', help='abstract topology platform (tpu on workers; attached-cpu = local machinery smoke)')
    parser.add_argument('--mesh', type=int, nargs=2, default=(2, 4), metavar=('DATA', 'MODEL'))
    parser.add_argument('--cache-probe', action='store_true',
                        help='also enable the persistent cache in a private tempdir and list entries')
    parser.add_argument('--skip-aot', action='store_true', help='source inspection only')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ray-url', default='http://127.0.0.1:8265')
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    if args.mode == 'remote':
        remote(args)
    elif args.mode == 'submit':
        if args.output is None:
            parser.error('--output is required for submit')
        submit(args)
    else:
        report = probe(args)
        text = json.dumps(report, indent=2, default=str) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            sys.stdout.write(text)


if __name__ == '__main__':
    main()
