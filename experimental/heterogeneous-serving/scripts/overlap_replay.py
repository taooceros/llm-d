"""Capture real startup modules and replay them with an offline TPU client.

No array creation, executable invocation, or real TPU backend initialization is
allowed in replay. The bundle retains the engine's exact HLO/options; a cache
key mismatch is reported, never patched over or counted as a cache hit.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import threading
import time


@contextmanager
def capture(directory):
    import jax
    from jax._src import compiler, distributed
    # Implicit TPU/PJRT multi-controller setup can leave JAX's explicit
    # distributed State.process_id at its default zero on every host.
    # The actual PJRT process index, not that optional state, elects a writer.
    if jax.process_index() != 0:
        yield
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original = compiler.compile_or_get_cached
    lock = threading.Lock()
    signature = inspect.signature(original)
    seen = set()
    def record(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        backend, module, devices, options = (p[k] for k in
                ('backend', 'computation', 'devices', 'compile_options'))
        if p['host_callbacks']:
            raise ValueError('offline precompile does not support host callbacks')
        key = compiler._get_cache_key(options, backend, module, devices)
        if key is None:
            raise ValueError('capture requires enabled persistent compilation cache')
        with lock:
            fresh = key not in seen
            seen.add(key)
        if fresh:
            stream = io.BytesIO()
            module.operation.write_bytecode(stream)
            mlir = stream.getvalue()
            config = options.SerializeAsString()
            (directory / (key + '.mlir')).write_bytes(mlir)
            (directory / (key + '.options')).write_bytes(config)
            row = {'key': key, 'device_ids': [d.id for d in devices.flat],
                   'device_shape': list(devices.shape),
                   'process_indices': [d.process_index for d in devices.flat],
                   'device_kind': devices.flat[0].device_kind,
                   'module_sha256': hashlib.sha256(mlir).hexdigest(),
                   'options_sha256': hashlib.sha256(config).hexdigest(),
                   'jax_version': __import__('jax').__version__,
                   "compiler_env": {k: os.environ.get(k) for k in ("XLA_FLAGS", "LIBTPU_INIT_ARGS")},
                   "writer_pjrt_process_index": jax.process_index(),
                   "writer_cache_process_id": distributed.global_state.process_id,
                   "writer_host": __import__("socket").gethostname(),
                   "writer_pid": os.getpid(),
                   'captured_at': time.time()}
            (directory / (key + '.json')).write_text(json.dumps(row, indent=2))
        return original(*args, **kwargs)
    compiler.compile_or_get_cached = record
    try:
        yield
    finally:
        compiler.compile_or_get_cached = original


def replay(directory, cache_dir, topology_name, *, enabled=False):
    if not enabled:
        return {'enabled': False, 'executables': [], 'device_launches': 0}
    os.environ['JAX_PLATFORMS'] = 'cpu'
    descriptors = [json.loads(path.read_text()) for path in sorted(Path(directory).glob("*.json"))]
    compiler_envs = [row["compiler_env"] for row in descriptors]
    if compiler_envs and any(value != compiler_envs[0] for value in compiler_envs):
        raise ValueError("one replay bundle contains differing compiler environments")
    for key, value in (compiler_envs[0] if compiler_envs else {}).items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    import jax
    import numpy as np
    from jax.experimental import topologies
    from jax._src import compiler, xla_bridge
    from jax._src.interpreters import mlir
    from jax._src.lib import xla_client as xc
    from jaxlib.mlir import ir
    from hetero.compile_overlap import StartupProfile
    from hetero.compile_cache import configure_compile_cache, verify_compile_cache
    if 'tpu' in xla_bridge._backends:
        raise RuntimeError('offline replay refuses an initialized real TPU backend')
    jax.config.update('jax_compilation_cache_dir', cache_dir)
    expected_cache = configure_compile_cache({
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
        "JAX_LOG_COMPILES": "0", "JAX_EXPLAIN_CACHE_MISSES": "0"})
    topology = (topologies.get_attached_topology("cpu") if topology_name == "attached:cpu"
                else topologies.get_topology_desc(platform="tpu", topology_name=topology_name))
    available = {d.id: d for d in topology.devices}
    profile = StartupProfile()
    started, cpu = time.time(), time.process_time()
    rows = []
    with profile.hooks():
        for descriptor in descriptors:
            key = descriptor['key']
            t0, cpu0 = time.time(), time.process_time()
            row = {'expected_key': key, 'start': t0}
            try:
                if descriptor['jax_version'] != jax.__version__:
                    raise ValueError('JAX version differs from captured module')
                devices = np.asarray([available[i] for i in descriptor['device_ids']], dtype=object).reshape(descriptor['device_shape'])
                backend = devices.flat[0].client
                options = xc.CompileOptions.ParseFromString((Path(directory) / (key + '.options')).read_bytes())
                with mlir.make_ir_context():
                    module = ir.Module.parse((Path(directory) / (key + '.mlir')).read_bytes())
                    actual_key = compiler._get_cache_key(options, backend, module, devices)
                    row['actual_key'] = actual_key
                    row['key_matches'] = actual_key == key
                    # Even a mismatch is compiled as evidence, but never counted
                    # as target-cache savings; no key rewriting is permitted.
                    params = dict(backend=backend, computation=module, devices=devices,
                                  compile_options=options, host_callbacks=[],
                                  executable_devices=xc.DeviceList(tuple(devices.flat)))
                    compiler.compile_or_get_cached(**params)
                    row['compiled'] = True
            except Exception as exc:
                row.update(compiled=False, error=f'{type(exc).__name__}: {exc}')
            row.update(end=time.time(), process_cpu_s=time.process_time() - cpu0)
            rows.append(row)
    ended = time.time()
    verify_compile_cache(expected_cache)
    if 'tpu' in xla_bridge._backends:
        raise RuntimeError('offline replay unexpectedly initialized TPU backend')
    return {'enabled': True, 'start': started, 'end': ended,
            'wall_s': ended - started, 'process_cpu_s': time.process_time() - cpu,
            'executables': rows, 'profile': profile.report(started, ended),
            'device_launches': 0, 'real_tpu_backend_initialized': False,
            'effective_cache_config': {
                'jax_compilation_cache_dir': jax.config.jax_compilation_cache_dir,
                **{key: getattr(jax.config, key) for key in expected_cache}},
            'all_keys_match': bool(rows) and all(r.get('key_matches') for r in rows),
            'all_compiled': bool(rows) and all(r.get('compiled') for r in rows)}
