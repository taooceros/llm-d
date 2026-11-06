"""CPU-only probe: can a TP8 v5e PJRT executable be built and held without a TPU, and what still needs a client?

Writes one JSON document to argv[1]. Uses the real libtpu of the qualified pm-vllm stack in compile-only mode.
"""
import ctypes
import glob
import hashlib
import json
import os
import struct
import sys
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
OUT = sys.argv[1]
res = {'pid': os.getpid(), 'accel_device_nodes': sorted(glob.glob('/dev/accel*') + glob.glob('/dev/vfio/*'))}
clock = time.perf_counter
START = clock()


def error_text(exc):
    return f'{type(exc).__name__}: {exc}'.splitlines()[0][:600]


import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlib  # noqa: E402
import libtpu  # noqa: E402
import numpy as np  # noqa: E402
from jax._src.lib import _jax  # noqa: E402
from jax._src.lib import xla_client as xc  # noqa: E402
from jax.experimental import topologies  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

lib = ctypes.CDLL(libtpu.get_library_path())
lib.GetPjrtApi.restype = ctypes.c_void_p
api = lib.GetPjrtApi()
size, _, vsize, _, major, minor = struct.unpack_from('<QQQQii', ctypes.string_at(api, 40))
slots = (size - 16 - vsize) // 8
fns = struct.unpack(f'<{slots}Q', ctypes.string_at(api + 16 + vsize, 8 * slots))
res['versions'] = {'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'libtpu': libtpu.get_library_path(),
                   'pjrt_c_api': f'{major}.{minor}', 'pjrt_api_fn_slots': slots,
                   'pjrt_api_null_slots': [i for i, fn in enumerate(fns) if not fn]}

topo, errors = None, {}
for name in ('v5e:4x2', 'v5e:2x4'):
    try:
        t0 = clock()
        topo = topologies.get_topology_desc(topology_name=name, platform='tpu')
        res['topology'] = {'name': name, 'create_s': round(clock() - t0, 3)}
        break
    except Exception as exc:  # record the compiler's own rejection, then try the transposed name
        errors[name] = error_text(exc)
res['topology_errors'] = errors
if topo is None:
    json.dump(res, open(OUT, 'x'), indent=2)
    raise SystemExit('no compile-only topology')
devs = sorted(topo.devices, key=lambda d: d.id)
res['devices'] = [{'id': d.id, 'coords': list(d.coords), 'process_index': d.process_index,
                   'kind': d.device_kind} for d in devs]
client = devs[0].client
res['client'] = {'type': type(client).__name__, 'platform': client.platform,
                 'platform_version': client.platform_version}
fd_targets = []
for fd in os.listdir('/proc/self/fd'):
    try:
        fd_targets.append(os.readlink(f'/proc/self/fd/{fd}'))
    except OSError:  # the listing's own fd closes before readlink
        pass
res['libtpu_fds'] = sorted({p for p in fd_targets if 'libtpu' in p})

mesh = Mesh(np.array(devs), ('tp',))
L, B, D, F = (int(os.environ.get(k, v)) for k, v in (('PJ_L', '60'), ('PJ_B', '32'), ('PJ_D', '5376'),
                                                     ('PJ_F', '21504')))
res['program'] = {'layers': L, 'tokens': B, 'hidden': D, 'ffn': F, 'dtype': 'bfloat16',
                  'shape': 'per layer: q,k,v,o column/row-parallel + MLP; GSPMD inserts the TP all-reduces'}


def sharded(shape, *spec):
    return jax.ShapeDtypeStruct(shape, jnp.bfloat16, sharding=NamedSharding(mesh, P(*spec)))


def model(x, wq, wk, wv, wo, w1, w2):
    for i in range(L):
        q, k, v = (jnp.dot(x, w[i]) for w in (wq, wk, wv))
        x = x + jnp.dot(jax.nn.softmax(q * k, axis=-1) * v, wo[i])
        x = x + jnp.dot(jax.nn.gelu(jnp.dot(x, w1[i])), w2[i])
    return x


avals = (sharded((B, D)), *(sharded((L, D, D), None, None, 'tp') for _ in range(3)),
         sharded((L, D, D), None, 'tp', None), sharded((L, D, F), None, None, 'tp'),
         sharded((L, F, D), None, 'tp', None))
t0 = clock()
lowered = jax.jit(model, out_shardings=NamedSharding(mesh, P())).lower(*avals)
res['lower_s'] = round(clock() - t0, 3)
t0 = clock()
compiled = lowered.compile()
res['compile_s'] = round(clock() - t0, 3)
exe = compiled.runtime_executable()
stats = exe.get_compiled_memory_stats()
res['executable'] = {
    'type': f'{type(exe).__module__}.{type(exe).__name__}',
    'is_loaded_executable': isinstance(exe, _jax.LoadedExecutable),
    'has_execute': hasattr(exe, 'execute'),
    'hlo_modules': len(exe.hlo_modules()),
    'generated_code_bytes': stats.generated_code_size_in_bytes,
    'argument_bytes_per_chip': stats.argument_size_in_bytes,
    'temp_bytes_per_chip': stats.temp_size_in_bytes,
    'all_reduce_ops_in_optimized_hlo': compiled.as_text().count(' all-reduce('),
}
t0 = clock()
blob = exe.serialize()
res['serialize'] = {'s': round(clock() - t0, 3), 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}

attempts = {}
try:
    jax.device_put(np.zeros((B, D), np.float32), NamedSharding(mesh, P()))
    attempts['device_put_to_compile_only_devices'] = 'succeeded'
except Exception as exc:
    attempts['device_put_to_compile_only_devices'] = error_text(exc)
try:
    client.deserialize_executable(blob, xc.DeviceList(tuple(devs)), None)
    attempts['deserialize_executable_on_compile_only_client'] = 'succeeded'
except Exception as exc:
    attempts['deserialize_executable_on_compile_only_client'] = error_text(exc)
res['attempts'] = attempts
res['jax_backends_initialized'] = sorted(getattr(sys.modules['jax._src.xla_bridge'], '_backends', {}) or {})
res['wall_s'] = round(clock() - START, 3)
with open(OUT, 'x') as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
