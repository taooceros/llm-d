import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ['JAX_PLATFORMS'] = 'cpu'
from scripts.overlap_key_only import compare_one, index_descriptors, read_verified


class DescriptorTests(unittest.TestCase):
    def test_grouping_preserves_distinct_compile_options(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            row = {'key': 'test-key', 'module_sha256': 'same-module',
                   'options_sha256': 'same-options', 'device_ids': [0],
                   'device_shape': [1, 1], 'jax_version': '0.10.2',
                   'compiler_env': {'XLA_FLAGS': None, 'LIBTPU_INIT_ARGS': None}}
            for instance in ('small_a', 'small_b'):
                p = root / instance / 'warm'
                p.mkdir(parents=True)
                (p / 'one.json').write_text(json.dumps(row))
            row['options_sha256'] = 'different-options'
            (p / 'two.json').write_text(json.dumps(row))
            groups, phases, env = index_descriptors(root)
            self.assertEqual(sorted(len(g['paths']) for g in groups), [1, 2])
            self.assertEqual(sum(phases.values()), 3)
            self.assertEqual(env, row['compiler_env'])
            row['compiler_env']['XLA_FLAGS'] = '--xla_different=true'
            (p / 'two.json').write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'environments differ'):
                index_descriptors(root)

    def test_payload_hash_mismatch_is_not_silently_replayed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'module.mlir'
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'payload hash mismatch'):
                read_verified(path, hashlib.sha256(b'original').hexdigest())


class RealCPUKeyTests(unittest.TestCase):
    def test_key_roundtrip_and_negative_result_without_compilation(self):
        import jax
        import numpy as np
        from jax._src import compilation_cache, compiler, xla_bridge
        from jax._src.interpreters import mlir
        from jax._src.lib import xla_client as xc
        from jaxlib.mlir import ir

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compilation_cache.reset_cache()
            jax.config.update('jax_compilation_cache_dir', str(root / 'never-written-cache'))
            backend = xla_bridge.get_backend('cpu')
            device = backend.devices()[0]
            devices = np.asarray([device], dtype=object).reshape(1, 1)
            options = xc.CompileOptions()
            options.num_replicas = 1
            options.num_partitions = 1
            with mlir.make_ir_context():
                module = ir.Module.parse('''module @key_only_smoke {
                  func.func public @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    return %arg0 : tensor<2xf32>
                  }
                }''')
                expected = compiler._get_cache_key(options, backend, module, devices)
                stream = io.BytesIO()
                module.operation.write_bytecode(stream)
            data = stream.getvalue()
            config = options.SerializeAsString()
            path = root / 'captured.json'
            path.with_suffix('.mlir').write_bytes(data)
            path.with_suffix('.options').write_bytes(config)
            descriptor = {'key': expected, 'module_sha256': hashlib.sha256(data).hexdigest(),
                          'options_sha256': hashlib.sha256(config).hexdigest(),
                          'device_ids': [device.id], 'device_shape': [1, 1],
                          'jax_version': jax.__version__}
            group = {'descriptor': descriptor, 'paths': [str(path)]}
            result = compare_one(group, {device.id: device})
            self.assertEqual(result['status'], 'match', result)
            self.assertEqual(result['actual_key'], expected)
            bad = copy.deepcopy(group)
            bad['descriptor']['key'] = 'deliberately-not-the-captured-key'
            result = compare_one(bad, {device.id: device})
            self.assertEqual(result['status'], 'mismatch', result)
            self.assertEqual(result['actual_key'], expected)
            path.with_suffix('.options').write_bytes(b'corrupt')
            self.assertEqual(compare_one(group, {device.id: device})['status'], 'error')
            self.assertFalse((root / 'never-written-cache').exists())
            self.assertNotIn('tpu', xla_bridge._backends)
            compilation_cache.reset_cache()


if __name__ == '__main__':
    unittest.main()
