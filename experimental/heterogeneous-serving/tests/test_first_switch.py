"""First switch (F0): inline provisioning in a normal deploy, run scopes, segment accounting.

The bank written by a checkpoint-loaded engine and read by a snapshot-restored one is exercised in
tests/test_exec_bank.py (``read_restored``); here: the engine settings, the loader-path independence of
the bank identity, run-scope files/cleanup (incl. the RCM host action), the normal-loader checkpoint
clock, the deploy anatomy / provisioning flags and the first-switch summary.
"""
import json
import pathlib
import sys
import tempfile
import time
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hetero import exec_bank as eb  # noqa: E402
from hetero import run_scope, switch  # noqa: E402
from hetero.engine_actor import EngineSettings  # noqa: E402
from hetero.weight_snapshot import prepare, restore, snapshot  # noqa: E402


class SettingsTests(unittest.TestCase):
    def test_snapshot_root_default_and_validation(self):
        self.assertEqual(EngineSettings().weight_snapshot_root, "/dev/shm/omp_weights")
        s = EngineSettings(weight_snapshot="write", exec_bank="write",
                           **run_scope.scope_roots("first-abc-r1"))
        self.assertEqual(s.weight_snapshot_root, "/dev/shm/omp_weights_runs/first-abc-r1")
        self.assertEqual(s.exec_bank_dir, "/root/.cache/omp_exec_bank_runs/first-abc-r1")
        for bad in ({"weight_snapshot_root": "relative"}, {"weight_snapshot_root": 3}):
            with self.assertRaises(ValueError, msg=bad):
                EngineSettings(**bad)

    def test_scope_names(self):
        for bad in ("", "../x", "a/b", "-lead", "a..b", "x" * 129, None):
            with self.assertRaises(ValueError, msg=bad):
                run_scope.check_scope(bad)
        self.assertEqual(run_scope.check_scope("first-gate-1a2b3c4"), "first-gate-1a2b3c4")


class IdentityLoaderPathTests(unittest.TestCase):
    """The bank identity reads only ``runtime_env`` and ``tp_size`` from the instance payload, so the
    loader path (normal + inline snapshot vs restore), the snapshot/bank roots, the pre-spawn gate and
    the per-deploy nonce/generation cannot change the digest that names (and gates) the bank."""

    def test_digest_independent_of_loader_fields(self):
        hf = NS(to_dict=lambda: {"hidden_size": 5376})
        config = NS(model_config=NS(model="/models/gemma-4-31b", revision=None, dtype="torch.bfloat16",
                                    quantization=None, max_model_len=131072, max_logprobs=32,
                                    enforce_eager=False, hf_config=hf),
                    cache_config=NS(cache_dtype="bfloat16", block_size=512, enable_prefix_caching=False),
                    scheduler_config=NS(max_num_seqs=512, max_num_batched_tokens=2048, async_scheduling=True),
                    speculative_config=None,
                    additional_config={"activation_dtype": None, "matmul_precision": "default"})
        runner = NS(vllm_config=config, mesh=jax.make_mesh((1, 1), ("data", "model")),
                    kv_caches=[jnp.zeros((4, 8))])
        base = {"tp_size": 16, "runtime_env": {"TPU_PROCESS_ADDRESSES": "10.0.0.1:8471,10.0.0.2:8471",
                                               "TPU_TOPOLOGY": "4x4"}}
        normal_write = {**base, "weight_snapshot": "write", "fast_graph": True, "exec_bank": "write",
                        "exec_bank_nonce": "a" * 32, "weight_snapshot_generation": "b" * 32,
                        "prespawn_gate": {"name": "gate"},
                        **run_scope.scope_roots("first-abc-r1")}
        restore_read = {**base, "weight_snapshot": "read", "weight_snapshot_verify": False, "fast_graph": True,
                        "exec_bank": "read", "exec_bank_nonce": "c" * 32,
                        "weight_snapshot_generation": "d" * 32, "weight_snapshot_root": "/dev/shm/omp_weights",
                        "exec_bank_dir": "/root/.cache/omp_exec_bank"}

        def digest(payload):
            identity = eb.engine_identity(runner, payload, backend_revision="b", runtime_revision="r",
                                          runtime=eb.runtime_identity({}), files={"m": "x"}, num_blocks=283)
            return eb.digest(identity["instance"])

        self.assertEqual(digest(normal_write), digest(restore_read))
        self.assertNotEqual(digest(normal_write), digest({**normal_write, "tp_size": 8}))


def _tree(mesh):
    n = len(mesh.devices.flat)
    return {"w": jax.device_put(jnp.arange(n * 8, dtype=jnp.bfloat16).reshape(n * 2, 4),
                                NamedSharding(mesh, PartitionSpec("tp", None)))}


class RunScopeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        for name, sub in (("SNAPSHOT_RUNS_PARENT", "shm/omp_weights_runs"),
                          ("BANK_RUNS_PARENT", "cache/omp_exec_bank_runs")):
            patcher = mock.patch.object(run_scope, name, str(self.root / sub))
            patcher.start()
            self.addCleanup(patcher.stop)
        # Neighbours a scope cleanup must never touch: the default roots and another run's scope.
        self.default_snapshot = self.root / "shm/omp_weights/H/large_a/rank0"
        self.default_bank = self.root / "cache/omp_exec_bank/digest/rank0"
        for path in (self.default_snapshot, self.default_bank):
            path.mkdir(parents=True)
            (path / "keep.bin").write_bytes(b"x" * 10)
        self.mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("tp",))
        self.context = {"model_path": "/models/test", "tp_size": 1, "dtype": "bfloat16"}

    def _populate(self, scope):
        roots = run_scope.scope_roots(scope)
        snap_dir = pathlib.Path(roots["weight_snapshot_root"]) / "H" / "large_a" / "rank0"
        report = snapshot(snap_dir, _tree(self.mesh), self.context, generation="e" * 32)
        bank_dir = pathlib.Path(roots["exec_bank_dir"]) / ("f" * 64) / "rank0"
        bank_dir.mkdir(parents=True)
        (bank_dir / "run_model-1.bin").write_bytes(b"y" * 7)
        (bank_dir / "manifest.json").write_text(json.dumps({
            "identity_digest": "f" * 64, "skipped": [],
            "entries": {"run_model-1.bin": {"fn": "run_model", "key": "k", "sha256": "s", "bytes": 7}}}))
        return report

    def test_files_report_and_remove_only_the_scope(self):
        report = self._populate("first-abc-r1")
        self._populate("first-abc-r2")
        files = run_scope.scope_files("first-abc-r1")
        self.assertEqual([m["generation"] for m in files["snapshot_manifests"]], ["e" * 32])
        self.assertEqual(files["snapshot_manifests"][0]["bytes"], report["bytes"])
        self.assertEqual(files["bank_manifests"][0]["entries"], 1)
        removed = run_scope.remove_scope("first-abc-r1")
        self.assertTrue(all(r["existed"] and r["absent_after"] for r in removed.values()), removed)
        self.assertGreater(removed["weight_snapshot_root"]["bytes"], report["bytes"] - 1)
        self.assertEqual(run_scope.scope_files("first-abc-r1"),
                         {"scope": "first-abc-r1", "snapshot_manifests": [], "bank_manifests": []})
        self.assertEqual(len(run_scope.scope_files("first-abc-r2")["snapshot_manifests"]), 1)
        self.assertTrue((self.default_snapshot / "keep.bin").exists())
        self.assertTrue((self.default_bank / "keep.bin").exists())
        # Idempotent: a second cleanup (the EXIT trap after a normal close) is a no-op.
        self.assertFalse(any(r["existed"] for r in run_scope.remove_scope("first-abc-r1").values()))

    def test_scope_snapshot_passes_manifest_and_restore_checks(self):
        self._populate("first-abc-r1")
        directory = pathlib.Path(run_scope.scope_roots("first-abc-r1")["weight_snapshot_root"]) / "H/large_a/rank0"
        clocks = {}
        prepared = prepare(directory, self.context, self.mesh, clocks, verify_checksums=False)
        self.assertEqual(prepared[0]["version"], 2)
        self.assertEqual(prepared[0]["generation"], "e" * 32)
        tree = _tree(self.mesh)
        abstract = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), tree)
        restored = restore(abstract, self.mesh, prepared, clocks)
        np.testing.assert_array_equal(np.asarray(restored["w"]), np.asarray(tree["w"]))

    def test_rcm_host_action_scope_cleanup_and_files(self):
        from scripts import snapshot_validate

        self._populate("first-abc-r1")
        row = snapshot_validate.host_action("files", 0, run_scope="first-abc-r1")
        self.assertEqual(len(row["scope_files"]["snapshot_manifests"]), 1)
        with self.assertRaisesRegex(RuntimeError, "no snapshot or bank manifest"):
            snapshot_validate.host_action("files", 0, run_scope="first-abc-r9")
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            snapshot_validate.host_action("poison", 0, run_scope="first-abc-r1")
        row = snapshot_validate.host_action("cleanup", 0, run_scope="first-abc-r1")
        self.assertFalse(row["cleaned"])  # the default /dev/shm/omp_weights root was not removed
        self.assertTrue(all(r["absent_after"] for r in row["scope_removed"].values()))
        self.assertTrue((self.default_snapshot / "keep.bin").exists())


class CheckpointClockTests(unittest.TestCase):
    def setUp(self):
        # hetero.tpu_worker imports ray and tpu_inference at import time; compile only its helpers.
        import ast
        import contextlib
        from typing import Any, Dict

        tree = ast.parse((ROOT / "hetero/tpu_worker.py").read_text())
        keep = [node for node in tree.body if (isinstance(node, ast.FunctionDef) and node.name in (
            "_new_load_stats", "_checkpoint_clock")) or (isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "_LOADER_HOOKS" for t in node.targets))]
        self.assertEqual(len(keep), 3)
        self.ns = {"contextlib": contextlib, "time": time, "Dict": Dict, "Any": Any}
        exec(compile(ast.Module(body=keep, type_ignores=[]), "tpu_worker_helpers", "exec"), self.ns)

    def test_splits_weight_load_and_restores_every_patched_attribute(self):
        class Tensor:  # the torch.Tensor surface the clock reads (bf16, 4 elements)
            def numel(self):
                return 4

            def element_size(self):
                return 2

        weight_utils = types.ModuleType("tpu_inference.models.jax.utils.weight_utils")
        unquantized = types.ModuleType("tpu_inference.layers.jax.quantization.unquantized")

        def convert(tensor):
            time.sleep(0.03)  # page faults + permute + t2j
            return tensor

        def place(param, weight, param_name=None):
            time.sleep(0.01)  # shard selection + H2D

        weight_utils.jax_array_from_reshaped_torch = convert
        weight_utils.assign_and_shard_param = place
        unquantized.jax_array_from_reshaped_torch = convert  # imported by name, as in tpu_inference
        unquantized.assign_and_shard_param = place

        class DefaultModelLoader:
            def get_all_weights(self, model_config, model):
                for i in range(4):
                    yield f"w{i}", Tensor()  # lazy mmaps: opening is cheap

            def load_weights(self, model, model_config):
                for i, (_, tensor) in enumerate(self.get_all_weights(model_config, model)):
                    module = weight_utils if i % 2 else unquantized  # both call sites are timed
                    module.assign_and_shard_param(None, module.jax_array_from_reshaped_torch(tensor))

        modules = {name: types.ModuleType(name) for name in (
            "vllm", "vllm.model_executor", "vllm.model_executor.model_loader",
            "vllm.model_executor.model_loader.default_loader")}
        modules["vllm.model_executor.model_loader.default_loader"].DefaultModelLoader = DefaultModelLoader
        modules[weight_utils.__name__] = weight_utils
        modules[unquantized.__name__] = unquantized
        original = (DefaultModelLoader.get_all_weights, DefaultModelLoader.load_weights)
        stats, clocks = self.ns["_new_load_stats"](), {}
        with mock.patch.dict(sys.modules, modules):
            with self.ns["_checkpoint_clock"](stats, clocks):
                DefaultModelLoader().load_weights(None, None)
        self.assertEqual((DefaultModelLoader.get_all_weights, DefaultModelLoader.load_weights), original)
        for module in (weight_utils, unquantized):
            self.assertIs(module.jax_array_from_reshaped_torch, convert)
            self.assertIs(module.assign_and_shard_param, place)
        self.assertEqual(len(stats["hooks"]), 4)
        self.assertEqual((stats["tensors"], stats["bytes"]), (4, 4 * 4 * 2))
        self.assertEqual((stats["host_convert_calls"], stats["shard_place_calls"]), (4, 4))
        self.assertGreaterEqual(stats["host_convert_s"], 0.115)
        self.assertGreaterEqual(stats["shard_place_s"], 0.035)
        self.assertLess(stats["read_s"], 0.02)
        window = clocks["weight_load"]["end"] - clocks["weight_load"]["start"]
        self.assertGreaterEqual(window + 1e-3, stats["host_convert_s"] + stats["shard_place_s"] + stats["read_s"])

    def test_nested_and_concurrent_calls_count_wall_time_once(self):
        """F1 attempt 1 summed host_convert to 780 s inside a 565 s window: a hooked name reached
        through another hooked wrapper was timed twice.  Only the outermost call counts now, and
        concurrent calls count their wall-clock union."""
        import threading

        weight_utils = types.ModuleType("tpu_inference.models.jax.utils.weight_utils")
        unquantized = types.ModuleType("tpu_inference.layers.jax.quantization.unquantized")

        def convert(tensor):
            time.sleep(0.05)
            return tensor

        def place(param, weight, param_name=None):
            time.sleep(0.02)

        weight_utils.jax_array_from_reshaped_torch = convert
        weight_utils.assign_and_shard_param = place
        # unquantized's names resolve through weight_utils' (patched) attributes: the double-wrap case.
        unquantized.jax_array_from_reshaped_torch = lambda t: weight_utils.jax_array_from_reshaped_torch(t)
        unquantized.assign_and_shard_param = lambda p, w, param_name=None: weight_utils.assign_and_shard_param(p, w)

        class DefaultModelLoader:
            def get_all_weights(self, model_config, model):
                yield from ()

            def load_weights(self, model, model_config):
                for _ in range(3):  # nested: unquantized -> weight_utils
                    unquantized.assign_and_shard_param(None, unquantized.jax_array_from_reshaped_torch(1))
                threads = [threading.Thread(target=weight_utils.jax_array_from_reshaped_torch, args=(1,))
                           for _ in range(4)]  # concurrent: 4 x 50 ms overlapping
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        modules = {name: types.ModuleType(name) for name in (
            "vllm", "vllm.model_executor", "vllm.model_executor.model_loader",
            "vllm.model_executor.model_loader.default_loader")}
        modules["vllm.model_executor.model_loader.default_loader"].DefaultModelLoader = DefaultModelLoader
        modules[weight_utils.__name__] = weight_utils
        modules[unquantized.__name__] = unquantized
        stats, clocks = self.ns["_new_load_stats"](), {}
        with mock.patch.dict(sys.modules, modules):
            with self.ns["_checkpoint_clock"](stats, clocks):
                loader = DefaultModelLoader()
                loader.load_weights(None, None)
                loader.load_weights(None, None)  # a second load_weights extends one window
        self.assertIs(weight_utils.jax_array_from_reshaped_torch, convert)
        self.assertEqual(stats["weight_load_calls"], 2)
        # 2 loads x (3 outer + 3 nested + 4 threads): threads overlapping an active call count as nested.
        self.assertEqual(stats["host_convert_calls"] + stats["host_convert_nested_calls"], 20)
        self.assertGreaterEqual(stats["host_convert_nested_calls"], 6)
        self.assertEqual(stats["shard_place_nested_calls"], 6)
        self.assertEqual(stats["shard_place_calls"], 6)
        # 2 x (3 x 50 ms nested + ~50 ms for 4 concurrent) = ~0.4 s, not 2 x (0.3 + 0.2) = 1.0 s.
        self.assertGreaterEqual(stats["host_convert_s"], 0.39)
        self.assertLess(stats["host_convert_s"], 0.6)
        self.assertLess(stats["shard_place_s"], 0.2)
        window = clocks["weight_load"]["end"] - clocks["weight_load"]["start"]
        self.assertGreaterEqual(window + 1e-3, stats["host_convert_s"] + stats["shard_place_s"])


def _worker(rank, *, t0, normal, gen="g" * 32, bank_entries=187, write_error=None, digest="d1"):
    ph = {"init_device": {"start": t0, "end": t0 + 4}}
    if normal:
        ph.update(load_model={"start": t0 + 4, "end": t0 + 104}, graph_build={"start": t0 + 5, "end": t0 + 7},
                  weight_load={"start": t0 + 8, "end": t0 + 100},
                  snapshot_write={"start": t0 + 104.5, "end": t0 + 111.5})
    else:
        ph.update(load_model={"start": t0 + 4, "end": t0 + 9},
                  snapshot_device_placement={"start": t0 + 6, "end": t0 + 8})
    ph.update(determine_available_memory={"start": t0 + 112, "end": t0 + 113},
              initialize_from_config={"start": t0 + 113, "end": t0 + 116},
              compile_or_warm_up_model={"start": t0 + 116, "end": t0 + 150},
              exec_bank_write={"start": t0 + 150, "end": t0 + 152.5})
    return {"executor_rank": rank, "phases": ph,
            "normal_load": {"read_s": 0.2, "host_convert_s": 80.0, "shard_place_s": 6.0} if normal else None,
            "weight_snapshot": {"mode": "write", "restored": False,
                                "snapshot": {"generation": gen, "bytes": 15824457568, "snapshot_s": 7.0,
                                             "directory": "/dev/shm/omp_weights_runs/x/H/large_a/rank0"}},
            "exec_bank": {"mode": "write", "identity_digest": digest, "write_error": write_error,
                          "written": {"entries": bank_entries, "bytes": 1, "skipped": [], "path": "p"}}}


def _deploy(workers):
    return {"instances": [{"descriptor": {"instance_id": "large_a"}, "initialization": {
        "instance_id": "large_a", "engine_init_started_at": 0.0, "worker_init_phases": workers}}]}


class AnatomyTests(unittest.TestCase):
    def test_normal_loader_subphases_and_engine_ready_after_the_bank_write(self):
        a = switch.deploy_anatomy(_deploy([_worker(0, t0=10.0, normal=True), _worker(1, t0=10.5, normal=True)]))
        self.assertEqual(a["last_engine_ready"], 10.5 + 152.5)  # bank write end, not compile end
        med = a["phase_median_s"]
        for key, value in (("weight_load", 92.0), ("graph_build", 2.0), ("snapshot_write", 7.0),
                           ("exec_bank_write", 2.5), ("checkpoint_iter", 0.2), ("host_convert", 80.0),
                           ("shard_place", 6.0), ("load_model_unattributed", 6.0)):
            self.assertAlmostEqual(med[key], value, msg=key)

    def test_restore_rows_keep_the_o3_semantics(self):
        a = switch.deploy_anatomy(_deploy([_worker(0, t0=0.0, normal=False)]))
        self.assertAlmostEqual(a["phase_median_s"]["load_model_unattributed"], 3.0)
        self.assertNotIn("host_convert", a["phase_median_s"])

    def test_provisioned_flags(self):
        ok = switch.provisioned_flags(_deploy([_worker(0, t0=0, normal=True), _worker(1, t0=0, normal=True)]))
        self.assertTrue(ok["all_provisioned"], ok)
        self.assertEqual(ok["per_instance"]["large_a"]["1"]["bank_entries"], 187)
        for bad in ([_worker(0, t0=0, normal=True), _worker(1, t0=0, normal=True, gen="h" * 32)],
                    [_worker(0, t0=0, normal=True), _worker(1, t0=0, normal=True, gen=None)]):
            flags = switch.provisioned_flags(_deploy(bad))
            self.assertFalse(flags["snapshot_published"])
            self.assertTrue(flags["bank_written"])
        for bad in ([_worker(0, t0=0, normal=True, write_error="OSError: x")],
                    [_worker(0, t0=0, normal=True, bank_entries=0)],
                    [_worker(0, t0=0, normal=True), _worker(1, t0=0, normal=True, digest="d2")]):
            flags = switch.provisioned_flags(_deploy(bad))
            self.assertTrue(flags["snapshot_published"])
            self.assertFalse(flags["bank_written"])
            self.assertFalse(flags["all_provisioned"])
        self.assertFalse(switch.provisioned_flags(None)["all_provisioned"])


class FirstSummaryTests(unittest.TestCase):
    def test_cold_means_warm_beside_and_amortization(self):
        from scripts import switch_dry_run as dry
        from scripts import switch_summary

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        artifacts, out = root / "artifacts", root / "out"
        out.mkdir()
        runs, reports = [], {}
        for rep, kind, run_id in ((1, "cold", "first-abc-ov-r1"), (2, "cold", "first-abc-ov-r2"),
                                  (3, "warm", "first-abc-ov-warm")):
            fabric, client, kw = dry.run_dry("full", artifacts, scale=0.05, run_id=run_id)
            reports[kind, rep] = switch.run_switch(fabric, client, **kw)
            runs.append({"mode": "first", "rep": rep, "attempt": 1, "run_id": run_id, "ok": True, "kind": kind})
        (out / "campaign.json").write_text(json.dumps({"mode": "first", "run_base": "first-abc-ov"}))
        (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
        summary = switch_summary.summarize(out, artifacts=artifacts)
        self.assertEqual([r["label"] for r in summary["reps"]], ["r1", "r2", "warm"])
        self.assertEqual([r["kind"] for r in summary["reps"]], ["cold", "cold", "warm"])
        cold = [reports["cold", 1]["endpoints"]["C_production_s"], reports["cold", 2]["endpoints"]["C_production_s"]]
        warm = reports["warm", 3]["endpoints"]["C_production_s"]
        a = summary["amortized"]
        self.assertAlmostEqual(a["C_first_s"], round(sum(cold) / 2, 3), places=3)
        self.assertEqual(a["C_warm_s"], warm)
        self.assertAlmostEqual(a["per_switch_s"]["10"], round((a["C_first_s"] + 9 * warm) / 10, 3), places=3)
        self.assertEqual(a["per_switch_s"]["1"], round(a["C_first_s"], 3))
        self.assertEqual(a["per_switch_O3_warm_s"]["1"], round(a["C_first_s"], 3))
        self.assertAlmostEqual(a["per_switch_O3_warm_s"]["3"], round((a["C_first_s"] + 2 * 27.51) / 3, 3), places=3)
        self.assertAlmostEqual(a["separate_pass_s"]["1"], 350.8 + 27.51, places=3)
        self.assertAlmostEqual(a["separate_pass_s"]["10"], round((350.8 + 10 * 27.51) / 10, 3), places=3)
        text = switch_summary.markdown(summary)
        self.assertIn("Amortized per-switch cost", text)
        self.assertIn("| metric | r1 | r2 | warm | mean (cold) |", text)
        self.assertNotIn("Harness H -> S", text)


if __name__ == "__main__":
    unittest.main()
