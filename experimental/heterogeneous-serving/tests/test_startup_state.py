"""Autoscale startup-state shipping (scripts/startup_state.py) and the placement-free snapshot context.

A snapshot written for H:large_a in one run scope is listed, copied file by file (new mtimes, as any
transport produces), published for H:large_b in another scope, and must then pass the same
``prepare``/``restore`` a restoring worker runs; the bank lands atomically with ``relocation.json`` and
passes ``read_rank_bank``; the TCP peer transport delivers byte-identical files.
"""
import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile
import threading
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hetero import run_scope  # noqa: E402
from hetero import exec_bank  # noqa: E402
from hetero.weight_snapshot import SnapshotRefused, prepare, relative_chip_coords, restore, snapshot  # noqa: E402

_spec = importlib.util.spec_from_file_location("startup_state", ROOT / "scripts" / "startup_state.py")
ss = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("startup_state", ss)  # cloudpickle's by-value registration needs it importable
_spec.loader.exec_module(ss)

DIGEST = "f" * 64


class PlacementContextTests(unittest.TestCase):
    def test_congruent_placements_share_the_context_and_others_do_not(self):
        a = {"10.0.0.1": [[0, 0, 0], [1, 0, 0]], "10.0.0.2": [[2, 0, 0], [3, 0, 0]]}
        b = {"10.0.0.7": [[1, 4, 0], [0, 4, 0]], "10.0.0.9": [[3, 4, 0], [2, 4, 0]]}  # shifted, chips reordered
        self.assertEqual(relative_chip_coords(["10.0.0.1", "10.0.0.2"], a),
                         relative_chip_coords(["10.0.0.7", "10.0.0.9"], b))
        # Swapped rank order is a different placement (rank r's shards would land on other chips).
        self.assertNotEqual(relative_chip_coords(["10.0.0.1", "10.0.0.2"], a),
                            relative_chip_coords(["10.0.0.9", "10.0.0.7"], b))
        column = {"10.0.0.1": [[0, 0, 0], [0, 1, 0]], "10.0.0.2": [[0, 2, 0], [0, 3, 0]]}
        self.assertNotEqual(relative_chip_coords(["10.0.0.1", "10.0.0.2"], a),
                            relative_chip_coords(["10.0.0.1", "10.0.0.2"], column))


def _tree(mesh):
    n = len(mesh.devices.flat)
    return {"w": jax.device_put(jnp.arange(n * 16, dtype=jnp.bfloat16).reshape(n * 4, 4),
                                NamedSharding(mesh, PartitionSpec("tp", None))),
            "b": jax.device_put(jnp.ones((n * 2,), dtype=jnp.bfloat16), NamedSharding(mesh, PartitionSpec("tp")))}


class ShipTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        for name, sub in (("SNAPSHOT_RUNS_PARENT", "shm/omp_weights_runs"),
                          ("BANK_RUNS_PARENT", "cache/omp_exec_bank_runs"),
                          ("XLA_RUNS_PARENT", "cache/omp_xla_runs"),
                          ("CKPT_RUNS_PARENT", "shm/omp_ckpt_runs")):
            patcher = mock.patch.object(run_scope, name, str(self.root / sub))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("tp",))
        self.context = {"model_path": "/models/test", "tp_size": 1, "dtype": "bfloat16",
                        "chip_coords_by_rank": [[[0, 0, 0]]]}
        self.tree = _tree(self.mesh)
        # Seed scope: large_a's snapshot, bank (one digest) and XLA cache (nested names).
        snap = pathlib.Path(run_scope.scope_roots("seed")["weight_snapshot_root"]) / "H/large_a/rank0"
        snapshot(snap, self.tree, self.context, generation="e" * 32)
        bank = pathlib.Path(run_scope.scope_roots("seed")["exec_bank_dir"]) / DIGEST / "rank0"
        bank.mkdir(parents=True)
        entries = {}
        for name, data in (("run_model-1.bin", b"y" * 7), ("sample-2.bin", b"z" * 3)):
            (bank / name).write_bytes(data)
            entries[name] = {"key": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        (bank / "manifest.json").write_text(json.dumps({"identity_digest": DIGEST, "entries": entries,
                                                        "identity": {"rank": {"process_index": 0}}}))
        xla = pathlib.Path(run_scope.scope_xla_dir("seed"))
        (xla / "sub").mkdir(parents=True)
        (xla / "jit_a-cache").write_bytes(b"a" * 5)
        (xla / "sub" / "jit_b-cache").write_bytes(b"b" * 9)

    def _ship(self, parts, target="H:large_b"):
        """What every transport does: list on the source, copy each file to its dest_path (fresh mtimes)."""
        ss.prepare_target("arm", target, 0, parts)
        files = ss.source_files("seed", "H:large_a", 0, parts)
        for f in files:
            dest = ss.dest_path("arm", target, 0, f["part"], f["rel"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f["path"], dest)
        return files, ss.finalize("arm", target, 0, parts, {"method": "test", "source_host": "h0"})

    def test_relocated_snapshot_restores_on_the_target_placement(self):
        files, report = self._ship(["snapshot", "bank", "xla"])
        target = pathlib.Path(run_scope.scope_roots("arm")["weight_snapshot_root"]) / "H/large_b/rank0"
        self.assertTrue((target / "manifest.json").exists())
        self.assertFalse((target / "manifest.json.incoming").exists())
        self.assertEqual(report["snapshot"]["generation"], "e" * 32)
        clocks = {}
        prepared = prepare(target, self.context, self.mesh, clocks, verify_checksums=False)
        abstract = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), self.tree)
        restored = restore(abstract, self.mesh, prepared, clocks)
        for key in self.tree:
            np.testing.assert_array_equal(np.asarray(restored[key]), np.asarray(self.tree[key]))
        # Bank: published atomically with provenance, staging gone, entries sha256-verified.
        bank = pathlib.Path(run_scope.scope_roots("arm")["exec_bank_dir"])
        self.assertEqual(sorted(p.name for p in bank.iterdir()), [DIGEST])
        relocation = json.loads((bank / DIGEST / "rank0" / "relocation.json").read_text())
        self.assertEqual((relocation["method"], relocation["source_host"]), ("test", "h0"))
        self.assertEqual(len(exec_bank.read_rank_bank(str(bank), DIGEST, 0)["files"]), 2)
        self.assertEqual(report["bank"]["entries"], 2)
        # XLA cache: every file, nested names kept.
        xla = pathlib.Path(run_scope.scope_xla_dir("arm"))
        self.assertEqual((xla / "sub" / "jit_b-cache").read_bytes(), b"b" * 9)
        self.assertEqual(report["xla"]["files"], 2)
        self.assertEqual({f["part"] for f in files}, {"snapshot", "bank", "xla"})
        # Delivery marker: published last, where a state-gated worker of H:large_b rank 0 looks.
        from hetero.state_delivery import marker_path
        marker = marker_path(run_scope.scope_roots("arm")["weight_snapshot_root"], "H", "large_b", 0)
        self.assertEqual(report["marker"], str(marker))
        delivered = json.loads(marker.read_text())
        self.assertEqual((delivered["rank"], delivered["parts"], delivered["provenance"]["source_host"]),
                         (0, ["snapshot", "bank", "xla"], "h0"))
        with self.assertRaisesRegex(RuntimeError, "delivery marker"):
            ss.prepare_target("arm", "H:large_b", 0, ["xla"])
        marker.unlink()
        # A second fetch into the same scope is refused part by part; cleanup removes all three roots.
        for part in ("snapshot", "bank", "xla"):
            with self.assertRaises(RuntimeError, msg=part):
                ss.prepare_target("arm", "H:large_b", 0, [part])
        checkpoint = pathlib.Path(run_scope.scope_checkpoint_dir("arm")) / "gemma"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"w" * 11)
        removed = run_scope.remove_scope("arm")
        self.assertEqual(set(removed), {"weight_snapshot_root", "exec_bank_dir", "compilation_cache_dir",
                                        "checkpoint_dir"})
        self.assertTrue(all(r["absent_after"] for r in removed.values()))

    def test_unpublished_or_tampered_snapshot_is_refused(self):
        ss.prepare_target("arm", "H:large_b", 0, ["snapshot"])
        target = pathlib.Path(run_scope.scope_roots("arm")["weight_snapshot_root"]) / "H/large_b/rank0"
        for f in ss.source_files("seed", "H:large_a", 0, ["snapshot"]):
            dest = ss.dest_path("arm", "H:large_b", 0, f["part"], f["rel"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f["path"], dest)
        with self.assertRaises(SnapshotRefused):  # manifest still under its incoming name
            prepare(target, self.context, self.mesh, {}, verify_checksums=False)
        shard = next(p for p in target.iterdir() if p.suffix == ".raw")
        with shard.open("ab") as handle:
            handle.write(b"\0")
        with self.assertRaises(RuntimeError):
            ss.finalize("arm", "H:large_b", 0, ["snapshot"], {})

    def test_names_cannot_escape_their_part(self):
        for part, rel in (("xla", "../x"), ("xla", "/etc/passwd"), ("snapshot", "a/b.raw"),
                          ("bank", "digest"), ("bank", ".incoming/rank0"), ("snapshot", "")):
            with self.assertRaises(ValueError, msg=(part, rel)):
                ss.dest_path("arm", "H:large_b", 0, part, rel)
        self.assertEqual(ss.gcs_object("p/q", 2, "bank", f"{DIGEST}/m.json"), f"p/q/rank2/bank/{DIGEST}/m.json")
        self.assertEqual(ss.gcs_object("p/q", 2, "xla", "sub/k"), "p/q/rank2/xla/sub/k")
        self.assertEqual(ss.split_gcs("gs://b/omp/x/"), ("b", "omp/x"))
        self.assertEqual(ss.parse_parts("xla,snapshot"), ["snapshot", "xla"])
        for bad in ("", "snapshot,snapshot", "weights"):
            with self.assertRaises(ValueError):
                ss.parse_parts(bad)


class PeerTransportTests(unittest.TestCase):
    def test_remote_functions_pickle_by_value(self):
        # Ray runs the script as __main__ and so pickles its classes and functions by value, globals
        # included (a module-level struct.Struct broke every peer fetch on the cluster).
        from ray import cloudpickle
        cloudpickle.register_pickle_by_value(ss)
        try:
            for obj in (ss.Receiver, ss.send_files, ss.prepare_target, ss.finalize, ss.source_files,
                        ss.gcs_upload, ss.gcs_fetch, ss.fuse_fetch, ss.gcs_delete, ss.gcs_absent,
                        ss.probe_write, ss.probe_cleanup, ss.checkpoint_fetch, ss.gcs_download_dir,
                        ss.checkpoint_fuse_fetch, ss.ckpt_io_variant, ss.io_cleanup):
                cloudpickle.dumps(obj)
        finally:
            cloudpickle.unregister_pickle_by_value(ss)

    def test_streams_deliver_every_file_once_and_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = pathlib.Path(tmp, "src"), pathlib.Path(tmp, "dst")
            src.mkdir()
            rng = np.random.default_rng(0)
            items, plan = [], []
            for i, size in enumerate((0, 1, 3 << 20, (17 << 20) + 5, 4096)):
                path = src / f"f{i}"
                path.write_bytes(rng.integers(0, 256, size, dtype=np.uint8).tobytes())
                items.append([i, str(path), size])
                plan.append([str(dst / "nested" / f"f{i}"), size])
            receiver = ss.Receiver(plan, connections=3, timeout=60)
            box = {}
            thread = threading.Thread(target=lambda: box.update(r=receiver.run()))
            thread.start()
            sent = ss.send_files("127.0.0.1", receiver.port(), items, streams=3, timeout=60)
            thread.join(60)
            self.assertEqual(box["r"]["files"], len(items))
            self.assertEqual(box["r"]["bytes"], sent["bytes"])
            for (_, path, _), (dest, _) in zip(items, plan):
                self.assertEqual(pathlib.Path(dest).read_bytes(), pathlib.Path(path).read_bytes())

    def test_missing_file_fails_the_receiver(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp, "a")
            path.write_bytes(b"abc")
            receiver = ss.Receiver([[str(pathlib.Path(tmp, "out", "a")), 3], [str(pathlib.Path(tmp, "out", "b")), 2]],
                                   connections=1, timeout=30)
            box = {}

            def run():
                try:
                    receiver.run()
                except RuntimeError as exc:
                    box["error"] = str(exc)

            thread = threading.Thread(target=run)
            thread.start()
            ss.send_files("127.0.0.1", receiver.port(), [[0, str(path), 3]], streams=1, timeout=30)
            thread.join(30)
            self.assertIn("missing=1", box["error"])


class CheckpointReadTests(unittest.TestCase):
    """fetch-checkpoint --method fuse and the ckpt-io probe, against a directory standing in for /models."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        for target, name, value in ((ss, "FUSE_MOUNT", self.root / "models"),
                                    (run_scope, "CKPT_RUNS_PARENT", str(self.root / "shm/omp_ckpt_runs")),
                                    (ss, "IO_ROOT", str(self.root / "shm/omp_ckpt_io"))):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.ckpt = self.root / "models" / "gemma-test"
        self.ckpt.mkdir(parents=True)
        rng = np.random.default_rng(1)
        self.files = {"model-00001-of-00002.safetensors": (3 << 20) + 7, "model-00002-of-00002.safetensors": 1,
                      "config.json": 11, "empty.txt": 0}
        for name, size in self.files.items():
            (self.ckpt / name).write_bytes(rng.integers(0, 256, size, dtype=np.uint8).tobytes())

    def test_chunks_cover_every_byte_once(self):
        jobs = ss.chunk_jobs([("a", 10), ("b", 0), ("c", 4)], 4)
        self.assertEqual(jobs, [("a", 0, 4), ("a", 4, 4), ("a", 8, 2), ("c", 0, 4)])

    def test_fuse_checkpoint_copy_is_byte_identical_and_refuses_a_second_copy(self):
        report = ss.checkpoint_fuse_fetch("gemma-test", "arm", workers=3, chunk_bytes=1 << 20)
        dest = pathlib.Path(run_scope.scope_checkpoint_dir("arm")) / "gemma-test"
        self.assertEqual(report["directory"], str(dest))
        self.assertEqual(sorted(p.name for p in dest.iterdir()), sorted(n for n, s in self.files.items() if s))
        for path in dest.iterdir():
            self.assertEqual(path.read_bytes(), (self.ckpt / path.name).read_bytes(), path.name)
        self.assertEqual(report["bytes"], sum(self.files.values()))
        with self.assertRaises(RuntimeError):
            ss.checkpoint_fuse_fetch("gemma-test", "arm", workers=3)

    def test_io_variants_read_every_safetensors_byte(self):
        safetensors = sum(s for n, s in self.files.items() if n.endswith(".safetensors"))
        for variant in ("seq", "par-3", "vpf"):
            row = ss.ckpt_io_variant(ss.FUSE_BUCKET, "gemma-test", variant)
            self.assertEqual(row["bytes"], safetensors, variant)
            self.assertTrue(0.0 <= row["resident_before"] <= 1.0)
        with self.assertRaises(ValueError):
            ss.ckpt_io_variant(ss.FUSE_BUCKET, "gemma-test", "mmap")

    def test_cli_refuses_unknown_variants_and_peer_checkpoints(self):
        for argv in (["ckpt-io", "--gcs", "gs://b/m", "--target", "H:large_b", "--variants", "par-x"],
                     ["ckpt-io", "--gcs", "gs://b/m", "--target", "H:large_b", "--variants", "st-lazy,mmap"],
                     ["ckpt-io", "--gcs", "gs://b/m", "--target", "H:large_b", "--variants", "st-lazy+seq"],
                     ["ckpt-io", "--gcs", "gs://b/m", "--target", "H:large_b", "--variants", "st-lazy+gcs-8"],
                     ["fetch-checkpoint", "--gcs", "gs://b/m", "--target", "H:large_b", "--target-scope", "arm",
                      "--method", "peer"]):
            with mock.patch.object(sys, "argv", ["startup_state.py", *argv]), \
                    mock.patch("sys.stderr"), self.assertRaises(SystemExit, msg=argv):
                ss.main()


if __name__ == "__main__":
    unittest.main()
