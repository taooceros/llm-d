"""The frozen-cohort gate that decides whether a timing run may start at all."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hetero.experiment import (ADMITTED_CAP, ARMS, BASE_POLICY, WARMUP_MAX_TOKENS,
                               WARMUP_REQUESTS, _frozen_cohort)
from hetero.engine_actor import EngineSettings
from hetero.native_backend import BACKEND_REVISION, RUNTIME_REVISION
from hetero.manifest import load_manifest


def _freeze(**overrides) -> dict:
    names = sorted({str(path.relative_to(ROOT)) for path in (ROOT / "hetero").rglob("*")
                    if path.is_file() and path.suffix in (".py", ".json")} | {
        "scripts/hetero_ctl.py", "deploy/k8s/omp-gateway.yaml", "deploy/k8s/omp-router.yaml"})
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    settings = EngineSettings().to_dict()
    # These CPU guard fixtures do not assert physical qualification. The
    # freezer separately validates the raw reports before issuing a real cohort.
    evidence_path = "results/hetero/20260918/minimal_fp32_state_continuation.json"
    evidence = {"path": evidence_path,
                "file_sha256": hashlib.sha256((ROOT / evidence_path).read_bytes()).hexdigest()}
    manifests = []
    for name in ("W0", "W1", "W2", "E_SHORT", "E_LONG"):
        manifest = load_manifest(name, ROOT)
        path = f"data/manifests/{name}.json"
        manifests.append({**{k: v for k, v in manifest.items() if k != "requests"},
                          "path": path, "file_sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()})
    payload = {
        "source_freeze": {
            "status": "qualified", "commit_id": "0" * 40, "source_sha256": sources,
            "tree_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "timing_authorization": {"authorized_arms": ["H", "S", "M0"],
                                     "unauthorized_reason": "migration not admitted in this fixture"},
        },
        "qualified_runtime": {"backend_revision": BACKEND_REVISION, "runtime_revision": RUNTIME_REVISION,
                              "engine_settings": settings,
                              "evidence": {key: dict(evidence) for key in (
                                  "gate_H", "gate_S", "gate_M", "lifecycle", "continuation", "recompute", "kv_host")}},
        "engine_settings": settings, "manifests": manifests,
        "routing_and_migration_policy": BASE_POLICY, "arms": ARMS,
        "admitted_population_cap": ADMITTED_CAP,
        "warmup": {"requests": WARMUP_REQUESTS, "max_tokens": WARMUP_MAX_TOKENS},
    }
    payload["source_freeze"].update(overrides.pop("source_freeze", {}))
    payload.update(overrides)
    payload["predeclaration_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _write(payload: dict) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(payload, handle)
    handle.close()
    return handle.name


class FrozenCohortTests(unittest.TestCase):
    def setUp(self):
        self.paths = []
        self.addCleanup(lambda: [pathlib.Path(path).unlink(missing_ok=True) for path in self.paths])
        _frozen_cohort(self.write(_freeze()), "H")

    def write(self, payload):
        path = _write(payload)
        self.paths.append(path)
        return path

    def test_unauthorized_migration_arm_cannot_be_timed(self) -> None:
        path = self.write(_freeze())
        for arm in ("MR", "MK-H"):
            with self.assertRaises(ValueError):
                _frozen_cohort(path, arm)

    def test_missing_protocol_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            _frozen_cohort(None, "H")

    def test_pending_source_freeze_blocks_every_arm(self) -> None:
        path = self.write(_freeze(source_freeze={"status": "pending"}))
        with self.assertRaises(ValueError):
            _frozen_cohort(path, "H")

    def test_edited_amendment_body_is_rejected(self) -> None:
        payload = _freeze()
        payload["source_freeze"]["timing_authorization"]["authorized_arms"].append("MR")
        with self.assertRaises(ValueError):
            _frozen_cohort(self.write(payload), "MR")

    def test_source_drift_since_the_freeze_is_rejected(self) -> None:
        payload = _freeze()
        payload["source_freeze"]["source_sha256"]["hetero/experiment.py"] = "d" * 64
        payload["predeclaration_sha256"] = hashlib.sha256(
            json.dumps({k: v for k, v in payload.items() if k != "predeclaration_sha256"},
                       sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with self.assertRaises(ValueError):
            _frozen_cohort(self.write(payload), "H")

    def test_current_source_without_qualified_runtime_is_refused(self):
        with self.assertRaises(ValueError):
            _frozen_cohort(self.write(_freeze(qualified_runtime=None)), "H")


if __name__ == "__main__":
    unittest.main()
