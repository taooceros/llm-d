"""An existing TPU pool must never be implicitly replaced on config drift."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


class PoolDriftSafetyTest(unittest.TestCase):
    def test_drift_aborts_before_creating_an_earlier_missing_pool(self):
        script = Path(__file__).resolve().parents[1] / "scripts/py/cluster_setup.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "calls.jsonl"
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({
                "project_id": "test-project", "region": "us-west1", "zone": "us-west1-c",
                "cluster_name": "test-cluster", "network": "test-net", "subnetwork": "test-subnet",
                "node_pools": [
                    {"name": "missing-pool", "machine_type": "ct5lp-hightpu-4t"},
                    {"name": "occupied-pool", "machine_type": "ct5lp-hightpu-4t"},
                ],
            }))
            executable = root / "gcloud"
            executable.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["CALL_LOG"], "a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:4] == ["container", "clusters", "list"]:
    print(json.dumps([{"name": "test-cluster", "status": "RUNNING",
                       "network": "test-net", "subnetwork": "test-subnet"}]))
elif sys.argv[1:4] == ["container", "node-pools", "list"]:
    print(json.dumps([{"name": "occupied-pool", "status": "RUNNING",
                       "config": {"machineType": "ct5lp-hightpu-8t"}}]))
else:
    sys.exit(99)
''')
            executable.chmod(0o755)
            result = subprocess.run(
                [sys.executable, str(script), f"--config={config}", "--yes"],
                cwd=root, env={**os.environ, "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                               "CALL_LOG": str(log), "PYTHONDONTWRITEBYTECODE": "1"},
                text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("occupied-pool", result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertTrue(calls)
            self.assertTrue(all("list" in call for call in calls), calls)


if __name__ == "__main__":
    unittest.main()
