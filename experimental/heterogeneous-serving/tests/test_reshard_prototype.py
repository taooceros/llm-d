"""Value preservation and byte-bound regressions for the CPU-only experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def tiny_inventory():
    hosts = []
    for y in range(4):
        for x in range(2):
            rank = y * 2 + x
            hosts.append({"node_ip": f"host{rank}", "ray_node_id": str(rank),
                          "hostname": f"host{rank}", "tpu_worker_id": rank,
                          "tpu_name": "test", "tpu_count": 4,
                          "chip_coords": [[x * 2 + dx, y * 2 + dy, 0]
                                          for dy in range(2) for dx in range(2)],
                          "jax_process_index": rank, "jax_global_device_count": 32})
    return {"slice_name": "test", "hosts": hosts, "chip_dims": [4, 8, 1],
            "host_chip_dims": [2, 2, 1], "host_grid_dims": [2, 4, 1]}


class ReshardPrototype(unittest.TestCase):
    def test_round_trip_preserves_all_parameter_values_and_replica_byte_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            inventory = Path(tmp) / "inventory.json"
            output = Path(tmp) / "result.json"
            inventory.write_text(json.dumps(tiny_inventory()))
            environment = dict(os.environ, JAX_PLATFORMS="cpu",
                               XLA_FLAGS="--xla_force_host_platform_device_count=16")
            subprocess.run([sys.executable, str(ROOT / "scripts/reshard_prototype.py"),
                            "--inventory", str(inventory), "--output", str(output),
                            "--width", "16", "--layers", "1"],
                           env=environment, cwd=ROOT, capture_output=True, text=True,
                           check=True, timeout=120)
            result = json.loads(output.read_text())
        self.assertEqual(result["equality"], {"S_to_H": True, "H_to_S_both_replicas": True})
        merge = result["prototype_movement_lower_bound"]["S_to_H"]
        split = result["prototype_movement_lower_bound"]["H_to_S"]
        # Expanding one model into two replicas adds one full copy. Existing
        # local slices must be retained symmetrically in both directions.
        self.assertEqual(split["received_bytes"] - merge["received_bytes"],
                         result["tp_sharded_parameter_bytes"])
        transitions = result["transitions"]
        self.assertEqual(transitions["H->S"]["received_bytes_estimate"] -
                         transitions["S->H"]["received_bytes_estimate"], 125_000_000_000)
        self.assertEqual(transitions["M->S"]["received_bytes_estimate"],
                         transitions["H->S"]["received_bytes_estimate"] / 2)


if __name__ == "__main__":
    unittest.main()
