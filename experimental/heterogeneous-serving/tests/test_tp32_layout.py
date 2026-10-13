import argparse
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hetero.engine_actor import EngineSettings  # noqa: E402
from hetero.experiment import ARMS, PILOT_ARMS  # noqa: E402
from hetero.layout import LayoutRequest, resolve_layout  # noqa: E402
from hetero.slice_inventory import SliceInventory  # noqa: E402

LAYOUT_FILE = ROOT / "deploy/layouts/T32.json"


def v5e_32_inventory() -> SliceInventory:
    hosts = []
    for gy in range(4):
        for gx in range(2):
            index = len(hosts)
            hosts.append({
                "node_ip": f"10.0.0.{index + 1}", "ray_node_id": f"node{index}",
                "hostname": f"host{index}", "tpu_worker_id": index, "tpu_name": "slice",
                "tpu_count": 4, "jax_process_index": index, "jax_global_device_count": 32,
                "chip_coords": [[2 * gx + dx, 2 * gy + dy, 0] for dy in (0, 1) for dx in (0, 1)],
            })
    return SliceInventory.from_dict({
        "slice_name": "slice", "hosts": hosts, "chip_dims": [4, 8, 1],
        "host_chip_dims": [2, 2, 1], "host_grid_dims": [2, 4, 1]})


class TP32LayoutTest(unittest.TestCase):
    def test_layout_file_resolves_one_engine_over_whole_slice(self):
        request = LayoutRequest.from_dict(json.loads(LAYOUT_FILE.read_text()))
        resolved = resolve_layout(v5e_32_inventory(), request).to_dict()
        self.assertEqual(resolved["layout_id"], "T32")
        (instance,) = resolved["instances"]
        self.assertEqual(instance["instance_id"], "large")
        self.assertEqual(instance["tp_size"], 32)
        self.assertEqual(len(instance["host_ips"]), 8)

    def test_routing_policy_reads_layout_file_roles(self):
        from hetero.cli import _routing_policy

        args = argparse.Namespace(policy_file=None, policy=None, layout="T32",
                                  layout_file=str(LAYOUT_FILE), migration=False)
        policy = _routing_policy(args)
        self.assertEqual(policy["large_instances"], ["large"])
        self.assertEqual(policy["small_instances"], [])

    def test_t32_is_pilot_only_arm(self):
        self.assertNotIn("T32", ARMS)
        self.assertEqual(PILOT_ARMS["T32"]["layout"], "T32")
        self.assertFalse(PILOT_ARMS["T32"]["migration_enabled"])


class MMEncoderTPModeTest(unittest.TestCase):
    def test_default_omits_backend_key(self):
        config = EngineSettings.from_dict({}).additional_config({"p": 1})
        self.assertNotIn("mm-encoder-tp-mode", config)
        self.assertEqual(config["omp_hetero"], {"p": 1})
        # Full-bf16 default (chain 69a705c): activation_dtype unset, the model dtype applies.
        self.assertIsNone(config["activation_dtype"])

    def test_data_mode_passes_through(self):
        settings = EngineSettings.from_dict({"mm_encoder_tp_mode": "data"})
        self.assertEqual(settings.additional_config({})["mm-encoder-tp-mode"], "data")

    def test_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "mm_encoder_tp_mode"):
            EngineSettings.from_dict({"mm_encoder_tp_mode": "tensor"})


if __name__ == "__main__":
    unittest.main()
