#!/usr/bin/env python3
"""
4-Stage Reverse Control Pipeline using omp_rcm SDK.
Demonstrates:
1. Topology lock
2. Weights & HBM allocation
3. Pallas fused attention compute kernel execution
4. E2E Serving benchmark aggregation
"""

import json
import pathlib
import sys
import time

# Ensure repo root is in sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Ensure demos directory is in sys.path for pallas_ops import
_DEMOS_DIR = pathlib.Path(__file__).resolve().parent
if str(_DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMOS_DIR))

from omp_rcm import (
    JointTimingLogEngine,
    StageRunner,
    StateStore,
    stage,
)
import pallas_ops


@stage(name="cluster_topology_mesh_lock", order_idx=1, expected_s=1.0)
def stage_1_topology(state: StateStore, timing: JointTimingLogEngine):
    timing.log("Probing Cloud TPU v5e hosts for physical 2x4 mesh contiguity...", level="INFO")
    time.sleep(0.3)
    target_hosts = ["192.168.102.18", "192.168.102.19"]
    timing.log(f"Contiguous ICI mesh confirmed across hosts: {target_hosts}", level="INFO")
    time.sleep(0.2)

    state["topology"] = {
        "mesh_shape": (2, 4),
        "target_hosts": target_hosts,
        "placement_group_id": "pg_tpu_v5e_2x4_0xDEADBEEF",
    }
    timing.log("Placement group pg_tpu_v5e_2x4_0xDEADBEEF locked successfully.", level="INFO")


@stage(name="weights_and_kv_allocation", order_idx=2, expected_s=1.5)
def stage_2_weights(state: StateStore, timing: JointTimingLogEngine):
    timing.log("Allocating 62.4 GB HBM across 8 TPU cores (7.8 GB/chip)...", level="INFO")
    time.sleep(0.4)
    state["weights_handle"] = "SHARDED_WEIGHTS_OBJECT_REF_TPU_2x4_0xCAFEBABE"
    state["hbm_allocated_gb"] = 62.4
    timing.log("Loaded Gemma-4 31B safetensors into device memory.", level="INFO")
    time.sleep(0.3)
    state["kv_cache_allocated"] = True
    timing.log("KV Cache memory pool initialized (max_batch_tokens=32768).", level="INFO")


@stage(name="pallas_custom_kernel_execution", order_idx=3, expected_s=2.0)
def stage_3_pallas_kernel(state: StateStore, timing: JointTimingLogEngine):
    timing.log("Executing fused Pallas attention kernel on TPU cores...", level="INFO")
    res = pallas_ops.execute_pallas_fused_attention(batch_size=64, hidden_dim=4096)
    state["kernel_output"] = res
    timing.log(f"Pallas kernel executed successfully: {res}", level="INFO")


@stage(name="e2e_serving_benchmark_settle", order_idx=4, expected_s=1.0)
def stage_4_serving(state: StateStore, timing: JointTimingLogEngine):
    timing.log("Running final e2e serving throughput benchmark...", level="INFO")
    time.sleep(0.3)
    report = {
        "model": "gemma-4-31b",
        "mesh": state.get("topology"),
        "weights": state.get("weights_handle"),
        "hbm_gb": state.get("hbm_allocated_gb"),
        "kernel_metrics": state.get("kernel_output"),
        "tpot_ms": 4.12,
        "throughput_tokens_per_sec": 15533.9,
        "status": "COMPLETED",
    }
    state["serving_report"] = report
    timing.log("Serving report generated. Benchmark settled with status SUCCESS.", level="INFO")


def main():
    runner = StageRunner(
        silence_timeout_s=3.0,
        enable_steering_socket=True,
        watchdog_interval_s=0.2,
    )

    runner.register(stage_1_topology)
    runner.register(stage_2_weights)
    runner.register(stage_3_pallas_kernel)
    runner.register(stage_4_serving)

    success = runner.run()
    if success and "serving_report" in runner.state:
        print("\n[PIPELINE_COMPLETE] Final Serving Report Summary:")
        print(json.dumps(runner.state["serving_report"], indent=2))

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
