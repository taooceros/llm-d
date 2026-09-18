#!/usr/bin/env python3
"""
Interactive Multi-Stage Gemma-4 31B Workload using omp_rcm SDK.
Demonstrates fault injection (--force-hang, --force-error), auto-steering (--auto-steer),
and live watchdog monitoring.
"""

import argparse
import json
import os
import pathlib
import sys
import threading
import time

# Ensure repo root is on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omp_rcm import (
    DiagnosticTrap,
    JointTimingLogEngine,
    StageRunner,
    StateStore,
    stage,
)


@stage(name="tpu_topology_discovery", order_idx=1, expected_s=1.0)
def stage_topology(state: StateStore, timing: JointTimingLogEngine):
    timing.log("Probing cluster nodes for Cloud TPU v5e devices...", level="INFO")
    time.sleep(0.3)
    target_hosts = ["192.168.102.18", "192.168.102.19"]
    timing.log(f"Discovered 2 TPU worker hosts: {target_hosts}", level="INFO")
    time.sleep(0.2)

    state["topology"] = {
        "type": "2x4_tpu_v5e",
        "mesh_shape": (2, 4),
        "target_hosts": target_hosts,
        "ici_contiguous": True,
    }
    timing.log("Explicit node-pinned placement group locked successfully.", level="INFO")


@stage(name="weights_load_and_compile", order_idx=2, expected_s=2.0)
def stage_weights(state: StateStore, timing: JointTimingLogEngine):
    timing.log("Initializing Ray distributed worker group on TPU mesh...", level="INFO")
    time.sleep(0.4)

    # Check for injected fault
    if state.get("force_error") and state.get_param("batch_size", 4) < 32:
        timing.error("Validating execution parameters: batch_size=4 is invalid for TPU v5e Megascale")
        raise ValueError(
            "ConfigValidationError: batch_size=4 causes Megascale ICI under-utilization. "
            "Minimum required batch_size for 2x4 mesh is 32 or 64."
        )

    timing.log("Loading Gemma-4 31B sharded safetensors into HBM (TP=8)...", level="INFO")
    time.sleep(0.5)
    state["weights_handle"] = "SHARDED_WEIGHTS_OBJECT_REF_TPU_2x4_0xCAFEBABE"
    state["hbm_allocated_gb"] = 62.4
    timing.log(f"Allocated {state['hbm_allocated_gb']} GB HBM across 8 TPU cores (7.8 GB/chip)", level="INFO")

    timing.log("Compiling XLA ahead-of-time compute kernels...", level="INFO")
    time.sleep(0.4)
    state["compiled_kernels"] = ["gemm_pallas_fused", "paged_attention_v2", "rmsnorm_quant"]
    timing.log("XLA kernel compilation completed.", level="INFO")


@stage(name="decoupled_prefill_benchmark", order_idx=3, expected_s=2.0)
def stage_prefill(state: StateStore, timing: JointTimingLogEngine):
    timing.log("Executing isolated prefill benchmark (max_tokens=1)...", level="INFO")

    # Injected hang simulation
    if state.get("force_hang") and not state.get("_hang_recovered"):
        timing.warn("Simulating unrecoverable driver hang / deadlock in XLA Megascale runtime (60.0s sleep)...")
        state["_hang_recovered"] = True
        time.sleep(60.0)

    prompt_lens = [512, 1024, 2048]
    prefill_results = {}
    for plen in prompt_lens:
        timing.log(f"Prefill sequence length {plen} tokens...", level="INFO")
        time.sleep(0.2)
        ttft_ms = 12.5 + (plen / 1024.0) * 8.2
        throughput_tok_s = (plen / (ttft_ms / 1000.0))
        prefill_results[plen] = {
            "ttft_ms": round(ttft_ms, 2),
            "throughput_tokens_per_sec": round(throughput_tok_s, 1),
        }

    state["prefill_metrics"] = prefill_results
    timing.log(f"Prefill benchmark complete: TTFT @ 2048 = {prefill_results[2048]['ttft_ms']}ms", level="INFO")


@stage(name="pure_decode_dynamic_batch", order_idx=4, expected_s=2.0)
def stage_decode(state: StateStore, timing: JointTimingLogEngine):
    timing.log("Benchmarking pure decode iterations across dynamic batch sizes...", level="INFO")
    if state.get("force_kernel_bug"):
        timing.error("Pallas compute failure: JAX/Pallas custom kernel layout mismatch")
        raise RuntimeError("PallasAttentionKernelFault: Fused attention kernel crashed due to missing unrolled loop stride in TPU v5e Megascale")

    batch_size = state.get_param("batch_size", 64)
    timing.log(f"Operating decode with active batch_size={batch_size}...", level="INFO")
    decode_results = {}
    for step in range(1, 4):
        time.sleep(0.2)
        step_latency_ms = 3.8 + (batch_size * 0.02)
        throughput = (batch_size * 1000.0) / step_latency_ms
        timing.log(f"Decode step {step}/3: latency={step_latency_ms:.2f}ms, throughput={throughput:.1f} tok/s", level="INFO")
        decode_results[f"step_{step}"] = {
            "step_latency_ms": round(step_latency_ms, 2),
            "throughput": round(throughput, 1),
        }

    state["decode_metrics"] = decode_results
    timing.log("Pure decode dynamic batch benchmark settled.", level="INFO")


@stage(name="generate_serving_report", order_idx=5, expected_s=1.0)
def stage_report(state: StateStore, timing: JointTimingLogEngine):
    timing.log("Aggregating serving metrics into final JSON artifact...", level="INFO")
    time.sleep(0.2)
    report = {
        "model": "gemma-4-31b",
        "topology": state.get("topology"),
        "prefill": state.get("prefill_metrics"),
        "decode": state.get("decode_metrics"),
        "hbm_allocated_gb": state.get("hbm_allocated_gb"),
        "status": "COMPLETED",
    }
    state["final_report"] = report
    timing.log("Report generation finished. Artifact ready.", level="INFO")


def auto_steer_agent(runner: StageRunner, stop_event: threading.Event):
    """Simulates an OMP agent side watcher reacting to diagnostic trap snapshots."""
    traps_dir = runner.trap.traps_dir
    observed_traps = set()
    start_time = time.time()

    while not stop_event.is_set():
        time.sleep(0.1)
        if not traps_dir.exists():
            continue

        for trap_file in sorted(traps_dir.glob("*.json")):
            if trap_file.name == "latest.json":
                continue

            try:
                with open(trap_file, "r", encoding="utf-8") as f:
                    snapshot = json.load(f)

                ts = snapshot.get("timestamp", 0)
                if ts < start_time:
                    continue

                trap_key = f"{trap_file.name}_{ts}"
                if trap_key in observed_traps:
                    continue

                stage_name = snapshot.get("stage")
                reason = snapshot.get("reason", "")
                observed_traps.add(trap_key)

                # Diagnostic decision engine
                if "ConfigValidationError" in reason or "Invalid batch size" in reason:
                    print(
                        f"\n[SIDE_WATCHER_AGENT] Diagnosed config error in '{stage_name}'. "
                        f"Mutating parameter: batch_size=64 and retrying...",
                        flush=True,
                    )
                    runner.steering.inject_action({
                        "op": "MUTATE_CONFIG",
                        "params": {"batch_size": 64},
                    })
                elif "SILENCE_TIMEOUT" in reason or "SIGUSR1" in reason or "WATCHDOG" in reason or "SLA" in reason:
                    live_stack = snapshot.get("live_stack") or snapshot.get("exception", {}).get("traceback", "N/A")
                    print(
                        f"\n[SIDE_WATCHER_AGENT] Diagnosed watchdog stall/pause in '{stage_name}'.\n"
                        f"--- Captured Live Stack in Trapped Worker ---\n"
                        f"{live_stack.strip()}\n"
                        f"----------------------------------------------\n"
                        f"Clearing hang flag and issuing RETRY with fresh clocks...",
                        flush=True,
                    )
                    runner.state["force_hang"] = False
                    runner.steering.inject_action({"op": "RETRY"})
                elif "PallasAttentionKernelFault" in reason or "KernelFault" in reason:
                    print(
                        f"\n[SIDE_WATCHER_AGENT] Diagnosed Pallas kernel runtime fault in '{stage_name}'.\n"
                        f"--- Emitting In-Memory LIVE_PATCH for stage '{stage_name}' (Zero-Disk I/O) ---\n",
                        flush=True,
                    )
                    runner.steering.inject_action({
                        "op": "LIVE_PATCH",
                        "stage": stage_name,
                        "code": (
                            "def pure_decode_dynamic_batch(state, timing):\n"
                            "    timing.log('Operating with LIVE_PATCHED fused attention kernel!', level='INFO')\n"
                            "    batch_size = state.get_param('batch_size', 64)\n"
                            "    decode_results = {}\n"
                            "    for step in range(1, 4):\n"
                            "        time.sleep(0.1)\n"
                            "        step_latency_ms = 2.9 + (batch_size * 0.015)\n"
                            "        throughput = (batch_size * 1000.0) / step_latency_ms\n"
                            "        timing.log(f'Live-patched decode step {step}/3: latency={step_latency_ms:.2f}ms', level='INFO')\n"
                            "        decode_results[f'step_{step}'] = {'step_latency_ms': round(step_latency_ms, 2), 'throughput': round(throughput, 1)}\n"
                            "    state['decode_metrics'] = decode_results\n"
                            "    timing.log('Pure decode dynamic batch settled with live-patched kernel.', level='INFO')\n"
                        ),
                    })
                else:
                    print(
                        f"\n[SIDE_WATCHER_AGENT] Diagnosed fault in '{stage_name}'. "
                        f"Issuing default RETRY...",
                        flush=True,
                    )
                    runner.steering.inject_action({"op": "RETRY"})

            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="OMP Reverse Control Mechanism Demo Workload")
    parser.add_argument("--force-hang", action="store_true", help="Inject unrecoverable silence hang into prefill stage")
    parser.add_argument("--force-error", action="store_true", help="Inject invalid batch_size config error into weights stage")
    parser.add_argument("--force-kernel-bug", action="store_true", help="Inject Pallas kernel failure into decode stage (resolvable via LIVE_PATCH)")
    parser.add_argument("--auto-steer", action="store_true", help="Spawn autonomous side watcher to resolve diagnostic traps")
    parser.add_argument("--silence-timeout", type=float, default=2.0, help="Silence watchdog timeout in seconds")
    args = parser.parse_args()

    runner = StageRunner(
        silence_timeout_s=args.silence_timeout,
        enable_steering_socket=True,
        watchdog_interval_s=0.2,
    )

    # Register stages
    runner.register(stage_topology)
    runner.register(stage_weights)
    runner.register(stage_prefill)
    runner.register(stage_decode)
    runner.register(stage_report)

    # Configure initial fault states
    if args.force_hang:
        runner.state["force_hang"] = True
    if args.force_error:
        runner.state["force_error"] = True
        runner.state.set_param("batch_size", 4)
    if args.force_kernel_bug:
        runner.state["force_kernel_bug"] = True
    stop_watcher = threading.Event()
    watcher_thread = None
    if args.auto_steer:
        watcher_thread = threading.Thread(target=auto_steer_agent, args=(runner, stop_watcher), daemon=True)
        watcher_thread.start()

    try:
        success = runner.run()
        print(f"\n[DEMO_SUMMARY] Workload completed with success={success}")
        if success and "final_report" in runner.state:
            print("[DEMO_SUMMARY] Final Report Summary:")
            print(json.dumps(runner.state["final_report"], indent=2))
        return 0 if success else 1
    finally:
        stop_watcher.set()
        if watcher_thread:
            watcher_thread.join(timeout=1.0)


if __name__ == "__main__":
    sys.exit(main())
