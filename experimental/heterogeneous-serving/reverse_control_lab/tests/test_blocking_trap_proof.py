"""
Strict Proof of Reverse Control Blocking Trap & In-Flight Resolution.
Proves beyond doubt that:
1. StageRunner freezes completely in enter_diagnostic_trap() when an error occurs.
2. Downstream stages are strictly blocked from executing while in the trap.
3. Only when an external steering directive is injected does the stage unpause.
4. The injected mutation/hot-patch takes effect on the retry attempt.
"""

import json
import pathlib
import sys
import threading
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omp_rcm import (
    StageRunner,
    StateStore,
    JointTimingLogEngine,
    stage,
)

TRAPS_DIR = pathlib.Path("/tmp/omp_rcm/traps")
ACTION_FILE = pathlib.Path("/tmp/omp_rcm/control_action.json")


def run_proof():
    TRAPS_DIR.mkdir(parents=True, exist_ok=True)
    if ACTION_FILE.exists():
        ACTION_FILE.unlink()

    runner = StageRunner(
        silence_timeout_s=5.0,
        enable_steering_socket=False,
    )

    stage_events = []
    timestamps = {}

    @stage(name="stage_1_alloc", order_idx=1, expected_s=0.5)
    def stage_1(state: StateStore, timing: JointTimingLogEngine):
        stage_events.append("stage_1_started")
        state["hbm_allocated_gb"] = 62.4
        state.set_param("batch_size", 4)  # Invalid batch size
        timing.log("Stage 1: HBM allocated with initial batch_size=4", level="INFO")
        stage_events.append("stage_1_finished")

    @stage(name="stage_2_validate_and_compile", order_idx=2, expected_s=0.5)
    def stage_2(state: StateStore, timing: JointTimingLogEngine):
        bs = state.get_param("batch_size")
        stage_events.append(f"stage_2_attempt_with_bs_{bs}")
        timing.log(f"Stage 2 executing with batch_size={bs}", level="INFO")

        if bs < 32:
            raise ValueError(f"ConfigValidationError: batch_size={bs} < 32 is invalid for TPU v5e Megascale")

        state["compiled_grid"] = f"AOT_COMPILED_BS_{bs}"
        stage_events.append("stage_2_success")

    @stage(name="stage_3_downstream_benchmark", order_idx=3, expected_s=0.5)
    def stage_3(state: StateStore, timing: JointTimingLogEngine):
        stage_events.append("stage_3_started")
        grid = state.get("compiled_grid")
        bs = state.get_param("batch_size")
        timing.log(f"Stage 3 consuming compiled grid: {grid} (bs={bs})", level="INFO")
        assert grid == "AOT_COMPILED_BS_64", f"Unexpected grid: {grid}"
        assert bs == 64, f"Unexpected batch_size: {bs}"
        stage_events.append("stage_3_finished")

    runner.register(stage_1)
    runner.register(stage_2)
    runner.register(stage_3)

    trap_file = TRAPS_DIR / "stage_2_validate_and_compile.json"
    if trap_file.exists():
        trap_file.unlink()

    # External Watcher that intentionally waits 1.5 seconds before solving the trap
    def external_steerer():
        start_wait = time.time()
        # Wait until the trap file is created by Stage 2's failure
        while not trap_file.exists() and time.time() - start_wait < 5.0:
            time.sleep(0.02)

        assert trap_file.exists(), "Trap file was never created!"
        timestamps["trap_observed_at"] = time.time()
        print(f"\n[EXTERNAL_WATCHER] Trap observed at t={timestamps['trap_observed_at'] - timestamps['run_started_at']:.3f}s", flush=True)

        # Inspect trap snapshot
        with open(trap_file, "r", encoding="utf-8") as f:
            snapshot = json.load(f)

        print(f"[EXTERNAL_WATCHER] Snapshot reason: {snapshot['reason']}", flush=True)
        assert "ConfigValidationError: batch_size=4" in snapshot["reason"]

        # ASSERTION BARRIER 1: Verify Stage 3 has NOT run while trapped
        assert "stage_3_started" not in stage_events, "Stage 3 started prematurely while trapped!"
        assert "stage_2_success" not in stage_events, "Stage 2 marked success while trapped!"

        # Intentionally hold the trap open for 1.5 seconds
        hold_duration = 1.5
        print(f"[EXTERNAL_WATCHER] Intentionally holding trap open for {hold_duration}s to prove pipeline is frozen...", flush=True)
        time.sleep(hold_duration)

        # ASSERTION BARRIER 2: After 1.5 seconds, verify pipeline is STILL frozen
        assert "stage_3_started" not in stage_events, "Stage 3 started during hold period!"
        assert "stage_2_success" not in stage_events, "Stage 2 succeeded during hold period!"
        print("[EXTERNAL_WATCHER] Confirmed: Pipeline remained 100% frozen during hold period.", flush=True)

        # Inject the fix
        print("[EXTERNAL_WATCHER] Injecting MUTATE_CONFIG directive: batch_size=64...", flush=True)
        timestamps["fix_injected_at"] = time.time()
        with open(ACTION_FILE, "w", encoding="utf-8") as f:
            json.dump({"op": "MUTATE_CONFIG", "params": {"batch_size": 64}}, f)

    steerer_thread = threading.Thread(target=external_steerer, daemon=True)

    timestamps["run_started_at"] = time.time()
    steerer_thread.start()

    success = runner.run()
    timestamps["run_finished_at"] = time.time()
    steerer_thread.join(timeout=3.0)

    total_duration = timestamps["run_finished_at"] - timestamps["run_started_at"]
    print(f"\n[PROOF_SUMMARY] Total Run Duration: {total_duration:.3f}s (held for >= 1.5s)")
    print(f"[PROOF_SUMMARY] Stage Event Sequence: {stage_events}")
    print(f"[PROOF_SUMMARY] Final State: batch_size={runner.state.get_param('batch_size')}, grid={runner.state.get('compiled_grid')}")

    # Rigorous Proof Assertions
    assert success is True, "Run failed!"
    assert total_duration >= 1.5, f"Run finished in {total_duration:.3f}s without waiting for the 1.5s hold!"
    assert stage_events == [
        "stage_1_started",
        "stage_1_finished",
        "stage_2_attempt_with_bs_4",  # First attempt fails & traps
        "stage_2_attempt_with_bs_64", # Second attempt succeeds after mutation
        "stage_2_success",
        "stage_3_started",            # Downstream stage executes ONLY after resolution
        "stage_3_finished",
    ], f"Invalid event sequence: {stage_events}"

    print("\n>>> ALL PROOF ASSERTIONS PASSED: Trap genuinely froze execution and mutation resolved it! <<<\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_proof())
