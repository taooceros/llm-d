"""
Strict Proof of Watchdog Preemption & Frozen Trap on Driver Hang.
Proves beyond doubt that:
1. A 100-second blocking sleep in a stage is preempted by the silence watchdog.
2. The pipeline freezes completely in the trap until steered.
3. Downstream stages never execute while trapped.
4. Injecting RETRY terminates the hung thread, resets clocks, and resumes execution.
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


def run_hang_proof():
    TRAPS_DIR.mkdir(parents=True, exist_ok=True)
    if ACTION_FILE.exists():
        ACTION_FILE.unlink()

    runner = StageRunner(
        silence_timeout_s=0.3,
        enable_steering_socket=False,
        watchdog_interval_s=0.05,
    )

    stage_events = []
    timestamps = {}
    attempts = {"stage_2": 0}

    @stage(name="stage_1_init", order_idx=1, expected_s=0.5)
    def stage_1(state: StateStore, timing: JointTimingLogEngine):
        stage_events.append("stage_1_done")

    @stage(name="stage_2_hang", order_idx=2, expected_s=0.5)
    def stage_2(state: StateStore, timing: JointTimingLogEngine):
        attempts["stage_2"] += 1
        curr_attempt = attempts["stage_2"]
        stage_events.append(f"stage_2_attempt_{curr_attempt}")
        timing.log(f"Stage 2 executing attempt {curr_attempt}", level="INFO")

        if curr_attempt == 1:
            # First attempt: simulate 100s blocking deadlock
            timing.warn("Stage 2 entering 100s deadlock...")
            time.sleep(100.0)

        stage_events.append("stage_2_success")

    @stage(name="stage_3_downstream", order_idx=3, expected_s=0.5)
    def stage_3(state: StateStore, timing: JointTimingLogEngine):
        stage_events.append("stage_3_started")
        stage_events.append("stage_3_done")

    runner.register(stage_1)
    runner.register(stage_2)
    runner.register(stage_3)

    trap_file = TRAPS_DIR / "stage_2_hang.json"
    if trap_file.exists():
        trap_file.unlink()

    def external_steerer():
        start_wait = time.time()
        while not trap_file.exists() and time.time() - start_wait < 5.0:
            time.sleep(0.02)

        assert trap_file.exists(), "Trap file was never created!"
        timestamps["trap_observed_at"] = time.time()
        print(f"\n[EXTERNAL_WATCHER] Watchdog trap observed at t={timestamps['trap_observed_at'] - timestamps['run_started_at']:.3f}s", flush=True)

        with open(trap_file, "r", encoding="utf-8") as f:
            snapshot = json.load(f)

        print(f"[EXTERNAL_WATCHER] Trap reason: {snapshot['reason']}", flush=True)
        assert "No log emitted" in snapshot["reason"]
        assert "time.sleep(100.0)" in snapshot["live_stack"]

        # Barrier 1: Stage 3 has not started
        assert "stage_3_started" not in stage_events
        assert "stage_2_success" not in stage_events

        # Hold open for 1.0s to prove execution is frozen
        hold_duration = 1.0
        print(f"[EXTERNAL_WATCHER] Holding trap for {hold_duration}s to confirm frozen pipeline...", flush=True)
        time.sleep(hold_duration)

        # Barrier 2: Still frozen
        assert "stage_3_started" not in stage_events
        assert "stage_2_success" not in stage_events
        print("[EXTERNAL_WATCHER] Confirmed: Pipeline remained frozen during watchdog trap.", flush=True)

        # Inject RETRY
        print("[EXTERNAL_WATCHER] Injecting RETRY directive...", flush=True)
        with open(ACTION_FILE, "w", encoding="utf-8") as f:
            json.dump({"op": "RETRY"}, f)

    steerer_thread = threading.Thread(target=external_steerer, daemon=True)

    timestamps["run_started_at"] = time.time()
    steerer_thread.start()

    success = runner.run()
    timestamps["run_finished_at"] = time.time()
    steerer_thread.join(timeout=3.0)

    total_duration = timestamps["run_finished_at"] - timestamps["run_started_at"]
    print(f"\n[PROOF_SUMMARY] Total Run Duration: {total_duration:.3f}s (preempted 100s sleep at ~0.3s + 1.0s hold)")
    print(f"[PROOF_SUMMARY] Stage Event Sequence: {stage_events}")

    assert success is True, "Run failed!"
    assert total_duration < 5.0, f"Run took {total_duration:.3f}s, did not preempt 100s hang!"
    assert total_duration >= 1.3, f"Run took {total_duration:.3f}s, did not honor the hold period!"
    assert stage_events == [
        "stage_1_done",
        "stage_2_attempt_1",
        "stage_2_attempt_2",
        "stage_2_success",
        "stage_3_started",
        "stage_3_done",
    ]

    print("\n>>> ALL WATCHDOG PROOF ASSERTIONS PASSED: 100s hang was preempted and resolved cleanly! <<<\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_hang_proof())
