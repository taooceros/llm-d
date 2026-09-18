"""
Focused verification tests for Tier 3:
- D1: ctx.once(key, fn, deps=()) memoization in StateStore
- D2: Trap banner lists preserved sub-steps; --fresh invalidates memoization
- E3: Heartbeat in live_status.json, status distinguishes running vs defunct/zombie
- F3: Documented OMP_EVENT grammar and parse_omp_event helper
"""

import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

# Ensure repo root is on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omp_rcm import (
    BASE_RCM_DIR,
    CONTROL_ACTION_FILE,
    DAEMON_LOG_FILE,
    DAEMON_PID_FILE,
    LATEST_TRAP_FILE,
    LIVE_STATUS_FILE,
    TRAPS_DIR,
    StageMetadata,
    StageRunner,
    StateStore,
    cli_main,
    format_trap_system_directive,
    is_pid_zombie,
    parse_omp_event,
)
from omp_rcm.core import ensure_rcm_dir


class TestTier3Ergonomics(unittest.TestCase):

    def setUp(self):
        self.test_dir = pathlib.Path(tempfile.mkdtemp(prefix="rcm_tier3_test_"))
        ensure_rcm_dir(self.test_dir, 0o700)
        ensure_rcm_dir(TRAPS_DIR, 0o700)
        LATEST_TRAP_FILE.unlink(missing_ok=True)
        CONTROL_ACTION_FILE.unlink(missing_ok=True)

    def tearDown(self):
        LATEST_TRAP_FILE.unlink(missing_ok=True)
        CONTROL_ACTION_FILE.unlink(missing_ok=True)
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_d1_ctx_once_memoization(self):
        """D1: ctx.once memoizes expensive sub-steps across retries unless deps change."""
        state = StateStore()
        exec_counts = {"calc": 0}

        def expensive_calc():
            exec_counts["calc"] += 1
            return 999

        # First call: executes fn
        v1 = state.once("my_calc", expensive_calc, deps=("v1",))
        self.assertEqual(v1, 999)
        self.assertEqual(exec_counts["calc"], 1)
        self.assertEqual(state.get("my_calc"), 999)
        self.assertEqual(state.get_once_keys(), ["my_calc"])

        # Second call with same deps: returns cached, does NOT re-execute
        v2 = state.once("my_calc", expensive_calc, deps=("v1",))
        self.assertEqual(v2, 999)
        self.assertEqual(exec_counts["calc"], 1)

        # Call with different deps: re-executes
        v3 = state.once("my_calc", expensive_calc, deps=("v2",))
        self.assertEqual(v3, 999)
        self.assertEqual(exec_counts["calc"], 2)

        # Invalidate single key: re-executes
        state.invalidate_once("my_calc")
        self.assertEqual(state.get_once_keys(), [])
        v4 = state.once("my_calc", expensive_calc, deps=("v2",))
        self.assertEqual(v4, 999)
        self.assertEqual(exec_counts["calc"], 3)

    def test_d2_banner_shows_preserved_substeps(self):
        """D2: Banner lists satisfied once keys as will NOT re-run on resume."""
        trap_data = {
            "stage": "heavy_compute",
            "order_idx": 2,
            "reason": "EXCEPTION: RuntimeError: Test error",
            "once_keys": ["weights_sharded_ref", "kv_cache_allocated"],
            "state_keys": ["weights_sharded_ref", "kv_cache_allocated", "step"],
        }
        banner = format_trap_system_directive(trap_data, daemon_pid=1234)
        self.assertIn("Preserved Sub-steps (will NOT re-run on resume):", banner)
        self.assertIn("`weights_sharded_ref`", banner)
        self.assertIn("`kv_cache_allocated`", banner)

    def test_d2_fresh_invalidates_memoized_cache(self):
        """D2: Action with fresh=True clears memoized once cache in StageRunner."""
        runner = StageRunner(
            checkpoint_path=self.test_dir / "checkpoint.json",
            telemetry_path=self.test_dir / "telemetry.jsonl",
            traps_dir=self.test_dir / "traps",
            action_file=self.test_dir / "action.json",
            socket_path=self.test_dir / "control.sock",
            trap_timeout_s=2.0,
        )

        runner.state.once("cached_step", lambda: "val1", deps=())
        self.assertIn("cached_step", runner.state.get_once_keys())

        # Write fresh action to action file
        (self.test_dir / "action.json").write_text(
            json.dumps({"op": "RETRY", "fresh": True}) + "\n",
            encoding="utf-8",
        )

        # Enter trap which polls action and clears once cache
        action = runner.enter_diagnostic_trap(
            stage_name="test_stage",
            reason="simulated stall",
        )
        self.assertTrue(action.get("fresh"))
        self.assertEqual(runner.state.get_once_keys(), [])

    def test_e3_heartbeat_and_zombie_status(self):
        """E3: live_status heartbeat and zombie/stale distinction in cmd_status."""
        # Check active non-zombie PID
        self.assertFalse(is_pid_zombie(os.getpid()))

        # Mock running daemon with active process
        sleep_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
        DAEMON_PID_FILE.write_text(str(sleep_proc.pid), encoding="utf-8")
        try:
            # 1. Live status with fresh heartbeat
            now = time.time()
            st_fresh = {
                "timestamp": now,
                "heartbeat_s": now,
                "current_stage": "stage_alpha",
                "clocks": {"t_stage_s": 1.0, "t_total_s": 1.0},
            }
            LIVE_STATUS_FILE.write_text(json.dumps(st_fresh), encoding="utf-8")

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                rc = cli_main(["status", "--json"])
                out = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            self.assertEqual(rc, 0)
            parsed = json.loads(out)
            self.assertEqual(parsed["daemon_pid"], sleep_proc.pid)
            self.assertTrue(parsed["alive"])
            self.assertFalse(parsed["zombie"])
            self.assertFalse(parsed["stale_heartbeat"])

            # 2. Stale heartbeat (>60s old)
            st_stale = dict(st_fresh)
            st_stale["heartbeat_s"] = now - 120.0
            LIVE_STATUS_FILE.write_text(json.dumps(st_stale), encoding="utf-8")

            sys.stdout = io.StringIO()
            try:
                rc = cli_main(["status", "--json"])
                out_stale = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            parsed_stale = json.loads(out_stale)
            self.assertTrue(parsed_stale["stale_heartbeat"])
        finally:
            sleep_proc.terminate()
            sleep_proc.wait()
            DAEMON_PID_FILE.unlink(missing_ok=True)
            LIVE_STATUS_FILE.unlink(missing_ok=True)

    def test_f3_parse_omp_event(self):
        """F3: parse_omp_event accurately parses standardized telemetry lines."""
        # 1. Standard STAGE_ENTERED
        line1 = "[OMP_EVENT: STAGE_ENTERED stage=load_weights stage_idx=2 expected_s=3.5]"
        ev1 = parse_omp_event(line1)
        self.assertIsNotNone(ev1)
        self.assertEqual(ev1["event"], "STAGE_ENTERED")
        self.assertEqual(ev1["stage"], "load_weights")
        self.assertEqual(ev1["stage_idx"], 2)
        self.assertEqual(ev1["expected_s"], 3.5)

        # 2. TRAP_PAUSED with quoted reason
        line2 = '[OMP_EVENT: TRAP_PAUSED stage=eval elapsed_s=1.2 expected_s=2.0 reason="EXCEPTION: ValueError: Bad shape" trap_file="/tmp/t.json"]'
        ev2 = parse_omp_event(line2)
        self.assertIsNotNone(ev2)
        self.assertEqual(ev2["event"], "TRAP_PAUSED")
        self.assertEqual(ev2["stage"], "eval")
        self.assertEqual(ev2["reason"], "EXCEPTION: ValueError: Bad shape")
        self.assertEqual(ev2["trap_file"], "/tmp/t.json")

        # 3. RUN_FINISHED
        line3 = "[OMP_EVENT: RUN_FINISHED status=SUCCESS total_elapsed_s=12.4]"
        ev3 = parse_omp_event(line3)
        self.assertIsNotNone(ev3)
        self.assertEqual(ev3["event"], "RUN_FINISHED")
        self.assertEqual(ev3["status"], "SUCCESS")
        self.assertEqual(ev3["total_elapsed_s"], 12.4)

        # 4. Non-event line returns None
        self.assertIsNone(parse_omp_event("Regular application stdout log line"))
        self.assertIsNone(parse_omp_event("[INFO] Standard log message"))


if __name__ == "__main__":
    unittest.main()
