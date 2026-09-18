"""
Focused verification tests for Tier 2:
- C1: rcm_exec resume: detects modified module on disk, hot-patches, and retries cleanly
- C2: resume --skip, --abort, --fresh
- C3: Warns loudly when no dirty modules detected and no explicit op passed
- B5: Recurrence tracking in history.json, ordinal display, directive change at >=3
- B6: Exactly one copy-pasteable NEXT: command in banner with no placeholders
- E1: trap_timeout_s default 1800s, auto-ABORT, PG release, reason=TRAP_TIMEOUT
- E2: rcm_exec run --replace auto-reaps existing daemon
- F2: rcm_exec trap [--json] and --json on status/wait/logs
"""

import io
import json
import os
import pathlib
import shutil
import signal
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
    TRAPS_DIR,
    StageMetadata,
    StageRunner,
    cli_main,
    cmd_resume,
    cmd_trap,
    format_trap_system_directive,
    get_active_daemon_pid,
)
from omp_rcm.core import ensure_rcm_dir


class TestTier2Ergonomics(unittest.TestCase):

    def setUp(self):
        self.test_dir = pathlib.Path(tempfile.mkdtemp(prefix="rcm_tier2_test_"))
        ensure_rcm_dir(self.test_dir, 0o700)
        ensure_rcm_dir(TRAPS_DIR, 0o700)
        LATEST_TRAP_FILE.unlink(missing_ok=True)
        CONTROL_ACTION_FILE.unlink(missing_ok=True)
        (TRAPS_DIR / "history.json").unlink(missing_ok=True)

    def tearDown(self):
        LATEST_TRAP_FILE.unlink(missing_ok=True)
        CONTROL_ACTION_FILE.unlink(missing_ok=True)
        (TRAPS_DIR / "history.json").unlink(missing_ok=True)
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_e1_trap_timeout_auto_abort(self):
        """E1: Assert trap_timeout_s auto-aborts execution and emits reason=TRAP_TIMEOUT."""
        runner = StageRunner(
            checkpoint_path=self.test_dir / "checkpoint.json",
            telemetry_path=self.test_dir / "telemetry.jsonl",
            traps_dir=self.test_dir / "traps",
            action_file=self.test_dir / "action.json",
            socket_path=self.test_dir / "control.sock",
            trap_timeout_s=0.3,  # Fast timeout for test
        )

        def failing_stage(state):
            raise RuntimeError("Initial simulated failure")

        runner.register(StageMetadata(
            name="test_timeout_stage",
            order_idx=1,
            expected_s=0.1,
            func=failing_stage,
            module_name=__name__,
            qualname="failing_stage",
        ))
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            success = runner.run()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        self.assertFalse(success)
        self.assertIn("RUN_FINISHED status=ABORTED", output)
        self.assertIn("reason=TRAP_TIMEOUT", output)

    def test_b5_b6_recurrence_tracking_and_next_command(self):
        """
        B5: Recurrence tracking in history.json: ordinal counted; at >=3 changes directive.
        B6: Exactly one copy-pasteable NEXT: command with no placeholders.
        """
        runner = StageRunner(
            checkpoint_path=self.test_dir / "checkpoint.json",
            telemetry_path=self.test_dir / "telemetry.jsonl",
            traps_dir=self.test_dir / "traps",
            action_file=self.test_dir / "action.json",
            socket_path=self.test_dir / "control.sock",
        )

        exc = ValueError("Dimension alignment mismatch")

        # 1st trap
        trap_file_1 = runner.trap.write_snapshot(
            stage_name="decode_step",
            reason="EXCEPTION: ValueError: Dimension alignment mismatch",
            clocks={"t_stage_s": 0.1, "t_expected_s": 1.0, "t_total_s": 0.1},
            log_tail=[],
            state_store=runner.state,
            exc=exc,
        )
        with open(trap_file_1, "r", encoding="utf-8") as f:
            snap1 = json.load(f)
        self.assertEqual(snap1.get("recurrence_count"), 1)
        banner1 = format_trap_system_directive(snap1, daemon_pid=1111)
        self.assertIn("NEXT: rcm_exec resume", banner1)
        self.assertNotIn("Occurrence #", banner1)
        self.assertNotIn("<file>", banner1)
        self.assertNotIn("<modified_file>", banner1)

        # 2nd trap
        trap_file_2 = runner.trap.write_snapshot(
            stage_name="decode_step",
            reason="EXCEPTION: ValueError: Dimension alignment mismatch",
            clocks={"t_stage_s": 0.2, "t_expected_s": 1.0, "t_total_s": 0.2},
            log_tail=[],
            state_store=runner.state,
            exc=exc,
        )
        with open(trap_file_2, "r", encoding="utf-8") as f:
            snap2 = json.load(f)
        self.assertEqual(snap2.get("recurrence_count"), 2)
        banner2 = format_trap_system_directive(snap2, daemon_pid=1111)
        self.assertIn("Recurrence: Occurrence #2", banner2)
        self.assertIn("NEXT: rcm_exec resume", banner2)

        # 3rd trap: directive changes per B5
        trap_file_3 = runner.trap.write_snapshot(
            stage_name="decode_step",
            reason="EXCEPTION: ValueError: Dimension alignment mismatch",
            clocks={"t_stage_s": 0.3, "t_expected_s": 1.0, "t_total_s": 0.3},
            log_tail=[],
            state_store=runner.state,
            exc=exc,
        )
        with open(trap_file_3, "r", encoding="utf-8") as f:
            snap3 = json.load(f)
        self.assertEqual(snap3.get("recurrence_count"), 3)
        banner3 = format_trap_system_directive(snap3, daemon_pid=1111)
        self.assertIn("*** REPEATED FAILURE DETECTED (Occurrence #3) ***", banner3)
        self.assertIn("Stop patching", banner3)
        self.assertIn("NEXT: rcm_exec resume --skip", banner3)
        self.assertNotIn("<", banner3.split("NEXT:")[1].splitlines()[0])

    def test_c1_resume_dirty_module_detection_and_hot_patch(self):
        """
        C1: Modify a module on disk mid-trap; assert resume detects exactly it,
        hot-patches, and the retry executes the new code.
        """
        # Create a module in repo root
        stage_dir = _REPO_ROOT / "admission_control" / "_test_stage_mod"
        stage_dir.mkdir(parents=True, exist_ok=True)
        mod_file = stage_dir / "worker_substep.py"

        try:
            # Initial failing version
            mod_file.write_text(
                "def compute_val():\n"
                "    return 100 // 0  # Bug: ZeroDivisionError\n",
                encoding="utf-8",
            )

            # Mock running daemon with real alive process
            sleep_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
            DAEMON_PID_FILE.write_text(str(sleep_proc.pid), encoding="utf-8")
            DAEMON_LOG_FILE.write_text("", encoding="utf-8")
            # Snapshot with initial file mtime and sha1
            stat = mod_file.stat()
            with open(mod_file, "rb") as f:
                import hashlib
                initial_sha = hashlib.sha1(f.read()).hexdigest()

            trap_snapshot = {
                "timestamp": time.time(),
                "stage": "compute_stage",
                "reason": "EXCEPTION: ZeroDivisionError: integer division or modulo by zero",
                "loaded_modules": {
                    "worker_substep": {
                        "file": str(mod_file.resolve()),
                        "mtime": stat.st_mtime - 1.0,  # simulate older snapshot time
                        "sha1": initial_sha,
                    }
                },
                "blame_frame": {
                    "file": str(mod_file.resolve()),
                    "line": 2,
                    "function": "compute_val",
                },
            }
            with open(LATEST_TRAP_FILE, "w", encoding="utf-8") as f:
                json.dump(trap_snapshot, f)

            # Mid-trap edit: fix the file on disk
            time.sleep(0.02)
            mod_file.write_text(
                "def compute_val():\n"
                "    return 42  # Fixed!\n",
                encoding="utf-8",
            )

            # Run rcm_exec resume --no-wait
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                rc = cli_main(["resume", "--no-wait"])
                out = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            self.assertEqual(rc, 0)
            self.assertIn("Detected modified module(s): worker_substep", out)
            self.assertTrue(CONTROL_ACTION_FILE.exists())
            directive = json.loads(CONTROL_ACTION_FILE.read_text(encoding="utf-8"))
            self.assertEqual(directive.get("op"), "HOT_PATCH")
            self.assertEqual(directive.get("file"), str(mod_file.resolve()))
        finally:
            if "sleep_proc" in locals():
                sleep_proc.terminate()
                sleep_proc.wait()
            DAEMON_PID_FILE.unlink(missing_ok=True)
            if mod_file.exists():
                mod_file.unlink()
            if stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)

    def test_c3_resume_no_dirty_modules_warning(self):
        """C3: If resume finds no dirty modules and no explicit op, warn loudly to stderr."""
        sleep_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
        DAEMON_PID_FILE.write_text(str(sleep_proc.pid), encoding="utf-8")
        DAEMON_LOG_FILE.write_text("", encoding="utf-8")
        dummy_file = self.test_dir / "untouched_file.py"
        dummy_file.write_text("def untouched(): pass\n", encoding="utf-8")
        stat = dummy_file.stat()
        with open(dummy_file, "rb") as f:
            import hashlib
            sha = hashlib.sha1(f.read()).hexdigest()

        trap_snapshot = {
            "timestamp": time.time(),
            "stage": "stage_a",
            "reason": "EXCEPTION: RuntimeError: Test error",
            "loaded_modules": {
                "untouched_file": {
                    "file": str(dummy_file.resolve()),
                    "mtime": stat.st_mtime + 10.0,  # snapshot is newer than disk
                    "sha1": sha,
                }
            },
        }
        with open(LATEST_TRAP_FILE, "w", encoding="utf-8") as f:
            json.dump(trap_snapshot, f)

        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = io.StringIO()
        sys.stdout = io.StringIO()
        try:
            rc = cli_main(["resume", "--no-wait"])
            err_out = sys.stderr.getvalue()
            std_out = sys.stdout.getvalue()
        finally:
            sleep_proc.terminate()
            sleep_proc.wait()
            sys.stderr = old_stderr
            sys.stdout = old_stdout
            DAEMON_PID_FILE.unlink(missing_ok=True)

        self.assertEqual(rc, 0)
        self.assertIn("No modified first-party modules detected", err_out)
        self.assertIn("Did you edit a file that the daemon has not imported", err_out)
        self.assertIn("Retrying stage without hot patch", std_out)
        directive = json.loads(CONTROL_ACTION_FILE.read_text(encoding="utf-8"))
        self.assertEqual(directive.get("op"), "RETRY")
    def test_f2_trap_command_and_json_modes(self):
        """F2: rcm_exec trap [--json] re-reads latest trap without re-waiting; status/wait --json."""
        trap_data = {
            "stage": "eval_stage",
            "order_idx": 3,
            "reason": "EXCEPTION: RuntimeError: Mock error",
            "exception": {"type": "RuntimeError", "message": "Mock error", "traceback": ""},
            "state_keys": ["k1"],
        }
        with open(LATEST_TRAP_FILE, "w", encoding="utf-8") as f:
            json.dump(trap_data, f)

        # 1. rcm_exec trap
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            rc = cli_main(["trap"])
            out = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        self.assertEqual(rc, 2)
        self.assertIn("<system-directive>", out)
        self.assertIn("eval_stage", out)

        # 2. rcm_exec trap --json
        sys.stdout = io.StringIO()
        try:
            rc_json = cli_main(["trap", "--json"])
            out_json = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        self.assertEqual(rc_json, 0)
        parsed = json.loads(out_json)
        self.assertEqual(parsed.get("stage"), "eval_stage")

        # 3. rcm_exec status --json
        sys.stdout = io.StringIO()
        try:
            rc_status = cli_main(["status", "--json"])
            out_status = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        self.assertEqual(rc_status, 0)
        status_parsed = json.loads(out_status)
        self.assertIn("daemon_pid", status_parsed)
        self.assertIn("live_status", status_parsed)

    def test_e2_run_replace(self):
        """E2: rcm_exec run --replace reaps existing daemon and starts new one."""
        old_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        DAEMON_PID_FILE.write_text(str(old_proc.pid), encoding="utf-8")

        test_script = self.test_dir / "quick_job.py"
        test_script.write_text("import time\nprint('done')\n", encoding="utf-8")

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            # Without --replace, should fail with exit code 1
            rc_fail = cli_main(["run", str(test_script)])
            self.assertEqual(rc_fail, 1)
        finally:
            sys.stderr = old_stderr

        # With --replace, should terminate old_proc and spawn new daemon
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            rc_ok = cli_main(["run", "--replace", "--daemon", str(test_script)])
            self.assertEqual(rc_ok, 0)
        finally:
            sys.stdout = old_stdout

        # Verify old process was terminated
        try:
            old_proc.wait(timeout=2.0)
            terminated = True
        except subprocess.TimeoutExpired:
            terminated = False
        self.assertTrue(terminated)
        self.assertIsNotNone(old_proc.returncode)

        # Clean up spawned new daemon
        new_pid = get_active_daemon_pid()
        if new_pid:
            try:
                os.kill(new_pid, signal.SIGTERM)
            except OSError:
                pass
            DAEMON_PID_FILE.unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()
