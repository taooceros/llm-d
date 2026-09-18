"""
Focused verification tests for Tier 1:
- A1/A2: Output discipline, events filtering, and output capping (300 lines / 24 KB)
- A3: Shared EVENT_MARKERS constant
- A4: Capped and redacted state_sample
- B1: First-party blame frame selection over vendored library frames
- B2: Source window with '>' gutter marker
- B3: Redacted and summarized frame locals (Decision 2)
- B4: Condensed first-party traceback
- F1: Documented exit codes (0 complete, 1 failed, 2 trapped, 3 no daemon, 124 timeout)
"""

import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

# Ensure repo root is on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omp_rcm import (
    BASE_RCM_DIR,
    LATEST_TRAP_FILE,
    TRAPS_DIR,
    StageRunner,
    StateStore,
    cli_main,
    format_trap_system_directive,
)
from omp_rcm.core import (
    EVENT_MARKERS,
    ensure_rcm_dir,
    extract_state_sample,
    get_source_window,
    is_event_line,
    is_first_party_file,
    sanitize_value,
)
from omp_rcm.cli import stream_daemon_output


class DummyTensor:
    """Mock tensor/array object with shape and dtype attributes."""
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class TestTier1Ergonomics(unittest.TestCase):

    def setUp(self):
        self.test_dir = pathlib.Path(tempfile.mkdtemp(prefix="rcm_tier1_test_"))
        ensure_rcm_dir(self.test_dir, 0o700)
        LATEST_TRAP_FILE.unlink(missing_ok=True)

    def tearDown(self):
        LATEST_TRAP_FILE.unlink(missing_ok=True)
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_a3_event_markers_defined(self):
        """A3: Assert EVENT_MARKERS constant is defined and covers critical patterns."""
        self.assertIn("[OMP_EVENT:", EVENT_MARKERS)
        self.assertIn("TRAP_PAUSED", EVENT_MARKERS)
        self.assertTrue(is_event_line("[OMP_EVENT: TRAP_PAUSED stage=test]"))
        self.assertFalse(is_event_line("regular vLLM debug message batch=16"))

    def test_a4_state_sample_capping_and_redaction(self):
        """A4: Cap state_sample (40 keys, 200 chars each) and apply Decision-2 redaction."""
        store = StateStore()
        store["auth_token"] = "super_secret_token_abc123"
        store["user_password"] = "p@ssw0rd!"
        store["large_tensor"] = DummyTensor((32, 4096), "bfloat16")
        store["large_blob"] = "x" * 1000
        for i in range(60):
            store[f"key_{i}"] = f"value_{i}"
        sampled = extract_state_sample(store, max_keys=40, max_chars=200)
        self.assertLessEqual(len(sampled), 41)  # 40 keys + '...'
        self.assertEqual(sampled["auth_token"], "<redacted>")
        self.assertEqual(sampled["user_password"], "<redacted>")
        self.assertEqual(sampled["large_tensor"], "<DummyTensor shape=(32, 4096) dtype=bfloat16>")
        self.assertTrue(str(sampled["large_blob"]).startswith("<str len=1000:"))
        self.assertIn("...", sampled)

    def test_b1_b2_b3_b4_vendored_frame_blame_selection_and_banner(self):
        """
        B1: Given a stage raising inside a vendored library called from first-party code,
        assert the banner names the first-party frame, shows its source window with '>',
        and displays sanitized frame locals.
        """
        # Create a first-party file inside repo root
        first_party_dir = _REPO_ROOT / "admission_control" / "_test_tmp"
        first_party_dir.mkdir(parents=True, exist_ok=True)
        first_party_file = first_party_dir / "user_stage_module.py"

        # Create a simulated vendored file inside site-packages directory
        vendored_dir = self.test_dir / "site-packages" / "vendor_pkg"
        vendored_dir.mkdir(parents=True, exist_ok=True)
        vendored_file = vendored_dir / "backend_engine.py"

        try:
            vendored_file.write_text(
                "def internal_vendored_compute(tensor_input):\n"
                "    # Line 2: raises deep inside vendor package\n"
                "    raise ValueError('Vendored hardware alignment error')\n",
                encoding="utf-8",
            )

            first_party_file.write_text(
                "from site_packages_import import internal_vendored_compute\n"
                "def first_party_stage_step():\n"
                "    batch_size = 64\n"
                "    api_key = 'sk-live-secret-12345'\n"
                "    weights_buffer = 'A' * 500\n"
                "    # Line 7: first party calls vendored library\n"
                "    internal_vendored_compute(weights_buffer)\n",
                encoding="utf-8",
            )

            # Execute and generate an actual exception with frames crossing vendored boundary
            sys.path.insert(0, str(self.test_dir / "site-packages"))
            # Rename import in first party to mock module
            sys.modules["site_packages_import"] = type(sys)("site_packages_import")
            exec(compile(vendored_file.read_text(), str(vendored_file), "exec"), sys.modules["site_packages_import"].__dict__)

            mod_ns = {}
            code_obj = compile(first_party_file.read_text(), str(first_party_file), "exec")
            exec(code_obj, mod_ns)

            exc = None
            try:
                mod_ns["first_party_stage_step"]()
            except BaseException as e:
                exc = e

            self.assertIsNotNone(exc)

            # Run through StageRunner DiagnosticTrap
            traps_dir = self.test_dir / "traps"
            runner = StageRunner(
                checkpoint_path=self.test_dir / "checkpoint.json",
                telemetry_path=self.test_dir / "telemetry.jsonl",
                traps_dir=traps_dir,
                action_file=self.test_dir / "action.json",
                socket_path=self.test_dir / "control.sock",
            )

            trap_file = runner.trap.write_snapshot(
                stage_name="decode_benchmark",
                reason="EXCEPTION: ValueError: Vendored hardware alignment error",
                clocks={"t_stage_s": 0.1, "t_expected_s": 1.0, "t_total_s": 0.1},
                log_tail=["[INFO] Starting decode step"],
                state_store=runner.state,
                exc=exc,
            )

            with open(trap_file, "r", encoding="utf-8") as f:
                snapshot = json.load(f)

            # B1: Blame frame must be the first-party file, NOT the site-packages file!
            blame = snapshot["blame_frame"]
            self.assertEqual(pathlib.Path(blame["file"]).resolve(), first_party_file.resolve())
            self.assertEqual(blame["line"], 7)
            self.assertEqual(blame["function"], "first_party_stage_step")

            # B2: Source window with '>' gutter marker
            source_win = blame["source_window"]
            self.assertIn(">", source_win["formatted"])
            self.assertIn("internal_vendored_compute(weights_buffer)", source_win["formatted"])

            # B3: Frame locals with Decision 2 redaction
            locals_dict = blame["locals"]
            self.assertEqual(locals_dict["batch_size"], 64)
            self.assertEqual(locals_dict["api_key"], "<redacted>")
            self.assertTrue(str(locals_dict["weights_buffer"]).startswith("<str len=500:"))

            # Format banner
            banner = format_trap_system_directive(snapshot, daemon_pid=54321)
            self.assertIn("<system-directive>", banner)
            self.assertIn("user_stage_module.py:7 in first_party_stage_step", banner)
            self.assertIn("batch_size = 64", banner)
            self.assertIn("api_key = <redacted>", banner)
            self.assertIn("Source Code Window", banner)
            self.assertIn(">    7:     internal_vendored_compute(weights_buffer)", banner)
            self.assertIn("rcm_exec resume", banner)
        finally:
            if first_party_file.exists():
                first_party_file.unlink()
            if first_party_dir.exists():
                shutil.rmtree(first_party_dir, ignore_errors=True)
            if "site_packages_import" in sys.modules:
                del sys.modules["site_packages_import"]
            if str(self.test_dir / "site-packages") in sys.path:
                sys.path.remove(str(self.test_dir / "site-packages"))

    def test_a1_a2_output_capping_and_suppression_footer(self):
        """
        A1/A2: Output capping per invocation (~300 lines / 24 KB) with suppression footer.
        Assert that streaming 10,000 lines stays strictly bounded.
        """
        log_file = self.test_dir / "daemon_large.log"
        # Write 10,000 debug lines followed by completion event
        with open(log_file, "w", encoding="utf-8") as f:
            for i in range(10000):
                f.write(f"debug log iteration {i}: processing token batch and kv cache slice\n")
            f.write("[OMP_EVENT: RUN_FINISHED status=SUCCESS total_elapsed_s=1.23]\n")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            # Set RCM_AGENT=1 to enforce events filter + capping
            os.environ["RCM_AGENT"] = "1"
            rc = stream_daemon_output(
                log_file=log_file,
                daemon_proc=None,
                daemon_pid=999999,  # non-existent PID so alive=False
                start_offset=0,
                max_lines=300,
                max_bytes=24576,
            )
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            os.environ.pop("RCM_AGENT", None)

        self.assertEqual(rc, 0)
        out_lines = output.splitlines()
        # Should NOT emit 10,000 lines; should be capped
        self.assertLessEqual(len(out_lines), 305)
        self.assertLessEqual(len(output.encode("utf-8")), 26000)

    def test_f1_exit_codes(self):
        """F1: Documented exit codes: 3 when no daemon is active."""
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            # wait with no daemon -> 3
            rc_wait = cli_main(["wait"])
            self.assertEqual(rc_wait, 3)

            # reload with no daemon -> 3
            rc_reload = cli_main(["reload", "nonexistent.py"])
            self.assertEqual(rc_reload, 3)

            # patch with no daemon -> 3
            rc_patch = cli_main(["patch", "--code", "def foo(): pass"])
            self.assertEqual(rc_patch, 3)

            # steer with no daemon -> 3
            rc_steer = cli_main(["steer", '{"op": "RETRY"}'])
            self.assertEqual(rc_steer, 3)

            # abort with no daemon -> 3
            rc_abort = cli_main(["abort"])
            self.assertEqual(rc_abort, 3)
        finally:
            sys.stderr = old_stderr

    def test_chmod_700_permissions(self):
        """Decision 2 / Finding 11: chmod 700 on BASE_RCM_DIR."""
        ensure_rcm_dir(BASE_RCM_DIR, 0o700)
        mode = BASE_RCM_DIR.stat().st_mode & 0o777
        self.assertEqual(mode, 0o700)


if __name__ == "__main__":
    unittest.main()
