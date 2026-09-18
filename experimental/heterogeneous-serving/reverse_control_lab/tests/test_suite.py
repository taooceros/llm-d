"""
Automated Verification Suite for Reverse Control Mechanism (RCM) SDK.
Validates all 5 architectural slices using the `omp_rcm` SDK package:
1. Minimal Standalone Stage Machine & Checkpointing
2. Joint Time-Log Engine & Normality Evaluation
3. State Preservation & In-Flight Parameter Mutation Steering
4. Hot Module Patching & Code Reloading
5. Asymmetric Watchdog & SIGUSR1 Interruption
"""

import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import time
import unittest
# Ensure repo root is on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omp_rcm import (
    DiagnosticTrap,
    JointTimingLogEngine,
    StageMetadata,
    StageRunner,
    StateStore,
    SteeringReceiver,
    WatchdogTimer,
    stage,
    apply_unified_diff,
    RayJobSupervisor,
    read_active_ray_job,
    read_all_ray_jobs,
    format_ray_job_summary,
    format_ray_jobs_table,
    RAY_JOB_FILE,
    RAY_JOBS_FILE,
    DAEMON_PID_FILE,
    DAEMON_LOG_FILE,
    LATEST_TRAP_FILE,
    TRAPS_DIR,
    cli_main,
    format_trap_system_directive,
)

class TestReverseControlSDK(unittest.TestCase):

    def setUp(self):
        self.test_dir = pathlib.Path(tempfile.mkdtemp(prefix="rcm_test_"))
        self.checkpoint_path = self.test_dir / "checkpoint.json"
        self.telemetry_path = self.test_dir / "telemetry.jsonl"
        self.traps_dir = self.test_dir / "traps"
        self.socket_path = self.test_dir / "control.sock"
        self.action_file = self.test_dir / "control_action.json"

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_minimal_stages(self):
        """Slice 1: Test clean sequential stage progression and checkpoint JSON generation."""
        runner = StageRunner(
            checkpoint_path=self.checkpoint_path,
            telemetry_path=self.telemetry_path,
            traps_dir=self.traps_dir,
            action_file=self.action_file,
            socket_path=self.socket_path,
            silence_timeout_s=10.0,
            enable_steering_socket=False,
        )

        events_observed = []

        @stage(name="topology_discovery", order_idx=1, expected_s=0.2)
        def stage_1(state: StateStore):
            events_observed.append("stage_1_executed")
            state["topology"] = "2x4_tpu_v5e"
            time.sleep(0.05)

        @stage(name="weight_loading", order_idx=2, expected_s=0.2)
        def stage_2(state: StateStore):
            events_observed.append("stage_2_executed")
            self.assertEqual(state["topology"], "2x4_tpu_v5e")
            state["weights"] = "gemma-4-31b-sharded"
            time.sleep(0.05)

        @stage(name="prefill_benchmark", order_idx=3, expected_s=0.2)
        def stage_3(state: StateStore):
            events_observed.append("stage_3_executed")
            self.assertEqual(state["weights"], "gemma-4-31b-sharded")
            state["tpot_ms"] = 4.2
            time.sleep(0.05)

        runner.register(stage_1)
        runner.register(stage_2)
        runner.register(stage_3)

        success = runner.run()
        self.assertTrue(success)
        self.assertEqual(events_observed, ["stage_1_executed", "stage_2_executed", "stage_3_executed"])

        # Verify Checkpoint JSON
        self.assertTrue(self.checkpoint_path.exists())
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        self.assertEqual(checkpoint["status"], "SUCCESS")
        self.assertEqual(checkpoint["completed_stages"], ["topology_discovery", "weight_loading", "prefill_benchmark"])
        self.assertEqual(checkpoint["total_stages"], 3)
        self.assertIn("topology", checkpoint["state_keys"])
        self.assertIn("weights", checkpoint["state_keys"])
        self.assertIn("tpot_ms", checkpoint["state_keys"])

    def test_joint_time_log_normality(self):
        """Slice 2: Verify silent normal operation, anomaly warnings, and stall timeouts."""
        engine = JointTimingLogEngine(
            buffer_size=10,
            telemetry_path=self.telemetry_path,
            silence_timeout_s=0.3,
        )

        engine.start_run()
        engine.start_stage("matrix_multiplication", expected_s=0.2)

        # 1. Immediate check: HEALTHY
        status, reason = engine.check_normality()
        self.assertEqual(status, "HEALTHY")
        self.assertIsNone(reason)

        # Log entries
        engine.log("Executing GEMM iteration 1", level="INFO")
        engine.log("Executing GEMM iteration 2", level="INFO")
        tail = engine.get_tail_logs()
        self.assertEqual(len(tail), 2)
        self.assertIn("GEMM iteration 1", tail[0])

        # 2. Simulate 1.3x expected elapsed time (0.26s > 0.25s)
        engine.t_stage_start = time.time() - 0.27
        status, reason = engine.check_normality()
        self.assertEqual(status, "ANOMALY_WARNING")
        self.assertIn("exceeding 1.25x expected", reason)

        # 3. Simulate 2.1x expected elapsed time (0.45s > 0.40s)
        engine.t_stage_start = time.time() - 0.45
        status, reason = engine.check_normality()
        self.assertEqual(status, "SLA_BREACH")
        self.assertIn("exceeding 2.0x SLA", reason)

        # 4. Simulate silence timeout
        engine.t_last_log = time.time() - 0.35
        status, reason = engine.check_normality()
        self.assertEqual(status, "SILENCE_TIMEOUT")
        self.assertIn("No log emitted for", reason)

        # Verify telemetry log file
        self.assertTrue(self.telemetry_path.exists())
        with open(self.telemetry_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["message"], "Executing GEMM iteration 1")

    def test_state_preservation_and_steering(self):
        """Slice 3: Injects synthetic error, mutates config via steering, retries and succeeds."""
        runner = StageRunner(
            checkpoint_path=self.checkpoint_path,
            telemetry_path=self.telemetry_path,
            traps_dir=self.traps_dir,
            action_file=self.action_file,
            socket_path=self.socket_path,
            silence_timeout_s=10.0,
            enable_steering_socket=False,
        )

        attempt_counts = {"stage_2": 0}

        @stage(name="load_weights", order_idx=1, expected_s=0.2)
        def stage_1(state: StateStore):
            state["weights_handle"] = "TPU_MESH_OBJECT_ADDR_0xDEADBEEF"
            state.set_param("batch_size", 8)

        @stage(name="run_batch_benchmark", order_idx=2, expected_s=0.2)
        def stage_2(state: StateStore):
            attempt_counts["stage_2"] += 1
            bs = state.get_param("batch_size")
            # Weight object MUST still be in state
            self.assertEqual(state["weights_handle"], "TPU_MESH_OBJECT_ADDR_0xDEADBEEF")

            if bs < 32:
                raise ValueError(f"Invalid batch size {bs}: TPU v5e requires batch_size >= 32 for saturated throughput")

            state["throughput_tokens_per_sec"] = bs * 128.5

        runner.register(stage_1)
        runner.register(stage_2)

        # Side watcher thread acting as OMP agent resolving the trap
        def watcher_agent():
            trap_file = self.traps_dir / "run_batch_benchmark.json"
            # Wait for trap file
            start = time.time()
            while not trap_file.exists() and time.time() - start < 3.0:
                time.sleep(0.05)

            if trap_file.exists():
                with open(trap_file, "r", encoding="utf-8") as f:
                    snapshot = json.load(f)
                self.assertEqual(snapshot["stage"], "run_batch_benchmark")
                self.assertIn("Invalid batch size 8", snapshot["exception"]["message"])

                # Inject MUTATE_CONFIG directive
                runner.steering.inject_action({
                    "op": "MUTATE_CONFIG",
                    "params": {"batch_size": 64},
                })

        watcher_thread = threading.Thread(target=watcher_agent, daemon=True)
        watcher_thread.start()

        success = runner.run()
        watcher_thread.join(timeout=2.0)

        self.assertTrue(success)
        self.assertEqual(attempt_counts["stage_2"], 2)
        self.assertEqual(runner.state.get_param("batch_size"), 64)
        self.assertAlmostEqual(runner.state["throughput_tokens_per_sec"], 64 * 128.5)

    def test_hot_patch_module_reload(self):
        """Slice 3: Modifies module on disk, sends HOT_PATCH, verifies reloaded logic executes on retry."""
        mod_dir = self.test_dir / "modules"
        mod_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(mod_dir))

        mod_file = mod_dir / "tpu_kernel_plugin.py"
        # 1. Write initial buggy version
        with open(mod_file, "w", encoding="utf-8") as f:
            f.write(
                "def execute_kernel(val):\n"
                "    raise RuntimeError('XLA uncompiled kernel compilation error: Missing custom Pallas layout')\n"
            )

        runner = StageRunner(
            checkpoint_path=self.checkpoint_path,
            telemetry_path=self.telemetry_path,
            traps_dir=self.traps_dir,
            action_file=self.action_file,
            socket_path=self.socket_path,
            silence_timeout_s=10.0,
            enable_steering_socket=False,
        )

        @stage(name="compile_pallas_kernel", order_idx=1, expected_s=0.2)
        def stage_kernel(state: StateStore):
            import tpu_kernel_plugin
            result = tpu_kernel_plugin.execute_kernel(42)
            state["kernel_result"] = result

        runner.register(stage_kernel)

        # Side watcher agent thread: fixes the file on disk and issues HOT_PATCH
        def watcher_fixer():
            trap_file = self.traps_dir / "compile_pallas_kernel.json"
            start = time.time()
            while not trap_file.exists() and time.time() - start < 3.0:
                time.sleep(0.05)

            if trap_file.exists():
                # Fix the file on disk
                with open(mod_file, "w", encoding="utf-8") as f:
                    f.write(
                        "def execute_kernel(val):\n"
                        "    return val * 100\n"
                    )

                # Send HOT_PATCH action
                runner.steering.inject_action({
                    "op": "HOT_PATCH",
                    "module": "tpu_kernel_plugin",
                })

        watcher_thread = threading.Thread(target=watcher_fixer, daemon=True)
        watcher_thread.start()

        success = runner.run()
        watcher_thread.join(timeout=2.0)

        self.assertTrue(success)
        self.assertEqual(runner.state["kernel_result"], 4200)

        # Clean up sys.path
        if str(mod_dir) in sys.path:
            sys.path.remove(str(mod_dir))

    def test_sigusr1_asymmetric_interrupt(self):
        """Slice 4: Watchdog / signal triggers non-destructive pause and clean resolution."""
        runner = StageRunner(
            checkpoint_path=self.checkpoint_path,
            telemetry_path=self.telemetry_path,
            traps_dir=self.traps_dir,
            action_file=self.action_file,
            socket_path=self.socket_path,
            silence_timeout_s=10.0,
            enable_steering_socket=False,
            watchdog_interval_s=0.05,
        )

        stage_reached = threading.Event()
        attempts = {"count": 0}

        @stage(name="streaming_generation", order_idx=1, expected_s=1.0)
        def stage_streaming(state: StateStore):
            attempts["count"] += 1
            stage_reached.set()
            if attempts["count"] == 1:
                time.sleep(10.0)
            state["completed"] = True

        runner.register(stage_streaming)

        def interrupter_agent():
            stage_reached.wait(timeout=2.0)
            time.sleep(0.05)
            # Trigger interrupt
            runner.watchdog.trigger_interrupt()

            # Wait for trap
            trap_file = self.traps_dir / "streaming_generation.json"
            start = time.time()
            while not trap_file.exists() and time.time() - start < 3.0:
                time.sleep(0.05)

            if trap_file.exists():
                with open(trap_file, "r", encoding="utf-8") as f:
                    snapshot = json.load(f)
                self.assertEqual(snapshot["stage"], "streaming_generation")
                self.assertIn("SIGUSR1", snapshot["reason"])

                # Retry stage
                runner.steering.inject_action({"op": "RETRY"})
        interrupter_thread = threading.Thread(target=interrupter_agent, daemon=True)
        interrupter_thread.start()

        success = runner.run()
        interrupter_thread.join(timeout=2.0)

        self.assertTrue(success)
        self.assertTrue(runner.state["completed"])

    def test_silence_watchdog_preemption_and_retry(self):
        """Slice 4: Tests silence timeout watchdog preempting a hung stage and retrying cleanly."""
        runner = StageRunner(
            checkpoint_path=self.checkpoint_path,
            telemetry_path=self.telemetry_path,
            traps_dir=self.traps_dir,
            action_file=self.action_file,
            socket_path=self.socket_path,
            silence_timeout_s=0.2,
            enable_steering_socket=False,
            watchdog_interval_s=0.05,
        )

        stage1_attempts = {"count": 0}

        @stage(name="hang_stage", order_idx=1, expected_s=0.5)
        def stage_1(state: StateStore, timing: JointTimingLogEngine):
            stage1_attempts["count"] += 1
            if stage1_attempts["count"] == 1:
                time.sleep(5.0)  # Hung sleep, should be preempted by 0.2s silence watchdog
            timing.log("Stage 1 succeeded", level="INFO")
            state["s1"] = True

        @stage(name="subsequent_stage", order_idx=2, expected_s=0.5)
        def stage_2(state: StateStore, timing: JointTimingLogEngine):
            timing.log("Stage 2 succeeded", level="INFO")
            state["s2"] = True

        runner.register(stage_1)
        runner.register(stage_2)

        def watcher():
            trap_file = self.traps_dir / "hang_stage.json"
            start = time.time()
            while not trap_file.exists() and time.time() - start < 3.0:
                time.sleep(0.05)
            if trap_file.exists():
                runner.steering.inject_action({"op": "RETRY"})

        w_thread = threading.Thread(target=watcher, daemon=True)
        w_thread.start()

        success = runner.run()
        w_thread.join(timeout=2.0)

        self.assertTrue(success)
        self.assertEqual(stage1_attempts["count"], 2)
        self.assertTrue(runner.state["s1"])
        self.assertTrue(runner.state["s2"])

    def test_skip_stage_and_abort(self):
        """Slice 3: Tests skipping a failing stage and proceeding, and aborting execution."""
        runner = StageRunner(
            checkpoint_path=self.checkpoint_path,
            telemetry_path=self.telemetry_path,
            traps_dir=self.traps_dir,
            action_file=self.action_file,
            socket_path=self.socket_path,
            silence_timeout_s=5.0,
            enable_steering_socket=False,
        )

        @stage(name="failing_stage", order_idx=1, expected_s=0.2)
        def stage_1(state: StateStore):
            raise ValueError("Intentional error to test skip")

        @stage(name="next_stage", order_idx=2, expected_s=0.2)
        def stage_2(state: StateStore):
            state["s2_reached"] = True

        runner.register(stage_1)
        runner.register(stage_2)

        def watcher():
            trap_file = self.traps_dir / "failing_stage.json"
            start = time.time()
            while not trap_file.exists() and time.time() - start < 3.0:
                time.sleep(0.05)
            if trap_file.exists():
                runner.steering.inject_action({"op": "SKIP_STAGE"})

        w_thread = threading.Thread(target=watcher, daemon=True)
        w_thread.start()

        success = runner.run()
        w_thread.join(timeout=2.0)

        self.assertTrue(success)
        self.assertTrue(runner.state["s2_reached"])

    def test_trap_timeout_expiration(self):
        """Slice 4: Tests trap timeout aborts execution when no directive is received."""
        runner = StageRunner(
            checkpoint_path=self.checkpoint_path,
            telemetry_path=self.telemetry_path,
            traps_dir=self.traps_dir,
            action_file=self.action_file,
            socket_path=self.socket_path,
            silence_timeout_s=5.0,
            enable_steering_socket=False,
            trap_timeout_s=0.3,
        )

        @stage(name="failing_stage", order_idx=1, expected_s=0.2)
        def stage_1(state: StateStore):
            raise RuntimeError("Unrecoverable error with no watcher")

        runner.register(stage_1)
        success = runner.run()
        self.assertFalse(success)

    def test_state_store_complex_objects(self):
        """Tests StateStore handling non-JSON serializable objects gracefully."""
        state = StateStore()
        state["number"] = 123
        state["string"] = "hello"
        state["list"] = [1, 2, 3]
        state["custom_obj"] = object()
        state["thread"] = threading.current_thread()

        d = state.to_dict()
        self.assertEqual(d["number"], 123)
        self.assertEqual(d["string"], "hello")
        self.assertEqual(d["list"], [1, 2, 3])
        self.assertIn("object", d["custom_obj"])
        self.assertIn("_MainThread", d["thread"])

    def test_strict_blocking_barrier_and_mutation(self):
        """Proves that StageRunner freezes completely while trapped and only resumes on steering."""
        runner = StageRunner(
            checkpoint_path=self.checkpoint_path,
            telemetry_path=self.telemetry_path,
            traps_dir=self.traps_dir,
            action_file=self.action_file,
            socket_path=self.socket_path,
            silence_timeout_s=5.0,
            enable_steering_socket=False,
        )

        events = []

        @stage(name="s1", order_idx=1, expected_s=0.5)
        def s1(state: StateStore):
            events.append("s1_done")
            state.set_param("val", 10)

        @stage(name="s2", order_idx=2, expected_s=0.5)
        def s2(state: StateStore):
            val = state.get_param("val")
            events.append(f"s2_val_{val}")
            if val < 50:
                raise ValueError(f"Val too low: {val}")
            events.append("s2_success")

        @stage(name="s3", order_idx=3, expected_s=0.5)
        def s3(state: StateStore):
            events.append("s3_done")

        runner.register(s1)
        runner.register(s2)
        runner.register(s3)

        trap_file = self.traps_dir / "s2.json"

        def steerer():
            start = time.time()
            while not trap_file.exists() and time.time() - start < 3.0:
                time.sleep(0.02)
            self.assertTrue(trap_file.exists())

            # Assert s3 has not executed
            self.assertNotIn("s3_done", events)
            self.assertNotIn("s2_success", events)

            # Hold open for 0.4s to prove pipeline is frozen
            time.sleep(0.4)
            self.assertNotIn("s3_done", events)
            self.assertNotIn("s2_success", events)

            # Mutate and solve
            runner.steering.inject_action({"op": "MUTATE_CONFIG", "params": {"val": 100}})

        t = threading.Thread(target=steerer, daemon=True)
        t.start()

        t_start = time.time()
        success = runner.run()
        t_elapsed = time.time() - t_start
        t.join(timeout=2.0)

        self.assertTrue(success)
        self.assertGreaterEqual(t_elapsed, 0.4)
        self.assertEqual(events, ["s1_done", "s2_val_10", "s2_val_100", "s2_success", "s3_done"])
        self.assertEqual(runner.state.get_param("val"), 100)

    def test_watchdog_preemption_and_freeze_barrier(self):
        """Proves that a 100s blocking sleep is preempted by silence watchdog and frozen until steered."""
        runner = StageRunner(
            checkpoint_path=self.checkpoint_path,
            telemetry_path=self.telemetry_path,
            traps_dir=self.traps_dir,
            action_file=self.action_file,
            socket_path=self.socket_path,
            silence_timeout_s=0.2,
            enable_steering_socket=False,
            watchdog_interval_s=0.05,
        )

        events = []
        attempts = {"s2": 0}

        @stage(name="s1", order_idx=1, expected_s=0.5)
        def s1(state: StateStore):
            events.append("s1_done")

        @stage(name="s2", order_idx=2, expected_s=0.5)
        def s2(state: StateStore):
            attempts["s2"] += 1
            events.append(f"s2_attempt_{attempts['s2']}")
            if attempts["s2"] == 1:
                time.sleep(100.0)  # Simulated deadlock
            events.append("s2_success")

        @stage(name="s3", order_idx=3, expected_s=0.5)
        def s3(state: StateStore):
            events.append("s3_done")

        runner.register(s1)
        runner.register(s2)
        runner.register(s3)

        trap_file = self.traps_dir / "s2.json"

        def steerer():
            start = time.time()
            while not trap_file.exists() and time.time() - start < 3.0:
                time.sleep(0.02)
            self.assertTrue(trap_file.exists())

            self.assertNotIn("s3_done", events)
            self.assertNotIn("s2_success", events)

            time.sleep(0.3)
            self.assertNotIn("s3_done", events)
            self.assertNotIn("s2_success", events)

            runner.steering.inject_action({"op": "RETRY"})

        t = threading.Thread(target=steerer, daemon=True)
        t.start()

        t_start = time.time()
        success = runner.run()
        t_elapsed = time.time() - t_start
        t.join(timeout=2.0)

        self.assertTrue(success)
        self.assertLess(t_elapsed, 4.0)  # Preempted 100s
        self.assertGreaterEqual(t_elapsed, 0.5)  # 0.2s silence + 0.3s hold
        self.assertEqual(events, ["s1_done", "s2_attempt_1", "s2_attempt_2", "s2_success", "s3_done"])

    def test_cli_status_and_system_directive_generation(self):
        """Validates that CLI system directive format and idle status behave correctly."""
        trap_data = {
            "stage": "decode_stage",
            "order_idx": 2,
            "reason": "EXCEPTION: RuntimeError: Stride alignment error",
            "exception": {
                "type": "RuntimeError",
                "message": "Stride alignment error",
                "traceback": '  File "/tmp/kernel.py", line 42, in compute\n    raise RuntimeError("Stride alignment error")',
            },
            "state_keys": ["weights", "placement_group"],
            "log_tail": ["[INFO] Initializing", "[ERROR] Stride mismatch"],
        }
        banner = format_trap_system_directive(trap_data, daemon_pid=12345)
        self.assertIn("<system-directive>", banner)
        self.assertIn("</system-directive>", banner)
        self.assertIn("PID 12345 PAUSED", banner)
        self.assertIn("/tmp/kernel.py:42", banner)
        self.assertIn("xd://checkpoint", banner)
        self.assertIn("xd://rewind", banner)

        # Test CLI status with no active daemon
        rc = cli_main(["status"])
        self.assertEqual(rc, 0)

    def test_apply_unified_diff(self):
        """Validates that apply_unified_diff correctly patches target source text."""
        original = (
            "def compute(x: int) -> int:\n"
            "    val = x * 2\n"
            "    return val\n"
        )
        diff = (
            "--- a/compute.py\n"
            "+++ b/compute.py\n"
            "@@ -2,2 +2,2 @@\n"
            "-    val = x * 2\n"
            "+    val = x * 4\n"
        )
        patched = apply_unified_diff(original, diff)
        self.assertEqual(
            patched,
            "def compute(x: int) -> int:\n"
            "    val = x * 4\n"
            "    return val\n",
        )

    def test_ray_job_supervisor_lifecycle_and_streaming(self):
        """Validates that RayJobSupervisor submits, polls, streams logs, and extracts measurements."""
        job_file = self.test_dir / "ray_job.json"

        class MockJobStatus:
            def __init__(self, name: str, terminal: bool):
                self.name = name
                self._terminal = terminal
            def is_terminal(self):
                return self._terminal

        class MockRayClient:
            def __init__(self):
                self.call_count = 0
                self.stopped = False

            def submit_job(self, entrypoint: str, runtime_env: dict, metadata: dict):
                return "raysubmit_mock_abc123"

            def get_job_status(self, submission_id: str):
                self.call_count += 1
                if self.stopped:
                    return MockJobStatus("STOPPED", True)
                if self.call_count < 3:
                    return MockJobStatus("RUNNING", False)
                return MockJobStatus("SUCCEEDED", True)

            def get_job_logs(self, submission_id: str):
                if self.call_count == 1:
                    return "[INFO] Worker initializing...\n[OMP_EVENT: TPU_WARMUP_STARTED chips=8]\n"
                elif self.call_count == 2:
                    return (
                        "[INFO] Worker initializing...\n[OMP_EVENT: TPU_WARMUP_STARTED chips=8]\n"
                        "[INFO] Step 1 complete.\n[OMP_EVENT: CLUSTER_JOB_RUNNING step=1/2]\n"
                    )
                else:
                    return (
                        "[INFO] Worker initializing...\n[OMP_EVENT: TPU_WARMUP_STARTED chips=8]\n"
                        "[INFO] Step 1 complete.\n[OMP_EVENT: CLUSTER_JOB_RUNNING step=1/2]\n"
                        "EMPIRICAL TPU HARDWARE MEASUREMENTS (JSON OUTPUT)\n"
                        "================================================\n"
                        '{"empirical_measurements": {"goodput": 1437.2, "preemptions": 0}}\n\n'
                        "[OMP_EVENT: TPU_JOB_COMPLETED]\n"
                    )

            def stop_job(self, submission_id: str):
                self.stopped = True

        mock_client = MockRayClient()
        timing = JointTimingLogEngine(telemetry_path=self.telemetry_path, silence_timeout_s=10.0)
        supervisor = RayJobSupervisor(
            ray_url="http://127.0.0.1:8265",
            client=mock_client,
            timing_engine=timing,
            job_file=job_file,
        )

        sub_id = supervisor.submit_job(
            entrypoint="python3 run_test.py",
            metadata={"topology": "2x4"},
        )
        self.assertEqual(sub_id, "raysubmit_mock_abc123")
        self.assertTrue(job_file.exists())

        # Test reading active job file
        active_data = read_active_ray_job(job_file=job_file)
        self.assertIsNotNone(active_data)
        self.assertEqual(active_data["submission_id"], "raysubmit_mock_abc123")
        self.assertEqual(active_data["status"], "PENDING")

        summary = format_ray_job_summary(active_data)
        self.assertIn("raysubmit_mock_abc123", summary)
        self.assertIn("http://127.0.0.1:8265", summary)

        # Supervise to completion
        state = StateStore()
        res = supervisor.supervise_until_completion(
            poll_interval_s=0.01,
            stdout_stream=False,
            stage_name="test_stage",
            state_store=state,
        )
        self.assertEqual(res["status"], "SUCCEEDED")
        self.assertIn("empirical_measurements", res["measurements"])
        self.assertEqual(res["measurements"]["empirical_measurements"]["goodput"], 1437.2)

        # Verify telemetry file has forwarded events
        telemetry_lines = self.telemetry_path.read_text(encoding="utf-8").splitlines()
        self.assertTrue(any("TPU_WARMUP_STARTED" in l for l in telemetry_lines))
        self.assertTrue(any("CLUSTER_JOB_RUNNING" in l for l in telemetry_lines))

    def test_ray_job_supervisor_abort(self):
        """Validates that stop_job cancels the Ray job and updates status to STOPPED."""
        job_file = self.test_dir / "ray_job_abort.json"

        class MockRayClient:
            def __init__(self):
                self.stopped = False
            def submit_job(self, entrypoint: str, runtime_env: dict, metadata: dict):
                return "raysubmit_abort_test"
            def get_job_status(self, submission_id: str):
                return "STOPPED" if self.stopped else "RUNNING"
            def get_job_logs(self, submission_id: str):
                return "Executing..."
            def stop_job(self, submission_id: str):
                self.stopped = True

        mock_client = MockRayClient()
        supervisor = RayJobSupervisor(
            ray_url="http://127.0.0.1:8265",
            client=mock_client,
            job_file=job_file,
        )
        supervisor.submit_job(entrypoint="python3 test.py")
        self.assertEqual(supervisor.active_job.status, "PENDING")

        stopped = supervisor.stop_job()
        self.assertTrue(stopped)
        self.assertTrue(mock_client.stopped)
        self.assertEqual(supervisor.active_job.status, "STOPPED")

        active_data = read_active_ray_job(job_file=job_file)
        self.assertEqual(active_data["status"], "STOPPED")

    def test_state_store_ray_job_helpers(self):
        """Validates StateStore set_ray_job and get_ray_job."""
        state = StateStore()
        self.assertIsNone(state.get_ray_job())
        state.set_ray_job({"submission_id": "sub_1", "status": "RUNNING"})
        self.assertEqual(state.get_ray_job()["submission_id"], "sub_1")

    def test_ray_job_supervisor_multi_job_supervision(self):
        """Validates that multiple concurrent Ray cluster jobs are supervised and multiplexed."""
        job_file = self.test_dir / "ray_job.json"
        jobs_file = self.test_dir / "ray_jobs.json"

        class MultiMockRayClient:
            def __init__(self):
                self.calls = {"job_1": 0, "job_2": 0}
                self.stopped = set()

            def submit_job(self, entrypoint: str, runtime_env: dict, metadata: dict):
                sub_id = f"raysubmit_{metadata.get('slice', 'unknown')}"
                self.calls[sub_id] = 0
                return sub_id

            def get_job_status(self, submission_id: str):
                if submission_id in self.stopped:
                    return "STOPPED"
                self.calls[submission_id] = self.calls.get(submission_id, 0) + 1
                if self.calls[submission_id] < 2:
                    return "RUNNING"
                return "SUCCEEDED"

            def get_job_logs(self, submission_id: str):
                cnt = self.calls.get(submission_id, 0)
                if cnt == 1:
                    return f"[INFO] {submission_id} step 1\n"
                return (
                    f"[INFO] {submission_id} step 1\n"
                    f"[INFO] {submission_id} complete\n"
                    f"TOPOLOGY_RESULT_JSON={{\"job\": \"{submission_id}\", \"goodput\": 1250.0}}\n"
                )

            def stop_job(self, submission_id: str):
                self.stopped.add(submission_id)

        client = MultiMockRayClient()
        timing = JointTimingLogEngine(telemetry_path=self.telemetry_path, silence_timeout_s=10.0)
        supervisor = RayJobSupervisor(
            client=client,
            timing_engine=timing,
            job_file=job_file,
            jobs_file=jobs_file,
        )

        sub_1 = supervisor.submit_job(
            entrypoint="python3 prefill.py",
            name="Prefill-2x4",
            metadata={"slice": "prefill"},
        )
        sub_2 = supervisor.submit_job(
            entrypoint="python3 decode.py",
            name="Decode-2x4",
            metadata={"slice": "decode"},
        )

        self.assertEqual(len(supervisor.jobs), 2)
        self.assertEqual(supervisor.active_job_id, sub_2)

        # Verify multi-job state file
        all_jobs_data = read_all_ray_jobs(jobs_file=jobs_file)
        self.assertEqual(all_jobs_data["total_jobs"], 2)
        self.assertIn(sub_1, all_jobs_data["jobs"])
        self.assertIn(sub_2, all_jobs_data["jobs"])

        table = format_ray_jobs_table(all_jobs_data)
        self.assertIn("Prefill-2x4", table)
        self.assertIn("Decode-2x4", table)

        # Supervise both jobs concurrently
        state = StateStore()
        res = supervisor.supervise_jobs(
            submission_ids=[sub_1, sub_2],
            poll_interval_s=0.01,
            stdout_stream=False,
            stage_name="multi_job_stage",
            state_store=state,
        )
        self.assertEqual(res["status"], "SUCCEEDED")
        self.assertIn(sub_1, res["all_measurements"])
        self.assertIn(sub_2, res["all_measurements"])
        self.assertEqual(res["all_measurements"][sub_1]["goodput"], 1250.0)
        self.assertEqual(res["all_measurements"][sub_2]["goodput"], 1250.0)

    def test_ray_job_supervisor_stop_all(self):
        """Validates that stop_all_jobs stops every active non-terminal Ray job."""
        job_file = self.test_dir / "ray_job.json"
        jobs_file = self.test_dir / "ray_jobs.json"

        class MultiStopMockClient:
            def __init__(self):
                self.stopped = set()
            def submit_job(self, entrypoint: str, runtime_env: dict, metadata: dict):
                return f"raysubmit_{metadata.get('id')}"
            def get_job_status(self, submission_id: str):
                return "STOPPED" if submission_id in self.stopped else "RUNNING"
            def get_job_logs(self, submission_id: str):
                return ""
            def stop_job(self, submission_id: str):
                self.stopped.add(submission_id)

        client = MultiStopMockClient()
        supervisor = RayJobSupervisor(client=client, job_file=job_file, jobs_file=jobs_file)
        s1 = supervisor.submit_job(entrypoint="p1.py", metadata={"id": "j1"})
        s2 = supervisor.submit_job(entrypoint="p2.py", metadata={"id": "j2"})

        stopped_cnt = supervisor.stop_all_jobs()
        self.assertEqual(stopped_cnt, 2)
        self.assertTrue(supervisor.jobs[s1].terminal)
        self.assertTrue(supervisor.jobs[s2].terminal)
        self.assertEqual(client.stopped, {s1, s2})

    def test_cli_wait_immediate_trap(self):
        """Validates that rcm_exec wait immediately catches an active trap and exits code 2."""
        TRAPS_DIR.mkdir(parents=True, exist_ok=True)
        trap_data = {
            "stage": "prefill_stage",
            "reason": "OOM simulated",
            "exception": {"type": "RuntimeError", "message": "CUDA OOM", "traceback": ""},
            "state_keys": ["weights"],
        }
        with open(LATEST_TRAP_FILE, "w", encoding="utf-8") as f:
            json.dump(trap_data, f)

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            code = cli_main(["wait"])
            out = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            LATEST_TRAP_FILE.unlink(missing_ok=True)

        self.assertEqual(code, 2)
        self.assertIn("RCM DIAGNOSTIC TRAP", out)
        self.assertIn("prefill_stage", out)

    def test_cli_wait_timeout(self):
        """Validates that rcm_exec wait exits with code 124 when timeout expires."""
        import subprocess
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        DAEMON_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
        LATEST_TRAP_FILE.unlink(missing_ok=True)

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            code = cli_main(["wait", "--timeout", "0.2"])
            out = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            proc.terminate()
            proc.wait()
            DAEMON_PID_FILE.unlink(missing_ok=True)

        self.assertEqual(code, 124)
        self.assertIn("timed out", out)


if __name__ == "__main__":
    unittest.main()
