"""Autoscale boot overlap (hetero/state_delivery.py): the fetch runs as a child of the deploy driver
and the state-gated workers wait for their rank's delivery marker.

The child is a real subprocess that speaks the evidence protocol, so these tests cover the parsing
of the payload a supervisor would scrape, the defusing of its event lines, and failure detection.
"""
import json
import pathlib
import sys
import tempfile
import textwrap
import threading
import time
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hetero import state_delivery as sd  # noqa: E402
from hetero.evidence import BANNER  # noqa: E402


def _child(tmp, body):
    """A python script standing in for ``startup_state.py fetch --remote``."""
    script = pathlib.Path(tmp) / "child.py"
    script.write_text(textwrap.dedent(body))
    return {"argv": ["python3", "-u", str(script), "fetch", "--remote"], "timeout_s": 60}


class MarkerTests(unittest.TestCase):
    def test_worker_waits_for_its_rank_marker_and_reads_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = sd.marker_path(tmp, "H", "large_b", 2)
            self.assertEqual(path, pathlib.Path(tmp) / ".delivered/H/large_b/rank2.json")
            threading.Timer(0.3, sd.write_marker, (path, {"rank": 2, "parts": ["snapshot"]})).start()
            gate = sd.wait_marker(path, timeout_s=10, poll_s=0.01)
            self.assertGreaterEqual(gate["waited_s"], 0.2)
            self.assertEqual(gate["delivered"]["rank"], 2)
            self.assertFalse(path.with_name(path.name + ".incoming").exists())

    def test_missing_marker_times_out(self):
        clock = iter(range(100)).__next__
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TimeoutError):
                sd.wait_marker(sd.marker_path(tmp, "H", "large_b", 0), timeout_s=5,
                               clock=lambda: float(clock()), sleep=lambda s: None)


class DeliveryTests(unittest.TestCase):
    def test_payload_is_captured_and_events_defused(self):
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            spec = _child(tmp, f"""
                import json
                print("[INFO] fetching", flush=True)
                print("[OMP_EVENT: MEASUREMENTS_EMITTED {{}}]", flush=True)
                print("\\n" + "=" * 78 + "\\n{BANNER}\\n" + "=" * 78)
                print(json.dumps({{"ok": True, "wall_s": 1.5, "method": "peer"}}) + "\\n")
            """)
            delivery = sd.Delivery(spec, log=logs.append)
            self.assertEqual(delivery.argv[0], sys.executable)
            evidence = delivery.finish()
        self.assertEqual(evidence["fetch"], {"ok": True, "wall_s": 1.5, "method": "peer"})
        self.assertEqual(evidence["returncode"], 0)
        self.assertGreater(evidence["wall_s"], 0)
        self.assertIn("state delivery: [INFO] fetching", logs)
        self.assertFalse(any("[OMP_EVENT:" in line or BANNER in line or '"ok"' in line for line in logs), logs)

    def test_failed_child_fails_check_and_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _child(tmp, """
                import sys
                print("RuntimeError: target already holds a delivery marker", flush=True)
                sys.exit(3)
            """)
            delivery = sd.Delivery(spec, log=lambda line: None)
            delivery.proc.wait(timeout=30)
            with self.assertRaisesRegex(RuntimeError, "exit 3.*delivery marker"):
                delivery.check()
            with self.assertRaises(RuntimeError):
                delivery.finish()

    def test_child_without_a_successful_payload_fails_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _child(tmp, f"""
                import json
                print("{BANNER}")
                print(json.dumps({{"ok": False, "error": "x"}}))
            """)
            with self.assertRaisesRegex(RuntimeError, "payload ok=False"):
                sd.Delivery(spec, log=lambda line: None).finish()

    def test_spec_must_name_the_remote_driver(self):
        for bad in ({"argv": []}, {"argv": ["python3", "x.py"]}, {"argv": ["python3", "--remote"], "timeout_s": 0},
                    {"argv": ["python3", "--remote"], "extra": 1}, [], {"argv": ["python3", "--remote"], "timeout_s": True}):
            with self.assertRaises(ValueError, msg=bad):
                sd.validate_spec(bad)
        self.assertEqual(sd.validate_spec({"argv": ["python3", "--remote"]})["timeout_s"], 1800.0)


class AwaitInitializedTests(unittest.TestCase):
    def _fake_ray(self, ready_after):
        started = time.time()

        def wait(refs, num_returns, timeout):
            time.sleep(min(timeout, 0.01))
            done = [r for r in refs if time.time() - started > ready_after[r]]
            return done, [r for r in refs if r not in done]

        return types.SimpleNamespace(wait=wait, get=lambda ref, timeout=None: {"instance": ref})

    def test_failed_delivery_fails_the_deploy_before_the_engines_finish(self):
        failing = mock.Mock()
        failing.check.side_effect = RuntimeError("state delivery failed (exit 1)")
        with mock.patch.dict(sys.modules, {"ray": self._fake_ray({"fa": 0.0, "fb": 3600.0})}):
            started = time.time()
            with self.assertRaisesRegex(RuntimeError, "state delivery failed"):
                sd.await_initialized({"large_a": "fa", "large_b": "fb"}, timeout_s=600, delivery=failing,
                                     poll_s=0.01)
            self.assertLess(time.time() - started, 5)

    def test_reports_keep_instance_order(self):
        healthy = mock.Mock()
        with mock.patch.dict(sys.modules, {"ray": self._fake_ray({"fa": 0.05, "fb": 0.0})}):
            reports = sd.await_initialized({"large_a": "fa", "large_b": "fb"}, timeout_s=60, delivery=healthy,
                                           poll_s=0.01)
        self.assertEqual(list(reports), ["large_a", "large_b"])
        self.assertEqual(reports["large_a"], {"instance": "fa"})


class SettingsTests(unittest.TestCase):
    def test_state_gate_setting(self):
        from hetero.engine_actor import EngineSettings

        self.assertEqual(EngineSettings().state_gate_timeout_s, 0.0)
        self.assertEqual(EngineSettings(weight_snapshot="read", state_gate_timeout_s=900).state_gate_timeout_s, 900)
        for bad in ({"state_gate_timeout_s": -1}, {"state_gate_timeout_s": True},
                    {"state_gate_timeout_s": 60, "prespawn": True}):
            with self.assertRaises(ValueError, msg=bad):
                EngineSettings(**bad)


if __name__ == "__main__":
    unittest.main()
