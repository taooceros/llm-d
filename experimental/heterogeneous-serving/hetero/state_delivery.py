"""Autoscale boot overlap: a new instance boots while its startup state is still being delivered.

Scaling up onto fresh hosts takes two independent pieces of work:

* deliver the startup state (weight snapshot, executable bank, XLA cache) onto the hosts, which is
  ``scripts/startup_state.py fetch``;
* bring up the engine process tree: placement, engine and worker actors, Python imports, and vLLM
  and tpu-inference construction. This takes ~22 s before the first device touch and needs no state.

``deploy(state_delivery=...)`` runs the fetch's remote driver as a child process (``Delivery``) while
the engines come up. A worker whose engine setting ``state_gate_timeout_s`` is set waits right
before its first device touch until the fetch has published that worker's rank. The fetch's
``finalize`` signals this with a marker file (``marker_path``). It writes the marker after the
snapshot manifest and the bank are published and every XLA cache file has landed. Waiting before the
device touch, not before ``load_model``, matters: device init can already compile, and JAX's cache
``put`` writes non-atomically into the directory the fetch is filling.

This is not the reconfiguration pre-spawn (hetero/prespawn.py). In that flow the target's workers are
CPU-only and wait for a source engine to release the TPUs. Here the TPUs are free; the workers hold
them from the start and wait only for their host's startup state.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .evidence import BANNER

DELIVERED_DIR = ".delivered"


def marker_path(weight_snapshot_root: str, layout_id: str, instance_id: str, rank: int) -> Path:
    """Where the fetch publishes "rank ``rank`` of this instance holds its startup state"."""
    return Path(weight_snapshot_root) / DELIVERED_DIR / layout_id / instance_id / f"rank{int(rank)}.json"


def write_marker(path: Path, report: Dict[str, Any]) -> None:
    """Atomically publish a delivery marker (rename into place, then sync the directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".incoming")
    with open(staged, "w") as handle:
        json.dump(report, handle, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def wait_marker(path: Path, *, timeout_s: float, poll_s: float = 0.05,
                clock: Callable[[], float] = time.time,
                sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """Block until ``path`` exists, then return the wait and the marker contents. Raise on timeout."""
    started = clock()
    polls = 0
    while True:
        polls += 1
        if path.exists():
            seen = clock()
            return {"marker": str(path), "started_at": started, "seen_at": seen,
                    "waited_s": seen - started, "polls": polls,
                    "delivered": json.loads(path.read_text())}
        if clock() - started > timeout_s:
            raise TimeoutError(f"startup state not delivered within {timeout_s:.0f} s: no {path}")
        sleep(poll_s)


def validate_spec(spec: Any) -> Dict[str, Any]:
    """``{"argv": [...], "timeout_s": N}``. ``argv`` is the fetch's remote-driver command line,
    as printed by ``startup_state.py fetch --print-remote-argv``."""
    if not isinstance(spec, dict) or set(spec) - {"argv", "timeout_s"}:
        raise ValueError("state delivery: a JSON object with keys argv and timeout_s")
    argv, timeout_s = spec.get("argv"), spec.get("timeout_s", 1800.0)
    if not (isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv)):
        raise ValueError("state delivery argv: a non-empty list of strings")
    if "--remote" not in argv:
        raise ValueError("state delivery argv must run the fetch's remote driver (--remote)")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not timeout_s > 0:
        raise ValueError("state delivery timeout_s must be a positive number")
    return {"argv": list(argv), "timeout_s": float(timeout_s)}


class Delivery:
    """The fetch's remote driver running as a child of the deploy driver.

    Its log lines are forwarded as plain log lines. Its ``[OMP_EVENT: ...]`` lines are defused,
    and its measurement payload is captured, not re-emitted. That keeps the deploy job's one scraped
    payload unambiguous.
    """

    def __init__(self, spec: Dict[str, Any], *, log: Callable[[str], None]) -> None:
        spec = validate_spec(spec)
        argv = spec["argv"]
        # The job's own interpreter: same runtime env (working_dir, pip) as the deploy driver.
        self.argv: List[str] = [sys.executable, *argv[1:]] if Path(argv[0]).name.startswith("python") else argv
        self.timeout_s = spec["timeout_s"]
        self._log = log
        self.payload: Optional[Dict[str, Any]] = None
        self.tail: collections.deque = collections.deque(maxlen=40)
        self.started_at = time.time()
        self.exited_at: Optional[float] = None
        self.proc = subprocess.Popen(self.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     bufsize=1)
        self._reader = threading.Thread(target=self._read, name="state-delivery-log", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        after_banner = False
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            if line == BANNER:
                after_banner = True
                continue
            if line and set(line) == {"="}:
                continue
            if after_banner and line.startswith("{"):
                try:
                    self.payload = json.loads(line)
                except ValueError:
                    self.tail.append(line[:300])
                after_banner = False
                continue
            if line:
                self.tail.append(line[:300])
                self._log("state delivery: " + line.replace("[OMP_EVENT:", "(event:"))
        self.exited_at = time.time()

    def check(self) -> None:
        """Raise as soon as the delivery has failed or overrun its timeout."""
        code = self.proc.poll()
        if code not in (None, 0):
            self._reader.join(timeout=10)
            raise RuntimeError(f"state delivery failed (exit {code}): " + " | ".join(list(self.tail)[-8:]))
        if code is None and time.time() - self.started_at > self.timeout_s:
            raise TimeoutError(f"state delivery still running after {self.timeout_s:.0f} s")

    def finish(self) -> Dict[str, Any]:
        """Wait for the child to exit and return its evidence. Raise unless it reported success."""
        remaining = max(1.0, self.started_at + self.timeout_s - time.time())
        try:
            code = self.proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            self.abort()
            raise TimeoutError(f"state delivery still running after {self.timeout_s:.0f} s") from None
        self._reader.join(timeout=30)
        if code != 0 or not self.payload or self.payload.get("ok") is not True:
            raise RuntimeError(f"state delivery failed (exit {code}, payload ok="
                               f"{(self.payload or {}).get('ok')}): " + " | ".join(list(self.tail)[-8:]))
        return self.evidence()

    def evidence(self) -> Dict[str, Any]:
        return {"argv": self.argv, "timeout_s": self.timeout_s, "started_at": self.started_at,
                "exited_at": self.exited_at, "returncode": self.proc.poll(),
                "wall_s": None if self.exited_at is None else self.exited_at - self.started_at,
                "fetch": self.payload}

    def abort(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=15)


def await_initialized(futures: Dict[str, Any], *, timeout_s: float, delivery: Optional[Delivery],
                      poll_s: float = 1.0) -> Dict[str, Any]:
    """``ray.get`` every engine's initialize future. A failed delivery fails the deploy immediately,
    rather than after the workers' gate timeout."""
    import ray

    if delivery is None:
        return {iid: ray.get(future, timeout=timeout_s) for iid, future in futures.items()}
    deadline = time.time() + timeout_s
    pending = {future: iid for iid, future in futures.items()}
    reports: Dict[str, Any] = {}
    while pending:
        done, _ = ray.wait(list(pending), num_returns=len(pending), timeout=poll_s)
        for future in done:
            reports[pending.pop(future)] = ray.get(future)
        if pending:
            delivery.check()
            if time.time() > deadline:
                raise TimeoutError(f"engines {sorted(pending.values())} not initialized within {timeout_s:.0f} s")
    return {iid: reports[iid] for iid in futures}
