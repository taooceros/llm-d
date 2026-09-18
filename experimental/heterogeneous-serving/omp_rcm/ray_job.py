"""
Reverse Control Mechanism (RCM) Ray Job Supervisor.
===================================================
Directly supervises, monitors, and steers Ray cluster workloads within the
RCM runtime, eliminating the need for agents to monitor jobs through the Ray API.

Supports both single-job and concurrent multi-job execution across Ray clusters.

Key Features:
- Submits and tracks multiple concurrent/sequential Ray cluster jobs.
- Continuously streams remote logs and forwards [OMP_EVENT: ...] lines into RCM telemetry.
- Multiplexes concurrent job streams with clean tags ([job_name|submission_id]).
- Intercepts remote failures, stalls, and anomaly patterns as native RCM DiagnosticTraps.
- Supports in-flight steering (ABORT cleanly stops remote Ray jobs; RETRY/MUTATE_CONFIG adjusts runtime).
- Persists live state to /tmp/omp_rcm/ray_jobs.json and /tmp/omp_rcm/ray_job.json for zero-token CLI inspection.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .core import (
    BASE_RCM_DIR,
    DiagnosticTrap,
    JointTimingLogEngine,
    LIVE_STATUS_FILE,
    StateStore,
    SteeringReceiver,
    TELEMETRY_FILE,
)

RAY_JOB_FILE = BASE_RCM_DIR / "ray_job.json"
RAY_JOBS_FILE = BASE_RCM_DIR / "ray_jobs.json"
DEFAULT_RAY_URL = os.environ.get("RAY_DASHBOARD_URL", "http://127.0.0.1:8265")


@dataclasses.dataclass
class RayJobInfo:
    """Structured container for Ray cluster job metadata and live execution status."""

    submission_id: str
    ray_url: str
    entrypoint: str
    status: str = "PENDING"
    name: Optional[str] = None
    start_time: float = dataclasses.field(default_factory=time.time)
    updated_time: float = dataclasses.field(default_factory=time.time)
    elapsed_s: float = 0.0
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    error: Optional[str] = None
    terminal: bool = False
    last_event: Optional[str] = None
    log_tail: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "name": self.name,
            "ray_url": self.ray_url,
            "entrypoint": self.entrypoint,
            "status": self.status,
            "start_time": self.start_time,
            "updated_time": self.updated_time,
            "elapsed_s": round(self.elapsed_s, 2),
            "metadata": self.metadata,
            "error": self.error,
            "terminal": self.terminal,
            "last_event": self.last_event,
            "log_tail": self.log_tail[-20:],
        }


class RayJobSupervisor:
    """Manages the full lifecycle of one or more Ray cluster jobs from inside the RCM runtime."""

    def __init__(
        self,
        ray_url: str = DEFAULT_RAY_URL,
        client: Optional[Any] = None,
        timing_engine: Optional[JointTimingLogEngine] = None,
        trap: Optional[DiagnosticTrap] = None,
        steering: Optional[SteeringReceiver] = None,
        job_file: pathlib.Path = RAY_JOB_FILE,
        jobs_file: pathlib.Path = RAY_JOBS_FILE,
    ):
        self.ray_url = ray_url
        self._client = client
        self.timing_engine = timing_engine
        self.trap = trap or DiagnosticTrap()
        self.steering = steering
        self.job_file = pathlib.Path(job_file)
        self.jobs_file = pathlib.Path(jobs_file)
        self.job_file.parent.mkdir(parents=True, exist_ok=True)
        self.jobs_file.parent.mkdir(parents=True, exist_ok=True)

        self.jobs: Dict[str, RayJobInfo] = {}
        self.active_job_id: Optional[str] = None
        self._log_offsets: Dict[str, int] = {}
        self._full_outputs: Dict[str, str] = {}

    @property
    def active_job(self) -> Optional[RayJobInfo]:
        """Returns the currently active or most recently registered RayJobInfo."""
        if self.active_job_id and self.active_job_id in self.jobs:
            return self.jobs[self.active_job_id]
        if self.jobs:
            return list(self.jobs.values())[-1]
        return None

    @active_job.setter
    def active_job(self, job: Optional[RayJobInfo]) -> None:
        if job is not None:
            self.jobs[job.submission_id] = job
            self.active_job_id = job.submission_id
        else:
            self.active_job_id = None

    def get_client(self) -> Any:
        """Retrieves or lazy-initializes the Ray JobSubmissionClient."""
        if self._client is None:
            try:
                from ray.job_submission import JobSubmissionClient

                self._client = JobSubmissionClient(self.ray_url)
            except ImportError as e:
                raise RuntimeError(
                    f"Ray JobSubmissionClient unavailable: {e}. Ensure 'ray[default]' is installed."
                ) from e
        return self._client

    def _persist_job_state(self) -> None:
        """Atomically saves all tracked jobs to ray_jobs.json and ray_job.json, and updates live_status.json."""
        now = time.time()
        for j in self.jobs.values():
            if not j.terminal:
                j.updated_time = now
                j.elapsed_s = now - j.start_time

        all_data = {
            "active_job_id": self.active_job_id,
            "total_jobs": len(self.jobs),
            "jobs": {k: v.to_dict() for k, v in self.jobs.items()},
        }

        # 1. Save all jobs to ray_jobs.json
        try:
            temp_jobs_file = self.jobs_file.with_suffix(".tmp")
            with open(temp_jobs_file, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2)
            temp_jobs_file.replace(self.jobs_file)
        except Exception:
            pass

        # 2. Save active/latest job to ray_job.json (for single-job backwards compatibility)
        if self.active_job:
            try:
                temp_file = self.job_file.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(self.active_job.to_dict(), f, indent=2)
                temp_file.replace(self.job_file)
            except Exception:
                pass

        # 3. Update live_status.json if present
        if LIVE_STATUS_FILE.exists():
            try:
                with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                    live_data = json.load(f)
                if self.active_job:
                    live_data["ray_job"] = self.active_job.to_dict()
                live_data["ray_jobs"] = [j.to_dict() for j in self.jobs.values()]
                temp_live = LIVE_STATUS_FILE.with_suffix(".tmp")
                with open(temp_live, "w", encoding="utf-8") as f:
                    json.dump(live_data, f, indent=2)
                temp_live.replace(LIVE_STATUS_FILE)
            except Exception:
                pass

    def submit_job(
        self,
        entrypoint: str,
        runtime_env: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> str:
        """Submits a new job to the Ray cluster and begins RCM supervision."""
        client = self.get_client()

        job_label = name or f"job_{len(self.jobs) + 1}"
        if self.timing_engine:
            self.timing_engine.log(
                f"Submitting Ray cluster job [{job_label}] to {self.ray_url}: {entrypoint}",
                level="INFO",
            )

        submission_id = client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env or {},
            metadata=metadata or {},
        )

        now = time.time()
        job_info = RayJobInfo(
            submission_id=submission_id,
            name=job_label,
            ray_url=self.ray_url,
            entrypoint=entrypoint,
            status="PENDING",
            start_time=now,
            updated_time=now,
            metadata=metadata or {},
        )
        self.jobs[submission_id] = job_info
        self.active_job_id = submission_id
        self._log_offsets[submission_id] = 0
        self._full_outputs[submission_id] = ""
        self._persist_job_state()

        if self.timing_engine:
            self.timing_engine.log(
                f"Ray Job [{job_label}] submitted successfully. ID: {submission_id}",
                level="INFO",
            )
            self.timing_engine.touch()

        print(
            f"[OMP_EVENT: CLOUD_JOB_SUBMITTED submission_id={submission_id} name=\"{job_label}\" ray_url=\"{self.ray_url}\"]",
            flush=True,
        )
        return submission_id

    def attach_job(
        self,
        submission_id: str,
        entrypoint: str = "attached_job",
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> RayJobInfo:
        """Attaches supervision to an already-running Ray job submission ID."""
        now = time.time()
        job_label = name or f"job_{len(self.jobs) + 1}"
        job_info = RayJobInfo(
            submission_id=submission_id,
            name=job_label,
            ray_url=self.ray_url,
            entrypoint=entrypoint,
            status="RUNNING",
            start_time=now,
            updated_time=now,
            metadata=metadata or {},
        )
        self.jobs[submission_id] = job_info
        self.active_job_id = submission_id
        self._log_offsets[submission_id] = 0
        self._full_outputs[submission_id] = ""
        self._persist_job_state()
        return job_info

    def get_job(self, submission_id: Optional[str] = None) -> Optional[RayJobInfo]:
        """Retrieves RayJobInfo for a given submission ID or the active job."""
        if submission_id:
            return self.jobs.get(submission_id)
        return self.active_job

    def list_jobs(self, active_only: bool = False) -> List[RayJobInfo]:
        """Returns all tracked Ray jobs or only non-terminal active ones."""
        if active_only:
            return [j for j in self.jobs.values() if not j.terminal]
        return list(self.jobs.values())

    def poll_job_status(self, submission_id: Optional[str] = None) -> Tuple[str, bool]:
        """Queries current status from Ray for a job and returns (status_str, is_terminal)."""
        target = self.get_job(submission_id)
        if not target:
            return "UNKNOWN", True

        client = self.get_client()
        raw_status = client.get_job_status(target.submission_id)

        if hasattr(raw_status, "name"):
            status_str = raw_status.name
            is_terminal = getattr(raw_status, "is_terminal", lambda: status_str in ("SUCCEEDED", "FAILED", "STOPPED"))()
        else:
            status_str = str(raw_status).upper()
            is_terminal = status_str in ("SUCCEEDED", "FAILED", "STOPPED")

        target.status = status_str
        target.terminal = is_terminal
        self._persist_job_state()
        return status_str, is_terminal

    def poll_job_logs(
        self,
        submission_id: Optional[str] = None,
        stdout_stream: bool = True,
        stage_name: str = "execute_policy",
        tag_prefix: bool = False,
    ) -> str:
        """Pulls incremental logs from Ray for a given job, updates timing, and streams lines."""
        target = self.get_job(submission_id)
        if not target:
            return ""

        sub_id = target.submission_id
        offset = self._log_offsets.get(sub_id, 0)
        client = self.get_client()

        try:
            logs = client.get_job_logs(sub_id)
        except Exception as e:
            if self.timing_engine:
                self.timing_engine.log(f"Notice fetching Ray logs for {sub_id}: {e}", level="WARN")
            return ""

        if len(logs) > offset:
            new_text = logs[offset:]
            self._full_outputs[sub_id] = logs
            self._log_offsets[sub_id] = len(logs)

            prefix = f"[{target.name or sub_id[:8]}] " if tag_prefix else ""
            if stdout_stream:
                if tag_prefix:
                    for l in new_text.splitlines(keepends=True):
                        sys.stdout.write(f"{prefix}{l}")
                else:
                    sys.stdout.write(new_text)
                sys.stdout.flush()

            for line in new_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue

                target.log_tail.append(stripped)
                if len(target.log_tail) > 100:
                    target.log_tail.pop(0)

                if "[OMP_EVENT:" in stripped:
                    target.last_event = stripped
                    print(f"[FORWARDED_EVENT] {prefix}{stripped}", flush=True)

                if self.timing_engine:
                    self.timing_engine.log(f"{prefix}{stripped}", stage_name=stage_name)
                    self.timing_engine.touch()

            self._persist_job_state()
            return new_text
        return ""

    def stop_job(self, submission_id: Optional[str] = None) -> bool:
        """Explicitly cancels a specific Ray job (or active job) on the cluster."""
        target = self.get_job(submission_id)
        if not target:
            return False

        sub_id = target.submission_id
        if self.timing_engine:
            self.timing_engine.log(f"Stopping Ray cluster job {sub_id}...", level="WARN")

        success = False
        try:
            client = self.get_client()
            client.stop_job(sub_id)
            success = True
        except Exception as e:
            if self.timing_engine:
                self.timing_engine.log(f"Notice stopping Ray job {sub_id}: {e}", level="WARN")

        target.status = "STOPPED"
        target.terminal = True
        self._persist_job_state()
        print(f"[OMP_EVENT: CLOUD_JOB_STOPPED submission_id={sub_id}]", flush=True)
        return success

    def stop_all_jobs(self) -> int:
        """Cancels all active non-terminal Ray jobs registered with this supervisor."""
        active = self.list_jobs(active_only=True)
        count = 0
        for job in active:
            if self.stop_job(job.submission_id):
                count += 1
        return count

    def supervise_jobs(
        self,
        submission_ids: Optional[List[str]] = None,
        poll_interval_s: float = 1.5,
        timeout_s: Optional[float] = None,
        stdout_stream: bool = True,
        stage_name: str = "execute_policy",
        state_store: Optional[StateStore] = None,
    ) -> Dict[str, Any]:
        """Monitors one or multiple concurrent Ray jobs until all complete or an error occurs."""
        target_ids = submission_ids or list(self.jobs.keys())
        if not target_ids:
            raise RuntimeError("No Ray jobs to supervise.")

        multi = len(target_ids) > 1
        steering = self.steering or SteeringReceiver()
        t_start = time.time()

        try:
            while True:
                all_terminal = True
                any_failed = False

                for sub_id in target_ids:
                    try:
                        status_str, is_terminal = self.poll_job_status(sub_id)
                        self.poll_job_logs(
                            sub_id,
                            stdout_stream=stdout_stream,
                            stage_name=stage_name,
                            tag_prefix=multi,
                        )
                    except Exception as e:
                        if self.timing_engine:
                            self.timing_engine.log(f"Transient error polling Ray job {sub_id}: {e}", level="WARN")
                        is_terminal = False

                    if not is_terminal:
                        all_terminal = False
                    elif status_str != "SUCCEEDED":
                        any_failed = True

                # Check for RCM Normality Traps via timing engine
                if self.timing_engine:
                    norm_status, reason = self.timing_engine.check_normality()
                    if norm_status in ("CRITICAL_PATTERN", "REPETITIVE_STALL", "SILENCE_TIMEOUT") and reason:
                        clocks = self.timing_engine.get_clocks()
                        trap_file = self.trap.write_snapshot(
                            stage_name=stage_name,
                            reason=reason,
                            clocks=clocks,
                            log_tail=self.timing_engine.get_tail_logs(25),
                            state_store=state_store or StateStore(),
                            allowed_actions=["RESUME", "ABORT", "MUTATE_CONFIG", "RETRY"],
                        )
                        print(
                            f"\n[OMP_EVENT: TRAP_PAUSED stage={stage_name} "
                            f"elapsed_s={clocks['t_stage_s']} expected_s={clocks['t_expected_s']} "
                            f"reason=\"{reason}\" trap_file=\"{trap_file}\"]",
                            flush=True,
                        )

                        action = None
                        while action is None:
                            action = steering.poll_action(timeout_s=0.5)

                        op = action.get("op", "RESUME").upper()
                        print(f"[OMP_EVENT: TRAP_RESOLVED stage={stage_name} action={op}]", flush=True)

                        if op == "ABORT":
                            self.stop_all_jobs()
                            raise RuntimeError(f"Remote TPU job(s) aborted via RCM steering directive: {reason}")
                        elif op == "MUTATE_CONFIG":
                            if state_store is not None and "params" in action:
                                state_store.mutate_params(action["params"])
                            self.timing_engine.touch()
                            self.timing_engine.extend_stage(120.0)
                        elif op == "RETRY":
                            self.timing_engine.touch()
                            self.timing_engine.extend_stage(180.0)

                # Check timeout
                if timeout_s is not None and (time.time() - t_start) > timeout_s:
                    self.stop_all_jobs()
                    raise TimeoutError(f"Ray jobs timed out after {timeout_s}s: {target_ids}")

                if all_terminal:
                    break

                time.sleep(poll_interval_s)

            # Final log flush for each job
            for sub_id in target_ids:
                self.poll_job_logs(sub_id, stdout_stream=stdout_stream, stage_name=stage_name, tag_prefix=multi)
                self.poll_job_status(sub_id)

            failed_jobs = [j for j in [self.jobs[s] for s in target_ids] if j.status != "SUCCEEDED"]
            if failed_jobs:
                err_summary = ", ".join(f"{j.submission_id} [{j.status}]" for j in failed_jobs)
                print(f"[OMP_EVENT: CLOUD_JOBS_FAILED failed=\"{err_summary}\"]", flush=True)
                raise RuntimeError(f"One or more Ray cluster jobs failed: {err_summary}")

            elapsed_total = round(time.time() - t_start, 2)
            print(
                f"[OMP_EVENT: CLOUD_JOBS_SUCCEEDED count={len(target_ids)} elapsed_s={elapsed_total}]",
                flush=True,
            )

            # Extract measurements
            measurements = {}
            for sub_id in target_ids:
                out = self._full_outputs.get(sub_id, "")
                meas = self.extract_json_measurements(out)
                if meas:
                    measurements[sub_id] = meas

            # Backwards compatibility: primary job output and measurements
            primary_id = target_ids[-1]
            primary_out = self._full_outputs.get(primary_id, "")
            primary_meas = measurements.get(primary_id, {})

            return {
                "status": "SUCCEEDED",
                "submission_ids": target_ids,
                "submission_id": primary_id,
                "full_output": primary_out,
                "measurements": primary_meas,
                "all_measurements": measurements,
                "all_outputs": self._full_outputs,
                "jobs": {s: self.jobs[s].to_dict() for s in target_ids},
            }
        finally:
            self._persist_job_state()

    def supervise_until_completion(
        self,
        submission_id: Optional[str] = None,
        poll_interval_s: float = 1.5,
        timeout_s: Optional[float] = None,
        stdout_stream: bool = True,
        stage_name: str = "execute_policy",
        state_store: Optional[StateStore] = None,
    ) -> Dict[str, Any]:
        """Supervises a specific job (or active jobs) until completion."""
        target_ids = [submission_id] if submission_id else (list(self.jobs.keys()) or [self.active_job_id])
        target_ids = [t for t in target_ids if t]
        return self.supervise_jobs(
            submission_ids=target_ids,
            poll_interval_s=poll_interval_s,
            timeout_s=timeout_s,
            stdout_stream=stdout_stream,
            stage_name=stage_name,
            state_store=state_store,
        )

    @staticmethod
    def extract_json_measurements(output_text: str) -> Dict[str, Any]:
        """Extracts EMPIRICAL TPU HARDWARE MEASUREMENTS JSON payload from job logs."""
        patterns = [
            r"EMPIRICAL TPU HARDWARE MEASUREMENTS \(JSON OUTPUT\)\n=+\n(\{.*?\})\n\n\[OMP_EVENT",
            r"EMPIRICAL TPU HARDWARE MEASUREMENTS \(JSON OUTPUT\)\n=+\n(\{.*?\})\n=+",
            r"TOPOLOGY_RESULT_JSON=(\{.*?\})$",
        ]
        for pat in patterns:
            match = re.search(pat, output_text, re.DOTALL | re.MULTILINE)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
        return {}


def read_active_ray_job(job_file: pathlib.Path = RAY_JOB_FILE) -> Optional[Dict[str, Any]]:
    """Reads the active/latest Ray job state file if present and valid."""
    path = pathlib.Path(job_file)
    if not path.exists():
        if LIVE_STATUS_FILE.exists():
            try:
                with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                    st = json.load(f)
                return st.get("ray_job")
            except Exception:
                pass
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_all_ray_jobs(jobs_file: pathlib.Path = RAY_JOBS_FILE) -> Dict[str, Any]:
    """Reads all tracked Ray jobs from ray_jobs.json or live_status.json."""
    path = pathlib.Path(jobs_file)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback to single ray_job.json
    single = read_active_ray_job()
    if single:
        return {
            "active_job_id": single.get("submission_id"),
            "total_jobs": 1,
            "jobs": {single.get("submission_id"): single},
        }

    # Fallback to live_status.json
    if LIVE_STATUS_FILE.exists():
        try:
            with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                st = json.load(f)
            jobs_list = st.get("ray_jobs", [])
            if jobs_list:
                jobs_dict = {j["submission_id"]: j for j in jobs_list if "submission_id" in j}
                return {
                    "active_job_id": st.get("ray_job", {}).get("submission_id"),
                    "total_jobs": len(jobs_dict),
                    "jobs": jobs_dict,
                }
        except Exception:
            pass

    return {"active_job_id": None, "total_jobs": 0, "jobs": {}}


def format_ray_jobs_table(jobs_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
    """Renders an executive summary table/roster of all tracked Ray cluster jobs."""
    if isinstance(jobs_data, list):
        jobs_map = {j.get("submission_id", f"job_{i}"): j for i, j in enumerate(jobs_data)}
        active_id = None
    else:
        jobs_map = jobs_data.get("jobs", {})
        active_id = jobs_data.get("active_job_id")

    if not jobs_map:
        return "No active or recorded Ray cluster jobs found."

    total = len(jobs_map)
    active_count = sum(1 for j in jobs_map.values() if not j.get("terminal", False))

    lines = []
    lines.append(f"=== RCM Ray Cluster Jobs ({active_count} Active, {total - active_count} Settled) ===")
    header = f"{'':2} {'Submission ID':28} {'Status':11} {'Elapsed':8} {'Name/Topology':20} {'Last Event / Error'}"
    lines.append(header)
    lines.append("-" * len(header))

    for sub_id, j in jobs_map.items():
        is_active = (sub_id == active_id)
        marker = "* " if is_active else "  "
        status = j.get("status", "UNKNOWN")
        elapsed = f"{j.get('elapsed_s', 0.0):.1f}s"
        name = j.get("name") or j.get("metadata", {}).get("topology") or j.get("metadata", {}).get("policy") or "cluster_job"
        name = str(name)[:19]

        detail = j.get("last_event") or j.get("error") or ""
        if len(detail) > 40:
            detail = detail[:37] + "..."

        row = f"{marker}{sub_id:28} {status:11} {elapsed:8} {name:20} {detail}"
        lines.append(row)

    return "\n".join(lines)


def format_ray_job_summary(job_data: Dict[str, Any]) -> str:
    """Renders a concise, human-readable summary of a Ray cluster job."""
    lines = []
    sub_id = job_data.get("submission_id", "UNKNOWN")
    status = job_data.get("status", "UNKNOWN")
    name = job_data.get("name")
    ray_url = job_data.get("ray_url", DEFAULT_RAY_URL)
    elapsed = job_data.get("elapsed_s", 0.0)
    entrypoint = job_data.get("entrypoint", "N/A")
    last_event = job_data.get("last_event")

    header = f"Ray Cluster Job: {sub_id} [{status}]"
    if name:
        header = f"Ray Cluster Job: {name} ({sub_id}) [{status}]"
    lines.append(header)
    lines.append(f"  Cluster URL:  {ray_url}")
    lines.append(f"  Elapsed Time: {elapsed}s")
    lines.append(f"  Entrypoint:   {entrypoint}")
    if last_event:
        lines.append(f"  Last Event:   {last_event}")
    if job_data.get("error"):
        lines.append(f"  Error:        {job_data.get('error')}")

    tail = job_data.get("log_tail", [])
    if tail:
        lines.append(f"  Recent Logs ({len(tail)} lines):")
        for log_line in tail[-5:]:
            lines.append(f"    | {log_line}")

    return "\n".join(lines)
