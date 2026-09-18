"""
OMP RCM Log Inspection Utility (Zero-Token-Waste Diagnostics)
=============================================================
Provides fine-grained, token-efficient log filtering and status inspection
for Oh My Pi (OMP) agents and human engineers.

Usage:
  python3 -m omp_rcm.inspect
  python3 -m omp_rcm.inspect --tail 30
  python3 -m omp_rcm.inspect --stage policy_a_greedy_preemption_stress
  python3 -m omp_rcm.inspect --level WARN,ERROR
  python3 -m omp_rcm.inspect --grep "preempt|exhaust"
  python3 -m omp_rcm.inspect --summary
"""

import os
import sys
import json
import time
import argparse
import pathlib
import re
from typing import List, Dict, Any, Optional

BASE_RCM_DIR = pathlib.Path("/tmp/omp_rcm")
TELEMETRY_FILE = BASE_RCM_DIR / "telemetry.jsonl"
LIVE_STATUS_FILE = BASE_RCM_DIR / "live_status.json"
LIVE_LOGS_FILE = BASE_RCM_DIR / "live_logs.txt"
LATEST_TRAP_FILE = BASE_RCM_DIR / "traps" / "latest.json"
RAY_JOB_FILE = BASE_RCM_DIR / "ray_job.json"
RAY_JOBS_FILE = BASE_RCM_DIR / "ray_jobs.json"

def get_live_status() -> Optional[Dict[str, Any]]:
    """Returns the atomic live status dictionary if available."""
    if LIVE_STATUS_FILE.exists():
        try:
            with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def get_latest_trap() -> Optional[Dict[str, Any]]:
    """Returns the latest diagnostic trap snapshot if available."""
    if LATEST_TRAP_FILE.exists():
        try:
            with open(LATEST_TRAP_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None
def get_active_ray_job() -> Optional[Dict[str, Any]]:
    """Returns active Ray cluster job data from ray_job.json or live_status.json."""
    if RAY_JOB_FILE.exists():
        try:
            with open(RAY_JOB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    status = get_live_status()
    if status and "ray_job" in status:
        return status.get("ray_job")
    return None
def get_all_ray_jobs() -> Dict[str, Any]:
    """Returns all tracked Ray cluster jobs from ray_jobs.json or live_status.json."""
    if RAY_JOBS_FILE.exists():
        try:
            with open(RAY_JOBS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    single = get_active_ray_job()
    if single:
        return {"active_job_id": single.get("submission_id"), "total_jobs": 1, "jobs": {single.get("submission_id"): single}}
    return {"active_job_id": None, "total_jobs": 0, "jobs": {}}




def inspect_logs(
    telemetry_path: pathlib.Path = TELEMETRY_FILE,
    tail: int = 30,
    stage: Optional[str] = None,
    level: Optional[str] = None,
    grep: Optional[str] = None,
    since_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Parses out-of-band JSONL telemetry and returns filtered structured entries.
    """
    if not telemetry_path.exists():
        return []

    levels_set = set(level.upper().split(",")) if level else None
    grep_pattern = re.compile(grep, re.IGNORECASE) if grep else None
    now = time.time()

    matched: List[Dict[str, Any]] = []
    try:
        with open(telemetry_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    entry = json.loads(line_str)
                except Exception:
                    continue

                if stage and entry.get("stage", "") != stage:
                    continue
                if levels_set and entry.get("level", "INFO") not in levels_set:
                    continue
                if since_s is not None and (now - entry.get("timestamp", now)) > since_s:
                    continue
                if grep_pattern and not grep_pattern.search(entry.get("message", "")):
                    continue

                matched.append(entry)
    except Exception:
        pass

    if tail and len(matched) > tail:
        matched = matched[-tail:]
    return matched


def format_inspected_logs(entries: List[Dict[str, Any]], relative_to: Optional[float] = None) -> str:
    """Formats log entries into token-compact single-line strings with relative offsets."""
    if not entries:
        return "<no matching logs found>"

    ref_t = relative_to or time.time()
    lines = []
    for e in entries:
        rel = e.get("timestamp", ref_t) - ref_t
        st = e.get("stage", "GLOBAL")
        lvl = e.get("level", "INFO")
        msg = e.get("message", "")
        lines.append(f"[{rel:+.2f}s] [{st}] [{lvl}] {msg}")
    return "\n".join(lines)


def format_summary() -> str:
    """Renders a concise, token-efficient executive summary of RCM state."""
    status = get_live_status()
    trap = get_latest_trap()

    out = []
    out.append("=== OMP RCM Live Execution Summary ===")
    if status:
        clocks = status.get("clocks", {})
        out.append(f"Current Stage: {status.get('current_stage', 'UNKNOWN')}")
        out.append(
            f"Clocks: Total={clocks.get('t_total_s', 0)}s | "
            f"Stage={clocks.get('t_stage_s', 0)}s / Exp={clocks.get('t_expected_s', 0)}s | "
            f"Silence(Log)={clocks.get('t_since_last_log_s', 0)}s | "
            f"Silence(Prog)={clocks.get('t_since_last_progress_s', 0)}s"
        )
        out.append(f"Repetitive Count: {status.get('repetition_count', 0)} (Allowed: {status.get('allow_repetitive', False)})")
    else:
        out.append("Live status file not active.")

    if trap:
        out.append(f"\n[Latest Trap] Stage={trap.get('stage')} Reason=\"{trap.get('reason')}\"")
        exc = trap.get("exception")
        if exc:
            out.append(f"Trap Exception: {exc.get('type')}: {exc.get('message')}")
        out.append(f"Allowed Actions: {', '.join(trap.get('allowed_actions', []))}")
    all_jobs_data = get_all_ray_jobs()
    jobs_map = all_jobs_data.get("jobs", {})
    if len(jobs_map) > 1:
        from .ray_job import format_ray_jobs_table
        out.append("\n" + format_ray_jobs_table(all_jobs_data))
    elif jobs_map:
        ray_job = list(jobs_map.values())[0]
        out.append(f"\n[Ray Cluster Job] {ray_job.get('name') or ray_job.get('submission_id')} [{ray_job.get('status')}]")
        out.append(f"Ray URL: {ray_job.get('ray_url')} | Elapsed: {ray_job.get('elapsed_s', 0)}s")
        if ray_job.get("entrypoint"):
            out.append(f"Entrypoint: {ray_job.get('entrypoint')}")
        if ray_job.get("last_event"):
            out.append(f"Last Event: {ray_job.get('last_event')}")
        if ray_job.get("error"):
            out.append(f"Error: {ray_job.get('error')}")
    else:
        ray_job = get_active_ray_job()
        if ray_job:
            out.append(f"\n[Ray Cluster Job] {ray_job.get('submission_id')} [{ray_job.get('status')}]")
            out.append(f"Ray URL: {ray_job.get('ray_url')} | Elapsed: {ray_job.get('elapsed_s', 0)}s")
            if ray_job.get("entrypoint"):
                out.append(f"Entrypoint: {ray_job.get('entrypoint')}")
            if ray_job.get("last_event"):
                out.append(f"Last Event: {ray_job.get('last_event')}")
            if ray_job.get("error"):
                out.append(f"Error: {ray_job.get('error')}")

    out.append("\n=== Recent Telemetry Tail (Last 10 Lines) ===")
    tail = inspect_logs(tail=10)
    out.append(format_inspected_logs(tail))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="OMP RCM Log Inspection Utility")
    parser.add_argument("--tail", "-n", type=int, default=30, help="Number of recent lines to display")
    parser.add_argument("--stage", "-s", type=str, default=None, help="Filter by stage name")
    parser.add_argument("--level", "-l", type=str, default=None, help="Filter by log level (e.g. WARN,ERROR)")
    parser.add_argument("--grep", "-g", type=str, default=None, help="Regex pattern filter")
    parser.add_argument("--since", type=float, default=None, help="Only show logs within last N seconds")
    parser.add_argument("--summary", action="store_true", help="Print executive state summary")
    parser.add_argument("--json", action="store_true", help="Output raw JSON array")

    args = parser.parse_args()

    if args.summary:
        print(format_summary())
        return

    logs = inspect_logs(
        tail=args.tail,
        stage=args.stage,
        level=args.level,
        grep=args.grep,
        since_s=args.since,
    )

    if args.json:
        print(json.dumps(logs, indent=2))
    else:
        print(format_inspected_logs(logs))


if __name__ == "__main__":
    main()
