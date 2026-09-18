"""
OMP Reverse Control Mechanism (RCM) CLI Wrapper & Process Supervisor.
Provides synchronous execution, zero-token diagnostic trap interception,
and in-flight live code / parameter steering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from .core import (
    BASE_RCM_DIR,
    CONTROL_ACTION_FILE,
    CONTROL_SOCK_FILE,
    LATEST_TRAP_FILE,
    LIVE_STATUS_FILE,
    TELEMETRY_FILE,
    TRAPS_DIR,
    RAY_JOB_FILE,
    EVENT_MARKERS,
    is_event_line,
    ensure_rcm_dir,
    get_source_window,
    is_first_party_file,
    get_repo_root,
)
from .ray_job import (
    read_active_ray_job,
    read_all_ray_jobs,
    format_ray_job_summary,
    format_ray_jobs_table,
)
from .inspect import (
    format_inspected_logs,
    format_summary,
    get_live_status,
    inspect_logs,
)

DAEMON_PID_FILE = BASE_RCM_DIR / "daemon.pid"
DAEMON_LOG_FILE = BASE_RCM_DIR / "daemon_raw.log"


def is_pid_alive(pid: int) -> bool:
    """Checks if a process ID is currently running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_pid_zombie(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        proc_status = pathlib.Path(f"/proc/{pid}/status")
        if proc_status.exists():
            for line in proc_status.read_text(encoding="utf-8").splitlines():
                if line.startswith("State:"):
                    return "Z" in line
    except Exception:
        pass
    return False


def get_active_daemon_pid() -> Optional[int]:
    """Retrieves the active daemon PID from DAEMON_PID_FILE if running."""
    if not DAEMON_PID_FILE.exists():
        return None
    try:
        pid = int(DAEMON_PID_FILE.read_text(encoding="utf-8").strip())
        if is_pid_alive(pid):
            return pid
        else:
            DAEMON_PID_FILE.unlink(missing_ok=True)
            return None
    except Exception:
        return None


def read_latest_trap(timeout_s: float = 1.0) -> Optional[Dict[str, Any]]:
    """Polls briefly for latest.json to be written by the trapped worker."""
    start_t = time.time()
    while (time.time() - start_t) < timeout_s:
        if LATEST_TRAP_FILE.exists():
            try:
                with open(LATEST_TRAP_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        time.sleep(0.05)
    return None


def format_trap_system_directive(trap_data: Dict[str, Any], daemon_pid: int) -> str:
    """
    Formats the authoritative <system-directive> banner presented to the LLM agent
    when a diagnostic trap pauses the background worker.
    """
    stage_name = trap_data.get("stage", "unknown_stage")
    order_idx = trap_data.get("order_idx", "?")
    reason = trap_data.get("reason", "Anomaly / Exception")
    exc_raw = trap_data.get("exception")
    if isinstance(exc_raw, dict):
        exc_type = exc_raw.get("type", "DiagnosticTrap")
        exc_msg = exc_raw.get("message", reason)
        tb_str = exc_raw.get("traceback") or ""
    elif isinstance(exc_raw, str):
        exc_type = exc_raw.split(":", 1)[0].strip() if ":" in exc_raw else "DiagnosticTrap"
        exc_msg = exc_raw
        tb_str = trap_data.get("traceback") or exc_raw
    else:
        exc_type = "DiagnosticTrap"
        exc_msg = reason
        tb_str = trap_data.get("traceback") or ""
    state_keys = trap_data.get("state_keys", [])

    # Blame frame from snapshot or resolved from traceback (B1)
    blame_frame = trap_data.get("blame_frame") or {}
    parent_frame = trap_data.get("parent_frame") or {}
    first_party_tb = trap_data.get("first_party_traceback") or []
    repo_root = get_repo_root()

    loc_str = "unknown"
    file_candidate = None
    line_candidate = None
    func_candidate = "?"

    if blame_frame.get("file"):
        file_candidate = blame_frame.get("file")
        line_candidate = blame_frame.get("line")
        func_candidate = blame_frame.get("function", "?")
        loc_str = f"{file_candidate}:{line_candidate}"
        if func_candidate and func_candidate != "?":
            loc_str += f" in {func_candidate}"
    elif tb_str:
        cand_frames = []
        for line in tb_str.splitlines():
            line_str = line.strip()
            if line_str.startswith('File "') and '", line ' in line_str:
                try:
                    parts = line_str.split('File "')[1].split('", line ')
                    f_path = parts[0]
                    line_no = parts[1].split(",")[0].strip()
                    fn = parts[1].split("in ")[1].strip() if "in " in parts[1] else "?"
                    cand_frames.append((f_path, line_no, fn))
                except Exception:
                    pass
        selected = None
        for f_p, l_no, fn in reversed(cand_frames):
            if is_first_party_file(f_p, repo_root):
                selected = (f_p, l_no, fn)
                break
        if not selected and cand_frames:
            selected = cand_frames[-1]
        if selected:
            file_candidate, line_candidate, func_candidate = selected
            loc_str = f"{file_candidate}:{line_candidate}"
            if func_candidate and func_candidate != "?":
                loc_str += f" in {func_candidate}"

    # Source code window (B2)
    source_window = blame_frame.get("source_window") or trap_data.get("source_window")
    source_formatted = None
    if source_window and isinstance(source_window, dict) and source_window.get("formatted"):
        source_formatted = source_window["formatted"]
    elif file_candidate and line_candidate:
        try:
            l_int = int(line_candidate)
            sw = get_source_window(file_candidate, l_int, window=15)
            if sw.get("formatted"):
                source_formatted = sw["formatted"]
        except Exception:
            pass

    # Frame locals (B3)
    blame_locals = blame_frame.get("locals", {})
    parent_locals = parent_frame.get("locals", {})

    log_tail = trap_data.get("log_tail", [])
    log_tail_formatted = "\n  ".join(log_tail[-4:]) if log_tail else "  (none)"

    rec_count = trap_data.get("recurrence_count", 1)
    banner = [
        "<system-directive>",
        "========================= RCM DIAGNOSTIC TRAP =========================",
        f"Stage:      {stage_name} (Order: {order_idx})",
        f"Reason:     {reason}",
        f"Exception:  {exc_type}: {exc_msg}",
        f"Location:   {loc_str}",
        f"Daemon:     PID {daemon_pid} PAUSED in background (In-memory state preserved)",
        f"State Keys: {list(state_keys)}",
    ]
    once_keys = trap_data.get("once_keys", [])
    if once_keys:
        pres_text = ", ".join(f"`{k}`" for k in once_keys)
        banner.append(f"Preserved Sub-steps (will NOT re-run on resume): {pres_text}")

    if rec_count > 1:
        banner.append(f"Recurrence: Occurrence #{rec_count}")
    if source_formatted:
        banner.append("")
        banner.append(f"Source Code Window ({file_candidate}):")
        banner.append(source_formatted)

    if blame_locals:
        banner.append("")
        banner.append(f"Blame Frame Locals ({func_candidate}):")
        for k, v in list(blame_locals.items())[:25]:
            banner.append(f"  {k} = {v}")

    if parent_locals:
        p_func = parent_frame.get("function", "parent")
        banner.append("")
        banner.append(f"Parent Frame Locals ({p_func}):")
        for k, v in list(parent_locals.items())[:20]:
            banner.append(f"  {k} = {v}")

    if first_party_tb:
        banner.append("")
        banner.append("Condensed Traceback (first-party):")
        for entry in first_party_tb:
            f_rel = entry.get("file", "")
            try:
                f_rel = str(pathlib.Path(f_rel).relative_to(repo_root))
            except Exception:
                pass
            banner.append(f"  File \"{f_rel}\", line {entry.get('line')}, in {entry.get('function')}")
            if entry.get("code"):
                banner.append(f"    {entry.get('code')}")

    banner.append("")
    banner.append("Recent Logs:")
    banner.append(f"  {log_tail_formatted}")
    banner.append("")
    if rec_count >= 3:
        banner.append("MANDATORY CONTEXT-REWIND PROTOCOL (REPEATED FAILURE):")
        banner.append(f"*** REPEATED FAILURE DETECTED (Occurrence #{rec_count}) ***")
        banner.append("Stop patching: this fault has occurred multiple times without resolution.")
        banner.append("Inspect state store, check parameters, and consider MUTATE_CONFIG or SKIP_STAGE.")
        banner.append("1. MUST invoke checkpoint before exploratory inspection or modifying state:")
        banner.append(f'   write to xd://checkpoint with {{"goal": "Triage repeat fault in {stage_name}"}}')
        banner.append("2. Inspect state store or modify workload parameters via steer.")
        banner.append("3. MUST invoke rewind after making changes to erase exploration tokens:")
        banner.append(f'   write to xd://rewind with {{"report": "Triaged repeat fault in {stage_name}"}}')
        banner.append("4. Resume execution via:")
        banner.append("   NEXT: rcm_exec resume --skip")
        banner.append("   (or `rcm_exec steer '{\"op\": \"MUTATE_CONFIG\", \"params\": {...}}'`)")
        banner.append("   (or `rcm_exec resume --abort`)")
    else:
        banner.append("TRIAGE & CONTEXT-PRESERVATION PROTOCOL:")
        banner.append("- Direct Fix (3 tool calls): If the blame frame & source window above are sufficient,")
        banner.append("  apply the edit directly and run:")
        banner.append("    NEXT: rcm_exec resume")
        banner.append("- Exploratory Debugging: If you must read additional files or query logs,")
        banner.append("  MUST wrap exploration in checkpoint / rewind to prevent logs and exploration")
        banner.append("  tokens from flowing into and wasting the context window:")
        banner.append(f'    1. write to xd://checkpoint with {{"goal": "Triage {stage_name}: {exc_msg[:60]}"}}')
        banner.append("    2. Read files / logs and apply surgical fix via Edit tool")
        banner.append(f'    3. write to xd://rewind with {{"report": "Fixed {stage_name} {loc_str}"}}')
        banner.append("    4. NEXT: rcm_exec resume")
        banner.append("       (or `rcm_exec steer '{\"op\": \"RETRY\"}'`)")
    banner.append("</system-directive>")
    return "\n".join(banner)

def stream_daemon_output(
    log_file: pathlib.Path,
    daemon_proc: Optional[subprocess.Popen],
    daemon_pid: int,
    start_offset: int = 0,
    stream_all: bool = False,
    quiet: bool = False,
    target_stage: Optional[str] = None,
    timeout_s: Optional[float] = None,
    poll_interval_s: float = 0.05,
    max_lines: Optional[int] = None,
    max_bytes: Optional[int] = None,
    heartbeat_interval_s: Optional[float] = None,
) -> int:
    """
    Unified daemon log streaming and supervisor loop.
    A1: Defaults to events-only filter when RCM_AGENT=1 or stdout is not a TTY.
        --stream opts into full firehose.
    A2: Caps output per invocation (~300 lines / 24 KB) with suppression footer.
        Emits a one-line heartbeat every ~5 min (stage + clocks).
    A3: Uses shared EVENT_MARKERS constant and is_event_line.
    F1: Standardized exit codes: 0 complete, 1 failed, 2 trapped, 3 no daemon, 124 timeout.
    """
    if max_lines is None:
        max_lines = int(os.environ.get("RCM_MAX_OUTPUT_LINES", "300"))
    if max_bytes is None:
        max_bytes = int(os.environ.get("RCM_MAX_OUTPUT_BYTES", "24576"))
    if heartbeat_interval_s is None:
        heartbeat_interval_s = float(os.environ.get("RCM_HEARTBEAT_INTERVAL_S", "300.0"))

    is_agent = (os.environ.get("RCM_AGENT") == "1") or (not sys.stdout.isatty())
    filter_events = not stream_all and (is_agent or quiet)

    curr_offset = start_offset
    line_buf = ""
    trap_detected = False
    emitted_lines = 0
    emitted_bytes = 0
    suppressed_lines = 0
    start_time = time.time()
    last_heartbeat_time = time.time()

    def print_suppression_footer() -> None:
        nonlocal suppressed_lines
        if suppressed_lines > 0:
            print(f"\n… {suppressed_lines} lines suppressed — rcm_exec logs --tail 50", flush=True)
            suppressed_lines = 0

    def emit_line(line_to_emit: str) -> None:
        nonlocal emitted_lines, emitted_bytes, suppressed_lines
        if quiet:
            return
        if filter_events and not is_event_line(line_to_emit):
            return

        enc_len = len(line_to_emit.encode("utf-8", errors="replace")) + 1
        if emitted_lines >= max_lines or (emitted_bytes + enc_len) > max_bytes:
            suppressed_lines += 1
        else:
            sys.stdout.write(line_to_emit + "\n")
            sys.stdout.flush()
            emitted_lines += 1
            emitted_bytes += enc_len

    def check_heartbeat() -> None:
        nonlocal last_heartbeat_time
        now = time.time()
        if (now - last_heartbeat_time) >= heartbeat_interval_s:
            last_heartbeat_time = now
            try:
                st = get_live_status()
                st_stage = (st.get("current_stage") if st else None) or "idle"
                clocks = (st.get("clocks") if st else None) or {}
                t_stage = clocks.get("t_stage_s", 0.0)
                t_exp = clocks.get("t_expected_s", 0.0)
                t_tot = clocks.get("t_total_s", 0.0)
                hb_msg = f"[rcm_exec heartbeat] stage={st_stage} elapsed_s={t_stage:.1f}s/{t_exp:.1f}s total_s={t_tot:.1f}s"
                sys.stdout.write(hb_msg + "\n")
                sys.stdout.flush()
            except Exception:
                pass

    while True:
        elapsed = time.time() - start_time
        if timeout_s is not None and elapsed >= timeout_s:
            print_suppression_footer()
            print(
                f"\n[rcm_exec] Wait timed out after {elapsed:.1f}s (timeout={timeout_s}s). "
                f"Daemon PID {daemon_pid} is still running.",
                flush=True,
            )
            return 124

        check_heartbeat()
        alive = is_pid_alive(daemon_pid)

        if LATEST_TRAP_FILE.exists():
            trap_detected = True

        if log_file.exists():
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                f.seek(curr_offset)
                chunk = f.read()
                curr_offset = f.tell()

            if chunk:
                line_buf += chunk
                while "\n" in line_buf:
                    line, line_buf = line_buf.split("\n", 1)
                    emit_line(line)

                    if "[OMP_EVENT: TRAP_PAUSED" in line:
                        trap_detected = True
                    elif target_stage and f"[OMP_EVENT: STAGE_SETTLED stage={target_stage}" in line:
                        print_suppression_footer()
                        if not quiet:
                            print(f"\n[rcm_exec] Target stage '{target_stage}' settled successfully.", flush=True)
                        return 0
                    elif "[OMP_EVENT: RUN_FINISHED status=SUCCESS" in line:
                        print_suppression_footer()
                        if not quiet and not filter_events:
                            print("\n[rcm_exec] Workload finished successfully (SUCCESS).", flush=True)
                        return 0
                    elif (
                        "[OMP_EVENT: RUN_FINISHED status=FAILED" in line
                        or "[OMP_EVENT: RUN_FINISHED status=ABORTED" in line
                    ):
                        print_suppression_footer()
                        if not quiet:
                            print(f"\n[rcm_exec] Workload finished with failure: {line.strip()}", flush=True)
                        return 1

        if trap_detected:
            time.sleep(0.1)  # Brief grace period for latest.json write
            print_suppression_footer()
            trap_data = read_latest_trap() or {}
            print("\n" + format_trap_system_directive(trap_data, daemon_pid), flush=True)
            return 2

        if not alive:
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(curr_offset)
                    rest = f.read()
                    if rest:
                        line_buf += rest
                        while "\n" in line_buf:
                            l, line_buf = line_buf.split("\n", 1)
                            emit_line(l)
                            if "[OMP_EVENT: TRAP_PAUSED" in l:
                                trap_detected = True
                            elif target_stage and f"[OMP_EVENT: STAGE_SETTLED stage={target_stage}" in l:
                                print_suppression_footer()
                                return 0
                            elif "[OMP_EVENT: RUN_FINISHED status=SUCCESS" in l:
                                print_suppression_footer()
                                return 0
                            elif (
                                "[OMP_EVENT: RUN_FINISHED status=FAILED" in l
                                or "[OMP_EVENT: RUN_FINISHED status=ABORTED" in l
                            ):
                                print_suppression_footer()
                                return 1
                        if line_buf:
                            emit_line(line_buf)
                            line_buf = ""

            print_suppression_footer()
            trap_data = read_latest_trap(timeout_s=0.1)
            if trap_data or trap_detected:
                trap_data = trap_data or {}
                print("\n" + format_trap_system_directive(trap_data, daemon_pid), flush=True)
                return 2

            if "[OMP_EVENT: RUN_FINISHED status=SUCCESS" in line_buf:
                return 0
            if daemon_proc and daemon_proc.returncode is not None:
                return daemon_proc.returncode
            return 1

        time.sleep(poll_interval_s)


def cmd_run(args: argparse.Namespace) -> int:
    """Spawns an RCM workload as a supervised background daemon."""
    existing_pid = get_active_daemon_pid()
    if existing_pid:
        if getattr(args, "replace", False):
            print(f"[rcm_exec] Replacing existing worker daemon PID {existing_pid}...", flush=True)
            try:
                os.kill(existing_pid, signal.SIGTERM)
                for _ in range(20):
                    if not is_pid_alive(existing_pid):
                        break
                    time.sleep(0.1)
                if is_pid_alive(existing_pid):
                    os.kill(existing_pid, signal.SIGKILL)
            except OSError:
                pass
            DAEMON_PID_FILE.unlink(missing_ok=True)
        else:
            print(f"[RCM] Active worker daemon PID {existing_pid} is currently running.", file=sys.stderr)
            trap_data = read_latest_trap(timeout_s=0.1)
            if trap_data:
                print(f"[RCM] Worker is currently TRAPPED in stage '{trap_data.get('stage')}'.", file=sys.stderr)
                print("[RCM] Use `rcm_exec resume` to resume,", file=sys.stderr)
                print("[RCM] or run `rcm_exec abort` to terminate the existing worker.", file=sys.stderr)
            else:
                print("[RCM] Worker is currently running. Use `rcm_exec status` or `rcm_exec abort`.", file=sys.stderr)
            return 1

    ensure_rcm_dir(BASE_RCM_DIR, 0o700)
    ensure_rcm_dir(TRAPS_DIR, 0o700)
    LATEST_TRAP_FILE.unlink(missing_ok=True)
    CONTROL_ACTION_FILE.unlink(missing_ok=True)
    (TRAPS_DIR / "history.json").unlink(missing_ok=True)  # Reset recurrence tracking on fresh run (B5)
    if DAEMON_LOG_FILE.exists():
        DAEMON_LOG_FILE.unlink()

    script_path = pathlib.Path(args.script).resolve()
    script_cmd = [sys.executable, "-u", str(script_path)] + (args.script_args or [])
    env = dict(os.environ)
    env["RCM_DAEMON_WORKER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    # Ensure repository directories are in PYTHONPATH
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    adm_root = repo_root / "admission-control-vllm" if repo_root.name != "admission-control-vllm" else repo_root
    py_paths = [str(repo_root), str(adm_root)]
    if "PYTHONPATH" in env:
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_paths)

    log_f = open(DAEMON_LOG_FILE, "w", encoding="utf-8")
    proc = subprocess.Popen(
        script_cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=os.getcwd(),
        start_new_session=True,
    )
    DAEMON_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"[rcm_exec] Started background worker daemon PID {proc.pid}", flush=True)
    if getattr(args, "daemon", False):
        print(f"[rcm_exec] Workload spawned as background daemon PID {proc.pid}.", flush=True)
        print("[rcm_exec] Run 'rcm_exec wait' to wait for diagnostic trap or completion.", flush=True)
        print("[rcm_exec] Run 'rcm_exec status' or 'rcm_exec logs' to monitor.", flush=True)
        return 0

    return stream_daemon_output(
        DAEMON_LOG_FILE,
        proc,
        proc.pid,
        start_offset=0,
        stream_all=getattr(args, "stream", False),
    )


def cmd_reload(args: argparse.Namespace) -> int:
    """Hot-reloads a modified source file on disk into the paused worker daemon."""
    daemon_pid = get_active_daemon_pid()
    if not daemon_pid:
        print("[RCM ERROR] No active RCM worker daemon found. Run `rcm_exec run <script>` first.", file=sys.stderr)
        return 3

    target_file = pathlib.Path(args.file).resolve()
    if not target_file.exists():
        print(f"[RCM ERROR] File does not exist: {target_file}", file=sys.stderr)
        return 1

    offset = DAEMON_LOG_FILE.stat().st_size if DAEMON_LOG_FILE.exists() else 0
    LATEST_TRAP_FILE.unlink(missing_ok=True)

    directive = {
        "op": "HOT_PATCH",
        "file": str(target_file),
    }
    CONTROL_ACTION_FILE.write_text(json.dumps(directive) + "\n", encoding="utf-8")
    print(f"[rcm_exec] Injected HOT_PATCH for '{target_file.name}' to worker PID {daemon_pid}...", flush=True)
    if getattr(args, "no_wait", False):
        print(f"[rcm_exec] Injected HOT_PATCH to worker PID {daemon_pid}. Run 'rcm_exec wait' to wait for trap or completion.", flush=True)
        return 0

    return stream_daemon_output(
        DAEMON_LOG_FILE,
        None,
        daemon_pid,
        start_offset=offset,
        stream_all=getattr(args, "stream", False),
    )


def cmd_patch(args: argparse.Namespace) -> int:
    """Applies an in-memory live patch or disk unified diff to the paused worker daemon."""
    daemon_pid = get_active_daemon_pid()
    if not daemon_pid:
        print("[RCM ERROR] No active RCM worker daemon found. Run `rcm_exec run <script>` first.", file=sys.stderr)
        return 3

    directive: Dict[str, Any] = {"op": "LIVE_PATCH"}
    patch_text = None
    target_file = args.file

    if args.patch_target and not args.file and not args.stage and not args.target:
        p_cand = pathlib.Path(args.patch_target)
        if p_cand.exists() and p_cand.is_file():
            patch_text = p_cand.read_text(encoding="utf-8")
            for line in patch_text.splitlines():
                if line.startswith("--- a/") or line.startswith("+++ b/"):
                    cand = line.split("/", 1)[1].strip()
                    if pathlib.Path(cand).exists():
                        target_file = cand
                        break
        else:
            patch_text = args.patch_target

    if args.patch:
        p_cand = pathlib.Path(args.patch)
        if p_cand.exists() and p_cand.is_file():
            patch_text = p_cand.read_text(encoding="utf-8")
        else:
            patch_text = args.patch

    if target_file:
        directive["file"] = str(pathlib.Path(target_file).resolve())
    if patch_text:
        directive["patch"] = patch_text
    if args.code:
        directive["code"] = args.code
    if args.stage:
        directive["stage"] = args.stage
    if args.target:
        directive["target"] = args.target

    if not ("patch" in directive or "code" in directive):
        print("[RCM ERROR] Must specify --patch, --code, or a valid patch argument.", file=sys.stderr)
        return 1

    offset = DAEMON_LOG_FILE.stat().st_size if DAEMON_LOG_FILE.exists() else 0
    LATEST_TRAP_FILE.unlink(missing_ok=True)

    CONTROL_ACTION_FILE.write_text(json.dumps(directive) + "\n", encoding="utf-8")
    label = directive.get("file") or directive.get("stage") or directive.get("target") or "live_patch"
    print(f"[rcm_exec] Injected LIVE_PATCH ({label}) to worker PID {daemon_pid}...", flush=True)

    if getattr(args, "no_wait", False):
        print(f"[rcm_exec] Injected LIVE_PATCH to worker PID {daemon_pid}. Run 'rcm_exec wait' to wait for trap or completion.", flush=True)
        return 0

    return stream_daemon_output(
        DAEMON_LOG_FILE,
        None,
        daemon_pid,
        start_offset=offset,
        stream_all=getattr(args, "stream", False),
    )

def cmd_steer(args: argparse.Namespace) -> int:
    """Injects a raw JSON steering directive into the paused worker daemon."""
    daemon_pid = get_active_daemon_pid()
    if not daemon_pid:
        print("[RCM ERROR] No active RCM worker daemon found. Run `rcm_exec run <script>` first.", file=sys.stderr)
        return 3

    try:
        directive = json.loads(args.directive)
    except json.JSONDecodeError as e:
        print(f"[RCM ERROR] Invalid JSON steering directive: {e}", file=sys.stderr)
        return 1

    offset = DAEMON_LOG_FILE.stat().st_size if DAEMON_LOG_FILE.exists() else 0
    LATEST_TRAP_FILE.unlink(missing_ok=True)

    CONTROL_ACTION_FILE.write_text(json.dumps(directive) + "\n", encoding="utf-8")
    print(f"[rcm_exec] Injected directive '{directive.get('op')}' to worker PID {daemon_pid}...", flush=True)
    if getattr(args, "no_wait", False):
        print(f"[rcm_exec] Injected directive '{directive.get('op')}' to worker PID {daemon_pid}. Run 'rcm_exec wait' to wait for trap or completion.", flush=True)
        return 0

    return stream_daemon_output(
        DAEMON_LOG_FILE,
        None,
        daemon_pid,
        start_offset=offset,
        stream_all=getattr(args, "stream", False),
    )

def cmd_resume(args: argparse.Namespace) -> int:
    """
    C1: Diffs loaded modules from trap snapshot against disk, hot-patches dirty modules,
        retries stage, and streams output until next trap or completion.
    C2: Supports --skip, --abort, and --fresh flags.
    C3: Warns loudly if no dirty modules are detected and no explicit op was passed.
    """
    daemon_pid = get_active_daemon_pid()
    if not daemon_pid:
        print("[RCM ERROR] No active RCM worker daemon found. Run `rcm_exec run <script>` first.", file=sys.stderr)
        return 3

    # Handle explicit --abort (C2)
    if getattr(args, "abort", False):
        print("[rcm_exec resume] Aborting workload upon request...", flush=True)
        return cmd_abort(args)

    # Handle explicit --skip (C2)
    if getattr(args, "skip", False):
        offset = DAEMON_LOG_FILE.stat().st_size if DAEMON_LOG_FILE.exists() else 0
        LATEST_TRAP_FILE.unlink(missing_ok=True)
        directive = {"op": "SKIP_STAGE"}
        CONTROL_ACTION_FILE.write_text(json.dumps(directive) + "\n", encoding="utf-8")
        print(f"[rcm_exec resume] Injected SKIP_STAGE to worker PID {daemon_pid}...", flush=True)
        if getattr(args, "no_wait", False):
            return 0
        return stream_daemon_output(
            DAEMON_LOG_FILE,
            None,
            daemon_pid,
            start_offset=offset,
            stream_all=getattr(args, "stream", False),
        )

    # C1: Diff loaded modules from snapshot against disk
    trap_data = read_latest_trap(timeout_s=0.5)
    if not trap_data and LATEST_TRAP_FILE.exists():
        try:
            with open(LATEST_TRAP_FILE, "r", encoding="utf-8") as f:
                trap_data = json.load(f)
        except Exception:
            pass

    loaded_modules = trap_data.get("loaded_modules", {}) if trap_data else {}
    dirty_modules: List[Tuple[str, str]] = []

    for mod_name, info in loaded_modules.items():
        f_path_str = info.get("file")
        if not f_path_str:
            continue
        f_p = pathlib.Path(f_path_str)
        if f_p.exists():
            orig_mtime = info.get("mtime", 0.0)
            orig_sha1 = info.get("sha1", "")
            try:
                cur_mtime = f_p.stat().st_mtime
                cur_sha1 = ""
                with open(f_p, "rb") as f:
                    cur_sha1 = hashlib.sha1(f.read()).hexdigest()
                if cur_mtime > orig_mtime or (orig_sha1 and cur_sha1 != orig_sha1):
                    dirty_modules.append((mod_name, str(f_p.resolve())))
            except OSError:
                pass

    # Fallback to blame frame file if not caught in loaded_modules
    blame_file = trap_data.get("blame_frame", {}).get("file") if trap_data else None
    if blame_file and blame_file != "unknown":
        b_p = pathlib.Path(blame_file)
        if b_p.exists():
            b_resolved = str(b_p.resolve())
            if not any(f == b_resolved for _, f in dirty_modules):
                try:
                    b_mtime = b_p.stat().st_mtime
                    trap_time = trap_data.get("timestamp", 0.0) if trap_data else 0.0
                    if b_mtime > trap_time:
                        dirty_modules.append((b_p.stem, b_resolved))
                except OSError:
                    pass

    offset = DAEMON_LOG_FILE.stat().st_size if DAEMON_LOG_FILE.exists() else 0
    LATEST_TRAP_FILE.unlink(missing_ok=True)

    directive: Dict[str, Any] = {}
    if getattr(args, "fresh", False):
        directive["fresh"] = True

    if dirty_modules:
        mod_names = [m for m, _ in dirty_modules]
        print(f"[rcm_exec resume] Detected modified module(s): {', '.join(mod_names)}. Hot-patching and resuming...", flush=True)
        if len(dirty_modules) == 1:
            directive["op"] = "HOT_PATCH"
            directive["module"] = dirty_modules[0][0]
            directive["file"] = dirty_modules[0][1]
        else:
            directive["op"] = "HOT_PATCH"
            directive["files"] = [f for _, f in dirty_modules]
    else:
        # C3: Warn loudly if no dirty modules detected
        print("[RCM WARNING] No modified first-party modules detected since trap entry!", file=sys.stderr)
        print("[RCM WARNING] Did you edit a file that the daemon has not imported, or forget to save?", file=sys.stderr)
        print("[rcm_exec resume] Retrying stage without hot patch...", flush=True)
        directive["op"] = "RETRY"

    CONTROL_ACTION_FILE.write_text(json.dumps(directive) + "\n", encoding="utf-8")
    if getattr(args, "no_wait", False):
        print(f"[rcm_exec resume] Injected {directive['op']} to worker PID {daemon_pid}.", flush=True)
        return 0

    return stream_daemon_output(
        DAEMON_LOG_FILE,
        None,
        daemon_pid,
        start_offset=offset,
        stream_all=getattr(args, "stream", False),
    )


def cmd_trap(args: argparse.Namespace) -> int:
    """Displays or exports the latest diagnostic trap snapshot without re-waiting (F2)."""
    trap_data = read_latest_trap(timeout_s=0.2)
    if not trap_data and LATEST_TRAP_FILE.exists():
        try:
            with open(LATEST_TRAP_FILE, "r", encoding="utf-8") as f:
                trap_data = json.load(f)
        except Exception:
            pass

    if not trap_data:
        if getattr(args, "json", False):
            print(json.dumps({"error": "No active or recorded diagnostic trap found", "status": "NO_TRAP"}, indent=2))
        else:
            print("[rcm_exec] No active or recorded diagnostic trap found.", file=sys.stderr)
        return 3

    if getattr(args, "json", False):
        print(json.dumps(trap_data, indent=2))
        return 0

    daemon_pid = get_active_daemon_pid() or trap_data.get("pid", 0)
    print(format_trap_system_directive(trap_data, daemon_pid))
    return 2


def cmd_wait(args: argparse.Namespace) -> int:
    """Waits for active RCM worker daemon to reach a diagnostic trap, complete, or hit timeout."""
    daemon_pid = get_active_daemon_pid()
    timeout_s: Optional[float] = getattr(args, "timeout", None)
    target_stage: Optional[str] = getattr(args, "stage", None)
    stream_all: bool = getattr(args, "stream", False)
    quiet: bool = getattr(args, "quiet", False)
    json_mode: bool = getattr(args, "json", False)
    poll_interval_s: float = getattr(args, "poll", 0.05) or 0.05

    # 1. Immediate trap check
    trap_data = read_latest_trap(timeout_s=0.1)
    if trap_data:
        if json_mode:
            print(json.dumps({"status": "TRAPPED", "exit_code": 2, "trap": trap_data}, indent=2))
        else:
            pid = daemon_pid or trap_data.get("pid") or 0
            print(format_trap_system_directive(trap_data, pid), flush=True)
        return 2

    # 2. Daemon running check (F1: exit 3 when no daemon running)
    if not daemon_pid:
        status = get_live_status()
        if status and status.get("status") in ("SUCCESS", "COMPLETED"):
            if json_mode:
                print(json.dumps({"status": "SUCCESS", "exit_code": 0, "live_status": status}, indent=2))
            elif not quiet:
                print(f"[rcm_exec] Workload already finished with status {status.get('status')}.")
            return 0
        elif status and status.get("status") in ("FAILED", "ABORTED"):
            if json_mode:
                print(json.dumps({"status": "FAILED", "exit_code": 1, "live_status": status}, indent=2))
            elif not quiet:
                print(f"[rcm_exec] Workload already finished with status {status.get('status')}.")
            return 1
        if json_mode:
            print(json.dumps({"status": "NO_DAEMON", "exit_code": 3}, indent=2))
        else:
            print("[rcm_exec] No active RCM worker daemon running.", file=sys.stderr)
        return 3

    stage_desc = f"stage '{target_stage}'" if target_stage else "any trap or completion"
    timeout_desc = f"{timeout_s}s" if timeout_s is not None else "indefinite"
    if not quiet and not json_mode:
        print(f"[rcm_exec] Waiting for daemon PID {daemon_pid} ({stage_desc}, timeout: {timeout_desc})...", flush=True)

    curr_offset = DAEMON_LOG_FILE.stat().st_size if DAEMON_LOG_FILE.exists() else 0
    rc = stream_daemon_output(
        log_file=DAEMON_LOG_FILE,
        daemon_proc=None,
        daemon_pid=daemon_pid,
        start_offset=curr_offset,
        stream_all=stream_all,
        quiet=quiet or json_mode,
        target_stage=target_stage,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )
    if json_mode:
        if rc == 2:
            latest = read_latest_trap(timeout_s=0.2) or {}
            print(json.dumps({"status": "TRAPPED", "exit_code": 2, "trap": latest}, indent=2))
        elif rc == 0:
            print(json.dumps({"status": "SUCCESS", "exit_code": 0}, indent=2))
        elif rc == 124:
            print(json.dumps({"status": "TIMEOUT", "exit_code": 124}, indent=2))
        else:
            print(json.dumps({"status": "FAILED", "exit_code": rc}, indent=2))
    return rc


def cmd_status(args: argparse.Namespace) -> int:
    """Queries live worker status, stage timing, diagnostic trap details, and active Ray jobs (E3)."""
    daemon_pid = get_active_daemon_pid()
    all_jobs = read_all_ray_jobs()
    jobs_map = all_jobs.get("jobs", {})

    zombie = is_pid_zombie(daemon_pid) if daemon_pid else False
    live_st = get_live_status()
    heartbeat_s = live_st.get("heartbeat_s") or live_st.get("timestamp") if live_st else None
    trap_data = read_latest_trap(timeout_s=0.05)
    stale_heartbeat = False
    if daemon_pid and not zombie and not trap_data and heartbeat_s:
        if (time.time() - heartbeat_s) > 60.0:
            stale_heartbeat = True

    if getattr(args, "json", False):
        status_payload = {
            "daemon_pid": daemon_pid,
            "alive": (is_pid_alive(daemon_pid) and not zombie) if daemon_pid else False,
            "zombie": zombie,
            "heartbeat_s": heartbeat_s,
            "stale_heartbeat": stale_heartbeat,
            "live_status": live_st,
            "latest_trap": trap_data,
            "ray_jobs": all_jobs,
        }
        print(json.dumps(status_payload, indent=2))
        return 0

    if not daemon_pid:
        print("Status: Idle (No active RCM worker daemon)")
        if len(jobs_map) > 1:
            print("\nRecent Ray Cluster Jobs:")
            print(format_ray_jobs_table(all_jobs))
        elif jobs_map:
            print("\nRecent Ray Cluster Job:")
            print(format_ray_job_summary(list(jobs_map.values())[0]))
        return 0

    if zombie:
        print(f"Status: Defunct / Zombie Daemon PID {daemon_pid} (terminated or un-reaped)")
    else:
        print(f"Status: Active Worker Daemon PID {daemon_pid}")
        if stale_heartbeat:
            print(f"  [Warning: Daemon heartbeat is stale ({int(time.time() - heartbeat_s)}s since last beat)]")

    if LIVE_STATUS_FILE.exists():
        try:
            with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                st = json.load(f)
            print(f"Current Stage: {st.get('current_stage')}")
            print(f"Clocks: {st.get('clocks')}")
        except Exception:
            pass

    if len(jobs_map) > 1:
        print("\n" + format_ray_jobs_table(all_jobs))
    elif jobs_map:
        print("\n" + format_ray_job_summary(list(jobs_map.values())[0]))

    if trap_data:
        print(f"\nState: TRAP_PAUSED in stage '{trap_data.get('stage')}'")
        print(f"Reason: {trap_data.get('reason')}")
    elif zombie:
        print("State: ZOMBIE")
    else:
        print("State: RUNNING")
    return 0


def cmd_ray_status(args: argparse.Namespace) -> int:
    """Directly inspects active or recorded Ray cluster jobs from RCM runtime."""
    target_id = getattr(args, "job_id", None)
    all_jobs = read_all_ray_jobs()
    jobs_map = all_jobs.get("jobs", {})

    if not jobs_map:
        print("No active or recorded Ray cluster jobs found in RCM runtime.")
        return 0

    if target_id:
        # Look up by exact ID, prefix, or name
        target = jobs_map.get(target_id)
        if not target:
            for sub_id, j in jobs_map.items():
                if sub_id.startswith(target_id) or j.get("name") == target_id:
                    target = j
                    break
        if not target:
            print(f"No Ray cluster job matching '{target_id}' found in RCM runtime.")
            return 1

        print(f"=== RCM Ray Cluster Job Inspection: {target.get('submission_id')} ===")
        print(format_ray_job_summary(target))
        if getattr(args, "logs", False):
            tail = target.get("log_tail", [])
            if tail:
                print("\nRemote Log Tail:")
                for l in tail:
                    print(f"  {l}")
        return 0

    # Multiple jobs: display table roster
    if len(jobs_map) > 1:
        print(format_ray_jobs_table(all_jobs))
        active_id = all_jobs.get("active_job_id")
        active_job = jobs_map.get(active_id)
        if active_job and getattr(args, "logs", False):
            print(f"\nActive Job ({active_job.get('submission_id')}) Remote Log Tail:")
            for l in active_job.get("log_tail", []):
                print(f"  {l}")
    else:
        single = list(jobs_map.values())[0]
        print(format_ray_job_summary(single))
        if getattr(args, "logs", False):
            tail = single.get("log_tail", [])
            if tail:
                print("\nRemote Log Tail:")
                for l in tail:
                    print(f"  {l}")
    return 0
def cmd_logs(args: argparse.Namespace) -> int:
    """Inspects out-of-band JSONL telemetry with fine-grained filtering."""
    if args.summary:
        print(format_summary())
        return 0
    job_filter = getattr(args, "job", None)
    grep_pat = args.grep
    if getattr(args, "ray", False):
        ray_pat = r"(\[OMP_EVENT:|\bRay\b|\bTPU\b|FORWARDED_EVENT)"
        grep_pat = f"({grep_pat})|{ray_pat}" if grep_pat else ray_pat
    if job_filter:
        grep_pat = f"({grep_pat}).*{job_filter}" if grep_pat else job_filter
    entries = inspect_logs(
        telemetry_path=TELEMETRY_FILE,
        tail=args.tail,
        stage=args.stage,
        level=args.level,
        grep=grep_pat,
        since_s=args.since,
    )
    if getattr(args, "json", False):
        print(json.dumps(entries, indent=2))
        return 0
    print(format_inspected_logs(entries))
    return 0

def cmd_abort(args: argparse.Namespace) -> int:
    """Gracefully halts active worker daemon and terminates remote Ray cluster jobs."""
    target_job_id = getattr(args, "job", None)
    all_jobs = read_all_ray_jobs()
    jobs_map = all_jobs.get("jobs", {})

    # Stop specific Ray job if requested
    if target_job_id:
        target = jobs_map.get(target_job_id)
        if not target:
            for sub_id, j in jobs_map.items():
                if sub_id.startswith(target_job_id) or j.get("name") == target_job_id:
                    target = j
                    break
        if target and not target.get("terminal", False):
            sub_id = target.get("submission_id")
            ray_url = target.get("ray_url", "http://127.0.0.1:8265")
            print(f"Cancelling remote Ray cluster job {sub_id} on {ray_url}...", flush=True)
            try:
                from ray.job_submission import JobSubmissionClient
                client = JobSubmissionClient(ray_url)
                client.stop_job(sub_id)
                print(f"Remote Ray cluster job {sub_id} stopped.", flush=True)
            except Exception as e:
                print(f"Notice cancelling Ray job {sub_id}: {e}", flush=True)
    else:
        # Stop ALL active Ray cluster jobs
        for sub_id, j in jobs_map.items():
            if not j.get("terminal", False):
                ray_url = j.get("ray_url", "http://127.0.0.1:8265")
                print(f"Cancelling remote Ray cluster job {sub_id} on {ray_url}...", flush=True)
                try:
                    from ray.job_submission import JobSubmissionClient
                    client = JobSubmissionClient(ray_url)
                    client.stop_job(sub_id)
                    print(f"Remote Ray cluster job {sub_id} stopped.", flush=True)
                except Exception as e:
                    print(f"Notice cancelling Ray job {sub_id}: {e}", flush=True)
    daemon_pid = get_active_daemon_pid()
    if not daemon_pid:
        print("No active RCM worker daemon to abort.")
        return 3

    directive = {"op": "ABORT"}
    CONTROL_ACTION_FILE.write_text(json.dumps(directive) + "\n", encoding="utf-8")
    print(f"Sent ABORT directive to worker PID {daemon_pid}. Waiting for termination...", flush=True)

    for _ in range(30):
        if not is_pid_alive(daemon_pid):
            break
        time.sleep(0.1)

    if is_pid_alive(daemon_pid):
        print(f"Worker PID {daemon_pid} did not exit gracefully; terminating...", flush=True)
        try:
            os.kill(daemon_pid, signal.SIGTERM)
            time.sleep(0.5)
            if is_pid_alive(daemon_pid):
                os.kill(daemon_pid, signal.SIGKILL)
        except OSError:
            pass

    DAEMON_PID_FILE.unlink(missing_ok=True)
    print("Worker daemon stopped successfully.")
    return 0

def build_parser() -> argparse.ArgumentParser:
    """Constructs the unified CLI argument parser for RCM."""
    parser = argparse.ArgumentParser(
        prog="rcm_exec",
        description="Oh My Pi (OMP) Reverse Control Mechanism (RCM) CLI Supervisor",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run RCM script in background and stream logs")
    p_run.add_argument("script", help="Path to Python script")
    p_run.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments to script")
    p_run.add_argument("--daemon", "-d", "--detach", "--background", action="store_true", help="Spawn worker in background and return immediately")
    p_run.add_argument("--stream", action="store_true", help="Stream all stdout logs while running (default: events only in non-TTY/agent mode)")
    p_run.add_argument("--replace", action="store_true", help="Auto-reap existing worker daemon before running")
    p_run.set_defaults(func=cmd_run)

    # wait
    p_wait = subparsers.add_parser("wait", help="Wait for active worker daemon to hit diagnostic trap or complete")
    p_wait.add_argument("--timeout", "-t", type=float, default=None, help="Maximum seconds to wait before timing out (exit 124)")
    p_wait.add_argument("--stage", "-s", type=str, default=None, help="Wait until a specific stage completes or traps")
    p_wait.add_argument("--stream", action="store_true", help="Stream all stdout logs while waiting (default: events only)")
    p_wait.add_argument("--quiet", "-q", action="store_true", help="Do not print intermediate events, only trap or final status")
    p_wait.add_argument("--poll", "-p", type=float, default=0.05, help="Polling interval in seconds (default: 0.05)")
    p_wait.add_argument("--json", action="store_true", help="Output final wait result as JSON")
    p_wait.set_defaults(func=cmd_wait)

    # resume
    p_resume = subparsers.add_parser("resume", help="Diff loaded modules, hot-patch changes, retry stage, and wait")
    p_resume.add_argument("--skip", action="store_true", help="Skip the currently trapped stage and advance")
    p_resume.add_argument("--abort", action="store_true", help="Abort the workload and release resources")
    p_resume.add_argument("--fresh", action="store_true", help="Clear memoized stage caches and retry cleanly")
    p_resume.add_argument("--stream", action="store_true", help="Stream all stdout logs while waiting (default: events only)")
    p_resume.add_argument("--no-wait", "-d", action="store_true", help="Inject resume directive without waiting for logs")
    p_resume.set_defaults(func=cmd_resume)

    # trap
    p_trap = subparsers.add_parser("trap", help="Display latest diagnostic trap snapshot without re-waiting")
    p_trap.add_argument("--json", action="store_true", help="Output trap snapshot as JSON")
    p_trap.set_defaults(func=cmd_trap)

    p_wait_trap = subparsers.add_parser("wait-trap", help="Alias for wait")
    p_wait_trap.add_argument("--timeout", "-t", type=float, default=None, help="Maximum seconds to wait before timing out")
    p_wait_trap.add_argument("--stage", "-s", type=str, default=None, help="Wait until a specific stage completes or traps")
    p_wait_trap.add_argument("--stream", action="store_true", help="Stream all stdout logs while waiting")
    p_wait_trap.add_argument("--quiet", "-q", action="store_true", help="Do not print intermediate events")
    p_wait_trap.add_argument("--poll", "-p", type=float, default=0.05, help="Polling interval in seconds")
    p_wait_trap.add_argument("--json", action="store_true", help="Output final wait result as JSON")
    p_wait_trap.set_defaults(func=cmd_wait)
    # reload
    p_reload = subparsers.add_parser("reload", help="Hot-reload modified source file and resume execution")
    p_reload.add_argument("file", help="Path to modified Python file")
    p_reload.add_argument("--no-wait", "-d", action="store_true", help="Inject reload without waiting for logs")
    p_reload.add_argument("--stream", action="store_true", help="Stream all stdout logs after reload")
    p_reload.set_defaults(func=cmd_reload)

    # patch
    p_patch = subparsers.add_parser("patch", help="Apply in-memory live patch or disk unified diff")
    p_patch.add_argument("patch_target", nargs="?", help="Optional patch file or diff string")
    p_patch.add_argument("--file", help="Path to target file on disk")
    p_patch.add_argument("--patch", help="Unified diff string or path to .patch file")
    p_patch.add_argument("--code", help="Replacement Python source code string")
    p_patch.add_argument("--stage", help="Target stage name in StageRunner")
    p_patch.add_argument("--target", help="Target callable dotted path (e.g. module.function)")
    p_patch.add_argument("--no-wait", "-d", action="store_true", help="Inject patch without waiting for logs")
    p_patch.add_argument("--stream", action="store_true", help="Stream all stdout logs after patch")
    p_patch.set_defaults(func=cmd_patch)
    # steer
    p_steer = subparsers.add_parser("steer", help="Inject steering JSON directive and resume")
    p_steer.add_argument("directive", help="JSON directive string, e.g. '{\"op\": \"RETRY\"}'")
    p_steer.add_argument("--no-wait", "-d", action="store_true", help="Inject directive without waiting for logs")
    p_steer.add_argument("--stream", action="store_true", help="Stream all stdout logs after steer")
    p_steer.set_defaults(func=cmd_steer)
    # status
    p_status = subparsers.add_parser("status", help="Query live worker status")
    p_status.add_argument("--json", action="store_true", help="Output status as JSON")
    p_status.set_defaults(func=cmd_status)

    # logs / inspect
    p_logs = subparsers.add_parser("logs", help="Inspect out-of-band JSONL telemetry logs")
    p_logs.add_argument("--tail", type=int, default=30, help="Number of latest log lines (default: 30)")
    p_logs.add_argument("--stage", type=str, default=None, help="Filter by stage name")
    p_logs.add_argument("--level", type=str, default=None, help="Filter by comma-separated levels (e.g. WARN,ERROR)")
    p_logs.add_argument("--grep", type=str, default=None, help="Regex filter for message content")
    p_logs.add_argument("--since", type=float, default=None, help="Show logs emitted in last N seconds")
    p_logs.add_argument("--summary", action="store_true", help="Print structured state and trap summary")
    p_logs.add_argument("--json", action="store_true", help="Output logs as JSON")
    p_logs.set_defaults(func=cmd_logs)
    p_logs.add_argument("--ray", action="store_true", help="Filter for Ray cluster events and forwarded logs")

    p_logs.add_argument("--job", type=str, default=None, help="Filter telemetry by specific Ray job ID or name")

    # ray-status / jobs
    p_ray = subparsers.add_parser("ray-status", help="Directly query active/recent Ray cluster job details")
    p_ray.add_argument("job_id", nargs="?", help="Optional specific submission ID or name to inspect")
    p_ray.add_argument("--logs", action="store_true", help="Include full recent remote log tail")
    p_ray.set_defaults(func=cmd_ray_status)

    p_jobs = subparsers.add_parser("jobs", help="List and inspect Ray cluster jobs in RCM runtime")
    p_jobs.add_argument("job_id", nargs="?", help="Optional specific submission ID or name to inspect")
    p_jobs.add_argument("--logs", action="store_true", help="Include full recent remote log tail")
    p_jobs.set_defaults(func=cmd_ray_status)

    # abort
    p_abort = subparsers.add_parser("abort", help="Abort active worker daemon and cancel remote Ray cluster jobs")
    p_abort.add_argument("--job", type=str, default=None, help="Optional specific Ray job ID to cancel")
    p_abort.set_defaults(func=cmd_abort)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
