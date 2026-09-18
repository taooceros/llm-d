"""
Reverse Control Mechanism (RCM) Core Engine for Oh My Pi (OMP).
Script as Control Flow, Agent as Side Watcher & Joint Time-Log Analyst.

Key Architecture:
- StageRunner: Sequential deterministic stage machine with discrete event emission.
- StateStore: Thread-safe persistent state memory across stage retries and hot-patches.
- JointTimingLogEngine: High-frequency telemetry sink, windowed FIFO buffer, and normality evaluator.
- DiagnosticTrap: Out-of-band diagnostic snapshot generator for zero-context-token analysis.
- SteeringReceiver: Multi-channel in-flight steering receiver (stdin, unix socket, file, programmatic).
- WatchdogTimer: Periodic SLA/silence monitor and non-destructive SIGUSR1 asymmetric interrupter.
"""

from __future__ import annotations

import collections
import ctypes
import dataclasses
import importlib
import hashlib
import inspect
import json
import linecache
import os
import pathlib
import re
import select
import shlex
import signal
import socket
import sys
import threading
import time
import traceback
import types
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# Global default paths for out-of-band communication
BASE_RCM_DIR = pathlib.Path("/tmp/omp_rcm")
CHECKPOINT_FILE = BASE_RCM_DIR / "checkpoint.json"
TELEMETRY_FILE = BASE_RCM_DIR / "telemetry.jsonl"
CONTROL_SOCK_FILE = BASE_RCM_DIR / "control.sock"
CONTROL_ACTION_FILE = BASE_RCM_DIR / "control_action.json"
TRAPS_DIR = BASE_RCM_DIR / "traps"
LATEST_TRAP_FILE = TRAPS_DIR / "latest.json"
LIVE_STATUS_FILE = BASE_RCM_DIR / "live_status.json"
LIVE_LOGS_FILE = BASE_RCM_DIR / "live_logs.txt"
RAY_JOB_FILE = BASE_RCM_DIR / "ray_job.json"

EVENT_MARKERS = (
    "[OMP_EVENT:",
    "FORWARDED_EVENT",
    "[CRITICAL]",
    "[ERROR]",
    "TRAP_PAUSED",
)


def is_event_line(line: str) -> bool:
    """Checks if a log line matches any known RCM event marker."""
    return any(marker in line for marker in EVENT_MARKERS)


OMP_EVENT_GRAMMAR = """
Formal Grammar for OMP Structured Telemetry Events (F3):
========================================================
Line format:
  [OMP_EVENT: <EVENT_TYPE> [key=value | key="quoted string"]...]

Field parsing rules:
  - Tokens are whitespace-delimited outside double quotes.
  - Quoted values start and end with double quotes (") and may contain spaces and escaped quotes.
  - Numeric values (integers and floats) are parsed into numbers when possible.
  - Boolean flags without '=' evaluate to True.

Standard Event Types:
  - RUN_STARTED total_stages=<int>
  - STAGE_ENTERED stage=<name> stage_idx=<int> expected_s=<float>
  - STAGE_SETTLED stage=<name> elapsed_s=<float>
  - STAGE_SKIPPED stage=<name>
  - TRAP_PAUSED stage=<name> elapsed_s=<float> expected_s=<float> reason="<str>" trap_file="<path>"
  - TRAP_RESOLVED stage=<name> action=<OP>
  - ANOMALY_WARNING stage=<name> elapsed_s=<float> expected_s=<float> reason="<str>"
  - RUN_FINISHED status=<SUCCESS|FAILED|ABORTED> total_elapsed_s=<float> [reason="<str>"]
  - TPU_JOB_COMPLETED
  - TPU_JOB_FAILED error="<str>"
"""


def parse_omp_event(line: str) -> Optional[Dict[str, Any]]:
    """
    Parses a structured [OMP_EVENT: <EVENT_TYPE> k=v ...] line into a dictionary.
    Returns None if the line does not contain an [OMP_EVENT: ...] envelope.
    """
    if "[OMP_EVENT:" not in line:
        return None
    try:
        start_idx = line.index("[OMP_EVENT:") + len("[OMP_EVENT:")
        end_idx = line.rindex("]")
        content = line[start_idx:end_idx].strip()
    except ValueError:
        return None

    tokens = shlex.split(content)
    if not tokens:
        return None

    event_type = tokens[0]
    data: Dict[str, Any] = {"event": event_type}
    for tok in tokens[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                if "." in v:
                    data[k] = float(v)
                else:
                    data[k] = int(v)
            except ValueError:
                data[k] = v
        else:
            data[tok] = True
    return data

def ensure_rcm_dir(path: Union[pathlib.Path, str] = BASE_RCM_DIR, mode: int = 0o700) -> pathlib.Path:
    """Ensures directory exists with strict permissions (chmod 700)."""
    p = pathlib.Path(path)
    p.mkdir(mode=mode, parents=True, exist_ok=True)
    try:
        os.chmod(p, mode)
    except OSError:
        pass
    return p


REDACT_NAME_RE = re.compile(r"(?i)(key|token|secret|password|credential|auth)")
EXCLUDED_FRAME_RE = re.compile(r"(site-packages|dist-packages|/\.venv|/usr/lib/python)")


def sanitize_value(val: Any, name: str = "", max_chars: int = 200) -> Any:
    """
    Sanitizes values according to Decision 2:
    - Name-based redaction: (?i)(key|token|secret|password|credential|auth)
    - Size-based elision: len(repr(v)) > max_chars -> type/shape summary
      Fast path for numpy/torch/jax arrays.
    """
    if name and REDACT_NAME_RE.search(str(name)):
        return "<redacted>"

    # Fast path for numpy/torch/jax arrays or array-like objects with shape and dtype
    if hasattr(val, "shape") and hasattr(val, "dtype"):
        try:
            return f"<{type(val).__name__} shape={tuple(val.shape)} dtype={val.dtype}>"
        except Exception:
            pass

    # Basic JSON types
    if val is None or isinstance(val, (bool, int, float)):
        return val

    if isinstance(val, str):
        if len(val) > max_chars:
            return f"<str len={len(val)}: {val[:max_chars // 2]!r}...>"
        return val

    if isinstance(val, bytes):
        return f"<bytes len={len(val)}: {val[:min(len(val), 20)]!r}...>"

    if isinstance(val, (list, tuple)):
        try:
            r = repr(val)
        except Exception:
            r = f"<{type(val).__name__}>"
        if len(r) > max_chars:
            return f"<{type(val).__name__} len={len(val)}>"
        return r

    if isinstance(val, dict):
        try:
            r = repr(val)
        except Exception:
            r = "<dict>"
        if len(r) > max_chars:
            return f"<dict len={len(val)}>"
        return r

    try:
        r = repr(val)
    except Exception:
        r = f"<{type(val).__name__}>"
    if len(r) > max_chars:
        return f"<{type(val).__name__}: {r[:max_chars // 2]}...>"
    return r


def extract_state_sample(
    state_store: Any,
    max_keys: int = 40,
    max_chars: int = 200,
) -> Dict[str, Any]:
    """Serializes at most max_keys from state_store, sanitized and capped."""
    if state_store is None:
        return {}
    if hasattr(state_store, "_store"):
        raw = dict(state_store._store)
    elif hasattr(state_store, "to_dict"):
        raw = state_store.to_dict()
    elif isinstance(state_store, dict):
        raw = dict(state_store)
    else:
        return {}

    sampled: Dict[str, Any] = {}
    keys = list(raw.keys())
    for k in keys[:max_keys]:
        k_str = str(k)
        sampled[k_str] = sanitize_value(raw[k], name=k_str, max_chars=max_chars)

    if len(keys) > max_keys:
        sampled["..."] = f"{len(keys) - max_keys} keys omitted"
    return sampled


def get_repo_root() -> pathlib.Path:
    """Finds repository root directory."""
    pkg_dir = pathlib.Path(__file__).resolve().parent
    cand_dirs = [pkg_dir.parent, pathlib.Path.cwd()]
    for cand in cand_dirs:
        try:
            resolved = cand.resolve()
            if (resolved / ".jj").exists() or (resolved / ".git").exists():
                return resolved
            if (resolved / "omp_rcm").exists() and (resolved / "admission_control").exists():
                return resolved
        except Exception:
            pass
    return pkg_dir.parent.resolve()


def is_first_party_file(f_path: str, repo_root: Optional[pathlib.Path] = None) -> bool:
    """Checks if a file belongs to the first-party repository and is not vendored/library."""
    if not f_path or f_path.startswith("<") or f_path.endswith(">"):
        return False
    if EXCLUDED_FRAME_RE.search(f_path):
        return False
    root = repo_root or get_repo_root()
    try:
        f_resolved = str(pathlib.Path(f_path).resolve())
        root_resolved = str(root.resolve())
        return f_resolved.startswith(root_resolved)
    except Exception:
        return False


def get_source_window(
    filename: str,
    lineno: int,
    window: int = 15,
) -> Dict[str, Any]:
    """Extracts ±15 lines of source code around lineno with a '>' gutter marker."""
    p = pathlib.Path(filename)
    if not p.is_absolute():
        repo_root = get_repo_root()
        cand = repo_root / p
        if cand.exists():
            p = cand
    if not p.exists():
        return {
            "file": filename,
            "line": lineno,
            "formatted": "",
            "lines": [],
        }

    str_path = str(p.resolve())
    linecache.checkcache(str_path)
    start_line = max(1, lineno - window)
    end_line = lineno + window

    lines: List[Dict[str, Any]] = []
    lines_formatted: List[str] = []

    for cur_line in range(start_line, end_line + 1):
        text = linecache.getline(str_path, cur_line)
        if not text and cur_line > lineno:
            break
        text_clean = text.rstrip("\r\n")
        marker = ">" if cur_line == lineno else " "
        formatted = f"{marker} {cur_line:4d}: {text_clean}"
        lines_formatted.append(formatted)
        lines.append({
            "line": cur_line,
            "marker": marker,
            "text": text_clean,
        })

    return {
        "file": str_path,
        "line": lineno,
        "start_line": start_line,
        "end_line": start_line + len(lines) - 1 if lines else start_line,
        "formatted": "\n".join(lines_formatted),
        "lines": lines,
    }


def extract_frame_locals(frame: Any, max_vars: int = 40, max_chars: int = 200) -> Dict[str, Any]:
    """Captures sanitized local variables from a Python frame."""
    if not hasattr(frame, "f_locals"):
        return {}
    res: Dict[str, Any] = {}
    f_locals = getattr(frame, "f_locals", {})
    keys = [k for k in f_locals.keys() if not (k.startswith("__") and k.endswith("__"))]
    for k in keys[:max_vars]:
        res[k] = sanitize_value(f_locals[k], name=k, max_chars=max_chars)
    if len(keys) > max_vars:
        res["..."] = f"{len(keys) - max_vars} vars omitted"
    return res


def analyze_frames(
    exc: Optional[BaseException] = None,
    frame: Optional[Any] = None,
    live_traceback: Optional[str] = None,
    repo_root: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """
    Selects blame frame according to B1:
    Deepest frame under repo root and NOT matching (site-packages|dist-packages|/.venv|/usr/lib/python).
    Falls back to deepest frame.
    Extracts source window (B2), frame locals (B3), and first-party traceback (B4).
    """
    root = repo_root or get_repo_root()
    extracted_frames: List[Tuple[Any, int, str, str]] = []

    if exc is not None and exc.__traceback__ is not None:
        tb = exc.__traceback__
        while tb:
            f = tb.tb_frame
            extracted_frames.append((f, tb.tb_lineno, f.f_code.co_filename, f.f_code.co_name))
            tb = tb.tb_next
    elif frame is not None:
        curr = frame
        tmp = []
        while curr:
            tmp.append((curr, curr.f_lineno, curr.f_code.co_filename, curr.f_code.co_name))
            curr = curr.f_back
        tmp.reverse()
        extracted_frames = tmp

    tb_string_frames: List[Tuple[str, int, str]] = []
    if not extracted_frames and live_traceback:
        for line in live_traceback.splitlines():
            line_s = line.strip()
            if line_s.startswith('File "') and '", line ' in line_s:
                try:
                    parts = line_s.split('File "')[1].split('", line ')
                    f_p = parts[0]
                    rest = parts[1]
                    l_no = int(rest.split(",")[0].strip())
                    fn = rest.split("in ")[1].strip() if "in " in rest else "?"
                    tb_string_frames.append((f_p, l_no, fn))
                except Exception:
                    pass

    blame_file = "unknown"
    blame_line = 0
    blame_func = "?"
    blame_frame_obj = None
    parent_frame_obj = None
    parent_file = None
    parent_line = None
    parent_func = None

    if extracted_frames:
        blame_idx = -1
        # B1: Scan backwards for deepest first-party frame
        for i in range(len(extracted_frames) - 1, -1, -1):
            f_obj, l_no, f_p, fn = extracted_frames[i]
            if is_first_party_file(f_p, root):
                blame_idx = i
                break
        # Fallback to deepest frame
        if blame_idx == -1:
            blame_idx = len(extracted_frames) - 1

        blame_frame_obj, blame_line, blame_file, blame_func = extracted_frames[blame_idx]
        if blame_idx > 0:
            parent_frame_obj, parent_line, parent_file, parent_func = extracted_frames[blame_idx - 1]
    elif tb_string_frames:
        blame_idx = -1
        for i in range(len(tb_string_frames) - 1, -1, -1):
            f_p, l_no, fn = tb_string_frames[i]
            if is_first_party_file(f_p, root):
                blame_idx = i
                break
        if blame_idx == -1:
            blame_idx = len(tb_string_frames) - 1

        blame_file, blame_line, blame_func = tb_string_frames[blame_idx]
        if blame_idx > 0:
            parent_file, parent_line, parent_func = tb_string_frames[blame_idx - 1]

    # Source window for blame frame (B2)
    source_window = get_source_window(blame_file, blame_line, window=15) if blame_file != "unknown" and blame_line > 0 else {}

    # Frame locals (B3)
    blame_locals = extract_frame_locals(blame_frame_obj) if blame_frame_obj else {}
    parent_locals = extract_frame_locals(parent_frame_obj) if parent_frame_obj else {}

    # Condensed first-party traceback (B4)
    first_party_traceback: List[Dict[str, Any]] = []
    if extracted_frames:
        for f_obj, l_no, f_p, fn in extracted_frames:
            if is_first_party_file(f_p, root):
                first_party_traceback.append({
                    "file": f_p,
                    "line": l_no,
                    "function": fn,
                    "code": linecache.getline(f_p, l_no).strip() if pathlib.Path(f_p).exists() else "",
                })
    elif tb_string_frames:
        for f_p, l_no, fn in tb_string_frames:
            if is_first_party_file(f_p, root):
                first_party_traceback.append({
                    "file": f_p,
                    "line": l_no,
                    "function": fn,
                    "code": linecache.getline(f_p, l_no).strip() if pathlib.Path(f_p).exists() else "",
                })

    full_tb_str = None
    if exc is not None:
        full_tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    elif live_traceback is not None:
        full_tb_str = live_traceback

    return {
        "blame_frame": {
            "file": blame_file,
            "line": blame_line,
            "function": blame_func,
            "locals": blame_locals,
            "source_window": source_window,
        },
        "parent_frame": {
            "file": parent_file,
            "line": parent_line,
            "function": parent_func,
            "locals": parent_locals,
        } if parent_file else None,
        "first_party_traceback": first_party_traceback,
        "full_traceback": full_tb_str,
    }


def snapshot_loaded_modules(repo_root: Optional[pathlib.Path] = None) -> Dict[str, Dict[str, Any]]:
    """Snapshots {module_name: {file, mtime, sha1}} for all loaded first-party modules (C1)."""
    root = repo_root or get_repo_root()
    result: Dict[str, Dict[str, Any]] = {}
    for mod_name, mod in list(sys.modules.items()):
        if not mod or not hasattr(mod, "__file__") or not mod.__file__:
            continue
        f_path = str(pathlib.Path(mod.__file__).resolve())
        if is_first_party_file(f_path, root):
            try:
                st = os.stat(f_path)
                mtime = st.st_mtime
                with open(f_path, "rb") as f:
                    content_sha = hashlib.sha1(f.read()).hexdigest()
                result[mod_name] = {
                    "file": f_path,
                    "mtime": mtime,
                    "sha1": content_sha,
                }
            except OSError:
                continue
    return result
class AbortExecution(Exception):
    """Raised when an explicit ABORT directive is received."""
    pass


class SkipStage(Exception):
    """Raised when a stage is explicitly skipped via steering."""
    pass


class TrapInterrupt(BaseException):
    """Raised inside a worker thread when interrupted by a watchdog trap or signal."""
    pass


@dataclasses.dataclass
class StageMetadata:
    name: str
    order_idx: int
    expected_s: float
    func: Callable[..., Any]
    module_name: str
    qualname: str
    doc: Optional[str] = None
    critical_patterns: Optional[List[Tuple[re.Pattern, str]]] = None
    allow_repetitive: bool = False


def stage(
    name: Optional[str] = None,
    order_idx: int = 0,
    expected_s: float = 5.0,
    critical_patterns: Optional[List[Union[str, Tuple[str, str]]]] = None,
    allow_repetitive: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a function as a managed Reverse Control Stage."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        stage_name = name or func.__name__
        compiled_patterns = []
        if critical_patterns:
            for item in critical_patterns:
                if isinstance(item, tuple):
                    pat, reason = item
                    compiled_patterns.append((re.compile(pat, re.IGNORECASE) if isinstance(pat, str) else pat, reason))
                elif isinstance(item, (str, re.Pattern)):
                    pat = item
                    compiled_patterns.append((re.compile(pat, re.IGNORECASE) if isinstance(pat, str) else pat, f"Critical pattern matched: {pat}"))
        meta = StageMetadata(
            name=stage_name,
            order_idx=order_idx,
            expected_s=expected_s,
            func=func,
            module_name=func.__module__,
            qualname=func.__qualname__,
            doc=func.__doc__,
            critical_patterns=compiled_patterns if compiled_patterns else None,
            allow_repetitive=allow_repetitive,
        )
        setattr(func, "__rcm_stage__", meta)
        return func
    return decorator


class StateStore:
    """Thread-safe context store holding persistent objects across stages & retries."""

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self._lock = threading.RLock()
        self._store: Dict[str, Any] = dict(initial_state or {})
        self._params: Dict[str, Any] = {}
        self._once_cache: Dict[str, Tuple[Any, Tuple[Any, ...]]] = {}

    def once(self, key: str, fn: Callable[[], Any], deps: Tuple[Any, ...] = ()) -> Any:
        """
        D1: Memoizes expensive sub-steps within or across stages.
        If key exists in _once_cache and cached deps match deps, returns cached value.
        Otherwise executes fn(), caches (value, deps), and stores in _store[key].
        On retry, the sub-step is skipped unless invalidated via invalidate_once() or --fresh.
        """
        with self._lock:
            cached = self._once_cache.get(key)
            if cached is not None:
                cached_val, cached_deps = cached
                if cached_deps == tuple(deps):
                    return cached_val
            val = fn()
            self._once_cache[key] = (val, tuple(deps))
            self._store[key] = val
            return val

    def invalidate_once(self, key: Optional[str] = None) -> None:
        """Invalidates once-memoized cache entry or all entries if key is None."""
        with self._lock:
            if key is None:
                self._once_cache.clear()
            else:
                self._once_cache.pop(key, None)

    def get_once_keys(self) -> List[str]:
        """Returns list of currently satisfied once-memoized keys (D2)."""
        with self._lock:
            return list(self._once_cache.keys())
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value

    def update(self, mapping: Dict[str, Any]) -> None:
        with self._lock:
            self._store.update(mapping)

    def set_param(self, key: str, value: Any) -> None:
        with self._lock:
            self._params[key] = value
            self._store[key] = value

    def get_param(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._params.get(key, default)

    def mutate_params(self, params: Dict[str, Any]) -> None:
        with self._lock:
            self._params.update(params)
            self._store.update(params)

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())
    def set_ray_job(self, job_info: Any) -> None:
        with self._lock:
            self._store["ray_job"] = job_info

    def get_ray_job(self) -> Any:
        with self._lock:
            return self._store.get("ray_job")


    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            # Serialise json-serialisable items, format others as repr
            out = {}
            for k, v in self._store.items():
                try:
                    json.dumps(v)
                    out[k] = v
                except Exception:
                    out[k] = sanitize_value(v, name=k)
            return out

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._store[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._store


DEFAULT_CRITICAL_ANOMALY_PATTERNS: List[Tuple[re.Pattern, str]] = []


class JointTimingLogEngine:
    """Tracks hierarchical clocks, sliding log buffer, script-defined patterns, and evaluates execution normality."""

    def __init__(
        self,
        buffer_size: int = 30,
        telemetry_path: pathlib.Path = TELEMETRY_FILE,
        silence_timeout_s: float = 30.0,
    ):
        self.buffer_size = buffer_size
        self.telemetry_path = pathlib.Path(telemetry_path)
        self.silence_timeout_s = silence_timeout_s

        self._lock = threading.RLock()
        self.log_buffer: collections.deque[Dict[str, Any]] = collections.deque(maxlen=buffer_size)

        self.t_total_start: float = time.time()
        self.t_stage_start: float = self.t_total_start
        self.t_last_log: float = self.t_total_start
        self.t_last_progress: float = self.t_total_start
        self.current_stage: Optional[str] = None
        self.current_expected_s: float = 5.0
        self.warned_anomalies: Set[str] = set()

        # Script-defined critical patterns and repetition detection
        self.critical_patterns: List[Tuple[re.Pattern, str]] = list(DEFAULT_CRITICAL_ANOMALY_PATTERNS)
        self.suppressed_patterns: Set[str] = set()
        self.allow_repetitive: bool = False
        self.recent_signatures: collections.deque[Tuple[float, str]] = collections.deque(maxlen=40)
        self.pending_critical_anomaly: Optional[str] = None
        self.repetition_count: int = 0
        self.last_raw_message: str = ""

        self._ensure_telemetry_dir()

    def register_critical_pattern(
        self,
        pattern: Union[str, re.Pattern],
        reason: str = "Critical anomaly pattern matched",
    ) -> None:
        """Registers a domain-specific pattern that triggers an immediate DiagnosticTrap when matched."""
        with self._lock:
            compiled = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern
            self.critical_patterns.append((compiled, reason))

    def clear_critical_patterns(self) -> None:
        """Clears all registered critical anomaly patterns."""
        with self._lock:
            self.critical_patterns.clear()
            self.pending_critical_anomaly = None

    def suppress_pattern(self, pattern_str: str) -> None:
        """Suppresses a pattern string from triggering diagnostic traps (agent override)."""
        with self._lock:
            self.suppressed_patterns.add(pattern_str)

    def unsuppress_pattern(self, pattern_str: str) -> None:
        with self._lock:
            self.suppressed_patterns.discard(pattern_str)

    def set_allow_repetitive(self, allow: bool) -> None:
        """Controls whether repeating log messages count as genuine forward progress."""
        with self._lock:
            self.allow_repetitive = allow

    def _ensure_telemetry_dir(self) -> None:
        try:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def start_run(self) -> None:
        with self._lock:
            now = time.time()
            self.t_total_start = now
            self.t_stage_start = now
            self.t_last_log = now
            self.t_last_progress = now
            self.warned_anomalies.clear()
            self.pending_critical_anomaly = None
            self.repetition_count = 0

    def start_stage(self, stage_name: str, expected_s: float) -> None:
        with self._lock:
            now = time.time()
            self.current_stage = stage_name
            self.current_expected_s = expected_s
            self.t_stage_start = now
            self.t_last_log = now
            self.t_last_progress = now
            self.warned_anomalies.discard(stage_name)
            self.pending_critical_anomaly = None
            self.repetition_count = 0

    def log(self, message: str, level: str = "INFO", stage_name: Optional[str] = None) -> None:
        with self._lock:
            now = time.time()
            st_name = stage_name or self.current_stage or "GLOBAL"
            self.t_last_log = now
            self.last_raw_message = message
            entry = {
                "timestamp": now,
                "stage": st_name,
                "level": level.upper(),
                "message": message,
            }
            self.log_buffer.append(entry)
            try:
                with open(self.telemetry_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass

            # 1. Critical pattern matching
            for pattern, desc in self.critical_patterns:
                if pattern.search(message):
                    pat_str = pattern.pattern
                    if pat_str not in self.suppressed_patterns:
                        self.pending_critical_anomaly = f"Critical log pattern matched: '{desc}' [matched: '{message.strip()[:120]}']"
                        break

            # 2. Normalized log signature for repetition detection (strip hex, IPs, timestamps, numbers)
            norm_sig = re.sub(r"\b(0x[0-9a-fA-F]+|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|\d+:\d+:\d+|\d+)\b", "#", message)
            norm_sig = re.sub(r"\s+", " ", norm_sig).strip()

            recent_same = sum(1 for t, s in self.recent_signatures if s == norm_sig and (now - t) < 60.0)
            self.recent_signatures.append((now, norm_sig))

            # If repeating warnings/errors, treat as pseudo-activity (do NOT bump progress clock)
            if recent_same >= 2:
                self.repetition_count += 1
            else:
                self.repetition_count = 0
                self.t_last_progress = now

            self._write_live_status(now)

    def _write_live_status(self, now: Optional[float] = None) -> None:
        """Writes a compact, atomic live status and rolling log view for out-of-band inspection."""
        try:
            t_now = now or time.time()
            tail_formatted = []
            for entry in list(self.log_buffer)[-50:]:
                rel = entry["timestamp"] - t_now
                tail_formatted.append(f"[{rel:+.2f}s] [{entry['stage']}] [{entry['level']}] {entry['message']}")

            ray_job_data = None
            if RAY_JOB_FILE.exists():
                try:
                    with open(RAY_JOB_FILE, "r", encoding="utf-8") as f:
                        ray_job_data = json.load(f)
                except Exception:
                    pass

            status_obj = {
                "timestamp": t_now,
                "heartbeat_s": t_now,
                "current_stage": self.current_stage,
                "clocks": self.get_clocks(),
                "repetition_count": self.repetition_count,
                "allow_repetitive": self.allow_repetitive,
                "critical_patterns_count": len(self.critical_patterns),
                "suppressed_patterns": list(self.suppressed_patterns),
                "recent_logs_tail": tail_formatted[-20:],
                "ray_job": ray_job_data,
            }
            ensure_rcm_dir(BASE_RCM_DIR, 0o700)
            with open(LIVE_STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump(status_obj, f, indent=2)
            with open(LIVE_LOGS_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(tail_formatted) + "\n")
        except Exception:
            pass

    def warn(self, message: str, stage_name: Optional[str] = None) -> None:
        self.log(message, level="WARN", stage_name=stage_name)

    def error(self, message: str, stage_name: Optional[str] = None) -> None:
        self.log(message, level="ERROR", stage_name=stage_name)
    def touch(self) -> None:
        """Updates last log and progress timestamp to reset silence timer."""
        with self._lock:
            now = time.time()
            self.t_last_log = now
            self.t_last_progress = now

    def extend_stage(self, additional_s: float) -> None:
        """Extends the current stage expected duration after a resume action."""
        with self._lock:
            self.current_expected_s += max(additional_s, 2.0)
            now = time.time()
            self.t_last_log = now
            self.t_last_progress = now
            self.warned_anomalies.discard(self.current_stage)

    def get_clocks(self) -> Dict[str, float]:
        with self._lock:
            now = time.time()
            return {
                "t_total_s": round(now - self.t_total_start, 3),
                "t_stage_s": round(now - self.t_stage_start, 3),
                "t_expected_s": round(self.current_expected_s, 3),
                "t_since_last_log_s": round(now - self.t_last_log, 3),
                "t_since_last_progress_s": round(now - self.t_last_progress, 3),
            }

    def get_tail_logs(self, n: Optional[int] = None) -> List[str]:
        with self._lock:
            now = time.time()
            logs = list(self.log_buffer)
            if n:
                logs = logs[-n:]
            formatted = []
            for entry in logs:
                rel = entry["timestamp"] - now
                formatted.append(f"[{rel:+.2f}s] [{entry['stage']}] [{entry['level']}] {entry['message']}")
            return formatted

    def check_normality(self) -> Tuple[str, Optional[str]]:
        """
        Evaluates execution state against time-log contracts:
        - CRITICAL_PATTERN: Log stream matched fatal infrastructure pattern
        - REPETITIVE_STALL: Identical logs loop without genuine progress (pseudo-activity)
        - SILENCE_TIMEOUT: No log emitted for silence_timeout_s
        - SLA_BREACH: T_stage > 2.0 * T_expected
        - ANOMALY_WARNING: 1.25 * T_expected < T_stage <= 2.0 * T_expected
        """
        with self._lock:
            if not self.current_stage:
                return "HEALTHY", None

            now = time.time()
            elapsed_stage = now - self.t_stage_start
            silence_log_s = now - self.t_last_log
            silence_progress_s = now - self.t_last_progress
            expected = self.current_expected_s

            # 1. Critical pattern anomaly
            if self.pending_critical_anomaly:
                reason = self.pending_critical_anomaly
                self.pending_critical_anomaly = None
                return "CRITICAL_PATTERN", reason

            # 2. True silence timeout (zero logs emitted)
            if silence_log_s > self.silence_timeout_s:
                return "SILENCE_TIMEOUT", f"No log emitted for {silence_log_s:.1f}s (threshold: {self.silence_timeout_s:.1f}s)"

            # 3. Repetitive stall / pseudo-activity loop
            if not self.allow_repetitive and silence_progress_s > self.silence_timeout_s:
                return "REPETITIVE_STALL", (
                    f"Repetitive log stall detected for {silence_progress_s:.1f}s "
                    f"(repetition count: {self.repetition_count}, last: '{self.last_raw_message.strip()[:75]}...') "
                    f"without genuine progress."
                )

            # 4. SLA deadline monitoring
            if expected > 0:
                if elapsed_stage > 2.0 * expected:
                    return "SLA_BREACH", f"Stage '{self.current_stage}' elapsed {elapsed_stage:.2f}s exceeding 2.0x SLA ({expected * 2.0:.2f}s)"
                elif elapsed_stage > 1.25 * expected:
                    reason = f"Stage '{self.current_stage}' elapsed {elapsed_stage:.2f}s exceeding 1.25x expected ({expected * 1.25:.2f}s)"
                    if self.current_stage not in self.warned_anomalies:
                        self.warned_anomalies.add(self.current_stage)
                        return "ANOMALY_WARNING", reason
                    return "HEALTHY", None

            return "HEALTHY", None

class DiagnosticTrap:
    """Captures and persists diagnostic snapshots when execution stalls or faults."""

    def __init__(self, traps_dir: pathlib.Path = TRAPS_DIR):
        self.traps_dir = ensure_rcm_dir(traps_dir, 0o700)

    def write_snapshot(
        self,
        stage_name: str,
        reason: str,
        clocks: Dict[str, float],
        log_tail: List[str],
        state_store: StateStore,
        exc: Optional[BaseException] = None,
        live_traceback: Optional[str] = None,
        frame: Optional[Any] = None,
        allowed_actions: Optional[List[str]] = None,
    ) -> pathlib.Path:
        if allowed_actions is None:
            allowed_actions = ["RESUME", "RETRY", "MUTATE_CONFIG", "HOT_PATCH", "LIVE_PATCH", "SKIP_STAGE", "ABORT"]

        # State sample capped & redacted per A4 and Decision 2
        state_sample = extract_state_sample(state_store, max_keys=40, max_chars=200)

        # Frame analysis for B1, B2, B3, B4
        repo_root = get_repo_root()
        frame_analysis = analyze_frames(
            exc=exc,
            frame=frame,
            live_traceback=live_traceback,
            repo_root=repo_root,
        )

        tb_str = frame_analysis.get("full_traceback")
        if tb_str is None:
            if exc is not None:
                tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            elif live_traceback is not None:
                tb_str = live_traceback

        # Loaded modules snapshot for C1 diffing
        loaded_mods = snapshot_loaded_modules(repo_root)

        # Recurrence tracking (B5) in traps/history.json
        ensure_rcm_dir(self.traps_dir, 0o700)
        history_file = self.traps_dir / "history.json"
        history: Dict[str, Any] = {}
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = {}

        b_file = frame_analysis.get("blame_frame", {}).get("file", "unknown")
        b_line = frame_analysis.get("blame_frame", {}).get("line", 0)
        exc_type_name = type(exc).__name__ if exc else "DiagnosticTrap"
        sig_str = f"{stage_name}_{exc_type_name}_{b_file}:{b_line}"
        sig_hash = hashlib.sha1(sig_str.encode("utf-8")).hexdigest()

        rec_entry = history.get(sig_hash, {"count": 0})
        rec_count = rec_entry.get("count", 0) + 1
        history[sig_hash] = {
            "count": rec_count,
            "stage": stage_name,
            "exc_type": exc_type_name,
            "location": f"{b_file}:{b_line}",
            "last_time": time.time(),
        }
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception:
            pass

        snapshot = {
            "timestamp": time.time(),
            "stage": stage_name,
            "reason": reason,
            "clocks": clocks,
            "exception": {
                "type": type(exc).__name__ if exc else (
                    "CriticalPatternTrap" if "Critical" in reason or "pattern" in reason.lower() else (
                        "RepetitiveStallTrap" if "Repetitive" in reason or "loop" in reason.lower() else (
                            "WatchdogInterrupt" if "WATCHDOG" in reason or "SLA" in reason or "SILENCE" in reason else "Interrupt"
                        )
                    )
                ),
                "message": str(exc) if exc else reason,
                "traceback": tb_str,
            } if (exc or live_traceback or "Critical" in reason or "Repetitive" in reason) else None,
            "live_stack": live_traceback,
            "log_tail": log_tail,
            "state_keys": list(state_store.keys()) if (state_store is not None and hasattr(state_store, "keys")) else [],
            "once_keys": state_store.get_once_keys() if (state_store is not None and hasattr(state_store, "get_once_keys")) else [],
            "state_sample": state_sample,
            "blame_frame": frame_analysis.get("blame_frame"),
            "parent_frame": frame_analysis.get("parent_frame"),
            "first_party_traceback": frame_analysis.get("first_party_traceback", []),
            "source_window": frame_analysis.get("blame_frame", {}).get("source_window"),
            "loaded_modules": loaded_mods,
            "recurrence_count": rec_count,
            "recurrence_sig": sig_hash,
            "allowed_actions": allowed_actions,
        }
        ensure_rcm_dir(self.traps_dir, 0o700)
        stage_file = self.traps_dir / f"{stage_name}.json"
        with open(stage_file, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

        try:
            ensure_rcm_dir(LATEST_TRAP_FILE.parent, 0o700)
            with open(LATEST_TRAP_FILE, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
        except Exception:
            pass
        return stage_file

class SteeringReceiver:
    """Multi-transport steering receiver supporting stdin, unix socket, control file, and in-memory queue."""

    def __init__(
        self,
        socket_path: pathlib.Path = CONTROL_SOCK_FILE,
        action_file: pathlib.Path = CONTROL_ACTION_FILE,
        enable_socket: bool = True,
    ):
        self.socket_path = pathlib.Path(socket_path)
        self.action_file = pathlib.Path(action_file)
        self.enable_socket = enable_socket
        self.queue: collections.deque[Dict[str, Any]] = collections.deque()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._server_sock: Optional[socket.socket] = None
        self._sock_thread: Optional[threading.Thread] = None

        if self.enable_socket:
            self._start_socket_listener()

    def _start_socket_listener(self) -> None:
        try:
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            if self.socket_path.exists():
                try:
                    self.socket_path.unlink()
                except Exception:
                    pass

            self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server_sock.bind(str(self.socket_path))
            self._server_sock.listen(5)
            self._server_sock.settimeout(0.5)

            self._sock_thread = threading.Thread(target=self._socket_loop, daemon=True, name="rcm-steering-sock")
            self._sock_thread.start()
        except Exception:
            # Fall back gracefully if Unix domain socket creation is not permitted
            self._server_sock = None

    def _socket_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._server_sock is None:
                    break
                conn, _ = self._server_sock.accept()
                with conn:
                    data = conn.recv(65536).decode("utf-8")
                    if data:
                        try:
                            action = json.loads(data.strip())
                            self.inject_action(action)
                            conn.sendall(b'{"status": "RECEIVED"}\n')
                        except Exception as e:
                            conn.sendall(f'{{"status": "ERROR", "error": "{str(e)}"}}\n'.encode("utf-8"))
            except socket.timeout:
                continue
            except Exception:
                if self._stop_event.is_set():
                    break

    def flush(self) -> None:
        """Clears queued and pending actions."""
        with self._lock:
            self.queue.clear()
        if self.action_file.exists():
            try:
                self.action_file.unlink(missing_ok=True)
            except Exception:
                pass

    def inject_action(self, action: Dict[str, Any]) -> None:
        with self._lock:
            self.queue.append(action)

    def poll_action(self, timeout_s: float = 0.0) -> Optional[Dict[str, Any]]:
        """Polls queue, file, or stdin for incoming steering directives."""
        start_t = time.time()
        while True:
            # 1. Check in-memory queue
            with self._lock:
                if self.queue:
                    return self.queue.popleft()

            # 2. Check control action file
            if self.action_file.exists():
                try:
                    with open(self.action_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        action = json.loads(content)
                        # Only delete file if successfully parsed as JSON
                        self.action_file.unlink(missing_ok=True)
                        return action
                except (json.JSONDecodeError, OSError):
                    # Incomplete write or transient lock, retry on next poll
                    pass
                except Exception:
                    pass

            # 3. Check stdin non-blocking (works with ttys, ptys, pipes)
            if sys.stdin:
                try:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
                    if rlist:
                        line = sys.stdin.readline()
                        if line:
                            line_str = line.strip()
                            if line_str:
                                try:
                                    action = json.loads(line_str)
                                    return action
                                except json.JSONDecodeError:
                                    shorthand = line_str.upper()
                                    if shorthand in {"RESUME", "RETRY", "SKIP", "SKIP_STAGE", "ABORT"}:
                                        op = "SKIP_STAGE" if shorthand == "SKIP" else shorthand
                                        return {"op": op}
                except (ValueError, OSError):
                    pass

            elapsed = time.time() - start_t
            if elapsed >= timeout_s:
                break
            time.sleep(min(0.05, max(0.01, timeout_s - elapsed)))

        return None

    def close(self) -> None:
        self._stop_event.set()
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except Exception:
                pass


class WatchdogTimer:
    """Active background watchdog monitoring health, silence timeouts, and SIGUSR1 interrupts."""

    def __init__(
        self,
        timing_engine: JointTimingLogEngine,
        on_trap_trigger: Callable[[str, str], None],
        check_interval_s: float = 0.5,
    ):
        self.timing_engine = timing_engine
        self.on_trap_trigger = on_trap_trigger
        self.check_interval_s = check_interval_s

        self._stop_event = threading.Event()
        self._paused_event = threading.Event()
        self._interrupt_requested = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._orig_sigusr1 = None

    def start(self) -> None:
        self._stop_event.clear()
        self._paused_event.clear()
        self._interrupt_requested.clear()

        # Register SIGUSR1 if possible
        try:
            if hasattr(signal, "SIGUSR1") and threading.current_thread() is threading.main_thread():
                self._orig_sigusr1 = signal.getsignal(signal.SIGUSR1)
                signal.signal(signal.SIGUSR1, self._handle_sigusr1)
        except (ValueError, AttributeError):
            pass

        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="rcm-watchdog")
            self._thread.start()

    def _handle_sigusr1(self, signum: int, frame: Any) -> None:
        self._interrupt_requested.set()

    def trigger_interrupt(self) -> None:
        self._interrupt_requested.set()

    def pause(self) -> None:
        self._paused_event.set()

    def resume(self) -> None:
        self._paused_event.clear()
        self._interrupt_requested.clear()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.check_interval_s)
            if self._stop_event.is_set() or self._paused_event.is_set():
                continue

            try:
                self.timing_engine._write_live_status()
            except Exception:
                pass
            # Check explicit interrupt
            if self._interrupt_requested.is_set():
                self._interrupt_requested.clear()
                self.on_trap_trigger("SIGUSR1_INTERRUPT", "Process received SIGUSR1 signal for asymmetric steering")
                continue

            status, reason = self.timing_engine.check_normality()
            if status == "ANOMALY_WARNING" and reason:
                clocks = self.timing_engine.get_clocks()
                print(
                    f"[OMP_EVENT: ANOMALY_WARNING stage={self.timing_engine.current_stage} "
                    f"elapsed_s={clocks['t_stage_s']} expected_s={clocks['t_expected_s']} "
                    f"reason=\"{reason}\"]",
                    flush=True,
                )
            elif status in ("SLA_BREACH", "SILENCE_TIMEOUT", "CRITICAL_PATTERN", "REPETITIVE_STALL") and reason:
                self.on_trap_trigger(status, reason)

    def stop(self) -> None:
        self._stop_event.set()
        if self._orig_sigusr1 and hasattr(signal, "SIGUSR1"):
            try:
                if threading.current_thread() is threading.main_thread():
                    signal.signal(signal.SIGUSR1, self._orig_sigusr1)
            except (ValueError, AttributeError):
                pass


def _transplant_callable(old_func: Any, new_func: Any) -> bool:
    """Safely transplants bytecode and metadata from new_func into old_func in-place."""
    if not (inspect.isfunction(old_func) and inspect.isfunction(new_func)):
        return False
    if old_func.__code__.co_freevars == new_func.__code__.co_freevars:
        try:
            old_func.__code__ = new_func.__code__
            old_func.__defaults__ = new_func.__defaults__
            old_func.__kwdefaults__ = new_func.__kwdefaults__
            old_func.__annotations__ = getattr(new_func, "__annotations__", {})
            old_func.__doc__ = new_func.__doc__
            return True
        except Exception:
            return False
    return False


def apply_unified_diff(original_text: str, patch_text: str) -> str:
    """Applies a standard unified diff string to original_text."""
    orig_lines = original_text.splitlines(keepends=True)
    patch_lines = patch_text.splitlines(keepends=True)

    out_lines: List[str] = []
    orig_idx = 0
    i = 0
    n = len(patch_lines)

    while i < n and (patch_lines[i].startswith("---") or patch_lines[i].startswith("+++")):
        i += 1

    while i < n:
        line = patch_lines[i]
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if not m:
                i += 1
                continue
            hunk_orig_start = int(m.group(1)) - 1
            while orig_idx < hunk_orig_start and orig_idx < len(orig_lines):
                out_lines.append(orig_lines[orig_idx])
                orig_idx += 1
            i += 1
            while i < n and not patch_lines[i].startswith("@@"):
                hline = patch_lines[i]
                if hline.startswith("-"):
                    orig_idx += 1
                elif hline.startswith("+"):
                    out_lines.append(hline[1:])
                elif hline.startswith(" "):
                    if orig_idx < len(orig_lines):
                        out_lines.append(orig_lines[orig_idx])
                    orig_idx += 1
                i += 1
        else:
            i += 1

    while orig_idx < len(orig_lines):
        out_lines.append(orig_lines[orig_idx])
        orig_idx += 1

    return "".join(out_lines)

class StageRunner:
    """Unified Reverse Control Engine executing stages with discrete event logging & steering."""

    def __init__(
        self,
        checkpoint_path: pathlib.Path = CHECKPOINT_FILE,
        telemetry_path: pathlib.Path = TELEMETRY_FILE,
        traps_dir: pathlib.Path = TRAPS_DIR,
        action_file: pathlib.Path = CONTROL_ACTION_FILE,
        socket_path: pathlib.Path = CONTROL_SOCK_FILE,
        silence_timeout_s: float = 30.0,
        enable_steering_socket: bool = True,
        watchdog_interval_s: float = 0.5,
        trap_timeout_s: Optional[float] = 300.0,
    ):
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.silence_timeout_s = silence_timeout_s
        self.watchdog_interval_s = watchdog_interval_s
        env_timeout = os.environ.get("RCM_TRAP_TIMEOUT_S")
        if env_timeout:
            try:
                trap_timeout_s = float(env_timeout)
            except ValueError:
                pass
        self.trap_timeout_s = trap_timeout_s
        self.stages: Dict[str, StageMetadata] = {}
        self.state = StateStore()
        self.timing_engine = JointTimingLogEngine(
            telemetry_path=pathlib.Path(telemetry_path),
            silence_timeout_s=silence_timeout_s,
        )
        self.trap = DiagnosticTrap(traps_dir=pathlib.Path(traps_dir))
        self.steering = SteeringReceiver(
            socket_path=pathlib.Path(socket_path),
            action_file=pathlib.Path(action_file),
            enable_socket=enable_steering_socket,
        )
        self._trap_event = threading.Event()
        self._trap_reason: Optional[str] = None
        self._trap_status: Optional[str] = None
        self.watchdog = WatchdogTimer(
            timing_engine=self.timing_engine,
            on_trap_trigger=self._on_watchdog_trap,
            check_interval_s=watchdog_interval_s,
        )
        self._ray_supervisor: Optional[Any] = None

    @property
    def ray_supervisor(self) -> Any:
        """Lazy-initializes and returns the RayJobSupervisor tied to this StageRunner."""
        if self._ray_supervisor is None:
            from .ray_job import RayJobSupervisor
            self._ray_supervisor = RayJobSupervisor(
                timing_engine=self.timing_engine,
                trap=self.trap,
                steering=self.steering,
            )
        return self._ray_supervisor


        ensure_rcm_dir(self.checkpoint_path.parent, 0o700)

    def register(self, func_or_meta: Union[Callable[..., Any], StageMetadata]) -> None:
        if isinstance(func_or_meta, StageMetadata):
            self.stages[func_or_meta.name] = func_or_meta
        elif hasattr(func_or_meta, "__rcm_stage__"):
            meta = getattr(func_or_meta, "__rcm_stage__")
            self.stages[meta.name] = meta
        else:
            meta = StageMetadata(
                name=func_or_meta.__name__,
                order_idx=len(self.stages),
                expected_s=5.0,
                func=func_or_meta,
                module_name=func_or_meta.__module__,
                qualname=func_or_meta.__qualname__,
                doc=func_or_meta.__doc__,
            )
            self.stages[meta.name] = meta

    def auto_discover(self, module_or_dict: Union[Any, Dict[str, Any]]) -> None:
        """Discovers all @stage decorated functions in a module or global dict."""
        items = vars(module_or_dict).values() if not isinstance(module_or_dict, dict) else module_or_dict.values()
        for item in items:
            if callable(item) and hasattr(item, "__rcm_stage__"):
                self.register(item)

    def _on_watchdog_trap(self, status: str, reason: str) -> None:
        self._trap_status = status
        self._trap_reason = reason
        self._trap_event.set()

    def _write_checkpoint(
        self,
        current_stage: Optional[str],
        completed_stages: List[str],
        status: str = "RUNNING",
    ) -> None:
        now = time.time()
        elapsed_total = now - self.timing_engine.t_total_start
        data = {
            "timestamp": now,
            "status": status,
            "current_stage": current_stage,
            "completed_stages": completed_stages,
            "total_stages": len(self.stages),
            "elapsed_total_s": round(elapsed_total, 3),
            "state_keys": self.state.keys(),
        }
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def handle_hot_patch(self, module_name: Optional[str] = None, file_path: Optional[str] = None) -> bool:
        """Hot-reloads a module and refreshes stage functions and transplanted callables."""
        try:
            if file_path:
                target_p = pathlib.Path(file_path).resolve()
                for m_name, m_obj in list(sys.modules.items()):
                    if getattr(m_obj, "__file__", None) and pathlib.Path(m_obj.__file__).resolve() == target_p:
                        module_name = m_name
                        break
            if not module_name:
                self.timing_engine.error(f"Cannot resolve module for file: {file_path}")
                return False

            if module_name in sys.modules:
                mod = sys.modules[module_name]
                old_functions = {k: v for k, v in mod.__dict__.items() if inspect.isfunction(v)}
                importlib.invalidate_caches()
                if getattr(mod, "__spec__", None) is not None:
                    importlib.reload(mod)
                elif file_path or hasattr(mod, "__file__"):
                    target_file = file_path or getattr(mod, "__file__")
                    with open(target_file, "r", encoding="utf-8") as f:
                        code = compile(f.read(), str(target_file), "exec")
                    exec(code, mod.__dict__)
                else:
                    importlib.reload(mod)

                # Transplant callable bytecode so callers holding old function references update in-place
                for fn_name, old_fn in old_functions.items():
                    new_fn = mod.__dict__.get(fn_name)
                    if new_fn and inspect.isfunction(new_fn):
                        _transplant_callable(old_fn, new_fn)
            else:
                mod = importlib.import_module(module_name)

            # Re-bind registered stages from this module
            for stage_name, meta in list(self.stages.items()):
                if meta.module_name == module_name:
                    fn_name = meta.qualname.split(".")[-1]
                    if hasattr(mod, fn_name):
                        new_fn = getattr(mod, fn_name)
                        new_meta = getattr(new_fn, "__rcm_stage__", None)
                        if new_meta:
                            self.stages[stage_name] = new_meta
                        else:
                            meta.func = new_fn
            self.timing_engine.log(f"Hot-patched module '{module_name}' successfully", level="INFO")
            return True
        except Exception as e:
            self.timing_engine.error(f"Failed to hot-patch module '{module_name}': {e}")
            return False
    def _terminate_worker_thread(self, worker_thread: threading.Thread, timeout_s: float = 2.0) -> bool:
        """Terminates a worker thread by injecting TrapInterrupt (BaseException) and waiting for join."""
        if not worker_thread.is_alive() or not worker_thread.ident:
            return True
        try:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(worker_thread.ident),
                ctypes.py_object(TrapInterrupt),
            )
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(worker_thread.ident),
                    None,
                )
        except Exception:
            pass

        start_wait = time.time()
        while worker_thread.is_alive() and (time.time() - start_wait) < timeout_s:
            worker_thread.join(timeout=0.05)

        return not worker_thread.is_alive()

    def handle_live_patch(self, action: Dict[str, Any]) -> bool:
        """
        Applies in-memory code, callable transplantation, or disk patch live to the running system.
        Directive format:
          {
            "op": "LIVE_PATCH",
            "code": "def ...",               # Python source code string
            "stage": "stage_name",           # Optional stage name in runner
            "target": "pkg.mod.func",         # Optional dotted callable / attribute target
            "module": "mod_name",            # Optional module name
            "file": "path/to/file.py",       # Optional file path
            "patch": "@@ ... @@",            # Optional unified diff text
            "params": {...},                 # Optional StateStore parameter mutations
          }
        """
        if "params" in action and isinstance(action["params"], dict):
            self.state.mutate_params(action["params"])

        file_path_str = action.get("file") or action.get("file_path")
        patch_text = action.get("patch")
        code_str = action.get("code")
        stage_name = action.get("stage") or (
            self.timing_engine.current_stage
            if not action.get("target") and not action.get("module") and not file_path_str
            else None
        )
        target = action.get("target")
        module_name = action.get("module")
        target_label = target or stage_name or module_name or (str(file_path_str) if file_path_str else "live_patch")

        # 1. Disk-backed file patch or rewrite
        if file_path_str:
            try:
                f_path = pathlib.Path(file_path_str).resolve()
                if not f_path.exists():
                    err = f"Target patch file does not exist: {f_path}"
                    print(f"[OMP_EVENT: LIVE_PATCH_FAILED target={target_label} error=\"{err}\"]", flush=True)
                    self.timing_engine.error(f"LIVE_PATCH failed: {err}")
                    return False
                if patch_text:
                    orig_content = f_path.read_text(encoding="utf-8")
                    new_content = apply_unified_diff(orig_content, patch_text)
                    f_path.write_text(new_content, encoding="utf-8")
                elif code_str:
                    f_path.write_text(code_str, encoding="utf-8")

                mod_to_reload = module_name
                if not mod_to_reload:
                    for m_name, m_obj in list(sys.modules.items()):
                        if getattr(m_obj, "__file__", None) and pathlib.Path(m_obj.__file__).resolve() == f_path:
                            mod_to_reload = m_name
                            break
                if mod_to_reload:
                    self.handle_hot_patch(mod_to_reload, str(f_path))

                print(f"[OMP_EVENT: LIVE_PATCH_APPLIED target={target_label} stage={self.timing_engine.current_stage}]", flush=True)
                self.timing_engine.log(f"Live-patched file '{f_path}' successfully", level="INFO")
                return True
            except Exception as e:
                err = f"File patch exception: {e}"
                print(f"[OMP_EVENT: LIVE_PATCH_FAILED target={target_label} error=\"{err}\"]", flush=True)
                self.timing_engine.error(f"LIVE_PATCH failed: {err}")
                return False

        # 2. Pure in-memory code patch
        if code_str:
            safe_ident = re.sub(r"[^a-zA-Z0-9_]", "_", str(target_label))
            filename = f"<rcm_live_patch_{safe_ident}>"
            try:
                compiled = compile(code_str, filename, "exec")
            except Exception as e:
                err = f"Syntax/Compilation error: {e}"
                print(f"[OMP_EVENT: LIVE_PATCH_FAILED target={target_label} error=\"{err}\"]", flush=True)
                self.timing_engine.error(f"LIVE_PATCH compilation error: {err}")
                return False

            linecache.cache[filename] = (
                len(code_str),
                None,
                [line + "\n" for line in code_str.splitlines()],
                filename,
            )

            # A. Target is a registered stage
            if stage_name and stage_name in self.stages:
                st_meta = self.stages[stage_name]
                target_mod = sys.modules.get(st_meta.module_name)
                exec_env = dict(target_mod.__dict__) if target_mod else dict(globals())
                try:
                    exec(compiled, exec_env)
                except Exception as e:
                    err = f"Runtime execution error during live patch: {e}"
                    print(f"[OMP_EVENT: LIVE_PATCH_FAILED target={target_label} error=\"{err}\"]", flush=True)
                    self.timing_engine.error(f"LIVE_PATCH execution error: {err}")
                    return False

                fn_name = st_meta.qualname.split(".")[-1]
                new_fn = exec_env.get(fn_name) or exec_env.get(stage_name)
                if new_fn is None:
                    for k, v in exec_env.items():
                        if callable(v) and (hasattr(v, "__rcm_stage__") or (target_mod and k not in target_mod.__dict__)):
                            new_fn = v
                            break

                if new_fn is None or not callable(new_fn):
                    err = f"No replacement callable matching stage '{stage_name}' found in patch code"
                    print(f"[OMP_EVENT: LIVE_PATCH_FAILED target={target_label} error=\"{err}\"]", flush=True)
                    self.timing_engine.error(f"LIVE_PATCH error: {err}")
                    return False

                old_fn = st_meta.func
                if inspect.isfunction(old_fn) and inspect.isfunction(new_fn):
                    _transplant_callable(old_fn, new_fn)

                if hasattr(new_fn, "__rcm_stage__"):
                    new_meta = getattr(new_fn, "__rcm_stage__")
                    self.stages[stage_name] = new_meta
                else:
                    st_meta.func = new_fn

                if target_mod and hasattr(target_mod, fn_name):
                    setattr(target_mod, fn_name, new_fn)

                print(f"[OMP_EVENT: LIVE_PATCH_APPLIED target={target_label} stage={stage_name}]", flush=True)
                self.timing_engine.log(f"Live-patched stage '{stage_name}' successfully", level="INFO")
                return True

            # B. Target is a dotted path (e.g. pallas_ops.execute_pallas_fused_attention)
            elif target:
                parts = target.split(".")
                mod_obj = None
                parent = None
                attr_name = parts[-1]

                for i in range(len(parts), 0, -1):
                    cand_mod = ".".join(parts[:i])
                    if cand_mod in sys.modules:
                        mod_obj = sys.modules[cand_mod]
                        curr = mod_obj
                        valid = True
                        for p in parts[i:-1]:
                            curr = getattr(curr, p, None)
                            if curr is None:
                                valid = False
                                break
                        if valid:
                            parent = curr
                            break

                if parent is None:
                    main_mod = sys.modules.get("__main__")
                    if main_mod and hasattr(main_mod, parts[0]):
                        curr = main_mod
                        valid = True
                        for p in parts[:-1]:
                            curr = getattr(curr, p, None)
                            if curr is None:
                                valid = False
                                break
                        if valid:
                            parent = curr
                    elif parts[0] in globals():
                        curr = globals()
                        valid = True
                        for p in parts[:-1]:
                            curr = getattr(curr, p) if hasattr(curr, p) else (curr.get(p) if isinstance(curr, dict) else None)
                            if curr is None:
                                valid = False
                                break
                        if valid:
                            parent = curr

                if parent is None:
                    err = f"Target '{target}' could not be resolved in active Python runtime"
                    print(f"[OMP_EVENT: LIVE_PATCH_FAILED target={target_label} error=\"{err}\"]", flush=True)
                    self.timing_engine.error(f"LIVE_PATCH error: {err}")
                    return False

                old_obj = getattr(parent, attr_name, None) if hasattr(parent, attr_name) else (
                    parent.get(attr_name) if isinstance(parent, dict) else None
                )
                exec_env = dict(parent.__dict__) if hasattr(parent, "__dict__") else (
                    dict(mod_obj.__dict__) if mod_obj else dict(globals())
                )

                try:
                    exec(compiled, exec_env)
                except Exception as e:
                    err = f"Runtime execution error during target patch: {e}"
                    print(f"[OMP_EVENT: LIVE_PATCH_FAILED target={target_label} error=\"{err}\"]", flush=True)
                    self.timing_engine.error(f"LIVE_PATCH execution error: {err}")
                    return False

                new_fn = exec_env.get(attr_name)
                if new_fn is None:
                    for k, v in exec_env.items():
                        if callable(v) and (not hasattr(parent, k) and k != "__builtins__"):
                            new_fn = v
                            break

                if new_fn is None:
                    err = f"Callable '{attr_name}' was not defined in patch code"
                    print(f"[OMP_EVENT: LIVE_PATCH_FAILED target={target_label} error=\"{err}\"]", flush=True)
                    self.timing_engine.error(f"LIVE_PATCH error: {err}")
                    return False

                if old_obj and inspect.isfunction(old_obj) and inspect.isfunction(new_fn):
                    _transplant_callable(old_obj, new_fn)

                if hasattr(parent, attr_name):
                    setattr(parent, attr_name, new_fn)
                elif isinstance(parent, dict):
                    parent[attr_name] = new_fn

                for s_name, meta in self.stages.items():
                    if meta.func is old_obj or meta.qualname == target:
                        meta.func = new_fn

                print(f"[OMP_EVENT: LIVE_PATCH_APPLIED target={target_label} stage={self.timing_engine.current_stage}]", flush=True)
                self.timing_engine.log(f"Live-patched target '{target}' successfully", level="INFO")
                return True

            # C. Module-level in-memory execution with transactional rollback
            elif module_name:
                mod = sys.modules.get(module_name)
                if mod is None:
                    try:
                        mod = importlib.import_module(module_name)
                    except Exception as e:
                        err = f"Module '{module_name}' could not be imported: {e}"
                        print(f"[OMP_EVENT: LIVE_PATCH_FAILED target={target_label} error=\"{err}\"]", flush=True)
                        self.timing_engine.error(f"LIVE_PATCH error: {err}")
                        return False

                backup_dict = dict(mod.__dict__)
                try:
                    exec(compiled, mod.__dict__)
                except Exception as e:
                    mod.__dict__.clear()
                    mod.__dict__.update(backup_dict)
                    err = f"Execution error in module '{module_name}': {e}"
                    print(f"[OMP_EVENT: LIVE_PATCH_FAILED target={target_label} error=\"{err}\"]", flush=True)
                    self.timing_engine.error(f"LIVE_PATCH execution error: {err}")
                    return False

                for k, v in mod.__dict__.items():
                    if k in backup_dict and inspect.isfunction(backup_dict[k]) and inspect.isfunction(v):
                        _transplant_callable(backup_dict[k], v)

                for s_name, meta in list(self.stages.items()):
                    if meta.module_name == module_name:
                        fn_name = meta.qualname.split(".")[-1]
                        if hasattr(mod, fn_name):
                            new_fn = getattr(mod, fn_name)
                            new_meta = getattr(new_fn, "__rcm_stage__", None)
                            if new_meta:
                                self.stages[s_name] = new_meta
                            else:
                                meta.func = new_fn

                print(f"[OMP_EVENT: LIVE_PATCH_APPLIED target={target_label} stage={self.timing_engine.current_stage}]", flush=True)
                self.timing_engine.log(f"Live-patched module '{module_name}' successfully", level="INFO")
                return True

        err = "LIVE_PATCH directive must supply either 'code' (with 'stage', 'target', or 'module') or 'file'/'patch'"
        print(f"[OMP_EVENT: LIVE_PATCH_FAILED target={target_label} error=\"{err}\"]", flush=True)
        self.timing_engine.error(f"LIVE_PATCH error: {err}")
        return False

    def enter_diagnostic_trap(
        self,
        stage_name: str,
        reason: str,
        exc: Optional[BaseException] = None,
        live_traceback: Optional[str] = None,
        frame: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Pauses stage execution, dumps diagnostic snapshot, emits discrete trap event,
        and blocks on the steering channel until a resolution directive is received.
        """
        self.watchdog.pause()
        clocks = self.timing_engine.get_clocks()
        log_tail = self.timing_engine.get_tail_logs(30)

        trap_path = self.trap.write_snapshot(
            stage_name=stage_name,
            reason=reason,
            clocks=clocks,
            log_tail=log_tail,
            state_store=self.state,
            exc=exc,
            live_traceback=live_traceback,
            frame=frame,
        )

        print(
            f"[OMP_EVENT: TRAP_PAUSED stage={stage_name} elapsed_s={clocks['t_stage_s']} "
            f"expected_s={clocks['t_expected_s']} reason=\"{reason}\" trap_file=\"{trap_path}\"]",
            flush=True,
        )

        # Block until steering directive is received
        action = None
        start_wait = time.time()
        while action is None:
            action = self.steering.poll_action(timeout_s=0.2)
            if self.trap_timeout_s is not None and (time.time() - start_wait) > self.trap_timeout_s:
                raise AbortExecution(f"Trap timeout expired ({self.trap_timeout_s}s) waiting for steering directive")

        op = action.get("op", "RESUME").upper()
        print(f"[OMP_EVENT: TRAP_RESOLVED stage={stage_name} action={op}]", flush=True)

        self._trap_event.clear()
        self._trap_reason = None
        self._trap_status = None
        self.watchdog.resume()
        if action.get("fresh", False):
            if hasattr(self.state, "invalidate_once"):
                self.state.invalidate_once()
        return action

    def run(self) -> bool:
        """Executes all registered stages in defined order with reverse control steering."""
        sorted_stages = sorted(self.stages.values(), key=lambda s: (s.order_idx, s.name))
        completed_stages: List[str] = []

        self.steering.flush()
        self.timing_engine.start_run()
        self.watchdog.start()

        print(f"[OMP_EVENT: RUN_STARTED total_stages={len(sorted_stages)}]", flush=True)
        self._write_checkpoint(current_stage=None, completed_stages=completed_stages, status="RUNNING")

        stage_idx = 0
        try:
            while stage_idx < len(sorted_stages):
                st = sorted_stages[stage_idx]
                stage_name = st.name
                expected_s = st.expected_s

                print(
                    f"[OMP_EVENT: STAGE_ENTERED stage={stage_name} stage_idx={stage_idx + 1} "
                    f"expected_s={expected_s}]",
                    flush=True,
                )

                # Retry loop for the current stage
                retry_stage = True
                while retry_stage:
                    retry_stage = False
                    skip_stage = False
                    st = self.stages[stage_name]
                    self._trap_event.clear()
                    self.timing_engine.clear_critical_patterns()
                    if st.critical_patterns:
                        for pat, reason in st.critical_patterns:
                            self.timing_engine.register_critical_pattern(pat, reason)
                    self.timing_engine.set_allow_repetitive(st.allow_repetitive)

                    self.timing_engine.start_stage(stage_name, expected_s)
                    self._write_checkpoint(current_stage=stage_name, completed_stages=completed_stages, status="RUNNING")
                    # Worker execution wrapper
                    worker_res: Dict[str, Any] = {"exc": None, "completed": False}

                    def run_worker_fn():
                        try:
                            sig = inspect.signature(st.func)
                            if len(sig.parameters) == 0:
                                st.func()
                            elif len(sig.parameters) == 1:
                                st.func(self.state)
                            else:
                                st.func(self.state, self.timing_engine)
                            worker_res["completed"] = True
                        except TrapInterrupt:
                            worker_res["exc"] = TrapInterrupt("Worker interrupted by RCM steering directive")
                        except BaseException as ex:
                            worker_res["exc"] = ex
                    worker_thread = threading.Thread(
                        target=run_worker_fn,
                        daemon=True,
                        name=f"rcm-worker-{stage_name}",
                    )
                    worker_thread.start()

                    # Supervisor loop actively monitoring worker and trap signals
                    while worker_thread.is_alive():
                        worker_thread.join(timeout=0.05)

                        # Check for in-flight steering directives while worker is running
                        if not self._trap_event.is_set():
                            in_flight_action = self.steering.poll_action(timeout_s=0.0)
                            if in_flight_action:
                                op = in_flight_action.get("op", "").upper()
                                if op == "LIVE_PATCH":
                                    self._terminate_worker_thread(worker_thread)
                                    patch_ok = self.handle_live_patch(in_flight_action)
                                    if patch_ok:
                                        st = self.stages[stage_name]
                                        retry_stage = True
                                        break
                                elif op == "MUTATE_CONFIG":
                                    self._terminate_worker_thread(worker_thread)
                                    if "params" in in_flight_action:
                                        self.state.mutate_params(in_flight_action["params"])
                                    retry_stage = True
                                    break
                                elif op == "HOT_PATCH":
                                    self._terminate_worker_thread(worker_thread)
                                    files = in_flight_action.get("files")
                                    if files and isinstance(files, list):
                                        for f_item in files:
                                            m_n = f_item if isinstance(f_item, str) and not f_item.endswith(".py") else None
                                            f_p = f_item if isinstance(f_item, str) and (f_item.endswith(".py") or "/" in f_item) else None
                                            self.handle_hot_patch(m_n, f_p)
                                    else:
                                        mod_name = in_flight_action.get("module")
                                        f_path = in_flight_action.get("file") or in_flight_action.get("file_path")
                                        self.handle_hot_patch(mod_name, f_path)
                                    st = self.stages[stage_name]
                                    retry_stage = True
                                    break
                                elif op == "RETRY":
                                    self._terminate_worker_thread(worker_thread)
                                    retry_stage = True
                                    break
                                elif op in ("SKIP", "SKIP_STAGE"):
                                    self._terminate_worker_thread(worker_thread)
                                    skip_stage = True
                                    break
                                elif op == "ABORT":
                                    self._terminate_worker_thread(worker_thread)
                                    raise AbortExecution("Aborted by in-flight steering directive")

                        if self._trap_event.is_set():
                            # Watchdog or Signal triggered trap while worker is running
                            live_tb = None
                            if worker_thread.ident:
                                frame = sys._current_frames().get(worker_thread.ident)
                                if frame:
                                    live_tb = "".join(traceback.format_stack(frame))

                            action = self.enter_diagnostic_trap(
                                stage_name=stage_name,
                                reason=self._trap_reason or "WATCHDOG_INTERRUPT",
                                live_traceback=live_tb,
                                frame=frame,
                            )
                            op = action.get("op", "RESUME").upper()
                            if op == "RESUME":
                                # Worker thread is still alive and running: grant extension & touch
                                self.timing_engine.extend_stage(self.timing_engine.current_expected_s)
                                continue
                            else:
                                # For RETRY, HOT_PATCH, LIVE_PATCH, MUTATE_CONFIG, SKIP, ABORT: terminate stale thread
                                self._terminate_worker_thread(worker_thread)

                                if op == "RETRY":
                                    retry_stage = True
                                    break
                                elif op == "LIVE_PATCH":
                                    patch_ok = self.handle_live_patch(action)
                                    if patch_ok:
                                        st = self.stages[stage_name]
                                    retry_stage = True
                                    break
                                elif op == "MUTATE_CONFIG":
                                    if "params" in action:
                                        self.state.mutate_params(action["params"])
                                    if "suppress_patterns" in action:
                                        for pat in action["suppress_patterns"]:
                                            self.timing_engine.suppress_pattern(pat)
                                    if "allow_repetitive" in action:
                                        self.timing_engine.set_allow_repetitive(bool(action["allow_repetitive"]))
                                    if "critical_patterns" in action:
                                        for item in action["critical_patterns"]:
                                            pat = item[0] if isinstance(item, (list, tuple)) else item
                                            reason = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else "Custom critical pattern"
                                            self.timing_engine.register_critical_pattern(pat, reason)
                                    retry_stage = True
                                    break
                                elif op == "HOT_PATCH":
                                    mod_name = action.get("module")
                                    f_path = action.get("file") or action.get("file_path")
                                    self.handle_hot_patch(mod_name, f_path)
                                    st = self.stages[stage_name]
                                    retry_stage = True
                                    break
                                elif op in ("SKIP", "SKIP_STAGE"):
                                    skip_stage = True
                                    break
                                elif op == "ABORT":
                                    raise AbortExecution("Aborted by steering directive")
                    # If retry_stage was set in supervisor loop, restart this stage
                    if retry_stage:
                        continue

                    # If skip_stage was set in supervisor loop, break retry loop to advance
                    if skip_stage:
                        print(f"[OMP_EVENT: STAGE_SKIPPED stage={stage_name}]", flush=True)
                        break

                    # Worker finished execution: check for exceptions
                    exc = worker_res.get("exc")
                    if exc is not None and not isinstance(exc, TrapInterrupt):
                        if isinstance(exc, (AbortExecution, KeyboardInterrupt)):
                            raise exc
                        if isinstance(exc, SkipStage):
                            print(f"[OMP_EVENT: STAGE_SKIPPED stage={stage_name}]", flush=True)
                            skip_stage = True
                            break

                        action = self.enter_diagnostic_trap(
                            stage_name=stage_name,
                            reason=f"EXCEPTION: {type(exc).__name__}: {str(exc)}",
                            exc=exc,
                        )
                        op = action.get("op", "RETRY").upper()
                        if op in ("RETRY", "RESUME"):
                            retry_stage = True
                            continue
                        elif op == "LIVE_PATCH":
                            patch_ok = self.handle_live_patch(action)
                            if patch_ok:
                                st = self.stages[stage_name]
                            retry_stage = True
                            continue
                        elif op == "MUTATE_CONFIG":
                            if "params" in action:
                                self.state.mutate_params(action["params"])
                            retry_stage = True
                            continue
                        elif op == "HOT_PATCH":
                            files = action.get("files")
                            if files and isinstance(files, list):
                                for f_item in files:
                                    m_n = f_item if isinstance(f_item, str) and not f_item.endswith(".py") else None
                                    f_p = f_item if isinstance(f_item, str) and (f_item.endswith(".py") or "/" in f_item) else None
                                    self.handle_hot_patch(m_n, f_p)
                            else:
                                mod_name = action.get("module")
                                f_path = action.get("file") or action.get("file_path")
                                self.handle_hot_patch(mod_name, f_path)
                            st = self.stages[stage_name]
                            retry_stage = True
                            continue
                        elif op in ("SKIP", "SKIP_STAGE"):
                            skip_stage = True
                            break
                        elif op == "ABORT":
                            raise AbortExecution("Aborted by steering directive")
                        else:
                            retry_stage = True
                            continue

                if not skip_stage:
                    completed_stages.append(stage_name)
                    stage_clocks = self.timing_engine.get_clocks()
                    print(f"[OMP_EVENT: STAGE_SETTLED stage={stage_name} elapsed_s={stage_clocks['t_stage_s']}]", flush=True)

                stage_idx += 1

            total_clocks = self.timing_engine.get_clocks()
            self._write_checkpoint(current_stage=None, completed_stages=completed_stages, status="SUCCESS")
            print(f"[OMP_EVENT: RUN_FINISHED status=SUCCESS total_elapsed_s={total_clocks['t_total_s']}]", flush=True)
            return True
        except (AbortExecution, KeyboardInterrupt) as ab_exc:
            total_clocks = self.timing_engine.get_clocks()
            self._write_checkpoint(current_stage=None, completed_stages=completed_stages, status="ABORTED")
            reason_suffix = " reason=TRAP_TIMEOUT" if "Trap timeout expired" in str(ab_exc) else ""
            print(f"[OMP_EVENT: RUN_FINISHED status=ABORTED total_elapsed_s={total_clocks['t_total_s']}{reason_suffix}]", flush=True)
            return False
        except Exception:
            traceback.print_exc()
            total_clocks = self.timing_engine.get_clocks()
            self._write_checkpoint(current_stage=None, completed_stages=completed_stages, status="FAILED")
            print(f"[OMP_EVENT: RUN_FINISHED status=FAILED total_elapsed_s={total_clocks['t_total_s']}]", flush=True)
            return False
        finally:
            self.watchdog.stop()
            self.steering.close()
            if self._ray_supervisor:
                try:
                    self._ray_supervisor.stop_all_jobs()
                except Exception:
                    pass
            try:
                import ray
                from ray.util.placement_group import placement_group_table, remove_placement_group
                if ray.is_initialized():
                    for pg_id, pg_info in placement_group_table().items():
                        if pg_info.get("state") in ["CREATED", "PENDING"]:
                            try:
                                remove_placement_group(ray.util.placement_group(pg_id))
                            except Exception:
                                pass
            except Exception:
                pass
