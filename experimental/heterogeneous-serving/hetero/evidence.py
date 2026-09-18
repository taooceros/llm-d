"""Structured evidence emission for RCM-supervised Ray jobs.

The RCM ``RayJobSupervisor`` scrapes a single measurement payload out of the
remote job log using the banner below (see
``omp_rcm/ray_job.py::RayJobSupervisor.extract_json_measurements``).  Every
remote entry point in :mod:`hetero` emits exactly one such payload plus a
stream of ``[OMP_EVENT: ...]`` lines for live supervision.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict

BANNER = "EMPIRICAL TPU HARDWARE MEASUREMENTS (JSON OUTPUT)"
_RULE = "=" * 78

_T0 = time.time()


def emit_event(__event: str, **fields: Any) -> None:
    """Emits one structured RCM telemetry event line.

    The event name is positional-only so callers may pass a ``name`` field.
    """
    if fields:
        payload = json.dumps(fields, separators=(",", ":"), default=str)
        print(f"[OMP_EVENT: {__event} {payload}]", flush=True)
    else:
        print(f"[OMP_EVENT: {__event}]", flush=True)


def emit_log(message: str, level: str = "INFO") -> None:
    print(f"[{level}] +{time.time() - _T0:8.2f}s {message}", flush=True)


def emit_measurements(payload: Dict[str, Any]) -> None:
    """Emits the single scraped measurement payload for this job."""
    body = json.dumps(payload, separators=(",", ":"), default=str)
    out = sys.stdout
    out.write(f"\n{_RULE}\n{BANNER}\n{_RULE}\n{body}\n\n")
    out.flush()
    emit_event("MEASUREMENTS_EMITTED", bytes=len(body))
