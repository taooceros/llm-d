"""Per-run snapshot and executable-bank namespaces for first-switch (cold) runs.

A layout's first switch writes its weight snapshot and executable bank inside the
switch deploy. To emulate "first" without wiping warm state, each run writes both
into a namespace of its own and removes exactly that namespace afterwards:

    snapshot  /dev/shm/omp_weights_runs/<scope>/<layout>/<instance>/rank<k>/
    bank      /root/.cache/omp_exec_bank_runs/<scope>/<identity digest>/rank<k>/

The default roots (``/dev/shm/omp_weights``, ``/root/.cache/omp_exec_bank``) and the
persistent compile cache are never inside a scope, so scope cleanup cannot reach them.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict

SNAPSHOT_RUNS_PARENT = "/dev/shm/omp_weights_runs"
BANK_RUNS_PARENT = "/root/.cache/omp_exec_bank_runs"
_SCOPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def check_scope(scope: str) -> str:
    if not isinstance(scope, str) or not _SCOPE.fullmatch(scope) or ".." in scope:
        raise ValueError(f"invalid run scope {scope!r}: [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}, no '..'")
    return scope


def scope_roots(scope: str) -> Dict[str, str]:
    """{"weight_snapshot_root", "exec_bank_dir"} of one run scope (engine-setting names)."""
    check_scope(scope)
    return {"weight_snapshot_root": str(Path(SNAPSHOT_RUNS_PARENT) / scope),
            "exec_bank_dir": str(Path(BANK_RUNS_PARENT) / scope)}


def _tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())


def scope_files(scope: str) -> Dict[str, Any]:
    """Completeness of one scope on this host: snapshot manifests + shard sizes, bank manifests."""
    import json

    roots = {k: Path(v) for k, v in scope_roots(scope).items()}
    snap, bank = roots["weight_snapshot_root"], roots["exec_bank_dir"]
    manifests = []
    for path in sorted(snap.glob("*/*/rank*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("version") != 2 or not manifest.get("arrays"):
            raise RuntimeError(f"incomplete snapshot manifest: {path}")
        for array in manifest["arrays"]:
            for shard in array["shards"]:
                file = path.parent / shard["file"]
                if file.parent != path.parent or not file.is_file() or file.stat().st_size != shard["bytes"]:
                    raise RuntimeError(f"missing/wrong-sized snapshot shard: {file}")
        manifests.append({"path": str(path), "generation": manifest["generation"],
                          "arrays": len(manifest["arrays"]),
                          "bytes": sum(s["bytes"] for a in manifest["arrays"] for s in a["shards"])})
    banks = []
    for path in sorted(bank.glob("*/rank*/manifest.json")):
        manifest = json.loads(path.read_text())
        banks.append({"path": str(path), "identity_digest": manifest.get("identity_digest"),
                      "entries": len(manifest.get("entries") or {}),
                      "skipped": len(manifest.get("skipped") or []),
                      "bytes": sum(e["bytes"] for e in (manifest.get("entries") or {}).values())})
    return {"scope": scope, "snapshot_manifests": manifests, "bank_manifests": banks}


def remove_scope(scope: str) -> Dict[str, Any]:
    """Delete this host's two directories of ``scope`` (nothing else); returns bytes removed."""
    removed = {}
    for name, root in scope_roots(scope).items():
        path = Path(root)
        if path.is_symlink():
            raise RuntimeError(f"refusing to remove a symlinked scope root: {path}")
        existed = path.exists()
        size = _tree_bytes(path) if existed else 0
        if existed:
            shutil.rmtree(path)
        removed[name] = {"path": str(path), "existed": existed, "bytes": size,
                         "absent_after": not path.exists()}
    return removed
