"""Hash-pinned, offline native FP32-state source preparation.

No model/backend modules are imported here. Preparation changes files, not live
Python objects: serving processes must start only after successful preparation.
Replacement is atomic per file, not a distributed transaction. A failed apply
must be retried or repaired before any serving process is started.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Sequence


_PATCH_BYTES = Path(__file__).with_name("native_backend_patch.json").read_bytes()
BACKEND_REVISION = hashlib.sha256(_PATCH_BYTES).hexdigest()
_PATCH = json.loads(_PATCH_BYTES)
if _PATCH["format"] != 1:
    raise RuntimeError("Unsupported native backend patch format")

# The Ray working directory is immutable per submission. Capture its serving
# code identity once so attachments cannot mix old actors with new control code.
_SOURCE_ROOT = Path(__file__).parent
RUNTIME_REVISION = hashlib.sha256(json.dumps({
    str(path.relative_to(_SOURCE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(_SOURCE_ROOT.rglob("*"))
    if path.is_file() and path.suffix in (".py", ".json")
}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class BackendPreparationError(RuntimeError):
    """A rejected preparation, retaining every available per-node receipt."""

    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


def _sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def _inspect(root: Path | None = None) -> dict[str, Any]:
    files = []
    for patch in _PATCH["files"]:
        path = Path(patch["path"])
        if root is not None:
            path = root / path.relative_to("/")
        item = {
            "name": patch["name"],
            "path": str(path),
            "original_sha256": patch["original_sha256"],
            "patched_sha256": patch["patched_sha256"],
        }
        try:
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise RuntimeError("Native source must be a regular, non-symlink file")
            actual = _sha256(path.read_bytes())
            item["actual_sha256"] = actual
            item["state"] = ("patched" if actual == patch["patched_sha256"] else
                             "original" if actual == patch["original_sha256"] else
                             "unknown")
        except (OSError, RuntimeError) as exc:
            item.update(state="unavailable", actual_sha256=None, error=str(exc))
        files.append(item)
    return {
        "backend_revision": BACKEND_REVISION,
        "runtime_revision": RUNTIME_REVISION,
        "ready": all(item["state"] == "patched" for item in files),
        "compatible": all(item["state"] in ("original", "patched") for item in files),
        "files": files,
    }


def verify_backend() -> dict[str, Any]:
    """Require exact patched on-disk sources before importing native backends."""
    evidence = _inspect()
    if not evidence["ready"]:
        raise BackendPreparationError(
            "Native FP32 backend is not prepared; run backend preparation before serving",
            evidence,
        )
    return evidence


def _apply_local(root: Path | None = None) -> dict[str, Any]:
    before = _inspect(root)
    if not before["compatible"]:
        raise BackendPreparationError("Unknown native source; no files changed", before)
    staged = []
    try:
        # Validate and stage every local output before replacing any source.
        for patch, item in zip(_PATCH["files"], before["files"]):
            if item["state"] == "patched":
                continue
            path = Path(item["path"])
            original = path.read_bytes()
            if _sha256(original) != patch["original_sha256"]:
                raise BackendPreparationError("Native source changed after preflight", _inspect(root))
            output = original.decode("utf-8")
            for replacement in patch["replacements"]:
                if output.count(replacement["old"]) != 1:
                    raise RuntimeError(f"Non-unique native replacement in {path}")
                output = output.replace(replacement["old"], replacement["new"], 1)
            data = output.encode("utf-8")
            if _sha256(data) != patch["patched_sha256"]:
                raise RuntimeError(f"Native patch output hash mismatch for {path}")
            ast.parse(output, filename=str(path))
            fd, name = tempfile.mkstemp(prefix=f".{path.name}.fp32-", dir=path.parent)
            temporary = Path(name)
            staged.append((path, temporary, patch["original_sha256"]))
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        current = _inspect(root)
        if current["files"] != before["files"]:
            raise BackendPreparationError("Native sources changed during staging", current)
        for path, temporary, original_hash in staged:
            if path.is_symlink() or _sha256(path.read_bytes()) != original_hash:
                raise BackendPreparationError("Native source changed before replacement", _inspect(root))
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        for _, temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    evidence = _inspect(root)
    if not evidence["ready"]:
        raise BackendPreparationError("Native source verification failed after apply", evidence)
    return evidence


def _node_operation(apply: bool, expected_revision: str) -> dict[str, Any]:
    import ray

    identity = {
        "node_id": ray.get_runtime_context().get_node_id(),
        "host_ip": ray.util.get_node_ip_address(),
    }
    try:
        if expected_revision != BACKEND_REVISION:
            raise BackendPreparationError("Native patch revision differs on this node", _inspect())
        evidence = _apply_local() if apply else _inspect()
        return {**evidence, **identity}
    except Exception as exc:
        evidence = exc.evidence if isinstance(exc, BackendPreparationError) else _inspect()
        return {**evidence, **identity, "error": str(exc), "compatible": False, "ready": False}


def prepare_backend(host_ips: Sequence[str], *, apply: bool = False) -> dict[str, Any]:
    """Check all target hosts and the Ray head, then optionally apply the patch.

    Tasks are hard-pinned to nodes, reserve no accelerator resources, and import
    no accelerator runtime. Every preflight result must be compatible before
    the first apply task is submitted. This requires an already connected Ray
    session; it neither starts a cluster nor creates placement groups/engines.
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if not ray.is_initialized():
        raise RuntimeError("Connect to Ray before preparing the native backend")
    if isinstance(host_ips, str) or not host_ips:
        raise ValueError("At least one target host IP is required")
    nodes = [node for node in ray.nodes() if node["Alive"]]
    heads = [node for node in nodes if "node:__internal_head__" in node["Resources"]]
    if len(heads) != 1:
        raise RuntimeError("Cannot identify exactly one live Ray head")
    targets = {heads[0]["NodeID"]: heads[0]}
    for host_ip in sorted(set(host_ips)):
        matches = [node for node in nodes if node["NodeManagerAddress"] == host_ip]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one live Ray node for {host_ip}, found {len(matches)}")
        targets[matches[0]["NodeID"]] = matches[0]
    ordered = sorted(targets.values(), key=lambda node: (node["NodeManagerAddress"], node["NodeID"]))
    operation = ray.remote(num_cpus=0, num_gpus=0, max_retries=0)(_node_operation)

    def collect(do_apply: bool) -> list[dict[str, Any]]:
        pending = [(node, operation.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
        ).remote(do_apply, BACKEND_REVISION)) for node in ordered]
        results = []
        for node, reference in pending:
            try:
                receipt = ray.get(reference, timeout=120)
                if (receipt["node_id"] != node["NodeID"]
                        or receipt["host_ip"] != node["NodeManagerAddress"]
                        or receipt["runtime_revision"] != RUNTIME_REVISION):
                    raise RuntimeError("Native preflight node or runtime revision differs")
            except Exception as exc:
                ray.cancel(reference, force=True)
                receipt = {"node_id": node["NodeID"], "host_ip": node["NodeManagerAddress"],
                           "backend_revision": BACKEND_REVISION, "compatible": False,
                           "ready": False, "files": [], "error": str(exc)}
            receipt["is_head"] = node["NodeID"] == heads[0]["NodeID"]
            results.append(receipt)
        return results

    preflight = collect(False)
    evidence = {"backend_revision": BACKEND_REVISION, "apply": apply,
                "preflight": preflight, "nodes": preflight,
                "ready": all(node["ready"] for node in preflight)}
    if not all(node["compatible"] and not node.get("error") for node in preflight):
        raise BackendPreparationError("Native backend preflight failed; apply was not started", evidence)
    if apply:
        evidence["nodes"] = collect(True)
        evidence["ready"] = all(node["ready"] and not node.get("error") for node in evidence["nodes"])
        if not evidence["ready"]:
            raise BackendPreparationError("Native backend apply failed; do not start serving", evidence)
    return evidence
