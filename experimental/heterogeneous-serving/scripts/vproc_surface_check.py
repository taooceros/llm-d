#!/usr/bin/env python3
"""Static scan of tpu-inference and the vLLM TPU path for device/process-surface uses (plan C3).

    python scripts/vproc_surface_check.py [--tpu-inference DIR] [--vllm DIR] [--out JSON]

Finds engine-reachable uses of: ``jax.devices/local_devices/device_count/
local_device_count/process_index/process_count/process_indices``, ``xla_bridge``
(``xb.*``), ``jax.distributed.*`` and ``global_state``, ``multihost_utils.*``, device
``.process_index``, ``from jax import devices``-style bindings, implicit
physical-device defaults (``jax.make_mesh`` without ``devices=``), TPU env writes,
``jax.clear_caches`` and compilation-cache configuration. Each site is classified:

  covered    its API is a key of ``hetero.vproc.view.VIRTUALIZED`` (or the pinned list)
  guarded    ``hetero.vproc.engine_vllm`` has a guard for it (names verified at run time)
  harmless   cannot observe or change process/device identity (reason given)
  UNCOVERED  none of the above

Exit 1 if an UNCOVERED site is in a file on the engine path. Output: JSON with
file:line, snippet, class, fork SHA per site (default results/vproc/surface-check.json).
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[1]

# view.VIRTUALIZED keys as pinned in the C2/C3 contract; used when view.py is absent.
PINNED_VIRTUALIZED = (
    "jax.devices", "jax.device_count", "jax.local_devices",
    "jax.process_index", "jax.process_count", "jax.process_indices",
    "jax.experimental.multihost_utils", "tpu_inference.utils.xb",
    "tpu_inference.distributed.utils.get_device_topology_order_id",
    "jax._src.compiler._cache_write", "jax.distributed.coordination",
)

# guard name -> where the adapter implements it (hetero/vproc/engine_vllm.py)
ADAPTER_GUARDS = {
    "forbid_distributed": "jax.distributed.initialize/shutdown raise while an engine lives",
    "frozen_tpu_env": "TPU-backend env vars may not change during engine build; restored + raise",
    "check_view_config": "rejects PP>1, DP>1, TPU_MULTIPROCESS_DP, KV connectors, non-TP sharding",
    "preinit_environment": "applies env_override.py's XLA_FLAGS/LIBTPU_INIT_ARGS before the backend starts",
    "sync_compilation_cache": "resets JAX's cache so the engine's jax_compilation_cache_dir takes effect",
}

SURFACE_FUNCS = ("devices", "local_devices", "device_count", "local_device_count",
                 "process_index", "process_count", "process_indices")
MULTIHOST_SCOPED = {"process_allgather", "broadcast_one_to_all", "sync_global_devices", "assert_equal"}
XB_PHYSICAL = {"devices", "device_count", "process_index", "process_count", "process_indices",
               "get_backend", "backends"}
XB_PROXY_PHYSICAL = XB_PHYSICAL - {"devices"}
COORDINATION_ATTRS = {"client", "service", "coordinator_address", "slice_index", "preemption_sync_manager"}
MODULE_ROOTS = {"jax", "jax.distributed", "jax._src.distributed", "jax.experimental.multihost_utils",
                "jax._src.xla_bridge", "tpu_inference.utils.xb", "jax.experimental", "jax._src",
                "jax.experimental.mesh_utils", "jax.experimental.compilation_cache"}
PROXIED_BINDINGS = {("tpu_inference.utils", "xb")}  # view proxies this module-level name
CACHE_CONFIG_PREFIXES = ("jax_compilation_cache", "jax_persistent_cache", "jax_enable_compilation_cache")
IMPLICIT_DEVICES = {  # callee -> (positional index of `devices`, keyword)
    "jax.make_mesh": (None, "devices"),
    "jax.experimental.mesh_utils.create_device_mesh": (1, "devices"),
    "jax.experimental.mesh_utils.create_hybrid_device_mesh": (2, "devices"),
}

# (module, function) -> guard for TPU env writes
ENV_GUARDS = {
    ("tpu_inference.worker.tpu_worker", "_setup_dp_chip_isolation"): "check_view_config",
    ("tpu_inference.worker.tpu_worker", "init_device"): "check_view_config",
    ("tpu_inference.platforms.tpu_platform", "_resolve_multiprocess_dp"): "check_view_config",
    ("tpu_inference.env_override", "<module>"): "preinit_environment",
}

# Dynamic-key env writes reviewed by hand: (module, innermost function) -> why no TPU variable is written.
REVIEWED_DYNAMIC_ENV = {
    ("vllm.envs", "_get_or_set_default"): "only user is VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME (vllm/envs.py:1823)",
    ("vllm.model_executor.model_loader.runai_streamer_loader", "__init__"):
        "keys are RUNAI_STREAMER_* (runai_streamer_loader.py:50-73)",
}

# Engine path, file level. tpu_inference: everything except the listed prefixes.
TPU_INFERENCE_OFF_PATH = {
    "tpu_inference/executors/": "Ray executors; the engine uses hetero.vproc.vllm_executor instead",
    "tpu_inference/core/": "disaggregated-serving EngineCore (core_tpu.py); not used by view engines",
    "tpu_inference/offload/": "KV offload connectors; check_view_config rejects kv_transfer_config",
    "tpu_inference/distributed/tpu_": "KV transfer connectors; check_view_config rejects kv_transfer_config",
    "tpu_inference/distributed/kv_transfer.py": "KV transfer connectors; check_view_config rejects kv_transfer_config",
    "tpu_inference/distributed/transfer_stats.py": "KV transfer connectors",
}
# vllm: only these modules are on the TPU worker path (everything else is other platforms/frontends).
VLLM_ON_PATH = (
    "vllm/platforms/", "vllm/usage/", "vllm/v1/utils.py", "vllm/v1/worker/worker_base.py",
    "vllm/v1/executor/abstract.py", "vllm/v1/engine/core.py", "vllm/distributed/parallel_state.py",
    "vllm/distributed/utils.py", "vllm/config/parallel.py", "vllm/envs.py",
    "vllm/model_executor/model_loader/",
)
SKIP_DIRS = {"tests", "test", "benchmarks", "examples", "docs", "__pycache__", ".git", "tools", "scripts"}


def is_tpu_env_key(key: str) -> bool:
    return "TPU_" in key or key.startswith(("LIBTPU", "JAX_", "XLA_", "MEGASCALE_"))


def load_virtualized() -> Tuple[Set[str], str]:
    sys.path.insert(0, str(REPO))
    try:
        from hetero.vproc import view  # type: ignore
        return set(view.VIRTUALIZED), "hetero.vproc.view"
    except ImportError:
        return set(PINNED_VIRTUALIZED), "pinned"
    finally:
        sys.path.remove(str(REPO))


def verify_guards() -> Dict[str, bool]:
    sys.path.insert(0, str(REPO))
    try:
        from hetero.vproc import engine_vllm
        return {name: callable(getattr(engine_vllm, name, None)) for name in ADAPTER_GUARDS}
    except ImportError:
        return {name: False for name in ADAPTER_GUARDS}
    finally:
        sys.path.remove(str(REPO))


def dotted(node: ast.AST) -> Optional[List[str]]:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return None


class Scanner(ast.NodeVisitor):
    """One file. ``module`` is the dotted module name (``tpu_inference.utils``)."""

    def __init__(self, source: str, module: str, virtualized: Set[str], guards: Dict[str, bool]):
        self.lines = source.splitlines()
        self.module = module
        self.virtualized = virtualized
        self.guards = guards
        self.sites: List[Dict[str, Any]] = []
        self.aliases: Dict[str, str] = {}
        self.early: Set[str] = set()  # names bound at module level to surface functions
        self.scope: List[str] = []
        self.depth_function = 0
        self.tree = ast.parse(source)
        self._collect_imports(self.tree, 0)

    # -- imports -----------------------------------------------------------
    def _collect_imports(self, node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            inner = depth + (1 if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else 0)
            if isinstance(child, ast.Import):
                for a in child.names:
                    self.aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(child, ast.ImportFrom):
                base = child.module or ""
                if child.level:
                    package = self.module.split(".")[:-child.level]
                    base = ".".join(package + ([child.module] if child.module else []))
                for a in child.names:
                    canon = f"{base}.{a.name}" if base else a.name
                    self.aliases[a.asname or a.name] = canon
                    if depth == 0 and self._is_surface_function(canon):
                        self.early.add(a.asname or a.name)
            self._collect_imports(child, inner)

    @staticmethod
    def _is_surface_function(canon: str) -> bool:
        """Functions the view replaces by patching a module attribute (so a from-import copy bypasses it)."""
        if canon in {f"jax.{f}" for f in SURFACE_FUNCS} | {"jax.distributed.initialize", "jax.distributed.shutdown",
                                                           "jax._src.distributed.initialize",
                                                           "jax._src.distributed.shutdown"}:
            return True
        for root in ("jax._src.xla_bridge.", "jax.lib.xla_bridge."):
            if canon.startswith(root) and canon.count(".") == root.count(".") \
                    and canon.rsplit(".", 1)[1] in set(SURFACE_FUNCS) | XB_PHYSICAL:
                return True
        return False

    def canonical(self, parts: Sequence[str]) -> str:
        head = self.aliases.get(parts[0], parts[0])
        canon = ".".join([head, *parts[1:]])
        if canon.startswith("jax.lib.xla_bridge"):
            canon = "jax._src.xla_bridge" + canon[len("jax.lib.xla_bridge"):]
        if canon.startswith("jax._src.xla_bridge") and (self.module, parts[0]) in PROXIED_BINDINGS:
            canon = "tpu_inference.utils.xb" + canon[len("jax._src.xla_bridge"):]
        return canon

    # -- site recording ------------------------------------------------------
    @property
    def function(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def _covered(self, key: str, api: str, reason: str) -> Dict[str, str]:
        if key in self.virtualized:
            return {"class": "covered", "key": key, "reason": reason}
        return {"class": "UNCOVERED", "key": key, "reason": f"{reason}; key {key!r} is not in VIRTUALIZED"}

    def _guarded(self, guard: str, reason: str) -> Dict[str, str]:
        if self.guards.get(guard):
            return {"class": "guarded", "guard": guard, "reason": f"{reason}: {ADAPTER_GUARDS[guard]}"}
        return {"class": "UNCOVERED", "reason": f"{reason}; adapter guard {guard!r} is missing"}

    def record(self, node: ast.AST, api: str, verdict: Dict[str, str]) -> None:
        line = getattr(node, "lineno", 0)
        text = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        self.sites.append({"line": line, "snippet": text[:200], "api": api, "function": self.function, **verdict})

    # -- rules -------------------------------------------------------------
    def evaluate(self, canon: str, node: ast.AST, head: str) -> Optional[Tuple[str, Dict[str, str]]]:
        """(api, verdict) for a call/attribute chain, or None when not surface."""
        if head in self.early and canon.count(".") >= 1 and self._is_surface_function(canon):
            return canon, {"class": "UNCOVERED", "reason": f"{head!r} is bound at import time (from-import); "
                           "attribute patching by the view cannot reach it"}
        if canon == "jax.local_device_count" and canon not in self.virtualized:
            return canon, {"class": "harmless", "reason": "host granularity: every local chip belongs to the engine, "
                           "so the physical value is the engine's value"}
        for f in SURFACE_FUNCS:
            if canon == f"jax.{f}":
                note = {"devices": "virtual view devices; devices('cpu') puts this host's CPU first",
                        "local_devices": "host granularity (unchanged); process_index= remapped",
                        "process_index": "rank within the view", "process_count": "hosts in the view",
                        "process_indices": "view ranks"}.get(f, "view-scoped")
                return canon, self._covered(canon, canon, note)
        if canon.startswith("tpu_inference.utils.xb."):
            f = canon.split(".")[-1]
            if f == "devices":
                return canon, self._covered("tpu_inference.utils.xb", canon, "proxy module: devices() is virtual")
            if f in XB_PROXY_PHYSICAL:
                return canon, {"class": "UNCOVERED", "reason": f"proxy forwards xb.{f} unvirtualized (physical)"}
            return canon, {"class": "harmless", "reason": f"xb.{f}: no process/device identity (local view unchanged)"}
        if canon.startswith("jax._src.xla_bridge."):
            f = canon.split(".")[-1]
            if f in XB_PHYSICAL:
                return canon, {"class": "UNCOVERED", "reason": f"xla_bridge.{f} is physical; only jax.* and "
                               "tpu_inference.utils.xb are virtualized"}
            return canon, {"class": "harmless", "reason": f"xla_bridge.{f}: no process/device identity"}
        if canon in ("jax.distributed.initialize", "jax.distributed.shutdown",
                     "jax._src.distributed.initialize", "jax._src.distributed.shutdown"):
            return canon, self._guarded("forbid_distributed", f"{canon} raises under the adapter")
        if canon == "jax.distributed.is_initialized":
            return canon, {"class": "harmless", "reason": "read-only runtime state"}
        if canon.startswith("jax._src.distributed.global_state"):
            rest = canon[len("jax._src.distributed.global_state"):].lstrip(".").split(".")[0]
            if rest in COORDINATION_ATTRS:
                return canon, self._covered("jax.distributed.coordination", canon, "coordination service access")
            if rest in ("initialize", "shutdown"):
                return canon, self._guarded("forbid_distributed", f"global_state.{rest} raises under the adapter")
            if rest in ("process_id", "num_processes"):
                return canon, {"class": "UNCOVERED", "reason": f"global_state.{rest} is the physical process id/count "
                               "(plan C2 rejects remapping it)"}
            return canon, {"class": "UNCOVERED", "reason": "unknown distributed global_state access"}
        if canon.startswith("jax.experimental.multihost_utils."):
            f = canon.split(".")[-1]
            if f in MULTIHOST_SCOPED:
                return canon, self._covered("jax.experimental.multihost_utils", canon,
                                            "scoped through the virtual process_* names")
            return canon, {"class": "UNCOVERED", "reason": f"multihost_utils.{f} is not one of "
                           f"{sorted(MULTIHOST_SCOPED)} (the C2 table covers only those)"}
        if canon.startswith(("jax._src.compiler._cache_write", "jax._src.compiler.compile_or_get_cached")):
            return canon, self._covered("jax._src.compiler._cache_write", canon, "view rank 0 writes the cache")
        if canon == "jax.clear_caches":
            return canon, {"class": "harmless", "reason": "one engine per host-kernel process (plan C3); "
                           "only the kernel's reshard programs recompile"}
        if canon == "jax.live_arrays":
            return canon, {"class": "harmless", "reason": "process-wide array list, used only for Pathways "
                           "HBM accounting (tpu_inference/utils.py:126)"}
        if canon.startswith("jax.experimental.compilation_cache.compilation_cache."):
            f = canon.split(".")[-1]
            if f in ("set_cache_dir", "initialize_cache"):
                return canon, self._guarded("sync_compilation_cache", f"compilation_cache.{f}")
            if f == "reset_cache":
                return canon, {"class": "harmless", "reason": "resets process-wide cache state"}
        if canon in IMPLICIT_DEVICES and isinstance(node, ast.Call):
            index, keyword = IMPLICIT_DEVICES[canon]
            explicit = any(k.arg == keyword for k in node.keywords) or (index is not None and len(node.args) > index)
            if not explicit:
                return canon, {"class": "UNCOVERED", "reason": f"{canon} without devices= defaults to the physical "
                               "device list (xb.devices() inside jax._src)"}
        return None

    # -- visitors ----------------------------------------------------------
    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.depth_function += 1
        self.generic_visit(node)
        self.depth_function -= 1
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for a in node.names:
            name = a.asname or a.name
            if name in self.early and self.depth_function == 0:
                canon = self.aliases[name]
                self.record(node, canon, {"class": "UNCOVERED", "reason": f"`from ... import {a.name}` binds the "
                            "function at import time; the view patches module attributes, so this copy stays physical"})

    def visit_Call(self, node: ast.Call) -> None:
        parts = dotted(node.func)
        if parts:
            canon = self.canonical(parts)
            result = self.evaluate(canon, node, parts[0])
            if result:
                self.record(node, *result)
            elif canon == "jax.config.update" and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str) and node.args[0].value.startswith(CACHE_CONFIG_PREFIXES):
                option = node.args[0].value
                if option == "jax_compilation_cache_dir":
                    verdict = self._guarded("sync_compilation_cache", "cache dir set at runner init")
                else:
                    verdict = {"class": "harmless", "reason": "read at compile time; no process/device identity"}
                self.record(node, f"jax.config.update({option!r})", verdict)
            elif not self._env_call(node, canon) and parts[-1] == "process_index" and parts[0] not in self.aliases:
                self._device_attr(node, parts)
        else:
            self.visit(node.func)  # non-pure base such as jax.devices()[0].coords()
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        parts = dotted(node)
        if parts:
            canon = self.canonical(parts)
            if parts[0] in self.aliases:
                result = self.evaluate(canon, node, parts[0])
                if result:
                    self.record(node, *result)
            elif node.attr == "process_index":
                self._device_attr(node, parts)
            return
        if node.attr == "process_index":
            self._device_attr(node, ["<expr>", "process_index"])
        self.visit(node.value)

    def _device_attr(self, node: ast.AST, parts: Sequence[str]) -> None:
        key = f"{self.module}.{self.scope[-1]}" if self.scope else self.module
        api = ".".join(parts)
        if key in self.virtualized:
            verdict = {"class": "covered", "key": key,
                       "reason": f"device .process_index read inside {key} (wrapped to rank within the view)"}
        else:
            verdict = {"class": "UNCOVERED", "reason": "device .process_index is a read-only C++ attribute "
                       "(physical host); no virtualized function wraps this site"}
        self.record(node, api, verdict)

    # -- environment writes -------------------------------------------------
    def _env_verdict(self, node: ast.AST, key: Optional[str], api: str) -> None:
        if key is not None and not is_tpu_env_key(key):
            return
        guard = ENV_GUARDS.get((self.module, self.scope[-1] if self.scope else "<module>"))
        label = f"{api}[{key!r}]" if key is not None else f"{api}[<dynamic key>]"
        reviewed = REVIEWED_DYNAMIC_ENV.get((self.module, self.scope[-1] if self.scope else "<module>"))
        if key is None and reviewed:
            verdict = {"class": "harmless", "reason": f"dynamic key, reviewed: {reviewed}"}
        elif key is None:
            verdict = {"class": "UNCOVERED", "reason": "dynamic environment key could be a TPU variable"}
        elif guard:
            verdict = self._guarded(guard, f"TPU env write {key}")
        else:
            verdict = {"class": "UNCOVERED", "reason": f"TPU env write {key} outside the adapter-guarded sites; "
                       "the backend is already up in a host kernel"}
        self.record(node, label, verdict)

    @staticmethod
    def _const(node: Optional[ast.AST]) -> Optional[str]:
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    def _is_environ(self, node: ast.AST) -> bool:
        parts = dotted(node)
        return bool(parts) and self.canonical(parts) == "os.environ"

    def _env_call(self, node: ast.Call, canon: str) -> bool:
        if canon in ("os.environ.setdefault", "os.environ.pop", "os.putenv", "os.unsetenv"):
            key = self._const(node.args[0]) if node.args else None
            self._env_verdict(node, key, canon)
            return True
        if canon == "os.environ.update":
            keys: List[Optional[str]] = []
            if node.args and isinstance(node.args[0], ast.Dict):
                keys += [self._const(k) for k in node.args[0].keys]
            elif node.args:
                keys.append(None)
            keys += [k.arg for k in node.keywords if k.arg]
            if any(k.arg is None for k in node.keywords):
                keys.append(None)
            for key in keys or [None]:
                self._env_verdict(node, key, canon)
            return True
        if canon == "os.environ.clear":
            self.record(node, canon, {"class": "UNCOVERED", "reason": "clears every environment variable"})
            return True
        return False

    def _subscript_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Subscript) and self._is_environ(target.value):
            self._env_verdict(target, self._const(target.slice), "os.environ")
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._subscript_target(element)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._subscript_target(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._subscript_target(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._subscript_target(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._subscript_target(target)


def scan_source(source: str, module: str, virtualized: Optional[Iterable[str]] = None,
                guards: Optional[Dict[str, bool]] = None) -> List[Dict[str, Any]]:
    """Sites of one source text (no file system). ``virtualized``/``guards`` default to pinned/all-present."""
    scanner = Scanner(source, module, set(PINNED_VIRTUALIZED if virtualized is None else virtualized),
                      {name: True for name in ADAPTER_GUARDS} if guards is None else guards)
    scanner.visit(scanner.tree)
    return sorted(scanner.sites, key=lambda s: (s["line"], s["api"]))


# -- trees ----------------------------------------------------------------


def git_sha(path: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": "unknown", "dirty": None}


def engine_path_tpu_inference(rel: str) -> Tuple[bool, str]:
    for prefix, reason in TPU_INFERENCE_OFF_PATH.items():
        if rel.startswith(prefix):
            return False, reason
    return True, "imported by the TPU worker/runner/model path"


def engine_path_vllm(rel: str) -> Tuple[bool, str]:
    if rel.startswith(VLLM_ON_PATH):
        return True, "imported by TPUWorker / vLLM config for the TPU platform"
    return False, "not on the TPU worker path (other platforms, frontends, GPU workers)"


def scan_tree(root: Path, package: str, fork: str, on_path: Callable[[str], Tuple[bool, str]],
              virtualized: Set[str], guards: Dict[str, bool], info: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    base = root / package
    for path in sorted(base.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if SKIP_DIRS & set(path.relative_to(root).parts[:-1]):
            continue
        module = rel[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        try:
            sites = scan_source(path.read_text(), module, virtualized, guards)
        except SyntaxError as error:
            out.append({"file": rel, "line": error.lineno or 0, "snippet": "", "api": "<syntax error>",
                        "class": "UNCOVERED", "reason": str(error), "function": "<module>"})
            sites = []
        engine, why = on_path(rel)
        for site in sites:
            out.append({"fork": fork, "file": rel, "location": f"{fork}/{rel}:{site['line']}",
                        "engine_path": engine, "path_reason": why, "fork_sha": info["sha"], **site})
    return out


def summarize(sites: List[Dict[str, Any]]) -> Dict[str, Any]:
    def count(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for row in rows:
            result[row["class"]] = result.get(row["class"], 0) + 1
        return dict(sorted(result.items()))
    engine = [s for s in sites if s.get("engine_path")]
    return {"sites": len(sites), "by_class": count(sites), "engine_path_sites": len(engine),
            "engine_path_by_class": count(engine),
            "uncovered_engine_path": sum(1 for s in engine if s["class"] == "UNCOVERED")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    home = Path.home() / "heterogeneous-serving-tpu"
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tpu-inference", type=Path, default=home / "tpu-inference")
    parser.add_argument("--vllm", type=Path, default=home / "vllm")
    parser.add_argument("--out", type=Path, default=REPO / "results/vproc/surface-check.json")
    args = parser.parse_args(argv)
    for name, path in (("tpu-inference", args.tpu_inference), ("vllm", args.vllm)):
        if not (path / ("tpu_inference" if name == "tpu-inference" else "vllm")).is_dir():
            print(f"missing fork {name}: {path}", file=sys.stderr)
            return 2

    virtualized, source = load_virtualized()
    guards = verify_guards()
    forks = {"tpu-inference": git_sha(args.tpu_inference), "vllm": git_sha(args.vllm)}
    sites = scan_tree(args.tpu_inference, "tpu_inference", "tpu-inference", engine_path_tpu_inference,
                      virtualized, guards, forks["tpu-inference"])
    sites += scan_tree(args.vllm, "vllm", "vllm", engine_path_vllm, virtualized, guards, forks["vllm"])
    report = {
        "forks": forks, "virtualized_keys": sorted(virtualized), "virtualized_source": source,
        "adapter_guards": {name: {"present": guards[name], "does": text} for name, text in ADAPTER_GUARDS.items()},
        "engine_path_rules": {"tpu-inference": "all of tpu_inference/ except " + ", ".join(TPU_INFERENCE_OFF_PATH),
                              "vllm": "only " + ", ".join(VLLM_ON_PATH)},
        "summary": summarize(sites), "sites": sites,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    summary = report["summary"]
    print(f"{summary['sites']} sites; engine path {summary['engine_path_by_class']}; "
          f"uncovered on engine path: {summary['uncovered_engine_path']} -> {args.out}")
    for site in sites:
        if site["engine_path"] and site["class"] == "UNCOVERED":
            print(f"UNCOVERED {site['location']}: {site['snippet']}\n    {site['reason']}")
    return 1 if summary["uncovered_engine_path"] else 0


if __name__ == "__main__":
    sys.exit(main())
