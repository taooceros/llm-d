#!/usr/bin/env python3
"""Standalone overlap smoke, independent of vLLM.

CPU: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
     timeout 90s .venv/bin/python scripts/compile_execution_smoke.py --cpu-local \
     --depth 128 --batch 8 --width 128 --min-compile-s 0.01 --min-contained-steps 1 --output /tmp/smoke.json
TPU (authorized runs only): flock /tmp/hetero-cluster.lock .venv/bin/python rcm_exec run \
     scripts/compile_execution_smoke.py --inventory /tmp/hetero-inventory-only.json --output OUTPUT

Unified-runtime mode (plan U1/U2, hetero/unified_runtime.py): --experiment u1|u2|u1d
with --unified '{"field": value}' overrides of UnifiedConfig. CPU emulation runs four
local processes x four CPU devices in one jax.distributed runtime (gloo):
     timeout 300s .venv/bin/python scripts/compile_execution_smoke.py --cpu-local \
     --experiment u2 --unified '{"model_layers": 4, "model_d_model": 64, ...}' --output /tmp/u2.json

A passed result proves only this synthetic attached-topology/subset mechanism.
The source is precompiled and target tracing/lowering happen before measurement.
The default lower/origin half passed on libtpu 0.0.43; --target-half upper
retains the non-origin target-load bounds-failure reproducer. The short CPU
example relaxes the duration gate; default TPU acceptance requires >=5 s.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hetero.compile_execution_smoke import Config, Session, verdict
from hetero.unified_runtime import PROTOCOLS as UNIFIED_PROTOCOLS, UnifiedConfig

SUMMARY_KEYS = ("experiment", "status", "failures", "traceback", "rates_steps_per_s", "independence",
                "b2", "compiles", "e_target_compile_impact", "cross_half_max_abs_diff",
                "weight_bytes_per_chip_tp8", "target_compile", "serving_impact",
                "overlap_evidence_problems", "target_steps_per_s", "max_abs_diff_tp16_vs_tp8",
                "probes")


def header(config, mode, unified=None):
    extra = {} if unified is None else {"experiment": unified.experiment,
                                        "unified_config": dataclasses.asdict(unified)}
    return {"mode": mode, "config": dataclasses.asdict(config), **extra, "status": "failed",
            "source_revision": os.environ.get("SMOKE_SOURCE_REVISION"),
            "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                              for path in ("scripts/compile_execution_smoke.py",
                                           "hetero/compile_execution_smoke.py",
                                           "hetero/unified_runtime.py")},
            "limitations": OVERLAP_LIMITATIONS if unified is None else UNIFIED_LIMITATIONS}


OVERLAP_LIMITATIONS = [
    "Synthetic mechanism, not vLLM throughput or reconfiguration cost.",
    "Target is a subset of the attached topology, not a newly initialized smaller runtime.",
    "Trace/lower and source preparation are outside overlap windows.",
    "Only rank-zero monotonic timestamps establish native-call containment.",
    "Driver batch barriers/gaps are included in wall rates; all ranks execute equal step counts.",
    "Compiler CPU process totals include concurrent serving/native pools."]
UNIFIED_LIMITATIONS = [
    "Synthetic programs (TP-sharded MLP stack), not vLLM engines or their executable sets.",
    "Each half is driven by its own two processes through per-process command threads; "
    "driver queue refills (depth 3, ~0.1 s batches) can add gaps, included in wall rates.",
    "Overlap containment is per rank on its own monotonic clock: 'call' windows include "
    "trace/lower, 'native' windows only backend-compile intervals.",
    "Cross-host switch segments use epoch stamps shifted by a min-RTT driver clock estimate "
    "(uncertainty ~RTT/2); per-rank monotonic splits are reported alongside.",
    "The reverse switch reuses the TP8 executables loaded at setup; it has no compile by design.",
    "Steps block per step (block_until_ready), as a decode loop with no dispatch-ahead."]


def protocol(config, all_workers, rank_zero, emit, progress):
    plan = rank_zero("prepare_target", timeout_s=300)
    progress["target_plan"] = plan
    emit("SMOKE_TARGET_LOWERED", **plan)

    def controls(phase):
        until = time.monotonic() + config.control_s
        batches = 0
        while time.monotonic() < until:
            all_workers("steps", phase, config.step_batch, timeout_s=60)
            batches += 1
        emit("SMOKE_CONTROL_DONE", phase=phase, batches=batches)

    controls("baseline")
    rank_zero("start_compile", timeout_s=30)
    deadline = time.monotonic() + config.compile_timeout_s
    last_event = time.monotonic()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("native compile/source-dispatch overlap exceeded bounded deadline")
        all_workers("steps", "overlap", config.step_batch, timeout_s=remaining)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("source batch completed beyond native compile deadline")
        status = rank_zero("compile_status", timeout_s=remaining)
        if status["done"]:
            emit("SMOKE_NATIVE_RETURNED", **status["result"])
            if not status["result"]["ok"]:
                raise RuntimeError(status["result"]["traceback"])
            break
        if time.monotonic() - last_event > 5:
            emit("SMOKE_SOURCE_PROGRESS", completed_batch_steps=config.step_batch)
            last_event = time.monotonic()
    controls("postcontrol")
    progress["source_validation"] = all_workers("validate_source", timeout_s=60)
    # A native loader abort may kill rank zero. Save completed overlap first.
    progress["preload_workers"] = all_workers("report", timeout_s=60)
    # Every source batch is finished on every rank before serialization or load.
    exported = rank_zero("export_target", timeout_s=120)
    blob = exported.pop("blob")
    options = exported.pop("compile_options")
    if exported["compile_options_sha256"] != plan["compile_options_sha256"]:
        raise ValueError("compile options changed during native compilation")
    progress["serialized_target"] = exported
    emit("SMOKE_TARGET_SERIALIZED", **exported)
    target_checks = all_workers("load_and_check_target", blob, options, timeout_s=180)
    emit("SMOKE_TARGET_CHECKED", checks=target_checks)
    reports = all_workers("report", timeout_s=60)
    progress.pop("preload_workers", None)
    return {"workers": reports, "serialized_target": exported, **verdict(config, reports)}


def cpu_local(config):
    from hetero.evidence import emit_event
    result = header(config, "cpu_local")
    session = None
    try:
        session = Session(config, "cpu")

        def one(method, *args, timeout_s):
            # Native calls are bounded by the caller's process timeout in CPU mode.
            return getattr(session, method)(*args)

        def all_workers(method, *args, timeout_s):
            return [one(method, *args, timeout_s=timeout_s)]

        result.update(protocol(config, all_workers, one, emit_event, result))
    except Exception:
        result["traceback"] = traceback.format_exc()
        if session:
            result["partial_worker"] = session.report()
    return result


def cpu_unified(config, unified):
    from hetero.evidence import emit_event
    from hetero.unified_runtime import CpuFleet
    result = header(config, "cpu_unified_submeshes", unified)
    fleet = None
    try:
        fleet = CpuFleet(unified)
        result.update(UNIFIED_PROTOCOLS[unified.experiment](unified, fleet, emit_event, result))
    except Exception:
        result["traceback"] = traceback.format_exc()
    finally:
        if fleet is not None:
            fleet.close()
    return result


class RayFleet:
    """Ray actors as a unified-runtime fleet; rank == actor index == JAX process id."""

    def __init__(self, actors):
        self.actors, self.ranks = actors, list(range(len(actors)))

    def call(self, calls, timeout_s):
        import ray
        return ray.get([self.actors[r].call.remote(m, *a) for r, m, a in calls], timeout=timeout_s)


def remote(config, timeout, unified=None):
    import ray
    from ray.util.placement_group import placement_group, placement_group_table, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from ray.util.state import get_actor
    from hetero.layout import builtin_layout, resolve_layout
    from hetero.slice_inventory import SliceInventory, discover_ray_tpu_nodes
    from hetero.evidence import emit_event, emit_measurements

    mode = "tpu_16_to_8_attached_submesh" if unified is None else "tpu_16_unified_submeshes"
    result = header(config, mode, unified)
    actors, actor_ids, pg = [], [], None
    ray.init(address="auto", namespace="compile-execution-smoke")
    inventory = SliceInventory.from_dict(json.loads(os.environ["SMOKE_INVENTORY"]))
    instance = resolve_layout(inventory, builtin_layout("H")).instances[0]
    coordinator = f"{instance.coordinator_ip}:12355"

    @ray.remote(num_cpus=1, resources={"TPU": 4}, max_restarts=0)
    class Controller:
        def __init__(self, env, rank, count):
            env.pop("JAX_COMPILATION_CACHE_DIR", None)
            os.environ.update(env)
            os.environ["TPU_VISIBLE_CHIPS"] = "0,1,2,3"
            os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
            os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                os.environ.pop(key, None)
            self.rank, self.count = rank, count

        def initialize(self):
            import jax
            jax.distributed.initialize(coordinator_address=coordinator,
                num_processes=self.count, process_id=self.rank, initialization_timeout=180)
            if unified is None:
                self.session = Session(config, "tpu")
            else:
                from hetero.unified_runtime import UnifiedWorker
                self.session = UnifiedWorker(unified, "tpu")
            return self.session.identity

        def call(self, method, *args):
            return getattr(self.session, method)(*args)

    try:
        nodes = discover_ray_tpu_nodes()
        observed = {(n["node_ip"], n["ray_node_id"], n["tpu_count"]) for n in nodes}
        expected = {(h.node_ip, h.ray_node_id, h.tpu_count) for h in inventory.hosts}
        if observed != expected:
            raise RuntimeError("inventory node identities changed; refresh inventory before reserving")
        before = ray.available_resources()
        if before.get("TPU", 0) != inventory.total_chips:
            raise RuntimeError("slice is not idle; refusing to reserve")
        result.update(instance=instance.to_dict(), coordinator=coordinator,
                      nodes_before=nodes, resources_before=before)
        name = f"compile-execution-smoke-{mode}-{time.time_ns()}"
        pg = placement_group([{"CPU": 1, "TPU": 4, f"node:{h.node_ip}": .001}
                              for h in instance.hosts], strategy="STRICT_SPREAD", name=name)
        result["placement_group_id"] = pg.id.hex()
        ray.get(pg.ready(), timeout=120)
        emit_event("SMOKE_RESERVED", placement_group=result["placement_group_id"])
        for i, host in enumerate(instance.hosts):
            actor = Controller.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=i),
                resources={"TPU": 4, f"node:{host.node_ip}": .001}).remote(
                    instance.process_env(host.process_rank), host.process_rank, len(instance.hosts))
            actors.append(actor)
            actor_ids.append(actor._actor_id.hex())
        result["actor_ids"] = actor_ids
        result["initialized"] = ray.get([a.initialize.remote() for a in actors], timeout=300)
        emit_event("SMOKE_INITIALIZED", hosts=len(actors))

        def all_workers(method, *args, timeout_s):
            return ray.get([a.call.remote(method, *args) for a in actors], timeout=timeout_s)

        def rank_zero(method, *args, timeout_s):
            return ray.get(actors[0].call.remote(method, *args), timeout=timeout_s)

        if unified is None:
            result.update(protocol(config, all_workers, rank_zero, emit_event, result))
        else:
            result.update(UNIFIED_PROTOCOLS[unified.experiment](
                unified, RayFleet(actors), emit_event, result))
    except Exception:
        result["traceback"] = traceback.format_exc()
        emit_event("SMOKE_FAILED", traceback=result["traceback"])
        for index, actor in enumerate(actors):
            try:
                report = ray.get(actor.call.remote("report"), timeout=3)
                result.setdefault("partial_workers", []).append(report)
            except Exception as exc:
                result.setdefault("partial_report_errors", []).append({"worker": index, "error": str(exc)})
    finally:
        cleanup = {"ok": False, "actor_ids": actor_ids, "kill_errors": []}
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception as exc:
                cleanup["kill_errors"].append(str(exc))
        if pg is not None:
            remove_placement_group(pg)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                states = {}
                for actor_id in actor_ids:
                    state = get_actor(actor_id, timeout=3)
                    states[actor_id] = None if state is None else state.state
                pg_state = None if pg is None else placement_group_table(pg).get("state")
                available = ray.available_resources()
                cleanup.update(actor_states=states, placement_group_state=pg_state,
                               resources_after=available)
                cleanup["ok"] = (all(s in (None, "DEAD") for s in states.values())
                    and pg_state in (None, "REMOVED")
                    and available.get("TPU", 0) == inventory.total_chips)
                if cleanup["ok"]:
                    break
            except Exception as exc:
                cleanup["last_verification_error"] = str(exc)
            time.sleep(1)
        result["cleanup"] = cleanup
        if not cleanup["ok"]:
            result["status"] = "failed_cleanup"
        emit_event("SMOKE_RELEASED", **cleanup)
        emit_measurements({"result": result})
    return result


def local(args, config, unified=None):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor

    @stage(name="compile_execution_smoke", order_idx=0, expected_s=args.timeout)
    def submit(state, timing):
        supervisor = RayJobSupervisor(ray_url="http://127.0.0.1:8265", timing_engine=timing)
        argv = ["python3", "-u", "scripts/compile_execution_smoke.py", "--remote", "--timeout", str(args.timeout)]
        for field in dataclasses.fields(config):
            argv.extend(["--" + field.name.replace("_", "-"), str(getattr(config, field.name))])
        if unified is not None:
            argv.extend(["--experiment", unified.experiment,
                         "--unified", json.dumps(dataclasses.asdict(unified), sort_keys=True)])
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        submission = supervisor.submit_job(entrypoint=shlex.join(argv),
            runtime_env={"working_dir": str(ROOT),
                "excludes": [".venv*", ".git", ".jj", "data", "results", "**/results/**",
                             "docs", "reports", "formalization", "__pycache__", "*.pyc"],
                "env_vars": {"SMOKE_INVENTORY": args.inventory.read_text(),
                             "SMOKE_SOURCE_REVISION": revision, "PYTHONUNBUFFERED": "1"}},
            name="compile-execution-smoke" if unified is None else f"unified-runtime-{unified.experiment}",
            metadata={"purpose": "standalone-compile-execution-overlap" if unified is None
                      else f"unified-runtime-{unified.experiment}"})
        state.set("submission_id", submission)
        result = supervisor.supervise_until_completion(submission_id=submission,
            timeout_s=args.timeout, stage_name="compile_execution_smoke", state_store=state)
        payload = {"submission_id": submission, "source_revision": revision,
                   "saved_at": time.time(), "status": result["status"],
                   "entrypoint": shlex.join(argv), "measurements": result["measurements"]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[OMP_EVENT: SMOKE_SAVED path={args.output}]", flush=True)

    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(submit)
    if not runner.run():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--remote", action="store_true")
    modes.add_argument("--cpu-local", action="store_true")
    parser.add_argument("--inventory", type=Path, default=Path("/tmp/hetero-inventory-only.json"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/compile-execution-smoke.json"))
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--experiment", choices=("overlap",) + tuple(UNIFIED_PROTOCOLS),
                        default="overlap")
    parser.add_argument("--unified", default="{}",
                        help="JSON overrides of hetero.unified_runtime.UnifiedConfig")
    defaults = Config()
    for field in dataclasses.fields(defaults):
        value = getattr(defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    config = Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(defaults)})
    config.validate()
    unified = None
    if args.experiment != "overlap":
        overrides = json.loads(args.unified)
        overrides["experiment"] = args.experiment
        unified = UnifiedConfig(**overrides)
        unified.validate()
    if args.cpu_local:
        result = cpu_local(config) if unified is None else cpu_unified(config, unified)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if unified is None:
            print(json.dumps({k: v for k, v in result.items() if k not in ("workers", "partial_worker")}))
        else:
            print(json.dumps({k: result[k] for k in SUMMARY_KEYS if k in result}, default=str))
        raise SystemExit(0 if result["status"] == "passed" else 1)
    if args.remote:
        remote(config, args.timeout, unified)
    else:
        local(args, config, unified)


if __name__ == "__main__":
    main()
