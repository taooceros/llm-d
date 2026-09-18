#!/usr/bin/env python3
"""Remote entry point for the heterogeneous serving fabric.

Runs *inside* the Ray cluster as an RCM-supervised Ray job:

    rcm_exec run scripts/hetero_ctl.py -- inventory
    rcm_exec run scripts/hetero_ctl.py -- plan --layout M

Every command emits ``[OMP_EVENT: ...]`` telemetry plus exactly one structured
measurement payload scraped by ``RayJobSupervisor``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ray  # noqa: E402
from hetero.deployment import (  # noqa: E402
    DeploymentConfig,
    deploy,
    drain_deployment,
    inspect_deployment,
    plan_deployment,
)
from hetero.evidence import emit_event, emit_log, emit_measurements  # noqa: E402
from hetero.layout import (  # noqa: E402
    LayoutRequest,
    builtin_layout,
    resolve_layout,
)
from hetero.placement import list_owned_placement_groups  # noqa: E402
from hetero.registry import find_registry  # noqa: E402
from hetero.slice_inventory import SliceInventory, observe_slice  # noqa: E402


def _connect() -> None:
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
    ctx = ray.get_runtime_context()
    emit_event(
        "RAY_CONNECTED",
        job_id=ctx.get_job_id(),
        node_id=ctx.get_node_id(),
        ray_version=ray.__version__,
    )


def _layout_requests(args: argparse.Namespace) -> list[LayoutRequest]:
    if args.layout_file:
        raw = os.environ.get("OMP_HETERO_LAYOUT_JSON")
        data = json.loads(raw if raw is not None else pathlib.Path(args.layout_file).read_text())
        rows = data if isinstance(data, list) else [data]
        return [LayoutRequest.from_dict(row) for row in rows]
    return [builtin_layout(name) for name in args.layout.split(",") if name]


def cmd_inventory(args: argparse.Namespace) -> Dict[str, Any]:
    """Observes physical slice geometry with an owned, short-lived probe."""
    owned = list_owned_placement_groups()
    live = [row for row in owned if row["state"] in ("CREATED", "PENDING", "RESCHEDULING")]
    if live and not args.force:
        raise RuntimeError(
            "refusing to initialize a full-slice probe runtime while owned serving "
            f"instances hold chips: {[r['name'] for r in live]}. Drain them first or "
            "pass --force if you have verified they are idle and released."
        )
    inventory = observe_slice(
        probe_cpu=args.probe_cpu,
        ready_timeout_s=args.ready_timeout_s,
        probe_timeout_s=args.probe_timeout_s,
    )
    return {"action": "inventory", "inventory": inventory.to_dict()}


def cmd_plan(args: argparse.Namespace) -> Dict[str, Any]:
    """Non-mutating: resolves and validates layouts against observed geometry."""
    inventory, source = _resolve_inventory(args, allow_probe=False)
    plans = []
    for request in _layout_requests(args):
        resolved = resolve_layout(inventory, request)
        body = resolved.to_dict()
        emit_event(
            "LAYOUT_RESOLVED",
            layout_id=body["layout_id"],
            digest=body["layout_digest"],
            instances=[i["instance_id"] for i in body["instances"]],
        )
        for inst in body["instances"]:
            emit_log(
                f"{body['layout_id']}/{inst['instance_id']}: tp={inst['tp_size']} "
                f"chips={inst['chip_box']} topology={inst['subslice_topology']} "
                f"host_bounds={inst['host_bounds']} hosts={inst['host_ips']}"
            )
        plans.append({"request": request.to_dict(), "resolved_layout": body})
    return {
        "action": "plan",
        "inventory_source": source,
        "inventory": inventory.to_dict(),
        "plans": plans,
        "owned_placement_groups": list_owned_placement_groups(),
    }


def _engine_settings(args: argparse.Namespace) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    if args.engine_settings:
        settings.update(json.loads(args.engine_settings))
    if args.model:
        settings["model"] = args.model
    if args.max_model_len:
        settings["max_model_len"] = args.max_model_len
    if args.max_num_seqs:
        settings["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens:
        settings["max_num_batched_tokens"] = args.max_num_batched_tokens
    return settings


def _resolve_inventory(args: argparse.Namespace, *, allow_probe: bool = True) -> tuple[SliceInventory, str]:
    """Prefers a recorded inventory so deploy never re-grabs live chips."""
    if args.inventory_file:
        raw = os.environ.get("OMP_HETERO_INVENTORY_JSON")
        inventory = SliceInventory.from_dict(json.loads(
            raw if raw is not None else pathlib.Path(args.inventory_file).read_text()
        ))
        current = _node_generation_now()
        if inventory.ray_node_generation != current:
            raise RuntimeError(
                "saved inventory belongs to a different node generation "
                f"({inventory.ray_node_generation} != {current}); run explicit inventory"
            )
        return inventory, f"file:{args.inventory_file}"
    registry = find_registry()
    if registry is not None and not args.reobserve:
        saved = ray.get(registry.get_inventory.remote(), timeout=60)
        if saved:
            inventory = SliceInventory.from_dict(saved)
            current = _node_generation_now()
            if inventory.ray_node_generation == current:
                emit_log("reusing recorded slice inventory tied to the same allocation")
                return inventory, "registry"
            emit_log(
                "recorded inventory belongs to a different node generation "
                f"({inventory.ray_node_generation} != {current}); re-observing"
            )
    if not allow_probe:
        raise RuntimeError(
            "No physical inventory tied to the current allocation. Run the explicit "
            "inventory command on available owned resources first; plan/preflight "
            "never initialize TPU runtimes or create placement groups."
        )
    live = [
        row
        for row in list_owned_placement_groups()
        if row["state"] in ("CREATED", "PENDING", "RESCHEDULING")
    ]
    if live and not args.force:
        raise RuntimeError(
            "no usable recorded inventory and owned instances still hold chips: "
            f"{[r['name'] for r in live]}. Drain them before re-observing."
        )
    return (
        observe_slice(
            probe_cpu=args.probe_cpu,
            ready_timeout_s=args.ready_timeout_s,
            probe_timeout_s=args.probe_timeout_s,
        ),
        "observed",
    )


def _node_generation_now() -> str:
    from hetero.slice_inventory import _node_generation, discover_ray_tpu_nodes

    return _node_generation(discover_ray_tpu_nodes())


def _deployment_config(args: argparse.Namespace) -> DeploymentConfig:
    requests = _layout_requests(args)
    if len(requests) != 1:
        raise ValueError("deploy operates on exactly one layout")
    return DeploymentConfig(
        layout=requests[0],
        engine_settings=_engine_settings(args),
        cpu_per_host=args.cpu_per_host,
        pg_ready_timeout_s=args.ready_timeout_s,
        engine_init_timeout_s=args.engine_init_timeout_s,
    )


def cmd_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    """Non-mutating preflight of a full deployment."""
    inventory, source = _resolve_inventory(args, allow_probe=False)
    config = _deployment_config(args)
    _, preflight = plan_deployment(inventory, config)
    for action in preflight["actions"]:
        emit_log(
            f"{action['instance_id']}: pg={action['placement_group_action']} "
            f"actor={action['engine_actor_action']} hosts={action['hosts']}"
        )
    return {
        "action": "preflight",
        "inventory_source": source,
        "inventory": inventory.to_dict(),
        "preflight": preflight,
    }


def cmd_deploy(args: argparse.Namespace) -> Dict[str, Any]:
    """Reconciles owned engines and their gateway as one deployment."""
    inventory, source = _resolve_inventory(args)
    config = _deployment_config(args)
    report = deploy(inventory, config)
    gateway = cmd_gateway_up(args)["gateway"]
    return {"action": "deploy", "inventory_source": source,
            "deployment": report, "gateway": gateway}


def cmd_inspect(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "action": "inspect",
        "inspection": inspect_deployment(args.layout if args.layout else None),
    }

def cmd_cache_layout(args: argparse.Namespace) -> Dict[str, Any]:
    """Inspect actual native groups/arrays/shards without allocating an engine."""
    registry = find_registry()
    if registry is None:
        raise RuntimeError("no serving-instance registry")
    rows = ray.get(registry.snapshot.remote(args.layout), timeout=60)
    selected = set(args.instances.split(",")) if args.instances else None
    rows = [r for r in rows if selected is None or r["instance_id"] in selected]
    if not rows or (selected is not None and selected != {r["instance_id"] for r in rows}):
        raise RuntimeError("requested live instances are not registered")
    pending = {row["instance_id"]: ray.get_actor(
        row["engine_actor_name"], namespace=row["ray_namespace"]
    ).kv_layout.remote() for row in rows}
    layouts = {}
    for instance_id, reference in pending.items():
        try:
            layouts[instance_id] = {"supported": True, "layout": ray.get(reference, timeout=300)}
        except Exception as exc:
            layouts[instance_id] = {"supported": False, "error": str(exc)}
    return {"action": "cache-layout", "layout_id": args.layout,
            "descriptors": rows, "cache_layouts": layouts,
            "all_supported": all(row["supported"] for row in layouts.values())}


def cmd_drain(args: argparse.Namespace) -> Dict[str, Any]:
    instances = args.instances.split(",") if args.instances else None
    return {
        "action": "drain",
        "drain": drain_deployment(
            args.layout, instances, remove=not args.keep_resources, timeout_s=args.drain_timeout_s
        ),
    }


def cmd_gate(args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 1 exit gate: concurrent real serving on every logical instance."""
    from hetero.bringup_gate import run_bringup_gate

    inventory, source = _resolve_inventory(args)
    config = _deployment_config(args)
    report = run_bringup_gate(
        inventory,
        config,
        prompts_per_instance=args.prompts_per_instance,
        decode_tokens=args.decode_tokens,
        restart_instance=args.restart_instance,
    )
    return {"action": "gate", "inventory_source": source, "gate": report}


def cmd_release(args: argparse.Namespace) -> Dict[str, Any]:
    """Removes owned placement groups and engine actors by name, nothing else."""
    from hetero.deployment import ENGINE_ACTOR_PREFIX, engine_actor_name
    from hetero.placement import release_placement_group
    from hetero.registry import REGISTRY_NAMESPACE

    wanted = set(args.instances.split(",")) if args.instances else None
    rows = [
        row
        for row in list_owned_placement_groups(args.layout)
        if wanted is None or row["instance_id"] in wanted
    ]
    released = []
    for row in rows:
        entry = {"name": row["name"], "state": row["state"]}
        try:
            actor = ray.get_actor(
                engine_actor_name(row["layout_id"], row["instance_id"]),
                namespace=REGISTRY_NAMESPACE,
            )
            ray.kill(actor, no_restart=True)
            entry["actor_killed"] = True
        except ValueError:
            entry["actor_killed"] = False
        if row["state"] != "REMOVED":
            entry["pg_removed"] = release_placement_group(row["name"])
        registry = find_registry()
        if registry is not None:
            entry["withdrawn"] = ray.get(
                registry.withdraw.remote(row["instance_id"]), timeout=60
            )
        released.append(entry)
        emit_log(f"released {row['name']}")
    return {
        "action": "release",
        "released": released,
        "owned_placement_groups": list_owned_placement_groups(),
    }


def _routing_policy(args: argparse.Namespace) -> Dict[str, Any]:
    from hetero.layout import BUILTIN_LAYOUTS

    if args.policy_file:
        return json.loads(pathlib.Path(args.policy_file).read_text())
    if args.policy:
        return json.loads(args.policy)
    names = BUILTIN_LAYOUTS.get(args.layout, ())
    small = tuple(n for n, tp in names if tp == 8)
    large = tuple(n for n, tp in names if tp >= 16)
    if not large:
        # An all-small layout has no large destination; migration is impossible.
        large = ()
    return {
        "small_instances": list(small),
        "large_instances": list(large),
        "migration_enabled": bool(args.migration),
    }


def cmd_gateway_up(args: argparse.Namespace) -> Dict[str, Any]:
    from hetero.gateway.lifecycle import gateway_up

    head_ip = args.head_node_ip or _head_node_ip()
    report = gateway_up(
        layout_id=args.layout,
        policy=_routing_policy(args),
        head_node_ip=head_ip,
        bind_labels=not args.no_bind_labels,
    )
    return {"action": "gateway-up", "gateway": report}


def cmd_gateway_down(args: argparse.Namespace) -> Dict[str, Any]:
    from hetero.gateway.lifecycle import gateway_down

    return {"action": "gateway-down", "gateway": gateway_down(remove=not args.keep_resources)}


def cmd_gateway_status(args: argparse.Namespace) -> Dict[str, Any]:
    from hetero.gateway.lifecycle import gateway_status

    return {"action": "gateway-status", "gateway": gateway_status()}


def _head_node_ip() -> str:
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        resources = node.get("Resources", {}) or {}
        if any(k == "node:__internal_head__" for k in resources):
            return node["NodeManagerAddress"]
    raise RuntimeError("could not identify the Ray head node")


def cmd_gateway_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    from hetero.experiment import BASE_POLICY, _instance_roles
    from hetero.qualify_migration import run_hardware_qualification

    registry = find_registry()
    if registry is None:
        raise RuntimeError("no registered serving instances")
    descriptors = ray.get(registry.snapshot.remote(args.layout), timeout=60)
    instances = {row["instance_id"] for row in descriptors}
    if not instances:
        raise RuntimeError("layout has no registered serving instances")
    policy = dict(BASE_POLICY, **_instance_roles(args.layout, instances))
    policy["migration_enabled"] = False
    if not policy["small_instances"] or not policy["large_instances"]:
        policy["large_fresh_traffic_share"] = 0.0
    report = run_hardware_qualification(
        requests=len(instances) * args.prompts_per_instance, max_tokens=args.decode_tokens,
        policy=policy, require_migration=False,
    )
    observed = {row.get("initial_instance") for row in report["records"] if not row.get("error")}
    report["all_instances_served"] = observed == instances
    report["passed"] = report["passed"] and report["all_instances_served"]
    return {"action": "gateway-smoke", "serving": report}


def cmd_qualify_migration(args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 3 gate: continuation and ownership qualification under recompute."""
    from hetero.qualify_migration import (
        QUALIFICATION_POLICY,
        run_hardware_qualification,
        run_protocol_probes,
    )

    probes = run_protocol_probes()
    emit_log(f"protocol probes passed={probes['passed']}")
    policy = json.loads(args.policy) if args.policy else dict(QUALIFICATION_POLICY)
    hardware = run_hardware_qualification(
        requests=args.qual_requests,
        max_tokens=args.qual_max_tokens,
        prompt_offset=args.qual_prompt_offset,
        policy=policy,
    )
    return {
        "action": "qualify-migration",
        "protocol_probes": probes,
        "hardware": hardware,
        "passed": bool(probes["passed"] and hardware["passed"]),
    }


def _json_input_path(key: str, option: str, directory: str) -> pathlib.Path:
    import hashlib

    raw = os.environ.get(key)
    if raw is None:
        raise ValueError(f"submit the frozen JSON using hetero_ctl.py {option} <path>")
    content = raw.encode("utf-8")
    path = pathlib.Path("/tmp/omp_hetero") / directory / (hashlib.sha256(content).hexdigest() + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def cmd_qualify_continuation(args: argparse.Namespace) -> Dict[str, Any]:
    from hetero.qualify_continuation import run_continuation_qualification
    from hetero.gateway.lifecycle import find_gateway
    from hetero.experiment import _write_iterations

    gateway = find_gateway()
    if gateway is None:
        raise RuntimeError("deploy the gateway and idle instances before qualification")
    ray.get(gateway.start_capture.remote(), timeout=300)
    try:
        report = run_continuation_qualification(protocol_path=str(_json_input_path(
            "OMP_HETERO_PROTOCOL_JSON", "--protocol", "protocols"
        )))
    finally:
        captures = ray.get(gateway.stop_capture.remote(), timeout=600)
    report["iteration_stats"] = _write_iterations(
        pathlib.Path(report["artifact_dir"]), "continuation", 1, captures
    )
    report["passed"] = report["passed"] and report["iteration_stats"]["one_to_one"]
    return {"action": "qualify-continuation", "qualification": report}


def cmd_qualify_lifecycle(args: argparse.Namespace) -> Dict[str, Any]:
    from hetero.qualify_lifecycle import run_lifecycle_qualification

    if not args.restart_instance:
        raise ValueError("qualify-lifecycle requires --restart-instance <instance-id>")
    inventory, source = _resolve_inventory(args, allow_probe=False)
    report = run_lifecycle_qualification(
        inventory, _deployment_config(args), instance_id=args.restart_instance,
        requests_per_stage=args.prompts_per_instance, max_tokens=args.decode_tokens,
        prompt_offset=args.qual_prompt_offset, timeout_s=args.drain_timeout_s,
    )
    return {
        "action": "qualify-lifecycle", "inventory_source": source, "qualification": report,
        "artifact_dir": str(pathlib.Path(report["artifact_path"]).parent),
    }


def cmd_mechanism_cost(args: argparse.Namespace) -> Dict[str, Any]:
    from hetero.mechanism_cost import FIDELITY_PROTOCOL_PATH, run_mechanism_cost

    protocol = _json_input_path("OMP_HETERO_PROTOCOL_JSON", "--protocol", "protocols")
    evaluation = _json_input_path(
        "OMP_HETERO_EVALUATION_PROTOCOL_JSON", "--evaluation-protocol", "protocols"
    )
    qualification = _json_input_path(
        "OMP_HETERO_QUALIFICATION_REPORT_JSON", "--qualification-report", "qualifications"
    )
    fidelity = _json_input_path(
        "OMP_HETERO_FIDELITY_PROTOCOL_JSON", "--protocol", "protocols"
    )
    # The helper validates this frozen prerequisite against its pinned hash.
    fidelity_path = ROOT / FIDELITY_PROTOCOL_PATH
    fidelity_path.parent.mkdir(parents=True, exist_ok=True)
    fidelity_path.write_bytes(fidelity.read_bytes())
    report = run_mechanism_cost(
        protocol_path=str(protocol), qualification_report_path=str(qualification),
        evaluation_protocol_path=str(evaluation),
    )
    return {"action": "mechanism-cost", "mechanism_cost": report}


def cmd_build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    """Freezes a workload manifest on the cluster, where the tokenizer lives."""
    from hetero.manifest import build_manifest, manifest_path

    built = []
    for name in (args.workloads or "W0,W1,W2").split(","):
        name = name.strip()
        manifest = build_manifest(name, args.model)
        path = manifest_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, separators=(",", ":")))
        staged = pathlib.Path(
            os.environ.get("OMP_HETERO_ARTIFACTS", "/tmp/omp_hetero/artifacts")
        ) / "manifests"
        staged.mkdir(parents=True, exist_ok=True)
        (staged / f"{name}.json").write_text(json.dumps(manifest, separators=(",", ":")))
        summary = {k: v for k, v in manifest.items() if k != "requests"}
        summary["path"] = str(path)
        emit_log(
            f"manifest {name}: {manifest['num_requests']} requests, "
            f"{manifest['useful_output_tokens']} useful tokens, "
            f"sha256={manifest['manifest_sha256'][:16]}"
        )
        built.append(summary)
    return {
        "action": "build-manifest",
        "manifests": built,
        "artifact_dir": str(
            pathlib.Path(os.environ.get("OMP_HETERO_ARTIFACTS", "/tmp/omp_hetero/artifacts"))
            / "manifests"
        ),
    }


def cmd_run_arm(args: argparse.Namespace) -> Dict[str, Any]:
    """Runs one whole-workload arm repetition: the replication unit."""
    from hetero.experiment import run_arm

    overrides = json.loads(args.policy) if args.policy else None
    return {
        "action": "run-arm",
        "result": run_arm(
            arm=args.arm,
            workload=args.workload,
            repetition=args.repetition,
            run_id=args.run_id,
            timeout_s=args.arm_timeout,
            policy_overrides=overrides,
            protocol_path=str(_json_input_path("OMP_HETERO_PROTOCOL_JSON", "--protocol", "protocols")),
        ),
    }


def cmd_run_sequence(args: argparse.Namespace) -> Dict[str, Any]:
    """Runs a balanced multi-arm, multi-repetition block inside one job."""
    from hetero.experiment import run_sequence

    return {
        "action": "run-sequence",
        "result": run_sequence(
            arms=[a.strip() for a in args.arms.split(",") if a.strip()],
            workload=args.workload,
            repetitions=args.repetitions,
            run_id=args.run_id,
            timeout_s=args.arm_timeout,
            protocol_path=str(_json_input_path("OMP_HETERO_PROTOCOL_JSON", "--protocol", "protocols")),
        ),
    }


COMMANDS = {
    "inventory": cmd_inventory,
    "plan": cmd_plan,
    "preflight": cmd_preflight,
    "deploy": cmd_deploy,
    "inspect": cmd_inspect,
    "drain": cmd_drain,
    "gate": cmd_gate,
    "release": cmd_release,
    "gateway-up": cmd_gateway_up,
    "gateway-down": cmd_gateway_down,
    "gateway-status": cmd_gateway_status,
    "qualify-migration": cmd_qualify_migration,
    "cache-layout": cmd_cache_layout,
    "qualify-continuation": cmd_qualify_continuation,
    "qualify-lifecycle": cmd_qualify_lifecycle,
    "mechanism-cost": cmd_mechanism_cost,
    "gateway-smoke": cmd_gateway_smoke,
    "build-manifest": cmd_build_manifest,
    "run-arm": cmd_run_arm,
    "run-sequence": cmd_run_sequence,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--layout", default="M", help="built-in layout id (M, H, S)")
    parser.add_argument("--layout-file", help="explicit layout request JSON")
    parser.add_argument("--inventory-file", help="saved slice inventory JSON")
    parser.add_argument("--probe-cpu", type=float, default=1.0)
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    parser.add_argument("--probe-timeout-s", type=float, default=900.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reobserve", action="store_true")
    parser.add_argument("--cpu-per-host", type=float, default=8.0)
    parser.add_argument("--engine-init-timeout-s", type=float, default=5400.0)
    parser.add_argument("--engine-settings", help="JSON object of engine settings")
    parser.add_argument("--model", default="/models/gemma-4-31b")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--instances", help="comma separated instance ids")
    parser.add_argument("--keep-resources", action="store_true")
    parser.add_argument("--drain-timeout-s", type=float, default=600.0)
    parser.add_argument("--prompts-per-instance", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--restart-instance", help="instance id to restart during the gate")
    parser.add_argument("--policy", help="JSON routing policy object")
    parser.add_argument("--policy-file", help="routing policy JSON path")
    parser.add_argument("--migration", action="store_true")
    parser.add_argument("--head-node-ip")
    parser.add_argument("--no-bind-labels", action="store_true")
    parser.add_argument("--qual-requests", type=int, default=48)
    parser.add_argument("--qual-max-tokens", type=int, default=1024)
    parser.add_argument("--qual-prompt-offset", type=int, default=0)
    parser.add_argument("--workloads", default=None, help="comma separated: W0,W1,W2")
    parser.add_argument("--arm", default="M0")
    parser.add_argument("--workload", default="W0")
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--run-id", default="run")
    parser.add_argument("--arm-timeout", type=float, default=7200.0)
    parser.add_argument("--arms", default="M0,MR")
    parser.add_argument("--repetitions", type=int, default=3)
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    _connect()
    emit_event("COMMAND_BEGIN", command=args.command)
    payload = COMMANDS[args.command](args)
    payload["elapsed_s"] = round(time.time() - started, 3)
    payload["ok"] = True
    emit_measurements(payload)
    emit_event("COMMAND_COMPLETE", command=args.command, elapsed_s=payload["elapsed_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
