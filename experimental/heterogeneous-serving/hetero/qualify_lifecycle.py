"""Driver-side scoped restart qualification on an already deployed fabric.

This is an administrative, untimed qualification, never a workload arm. It
uses real ShareGPT requests through the same Envoy ExtProc transport as the
experiment driver. Only the selected instance is drained/recreated. A failed
transition stays fenced for explicit operator recovery; no all-engine restart,
policy workaround, journal reset, or automatic retry hides a failed check.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import pathlib
import time
import uuid
from typing import Any, Dict, Optional

import ray

from .deployment import DeploymentConfig, _find_engine_actor, deploy, drain_deployment
from .evidence import emit_event
from .gateway.lifecycle import find_gateway
from .placement import list_owned_placement_groups, pg_name
from .registry import find_registry
from .slice_inventory import SliceInventory
from .workload import load_catalogue, prompt_payload_digest, unique_prompts


def run_lifecycle_qualification(
    inventory: SliceInventory,
    config: DeploymentConfig,
    *,
    instance_id: str,
    requests_per_stage: int = 24,
    max_tokens: int = 32,
    catalogue_name: str = "small",
    prompt_offset: int = 0,
    timeout_s: float = 600.0,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Invoke from the existing Ray/RCM driver; this function submits no job.

    Requires an idle complete deployment with the requested static config and
    a live gateway. Five distinct prompt batches prove full-fabric service,
    withdrawal, service during native restart, ready-but-excluded service, and
    fresh-generation readmission. Returned timings are operation evidence, not
    throughput or latency comparisons. The report is checkpointed on failure.
    """
    from .experiment import ARTIFACT_ROOT, INGRESS_URL, _dispatch_http, _instance_roles

    if requests_per_stage < 1 or max_tokens < 1 or timeout_s <= 0:
        raise ValueError("lifecycle qualification requires positive request/token counts and timeout")
    gateway, registry = find_gateway(), find_registry()
    if gateway is None or registry is None:
        raise RuntimeError("lifecycle qualification requires the existing gateway and registry")
    before = ray.get(gateway.status.remote(), timeout=60)
    if (before["layout_id"] != config.layout.layout_id or before.get("maintenance")
            or before.get("capturing") or not before["discovery"].get("ready")):
        raise RuntimeError("qualification requires a complete idle static fabric, outside maintenance/capture")
    rows_before = ray.get(registry.snapshot.remote(config.layout.layout_id), timeout=60)
    originals = {row["instance_id"]: row for row in rows_before}
    if instance_id not in originals or len(originals) < 2:
        raise ValueError("select one existing instance with at least one peer")
    names, peers = set(originals), set(originals) - {instance_id}
    # Validate the complete desired configuration before any destructive action,
    # without initializing, reconfiguring, or recreating a peer.
    from .layout import resolve_layout

    resolved = resolve_layout(inventory, config.layout)
    if (resolved.to_dict()["layout_digest"] != originals[instance_id]["config_digest"]
            or {item.instance_id for item in resolved.instances} != names):
        raise RuntimeError("qualification config differs from the deployed static layout")
    for instance in resolved.instances:
        actor = _find_engine_actor(config.layout.layout_id, instance.instance_id)
        verdict = ray.get(actor.matches.remote(instance.to_dict(), config.engine_settings), timeout=60)
        if not verdict["matches"]:
            raise RuntimeError(f"qualification engine settings differ: {verdict}")
    actor_ids_before = {
        name: _find_engine_actor(config.layout.layout_id, name)._actor_id.hex() for name in names
    }
    gateway_id = gateway._actor_id.hex()
    groups_before = [row for row in list_owned_placement_groups() if row["state"] != "REMOVED"]
    idle = ray.get(gateway.await_idle_instances.remote(timeout_s), timeout=timeout_s + 60)
    if set(idle) != names or any(not row.get("idle") for row in idle.values()):
        raise RuntimeError(f"native engines are not idle: {idle}")
    catalogue = load_catalogue(catalogue_name)
    prompts = unique_prompts(catalogue, requests_per_stage * 5, offset=prompt_offset)
    invocation = uuid.uuid4().hex
    path = pathlib.Path(output_path) if output_path else ARTIFACT_ROOT / f"lifecycle_{invocation}" / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    original_policy = before["core"]["config"]["policy"]
    diagnostic_policy = {
        **original_policy, **_instance_roles(config.layout.layout_id, sorted(names)),
        "migration_enabled": False, "large_fresh_traffic_share": 0.5,
        "reserve_large_headroom_blocks": 0,
    }
    report: Dict[str, Any] = {
        "qualification": "gateway_scoped_lifecycle", "passed": False,
        "layout_id": config.layout.layout_id, "instance_id": instance_id,
        "gateway_actor_id": gateway_id, "ingress_url": INGRESS_URL,
        "catalogue_sha256": catalogue["sha256"], "prompt_sha256": prompt_payload_digest(prompts),
        "requests_per_stage": requests_per_stage, "max_tokens": max_tokens,
        "before": before, "registry_before": rows_before, "native_idle_before": idle,
        "actor_ids_before": actor_ids_before, "placement_groups_before": groups_before,
        "stages": {}, "artifact_path": str(path),
    }

    def save() -> None:
        path.write_text(json.dumps(report, indent=2) + "\n")

    def probe(stage: str, index: int, expected: set[str]) -> None:
        selected = prompts[index * requests_per_stage:(index + 1) * requests_per_stage]
        payload = [{"request_id": f"lifecycle-{invocation}-{stage}-{n}", "prompt": row["prompt"],
                    "max_tokens": max_tokens, "ignore_eos": True}
                   for n, row in enumerate(selected)]
        records = asyncio.run(_dispatch_http(payload, concurrency=requests_per_stage, timeout_s=timeout_s))
        observed = {row.get("initial_instance") for row in records if not row.get("error")}
        valid = (len(records) == len(payload)
                 and {row["request_id"] for row in records} == {row["request_id"] for row in payload}
                 and observed == expected
                 and all(not row.get("error") and row.get("useful_output_tokens") == max_tokens
                         and len(row.get("segments", [])) == 1
                         and row["segments"][0]["instance_id"] in expected for row in records))
        authoritative = {row["request_id"]: row for row in ray.get(gateway.records.remote(), timeout=60)}
        valid = valid and all(
            row.get("output_token_ids") == authoritative.get(row["request_id"], {}).get("output_token_ids")
            for row in records
        )
        report["stages"][stage] = {"passed": valid, "expected_instances": sorted(expected),
                                   "observed_instances": sorted(str(name) for name in observed),
                                   "client_records": records}
        save()
        if not valid:
            raise RuntimeError(f"ExtProc lifecycle probe failed at {stage}")

    save()
    emit_event("LIFECYCLE_QUALIFICATION_BEGIN", instance_id=instance_id, artifact_path=str(path))
    try:
        report["policy"] = ray.get(gateway.set_policy.remote(diagnostic_policy), timeout=60)
        probe("before", 0, names)
        report["drain"] = drain_deployment(config.layout.layout_id, [instance_id], timeout_s=timeout_s)
        maintenance_id = report["drain"]["maintenance"]["id"]
        withdrawn = ray.get(gateway.status.remote(), timeout=60)
        report["withdrawn"] = withdrawn
        if (instance_id in withdrawn["bindings"] or not withdrawn["discovery"].get("ready")
                or any(row["instance_id"] == instance_id for row in withdrawn["discovery"]["resolved"])
                or any(row["instance_id"] == instance_id for row in report["drain"]["registry_snapshot"])):
            raise RuntimeError("retiring generation/address was not withdrawn")
        probe("withdrawn", 1, peers)
        # Actual peer traffic overlaps the target's native initialization. This
        # is not an artificial scheduler or a simulated restart duration.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            restarted = executor.submit(deploy, inventory, config, instance_ids=[instance_id])
            report["restart_probe_started_at_s"] = time.time()
            report["restart_pending_at_probe"] = not restarted.done()
            probe("restarting", 2, peers)
            report["restart_pending_after_probe"] = not restarted.done()
            report["redeploy"] = restarted.result()
        if not report["restart_pending_at_probe"]:
            raise RuntimeError("no peer probe overlapped the native restart")
        report["ready_but_excluded"] = ray.get(gateway.reconcile.remote(), timeout=300)
        probe("ready_but_excluded", 3, peers)
        report["promotion"] = ray.get(gateway.finish_maintenance.remote(maintenance_id=maintenance_id), timeout=300)
        probe("restored", 4, names)
        after = ray.get(gateway.status.remote(), timeout=60)
        rows_after = ray.get(registry.snapshot.remote(config.layout.layout_id), timeout=60)
        current = {row["instance_id"]: row for row in rows_after}
        actor_ids_after = {
            name: _find_engine_actor(config.layout.layout_id, name)._actor_id.hex() for name in names
        }
        groups_after = [row for row in list_owned_placement_groups() if row["state"] != "REMOVED"]
        target_pg = pg_name(config.layout.layout_id, instance_id)
        unaffected = lambda rows: {(row["name"], row["pg_id"]) for row in rows if row["name"] != target_pg}
        checks = {
            "gateway_actor_preserved": find_gateway()._actor_id.hex() == gateway_id,
            "peer_actors_preserved": all(actor_ids_after[name] == actor_ids_before[name] for name in peers),
            "peer_bindings_preserved": all(
                all(current[name][key] == originals[name][key] for key in (
                    "generation", "endpoint", "placement_group_id", "engine_actor_name", "config_digest"))
                for name in peers
            ),
            "unselected_groups_preserved": unaffected(groups_before) == unaffected(groups_after),
            "new_target_actor": actor_ids_after[instance_id] != actor_ids_before[instance_id],
            "new_target_generation": current[instance_id]["generation"] > originals[instance_id]["generation"],
            "fresh_gateway_binding": after["bindings"][instance_id]["generation"] == current[instance_id]["generation"],
            "fresh_discovery_binding": any(row["instance_id"] == instance_id
                and row["generation"] == current[instance_id]["generation"]
                and row["endpoint"] == current[instance_id]["endpoint"] and row["routable"]
                for row in after["discovery"]["resolved"]),
            "no_duplicate_groups": len({row["name"] for row in groups_after}) == len(groups_after),
            "no_migration_or_owner_left": not after["core"]["backlog"]["inflight"]
                and not after["core"]["backlog"]["response_tasks"] and after["core"]["recovery"]["ready"]
                and all(row["state"] in ("released", "aborted") for row in after["core"]["state"]["transactions"]),
            "full_static_fabric_restored": set(current) == names and not after["maintenance"] and after["discovery"]["ready"],
        }
        report.update(after=after, registry_after=rows_after, actor_ids_after=actor_ids_after,
                      placement_groups_after=groups_after, checks=checks, passed=all(checks.values()))
        if not report["passed"]:
            raise RuntimeError("scoped lifecycle invariants failed")
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["passed"] = False
        raise RuntimeError(f"lifecycle qualification failed; evidence: {path}") from exc
    finally:
        try:
            status = ray.get(gateway.status.remote(), timeout=60)
            report["final_status"] = status
            # Failure is deliberately not an implicit unfence/redeploy/reset.
            if (not status["maintenance"] and not status.get("capturing")
                    and not status["core"]["backlog"]["response_tasks"]
                    and not status["core"]["backlog"]["inflight"]
                    and status["core"]["recovery"]["ready"]
                    and all(row["state"] in ("released", "aborted")
                            for row in status["core"]["state"]["transactions"])):
                report["restored_policy"] = ray.get(gateway.set_policy.remote(original_policy), timeout=60)
        except Exception as exc:
            report["finalization_error"] = f"{type(exc).__name__}: {exc}"
            report["passed"] = False
            if "error" not in report:
                raise RuntimeError(f"lifecycle finalization failed; evidence: {path}") from exc
        finally:
            save()
    emit_event("LIFECYCLE_QUALIFICATION_COMPLETE", passed=True, instance_id=instance_id, artifact_path=str(path))
    return report
