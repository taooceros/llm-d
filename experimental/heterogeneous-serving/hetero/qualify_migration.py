"""Phase 3 qualification: continuation and ownership under recompute migration.

Two layers of evidence:

* **Protocol probes** exercise the authoritative ownership/reservation state
  machine directly: duplicate and stale proposals, two small instances racing
  for the same destination credit, destination refusal, delayed commits and
  cancellation.  These are deterministic and run in-process at the gateway.
* **Hardware qualification** runs a real ShareGPT batch through the llm-d
  gateway on the live engines until the frozen policy actually migrates, then
  checks the delivered result against the original generation contract.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import ray

from .evidence import emit_event, emit_log
from .gateway.policy import RoutingPolicy
from .gateway.state import GatewayState, OwnershipState, TransactionState
from .workload import load_catalogue, prompt_payload_digest, unique_prompts

QUALIFICATION_POLICY = {
    "small_instances": ["small_a", "small_b"],
    "large_instances": ["large"],
    "migration_enabled": True,
    "migration_mechanism": "recompute",
    "large_fresh_traffic_share": 0.0,
    "min_generated_tokens": 64,
    "min_context_tokens": 512,
    "min_remaining_tokens": 128,
    "max_migrations_per_request": 1,
    "max_inflight_migrations_per_destination": 2,
    "max_inflight_migrations_total": 3,
    "source_pressure_free_block_fraction": 0.35,
    "reserve_large_headroom_blocks": 0,
}


# --------------------------------------------------------------- protocol


def run_protocol_probes() -> Dict[str, Any]:
    """Deterministic checks of the authoritative handoff state machine."""
    checks: Dict[str, Any] = {}

    def fresh() -> GatewayState:
        state = GatewayState()
        for name, total in (("small_a", 205), ("small_b", 205), ("large", 566)):
            state.observe_capacity(
                name,
                total_blocks=total,
                free_blocks=total,
                block_size=256,
                running=0,
                waiting=0,
                resident_growth_allowance=0,
            )
        return state

    # 1. Duplicate proposal for the same request is reconciled, not scheduled twice.
    state = fresh()
    state.claim("r1", "small_a", 1)
    first, err1 = state.open_transaction(
        request_id="r1", source_instance="small_a", destination_instance="large",
        mechanism="recompute", blocks=4, growth_blocks=2, destination_generation=1,
        max_inflight_per_destination=2,
    )
    second, err2 = state.open_transaction(
        request_id="r1", source_instance="small_a", destination_instance="large",
        mechanism="recompute", blocks=4, growth_blocks=2, destination_generation=1,
        max_inflight_per_destination=2,
    )
    checks["duplicate_proposal_declined"] = {
        "first_opened": first is not None,
        "second_declined": second is None and err2 == "duplicate_proposal",
        "reason": err2,
    }

    # 2. A proposal from a stale owner is refused.
    stale, err = state.open_transaction(
        request_id="r1", source_instance="small_b", destination_instance="large",
        mechanism="recompute", blocks=4, growth_blocks=2, destination_generation=1,
        max_inflight_per_destination=2,
    )
    checks["stale_owner_declined"] = {"declined": stale is None, "reason": err}

    # 3. Simultaneous proposals from both small instances cannot double-spend
    #    the same destination credit.
    state = fresh()
    state.observe_capacity(
        "large", total_blocks=566, free_blocks=10, block_size=256, running=0,
        waiting=0, resident_growth_allowance=0,
    )
    state.claim("a1", "small_a", 1)
    state.claim("b1", "small_b", 1)
    ta, _ = state.open_transaction(
        request_id="a1", source_instance="small_a", destination_instance="large",
        mechanism="recompute", blocks=8, growth_blocks=2, destination_generation=1,
        max_inflight_per_destination=4,
    )
    tb, errb = state.open_transaction(
        request_id="b1", source_instance="small_b", destination_instance="large",
        mechanism="recompute", blocks=8, growth_blocks=2, destination_generation=1,
        max_inflight_per_destination=4,
    )
    checks["reservation_is_atomic"] = {
        "first_reserved": ta is not None,
        "second_declined": tb is None,
        "reason": errb,
        "destination_admissible_after": state.capacity("large").admissible_blocks,
    }

    # 4. Destination refusal returns the credit and restores the source owner.
    state.abort(ta.transaction_id, "destination_refused")
    held_before_cleanup = state.capacity("large").admissible_blocks
    state.finish_abort(ta.transaction_id, {"aborted": True}, {"resumed": True})
    owner = state.owner("a1")
    checks["abort_restores_owner_and_credit"] = {
        "held_until_cleanup": held_before_cleanup == 0,
        "owner_state": owner.state.value,
        "owner_instance": owner.instance_id,
        "destination_admissible": state.capacity("large").admissible_blocks,
    }

    # 5. Commit is idempotent and a late timeout cannot roll it back.
    state = fresh()
    state.claim("r2", "small_a", 1)
    txn, _ = state.open_transaction(
        request_id="r2", source_instance="small_a", destination_instance="large",
        mechanism="recompute", blocks=4, growth_blocks=2, destination_generation=7,
        max_inflight_per_destination=2,
    )
    state.mark_reserved(txn.transaction_id, {"accepted": True, "native_reserved_blocks": 6})
    state.mark_quiesced(txn.transaction_id, 1200)
    state.mark_prepared(txn.transaction_id, replay_tokens=1200)
    committed = state.commit(txn.transaction_id, 7)
    again = state.commit(txn.transaction_id, 7)
    late_abort = state.abort(txn.transaction_id, "late_timeout")
    owner = state.owner("r2")
    checks["commit_idempotent_and_final"] = {
        "committed": committed is not None,
        "second_commit_same": again is not None
        and again.transaction_id == committed.transaction_id,
        "late_abort_did_not_roll_back": late_abort.state.value == "committed",
        "owner_instance": owner.instance_id,
        "owner_epoch": owner.epoch,
    }

    # 6. Release is only legal after commit; duplicates are absorbed.
    state.mark_activated(txn.transaction_id, {"activated": True})
    released = state.release(txn.transaction_id, {"released": True})
    released_again = state.release(txn.transaction_id, {"released": True})
    checks["release_after_commit_only"] = {
        "released": released is not None,
        "duplicate_release_absorbed": released_again is not None
        and released_again.state.value == "released",
    }

    # 7. Cancellation releases reservations without reviving the request.
    state = fresh()
    state.claim("r3", "small_a", 1)
    txn3, _ = state.open_transaction(
        request_id="r3", source_instance="small_a", destination_instance="large",
        mechanism="recompute", blocks=4, growth_blocks=2, destination_generation=1,
        max_inflight_per_destination=2,
    )
    state.cancel("r3")
    state.finish_abort(txn3.transaction_id, {"aborted": True}, {"resumed": True})
    state.record_cancellation("r3", {"cancelled": True})
    checks["cancel_releases_reservation"] = {
        "transaction_state": state.transaction(txn3.transaction_id).state.value,
        "owner_state": state.owner("r3").state.value,
        "destination_admissible": state.capacity("large").admissible_blocks,
    }

    passed = (
        checks["duplicate_proposal_declined"]["second_declined"]
        and checks["stale_owner_declined"]["declined"]
        and checks["reservation_is_atomic"]["second_declined"]
        and checks["abort_restores_owner_and_credit"]["owner_state"] == "owned"
        and checks["abort_restores_owner_and_credit"]["held_until_cleanup"]
        and checks["commit_idempotent_and_final"]["committed"]
        and checks["commit_idempotent_and_final"]["late_abort_did_not_roll_back"]
        and checks["release_after_commit_only"]["released"]
        and checks["cancel_releases_reservation"]["transaction_state"] == "aborted"
        and checks["cancel_releases_reservation"]["owner_state"] == "cancelled"
    )
    return {"checks": checks, "passed": bool(passed)}


# --------------------------------------------------------------- hardware


def run_hardware_qualification(
    *,
    requests: int = 48,
    max_tokens: int = 1024,
    prompt_offset: int = 0,
    policy: Optional[Dict[str, Any]] = None,
    catalogue_name: str = "small",
    timeout_s: float = 1800.0,
    require_migration: bool = True,
) -> Dict[str, Any]:
    """Runs a real ShareGPT batch until the frozen policy actually migrates."""
    from .gateway.lifecycle import find_gateway
    from .experiment import _dispatch_http, _write_iterations, _write_gz, ARTIFACT_ROOT, INGRESS_URL

    gateway = find_gateway()
    if gateway is None:
        raise RuntimeError("gateway actor is not running; start it with `gateway-up`")
    applied = ray.get(
        gateway.set_policy.remote(policy or QUALIFICATION_POLICY), timeout=60
    )
    ray.get(gateway.reset_records.remote(), timeout=60)
    ray.get(gateway.reconcile.remote(bind_labels=False), timeout=300)
    ray.get(gateway.resume_admission.remote(), timeout=120)
    ray.get(gateway.await_idle_instances.remote(300.0), timeout=360)
    ray.get(gateway.start_capture.remote(), timeout=300)
    catalogue = load_catalogue(catalogue_name)
    prompts = unique_prompts(catalogue, requests, offset=prompt_offset)
    invocation = uuid.uuid4().hex
    payload = [
        {
            "request_id": f"qual-{invocation}-{index:04d}",
            "prompt": record["prompt"],
            "max_tokens": max_tokens,
            "ignore_eos": True,
        }
        for index, record in enumerate(prompts)
    ]
    emit_event("MIGRATION_QUALIFICATION_BEGIN", requests=len(payload), max_tokens=max_tokens)
    started = time.perf_counter()
    client_records = asyncio.run(_dispatch_http(
        payload, concurrency=len(payload), timeout_s=timeout_s
    ))
    wall_s = time.perf_counter() - started
    artifact_dir = ARTIFACT_ROOT / f"qualification_{time.time_ns()}"
    artifact_dir.mkdir(parents=True)
    _write_gz(artifact_dir / "client_records.json.gz", client_records)
    emit_event("MIGRATION_CLIENT_RECORDS_SAVED", artifact_dir=str(artifact_dir))
    captures = ray.get(gateway.stop_capture.remote(), timeout=600)
    iteration_stats = _write_iterations(artifact_dir, "qualification", 1, captures)
    records = ray.get(gateway.records.remote(), timeout=120)
    by_id = {r["request_id"]: r for r in records}
    client_matches = (
        len(client_records) == len(payload) == len(records) == len(by_id)
        and {r["request_id"] for r in client_records} == {r["request_id"] for r in payload}
        and all(
            not row.get("error")
            and row["output_token_ids"] == by_id.get(row["request_id"], {}).get("output_token_ids")
            for row in client_records
        )
    )
    status = ray.get(gateway.status.remote(), timeout=120)
    traces = ray.get(gateway.traces.remote(8000), timeout=120)

    transactions = status["core"]["state"]["transactions"]
    committed = [t for t in transactions if t["state"] in ("committed", "released")]
    declined = [t for t in transactions if t["state"] in ("aborted",)]
    migrated_records = [r for r in records if len(r.get("segments", [])) > 1]

    accounting = _check_accounting(records, max_tokens)
    ordering = _check_handoff_ordering(committed)
    authority = _check_gateway_authority(records, transactions)

    result = {
        "qualification_mode": "migration" if require_migration else "serving_only",
        "policy": applied,
        "ingress_url": INGRESS_URL,
        "ingress_protocols": sorted({r["ingress_protocol"] for r in client_records
                                      if "ingress_protocol" in r}),
        "client_delivery_matches": client_matches,
        "artifact_dir": str(artifact_dir),
        "iteration_stats": iteration_stats,
        "requests": len(payload),
        "max_tokens": max_tokens,
        "prompt_digest": prompt_payload_digest(prompts),
        "catalogue_sha256": catalogue["sha256"],
        "wall_s": round(wall_s, 3),
        "migrations_committed": len(committed),
        "migrations_declined": len(declined),
        "records_with_migration": len(migrated_records),
        "counters": status["core"]["state"]["counters"],
        "transactions": transactions,
        "accounting": accounting,
        "handoff_ordering": ordering,
        "gateway_authority": authority,
        "traces": traces[-2000:],
        "records": records,
    }
    result["passed"] = bool(
        (len(committed) > 0 if require_migration else len(committed) == 0)
        and client_matches
        and iteration_stats["one_to_one"]
        and accounting["all_budgets_exact"]
        and accounting["no_duplicate_delivery"]
        and (ordering["all_ordered"] if require_migration else not committed)
        and authority["all_continuations_authorized"]
    )
    emit_event(
        "MIGRATION_QUALIFICATION_RESULT",
        passed=result["passed"],
        committed=len(committed),
        declined=len(declined),
    )
    return result


def _check_accounting(
    records: List[Dict[str, Any]], max_tokens: int | Dict[str, int]
) -> Dict[str, Any]:
    """Per-request committed-prefix preservation and exact generation budget."""
    rows = []
    for record in records:
        if record.get("error"):
            rows.append({"request_id": record.get("request_id"), "error": record["error"]})
            continue
        delivered = len(record["output_token_ids"])
        segment_tokens = sum(
            s.get("output_tokens", s.get("new_output_tokens", 0)) for s in record["segments"]
        )
        assembled = [token for segment in record["segments"]
                     for token in segment.get("output_token_ids", segment.get("new_output_token_ids", []))]
        budget = max_tokens.get(record["request_id"]) if isinstance(max_tokens, dict) else max_tokens
        rows.append(
            {
                "request_id": record["request_id"],
                "delivered_tokens": delivered,
                "accounted_tokens": record["useful_output_tokens"],
                "segment_tokens": segment_tokens,
                "budget_exact": delivered == budget,
                "no_double_count": (
                    delivered == record["useful_output_tokens"] == segment_tokens
                    and assembled == record["output_token_ids"]
                ),
                "segments": len(record["segments"]),
                "instances": [
                    s.get("segment_instance") or s.get("instance_id") for s in record["segments"]
                ],
            }
        )
    valid = [r for r in rows if "error" not in r]
    return {
        "per_request": rows,
        "failed_requests": [r for r in rows if "error" in r],
        "all_budgets_exact": bool(valid) and len(valid) == len(rows) and all(r["budget_exact"] for r in valid),
        "no_duplicate_delivery": bool(valid) and len(valid) == len(rows) and all(r["no_double_count"] for r in valid),
        "total_useful_tokens": sum(r["delivered_tokens"] for r in valid),
    }


def _check_handoff_ordering(committed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reserve -> quiesce -> prepare -> commit -> release, in that order."""
    rows = []
    expected = ["reserved", "native_reserved", "quiesced", "prepared", "committed", "released"]
    for txn in committed:
        names = [name for name, _ in txn["timeline"]]
        times = [ts for _, ts in txn["timeline"]]
        ordered = names[: len(expected)] == expected and times == sorted(times)
        rows.append(
            {
                "transaction_id": txn["transaction_id"],
                "timeline": names,
                "ordered": ordered,
                "source": txn["source_instance"],
                "destination": txn["destination_instance"],
                "epoch": txn["destination_epoch"],
                "snapshot_tokens": txn["snapshot_tokens"],
                "replay_tokens": txn["replay_tokens"],
                "handoff_s": round(
                    (txn["committed_at"] or 0) - txn["created_at"], 4
                ),
            }
        )
    return {"per_transaction": rows, "all_ordered": bool(rows) and all(r["ordered"] for r in rows)}


def _check_gateway_authority(
    records: List[Dict[str, Any]], transactions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """No continuation exists without a gateway-authorised transaction."""
    authorized = {t["transaction_id"]: t for t in transactions}
    rows = []
    for record in records:
        for segment in record.get("segments", []):
            txn_id = segment.get("transaction_id")
            if txn_id is None:
                continue
            rows.append(
                {
                    "request_id": record["request_id"],
                    "transaction_id": txn_id,
                    "authorized": txn_id in authorized,
                    "committed": authorized.get(txn_id, {}).get("state")
                    in ("committed", "released"),
                }
            )
    return {
        "continuations": rows,
        "all_continuations_authorized": all(r["authorized"] and r["committed"] for r in rows),
        "count": len(rows),
    }
