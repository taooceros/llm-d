"""Real native continuation qualification, never headline serving evidence.

Run only inside Main's RCM/RayJobSupervisor job against already resident M
engines.  This module neither launches nor stops engines.  Every generated
history comes from the mounted model and every handoff uses a private durable
GatewayState ledger, real native reservations and authoritative certificates.

The prospective protocol is byte-pinned.  Full token/logprob vectors and native
receipts live in artifacts; the returned report contains concise verdicts and
artifact identities.  A failed criterion is never turned into a new threshold.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import pathlib
import time
import uuid
from typing import Any, Dict, Optional, Sequence

from .gateway.state import COMMITTED_STATES, GatewayState, TransactionState
from .manifest import _assign_unique, _tokenise_catalogue, load_manifest, manifest_path

PROTOCOL_PATH = "results/hetero/continuation_qualification_protocol.json"
PROTOCOL_SHA256 = "e8fb4c25c1e615aba24560145d82c5013d5f8a646c717903199406edaf2529f5"
EXCLUDED_WORKLOADS = ("W0", "W1", "W2", "E_SHORT", "E_LONG")
SUFFIX_TOKENS = 32
TOP_K = 32


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _json_safe(value: Any) -> Any:
    """Preserve nonfinite observations without emitting nonstandard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite": repr(value)}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _save(directory: pathlib.Path, name: str, value: Any) -> Dict[str, Any]:
    raw = json.dumps(_json_safe(value), separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    path = directory / name
    if path.suffix == ".gz":
        with gzip.open(path, "wb") as stream:
            stream.write(raw)
    else:
        path.write_bytes(raw)
    return {"path": str(path), "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "uncompressed_bytes": len(raw)}


def _select_cases(protocol: Dict[str, Any], catalogue_name: str,
                  manifest_root: Optional[pathlib.Path]) -> Dict[str, Any]:
    # Reuse the workload builder's real mounted-model tokenizer and assignment
    # convention.  Its shortfall fallback is explicitly forbidden here.
    excluded = set()
    manifests = {}
    for name in EXCLUDED_WORKLOADS:
        manifest = load_manifest(name, manifest_root)
        ids = {row["source_id"] for row in manifest["requests"]}
        excluded.update(ids)
        manifests[name] = {
            "path": str(manifest_path(name, manifest_root)),
            "file_sha256": hashlib.sha256(manifest_path(name, manifest_root).read_bytes()).hexdigest(),
            "manifest_sha256": manifest["manifest_sha256"],
            "source_ids": sorted(ids),
        }
    tokenised = _tokenise_catalogue(protocol["model"], catalogue_name)
    eligible = [row for row in tokenised["rows"] if row["id"] not in excluded]
    demands = protocol["cross_configuration"]["history_cases"]
    assignments, stats = _assign_unique(eligible, [row["prompt_tokens"] for row in demands])
    if stats["shortfalls"] or len({row["source_id"] for row in assignments}) != len(demands):
        raise ValueError("held-out real ShareGPT conversations cannot satisfy exact protocol cases")
    cases = []
    for index, (bound, demand) in enumerate(zip(assignments, demands)):
        tokens = bound.pop("token_ids")
        if len(tokens) != demand["prompt_tokens"] or bound["source_id"] in excluded:
            raise ValueError("qualification prompt length/source exclusion violated")
        cases.append({
            "case_id": f"p{demand['prompt_tokens']}-h{demand['generated_history_tokens']}",
            **demand, **bound, "prompt_token_ids": tokens,
            "prompt_token_ids_sha256": _digest(tokens),
            "original_max_tokens": demand["generated_history_tokens"] + SUFFIX_TOKENS,
            "selection_order": index,
        })
    return {"catalogue": catalogue_name, "catalogue_path": tokenised["path"],
            "catalogue_sha256": tokenised["sha256"], "model": protocol["model"],
            "tokenizer": {"from_pretrained": protocol["model"], "add_special_tokens": True,
                          "truncation": "real leading tokens, never padding"},
            "excluded_workloads": manifests, "excluded_source_ids": sorted(excluded),
            "cases": cases}


def _compare_diagnostics(left: Dict[str, Any], right: Dict[str, Any],
                         criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Compare normalized model logprobs, not a renormalized top-k subset."""
    tables = []
    errors = []
    finite = True
    for label, diagnostic in (("left", left), ("right", right)):
        pairs = [(int(token), float(value)) for token, value in diagnostic["top_logprobs"].items()]
        table_finite = all(math.isfinite(value) for _, value in pairs)
        finite = finite and table_finite
        if len({token for token, _ in pairs}) != len(pairs):
            errors.append(f"{label}: duplicate normalized token identity")
        if len(pairs) < TOP_K:
            errors.append(f"{label}: fewer than {TOP_K} reported token probabilities")
        if any(value > 0 for _, value in pairs):
            errors.append(f"{label}: positive normalized log probability")
        # Never let a NaN disappear simply because it sorts outside shared top32.
        ordered = sorted(pairs, key=lambda pair: (-pair[1], pair[0])) if table_finite else pairs
        tables.append(dict(ordered[:TOP_K]))
    common = set(tables[0]) & set(tables[1])
    differences = [abs(tables[0][token] - tables[1][token]) for token in sorted(common)]
    maximum = max(differences) if differences and finite else None
    intersection = len(common) / TOP_K
    overlap_passed = intersection >= criteria["top32_intersection_fraction_min"]
    difference_passed = maximum is not None and maximum <= criteria[
        "shared_top32_max_absolute_log_probability_difference_max"]
    finite_passed = finite or not criteria["all_reported_values_finite"]
    return {"passed": not errors and overlap_passed and difference_passed and finite_passed,
            "argmax_equal": int(left["token_id"]) == int(right["token_id"]),
            "left_argmax": int(left["token_id"]), "right_argmax": int(right["token_id"]),
            "top32_intersection_count": len(common), "top32_intersection_fraction": intersection,
            "shared_top32_max_absolute_log_probability_difference": maximum,
            "all_reported_values_finite": finite,
            "intersection_passed": overlap_passed, "difference_passed": difference_passed,
            "finite_passed": finite_passed, "invalid_diagnostic": errors}


def _suffix_comparison(expected: Sequence[int], observed: Sequence[int]) -> Dict[str, Any]:
    divergences = [{"offset": offset,
                    "expected_token_id": expected[offset] if offset < len(expected) else None,
                    "observed_token_id": observed[offset] if offset < len(observed) else None}
                   for offset in range(max(len(expected), len(observed)))
                   if (offset >= len(expected) or offset >= len(observed)
                       or expected[offset] != observed[offset])]
    return {"passed": len(expected) == len(observed) == SUFFIX_TOKENS and not divergences,
            "expected_tokens": len(expected), "observed_tokens": len(observed),
            "divergence_count": len(divergences),
            "first_divergence_offset": divergences[0]["offset"] if divergences else None,
            "divergences": divergences}


def _snapshot_contract(snapshot: Dict[str, Any], case: Dict[str, Any],
                       reference: Sequence[int]) -> Dict[str, Any]:
    prompt, history = snapshot["prompt_token_ids"], snapshot["generated_token_ids"]
    context = len(prompt) + len(history)
    sampling = snapshot["sampling"]
    checks = {
        "original_prompt_boundary": prompt == case["prompt_token_ids"]
                                    and snapshot["prompt_len"] == len(prompt),
        "exact_generated_checkpoint": len(history) == case["generated_history_tokens"]
                                      and snapshot["generated_len"] == len(history),
        "real_reference_prefix_equal": history == list(reference[:len(history)]),
        "original_generation_budget": sampling["max_tokens"] == case["original_max_tokens"],
        "remaining_original_budget": snapshot["remaining_tokens"] == SUFFIX_TOKENS
                                     and sampling["max_tokens"] - len(history) == SUFFIX_TOKENS,
        "greedy_ignore_eos": sampling["temperature"] == 0 and sampling["ignore_eos"] is True,
        "final_generated_but_uncomputed_token": context - snapshot["num_computed_tokens"] == 1,
        "supported_native_snapshot": not snapshot["unsupported_continuation"],
    }
    return {"passed": all(checks.values()), "checks": checks,
            "prompt_tokens": len(prompt), "generated_tokens": len(history),
            "num_computed_tokens": snapshot["num_computed_tokens"],
            "uncomputed_tokens": context - snapshot["num_computed_tokens"],
            "history_sha256": _digest(prompt + history),
            "generated_token_ids_sha256": _digest(history),
            "snapshot_at_s": snapshot["snapshot_at_s"]}


def _cache_coverage(addressing: Dict[str, Any], layout: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups = []
    descriptions = layout[0]["groups"]
    if len(addressing["groups"]) != len(descriptions):
        raise ValueError("observed snapshot cache groups disagree with worker layout")
    for group, description in zip(addressing["groups"], descriptions):
        sizes = {row["block_size"] for row in description["layers"]}
        if len(sizes) != 1:
            raise ValueError("native group has mixed logical page sizes")
        size = sizes.pop()
        logical = group["logical_block_indices"]
        expected_pages = math.ceil(addressing["num_computed_tokens"] / size)
        omitted = sorted(set(range(expected_pages)) - set(logical))
        partial = [{"logical_block_index": index, "valid_token_range": interval}
                   for index, interval in zip(logical, group["valid_token_ranges"])
                   if interval[1] - interval[0] < size]
        windows = sorted({int(row["sliding_window"]) for row in description["layers"]
                          if row.get("sliding_window") is not None})
        groups.append({"group_id": group["group_id"], "block_size": size,
                       "spec_types": sorted({row["spec_type"] for row in description["layers"]}),
                       "sliding_windows": windows, "live_pages": len(logical),
                       "null_or_evicted_logical_pages": omitted, "partial_pages": partial})
    return {"native_group_count": len(groups), "grouped_cache_observed": len(groups) > 1,
            "partial_page_observed": any(row["partial_pages"] for row in groups),
            "sliding_window_present": any(row["sliding_windows"] for row in groups),
            "null_page_omission_observed": any(row["null_or_evicted_logical_pages"] for row in groups),
            "groups": groups}


class ContinuationQualificationSession:
    """Isolated diagnostic owner; reusable exact-history and native-transfer seams.

    For cost-under-background-load work, call start_checkpoint first, establish
    real background readiness, then transfer_checkpoint.  Only the latter's
    proposal clock includes reservation/quiescence/transport/commit/release.
    Never share a checkpoint between mechanisms: generate two real checkpoints
    and verify their exact histories.  close() reports unresolved identities; it
    never kills engines or resumes an uncertain source.
    """

    def __init__(self, protocol: Dict[str, Any], directory: pathlib.Path, timeout_s: float = 900.0) -> None:
        import ray

        self.ray = ray
        self.protocol = protocol
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        ledger = self.directory / "ownership.jsonl"
        if ledger.exists():
            raise FileExistsError("diagnostic ownership requires a new private ledger directory")
        self.state = GatewayState(str(ledger))
        self.actors: Dict[str, Any] = {}
        self.descriptors: Dict[str, Dict[str, Any]] = {}
        self.layouts: Dict[str, Any] = {}
        self.initial_idle: Dict[str, Any] = {}
        self.unresolved: list[Dict[str, Any]] = []
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.before: list[Dict[str, Any]] = []
        self.model_execution_started = False
        self.model_execution_attempted = False
        self.prefix = f"cq-{uuid.uuid4().hex}"
        self.source = protocol["same_configuration"]["source"].split()[-1]
        self.same = protocol["same_configuration"]["destination"].split()[-1]
        self.cross = protocol["cross_configuration"]["destination"].split()[-1]

    def event(self, event: str, **fields: Any) -> None:
        with (self.directory / "rpc_evidence.jsonl").open("a") as stream:
            stream.write(json.dumps(_json_safe({"event": event, "at_s": time.time(), **fields}),
                                    separators=(",", ":"), allow_nan=False) + "\n")

    def get(self, reference: Any) -> Any:
        # A timeout never cancels a possibly committed remote operation.
        return self.ray.get(reference, timeout=self.timeout_s)

    def call(self, instance: str, method: str, **arguments: Any) -> Any:
        identity = {key: arguments[key] for key in ("request_id", "transaction_id") if key in arguments}
        self.event("rpc_started", instance=instance, method=method, **identity)
        try:
            result = self.get(getattr(self.actors[instance], method).remote(**arguments))
        except Exception as exc:
            self.event("rpc_error", instance=instance, method=method, error=str(exc), **identity)
            raise
        self.event("rpc_acknowledged", instance=instance, method=method, **identity)
        return result

    def receipt(self, instance: str, txn: Any, method: str, key: str, **arguments: Any) -> Dict[str, Any]:
        try:
            result = self.call(instance, method, **arguments)
        except Exception:
            status = self.call(instance, "continuation_status", transaction_id=txn.transaction_id)
            if (status.get("generation") != self.descriptors[instance]["generation"]
                    or status.get("transaction_id") != txn.transaction_id):
                raise RuntimeError("reconciliation returned a different transaction generation")
            result = status.get(key)
            if not isinstance(result, dict):
                raise RuntimeError(f"{method}: native decision remains unresolved")
        if (result.get("instance_id") != instance
                or result.get("generation") != self.descriptors[instance]["generation"]
                or result.get("transaction_id") != txn.transaction_id):
            raise RuntimeError(f"{method}: native receipt identity mismatch")
        self.event("native_receipt", method=method, receipt=result)
        return result

    @staticmethod
    def source_arguments(txn: Any) -> Dict[str, Any]:
        return {"request_id": txn.request_id, "transaction_id": txn.transaction_id,
                "source_generation": txn.source_generation, "source_epoch": txn.source_epoch}

    def attach(self) -> None:
        """Attach only existing descriptors and inspect actual native KV layout."""
        from .registry import find_registry

        if not self.ray.is_initialized():
            raise RuntimeError("initialize Ray in the RCM-supervised entrypoint before qualification")
        self.registry = find_registry()
        if self.registry is None:
            raise RuntimeError("qualification requires already resident registered M engines")
        self.before = self.get(self.registry.snapshot.remote())
        _save(self.directory, "registry_before.json", self.before)
        by_name = {row["instance_id"]: row for row in self.before}
        roles = ((self.source, 8), (self.same, 8), (self.cross, 16))
        if len({name for name, _ in roles}) != 3:
            raise ValueError("protocol needs three distinct resident logical engines")
        for name, tp in roles:
            descriptor = by_name[name]
            if (descriptor["tp_size"] != tp or descriptor["layout_id"] != "M"
                    or descriptor["state"] != "ready" or descriptor["model"] != self.protocol["model"]):
                raise RuntimeError(f"unqualified descriptor for {name}: {descriptor}")
            self.descriptors[name] = descriptor
            self.actors[name] = self.ray.get_actor(descriptor["engine_actor_name"],
                                                  namespace=descriptor["ray_namespace"])
            for method in ("generate_qualification", "await_qualification_checkpoint", "next_token_diagnostic"):
                if not hasattr(self.actors[name], method):
                    raise RuntimeError(f"{name} needs the final diagnostic build redeployed: missing {method}")
            actual = self.call(name, "identity")
            if any(actual[key] != descriptor[key] for key in
                   ("instance_id", "layout_id", "generation", "tp_size")):
                raise RuntimeError(f"descriptor/actor identity mismatch for {name}")
            settings = actual["engine_settings"]
            if settings["enable_prefix_caching"] or settings["model"] != self.protocol["model"]:
                raise RuntimeError("qualification requires the protocol model without prefix caching")
            _save(self.directory, f"identity-{name}.json", actual)
            idle = self.call(name, "await_idle", timeout_s=self.timeout_s)
            if not idle["idle"] or idle["draining"]:
                raise RuntimeError(f"qualification requires an isolated accepting engine: {idle}")
            self.initial_idle[name] = idle
            # A projection warning is not evidence of actual cache sharding.
            # A rejected worker layout is recorded, never replaced by recompute.
            self.layouts[name] = self.call(name, "kv_layout")
            _save(self.directory, f"kv-layout-{name}.json.gz", self.layouts[name])

    def unresolved_request(self, request_id: str, error: Exception, txn: Any = None) -> None:
        row = {"request_id": request_id, "error": f"{type(error).__name__}: {error}",
               "owner": self.state.owner(request_id).to_dict() if self.state.owner(request_id) else None}
        if txn is not None:
            row.update(transaction_id=txn.transaction_id, source_instance=txn.source_instance,
                       source_generation=txn.source_generation, destination_instance=txn.destination_instance,
                       destination_generation=txn.destination_generation, state=txn.state.value)
            try:
                self.state.mark_unresolved(txn.transaction_id, row["error"])
            except Exception as storage_error:
                row["ledger_error"] = str(storage_error)
        self.unresolved.append(row)
        self.event("fail_closed_unresolved", **row)

    def cancel_owned(self, request_id: str, instance: str) -> Dict[str, Any]:
        owner = self.state.owner(request_id)
        self.state.cancel(request_id)
        receipt = self.call(instance, "cancel_request", request_id=request_id,
                            epoch=owner.epoch, generation=owner.instance_generation)
        if (receipt.get("request_id") != request_id or receipt.get("instance_id") != instance
                or receipt.get("generation") != owner.instance_generation or receipt.get("epoch") != owner.epoch):
            raise RuntimeError("cancellation receipt identity mismatch")
        self.state.record_cancellation(request_id, receipt)
        self.event("cancellation_acknowledged", receipt=receipt)
        return receipt

    def owned_generation(self, instance: str, method: str, label: str, **arguments: Any) -> Dict[str, Any]:
        rid = f"{self.prefix}-{label}"
        self.state.claim(rid, instance, self.descriptors[instance]["generation"])
        self.model_execution_attempted = True
        try:
            result = self.call(instance, method, request_id=rid, **arguments)
            if result.get("request_id") != rid or result.get("instance_id") != instance:
                raise RuntimeError("model result identity mismatch")
            self.model_execution_started = True
            self.state.finish(rid, result.get("output_tokens", 1))
            return result
        except Exception:
            try:
                self.cancel_owned(rid, instance)
            except Exception as cleanup_error:
                self.unresolved_request(rid, cleanup_error)
            raise

    def start_checkpoint(self, case: Dict[str, Any], label: str) -> Dict[str, Any]:
        """Generate a real held checkpoint without reducing its original budget."""
        rid = f"{self.prefix}-{label}"
        self.state.claim(rid, self.source, self.descriptors[self.source]["generation"])
        checkpoint = {"request_id": rid, "case": case, "source_started_at_s": time.time(),
                      "source_reference": None, "transaction_id": None, "resolved": False}
        self.checkpoints[rid] = checkpoint
        try:
            self.model_execution_attempted = True
            checkpoint["source_reference"] = self.actors[self.source].generate_qualification.remote(
                request_id=rid, prompt_token_ids=case["prompt_token_ids"],
                max_tokens=case["original_max_tokens"], checkpoint_tokens=case["generated_history_tokens"])
            checkpoint["native_checkpoint"] = self.call(self.source, "await_qualification_checkpoint", request_id=rid,
                checkpoint_tokens=case["generated_history_tokens"], timeout_s=self.timeout_s)
            self.model_execution_started = True
            checkpoint["checkpoint_ready_at_s"] = time.time()
            self.event("exact_checkpoint_ready", request_id=rid, checkpoint=checkpoint["native_checkpoint"])
            return checkpoint
        except Exception:
            self.abort_checkpoint(checkpoint)
            raise

    def abort_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel an unused held checkpoint; never bypass transaction cleanup."""
        if checkpoint["resolved"]:
            return {"acknowledged": True, "already_resolved": True}
        txn_id = checkpoint["transaction_id"]
        if txn_id is not None:
            return self.cleanup_trial(self.state.transaction(txn_id), checkpoint)
        try:
            receipt = self.cancel_owned(checkpoint["request_id"], self.source)
            checkpoint["resolved"] = True
            return {"acknowledged": True, "receipt": receipt}
        except Exception as exc:
            self.unresolved_request(checkpoint["request_id"], exc)
            return {"acknowledged": False, "error": str(exc)}

    def finish_commit(self, txn: Any) -> None:
        self.state.ensure_healthy()
        if txn.state not in COMMITTED_STATES:
            raise RuntimeError("activation requires authoritative durable ownership")
        if txn.state is TransactionState.RELEASED:
            return
        if not txn.activation_receipt:
            activation = self.receipt(txn.destination_instance, txn, "commit_continuation", "activation",
                                      transaction_id=txn.transaction_id, ownership=txn.ownership_certificate)
            self.state.mark_activated(txn.transaction_id, activation)
        release = self.receipt(txn.source_instance, txn, "release_source", "release",
                               request_id=txn.request_id, transaction_id=txn.transaction_id,
                               ownership=txn.ownership_certificate)
        self.state.release(txn.transaction_id, release)

    def abort_precommit(self, txn: Any, reason: str) -> None:
        self.state.ensure_healthy()
        if txn.state in COMMITTED_STATES:
            raise RuntimeError("committed ownership must never be rolled back")
        self.state.abort(txn.transaction_id, reason)
        abort = self.receipt(txn.destination_instance, txn, "abort_continuation", "abort",
                             transaction_id=txn.transaction_id)
        if not abort.get("aborted"):
            raise RuntimeError("destination cleanup was not acknowledged; source remains fenced")
        resume = self.receipt(txn.source_instance, txn, "unquiesce", "resume", **self.source_arguments(txn))
        self.state.finish_abort(txn.transaction_id, abort, resume)

    def cleanup_trial(self, txn: Any, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Reconcile the durable decision, never guess a lost ACK or resume two owners."""
        try:
            if txn.state in COMMITTED_STATES:
                self.finish_commit(txn)
                try:
                    result = self.call(txn.destination_instance, "await_continuation",
                                       transaction_id=txn.transaction_id, timeout_s=self.timeout_s)
                    self.state.finish(txn.request_id, len(txn.snapshot["generated_token_ids"])
                                      + result["new_output_tokens"])
                except Exception:
                    self.state.cancel(txn.request_id)
                    cancellation = self.receipt(txn.destination_instance, txn, "cancel_continuation",
                                                "cancellation", transaction_id=txn.transaction_id,
                                                ownership=txn.ownership_certificate)
                    self.state.record_cancellation(txn.request_id, cancellation)
            else:
                self.abort_precommit(txn, "qualification_error")
            result = self.get(checkpoint["source_reference"])
            if txn.state is TransactionState.ABORTED:
                self.state.finish(txn.request_id, result["output_tokens"])
            checkpoint["resolved"] = True
            return {"acknowledged": True, "transaction_state": txn.state.value, "source_result": result}
        except Exception as exc:
            self.unresolved_request(txn.request_id, exc, txn)
            return {"acknowledged": False, "error": str(exc)}

    def transfer_checkpoint(self, checkpoint: Dict[str, Any], *, destination: str, mechanism: str,
                            reference_token_ids: Sequence[int], reference_text: Optional[str] = None,
                            abort_prepared: bool = False, diagnostic: bool = True) -> Dict[str, Any]:
        """Reserve before coordinator quiescence; return actual native evidence.

        Source-history construction is outside timings.  Clock marks are local
        observed wall/monotonic/CPU values, not modeled transport costs.  The
        destination reports its own actual first-token timestamp.  No unrelated
        engine is drained, paused or stopped by this helper. Fidelity retains
        diagnostic=True; cost runs use False to exclude top-32 collection.
        """
        if type(diagnostic) is not bool:
            raise ValueError("diagnostic must be a boolean")
        case, rid = checkpoint["case"], checkpoint["request_id"]
        if checkpoint["resolved"] or checkpoint["transaction_id"] is not None:
            raise ValueError("a native checkpoint may be consumed exactly once")
        if self.unresolved:
            raise RuntimeError("session is fail-closed with unresolved native ownership")
        detail: Dict[str, Any] = {"request_id": rid, "case_id": case["case_id"],
            "source": self.source, "destination": destination, "mechanism": mechanism,
            "abort_prepared": abort_prepared, "diagnostic": diagnostic, "passed": False, "timings": {},
            "checkpoint": checkpoint["native_checkpoint"],
            "source_history_construction_included": False}
        def mark(name: str) -> None:
            detail["timings"][name] = {"wall_s": time.time(), "monotonic_s": time.monotonic(),
                                       "control_cpu_s": time.process_time()}
        txn = None
        mark("proposal")
        try:
            sampled_at = time.monotonic()
            capacity = self.call(destination, "kv_capacity")
            self.state.observe_capacity(destination, total_blocks=capacity["total_blocks"],
                free_blocks=capacity["free_blocks"], block_size=capacity["block_size"],
                running=capacity["num_running"], waiting=capacity["num_waiting"],
                resident_growth_allowance=0, sample_started_at=sampled_at)
            maximum = case["prompt_tokens"] + case["original_max_tokens"]
            blocks = math.ceil((case["prompt_tokens"] + case["generated_history_tokens"]) / capacity["block_size"])
            txn, decline = self.state.open_transaction(request_id=rid, source_instance=self.source,
                destination_instance=destination, mechanism=mechanism, blocks=blocks,
                growth_blocks=math.ceil(maximum / capacity["block_size"]) - blocks,
                destination_generation=self.descriptors[destination]["generation"],
                max_inflight_per_destination=1, max_context_tokens=maximum)
            if txn is None:
                raise RuntimeError(f"diagnostic transaction refused: {decline}")
            checkpoint["transaction_id"] = txn.transaction_id
            detail["transaction_id"] = txn.transaction_id
            identity = {**self.source_arguments(txn), "source_instance": self.source,
                        "destination_epoch": txn.destination_epoch,
                        "destination_generation": txn.destination_generation,
                        "max_context_tokens": maximum, "mechanism": mechanism}
            mark("reservation_start")
            reservation = self.receipt(destination, txn, "reserve_continuation", "reservation", payload=identity)
            self.state.mark_reserved(txn.transaction_id, reservation)
            detail["reservation"] = reservation
            mark("reservation_acknowledged")
            snapshot = self.call(self.source, "quiesce", **self.source_arguments(txn), timeout_s=self.timeout_s)
            mark("source_quiesced")
            detail["snapshot"] = snapshot
            for key, value in self.source_arguments(txn).items():
                if snapshot.get(key) != value:
                    raise RuntimeError(f"snapshot transaction identity mismatch: {key}")
            self.state.mark_quiesced(txn.transaction_id,
                                    len(snapshot["prompt_token_ids"]) + len(snapshot["generated_token_ids"]), snapshot)
            contract = _snapshot_contract(snapshot, case, reference_token_ids)
            detail["snapshot_contract"] = contract
            detail["cache_coverage"] = _cache_coverage(snapshot["kv_addressing"], self.layouts[self.source])
            if not all(value for key, value in contract["checks"].items() if key != "real_reference_prefix_equal"):
                raise RuntimeError("checkpoint violated the unchanged original continuation contract")
            payload = {**snapshot, **identity, "qualification_top32": diagnostic}
            mark("prepare_start")
            if mechanism == "kv_host":
                payload["kv_export"] = {"object_ref": self.actors[self.source].export_request_kv.remote(
                    **self.source_arguments(txn))}
            prepared = self.receipt(destination, txn, "prepare_continuation", "preparation", payload=payload)
            mark("prepared")
            detail["preparation"] = prepared
            if not prepared.get("accepted"):
                raise RuntimeError(f"native preparation refused: {prepared.get('reasons')}")
            transferred = prepared["transferred_bytes"]
            if mechanism == "kv_host" and (transferred <= 0 or prepared["num_computed_tokens"] != snapshot["num_computed_tokens"]):
                raise RuntimeError("host continuation failed to retain actual imported computed KV")
            if mechanism == "kv_host":
                native_evidence = prepared["evidence"]
                imports = native_evidence.get("import", [])
                source_addressing = native_evidence.get("source_addressing", {})
                destination_addressing = native_evidence.get("destination_addressing", {})
                source_groups = source_addressing.get("groups", [])
                destination_groups = destination_addressing.get("groups", [])
                import_contract = {
                    "every_worker_import_acknowledged": sorted(row["process_index"] for row in imports)
                        == list(range(self.layouts[destination][0]["process_count"]))
                        and all(row["arrays_written"] > 0 and row["bytes"] > 0 for row in imports),
                    "source_addressing_matches_fenced_snapshot": source_addressing == snapshot["kv_addressing"],
                    "destination_computed_frontier_preserved": destination_addressing.get("num_computed_tokens")
                        == snapshot["num_computed_tokens"],
                    "logical_page_coverage_preserved": len(source_groups) == len(destination_groups)
                        and all(source_group[key] == destination_group[key]
                                for source_group, destination_group in zip(source_groups, destination_groups)
                                for key in ("group_id", "logical_block_indices", "valid_token_ranges")),
                    "no_recompute_fallback": prepared["rebuild_tokens"] == 0,
                }
                detail["native_import_contract"] = import_contract
                if not all(import_contract.values()):
                    raise RuntimeError("actual native import acknowledgement/coverage contract failed")
            if mechanism == "recompute" and (transferred != 0 or prepared["num_computed_tokens"] != 0):
                raise RuntimeError("recompute continuation did not start from native recomputation")
            self.state.mark_prepared(txn.transaction_id, replay_tokens=prepared["rebuild_tokens"],
                                     transferred_bytes=transferred, receipt=prepared)
            if abort_prepared:
                self.abort_precommit(txn, "qualification_prepared_abort")
                mark("abort_acknowledged")
                source_result = self.get(checkpoint["source_reference"])
                detail["source_result"] = source_result
                self.state.finish(rid, source_result["output_tokens"])
                detail["passed"] = (source_result["output_token_ids"] == list(reference_token_ids)
                                    and source_result["prompt_tokens"] == case["prompt_tokens"]
                                    and source_result["finish_reason"] == "length" and contract["passed"])
            else:
                if self.state.commit(txn.transaction_id, txn.destination_generation) is None:
                    raise RuntimeError("authoritative ownership commit rejected")
                mark("ownership_committed")
                self.finish_commit(txn)
                mark("source_release_acknowledged")
                source_result = self.get(checkpoint["source_reference"])
                result = self.call(destination, "await_continuation", transaction_id=txn.transaction_id,
                                   timeout_s=self.timeout_s)
                mark("continuation_completed")
                detail.update(source_result=source_result, result=result)
                suffix = result["new_output_token_ids"]
                if (result.get("request_id") != rid or result.get("transaction_id") != txn.transaction_id
                        or result.get("generation") != txn.destination_generation
                        or result.get("instance_id") != destination):
                    raise RuntimeError("continuation result identity mismatch")
                detail["output_contract"] = {
                    "source_prefix_delivered_once": source_result["output_token_ids"] == snapshot["generated_token_ids"]
                                                    and source_result["finish_reason"] == "migrated",
                    "original_prompt_boundary": source_result["prompt_tokens"] == case["prompt_tokens"],
                    "remaining_original_budget": len(suffix) == result["new_output_tokens"] == SUFFIX_TOKENS
                                                  and result["finish_reason"] == "length",
                    "committed_text_prefix_preserved": source_result["text"] == snapshot["generated_text"]
                        and result["full_output_text"] == snapshot["generated_text"] + result["text"],
                }
                if diagnostic:
                    detail["output_contract"]["first_step_diagnostic_matches_suffix"] = (
                        bool(suffix) and result["first_token_id"] == suffix[0]
                        and result["logprobs_origin"] == "prepared_native_continuation")
                    detail["first_step_diagnostic"] = {"token_id": result["first_token_id"],
                                                         "top_logprobs": result["first_token_logprobs"]}
                else:
                    detail["output_contract"]["ordinary_path_without_top32"] = not any(
                        key in result for key in ("first_token_id", "first_token_logprobs", "logprobs_origin"))
                if destination == self.same and reference_text is not None:
                    detail["output_contract"]["same_tp_complete_text"] = result["full_output_text"] == reference_text
                detail["suffix_vs_uninterrupted_tp8"] = _suffix_comparison(
                    reference_token_ids[case["generated_history_tokens"]:], suffix)
                self.state.finish(rid, len(snapshot["generated_token_ids"]) + len(suffix))
                detail["passed"] = contract["passed"] and all(detail["output_contract"].values())
                if destination == self.same:
                    detail["passed"] = detail["passed"] and detail["suffix_vs_uninterrupted_tp8"]["passed"]
            checkpoint["resolved"] = True
        except Exception as exc:
            detail["error"] = f"{type(exc).__name__}: {exc}"
            detail["cleanup"] = self.cleanup_trial(txn, checkpoint) if txn else self.abort_checkpoint(checkpoint)
        finally:
            detail["transaction"] = txn.to_dict() if txn is not None else None
            detail["artifact"] = _save(self.directory, f"{rid}.json.gz", detail)
        return detail

    def close(self) -> Dict[str, Any]:
        """Save ownership and observed resident state; never stop an engine."""
        for checkpoint in self.checkpoints.values():
            if not checkpoint["resolved"] and not any(row["request_id"] == checkpoint["request_id"] for row in self.unresolved):
                self.abort_checkpoint(checkpoint)
        evidence: Dict[str, Any] = {"unresolved": self.unresolved, "idle": {}, "native_state": {},
                                    "resident_instances_preserved": False, "all_native_pages_reclaimed": False}
        try:
            for name in self.actors:
                # An unresolved fence is deliberately left held.  Do not spend
                # the result timeout waiting for work we must not resume.
                if not self.unresolved:
                    evidence["idle"][name] = self.call(name, "await_idle", timeout_s=self.timeout_s)
                evidence["native_state"][name] = self.call(name, "migration_state")
            evidence["all_native_pages_reclaimed"] = len(self.initial_idle) == 3 and all(
                name in evidence["idle"] and evidence["idle"][name]["free_blocks"] == before["free_blocks"]
                for name, before in self.initial_idle.items())
            after = self.get(self.registry.snapshot.remote())
            evidence["registry_after"] = _save(self.directory, "registry_after.json", after)
            by_name = {row["instance_id"]: row for row in after}
            evidence["resident_instances_preserved"] = bool(self.before) and all(
                row["instance_id"] in by_name and by_name[row["instance_id"]]["generation"] == row["generation"]
                and (row["state"] != "ready" or by_name[row["instance_id"]]["state"] == "ready")
                for row in self.before)
        except Exception as exc:
            evidence["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            evidence["ownership_artifact"] = _save(self.directory, "ownership_final.json.gz", self.state.snapshot())
            self.state.close()
        evidence["passed"] = (not self.unresolved and "error" not in evidence
                              and evidence["resident_instances_preserved"]
                              and evidence["all_native_pages_reclaimed"]
                              and all(not row["quiesced"] and not row["prepared"] and not row["active_continuations"]
                                      for row in evidence["native_state"].values())
                              and len(evidence["idle"]) == 3
                              and all(row["idle"] and not row["draining"] for row in evidence["idle"].values()))
        return evidence


def _trial_summary(trial: Dict[str, Any]) -> Dict[str, Any]:
    summary = {key: trial[key] for key in ("request_id", "transaction_id", "destination", "mechanism",
               "abort_prepared", "passed", "error", "artifact", "snapshot_contract", "output_contract") if key in trial}
    if "suffix_vs_uninterrupted_tp8" in trial:
        summary["suffix_vs_uninterrupted_tp8"] = {
            key: value for key, value in trial["suffix_vs_uninterrupted_tp8"].items() if key != "divergences"}
    if "preparation" in trial:
        summary["transport"] = {key: trial["preparation"][key] for key in
                                ("num_computed_tokens", "rebuild_tokens", "transferred_bytes", "native_reserved_blocks")}
    return summary


def _matched_pair(recompute: Dict[str, Any], host: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
    snapshots = [trial.get("snapshot") for trial in (recompute, host)]
    matched = all(snapshots) and all(snapshots[0][key] == snapshots[1][key] for key in
        ("prompt_token_ids", "generated_token_ids", "sampling", "num_computed_tokens", "remaining_tokens"))
    evidence: Dict[str, Any] = {"identical_histories_and_contract": bool(matched), "passed": False,
                              "transport_attribution_resolved": False}
    if not matched or not all("result" in trial and "first_step_diagnostic" in trial for trial in (recompute, host)):
        evidence["unresolved"] = "missing completed native path or nonidentical real source histories"
        return evidence
    evidence["prepared_first_step_logprobs"] = _compare_diagnostics(
        recompute["first_step_diagnostic"], host["first_step_diagnostic"], criteria)
    evidence["host_suffix_vs_recompute"] = _suffix_comparison(
        recompute["result"]["new_output_token_ids"], host["result"]["new_output_token_ids"])
    evidence["transport_attribution_resolved"] = evidence["host_suffix_vs_recompute"]["passed"]
    if not evidence["transport_attribution_resolved"]:
        evidence["unresolved"] = (
            "actual imported-KV continuation diverges from same-history destination recompute; "
            "not attributed to inherent TP drift")
    evidence["passed"] = (recompute["passed"] and host["passed"]
                          and evidence["prepared_first_step_logprobs"]["passed"]
                          and evidence["transport_attribution_resolved"])
    return evidence


def run_continuation_qualification(
    *, protocol_path: str = PROTOCOL_PATH, manifest_root: Optional[str] = None,
    artifact_root: Optional[str] = None, catalogue_name: str = "large", timeout_s: float = 900.0,
) -> Dict[str, Any]:
    """Qualify already resident real TP8/TP8/TP16 engines; never launch hardware.

    Call synchronously from the Ray-connected RCM-supervised CLI.  Always inspect
    passed and cleanup.unresolved; a report is not a qualification certificate
    merely because this function returned normally.  Saved direct-engine data
    cannot replace Main's Envoy ownership qualification or headline experiments.
    """
    root = pathlib.Path(artifact_root or os.environ.get("OMP_HETERO_ARTIFACTS", "/tmp/omp_hetero/artifacts"))
    directory = root / f"continuation-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    report: Dict[str, Any] = {
        "qualification": "real-native-continuation", "artifact_dir": str(directory), "passed": False,
        "evidence_scope": "direct-engine numerical/semantic qualification only; not HTTP ownership or headline results",
        "hardware_model_execution_started": False, "hardware_model_execution_attempted": False,
        "cases": [], "errors": [],
        "passed_flags": {}, "protocol_path": protocol_path,
        "fresh_diagnostic_scope": "full-history recomputation for TP drift, never evidence of KV restoration",
        "prepared_diagnostic_scope": "actual imported/recomputed native continuation first step, not ordinary migrated logprob support",
    }
    session = None
    trials = []
    pair_results = []
    fresh_results = []
    prepared_cross_results = []
    aborts = []
    selection = None
    try:
        protocol_file = pathlib.Path(protocol_path)
        if not protocol_file.is_absolute():
            protocol_file = pathlib.Path(__file__).resolve().parents[1] / protocol_file
        raw = protocol_file.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        report["protocol_sha256"] = digest
        if digest != PROTOCOL_SHA256:
            raise ValueError("prospective continuation protocol bytes changed; refusing new criteria")
        protocol_copy = directory / "prospective_protocol.json"
        protocol_copy.write_bytes(raw)
        report["protocol_artifact"] = {"path": str(protocol_copy), "sha256": digest, "bytes": len(raw)}
        protocol = json.loads(raw)
        criteria = protocol["cross_configuration"]["criteria"]
        report["criteria"] = criteria
        selection = _select_cases(protocol, catalogue_name, pathlib.Path(manifest_root) if manifest_root else None)
        report["prompt_selection_artifact"] = _save(directory, "selected_real_prompts.json.gz", selection)
        session = ContinuationQualificationSession(protocol, directory, timeout_s)
        session.attach()
        for case in selection["cases"]:
            row: Dict[str, Any] = {key: case[key] for key in ("case_id", "source_id", "source_index",
                "prompt_tokens", "generated_history_tokens", "original_max_tokens", "prompt_token_ids_sha256")}
            row["trials"] = []
            report["cases"].append(row)
            reference = session.owned_generation(session.source, "generate", f"{case['case_id']}-reference",
                prompt_token_ids=case["prompt_token_ids"], max_tokens=case["original_max_tokens"],
                temperature=0.0, ignore_eos=True)
            row["reference_artifact"] = _save(directory, f"{case['case_id']}-reference.json.gz", reference)
            tokens = reference["output_token_ids"]
            if (len(tokens) != case["original_max_tokens"] or reference["output_tokens"] != len(tokens)
                    or reference["prompt_tokens"] != case["prompt_tokens"] or reference["finish_reason"] != "length"):
                raise RuntimeError("real uninterrupted reference violated original prompt/output contract")
            history = case["prompt_token_ids"] + tokens[:case["generated_history_tokens"]]
            fresh = {}
            for instance in (session.source, session.cross):
                fresh[instance] = session.owned_generation(instance, "next_token_diagnostic",
                    f"{case['case_id']}-fresh-{instance}", token_ids=history)
            fresh_comparison = _compare_diagnostics(fresh[session.source], fresh[session.cross], criteria)
            fresh_results.append(fresh_comparison)
            row["fresh_tp8_vs_tp16"] = fresh_comparison
            row["fresh_diagnostics_artifact"] = _save(directory, f"{case['case_id']}-fresh-top32.json.gz",
                {"history_token_ids": history, "history_sha256": _digest(history), "diagnostics": fresh})
            for destination in (session.same, session.cross):
                paired = {}
                for mechanism in protocol["mechanisms"]:
                    checkpoint = session.start_checkpoint(case, f"{case['case_id']}-{destination}-{mechanism}")
                    trial = session.transfer_checkpoint(checkpoint, destination=destination,
                        mechanism=mechanism, reference_token_ids=tokens, reference_text=reference["text"])
                    trials.append(trial)
                    paired[mechanism] = trial
                    row["trials"].append(_trial_summary(trial))
                    if session.unresolved:
                        raise RuntimeError("unresolved native identity; no further model requests will be started")
                    if destination == session.cross and "first_step_diagnostic" in trial:
                        common_history = trial["snapshot"]["prompt_token_ids"] + trial["snapshot"]["generated_token_ids"] == history
                        comparison = _compare_diagnostics(fresh[session.source], trial["first_step_diagnostic"], criteria)
                        comparison["identical_history"] = common_history
                        comparison["passed"] = comparison["passed"] and common_history
                        prepared_cross_results.append(comparison)
                        row.setdefault("tp8_vs_prepared_tp16", {})[mechanism] = comparison
                pair = _matched_pair(paired["recompute"], paired["kv_host"], criteria)
                pair_results.append(pair)
                pair_artifact = _save(directory, f"{case['case_id']}-{destination}-matched-pair.json.gz", pair)
                concise_pair = {key: value for key, value in pair.items() if key != "host_suffix_vs_recompute"}
                if "host_suffix_vs_recompute" in pair:
                    concise_pair["host_suffix_vs_recompute"] = {key: value for key, value in pair["host_suffix_vs_recompute"].items()
                                                                 if key != "divergences"}
                row.setdefault("matched_pairs", {})[destination] = {**concise_pair, "artifact": pair_artifact}
            # Exercise actual prepared-page cleanup too, including imported KV,
            # and require resumed source completion to equal its real reference.
            if case is selection["cases"][0]:
                for mechanism in protocol["mechanisms"]:
                    checkpoint = session.start_checkpoint(case, f"{case['case_id']}-{mechanism}-prepared-abort")
                    trial = session.transfer_checkpoint(checkpoint, destination=session.same,
                        mechanism=mechanism, reference_token_ids=tokens, reference_text=reference["text"], abort_prepared=True)
                    aborts.append(trial)
                    row.setdefault("prepared_abort", []).append(_trial_summary(trial))
                    if session.unresolved:
                        raise RuntimeError("prepared abort cleanup unresolved; no further model execution")
        # Manifest content/source exclusions must remain the ones selected before
        # execution; their normal prompt-budget hash does not cover source IDs.
        report["exclusion_manifests_unchanged"] = all(
            hashlib.sha256(pathlib.Path(row["path"]).read_bytes()).hexdigest() == row["file_sha256"]
            for row in selection["excluded_workloads"].values())
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if session is not None:
            report["hardware_model_execution_started"] = session.model_execution_started
            report["hardware_model_execution_attempted"] = session.model_execution_attempted
            report["cleanup"] = session.close()
        else:
            report["cleanup"] = {"passed": False, "unresolved": [], "not_started": True}
    expected_cases = len(selection["cases"]) if selection else 3
    same_trials = [trial for trial in trials if session and trial["destination"] == session.same]
    coverage = [trial["cache_coverage"] for trial in trials
                if trial.get("passed") and trial["mechanism"] == "kv_host" and "cache_coverage" in trial]
    observed_layouts = list(session.layouts.values()) if session else []
    layouts_observed = len(observed_layouts) == 3 and all(observed_layouts)
    sliding = any(layer.get("sliding_window") is not None for layout in observed_layouts
                  for group in layout[0]["groups"] for layer in group["layers"])
    grouped = any(len(layout[0]["groups"]) > 1 for layout in observed_layouts)
    report["boundary_coverage"] = {
        "actual_imported_grouped_cache": any(row["grouped_cache_observed"] for row in coverage),
        "grouped_cache_requirement": ("present" if grouped else "not_present_in_observed_native_layout")
                                      if layouts_observed else "unobserved",
        "actual_imported_partial_page": any(row["partial_page_observed"] for row in coverage),
        "sliding_window_present": sliding,
        "actual_null_page_omission": any(row["null_page_omission_observed"] for row in coverage),
        "sliding_window_requirement": ("covered" if any(row["null_page_omission_observed"] for row in coverage)
                                       else ("unresolved" if sliding else "not_present_in_observed_layout"))
                                      if layouts_observed else "unobserved",
    }
    report["passed_flags"] = {
        "prospective_protocol_unchanged": report.get("protocol_sha256") == PROTOCOL_SHA256,
        "held_out_source_manifests_unchanged": report.get("exclusion_manifests_unchanged", False),
        "all_required_native_paths_completed": len(trials) == expected_cases * 4 and all(row["passed"] for row in trials),
        "same_tp8_32_token_suffix_exact": len(same_trials) == expected_cases * 2 and all(row["passed"] for row in same_trials),
        "cross_tp_fresh_top32": len(fresh_results) == expected_cases and all(row["passed"] for row in fresh_results),
        "cross_tp_actual_prepared_top32": len(prepared_cross_results) == expected_cases * 2
                                          and all(row["passed"] for row in prepared_cross_results),
        "host_vs_recompute_matched_history": len(pair_results) == expected_cases * 2
                                            and all(row["passed"] for row in pair_results),
        "prepared_abort_and_original_source_resume": len(aborts) == 2 and all(row["passed"] for row in aborts),
        "partial_page_covered_by_actual_import": report["boundary_coverage"]["actual_imported_partial_page"],
        "sliding_null_pages_covered_where_present": layouts_observed and (
            not sliding or report["boundary_coverage"]["actual_null_page_omission"]),
        "grouped_cache_covered_where_present": layouts_observed and (
            not grouped or report["boundary_coverage"]["actual_imported_grouped_cache"]),
        "acknowledged_cleanup_and_resident_preservation": report["cleanup"]["passed"],
    }
    report["passed"] = not report["errors"] and all(report["passed_flags"].values())
    report["checkpoint_seams"] = {
        "start_checkpoint": "Real source generate_qualification -> await_qualification_checkpoint; original max_tokens unchanged; excluded from transfer clock.",
        "background_boundary": "After start_checkpoint, caller may establish fixed real background readiness and per-iteration captures; no experiment is run here.",
        "transfer_checkpoint": "Proposal -> native reserve ACK -> coordinator quiesce snapshot -> actual export/prepare -> GatewayState WAL commit -> activate -> release -> result.",
        "matching": "Compare snapshot original prompt, full generated IDs, sampling, computed frontier and remaining budget byte-for-byte before paired attribution.",
        "observables": "timings holds local wall/monotonic/CPU marks; transaction holds all native receipts; result.first_token_at_s is actual destination delivery; preparation retains real export/import metadata.",
        "failure": "Call abort_checkpoint on unused checkpoints; never resume before destination abort ACK; close returns exact unresolved ownership identities without restarting or killing resident engines.",
    }
    report["report_artifact"] = _save(directory, "report.json", report)
    return report

