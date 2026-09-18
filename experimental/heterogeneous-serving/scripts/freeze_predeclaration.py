#!/usr/bin/env python3
"""Freeze a new prospective amendment locally; never rewrite an earlier declaration."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hetero.manifest import load_manifest
from hetero.predeclare import PREDECLARATION_PATH, build_predeclaration

AMENDMENT = "results/hetero/evaluation_amendment_20260916.json"
WORKLOADS = ("W0", "W1", "W2", "E_SHORT", "E_LONG")


SOURCE_FROZEN_AMENDMENT = "results/hetero/evaluation_fp32_state_20260918_source_frozen.json"
EVIDENCE_KEYS = ("gate_H", "gate_S", "gate_M", "lifecycle", "continuation", "recompute", "kv_host")


def _source_freeze() -> dict:
    """Hash the exact deployed source tree plus deployment manifests."""
    import subprocess

    files = sorted([*(ROOT / "hetero").rglob("*.py"), *(ROOT / "hetero").rglob("*.json"),
                    ROOT / "scripts/hetero_ctl.py", ROOT / "deploy/k8s/omp-gateway.yaml",
                    ROOT / "deploy/k8s/omp-router.yaml"])
    digests = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in files if path.exists()}
    commit = subprocess.run(["jj", "log", "-r", "@-", "--no-graph", "-T", "commit_id"],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["jj", "diff", "--stat"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout
    dirty = sorted(line.split("|")[0].strip() for line in status.splitlines()
                   if "|" in line and line.split("|")[0].strip() in digests)
    if dirty:
        raise ValueError(f"refusing to freeze uncommitted deployed source: {dirty}")
    return {
        "status": "pending",
        "commit_id": commit,
        "source_sha256": digests,
        "tree_sha256": hashlib.sha256(
            json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "scope": "Exact deployed source for every subsequent timing run. Any change to these files "
                 "invalidates this freeze and requires a new amendment and requalification.",
    }

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _all_passed(checks: dict, context: str) -> None:
    _require(bool(checks) and all(value is True for value in checks.values()), context)


def _capture(stats: dict) -> None:
    _require(stats["one_to_one"] is True and stats["rows"] > 0
             and stats["unmatched_retirements"] == stats["rows_without_retirement"] == 0,
             "capture has missing or unmatched retirement")
    workers = stats["per_instance"]
    _require(bool(workers), "missing per-instance capture")
    for row in workers.values():
        audit = row["capture_audit"]
        _require(audit["rows"] == audit["retired_rows"]
                 and audit["open_rows"] == audit["unmatched_retirements"] == audit["inflight_batches"] == 0,
                 "native iterations did not retire one-to-one")
    _require(sum(row["capture_audit"]["rows"] for row in workers.values()) == stats["rows"],
             "capture totals differ")


def _gate(gate: dict, layout: str, settings: dict) -> None:
    from hetero.bringup_gate import _evaluate_gate
    from hetero.native_backend import BACKEND_REVISION, RUNTIME_REVISION
    native_hashes = {row["name"]: row["patched_sha256"] for row in
                     json.loads((ROOT / "hetero/native_backend_patch.json").read_text())["files"]}

    _require(gate["layout_id"] == layout and gate["gate_passed"]["passed"] is True,
             f"gate {layout} failed or wrong layout")
    _all_passed(gate["gate_passed"]["checks"], f"gate {layout} has failed checks")
    _require(_evaluate_gate(gate)["passed"], f"gate {layout} raw checks failed")
    deployment = gate["deployment"]
    _require(deployment["desired_configuration"]["engine_settings"] == settings,
             f"gate {layout} desired EngineSettings differ")
    instances = deployment["instances"]
    expected_tp = {"H": [16, 16], "S": [8, 8, 8, 8], "M": [8, 8, 16]}[layout]
    _require(sorted(row["descriptor"]["tp_size"] for row in instances) == expected_tp,
             f"gate {layout} does not cover the full 32-chip layout")
    for instance in instances:
        initialization = instance["initialization"]
        _require(initialization["engine_settings"] == settings, "effective EngineSettings differ")
        workers = initialization["worker_evidence"]
        _require(len(workers) == instance["descriptor"]["tp_size"] // 4, "missing native worker evidence")
        for worker in workers:
            backend = worker["native_backend"]
            precision = worker["precision"]
            for receipt in (backend, precision):
                _require(receipt["backend_revision"] == BACKEND_REVISION
                         and receipt["runtime_revision"] == RUNTIME_REVISION,
                         "worker backend/runtime revision differs")
            _require(backend["ready"] is True and len(backend["files"]) == len(native_hashes)
                     and all(row["state"] == "patched" and row["actual_sha256"] == row["patched_sha256"]
                             for row in backend["files"])
                     and {row["name"]: row["actual_sha256"] for row in backend["files"]} == native_hashes,
                     "worker native backend is not fully patched")
            expected = {"model_dtype": settings["dtype"], "activation_dtype": settings["activation_dtype"],
                        "configured_cache_dtype": settings["kv_cache_dtype"],
                        "runner_cache_dtype": settings["kv_cache_dtype"],
                        "matmul_precision": settings["matmul_precision"]}
            _require(precision["verified_before_cache_allocation"] is True
                     and all(precision[key] == value for key, value in expected.items()),
                     "effective native precision differs")
            for key, dtype in (("parameter_dtypes", settings["dtype"]),
                               ("decoder_norm_dtypes", settings["activation_dtype"]),
                               ("attention_cache_dtypes", settings["kv_cache_dtype"])):
                _require(set(precision[key]) == {dtype} and precision[key][dtype] > 0,
                         f"native {key} differs")
        caches = gate["worker_kv_evidence"][instance["descriptor"]["instance_id"]]
        _require(len(caches) == len(workers), "missing allocated native KV evidence")
        for cache in caches:
            _require(cache["backend_revision"] == BACKEND_REVISION
                     and cache["configured_cache_dtype"] == settings["kv_cache_dtype"]
                     and cache["runner_cache_dtype"].removeprefix("torch.") == settings["kv_cache_dtype"]
                     and cache["cache_dtypes"] == [settings["kv_cache_dtype"]]
                     and cache["num_cache_arrays"] > 0 and cache["total_cache_bytes"] > 0,
                     "allocated native KV precision differs")


def _continuation(report: dict, numerical: dict, protocol_sha256: str) -> None:
    _require(report["passed"] is True and not report["errors"]
             and report["hardware_model_execution_started"] is True
             and report["protocol_sha256"] == protocol_sha256
             and report["criteria"] == numerical["cross_configuration"]["criteria"],
             "continuation failed or frozen numerical protocol changed")
    required_flags = {
        "prospective_protocol_unchanged", "held_out_source_manifests_unchanged",
        "all_required_native_paths_completed", "same_tp8_32_token_suffix_exact",
        "cross_tp_fresh_top32", "cross_tp_actual_prepared_top32", "host_vs_recompute_matched_history",
        "prepared_abort_and_original_source_resume", "partial_page_covered_by_actual_import",
        "sliding_null_pages_covered_where_present", "grouped_cache_covered_where_present",
        "acknowledged_cleanup_and_resident_preservation",
    }
    _require(required_flags <= report["passed_flags"].keys(), "incomplete continuation flags")
    _all_passed(report["passed_flags"], "continuation criterion failed")
    cases = report["cases"]
    expected = numerical["cross_configuration"]["history_cases"]
    _require([(row["prompt_tokens"], row["generated_history_tokens"]) for row in cases]
             == [(row["prompt_tokens"], row["generated_history_tokens"]) for row in expected],
             "continuation does not cover every frozen history case")
    criteria = report["criteria"]
    def diagnostic(row: dict) -> None:
        overlap = row["top32_intersection_fraction"]
        difference = row["shared_top32_max_absolute_log_probability_difference"]
        _require(row["passed"] is True and row["all_reported_values_finite"] is True
                 and not row["invalid_diagnostic"] and math.isfinite(overlap) and math.isfinite(difference)
                 and criteria["top32_intersection_fraction_min"] <= overlap <= 1
                 and 0 <= difference <= criteria["shared_top32_max_absolute_log_probability_difference_max"],
                 "continuation numerical observation violates frozen criterion")
    for case in cases:
        diagnostic(case["fresh_tp8_vs_tp16"])
        _require(set(case["tp8_vs_prepared_tp16"]) == {"recompute", "kv_host"}, "missing prepared diagnostic")
        for row in case["tp8_vs_prepared_tp16"].values():
            diagnostic(row)
            _require(row["identical_history"] is True, "prepared history differs")
        trials = case["trials"]
        _require(len(trials) == 4 and {(t["destination"], t["mechanism"]) for t in trials}
                 == set(itertools.product(("small_b", "large"), ("recompute", "kv_host"))),
                 "missing native continuation path")
        for trial in trials:
            _require(trial["passed"] is True and not trial.get("error")
                     and trial["snapshot_contract"]["passed"] is True, "native continuation trial failed")
            _all_passed(trial["snapshot_contract"]["checks"], "snapshot contract failed")
            _all_passed(trial["output_contract"], "original output contract failed")
            if trial["destination"] == "small_b":
                suffix = trial["suffix_vs_uninterrupted_tp8"]
                _require(suffix["passed"] is True and suffix["expected_tokens"] == suffix["observed_tokens"] == 32
                         and suffix["divergence_count"] == 0, "same-TP suffix differs")
        _require(set(case["matched_pairs"]) == {"small_b", "large"}, "missing transport attribution")
        for pair in case["matched_pairs"].values():
            _require(pair["passed"] is True and pair["identical_histories_and_contract"] is True
                     and pair["transport_attribution_resolved"] is True, "transport attribution failed")
            diagnostic(pair["prepared_first_step_logprobs"])
            suffix = pair["host_suffix_vs_recompute"]
            _require(suffix["passed"] is True and suffix["expected_tokens"] == suffix["observed_tokens"] == 32
                     and suffix["divergence_count"] == 0, "host continuation differs from matched recompute")
    aborts = [row for case in cases for row in case.get("prepared_abort", [])]
    _require(len(aborts) == 2 and {row["mechanism"] for row in aborts} == {"recompute", "kv_host"}
             and all(row["passed"] is True and row["abort_prepared"] is True for row in aborts),
             "prepared-abort source resumption is not qualified")
    cleanup = report["cleanup"]
    _require(cleanup["passed"] is True and not cleanup["unresolved"] and not cleanup.get("error")
             and cleanup["resident_instances_preserved"] is True and cleanup["all_native_pages_reclaimed"] is True
             and set(cleanup["idle"]) == set(cleanup["native_state"]) == {"small_a", "small_b", "large"}
             and all(row["idle"] is True and not row["draining"] for row in cleanup["idle"].values())
             and all(not row["quiesced"] and not row["prepared"] and not row["active_continuations"]
                     for row in cleanup["native_state"].values()), "continuation cleanup unresolved")
    _capture(report["iteration_stats"])


def _migration(measurements: dict, mechanism: str) -> None:
    from hetero.qualify_migration import _check_accounting, _check_gateway_authority, _check_handoff_ordering

    hardware = measurements["hardware"]
    _require(measurements["passed"] is True and measurements["protocol_probes"]["passed"] is True
             and hardware["passed"] is True and hardware["qualification_mode"] == "migration"
             and hardware["policy"]["migration_enabled"] is True
             and hardware["policy"]["migration_mechanism"] == mechanism
             and hardware["ingress_protocols"] == ["envoy-ext-proc-v3"] and bool(hardware["ingress_url"])
             and hardware["client_delivery_matches"] is True, f"{mechanism}: missing actual ExtProc qualification")
    transactions = hardware["transactions"]
    committed = [row for row in transactions if row["state"] in ("committed", "released")]
    records = hardware["records"]
    _require(len(committed) == hardware["migrations_committed"] > 0
             and all(row["mechanism"] == mechanism and row["state"] == "released" for row in committed)
             and len(records) == hardware["requests"] > 0
             and hardware["records_with_migration"] == sum(len(row["segments"]) > 1 for row in records) > 0,
             f"{mechanism}: no completed actual migrations")
    _require(all(row["transferred_bytes"] > 0 if mechanism == "kv_host" else row["replay_tokens"] > 0
                 for row in committed), f"{mechanism}: missing native transfer/replay work")
    accounting = _check_accounting(records, hardware["max_tokens"])
    _require(accounting == hardware["accounting"] and accounting["all_budgets_exact"]
             and accounting["no_duplicate_delivery"], f"{mechanism}: native output accounting failed")
    ordering = _check_handoff_ordering(committed)
    authority = _check_gateway_authority(records, transactions)
    _require(ordering == hardware["handoff_ordering"] and ordering["all_ordered"]
             and authority == hardware["gateway_authority"] and authority["all_continuations_authorized"]
             and authority["count"] > 0, f"{mechanism}: ownership/commit ordering failed")
    _capture(hardware["iteration_stats"])


def _qualification_bundle(path: pathlib.Path, source: dict, numerical: dict, protocol_sha256: str) -> dict:
    """Validate seven raw hetero_ctl wrappers, never a summary-only certificate.

    Bundle fields: backend_revision, runtime_revision, full engine_settings,
    evidence: {gate_H, gate_S, gate_M, lifecycle, continuation, recompute, kv_host},
    each containing repository-relative path and file_sha256.
    """
    from hetero.engine_actor import EngineSettings
    from hetero.native_backend import BACKEND_REVISION, RUNTIME_REVISION
    from hetero.qualify_continuation import PROTOCOL_SHA256

    bundle = json.loads(path.read_text())
    _require(set(bundle) == {"backend_revision", "runtime_revision", "engine_settings", "evidence"},
             "qualification bundle requires revisions, full engine_settings and evidence")
    _require(bundle["backend_revision"] == BACKEND_REVISION and bundle["runtime_revision"] == RUNTIME_REVISION,
             "qualification bundle does not match current native backend/runtime")
    settings = bundle["engine_settings"]
    _require(EngineSettings.from_dict(settings).to_dict() == settings, "EngineSettings must be complete and explicit")
    _require(settings["dtype"] == "bfloat16" and settings["enable_prefix_caching"] is False
             and settings["model"] == numerical["model"], "qualification configuration differs from frozen model contract")
    _require(protocol_sha256 == PROTOCOL_SHA256, "original numerical protocol bytes changed")
    _require(set(bundle["evidence"]) == set(EVIDENCE_KEYS), "all seven physical qualification artifacts are required")
    for key in EVIDENCE_KEYS:
        ref = bundle["evidence"][key]
        _require(set(ref) == {"path", "file_sha256"}, f"{key}: expected path/file_sha256")
        evidence_path = ROOT / ref["path"]
        _require(not pathlib.Path(ref["path"]).is_absolute() and evidence_path.resolve().is_relative_to(ROOT),
                 f"{key}: evidence must be repository-relative")
        raw = evidence_path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == ref["file_sha256"], f"{key}: evidence file hash differs")
        wrapper = json.loads(raw)
        deployed_sources = {name: digest for name, digest in wrapper["source_digest"].items()
                            if not name.startswith("input:")}
        _require(wrapper["status"] == "SUCCEEDED" and deployed_sources == source["source_sha256"],
                 f"{key}: unsuccessful wrapper or stale deployed source")
        if key == "continuation":
            _require(wrapper["source_digest"].get("input:OMP_HETERO_PROTOCOL_JSON") == protocol_sha256,
                     "continuation submitted protocol differs from the frozen numerical protocol")
        measurements = wrapper["measurements"]
        _require(measurements["ok"] is True, f"{key}: command failed")
        try:
            if key.startswith("gate_"):
                _gate(measurements["gate"], key[-1], settings)
            elif key == "continuation":
                _continuation(measurements["qualification"], numerical, protocol_sha256)
            elif key == "lifecycle":
                report = measurements["qualification"]
                _require(report["passed"] is True and not report.get("error") and not report.get("finalization_error"),
                         "lifecycle qualification failed")
                _all_passed(report["checks"], "lifecycle invariants failed")
            else:
                _migration(measurements, key)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{key}: invalid raw qualification evidence: {exc}") from exc
    return bundle



def _manifests() -> list[dict]:
    summaries = []
    sources: set[str] = set()
    for name in WORKLOADS:
        manifest = load_manifest(name, ROOT)
        rows = manifest["requests"]
        ids = {row["source_id"] for row in rows}
        prompts = {tuple(row["prompt_token_ids"]) for row in rows}
        if len(ids) != len(rows) or len(prompts) != len(rows) or sources & ids:
            raise ValueError(f"{name}: conversations/prompts not unique or conversations overlap")
        if (len(rows) != manifest["num_requests"]
                or sum(row["max_tokens"] for row in rows) != manifest["useful_output_tokens"]
                or any(len(row["prompt_token_ids"]) != row["demanded_prompt_tokens"] for row in rows)):
            raise ValueError(f"{name}: demand or token budget mismatch")
        sources.update(ids)
        summary = {key: value for key, value in manifest.items() if key != "requests"}
        summary["path"] = f"data/manifests/{name}.json"
        summary["file_sha256"] = hashlib.sha256((ROOT / summary["path"]).read_bytes()).hexdigest()
        summaries.append(summary)
    return summaries


def build_amendment(*, source_frozen: bool = False,
                    qualification_bundle: pathlib.Path | None = None) -> dict:
    original = json.loads((ROOT / PREDECLARATION_PATH).read_text())
    numerical_path = "results/hetero/continuation_qualification_protocol.json"
    numerical = json.loads((ROOT / numerical_path).read_text())
    qualified = None
    source = None
    if source_frozen:
        if qualification_bundle is None:
            raise ValueError("--qualification-bundle is required for a new FP32 source freeze; no timing is authorized")
        source = _source_freeze()
        qualified = _qualification_bundle(qualification_bundle, source, numerical,
            hashlib.sha256((ROOT / numerical_path).read_bytes()).hexdigest())
        source["status"] = "qualified"
        source["timing_authorization"] = {
            "authorized_arms": ["H", "S", "M0", "MR", "MK-H"],
            "authorized_reason": "All seven raw physical gates passed with identical frozen source, native backend and FP32 engine settings under unchanged numerical criteria.",
            "unauthorized_arms": [],
        }
    elif qualification_bundle is not None:
        raise ValueError("qualification bundle requires a source freeze")
    # Six blocks balance every layout position and both orders of every pair.
    orders = list(itertools.permutations(("H", "S", "M")))
    m_orders = list(itertools.permutations(("M0", "MR", "MK-H")))
    schedule = []
    for workload in WORKLOADS:
        for index, order in enumerate(orders):
            arms = []
            for layout in order:
                arms.extend(m_orders[index] if workload == "W0" and layout == "M"
                            else (("M0", "MR") if index % 2 == 0 else ("MR", "M0"))
                            if layout == "M" and workload in ("W1", "W2")
                            else [] if layout == "M" else [layout])
            schedule.append({"workload": workload, "repetition": index + 1, "arms": arms})
    payload = build_predeclaration(
        ROOT, manifests=_manifests(),
        layouts={"H": {"shape": "2 x TP16", "allocated_chips": 32},
                 "S": {"shape": "4 x TP8", "allocated_chips": 32},
                 "M": {"shape": "TP8 + TP8 + TP16", "allocated_chips": 32}},
        engine_settings=qualified["engine_settings"] if qualified else original["engine_settings"],
        qualification={"status": "qualified" if qualified else "pending",
                       "timing_authorized": qualified is not None,
                       "authorized_scope": "H/S/M0/MR/MK-H, identical qualified FP32 runtime only" if qualified else "none"},
        arm_order=schedule, repetitions=6,
    )
    payload.pop("predeclaration_sha256")
    payload.update(
        kind="prospective_evaluation_amendment",
        supersedes_for_future_runs_only={"path": PREDECLARATION_PATH,
                                       "sha256": original["predeclaration_sha256"]},
        source_freeze=source if source_frozen else {
            "status": "pending", "snapshot_scope": "Local source snapshot only, not qualified loaded source.",
            "required": "Create a new immutable final freeze after source changes settle; qualify identical loaded source/configuration/images before timing. No code changes within matched arms."},
        frozen_treatment="MR",
        numerical_qualification={"path": numerical_path,
                                 "file_sha256": hashlib.sha256((ROOT / numerical_path).read_bytes()).hexdigest(),
                                 "same_configuration": numerical["same_configuration"],
                                 "cross_configuration": numerical["cross_configuration"],
                                 "failure_rule": numerical["failure_rule"]},
        admission_symmetry="M0/MR/MK-H share fresh routing, large share, headroom, eligibility, engine limits and ingress. Only migration enablement/mechanism differ. No future-length oracle privilege.",
        resource_matching="32 allocated TPU chips for every arm, including idle chips; matched model, CPU/network allocation, ingress, instrumentation and workload token budget. Record any differences.",
        historical_exclusions=["The original39-run cohort remains excluded from this qualified FP32 cohort; never pool, relabel or overwrite it.",
                               "All 18 direct-Ray rows remain diagnostic, unqualified and unmatched to new HTTP runs.",
                               "Archived duplicate-prompt manifests are not the corrected manifest cohort.",
                               "Synthetic-prompt historical efficiency numbers are not real-conversation evidence."],
        optional_ici={"status": "deferred", "reason": "No qualified ICI transport; no host fallback labeled ICI."},
        measurement_groups={"efficiency": "results/hetero/efficiency_protocol.json",
                            "mechanism_cost": "results/hetero/mechanism_cost_protocol.json",
                            "end_to_end": "W0: H/S/M0/MR/MK-H; W1/W2: H/S/M0/frozen MR"},
    )
    if qualified is not None:
        payload["qualified_runtime"] = qualified
        # Do not carry the original BF16 illustrative byte estimate into FP32.
        payload["hypotheses"] = [dict(row, note="Host KV costs use actual allocated FP32 cache shapes and measured payload bytes; transport benefit remains an empirical contrast.")
                                 if row["id"] == "H3" else row for row in payload["hypotheses"]]
    payload["minimum_useful_effect"]["declared_before_any_timing_run"] = False
    payload["minimum_useful_effect"]["scope"] = "Unchanged 5% threshold, frozen before corrected eligible timing; earlier diagnostics already observed."
    payload["warmup"].update(
        catalogue_window_start=27000, prompt_token_limit=256,
        exclusion="All five evaluation source-ID sets and token vectors; eight unique real conversations, record exact IDs/tokens/catalogue hash. Drain warmup and verify native idle before timing.",
        selection_state="Exact warmup token vectors must be captured and validated on the mounted tokenizer before final source freeze.",
    )
    payload["analysis"]["secondary_metrics"] = [
        "useful tokens/s per all 32 allocated chips", "common deep-backlog native useful-token throughput",
        "terminal drain duration/fraction", "per-iteration phase, latency, batch/context and retirement eligibility",
        "initialization separately and initialization plus serving", "migration eligibility/proposals/commits/fallback coverage",
        "actual host KV payload bytes separately from measured network bytes; missing network telemetry stays null",
    ]
    payload["analysis"]["sustained_progress"] = {
        "begin": "client completion ceil(0.10 * request_count)",
        "end": "client completion request_count - admitted_population_cap",
        "tokens": "Native emitted useful outputs in (begin,end]; do not count replay or duplicates.",
        "terminal_drain": "Time from end boundary to last client completion; whole run if requests <= admitted cap.",
        "missing_interval": "Report whole makespan only; never invent sustained evidence or silently deepen a workload.",
        "clock": "Host timestamps, not physical device/kernel timing.",
    }
    payload["analysis"]["pairing"] = "Within workload, pair whole-run repetition and exact evaluation/source/config/manifest/budget cohort. Six balanced pairs; report each effect, median and range without p-values. No cherry-picking retries."
    payload["predeclaration_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-freeze", action="store_true",
                        help="Freeze committed deployed source after all seven physical qualifications")
    parser.add_argument("--qualification-bundle", type=pathlib.Path,
                        help="Required for new source freezes: revisions, full engine_settings, and seven raw evidence path/file_sha256 references")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--check", action="store_true", help="Validate current manifests and prospective plan without writing")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / (SOURCE_FROZEN_AMENDMENT if args.source_freeze else AMENDMENT)
    if args.qualification_bundle and not args.source_freeze:
        parser.error("--qualification-bundle requires --source-freeze")
    # Checking an immutable historical artifact neither imports current settings
    # into it nor attempts to create/requalify a new freeze.
    if args.check and args.output.exists():
        frozen = json.loads(args.output.read_text())
    else:
        payload = build_amendment(source_frozen=args.source_freeze,
                                  qualification_bundle=args.qualification_bundle)
        frozen = payload
    if args.check:
        body = {key: value for key, value in frozen.items() if key != "predeclaration_sha256"}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != frozen["predeclaration_sha256"] or frozen["manifests"] != _manifests():
            raise ValueError("amendment hash or current manifest provenance differs from frozen artifact")
        for workload in WORKLOADS:
            blocks = [row for row in frozen["arm_order"] if row["workload"] == workload]
            if len(blocks) != 6 or {row["repetition"] for row in blocks} != set(range(1, 7)):
                raise ValueError(f"{workload}: expected six independent balanced repetitions")
            arms = {arm for row in blocks for arm in row["arms"]}
            for left, right in itertools.combinations(sorted(arms), 2):
                if any(row["arms"].count(left) != 1 or row["arms"].count(right) != 1 for row in blocks):
                    raise ValueError(f"{workload}: incomplete paired block")
                forward = sum(row["arms"].index(left) < row["arms"].index(right) for row in blocks)
                if forward * 2 != len(blocks):
                    raise ValueError(f"{workload}: unbalanced {left}/{right} order")
        print("Validated immutable hash, five corrected manifests and six balanced pairs; "
              "historical check does not authorize timing with current code.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects the original declaration if explicitly selected.
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Frozen prospective amendment {args.output}; "
          + ("H/S/M0/MR/MK-H timing authorized for the qualified FP32 runtime only."
             if args.source_freeze else "source freeze and qualification pending; no timing authorized."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
