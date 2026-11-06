#!/usr/bin/env python3
"""Turn FastSnapshot / OverlapCompile result JSON into audited report tables.

Reads whatever exists under the two owners' result directories, prints Markdown
tables (median, min-max per worker/host), computes the switch cost C on the
report definition when the runs provide it, runs direction-matched break-even
on the measured throughput table, and enforces the Auditor's acceptance gates
(~/hetero-wt/Audit/results/audit/claims.md).  Every row that fails a gate is
printed with `NOT ACCEPTED` and its reason; nothing is imputed.

Gates (raw JSON plus explicit owner qualification; exclusions always veto):
  Q  qualification     missing qualification is unqualified, not a warm control;
                       resolved production config and effective cache must match.
  G1 d2h_complete      every array acquired on the host before the timer stops
                       (bench: sample.verified and verify.ok; writer: did_copy
                       arrays or the host cache was empty before the write).
  G2 durable           the bytes were flushed to shared memory inside the timer
                       (flush_s recorded for every chip, fsync+publish inside
                       snapshot_s; files.json confirms published sizes;
                       bench: memcmp of destination == source).
  G3 no_noop           a 'reused' destination counts only if it was poisoned
                       beforehand; reused_files on a cold run is a failure;
                       cold file bytes must equal the live source fingerprint
                       and restored weights must equal the source fingerprint.
  G4 listener_hits     cache hits come from the compile_or_get_cached read hook
                       (rows carrying `hit`), never from expected-key matching.
  G5 fresh_cache_dir   cold runs use a compilation cache directory that is
                       unique to that run id and absent from any earlier run.
  G6 like_for_like     serving impact compares pilots with the same layout,
                       policy and manifest, all requests complete, no errors.
  C  switch cost       C = first dispatch - last old completion + ramp deficit
                       (first 120 s at the median middle-60% 10 s rate).

Usage:
  reconfig_results_tables.py [--fast DIR] [--overlap DIR] [--baseline DIR]
      [--table throughput_table_measured.json] [--out current_tables.md]
      [--json current_tables.json] [--helper DIR] [--audit DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import csv
import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reconfig_breakeven import breakeven  # noqa: E402

FAST = Path("/home/hongtao/hetero-wt/FastSnapshot/results/restore/FastSnapshot")
OVERLAP = Path("/home/hongtao/hetero-wt/OverlapCompile/results/overlap/OverlapCompile")
BASELINE = Path("/home/hongtao/hetero-wt/ShardedRestore/results/restore/ShardedRestore")
TABLE = Path("/home/hongtao/hetero-research/results/reconfig/throughput_table_measured.json")
HELPER = Path("/home/hongtao/hetero-wt/SnapshotHelper/results/restore/SnapshotHelper")
AUDIT = Path("/home/hongtao/hetero-wt/AuditFollowup/results/audit")

RESTORE_PHASES = ("init_device", "load_model", "snapshot_manifest_verify", "snapshot_read_tmpfs",
                  "snapshot_verify", "snapshot_generation_vote", "snapshot_device_placement", "initialize_from_config",
                  "compile_or_warm_up_model")
WRITE_PARTS = ("metadata_s", "enqueue_s", "transfer_s", "transfer_wait_s", "allocation_s",
               "copy_s", "checksum_s", "flush_s", "publish_s")
# Report layouts and the engine tensor-parallel degree each one deploys.
LAYOUT_TP = {"H": {16}, "S": {8}, "M": {16, 8}}


# ----------------------------------------------------------------------------- helpers
def load(path):
    return json.loads(Path(path).read_text())


def measurements(path):
    doc = load(path)
    return doc.get("measurements", doc), doc


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def fmt(s, digits=2):
    if s is None:
        return "—"
    if s["n"] == 1:
        return f"{s['median']:.{digits}f}"
    return f"{s['median']:.{digits}f} ({s['min']:.{digits}f}–{s['max']:.{digits}f}, n={s['n']})"


def verdict(gates):
    failed = [g for g, ok in gates.items() if ok is not True]
    return "accepted" if not failed else "NOT ACCEPTED: " + ", ".join(failed)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def layout_of_tp(tp_sizes):
    """Direction label for a set of engine TP sizes; None when it matches no report layout."""
    for name, tps in LAYOUT_TP.items():
        if set(tp_sizes) == tps:
            return name
    return None


# ----------------------------------------------------------------------------- FastSnapshot bench
def bench_rows(path):
    """Component bandwidth cells from snapshot_bandwidth.py output (bandwidth*.json)."""
    m, doc = measurements(path)
    rows = []
    for op in m.get("ops", []):
        kind = op["op"].get("kind")
        for host in op.get("hosts", []):
            if host.get("status") in ("not_applicable", "skipped", "unavailable"):
                continue
            if kind in ("d2h", "h2d"):
                samples = host.get("samples") or []
                verified = bool(samples) and all(
                    s.get("verified") is True and (s.get("verify") or {}).get("ok") is True for s in samples)
                gates = {"op_ok": host.get("status") == "ok"}
                if kind == "d2h":
                    # every array acquired and byte-verified on the host before the timer stopped
                    gates["d2h_complete"] = verified
                    gates["counted_full_payload"] = bool(samples) and all(
                        (s.get("verify") or {}).get("bytes") == host.get("payload_bytes") for s in samples)
                    gates["fresh_source"] = all(s.get("host_cache_state") == "cold" for s in samples)
                else:
                    # device arrays ready and checked, no host pointer aliasing a device buffer
                    gates["h2d_complete"] = verified and all(
                        (s.get("device_check") or {}).get("ok") is True
                        and not s.get("host_pointer_aliases_device_buffer") for s in samples)
                rows.append({
                    "file": path.name, "op": op["id"], "host": host.get("host"),
                    "payload_bytes": host.get("payload_bytes"),
                    "elapsed_s": stats([s.get("elapsed_s") for s in samples]),
                    "gbps": stats([s.get("gbps") for s in samples]),
                    "gates": gates, "verdict": verdict(gates),
                })
            elif kind == "host_copy":
                for cell in host.get("cells") or []:
                    gates = {
                        "cell_ok": cell.get("status") == "ok",
                        "durable": cell.get("dest_equals_source_memcmp") is True,
                        "no_noop": cell.get("mode") != "reused" or cell.get("poison_present_before_copy") is True,
                        # a host copy never includes D2H: it may not be reported as a snapshot time
                        "d2h_complete": False,
                    }
                    rows.append({
                        "file": path.name, "op": f"{op['id']}/{cell.get('mode')}/{cell.get('route')}"
                                f"/t{cell.get('threads')}/{cell.get('digest')}",
                        "host": host.get("host"), "payload_bytes": host.get("payload_bytes"),
                        "elapsed_s": stats([cell.get("elapsed_s")]), "gbps": stats([cell.get("gbps")]),
                        "gates": gates,
                        "verdict": verdict({k: v for k, v in gates.items() if k != "d2h_complete"})
                        + " (host copy only; not a snapshot time)",
                    })
    return {"file": str(path), "submission_id": doc.get("submission_id"), "ok": m.get("ok"),
            "failed_ops": m.get("failed_ops"), "rows": rows}


# ----------------------------------------------------------------------------- FastSnapshot writes
def fingerprint_passed(path):
    """weight_equality.passed from a *-fingerprint.json; None when the file is absent."""
    if path is None or not Path(path).is_file():
        return None
    m, _ = measurements(path)
    return (m.get("weight_equality") or {}).get("passed")


def snapshot_write_rows(path, label, poison=None, files=None, baseline=None, source_fingerprint=None):
    """Per-worker write times from snapshot_validate.py snapshot action (*-cold/-reused.json).

    `did_copy_arrays`/`host_cache_bytes_before` only describe the Python-side copy;
    they cannot prove the PJRT internal host cache was cold, so a second run on the
    same live source is flagged `source_possibly_warm` (a caveat column, not a gate).
    """
    m, doc = measurements(path)
    rows = []
    bytes_equal = fingerprint_passed(source_fingerprint)
    poisoned = None
    if poison is not None:
        pm, _ = measurements(poison)
        poisoned = {h.get("host"): h.get("poisoned_bytes", 0) for h in pm.get("hosts", [])}
    published = None
    if files is not None:
        fm, _ = measurements(files)
        published = fm
    for engine, workers in (m.get("workers") or {}).items():
        for w in workers:
            t = w.get("timings") or {}
            chips = w.get("chip_timings") or []
            cold_source = w.get("host_cache_bytes_before", None) == 0
            gates = {
                "run_succeeded": doc.get("status") == "SUCCEEDED" and m.get("ok") is True,
                "d2h_complete": bool(chips) and (
                    all(c.get("did_copy_arrays", 0) > 0 for c in chips) or cold_source)
                    and t.get("transfer_s", 0) > 0,
                "durable": bool(chips) and all("flush_s" in c for c in chips)
                    and "publish_s" in t and w.get("snapshot_s", 0) >= t.get("transfer_s", 0),
                "no_noop": (w.get("reused_files", 0) == 0 if label == "cold" else
                            bool(poisoned) and all(v > 0 for v in poisoned.values())),
                "checksum_declared": "checksum_kind" in w,
            }
            gates["published_files_valid"] = published is not None and published.get("ok") is True
            if label == "cold":
                # cold file bytes must equal the live device weights (S-source-fingerprint.json)
                gates["cold_bytes_equal_source"] = bytes_equal is True
            old = None
            if baseline:
                old = next((r for r in baseline.get(engine, [])
                            if r.get("process_rank") == w.get("process_rank")), None)
            rows.append({
                "file": path.name, "label": label, "engine": engine, "process_rank": w.get("process_rank"),
                "bytes": w.get("bytes"), "snapshot_s": w.get("snapshot_s"),
                "GBps": w["bytes"] / w["snapshot_s"] / 1e9 if w.get("snapshot_s") else None,
                "parts": {k: t.get(k) for k in WRITE_PARTS},
                "reused_files": w.get("reused_files"), "new_files": w.get("new_files"),
                "checksum_kind": w.get("checksum_kind"),
                "baseline_s": old.get("snapshot_s") if old else None,
                "source_possibly_warm": label != "cold" or not cold_source,
                "scope": "cold-destination snapshot publication" if label == "cold" else "reused-destination publication; PJRT source possibly cached, not cold D2H",
                "cold_snapshot_proof": label == "cold" and cold_source and verdict(gates) == "accepted",
                "fingerprint_verified_outside_timer": bytes_equal is True,
                "phase_semantics": w.get("phase_semantics"),
                "gates": gates, "verdict": verdict(gates),
            })
    return {"file": str(path), "submission_id": doc.get("submission_id"), "label": label, "rows": rows}


# ----------------------------------------------------------------------------- restores
def worker_phase_lists(m):
    """Yield (engine, worker) from either the validate/check or the deploy schema."""
    if "worker_init_phases" in m and isinstance(m["worker_init_phases"], dict):
        for engine, workers in m["worker_init_phases"].items():
            for w in workers:
                yield engine, w, None
    for inst in (m.get("deployment") or {}).get("instances", []):
        ini = inst.get("initialization") or {}
        for w in ini.get("worker_init_phases", []):
            yield inst["descriptor"]["instance_id"], w, ini


def restore_rows(path, release=None, require_restored=True, restored_fingerprint=None, qualification=None):
    m, doc = measurements(path)
    weights_equal = fingerprint_passed(restored_fingerprint)
    release_t = None
    release_ok = None
    if release is not None:
        rel = load(release)
        states = {r.get("state") for r in rel.get("measurements", {}).get("released", [])}
        release_ok = rel.get("status") == "SUCCEEDED" and states == {"REMOVED"}
        release_t = rel.get("saved_at")
    rows = []
    for engine, w, ini in worker_phase_lists(m):
        ph = w.get("phases") or {}
        ws = w.get("weight_snapshot") or {}
        d = {k: ph[k]["end"] - ph[k]["start"] for k in RESTORE_PHASES if k in ph}
        gates = {"run_succeeded": doc.get("status") == "SUCCEEDED" and m.get("ok") is True}
        if require_restored:
            gates["restored"] = ws.get("restored") is True
            gates["no_fallback"] = not ws.get("fallback_reason")
            gates["restored_weights_equal_source"] = weights_equal is True
            gates["producer_released_first"] = bool(release_ok) and release_t is not None \
                and "load_model" in ph and release_t < ph["load_model"]["start"]
            gates["h2d_complete"] = "snapshot_device_placement" in ph and "load_model" in ph \
                and ph["snapshot_device_placement"]["end"] <= ph["load_model"]["end"]
        else:
            profiles = w.get("startup_profiles") or {}
            gates["startup_profiles_present"] = bool(profiles)
            for profile in profiles.values():
                for gate, passed in profile_gates(ini, profile, qualification or {}).items():
                    gates[gate] = gates.get(gate, True) and passed
        rows.append({"file": path.name, "engine": engine, "executor_rank": w.get("executor_rank"),
                     "phases_s": d, "weight_snapshot": ws,
                     "scope": "independent restore correctness/component timing" if require_restored else "startup attribution",
                     "evidence_qualification": qualification,
                     "checksum_verified_in_path": ws.get("verified_checksums") is True and ws.get("verified_bytes", 0) > 0,
                     "fingerprint_verified_outside_timer": weights_equal is True,
                     "unattributed_load_s": (d["load_model"] - sum(v for k, v in d.items() if k.startswith("snapshot_"))
                                             if "load_model" in d else None),
                     "gates": gates, "verdict": verdict(gates)})
    extra = {k: m.get(k) for k in ("greedy_equality", "weight_equality") if k in m}
    return {"file": str(path), "submission_id": doc.get("submission_id"), "rows": rows, **extra}


# ----------------------------------------------------------------------------- OverlapCompile
def overlap_qualification(path, doc, report=None):
    """Join owner exclusions by artifact identity, never by a cold/warm filename."""
    report = report or {}
    run = next((r for r in report.get("runs", [])
                if r.get("run_id") == doc.get("submission_id")
                and r.get("source") and str(path).endswith(r["source"])), {})
    qualification = run.get("qualification", "unqualified")
    reasons = list(run.get("exclusion_reasons", []))
    if run and report.get("status", "").startswith("diagnostic"):
        qualification = "diagnostic_only"
        reasons += report.get("limitations", [])
    return {"qualification": qualification, "exclusion_reasons": reasons,
            "source": run.get("source"), "run_id": run.get("run_id")}


def profile_gates(ini, profile, qualification):
    config = (ini or {}).get("resolved_engine_config") or {}
    requested = ((ini or {}).get("engine_settings") or {}).get("compilation_cache_dir")
    expected = {"max_model_len": 16384, "max_num_seqs": 512,
                "max_num_batched_tokens": 2048, "enable_prefix_caching": False}
    return {
        "evidence_qualified": qualification.get("qualification") == "controlled"
            and not qualification.get("exclusion_reasons"),
        "production_config": all(k in config and config[k] == v for k, v in expected.items()),
        "effective_cache_matches_requested": bool(requested)
            and profile.get("compilation_cache_dir") == requested,
    }


def warmup_rows(path, run_dir_name, earlier_cache_dirs, qualification=None):
    """Keep diagnostics visible, but never treat warm status as control qualification."""
    m, doc = measurements(path)
    qualification = qualification or {"qualification": "unqualified", "exclusion_reasons": []}
    rows = []
    for engine, w, ini in worker_phase_lists(m):
        profiles = w.get("startup_profiles") or {}
        for phase in ("compile_or_warm_up_model", "initialize_from_config"):
            p = profiles.get(phase)
            if p is None:
                continue
            intervals = p.get("intervals") or []
            reads = [r for r in intervals if "hit" in r]
            hits = sum(1 for r in reads if r["hit"])
            misses = len(reads) - hits
            counter = sum(int(e.get("metadata", {}).get("count", 1)) for e in p.get("monitoring_events", [])
                          if e.get("name") == "/jax/compilation_cache/cache_hits")
            cache_dir = p.get("compilation_cache_dir") or ""
            is_cold = "cold" in path.name or "cold" in cache_dir
            gates = profile_gates(ini, p, qualification)
            gates["run_succeeded"] = doc.get("status") == "SUCCEEDED" and m.get("ok") is True
            gates.update({"listener_hits": bool(reads) or (hits == 0 and misses == 0 and not counter),
                          "hits_from_hook_only": counter in (0, hits)})
            if is_cold:
                gates["fresh_cache_dir"] = bool(cache_dir) and run_dir_name in cache_dir \
                    and cache_dir not in earlier_cache_dirs
            rows.append({"file": str(path.relative_to(path.parents[1])), "engine": engine,
                         "executor_rank": w.get("executor_rank"), "phase": phase,
                         "wall_s": p.get("wall_s"), "exclusive_wall_s": p.get("exclusive_wall_s"),
                         "cache_hits": hits, "cache_misses": misses, "jax_cache_hit_counter": counter,
                         "executables": len(p.get("executables") or []),
                         "cache_dir": cache_dir, "cold": is_cold,
                         "cache_claim": "cold candidate" if is_cold else "not established as cold",
                         "evidence_qualification": qualification,
                         "device_module_execution_wall_s": (p.get("device_trace") or {}).get(
                             "device_module_execution_wall_s"),
                         "gates": gates, "verdict": verdict(gates)})
    return {"file": str(path), "submission_id": doc.get("submission_id"), "rows": rows}


def startup_projection_rows(path):
    """Read the bounded owner projection, not repeated multi-GB deployment profiles."""
    doc = load(path)
    if doc.get("schema") != "overlap-startup-table-projection-v1":
        raise ValueError(f"Unsupported startup projection schema in {path}")
    checks = []
    for entry in doc.get("sources", []):
        source = path.parents[3] / entry["path"]
        matched = False
        if source.is_file():
            with source.open("rb") as stream:
                matched = hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
        checks.append(matched)
    sources_verified = bool(checks) and all(checks)
    runs = []
    for run in doc.get("runs", []):
        workers = run.get("workers") or []
        identities = {(w.get("instance"), w.get("executor_rank")) for w in workers}
        hosts = {w.get("host") for w in workers}
        q = {key: run.get(key) for key in ("qualification", "exclusion_reasons", "source", "run_id")}
        q["exclusion_reasons"] = q.get("exclusion_reasons") or []
        rows = []
        for w in workers:
            cache = w.get("compile_cache") or {}
            ini = {"resolved_engine_config": w.get("resolved_engine_config"),
                   "engine_settings": {"compilation_cache_dir": w.get("requested_cache_dir")}}
            gates = profile_gates(ini, {"compilation_cache_dir": w.get("phase_cache_dir")}, q)
            hits, misses = w.get("cache_hits"), w.get("cache_misses")
            gates.update({
                "projection_sources_verified": sources_verified,
                "eight_unique_workers": len(workers) == len(identities) == len(hosts) == 8 and None not in hosts,
                "rpc_cache_matches_requested": bool(w.get("requested_cache_dir"))
                    and cache.get("dir") == w.get("requested_cache_dir"),
                "coverage_matched_thresholds": w.get("phase_min_compile_time_secs") == 0
                    and w.get("phase_min_entry_size_bytes") == 0 and cache.get("min_compile_time_secs") == 0
                    and cache.get("min_entry_size_bytes") == 0,
                "logging_disabled": cache.get("log_compiles") is False and cache.get("explain_cache_misses") is False,
                "requested_controls": all((w.get("requested_worker_env") or {}).get(k) == "0" for k in (
                    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
                    "JAX_LOG_COMPILES", "JAX_EXPLAIN_CACHE_MISSES")),
                "keyed_cache_evidence": hits is not None and misses is not None
                    and w.get("keyed_cache_read_count") == hits + misses
                    and w.get("successful_keyed_deserialization_count") == hits
                    and w.get("cache_hit_listener_count") == hits,
            })
            rows.append({"file": run["source"], "engine": w.get("instance"), "executor_rank": w.get("executor_rank"),
                         "phase": run.get("phase"), "wall_s": w.get("host_call_wall_s"),
                         "exclusive_wall_s": w.get("exclusive_host_call_categories_s"),
                         "cache_hits": hits, "cache_misses": misses,
                         "jax_cache_hit_counter": w.get("cache_hit_listener_count"),
                         "executables": w.get("executable_count"), "cache_dir": w.get("phase_cache_dir"),
                         "cache_claim": "observed keyed reads; cold/warm labels alone are not cache proof",
                         "device_module_execution_wall_s": None, "evidence_qualification": q,
                         "source_qualified": run.get("source_qualified") is True,
                         "scope": "host-call observation only; no device attribution or loaded-runtime equivalence",
                         "projection_worker": w, "gates": gates, "verdict": verdict(gates)})
        runs.append({"file": run["source"], "submission_id": run.get("run_id"), "rows": rows,
                     "raw_archive": run.get("raw_archive"), "raw_uncompressed_sha256": run.get("raw_uncompressed_sha256")})
    metadata = {k: v for k, v in doc.items() if k != "runs"}
    metadata.update({"file": str(path), "source_manifest_verified": sources_verified})
    return runs, metadata


# ----------------------------------------------------------------------------- pilots / C
def rate_series(rows, start, bucket=10.0):
    acc = {}
    for r in rows:
        i = int((float(r["completed_at_s"]) - start) // bucket)
        acc[i] = acc.get(i, 0) + int(float(r.get("emitted_output_tokens") or 0))
    return [acc.get(i, 0) / bucket for i in range(max(acc) + 1)] if acc else []


def pilot_from_artifacts(directory):
    """Serving-impact row from a run's client artifacts (records + iterations + summary)."""
    directory = Path(directory)
    records_path = next((p for p in directory.glob("*_records.json.gz") if "gateway" not in p.name), None)
    if records_path is None:
        return None
    records = json.load(gzip.open(records_path))
    iterations = list(csv.DictReader(gzip.open(
        records_path.with_name(records_path.name.replace("_records.json.gz", "_iterations.csv.gz")), "rt")))
    summary = load(records_path.with_name(records_path.name.replace("_records.json.gz", "_summary.json")))
    first = min(r["client_started_at_s"] for r in records)
    last = max(r["client_finished_at_s"] for r in records)
    series = rate_series(iterations, first)
    n = len(series)
    steady = statistics.median(series[int(.2 * n):max(int(.8 * n), int(.2 * n) + 1)]) if series else None
    deficit = sum(max(steady - x, 0) for x in series[:12]) * 10 / steady if steady else None
    return {"run_id": directory.name, "first_dispatch": first, "last_finished_request": last,
            "makespan_s": summary.get("makespan_s"), "useful_tokens_per_s": summary.get("useful_tokens_per_s"),
            "steady_tokens_per_s": steady, "ramp_deficit_s": deficit,
            "all_requests_complete": summary.get("all_requests_complete"),
            "policy": summary.get("policy"), "manifest": summary.get("manifest") or summary.get("workload"),
            "layout": summary.get("layout"), "during_run": summary.get("during_run"),
            "manifest_sha256": summary.get("manifest_sha256"),
            "catalogue_sha256": (summary.get("warmup") or {}).get("catalogue_sha256"),
            **{key: summary.get(key) for key in ("expected_catalogue_sha256", "source_qualified", "control_qualified",
                "unfrozen_pilot", "runtime_revision", "backend_revision")}}


def known_identity(value):
    return isinstance(value, str) and bool(value) and value.upper() not in {"UNFROZEN", "UNKNOWN"}


def comparison_gates(source, target):
    """Actual inputs and loaded source/control proof, not a workload name or token count."""
    gates = {}
    for key in ("manifest_sha256", "catalogue_sha256"):
        gates["same_" + key] = known_identity(source.get(key)) and source.get(key) == target.get(key)
    for label, pilot in (("source", source), ("target", target)):
        gates[label + "_catalogue_matches_manifest"] = known_identity(pilot.get("catalogue_sha256")) \
            and pilot.get("catalogue_sha256") == pilot.get("expected_catalogue_sha256")
        gates[label + "_qualified_source"] = pilot.get("source_qualified") is True
        gates[label + "_qualified_control"] = pilot.get("control_qualified") is True
        gates[label + "_frozen_runtime"] = pilot.get("unfrozen_pilot") is False \
            and known_identity(pilot.get("runtime_revision")) and known_identity(pilot.get("backend_revision"))
    gates["matching_runtime"] = known_identity(source.get("runtime_revision")) \
        and source.get("runtime_revision") == target.get("runtime_revision") \
        and known_identity(source.get("backend_revision")) \
        and source.get("backend_revision") == target.get("backend_revision")
    return gates


def serving_gates(control, treated):
    gates = comparison_gates(control, treated)
    gates.update({
        "same_layout": bool(control.get("layout")) and control.get("layout") == treated.get("layout"),
        "same_policy": control.get("policy") is not None and control.get("policy") == treated.get("policy"),
        "all_requests_complete": control.get("all_requests_complete") is True
            and treated.get("all_requests_complete") is True,
        "no_run_error": not (control.get("during_run") or {}).get("error")
            and not (treated.get("during_run") or {}).get("error"),
    })
    return gates


def switch_row(t, source, target, file):
    """Observed timestamp arithmetic and performance qualification are different claims."""
    cost, gap, deficit = t.get("cost_s"), t.get("dispatch_gap_s"), t.get("ramp_deficit_s")
    last, first = t.get("last_finished_request"), t.get("first_dispatch")
    gates = {
        "last_completion_anchor": last is not None and last == source.get("last_finished_request"),
        "first_dispatch_anchor": first is not None and first == target.get("first_dispatch"),
        "ramp_deficit_measured": deficit is not None and deficit == target.get("ramp_deficit_s"),
        "gap_matches_anchors": gap is not None and first is not None and last is not None
            and abs(gap - (first - last)) < 1e-6,
        "cost_is_gap_plus_deficit": cost is not None and gap is not None and deficit is not None
            and abs(cost - (gap + deficit)) < 1e-6,
        "source_all_requests_complete": source.get("all_requests_complete") is True,
        "target_all_requests_complete": target.get("all_requests_complete") is True,
        "no_run_error": bool(source) and bool(target) and not (target.get("during_run") or {}).get("error")
            and not (source.get("during_run") or {}).get("error"),
    }
    performance = comparison_gates(source, target)
    observed = verdict(gates) == "accepted"
    eligible = observed and verdict(performance) == "accepted"
    return {"file": file, "from": t.get("from"), "to": t.get("to"), "treatment": t.get("treatment"),
            "cost_s": cost, "dispatch_gap_s": gap, "ramp_deficit_s": deficit,
            "previous_run_id": source.get("run_id"), "target_run_id": target.get("run_id"),
            "observation_gates": gates, "observation_verdict": verdict(gates),
            "performance_gates": performance, "performance_eligible": eligible,
            "comparison_provenance": {"source": source, "target": target},
            "scope": "qualified transition observation" if eligible else "unqualified/unfrozen pilot observation only" if observed
                else "incomplete or rejected transition evidence; no accepted C",
            "cost_ledger": t.get("cost_ledger"), "limitations": t.get("limitations", []),
            "verdict": "accepted" if eligible else "OBSERVED ONLY" if observed else verdict(gates)}


def switch_cost_rows(fast_dir, overlap_dir):
    rows = []
    compact_ids = set()
    for path in sorted(overlap_dir.glob("*/*-transition-observed.json")):
        doc = load(path)
        inputs = doc.get("equal_input_provenance") or {}
        backend = doc.get("loaded_backend_provenance") or {}
        pilots = {}
        for label in ("source", "target"):
            pilot = dict(doc.get(label) or {})
            manifest = next((m for m in inputs.get("manifests", {}).values()
                             if m.get("manifest_sha256") == pilot.get("manifest_sha256")), {})
            pilot.update({"catalogue_sha256": inputs.get("catalogue_sha256"),
                          "expected_catalogue_sha256": manifest.get("catalogue_sha256"),
                          "backend_revision": backend.get("backend_revision"),
                          "runtime_revision": backend.get("runtime_revision"),
                          "unfrozen_pilot": backend.get("backend_revision") == "UNFROZEN"
                              or pilot.get("unfrozen_pilot"),
                          "source_qualified": backend.get("runtime_equivalence_verified") is True,
                          "control_qualified": pilot.get("control_qualified") is True})
            pilots[label] = pilot
        source, target = pilots["source"], pilots["target"]
        direction = doc.get("direction", "").split("_to_")
        t = {**doc, "cost_s": doc.get("C_s"), "last_finished_request": source.get("last_finished_request"),
             "first_dispatch": target.get("first_dispatch"),
             "from": direction[0] if len(direction) == 2 else None,
             "to": direction[1] if len(direction) == 2 else None}
        row = switch_row(t, source, target, str(path.relative_to(overlap_dir)))
        row.update({"input_provenance": inputs, "loaded_backend_provenance": backend,
                    "provisioning": doc.get("provisioning"),
                    "first_use_including_provisioning_s": doc.get("first_use_including_provisioning_s"),
                    "owner_scope": doc.get("scope"), "target_controls": doc.get("target_controls")})
        rows.append(row)
        if source.get("run_id") and target.get("run_id"):
            compact_ids.add((source["run_id"], target["run_id"]))
    for path in sorted(list(fast_dir.glob("cost-*.json")) + list(overlap_dir.glob("**/cost-*.json"))):
        for r in load(path):
            if r.get("cost_s") is None:
                continue
            # Older timers without pilot proof remain retained, not silently accepted.
            provenance = r.get("comparison_provenance") or {}
            rows.append(switch_row({**r, "from": r.get("previous"), "to": r.get("layout"),
                                    "treatment": r.get("treatment", "restore")},
                                   provenance.get("source", {}), provenance.get("target", {}), str(path)))
    for path in sorted(overlap_dir.glob("**/overlap_switch.json")):
        impact = path.with_name("serving_impact.json")
        pilots = load(impact).get("pilots", {}) if impact.is_file() else {}
        for t in load(path).get("transitions", []):
            source = pilots.get(t.get("previous_run_id")) or pilots.get(t.get("source")) or {}
            target = pilots.get(t.get("target_run_id")) or pilots.get(t.get("target")) or {}
            if (source.get("run_id"), target.get("run_id")) in compact_ids:
                continue
            rows.append(switch_row(t, source, target, str(path.relative_to(overlap_dir))))
    return rows


def breakeven_rows(table, cost_rows):
    """Keep incompatible candidates visible, but never compute a crossover for them."""
    out = []
    for c in cost_rows:
        a, b = c.get("from"), c.get("to")
        if not a or not b or c.get("cost_s") is None:
            continue
        provenance = c.get("comparison_provenance") or {}
        for workload, entry in table.items():
            layouts = {k: v for k, v in entry["makespan_s"].items() if v}
            if a not in layouts or b not in layouts:
                continue
            gates = {"observed_cost": c.get("observation_verdict") == "accepted"}
            for label, layout in (("source", a), ("target", b)):
                rate, pilot = layouts[layout], provenance.get(label, {})
                for key in ("manifest_sha256", "catalogue_sha256"):
                    gates[label + "_matching_" + key] = known_identity(rate.get(key)) and rate.get(key) == pilot.get(key)
                gates[label + "_rate_source_qualified"] = rate.get("source_qualified") is True and bool(rate.get("source"))
                gates[label + "_rate_control_qualified"] = rate.get("control_qualified") is True
                gates[label + "_positive_rate"] = isinstance(rate.get("value"), (int, float)) and rate["value"] > 0
            gates["same_actual_inputs"] = known_identity(provenance.get("source", {}).get("manifest_sha256")) \
                and provenance.get("source", {}).get("manifest_sha256") == provenance.get("target", {}).get("manifest_sha256") \
                and known_identity(provenance.get("source", {}).get("catalogue_sha256")) \
                and provenance.get("source", {}).get("catalogue_sha256") == provenance.get("target", {}).get("catalogue_sha256")
            tokens = entry.get("output_tokens")
            gates["positive_token_budget"] = isinstance(tokens, (int, float)) and tokens > 0
            compatible = verdict(gates) == "accepted"
            performance = comparison_gates(layouts[a], layouts[b])
            performance["qualified_transition"] = c.get("performance_eligible") is True
            for label, layout in (("source", a), ("target", b)):
                for key in ("runtime_revision", "backend_revision"):
                    performance[label + "_matching_" + key] = known_identity(layouts[layout].get(key)) \
                        and layouts[layout].get(key) == provenance.get(label, {}).get(key)
            qualified = compatible and verdict(performance) == "accepted"
            n = None
            if compatible:
                n = breakeven(tokens / layouts[a]["value"], tokens / layouts[b]["value"], c["cost_s"])
            out.append({"workload": workload, "from": a, "to": b, "treatment": c.get("treatment"),
                        "cost_s": c["cost_s"], "gain_per_phase_s": layouts[a]["value"] - layouts[b]["value"] if compatible else None,
                        "breakeven_tokens": None if n is None else round(n),
                        "breakeven_phase_multiple": None if n is None else n / tokens,
                        "sources": [layouts[a].get("source"), layouts[b].get("source")],
                        "throughput_provenance": {a: layouts[a], b: layouts[b]},
                        "cost_file": c["file"], "gates": gates, "model_compatible": compatible,
                        "performance_eligible": qualified, "performance_gates": performance, "crossover_observed": False,
                        "scope": "qualified rate-model estimate" if qualified else "compatible provenance-labeled estimate only" if compatible
                            else "ineligible: mismatched or unknown throughput provenance",
                        "verdict": "accepted estimate" if qualified else "ESTIMATE ONLY" if compatible else verdict(gates)})
    return out


def audited_claims(path):
    """Retain audit scope and verify its immutable input manifest against local files."""
    if not path.is_file():
        return {"file": str(path), "source_manifest_verified": False}
    doc = load(path)
    checks = []
    for entry in doc.get("source_manifest", []):
        source = path.parents[3] / entry["path"]
        ok = source.is_file() and source.stat().st_size == entry["bytes"]
        if ok:
            with source.open("rb") as stream:
                ok = hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
        checks.append({"path": str(source), "matched": ok})
    return {"file": str(path), **doc, "source_checks": checks,
            "source_manifest_verified": bool(checks) and all(c["matched"] for c in checks)}


def helper_rows(helper_dir, checksum_audit):
    """Synthetic components with stage boundaries, never full snapshot proof."""
    rows = []
    pinned = helper_dir / "pinned-bandwidth.json"
    if pinned.is_file():
        m, doc = measurements(pinned)
        probe = m.get("probe") or {}
        for c in probe.get("cases", []):
            gates = {"run_succeeded": doc.get("status") == "SUCCEEDED",
                     "payload_verified": c.get("ok") is True and c.get("bytes", 0) > 0
                         and c.get("verified_bytes") == c.get("bytes"),
                     "fresh_source": bool(c.get("source_cache_none_before"))
                         and all(c["source_cache_none_before"])}
            pinned_stage = c.get("mechanism") == "preallocated_pinned_donation"
            rows.append({"file": str(pinned), "submission_id": doc.get("submission_id"),
                         "component": c.get("mechanism"), "scope": probe.get("scope"),
                         "bytes": c.get("bytes"), "chips": c.get("chips"), "repetition": c.get("repetition"),
                         "elapsed_s": c.get("seconds"), "GBps": c.get("GBps"),
                         "timing_boundary": probe.get("timing", "") +
                             ("; ndarray acquisition/copy and final shared-file publication excluded"
                              if pinned_stage else "; ndarray acquisition included; file publication excluded"),
                         "full_snapshot_proof": False, "gates": gates, "verdict": verdict(gates)})
    checksum = helper_dir / "checksum-probe.json"
    if checksum.is_file():
        m, doc = measurements(checksum)
        probe = m.get("probe") or {}
        expected = checksum_audit.get("independent_digest_reconstruction") or {}
        for r in probe.get("runs", []):
            gates = {"run_succeeded": doc.get("status") == "SUCCEEDED" and probe.get("ok") is True,
                     "all_outputs_verified": r.get("checksum_outputs", 0) > 0
                         and r.get("verified_outputs") == r.get("checksum_outputs")
                         and r.get("mismatch_indices") == [],
                     "independent_digest_verified": checksum_audit.get("source_manifest_verified") is True
                         and r.get("checksum_output_sha256") == expected.get("sha256")}
            ready, readback = r.get("kernel_ready_wall_s_including_enqueue"), r.get("small_output_readback_s")
            rows.append({"file": str(checksum), "submission_id": doc.get("submission_id"),
                         "component": "synthetic weak sum/XOR reduction", "scope": checksum_audit.get("scope"),
                         "bytes": probe.get("source_bytes"), "repetition": r.get("repetition"),
                         "elapsed_s": ready + readback if ready is not None and readback is not None else None,
                         "kernel_ready_s": ready, "readback_s": readback, "enqueue_s": r.get("enqueue_s"),
                         "compile_excluded_s": probe.get("compile_s"), "GBps": None,
                         "integrity": checksum_audit.get("integrity"),
                         "timing_boundary": checksum_audit.get("timer_boundaries"),
                         "full_snapshot_proof": False, "gates": gates, "verdict": verdict(gates)})
    return rows


# ----------------------------------------------------------------------------- driver
def collect(fast_dir, overlap_dir, baseline_dir, table_path, helper_dir=HELPER, audit_dir=AUDIT):
    result = {"inputs": {"fast": str(fast_dir), "overlap": str(overlap_dir), "baseline": str(baseline_dir),
                         "table": str(table_path)},
              "bench": [], "snapshot_writes": [], "restores": [], "warmups": [], "pilots": {},
              "serving_impact": [], "switch_costs": [], "breakeven": []}
    result["audits"] = {name: audited_claims(audit_dir / f"followup_{name}_claims.json")
                        for name in ("restore", "checksum")}
    result["helper_components"] = helper_rows(helper_dir, result["audits"]["checksum"])
    result["owner_summaries"] = {name: load(fast_dir / f"{name}.json")
                                 for name in ("snapshot", "restore") if (fast_dir / f"{name}.json").is_file()}
    for path in sorted(fast_dir.glob("bandwidth*.json")):
        run = bench_rows(path)
        if run["rows"]:
            result["bench"].append(run)
    for path in sorted(fast_dir.glob("*-cold.json")) + sorted(fast_dir.glob("*-reused.json")):
        layout, label = path.name.rsplit(".", 1)[0].split("-", 1)
        baseline = None
        bpath = baseline_dir / f"{layout}-snapshot.json"
        if bpath.is_file():
            baseline = load(bpath)["measurements"].get("workers")
        poison = fast_dir / f"{layout}-poison.json"
        files = fast_dir / f"{layout}-files.json"
        result["snapshot_writes"].append(snapshot_write_rows(
            path, label, poison if poison.is_file() else None, files if files.is_file() else None, baseline,
            source_fingerprint=fast_dir / f"{layout}-source-fingerprint.json"))
    for path in sorted(fast_dir.glob("*-restored-validation.json")) + sorted(fast_dir.glob("*-worker-restore.json")):
        layout = path.name.split("-", 1)[0]
        release = fast_dir / f"{layout}-normal-release.json"
        result["restores"].append(restore_rows(path, release if release.is_file() else None,
                                               restored_fingerprint=fast_dir / f"{layout}-restored-fingerprint.json"))
    projection = overlap_dir / "startup_projection.json"
    if projection.is_file():
        result["warmups"], result["startup_projection"] = startup_projection_rows(projection)
        covered = {run["file"] for run in result["warmups"]}
        artifacts = {str(p).removesuffix(".gz") for p in overlap_dir.glob("*/*deploy.json*")
                     if not p.name.startswith("cleanup")}
        result["unprojected_startup_artifacts"] = [p for p in sorted(artifacts)
                                                   if not any(p.endswith(source) for source in covered)]
    else:
        seen_cache_dirs = set()
        report_path = overlap_dir / "warmup_split.json"
        qualification_report = load(report_path) if report_path.is_file() else {}
        for path in sorted(overlap_dir.glob("*/*deploy.json"), key=lambda p: p.stat().st_mtime):
            if path.name.startswith("cleanup"):
                continue
            _, doc = measurements(path)
            qualification = overlap_qualification(path, doc, qualification_report)
            run = warmup_rows(path, path.parent.name, seen_cache_dirs, qualification)
            for r in run["rows"]:
                if r["cache_dir"]:
                    seen_cache_dirs.add(r["cache_dir"])
            if run["rows"]:
                result["warmups"].append(run)
            m, _ = measurements(path)
            if run["rows"] or "deployment" in m:
                result["restores"].append(restore_rows(path, require_restored=False, qualification=qualification))
    result["switch_costs"] = switch_cost_rows(fast_dir, overlap_dir)
    seen_switch_ids = {(r.get("previous_run_id"), r.get("target_run_id")) for r in result["switch_costs"]}
    known_pilots = {p["run_id"]: p for r in result["switch_costs"]
                    for p in r["comparison_provenance"].values() if p.get("run_id")}
    for path in sorted(overlap_dir.glob("*/campaign.json")):
        camp = load(path)
        artifacts = overlap_dir.parents[1] / "hetero/artifacts"
        pilots = {}
        for label, run_id in camp.get("pilots", {}).items():
            p = known_pilots.get(run_id) or pilot_from_artifacts(artifacts / run_id)
            if p:
                pilots[label] = p
        result["pilots"][path.parent.name] = pilots
        for t in camp.get("transitions", []):
            src, tgt = pilots.get(t["source"]), pilots.get(t["target"])
            control = pilots.get(t["source"].replace("overlap", "undisturbed"))
            if src and control and src is not control:
                gates = serving_gates(control, src)
                result["serving_impact"].append({
                    "run": path.parent.name, "control": control["run_id"], "treated": src["run_id"],
                    "control_tps": control["useful_tokens_per_s"], "treated_tps": src["useful_tokens_per_s"],
                    "slowdown_pct": (1 - src["useful_tokens_per_s"] / control["useful_tokens_per_s"]) * 100
                    if control.get("useful_tokens_per_s") else None,
                    "gates": gates, "verdict": verdict(gates)})
            if src and tgt and (src.get("run_id"), tgt.get("run_id")) not in seen_switch_ids:
                first, last = tgt.get("first_dispatch"), src.get("last_finished_request")
                deficit = tgt.get("ramp_deficit_s")
                gap = first - last if first is not None and last is not None else None
                result["switch_costs"].append(switch_row(
                    {**t, "first_dispatch": first, "last_finished_request": last,
                     "cost_s": gap + deficit if gap is not None and deficit is not None else None,
                     "dispatch_gap_s": gap, "ramp_deficit_s": deficit}, src, tgt, str(path.relative_to(overlap_dir))))
                seen_switch_ids.add((src.get("run_id"), tgt.get("run_id")))
    table = load(table_path) if Path(table_path).is_file() else {}
    result["breakeven"] = breakeven_rows(table, result["switch_costs"])
    return result


def render(result):
    out = ["# Reconfiguration results — audited tables", "",
           f"Inputs: `{result['inputs']['fast']}`, `{result['inputs']['overlap']}`; "
           f"throughput table `{result['inputs']['table']}`.",
           "Rows marked `NOT ACCEPTED` fail an Auditor gate (see script docstring); they are kept so negative data is not lost.", ""]
    out += ["Acceptance is scoped to the row, not full reconfiguration. Warm cache names do not establish valid controls. "
            "Cold/reused snapshot labels describe destination pages; reused live sources may be PJRT-cached. "
            "Fingerprint validation is outside timers and is not in-path checksum verification.", ""]
    for name, audit in result.get("audits", {}).items():
        out += [f"Audit {name}: `{audit['file']}`; source manifest matched={audit['source_manifest_verified']}; "
                f"scope={audit.get('verdict', 'unavailable')}.", ""]
        for key in ("timers", "checksum_policy", "scope"):
            if audit.get(key):
                out += [audit[key], ""]
        equality = audit.get("byte_equality")
        if equality:
            out += ["Independent fingerprint chain (outside snapshot/restore timers): "
                    f"{equality['chain']}; {equality['workers']} workers, {equality['shards']} shards, "
                    f"{equality['bytes']} bytes; audit inputs matched={audit['source_manifest_verified']}.", ""]
    for summary in result.get("owner_summaries", {}).values():
        if summary.get("timing_note"):
            out += [summary["timing_note"], ""]
    if result.get("helper_components"):
        out += ["## Synthetic helper components — NOT full snapshot/restore", "",
                "Pinned DMA excludes copied ndarray extraction and shared-file publication; ordinary collector includes ndarray acquisition. "
                "Weak sum/XOR is noncryptographic and position-insensitive. SHA256 of its output vector is provenance, not source SHA256. "
                "No extrapolation from synthetic arrays to live model snapshots or physical bandwidth ceilings.", ""]
        rows = [(r["component"], r["repetition"], r["bytes"], f"{r['elapsed_s']:.6f}" if r["elapsed_s"] is not None else "—",
                 r.get("kernel_ready_s", "—"), r.get("readback_s", "—"), r["GBps"], r["timing_boundary"], r["verdict"])
                for r in result["helper_components"]]
        out += [md_table(["component", "repeat", "bytes", "host-consumed s (DMA: ready-only)", "kernel-ready incl enqueue s",
                          "readback s", "GB/s", "timing boundary", "component verdict"], rows), ""]
    for bench in result["bench"]:
        out += [f"## FastSnapshot component bandwidth — `{Path(bench['file']).name}` "
                f"(submission {bench['submission_id']}, measurements.ok={bench['ok']}, failed_ops={bench['failed_ops']})", ""]
        rows = [(r["op"], r["host"], r["payload_bytes"], fmt(r["elapsed_s"], 3), fmt(r["gbps"]), r["verdict"])
                for r in bench["rows"]]
        out += [md_table(["op", "host", "bytes", "elapsed s (median, min–max)", "GB/s", "gate verdict"], rows), ""]
    if result["snapshot_writes"]:
        out += ["## Snapshot write time per worker", ""]
        for run in result["snapshot_writes"]:
            out += [f"### `{Path(run['file']).name}` ({run['label']}, submission {run['submission_id']})", ""]
            rows = [(r["engine"], r["process_rank"], r["bytes"], f"{r['snapshot_s']:.2f}" if r["snapshot_s"] else "—",
                     f"{r['GBps']:.2f}" if r["GBps"] else "—",
                     ", ".join(f"{k}={v:.2f}" for k, v in r["parts"].items() if v is not None),
                     f"{r['baseline_s']:.1f}" if r["baseline_s"] else "—",
                     "possibly warm" if r["source_possibly_warm"] else "cold (Python cache)",
                     r["scope"], r["checksum_kind"], r["cold_snapshot_proof"], r["verdict"]) for r in run["rows"]]
            out += [md_table(["engine", "rank", "bytes", "snapshot_s", "GB/s", "parts (max per-chip active; overlapping)",
                              "baseline_s", "source", "scope", "checksum kind", "cold snapshot proof", "gate verdict"], rows), ""]
            acc = [r for r in run["rows"] if r["verdict"] == "accepted"]
            out += [f"Accepted workers: {len(acc)}/{len(run['rows'])}; snapshot_s {fmt(stats([r['snapshot_s'] for r in acc]))}; "
                    + "; ".join(f"{k} {fmt(stats([r['parts'][k] for r in acc]))}" for k in WRITE_PARTS), ""]
    if result["restores"]:
        out += ["## Restore / start-up time per worker", ""]
        for run in result["restores"]:
            out += [f"### `{Path(run['file']).name}` (submission {run['submission_id']})"
                    + (f" greedy_equality={run['greedy_equality']}" if "greedy_equality" in run else "")
                    + (f" weight_equality={run['weight_equality']}" if "weight_equality" in run else ""), ""]
            rows = [(r["engine"], r["executor_rank"],
                     ", ".join(f"{k}={v:.1f}" for k, v in r["phases_s"].items()),
                     f"{r['unattributed_load_s']:.1f}" if r["unattributed_load_s"] is not None else "—",
                     r["weight_snapshot"].get("restored"), r["checksum_verified_in_path"],
                     r["fingerprint_verified_outside_timer"], r["verdict"]) for r in run["rows"]]
            out += [md_table(["engine", "rank", "phases s", "load_model minus snapshot_* s", "restored",
                              "in-path checksum", "outside-timer fingerprint", "gate verdict"], rows), ""]
            acc = [r for r in run["rows"] if r["verdict"] == "accepted"]
            out += ["Accepted workers " + f"{len(acc)}/{len(run['rows'])}: " + "; ".join(
                f"{k} {fmt(stats([r['phases_s'].get(k) for r in acc]), 1)}" for k in RESTORE_PHASES
                if any(k in r["phases_s"] for r in acc)), ""]
    if result["warmups"]:
        out += ["## Warm-up / KV split and cache hits (OverlapCompile startup profiles)", ""]
        out += ["Accepted startup rows qualify host-call/configuration observations only, not loaded-runtime/source equivalence or device attribution. "
                "Unknown legacy counts remain unknown, not zero.", ""]
        if result.get("startup_projection"):
            projection = result["startup_projection"]
            out += [f"Compact projection: `{projection['file']}`; source hashes matched={projection['source_manifest_verified']}.", ""]
        if result.get("unprojected_startup_artifacts"):
            out += ["Unprojected archived startup artifacts (not accepted or used; raw paths retained): "
                    + "; ".join(result["unprojected_startup_artifacts"]), ""]
        for run in result["warmups"]:
            out += [f"### `{run['file']}` (submission {run['submission_id']})", ""]
            reasons = sorted({reason for r in run["rows"] for reason in r["evidence_qualification"]["exclusion_reasons"]})
            if reasons:
                out += ["Exclusions: " + "; ".join(reasons), ""]
            rows = [(r["engine"], r["executor_rank"], r["phase"], f"{r['wall_s']:.1f}" if r["wall_s"] else "—",
                     ", ".join(f"{k}={v:.1f}" for k, v in (r["exclusive_wall_s"] or {}).items())
                     if isinstance(r["exclusive_wall_s"], dict) else "—",
                     f"{r['cache_hits']}/{r['cache_misses']}", r["executables"],
                     f"{r['device_module_execution_wall_s']:.1f}" if r["device_module_execution_wall_s"] else "—",
                     r["cache_claim"], r["evidence_qualification"]["qualification"], r["verdict"]) for r in run["rows"]]
            out += [md_table(["engine", "rank", "phase", "wall s", "exclusive split s", "hits/misses (hook)",
                              "executables", "device exec s", "cache", "qualification", "gate verdict"], rows), ""]
    if result["serving_impact"]:
        out += ["## Serving impact of overlapped compile (control vs treated pilot)", ""]
        out += [md_table(["run", "control", "treated", "control tok/s", "treated tok/s", "slowdown %", "gate verdict"],
                         [(r["run"], r["control"], r["treated"], f"{r['control_tps']:.1f}", f"{r['treated_tps']:.1f}",
                           f"{r['slowdown_pct']:.2f}" if r["slowdown_pct"] is not None else "—", r["verdict"])
                          for r in result["serving_impact"]]), ""]
    out += ["## Switch cost C = first dispatch − last old completion + ramp deficit", ""]
    if result["switch_costs"]:
        out += [md_table(["source file", "from", "to", "treatment", "dispatch gap s", "ramp deficit s", "C s",
                          "observation verdict", "performance eligible", "scope/verdict"],
                         [(r["file"], r["from"], r["to"], r["treatment"],
                           f"{r['dispatch_gap_s']:.6f}" if r["dispatch_gap_s"] is not None else "—",
                           f"{r['ramp_deficit_s']:.6f}" if r["ramp_deficit_s"] is not None else "—",
                           f"{r['cost_s']:.6f}" if r["cost_s"] is not None else "—",
                           r["observation_verdict"], r["performance_eligible"], r["verdict"])
                          for r in result["switch_costs"]]), ""]
        for r in result["switch_costs"]:
            excluded = [gate for gate, passed in r["performance_gates"].items() if not passed]
            out += [f"{r['file']}: {r['scope']}; comparison exclusions: " + ", ".join(excluded), ""]
            if r.get("provisioning"):
                out += [f"Prior target provisioning={r['provisioning'].get('charged_s')} s; "
                        f"C plus provisioning={r.get('first_use_including_provisioning_s')} s. "
                        + r["provisioning"].get("scope", ""), ""]
            out += list(r.get("limitations", [])) + [""]
    else:
        out += ["No run provides C yet (no cost-*.json, overlap_switch.json or campaign.json with client artifacts).", ""]
    out += ["## Direction-matched break-even (reconfig_breakeven.py formula, measured throughput table)", ""]
    out += ["All numeric crossovers are rate-model estimates, not observed crossover experiments. "
            "Unknown or mismatched actual catalogue/manifest and missing source/control proof suppress the number. "
            "Compatible but unfrozen evidence is labeled ESTIMATE ONLY, never headline evidence.", ""]
    if result["breakeven"]:
        out += [md_table(["workload", "from→to", "treatment", "C s", "gain/phase s", "N* tokens", "× phase", "verdict"],
                         [(r["workload"], f"{r['from']}→{r['to']}", r["treatment"], f"{r['cost_s']:.1f}",
                           f"{r['gain_per_phase_s']:.1f}" if r["gain_per_phase_s"] is not None else "—",
                           "ineligible" if not r["model_compatible"] else "never pays" if r["breakeven_tokens"] is None
                           else f"{r['breakeven_tokens'] / 1e6:.2f}M",
                           "—" if r["breakeven_phase_multiple"] is None else f"{r['breakeven_phase_multiple']:.2f}",
                           r["verdict"]) for r in result["breakeven"]]), ""]
    else:
        out += ["No accepted direction-matched C available; nothing to break even. "
                "(A→B costs are never borrowed for B→A or for a third direction.)", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", type=Path, default=FAST)
    ap.add_argument("--overlap", type=Path, default=OVERLAP)
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--table", type=Path, default=TABLE)
    ap.add_argument("--out", type=Path, default=ROOT / "results/helper/ResultsFable/current_tables.md")
    ap.add_argument("--json", type=Path, default=ROOT / "results/helper/ResultsFable/current_tables.json")
    ap.add_argument("--helper", type=Path, default=HELPER)
    ap.add_argument("--audit", type=Path, default=AUDIT)
    args = ap.parse_args(argv)
    result = collect(args.fast, args.overlap, args.baseline, args.table, args.helper, args.audit)
    text = render(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    args.json.write_text(json.dumps(result, indent=1, default=str) + "\n")
    observed = sum(r["observation_verdict"] == "accepted" for r in result["switch_costs"])
    eligible = sum(r["performance_eligible"] for r in result["switch_costs"])
    print(f"wrote {args.out} ({len(text.splitlines())} lines); observed C rows: {observed}; performance-eligible: {eligible}")


if __name__ == "__main__":
    main()
