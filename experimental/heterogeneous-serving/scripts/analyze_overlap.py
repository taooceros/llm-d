#!/usr/bin/env python3
"""Consolidate retained overlap evidence; never impute missing measurements."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from reconfig_timeline import rate_series
from reconfig_breakeven import breakeven


def load(path):
    return json.loads(path.read_text())


def worker_rows(path, phase):
    evidence = load(path)
    measured = evidence.get("measurements") or {}
    deployment = measured.get("deployment") or {}
    exclusions = []
    if not measured.get("ok"):
        exclusions.append("deployment did not complete successfully")
    workers = []
    identities, hosts = set(), set()
    for instance in deployment.get('instances', []):
        name = instance['descriptor']['instance_id']
        initialization = instance["initialization"]
        settings = initialization.get("engine_settings", {})
        resolved = initialization.get("resolved_engine_config", {})
        for key, expected in (("max_model_len", 16384), ("max_num_seqs", 512),
                              ("max_num_batched_tokens", 2048), ("enable_prefix_caching", False)):
            if resolved.get(key) != expected:
                exclusions.append(f"{name}: effective {key} is not {expected}")
        for key in ("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
                    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
                    "JAX_LOG_COMPILES", "JAX_EXPLAIN_CACHE_MISSES"):
            if settings.get("worker_env", {}).get(key) != "0":
                exclusions.append(f"{name}: coverage-matched worker setting {key}=0 was not requested")
        for worker in initialization.get("worker_init_phases", []):
            profile = worker.get('startup_profiles', {}).get(phase)
            if profile is None:
                continue
            identity = (name, worker["executor_rank"])
            if identity in identities:
                exclusions.append(f"{name}: duplicated worker rank {worker['executor_rank']}")
            identities.add(identity)
            if not profile.get("host") or profile["host"] in hosts:
                exclusions.append(f"{name}: missing or duplicated worker host")
            hosts.add(profile.get("host"))
            effective = worker.get("compile_cache", {})
            if (profile.get("compilation_cache_dir") != settings.get("compilation_cache_dir")
                    or effective.get("dir") != settings.get("compilation_cache_dir")):
                exclusions.append(f"{name}: effective cache differs from requested cache")
            for key in ("persistent_cache_min_compile_time_secs", "persistent_cache_min_entry_size_bytes"):
                if profile.get(key) != 0:
                    exclusions.append(f"{name}: phase {key} is not coverage-matched zero")
            for key, expected in (("min_compile_time_secs", 0), ("min_entry_size_bytes", 0),
                                  ("log_compiles", False), ("explain_cache_misses", False)):
                if effective.get(key) != expected:
                    exclusions.append(f"{name}: effective cache {key} is not {expected}")
            if settings.get("startup_device_trace"):
                if profile.get("trace_options") != {"python_tracer_level": 0, "host_tracer_level": 2}:
                    exclusions.append(f"{name}: uncontrolled profiler settings")
                traces = profile.get("device_trace", {}).get("files", [])
                if not traces or any(not row.get("marker_window_us") or not row.get("device_lanes") for row in traces):
                    exclusions.append(f"{name}: missing phase annotation or TPU lanes")
            cache = [r for r in profile["intervals"] if "hit" in r]
            costs = []
            for executable in profile["executables"]:
                rows = [r for r in profile["intervals"]
                        if (r.get("executable") or {}).get("index") == executable["index"]]
                costs.append({**executable, "intervals": rows,
                              "inclusive_wall_s_by_kind": {kind: sum(r["end"]-r["start"] for r in rows if r["kind"]==kind)
                                                           for kind in {r["kind"] for r in rows}}})
            workers.append({'instance': name, 'executor_rank': worker['executor_rank'],
                            'cache_hits': sum(r['hit'] for r in cache),
                            'cache_misses': sum(not r['hit'] for r in cache),
                            "executable_count": len(profile["executables"]),
                            "per_executable": costs, "profile": profile,
                            "compile_cache": effective})
    if len(workers) != 8:
        exclusions.append(f"expected 8 worker profiles, observed {len(workers)}")
    return {"source": str(path.relative_to(ROOT)), "run_id": evidence.get("submission_id"),
            "qualification": "controlled" if not exclusions else "diagnostic_only",
            "exclusion_reasons": sorted(set(exclusions)), "workers": workers}


def pilot(run_id):
    directory = ROOT / 'results/hetero/artifacts' / run_id
    records_path = next(p for p in directory.glob('*_records.json.gz') if 'gateway' not in p.name)
    records = json.load(gzip.open(records_path))
    csv_path = records_path.with_name(records_path.name.replace('_records.json.gz', '_iterations.csv.gz'))
    iterations = list(csv.DictReader(gzip.open(csv_path, 'rt')))
    summary = load(records_path.with_name(records_path.name.replace("_records.json.gz", "_summary.json")))
    first = min(r['client_started_at_s'] for r in records)
    last = max(r['client_finished_at_s'] for r in records)
    decode = [row for row in iterations if row.get("phase") == "decode"]
    decode_tokens = sum(int(float(row.get("emitted_output_tokens") or 0)) for row in decode)
    series = rate_series(iterations, first)
    n = len(series)
    middle = series[int(.2*n):max(int(.8*n), int(.2*n)+1)]
    steady = statistics.median(middle)
    deficit = sum(max(steady-x, 0) for x in series[:12]) * 10 / steady if steady else None
    latencies = sorted(float(row["scheduled_to_output_s"]) for row in iterations
                       if row.get("scheduled_to_output_s") and row.get("phase") == "decode")
    return {'run_id': run_id, 'source': str(directory.relative_to(ROOT)), 'first_dispatch': first,
            'last_finished_request': last, 'makespan_s': summary['makespan_s'],
            'useful_tokens_per_s': summary['useful_tokens_per_s'],
            "decode_output_tokens": decode_tokens,
            "decode_tokens_per_s": decode_tokens / summary["makespan_s"],
            'steady_tokens_per_s': steady, 'ramp_deficit_s': deficit,
            'step_latency_p50_s': statistics.median(latencies) if latencies else None,
            'step_latency_p95_s': latencies[int(.95 * (len(latencies)-1))] if latencies else None,
            'all_requests_complete': summary['all_requests_complete'],
            'policy': summary['policy'], 'during_run': summary.get('during_run'),
            "manifest_sha256": summary["manifest_sha256"], "num_requests": summary["num_requests"],
            "useful_output_tokens": summary["useful_output_tokens"],
            "replayed_tokens": summary["replayed_tokens"]}


def deployment_provenance(path):
    evidence = load(path)
    instances = evidence["measurements"]["deployment"]["instances"]
    return {"source": str(path.relative_to(ROOT)), "submission_id": evidence["submission_id"],
            "instances": {row["descriptor"]["instance_id"]: {
                "requested": row["initialization"]["engine_settings"],
                "resolved": row["initialization"]["resolved_engine_config"]} for row in instances}}


def serving_signature(provenance):
    fields = ("model", "dtype", "kv_cache_dtype", "activation_dtype", "matmul_precision",
              "backend_revision", "model_build", "max_model_len", "max_num_seqs",
              "max_num_batched_tokens", "block_size", "gpu_memory_utilization", "enable_prefix_caching")
    return {json.dumps({key: row["resolved"][key] for key in fields}, sort_keys=True)
            for row in provenance["instances"].values()}


def routing_signature(policy):
    # Layout membership is the treatment, not a routing-knob difference.
    return {key: value for key, value in policy.items()
            if key not in ("small_instances", "large_instances")}


def provisioning_cost(directory):
    events = [json.loads(line) for line in (directory / "timeline.jsonl").read_text().splitlines()]
    def stamp(name):
        rows = [row["t"] for row in events if row["event"] == name]
        if len(rows) != 1:
            raise ValueError(f"missing/duplicated provisioning endpoint {name} in {directory}")
        return rows[0]
    start, ready, released = (stamp(name) for name in ("deploy_start:target-provision",
                             "deploy_end:target-provision", "released:target-provision"))
    if not start <= ready <= released:
        raise ValueError(f"invalid provisioning interval in {directory}")
    return {"start": start, "end": released, "charged_s": released - start,
            "normal_deploy_s": ready - start, "reference_snapshot_release_s": released - ready,
            "scope": "joint prior target graph/bundle, reference generation, weight snapshot and release; not pure compile"}


def cost_ledger(cost_s, source, control, provisioning):
    during = source.get("during_run") or {}
    result = during.get("result") or {}
    delta = source["makespan_s"] - control["makespan_s"]
    penalty = max(delta, 0)
    return {"raw_switch_C_s": cost_s, "source_serving_delta_s": delta,
            "source_serving_charge_s": penalty, "source_control_run_id": control["run_id"],
            "precompile_task_wall_s": (result["end"] - result["start"]) if "end" in result else None,
            "precompile_process_cpu_s": result.get("process_cpu_s"),
            "late_compile_after_source_s": max(result["end"] - source["last_finished_request"], 0) if "end" in result else None,
            "prior_target_provisioning": provisioning,
            "economic_cost_s": {"reused_provisioning": cost_s + penalty,
                                "first_use": cost_s + penalty + provisioning["charged_s"]},
            "accounting": "Precompile wall/CPU is retained, not added twice: hidden work overlaps source service; late join is already in C. Positive matched source slowdown is charged; a negative noisy delta is not credited."}


def precompile_cache_proof(source, deployment_path):
    during = source.get("during_run") or {}
    result = during.get("result") or {}
    compiled = [row for bundle in result.get("bundles", []) for row in bundle.get("executables", [])]
    matched = {row["actual_key"]: row for row in compiled
               if row.get("compiled") and row.get("key_matches")}
    workers = []
    for phase in ("initialize_from_config", "compile_or_warm_up_model"):
        for worker in worker_rows(deployment_path, phase)["workers"]:
            profile = worker["profile"]
            reads = [row for row in profile["intervals"] if "hit" in row]
            hits = [row for row in reads if row["hit"] and row["cache_key"] in matched
                    and matched[row["cache_key"]]["end"] <= row["start"]]
            loads = [row for row in profile["intervals"] if row["kind"] == "executable_load"
                     and any((row.get("executable") or {}).get("cache_key") == hit["cache_key"]
                             and hit["start"] <= row["start"] <= row["end"] <= hit["end"] for hit in hits)]
            listener_hits = [event for event in profile["monitoring_events"]
                             if event["name"].endswith("/cache_hits")]
            workers.append({"instance": worker["instance"], "executor_rank": worker["executor_rank"],
                            "phase": phase, "effective_cache_dir": profile.get("compilation_cache_dir"),
                            "completed_before_phase": result.get("end") is not None
                                and result["end"] <= profile["start"],
                            "precompiled_key_hits": hits, "all_keyed_reads": reads,
                            "successful_keyed_deserializations": loads,
                            "cache_hit_listener_events": listener_hits,
                            "monitoring_events": profile["monitoring_events"]})
    return {"background_error": during.get("error"), "background_ok": result.get("ok", False),
            "background_start": result.get("start"), "background_end": result.get("end"),
            "completed_while_source_served": result.get("end") is not None
                and result["end"] <= source["last_finished_request"],
            "actual_key_match_count": len(matched), "workers": workers,
            "limitation": "Key match alone is not savings: require target deserialize success/listener hit and measured full switch C."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT/'results/overlap/OverlapCompile')
    ap.add_argument("--campaign", type=Path, action="append", default=[])
    ap.add_argument("--check-deploy", type=Path, help="Check effective controls without rewriting reports")
    args = ap.parse_args()
    if args.check_deploy:
        rows = {phase: worker_rows(args.check_deploy.resolve(), phase)
                for phase in ("initialize_from_config", "compile_or_warm_up_model")}
        verdict = {phase: {key: row[key] for key in ("run_id", "qualification", "exclusion_reasons")}
                   for phase, row in rows.items()}
        print(json.dumps(verdict, indent=2))
        raise SystemExit(int(any(row["qualification"] != "controlled" for row in rows.values())))
    args.root = args.root.resolve()
    args.campaign = [path.resolve() for path in args.campaign]
    warm, kv = [], []
    for path in sorted(args.root.glob('*/*deploy.json')):
        if path.name.startswith('cleanup'):
            continue
        w = worker_rows(path, 'compile_or_warm_up_model')
        k = worker_rows(path, 'initialize_from_config')
        if w['workers']:
            warm.append(w)
        if k['workers']:
            kv.append(k)
    for name, rows in (('warmup_split.json', warm), ('kv_split.json', kv)):
        (args.root/name).write_text(json.dumps({'runs': rows}, indent=2)+'\n')
    if args.campaign:
        switches, all_pilots, break_even = [], {}, []
        for campaign_path in args.campaign:
            campaign = load(campaign_path)
            pilots = {name: pilot(run_id) for name, run_id in campaign["pilots"].items()}
            if not all(p["all_requests_complete"] for p in pilots.values()):
                raise ValueError(f"incomplete W0 requests in {campaign_path}")
            if not load(campaign_path.parent / "cleanup-tmpfs.json").get("measurements", {}).get("ok"):
                raise ValueError(f"normal cleanup not proven in {campaign_path}")
            workload_keys = ("manifest_sha256", "num_requests", "useful_output_tokens")
            if len({tuple(p[key] for key in workload_keys) for p in pilots.values()}) != 1:
                raise ValueError(f"unmatched W0 workload/output cohorts in {campaign_path}")
            provisioning = provisioning_cost(campaign_path.parent)
            provenance = {}
            for label, value in pilots.items():
                path = campaign_path.parent / campaign["deployments"][label]
                if path not in provenance:
                    provenance[path] = deployment_provenance(path)
                value["deployment"] = provenance[path]
            all_pilots.update({value["run_id"]: value for value in pilots.values()})
            for transition in campaign["transitions"]:
                source, target = (pilots[transition[k]] for k in ("source", "target"))
                gap = target["first_dispatch"] - source["last_finished_request"]
                if gap < 0 or target["ramp_deficit_s"] is None:
                    raise ValueError(f"invalid switch endpoints/ramp in {campaign_path}")
                deployment_path = campaign_path.parent / (transition["target"] + "-deploy.json")
                for phase in ("initialize_from_config", "compile_or_warm_up_model"):
                    if worker_rows(deployment_path, phase)["qualification"] != "controlled":
                        raise ValueError(f"uncontrolled target deployment {deployment_path}")
                for suffix in ("-check.json", "-validation.json", "-release.json"):
                    proof = load(campaign_path.parent / (transition["target"] + suffix))
                    if not (proof.get("measurements") or {}).get("ok"):
                        raise ValueError(f"target correctness/cleanup failed in {campaign_path}")
                validation_path = campaign_path.parent / (transition["target"] + "-validation.json")
                reference_path = campaign_path.parent / "target-reference.json"
                validation_evidence, reference_evidence = load(validation_path), load(reference_path)
                validated, reference = validation_evidence["measurements"], reference_evidence["measurements"]
                for key in ("layout", "manifest_sha256", "prompts", "sampling", "outputs"):
                    if key not in validated or validated[key] != reference[key]:
                        raise ValueError(f"target greedy output comparison failed: {key} in {campaign_path}")
                row = {**transition, "previous_run_id": source["run_id"], "target_run_id": target["run_id"],
                       "campaign": str(campaign_path.relative_to(ROOT)),
                       "last_finished_request": source["last_finished_request"],
                       "first_dispatch": target["first_dispatch"], "dispatch_gap_s": gap,
                       "ramp_deficit_s": target["ramp_deficit_s"],
                       "cost_s": gap + target["ramp_deficit_s"],
                       "output_equality": {
                           "reference_path": str(reference_path.relative_to(ROOT)),
                           "reference_submission_id": reference_evidence["submission_id"],
                           "validation_path": str(validation_path.relative_to(ROOT)),
                           "validation_submission_id": validation_evidence["submission_id"],
                           "manifest_sha256": reference["manifest_sha256"],
                           "per_instance": {name: {"equal": True, "prompts": len(outputs)}
                                            for name, outputs in validated["outputs"].items()}}}
                control = pilots["source-undisturbed"] if transition["treatment"] == "overlap" else source
                if source["policy"] != control["policy"]:
                    raise ValueError(f"unmatched source interference control in {campaign_path}")
                row["cost_ledger"] = cost_ledger(row["cost_s"], source, control, provisioning)
                if transition["treatment"] == "overlap":
                    row["cache_proof"] = precompile_cache_proof(source, deployment_path)
                signatures = serving_signature(control["deployment"]), serving_signature(target["deployment"])
                matched_runtime = len(signatures[0]) == 1 and signatures[0] == signatures[1]
                exclusions = []
                if not matched_runtime:
                    exclusions.append("runtime fingerprint or serving configuration differs")
                if routing_signature(control["policy"]) != routing_signature(target["policy"]):
                    exclusions.append("source/target routing knobs differ (layout bindings excluded)")
                if any(value["replayed_tokens"] for value in (control, target)):
                    exclusions.append("steady emitted-token rate includes replay")
                if not all(value["steady_tokens_per_s"] > 0 for value in (control, target)):
                    exclusions.append("steady-rate estimator is not positive")
                row["current_W0_rate_model_exclusions"] = exclusions
                row["matched_current_W0_rate_model_eligible"] = not exclusions
                if row["matched_current_W0_rate_model_eligible"]:
                    r_from, r_to = control["steady_tokens_per_s"], target["steady_tokens_per_s"]
                    for scope, cost in row["cost_ledger"]["economic_cost_s"].items():
                        n = breakeven(r_from, r_to, cost)
                        break_even.append({"workload": "W0", "from": transition["from"], "to": transition["to"],
                            "cost_s": cost, "cost_scope": scope, "raw_switch_C_s": row["cost_s"],
                            "r_from": r_from, "r_to": r_to, "breakeven_tokens": None if n is None else round(n),
                            "breakeven_phase_multiple": None if n is None else n / target["useful_output_tokens"],
                            "breakeven_requests": None if n is None else round(n / target["useful_output_tokens"] * target["num_requests"]),
                            "rate_evidence_scope": "matched current W0 steady-rate projection (median 10s bins in middle 20-80%)",
                            "runtime_equivalence_established": True, "crossover_observed": False,
                            "rate_control_run_id": control["run_id"], "previous_run_id": source["run_id"],
                            "target_run_id": target["run_id"], "treatment": transition["treatment"],
                            "campaign": str(campaign_path.relative_to(ROOT)),
                            "sources": [control["source"], target["source"]]})
                switches.append(row)
        (args.root / "overlap_switch.json").write_text(json.dumps({"transitions": switches}, indent=2) + "\n")
        (args.root / "serving_impact.json").write_text(json.dumps({"pilots": all_pilots}, indent=2) + "\n")
        # The existing command evaluates every direction for each C. Retain only
        # the direction actually measured, never extrapolate C to other switches.
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            output_path = Path(temp) / "breakeven.json"
            for switch in switches:
                for scope, cost in switch["cost_ledger"]["economic_cost_s"].items():
                    subprocess.run([sys.executable, str(ROOT / "scripts/reconfig_breakeven.py"),
                                    "--table", "/home/hongtao/hetero-research/results/reconfig/throughput_table_measured.json",
                                    "--cost", str(cost), "--json", str(output_path)], check=True,
                                   stdout=subprocess.DEVNULL)
                    break_even.extend({**entry, "treatment": switch["treatment"], "cost_scope": scope,
                                       "raw_switch_C_s": switch["cost_s"],
                                       "previous_run_id": switch["previous_run_id"], "target_run_id": switch["target_run_id"],
                                       "campaign": switch["campaign"],
                                       "rate_evidence_scope": "historical cross-revision rate model",
                                       "runtime_equivalence_established": False,
                                       "crossover_observed": False,
                                       "rate_table_source": "/home/hongtao/hetero-research/results/reconfig/throughput_table_measured.json"}
                                      for entry in load(output_path)
                                      if (entry["from"], entry["to"]) == (switch["from"], switch["to"]))
        (args.root / "breakeven.json").write_text(json.dumps(break_even, indent=2) + "\n")
    print(json.dumps({'warmup_runs': len(warm), 'kv_runs': len(kv)}))


if __name__ == '__main__':
    main()
