#!/usr/bin/env python3
"""O2/O3 overlap checks per run-switch rep, recomputed offline from the fetched artifacts.

For each ``results/hetero/artifacts/<run_id>/`` it reports and checks:
  * no second TPU owner: every pre-spawned worker parked before its release, both /proc self-checks
    clean, and the device handoff (release -> lock free -> /proc free -> groups openable -> devices);
  * bank consumption: every target worker's unanimity verdict ``use``, entries loaded, AOT buckets
    skipped, bank load seconds, compile_or_warm_up_model split (trace/lower must be ~0);
  * source impact: phase-1 delivery during / after the pre-spawn window (tok/s) and phase-1 length.
Serving-time bank counters (hits/misses/rejections) are read from the post-switch validation
reports given with ``--validation`` (full: ``switch-r<r>-H-validation.json``; partial:
``Mb-restored-validation-r<r>.json``).

Usage: overlap_check.py RUN_ID [RUN_ID ...] [--validation FILE ...] [--json OUT]
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import statistics
from typing import Any, Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "results" / "hetero" / "artifacts"


def _load(path: pathlib.Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def _result(body: Dict[str, Any]) -> Dict[str, Any]:
    body = body.get("measurements", body)
    return body.get("result", body)


def _span(phases: Dict[str, Any], name: str):
    v = phases.get(name)
    if isinstance(v, dict) and "start" in v and "end" in v:
        return round(v["end"] - v["start"], 3)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return round(v[1] - v[0], 3)
    return None


def check_run(run_id: str) -> Dict[str, Any]:
    art = ARTIFACTS / run_id
    report = _result(_load(art / "switch_report.json"))
    deploy = _load(art / "deploy_report.json.gz")
    timeline = report.get("timeline") or {}
    pre = timeline.get("prespawn") or {}
    workers_pre = pre.get("workers") or []
    target = []
    for inst in deploy.get("instances", []):
        iid = (inst.get("descriptor") or {}).get("instance_id") or inst.get("instance_id")
        for w in (inst.get("initialization") or {}).get("worker_init_phases", []):
            bank = w.get("exec_bank") or {}
            if bank.get("mode") is None and "prespawn_gate" not in w:
                continue  # an adopted (kept) engine of a partial switch
            profile = (w.get("startup_profiles") or {}).get("compile_or_warm_up_model") or {}
            split = profile.get("exclusive_wall_s") or {}
            phases = w.get("phases") or {}
            target.append({
                "instance": iid, "rank": bank.get("process_rank"),
                "bank_use": (bank.get("unanimity") or {}).get("use"),
                "bank_entries": (bank.get("loaded") or {}).get("entries"),
                "bank_load_s": (bank.get("loaded") or {}).get("load_s"),
                "aot": bank.get("run_compilation"),
                "init_device_s": _span(phases, "init_device"), "load_model_s": _span(phases, "load_model"),
                "initialize_from_config_s": _span(phases, "initialize_from_config"),
                "exec_bank_load_s": _span(phases, "exec_bank_load"),
                "compile_or_warm_up_s": _span(phases, "compile_or_warm_up_model"),
                "cwu_trace_s": split.get("trace", 0.0), "cwu_lower_s": split.get("lower", 0.0),
                "cwu_backend_compile_s": split.get("backend_compile_and_load", 0.0)})
    window = (report.get("prespawn") or {}).get("window") or {}
    stamps = report.get("stamps") or {}
    handoff = [w["devices_ready_at"] - w["released_at"] for w in workers_pre
               if w.get("devices_ready_at") is not None and w.get("released_at") is not None]
    checks = {
        "completed": report.get("completed") is True,
        "all_restored": (report.get("restored") or {}).get("all_restored") is True,
        "segments_reconcile": timeline.get("segments_reconcile") is True,
        "parked_before_T0": (report.get("prespawn") or {}).get("parked_before_T0") is True,
        "all_parked_before_release": pre.get("all_parked_before_release") is True,
        "all_self_checks_clean": pre.get("all_self_checks_clean") is True,
        "bank_used_everywhere": bool(target) and all(t["bank_use"] is True for t in target),
        "no_backend_compile": all((t["cwu_backend_compile_s"] or 0.0) < 0.05 for t in target),
    }
    return {"run_id": run_id, "checks": checks, "ok": all(checks.values()),
            "gap_dispatch_s": timeline.get("gap_dispatch_s"),
            "endpoints": report.get("endpoints") or timeline.get("endpoints"),
            "segments": timeline.get("segments"),
            "phase1_s": round(stamps["T0"] - stamps["phase1_begin"], 3) if "T0" in stamps and "phase1_begin" in stamps else None,
            "source_window": {k: window.get(k) for k in ("during", "after")},
            "handoff_release_to_devices_s": {"median": statistics.median(handoff), "max": max(handoff)} if handoff else None,
            "busy_group_hosts": sorted(w["host"] for w in workers_pre if (w.get("holders_seen") or {}).get("busy_groups")),
            "prespawn_workers": workers_pre, "target_workers": target}


def serving_counters(path: pathlib.Path) -> Dict[str, List[int]]:
    body = _load(path)["measurements"]
    totals: Dict[str, List[int]] = {}
    phases = body["worker_init_phases"]
    for workers in (phases.values() if isinstance(phases, dict) else [phases]):
        for w in workers:
            for fn, f in ((w.get("exec_bank") or {}).get("functions") or {}).items():
                t = totals.setdefault(fn, [0, 0, 0])
                t[0] += f.get("hits", 0)
                t[1] += f.get("misses", 0)
                t[2] += f.get("rejections", 0)
    return {"greedy_equality": (body.get("greedy_equality") or {}).get("passed"), "functions": totals,
            "misses": sum(t[1] for t in totals.values()), "rejections": sum(t[2] for t in totals.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--validation", nargs="*", type=pathlib.Path, default=[])
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()
    out = {"runs": [check_run(r) for r in args.runs],
           "validations": {str(p): serving_counters(p) for p in args.validation}}
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str))
    for run in out["runs"]:
        t = run["target_workers"]
        med = lambda k: statistics.median(x[k] for x in t if x[k] is not None) if t else None  # noqa: E731
        print(f"{run['run_id']}: ok={run['ok']} gap={run['gap_dispatch_s']} phase1={run['phase1_s']} "
              f"handoff={run['handoff_release_to_devices_s']} busy_hosts={run['busy_group_hosts']} "
              f"bank_load_s~{med('bank_load_s')} cwu_s~{med('compile_or_warm_up_s')} "
              f"source tok/s during={(run['source_window'].get('during') or {}).get('tokens_per_s')} "
              f"after={(run['source_window'].get('after') or {}).get('tokens_per_s')} failed={[k for k, v in run['checks'].items() if not v]}")
    for path, v in out["validations"].items():
        print(f"{path}: greedy={v['greedy_equality']} misses={v['misses']} rejections={v['rejections']}")


if __name__ == "__main__":
    main()
