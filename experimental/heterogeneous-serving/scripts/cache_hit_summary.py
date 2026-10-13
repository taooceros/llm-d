#!/usr/bin/env python3
"""Persistent-cache hits/misses and effective thresholds per worker and startup phase.

Reads deploy reports (``hetero_ctl deploy`` outputs or run-switch ``deploy_report.json.gz``; plain or gzip)
and counts, from each worker's startup profile, the persistent-cache lookups that hit (``cache_load``) or
missed (``cache_miss_lookup``), together with the phase's exclusive-wall split (trace, lower,
backend_compile_and_load, executable_load, ...).  The effective thresholds are read twice: from the
worker's ``compile_cache`` report (``omp_init_phases``) and from each startup profile.  A worker is
``zero_threshold`` only if every one of those reads is 0/0.

Usage: cache_hit_summary.py REPORT [REPORT ...] [--json OUT]
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import statistics
from typing import Any, Dict, Iterable, List

PHASES = ("initialize_from_config", "compile_or_warm_up_model")
SPLIT = ("trace", "lower", "backend_compile_and_load", "cache_lookup_and_decompress", "executable_load",
         "warmup_execute_and_sync")


def load(path: pathlib.Path) -> Dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        body = json.load(f)
    return body.get("measurements", {}).get("deployment", body)


def workers(deployment: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for instance in deployment.get("instances", []):
        instance_id = (instance.get("descriptor") or {}).get("instance_id") or instance.get("instance_id")
        for worker in (instance.get("initialization") or {}).get("worker_init_phases", []):
            yield instance_id, worker


def worker_row(instance_id: str, worker: Dict[str, Any]) -> Dict[str, Any]:
    cache = worker.get("compile_cache") or {}
    thresholds = [(cache.get("min_compile_time_secs"), cache.get("min_entry_size_bytes"))]
    row: Dict[str, Any] = {"instance_id": instance_id, "rank": worker.get("executor_rank"),
                           "host": cache.get("hostname"), "cache_dir": cache.get("dir"), "phases": {}}
    for phase in PHASES:
        profile = (worker.get("startup_profiles") or {}).get(phase)
        if profile is None:
            row["phases"][phase] = None
            continue
        thresholds.append((profile.get("persistent_cache_min_compile_time_secs"),
                           profile.get("persistent_cache_min_entry_size_bytes")))
        lookups = [r for r in profile.get("intervals", []) if "hit" in r]
        hits = sum(1 for r in lookups if r["hit"])
        split = profile.get("exclusive_wall_s") or {}
        row["phases"][phase] = {"lookups": len(lookups), "hits": hits, "misses": len(lookups) - hits,
                                "wall_s": profile.get("wall_s"),
                                **{f"{k}_s": split.get(k, 0.0) for k in SPLIT}}
    row["thresholds"] = sorted(set(thresholds), key=str)
    row["zero_threshold"] = all(t == (0, 0) for t in thresholds)
    return row


def summarize(paths: List[pathlib.Path]) -> Dict[str, Any]:
    rows = [dict(worker_row(i, w), report=str(p)) for p in paths for i, w in workers(load(p))]
    phases = {}
    for phase in PHASES:
        present = [r["phases"][phase] for r in rows if r["phases"].get(phase)]
        lookups = sum(p["lookups"] for p in present)
        hits = sum(p["hits"] for p in present)
        phases[phase] = {"workers": len(present), "lookups": lookups, "hits": hits,
                         "hit_rate": hits / lookups if lookups else None,
                         "min_worker_hit_rate": min((p["hits"] / p["lookups"] for p in present if p["lookups"]),
                                                    default=None),
                         **{f"median_{k}_s": statistics.median(p[f"{k}_s"] for p in present) if present else None
                            for k in ("wall",) + SPLIT}}
    return {"workers": rows, "phases": phases, "all_zero_threshold": bool(rows) and all(
        r["zero_threshold"] for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()
    summary = summarize(args.reports)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1))
    print(f"workers {len(summary['workers'])}  all 0/0 thresholds: {summary['all_zero_threshold']}")
    for phase, p in summary["phases"].items():
        rate = "n/a" if p["hit_rate"] is None else f"{p['hit_rate']:.1%}"
        print(f"{phase}: hits {p['hits']}/{p['lookups']} ({rate}; min worker "
              f"{p['min_worker_hit_rate']}) median wall {p['median_wall_s']}  "
              + " ".join(f"{k} {p[f'median_{k}_s']:.2f}" for k in SPLIT if p[f"median_{k}_s"] is not None))


if __name__ == "__main__":
    main()
