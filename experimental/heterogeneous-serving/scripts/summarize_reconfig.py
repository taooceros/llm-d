#!/usr/bin/env python3
"""End-to-end results of multi-phase reconfiguration jobs.

For every run directory given (results/hetero/artifacts/<run_id>), reads the per-request records
of each run in it and reports:
  job wall time   first dispatch of phase 1 -> last finished request of the last phase
  per phase       first dispatch, last finish and duration of each phase's requests
  transition      last finish of phase k -> first dispatch of phase k+1 (0 for one backlog)
Static arms run one composite manifest; their phases come from the request-id prefix
"<COMPOSITE>-p<phase>r<repeat>-".  Reconfigured arms run one manifest per phase in separate runs
(full redeploy: two mix-<tag> runs; partial: one run_id with both phases plus the transition
report, whose timestamps are added).

Usage: summarize_reconfig.py <run_dir>... [--json OUT]
"""
import argparse
import glob
import gzip
import json
import os
import re

PHASE = re.compile(r"-p(\d+)r\d+-")


def runs_in(run_dir):
    """(label, records) for each run inside a run directory, ordered by first dispatch."""
    out = []
    for path in glob.glob(os.path.join(run_dir, "*_records.json.gz")):
        if "gateway_records" in path:
            continue
        label = os.path.basename(path)[: -len("_records.json.gz")]
        out.append((label, json.load(gzip.open(path))))
    return sorted(out, key=lambda lr: min(r["client_started_at_s"] for r in lr[1]))


def rid(r):
    return r.get("canonical_request_id") or r["request_id"]


def summarise(run_dirs):
    phases = []
    for run_dir in run_dirs:
        for label, records in runs_in(run_dir):
            groups = {}
            for r in records:
                m = PHASE.search(rid(r))
                groups.setdefault(int(m.group(1)) if m else 0, []).append(r)
            for index in sorted(groups):
                rs = groups[index]
                phases.append({"run": os.path.basename(run_dir), "label": label, "phase": index,
                               "requests": len(rs),
                               "output_tokens": sum(int(r.get("output_tokens") or r.get("useful_output_tokens") or 0) for r in rs),
                               "first_dispatch": min(r["client_started_at_s"] for r in rs),
                               "last_finish": max(r["client_finished_at_s"] for r in rs)})
    phases.sort(key=lambda p: p["first_dispatch"])
    start = phases[0]["first_dispatch"]
    rows = []
    for i, p in enumerate(phases):
        rows.append({**p, "start_s": round(p["first_dispatch"] - start, 1),
                     "end_s": round(p["last_finish"] - start, 1),
                     "duration_s": round(p["last_finish"] - p["first_dispatch"], 1),
                     "gap_before_s": round(p["first_dispatch"] - phases[i - 1]["last_finish"], 1) if i else 0.0})
    result = {"runs": [os.path.basename(d) for d in run_dirs],
              "job_wall_s": round(max(p["last_finish"] for p in phases) - start, 1),
              "phases": rows}
    for run_dir in run_dirs:
        for path in glob.glob(os.path.join(run_dir, "transition_*.json")):
            t = json.load(open(path))
            result["transition"] = {"timeline": t.get("timeline"),
                                    "join_partition": {k: v for k, v in (t.get("join_partition") or {}).items()
                                                       if isinstance(v, dict)}}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="one job = space-separated dirs joined by ','")
    ap.add_argument("--json")
    args = ap.parse_args()
    jobs = [summarise(spec.split(",")) for spec in args.run_dirs]
    for job in jobs:
        print(f"{'+'.join(job['runs'])}: job wall {job['job_wall_s']} s")
        for p in job["phases"]:
            print(f"   {p['label']:16} phase {p['phase']} req={p['requests']:5} "
                  f"start={p['start_s']:7.1f} end={p['end_s']:7.1f} dur={p['duration_s']:7.1f} "
                  f"gap_before={p['gap_before_s']:6.1f}")
    if args.json:
        json.dump(jobs, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
