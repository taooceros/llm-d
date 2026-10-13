#!/usr/bin/env python3
"""Reconfiguration cost per transition from a reconfig_measure.sh campaign.

For every layout L deployed by the campaign this reports, on one wall clock:
  previous phase drained -> teardown (drain, gateway-down, release) -> deploy, split into
  backend/placement/actor/engine-init/serving stages and, inside engine init, the slowest
  worker's executor start-up, init_device, load_model, KV-cache init and compile/warm-up
  -> run job submitted -> first dispatch -> steady throughput.
"Steady throughput" is the median 10 s output-token rate over the middle 60% of the first run
after the deploy; the ramp deficit is the output the first 120 s would have produced at that rate
minus what it did produce, expressed in seconds of steady output.  Cost C for entering L is
  C = (first dispatch - previous phase drained) + ramp deficit.
The time from the previous drain is only meaningful when a previous layout ran in the campaign.

Usage: reconfig_timeline.py <commit> [--json OUT]
"""
import argparse
import collections
import csv
import glob
import gzip
import json
import statistics

PHASES = ("init_device", "load_model", "initialize_from_config", "compile_or_warm_up_model")


def first_run(commit, layout, slot=1):
    paths = glob.glob(f"results/hetero/artifacts/rc-{commit}-{layout}-s{slot}/*_records.json.gz")
    paths = [p for p in paths if "gateway" not in p]
    if not paths:
        return None
    records = json.load(gzip.open(paths[0]))
    rows = list(csv.DictReader(gzip.open(paths[0].replace("_records.json.gz", "_iterations.csv.gz"), "rt")))
    return records, rows


def rate_series(rows, start, bucket=10.0):
    acc = collections.Counter()
    for r in rows:
        acc[int((float(r["completed_at_s"]) - start) // bucket)] += int(r["emitted_output_tokens"] or 0)
    return [acc[i] / bucket for i in range(max(acc) + 1)] if acc else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("commit")
    ap.add_argument("--json")
    args = ap.parse_args()
    events = [json.loads(line) for line in open(f"results/reconfig/timeline-{args.commit}.jsonl")]
    out = []
    order = []
    for e in events:
        if e["event"] == "deploy_start" and e["layout"] not in order:
            order.append(e["layout"])
    for i, layout in enumerate(order):
        mine = {e["event"]: e["t"] for e in events if e["layout"] == layout}
        prev = order[i - 1] if i else None
        prev_drained = None
        if prev:
            # Drained = the previous layout's last request finished, not the run job's exit
            # (which also writes results); falls back to the run_end marker.
            slots = sorted(int(p.rsplit("-s", 1)[1]) for p in
                           glob.glob(f"results/hetero/artifacts/rc-{args.commit}-{prev}-s*"))
            last = first_run(args.commit, prev, slots[-1]) if slots else None
            if last:
                prev_drained = max(r["client_finished_at_s"] for r in last[0])
            else:
                prev_runs = [e["t"] for e in events if e["layout"] == prev and e["event"].startswith("run_end")]
                prev_drained = max(prev_runs) if prev_runs else None
        teardown = [e for e in events if e["layout"] == prev and e["event"] in
                    ("teardown_start", "drained", "gateway_down", "released")
                    and prev_drained and e["t"] >= prev_drained] if prev else []
        dep = json.load(open(f"results/reconfig/deploy-{args.commit}-{layout}.json"))["measurements"]
        clocks = dep["deployment"]["stage_clocks"]
        slowest = {}
        for inst in dep["deployment"]["instances"]:
            ini = inst["initialization"]
            t0 = ini["engine_init_started_at"]
            for w in ini["worker_init_phases"]:
                ph = w["phases"]
                first = min(v["start"] for v in ph.values())
                row = {"executor_startup_s": first - t0,
                       **{f"{k}_s": ph[k]["end"] - ph[k]["start"] for k in PHASES if k in ph},
                       "engine_init_s": ini["init_s"]}
                if row["engine_init_s"] >= slowest.get("engine_init_s", -1):
                    slowest = {"instance": inst["descriptor"]["instance_id"], **row}
        run = first_run(args.commit, layout)
        dispatch = steady = deficit = None
        if run:
            records, rows = run
            dispatch = min(r["client_started_at_s"] for r in records)
            series = rate_series(rows, dispatch)
            n = len(series)
            steady = statistics.median(series[int(0.2 * n):max(int(0.8 * n), int(0.2 * n) + 1)])
            head = series[:12]
            deficit = sum(max(steady - x, 0.0) for x in head) * 10.0 / steady
        row = {
            "layout": layout, "previous": prev,
            "previous_drained_t": prev_drained,
            "teardown_s": {e["event"]: round(e["t"] - prev_drained, 1) for e in teardown} if teardown else None,
            "deploy_wall_s": round(mine["deploy_end"] - mine["deploy_start"], 1),
            "deploy_stage_clocks": clocks,
            "slowest_engine_init": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in slowest.items()},
            "deploy_end_to_first_dispatch_s": round(dispatch - mine["deploy_end"], 1) if dispatch else None,
            "steady_tokens_per_s": round(steady, 1) if steady else None,
            "ramp_deficit_s": round(deficit, 1) if deficit is not None else None,
            "cost_s": (round(dispatch - prev_drained + deficit, 1)
                       if prev_drained and dispatch and deficit is not None else None),
        }
        out.append(row)
        print(json.dumps(row))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
