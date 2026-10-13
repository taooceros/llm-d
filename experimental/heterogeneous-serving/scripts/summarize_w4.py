#!/usr/bin/env python3
"""Summarise W4 (long-document + chat) runs: makespan, per-class finish times, placement.

Usage: summarize_w4.py <run_id> [<run_id> ...]
Reads results/pilot/<run_id>_W4_<arm>.json, data/manifests/W4.json and the
per-run artifacts under results/hetero/artifacts/<run_id>/.
"""
import collections
import csv
import glob
import gzip
import json
import os
import sys

ARMS = ("H", "M0")
FLAGS = ("all_requests_complete", "native_output_budget_exact", "iteration_capture_one_to_one")


def artifact(run_id, arm, suffix):
    # r[0-9] rather than r*: r* would also match r1_gateway_records.
    paths = glob.glob(f"results/hetero/artifacts/{run_id}/W4_{arm}_r[0-9]_{suffix}")
    return paths[0] if paths else None


def main(run_ids):
    request_class = {r["request_id"]: r["request_class"]
                     for r in json.load(open("data/manifests/W4.json"))["requests"]}
    for run_id in run_ids:
        for arm in ARMS:
            path = f"results/pilot/{run_id}_W4_{arm}.json"
            if not os.path.exists(path):
                continue
            result = json.load(open(path))["measurements"]["result"]
            complete = all(result.get(key) is True for key in FLAGS)
            print(f"== {run_id} {arm}: makespan={result['makespan_s']:.1f}s "
                  f"tok/s={result['useful_tokens_per_s']:.1f} complete={complete}")

            records = json.load(gzip.open(artifact(run_id, arm, "records.json.gz")))
            t0 = min(r["client_started_at_s"] for r in records)
            finish = collections.defaultdict(list)
            placement = collections.Counter()
            for r in records:
                rid = r.get("canonical_request_id") or r["request_id"].split(":")[-1]
                cls = request_class[rid]
                finish[cls].append(r["client_finished_at_s"] - t0)
                placement[(cls, r.get("initial_instance"))] += 1
            for cls in ("short", "long"):
                v = sorted(finish[cls])
                print(f"   {cls:5} n={len(v):3} p50_finish={v[len(v) // 2]:6.1f}s "
                      f"last_finish={v[-1]:6.1f}s placement="
                      f"{ {i: n for (c, i), n in sorted(placement.items()) if c == cls} }")

            rows = list(csv.DictReader(gzip.open(artifact(run_id, arm, "iterations.csv.gz"), "rt")))
            start = min(float(r["scheduled_at_s"]) for r in rows)
            by = collections.defaultdict(lambda: collections.Counter())
            for r in rows:
                s = by[r["instance_id"]]
                s["emitted"] += int(r["emitted_output_tokens"] or 0)
                s["prefill"] += int(r["prefill_tokens"] or 0)
                s["preempted"] += int(r["num_preempted"] or 0)
                s["replayed"] += int(r["native_preemption_replayed_tokens"] or 0)
                s["last"] = max(s["last"], float(r["completed_at_s"]) - start)
            for inst, s in sorted(by.items()):
                print(f"   {inst:8} decode_tok={s['emitted']:7d} prefill_tok={s['prefill']:7d} "
                      f"last_step={s['last']:6.1f}s preempt={s['preempted']:4d} "
                      f"replayed_tok={s['replayed']}")


if __name__ == "__main__":
    main(sys.argv[1:])
