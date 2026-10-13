#!/usr/bin/env python3
"""Summarise W3 campaign arms: makespan, throughput, migrations, native preemption waste.

Usage: summarize_w3.py <run_id> [<run_id> ...]
Reads results/pilot/<run_id>_W3_<arm>.json and, when present, the per-iteration CSV
under results/hetero/artifacts/<run_id>/.
"""
import csv
import glob
import gzip
import json
import os
import statistics
import sys

ARMS = ("S", "H", "M0", "MR", "MK-H")


def iterations(run_id, arm):
    paths = glob.glob(f"results/hetero/artifacts/{run_id}/W3_{arm}_r*_iterations.csv.gz")
    if not paths:
        return None
    rows = list(csv.DictReader(gzip.open(paths[0], "rt")))
    start = min(float(row["scheduled_at_s"]) for row in rows)
    by = {}
    per_instance = {}
    for row in rows:
        per_instance.setdefault(row["instance_id"], []).append(row)
        stats = by.setdefault(row["instance_id"], {"preempted": 0, "replayed": 0, "emitted": 0,
                                                   "busy_s": 0.0, "last_s": 0.0, "steps": 0,
                                                   "running": 0, "context": 0})
        stats["preempted"] += int(row.get("num_preempted") or 0)
        stats["replayed"] += int(row.get("native_preemption_replayed_tokens") or 0)
        stats["emitted"] += int(row.get("emitted_output_tokens") or 0)
        stats["busy_s"] += float(row["completed_at_s"]) - float(row["scheduled_at_s"])
        stats["last_s"] = max(stats["last_s"], float(row["completed_at_s"]) - start)
        stats["steps"] += 1
        stats["running"] += int(row["running_requests"])
        stats["context"] += int(row.get("context_tokens") or 0)
    # Step time is the gap between consecutive completions on one engine: scheduled_at ->
    # completed_at includes queueing, and iterations overlap.  A step "has prefill" when
    # prefill_tokens > 0 (new prompts or eviction replay).
    for inst, steps in per_instance.items():
        steps.sort(key=lambda row: float(row["completed_at_s"]))
        gaps = [(float(b["completed_at_s"]) - float(a["completed_at_s"]), b)
                for a, b in zip(steps, steps[1:])]
        decode = [(gap, row) for gap, row in gaps if int(row["prefill_tokens"] or 0) == 0]
        total = sum(gap for gap, _ in gaps)
        by[inst].update({
            "prefill_step_share": 1 - sum(gap for gap, _ in decode) / total,
            "decode_batch_p50": statistics.median(int(row["decode_tokens"]) for _, row in decode),
            "decode_step_p50_ms": statistics.median(gap for gap, _ in decode) * 1000,
            "decode_context_p50": statistics.median(float(row["context_tokens"] or 0)
                                                    for _, row in decode),
            "decode_tok_s": sum(int(row["emitted_output_tokens"] or 0) for _, row in decode)
                            / sum(gap for gap, _ in decode),
        })
    return by


def chips(instance_id):
    return 16 if instance_id.startswith("large") else 8


def main(run_ids):
    table = {}
    for run_id in run_ids:
        for arm in ARMS:
            path = f"results/pilot/{run_id}_W3_{arm}.json"
            if not os.path.exists(path):
                continue
            result = json.load(open(path))["measurements"]["result"]
            its = iterations(run_id, arm) or {}
            table.setdefault(arm, []).append({
                "makespan_s": result.get("makespan_s"),
                "tok_s": result.get("useful_tokens_per_s"),
                "complete": all(result.get(key) is True for key in (
                    "all_requests_complete", "native_output_budget_exact",
                    "iteration_capture_one_to_one")),
                "migrations": result.get("migrations_committed"),
                "preemptions": sum(s["preempted"] for s in its.values()) if its else None,
                "preempt_replayed_tokens": sum(s["replayed"] for s in its.values()) if its else None,
                "per_instance_tokens": {k: s["emitted"] for k, s in sorted(its.items())},
            })
    valid_uniforms = [a for a in ("S", "H")
                      if a in table and all(r["complete"] for r in table[a])]
    best_uniform = (min(statistics.median(r["makespan_s"] for r in table[a])
                        for a in valid_uniforms) if len(valid_uniforms) == 2 else None)
    def median_or_na(rows, key):
        values = [r[key] for r in rows]
        return statistics.median(values) if all(v is not None for v in values) else "N/A"
    print(f"{'arm':5} {'n':>2} {'makespan_s':>11} {'tok/s':>8} {'vs best uniform':>16} "
          f"{'migr':>5} {'preempt':>8} {'preempt replay tok':>19} complete")
    for arm in ARMS:
        if arm not in table:
            continue
        rows = table[arm]
        makespan = statistics.median(r["makespan_s"] for r in rows)
        gain = (f"{(best_uniform - makespan) / best_uniform * 100:+.1f}%"
                if best_uniform and all(r["complete"] for r in rows) else "N/A")
        print(f"{arm:5} {len(rows):>2} {makespan:>11.1f} {statistics.median(r['tok_s'] for r in rows):>8.1f} "
              f"{gain:>16} {statistics.median(r['migrations'] or 0 for r in rows):>5} "
              f"{median_or_na(rows, 'preemptions'):>8} "
              f"{median_or_na(rows, 'preempt_replayed_tokens'):>19} "
              f"{all(r['complete'] for r in rows)}")
        for r in rows:
            print(f"      tokens by instance: {r['per_instance_tokens']}")
    # Per-engine decode rate: output tokens per second per chip up to the engine's last step,
    # mean running sequences per chip, mean resident context tokens per step.
    for run_id in run_ids:
        for arm in ARMS:
            for inst, s in sorted((iterations(run_id, arm) or {}).items()):
                c = chips(inst)
                print(f"{arm:5} {inst:8} chips={c:2} out/s/chip={s['emitted'] / s['last_s'] / c:5.1f} "
                      f"running/chip={s['running'] / s['steps'] / c:4.2f} "
                      f"context={s['context'] / s['steps']:7.0f} last_step={s['last_s']:7.1f}s")
                print(f"      decode steps: batch p50={s['decode_batch_p50']:.0f} "
                      f"step p50={s['decode_step_p50_ms']:.1f} ms context p50={s['decode_context_p50']:.0f} "
                      f"tok/s/chip={s['decode_tok_s'] / c:.1f} prefill-step time share="
                      f"{s['prefill_step_share']:.1%}")


if __name__ == "__main__":
    main(sys.argv[1:])
