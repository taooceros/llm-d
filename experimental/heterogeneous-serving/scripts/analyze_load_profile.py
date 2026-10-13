#!/usr/bin/env python3
"""Validate and summarize exclusive worker load profiles from deploy JSON."""
import argparse
import json
import pathlib
import statistics


def profiles(node, path=""):
    if isinstance(node, dict):
        if "load_profile" in node:
            yield path, node
        for key, value in node.items():
            if key != "load_profile":
                yield from profiles(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from profiles(value, f"{path}/{index}")


def summarize(filename):
    payload = json.loads(pathlib.Path(filename).read_text())
    rows = []
    for path, worker in profiles(payload):
        p = worker["load_profile"]
        phase = worker["phases"]["load_model"]
        measured_load = phase["end"] - phase["start"]
        categories = p["categories_s"]
        category_sum = sum(categories.values())
        error = abs(category_sum - measured_load) / measured_load
        if error > 0.05:
            raise ValueError(f"{path}: category closure error {error:.2%}")
        raw = p["calls"].get("raw_batched_device_put")
        if not raw or not raw["local_bytes"]:
            raise ValueError(f"{path}: no measured raw TPU transfers")
        seconds = raw["exclusive_s"]
        row = {
            "worker_path": path, "executor_rank": worker["executor_rank"],
            "implementation": p["implementation"], "load_model_s": measured_load,
            "categories_s": categories, "closure_error_fraction": error,
            "raw_transfer_s": seconds, "raw_transfer_calls": raw["count"],
            "final_local_bytes": raw["local_bytes"],
            "observed_raw_transfer_GBps": raw["local_bytes"] / 1e9 / seconds,
            "snapshot_transfer_only_lower_bound_s": seconds,
            "snapshot_skippable_load_s_INFERENCE": measured_load - seconds,
            "calls": p["calls"], "top20": p["top20"]}
        rows.append(row)
    if not rows:
        raise ValueError(f"{filename}: no worker load profiles")
    fields = ("load_model_s", "raw_transfer_s", "final_local_bytes",
              "observed_raw_transfer_GBps", "raw_transfer_calls")
    return {"source": str(filename), "submission_id": payload["submission_id"], "workers": rows,
            "median": {key: statistics.median(row[key] for row in rows) for key in fields},
            "ranges": {key: [min(row[key] for row in rows), max(row[key] for row in rows)]
                       for key in fields},
            "median_calls": {
                name: {field: statistics.median(row["calls"][name][field] for row in rows)
                       for field in ("exclusive_s", "count")}
                for name in rows[0]["calls"]},
            "median_categories_s": {key: statistics.median(row["categories_s"][key] for row in rows)
                                    for key in rows[0]["categories_s"]},
            "interpretation": "Transfer-only lower bound is the sum of observed synchronized raw final-shard puts; it excludes snapshot I/O, reconstruction and coordination. It is not an achieved restore time."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {pathlib.Path(name).stem: summarize(name) for name in args.inputs}
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    for layout, summary in result.items():
        print(layout, "workers=", len(summary["workers"]))
        print(json.dumps(summary["median"], indent=2))
        print(json.dumps(summary["median_categories_s"], indent=2))
        print("max closure error", max(row["closure_error_fraction"] for row in summary["workers"]))


if __name__ == "__main__":
    main()
