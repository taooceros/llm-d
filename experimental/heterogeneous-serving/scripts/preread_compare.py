#!/usr/bin/env python3
"""Exactness of a pre-read snapshot against a device-produced (normal-loader) snapshot.

    scripts/preread_compare.py A.json B.json [--output RESULT.json]

A and B are ``scripts/snapshot_validate.py file-fingerprint`` outputs (``measurements.hosts[]
.fingerprints[]``, or the bare ``hosts`` form), or a JSON list of ``weight_snapshot.file_fingerprint``
records. The records of every rank of an instance are united. Both sides must then be identical in:
- per array: path, order, shape, dtype, spec, mesh shape/axes, and the device list (id,
  process_index, platform);
- per shard, keyed by device_id: index, shape, bytes, sha256.

Generation, mtime, file names and directories are ignored. Exit 0 when equal, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_fingerprints(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    data = data.get("measurements", data)
    if "hosts" in data:
        return [f for host in data["hosts"] for f in host.get("fingerprints", [])]
    raise ValueError(f"{path}: not a file-fingerprint result")


def unite(fingerprints: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{(layout/)instance: {"arrays": [meta...], "shards": {path: {device_id: row}}, "ranks": [...]}}."""
    out: Dict[str, Dict[str, Any]] = {}
    for record in fingerprints:
        if "array_meta" not in record:
            raise ValueError("fingerprint without array_meta (re-run file-fingerprint at this commit)")
        name = record["instance"]
        entry = out.setdefault(name, {"arrays": None, "shards": {}, "ranks": []})
        entry["ranks"].append(record["process_rank"])
        if entry["arrays"] is None:
            entry["arrays"] = record["array_meta"]
        elif entry["arrays"] != record["array_meta"]:
            raise ValueError(f"{name}: ranks disagree on the array layout")
        for shard in record["shards"]:
            per = entry["shards"].setdefault(shard["path"], {})
            if shard["device_id"] in per:
                raise ValueError(f"{name}: device {shard['device_id']} appears twice for {shard['path']}")
            per[shard["device_id"]] = {k: shard[k] for k in ("index", "shape", "dtype", "bytes", "sha256")}
    for entry in out.values():
        entry["ranks"].sort()
    return out


def compare(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    left, right = unite(a), unite(b)
    problems: List[str] = []
    if set(left) != set(right):
        problems.append(f"instances differ: {sorted(left)} vs {sorted(right)}")
    for name in sorted(set(left) & set(right)):
        x, y = left[name], right[name]
        if x["ranks"] != y["ranks"]:
            problems.append(f"{name}: ranks {x['ranks']} vs {y['ranks']}")
        if len(x["arrays"]) != len(y["arrays"]):
            problems.append(f"{name}: {len(x['arrays'])} vs {len(y['arrays'])} arrays")
        for ma, mb in zip(x["arrays"], y["arrays"]):
            if ma != mb:
                keys = sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))
                problems.append(f"{name}: array {ma.get('path')} differs in {keys}")
        for path in sorted(set(x["shards"]) | set(y["shards"])):
            sa, sb = x["shards"].get(path, {}), y["shards"].get(path, {})
            if set(sa) != set(sb):
                problems.append(f"{name}: {path} shard devices {sorted(sa)} vs {sorted(sb)}")
            for device in sorted(set(sa) & set(sb)):
                if sa[device] != sb[device]:
                    keys = sorted(k for k in sa[device] if sa[device][k] != sb[device].get(k))
                    problems.append(f"{name}: {path} device {device} differs in {keys}")
    shards = sum(len(v) for e in left.values() for v in e["shards"].values())
    return {"equal": not problems, "problems": problems[:200], "problem_count": len(problems),
            "instances": sorted(left), "arrays": {n: len(e["arrays"]) for n, e in left.items()},
            "shards_compared": shards,
            "ignored": ["generation", "mtime_ns", "file names", "directories"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = compare(load_fingerprints(args.a), load_fingerprints(args.b))
    text = json.dumps(result, indent=1)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
