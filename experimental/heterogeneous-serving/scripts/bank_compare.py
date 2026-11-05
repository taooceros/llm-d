#!/usr/bin/env python3
"""Compare executable banks across hosts from a ``snapshot_validate.py bank-manifests`` output.

Banks are grouped by (process rank, entry-set digest): the same rank of the same engine shape,
whichever placement wrote it. Within a group every entry's sha256 is compared across hosts, so
two congruent placements that compiled the same programs show whether the serialized
executables are byte-identical. Writes a JSON summary; ``--require-identical`` exits 1 unless
every group with banks on two or more placements is identical entry for entry.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def compare(hosts):
    groups = defaultdict(list)
    for host in hosts:
        for bank in host.get("banks") or []:
            if "error" in bank:
                continue
            groups[(bank["rank"], bank["entry_set_digest"])].append({**bank, "node_ip": host.get("node_ip")})
    out = []
    for (rank, entry_set), banks in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        keys = set(banks[0]["entries"])
        same_keys = all(set(b["entries"]) == keys for b in banks)
        identical = sum(1 for k in keys if len({b["entries"].get(k) for b in banks}) == 1)
        out.append({
            "rank": rank, "entry_set_digest": entry_set, "entries": len(keys),
            "banks": [{"node_ip": b["node_ip"], "path": b["path"], "format": b["format"],
                       "identity_digest": b["identity_digest"], "placement": b["placement"],
                       "relocated_from": (b.get("relocation") or {}).get("source_host"),
                       "manifest_sha256": b["manifest_sha256"], "bytes": b["bytes"]} for b in banks],
            "placements": len({json.dumps(b["placement"], sort_keys=True) for b in banks}),
            "same_entry_keys": same_keys,
            "entries_identical_across_hosts": identical,
            "identical": same_keys and identical == len(keys),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+",
                        help="snapshot_validate.py bank-manifests --output JSON (several roots allowed)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-identical", action="store_true")
    args = parser.parse_args()
    hosts = [h for path in args.manifests for h in json.loads(path.read_text())["measurements"]["hosts"]]
    groups = compare(hosts)
    multi = [g for g in groups if len(g["banks"]) > 1]
    summary = {"sources": [str(p) for p in args.manifests], "groups": groups,
               "groups_on_several_hosts": len(multi),
               "all_identical": bool(multi) and all(g["identical"] for g in multi)}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for g in groups:
        print(f"rank {g['rank']} entries {g['entries']} hosts {[b['node_ip'] for b in g['banks']]} "
              f"placements {g['placements']} identical {g['entries_identical_across_hosts']}/{g['entries']}")
    if args.require_identical and not summary["all_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
