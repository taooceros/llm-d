#!/usr/bin/env python3
"""Hard checks that a relocated executable bank was consumed (portable-bank campaign).

Input: a hetero_ctl deploy output or a run-switch artifact ``deploy_report.json.gz``, the
``bank_relocate.py`` output, and the instance(s) that read the relocated copy. For every
bank-reading worker it reports and checks:
  * unanimity ``use``, entries loaded, no backend compile in compile_or_warm_up_model;
  * relocated instances: ``relocation.source_host`` equals the relocation's source host of the
    same rank and the bank's recorded writer placement is the source placement; other
    bank-reading instances read banks they wrote themselves (no relocation record);
  * every bank-reading worker has the same identity digest and topology fingerprint, i.e. one
    bank serves both placements.

Usage: portable_check.py DEPLOY --relocate RELOCATE.json --relocated large_b [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from overlap_check import _load, target_workers  # noqa: E402


def check(deploy_path, relocate_path, relocated):
    body = _load(deploy_path)
    deploy = (body.get("measurements") or {}).get("deployment") or body
    relocation = json.loads(pathlib.Path(relocate_path).read_text())
    pairs = {p["rank"]: p for p in relocation["pairs"]}
    source_hosts = [pairs[r]["source_host"] for r in sorted(pairs)]
    rows = []
    for inst in deploy.get("instances", []):
        iid = (inst.get("descriptor") or {}).get("instance_id") or inst.get("instance_id")
        for w in (inst.get("initialization") or {}).get("worker_init_phases", []):
            bank = w.get("exec_bank") or {}
            if bank.get("mode") != "read":
                continue
            loaded = bank.get("loaded") or {}
            identity = bank.get("identity") or {}
            rows.append({
                "instance": iid, "rank": bank.get("process_rank"),
                "identity_digest": bank.get("identity_digest"),
                "topology_fingerprint": ((identity.get("instance") or {}).get("parallel") or {}).get("topology_fingerprint"),
                "local_device_coords": (identity.get("rank") or {}).get("local_device_coords"),
                "placement": identity.get("placement"),
                "written_placement": loaded.get("written_placement"),
                "relocated_from": (loaded.get("relocation") or {}).get("source_host"),
                "fallback_reason": bank.get("fallback_reason"), "local_error": bank.get("local_error"),
            })
    workers = {(t["instance"], t["rank"]): t for t in target_workers(deploy)}
    for row in rows:
        t = workers.get((row["instance"], row["rank"])) or {}
        row.update({k: t.get(k) for k in ("bank_use", "bank_entries", "bank_load_s", "restored",
                                          "compile_or_warm_up_s", "cwu_trace_s", "cwu_lower_s",
                                          "cwu_backend_compile_s")})
    moved = [r for r in rows if r["instance"] in relocated]
    own = [r for r in rows if r["instance"] not in relocated]
    checks = {
        "relocated_instance_present": bool(moved) and {r["instance"] for r in moved} == set(relocated),
        "bank_used_everywhere": bool(rows) and all(r["bank_use"] is True and (r["bank_entries"] or 0) > 0 for r in rows),
        "no_backend_compile": all((r["cwu_backend_compile_s"] or 0.0) < 0.05 for r in rows),
        "relocated_from_source_rank": all(r["relocated_from"] == pairs[r["rank"]]["source_host"] for r in moved),
        "written_on_source_placement": all((r["written_placement"] or {}).get("host_order") == source_hosts for r in moved),
        "reader_placement_differs": all((r["placement"] or {}).get("host_order") != source_hosts for r in moved),
        "own_banks_not_relocated": all(r["relocated_from"] is None for r in own),
        "one_identity_digest": len({r["identity_digest"] for r in rows}) == 1,
        "one_topology_fingerprint": len({json.dumps(r["topology_fingerprint"], sort_keys=True) for r in rows}) == 1,
    }
    return {"deploy": str(deploy_path), "relocate": str(relocate_path), "relocated": sorted(relocated),
            "checks": checks, "ok": all(checks.values()), "workers": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("deploy", type=pathlib.Path)
    parser.add_argument("--relocate", type=pathlib.Path, required=True)
    parser.add_argument("--relocated", required=True, help="comma-separated instances reading the relocated bank")
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()
    out = check(args.deploy, args.relocate, set(args.relocated.split(",")))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    for r in out["workers"]:
        print(f"{r['instance']} rank{r['rank']}: use={r['bank_use']} entries={r['bank_entries']} "
              f"load={r['bank_load_s']} cwu={r['compile_or_warm_up_s']} compile={r['cwu_backend_compile_s']} "
              f"from={r['relocated_from']}")
    print(f"ok={out['ok']} failed={[k for k, v in out['checks'].items() if not v]}")
    if not out["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
