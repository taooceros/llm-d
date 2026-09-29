#!/usr/bin/env python3
"""Summarise mechanism-cost trial reports (hetero_ctl --output files) side by side."""
import json
import statistics
import sys


def rows(path):
    report = json.load(open(path))["measurements"]["mechanism_cost"]
    for trial in report["trials"]:
        metrics = trial.get("metrics") or {}
        export = metrics.get("native_export") or {}
        imports = metrics.get("native_import_workers") or []
        yield {
            "label": path, "case": trial["case_id"], "mechanism": trial["mechanism"],
            "passed": trial["passed"],
            "first_token_s": metrics.get("proposal_to_first_destination_token_wall_s"),
            "prepare_s": metrics.get("native_prepare_s"),
            "export_s": export.get("export_s"),
            "source_paused_s": export.get("paused_s"),
            "destination_paused_s": imports[0].get("paused_s") if imports else None,
            "bytes": metrics.get("logical_payload_bytes"),
        }, report


def main(paths):
    for path in paths:
        collected = list(rows(path))
        report = collected[0][1] if collected else {}
        print(f"== {path}: passed={report.get('passed')} errors={report.get('errors')}")
        groups = {}
        for row, _ in collected:
            groups.setdefault((row["case"], row["mechanism"]), []).append(row)
        for (case, mechanism), items in sorted(groups.items()):
            def med(key):
                values = [r[key] for r in items if r[key] is not None]
                return round(statistics.median(values), 3) if values else None
            print(f"  {case:12} {mechanism:9} n={len(items)} pass={sum(r['passed'] for r in items)} "
                  f"first_tok={med('first_token_s')} prepare={med('prepare_s')} "
                  f"export={med('export_s')} src_pause={med('source_paused_s')} "
                  f"dst_pause={med('destination_paused_s')} bytes={med('bytes')}")


if __name__ == "__main__":
    main(sys.argv[1:])
