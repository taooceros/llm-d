#!/usr/bin/env python3
"""Summarise a scripts/switch_campaign.sh run: production C (full) / partial endpoints (partial).

Reads, never imputes:
  <OUT>/campaign.json, runs.jsonl, timeline.jsonl, switch-r<r>.json   (the driver's outputs)
  results/hetero/artifacts/<run_id>/                                   (fetched by hetero_ctl: switch_report.json,
      phase2_records.json.gz, phase2_r1_iterations.csv.gz, deploy_report.json.gz, drain_report.json.gz)
  results/hetero/artifacts/<tag>-ret-r<r>/                             (full: the harness return pilot, W0 on S)

Segments and endpoints are RECOMPUTED offline from the artifacts with hetero.switch.build_timeline (the
same function the run used) and compared with the report's in-process values.  For full mode the harness
H -> S cost is computed the way the published harness measures it:
    T0 = last phase-2 completion of the run-switch preceding the return
    first dispatch / ramp deficit = scripts/analyze_overlap.py:pilot of results/hetero/artifacts/<tag>-ret-r<r>
    C_harness = (first dispatch - T0) + ramp deficit
split into contiguous segments by the driver's timeline marks (local controller clock; T0 and first
dispatch are head-pod clocks, so the split carries any controller/head clock offset).

First-switch campaigns (``switch_campaign.sh first``) have cold reps r1..rN (pre-spawned target, checkpoint
load, inline snapshot + bank write into a per-run scope) and one warm follow-up (``kind: warm``, restore +
bank read from the last cold rep's scope); no harness return.  Means cover the cold reps; the warm rep is
reported beside them, plus the amortized per-switch cost (C_first + (N - 1) * C_warm) / N.

Usage: switch_summary.py <OUT> [--artifacts results/hetero/artifacts]   -> <OUT>/switch_summary.{json,md}
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pathlib
import statistics
import sys
from typing import Any, Callable, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hetero.switch import build_timeline  # noqa: E402

ARTIFACTS = ROOT / "results" / "hetero" / "artifacts"
CLOCK_NOTE = ("production segments: head-process clock (engine-host clocks inside deploy); "
              "harness split: controller-machine marks between head-pod T0 / first dispatch")


def read_json(path: pathlib.Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


AMORTIZE_N = (1, 3, 10)
O3_WARM_C = 27.51  # O3 full S->H production C, mean of 3 (bulk plan, O3 results)
# The O3 separate provisioning pass as the decision log (2026-10-12) counts it: H normal deploy 255.4 s +
# per-worker snapshot 7.2 s + bank-writing H restore deploy 88.2 s (hetero_ctl/rcm walls) = 350.8 s.
SEPARATE_PASS_S = 255.4 + 7.2 + 88.2


def load_run(out: pathlib.Path, run_id: str, artifacts: pathlib.Path, rep: Any) -> Dict[str, Any]:
    """Offline recompute of one run-switch from its fetched artifacts."""
    directory = artifacts / run_id
    report_path = directory / "switch_report.json"
    if report_path.exists():
        report = read_json(report_path)
    else:  # the compact report also travels in the hetero_ctl output
        report = read_json(out / f"switch-{label_of(rep)}.json")["measurements"]["result"]
    optional = lambda name: read_json(directory / name) if (directory / name).exists() else None  # noqa: E731
    records2 = read_json(directory / "phase2_records.json.gz")
    iter_path = directory / "phase2_r1_iterations.csv.gz"
    rows = []
    if iter_path.exists():
        with gzip.open(iter_path, "rt") as handle:
            rows = [(r["instance_id"], float(r["completed_at_s"]), int(float(r["emitted_output_tokens"] or 0)))
                    for r in csv.DictReader(handle) if r.get("completed_at_s")]
    again = build_timeline(
        mode=report["mode"], T0=report["T0"], stamps=report["stamps"], records2=records2,
        deploy_report=optional("deploy_report.json.gz"), drain_report=optional("drain_report.json.gz"),
        iter_rows=rows, plan=(report.get("preflight") or {}).get("plan"),
        static_target_s=report.get("static_target_s"))
    online = {s["segment"]: s["s"] for s in report.get("segments", [])}
    drift = max((abs(s["s"] - online.get(s["segment"], float("nan"))) for s in again["segments"]), default=None)
    return {"run_id": run_id, "rep": rep, "report": report, "timeline": again, "records2": records2,
            "recompute_max_segment_diff_s": drift, "completed": bool(report.get("completed")),
            "restored": report.get("restored", {}).get("all_restored"),
            "provisioned": report.get("provisioned", {}).get("all_provisioned"),
            "accounting_complete": bool(report.get("accounting", {}).get("phase2", {}).get("complete")
                                        and report.get("accounting", {}).get("phase1", {}).get("complete"))}


def label_of(rep: Any) -> str:
    return rep if isinstance(rep, str) else f"r{rep}"


def harness_return(run: Dict[str, Any], marks: Dict[str, float], base: str,
                   pilot_fn: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
    """Harness H -> S: T0 -> first dispatch of the return pilot, split by the timeline marks."""
    rep = run["rep"]
    label = f"ret-r{rep}"
    T0 = max(r["client_finished_at_s"] for r in run["records2"] if r.get("client_finished_at_s") is not None)
    pilot = pilot_fn(f"{base}-ret-r{rep}")
    first = pilot["first_dispatch"]
    chain = [("switch_tail", f"teardown_start:{label}"), ("teardown_drain_job", f"drained:{label}"),
             ("teardown_gateway_down_job", f"gateway_down:{label}"), ("teardown_release_job", f"released:{label}"),
             ("teardown_to_deploy_gap", f"deploy_start:{label}"), ("deploy_job", f"deploy_end:{label}"),
             ("deploy_to_check_gap", f"check_start:{label}"), ("check_job", f"check_end:{label}"),
             ("check_to_pilot_gap", f"pilot_start:{label}")]
    missing = [m for _, m in chain if m not in marks]
    segments, prev = [], T0
    if not missing:
        for name, mark in chain:
            segments.append({"segment": name, "s": round(marks[mark] - prev, 3)})
            prev = marks[mark]
        segments.append({"segment": "pilot_start_to_first_dispatch", "s": round(first - prev, 3)})
    gap = first - T0
    ramp = pilot["ramp_deficit_s"]
    return {"T0": T0, "first_dispatch": first, "gap_s": round(gap, 3), "ramp_deficit_s": round(ramp, 3),
            "C_harness_s": round(gap + ramp, 3), "segments": segments,
            "segment_sum_s": round(sum(s["s"] for s in segments), 3), "missing_marks": missing,
            "return_pilot_complete": pilot.get("all_requests_complete")}


def mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 3) if vals else None


def summarize(out: pathlib.Path, *, artifacts: pathlib.Path = ARTIFACTS,
              pilot_fn: Optional[Callable[[str], Dict[str, Any]]] = None) -> Dict[str, Any]:
    campaign = read_json(out / "campaign.json") if (out / "campaign.json").exists() else {}
    runs_log = read_jsonl(out / "runs.jsonl")
    ok_runs: Dict[Any, str] = {}
    for row in runs_log:
        if row.get("ok") and "-ret-" not in row["run_id"]:
            ok_runs["warm" if row.get("kind") == "warm" else int(row["rep"])] = row["run_id"]
    if not ok_runs:
        raise SystemExit(f"no successful run-switch recorded in {out}/runs.jsonl")
    marks = {e["event"]: e["t"] for e in read_jsonl(out / "timeline.jsonl")}
    mode = campaign.get("mode")
    first = mode == "first"
    base = campaign.get("run_base") or next(iter(ok_runs.values())).rsplit("-r", 1)[0]
    if pilot_fn is None and mode not in ("partial", "first"):
        import analyze_overlap

        pilot_fn = analyze_overlap.pilot
    reps = []
    for rep, run_id in sorted(ok_runs.items(), key=lambda kv: (isinstance(kv[0], str), str(kv[0]).zfill(4))):
        run = load_run(out, run_id, artifacts, rep)
        mode = mode or run["report"]["mode"]
        row: Dict[str, Any] = {
            "rep": rep, "label": label_of(rep), "kind": ("warm" if rep == "warm" else "cold") if first else None,
            "run_id": run_id, "completed": run["completed"], "all_restored": run["restored"],
            "all_provisioned": run["provisioned"],
            "accounting_complete": run["accounting_complete"],
            "recompute_max_segment_diff_s": run["recompute_max_segment_diff_s"],
            "segments": run["timeline"]["segments"], "segment_sum_s": run["timeline"]["segment_sum_s"],
            "segments_reconcile": run["timeline"]["segments_reconcile"],
            "endpoints": run["timeline"]["endpoints"],
            "drain_detail_s": run["timeline"]["drain_detail_s"],
            "deploy_anatomy": {k: v for k, v in (run["timeline"].get("deploy_anatomy") or {}).items()
                               if k in ("stage_clocks", "phase_median_s", "phase_max_s")}}
        if run["report"].get("prespawn"):  # OVERLAP=1: gate anatomy + source delivery around the window
            row["prespawn"] = {"anatomy": run["timeline"].get("prespawn"),
                               "window": run["report"]["prespawn"].get("window"),
                               "parked_before_T0": run["report"]["prespawn"].get("parked_before_T0")}
        if mode == "full":
            row["harness"] = harness_return(run, marks, base, pilot_fn)
        reps.append(row)
    summary: Dict[str, Any] = {"out": str(out), "mode": mode, "run_base": base, "clock_note": CLOCK_NOTE,
                               "reps": reps}
    if mode in ("full", "first"):
        keys = ["gap_dispatch_s", "gap_first_token_s", "ramp_makespan_s", "ramp_deficit_iter_s", "C_production_s",
                "C_production_ex_capture_s", "C_production_deficit_s", "capture_arm_s", "target_makespan_s"]
        timed = [r for r in reps if r["kind"] != "warm"]
        names = [s["segment"] for s in timed[0]["segments"]]
        phases = sorted({p for r in timed for p in r["deploy_anatomy"].get("phase_median_s", {})})
        summary["mean"] = {
            "endpoints": {k: mean([r["endpoints"].get(k) for r in timed]) for k in keys},
            "segments_s": {n: mean([next((s["s"] for s in r["segments"] if s["segment"] == n), None) for r in timed])
                           for n in names},
            "engine_phase_median_s": {p: mean([r["deploy_anatomy"]["phase_median_s"].get(p) for r in timed])
                                      for p in phases}}
        if mode == "full":
            summary["mean"].update(
                C_harness_s=mean([r["harness"]["C_harness_s"] for r in reps]),
                harness_gap_s=mean([r["harness"]["gap_s"] for r in reps]),
                harness_ramp_deficit_s=mean([r["harness"]["ramp_deficit_s"] for r in reps]))
        warm = next((r for r in reps if r["kind"] == "warm"), None)
        if first:
            c_first = summary["mean"]["endpoints"]["C_production_s"]
            c_warm = warm["endpoints"].get("C_production_s") if warm else None

            def per_switch(warm_c):
                return {str(n): None if c_first is None or warm_c is None
                        else round((c_first + (n - 1) * warm_c) / n, 3) for n in AMORTIZE_N}
            summary["amortized"] = {
                "C_first_s": c_first, "C_warm_s": c_warm, "n_cold": len(timed), "n_warm": int(warm is not None),
                "per_switch_s": per_switch(c_warm),
                # Second warm basis: the O3 production mean (n = 3, OverlapSwitch full-c762edb-ov).
                "C_warm_O3_s": O3_WARM_C, "per_switch_O3_warm_s": per_switch(O3_WARM_C),
                # The separate provisioning pass (SEPARATE_PASS_S, deploy walls) + O3 C per switch.
                "separate_pass_s": {str(n): round((SEPARATE_PASS_S + n * O3_WARM_C) / n, 3)
                                    for n in AMORTIZE_N}}
    else:
        flat = ["kept_gap_s", "join_delay_s", "C_eff_s", "target_makespan_s"]
        summary["mean"] = {"endpoints": {k: mean([r["endpoints"].get(k) for r in reps]) for k in flat},
                           "class_finish_s": {c: mean([(r["endpoints"].get("class_finish_s") or {}).get(c) for r in reps])
                                              for c in ("kept", "joiner")}}
    summary["n"] = len(reps)
    return summary


def fmt(value: Any) -> str:
    return "n/a" if value is None else (f"{value:.2f}" if isinstance(value, float) else str(value))


def markdown(summary: Dict[str, Any]) -> str:
    reps, lines = summary["reps"], []
    lines.append(f"# {summary['mode']} switch: {summary['run_base']} ({summary['n']} reps)\n")
    lines.append(f"_{summary['clock_note']}_\n")
    first = summary["mode"] == "first"
    head = "| " + " | ".join(["metric"] + [r.get("label") or f"r{r['rep']}" for r in reps]
                             + ["mean (cold)" if first else "mean"]) + " |"
    rule = "|" + "---|" * (len(reps) + 2)

    def row(label: str, getter: Callable[[Dict[str, Any]], Any], mean_value: Any = None) -> str:
        return "| " + " | ".join([label] + [fmt(getter(r)) for r in reps] + [fmt(mean_value)]) + " |"

    m = summary["mean"]
    if summary["mode"] in ("full", "first"):
        lines += ["## Production C (in-process switch)", head, rule]
        for key in ("gap_dispatch_s", "gap_first_token_s", "ramp_makespan_s", "ramp_deficit_iter_s", "C_production_s",
                    "C_production_ex_capture_s", "C_production_deficit_s"):
            lines.append(row(key, lambda r, k=key: r["endpoints"].get(k), m["endpoints"].get(key)))
        lines += ["", "## Production segments (s after T0, contiguous)", head, rule]
        for name in m["segments_s"]:
            cls = next((s["class"] for s in reps[0]["segments"] if s["segment"] == name), "")
            lines.append(row(f"{name} ({cls})", lambda r, n=name: next(
                (s["s"] for s in r["segments"] if s["segment"] == n), None), m["segments_s"][name]))
        lines.append(row("segment sum", lambda r: r["segment_sum_s"]))
        lines += ["", "## Engine start-up phases (median over target workers, s)", head, rule]
        for name, value in m["engine_phase_median_s"].items():
            lines.append(row(name, lambda r, n=name: r["deploy_anatomy"].get("phase_median_s", {}).get(n), value))
    if summary["mode"] == "first" and summary.get("amortized"):
        a = summary["amortized"]
        lines += ["", "## Amortized per-switch cost (C_first + (N - 1) * C_warm) / N",
                  f"C_first (mean of {a['n_cold']} cold) = {fmt(a['C_first_s'])} s; "
                  f"C_warm (this campaign, n = {a['n_warm']}) = {fmt(a['C_warm_s'])} s; "
                  f"C_warm (O3, n = 3) = {fmt(a['C_warm_O3_s'])} s", "",
                  "| N | per-switch, this warm | per-switch, O3 warm | separate pass (deploy walls) + O3 warm |",
                  "|---|---|---|---|"]
        lines += [f"| {n} | {fmt(a['per_switch_s'][n])} | {fmt(a['per_switch_O3_warm_s'][n])} | "
                  f"{fmt(a['separate_pass_s'][n])} |" for n in a["per_switch_s"]]
    if summary["mode"] == "full":
        lines += ["", "## Harness H -> S (published harness, return pilot)", head, rule,
                  row("gap_s", lambda r: r["harness"]["gap_s"], m["harness_gap_s"]),
                  row("ramp_deficit_s", lambda r: r["harness"]["ramp_deficit_s"], m["harness_ramp_deficit_s"]),
                  row("C_harness_s", lambda r: r["harness"]["C_harness_s"], m["C_harness_s"])]
        names = [s["segment"] for s in reps[0]["harness"]["segments"]]
        for name in names:
            lines.append(row(f"harness: {name}", lambda r, n=name: next(
                (s["s"] for s in r["harness"]["segments"] if s["segment"] == n), None),
                mean([next((s["s"] for s in r["harness"]["segments"] if s["segment"] == name), None) for r in reps])))
    elif summary["mode"] == "partial":
        lines += ["## Partial switch endpoints", head, rule]
        for key in ("kept_gap_s", "join_delay_s", "C_eff_s", "target_makespan_s"):
            lines.append(row(key, lambda r, k=key: r["endpoints"].get(k), m["endpoints"].get(key)))
        for cls in ("kept", "joiner"):
            lines.append(row(f"{cls}-class finish (s after T0)",
                             lambda r, c=cls: (r["endpoints"].get("class_finish_s") or {}).get(c), m["class_finish_s"][cls]))
        for key in ("deploy_ready_s", "join_called_s", "join_returned_s", "first_joiner_dispatch_s", "first_joiner_token_s"):
            lines.append(row(key, lambda r, k=key: (r["endpoints"].get("joiner_timeline") or {}).get(k)))
        for window in ("during_join", "after_join"):
            lines.append(row(f"kept-pair tokens/s {window}", lambda r, w=window: (
                (r["endpoints"].get("kept_pair_throughput") or {}).get(w) or {}).get("tokens_per_s")))
        lines += ["", "## Kept-side segments (s after T0)", head, rule]
        for name in [s["segment"] for s in reps[0]["segments"]]:
            lines.append(row(name, lambda r, n=name: next((s["s"] for s in r["segments"] if s["segment"] == n), None),
                             mean([next((s["s"] for s in r["segments"] if s["segment"] == name), None) for r in reps])))
    lines += ["", "## Run health", head, rule,
              row("completed (all requests exact)", lambda r: r["completed"]),
              row("all workers restored", lambda r: r["all_restored"]),
              row("snapshot + bank provisioned", lambda r: r.get("all_provisioned")),
              row("offline vs in-process max segment diff (s)", lambda r: r["recompute_max_segment_diff_s"])]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--artifacts", type=pathlib.Path, default=ARTIFACTS)
    args = parser.parse_args()
    summary = summarize(args.out, artifacts=args.artifacts)
    (args.out / "switch_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    text = markdown(summary)
    (args.out / "switch_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
