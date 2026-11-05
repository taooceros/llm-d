#!/usr/bin/env python3
"""Calibration of one TP8 instance (the bed's base) with the trace's length mix, before any arm runs.

Inputs: results/autoscale-burst/calibration/<name>/{summary.json, loadgen/client.jsonl, loadgen/router/samples.jsonl}
  cap-<N>        an instantaneous burst of 240 requests behind router cap N: saturation throughput per cap
  unloaded       80 trace-stratified requests sent one at a time: unloaded TTFT / TPOT medians -> SLO = 5x each
  rate-<r>       Poisson at r req/s for 240 s behind the chosen cap: P99 TTFT/TPOT, SLO attainment, router queue

    .venv/bin/python scripts/asb_calibrate.py            # prints and writes calibration/calibration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asb_metrics as am  # noqa: E402

CAL = Path(__file__).resolve().parents[1] / "results/autoscale-burst/calibration"
WARMUP_S = 30.0


def main() -> int:
    out: dict = {"caps": {}, "rates": {}}
    for d in sorted(CAL.glob("cap-*")):
        s = am.read_json(d / "summary.json")
        if s:
            out["caps"][s["cap"]] = {"saturation": s["saturation"], "ttft_p50": s["summary"]["ttft_p50"],
                                     "ok": s["summary"]["ok"], "n": s["summary"]["n"]}
    un = CAL / "unloaded"
    recs = am.read_jsonl(un / "loadgen/client.jsonl")
    if recs:
        s = am.summarize(recs)
        out["unloaded"] = {"n": s["n"], "ok": s["ok"], "ttft_p50": s["ttft_p50"], "tpot_p50": s["tpot_p50"],
                           "engine_ttft_p50": s["engine_ttft_p50"], "queue_p50": s["queue_p50"],
                           "ttft_p90": s["ttft_p90"], "tpot_p90": s["tpot_p90"]}
        out["slo"] = {"ttft_s": 5.0 * s["ttft_p50"], "tpot_s": 5.0 * s["tpot_p50"],
                      "rule": "5x the unloaded medians (BlitzScale), TTFT = router queue + engine TTFT"}
    slo = out.get("slo")
    for d in sorted(CAL.glob("rate-*"), key=lambda p: float(p.name.split("-", 1)[1])):
        recs = am.read_jsonl(d / "loadgen/client.jsonl")
        if not recs or not slo:
            continue
        rate = float(d.name.split("-", 1)[1])
        steady = [r for r in recs if (r.get("sched_t") or 0.0) >= WARMUP_S]
        s = am.summarize(steady, slo)
        samples = am.read_jsonl(d / "loadgen/router/samples.jsonl")
        q = sorted(x["queue_depth"] for x in samples)
        out["rates"][rate] = {"n": s["n"], "ok": s["ok"], "ttft_p50": s["ttft_p50"], "ttft_p99": s["ttft_p99"],
                              "tpot_p50": s["tpot_p50"], "tpot_p99": s["tpot_p99"],
                              "slo_attainment_pct": s["slo_attainment_pct"], "throughput_rps": s.get("throughput_rps"),
                              "queue_p50": am.pct(q, 50), "queue_p99": am.pct(q, 99), "queue_max": q[-1] if q else None}
    if slo and out["rates"]:
        ok99 = [r for r, v in out["rates"].items() if v["ttft_p99"] is not None and v["ttft_p99"] <= slo["ttft_s"]
                and v["tpot_p99"] is not None and v["tpot_p99"] <= slo["tpot_s"] and v["ok"] == v["n"]]
        ok90 = [r for r, v in out["rates"].items() if (v["slo_attainment_pct"] or 0) >= 90.0]
        out["capacity"] = {"max_rate_p99_within_slo": max(ok99) if ok99 else None,
                           "max_rate_attainment_90": max(ok90) if ok90 else None,
                           "rule": "highest tested Poisson rate whose P99 TTFT and P99 TPOT (after a 30 s warm-up) "
                                   "are within the SLO with every request served: C"}
    (CAL / "calibration.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
