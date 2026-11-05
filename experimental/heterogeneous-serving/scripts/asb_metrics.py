#!/usr/bin/env python3
"""Metrics of the autoscale-burst runs (shared by calibration, the summary and the figures).

Definitions (per request, from the open-loop client's records; durations only, each on one clock):
  TTFT = router queue wait (router clock) + engine TTFT (engine clock)
  TPOT = (engine finished - engine started - engine TTFT) / (output tokens - 1)
  SLO  = TTFT <= 5x and TPOT <= 5x their unloaded medians (BlitzScale); a request that failed (timeout, 429, 5xx,
         transport error) violates the SLO.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    for p in (path, path.with_suffix(path.suffix + ".gz")):
        if p.exists():
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rt") as fh:
                return [json.loads(line) for line in fh if line.strip()]
    return []


def read_json(path: Path):
    for p in (path, path.with_suffix(path.suffix + ".gz")):
        if p.exists():
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rt") as fh:
                return json.load(fh)
    return None


def pct(values, q: float):
    vals = sorted(v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def ok(rec: dict) -> bool:
    return rec.get("status") == 200 and rec.get("ttft_s") is not None and not rec.get("mismatch")


def meets_slo(rec: dict, slo: dict) -> bool:
    if not ok(rec):
        return False
    tpot = rec.get("tpot_s")
    return rec["ttft_s"] <= slo["ttft_s"] and (tpot is None or tpot <= slo["tpot_s"])


def summarize(recs: list[dict], slo: dict | None = None) -> dict:
    good = [r for r in recs if ok(r)]
    status: dict[str, int] = {}
    for r in recs:
        key = str(r.get("status"))
        status[key] = status.get(key, 0) + 1
    row = {"n": len(recs), "ok": len(good), "status": status,
           "ttft_p50": pct([r["ttft_s"] for r in good], 50), "ttft_p90": pct([r["ttft_s"] for r in good], 90),
           "ttft_p99": pct([r["ttft_s"] for r in good], 99),
           "tpot_p50": pct([r.get("tpot_s") for r in good], 50), "tpot_p90": pct([r.get("tpot_s") for r in good], 90),
           "tpot_p99": pct([r.get("tpot_s") for r in good], 99),
           "queue_p50": pct([r.get("queue_s") for r in good], 50), "queue_p99": pct([r.get("queue_s") for r in good], 99),
           "engine_ttft_p50": pct([r.get("engine_ttft_s") for r in good], 50),
           "e2e_p50": pct([r.get("e2e_s") for r in good], 50)}
    if slo:
        viol = sum(1 for r in recs if not meets_slo(r, slo))
        row["slo_violation_pct"] = 100.0 * viol / len(recs) if recs else None
        row["slo_attainment_pct"] = 100.0 - row["slo_violation_pct"] if recs else None
    if good:
        t0 = min(r["send_at"] for r in recs)
        t1 = max(r["done_at"] for r in good)
        row["duration_s"] = t1 - t0
        row["throughput_rps"] = len(good) / (t1 - t0) if t1 > t0 else None
        row["output_tok_s"] = sum(r.get("output_tokens") or 0 for r in good) / (t1 - t0) if t1 > t0 else None
    return row


def throughput_window(recs: list[dict], lo_frac: float = 0.1, hi_frac: float = 0.9) -> dict:
    """Saturation throughput of an instantaneous burst: completions per second between the lo/hi completion
    quantiles (drops the ramp-up and the drain tail)."""
    done = sorted(r["done_at"] for r in recs if ok(r))
    if len(done) < 10:
        return {}
    a, b = done[int(len(done) * lo_frac)], done[int(len(done) * hi_frac) - 1]
    n = int(len(done) * hi_frac) - int(len(done) * lo_frac)
    toks = sorted(((r["done_at"], r.get("output_tokens") or 0, r.get("prompt_tokens") or 0) for r in recs if ok(r)))
    sel = [t for t in toks if a <= t[0] <= b]
    return {"rps": n / (b - a) if b > a else None,
            "out_tok_s": sum(t[1] for t in sel) / (b - a) if b > a else None,
            "in_tok_s": sum(t[2] for t in sel) / (b - a) if b > a else None, "n": n, "window_s": b - a}
