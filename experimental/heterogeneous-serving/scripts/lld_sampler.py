#!/usr/bin/env python3
"""1-s sampler of the llm-d EPP's Prometheus endpoint (runs in the loadgen pod, same clock as the client).

    lld_sampler.py --url http://lld-s-epp.lld-s.svc:9090/metrics --out samples.jsonl [--period 1.0] [--until EPOCH]

Each line: {"t": epoch, "dt": scrape seconds, "m": {"<name>{<labels>}": value}} for the metric families whose
name starts with one of PREFIXES (EPP flow control, pool, objective, scheduler).  Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

PREFIXES = ("llm_d_epp_flow_control", "inference_pool", "inference_objective_running", "llm_d_epp_pool",
            "inference_extension_scheduler_e2e_duration_seconds_count", "llm_d_epp_saturation")


def parse(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#" or not line.startswith(PREFIXES):
            continue
        key, _, val = line.rpartition(" ")
        try:
            out[key] = float(val)
        except ValueError:
            pass
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--period", type=float, default=1.0)
    p.add_argument("--until", type=float, default=None)
    a = p.parse_args()
    nxt = time.time()
    with open(a.out, "a", buffering=1) as fh:
        while a.until is None or time.time() < a.until:
            t = time.time()
            row = {"t": t}
            try:
                with urllib.request.urlopen(a.url, timeout=max(0.5, a.period * 0.9)) as r:
                    row["m"] = parse(r.read().decode())
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            row["dt"] = round(time.time() - t, 4)
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            nxt += a.period
            time.sleep(max(0.0, nxt - time.time()))


if __name__ == "__main__":
    main()
