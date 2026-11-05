#!/usr/bin/env python3
"""Summary + figures of the autoscale-burst runs: results/autoscale-burst/<rev>/{summary.json, figures/*.png,
runs/<rid>/timeline.png}; README tables come from `--readme-tables` (markdown on stdout).

    .venv/bin/python scripts/asb_report.py --rev <rev> [--readme-tables]

Per run (all times relative to the client's start, replay seconds):
  recovery      per trigger k: trigger -> the new instance joined the router (healthy), -> its first dispatched
                request ("first request served"), -> its first completed request; and trigger 1 -> router queue
                back at/below the threshold for 10 s
  burst window  requests whose scheduled send time lies in the trace's burst (trace a: the Chutes window's burst
                minutes [0, burst_dur) rescaled; trace b: the high-rate segment): TTFT P99, TPOT P99, SLO-violation %
  run           the same over every request, max and integrated router queue (request-seconds), timeouts (429:
                admission timeout), 5xx, transport errors, instance-seconds (always-on instances from the client
                start, scaled-up ones from their trigger, to the last completion)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asb_metrics as am  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/autoscale-burst"
ARM_ORDER = ["static-base", "static-peak", "stock", "pm-gcs", "pm-peer", "pm-hot", "new-stock", "new-pm"]
ARM_LABEL = {"static-base": "1 static base", "static-peak": "2 static peak", "stock": "3 stock (standby)",
             "pm-gcs": "4 PM GCS (standby)", "pm-peer": "5 PM peer (standby)", "pm-hot": "6 PM hot standby",
             "new-stock": "7a stock (new capacity)", "new-pm": "7b PM (new capacity)"}
COLORS = {"static-base": "#7f7f7f", "static-peak": "#000000", "stock": "#d62728", "pm-gcs": "#1f77b4",
          "pm-peer": "#17becf", "pm-hot": "#2ca02c", "new-stock": "#ff7f0e", "new-pm": "#9467bd"}
BIN_S = 10.0


def med(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.median(vals) if vals else None


def burst_window(meta: dict) -> tuple[float, float]:
    if meta.get("kind") == "trace-a":
        k = meta["factor_k"]
        lead = -meta["rel_window_min"][0]
        end = min(meta["burst_dur_min"], meta["rel_window_min"][1])
        return lead * 60.0 / k, (lead + end) * 60.0 / k
    segs = meta.get("schedule_rps") or []
    # poisson: [{"dur_s", "rps", "n"}, ...] -> the highest-rate segment
    t, best = 0.0, None
    for seg in segs:
        dur, rate = float(seg["dur_s"]), float(seg["rps"])
        if best is None or rate > best[2]:
            best = (t, t + dur, rate)
        t += dur
    return (best[0], best[1]) if best else (0.0, float("inf"))


def load_run(d: Path) -> dict | None:
    rec = am.read_json(d / "run.json")
    if rec is None:
        return None
    lg = d / "loadgen"
    return {"dir": d, "rec": rec, "client": am.read_jsonl(lg / "client.jsonl"),
            "samples": am.read_jsonl(lg / "router/samples.jsonl"), "events": am.read_jsonl(lg / "router/events.jsonl"),
            "decisions": am.read_jsonl(lg / "router/decisions.jsonl"),
            "trace_meta": (am.read_json(lg / "trace.json") or {}).get("meta", {})}


def run_metrics(run: dict, slo: dict, threshold: float) -> dict:
    rec, client, samples = run["rec"], run["client"], run["samples"]
    t0 = rec.get("client_start_at") or (min(r["send_at"] for r in client) if client else 0.0)
    b0, b1 = burst_window(run["trace_meta"])
    burst = [r for r in client if b0 <= (r.get("sched_t") or 0.0) < b1]
    ends = [r["done_at"] for r in client if r.get("done_at")]
    t_end = max(ends) if ends else t0
    row = {"rid": rec["rid"], "arm": rec["arm"], "trace": rec["trace"], "rep": rec["rep"], "status": rec.get("status"),
           "error": rec.get("error"), "burst_window_s": [b0, b1], "duration_s": t_end - t0,
           "run": am.summarize(client, slo), "burst": am.summarize(burst, slo)}
    triggers = [d for d in run["decisions"] if d.get("kind") == "trigger"]
    row["triggers"] = [{"k": d["k"], "t": d["t"] - t0, "lane": d.get("lane"), "mode": d.get("mode")} for d in triggers]
    ev = run["events"]

    def first_event(kind, name):
        ts = [e["t_router"] for e in ev if e.get("kind") == kind and e.get("name") == name]
        return min(ts) if ts else None

    rec_rows = []
    for d in triggers:
        if d.get("mode") != "live":
            continue
        lane = d.get("lane")
        names = [lane, f"{lane}-ep"]
        joined = min([x for x in (first_event("joined", n) for n in names) if x] or [None]) if lane else None
        disp = min([x for x in (first_event("first_dispatch", n) for n in names) if x] or [None]) if lane else None
        done = min([x for x in (first_event("first_completion", n) for n in names) if x] or [None]) if lane else None
        rec_rows.append({"k": d["k"], "lane": lane, "trigger_s": d["t"] - t0,
                         "join_s": None if joined is None else joined - d["t"],
                         "first_served_s": None if disp is None else disp - d["t"],
                         "first_done_s": None if done is None else done - d["t"]})
    row["recovery"] = rec_rows
    row["recovery_first_served_s"] = rec_rows[0]["first_served_s"] if rec_rows else None
    row["recovery_join_s"] = rec_rows[0]["join_s"] if rec_rows else None
    # queue back at/below the threshold for >= 10 s, after trigger 1 (live or dry)
    qrec = None
    if triggers and samples:
        t1 = triggers[0]["t"]
        low_since = None
        for s in samples:
            if s["t"] < t1:
                continue
            if s["queue_depth"] <= threshold:
                low_since = s["t"] if low_since is None else low_since
                if s["t"] - low_since >= 10.0:
                    qrec = low_since - t1
                    break
            else:
                low_since = None
    row["queue_recovery_s"] = qrec
    q = [s["queue_depth"] for s in samples]
    row["queue_max"] = max(q) if q else None
    row["queue_integral_req_s"] = sum(q) * 1.0 if q else None  # 1-s samples
    st = row["run"]["status"]
    row["failures"] = {"timeouts_429": st.get("429", 0), "http_5xx": sum(v for k, v in st.items() if k.startswith("5")),
                       "transport_errors": st.get("error", 0) + st.get("None", 0)}
    # instance-seconds
    inst_s = 0.0
    n_static = 1 + (len(rec.get("lanes_order", [])) if rec["arm"] == "static-peak" else 0)
    inst_s += n_static * (t_end - t0)
    for r in rec_rows:
        inst_s += max(0.0, t_end - (t0 + r["trigger_s"]))
    row["instance_seconds"] = inst_s
    row["instances_max"] = max((s.get("active", 0) for s in samples), default=None)
    row["cost"] = chip_cost(rec, rec_rows, triggers, t0, t_end)
    n_met = sum(1 for r in client if am.meets_slo(r, slo))
    row["slo_met_requests"] = n_met
    row["chip_s_per_slo_met_request"] = row["cost"]["chip_s"] / n_met if n_met else None
    row["gke"] = []
    if rec["arm"] in NEW_ARMS:
        for r in rec_rows:
            g = ((rec.get("lanes") or {}).get(r["lane"]) or {}).get("gke") or {}
            row["gke"].append({"k": r["k"], "lane": r["lane"], **{key: g.get(key) for key in GKE_KEYS},
                               "failure_reasons": g.get("failure_reasons")})
    ver = rec.get("verification") or {}
    row["greedy_ok"] = {ln: bool((v.get("greedy") or {}).get("passed")) for ln, v in ver.items()}
    row["mechanism_ok"] = {ln: (v.get("mechanism") or {}).get("ok") for ln, v in ver.items()}
    row["mechanism"] = {ln: {k: (v.get("mechanism") or {}).get(k) for k in
                             ("snapshot_restored", "bank_installed", "xla_backend_compile_s", "gate_wait_s",
                              "load_format", "prespawn")} for ln, v in ver.items()}
    acts = rec.get("actuations") or []
    row["actuations"] = [{k: a.get(k) for k in ("k", "lane", "boot", "deploy_ok", "error")} |
                         {"seen_after_trigger_s": (a["seen_at"] - a["trigger_t"]) if a.get("trigger_t") else None,
                          "actuation_s": (a["done_at"] - a["trigger_t"]) if a.get("trigger_t") and a.get("done_at")
                          else None} for a in acts]
    booted = {a["lane"] for a in acts if a.get("deploy_ok")} | (set(rec.get("lanes_order", []))
                                                                  if rec["arm"] == "static-peak" else set())
    row["new_instances"] = sorted(booted)
    row["all_greedy_ok"] = all(row["greedy_ok"].get(ln) for ln in booted) if booted else True
    row["all_mechanism_ok"] = all(row["mechanism_ok"].get(ln) for ln in booted) if booted else True
    row["valid"] = bool(row["status"] == "ok" and row["all_greedy_ok"] and row["all_mechanism_ok"])
    row["peer_dip"] = peer_dip(run) if rec["arm"] == "pm-peer" else None
    return row


def token_rate(client: list[dict], t0: float, bin_s: float = 1.0) -> dict[str, dict[int, float]]:
    """Per instance output tokens/s: each request's tokens spread uniformly over its decode interval."""
    out: dict[str, dict[int, float]] = {}
    for r in client:
        if not am.ok(r) or not r.get("engine_started_at_s"):
            continue
        a = r["engine_started_at_s"] + (r.get("engine_ttft_s") or 0.0) - t0
        b = r["engine_finished_at_s"] - t0
        n = r.get("output_tokens") or 0
        if n <= 0 or b <= a:
            continue
        bins = out.setdefault(r.get("instance") or "?", {})
        rate = n / (b - a)
        x = a
        while x < b:
            nxt = min(b, (math.floor(x / bin_s) + 1) * bin_s)
            k = int(math.floor(x / bin_s))
            bins[k] = bins.get(k, 0.0) + rate * (nxt - x) / bin_s
            x = nxt
    return out


def peer_dip(run: dict) -> dict | None:
    """The serving base instance's output tokens/s while it streamed state to a pm-peer target (the transfer
    window of the target's state delivery) vs the 30 s before."""
    rec, client = run["rec"], run["client"]
    t0 = rec.get("client_start_at") or 0.0
    base = rec.get("base_lane")
    rates = token_rate(client, t0).get(base, {})
    rows = []
    for a in rec.get("actuations") or []:
        dep = am.read_json(run["dir"] / "lanes" / a["lane"] / "deploy.json")
        try:
            fetch = dep["measurements"]["deployment"]["state_delivery"]["fetch"]
            ts, te = fetch["transfer_started_at"] - t0, fetch["ended_at"] - t0
        except Exception:  # noqa: BLE001
            continue
        during = [rates.get(k, 0.0) for k in range(int(ts), int(te) + 1)]
        before = [rates.get(k, 0.0) for k in range(int(ts) - 30, int(ts))]
        rows.append({"lane": a["lane"], "transfer_s": te - ts, "start_s": ts,
                     "base_tok_s_during": statistics.mean(during) if during else None,
                     "base_tok_s_before": statistics.mean(before) if before else None,
                     "gbps": fetch.get("transfer_gbps")})
    return {"transfers": rows} if rows else None


def timeline(run: dict, slo: dict) -> dict:
    rec, client, samples = run["rec"], run["client"], run["samples"]
    t0 = rec.get("client_start_at") or 0.0
    nb = int(max([r.get("sched_t") or 0 for r in client] + [0]) // BIN_S) + 1
    bins = [{"t": i * BIN_S, "n": 0, "ttft": [], "tpot": []} for i in range(nb)]
    for r in client:
        i = int((r.get("sched_t") or 0.0) // BIN_S)
        if 0 <= i < nb:
            bins[i]["n"] += 1
            bins[i]["ttft"].append(r["ttft_s"] if am.ok(r) else float("inf"))
            if am.ok(r) and r.get("tpot_s") is not None:
                bins[i]["tpot"].append(r["tpot_s"])
    rows = [{"t": b["t"], "rate": b["n"] / BIN_S, "ttft_p50": am.pct(b["ttft"], 50), "ttft_p99": am.pct(b["ttft"], 99),
             "tpot_p99": am.pct(b["tpot"], 99)} for b in bins]
    qs = [{"t": s["t"] - t0, "queue": s["queue_depth"], "active": s.get("active")} for s in samples]
    return {"bins": rows, "samples": qs}


def plot_run(run: dict, m: dict, slo: dict, threshold: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tl = timeline(run, slo)
    t = [b["t"] for b in tl["bins"]]
    fig, ax = plt.subplots(4, 1, figsize=(9, 8.5), sharex=True)
    ax[0].plot(t, [b["rate"] for b in tl["bins"]], color="k", lw=1, label="request rate (req/s, 10 s)")
    ax0b = ax[0].twinx()
    ax0b.step([s["t"] for s in tl["samples"]], [s["active"] for s in tl["samples"]], where="post", color="tab:green",
              lw=1.5, label="instances serving")
    ax0b.set_ylim(0, 5)
    ax0b.set_ylabel("instances")
    ax[0].set_ylabel("req/s")
    ax[1].plot([s["t"] for s in tl["samples"]], [s["queue"] for s in tl["samples"]], color="tab:red", lw=1)
    ax[1].axhline(threshold, color="tab:red", ls=":", lw=0.8)
    ax[1].set_ylabel("router queue")
    ax[2].plot(t, [b["ttft_p50"] for b in tl["bins"]], lw=1, label="TTFT P50")
    ax[2].plot(t, [b["ttft_p99"] if b["ttft_p99"] != float("inf") else None for b in tl["bins"]], lw=1, label="TTFT P99")
    ax[2].axhline(slo["ttft_s"], color="gray", ls="--", lw=0.8, label="SLO")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("TTFT (s)")
    ax[2].legend(fontsize=7, loc="upper left")
    ax[3].plot(t, [b["tpot_p99"] for b in tl["bins"]], lw=1, color="tab:purple", label="TPOT P99")
    ax[3].axhline(slo["tpot_s"], color="gray", ls="--", lw=0.8)
    ax[3].set_ylabel("TPOT P99 (s)")
    ax[3].set_xlabel("replay time (s)")
    for a in ax:
        b0, b1 = m["burst_window_s"]
        a.axvspan(b0, b1, color="orange", alpha=0.06)
        for tr in m["triggers"]:
            a.axvline(tr["t"], color="tab:red", lw=0.7, ls="--")
        for r in m["recovery"]:
            if r["join_s"] is not None:
                a.axvline(r["trigger_s"] + r["join_s"], color="tab:green", lw=0.7)
    ax[0].set_title(f"{m['rid']}: {ARM_LABEL.get(m['arm'], m['arm'])} (red dashed: triggers, green: joins)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_summary(runs_by_arm: dict, metrics_by_arm: dict, slo: dict, threshold: float, trace: str, figdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    arms = [a for a in ARM_ORDER if a in runs_by_arm]
    # overlay timeline (first valid run of each arm)
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for arm in arms:
        run = next((r for r, m in zip(runs_by_arm[arm], metrics_by_arm[arm]) if m["valid"]), runs_by_arm[arm][0])
        tl = timeline(run, slo)
        t = [b["t"] for b in tl["bins"]]
        c = COLORS.get(arm)
        if arm == arms[0]:
            ax[0].plot(t, [b["rate"] for b in tl["bins"]], color="k", lw=1, label="request rate")
        ax[1].step([s["t"] for s in tl["samples"]], [s["active"] for s in tl["samples"]], where="post", color=c, lw=1.3,
                   label=ARM_LABEL[arm])
        ax[2].plot([s["t"] for s in tl["samples"]], [s["queue"] for s in tl["samples"]], color=c, lw=1)
        ax[3].plot(t, [b["ttft_p99"] if b["ttft_p99"] != float("inf") else 1e4 for b in tl["bins"]], color=c, lw=1)
    ax[0].set_ylabel("req/s (10 s)")
    ax[1].set_ylabel("instances serving")
    ax[1].legend(fontsize=7, ncol=2)
    ax[2].set_ylabel("router queue")
    ax[2].axhline(threshold, color="gray", ls=":", lw=0.8)
    ax[3].set_ylabel("TTFT P99 (s, 10 s bins)")
    ax[3].set_yscale("log")
    ax[3].axhline(slo["ttft_s"], color="gray", ls="--", lw=0.8)
    ax[3].set_xlabel("replay time (s)")
    ax[0].set_title(f"trace {trace}: one run per arm (r1 unless invalid)")
    fig.tight_layout()
    p = figdir / f"timeline-{trace}.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    out.append(p.name)
    # CDFs (burst window, all valid runs pooled)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for arm in arms:
        ttft, tpot = [], []
        for run, m in zip(runs_by_arm[arm], metrics_by_arm[arm]):
            b0, b1 = m["burst_window_s"]
            for r in run["client"]:
                if b0 <= (r.get("sched_t") or 0) < b1:
                    ttft.append(r["ttft_s"] if am.ok(r) else float("inf"))
                    if am.ok(r) and r.get("tpot_s") is not None:
                        tpot.append(r["tpot_s"])
        for a, vals in ((ax[0], ttft), (ax[1], tpot)):
            vals = sorted(vals)
            if vals:
                a.plot(vals, [(i + 1) / len(vals) for i in range(len(vals))], color=COLORS.get(arm), lw=1.2,
                       label=ARM_LABEL[arm])
    ax[0].set_xscale("log")
    ax[0].axvline(slo["ttft_s"], color="gray", ls="--", lw=0.8)
    ax[0].set_xlabel("TTFT (s), burst window (failed = off-axis)")
    ax[1].axvline(slo["tpot_s"], color="gray", ls="--", lw=0.8)
    ax[1].set_xlabel("TPOT (s), burst window")
    ax[0].set_ylabel("CDF")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    p = figdir / f"cdf-{trace}.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    out.append(p.name)
    # bars: SLO violation % and recovery
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, arm in enumerate(arms):
        ms = [m for m in metrics_by_arm[arm] if m["valid"]]
        v = [m["burst"].get("slo_violation_pct") for m in ms]
        r = [m["recovery_first_served_s"] for m in ms if m["recovery_first_served_s"] is not None]
        ax[0].bar(i, med(v) or 0, color=COLORS.get(arm))
        ax[0].scatter([i] * len(v), v, color="k", s=8, zorder=3)
        if r:
            ax[1].bar(i, med(r), color=COLORS.get(arm))
            ax[1].scatter([i] * len(r), r, color="k", s=8, zorder=3)
    for a in ax:
        a.set_xticks(range(len(arms)))
        a.set_xticklabels([ARM_LABEL[x] for x in arms], rotation=35, ha="right", fontsize=7)
    ax[0].set_ylabel("SLO violation % (burst window)")
    ax[1].set_ylabel("trigger 1 -> first request served by the new instance (s)")
    fig.tight_layout()
    p = figdir / f"bars-{trace}.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    out.append(p.name)
    return out


CHIPS_PER_HOST = 4  # ct5lp-hightpu-4t; a 2x4 slice = 2 hosts = 8 chips
NEW_ARMS = ("new-stock", "new-pm")
STANDBY_ARMS = ("stock", "pm-gcs", "pm-peer", "pm-hot")
GKE_KEYS = ("vm_created", "node_registered", "node_ready", "pod_scheduled", "image_pulled", "container_started",
            "ray_joined")


def chip_cost(rec: dict, rec_rows: list[dict], triggers: list[dict], t0: float, t_end: float) -> dict:
    """Chip-seconds over the run window [client start, last completion], 4 chips per host, split by state:
    idle (slice held, no engine), booting (trigger or acquisition -> the instance joined the router), serving
    (joined -> end). A slice counts from the moment the arm holds it: the base and static-peak slices and the
    standby slices (arms 3-6) for the whole window (they are held before the trigger), a new-capacity slice (arm 7)
    per host from its worker pod's PodScheduled time (the GKE wait before it is not held capacity)."""
    lanes = rec.get("lanes") or {}
    trig = {d.get("lane"): d["t"] for d in triggers if d.get("mode") == "live"}
    joined = {r["lane"]: t0 + r["trigger_s"] + r["join_s"] for r in rec_rows if r.get("join_s") is not None}

    def clamp(t: float) -> float:
        return min(max(t, t0), t_end)

    def host(lane: str, role: str, held: float, boot_from: float | None, serve_from: float | None) -> dict:
        held = clamp(held)
        b = clamp(boot_from) if boot_from is not None else None
        s = clamp(serve_from) if serve_from is not None else None
        idle_end = b if b is not None else (s if s is not None else t_end)
        boot_end = s if s is not None else t_end
        idle = max(0.0, idle_end - held)
        boot = max(0.0, boot_end - max(held, b)) if b is not None else 0.0
        serve = max(0.0, t_end - max(held, s)) if s is not None else 0.0
        return {"lane": lane, "role": role, "held_s": round(t_end - held, 1), "idle_chip_s": CHIPS_PER_HOST * idle,
                "booting_chip_s": CHIPS_PER_HOST * boot, "serving_chip_s": CHIPS_PER_HOST * serve}

    hosts: list[dict] = []
    arm = rec["arm"]
    for _ in range(2):  # the bed's base instance serves the whole window
        hosts.append(host(rec.get("base_lane", "base"), "base", t0, None, t0))
    for ln in rec.get("lanes_order", []):
        if arm == "static-peak":
            hosts += [host(ln, "static", t0, None, t0) for _ in range(2)]
        elif arm in STANDBY_ARMS:
            hosts += [host(ln, "standby", t0, trig.get(ln), joined.get(ln) if ln in trig else None)
                      for _ in range(2)]
        elif arm in NEW_ARMS and ln in trig:
            raw = (lanes.get(ln) or {}).get("gke_raw") or {}
            sched = [p.get("cond_PodScheduled") for p in raw.get("pods", []) if p.get("cond_PodScheduled")]
            basis = "pod_scheduled"
            if len(sched) < 2:  # no GKE record: the Ray join is a lower bound of the hold
                act = next((a for a in rec.get("actuations") or [] if a.get("lane") == ln), {})
                sched = [act.get("ray_joined_at") or joined.get(ln, t_end)] * 2
                basis = "ray_joined (lower bound)"
            for t_s in sched[:2]:
                h = host(ln, "new", t_s, t_s, joined.get(ln))
                h["basis"] = basis
                hosts.append(h)
    tot = {k: sum(h[k] for h in hosts) for k in ("idle_chip_s", "booting_chip_s", "serving_chip_s")}
    return {"chip_s": sum(tot.values()), **tot, "window_s": t_end - t0, "hosts": hosts}


def scalar_rows(m: dict) -> dict:
    return {"burst_ttft_p99": m["burst"].get("ttft_p99"), "burst_tpot_p99": m["burst"].get("tpot_p99"),
            "burst_slo_violation_pct": m["burst"].get("slo_violation_pct"),
            "run_slo_violation_pct": m["run"].get("slo_violation_pct"), "run_ttft_p99": m["run"].get("ttft_p99"),
            "recovery_join_s": m["recovery_join_s"], "recovery_first_served_s": m["recovery_first_served_s"],
            "queue_recovery_s": m["queue_recovery_s"], "queue_max": m["queue_max"],
            "queue_integral_req_s": m["queue_integral_req_s"], "timeouts_429": m["failures"]["timeouts_429"],
            "http_5xx": m["failures"]["http_5xx"], "transport_errors": m["failures"]["transport_errors"],
            "instance_seconds": m["instance_seconds"], "instances_max": m["instances_max"],
            "triggers": len(m["triggers"]),
            "run_slo_attainment_pct": m["run"].get("slo_attainment_pct"),
            "burst_slo_attainment_pct": m["burst"].get("slo_attainment_pct"),
            "chip_s": m["cost"]["chip_s"], "chip_s_serving": m["cost"]["serving_chip_s"],
            "chip_s_booting": m["cost"]["booting_chip_s"], "chip_s_idle": m["cost"]["idle_chip_s"],
            "chip_s_per_slo_met_request": m["chip_s_per_slo_met_request"],
            "gke_pod_scheduled_s": med([g.get("pod_scheduled") for g in m["gke"]]),
            "gke_ray_joined_s": med([g.get("ray_joined") for g in m["gke"]])}


def build(rev: str, plots: bool = True) -> dict:
    rdir = OUT_ROOT / rev
    conf = json.loads((rdir / "config.json").read_text())
    slo, threshold = conf["slo"], conf["threshold"]
    runs: dict[tuple, list] = {}
    for d in sorted((rdir / "runs").iterdir()) if (rdir / "runs").exists() else []:
        run = load_run(d)
        if run is None:
            continue
        runs.setdefault((run["rec"]["trace"], run["rec"]["arm"]), []).append(run)
    figdir = rdir / "figures"
    figdir.mkdir(exist_ok=True)
    summary: dict = {"rev": rev, "config": conf, "arms": {}, "figures": []}
    for trace in sorted({k[0] for k in runs}):
        by_arm_runs, by_arm_metrics = {}, {}
        for arm in ARM_ORDER:
            rs = sorted(runs.get((trace, arm), []), key=lambda r: r["rec"]["rep"])
            if not rs:
                continue
            ms = [run_metrics(r, slo, threshold) for r in rs]
            by_arm_runs[arm], by_arm_metrics[arm] = rs, ms
            valid = [m for m in ms if m["valid"]]
            scal = [scalar_rows(m) for m in valid]
            block = {"n": len(valid), "n_attempted": len(ms), "runs": [m["rid"] for m in valid],
                     "invalid": [{"rid": m["rid"], "status": m["status"], "error": (m["error"] or "")[:300],
                                  "greedy_ok": m["greedy_ok"], "mechanism_ok": m["mechanism_ok"]}
                                 for m in ms if not m["valid"]],
                     "median": {k: med([s[k] for s in scal]) for k in (scal[0] if scal else {})},
                     "per_run": [{"rid": m["rid"], **scalar_rows(m), "recovery": m["recovery"],
                                  "actuations": m["actuations"], "mechanism": m["mechanism"],
                                  "greedy_ok": m["greedy_ok"], "peer_dip": m["peer_dip"], "cost": m["cost"],
                                  "gke": m["gke"]} for m in ms]}
            summary["arms"].setdefault(trace, {})[arm] = block
            if plots:
                for r, m in zip(rs, ms):
                    plot_run(r, m, slo, threshold, r["dir"] / "timeline.png")
        if plots and by_arm_runs:
            summary["figures"] += plot_summary(by_arm_runs, by_arm_metrics, slo, threshold, trace, figdir)
    (rdir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def readme_tables(summary: dict) -> str:
    lines = []
    for trace, arms in summary["arms"].items():
        lines.append(f"\n### Trace {trace}\n")
        lines.append("| arm | n | recovery: trigger -> first served (s) | trigger -> queue <= thr (s) | burst TTFT P99 (s) "
                     "| burst TPOT P99 (s) | burst SLO viol. % | run SLO viol. % | max queue | queue integral (req-s) "
                     "| 429 / 5xx | instance-s | triggers |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        cost, gke = [], []
        for arm in ARM_ORDER:
            if arm not in arms:
                continue
            b = arms[arm]
            m = b["median"]
            per = [p for p in b["per_run"] if p["rid"] in b["runs"]]

            def runs_of(key, nd=1):
                return ", ".join(fmt(p.get(key), nd) for p in per)

            def cell(key, nd=1):
                return f"{fmt(m.get(key), nd)} ({runs_of(key, nd)})"
            lines.append(
                f"| {ARM_LABEL[arm]} | {b['n']} | {cell('recovery_first_served_s')} | {cell('queue_recovery_s')} "
                f"| {cell('burst_ttft_p99')} | {cell('burst_tpot_p99', 3)} | {cell('burst_slo_violation_pct')} "
                f"| {fmt(m.get('run_slo_violation_pct'))} | {cell('queue_max', 0)} "
                f"| {fmt(m.get('queue_integral_req_s'), 0)} | {fmt(m.get('timeouts_429'), 0)} / {fmt(m.get('http_5xx'), 0)} "
                f"| {fmt(m.get('instance_seconds'), 0)} | {fmt(m.get('triggers'), 0)} |")
            cost.append(f"| {ARM_LABEL[arm]} | {b['n']} | {cell('chip_s', 0)} | {fmt(m.get('chip_s_serving'), 0)} "
                        f"| {fmt(m.get('chip_s_booting'), 0)} | {fmt(m.get('chip_s_idle'), 0)} "
                        f"| {cell('run_slo_attainment_pct')} | {cell('burst_slo_attainment_pct')} "
                        f"| {cell('chip_s_per_slo_met_request', 2)} |")
            for p in per:
                for g, r in zip(p.get("gke") or [], p.get("recovery") or []):
                    gke.append(f"| {p['rid']} | {g['k']} ({g['lane']}) | {fmt(g.get('pod_scheduled'))} "
                               f"| {fmt(g.get('node_ready'))} | {fmt(g.get('ray_joined'))} "
                               f"| {fmt(r.get('first_served_s'))} | {', '.join(x for x in g.get('failure_reasons') or [] if x != 'FailedScheduling') or '-'} |")
        lines.append(f"\nCost and SLO attainment, trace {trace} (chip-seconds over the run window, 8 chips per slice; "
                     "serving / booting / idle are medians; SLO attainment = % of requests with TTFT <= SLO and "
                     "TPOT <= SLO, failed requests count as misses):\n")
        lines.append("| arm | n | chip-s | serving | booting | idle | run SLO attainment % | burst SLO attainment % "
                     "| chip-s per SLO-met request |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        lines += cost
        if gke:
            lines.append(f"\nNew capacity (arm 7), trace {trace}: seconds from the GKE trigger (RayCluster replicas 0 -> 1) "
                         "to the slice's last host at each stage, and controller trigger -> first served:\n")
            lines.append("| run | trigger (lane) | pod scheduled | node ready | Ray joined | first served | GKE failures (besides the pending pod's FailedScheduling) |")
            lines.append("|---|---|---|---|---|---|---|")
            lines += gke
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--readme-tables", action="store_true")
    args = ap.parse_args()
    summary = build(args.rev, plots=not args.no_plots)
    if args.readme_tables:
        print(readme_tables(summary))
    else:
        for trace, arms in summary["arms"].items():
            for arm, b in arms.items():
                print(trace, arm, b["n"], json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                                       for k, v in b["median"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
