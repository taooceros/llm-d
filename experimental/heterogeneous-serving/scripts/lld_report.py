#!/usr/bin/env python3
"""Analysis of the llm-d autoscaling runs (scripts/lld.py run): per-run metrics, the scale-up pipeline breakdown,
chip-second cost, per-arm medians, figures and the README tables.

    lld_report.py run <run dir>                      # one run -> <run dir>/metrics.json (+ timeline.png)
    lld_report.py build --rev <rev> [--readme-tables] # every run of results/llmd-autoscale/<rev>/runs ->
                                                     # summary.json, figures/, README tables

Clocks: client/sampler records are the loadgen pod's clock, engine timestamps the TPU hosts', marks the pods',
k8s condition/event times the API server's (1 s resolution), watcher rows the workstation's receipt time.
Per-request metrics (TTFT/TPOT/SLO) follow scripts/asb_metrics.py (TTFT = gateway queue + engine TTFT, with
the queue = engine arrival - client send behind the EPP).
"""
from __future__ import annotations

import argparse
import calendar
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import asb_metrics as am  # noqa: E402

RESULTS = ROOT / "results/llmd-autoscale"
SLO = {"ttft_s": 0.7882183790206909, "tpot_s": 0.06634625338868454}  # AutoscaleBurst 4daec1af/config.json
CHIPS = 8
TOL = 0.1  # HPA tolerance
# Cost window, the same for every arm: [client start, client start + trace duration + DRAIN_S]. DRAIN_S covers the last
# requests' completion (client end <= trace end + 16.1 s in every counted autoscaled run). Everything after it (the
# 300 s scale-down window, the teardown preStop checks, the CA's release wait for emptied spot slices) is reported
# separately as the "to idle" window: [client start, bed back at its initial size (+ new slices' nodes deleted)].
DRAIN_S = 20.0


GRACE_S = 1800  # LWS pods' terminationGracePeriodSeconds: deletionTimestamp = delete request + grace


def del_t(p: dict) -> float | None:
    d = ts(p.get("deleted")) if p else None
    return None if d is None else d - GRACE_S


def ts(s: str | None) -> float | None:
    if not s:
        return None
    s = s.rstrip("Z").split(".")[0]
    return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%S")))


def med(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(v) if v else None


def qty(v: str | None) -> float | None:
    if v is None:
        return None
    v = str(v)
    mult = {"m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9}  # Kubernetes resource.Quantity suffixes (decimal)
    return float(v[:-1]) * mult[v[-1]] if v and v[-1] in mult else float(v)


def burst_window(meta: dict) -> tuple[float, float]:
    if meta.get("kind") == "trace-a":
        k = meta["factor_k"]
        lead = -meta["rel_window_min"][0]
        end = min(meta["burst_dur_min"], meta["rel_window_min"][1])
        return lead * 60.0 / k, (lead + end) * 60.0 / k
    segs = meta.get("schedule_rps") or []
    t, best = 0.0, None
    for seg in segs:
        dur, rate = float(seg["dur_s"]), float(seg["rps"])
        if best is None or rate > best[2]:
            best = (t, t + dur, rate)
        t += dur
    return (best[0], best[1]) if best else (0.0, float("inf"))


def union_len(iv: list[tuple[float, float]], lo: float, hi: float) -> float:
    tot, cur = 0.0, None
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in iv if b > lo and a < hi):
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + (cur[1] - cur[0] if cur else 0.0)


# ---------------------------------------------------------------- loading

def load(d: Path) -> dict:
    run = am.read_json(d / "run.json")
    lg = d / "loadgen"
    pods = {}
    for p in sorted((d / "pods").glob("*")) if (d / "pods").exists() else []:
        pods[p.name] = {f.name.split(".")[0]: am.read_json(f.with_suffix("") if f.suffix == ".gz" else f)
                        for f in p.iterdir()}
    prom = {f.stem: am.read_json(f) for f in (d / "prom").glob("*.json")} if (d / "prom").exists() else {}
    return {"dir": d, "run": run, "client": am.read_jsonl(lg / "client.jsonl"),
            "samples": am.read_jsonl(lg / "samples.jsonl"), "trace": am.read_json(lg / "trace.json") or {},
            "state": am.read_jsonl(d / "k8s/state.jsonl"), "metric": am.read_jsonl(d / "k8s/metric.jsonl"),
            "nodes": am.read_jsonl(d / "k8s/nodes.jsonl"), "events": am.read_jsonl(d / "k8s/events.jsonl"),
            "prom": prom, "pods": pods}


def queue_series(R: dict) -> list[tuple[float, float]]:
    return sampler_series(R, "llm_d_epp_flow_control_queue_size{")


def running_series(R: dict) -> list[tuple[float, float]]:
    """EPP in-flight requests (the HPA's epp_running_requests = sum over the pool's objectives)."""
    return sampler_series(R, "inference_objective_running_requests")


def sampler_series(R: dict, prefix: str) -> list[tuple[float, float]]:
    out = []
    for s in R["samples"]:
        m = s.get("m")
        if m is None:
            continue
        out.append((s["t"], sum(v for k, v in m.items() if k.startswith(prefix))))
    return out


def prom_samples(R: dict, name: str = "epp_queue") -> list[tuple[float, float]]:
    """Raw scraped samples (summed over series) of prom/<name>.json: epp_queue (run.py), running (fetch-prom)."""
    try:
        vals = R["prom"][name]["data"]["result"]
    except (KeyError, TypeError):
        return []
    acc: dict[float, float] = {}
    for series in vals:
        for t, v in series.get("values", []):
            acc[float(t)] = acc.get(float(t), 0.0) + float(v)
    return sorted(acc.items())


def fetch_prom(d: Path) -> None:
    """Report-side: raw Prometheus samples of the HPA's second metric (EPP running requests) over the run window,
    into <run>/prom/running.json (run.py pulls only the queue and ready pods)."""
    import subprocess  # noqa: PLC0415
    from urllib.parse import urlencode  # noqa: PLC0415
    run = am.read_json(d / "run.json")
    mk = run["marks"]
    t0, t1 = mk.get("hpa_active", mk["prep_start"]), mk.get("end", mk.get("teardown_done", time.time()))
    q = urlencode({"query": f'inference_objective_running_requests{{job="lld-{run["bed"]}-epp"}}[{int(t1 - t0) + 60}s]',
                   "time": f"{t1:.3f}"})
    env = {**__import__("os").environ, "KUBECONFIG": str(Path.home() / ".kube/lld-usw1.yaml")}
    out = subprocess.run(["kubectl", "get", "--raw", "/api/v1/namespaces/llm-d-monitoring/services/"
                          f"http:llmd-kube-prometheus-stack-prometheus:9090/proxy/api/v1/query?{q}"],
                         capture_output=True, text=True, env=env, timeout=120, check=True).stdout
    (d / "prom").mkdir(exist_ok=True)
    (d / "prom/running.json").write_text(out)


# ---------------------------------------------------------------- per-run analysis

def pod_index(R: dict) -> dict:
    """pod name -> last seen row + first/last seen times (+ uid)."""
    idx: dict[str, dict] = {}
    for row in R["state"]:
        for p in row["pods"]:
            if p.get("worker") is None:
                continue
            key = f"{p['name']}-{p['uid'][:8]}"
            e = idx.setdefault(key, {"first_t": row["t"], "rows": 0})
            e.update(last_t=row["t"], last=p, rows=e["rows"] + 1)
            if p.get("ready") and "ready_t" not in e:  # first Ready: the condition's transition time (1 s)
                c = (p.get("cond") or {}).get("Ready")
                e["ready_t"] = ts(c[1]) if c and c[0] == "True" else row["t"]
                e["ready_seen"] = row["t"]
            if p.get("deleted") and "deleted_seen" not in e:
                e["deleted_seen"] = row["t"]
    return idx


def spec_series(R: dict) -> list[tuple[float, int]]:
    return [(r["t"], r["lws"]["spec_replicas"]) for r in R["state"] if r.get("lws")]


def decisions(R: dict) -> list[dict]:
    """HPA scale decisions = increases (and decreases) of the LWS spec.replicas, with the HPA's event time."""
    out, prev = [], None
    for t, n in spec_series(R):
        if prev is not None and n != prev:
            out.append({"t_seen": t, "from": prev, "to": n})
        prev = n
    ev = [e for e in R["events"] if e.get("reason") == "SuccessfulRescale"]
    for d in out:
        m = [(ts(e.get("last") or e.get("event_time") or e.get("first")), e) for e in ev
             if f"New size: {d['to']}" in (e.get("msg") or "")]
        m = [(c, e) for c, e in m if c is not None and d["t_seen"] - 30 <= c <= d["t_seen"] + 2]
        best = max(m, key=lambda x: x[0]) if m else None
        d["t_event"] = best[0] if best else None
        d["t"] = d["t_event"] if d["t_event"] is not None else d["t_seen"]
        d["msg"] = best[1]["msg"] if best else None
    return out


def leader_marks(R: dict, leaf: str) -> dict:
    mk = (R["pods"].get(leaf) or {}).get("marks") or []
    out = {}
    for m in mk:
        out.setdefault(m["event"], m)
    return out


def analyze(d: Path) -> dict:
    R = load(d)
    run = R["run"]
    t0 = run["client_start_at"]
    client = R["client"]
    meta = (R["trace"] or {}).get("meta", {})
    b0, b1 = burst_window(meta)
    burst = [r for r in client if b0 <= (r.get("sched_t") or 0.0) < b1]
    ends = [r["done_at"] for r in client if r.get("done_at")]
    t_client_end = max(ends) if ends else t0
    # arm key: the fewer-slices arms are run as s-pm with HPA maxReplicas 3 / 2 (plan reps 11-13 / 21-23)
    mr = int(run.get("max_replicas") or 4)
    arm_key = f"{run['arm']}-max{mr}" if run.get("hpa") and mr != 4 else run["arm"]
    row = {"rid": run["rid"], "arm": arm_key, "bed": run["bed"], "boot": run["boot"], "rep": run["rep"],
           "max_replicas": mr,
           "trace": run["trace"], "status": run["status"], "error": run.get("error"),
           "hpa": run.get("hpa"), "burst_window_s": [b0, b1], "duration_s": t_client_end - t0,
           "run": am.summarize(client, SLO), "burst": am.summarize(burst, SLO)}
    q = queue_series(R)
    row["queue_max"] = max((v for _, v in q), default=None)
    row["queue_integral_req_s"] = sum(v for _, v in q)  # 1-s samples
    st = row["run"]["status"]
    row["failures"] = {"http_429": st.get("429", 0), "http_5xx": sum(v for k, v in st.items() if k.startswith("5")),
                       "transport": st.get("error", 0) + st.get("None", 0)}
    # ---------------- scale-ups
    hm = (run.get("hpa") or {}).get("metrics") or []
    target = float(hm[0].get("external", {}).get("target", {}).get("value", "nan")) if hm else None
    run_target = float(hm[1]["external"]["target"]["averageValue"]) if len(hm) > 1 else None
    dec = decisions(R)
    pidx = pod_index(R)
    prom = prom_samples(R)
    prom_run = prom_samples(R, "running")
    ext = [(m["t"], qty(m.get("value"))) for m in R["metric"] if m.get("value") is not None]
    runq = running_series(R)
    cur = [(r["t"], (r.get("hpa") or {}).get("current") or (r.get("lws") or {}).get("status") or 1) for r in R["state"]]

    def replicas_at(t: float) -> int:
        n = cur[0][1] if cur else 1
        for tt, v in cur:
            if tt > t:
                break
            n = v
        return max(1, int(n))
    # one entry per group incarnation (a group index can be deleted by a scale-down and re-created by a later
    # scale-up): the leader pod + the worker of the same index created right after it
    groups: dict[str, dict] = {}
    leaders = [(leaf, e) for leaf, e in pidx.items() if e["last"]["worker"] == "0"]
    workers = [(leaf, e) for leaf, e in pidx.items() if e["last"]["worker"] != "0"]
    for leaf, e in leaders:
        g, tc = int(e["last"]["group"]), ts(e["last"].get("created")) or 0.0
        grp = groups.setdefault(leaf, {"g": g, "created": tc, "pods": {"leader": {"leaf": leaf, **e}}})
        cands = [(abs((ts(w["last"].get("created")) or 0.0) - tc), wl, w) for wl, w in workers
                 if int(w["last"]["group"]) == g and -5 <= (ts(w["last"].get("created")) or 0.0) - tc <= 120]
        if cands:
            _, wl, w = min(cands, key=lambda x: x[0])
            grp["pods"]["worker"] = {"leaf": wl, **w}

    def incarnation(g: int, t_dec: float) -> dict | None:
        c = [x for x in groups.values() if x["g"] == g and x["created"] >= t_dec - 3]
        return min(c, key=lambda x: x["created"]) if c else None
    ups = []
    r0 = run.get("replicas0", 1)
    prev_of = {id(x): (dec[i - 1]["t"] if i else t0 - 60) for i, x in enumerate(dec)}  # previous decision, any direction
    for dd in [x for x in dec if x["to"] > x["from"]]:
        prev_t = prev_of[id(dd)]
        # trigger condition: the first 1-s EPP sample since the previous decision in which the metric the HPA names
        # in its event exceeds (1 + tol) x its target: queue (Value 2) or running requests per replica (AverageValue 4)
        msg = dd.get("msg") or ""
        by_run = "epp_running_requests" in msg and "epp_queue_size" not in msg
        cq = next((t for t, v in q if prev_t < t <= dd["t"] + 1 and target and v > (1 + TOL) * target), None)
        cr = next((t for t, v in runq if prev_t < t <= dd["t"] + 1 and run_target
                   and v / replicas_at(t) > (1 + TOL) * run_target), None)
        cond = cr if by_run else (cq if "epp_queue_size" in msg or cr is None else min(x for x in (cq, cr) if x))
        trig_src = "sampler"
        if cond is None:
            # a sub-second spike the 1-s EPP sampler missed but a 10-s Prometheus scrape caught (a-s-stock-r33):
            # fall back to the first Prometheus sample since the previous decision that crosses the named target
            pr = next((t for t, v in prom_run if prev_t < t <= dd["t"] + 1 and run_target
                       and v / replicas_at(t) > (1 + TOL) * run_target), None)
            pq = next((t for t, v in prom if prev_t < t <= dd["t"] + 1 and target and v > (1 + TOL) * target), None)
            cand = pr if by_run else (pq if "epp_queue_size" in msg else min((x for x in (pq, pr) if x), default=None))
            if cand is not None:
                cond, trig_src = cand, "prometheus"
                cr = cand if cand == pr else cr
        trig_metric = "running" if cond is not None and cond == cr else ("queue" if cond is not None else None)
        if trig_metric == "running":
            vis = next((t for t, v in prom_run if t >= cond - 0.5 and v / replicas_at(t) > (1 + TOL) * run_target), None)
            ext_vis = None
        else:
            vis = next((t for t, v in prom if cond is not None and t >= cond - 0.5 and v > (1 + TOL) * target), None)
            ext_vis = next((t for t, v in ext if cond is not None and t >= cond - 0.5 and v > (1 + TOL) * target), None)
        for g in range(dd["from"], dd["to"]):
            grp = incarnation(g, dd["t"])
            rec = {"group": g, "decision_t": dd["t"] - t0, "decision_event": dd.get("t_event") is not None,
                   "decision_from_to": [dd["from"], dd["to"]], "hpa_msg": dd.get("msg"),
                   "trigger_condition_t": None if cond is None else cond - t0, "trigger_metric": trig_metric,
                   "trigger_source": trig_src if cond is not None else None,
                   "metric_visible_t": None if vis is None else vis - t0,
                   "external_api_visible_t": None if ext_vis is None else ext_vis - t0}
            if grp:
                L, W = grp["pods"].get("leader", {}), grp["pods"].get("worker", {})
                lp, wp = L.get("last", {}), W.get("last", {})
                created = [ts(x.get("created")) for x in (lp, wp) if x]
                sched = [ts((x.get("cond") or {}).get("PodScheduled", [None, None])[1]) for x in (lp, wp) if x]
                mk = leader_marks(R, L.get("leaf", ""))
                ready_cond = L.get("ready_t")  # the leader = the replica's serving pod
                endpoint = (mk.get("engine_ready") or {}).get("serving_url") or (mk.get("engine_ready") or {}).get("endpoint")
                ep_hostport = endpoint.split("//")[-1] if endpoint else None
                # this incarnation's first routed request (an endpoint = a slice's host, reused by later incarnations)
                born = ts(lp.get("created")) or 0.0
                gone_t = (L.get("last_t") or float("inf"))
                first_client = min((r["engine_started_at_s"] for r in client if r.get("instance") == ep_hostport
                                    and r.get("engine_started_at_s") and born <= r["engine_started_at_s"] <= gone_t),
                                   default=None)
                rel = lambda x: None if x is None else x - t0  # noqa: E731
                rec.update({
                    # the leader's creation (LWS creates the worker only once the leader is placed)
                    "pods_created_t": rel(ts(lp.get("created"))),
                    "worker_created_t": rel(ts(wp.get("created"))) if wp else None,
                    "pods_scheduled_t": rel(max(c for c in sched if c) if all(sched) and sched else None),
                    "container_start_t": rel((mk.get("container_start") or {}).get("t")),
                    "ray_joined_t": rel((mk.get("ray_joined") or {}).get("t")),
                    "inventory_t": rel((mk.get("inventory") or {}).get("t")),
                    "xla_fetch_s": ((mk.get("xla_fetch_end") or {}).get("t", 0) - (mk.get("xla_fetch_start") or {}).get("t", 0))
                    if mk.get("xla_fetch_end") else None,
                    "deploy_start_t": rel((mk.get("deploy_start") or {}).get("t")),
                    "deploy_end_t": rel((mk.get("deploy_end") or {}).get("t")),
                    "engine_ready_t": rel((mk.get("engine_ready") or {}).get("t")),
                    "pod_ready_t": rel(ready_cond),
                    "first_request_seen_t": rel((mk.get("first_request_seen") or {}).get("t")),
                    "first_served_client_t": rel(first_client), "endpoint": ep_hostport,
                    "nodes": sorted({x.get("node") for x in (lp, wp) if x.get("node")}),
                    "leader_leaf": L.get("leaf"), "deleted_t": rel(del_t(lp))})
                # new capacity: the cluster autoscaler's trigger (pod event) and the slice's nodes (k8s Node objects)
                pod_names = {x.get("name") for x in (lp, wp) if x}
                trig = [ts(e.get("first") or e.get("event_time")) for e in R["events"]
                        if e.get("reason") == "TriggeredScaleUp" and e.get("obj", "").split("/", 1)[-1] in pod_names]
                # only this incarnation's events (pod names recur across runs and incarnations)
                c0 = (rec["pods_created_t"] if rec["pods_created_t"] is not None else -1e9) + t0
                c1 = (rec["pods_scheduled_t"] if rec["pods_scheduled_t"] is not None else 1e9) + t0
                trig = [x for x in trig if x is not None and c0 - 5 <= x <= c1 + 1]
                ninfo = node_info(R)
                nodes = [ninfo[n] for n in rec["nodes"] if n in ninfo]
                if nodes and all(n["created"] for n in nodes):
                    rec["node_created_t"] = rel(max(n["created"] for n in nodes))
                    rec["node_ready_t"] = rel(max(n["ready"] for n in nodes)) if all(n["ready"] for n in nodes) else None
                    if (rec["node_created_t"] or -1e9) < (rec["pods_created_t"] or 0):  # pre-existing (standby) nodes
                        rec["node_created_t"] = rec["node_ready_t"] = None
                rec["ca_triggered_t"] = rel(min(trig)) if trig else None
                fs = rec["first_served_client_t"] if rec["first_served_client_t"] is not None else rec["first_request_seen_t"]
                rec["trigger_to_first_served_s"] = (fs - rec["trigger_condition_t"]) if (
                    fs is not None and rec["trigger_condition_t"] is not None) else None
                rec["decision_to_first_served_s"] = (fs - rec["decision_t"]) if fs is not None else None
            ups.append(rec)
    row["scale_ups"] = ups
    row["scale_downs"] = [{"t": x["t"] - t0, "from": x["from"], "to": x["to"]} for x in dec if x["to"] < x["from"]]
    row["replicas_max"] = max((n for _, n in spec_series(R)), default=None)
    # stages (first scale-up)
    row["pipeline"] = stages(ups[0]) if ups else None
    # ---------------- verification
    ver = {}
    for leaf, files in R["pods"].items():
        v = files.get("validate") or {}
        c = files.get("check") or {}
        ver[leaf] = {"greedy": greedy(v), "mechanism": mechanism((files.get("deploy") or {}), run["boot"]),
                     "check_ok": (c.get("measurements") or {}).get("ok")}
    row["verification"] = ver
    row["all_greedy_ok"] = bool(ver) and all(x["greedy"].get("passed") for x in ver.values())
    row["all_mechanism_ok"] = bool(ver) and all(x["mechanism"].get("ok") for x in ver.values())
    # ---------------- cost
    trace_dur = float(((R["trace"] or {}).get("requests") or [{"t": t_client_end - t0}])[-1]["t"])
    t_win = t0 + trace_dur + DRAIN_S
    row["cost"] = cost(R, run, t0, t_win, groups, clip=True)
    t_idle = max(t_client_end, (run.get("marks") or {}).get("tail_done") or t_client_end)
    row["cost_to_idle"] = cost(R, run, t0, t_idle, groups, clip=False)
    n_met = sum(1 for r in client if am.meets_slo(r, SLO))
    row["slo_met_requests"] = n_met
    row["chip_s_per_slo_met_request"] = row["cost"]["chip_s"] / n_met if n_met else None
    row["chip_s_to_idle_per_slo_met_request"] = row["cost_to_idle"]["chip_s"] / n_met if n_met else None
    t_tail = max(t_client_end, (run.get("marks") or {}).get("tail_done") or t_client_end)
    row["timeline"] = timeline(R, t0, t_tail)
    row["start_state"] = start_state(R, run)
    row["valid"] = bool(row["status"] == "ok" and row["all_greedy_ok"] and row["all_mechanism_ok"]
                        and row["start_state"].get("ok", True))
    return row


def start_state(R: dict, run: dict) -> dict:
    """The bed at client start (first watcher rows): only the initial LWS groups exist (every earlier replica's pods,
    and with them their /dev/shm emptyDir state, are gone), every standby node (bed s) runs its balloon, and on
    bed n no new-capacity node exists yet."""
    if not R["state"] or not R["nodes"]:
        return {"ok": None, "error": "no watcher rows"}
    st, nd = R["state"][0], R["nodes"][0]
    r0 = run.get("replicas0", 1)
    lws = [p for p in st["pods"] if p.get("worker") is not None]
    groups = sorted({int(p["group"]) for p in lws})
    out = {"t_rel": st["t"] - run["client_start_at"], "lws_pods": len(lws), "groups": groups,
           "lws_pods_ready": sum(1 for p in lws if p.get("ready")), "lws_restarts": sum(p.get("restarts") or 0 for p in lws)}
    ok = groups == list(range(r0)) and len(lws) == 2 * r0 and out["lws_pods_ready"] == 2 * r0
    if run["bed"] == "s":
        standby = sorted(n["name"] for n in nd["nodes"] if n["pool"] in ("lld-s1", "lld-s2", "lld-s3"))
        used = {p["node"] for p in lws}
        balloons = {}
        for p in st["pods"]:
            if p.get("app") == "lld-balloon" and p.get("phase") == "Running" and not p.get("deleted"):
                balloons[p["node"]] = balloons.get(p["node"], 0) + 1
        free = [n for n in standby if n not in used]
        out.update(standby_nodes=len(standby), standby_nodes_free=len(free),
                   balloons_on_free_standby=sum(balloons.get(n, 0) for n in free),
                   free_without_balloon=[n for n in free if not balloons.get(n)])
        ok = ok and not out["free_without_balloon"] and len(free) == len(standby) - 2 * max(0, r0 - 1)
    else:
        new = [n["name"] for n in nd["nodes"] if n["pool"] not in ("lld-n0",)]
        out["new_capacity_nodes_at_start"] = new
        ok = ok and (not new or r0 > 1)
    out["ok"] = bool(ok)
    return out


def node_info(R: dict) -> dict:
    """node name -> creationTimestamp (registered with the API server) and the first Ready=True transition."""
    out: dict[str, dict] = {}
    for row in R["nodes"]:
        for n in row["nodes"]:
            e = out.setdefault(n["name"], {"created": ts(n.get("created")), "ready": None, "pool": n.get("pool")})
            r = n.get("ready")
            if e["ready"] is None and r and r[0] == "True":
                e["ready"] = ts(r[1])
    return out


def stages(u: dict) -> dict:
    keys = ["trigger_condition_t", "metric_visible_t", "decision_t", "pods_created_t", "ca_triggered_t",
            "node_created_t", "node_ready_t", "pods_scheduled_t",
            "container_start_t", "ray_joined_t", "inventory_t", "deploy_start_t", "deploy_end_t", "engine_ready_t",
            "pod_ready_t", "first_served_client_t"]
    out, prev = {}, None
    for k in keys:
        v = u.get(k)
        if v is not None and prev is not None:
            out[f"{prev[0].removesuffix('_t')}->{k.removesuffix('_t')}"] = round(v - prev[1], 2)
        if v is not None:
            prev = (k, v)
    out["total_trigger_to_first_served"] = u.get("trigger_to_first_served_s")
    return out


def greedy(v: dict) -> dict:
    ref = am.read_json(ROOT / "scripts/lld/jax-S-small_a-reference-966399c.json")
    m = (v or {}).get("measurements") or {}
    if not m.get("outputs"):
        return {"passed": False, "error": "no validate outputs"}
    outs = m["outputs"].get("small_a") or next(iter(m["outputs"].values()))
    r = ref.get("measurements", ref)
    ro = r["outputs"].get("small_a") or next(iter(r["outputs"].values()))
    same = [a == b for a, b in zip(outs, ro)]
    return {"prompts": len(outs), "prompts_equal": sum(same), "passed": len(outs) == len(ro) == 32 and all(same)
            and m.get("manifest_sha256") == r.get("manifest_sha256")}


def mechanism(dep: dict, boot: str) -> dict:
    """asb_run.mechanism on the leader's deploy payload."""
    try:
        dpl = dep["measurements"]["deployment"]
        init = dpl["instances"][0]["initialization"]
        workers = init.get("worker_init_phases") or []
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)[:200]}
    compile_s = sum(float((prof.get("exclusive_wall_s") or {}).get("backend_compile_and_load") or 0.0)
                    for w in workers for prof in (w.get("startup_profiles") or {}).values())
    snap = [(w.get("weight_snapshot") or {}) for w in workers]
    bank = [(w.get("exec_bank") or {}) for w in workers]
    settings = init.get("engine_settings") or {}
    row = {"workers": len(workers), "snapshot_restored": [s.get("restored") for s in snap],
           "bank_installed": [b.get("installed") for b in bank], "xla_backend_compile_s": round(compile_s, 3),
           "gate_wait_s": [(w.get("state_gate") or {}).get("waited_s") for w in workers],
           "load_format": settings.get("load_format"), "kv_blocks": (init.get("kv_capacity") or {}).get("usable_blocks")}
    if boot == "stock":
        row["ok"] = (len(workers) == 2 and not any(row["snapshot_restored"]) and not any(row["bank_installed"])
                     and row["load_format"] == "runai_streamer")
    else:
        row["ok"] = (len(workers) == 2 and all(row["snapshot_restored"]) and all(row["bank_installed"])
                     and row["xla_backend_compile_s"] < 1.0)
    return row


def cost(R: dict, run: dict, t0: float, t_end: float, groups: dict, clip: bool = True) -> dict:
    """chip-seconds = 8 x seconds each 2x4 slice is held, by state, over [t0, t_end]. clip=True (the common window,
    see DRAIN_S): new-capacity slices count only inside it, too. clip=False (to idle): t_end = the bed back at its
    initial size, and an autoscaled arm's new-capacity slices count until their nodes are gone."""
    # node -> pool from the node watcher; pools of the bed
    node_pool, node_created, node_gone = {}, {}, {}
    for row in R["nodes"]:
        names = set()
        for n in row["nodes"]:
            node_pool[n["name"]] = n["pool"]
            node_created.setdefault(n["name"], ts(n["created"]))
            names.add(n["name"])
        for n in list(node_created):
            if n not in names and n not in node_gone and row["t"] > (node_created[n] or 0):
                node_gone[n] = row["t"]
    pools = sorted({p for p in node_pool.values()})
    client = R["client"]
    seg: dict[str, dict[str, float]] = {}
    for pool in pools:
        nodes = [n for n, p in node_pool.items() if p == pool]
        role = "new" if pool.startswith("lld-n") and pool != "lld-n0" else ("base" if pool.endswith("0") else "standby")
        if role == "new":
            lo = min((node_created[n] for n in nodes if node_created.get(n)), default=None)
            hi = max((node_gone.get(n, t_end) for n in nodes), default=None)
            if (clip or not run.get("hpa")) and hi is not None:  # common window; static fleet on CA pools: always-on
                hi = min(hi, t_end)
            if lo is None:
                continue
            # billing from the cluster autoscaler's scale-up trigger for the pods that landed here (the GCE VMs are
            # created right after it; the k8s Node registers only once they booted): an upper bound
            names = {(grp["pods"].get(r) or {}).get("last", {}).get("name") for grp in groups.values()
                     for r in ("leader", "worker")
                     if node_pool.get(((grp["pods"].get(r) or {}).get("last") or {}).get("node")) == pool}
            trig = [ts(e.get("first") or e.get("event_time")) for e in R["events"]
                    if e.get("reason") == "TriggeredScaleUp" and e.get("obj", "").split("/", 1)[-1] in names]
            trig = [x for x in trig if x is not None and lo - 900 <= x <= lo]
            lo = max(min([lo, *trig]), t0)
            # held = union of the slice's node lifetimes (a pool can be scaled up, released and scaled up again),
            # the first lifetime extended back to the CA trigger
            ivn = sorted((max(node_created[n], t0), node_gone.get(n, t_end)) for n in nodes if node_created.get(n))
            ivn[0] = (lo, ivn[0][1])
            held = union_len(ivn, lo, hi)
        else:
            lo, hi = t0, t_end
            held = hi - lo
        s = {"held": held}
        # groups on this pool
        ivs = []
        for g, grp in groups.items():
            lp = (grp["pods"].get("leader") or {}).get("last") or {}
            if node_pool.get(lp.get("node")) != pool:
                continue
            created = ts(lp.get("created"))
            sched = ts((lp.get("cond") or {}).get("PodScheduled", [None, None])[1]) or created
            ready = (grp["pods"].get("leader") or {}).get("ready_t")
            gone = (grp["pods"].get("leader") or {}).get("last_t")
            deleted = del_t(lp)
            ivs.append({"sched": sched, "ready": ready, "deleted": deleted, "gone": gone})
        occupied = sum(max(0.0, min(i["gone"] or hi, hi) - max(i["sched"] or lo, lo)) for i in ivs)
        booting = sum(max(0.0, min(i["ready"] or i["gone"] or hi, hi) - max(i["sched"] or lo, lo)) for i in ivs)
        serving_iv = [(max(i["ready"], lo), min(i["deleted"] or i["gone"] or hi, hi)) for i in ivs if i["ready"]]
        draining = sum(max(0.0, min(i["gone"] or hi, hi) - max(i["deleted"], lo)) for i in ivs if i["deleted"])
        serving = sum(max(0.0, b - a) for a, b in serving_iv)
        # busy: requests served by the endpoint(s) of the groups on this pool (endpoint ip = node ip)
        pool_eps = set()
        for g, grp in groups.items():
            lp = (grp["pods"].get("leader") or {}).get("last") or {}
            if node_pool.get(lp.get("node")) == pool:
                mk = leader_marks(R, (grp["pods"].get("leader") or {}).get("leaf", ""))
                er = mk.get("engine_ready") or {}
                ep = (er.get("serving_url") or er.get("endpoint") or "").split("//")[-1]
                if ep:
                    pool_eps.add(ep)
        reqs = [(r["engine_started_at_s"], r["engine_finished_at_s"]) for r in client
                if r.get("instance") in pool_eps and r.get("engine_started_at_s") and r.get("engine_finished_at_s")]
        busy = min(serving, union_len(reqs, lo, hi)) if serving_iv else 0.0
        s.update(role=role, booting=booting, serving_busy=busy, serving_idle=max(0.0, serving - busy),
                 draining_check=draining, unoccupied=max(0.0, held - occupied),
                 balloon_held=max(0.0, held - occupied) if role == "standby" else 0.0)
        if role == "new":  # unoccupied = provisioning (segment start -> first pod scheduled) + release wait
            segs: list[list[float]] = []
            for a, b in sorted(ivn):
                if segs and a <= segs[-1][1]:
                    segs[-1][1] = max(segs[-1][1], b)
                else:
                    segs.append([a, b])
            prov = 0.0
            for a, b in segs:
                # first group on the slice in this segment (a static fleet's pods were scheduled before t0)
                fs_ = min((i["sched"] for i in ivs if i["sched"] and i["sched"] <= b and (i["gone"] or b) >= a - 1),
                          default=b)
                prov += max(0.0, min(fs_, hi) - max(a, t0))
            s["provisioning"] = prov
            s["release_wait"] = max(0.0, s["unoccupied"] - s["provisioning"])
        seg[pool] = {k: (round(v * CHIPS, 1) if isinstance(v, float) else v) for k, v in s.items()}
    # fewer-slices arms (HPA maxReplicas < 4 on bed s): the arm may use only base + (max - 1) standby slices, so
    # 4 - max standby slice-windows are excluded from its ledger. LWS group indices are < max, and a re-created index
    # waits for the old pod's preStop, so at most max - 1 standby slices are ever occupied at once; but a re-created
    # group may land on any free standby slice, so over a run the groups can visit more distinct standby slices than
    # max - 1. Excluded: first the never-occupied standby slices (their balloons were the only tenant); if fewer
    # exist than 4 - max, the rest is taken as whole slice-windows of balloon-held time off the remaining standby
    # slices (pseudo-pool "excluded-slot"; the balloon-held time is >= that at every instant by the bound above).
    excluded, extra = [], {}
    mr = int(run.get("max_replicas") or 4)
    if run.get("hpa") and run.get("bed") == "s" and mr < 4:
        k = 4 - mr
        idle = sorted(p for p, v in seg.items() if v.get("role") == "standby" and v.get("balloon_held", 0) >= v["held"] - 1)
        excluded = idle[:k]
        for p in excluded:
            seg[p]["excluded"] = True
        seg = {p: v for p, v in seg.items() if not v.get("excluded")}
        # concurrency check: occupied standby slices at once (leader sched -> gone)
        evs = []
        for g, grp in groups.items():
            lp = (grp["pods"].get("leader") or {}).get("last") or {}
            if node_pool.get(lp.get("node"), "").startswith("lld-s") and not node_pool[lp["node"]].endswith("0"):
                a = ts((lp.get("cond") or {}).get("PodScheduled", [None, None])[1]) or ts(lp.get("created"))
                b = (grp["pods"].get("leader") or {}).get("last_t") or t_end
                if a is not None:
                    evs += [(a, 1), (b, -1)]
        cur = peak = 0
        for _, d in sorted(evs, key=lambda x: (x[0], x[1])):
            cur += d
            peak = max(peak, cur)
        extra = {"max_concurrent_standby_occupied": peak,
                 "distinct_standby_used": sorted(p for p, v in seg.items() if v.get("role") == "standby"
                                                 and v.get("balloon_held", 0) < v["held"] - 1)}
        short = k - len(excluded)
        if short > 0:
            w = round(-short * (t_end - t0) * CHIPS, 1)
            seg["excluded-slot"] = {"held": w, "role": "standby", "balloon_held": w, "unoccupied": w,
                                    "slices": short}
            extra["excluded_slot_slices"] = short
            extra["balloon_held_before_slot"] = round(sum(v.get("balloon_held", 0.0) for p, v in seg.items()
                                                          if p != "excluded-slot"), 1)
            if peak > mr - 1:
                extra["warning"] = f"{peak} standby slices occupied at once > max - 1 = {mr - 1}"
    tot = sum(v["held"] for v in seg.values())
    agg = {}
    for k in ("booting", "serving_busy", "serving_idle", "draining_check", "balloon_held", "provisioning",
              "release_wait", "unoccupied"):
        agg[k] = round(sum(v.get(k, 0.0) for v in seg.values() if isinstance(v.get(k, 0.0), float)), 1)
    return {"chip_s": round(tot, 1), "window_s": [0.0, t_end - t0], "by_state": agg, "by_pool": seg,
            "excluded_pools": excluded, **extra}


def timeline(R: dict, t0: float, t1: float, bin_s: float = 10.0) -> dict:
    n = int(math.ceil((t1 - t0) / bin_s)) + 1
    sends = [0] * n
    ttft = [[] for _ in range(n)]
    tpot = [[] for _ in range(n)]
    for r in R["client"]:
        i = int((r["send_at"] - t0) // bin_s)
        if 0 <= i < n:
            sends[i] += 1
            if am.ok(r):
                ttft[i].append(r["ttft_s"])
                if r.get("tpot_s") is not None:
                    tpot[i].append(r["tpot_s"])
    qmax = [0.0] * n
    for t, v in queue_series(R):
        i = int((t - t0) // bin_s)
        if 0 <= i < n:
            qmax[i] = max(qmax[i], v)
    desired = [None] * n
    ready = [None] * n
    hpa_desired = [None] * n
    hpa_current = [None] * n
    for row in R["state"]:
        i = int((row["t"] - t0) // bin_s)
        if 0 <= i < n:
            if row.get("lws"):
                desired[i] = row["lws"]["spec_replicas"]
            if row.get("hpa"):
                hpa_desired[i], hpa_current[i] = row["hpa"].get("desired"), row["hpa"].get("current")
            # ready replicas = Ready leaders (the LWS hpaPodSelector / the InferencePool endpoints)
            ready[i] = sum(1 for p in row["pods"] if p.get("worker") == "0" and p.get("ready")
                           and not p.get("deleted"))
    qmean = [0.0] * n
    qcnt = [0] * n
    for t, v in queue_series(R):
        i = int((t - t0) // bin_s)
        if 0 <= i < n:
            qmean[i] += v
            qcnt[i] += 1
    return {"bin_s": bin_s, "rate_rps": [x / bin_s for x in sends], "queue_max": qmax,
            "queue_mean": [q / c if c else None for q, c in zip(qmean, qcnt)], "replicas_desired": desired,
            "replicas_ready": ready, "hpa_desired": hpa_desired, "hpa_current": hpa_current,
            "ttft_p50": [am.pct(x, 50) for x in ttft], "ttft_p99": [am.pct(x, 99) for x in ttft],
            "tpot_p99": [am.pct(x, 99) for x in tpot]}


def plot_run(m: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tl = m["timeline"]
    x = [i * tl["bin_s"] for i in range(len(tl["rate_rps"]))]
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(x, tl["rate_rps"], lw=1)
    ax[0].set_ylabel("sent req/s")
    ax[1].step(x, [v if v is not None else float("nan") for v in tl["replicas_desired"]], where="post", label="desired")
    ax[1].step(x, [v if v is not None else float("nan") for v in tl["replicas_ready"]], where="post", label="ready")
    ax[1].set_ylabel("replicas")
    ax[1].legend(loc="upper right")
    ax[2].plot(x, tl["queue_max"], lw=1, color="tab:red")
    ax[2].set_ylabel("EPP queue (max/10 s)")
    ax[3].semilogy(x, [v or float("nan") for v in tl["ttft_p50"]], lw=1, label="TTFT p50")
    ax[3].semilogy(x, [v or float("nan") for v in tl["ttft_p99"]], lw=1, label="TTFT p99")
    ax[3].semilogy(x, [v or float("nan") for v in tl["tpot_p99"]], lw=1, label="TPOT p99")
    ax[3].axhline(SLO["ttft_s"], ls="--", lw=0.8, color="gray")
    ax[3].set_ylabel("s (by send time)")
    ax[3].legend(loc="upper right")
    for u in m["scale_ups"]:
        for a in ax:
            if u.get("decision_t") is not None:
                a.axvline(u["decision_t"], color="k", lw=0.6, ls=":")
            if u.get("first_served_client_t") is not None:
                a.axvline(u["first_served_client_t"], color="tab:green", lw=0.6, ls="--")
    ax[3].set_xlabel("s since client start (dotted: HPA decision, dashed: new replica's first request)")
    fig.suptitle(f"{m['rid']} ({m['arm']})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_cmd(d: Path, plot: bool = True) -> dict:
    if not (d / "prom/running.json").exists() and (d / "run.json").exists():
        try:
            fetch_prom(d)
        except Exception as exc:  # noqa: BLE001
            print(f"fetch_prom {d.name}: {exc!r}"[:300], file=sys.stderr)
    for f in (d / "pods").glob("*/*.json") if (d / "pods").exists() else []:  # check/validate payloads ~8 MB each
        if f.stat().st_size > 500_000:
            import subprocess  # noqa: PLC0415
            subprocess.run(["gzip", "-f", str(f)], check=False)
    m = analyze(d)
    (d / "metrics.json").write_text(json.dumps(m, indent=1, default=str))
    if plot:
        plot_run(m, d / "timeline.png")
    return m


# ---------------------------------------------------------------- campaign build

ARM_ORDER = ["s-pm", "s-stock", "s-pm-max3", "s-pm-max2", "s-stock-max3", "s-stock-max2", "n-pm", "n-stock",
             "static-base", "static-peak", "s-pm-stab120", "s-stock-stab120"]
ARM_LABEL = {"s-pm": "PM, standby (max 4)", "s-stock": "stock, standby (max 4)",
             "s-pm-max3": "PM, standby, max 3 replicas", "s-pm-max2": "PM, standby, max 2 replicas",
             "s-stock-max3": "stock, standby, max 3 replicas", "s-stock-max2": "stock, standby, max 2 replicas",
             "n-pm": "PM, new capacity (spot, CA)",
             "n-stock": "stock, new capacity (spot, CA)", "static-base": "static base (1 replica)",
             "static-peak": "static peak (4 replicas)", "s-pm-stab120": "PM, standby, 120 s up-stabilization",
             "s-stock-stab120": "stock, standby, 120 s up-stabilization"}
ASB = Path.home() / "hetero-wt/AutoscaleBurst/results/autoscale-burst/4daec1af/summary.json"
# pipeline stages (task order), each = (label, from key, to key); new capacity splits "decision -> scheduled"
PIPE = [("queue > target -> Prometheus", "trigger_condition_t", "metric_visible_t"),
        ("Prometheus -> HPA decision", "metric_visible_t", "decision_t"),
        ("decision -> pods created", "decision_t", "pods_created_t"),
        ("pods created -> scheduled", "pods_created_t", "pods_scheduled_t"),
        ("scheduled -> Ray joined", "pods_scheduled_t", "ray_joined_t"),
        ("Ray joined -> engine ready", "ray_joined_t", "engine_ready_t"),
        ("engine ready -> first routed", "engine_ready_t", "first_served_client_t")]
PIPE_GKE = [("  of which: pods created -> CA trigger", "pods_created_t", "ca_triggered_t"),
            ("  of which: CA trigger -> nodes registered", "ca_triggered_t", "node_created_t"),
            ("  of which: nodes registered -> Ready", "node_created_t", "node_ready_t"),
            ("  of which: nodes Ready -> pods scheduled", "node_ready_t", "pods_scheduled_t")]


def pipe_row(u: dict, stages_: list) -> dict:
    out = {}
    for label, a, b in stages_:
        va, vb = u.get(a), u.get(b)
        out[label] = (vb - va) if va is not None and vb is not None else None
    return out


def run_summary(m: dict) -> dict:
    """The per-run numbers the tables use."""
    c = m["cost"]["by_state"]
    ups = m.get("scale_ups") or []
    first = ups[0] if ups else {}
    serving = c.get("serving_busy", 0.0) + c.get("serving_idle", 0.0)
    return {
        "rid": m["rid"], "arm": m["arm"], "rep": m["rep"], "trace": m["trace"], "valid": m.get("valid"),
        "status": m["status"], "greedy_ok": m.get("all_greedy_ok"), "mechanism_ok": m.get("all_mechanism_ok"),
        "replicas_max": m.get("replicas_max"), "scale_ups": len(ups), "scale_downs": len(m.get("scale_downs") or []),
        # HPA flapping: replicas booted again after an earlier scale-down within the run (running-requests target)
        "reboots_after_scale_down": sum(1 for u in ups if any(s["t"] < u["decision_t"] for s in m.get("scale_downs") or [])),
        "trigger_to_first_served_s": first.get("trigger_to_first_served_s"),
        "decision_to_first_served_s": first.get("decision_to_first_served_s"),
        "trigger_to_first_served_all_s": [u.get("trigger_to_first_served_s") for u in ups],
        "decision_to_first_served_all_s": [u.get("decision_to_first_served_s") for u in ups],
        "first_trigger_t": first.get("trigger_condition_t"), "first_decision_t": first.get("decision_t"),
        "burst_ttft_p99": m["burst"].get("ttft_p99"), "burst_tpot_p99": m["burst"].get("tpot_p99"),
        "burst_slo_attainment_pct": m["burst"].get("slo_attainment_pct"),
        "run_ttft_p99": m["run"].get("ttft_p99"), "run_tpot_p99": m["run"].get("tpot_p99"),
        "run_slo_attainment_pct": m["run"].get("slo_attainment_pct"),
        "queue_max": m.get("queue_max"), "queue_integral_req_s": m.get("queue_integral_req_s"),
        "errors_429": m["failures"]["http_429"], "errors_5xx": m["failures"]["http_5xx"],
        "errors_transport": m["failures"]["transport"], "n_requests": m["run"].get("n"),
        "chip_s": m["cost"]["chip_s"], "chip_s_serving": serving, "chip_s_serving_busy": c.get("serving_busy"),
        "chip_s_serving_idle": c.get("serving_idle"), "chip_s_booting": c.get("booting"),
        "chip_s_balloon_held": c.get("balloon_held"), "chip_s_provisioning": c.get("provisioning"),
        "chip_s_release_wait": c.get("release_wait"), "chip_s_draining_check": c.get("draining_check"),
        "chip_s_excl_check": m["cost"]["chip_s"] - (c.get("draining_check") or 0.0),
        "chip_s_per_slo_met_request": m.get("chip_s_per_slo_met_request"),
        "chip_s_to_idle": (m.get("cost_to_idle") or {}).get("chip_s"),
        "chip_s_to_idle_per_slo_met_request": m.get("chip_s_to_idle_per_slo_met_request"),
        "chip_s_to_idle_by_state": (m.get("cost_to_idle") or {}).get("by_state"),
        "window_s": m["cost"]["window_s"][1], "window_to_idle_s": (m.get("cost_to_idle") or {}).get("window_s", [0, None])[1],
        "pipeline_first": pipe_row(first, PIPE + PIPE_GKE) if first else None,
        "pipeline_all": [pipe_row(u, PIPE + PIPE_GKE) for u in ups],
    }


def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)) or (isinstance(v, float) and nd == 0):
        return f"{v:.0f}"
    return f"{v:.{nd}f}"


def cell(rows: list[dict], key: str, nd=1) -> str:
    vals = [r.get(key) for r in rows]
    m = med(vals)
    return f"{fmt(m, nd)} ({', '.join(fmt(v, nd) for v in vals)})" if len(vals) > 1 else fmt(m, nd)


def build(rev: str, readme: bool, plot: bool = True) -> dict:
    base = RESULTS / rev
    runs = sorted(p for p in (base / "runs").glob("*") if (p / "run.json").exists())
    per_run, metrics = [], {}
    for d in runs:
        m = run_cmd(d, plot=plot)
        metrics[m["rid"]] = m
        per_run.append(run_summary(m))
    failed = sorted(p.name for p in (base / "runs-failed").glob("*")) if (base / "runs-failed").exists() else []
    arms: dict[str, dict] = {}
    for tr in sorted({r["trace"] for r in per_run}):
        for arm in ARM_ORDER + sorted({r["arm"] for r in per_run} - set(ARM_ORDER)):
            rows = sorted([r for r in per_run if r["arm"] == arm and r["trace"] == tr], key=lambda r: r["rep"])
            if not rows:
                continue
            valid = [r for r in rows if r["valid"]]
            keys = [k for k, v in valid[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)] if valid else []
            pipe_keys = [s[0] for s in PIPE + PIPE_GKE]
            pipe_first = {k: med([(r["pipeline_first"] or {}).get(k) for r in valid]) for k in pipe_keys}
            pipe_all = {k: med([p.get(k) for r in valid for p in r["pipeline_all"]]) for k in pipe_keys}
            arms[f"{tr}/{arm}"] = {
                "trace": tr, "arm": arm, "label": ARM_LABEL.get(arm, arm), "n": len(valid), "n_attempted": len(rows),
                "runs": [r["rid"] for r in valid], "invalid": [r["rid"] for r in rows if not r["valid"]],
                "median": {k: med([r[k] for r in valid]) for k in keys},
                "trigger_to_first_served_all_median": med([x for r in valid for x in r["trigger_to_first_served_all_s"]]),
                "pipeline_first_median": pipe_first, "pipeline_all_median": pipe_all}
    asb = am.read_json(ASB) or {}
    summary = {"rev": rev, "slo": SLO, "chips_per_slice": CHIPS, "arms": arms, "per_run": per_run,
               "failed_attempts": failed, "infra_ledger": ledger(rev) if (base / "infra/nodes.jsonl").exists() else None,
               "autoscale_burst_ref": {"file": str(ASB), "arms": {
                   k: {"n": v.get("n"), "median": v.get("median")} for k, v in ((asb.get("arms") or {}).get("a") or {}).items()}}}
    (base / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if plot:
        figures(base, per_run, metrics, arms)
    if readme:
        write_tables(base, summary)
    return summary


def tables_md(summary: dict) -> str:
    per_run = summary["per_run"]
    out = []
    for tr in sorted({a["trace"] for a in summary["arms"].values()}):
        arms = [a for a in summary["arms"].values() if a["trace"] == tr]
        rowsof = lambda a: [r for r in per_run if r["rid"] in a["runs"]]  # noqa: E731
        out.append(f"### Trace {tr}\n")
        out.append("Latency and SLO: median (per-run values, rep order). Recovery = trigger (the first 1-s EPP sample "
                   "in which the metric named by the HPA's event exceeds 1.1 x its target: queue > 2.2, or in-flight "
                   "requests per current replica > 4.4) -> the new replica's first routed request, first scale-up.\n")
        out.append("| arm | n | recovery: trigger -> first served (s) | all scale-ups, median (s) | burst TTFT P99 (s) | "
                   "burst TPOT P99 (s) | burst SLO attain. % | run SLO attain. % | run TTFT P99 (s) | EPP queue max | "
                   "queue integral (req-s) | 429 / 5xx / transport | replicas max | scale-ups / scale-downs / "
                   "re-boots after a scale-down (per run) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a in arms:
            rs = rowsof(a)
            if not rs:
                continue
            out.append(f"| {a['label']} | {a['n']} | {cell(rs, 'trigger_to_first_served_s')} | "
                       f"{fmt(a['trigger_to_first_served_all_median'])} | {cell(rs, 'burst_ttft_p99')} | "
                       f"{cell(rs, 'burst_tpot_p99', 3)} | {cell(rs, 'burst_slo_attainment_pct')} | "
                       f"{cell(rs, 'run_slo_attainment_pct')} | {cell(rs, 'run_ttft_p99')} | {cell(rs, 'queue_max', 0)} | "
                       f"{cell(rs, 'queue_integral_req_s', 0)} | {fmt(med([r['errors_429'] for r in rs]), 0)} / "
                       f"{fmt(med([r['errors_5xx'] for r in rs]), 0)} / {fmt(med([r['errors_transport'] for r in rs]), 0)} | "
                       f"{cell(rs, 'replicas_max', 0)} | "
                       f"{', '.join(str(r['scale_ups']) + '/' + str(r['scale_downs']) + '/' + str(r['reboots_after_scale_down']) for r in rs)} |")
        out.append("")
        win = med([r.get("window_s") for r in per_run if r["trace"] == tr])
        out.append("Cost in chip-seconds (8 chips per 2x4 slice) over the **common window** of every arm: [client "
                   f"start, trace end + 20 s drain] ({fmt(win)} s on trace {tr}; new-capacity slices from the CA scale-up "
                   "trigger, clipped to the window). The last two columns use the **to-idle window** instead: until "
                   "the bed is back at 1 replica (the HPA's 300 s scale-down window plus the teardown preStop checks) "
                   "and, on new capacity, until the cluster autoscaler deleted the emptied spot slices. Columns are "
                   "medians; total is median (per run).\n")
        out.append("| arm | n | total chip-s | serving busy | serving idle | booting | balloon-held standby | "
                   "provisioning (new) | release wait (new) | drain + greedy check | chip-s per SLO-met request | "
                   "to-idle chip-s | to-idle chip-s per SLO-met request |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a in arms:
            rs = rowsof(a)
            if not rs:
                continue
            mm = lambda k: fmt(med([r.get(k) for r in rs]), 0)  # noqa: E731
            out.append(f"| {a['label']} | {a['n']} | {cell(rs, 'chip_s', 0)} | {mm('chip_s_serving_busy')} | "
                       f"{mm('chip_s_serving_idle')} | {mm('chip_s_booting')} | {mm('chip_s_balloon_held')} | "
                       f"{mm('chip_s_provisioning')} | {mm('chip_s_release_wait')} | {mm('chip_s_draining_check')} | "
                       f"{cell(rs, 'chip_s_per_slo_met_request', 2)} | {cell(rs, 'chip_s_to_idle', 0)} | "
                       f"{cell(rs, 'chip_s_to_idle_per_slo_met_request', 2)} |")
        out.append("")
        dyn = [a for a in arms if any(r["scale_ups"] for r in rowsof(a))]
        if dyn:
            out.append("Scale-up pipeline, median over every scale-up of the arm's runs (s):\n")
            out.append("| stage | " + " | ".join(a["label"] for a in dyn) + " |")
            out.append("|---|" + "---|" * len(dyn))
            for label, _, _ in PIPE + PIPE_GKE:
                vals = [a["pipeline_all_median"].get(label) for a in dyn]
                if all(v is None for v in vals):
                    continue
                out.append(f"| {label} | " + " | ".join(fmt(v) for v in vals) + " |")
            out.append("| **total trigger -> first served** | " + " | ".join(
                fmt(a["trigger_to_first_served_all_median"]) for a in dyn) + " |")
            out.append("")
    return "\n".join(out)


def write_tables(base: Path, summary: dict) -> None:
    readme = RESULTS / "README.md"
    txt = readme.read_text()
    begin, end = "<!-- lld-tables:begin -->", "<!-- lld-tables:end -->"
    block = f"{begin}\n{tables_md(summary)}\n{end}"
    if begin in txt and end in txt:
        txt = txt.split(begin)[0] + block + txt.split(end, 1)[1]
    else:
        txt += "\n" + block + "\n"
    readme.write_text(txt)


def figures(base: Path, per_run: list[dict], metrics: dict, arms: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = base / "figures"
    fd.mkdir(exist_ok=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    # timeline per arm: reps overlaid
    for key, a in arms.items():
        ms = [metrics[r] for r in a["runs"] + a["invalid"] if r in metrics]
        if not ms:
            continue
        fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        for i, m in enumerate(ms):
            tl, c = m["timeline"], colors[i % len(colors)]
            x = [j * tl["bin_s"] for j in range(len(tl["rate_rps"]))]
            nan = lambda v: [float("nan") if y is None else y for y in v]  # noqa: E731
            if i == 0:
                ax[0].plot(x, tl["rate_rps"], lw=0.8, color="k")
            ax[1].step(x, nan(tl["replicas_desired"]), where="post", color=c, lw=0.8, ls=":")
            ax[1].step(x, nan(tl["replicas_ready"]), where="post", color=c, lw=1.2, label=f"r{m['rep']} ready")
            ax[2].plot(x, tl["queue_max"], lw=0.8, color=c)
            ax[3].semilogy(x, [v or float("nan") for v in tl["ttft_p99"]], lw=0.8, color=c)
            ax[3].semilogy(x, [v or float("nan") for v in tl["tpot_p99"]], lw=0.6, color=c, ls="--")
        ax[0].set_ylabel("sent req/s (10 s)")
        ax[1].set_ylabel("replicas\n(dotted: HPA/LWS desired)")
        ax[1].legend(loc="upper right", fontsize=8)
        ax[2].set_ylabel("EPP queue (max/10 s)")
        ax[3].axhline(SLO["ttft_s"], ls="-.", lw=0.8, color="gray")
        ax[3].axhline(SLO["tpot_s"], ls="-.", lw=0.8, color="silver")
        ax[3].set_ylabel("TTFT P99 (solid)\nTPOT P99 (dashed), s")
        ax[3].set_xlabel("s since client start")
        fig.suptitle(f"trace {a['trace']}: {a['label']} (n={a['n']})")
        fig.tight_layout()
        fig.savefig(fd / f"timeline-{a['trace']}-{a['arm']}.png", dpi=110)
        plt.close(fig)
    for tr in sorted({a["trace"] for a in arms.values()}):
        al = [a for a in arms.values() if a["trace"] == tr and a["n"]]
        if not al:
            continue
        # cost vs SLO bars
        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(al)), 5))
        parts = [("chip_s_serving_busy", "serving busy", "tab:green"), ("chip_s_serving_idle", "serving idle", "yellowgreen"),
                 ("chip_s_booting", "booting", "tab:orange"), ("chip_s_balloon_held", "balloon-held standby", "lightgray"),
                 ("chip_s_provisioning", "provisioning (new)", "tab:brown"), ("chip_s_release_wait", "release wait (new)", "rosybrown"),
                 ("chip_s_draining_check", "drain + check", "tab:gray")]
        xs = range(len(al))
        bottom = [0.0] * len(al)
        for k, lab, col in parts:
            v = [((a["median"].get(k) or 0.0) / 1000.0) for a in al]
            ax.bar(xs, v, bottom=bottom, color=col, label=lab, width=0.6)
            bottom = [b + y for b, y in zip(bottom, v)]
        ax.set_ylabel("k chip-s (median)")
        ax.set_xticks(list(xs), [a["label"].replace(", ", ",\n") for a in al], fontsize=8)
        ax2 = ax.twinx()
        for i, a in enumerate(al):
            vals = [r["run_slo_attainment_pct"] for r in per_run if r["rid"] in a["runs"]]
            bvals = [r["burst_slo_attainment_pct"] for r in per_run if r["rid"] in a["runs"]]
            ax2.scatter([i] * len(vals), vals, color="k", marker="o", s=18, zorder=3)
            ax2.scatter([i + 0.12] * len(bvals), bvals, color="tab:red", marker="^", s=18, zorder=3)
        ax2.set_ylim(0, 102)
        ax2.set_ylabel("SLO attainment % (o run, ^ burst)")
        ax.legend(loc="upper left", fontsize=7)
        ax.set_title(f"trace {tr}: cost vs SLO attainment")
        fig.tight_layout()
        fig.savefig(fd / f"cost-vs-slo-{tr}.png", dpi=110)
        plt.close(fig)
        # cost frontier from real runs: chip-s vs SLO attainment, one point per arm (median), per-run values faint
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
        mk = {"pm": "o", "stock": "s"}
        for j, (key, name) in enumerate((("burst_slo_attainment_pct", "burst"), ("run_slo_attainment_pct", "whole run"))):
            ax = axs[j]
            for i, a in enumerate(al):
                rs = [r for r in per_run if r["rid"] in a["runs"]]
                c = f"C{i % 10}"
                m_ = mk.get("stock" if "stock" in a["arm"] else "pm", "o") if a["arm"].startswith(("s-", "n-")) else "D"
                ax.scatter([r["chip_s"] / 1000 for r in rs], [r[key] for r in rs], color=c, marker=m_, s=14, alpha=0.35)
                x, y = med([r["chip_s"] for r in rs]), med([r[key] for r in rs])
                if x is None or y is None:
                    continue
                ax.scatter([x / 1000], [y], color=c, marker=m_, s=70, edgecolor="k", zorder=3,
                           label=f"{a['label']} (n={a['n']})")
                ax.annotate(a["arm"], (x / 1000, y), textcoords="offset points", xytext=(6, -10), fontsize=7)
            ax.set_xlabel("chip-s per run (k, median; faint = per run)")
            ax.set_ylabel(f"{name} SLO attainment %")
            ax.grid(alpha=0.3)
            ax.set_title(f"trace {tr}: cost frontier ({name})")
        axs[0].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(fd / f"frontier-{tr}.png", dpi=110)
        plt.close(fig)
        # pipeline breakdown (median over all scale-ups)
        dyn = [a for a in al if a["trigger_to_first_served_all_median"] is not None]
        if dyn:
            fig, ax = plt.subplots(figsize=(11, 0.8 * len(dyn) + 1.8))
            pc = ["tab:red", "tab:pink", "tab:purple", "tab:brown", "tab:blue", "tab:orange", "tab:green"]
            for i, a in enumerate(dyn):
                left = 0.0
                for j, (lab, _, _) in enumerate(PIPE):
                    v = a["pipeline_all_median"].get(lab) or 0.0
                    ax.barh(i, v, left=left, color=pc[j], label=lab if i == 0 else None, height=0.6)
                    left += v
                ax.text(left + 3, i, f"{left:.0f} s", va="center", fontsize=8)
            ax.set_yticks(range(len(dyn)), [a["label"] for a in dyn], fontsize=8)
            ax.set_xlabel("s (median stage durations over every scale-up)")
            ax.legend(fontsize=7, loc="lower right")
            ax.set_title(f"trace {tr}: llm-d scale-up pipeline")
            fig.tight_layout()
            fig.savefig(fd / f"pipeline-{tr}.png", dpi=110)
            plt.close(fig)


def ledger(rev: str, t_end: float | None = None) -> dict:
    """Campaign infra ledger: every TPU node the cluster held (infra/nodes.jsonl, 30-s polls from 08:31Z; nodes that
    existed before the first poll start at their creationTimestamp), 4 chips per node, split by pool and by what
    the bed was doing: counted-run windows (prep_start -> end, per bed), failed attempts, and the rest (setup,
    calibration, smoke, waits between runs)."""
    base = RESULTS / rev
    rows = am.read_jsonl(base / "infra/nodes.jsonl")
    life: dict[str, dict] = {}
    for r in rows:
        for n in r["nodes"]:
            e = life.setdefault(n["name"], {"pool": n["pool"], "spot": n.get("spot") == "true",
                                            "start": ts(n["created"]), "last": r["t"]})
            e["last"] = r["t"]
    t_last = rows[-1]["t"] if rows else None
    for e in life.values():  # still present at the last poll: held until t_end (deletion) if given
        e["end"] = (t_end or e["last"]) if e["last"] == t_last else e["last"] + 15.0
    windows = {}
    for sub in ("runs", "runs-failed"):
        for d in sorted((base / sub).glob("*")) if (base / sub).exists() else []:
            run = am.read_json(d / "run.json") or {}
            mk = run.get("marks") or {}
            if mk.get("prep_start"):
                windows.setdefault(run["bed"], []).append((sub, mk["prep_start"], mk.get("end", mk["prep_start"])))
    out = {"pools": {}, "total_chip_s": 0.0}
    for name, e in life.items():
        bed = "s" if e["pool"].startswith("lld-s") else "n"
        dur = e["end"] - e["start"]
        split = {"counted_runs": 0.0, "failed_attempts": 0.0}
        for sub, a, b in windows.get(bed, []):
            ov = max(0.0, min(b, e["end"]) - max(a, e["start"]))
            split["counted_runs" if sub == "runs" else "failed_attempts"] += ov
        split["other"] = dur - split["counted_runs"] - split["failed_attempts"]
        p = out["pools"].setdefault(e["pool"], {"spot": e["spot"], "nodes": 0, "chip_s": 0.0, "counted_runs": 0.0,
                                                "failed_attempts": 0.0, "other": 0.0})
        p["nodes"] += 1
        p["chip_s"] += 4 * dur
        for k, v in split.items():
            p[k] += 4 * v
    for p in out["pools"].values():
        for k in ("chip_s", "counted_runs", "failed_attempts", "other"):
            p[k] = round(p[k], 0)
    out["total_chip_s"] = sum(p["chip_s"] for p in out["pools"].values())
    out["total_chip_h"] = round(out["total_chip_s"] / 3600, 1)
    out["on_demand_chip_h"] = round(sum(p["chip_s"] for p in out["pools"].values() if not p["spot"]) / 3600, 1)
    out["spot_chip_h"] = round(sum(p["chip_s"] for p in out["pools"].values() if p["spot"]) / 3600, 1)
    (base / "infra/ledger.json").write_text(json.dumps(out, indent=1))
    return out


def calibration() -> dict:
    """llm-d chain calibration (one replica): unloaded medians -> 5x SLO; Poisson rates -> attainment (30 s warm-up
    dropped) vs AutoscaleBurst's (same traces/seed, its own router)."""
    cal = RESULTS / "calibration"
    asb = am.read_json(Path.home() / "hetero-wt/AutoscaleBurst/results/autoscale-burst/calibration/calibration.json") or {}
    out: dict = {"asb": {"unloaded": asb.get("unloaded"), "slo": asb.get("slo"), "rates": asb.get("rates")},
                 "slo_used": SLO, "rates": {}}
    recs = am.read_jsonl(cal / "unloaded/client.jsonl")
    if recs:
        s = am.summarize(recs)
        out["unloaded"] = {k: s[k] for k in ("n", "ok", "ttft_p50", "ttft_p90", "tpot_p50", "tpot_p90",
                                             "engine_ttft_p50", "queue_p50")}
        out["slo_5x"] = {"ttft_s": 5 * s["ttft_p50"], "tpot_s": 5 * s["tpot_p50"]}
    for d in sorted(cal.glob("rate-*"), key=lambda p: float(p.name.split("-", 1)[1])):
        recs = am.read_jsonl(d / "client.jsonl")
        if not recs:
            continue
        steady = [r for r in recs if (r.get("sched_t") or 0.0) >= 30.0]
        s = am.summarize(steady, SLO)
        q = [sum(v for k, v in (x.get("m") or {}).items() if k.startswith("llm_d_epp_flow_control_queue_size{"))
             for x in am.read_jsonl(d / "samples.jsonl")]
        run_ = [sum(v for k, v in (x.get("m") or {}).items() if k.startswith("inference_objective_running_requests"))
                for x in am.read_jsonl(d / "samples.jsonl")]
        out["rates"][d.name.split("-", 1)[1]] = {
            "n": s["n"], "ok": s["ok"], "ttft_p50": s["ttft_p50"], "ttft_p99": s["ttft_p99"], "tpot_p50": s["tpot_p50"],
            "tpot_p99": s["tpot_p99"], "slo_attainment_pct": s["slo_attainment_pct"],
            "throughput_rps": s.get("throughput_rps"), "epp_queue_p99": am.pct(q, 99), "epp_queue_max": max(q, default=None),
            "epp_running_mean": statistics.mean(run_) if run_ else None}
    ok90 = [float(r) for r, v in out["rates"].items() if (v["slo_attainment_pct"] or 0) >= 90.0]
    out["capacity_attainment_90"] = max(ok90) if ok90 else None
    (cal / "calibration.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("run"); q.add_argument("dir")
    q.add_argument("--no-plot", action="store_true")
    q = sp.add_parser("build"); q.add_argument("--rev", required=True)
    q.add_argument("--readme-tables", action="store_true"); q.add_argument("--no-plot", action="store_true")
    sp.add_parser("calibration")
    q = sp.add_parser("ledger"); q.add_argument("--rev", required=True); q.add_argument("--t-end", type=float, default=None)
    a = p.parse_args()
    if a.cmd == "build":
        s = build(a.rev, a.readme_tables, not a.no_plot)
        print(tables_md(s))
    elif a.cmd == "calibration":
        print(json.dumps(calibration(), indent=1, default=str))
    elif a.cmd == "ledger":
        print(json.dumps(ledger(a.rev, a.t_end), indent=1))
    if a.cmd == "run":
        m = run_cmd(Path(a.dir), not a.no_plot)
        brief = {k: m[k] for k in ("rid", "status", "queue_max", "replicas_max", "pipeline", "all_greedy_ok",
                                   "all_mechanism_ok")}
        brief["burst"] = {k: m["burst"].get(k) for k in ("n", "ttft_p99", "tpot_p99", "slo_violation_pct")}
        brief["run"] = {k: m["run"].get(k) for k in ("n", "ttft_p99", "slo_violation_pct", "status")}
        brief["cost"] = m["cost"]["by_state"] | {"chip_s": m["cost"]["chip_s"]}
        print(json.dumps(brief, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
