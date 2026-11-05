"""Paper figures + numbers.json for the llm-d autoscaling campaign (results/llmd-autoscale/4d97ec1a).

Reads the campaign summary.json, per-run metrics.json, loadgen/client.jsonl.gz and k8s/nodes.jsonl.gz, and the
replayed traces (results/autoscale-burst/4daec1af/traces, read via `git show HEAD:` if the sparse checkout lacks them).
usage: .venv/bin/python scripts/lld_paper.py [OUTDIR]   (default results/llmd-autoscale/paper)
"""
import gzip
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/llmd-autoscale/4d97ec1a"
TRACES = "results/autoscale-burst/4daec1af/traces"
PAPER = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/llmd-autoscale/paper"
OUT = PAPER / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def trace(tr):
    p = ROOT / TRACES / f"trace-{tr}.json"
    if p.exists():
        return json.loads(p.read_text())
    import subprocess  # noqa: PLC0415
    return json.loads(subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{TRACES}/trace-{tr}.json"],
                                     check=True, capture_output=True, text=True).stdout)
S = json.loads((BASE / "summary.json").read_text())
A = S["arms"]
RUNS = {r["rid"]: r for r in S["per_run"]}
C = 0.7  # req/s per replica (calibrated)
SLO_TTFT = S["slo"]["ttft_s"]
PM, STOCK, GREY = "#1f77b4", "#ff7f0e", "#7f7f7f"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def metrics(rid):
    return json.loads((BASE / "runs" / rid / "metrics.json").read_text())


def med(arm, key, tr="a"):
    return A[f"{tr}/{arm}"]["median"][key]


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- fig 1: workload
def fig_workload():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [2.6, 1]})
    for ax, tr, title in [(axs[0], "a", "Trace A: Chutes gemma-3-27b-it burst (replayed, k = 0.854)"),
                          (axs[1], "b", "Trace B: synthetic Poisson step")]:
        d = trace(tr)
        ts = [r["t"] for r in d["requests"]]
        b = 30.0
        n = int(max(ts) // b) + 1
        cnt = [0] * n
        for t in ts:
            cnt[int(t // b)] += 1
        x = [i * b / 60 for i in range(n)]
        ax.step(x, [c / b for c in cnt], where="post", color="k", lw=1, label="offered load (30 s bins)")
        for k in range(1, 5):
            ax.axhline(k * C, color=GREY, lw=0.7, ls=":")
            ax.text(1.005, k * C, f"{k}x C", va="center", fontsize=8, color=GREY,
                    transform=ax.get_yaxis_transform())
        bw = metrics(f"{tr}-s-pm-r1")["burst_window_s"]
        ax.axvspan(bw[0] / 60, bw[1] / 60, color="tab:red", alpha=0.07, label="burst window (SLO scored)")
        ax.set_xlabel("minutes since client start")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 3.2)
    axs[0].set_ylabel("requests / s")
    axs[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "fig1-workload.png")


# ---------------------------------------------------------------- timelines (fig 2, fig 7)
def series(rid):
    m = metrics(rid)
    tl = m["timeline"]
    x = [i * tl["bin_s"] / 60 for i in range(len(tl["rate_rps"]))]
    return m, tl, x


def smooth(v, w=3):
    out = []
    for i in range(len(v)):
        win = [y for y in v[max(0, i - w + 1): i + 1] if y is not None]
        out.append(sum(win) / len(win) if win else None)
    return out


def fig_timeline(spec, name, title, tmax_min):
    fig, axs = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1]})
    _, tl0, x0 = series(spec[0][0])
    axs[0].fill_between(x0, smooth(tl0["rate_rps"]), step="post", color="#dddddd", label="offered load (30 s avg)")
    for rid, label, color, ls in spec:
        m, tl, x = series(rid)
        cap = [None if v is None else v * C for v in tl["replicas_ready"]]
        axs[0].step(x, cap, where="post", color=color, ls=ls, lw=1.6, label=f"{label}: ready capacity")
        axs[1].plot(x, tl["queue_max"], color=color, ls=ls, lw=1.2, label=label)
        axs[2].semilogy(x, [v if v else float("nan") for v in tl["ttft_p99"]], color=color, ls=ls, lw=1.0,
                        label=label)
        u = (m.get("scale_ups") or [{}])[0]
        if u.get("first_served_client_t") is not None:
            for ax in axs:
                ax.axvline(u["first_served_client_t"] / 60, color=color, ls=ls, lw=0.6, alpha=0.6)
    t_first = min(((metrics(r).get("scale_ups") or [{}])[0].get("decision_t") or 1e9) for r, *_ in spec)
    for ax in axs:
        ax.axvline(t_first / 60, color="k", ls=":", lw=0.8)
    axs[1].text(t_first / 60, axs[1].get_ylim()[1] * 0.95, " first HPA scale-up decision", fontsize=8, va="top")
    axs[0].set_ylabel("req/s  (ready replicas x C)")
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, frameon=False, ncol=3)
    axs[1].set_ylabel("EPP queue (max / 10 s)")
    axs[2].axhline(SLO_TTFT, color="k", lw=0.8, ls="--")
    axs[2].text(0.2, SLO_TTFT * 1.15, f"TTFT SLO {SLO_TTFT:.3f} s", fontsize=8)
    axs[2].set_ylabel("TTFT P99 per 10 s (s)")
    axs[2].set_xlabel("minutes since client start (thin vertical lines: each arm's first new replica serves)")
    axs[2].set_xlim(0, tmax_min)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.suptitle(title, fontsize=11, y=0.995)
    save(fig, name)


# ---------------------------------------------------------------- fig 3: cost frontier
def fig_frontier():
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    k = lambda arm: med(arm, "chip_s") / 1000  # noqa: E731
    y = lambda arm: med(arm, "burst_slo_attainment_pct")  # noqa: E731
    for arm, color in [("s-pm", PM), ("s-pm-max3", PM), ("s-pm-max2", PM), ("s-stock", STOCK),
                       ("s-stock-max3", STOCK), ("n-pm", PM), ("n-stock", STOCK), ("s-pm-stab120", PM),
                       ("s-stock-stab120", STOCK), ("static-peak", GREY)]:
        for rid in A[f"a/{arm}"]["runs"]:
            r = RUNS[rid]
            ax.scatter(r["chip_s"] / 1000, r["burst_slo_attainment_pct"], s=10, color=color, alpha=0.3, zorder=2)
    pm = ["s-pm-max2", "s-pm-max3", "s-pm"]
    st = ["s-stock-max3", "s-stock"]
    ax.plot([k(a) for a in pm], [y(a) for a in pm], "-o", color=PM, ms=8, lw=2,
            label="TPUFlex, standby (2 / 3 / 4 slices)", zorder=3)
    ax.plot([k(a) for a in st], [y(a) for a in st], "-s", color=STOCK, ms=8, lw=2,
            label="stock, standby (3 / 4 slices)", zorder=3)
    ax.scatter([k("n-pm")], [y("n-pm")], marker="^", s=80, facecolor="white", edgecolor=PM, lw=2, zorder=3,
               label="TPUFlex, new spot capacity (max 4)")
    ax.scatter([k("n-stock")], [y("n-stock")], marker="^", s=80, facecolor="white", edgecolor=STOCK, lw=2,
               zorder=3, label="stock, new spot capacity (max 4)")
    ax.scatter([k("s-pm-stab120")], [y("s-pm-stab120")], marker="D", s=50, facecolor="white", edgecolor=PM, lw=2,
               zorder=3, label="TPUFlex, 120 s up-stabilization")
    ax.scatter([k("s-stock-stab120")], [y("s-stock-stab120")], marker="D", s=50, facecolor="white",
               edgecolor=STOCK, lw=2, zorder=3, label="stock, 120 s up-stabilization")
    ax.scatter([k("static-peak")], [y("static-peak")], marker="*", s=200, color=GREY, zorder=3,
               label="static peak (4 replicas always on)")
    for a, txt, dx, dy in [("s-pm-max2", "2 slices", -1.5, -3.2), ("s-pm-max3", "3 slices", -2, 1.5),
                           ("s-pm", "4", 0.8, -0.2), ("s-stock-max3", "3 slices", -2, -3.2),
                           ("s-stock", "4", 0.8, -0.2)]:
        ax.text(k(a) + dx, y(a) + dy, txt, fontsize=8)
    ax.annotate("", xy=(k("s-pm-max3"), y("s-pm-max3") - 1), xytext=(k("s-stock"), y("s-stock") + 1),
                arrowprops={"arrowstyle": "->", "color": "tab:green", "lw": 1.5})
    dk = 100 * (1 - k("s-pm-max3") / k("s-stock"))
    dy_ = y("s-pm-max3") - y("s-stock")
    ax.text((k("s-pm-max3") + k("s-stock")) / 2 + 1, (y("s-pm-max3") + y("s-stock")) / 2,
            f"-{dk:.0f} % chip-s\n+{dy_:.1f} pp SLO", color="tab:green", fontsize=9)
    ax.axhline(90, color="k", lw=0.6, ls=":")
    ax.text(41, 90.5, "90 % target", fontsize=8)
    ax.set_xlim(40, 96)
    ax.set_ylim(55, 100)
    ax.set_xlabel("chip-seconds per run (thousands; 8 chips per held slice; trace window + 20 s)")
    ax.set_ylabel("burst SLO attainment (%)")
    ax.text(41, 56.5, f"off-scale: static base (1 replica) {k('static-base'):.1f} k chip-s, "
                      f"{y('static-base'):.1f} %", fontsize=8, color=GREY)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_title("Trace A cost frontier (median of n = 2-3 runs; faint dots = individual runs)", fontsize=10)
    ax.grid(alpha=0.25)
    save(fig, "fig3-frontier.png")


# ---------------------------------------------------------------- fig 4: chip-s breakdown
ARMS4 = [("static-peak", "static peak (4 always on)"), ("s-pm", "PM, 4 slices"), ("s-stock", "stock, 4 slices"),
         ("s-pm-max3", "PM, 3 slices"), ("s-stock-max3", "stock, 3 slices"), ("s-pm-max2", "PM, 2 slices"),
         ("n-pm", "PM, new spot capacity"), ("n-stock", "stock, new spot capacity"),
         ("s-pm-stab120", "PM, 120 s stabilization"), ("s-stock-stab120", "stock, 120 s stabilization"),
         ("static-base", "static base (1 replica)")]
STATES = [("chip_s_serving_busy", "serving (busy)", "#2ca02c"), ("chip_s_serving_idle", "serving (idle)", "#98df8a"),
          ("chip_s_booting", "booting", "#ff7f0e"), ("chip_s_balloon_held", "held standby (placeholder)", "#c7c7c7"),
          ("chip_s_provisioning", "provisioning new nodes", "#8c564b"),
          ("chip_s_release_wait", "release wait (new)", "#c49c94"),
          ("chip_s_draining_check", "drain + greedy check", "#7f7f7f")]


def fig_breakdown():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True, gridspec_kw={"width_ratios": [2.2, 1]})
    ys = list(range(len(ARMS4)))[::-1]
    for y, (arm, _) in zip(ys, ARMS4):
        left = 0.0
        for key, lab, col in STATES:
            v = (med(arm, key) or 0) / 1000
            ax.barh(y, v, left=left, color=col, edgecolor="white", lw=0.4,
                    label=lab if y == ys[0] else None)
            left += v
        ax.text(left + 0.8, y, f"{med(arm, 'chip_s') / 1000:.1f} k", va="center", fontsize=8)
        pr = med(arm, "chip_s_per_slo_met_request")
        col = PM if "pm" in arm else (STOCK if "stock" in arm else GREY)
        ax2.barh(y, pr, color=col)
        ax2.text(pr + 0.8, y, f"{pr:.1f}", va="center", fontsize=8)
    ax.set_yticks(ys, [lab for _, lab in ARMS4])
    ax.set_xlabel("chip-seconds per run (thousands), by slice state")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=8,
               frameon=False, ncol=4)
    ax.set_xlim(0, 105)
    ax2.set_xlabel("chip-seconds per SLO-met request")
    ax2.set_xlim(0, 80)
    fig.suptitle("Trace A: what the chip-seconds pay for (median runs)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig4-cost-breakdown.png")


# ---------------------------------------------------------------- fig 5: pipeline
GROUPS = [("metrics + HPA decision", ["queue > target -> Prometheus", "Prometheus -> HPA decision"], "#d62728"),
          ("pods scheduled (incl. new nodes)", ["decision -> pods created", "pods created -> scheduled"], "#8c564b"),
          ("pod start + Ray join", ["scheduled -> Ray joined"], "#1f77b4"),
          ("engine boot", ["Ray joined -> engine ready"], "#ff7f0e"),
          ("first request routed", ["engine ready -> first routed"], "#2ca02c")]
ARMS5 = [("s-pm", "PM, standby"), ("s-stock", "stock, standby"), ("n-pm", "PM, new spot capacity"),
         ("n-stock", "stock, new spot capacity"), ("s-pm-stab120", "PM, standby, 120 s stab."),
         ("s-stock-stab120", "stock, standby, 120 s stab.")]


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ys = list(range(len(ARMS5)))[::-1]
    for y, (arm, _) in zip(ys, ARMS5):
        p = A[f"a/{arm}"]["pipeline_first_median"]
        left = 0.0
        for lab, keys, col in GROUPS:
            v = max(0.0, sum((p.get(k_) or 0) for k_ in keys))
            ax.barh(y, v, left=left, color=col, edgecolor="white", lw=0.4, label=lab if y == ys[0] else None)
            if v > 25:
                ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white")
            left += v
        tot = med(arm, "trigger_to_first_served_s")
        ax.text(left + 5, y, f"{tot:.0f} s (trigger -> first served)", va="center", fontsize=8)
    ax.set_yticks(ys, [lab for _, lab in ARMS5])
    ax.set_xlabel("seconds, first scale-up of each run (median over runs; stages are medians, so sums differ slightly)")
    ax.set_xlim(0, 820)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, frameon=False, ncol=5)
    ax.set_title("Trace A: where a scale-up's time goes under llm-d", fontsize=10)
    save(fig, "fig5-pipeline.png")


# ---------------------------------------------------------------- fig 6: burst TTFT CDF
def burst_ttfts(rid):
    m = metrics(rid)
    run = json.loads((BASE / "runs" / rid / "run.json").read_text())
    t0 = run["client_start_at"]
    lo, hi = m["burst_window_s"]
    out = []
    with gzip.open(BASE / "runs" / rid / "loadgen/client.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if lo <= r["send_at"] - t0 < hi:
                ok = r.get("status") == 200 and r.get("error") is None and r.get("ttft_s") is not None
                out.append(r["ttft_s"] if ok else float("inf"))
    return out


def fig_cdf():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for arm, lab, col, ls in [("static-peak", "static peak (4 always on)", GREY, "-"),
                              ("s-pm", "PM, 4 slices", PM, "-"), ("s-pm-max3", "PM, 3 slices", PM, "--"),
                              ("s-pm-max2", "PM, 2 slices", PM, ":"), ("s-stock", "stock, 4 slices", STOCK, "-"),
                              ("s-stock-max3", "stock, 3 slices", STOCK, "--"),
                              ("static-base", "static base (1 replica)", "k", ":")]:
        v = sorted(x for rid in A[f"a/{arm}"]["runs"] for x in burst_ttfts(rid))
        n = len(v)
        ax.step(v, [(i + 1) / n * 100 for i in range(n)], where="post", color=col, ls=ls, lw=1.5, label=lab)
    ax.axvline(SLO_TTFT, color="k", lw=0.8, ls="--")
    ax.text(SLO_TTFT * 1.08, 3, "TTFT SLO", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 1000)
    ax.set_ylim(0, 100)
    ax.set_xlabel("TTFT (s), burst-window requests pooled over runs")
    ax.set_ylabel("cumulative % of requests")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    ax.set_title("Trace A: burst TTFT distribution", fontsize=10)
    save(fig, "fig6-ttft-cdf.png")


# ---------------------------------------------------------------- fig 8: cross-check vs hand-written controller
def fig_crosscheck():
    asb = S["autoscale_burst_ref"]["arms"]
    hw = {"pm": asb["pm-gcs"]["median"], "stock": asb["stock"]["median"]}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    groups = ["hand-written controller\n(AutoscaleBurst, n=3)", "llm-d HPA chain\n(this campaign, n=3)"]
    rec = {"pm": [hw["pm"]["recovery_first_served_s"], med("s-pm", "trigger_to_first_served_s")],
           "stock": [hw["stock"]["recovery_first_served_s"], med("s-stock", "trigger_to_first_served_s")]}
    slo = {"pm": [100 - hw["pm"]["burst_slo_violation_pct"], med("s-pm", "burst_slo_attainment_pct")],
           "stock": [100 - hw["stock"]["burst_slo_violation_pct"], med("s-stock", "burst_slo_attainment_pct")]}
    for ax, dat, ylab in [(axs[0], rec, "trigger -> first served (s)"), (axs[1], slo, "burst SLO attainment (%)")]:
        for j, (key, col, lab) in enumerate([("pm", PM, "TPUFlex"), ("stock", STOCK, "stock")]):
            xs = [i + (j - 0.5) * 0.36 for i in range(2)]
            ax.bar(xs, dat[key], width=0.34, color=col, label=lab)
            for xx, vv in zip(xs, dat[key]):
                ax.text(xx, vv * 1.01, f"{vv:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(2), groups, fontsize=8)
        ax.set_ylabel(ylab)
    axs[0].legend(fontsize=8, frameon=False)
    axs[1].set_ylim(0, 105)
    fig.suptitle("Same trace, same 4 standby slices: hand-written controller vs llm-d", fontsize=10)
    fig.tight_layout()
    save(fig, "fig8-crosscheck.png")


# ---------------------------------------------------------------- spot (new capacity) timeline, replica units
def slices_series(rid, t0):
    out = []
    with gzip.open(BASE / "runs" / rid / "k8s/nodes.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            n = sum(1 for x in r["nodes"] if (x.get("pool") or "").startswith("lld-n") and not x.get("deleted"))
            out.append(((r["t"] - t0) / 60, n / 2))
    return out


def fig_spot(spec, name, title, tmax_min):
    fig, axs = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1]})
    _, tl0, x0 = series(spec[0][0])
    axs[0].fill_between(x0, [None if v is None else v / C for v in smooth(tl0["rate_rps"])], step="post",
                        color="#dddddd", label="offered load / C (replicas needed)")
    for rid, label, color in spec:
        m, tl, x = series(rid)
        t0 = json.loads((BASE / "runs" / rid / "run.json").read_text())["client_start_at"]
        ss = slices_series(rid, t0)
        axs[0].step([a for a, _ in ss], [b for _, b in ss], where="post", color=color, ls="--", lw=1.0,
                    label=f"{label}: TPU slices up")
        axs[0].step(x, tl["replicas_ready"], where="post", color=color, lw=1.8, label=f"{label}: replicas serving")
        axs[1].plot(x, tl["queue_max"], color=color, lw=1.2, label=label)
        axs[2].semilogy(x, [v if v else float("nan") for v in tl["ttft_p99"]], color=color, lw=1.0, label=label)
        u = (m.get("scale_ups") or [{}])[0]
        if u.get("first_served_client_t") is not None:
            for ax in axs:
                ax.axvline(u["first_served_client_t"] / 60, color=color, lw=0.6, alpha=0.6)
    t_first = min(((metrics(r).get("scale_ups") or [{}])[0].get("decision_t") or 1e9) for r, *_ in spec)
    for ax in axs:
        ax.axvline(t_first / 60, color="k", ls=":", lw=0.8)
    axs[1].text(t_first / 60, axs[1].get_ylim()[1] * 0.95, " first HPA scale-up decision", fontsize=8, va="top")
    axs[0].set_ylabel("replicas / slices")
    axs[0].set_ylim(0, 4.6)
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, frameon=False, ncol=3)
    axs[1].set_ylabel("EPP queue (max / 10 s)")
    axs[2].axhline(SLO_TTFT, color="k", lw=0.8, ls="--")
    axs[2].text(0.2, SLO_TTFT * 1.15, f"TTFT SLO {SLO_TTFT:.3f} s", fontsize=8)
    axs[2].set_ylabel("TTFT P99 per 10 s (s)")
    axs[2].set_xlabel("minutes since client start (thin vertical lines: each arm's first new replica serves)")
    axs[2].set_xlim(0, tmax_min)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.suptitle(title, fontsize=11, y=0.995)
    save(fig, name)




# ---------------------------------------------------------------- numbers.json (exact values for the paper)
KEYS = ["trigger_to_first_served_s", "decision_to_first_served_s", "burst_ttft_p99", "burst_tpot_p99",
        "burst_slo_attainment_pct", "run_slo_attainment_pct", "queue_max", "chip_s", "chip_s_per_slo_met_request",
        "chip_s_booting", "chip_s_balloon_held", "chip_s_serving_busy", "chip_s_serving_idle", "chip_s_to_idle",
        "reboots_after_scale_down", "n_requests"]


def numbers():
    out = {"source": "results/llmd-autoscale/4d97ec1a/summary.json (code rev 4d97ec1a)", "slo": S["slo"],
           "capacity_rps_per_replica": C, "chips_per_slice": S["chips_per_slice"],
           "cost_window": "[client start, trace end + 20 s]: 2830.6 s (trace a), 1099.5 s (trace b)",
           "burst_window_s": {"a": metrics("a-s-pm-r1")["burst_window_s"], "b": metrics("b-s-pm-r1")["burst_window_s"]},
           "arms": {}}
    for key, a in A.items():
        rows = [RUNS[r] for r in a["runs"]]
        out["arms"][key] = {"label": a["label"], "n": a["n"], "runs": a["runs"],
                            "median": {k: a["median"].get(k) for k in KEYS},
                            "per_run": {k: [r.get(k) for r in rows] for k in KEYS},
                            "pipeline_first_scaleup_median_s": a["pipeline_first_median"]}
    asb = S["autoscale_burst_ref"]["arms"]
    out["hand_written_controller_ref"] = {
        "source": "research/autoscale-burst @ 94974e1b, results/autoscale-burst summary.json (trace a, n=3)",
        "pm": {"recovery_first_served_s": asb["pm-gcs"]["median"]["recovery_first_served_s"],
               "burst_slo_attainment_pct": 100 - asb["pm-gcs"]["median"]["burst_slo_violation_pct"]},
        "stock": {"recovery_first_served_s": asb["stock"]["median"]["recovery_first_served_s"],
                  "burst_slo_attainment_pct": 100 - asb["stock"]["median"]["burst_slo_violation_pct"]}}
    (PAPER / "numbers.json").write_text(json.dumps(out, indent=1))
    print("wrote", PAPER / "numbers.json")


if __name__ == "__main__":
    fig_workload()
    fig_timeline([("a-s-pm-r1", "PM, 4 slices", PM, "-"), ("a-s-pm-r11", "PM, 3 slices", PM, "--"),
                  ("a-s-stock-r1", "stock, 4 slices", STOCK, "-"), ("a-s-stock-r31", "stock, 3 slices", STOCK, "--")],
                 "fig2-timeline-a.png",
                 "Trace A, one median run per arm: capacity, queue and tail latency", 48)
    fig_frontier()
    fig_breakdown()
    fig_pipeline()
    fig_cdf()
    fig_timeline([("b-s-pm-r1", "PM, 4 slices", PM, "-"), ("b-s-stock-r1", "stock, 4 slices", STOCK, "-")],
                 "fig7-timeline-b.png", "Trace B (n = 1): capacity, queue and tail latency", 18.5)
    fig_crosscheck()
    fig_spot([("a-n-pm-r2", "PM", PM), ("a-n-stock-r3", "stock", STOCK)], "fig9-spot-timeline.png",
             "Trace A on new spot capacity (GKE cluster autoscaler from 0 nodes), median run per arm", 48)
    numbers()
