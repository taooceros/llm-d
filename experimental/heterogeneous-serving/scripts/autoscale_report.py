#!/usr/bin/env python3
"""Summarize a ``switch_campaign.sh autoscale`` run: per arm, how long the new instance (large_b) took
from "hosts allocated, nothing local" to "serving", split into state delivery and engine boot.

    python3 scripts/autoscale_report.py results/switch/autoscale-<commit>[-ov] [--json out.json]

Per arm (row = one fresh-host boot; ``runs.jsonl`` names it):
  fetch_s     wall of the startup_state.py fetch remote driver (target refusal check, transfer,
              publication); 0 for ``raw``.  ``fetch_job_s`` adds Ray job submission (timeline marks).
              -ov arms: the same driver run as a child of the deploy job (deployment ``state_delivery``).
  ckpt_fetch  ckpt-gcs arms: wall of the fetch-checkpoint job's remote driver (every host pulls the checkpoint).
  delivery_s  fetch_s + ckpt_fetch wall.
  boot_s      large_b ``initialization.init_s``: engine actor start -> engine initialized.
  ready_s     delivery_s + boot_s (sequential arms only).  Shared by every arm and not included:
              host/pod provisioning, image pull, placement-group creation, gateway registration
              (deployment ``stage_clocks`` are listed).
  e2e_s       scale-up start -> large_b initialized, the same cut for every arm (needs the deployment's
              ``started_at``): sequential = delivery_s + control_s + boot_s; -ov = initialized - deploy start
              (the delivery child starts right after it).
  control_s   deploy start -> large_b engine init start (backend check, placement group, actors).
  pre_device_s  engine init start -> the last worker is ready for its first device touch (process
              spawn, imports, vLLM/tpu-inference construction; no startup state needed).
  gate_wait_s   -ov: the longest worker wait for its rank's delivery marker (0 = delivery was hidden).
  device_s    the last worker's first device touch -> engine initialized (init_device, load, compile/bank).
  phases      critical path over large_b's workers (max duration) of the dominant init phases.
  compile     rank-0 startup-profile interval seconds by kind (backend_compile_and_load = XLA compiles
              that missed every cache) and the number of compiles.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path

PHASES = ("state_gate", "init_device", "load_model", "initialize_from_config", "exec_bank_load",
          "compile_or_warm_up_model")
KINDS = ("trace", "lower", "cache_lookup_and_decompress", "executable_load", "backend_compile_and_load",
         "warmup_execute_and_sync", "aot_compile")


def load(directory, name):
    for path in (directory / name, directory / (name + ".gz")):
        if path.exists():
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt") as handle:
                return json.load(handle)
    return None


def marks(directory):
    rows = {}
    path = directory / "timeline.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row.get("event") or row.get("mark")] = row.get("t") or row.get("time") or row.get("ts")
    return rows


def instance(deploy, name):
    for row in deploy["measurements"]["deployment"]["instances"]:
        init = row.get("initialization") or {}
        if (row.get("instance_id") or init.get("instance_id")) == name:
            return init
    raise KeyError(f"no instance {name} in deploy evidence")


def summarize_boot(init):
    workers = init.get("worker_init_phases") or []
    started, init_s = init.get("engine_init_started_at"), init.get("init_s")
    critical = {}
    if started is not None and init_s is not None and workers:
        ready_at = started + init_s
        touch = [w["phases"]["init_device"]["start"] for w in workers
                 if isinstance(w.get("phases", {}).get("init_device"), dict)]
        gate = [w["phases"]["state_gate"] for w in workers
                if isinstance(w.get("phases", {}).get("state_gate"), dict)]
        if touch:
            ready_to_touch = max(g["start"] for g in gate) if gate else max(touch)
            critical = {"ready_at": ready_at, "pre_device_s": round(ready_to_touch - started, 2),
                        "gate_wait_s": round(max(g["end"] - g["start"] for g in gate), 2) if gate else 0.0,
                        "gate_wait_by_rank_s": [round(g["end"] - g["start"], 2) for g in gate],
                        "device_s": round(ready_at - max(touch), 2)}
    phases = {}
    for phase in PHASES:
        spans = [w["phases"][phase]["end"] - w["phases"][phase]["start"] for w in workers
                 if isinstance(w.get("phases", {}).get(phase), dict)]
        if spans:
            phases[phase] = round(max(spans), 2)
    weight_phases = {}
    for w in workers:
        for name, span in (w.get("phases") or {}).items():
            if isinstance(span, dict) and (name.startswith("snapshot_") or name.startswith("normal_load")
                                           or name in ("graph_build", "loader_entry")):
                if not name.startswith("snapshot_device_placement_chip"):
                    weight_phases[name] = max(weight_phases.get(name, 0.0), span["end"] - span["start"])
    rank0 = next((w for w in workers if w.get("executor_rank") == 0), workers[0] if workers else {})
    kinds, compiles = {k: 0.0 for k in KINDS}, 0
    for profile in (rank0.get("startup_profiles") or {}).values():
        for interval in profile.get("intervals", []) if isinstance(profile, dict) else []:
            kind = interval.get("kind")
            if kind in kinds:
                kinds[kind] += interval["end"] - interval["start"]
                compiles += kind == "backend_compile_and_load"
    snap = rank0.get("weight_snapshot") or {}
    bank = rank0.get("exec_bank") or {}
    return {"boot_s": init.get("init_s"), **critical, "phases": phases,
            "weight_phases": {k: round(v, 2) for k, v in sorted(weight_phases.items())},
            "compile_s_by_kind": {k: round(v, 2) for k, v in kinds.items() if v},
            "xla_compiles": compiles,
            "snapshot_restored": bool(snap.get("restored")), "snapshot_bytes_per_rank": snap.get("bytes"),
            "bank_installed": bool(bank.get("installed")), "bank_mode": bank.get("mode"),
            "compile_cache_dir": (rank0.get("compile_cache") or {}).get("dir")}


def summarize(directory):
    directory = Path(directory)
    runs = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines() if line.strip()]
    timeline = marks(directory)
    rows = []
    for run in runs:
        rid, arm = run["run_id"], run.get("kind")
        row = {"arm": arm, "run_id": rid, "ok": run.get("ok")}
        deploy = load(directory, f"{rid}-deploy.json")
        fetch = load(directory, f"{rid}-fetch.json")
        delivery = (deploy or {}).get("measurements", {}).get("deployment", {}).get("state_delivery")
        m = fetch["measurements"] if fetch else (delivery or {}).get("fetch")
        if m:
            row["fetch"] = {"method": m.get("method"), "parts": m.get("parts"), "bytes": m.get("bytes"),
                            "files": m.get("files"), "wall_s": round(m["wall_s"], 2),
                            "transfer_s": round(m["transfer_s"], 2), "finalize_s": round(m["finalize_s"], 2),
                            "transfer_gbps": round(m["transfer_gbps"], 3) if m.get("transfer_gbps") else None,
                            "bytes_by_part": {p: sum(r["by_part"].get(p, 0) for r in m["ranks"]) for p in m["parts"]}}
            start, end = timeline.get(f"fetch_start:{rid}"), timeline.get(f"fetch_end:{rid}")
            if start and end:
                row["fetch"]["job_s"] = round(end - start, 2)
        ckpt = load(directory, f"{rid}-ckpt-fetch.json")
        if ckpt:
            m = ckpt["measurements"]
            row["ckpt_fetch"] = {"bytes": m["bytes"], "files": m["files"], "wall_s": round(m["wall_s"], 2),
                                 "transfer_gbps": round(m["transfer_gbps"], 3),
                                 "per_host_gbps": [round(v, 3) for v in m["per_host_gbps"]]}
        row["delivery_s"] = round((row.get("fetch", {}).get("wall_s") or 0.0)
                                  + (row.get("ckpt_fetch", {}).get("wall_s") or 0.0), 2)
        row["overlap"] = delivery is not None
        if deploy:
            dep = deploy["measurements"]["deployment"]
            b = instance(deploy, "large_b")
            row.update(summarize_boot(b))
            row["seed_large_a_boot_s"] = instance(deploy, "large_a").get("init_s")
            row["stage_clocks"] = dep.get("stage_clocks")
            t0 = dep.get("started_at")
            if t0 is not None and b.get("engine_init_started_at") is not None and "ready_at" in row:
                row["control_s"] = round(b["engine_init_started_at"] - t0, 2)
                row["e2e_s"] = round(row["ready_at"] - t0 if delivery else
                                     row["delivery_s"] + row["ready_at"] - t0, 2)
            if delivery and t0 is not None:
                f = delivery.get("fetch") or {}
                rel = lambda t: None if t is None else round(t - t0, 2)  # noqa: E731
                row["delivery_timeline"] = {
                    "child_start": rel(delivery.get("started_at")), "driver_start": rel(f.get("started_at")),
                    "transfer_start": rel(f.get("transfer_started_at")),
                    "finalize_start": rel(f.get("finalize_started_at")), "published": rel(f.get("ended_at")),
                    "child_exit": rel(delivery.get("exited_at")),
                    "engine_init_start": rel(b.get("engine_init_started_at")),
                    "initialized": rel(row.get("ready_at"))}
                row["delivery_child_s"] = round(delivery["wall_s"], 2) if delivery.get("wall_s") else None
        validation = load(directory, f"{rid}-validation.json")
        if validation:
            row["greedy_ok"] = validation["measurements"].get("ok")
        if row.get("boot_s") is not None and not row["overlap"]:
            row["ready_s"] = round(row["delivery_s"] + row["boot_s"], 2)
        row.pop("ready_at", None)
        rows.append(row)
    seed = load(directory, "seed-deploy.json")
    upload = load(directory, "upload.json")
    result = {"directory": str(directory), "arms": rows}
    if seed:
        result["seed"] = {"large_a_write": summarize_boot(instance(seed, "large_a")),
                          "large_b_normal": summarize_boot(instance(seed, "large_b"))}
    if upload:
        m = upload["measurements"]
        result["upload"] = {"bytes": m.get("bytes"), "files": m.get("files"), "wall_s": round(m["wall_s"], 2),
                            "gbps": round(m["gbps"], 3)}
    by_arm = {}
    for row in rows:
        if row.get("ok") and row.get("boot_s") is not None:
            by_arm.setdefault(row["arm"], []).append(row)

    def med(rs, key):
        values = [r[key] for r in rs if r.get(key) is not None]
        return round(statistics.median(values), 2) if values else None

    result["by_arm"] = {arm: {"n": len(rs), "e2e_s": [r.get("e2e_s") for r in rs],
                              "ready_s": [r.get("ready_s") for r in rs],
                              **{f"{k}_median_s": med(rs, f"{k}_s")
                                 for k in ("e2e", "ready", "delivery", "control", "pre_device", "gate_wait",
                                           "device", "boot")}}
                        for arm, rs in by_arm.items()}
    return result


def table(result):
    lines = ["| arm | n | delivery s | control s | pre-device s | gate wait s | device s | boot s | e2e s | e2e runs |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(result["by_arm"].items(), key=lambda kv: (kv[1]["e2e_median_s"] is None,
                                                             kv[1]["e2e_median_s"] or kv[1]["ready_median_s"] or 0))
    for arm, s in order:
        lines.append(f"| {arm} | {s['n']} | {s['delivery_median_s']} | {s['control_median_s']} "
                     f"| {s['pre_device_median_s']} | {s['gate_wait_median_s']} | {s['device_median_s']} "
                     f"| {s['boot_median_s']} | {s['e2e_median_s']} | {', '.join(str(v) for v in s['e2e_s'])} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    for row in result["arms"]:
        print(json.dumps({k: row.get(k) for k in ("arm", "run_id", "ok", "greedy_ok", "e2e_s", "ready_s", "control_s",
                                                   "pre_device_s", "gate_wait_s", "device_s", "boot_s",
                                                   "delivery_timeline", "fetch", "ckpt_fetch",
                                                   "phases", "weight_phases", "xla_compiles", "compile_s_by_kind",
                                                   "snapshot_restored", "bank_installed")}, separators=(",", ":")))
    print(table(result))


if __name__ == "__main__":
    main()
