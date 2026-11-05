#!/usr/bin/env python3
"""G3 cluster gate: non-transparent in-engine layout switch of ONE engine over block A (results/in_engine_dp/).

Engine 'A' spans hosts (0, 1, 2, 3) of one JAX runtime, one vLLM + tpu-inference engine. Layouts (types.IN_ENGINE_LAYOUTS):
    H    = TP16            (mesh data=1, model=16)
    S-dp = TP8 x DP2       (single-process SPMD data parallelism, DPScheduler; data group g on hosts (2g, 2g+1))
The gate deploys H cold from the checkpoint (resident=False), checks tokens against the H reference, then
``BlockController.switch_layout``: S-dp (checked against the S reference) -> H (H reference) -> and ``--reps`` more
S-dp/H cycles, every visit checked. Per switch it records ``built``, controller phases, per-host worker segments,
frontend segments, compiles (backend compile events = persistent-cache misses; hits are not counted by the kernels,
so the new entries of the compile-cache dir are recorded instead), HBM, and whether runtime_id, host kernel pids and
the frontend actor are unchanged.

S-dp tokens that differ from the S reference are recorded per prompt (first differing position) and the gate keeps
going (``--stop-on-mismatch`` stops at the first mismatch of any kind); every later S-dp visit must equal the FIRST
S-dp visit exactly, and every H visit must equal the H reference. ``ok`` needs: no failure, H visits equal the H
reference, S-dp visits identical to each other, built == (first visit of the layout), runtime/pids/frontend unchanged,
every switch ok. ``summary.sdp_equals_s_reference`` is reported separately.

    rcm_exec run scripts/dp_switch_gate.py -- --reps 3 --inventory /tmp/hetero-inventory-only.json
    python scripts/dp_switch_gate.py --local --reps 2     # dry run: same body, local Ray, CPU block, synthetic engine
    python scripts/dp_switch_gate.py --dry-run            # build the submit payload, submit nothing

TPU_MULTIPROCESS_DP=0 is set in every rank env and in the frontend. The fresh compile-cache dir per run
(JAX_COMPILATION_CACHE_DIR and VLLM_XLA_CACHE_PATH) makes the first S-dp visit pay its compiles and every later one
hit the cache. On the cluster a failure never rebuilds the block: it stops the gate and becomes evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vproc_gate as vg  # noqa: E402  (same StageRunner/payload conventions as G1/G2)

ENGINE_ID = "A"
HOSTS = (0, 1, 2, 3)
H, SDP = (16, 1), (8, 2)
LAYOUT_NAMES = {H: "H", SDP: "S-dp"}
REFERENCE_TP = {H: "16", SDP: "8"}  # ShardedRestore references are keyed by tensor-parallel size
KERNEL_PREFIX = vg.KERNEL_PREFIX
LOCAL_REQUEST = vg.LOCAL_REQUEST
SAMPLING = vg.SAMPLING


class GateStop(RuntimeError):
    """``--stop-on-mismatch``: tokens differed."""


# ------------------------------------------------------------------------------ pure helpers


def compare_tokens(got, reference):
    """Per prompt: equal?, first differing position, lengths. ``got``/``reference``: ``[[int]]``."""
    rows = []
    for index in range(max(len(got), len(reference))):
        a = [int(t) for t in got[index]] if index < len(got) else None
        b = [int(t) for t in reference[index]] if index < len(reference) else None
        first = None
        if a is not None and b is not None and a != b:
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        rows.append({"prompt": index, "equal": a == b, "first_diff": first,
                     "got_len": None if a is None else len(a), "ref_len": None if b is None else len(b)})
    return {"equal": all(r["equal"] for r in rows), "prompts": len(rows),
            "mismatched": [r for r in rows if not r["equal"]]}


def same_actor(a, b):
    """The very same Ray actor (or the same object for in-process frontends)."""
    if hasattr(a, "_actor_id") or hasattr(b, "_actor_id"):
        return getattr(a, "_actor_id", None) == getattr(b, "_actor_id", None) is not None
    return a is b


def cache_entries(path):
    """Entries of the compile-cache dir (None when it is not visible from this process)."""
    if not path:
        return None
    try:
        return len(os.listdir(path))
    except OSError:
        return None


def switch_row(index, report, runtime_ids, pids, frontend_same, cache_before, cache_after):
    return {"index": index, "ok": report.get("ok"), "source": report.get("source_layout"),
            "target": report.get("target_layout"), "built": report.get("built"),
            # revisits reuse the parked layout's executables: the explicit expectation, not inferred from compile counts
            "cache_hit": None if report.get("built") is None else not report["built"],
            "built_by_host": next((p.get("built_by_host") for p in report.get("phases", [])
                                   if p.get("name") == "worker_switch"), None),
            "total_s": report.get("total_s"),
            "unaccounted_s": report.get("unaccounted_s"), "controller_segments": report.get("segments"),
            "worker_segments": report.get("worker_segments"), "frontend_segments": report.get("frontend_segments"),
            "compiles": report.get("compiles"), "compiles_by_phase": {p["name"]: p.get("compiles") for p in
                                                                      report.get("phases", []) if "compiles" in p},
            "cache_entries_before": cache_before, "cache_entries_after": cache_after,
            "cache_entries_added": None if None in (cache_before, cache_after) else cache_after - cache_before,
            "hbm": report.get("hbm"), "runtime_ids": runtime_ids, "host_pids": pids,
            "frontend_unchanged": frontend_same,
            "generation": report.get("generation"), "layout_id": report.get("layout_id"),
            "layout_digest": report.get("layout_digest"), "comparison": report.get("comparison")}


# ------------------------------------------------------------------------------ the gate body


def new_evidence(args, platform):
    return {"ok": False, "gate": "G3", "reps": args.reps, "run_id": args.run_id, "platform": platform,
            "engine_id": ENGINE_ID, "stop_on_mismatch": bool(args.stop_on_mismatch), "deploy": None, "switches": [],
            "checks": [], "phases": [], "first_sdp": None, "switch_failure": None, "cache_dir": None,
            "failure": None, "cleanup": None}


def serve_check(ctl, evidence, label, request, reference_for, first_sdp, stop_on_mismatch):
    """Serve ``request`` on the engine in its current layout and check it. Returns the visit row."""
    started = time.time()
    result = ctl.serve(ENGINE_ID, request, retry=False)
    spec = ctl.engines[ENGINE_ID]
    reference = reference_for(spec)
    against = compare_tokens(result["tokens"], reference)
    row = {"phase": label, "layout": list(spec.layout), "layout_name": LAYOUT_NAMES.get(spec.layout),
           "tp": spec.tp_size, "dp": spec.dp_size, "generation": spec.generation, "serve_s": result["serve"]["elapsed_s"],
           "wall_s": time.time() - started, "reference": f"tp{spec.tp_size}", "equal_reference": against["equal"],
           "prompts": against["prompts"], "mismatches": against["mismatched"], "ranks_agree": result["serve"]["ranks_agree"]}
    if spec.layout == SDP:
        if first_sdp["tokens"] is None:
            first_sdp["tokens"], first_sdp["label"] = result["tokens"], label
            row["first_sdp_visit"] = True
        else:
            same = compare_tokens(result["tokens"], first_sdp["tokens"])
            row.update(first_sdp_visit=False, equal_first_sdp=same["equal"], first_sdp_mismatches=same["mismatched"],
                       first_sdp_label=first_sdp["label"])
            ctl.measure.check_tokens(f"{label}/A-vs-first-S-dp", result["tokens"], first_sdp["tokens"])
    else:
        ctl.measure.check_tokens(f"{label}/A", result["tokens"], reference)
    evidence["checks"].append(row)
    evidence["phases"].append({"label": label, "layout": list(spec.layout), "serve_wall_s": row["wall_s"]})
    mismatch = (not against["equal"]) or row.get("equal_first_sdp") is False
    if mismatch and stop_on_mismatch:
        raise GateStop(f"{label}: tokens differ from {row['reference']} reference (first diffs "
                       f"{[(m['prompt'], m['first_diff']) for m in against['mismatched'][:5]]})")
    return row


def run_gate(reps, ctl, spec, request, reference_for, evidence, stop_on_mismatch=False, cache_dir=None):
    """H deploy (cold) -> S-dp -> H -> reps x (S-dp -> H); every visit checked."""
    describe = ctl.describe()
    runtime = {d.runtime_id for d in describe}
    pids = [d.pid for d in describe]
    evidence["cache_dir"] = cache_dir
    evidence["deploy"] = ctl.deploy([spec])
    actor = ctl.frontends[ENGINE_ID]
    first_sdp = {"tokens": None, "label": None}
    serve_check(ctl, evidence, "deploy_H", request, reference_for, first_sdp, stop_on_mismatch)
    sequence = [("S-dp", "sdp0"), ("H", "h0")] + [(t, f"{n}{i + 1}") for i in range(reps)
                                                  for t, n in (("S-dp", "sdp"), ("H", "h"))]
    for index, (target, label) in enumerate(sequence):
        cache_before = cache_entries(cache_dir)
        try:
            report = ctl.switch_layout(ENGINE_ID, target)
        except Exception as error:
            evidence["switch_failure"] = getattr(error, "report", None)
            raise
        describe = ctl.describe()
        runtime_ids = [d.runtime_id for d in describe]
        row = switch_row(index, report, runtime_ids, [d.pid for d in describe],
                         same_actor(ctl.frontends.get(ENGINE_ID), actor), cache_before, cache_entries(cache_dir))
        row["runtime_unchanged"] = set(runtime_ids) == runtime and len(set(runtime_ids)) == 1
        row["pids_unchanged"] = row["host_pids"] == pids
        row["label"] = label
        evidence["switches"].append(row)
        serve_check(ctl, evidence, f"{label}_{LAYOUT_NAMES[tuple(report['target_layout'])]}", request,
                    reference_for, first_sdp, stop_on_mismatch)
    after = ctl.describe()
    evidence["runtime_unchanged"] = {d.runtime_id for d in after} == runtime and len(runtime) == 1
    evidence["pids_unchanged"] = [d.pid for d in after] == pids
    evidence["frontend_unchanged"] = same_actor(ctl.frontends.get(ENGINE_ID), actor)
    evidence["first_sdp"] = first_sdp["label"]


def summarize(evidence):
    from hetero.vproc import measure
    switches, checks = evidence["switches"], evidence["checks"]
    sdp = [c for c in checks if c["layout"] == list(SDP)]
    hs = [c for c in checks if c["layout"] == list(H)]
    seen, expected_built = {H}, []
    for row in switches:
        target = tuple(row["target"] or ())
        expected_built.append(target not in seen)
        seen.add(target)
    summary = {
        "switches": [{k: row.get(k) for k in ("index", "label", "source", "target", "built", "ok", "total_s",
                                              "compiles", "cache_hit", "built_by_host", "cache_entries_added", "controller_segments",
                                              "frontend_segments", "runtime_unchanged", "pids_unchanged",
                                              "frontend_unchanged")}
                     | {"vs": measure.compare(row["total_s"]) if isinstance(row.get("total_s"), (int, float)) else None}
                     for row in switches],
        "visits": len(checks),
        "h_equals_reference": bool(hs) and all(c["equal_reference"] for c in hs),
        "sdp_equals_s_reference": bool(sdp) and all(c["equal_reference"] for c in sdp),
        "sdp_mismatches": [{"phase": c["phase"], "prompts": [(m["prompt"], m["first_diff"]) for m in c["mismatches"]]}
                           for c in sdp if not c["equal_reference"]],
        "sdp_visits_identical": bool(sdp) and all(c.get("equal_first_sdp", True) for c in sdp),
        "built_flags": [row["built"] for row in switches], "built_as_expected":
            [row["built"] for row in switches] == expected_built,
        "runtime_unchanged": evidence.get("runtime_unchanged") and all(r["runtime_unchanged"] for r in switches),
        "pids_unchanged": evidence.get("pids_unchanged") and all(r["pids_unchanged"] for r in switches),
        "frontend_unchanged": evidence.get("frontend_unchanged") and all(r["frontend_unchanged"] for r in switches),
        # Only the synthetic frontend compares per-rank outputs; a vLLM frontend reports None
        # (one output path), which is not a disagreement.
        "ranks_agree": all(c["ranks_agree"] is not False for c in checks) if checks else False}
    evidence["summary"] = summary
    evidence["ok"] = bool(evidence["failure"] is None and switches and summary["h_equals_reference"]
                          and summary["sdp_visits_identical"] and summary["built_as_expected"]
                          and summary["runtime_unchanged"] and summary["pids_unchanged"]
                          and summary["frontend_unchanged"] and summary["ranks_agree"]
                          and all(row["ok"] for row in switches))


def finish(args, evidence):
    summarize(evidence)
    vg.write(args.output, evidence)
    print(json.dumps(evidence["summary"], indent=1, default=str))
    print(f"ok={evidence['ok']} evidence={args.output}")


# ------------------------------------------------------------------------------ local dry run


def local(args):
    """Dry run of the remote body: local Ray, 4 x 4 CPU block, one synthetic engine over all four hosts."""
    import ray
    from hetero.vproc.controller import BlockController
    from hetero.vproc.endpoints import InMemoryEndpoints
    from hetero.vproc.host_kernel import kill_block, spawn_block
    from hetero.vproc.synthetic_engine import expected_tokens
    from hetero.vproc.types import EngineSpec
    os.environ["TPU_MULTIPROCESS_DP"] = "0"
    evidence = new_evidence(args, "cpu")
    ray.init(address="local", num_cpus=16, include_dashboard=False, logging_level="ERROR")
    actors, ctl = [], None
    try:
        actors = spawn_block(4, 4, name_prefix=KERNEL_PREFIX, actor_options={"namespace": "omp_hetero"},
                             preinit=None)
        spec = EngineSpec(ENGINE_ID, "synthetic", HOSTS, H[0], {"seed": 7}, 0, H[1])
        ctl = BlockController(actors, InMemoryEndpoints(), respawn=vg.refuse_respawn, engine_kind="synthetic",
                              engine_model={"seed": 7}, checksums=True)
        run_gate(args.reps, ctl, spec, LOCAL_REQUEST, lambda s: expected_tokens(s, LOCAL_REQUEST), evidence,
                 args.stop_on_mismatch)
    except Exception:
        evidence["failure"] = traceback.format_exc()
    finally:
        if ctl is not None:
            evidence["report"] = ctl.report()
            ctl.close()
        else:
            kill_block(actors)
        ray.shutdown()
    finish(args, evidence)
    if not evidence["ok"]:
        raise SystemExit(1)


# ------------------------------------------------------------------------------ cluster body


def remote(args):
    """Runs as the Ray job on the cluster head: block A, real vLLM + tpu-inference engine, H deployed cold."""
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from hetero.evidence import emit_measurements
    from hetero.layout import builtin_layout, resolve_layout
    from hetero.native_backend import prepare_backend
    from hetero.slice_inventory import SliceInventory
    from hetero.vproc.controller import BlockController
    from hetero.vproc.endpoints import InMemoryEndpoints
    from hetero.vproc.host_kernel import kill_block, spawn_block
    from hetero.vproc.types import EngineSpec
    os.environ["TPU_MULTIPROCESS_DP"] = "0"  # the frontend process (this driver and the actor's runtime_env)
    ray.init(address="auto", ignore_reinit_error=True)
    evidence = new_evidence(args, "tpu")
    actors, ctl, pg = [], None, None
    try:
        payload = json.loads((ROOT / os.environ["VPROC_PAYLOAD"]).read_text())
        if payload["sampling"] != SAMPLING:
            raise RuntimeError(f"reference sampling {payload['sampling']} != {SAMPLING}")
        inventory = SliceInventory.from_dict(payload["inventory"])
        box = resolve_layout(inventory, builtin_layout("H")).instances[0]
        evidence["box"] = box.to_dict()
        if any(row["state"] != "REMOVED" for row in ray.util.placement_group_table().values()):
            raise RuntimeError("refusing: placement groups owned by another experiment exist")
        evidence["backend"] = prepare_backend(box.host_ips, apply=False)
        if not evidence["backend"]["ready"]:
            raise RuntimeError("native backend must already be prepared; the gate does not mutate packages")
        pg = placement_group([{"TPU": 4, f"node:{host.node_ip}": 0.001} for host in box.hosts],
                             strategy="STRICT_SPREAD", name=f"dp-switch-gate-{args.run_id}")
        ray.get(pg.ready(), timeout=120)
        # fresh per run even when a run id is retried: unique suffix, and refuse a directory that already exists
        cache = f"/models/jax_cache/dpswitch-{args.run_id}-{int(time.time())}-{os.getpid()}"
        if os.path.exists(cache):
            raise RuntimeError(f"compile cache dir {cache} already exists; the gate needs a fresh one")
        evidence["cache_dir"] = cache
        rank_env = [{**box.process_env(rank), "JAX_COMPILATION_CACHE_DIR": cache, "VLLM_XLA_CACHE_PATH": cache,
                     "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1", "TPU_MULTIPROCESS_DP": "0"}
                    for rank in range(len(box.hosts))]
        actor_options = [{"namespace": "omp_hetero", "num_cpus": 0, "resources": {"TPU": 4}, "max_restarts": 0,
                          "scheduling_strategy": PlacementGroupSchedulingStrategy(
                              placement_group=pg, placement_group_bundle_index=rank,
                              placement_group_capture_child_tasks=True)}
                         for rank in range(len(box.hosts))]
        actors = spawn_block(len(box.hosts), 4, platform="tpu", rank_env=rank_env, actor_options=actor_options,
                             name_prefix=KERNEL_PREFIX, init_timeout_s=600)
        ctl = BlockController(actors, InMemoryEndpoints(), respawn=vg.refuse_respawn, engine_kind="vllm",
                              engine_model={}, checksums=False)
        request = {"prompt": payload["prompts"], "steps": SAMPLING["max_tokens"], "ignore_eos": SAMPLING["ignore_eos"]}
        references = payload["references"]
        spec = EngineSpec(ENGINE_ID, "vllm", HOSTS, H[0], {}, 0, H[1])
        run_gate(args.reps, ctl, spec, request,
                 lambda s: references[REFERENCE_TP[tuple(s.layout)]]["outputs"], evidence,
                 args.stop_on_mismatch, cache_dir=cache)
    except Exception:
        evidence["failure"] = traceback.format_exc()
    finally:
        if ctl is not None:
            try:
                evidence["report"] = ctl.report()
            except Exception:
                evidence["report_failure"] = traceback.format_exc()
            ctl.close()
        else:
            kill_block(actors)
        if pg is not None:
            remove_placement_group(pg)
        evidence["cleanup"] = {"kernels": len(actors), "placement_group_removed": pg is not None,
                               "time": time.time()}
        summarize(evidence)
        time.sleep(15)
        emit_measurements(evidence)


# ------------------------------------------------------------------------------ submit


def submit(args):
    payload = vg.build_payload(args)
    payload_path = ROOT / f"vproc-input-{args.run_id}.json"
    argv = ["python3", "-u", "scripts/dp_switch_gate.py", "--remote", "--reps", str(args.reps),
            "--run-id", args.run_id] + (["--stop-on-mismatch"] if args.stop_on_mismatch else [])
    env = {"PYTHONUNBUFFERED": "1", "VPROC_PAYLOAD": payload_path.name, "TPU_MULTIPROCESS_DP": "0"}
    if args.dry_run:
        print(json.dumps({"entrypoint": " ".join(argv), "env": env, "prompts": len(payload["prompts"]),
                          "references": {k: v["file"] for k, v in payload["references"].items()},
                          "reference_for": {LAYOUT_NAMES[k]: f"tp{v}" for k, v in REFERENCE_TP.items()},
                          "engine": {"id": ENGINE_ID, "hosts": list(HOSTS), "deploy_layout": list(H),
                                     "sequence": ["S-dp", "H"] + ["S-dp", "H"] * args.reps},
                          "output": str(args.output)}, indent=1))
        return
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor

    @stage(name="dp_switch_G3", order_idx=1, expected_s=args.timeout)
    def run(state, timing):
        supervisor = state.get("supervisor")
        if supervisor is None:
            supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
            state.set("supervisor", supervisor)
        job = state.get("submission_id")
        if job is None:
            payload_path.write_text(json.dumps(payload))
            try:
                job = supervisor.submit_job(entrypoint=" ".join(argv), runtime_env={
                    "working_dir": str(ROOT), "env_vars": env,
                    "excludes": [".venv*", ".git", ".jj", "__pycache__", "*.pyc", "results", "**/results/**",
                                 "reports", "data"]},
                    metadata={"purpose": "dp-switch-G3"}, name=f"dp-switch-G3-{args.run_id}")
            finally:
                payload_path.unlink(missing_ok=True)
            state.set("submission_id", job)
        result = supervisor.supervise_until_completion(submission_id=job, timeout_s=args.timeout,
                                                       stage_name="dp_switch_G3", state_store=state)
        measured = result["measurements"]
        vg.write(args.output, {"submission_id": job, "status": result["status"], "measurements": measured})
        # A failed gate is evidence, not an RCM trap holding the block.
        if not measured:
            raise RuntimeError("gate returned no machine-readable evidence")

    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(run)
    if not runner.run():
        raise SystemExit(1)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=3, help="S-dp/H cycles after the first S-dp -> H")
    parser.add_argument("--stop-on-mismatch", action="store_true", help="stop at the first token mismatch")
    parser.add_argument("--remote", action="store_true", help="cluster body (Ray job entrypoint)")
    parser.add_argument("--local", action="store_true", help="dry run on local Ray with a CPU block")
    parser.add_argument("--dry-run", action="store_true", help="build the payload, print the job, submit nothing")
    parser.add_argument("--inventory", type=Path, default=Path("/tmp/hetero-inventory-only.json"))
    parser.add_argument("--ray-url", default="http://127.0.0.1:8265")
    parser.add_argument("--run-id", default="dpsw-" + str(int(time.time())))
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = ROOT / f"results/in_engine_dp/G3-{'local' if args.local else args.run_id}.json"
    return args


if __name__ == "__main__":
    options = parse()
    if options.remote:
        remote(options)
    elif options.local:
        local(options)
    else:
        submit(options)
