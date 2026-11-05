#!/usr/bin/env python3
"""Hot standby for the autoscale-burst experiment (arm 6): a pre-spawned, parked engine opened by the controller.

On a standby slice whose TPUFlex state already sits in /dev/shm (startup_state.py fetch into the arm's run
scope), this Ray job pre-spawns the instance exactly as the warm-pm arm does (hetero/deployment.py
prespawn_deployment: CPU-only placement groups, engine actors, workers parked on the release gate before their
first device touch), waits until every worker is parked, then long-polls the in-cluster controller
(asb_loadgen.py router, GET /controller/trigger/<k>) and opens the gate the moment trigger k fires:
release_prepared -> commit_prespawned (restore from /dev/shm, serve on :8000).  deploy_prespawned does the same
with the gate opened at once; only the wait between park and release is new here.

    .venv/bin/python rcm_exec run scripts/asb_hot_standby.py --ray-url http://127.0.0.1:8271 \
        --inventory <inventory-live.json> --output <json> --gate-url http://asb-router-x.asb:9000/controller/trigger/1 \
        -- --layout S1 --model /models/gemma-4-31b --max-model-len 8192 --max-num-seqs 512 \
           --max-num-batched-tokens 2048 --engine-settings '{...}' --instance-engine-settings '{...}' \
           --partition --instances small_a
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNTIME_EXCLUDES = [".venv*", ".git", ".jj", "formalization", "skaffold", "__pycache__", "*.pyc", "docs", "results",
                    "experiments", "reports", "**/results/**", "**/reports/**", "data"]


def _post_event(url: str, body: dict) -> None:
    import urllib.request
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json"})
        urllib.request.urlopen(req, timeout=3).read()
    except Exception:  # noqa: BLE001 - the timeline event is best effort; the measurements carry the clocks
        pass


class GateClosed(Exception):
    """The run ended (controller closed) before this standby's trigger fired."""


def wait_open(gate_url: str, timeout_s: float, poll_wait_s: float = 20.0) -> dict:
    """Long-poll the controller until trigger k exists (HTTP 200 with the decision record); 410 = run closed."""
    import urllib.error
    import urllib.request
    deadline = time.time() + timeout_s
    errors = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{gate_url}?wait={poll_wait_s}", timeout=poll_wait_s + 10) as resp:
                if resp.status == 200:
                    return {"decision": json.loads(resp.read() or b"{}"), "seen_at": time.time(), "poll_errors": errors}
        except urllib.error.HTTPError as exc:
            if exc.code == 410:
                raise GateClosed(gate_url) from exc
            if exc.code not in (204, 404):
                errors += 1
                time.sleep(0.5)
        except Exception:  # noqa: BLE001 - router restarts between runs; keep polling
            errors += 1
            time.sleep(0.5)
    raise TimeoutError(f"gate {gate_url} not opened within {timeout_s:.0f} s")


def remote(argv: list[str]) -> int:
    """The Ray job: parse hetero.cli's deploy arguments plus ours, park, wait, release, commit."""
    from hetero import cli
    from hetero.deployment import (abort_prespawned, await_parked, commit_prespawned, prespawn_deployment,
                                   release_prepared)
    from hetero.evidence import emit_event, emit_measurements

    parser = cli.build_parser()
    parser.add_argument("--gate-url", required=True)
    parser.add_argument("--event-url")
    parser.add_argument("--lane", default="")
    parser.add_argument("--park-timeout-s", type=float, default=900.0)
    parser.add_argument("--open-timeout-s", type=float, default=14400.0)
    args = parser.parse_args(["deploy", *argv])
    if not args.partition or not args.instances:
        raise SystemExit("hot standby is a partition deploy: --partition --instances")
    started = time.time()
    cli._connect()
    inventory, source = cli._resolve_inventory(args)
    config = cli._deployment_config(args)
    instance_ids = args.instances.split(",")
    if {bool(config.settings_for(i)["prespawn"]) for i in instance_ids} != {True}:
        raise SystemExit("hot standby needs engine setting prespawn=true")
    prepared = prespawn_deployment(inventory, config, instance_ids=instance_ids,
                                   gate_timeout_s=args.open_timeout_s + 600.0, partition=True)
    clocks = {"started_at": started}
    try:
        parked = await_parked(prepared, timeout_s=args.park_timeout_s)
        clocks["parked_at"] = time.time()
        emit_event("ASB_HOT_STANDBY_PARKED", lane=args.lane, parked_s=round(clocks["parked_at"] - started, 2))
        if args.event_url:
            _post_event(args.event_url, {"kind": "hot_standby_parked", "lane": args.lane, "t": clocks["parked_at"]})
        opened = wait_open(args.gate_url, args.open_timeout_s)
        clocks["signal_seen_at"] = opened["seen_at"]
        emit_event("ASB_HOT_STANDBY_OPEN", lane=args.lane, decision=opened["decision"])
        released = release_prepared(prepared)
        clocks["released_at"] = time.time()
        if args.event_url:
            _post_event(args.event_url, {"kind": "gate_open", "lane": args.lane, "t": clocks["released_at"],
                                         "signal_seen_at": opened["seen_at"],
                                         "trigger_t": (opened["decision"] or {}).get("t")})
    except GateClosed:
        # The run ended without this standby's trigger: remove only what the pre-spawn created (actors, groups,
        # gate), so nothing of it can hold the slice's TPUs afterwards.
        aborted = abort_prespawned(prepared)
        clocks["closed_at"] = time.time()
        emit_measurements({"action": "deploy", "mode": "hot-standby", "inventory_source": source, "lane": args.lane,
                           "gate_url": args.gate_url, "opened": None, "closed": True, "aborted": aborted,
                           "clocks": clocks, "prespawn": prepared.evidence(), "ok": True,
                           "elapsed_s": round(time.time() - started, 3)})
        return 0
    except BaseException as exc:
        emit_event("DEPLOY_FAILED", layout_id=prepared.layout_id, error=str(exc)[:400])
        abort_prespawned(prepared)
        raise
    report = commit_prespawned(prepared)
    clocks["committed_at"] = time.time()
    report["started_at"] = started
    report["prespawn"]["parked"] = parked
    report["prespawn"]["release"] = released
    report["prespawn"]["gate_open_at"] = min(prepared.released_at.values()) if prepared.released_at else None
    if args.event_url:
        _post_event(args.event_url, {"kind": "hot_standby_committed", "lane": args.lane, "t": clocks["committed_at"]})
    emit_measurements({"action": "deploy", "mode": "hot-standby", "inventory_source": source, "lane": args.lane,
                       "gate_url": args.gate_url, "opened": opened, "clocks": clocks, "deployment": report,
                       "gateway": None, "ok": True, "elapsed_s": round(time.time() - started, 3)})
    return 0


def submit() -> int:
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ray-url", default="http://127.0.0.1:8265")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--timeout", type=float, default=21600.0)
    parser.add_argument("--gate-url", required=True)
    parser.add_argument("--event-url")
    parser.add_argument("--lane", default="")
    parser.add_argument("--open-timeout-s", type=float, default=14400.0)
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    raw = pathlib.Path(args.inventory).read_text()
    json.loads(raw)
    remote_args = [*extra, "--inventory-file", args.inventory, "--gate-url", args.gate_url, "--lane", args.lane,
                   "--open-timeout-s", str(args.open_timeout_s)]
    if args.event_url:
        remote_args += ["--event-url", args.event_url]

    @stage(name="hot_standby", order_idx=1, expected_s=args.timeout)
    def run_remote(state, timing):
        supervisor = state.get("supervisor")
        if supervisor is None:
            supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
            state.set("supervisor", supervisor)
        job = state.get("submission_id")
        if job is None:
            job = supervisor.submit_job(
                entrypoint=shlex.join(["python3", "-u", "scripts/asb_hot_standby.py", "--remote", *remote_args]),
                runtime_env={"working_dir": str(ROOT), "excludes": RUNTIME_EXCLUDES, "pip": ["xds-protos==1.84.0"],
                             "env_vars": {"PYTHONUNBUFFERED": "1", "OMP_HETERO_DATA_DIR": "/tmp/omp_hetero/data",
                                          "OMP_HETERO_INVENTORY_JSON": raw}},
                metadata={"purpose": "asb-hot-standby", "lane": args.lane}, name="asb-hot-standby")
            state.set("submission_id", job)
        result = supervisor.supervise_until_completion(submission_id=job, timeout_s=args.timeout,
                                                       stage_name="hot_standby", state_store=state)
        if not result["measurements"]:
            raise RuntimeError("hot standby produced no structured measurement payload")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"submission_id": job, "status": result["status"],
                                           "remote_arguments": remote_args, "saved_at": time.time(),
                                           "measurements": result["measurements"]}, indent=2) + "\n")
        state.set("submission_id", None)

    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(run_remote)
    return 0 if runner.run() else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--remote"]:
        sys.exit(remote(sys.argv[2:]))
    sys.exit(submit())
