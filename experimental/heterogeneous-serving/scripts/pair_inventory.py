#!/usr/bin/env python3
"""Slice inventory for ONE pair of a shared slice, without touching any other host's TPUs.

``hetero_ctl inventory`` (observe_slice) initializes JAX across the whole slice, which is forbidden while
another session serves on part of it.  This builds the same inventory object from:

  * the live Ray TPU nodes (ids, IPs, chip counts) and their ``ray_node_generation`` (no task runs on
    any host for this);
  * each host's physical ``TPU_WORKER_ID`` (from the KubeRay pod env, passed in with ``--worker-ids``)
    and the worker-id -> chip-coordinates map of an earlier full-slice ``hetero_ctl inventory`` of the
    same 4x8 pool shape (``--template``);
  * a two-host JAX probe on the target pair only: one actor per pair host inside a STRICT_SPREAD
    placement group holding exactly those hosts' 4 TPU chips, each applying the sub-slice process
    environment the deploy gives that rank (``ResolvedInstance.process_env``) and initializing the
    2-process runtime.  It proves the pair forms a working 2x4 ICI sub-slice in the resolved rank
    order and records the runtime chip coordinates; a sub-slice runtime reports coordinates relative
    to the sub-slice, so the absolute positions of the pair (and of every other host) come from the
    template map, not from this probe.

Ray driver with ``log_to_driver=False`` (other sessions' unattributed log lines must not reach it).

    OMP_RCM_DIR=/tmp/omp_rcm-<name> .venv/bin/python rcm_exec run scripts/pair_inventory.py \\
        --hosts 192.168.102.116,192.168.102.39 --instance small_a --layout S \\
        --template /tmp/hetero-inventory-only.json --worker-ids '{"192.168.102.116": 7, ...}' \\
        --hostnames '{"192.168.102.116": "gke-tpu-...", ...}' --output OUT.json --inventory-out INV.json
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_inventory(nodes, generation, worker_ids, hostnames, template):
    """The inventory dict: live node ids + physical worker ids, coordinates from the template map."""
    by_worker = {int(h["tpu_worker_id"]): h for h in template["hosts"]}
    live = {n["node_ip"]: n for n in nodes}
    if set(live) != set(worker_ids):
        raise RuntimeError(f"live Ray TPU hosts {sorted(live)} != --worker-ids hosts {sorted(worker_ids)}")
    if sorted(int(v) for v in worker_ids.values()) != sorted(by_worker):
        raise RuntimeError(f"worker ids {sorted(worker_ids.values())} != template's {sorted(by_worker)}")
    hosts = []
    for ip, wid in worker_ids.items():
        ref = by_worker[int(wid)]
        if int(live[ip]["tpu_count"]) != int(ref["tpu_count"]):
            raise RuntimeError(f"{ip}: {live[ip]['tpu_count']} chips, template worker {wid} has {ref['tpu_count']}")
        hosts.append({"node_ip": ip, "ray_node_id": live[ip]["ray_node_id"], "hostname": hostnames.get(ip, ""),
                      "tpu_worker_id": int(wid), "tpu_name": ref["tpu_name"], "tpu_count": int(ref["tpu_count"]),
                      "chip_coords": [list(c) for c in ref["chip_coords"]],
                      "jax_process_index": int(ref["jax_process_index"]),
                      "jax_global_device_count": int(ref["jax_global_device_count"])})
    hosts.sort(key=lambda h: tuple(min(c[a] for c in h["chip_coords"]) for a in range(3))[::-1])
    return {k: template[k] for k in ("slice_name", "chip_dims", "host_chip_dims", "host_grid_dims")} | {
        "hosts": hosts, "ray_session": "pair-inventory", "observed_at": time.time(),
        "ray_node_generation": generation}


def remote(args):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from hetero.evidence import emit_measurements
    from hetero.layout import builtin_layout, resolve_layout
    from hetero.slice_inventory import SliceInventory, _node_generation, discover_ray_tpu_nodes

    ray.init(address="auto", logging_level="WARNING", log_to_driver=False)
    started = time.time()
    nodes = discover_ray_tpu_nodes()
    generation = _node_generation(nodes)
    template = json.loads(args.template_json)
    template = template.get("measurements", {}).get("inventory", template)
    raw = build_inventory(nodes, generation, json.loads(args.worker_ids), json.loads(args.hostnames), template)
    inventory = SliceInventory.from_dict(raw)
    layout = resolve_layout(inventory, builtin_layout(args.layout))
    instance = next(i for i in layout.instances if i.instance_id == args.instance)
    pair = args.hosts.split(",")
    if list(instance.host_ips) != pair:
        raise RuntimeError(f"{args.layout}:{args.instance} resolves to {list(instance.host_ips)}, not {pair}")

    @ray.remote(num_cpus=1, max_restarts=0)
    class PairProbe:
        def facts(self):
            env = os.environ
            return {"node_ip": ray.util.get_node_ip_address(), "hostname": socket.gethostname(), "pid": os.getpid(),
                    "tpu_worker_id": env.get("TPU_WORKER_ID"), "tpu_name": env.get("TPU_NAME"),
                    "tpu_visible_chips": env.get("TPU_VISIBLE_CHIPS")}

        def observe(self, process_env):
            # As HeteroTPUWorker.init_device, before the first device touch.  Nothing is compiled; still keep
            # any cache write off the layout's /models FUSE default.
            os.environ.update(process_env, JAX_COMPILATION_CACHE_DIR="/tmp/omp_pair_probe_xla")
            t0 = time.time()
            import jax
            devices = jax.devices()
            local = jax.local_devices()
            return {"node_ip": ray.util.get_node_ip_address(), "process_index": int(jax.process_index()),
                    "process_count": int(jax.process_count()), "global_device_count": len(devices),
                    "local_device_count": len(local), "device_kind": local[0].device_kind if local else None,
                    "runtime_local_chip_coords": sorted([int(v) for v in d.coords] for d in local),
                    "runtime_global_chip_coords": sorted([int(v) for v in d.coords] for d in devices),
                    "init_s": round(time.time() - t0, 3), "jax_version": jax.__version__,
                    "env": {k: os.environ.get(k) for k in sorted(process_env)}}

    per_host = float(nodes[0]["tpu_count"])
    pg = placement_group([{"TPU": per_host, "CPU": 1.0, f"node:{ip}": 0.001} for ip in pair],
                         strategy="STRICT_SPREAD", name=f"omp-pair-inventory-{args.instance}")
    actors = []
    try:
        ray.get(pg.ready(), timeout=180)
        for index, ip in enumerate(pair):
            actors.append(PairProbe.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=index,
                    placement_group_capture_child_tasks=False),
                resources={"TPU": per_host, f"node:{ip}": 0.001}).remote())
        facts = ray.get([a.facts.remote() for a in actors], timeout=300)
        for ip, fact in zip(pair, facts):
            if fact["node_ip"] != ip:
                raise RuntimeError(f"probe for {ip} landed on {fact['node_ip']}")
        envs = [instance.process_env(h.process_rank) for h in instance.hosts]
        observed = ray.get([a.observe.remote(env) for a, env in zip(actors, envs)], timeout=600)
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:  # noqa: BLE001
                pass
        try:
            remove_placement_group(pg)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(15.0)  # let the probe processes exit before anything claims the chips

    by_ip = {h["node_ip"]: h for h in raw["hosts"]}
    origin = [b[0] for b in instance.chip_box]
    checks = []
    for host, fact, obs in zip(instance.hosts, facts, observed):
        relative = sorted([c[a] - origin[a] for a in range(3)] for c in by_ip[host.node_ip]["chip_coords"])
        checks.append({
            "node_ip": host.node_ip, "process_rank": host.process_rank,
            "physical_worker_id_env": fact["tpu_worker_id"],
            "worker_id_matches": str(fact["tpu_worker_id"]) == str(by_ip[host.node_ip]["tpu_worker_id"]),
            "process_index_matches": obs["process_index"] == host.process_rank,
            "global_devices_ok": obs["global_device_count"] == instance.tp_size and obs["process_count"] == len(pair),
            "expected_relative_local_coords": relative,
            "relative_position_match": obs["runtime_local_chip_coords"] == relative,
        })
    ok = all(c["worker_id_matches"] and c["process_index_matches"] and c["global_devices_ok"]
             and c["relative_position_match"] for c in checks)
    inventory_out = dict(raw, observed_by=(
        "scripts/pair_inventory.py: live Ray node ids + generation; physical TPU_WORKER_ID from the pod env; "
        "chip coordinates from the template's worker-id map; two-host JAX sub-slice probe on " + args.hosts),
        template_generation=template.get("ray_node_generation"), pair_probe={"hosts": pair, "checks": checks})
    emit_measurements({"ok": ok, "inventory": inventory_out, "nodes": nodes, "ray_node_generation": generation,
                       "instance": instance.to_dict(), "facts": facts, "observed": observed, "checks": checks,
                       "layout_instances": {i.instance_id: list(i.host_ips) for i in layout.instances},
                       "wall_s": time.time() - started})


def submit(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    template = Path(args.template).read_text()
    argv = ["python3", "-u", "scripts/pair_inventory.py", "--remote", "--hosts", args.hosts, "--instance",
            args.instance, "--layout", args.layout, "--template-json", template, "--worker-ids", args.worker_ids,
            "--hostnames", args.hostnames]

    @stage(name="pair_inventory", order_idx=1, expected_s=args.timeout)
    def run(state, timing):
        supervisor = state.get("supervisor")
        if supervisor is None:
            supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
            state.set("supervisor", supervisor)
        job = state.get("submission_id")
        if job is None:
            job = supervisor.submit_job(entrypoint=shlex.join(argv), runtime_env={
                "working_dir": str(ROOT), "excludes": [".venv*", ".git", ".jj", "__pycache__", "*.pyc", "results",
                                                       "**/results/**", "reports", "data"],
                "env_vars": {"PYTHONUNBUFFERED": "1"}},
                metadata={"purpose": "pair-inventory"}, name=f"pair-inventory-{args.instance}")
            state.set("submission_id", job)
        result = supervisor.supervise_until_completion(submission_id=job, timeout_s=args.timeout,
                                                       stage_name="pair_inventory", state_store=state)
        measured = result["measurements"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"submission_id": job, "status": result["status"],
                                           "measurements": measured}, indent=2) + "\n")
        if not measured or measured.get("ok") is not True:
            raise RuntimeError("pair inventory not ok: " + json.dumps((measured or {}).get("checks"))[:2000])
        args.inventory_out.write_text(json.dumps(measured["inventory"], indent=1) + "\n")

    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(run)
    if not runner.run():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hosts", required=True, help="the pair, rank order: ip0,ip1")
    parser.add_argument("--instance", required=True)
    parser.add_argument("--layout", default="S")
    parser.add_argument("--template", help="earlier full-slice inventory JSON of the same pool shape")
    parser.add_argument("--template-json", help=argparse.SUPPRESS)
    parser.add_argument("--worker-ids", required=True, help='JSON {"ip": TPU_WORKER_ID} for every TPU host')
    parser.add_argument("--hostnames", default="{}", help='JSON {"ip": hostname}')
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory-out", type=Path)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--ray-url", default="http://127.0.0.1:8265")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.remote:
        remote(args)
    else:
        if not (args.template and args.output and args.inventory_out):
            parser.error("--template, --output and --inventory-out are required")
        submit(args)


if __name__ == "__main__":
    main()
