#!/usr/bin/env python3
"""vLLM-TorchTPU weight-load probe on one TPU v5e host, or one engine across several hosts, of the shared Ray cluster.

Local mode (default): an RCM stage that submits a Ray job whose driver (this file with --remote) runs
scripts/torchtpu_load_host.sh <phase> inside Ray tasks pinned to the TPU worker pods.
- One host (--hosts IP, default 192.168.102.199): the task holds the host's 4 TPU chips and vLLM runs its own
  multiprocessing executor (TP <= 4).
- Several hosts (--hosts IP0,IP1 in host-rank order, or --pair small_d to resolve them from the hetero layout and
  inventory): Ray mode. One task per host, none holding TPU chips, every one with TTL_RAY_HOSTS / TTL_HOST_RANK;
  on host rank 0 the serve phase runs a single `vllm serve` driver with the Ray executor, whose worker actors take
  the chips through a placement group pinned to these hosts (torchtpu_ray_pin_plugin.py).
Phases: setup (isolated venv + vllm-torchtpu==0.29.0 + sliced checkpoint copy to tmpfs), serve (vllm serve with
full timestamped log, greedy request), cleanup, pmupload/pmdelete (TPUFlex state per host), fusebench, h2dbench.

Run: flock-held locks; .venv/bin/python rcm_exec run scripts/torchtpu_load_probe.py --phase setup --run-id R ...
"""
import argparse
import json
import os
import pathlib
import shlex
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--remote', action='store_true', help='internal: run as the Ray job driver')
parser.add_argument('--ray-url', default='http://127.0.0.1:8265')
parser.add_argument('--phase', choices=['setup', 'serve', 'cleanup', 'fusebench', 'pmupload', 'pmdelete', 'h2dbench', 'pmserve',
                                        'venvupload', 'chipmap'],
                    required=True)
parser.add_argument('--run-id', required=True)
parser.add_argument('--hosts', '--node-ip', dest='hosts', default='192.168.102.199',
                    help='worker pod IP(s) in host-rank order, comma-separated; 2+ hosts = one engine, Ray executor')
parser.add_argument('--pair', help='resolve --hosts from the hetero S layout instance (e.g. small_d) and --inventory')
parser.add_argument('--inventory', default='/tmp/hetero-inventory-only.json')
parser.add_argument('--peer-grace', type=float, default=300, help='seconds the other hosts may outlive host rank 0')
parser.add_argument('--timeout', type=float, default=3600)
parser.add_argument('--out-dir', type=pathlib.Path, default=ROOT / 'results/torchtpu-load')
parser.add_argument('--host-env', action='append', default=[], help='KEY=VALUE passed to the host script (TTL_*)')
args = parser.parse_args()

if args.pair and not args.remote:
    sys.path.insert(0, str(ROOT))
    from hetero import layout as _layout
    from hetero.slice_inventory import SliceInventory
    _inv = SliceInventory.from_dict(json.loads(pathlib.Path(args.inventory).read_text()))
    _inst = next(i for i in _layout.resolve_layout(_inv, _layout.builtin_layout('S')).instances
                 if i.instance_id == args.pair)
    args.hosts = ','.join(_inst.host_ips)
    print(f'pair {args.pair}: hosts {args.hosts} (inventory {args.inventory}, session {_inv.ray_session})', flush=True)


def remote_main():
    import time
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    # log_to_driver off: the driver would also receive every unattributed log line of the shared cluster's other
    # tenants ("(pid=, ip=...)": 635k lines of libtpu dumps from another probe in m8wpB2, a 480 MB job log; the Ray
    # head was OOM-killed soon after). The host scripts' own logs are pulled from the pods.
    ray.init(address='auto', logging_level='WARNING', log_to_driver=False)
    hosts = [h for h in args.hosts.split(',') if h]
    ray_mode = len(hosts) > 1
    nodes = {}
    for ip in hosts:
        node = next(n for n in ray.nodes() if n['Alive'] and n['NodeManagerAddress'] == ip and n['Resources'].get('TPU'))
        nodes[ip] = node
        print(f"target node {node['NodeID']} {ip} resources={node['Resources']}", flush=True)
    host_env = dict(kv.split('=', 1) for kv in args.host_env)
    if ray_mode:
        host_env.update({'TTL_RAY_HOSTS': ','.join(hosts), 'TTL_RAY_VERSION': ray.__version__,
                         'TTL_RAY_ADDRESS': ray.get_runtime_context().gcs_address})

    def task(phase, run_id, host_env, script_dir):
        import os, socket, subprocess, time
        env = {k: v for k, v in os.environ.items() if not k.startswith(('TPU_VISIBLE', 'RAY_'))}
        env.update({'TTL_RUN': run_id, **host_env})
        t0 = time.time()
        rc = subprocess.run(['bash', os.path.join(script_dir, 'torchtpu_load_host.sh'), phase], env=env).returncode
        return {'host': socket.gethostname(), 'rank': host_env.get('TTL_HOST_RANK', '0'), 'rc': rc,
                'seconds': round(time.time() - t0, 1)}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    refs = []
    for rank, ip in enumerate(hosts):
        # Ray mode: the host tasks hold no TPU chips; the engine's worker actors take them via its placement group.
        opts = dict(num_cpus=1, max_retries=0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nodes[ip]['NodeID'], soft=False))
        if not ray_mode:
            opts['resources'] = {'TPU': 4}
        env = dict(host_env, TTL_HOST_RANK=str(rank)) if ray_mode else host_env
        refs.append(ray.remote(**opts)(task).remote(args.phase, args.run_id, env, script_dir))
    if args.phase != 'serve':  # every host does the same work
        results = ray.get(refs)
        for r in results:
            print('TOPOLOGY_RESULT_JSON=' + json.dumps(r), flush=True)
        sys.exit(0 if all(r.get('rc') == 0 for r in results) else 1)
    results = [ray.get(refs[0])]  # serve: host rank 0 runs the engine; the other hosts follow it
    print('TOPOLOGY_RESULT_JSON=' + json.dumps(results[0]), flush=True)
    deadline = time.time() + args.peer_grace
    for ref in refs[1:]:
        ready, _ = ray.wait([ref], timeout=max(1.0, deadline - time.time()))
        if ready:
            results.append(ray.get(ref))
        else:
            ray.cancel(ref, force=True)
            results.append({'rc': 'cancelled', 'after_s': args.peer_grace})
        print('TOPOLOGY_RESULT_JSON=' + json.dumps(results[-1]), flush=True)
    sys.exit(0 if all(r.get('rc') == 0 for r in results) else 1)


if args.remote:
    remote_main()
    sys.exit(0)

sys.path.insert(0, str(ROOT))
from omp_rcm import StageRunner, stage
from omp_rcm.ray_job import RayJobSupervisor


@stage(name=f'torchtpu_{args.phase}', order_idx=1, expected_s=max(60.0, args.timeout))
def run_remote(state, timing):
    supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
    work = pathlib.Path(tempfile.mkdtemp(prefix='torchtpu_probe_'))
    for name in ('torchtpu_load_probe.py', 'torchtpu_load_host.sh', 'torchtpu_ckpt_copy.py', 'torchtpu_load_plugin.py',
                 'torchtpu_gcs_tools.py', 'torchtpu_fuse_bench.py', 'torchtpu_req_probe.py', 'pm_torchtpu_plugin.py',
                 'pm_artifacts.py', 'pm_zygote.py', 'pm_source_patches.py', 'torchtpu_h2d_bench.py',
                 'torchtpu_ray_pin_plugin.py', 'pm_ray_zygote.py', 'torchtpu_chipmap.py'):
        shutil.copy(HERE / name, work / name)
    remote_args = ['--remote', '--phase', args.phase, '--run-id', args.run_id, '--hosts', args.hosts,
                   '--peer-grace', str(args.peer_grace)]
    for kv in args.host_env:
        remote_args += ['--host-env', kv]
    sid = supervisor.submit_job(
        entrypoint=shlex.join(['python3', '-u', 'torchtpu_load_probe.py', *remote_args]),
        runtime_env={'working_dir': str(work), 'env_vars': {'PYTHONUNBUFFERED': '1'}},
        metadata={'purpose': 'torchtpu-weight-load-probe', 'phase': args.phase, 'run_id': args.run_id},
        name=f'torchtpu-{args.phase}',
    )
    try:
        result = supervisor.supervise_until_completion(
            submission_id=sid, timeout_s=args.timeout, stage_name=f'torchtpu_{args.phase}', state_store=state)
    finally:
        out = args.out_dir / args.run_id
        out.mkdir(parents=True, exist_ok=True)
        text = supervisor._full_outputs.get(sid, '')
        (out / f'rayjob_{args.phase}_{sid}.log').write_text(text)
    (out / f'{args.phase}_result.json').write_text(json.dumps(
        {'submission_id': sid, 'status': result['status'], 'measurements': result['measurements']}, indent=2) + '\n')
    print('[OMP_EVENT: TORCHTPU_PHASE_DONE phase=' + args.phase + ']', flush=True)


if __name__ == '__main__':
    runner = StageRunner(silence_timeout_s=5400)
    runner.register(run_remote)
    if not runner.run():
        raise SystemExit(1)
