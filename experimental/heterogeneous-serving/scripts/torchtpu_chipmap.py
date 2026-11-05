#!/usr/bin/env python3
"""TPU chip enumeration probe (v5e, 4 chips per host). Run while the chips are free, on every host of the slice at once.

vLLM-TorchTPU's Ray executor gives TP rank r the local rank k = r mod 4 on its host, and torch_tpu opens the chip at
local enumeration index k (Ray's TPU_VISIBLE_CHIPS=k, or LOCAL_RANK=k copied into it when unset). Which physical chip
that is, and so which slice device_id and coords rank r gets, differs between VMs. This probe binds chips exactly the
way torch_tpu does in vLLM, one process per chip in a torch_tpu (tpu_dist) world, and each process reports its JAX
device (id, coords) and torch_tpu's global_device_id:

  round "slice" (--hosts ip0,ip1 --host-rank h; the authoritative one): the 2x(2*NH) world of the pair, with the
      environment the Ray V2 executor gives its workers (RANK=4h+k, LOCAL_RANK=k, TORCH_TPU_TOPOLOGY=2,2NH,1, slice
      builder addresses on every host, TPU_WORKER_ID=h): index k -> slice device_id and coords;
  round "local_rank" (single-host 2x2 world, LOCAL_RANK=k) and round "visible" (TPU_VISIBLE_CHIPS=(k+1) mod 4 next to
      LOCAL_RANK=k: which variable picks the chip). A host's 2x2 orientation alone is arbitrary (no inter-host links),
      so these are evidence, not the binding map. OPT-IN (--local-rounds), never under a shared Ray cluster: the
      localhost slice-builder mode starts libtpu's tpu_driver with --tpu_slice_builder_dump_chip=true, whose output
      reaches Ray's log monitor as unattributed lines that every log_to_driver driver of the cluster receives (a
      probe doing this flooded other jobs with ~600k lines; the Ray head was OOM-killed minutes later, 10:42Z).
  plus one JAX process over the 4 chips (JAX numbers devices by coords, so local_hardware_id is not the enumeration
  index) and the VFIO group -> PCI address of each chip.

Derived (from the slice round, else the local one):
  chips[k] = {index k, coords, device_id}: stock binding, local rank k -> device_id;
  canonical: chip indices sorted by coords (z, y, x): local rank k takes canonical[k] under the canonical binding
      (torchtpu_ray_pin_plugin.py, TTL_CHIP_BIND=canonical), so rank r gets the same device_id and coords on any slice;
  context: the hosts and host rank the map holds for (the plugin refuses a map from another slice).

Usage: torchtpu_chipmap.py OUT.json [--hosts ip0,ip1 --host-rank h] [--local-rounds]   (vLLM-TorchTPU venv python)
"""
import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import time

JAX_ALL = r'''
import json, jax
print("CHIPMAP_JAX=" + json.dumps([{"id": int(d.id), "hw": getattr(d, "local_hardware_id", None),
    "coords": [int(c) for c in d.coords]} for d in jax.local_devices()]), flush=True)
'''

TORCH_ONE = r'''
import json, os, sys
import torch
import torch.distributed as dist
k = int(os.environ["RANK"])
dist.init_process_group("tpu_dist", init_method=sys.argv[1], rank=k, world_size=int(os.environ["WORLD_SIZE"]))
x = torch.ones(1, device="tpu") + 1
import jax
d = jax.local_devices()[0]
try:
    from torch_tpu._internal.distributed import tpu_distributed
    gid = int(tpu_distributed.global_device_id())
except Exception as e:
    gid = repr(e)
print("CHIPMAP_RANK=" + json.dumps({"rank": k, "local_rank": int(os.environ["LOCAL_RANK"]),
    "visible_chips": os.environ.get("TPU_VISIBLE_CHIPS"), "id": int(d.id), "coords": [int(c) for c in d.coords],
    "global_device_id": gid, "x": float(x.cpu()[0])}), flush=True)
dist.barrier()
dist.destroy_process_group()
'''


def tpu_env(nh, host_rank, hosts):
    return {'TPU_SKIP_MDS_QUERY': 'true', 'TPU_ACCELERATOR_TYPE': f'v5litepod-{4 * nh}', 'TPU_TOPOLOGY': f'2x{2 * nh}',
            'TPU_TOPOLOGY_WRAP': 'false,false,false', 'TPU_HOST_BOUNDS': f'1,{nh},1',
            'TPU_CHIPS_PER_HOST_BOUNDS': '2,2,1', 'TPU_WORKER_ID': str(host_rank),
            'TPU_WORKER_HOSTNAMES': ','.join(hosts)}


def _env(base, extra):
    return {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': os.environ.get('HOME', '/root'),
            'LANG': 'C.UTF-8', **base, **extra}


def free_ports(n):
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(('127.0.0.1', 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def jax_all():
    t0 = time.time()
    p = subprocess.run([sys.executable, '-c', JAX_ALL], env=_env(tpu_env(1, 0, ['localhost']), {'JAX_PLATFORMS': 'tpu'}),
                       capture_output=True, text=True, timeout=180)
    for line in p.stdout.splitlines():
        if line.startswith('CHIPMAP_JAX='):
            return {'devices': json.loads(line.split('=', 1)[1]), 'seconds': round(time.time() - t0, 1)}
    return {'devices': None, 'error': (p.stderr or p.stdout)[-1500:]}


def torch_round(hosts, host_rank, visible_shift=None, ports=None):
    """This host's 4 ranks of a torch_tpu world over `hosts` (LOCAL_RANK=k, + TPU_VISIBLE_CHIPS=(k+shift)%4)."""
    nh = len(hosts)
    if ports is None:  # single host: any free ports
        p = free_ports(6)
        master, init, sb = p[0], p[1], p[2:6]
    else:  # multi-host: fixed ports every host agrees on
        master, init, sb = ports, ports + 1, [ports + 10 + i for i in range(4)]
    head = '127.0.0.1' if nh == 1 else hosts[0]
    sb_hosts = ['localhost'] if nh == 1 else hosts
    common = {'WORLD_SIZE': str(4 * nh), 'LOCAL_WORLD_SIZE': '4', 'NNODES': str(nh), 'NODE_RANK': str(host_rank),
              'TPU_NUM_HOSTS': str(nh), 'MASTER_ADDR': head, 'MASTER_PORT': str(master),
              'TORCH_TPU_SLICEBUILDER_ADDRESSES': ','.join(f'{h}:{q}' for h in sb_hosts for q in sb),
              'TORCH_TPU_TOPOLOGY': f'2,{2 * nh},1', 'TORCH_TPU_XPROF_SESSION_ID': '1'}
    base = tpu_env(nh, host_rank, hosts if nh > 1 else ['localhost'])
    procs = []
    t0 = time.time()
    for k in range(4):
        e = dict(common, RANK=str(4 * host_rank + k), LOCAL_RANK=str(k))
        if visible_shift is not None:
            e['TPU_VISIBLE_CHIPS'] = e['TPU_VISIBLE_DEVICES'] = str((k + visible_shift) % 4)
        procs.append(subprocess.Popen([sys.executable, '-c', TORCH_ONE, f'tcp://{head}:{init}'],
                                      env=_env(base, e), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    ranks, errs = [], []
    for k, p in enumerate(procs):
        try:
            out, err = p.communicate(timeout=420)
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            out, err = p.communicate()
        got = [json.loads(line.split('=', 1)[1]) for line in out.splitlines() if line.startswith('CHIPMAP_RANK=')]
        ranks += got
        if not got:
            errs.append(f'local rank {k} rc={p.returncode}: {err[-800:]}')
    return {'hosts': hosts, 'host_rank': host_rank, 'ranks': sorted(ranks, key=lambda r: r['rank']), 'errors': errs,
            'seconds': round(time.time() - t0, 1)}


def vfio_pci():
    out = {}
    for g in sorted(glob.glob('/dev/vfio/[0-9]*')):
        n = os.path.basename(g)
        d = f'/sys/kernel/iommu_groups/{n}/devices'
        out[n] = sorted(os.listdir(d)) if os.path.isdir(d) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--hosts', default='')
    ap.add_argument('--host-rank', type=int, default=0)
    ap.add_argument('--port', type=int, default=18480)
    ap.add_argument('--local-rounds', action='store_true', help='single-host rounds too (see the module docstring)')
    a = ap.parse_args()
    hosts = [h for h in a.hosts.split(',') if h]
    res = {'host': socket.gethostname(), 'python': sys.executable, 'vfio_pci': vfio_pci(), 'probe_t': time.time(),
           'ok': False}
    try:
        import torch_tpu  # noqa: F401
    except Exception as e:  # noqa: BLE001
        res['error'] = f'no torch_tpu in {sys.executable}: {e!r}'
        hosts, a.local_rounds = [], False
    if a.local_rounds:
        res['jax_all4'] = jax_all()
        res['round_local_rank'] = torch_round(['localhost'], 0)
        res['round_visible'] = torch_round(['localhost'], 0, visible_shift=1)
        loc = {r['local_rank']: r['coords'] for r in res['round_local_rank']['ranks']}
        b = res['round_visible']['ranks']
        if len(loc) == 4 and len(b) == 4:  # which variable picked the chip when they disagree
            res['chip_selected_by'] = ('LOCAL_RANK' if all(r['coords'] == loc[r['local_rank']] for r in b) else
                                       'TPU_VISIBLE_CHIPS' if all(r['coords'] == loc[(r['local_rank'] + 1) % 4]
                                                                  for r in b) else 'neither')
    src = None
    if len(hosts) > 1:
        res['round_slice'] = torch_round(hosts, a.host_rank, ports=a.port)
        src = res['round_slice']
        res['context'] = {'hosts': hosts, 'host_rank': a.host_rank, 'topology': f'2x{2 * len(hosts)}'}
    elif 'round_local_rank' in res:
        src = res['round_local_rank']
        res['context'] = {'hosts': ['localhost'], 'host_rank': 0, 'topology': '2x2'}
    if src and len(src['ranks']) == 4:
        chips = [{'index': r['local_rank'], 'coords': r['coords'], 'device_id': r['id'],
                  'global_device_id': r['global_device_id']} for r in src['ranks']]
        res['chips'] = chips
        res['stock_device_id'] = [c['device_id'] for c in chips]
        canon = sorted(chips, key=lambda c: (c['coords'][2], c['coords'][1], c['coords'][0]))
        res['canonical'] = [c['index'] for c in canon]
        res['canonical_device_id'] = [c['device_id'] for c in canon]
        res['ok'] = sorted(c['index'] for c in chips) == [0, 1, 2, 3] and len({tuple(c['coords']) for c in chips}) == 4
    json.dump(res, open(out_path := a.out, 'w'), indent=1)
    print('CHIPMAP_JSON=' + json.dumps({k: res.get(k) for k in (
        'host', 'context', 'chips', 'stock_device_id', 'canonical', 'canonical_device_id', 'chip_selected_by',
        'vfio_pci', 'ok', 'error')}), flush=True)
    for k in ('round_local_rank', 'round_visible', 'round_slice'):
        if k in res:
            print(f'CHIPMAP_{k.upper()}=' + json.dumps(res[k])[:2500], flush=True)
    print(f'CHIPMAP_WROTE {out_path}', flush=True)
    sys.exit(0 if res['ok'] else 1)


if __name__ == '__main__':
    main()
