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
parser.add_argument('--phase', choices=['inspect', 'setup', 'serve', 'cleanup', 'fusebench',
                                       'pmupload', 'pmdelete', 'h2dbench',
                                       'rewind-install', 'rewind-advance', 'rewind-revise', 'rewind-refresh',
                                       'rewind-qualify', 'rewind-prepare', 'rewind-copy',
                                       'rewind-run', 'rewind-collect', 'rewind-cleanup', 'rewind-retire',
                                       'rewind-metadata'],
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
parser.add_argument('--rewind-bundle', type=pathlib.Path, help='pinned native tar.gz; required for rewind-install/rewind-revise/rewind-refresh')
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

if args.phase.startswith('rewind-'):
    if args.hosts != '192.168.102.112,192.168.102.114':
        parser.error('rewind phases require the explicit ordered small_c pair .112,.114')
    if args.phase in ('rewind-install', 'rewind-revise', 'rewind-refresh') and not args.remote:
        if not args.rewind_bundle or not args.rewind_bundle.is_file():
            parser.error(f'{args.phase} requires an existing --rewind-bundle')
    if args.phase == 'rewind-metadata' and not args.remote:
        if not args.out_dir.absolute().is_relative_to('/dev/shm'):
            parser.error('CPU-only model metadata must be collected into local /dev/shm, never the repository')


def remote_main():
    import time
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    ray.init(address='auto', logging_level='WARNING')
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
        if phase == 'inspect':
            import pathlib, shutil
            users = []
            for proc in pathlib.Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    devices = [os.readlink(fd) for fd in (proc / 'fd').iterdir()]
                    devices = sorted({p for p in devices if p.startswith(('/dev/accel', '/dev/vfio'))})
                    if devices:
                        users.append({'pid': int(proc.name), 'exe': os.readlink(proc / 'exe'),
                                      'devices': devices})
                except OSError:
                    continue
            python = subprocess.run(
                ['python3.12', '-c', 'import ctypes,sys; print(sys.version)'],
                text=True, capture_output=True, timeout=20)
            private_paths, private_runtime = {}, None, None
            if 'TTL_BASE' in host_env:
                for key, value in {'base': host_env['TTL_BASE'],
                                   'base_logs': str(pathlib.Path(host_env['TTL_BASE']) / 'logs'),
                                   'pm_root': host_env.get('TTL_PM_ROOT', '/dev/shm/pm_rewind_smallc'),
                                   'model_source': host_env.get('TTL_SRC', '/models/gemma-4-31b')}.items():
                    path = pathlib.Path(value)
                    if path.exists() or path.is_symlink():
                        info = path.lstat()
                        private_paths[key] = {'path': value, 'mode': oct(info.st_mode & 0o777),
                                              'uid': info.st_uid, 'symlink': path.is_symlink(),
                                              'entries': sorted(p.name for p in path.iterdir()) if path.is_dir() else []}
                        if key == 'base_logs' and path.is_dir() and not path.is_symlink():
                            private_paths[key]['directories'] = {
                                p.name: sorted(f.name for f in p.iterdir())
                                for p in path.iterdir() if p.is_dir() and not p.is_symlink()}
                    else:
                        private_paths[key] = {'path': value, 'exists': False}
                exe = pathlib.Path(host_env['TTL_BASE']) / 'venv/bin/python'
                if exe.is_file():
                    code = ('import ctypes,importlib.metadata as m,json,os,signal,sys;'
                            'lib=ctypes.CDLL(None);lib.gnu_get_libc_version.restype=ctypes.c_char_p;'
                            'print(json.dumps({"python":sys.version,"libc":lib.gnu_get_libc_version().decode(),'
                            '"pidfd_open":hasattr(os,"pidfd_open"),'
                            '"pidfd_send_signal":hasattr(signal,"pidfd_send_signal"),'
                            '"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))')
                    observed = subprocess.run([str(exe), '-c', code], text=True, capture_output=True,
                                              timeout=30, env=dict(env, PYTHONDONTWRITEBYTECODE='1'))
                    private_runtime = {'rc': observed.returncode, 'stdout': observed.stdout,
                                       'stderr': observed.stderr}
            return {'host': socket.gethostname(), 'rank': host_env.get('TTL_HOST_RANK', '0'),
                    'rc': python.returncode, 'python': python.stdout, 'python_error': python.stderr,
                    'uname': list(os.uname()), 'uid': os.getuid(), 'device_users': users,
                    'private_paths': private_paths, 'private_runtime': private_runtime,
                    'disks': {p: shutil.disk_usage(p)._asdict() for p in ('/', '/dev/shm')},
                    'meminfo': pathlib.Path('/proc/meminfo').read_text(),
                    'pod_geometry': {k: os.environ.get(k) for k in (
                        'TPU_WORKER_ID', 'TPU_WORKER_HOSTNAMES', 'TPU_HOST_BOUNDS',
                        'TPU_CHIPS_PER_HOST_BOUNDS', 'TPU_TOPOLOGY')}}
        if phase == 'rewind-metadata':
            # Read only the already sealed runtime/model. Never upload/replace a runtime
            # or copy a weight shard while collecting local CPU preparation inputs.
            code = """
import base64, hashlib, importlib.util, io, json, os, re, stat, tarfile
from pathlib import Path
base = Path(os.environ['TTL_BASE'])
s = base.lstat()
assert base == base.resolve() and stat.S_ISDIR(s.st_mode) and s.st_uid == os.getuid() and not s.st_mode & 0o077
seal_path = base / 'rewind-install.json'
helper = base / 'rewind-src/pm_rewind_tpu.py'
transition_path = base / 'rewind-advance.json'
revision_path = base / 'rewind-revise.json'
def verified():
    # Keep the pre-import and under-lock checks independent of helper vintage.
    unbound = object()
    def receipt(path, expected=unbound):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f'unsafe/missing rewind transition file: {path}')
        data = path.read_bytes()
        if expected is not unbound and hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f'rewind transition receipt changed: {path}')
        return json.loads(data)
    owner_path = base / 'rewind-owner.json'
    seal, owner = receipt(seal_path), receipt(owner_path)
    if helper.is_symlink() or not helper.is_file():
        raise RuntimeError('unsafe installed rewind helper')
    raw = helper.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if not (sha == seal['files'][str(helper)] == seal['source_hashes']['pm_rewind_tpu.py']
            == owner['source_hashes']['pm_rewind_tpu.py']):
        raise RuntimeError('installed helper seal/owner mismatch')
    refresh_path = base / 'rewind-refresh.json'
    historical_seal_path, historical_owner_path = seal_path, owner_path
    historical_seal, historical_owner = seal, owner
    refresh = None
    withdrawals = set(base.glob('rewind-refresh-parent-install-*.json'))
    archives = set(base.glob('rewind-refresh-archive-*'))
    pm = Path(seal['config']['pm_root'])
    if not (str(pm) == owner['config']['pm_root']
            == os.environ.get('TTL_PM_ROOT', '/dev/shm/pm_rewind_smallc')
            and pm.parent == Path('/dev/shm') and pm.resolve() == pm):
        raise RuntimeError('unsafe or mismatched sealed PM root')
    qualifications = set(pm.glob('rewind-refresh-qualifications-*'))
    images = set(pm.glob('rewind-refresh-images-*'))
    if any(helper.parent.glob('*.refresh-pending')):
        raise RuntimeError('interrupted refresh source publication; no automatic retry')
    if os.path.lexists(refresh_path):
        refresh = receipt(refresh_path)
        if not (refresh.get('phase') == 'rewind-refresh' and refresh.get('success') is True
                and refresh.get('state') == 'complete' and refresh.get('leases_must_stay_held') is False):
            raise RuntimeError('source refresh incomplete; no automatic retry')
        lineage = refresh['refresh']
        run = lineage['run_id']
        if not isinstance(run, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', run):
            raise RuntimeError('unsafe source refresh run ID')
        archive = base / ('rewind-refresh-archive-' + run)
        withdrawn = base / ('rewind-refresh-parent-install-' + run + '.json')
        if not (refresh['run_id'] == run and refresh['archive'] == lineage['archive'] == str(archive)
                and refresh['withdrawn_install_receipt'] == str(withdrawn)
                and archives == {archive} and withdrawals == {withdrawn}
                and archive.is_dir() and not archive.is_symlink() and archive.resolve() == archive
                and lineage == seal.get('source_refresh') == owner.get('source_refresh')):
            raise RuntimeError('source refresh lost its seal/owner or permanent withdrawal/archive evidence')
        retired_qualifications = pm / ('rewind-refresh-qualifications-' + run)
        retired_images = pm / ('rewind-refresh-images-' + run)
        if not (qualifications == {retired_qualifications} and images == {retired_images}
                and lineage['retired_qualifications'] == str(retired_qualifications)
                and lineage['retired_images']['archive'] == str(retired_images)
                and lineage['retired_images']['original'] == seal['config']['cache']
                and all(root.is_dir() and not root.is_symlink() and root.resolve() == root
                        for root in (retired_qualifications, retired_images))):
            raise RuntimeError('source refresh lost its permanent PM archives')
        receipt(pm / 'model-receipt.json', refresh['new_model_receipt_sha256'])
        if (receipt(seal_path, refresh['new_install_receipt_sha256']) != seal
                or receipt(owner_path, refresh['new_owner_sha256']) != owner
                or seal['source_hashes'] != owner['source_hashes']
                or seal['source_hashes'] != refresh['new_source_hashes']):
            raise RuntimeError('source refresh does not bind the active seal/owner')
        historical_seal_path = archive / 'rewind-install.json'
        historical_owner_path = archive / 'rewind-owner.json'
        historical_seal = receipt(historical_seal_path, lineage['parent_install_receipt_sha256'])
        historical_owner = receipt(historical_owner_path, lineage['parent_owner_sha256'])
        receipt(withdrawn, lineage['parent_install_receipt_sha256'])
        receipt(archive / 'model-receipt.json', lineage['parent_model_receipt_sha256'])
        for name, key in (('rewind-revise.json', 'revision_marker_sha256'),
                          ('rewind-advance.json', 'advance_marker_sha256')):
            receipt(base / name, lineage[key])
            receipt(archive / name, lineage[key])
        if not (historical_seal['source_hashes'] == historical_owner['source_hashes']
                == lineage['parent_source_hashes']
                and 'source_refresh' not in historical_seal and 'source_refresh' not in historical_owner
                and seal.get('source_revision') == historical_seal.get('source_revision')
                and owner.get('source_revision') == historical_owner.get('source_revision')):
            raise RuntimeError('source refresh changed historical source authority')
    elif (withdrawals or archives or qualifications or images
          or 'source_refresh' in seal or 'source_refresh' in owner
          or (os.path.lexists(pm / 'model-receipt.json')
              and 'source_refresh' in receipt(pm / 'model-receipt.json'))):
        raise RuntimeError('source refresh marker lost; no automatic retry')
    history = None
    if os.path.lexists(transition_path):
        history = receipt(transition_path)
        if not (history.get('success') is True and history.get('state') == 'complete'):
            raise RuntimeError('incomplete historical advance')
    elif (historical_seal.get('lineage') or historical_owner.get('lineage')
          or any(base.glob('rewind-advance-parent-install-*.json'))
          or any(base.glob('rewind-advance-archive-*'))):
        raise RuntimeError('historical advance marker lost; no automatic retry')
    active = history
    if os.path.lexists(revision_path):
        revision = receipt(revision_path)
        if not (history and revision.get('phase') == 'rewind-revise' and revision.get('success') is True
                and revision.get('state') == 'complete'
                and revision['revision'] == historical_seal.get('source_revision')
                == historical_owner.get('source_revision')):
            raise RuntimeError('incomplete or unverified historical source revision')
        receipt(transition_path, revision['revision']['advance_marker_sha256'])
        run = revision['run_id']
        if not isinstance(run, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', run):
            raise RuntimeError('unsafe source revision run ID')
        receipt(base / ('rewind-revision-parent-install-' + run + '.json'),
                revision['revision']['parent_install_receipt_sha256'])
        if not (receipt(historical_seal_path, revision['new_install_receipt_sha256']) == historical_seal
                and receipt(historical_owner_path, revision['new_owner_sha256']) == historical_owner
                and historical_seal['source_hashes'] == historical_owner['source_hashes']
                == revision['new_source_hashes']):
            raise RuntimeError('historical source revision seal/owner changed')
        active = revision
    elif (refresh or historical_seal.get('source_revision') or historical_owner.get('source_revision')
          or any(base.glob('rewind-revision-parent-install-*.json'))
          or any(base.glob('rewind-revision-archive-*'))):
        raise RuntimeError('source revision marker lost; no automatic retry')
    active = refresh or active
    if active and sha != active['new_source_hashes']['pm_rewind_tpu.py']:
        raise RuntimeError('unverified active rewind helper')
    return raw
raw = verified()
module = importlib.util.module_from_spec(importlib.util.spec_from_file_location('_sealed_cpu_metadata', helper))
exec(compile(raw, str(helper), 'exec'), module.__dict__)
c = module.Config()
lock = module.phase_lock(c)
try:
    if verified() != raw:
        raise RuntimeError('installed helper changed while acquiring phase lock')
    c.check_install()
    module.private(c.ckpt)
    if os.path.lexists(transition_path) or hasattr(module, 'require_model_receipt'):
        model = module.require_model_receipt(c)
    else:
        # Before the one-time transition the sealed legacy helper has no lineage API.
        model = module.load(c.pm / 'model-receipt.json')
        assert model['success'] and model['config_hash'] == c.config_hash and model['install_id'] == c.install_id
    names = sorted(name for name in model['files'] if Path(name).suffix in ('.json', '.model', '.txt', '.jinja'))
    assert 'config.json' in names and any(name.startswith('tokenizer') for name in names)
    files, blobs, total = {}, {}, 0
    for name in names:
        assert Path(name).name == name
        path = c.ckpt / name
        assert path.is_file() and not path.is_symlink()
        total += path.stat().st_size
        assert total <= 64 * 1024 * 1024
        data = path.read_bytes()
        actual = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        assert actual == model['files'][name]
        files[name] = actual
        if c.rank == 0:
            blobs['model/' + name] = data
    refresh_files, refresh_bytes = {}, 0
    refresh_path = base / 'rewind-refresh.json'
    if os.path.lexists(refresh_path):
        def small_refresh_receipt(path, expected=None):
            if (path.is_symlink() or not path.is_file() or path.resolve() != path
                    or not path.is_relative_to(base)):
                raise RuntimeError(f'unsafe metadata refresh receipt: {path}')
            with path.open('rb') as source:
                data = source.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise RuntimeError(f'metadata refresh receipt exceeds 4 MiB: {path}')
            sha = hashlib.sha256(data).hexdigest()
            if expected is not None and sha != expected:
                raise RuntimeError(f'metadata refresh receipt changed: {path}')
            return data, sha
        marker_data, marker_sha = small_refresh_receipt(refresh_path)
        refresh = json.loads(marker_data)
        lineage = refresh['refresh']
        refresh_run = lineage['run_id']
        if not isinstance(refresh_run, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', refresh_run):
            raise RuntimeError('unsafe metadata refresh run ID')
        refresh_archive = base / ('rewind-refresh-archive-' + refresh_run)
        receipts = [(refresh_path, marker_sha),
                    (base / ('rewind-refresh-parent-install-' + refresh_run + '.json'),
                     lineage['parent_install_receipt_sha256'])]
        receipts += [(refresh_archive / name, lineage[key]) for name, key in (
            ('rewind-install.json', 'parent_install_receipt_sha256'),
            ('rewind-owner.json', 'parent_owner_sha256'),
            ('model-receipt.json', 'parent_model_receipt_sha256'),
            ('rewind-revise.json', 'revision_marker_sha256'),
            ('rewind-advance.json', 'advance_marker_sha256'))]
        for path, expected in receipts:
            data, sha = small_refresh_receipt(path, expected)
            refresh_bytes += len(data)
            if refresh_bytes > 16 * 1024 * 1024 or total + refresh_bytes > 64 * 1024 * 1024:
                raise RuntimeError('metadata/refresh evidence exceeds its receipt budget')
            member = 'refresh-receipts/' + str(path.relative_to(base))
            refresh_files[member] = {'bytes': len(data), 'sha256': sha}
            blobs[member] = data
    report = {'success': True, 'phase': 'rewind-metadata', 'model_id': model['model_id'],
              'install_id': c.install_id, 'config_hash': c.config_hash, 'files': files,
              'bytes': total, 'weights_included': False, 'runtime_modified': False,
              'refresh_receipts': refresh_files, 'refresh_receipt_bytes': refresh_bytes}
    blobs['model-metadata-receipt.json'] = module.encoded(report)
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w:gz') as archive:
        for name, data in blobs.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o600
            archive.addfile(info, io.BytesIO(data))
    print(json.dumps({'measurement': report,
                      'evidence_tar_gz_b64': base64.b64encode(payload.getvalue()).decode('ascii')}))
finally:
    os.close(lock)
"""
            observed = subprocess.run([os.path.join(env['TTL_BASE'], 'venv/bin/python'), '-c', code],
                                      text=True, capture_output=True, timeout=180,
                                      env=dict(env, PYTHONDONTWRITEBYTECODE='1'))
            result = {'host': socket.gethostname(), 'rank': host_env['TTL_HOST_RANK'],
                      'rc': observed.returncode, 'seconds': round(time.time() - t0, 3)}
            if observed.returncode:
                result['error'] = observed.stderr
            else:
                result.update(json.loads(observed.stdout))
            return result
        if phase.startswith('rewind-'):
            import base64, io, pathlib, tarfile
            proc = subprocess.run(['bash', os.path.join(script_dir, 'torchtpu_load_host.sh'), phase],
                                  env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print(proc.stdout, end='', flush=True)
            records = [line.removeprefix('REWIND_RESULT_JSON=') for line in proc.stdout.splitlines()
                       if line.startswith('REWIND_RESULT_JSON=')]
            result = {'host': socket.gethostname(), 'rank': host_env['TTL_HOST_RANK'],
                      'rc': proc.returncode, 'seconds': round(time.time() - t0, 3)}
            try:
                if len(records) == 1:
                    result['measurement'] = json.loads(records[0])
                    if not result['measurement'].get('success'):
                        result['rc'] = result['rc'] or 1
                else:
                    result['rc'] = result['rc'] or 1
                    result['protocol_error'] = f'expected one helper result, received {len(records)}'
                base = pathlib.Path(env.get('TTL_BASE', '/root/pm_rewind_smallc'))
                log = base / 'logs' / run_id
                # Refresh exports receipts only, never even walks its retired logs,
                # images, qualification caches, or retained-runtime inventories.
                files = list(log.rglob('*')) if phase != 'rewind-refresh' and log.is_dir() else []
                files += [base / name for name in ('rewind-owner.json', 'rewind-install.json',
                                                   'rewind-install.log', 'rewind-advance.json',
                                                   'rewind-revise.json')
                          if (base / name).is_file()
                          and not (phase == 'rewind-refresh' and name == 'rewind-install.log')]
                import re
                refresh_blobs = {}
                def small_refresh_receipt(path):
                    if not os.path.lexists(path):
                        return None
                    if (path.is_symlink() or not path.is_file() or path.resolve() != path
                            or not path.is_relative_to(base)):
                        raise RuntimeError(f'unsafe refresh evidence receipt: {path}')
                    member = str(path.relative_to(base))
                    if member in refresh_blobs:
                        return refresh_blobs[member]
                    if len(refresh_blobs) >= 16:
                        raise RuntimeError('too many refresh evidence receipts')
                    with path.open('rb') as source:
                        data = source.read(4 * 1024 * 1024 + 1)
                    if len(data) > 4 * 1024 * 1024:
                        raise RuntimeError(f'refresh evidence receipt exceeds 4 MiB: {path}')
                    if sum(map(len, refresh_blobs.values())) + len(data) > 16 * 1024 * 1024:
                        raise RuntimeError('refresh evidence receipts exceed 16 MiB')
                    refresh_blobs[member] = data
                    return data
                # Collect only the revision marker and exact permanent withdrawal
                # receipt, not its potentially large retired archive or images.
                revision_path = base / 'rewind-revise.json'
                if revision_path.is_file() and not revision_path.is_symlink():
                    revision = json.loads(small_refresh_receipt(revision_path) if phase == 'rewind-refresh'
                                          else revision_path.read_text())
                    revision_run = revision.get('run_id', '')
                    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', revision_run):
                        raise RuntimeError('unsafe revision evidence run ID')
                    revision_archive = base / ('rewind-revision-archive-' + revision_run)
                    if revision.get('archive') != str(revision_archive):
                        raise RuntimeError('unknown revision evidence archive')
                    files.append(base / ('rewind-revision-parent-install-' + revision_run + '.json'))
                else:
                    # Withdrawal-before-marker interruption still has useful evidence.
                    files += list(base.glob('rewind-revision-parent-install-*.json'))
                refresh_runs = set()
                def refresh_run(value):
                    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', value):
                        raise RuntimeError('unsafe refresh evidence run ID')
                    refresh_runs.add(value)
                    if len(refresh_runs) > 1:
                        raise RuntimeError('multiple refresh evidence generations')
                marker_data = small_refresh_receipt(base / 'rewind-refresh.json')
                if marker_data is not None:
                    refresh = json.loads(marker_data)
                    refresh_run(refresh.get('run_id'))
                    if refresh.get('archive') != str(base / ('rewind-refresh-archive-' + refresh['run_id'])):
                        raise RuntimeError('unknown refresh evidence archive')
                # A withdrawal or archive can survive loss of the marker. Discover
                # only its name, then read an exact, small allowlist of receipts.
                for prefix, suffix in (('rewind-refresh-parent-install-', '.json'),
                                       ('rewind-refresh-archive-', '')):
                    for path in base.glob(prefix + '*' + suffix):
                        refresh_run(path.name[len(prefix):-len(suffix)] if suffix else path.name[len(prefix):])
                for refresh_id in refresh_runs:
                    small_refresh_receipt(base / ('rewind-refresh-parent-install-' + refresh_id + '.json'))
                    refresh_archive = base / ('rewind-refresh-archive-' + refresh_id)
                    if os.path.lexists(refresh_archive):
                        if (refresh_archive.is_symlink() or not refresh_archive.is_dir()
                                or refresh_archive.resolve() != refresh_archive):
                            raise RuntimeError('unsafe refresh evidence archive')
                        for name in ('rewind-install.json', 'rewind-owner.json', 'model-receipt.json',
                                     'rewind-revise.json', 'rewind-advance.json'):
                            small_refresh_receipt(refresh_archive / name)
                if phase == 'rewind-refresh':
                    for path in files:
                        small_refresh_receipt(path)
                    files = []
                qualification = None
                if phase == 'rewind-qualify' and result.get('measurement', {}).get('phase') == 'rewind-qualify':
                    owner = json.loads((base / 'rewind-owner.json').read_text())
                    qowner = json.loads((log / 'rewind-qualification-owner.json').read_text())
                    pm = pathlib.Path(owner['config']['pm_root'])
                    qualification = pm / 'qualifications' / run_id
                    if (str(pm) != env.get('TTL_PM_ROOT') or pm.parent != pathlib.Path('/dev/shm')
                            or qowner.get('output') != str(qualification)
                            or qowner.get('run_id') != run_id
                            or qowner.get('config_hash') != result['measurement'].get('config_hash')
                            or qualification.resolve() != qualification
                            or qualification.stat().st_uid != os.getuid()):
                        raise RuntimeError('unknown qualification evidence ownership')
                    files += list((qualification / 'cache' / 'build').glob('*.log'))
                    files += list(qualification.glob('run*/*.log'))
                payload = io.BytesIO()
                size = sum(map(len, refresh_blobs.values()))
                with tarfile.open(fileobj=payload, mode='w:gz') as archive:
                    for member, data in sorted(refresh_blobs.items()):
                        info = tarfile.TarInfo(member)
                        info.size, info.mode = len(data), 0o600
                        archive.addfile(info, io.BytesIO(data))
                    for path in sorted(set(files)):
                        if not path.is_file():
                            continue
                        if path.is_relative_to(base):
                            boundary, member = base, path.relative_to(base)
                        elif qualification is not None and path.is_relative_to(qualification):
                            boundary = qualification
                            member = pathlib.Path('qualification-native') / path.relative_to(qualification)
                        else:
                            raise RuntimeError(f'unknown evidence root: {path}')
                        if path.is_symlink() or not path.resolve().is_relative_to(boundary.resolve()):
                            raise RuntimeError(f'log path escaped owned root: {path}')
                        size += path.stat().st_size
                        if size > 64 * 1024 * 1024:
                            raise RuntimeError('owned evidence exceeds 64 MiB; collect it explicitly')
                        archive.add(path, arcname=str(member), recursive=False)
                result['evidence_tar_gz_b64'] = base64.b64encode(payload.getvalue()).decode('ascii')
            except Exception as exc:
                result['rc'] = result['rc'] or 1
                result['collection_error'] = repr(exc)
            return result
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
            print('\nTOPOLOGY_RESULT_JSON=' + json.dumps(r) + '\n', end='', flush=True)
        sys.exit(0 if all(r.get('rc') == 0 for r in results) else 1)
    results = [ray.get(refs[0])]  # serve: host rank 0 runs the engine; the other hosts follow it
    print('\nTOPOLOGY_RESULT_JSON=' + json.dumps(results[0]) + '\n', end='', flush=True)
    deadline = time.time() + args.peer_grace
    for ref in refs[1:]:
        ready, _ = ray.wait([ref], timeout=max(1.0, deadline - time.time()))
        if ready:
            results.append(ray.get(ref))
        else:
            ray.cancel(ref, force=True)
            results.append({'rc': 'cancelled', 'after_s': args.peer_grace})
        print('\nTOPOLOGY_RESULT_JSON=' + json.dumps(results[-1]) + '\n', end='', flush=True)
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
                 'torchtpu_ray_pin_plugin.py', 'pm_ray_zygote.py', 'pm_rewind.py', 'pm_rewind_native.py',
                 'pm_rewind_ray.py', 'pm_rewind_ray_probe.py', 'pm_rewind_probe.py', 'pm_rewind_tpu.py'):
        shutil.copy(HERE / name, work / name)
    if args.phase in ('rewind-install', 'rewind-revise', 'rewind-refresh'):
        shutil.copyfile(args.rewind_bundle, work / 'pm_rewind_native.tar.gz')
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
        import base64, hashlib
        hosts = []
        for line in text.splitlines():
            if line.startswith('TOPOLOGY_RESULT_JSON='):
                host = json.loads(line.removeprefix('TOPOLOGY_RESULT_JSON='))
                packed = host.pop('evidence_tar_gz_b64', None)
                if packed is not None:
                    rank = str(host['rank'])
                    if rank not in ('0', '1'):
                        raise RuntimeError(f'unexpected rewind host rank: {rank}')
                    data = base64.b64decode(packed, validate=True)
                    name = f'{args.phase}-host{rank}-evidence.tar.gz'
                    (out / name).write_bytes(data)
                    host['evidence'] = {'file': name, 'bytes': len(data),
                                        'sha256': hashlib.sha256(data).hexdigest()}
                hosts.append(host)
        (out / f'{args.phase}_hosts.json').write_text(json.dumps(hosts, indent=2) + '\n')
    if args.phase.startswith('rewind-'):
        if len(hosts) != 2 or sorted(str(h.get('rank')) for h in hosts) != ['0', '1']:
            raise RuntimeError('rewind result is missing one or both host reports')
    (out / f'{args.phase}_result.json').write_text(json.dumps(
        {'submission_id': sid, 'status': result['status'], 'measurements': result['measurements'],
         'hosts': hosts}, indent=2) + '\n')
    print('[OMP_EVENT: TORCHTPU_PHASE_DONE phase=' + args.phase + ']', flush=True)


if __name__ == '__main__':
    runner = StageRunner(silence_timeout_s=5400)
    runner.register(run_remote)
    if not runner.run():
        raise SystemExit(1)
