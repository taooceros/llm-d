#!/usr/bin/env python3
"""Owned TP8 experiment; CLI: rewind-{install,advance,revise,refresh,qualify,prepare,copy,run,collect,cleanup,retire}.

All public phases require TTL_RUN, TTL_RAY_HOSTS=192.168.102.112,192.168.102.114,
TTL_HOST_RANK=0|1 and TTL_RAY_ADDRESS. Run both hosts for rewind-run and rewind-cleanup.
Install after private-venv setup, before model copy/prepare. Never upload over the
installed rewind-src directory: use the explicit advance/revise/refresh transitions.
Cleanup/retirement recheck launch journals, owned process receipts and exact Ray
jobs; failed launches need not produce four workers. Unknown launch ownership is
never inferred safe from an empty proxy list.
TTL_REWIND_RESET_UNQUALIFIED=1 explicitly rolls back a failed pre-image CPU
qualification during install, preserving the private venv and archiving old proof.
TTL_REWIND_RESET_SETUP_RUN explicitly preserves one verified setup-only log directory.
Advance requires TTL_REWIND_EXPECT_OLD_{INSTALL_ID,CONFIG_HASH,MODEL_ID,SOURCE_SHA256};
SOURCE_SHA256 hashes the sorted compact JSON source_hashes map plus newline.
rewind-revise uses the same explicit old identities after a completed CPU advance,
before successful qualification/production preparation. It requires the exact helper
pair change and frozen CPU runtime SHA, preserving advance and retained runtime/model bytes.
rewind-refresh is the single reviewed successor to that exact revised install.
It additionally requires TTL_REWIND_EXPECT_OLD_NATIVE_BUNDLE_SHA256 and the original
bundle payload, archives qualification/prepare/failed serving without reuse, and
permanently withdraws the old seal before mutation. There is no retry/repair route.
TTL_REWIND_CACHE_MODE=strict (default) enforces the original seed executable bank.
The explicit functional mode measures restore/plain without cache-hit claims;
TPU init, terminal restore, native/CPU identity, requests and cleanup stay mandatory.
"""
import argparse
import ast
import configparser
import copy
import ctypes
import fcntl
import hashlib
import http.server
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HOSTS = ('192.168.102.112', '192.168.102.114')
SOURCES = ('pm_rewind_tpu.py', 'pm_rewind.py', 'pm_rewind_native.py', 'pm_rewind_ray.py',
           'pm_ray_zygote.py', 'pm_zygote.py', 'pm_source_patches.py',
           'pm_torchtpu_plugin.py', 'torchtpu_load_plugin.py', 'torchtpu_ray_pin_plugin.py',
           'torchtpu_req_probe.py', 'pm_rewind_ray_probe.py', 'torchtpu_ckpt_copy.py')
# This one-shot revision is bound to the reviewed early env_override CPU preload.
REVISION_RUNTIME_SHA256 = '4aeffa9dea2344d2c7a373a109f79a3830a491461fab96dda234b0b0fa490e84'
# Refresh is NOT a general source upgrade API: only this qualified revised parent.
REFRESH_RUNTIME_SHA256 = '8529998525c7d9c25e0981e72ceccf8a82f9a7d71ce32f19f51dd68d4d640691'
REFRESH_PARENT = {
    'INSTALL_ID': '8973dc3f12b3756ed7d1f2926e76b10c1f21bd61e6283f8aabd40281ef8e0125',
    'CONFIG_HASH': 'fd4e19123c6dc453ff05c6d8417b40aef76255fde286ec94f0e19a9cbe4261bd',
    'MODEL_ID': '7db39c880113f3cc9ddbe737ddb1513b32077ab56eda509256f5dba64c6f5ea6',
    'SOURCE_SHA256': '8354b1d5a2e83fd060c68f38de0509611530cb8a7ef5098393349db54ac2e7bd',
}
UNSET = 'ENABLE_PJRT_COMPATIBILITY,TPU_NAME,JAX_PLATFORMS'
LATE_WORKER = ('PM_RAY_ZYGOTE_SOCK', 'TTL_RUN', 'TTL_RAY_WORKER_ENV', 'PM_SNAP_MODE')
LATE_API = (*LATE_WORKER, 'RAY_ADDRESS', 'TTL_RAY_RUNTIME_ENV', 'TTL_RAY_JOBID_FILE',
            'TPU_WORKER_ID', 'VLLM_HOST_IP', 'PM_ZYGOTE_SOCK')
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CANCELLATION = {'deferred': False, 'pending': False}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def dump(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    with pending.open('wb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(encoded(value))
    pending.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(pid=None):
    raw = Path(f'/proc/{pid or os.getpid()}/stat').read_text()
    return {'pid': int(raw.split(' ', 1)[0]), 'start_ticks': int(raw.rpartition(')')[2].split()[19])}


def alive(who):
    try:
        current = identity(who['pid'])
        return who.get('start_ticks') is None or current['start_ticks'] == who['start_ticks']
    except FileNotFoundError:
        return False


def private(path, create=False):
    path = Path(os.path.abspath(path))
    require(str(path) == str(path.resolve()), f'symlink path refused: {path}')
    if create and not path.exists():
        for parent in path.parents:
            info = parent.lstat()
            require(stat.S_ISDIR(info.st_mode) and info.st_uid in (0, os.getuid())
                    and (not info.st_mode & 0o022 or (info.st_uid == 0 and info.st_mode & stat.S_ISVTX)),
                    f'untrusted parent: {parent}')
        path.mkdir(mode=0o700)
    for item in (path, *path.parents):
        s = item.lstat()
        require(stat.S_ISDIR(s.st_mode) and s.st_uid in (0, os.getuid()), f'untrusted directory: {item}')
        require(not s.st_mode & 0o022 or (s.st_uid == 0 and s.st_mode & stat.S_ISVTX),
                f'writable directory: {item}')
    require(path.stat().st_uid == os.getuid() and not path.stat().st_mode & 0o077,
            f'private 0700 directory required: {path}')
    return path


def stable_write(path, data, mode=0o600):
    path = Path(path)
    if path.exists() or path.is_symlink():
        s = path.lstat()
        require(stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid()
                and stat.S_IMODE(s.st_mode) == mode and path.read_bytes() == data,
                f'conflicting installation, use a new TTL_BASE: {path}')
        return
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(), mode)
        stream.write(data)


class Config:
    def __init__(self):
        e = os.environ
        require(e.get('TTL_RAY_HOSTS') == ','.join(HOSTS), 'ONLY ordered small_c hosts .112,.114 are allowed')
        require(e.get('TTL_HOST_RANK') in ('0', '1'), 'TTL_HOST_RANK must be 0 or 1')
        self.rank = int(e['TTL_HOST_RANK'])
        self.ip = HOSTS[self.rank]
        self.run = e.get('TTL_RUN', '')
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', self.run), 'new safe TTL_RUN required (1-48 chars)')
        self.base = Path(e.get('TTL_BASE', '/root/pm_rewind_smallc'))
        self.pm = Path(e.get('TTL_PM_ROOT', '/dev/shm/pm_rewind_smallc'))
        self.ckpt = Path(e.get('TTL_CKPT', str(self.pm / 'model')))
        self.cache = Path(e.get('TTL_REWIND_CACHE', str(self.pm / 'images')))
        for path in (self.base, self.pm, self.ckpt, self.cache):
            require(path.is_absolute() and '..' not in path.parts and str(path) == str(path.resolve()),
                    f'absolute non-symlink path required: {path}')
        require(self.base not in (Path('/root/torchtpu_load'), Path('/root'), Path('/tmp'), Path('/dev/shm')),
                'dedicated TTL_BASE required')
        require(self.pm.parent == Path('/dev/shm') and self.pm.name not in ('pm_torchtpu', 'torch_tpu_cache'),
                'TTL_PM_ROOT must be a dedicated direct child of /dev/shm')
        require(self.ckpt.is_relative_to(self.pm) and self.ckpt != self.pm, 'TTL_CKPT must be inside private TTL_PM_ROOT')
        require(self.cache.is_relative_to(self.pm) and self.cache != self.pm,
                'TTL_REWIND_CACHE must be inside private tmpfs PM_ROOT')
        self.address = e.get('TTL_RAY_ADDRESS', '')
        require(self.address and self.address not in ('auto', 'local'), 'explicit shared TTL_RAY_ADDRESS required')
        self.state_address = e.get('TTL_RAY_STATE_ADDRESS',
                                   'http://' + self.address.split('://')[-1].split(':')[0] + ':8265')
        self.arm = e.get('TTL_REWIND_ARM', 'restore')
        require(self.arm in ('seed', 'plain', 'restore'), 'TTL_REWIND_ARM must be seed|plain|restore')
        require(e.get('TTL_TP', '8') == '8', 'only TP8 is supported')
        self.port = int(e.get('TTL_PORT', '18080'))
        self.mark_port = int(e.get('TTL_MARK_PORT', '18470'))
        require(1024 <= self.port <= 65535 and 1024 <= self.mark_port <= 65535
                and self.port != self.mark_port, 'distinct nonprivileged service and marker ports required')
        self.timeout = float(e.get('TTL_READY_TIMEOUT', '2400'))
        self.src = self.base / 'rewind-src'
        self.python = self.base / 'venv/bin/python'
        self.native = self.base / 'native'
        self.log = self.base / 'logs' / self.run
        self.generation = self.base / 'runs' / self.run
        self.socket = self.generation / 'worker.sock'
        require(len(os.fsencode(self.socket)) < 104, 'BASE/run socket path too long')
        self.spec = self.configuration()
        self.config_hash = hashlib.sha256(encoded(self.spec)).hexdigest()

    def configuration(self):
        e = os.environ
        env = {
            'PATH': f'{self.base}/venv/bin:/usr/local/bin:/usr/bin:/bin', 'HOME': str(self.base),
            'LANG': 'C.UTF-8', 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'GCONV_PATH': '/usr/lib/x86_64-linux-gnu/gconv',
            'GLIBC_TUNABLES': 'glibc.pthread.rseq=0',
            'PYTHONPATH': str(self.src), 'LD_LIBRARY_PATH': '',
            'LD_PRELOAD': str(self.native / 'allocator/libtcmalloc_minimal.so.4'),
            'OMP_NUM_THREADS': e.get('TTL_OMP_NUM_THREADS', '4'),
            'MKL_NUM_THREADS': e.get('TTL_MKL_NUM_THREADS', '4'),
            'OPENBLAS_NUM_THREADS': e.get('TTL_OPENBLAS_NUM_THREADS', '1'),
            'TORCH_DEVICE_BACKEND_AUTOLOAD': '0',
            'HF_HOME': str(self.base / 'hf'), 'XDG_CACHE_HOME': str(self.base / 'xdg'),
            'VLLM_CACHE_ROOT': str(self.pm / 'bank'),
            'TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT': str(self.pm / 't3'),
            'TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT': str(self.pm / 't3'),
            'VLLM_PLUGINS': 'torchtpu,torchtpu_layers,ttl_probe,ttl_ray_pin,pm_torchtpu',
            'VLLM_USE_RAY_V2_EXECUTOR_BACKEND': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
            'RAY_LD_PRELOAD_ON_WORKERS': '0',
            'RAY_CLIENT_MODE': '0',
            'TTL_RAY_HOSTS': ','.join(HOSTS), 'TTL_RAY_WORKER_UNSET': UNSET,
            'TPU_SKIP_MDS_QUERY': '1', 'TPU_ACCELERATOR_TYPE': 'v5litepod-8',
            'TPU_TOPOLOGY': '4x2', 'TPU_TOPOLOGY_WRAP': 'false,false,false',
            'TPU_HOST_BOUNDS': '2,1,1', 'TPU_CHIPS_PER_HOST_BOUNDS': '2,2,1',
            'TPU_WORKER_HOSTNAMES': ','.join(HOSTS), 'TORCH_TPU_TOPOLOGY': '4,2,1',
            'TPU_MULTIHOST_BACKEND': 'ray', 'PM_RAY_ZYGOTE_STRICT': '1',
            'PM_REWIND_DEFER_TPU_INIT': '1',
            'PM_SNAP_DIR': str(self.pm / 'snap'), 'PM_SNAP_ASYNC': '1', 'PM_SNAP_THREADS': '4',
            'PM_SNAP_FOREACH': '1', 'PM_SNAP_FOREACH_MB': '100000',
            'PM_AOT_SLIM': '1', 'PM_AOT_PRELOAD': '1', 'PM_BANK_PRELOAD': '1',
            'PM_SAMPLER_BANK': '1', 'PM_OVERLAP_RENDERER': '1', 'PM_WARM': '0',
            'PM_SB_PRESEED': '1', 'TOKENIZERS_PARALLELISM': 'false',
            'PM_SKIP_MM_PROBE': '1', 'PM_FAST_REPLAY': '1',
        }
        for item in shlex.split(e.get('TTL_EXTRA_ENV', '')):
            key, sep, value = item.partition('=')
            require(sep and key in env and key.startswith('PM_')
                    and key not in ('PM_SNAP_DIR', 'PM_RAY_ZYGOTE_STRICT', 'PM_REWIND_DEFER_TPU_INIT'),
                    f'TTL_EXTRA_ENV may override only declared PM tuning flags: {key}')
            env[key] = value
        argv = ['serve', str(self.ckpt), '--served-model-name', 'gemma4',
                '--tensor-parallel-size', '8', '--distributed-executor-backend', 'ray',
                '--pipeline-parallel-size', '1', '--data-parallel-size', '1',
                '--max-model-len', e.get('TTL_MAXLEN', '8192'),
                '--max-num-seqs', e.get('TTL_MAXSEQS', '512'),
                '--max-num-batched-tokens', e.get('TTL_MAX_BATCHED_TOKENS', '2048'),
                '--kv-cache-memory-bytes', e.get('TTL_KV_BYTES', '7184619601'),
                '--limit-mm-per-prompt', '{"image":0,"video":0,"audio":0}',
                '--seed', '0', '--host', '0.0.0.0', '--port', str(self.port)]
        require(not e.get('TTL_VLLM_EXTRA'), 'use explicit TTL_MAXLEN/MAXSEQS/MAX_BATCHED_TOKENS/KV_BYTES, not TTL_VLLM_EXTRA')
        env['PM_REWIND_SERVE_ARGV'] = json.dumps(argv, separators=(',', ':'))
        return {'env': env, 'argv': argv, 'ray_address': self.address,
                'ray_state_address': self.state_address,
                'api_forkserver': e.get('TTL_ZYGOTE', '1') == '1',
                'base': str(self.base), 'pm_root': str(self.pm), 'cache': str(self.cache),
                'mark_port': self.mark_port}

    def check_install(self):
        private(self.base)
        lineage = advance_lineage(self)
        require(not lineage or self.arm != 'seed', 'advanced installs cannot reseed or rewrite old seed receipts')
        owner = load(self.base / 'rewind-owner.json')
        require(owner['uid'] == os.getuid() and owner['config'] == self.spec,
                'installed configuration drift; new BASE/PM_ROOT required')
        receipt = load(self.base / 'rewind-install.json')
        require(receipt['config_hash'] == self.config_hash and receipt['config'] == self.spec
                and receipt['source_hashes'] == owner['source_hashes']
                and receipt['native_bundle_sha256'] == owner['bundle_sha256'], 'install/owner config or source mismatch')
        self.install_id = hashlib.sha256(encoded({
            'sources': receipt['source_hashes'], 'native': receipt['native_bundle_sha256']})).hexdigest()
        for name, expected in receipt['files'].items():
            p = Path(name)
            require(p.is_file() and not p.is_symlink() and digest(p) == expected,
                    f'sealed source drift: {p}')
        return receipt


def checked(command, log, env=None, timeout=600):
    with Path(log).open('ab') as output:
        result = subprocess.run([str(x) for x in command], env=env, stdout=output,
                                stderr=subprocess.STDOUT, timeout=timeout)
    require(result.returncode == 0, f'command failed rc={result.returncode}; see {log}')


def reset_unqualified(c, payload, bundle_hash, source):
    """Locked rollback of a failed, unpublished qualification, never seal repair."""
    archive = c.log / 'reset-archive'
    result = {'success': False, 'cleanup_complete': False, 'state': 'validating',
              'run_id': c.run, 'host_rank': c.rank,
              'receipt_path': str(c.log / 'rewind-reset.json'),
              'archive': str(archive), 'deleted': [], 'preserved': [
                  str(c.base), str(c.base / 'venv'), str(phase_lock_path(c)),
                  str(c.base / 'hf'), str(c.base / 'xdg')]}
    c.reset_receipt = result
    journal_ready = False
    def root_identity(path):
        info = path.lstat()
        require(path.resolve() == path and stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(),
                f'reset root is not an owned real directory: {path}')
        return {'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid}
    def no_mounts(root):
        device = root.stat().st_dev
        require(not os.path.ismount(root), f'reset refuses mounted root: {root}')
        for directory, dirs, _ in os.walk(root, followlinks=False):
            for name in dirs:
                child = Path(directory) / name
                require(child.is_symlink() or
                        (child.stat().st_dev == device and not os.path.ismount(child)),
                        f'reset refuses nested mount: {child}')
    def gone(who, label):
        require(isinstance(who, dict) and type(who.get('pid')) is int and who['pid'] > 0
                and type(who.get('start_ticks')) is int and who['start_ticks'] > 0,
                f'missing real process identity: {label}')
        require(not alive(who), f'live reset-owned process: {label}: {who}')
    def archived_files(directory, measurement):
        previous = measurement.get('reset')
        root = directory / 'reset-archive'
        if previous is None:
            require(not os.path.lexists(root), f'unreceipted reset archive: {root}')
            return set()
        require(measurement.get('phase') == 'rewind-install' and measurement.get('success') is True
                and previous.get('success') is True and previous.get('cleanup_complete') is True
                and previous.get('state') == 'old_install_removed_venv_preserved'
                and not previous.get('leases_must_stay_held') and 'deleting' not in previous
                and previous.get('run_id') == directory.name and previous.get('host_rank') == c.rank
                and previous.get('new_config_hash') == measurement.get('config_hash')
                and previous.get('archive') == str(root)
                and previous.get('receipt_path') == str(directory / 'rewind-reset.json')
                and load(directory / 'rewind-reset.json') == previous,
                f'incomplete or mismatched prior reset: {directory}')
        require(previous.get('authorized_deletions')
                and set(previous.get('deleted', [])) == set(previous['authorized_deletions']) | {
                    str(c.base / name) for name in
                    ('rewind-install.log', 'rewind-install.json', 'rewind-owner.json')},
                f'incomplete prior reset deletion journal: {directory}')
        root_identity(root)
        listed = set()
        for entry in previous['archive_files']:
            path = Path(entry['archive'])
            require(path.is_absolute() and path.is_relative_to(root) and path != root
                    and path.resolve() == path and path.is_file() and not path.is_symlink()
                    and path not in listed and digest(path) == entry['sha256'],
                    f'unverified prior archive file: {path}')
            listed.add(path)
        actual = set()
        for path in root.rglob('*'):
            require(not path.is_symlink() and (path.is_dir() or path.is_file()),
                    f'unsafe prior archive entry: {path}')
            if path.is_file():
                actual.add(path)
        require(listed and actual == listed, f'unknown/missing prior archive files: {root}')
        return listed
    try:
        require(os.environ.get('TTL_REWIND_RESET_UNQUALIFIED') == '1', 'reset requires explicit opt-in')
        require(getattr(c, 'phase_lock_fd', None) is not None, 'reset requires the held phase lock')
        lock = os.fstat(c.phase_lock_fd)
        linked = phase_lock_path(c).lstat()
        require(lock.st_nlink == 1 and (lock.st_dev, lock.st_ino) == (linked.st_dev, linked.st_ino),
                'reset phase lock was replaced')
        require(not archive.exists() and not archive.is_symlink()
                and not os.path.lexists(c.log / 'rewind-reset.json')
                and not os.path.lexists(c.log / 'rewind_measurement.json'), 'reset needs a new run log directory')
        private(c.log, create=True)
        journal_ready = True
        sealed = load(c.base / 'rewind-install.json')
        old_owner = load(c.base / 'rewind-owner.json')
        helper = c.src / 'pm_rewind_tpu.py'
        require(helper.is_file() and not helper.is_symlink(), 'old sealed helper missing')
        helper_bytes = helper.read_bytes()
        helper_hash = hashlib.sha256(helper_bytes).hexdigest()
        require(sealed['files'].get(str(helper)) == helper_hash
                == old_owner['source_hashes'].get('pm_rewind_tpu.py'), 'old helper is not sealed')
        spec = importlib.util.spec_from_file_location('_pm_rewind_reset_old', helper)
        old_module = importlib.util.module_from_spec(spec)
        # Execute verified source bytes, never an unsealed/stale __pycache__ file.
        exec(compile(helper_bytes, str(helper), 'exec'), old_module.__dict__)
        old = old_module.Config()
        old_seal = old.check_install()
        expected = json.loads(encoded(c.spec))
        for key, value in (('GCONV_PATH', '/usr/lib/x86_64-linux-gnu/gconv'),
                           ('RAY_LD_PRELOAD_ON_WORKERS', '0')):
            if key not in old.spec['env'] and expected['env'].get(key) == value:
                expected['env'].pop(key)
        require(old.spec == expected,
                'reset allows only added frozen GCONV_PATH and RAY_LD_PRELOAD_ON_WORKERS=0')
        require(old_seal == sealed and old_owner['source_hashes'] == sealed['source_hashes']
                and old_owner['bundle_sha256'] == sealed['native_bundle_sha256'] == bundle_hash,
                'old source/native identity is inconsistent or native bundle changed')
        require(set(payload) == set(old_owner['source_hashes']), 'reset source inventory changed')
        allowed_changes = {'pm_rewind_tpu.py', 'pm_rewind.py', 'pm_rewind_ray_probe.py',
                           'pm_rewind_ray.py', 'pm_ray_zygote.py'}
        require(all(name in allowed_changes or hashlib.sha256(data).hexdigest() == old_owner['source_hashes'][name]
                    for name, data in payload.items()), 'reset would change retained runtime/plugin patch inputs')
        require(old_owner.get('retirement_roots') == [str(c.pm), str(c.base)]
                and old_owner.get('phase_lock') == str(phase_lock_path(c)),
                'old installation does not authorize these exact roots')
        for root in (c.base, c.pm):
            require(root_identity(root) == sealed['retirement_roots'].get(str(root)),
                    f'old owned root was replaced: {root}')
            no_mounts(root)
        venv_identity = root_identity(c.base / 'venv')
        result.update(old_config_hash=old.config_hash, new_config_hash=c.config_hash,
                      old_install_id=old.install_id, old_source_hashes=sealed['source_hashes'],
                      old_roots=sealed['retirement_roots'], venv_identity=venv_identity)
        require(not os.path.lexists(c.base / 'rewind-qualified.json'), 'qualified installation cannot be reset')
        require(not any((c.base / 'runs').iterdir()), 'service/run generation exists')
        empty = (c.cache, c.ckpt, c.pm / 'bank', c.pm / 'snap', c.pm / 't3')
        for directory in empty:
            private(directory)
            require(not any(directory.iterdir()), f'published image/model/terminal/cache state exists: {directory}')
        permitted = {c.pm, c.pm / 'qualifications'}
        for directory in empty:
            while directory != c.pm:
                permitted.add(directory)
                directory = directory.parent
        for directory, dirs, files in os.walk(c.pm, followlinks=False):
            here = Path(directory)
            if here == c.pm / 'qualifications':
                dirs[:] = []
                continue
            require(here in permitted and not files
                    and all(not (here / name).is_symlink() for name in dirs),
                    f'unknown generated PM state: {here}')
        logs = []
        qualifications = {}
        extras = {}
        unpublished = set()
        supervisors = set()
        setup_run = os.environ.get('TTL_REWIND_RESET_SETUP_RUN', '')
        require(not setup_run or (re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', setup_run)
                                  and setup_run != c.run), 'invalid preserved setup run ID')
        setup = None
        for directory in (c.base / 'logs').iterdir():
            if directory == c.log:
                continue
            require(directory.is_dir() and not directory.is_symlink(), f'unknown old log entry: {directory}')
            if directory.name == setup_run:
                root_identity(directory)  # Setup logs may be 0755 beneath the private BASE.
                require({path.name for path in directory.iterdir()} ==
                        {'host.log', 'install.log', 'pip_freeze.txt'}, 'unknown files in preserved setup directory')
                for path in directory.iterdir():
                    info = path.lstat()
                    require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                            and not info.st_mode & 0o022, f'unsafe setup log: {path}')
                setup = directory
                result['preserved'].append(str(directory))
                continue
            measurement = load(directory / 'rewind_measurement.json')
            require(measurement.get('config_hash') == old.config_hash
                    and measurement.get('phase') in ('rewind-install', 'rewind-qualify'),
                    f'old service/prep/copy/run or unknown log generation: {directory}')
            prior_files = archived_files(directory, measurement)
            owners = [path for path in directory.rglob('*owner.json') if path not in prior_files]
            qowner_path = directory / 'rewind-qualification-owner.json'
            require(all(path == qowner_path for path in owners)
                    and not any(path.name.startswith(('prepare-', 'api-help-', 'serve_', 'ray_job'))
                                for path in directory.iterdir()), f'non-qualification ownership: {directory}')
            if measurement['phase'] == 'rewind-qualify':
                record = load(qowner_path)
                qroot = c.pm / 'qualifications' / directory.name
                private(qroot)
                require(record.get('run_id') == directory.name and record.get('host_rank') == c.rank
                        and record.get('config_hash') == old.config_hash and record.get('output') == str(qroot),
                        'old qualification owner mismatch')
                gone(record.get('owner'), directory.name + '/owner')
                gone(record.get('probe'), directory.name + '/probe')
                qualified = load(directory / 'rewind-qualification-result.json')
                proof = load(qroot / 'report.json')
                require(qualified == measurement and qualified.get('success') is False
                        and qualified.get('cleanup_complete') is True
                        and qualified.get('shared_ray_used') is False
                        and not qualified.get('leases_must_stay_held')
                        and proof == load(directory / 'rewind-qualification.json')
                        and proof.get('success') is False and proof.get('cpu_only') is True
                        and not proof.get('generations') and not proof.get('capture')
                        and not proof.get('ray_driver_final') and not proof.get('cleanup_errors'),
                        f'qualification is not a closed unpublished failure: {directory.name}')
                cleanup = proof.get('probe_cleanup') or {}
                require(cleanup.get('cleanup') == 'complete' and not cleanup.get('errors')
                        and isinstance(cleanup.get('owned'), dict), 'probe cleanup is unproven')
                for who in cleanup['owned'].values():
                    require(who.get('reaped'), 'probe adopted child was not reaped')
                    gone(who, directory.name + '/adopted')
                commands = proof.get('commands', [])
                require(commands and all(command.get('name') in ('versions', 'dmtcp-version', 'prepare')
                                         and type(command.get('rc')) is int for command in commands),
                        'qualification progressed beyond the permitted capture attempt')
                for command in commands:
                    gone(command.get('identity'), directory.name + '/' + command['name'])
                receipts = proof.get('receipts', {})
                actual = list(qroot.rglob('supervisor.json'))
                require(actual and {encoded(load(path)) for path in actual} == {encoded(value) for value in receipts.values()}
                        and qualified.get('receipts') == receipts, 'unknown/missing qualification supervisors')
                supervisors.update(actual)
                for path in actual:
                    receipt = load(path)
                    retired_supervisor(receipt, str(path))
                    for who in [receipt['reaper'], *receipt['owned'].values()]:
                        gone(who, str(path))
                images = list(qroot.rglob('*.dmtcp*'))
                if images:
                    cache = qroot / 'cache'
                    private(cache)
                    require(len(images) == 1 and images[0].parent == cache
                            and images[0].suffix == '.dmtcp' and not images[0].is_symlink()
                            and images[0].is_file() and images[0].stat().st_uid == os.getuid()
                            and images[0].stat().st_size > 0
                            and stat.S_IMODE(images[0].stat().st_mode) == 0o400,
                            'only one unpublished private CPU capture may be removed')
                    require([(cmd['name'], cmd['rc']) for cmd in commands] ==
                            [('versions', 0), ('dmtcp-version', 0), ('prepare', 1)]
                            and commands[-1].get('cwd') == str(qroot)
                            and commands[-1].get('argv', [])[:6] == [
                                str(c.python), str(qroot / 'src/pm_rewind.py'), 'prepare',
                                str(cache), '--factory', 'pm_rewind_ray:prepare_workers']
                            and set(receipts) == {'cache/build'}
                            and actual == [cache / 'build/supervisor.json'],
                            'unpublished image lacks the failed prepare/sole capture receipt')
                    receipt = receipts['cache/build']
                    application = receipt['owned'].get(str(receipt.get('application_pid')), {})
                    require(receipt.get('result', {}).get('returncode') == 0
                            and application.get('returncode') == 0 and application.get('reaped') is True,
                            'unpublished capture did not exit successfully')
                    config = load(cache / 'config.json')
                    prepared_path = cache / 'prepared.json'
                    require(prepared_path.is_file() and not prepared_path.is_symlink(),
                            'missing unpublished capture preparation proof')
                    prepared = load(prepared_path)
                    require(config.get('cache') == str(cache)
                            and config.get('factory') == 'pm_rewind_ray:prepare_workers'
                            and config.get('real_pids') is True
                            and config.get('python') == str(c.python)
                            and config.get('bin_dir') == str(c.native / 'bin')
                            and prepared.get('native_threads')
                            and prepared.get('files', {}).get(str(qroot / 'src/pm_rewind.py'), {}).get('sha256')
                                == sealed['source_hashes']['pm_rewind.py'],
                            'unpublished capture is not the old sealed CPU preparation')
                    captures = [json.loads(line.split('PM_REWIND_RAY_CAPTURE ', 1)[1])
                                for line in (cache / 'build/backend.log').read_text().splitlines()
                                if line.startswith('PM_REWIND_RAY_CAPTURE ')]
                    require(len(captures) == 1, 'missing/duplicate CPU capture witness')
                    witness = captures[0]
                    zygote_path = c.src / 'pm_ray_zygote.py'
                    zygote_bytes = zygote_path.read_bytes()
                    require(hashlib.sha256(zygote_bytes).hexdigest() ==
                            sealed['files'].get(str(zygote_path)), 'old preload source is not sealed')
                    constants = {}
                    for statement in ast.parse(zygote_bytes, filename=str(zygote_path)).body:
                        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                            target = statement.targets[0]
                            if isinstance(target, ast.Name) and target.id in (
                                    'PRELOAD', 'CPU_PRELOAD', 'WORKER_CPU_PRELOAD'):
                                require(target.id not in constants, 'duplicate sealed preload assignment')
                                value = statement.value
                                if isinstance(value, ast.Name):
                                    require(value.id in constants, 'unknown/forward sealed preload alias')
                                    parsed = constants[value.id]
                                else:
                                    require(isinstance(value, (ast.Tuple, ast.List)),
                                            'sealed preload must be a literal tuple/list or known alias')
                                    parsed = []
                                    for element in value.elts:
                                        if isinstance(element, ast.Starred):
                                            require(isinstance(element.value, ast.Name)
                                                    and element.value.id in constants,
                                                    'sealed preload spread must name a known preceding tuple')
                                            parsed.extend(constants[element.value.id])
                                        else:
                                            require(isinstance(element, ast.Constant)
                                                    and isinstance(element.value, str),
                                                    'sealed preload elements must be literal module names')
                                            parsed.append(element.value)
                                constants[target.id] = tuple(parsed)
                    preload = constants.get('WORKER_CPU_PRELOAD', constants.get('CPU_PRELOAD'))
                    require(isinstance(preload, tuple) and preload
                            and all(isinstance(name, str) for name in preload),
                            'old sealed worker CPU preload contract is not explicit')
                    legacy = set(preload) == {'torch', 'torch_tpu', 'jax'}
                    expected_ray = 'ray' in preload
                    modules = witness.get('saved_module_ids', {})
                    require(set(modules) == set(preload)
                            and all(type(value) is int and value > 0 for value in modules.values())
                            and witness.get('ray_imported') is expected_ray
                            and (witness.get('ray_initialized') is False
                                 or (legacy and 'ray_initialized' not in witness)),
                            'capture does not satisfy its old sealed preload/Ray boundary')
                    result['old_cpu_preload'] = list(preload)
                    result['old_capture_scope'] = ('legacy three-module CPU' if legacy else
                        'expanded CPU, Ray uninitialized' if expected_ray else 'worker CPU, Ray-free')
                    require(witness.get('role') == 'workers'
                            and witness.get('pid') == witness.get('proc_pid') == witness.get('capture_pid')
                                == application.get('pid')
                            and witness.get('tpu_initialized') is False
                            and witness.get('proxy') is None,
                            'unpublished capture crossed the CPU boundary')
                    unpublished.update(images)
                require(qualified.get('extra_private_roots') == record.get('extra_private_roots'),
                        'qualification temp ownership differs')
                for name, expected_identity in record['extra_private_roots'].items():
                    temp = Path(name)
                    require(temp.parent == Path('/tmp') and temp.name.startswith('pmrr-')
                            and proof.get('private_temp') == str(temp)
                            and (qroot / 'private-temp').is_symlink()
                            and (qroot / 'private-temp').resolve() == temp
                            and root_identity(temp) == expected_identity, 'unknown/replaced qualification temp root')
                    require(temp not in extras, 'qualification temp root shared across generations')
                    no_mounts(temp)
                    extras[temp] = expected_identity
                temp = Path(proof['private_temp'])
                require(temp.is_relative_to(qroot) or temp in extras, 'unrecorded external qualification temp')
                qualifications[qroot] = directory.name
            else:
                require(not owners and measurement.get('success') is True
                        and measurement.get('source_hashes') == sealed['source_hashes']
                        and measurement.get('native_bundle_sha256') == bundle_hash, 'unknown install log ownership')
            logs.append(directory)
        require(not setup_run or setup is not None, 'explicit preserved setup directory was not found')
        require(qualifications and set((c.pm / 'qualifications').iterdir()) == set(qualifications),
                'unknown or missing failed qualification generation')
        for root in [*qualifications, *extras]:
            paths = list(root.rglob('*'))
            for path in paths:
                if path.is_symlink() and path.is_dir():
                    require(root in qualifications and path == root / 'private-temp'
                            and path.resolve() in extras, f'unknown qualification directory link: {path}')
            require(all(path in supervisors for path in paths if path.name == 'supervisor.json'),
                    f'unchecked supervisor outside qualification capture: {root}')
            require(not any(path.name in ('manifest.json', 'manifest.pending', 'ray-driver.json', 'ray-ready.json')
                            or path.name.startswith(('run', 'restore', 'generation', 'controller-'))
                            for path in paths)
                    and all(path in unpublished for path in paths if '.dmtcp' in path.name),
                    f'published image/restore/Ray driver/generation state exists: {root}')
        result['unpublished_images'] = [str(path) for path in sorted(unpublished)]
        metadata = []
        for filename, module in (('torchtpu_load_plugin.py', 'ttl_probe'),
                                 ('torchtpu_ray_pin_plugin.py', 'ttl_ray_pin'),
                                 ('pm_torchtpu_plugin.py', 'pm_torchtpu')):
            matches = [Path(name) for name in sealed['files']
                       if Path(name).name == module + '.py' and Path(name).is_relative_to(c.base / 'venv')]
            require(len(matches) == 1 and matches[0].read_bytes() == payload[filename],
                    f'retained plugin would change: {module}')
            directory = matches[0].parent / (module + '-0.0.0.dist-info')
            require(set(directory.iterdir()) == {directory / 'METADATA', directory / 'entry_points.txt'}
                    and all(str(path) in sealed['files'] and digest(path) == sealed['files'][str(path)]
                            for path in directory.iterdir()), f'unknown plugin metadata: {directory}')
            metadata.append(directory)
        removals = [*extras, c.pm, c.src, c.native, c.base / 'runs', *logs, *metadata]
        require(all(not source.is_relative_to(path) and not archive.is_relative_to(path) for path in removals),
                'incoming source/archive is inside the deletion set')
        identities = {str(path): root_identity(path) for path in removals}
        for path in removals:
            no_mounts(path)
        require(shutil.rmtree.avoids_symlink_attacks, 'symlink-safe rmtree required')
        private(archive, create=True)
        archived = []
        roots = [(path, archive / 'logs' / path.name) for path in logs]
        roots += [(path, archive / 'qualifications' / name) for path, name in qualifications.items()]
        roots += [(path, archive / 'temps' / path.name) for path in extras]
        files = [(c.base / name, archive / name)
                 for name in ('rewind-owner.json', 'rewind-install.json', 'rewind-install.log')]
        if setup is not None:
            files += [(path, archive / 'setup' / setup.name / path.name) for path in setup.iterdir()]
        for root, destination in roots:
            files += [(path, destination / path.relative_to(root)) for path in root.rglob('*')
                      if path.is_file() and (root in logs or path.suffix == '.json' or '.log' in path.name)]
        for path, destination in files:
            require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 * 1024 * 1024,
                    f'reset archive requires a small regular receipt/log: {path}')
            expected_hash = digest(path)
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            require(digest(destination) == expected_hash == digest(path), f'archive changed during copy: {path}')
            archived.append({'source': str(path), 'archive': str(destination), 'sha256': expected_hash})
        result.update(archive_files=archived, authorized_deletions=identities,
                      qualifications=sorted(qualifications.values()))
        dump(c.log / 'rewind-reset.json', result)
        old.check_install()  # Still the old seal; no source/metadata deletion has happened yet.
        for path in removals:
            require(root_identity(path) == identities[str(path)], f'reset deletion root changed: {path}')
            no_mounts(path)
            result.update(state='removing', deleting=str(path))
            dump(c.log / 'rewind-reset.json', result)
            shutil.rmtree(path)
            result['deleted'].append(str(path))
            result.pop('deleting', None)
            dump(c.log / 'rewind-reset.json', result)
        for name in ('rewind-install.log', 'rewind-install.json', 'rewind-owner.json'):
            path = c.base / name
            expected_hash = next(item['sha256'] for item in archived if item['source'] == str(path))
            require(digest(path) == expected_hash, f'old seal/log changed before removal: {path}')
            result.update(state='removing', deleting=str(path))
            dump(c.log / 'rewind-reset.json', result)
            path.unlink()
            result['deleted'].append(str(path))
            result.pop('deleting', None)
            dump(c.log / 'rewind-reset.json', result)
        require(root_identity(c.base) == sealed['retirement_roots'][str(c.base)]
                and root_identity(c.base / 'venv') == venv_identity, 'preserved BASE/venv identity changed')
        result.update(success=True, cleanup_complete=True, state='old_install_removed_venv_preserved')
        dump(c.log / 'rewind-reset.json', result)
        return result
    except BaseException as exc:
        result.update(success=False, error=f'{type(exc).__name__}: {exc}',
                      state='reset_failed', leases_must_stay_held=True)
        if journal_ready:
            dump(c.log / 'rewind-reset.json', result)
        raise


def install(c):
    lineage = advance_lineage(c)
    source = Path(os.environ.get('TTL_REWIND_SOURCE', str(Path(__file__).resolve().parent))).resolve()
    private(c.base)
    require(c.python.is_file(), 'run private venv setup with TTL_SKIP_COPY=1 first')
    payload = {name: (source / name).read_bytes() for name in SOURCES}
    bundle = source / 'pm_rewind_native.tar.gz'
    bundle_hash = digest(bundle)
    owner_path = c.base / 'rewind-owner.json'
    owner = {'format': 1, 'uid': os.getuid(), 'config': c.spec,
             'source_hashes': {n: hashlib.sha256(b).hexdigest() for n, b in payload.items()},
             'bundle_sha256': bundle_hash,
             'private_paths': [str(c.pm), str(c.cache), str(c.src), str(c.base / 'runs'),
                               str(c.pm / 'qualifications')],
             'retirement_roots': [str(c.pm), str(c.base)],
             'phase_lock': str(phase_lock_path(c))}
    revision = source_revision(c)
    if revision:
        owner['source_revision'] = revision['revision']
    if os.path.lexists(owner_path):
        require(owner_path.is_file() and not owner_path.is_symlink(), 'unsafe install owner')
    require(not lineage or (owner_path.is_file() and load(owner_path) == owner),
            'advanced/revised installs cannot reset or upgrade through rewind-install')
    if owner_path.is_file() and load(owner_path) == owner:
        receipt = c.check_install()
        # Advanced/revised installs are validation-only, including inode identity.
        if not lineage:
            for name, data in payload.items():
                stable_write(c.src / name, data)
        return {'success': True, 'phase': 'rewind-install', 'unchanged': True,
                'config_hash': c.config_hash, 'source_hashes': receipt['source_hashes'],
                'native_bundle_sha256': receipt['native_bundle_sha256']}
    # A crash after any reset deletion (including before the new seal is written)
    # must not become an implicit fresh install, with or without the reset flag.
    logs = c.base / 'logs'
    for directory in logs.iterdir() if logs.exists() else ():
        journal = directory / 'rewind-reset.json'
        measurement_path = directory / 'rewind_measurement.json'
        measurement = load(measurement_path) if measurement_path.is_file() else {}
        if not os.path.lexists(journal):
            error_path = directory / 'rewind-phase-error.json'
            error = load(error_path) if error_path.is_file() else {}
            previous = error.get('reset') or measurement.get('reset')
            if previous is not None:
                c.reset_receipt = previous
            require(previous is None and not os.path.lexists(directory / 'reset-archive'),
                    f'prior reset evidence lacks its journal; no automatic retry: {directory}')
            continue
        require(directory.is_dir() and not directory.is_symlink()
                and journal.is_file() and not journal.is_symlink(), 'unsafe prior reset journal')
        previous = load(journal)
        if not (owner_path.is_file() and previous.get('success') is True
                and previous.get('cleanup_complete') is True
                and previous.get('state') == 'old_install_removed_venv_preserved'
                and not previous.get('leases_must_stay_held') and 'deleting' not in previous
                and measurement.get('phase') == 'rewind-install' and measurement.get('success') is True
                and measurement.get('reset') == previous):
            c.reset_receipt = previous
            raise RuntimeError(f'prior reset/install is incomplete; no automatic retry: {journal}')
    reset = None
    reset_flag = os.environ.get('TTL_REWIND_RESET_UNQUALIFIED', '0')
    require(reset_flag in ('0', '1'), 'TTL_REWIND_RESET_UNQUALIFIED must be 0 or 1')
    if os.path.lexists(owner_path):
        require(reset_flag == '1', 'unknown/conflicting install: use new BASE and PM_ROOT')
        reset = reset_unqualified(c, payload, bundle_hash, source)
    for path in (c.src, c.pm, c.cache, c.base / 'runs', c.base / 'native'):
        require(not path.exists(), f'refusing to adopt unknown existing path: {path}')
    stable_write(owner_path, encoded(owner))
    for path in (c.src, c.pm, c.base / 'runs'):
        private(path, create=True)
    # BASE and PM_ROOT are newly owned; only direct/nested children are admitted.
    for path in (c.cache, c.pm / 'bank', c.pm / 'snap', c.pm / 't3', c.ckpt,
                 c.base / 'hf', c.base / 'xdg', c.pm / 'qualifications'):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        private(path)
    logs = c.base / 'logs'
    if not logs.exists():
        private(logs, create=True)
    else:
        info = logs.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
                and not info.st_mode & 0o022, 'untrusted setup log directory')
    for name, data in payload.items():
        stable_write(c.src / name, data)
    shim = f'#!{c.python}\nimport sys\nsys.path.insert(0, {str(c.src)!r})\nfrom pm_rewind_tpu import fresh_entry\nraise SystemExit(fresh_entry(sys.argv[1:]))\n'
    stable_write(c.src / 'fresh-entry', shim.encode(), 0o700)
    site = Path(subprocess.check_output([str(c.python), '-c',
                'import sysconfig;print(sysconfig.get_paths()["purelib"])'], text=True).strip())
    require(site.resolve().is_relative_to((c.base / 'venv').resolve()), 'venv site-packages escaped BASE')
    installed = [c.src / name for name in SOURCES] + [c.src / 'fresh-entry']
    core_entries = list(site.glob('vllm_torchtpu_core-*.dist-info/entry_points.txt'))
    require(len(core_entries) == 1, 'exactly one TorchTPU core plugin distribution required')
    plugins = configparser.ConfigParser(interpolation=None)
    plugins.read(core_entries[0])
    require(plugins.get('vllm.platform_plugins', 'torchtpu', fallback=None)
            == 'vllm_torchtpu.platforms:register_tpu_platform'
            and plugins.get('vllm.general_plugins', 'torchtpu_layers', fallback=None)
            == 'vllm_torchtpu.layers:register_layers', 'unexpected TorchTPU platform/layer plugin entrypoints')
    installed.extend(core_entries)
    for filename, module in (('torchtpu_load_plugin.py', 'ttl_probe'),
                             ('torchtpu_ray_pin_plugin.py', 'ttl_ray_pin'),
                             ('pm_torchtpu_plugin.py', 'pm_torchtpu')):
        target = site / (module + '.py')
        stable_write(target, payload[filename])
        metadata = site / (module + '-0.0.0.dist-info')
        require(not metadata.exists(), f'unknown existing plugin metadata: {metadata}')
        metadata.mkdir(mode=0o700)
        stable_write(metadata / 'METADATA', f'Metadata-Version: 2.1\nName: {module.replace("_", "-")}\nVersion: 0.0.0\n'.encode())
        stable_write(metadata / 'entry_points.txt', f'[vllm.general_plugins]\n{module} = {module}:register\n'.encode())
        installed += [target, metadata / 'METADATA', metadata / 'entry_points.txt']
    checked([c.python, c.src / 'pm_source_patches.py', 'apply', site], c.base / 'rewind-install.log')
    checked([c.python, c.src / 'pm_rewind_native.py', 'install', '--bundle', bundle,
             '--dest', c.native], c.base / 'rewind-install.log')
    for package in ('vllm', 'vllm_torchtpu', 'torch_tpu'):
        installed.extend(sorted((site / package).rglob('*.py')))
    installed.extend(p for p in c.native.rglob('*') if p.is_file())
    receipt = {'config_hash': c.config_hash, 'config': c.spec, 'source_hashes': owner['source_hashes'],
               'files': {str(p): digest(p) for p in installed}, 'python': str(c.python),
               'native_bundle_sha256': bundle_hash, 'installed_at': time.time(),
               'retirement_roots': {str(path): {'device': path.stat().st_dev, 'inode': path.stat().st_ino,
                                              'uid': path.stat().st_uid} for path in (c.pm, c.base)}}
    dump(c.base / 'rewind-install.json', receipt)
    return {'success': True, 'phase': 'rewind-install', 'config_hash': c.config_hash,
            'source_hashes': owner['source_hashes'], 'native_bundle_sha256': bundle_hash,
            'reset': reset}


def refreshed_model(parent, marker):
    refresh = marker['refresh']
    return dict(parent, install_id=marker['new_install_id'], source_refresh={
        'run_id': refresh['run_id'], 'parent_install_id': refresh['parent_install_id'],
        'parent_model_receipt_sha256': refresh['parent_model_receipt_sha256'],
        'revision_marker_sha256': refresh['revision_marker_sha256']})


def source_refresh(c):
    """Validate the sole successor; never reinterpret or rewrite earlier markers."""
    path = c.base / 'rewind-refresh.json'
    withdrawals = list(c.base.glob('rewind-refresh-parent-install-*.json'))
    archives = list(c.base.glob('rewind-refresh-archive-*'))
    qualifications = list(c.pm.glob('rewind-refresh-qualifications-*'))
    images = list(c.pm.glob('rewind-refresh-images-*'))
    require(not any(c.src.glob('*.refresh-pending')), 'interrupted refresh source publication')
    if not os.path.lexists(path):
        require(not withdrawals and not archives and not qualifications and not images,
                'lost/interrupted refresh; no automatic retry')
        for receipt in (c.base / 'rewind-install.json', c.base / 'rewind-owner.json',
                        c.pm / 'model-receipt.json'):
            if receipt.is_file():
                require(not load(receipt).get('source_refresh'), 'refresh marker lost')
        return None
    require(path.is_file() and not path.is_symlink(), 'unsafe refresh marker')
    marker = load(path)
    require(marker.get('phase') == 'rewind-refresh' and marker.get('success') is True
            and marker.get('state') == 'complete' and marker.get('leases_must_stay_held') is False,
            'refresh incomplete; retain leases and evidence, no automatic retry')
    refresh = marker['refresh']
    run_id = refresh['run_id']
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', run_id), 'invalid refresh run')
    archive = c.base / ('rewind-refresh-archive-' + run_id)
    withdrawn = c.base / ('rewind-refresh-parent-install-' + run_id + '.json')
    require(archives == [archive] and withdrawals == [withdrawn]
            and qualifications == [c.pm / ('rewind-refresh-qualifications-' + run_id)]
            and images == [c.pm / ('rewind-refresh-images-' + run_id)]
            and refresh['archive'] == str(archive)
            and refresh['retired_qualifications'] == str(qualifications[0])
            and refresh['retired_images']['archive'] == str(images[0])
            and refresh['retired_images']['original'] == str(c.cache)
            and marker['withdrawn_install_receipt'] == str(withdrawn),
            'refresh lost its permanent withdrawal/archive evidence')
    for root in (archive, qualifications[0], images[0]):
        private(root)
    for key in ('run_id', 'parent_install_id', 'parent_config_hash', 'archive'):
        require(marker.get(key) == refresh.get(key), f'refresh identity mismatch: {key}')
    for receipt, expected in (
            (withdrawn, refresh['parent_install_receipt_sha256']),
            (archive / 'rewind-install.json', refresh['parent_install_receipt_sha256']),
            (archive / 'rewind-owner.json', refresh['parent_owner_sha256']),
            (archive / 'model-receipt.json', refresh['parent_model_receipt_sha256']),
            (archive / 'rewind-revise.json', refresh['revision_marker_sha256']),
            (c.base / 'rewind-revise.json', refresh['revision_marker_sha256']),
            (archive / 'rewind-advance.json', refresh['advance_marker_sha256']),
            (c.base / 'rewind-advance.json', refresh['advance_marker_sha256']),
            (c.base / 'rewind-install.json', marker['new_install_receipt_sha256']),
            (c.base / 'rewind-owner.json', marker['new_owner_sha256']),
            (c.pm / 'model-receipt.json', marker['new_model_receipt_sha256'])):
        require(receipt.is_file() and not receipt.is_symlink() and digest(receipt) == expected,
                f'refresh receipt changed: {receipt}')
    parent = load(archive / 'rewind-install.json')
    parent_owner = load(archive / 'rewind-owner.json')
    parent_model = load(archive / 'model-receipt.json')
    expected_parent = {
        'INSTALL_ID': hashlib.sha256(encoded({'sources': parent['source_hashes'],
                         'native': parent['native_bundle_sha256']})).hexdigest(),
        'CONFIG_HASH': parent['config_hash'], 'MODEL_ID': parent_model['model_id'],
        'SOURCE_SHA256': hashlib.sha256(encoded(parent['source_hashes'])).hexdigest(),
    }
    require(expected_parent == REFRESH_PARENT and parent.get('source_revision')
            and not parent.get('source_refresh')
            and refresh['parent_install_id'] == expected_parent['INSTALL_ID']
            and refresh['parent_config_hash'] == expected_parent['CONFIG_HASH']
            and refresh['parent_model_id'] == expected_parent['MODEL_ID']
            and refresh['parent_source_hashes'] == parent['source_hashes'],
            'refresh is only authorized for the exact reviewed revised parent')
    installed = load(c.base / 'rewind-install.json')
    owner = load(c.base / 'rewind-owner.json')
    hashes = installed['source_hashes']
    changed = {name for name in parent['source_hashes'] if hashes.get(name) != parent['source_hashes'][name]}
    require(set(hashes) == set(parent['source_hashes']) == set(SOURCES)
            and changed == {'pm_rewind_tpu.py', 'pm_ray_zygote.py'}
            and hashes['pm_ray_zygote.py'] == REFRESH_RUNTIME_SHA256
            and sorted(changed) == refresh['changed_sources']
            and hashes == marker['new_source_hashes']
            and parent['config'] == c.spec and parent['config_hash'] == c.config_hash == marker['new_config_hash']
            and hashlib.sha256(encoded({'sources': hashes,
                'native': parent['native_bundle_sha256']})).hexdigest() == marker['new_install_id'],
            'refresh source/config contract changed')
    files = dict(parent['files'])
    files.update({str(c.src / name): hashes[name] for name in changed})
    require(installed == dict(parent, source_hashes=hashes, files=files,
                              installed_at=installed['installed_at'], source_refresh=refresh)
            and owner == dict(parent_owner, source_hashes=hashes, source_refresh=refresh)
            and load(c.pm / 'model-receipt.json') == refreshed_model(parent_model, marker),
            'refresh modified fields outside its source-only contract')
    for name, expected_identity in refresh['retained_identities'].items():
        require(file_identity(Path(name)) == expected_identity, f'refresh retained inode changed: {name}')
    for role, sha in refresh['retired_images']['manifests'].items():
        manifest = c.cache / role / 'manifest.json'
        require(not os.path.lexists(manifest) or (not manifest.is_symlink() and digest(manifest) != sha),
                f'refresh cannot reuse retired {role} image')
    return marker


def source_revision(c):
    """The source revision is active authority; advance remains historical evidence."""
    refreshed = source_refresh(c)
    current = Path(refreshed['archive']) if refreshed else c.base
    current_model = current / 'model-receipt.json' if refreshed else c.pm / 'model-receipt.json'
    path = c.base / 'rewind-revise.json'
    withdrawals = list(c.base.glob('rewind-revision-parent-install-*.json'))
    archives = list(c.base.glob('rewind-revision-archive-*'))
    if not os.path.lexists(path):
        require(not withdrawals and not archives, 'lost/interrupted source revision; no automatic retry')
        for receipt in (c.base / 'rewind-install.json', c.base / 'rewind-owner.json',
                        c.pm / 'model-receipt.json'):
            if receipt.is_file():
                require(not load(receipt).get('source_revision'), 'source revision marker lost')
        return None
    require(path.is_file() and not path.is_symlink(), 'unsafe source revision marker')
    marker = load(path)
    require(marker.get('phase') == 'rewind-revise' and marker.get('success') is True
            and marker.get('state') == 'complete' and not marker.get('leases_must_stay_held'),
            'source revision incomplete; retain leases and evidence, no automatic retry')
    revision = marker['revision']
    run_id = revision['run_id']
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', run_id), 'invalid source revision run')
    archive = c.base / ('rewind-revision-archive-' + run_id)
    withdrawn = c.base / ('rewind-revision-parent-install-' + run_id + '.json')
    require(archives == [archive] and withdrawals == [withdrawn]
            and revision['archive'] == str(archive)
            and marker['withdrawn_install_receipt'] == str(withdrawn),
            'source revision lost its permanent withdrawal/archive evidence')
    private(archive)
    for key in ('run_id', 'parent_install_id', 'parent_config_hash', 'archive'):
        require(marker.get(key) == revision.get(key), f'source revision identity mismatch: {key}')
    for path, expected in (
            (withdrawn, revision['parent_install_receipt_sha256']),
            (archive / 'rewind-install.json', revision['parent_install_receipt_sha256']),
            (archive / 'model-receipt.json', revision['parent_model_receipt_sha256']),
            (archive / 'rewind-owner.json', revision['parent_owner_sha256']),
            (archive / 'rewind-advance.json', revision['advance_marker_sha256']),
            (c.base / 'rewind-advance.json', revision['advance_marker_sha256']),
            (current / 'rewind-install.json', marker['new_install_receipt_sha256']),
            (current / 'rewind-owner.json', marker['new_owner_sha256']),
            (current_model, marker['new_model_receipt_sha256'])):
        require(path.is_file() and not path.is_symlink() and digest(path) == expected,
                f'source revision receipt changed: {path}')
    installed = load(current / 'rewind-install.json')
    parent = load(archive / 'rewind-install.json')
    owner = load(current / 'rewind-owner.json')
    require(installed.get('source_revision') == revision
            and owner.get('source_revision') == revision
            and installed.get('lineage') == parent.get('lineage')
            and installed['config'] == parent['config'] == c.spec
            and installed['config_hash'] == parent['config_hash'] == c.config_hash == marker['new_config_hash']
            and installed['native_bundle_sha256'] == parent['native_bundle_sha256']
            and installed['source_hashes'] == owner['source_hashes'] == marker['new_source_hashes']
            and hashlib.sha256(encoded({'sources': installed['source_hashes'],
                'native': installed['native_bundle_sha256']})).hexdigest() == marker['new_install_id'],
            'source revision does not describe its historical seal')
    changed = {name for name in parent['source_hashes']
               if installed['source_hashes'].get(name) != parent['source_hashes'][name]}
    require(set(installed['source_hashes']) == set(parent['source_hashes'])
            and changed == {'pm_rewind_tpu.py', 'pm_ray_zygote.py'}
            and installed['source_hashes']['pm_ray_zygote.py'] == REVISION_RUNTIME_SHA256
            and sorted(changed) == revision['changed_sources'],
            'source revision must match the exact helper pair and frozen CPU runtime')
    files = dict(parent['files'])
    files.update({str(c.src / name): installed['source_hashes'][name] for name in changed})
    require(installed == dict(parent, source_hashes=installed['source_hashes'], files=files,
                              installed_at=installed['installed_at'], source_revision=revision)
            and owner == dict(load(archive / 'rewind-owner.json'),
                              source_hashes=installed['source_hashes'], source_revision=revision)
            and revision['parent_source_hashes'] == parent['source_hashes']
            and revision['parent_config_hash'] == c.config_hash
            and revision['parent_install_id'] == hashlib.sha256(encoded({
                'sources': parent['source_hashes'], 'native': parent['native_bundle_sha256']})).hexdigest(),
            'revision modified fields outside its source-only contract')
    parent_model = load(archive / 'model-receipt.json')
    require(revision['parent_model_id'] == parent_model['model_id']
            and load(current_model) == revised_model(parent_model, marker),
            'revision model linkage differs from its retained parent')
    for name, expected_identity in revision['retained_identities'].items():
        require(file_identity(Path(name)) == expected_identity, f'retained inode changed: {name}')
    return marker


def advance_lineage(c):
    """Validate the historical advance against its seal, not a revised active seal."""
    revision = source_revision(c)
    path = c.base / 'rewind-advance.json'
    if not os.path.lexists(path):
        require(not revision and not any(c.base.glob('rewind-advance-parent-install-*.json')),
                'interrupted seal withdrawal; no automatic advance/install retry')
        require(not load(c.base / 'rewind-install.json').get('lineage')
                if (c.base / 'rewind-install.json').exists() else True,
                'advanced seal lost its transition marker')
        return None
    require(path.is_file() and not path.is_symlink(), 'unsafe advance marker')
    marker = load(path)
    require(marker.get('success') is True and marker.get('state') == 'complete',
            'rewind-advance incomplete; retain leases and evidence, no automatic retry')
    seal_path = (Path(revision['archive']) / 'rewind-install.json'
                 if revision else c.base / 'rewind-install.json')
    model_path = (Path(revision['archive']) / 'model-receipt.json'
                  if revision else c.pm / 'model-receipt.json')
    installed = load(seal_path)
    require(installed.get('lineage') == marker.get('lineage') and marker.get('lineage')
            and installed['source_hashes'] == marker.get('new_source_hashes')
            and installed['config_hash'] == marker.get('new_config_hash') == c.config_hash
            and hashlib.sha256(encoded({'sources': installed['source_hashes'],
                'native': installed['native_bundle_sha256']})).hexdigest() == marker.get('new_install_id'),
            'advance marker does not describe its historical seal')
    for key, value in marker['lineage'].items():
        require(marker.get(key) == value, f'advance lineage mismatch: {key}')
    for name, expected in marker['seed_receipts'].items():
        path = Path(name)
        require(path.is_file() and not path.is_symlink() and digest(path) == expected,
                f'old seed receipt changed after advance: {path}')
    require(digest(seal_path) == marker.get('new_install_receipt_sha256')
            and digest(model_path) == marker.get('new_model_receipt_sha256'),
            'advanced install/model receipt changed after commit')
    for role in ('workers', 'api'):
        manifest = c.cache / role / 'manifest.json'
        retired_hash = marker['retired_images']['files'].get(role + '/manifest.json')
        require(not manifest.exists() or digest(manifest) != retired_hash,
                f'retired parent {role} image cannot be reused')
    return marker


def require_model_receipt(c):
    """Accept a copied model or the exact inode-preserving, archived advance parent."""
    model = load(c.pm / 'model-receipt.json')
    require(model.get('success') and model.get('config_hash') == c.config_hash
            and model.get('install_id') == c.install_id, 'same-install model receipt required')
    lineage = advance_lineage(c)
    active_model = model
    refreshed = source_refresh(c)
    if refreshed:
        model = load(Path(refreshed['archive']) / 'model-receipt.json')
        require(active_model == refreshed_model(model, refreshed),
                'model is not the exact retained refresh parent')
    revision = source_revision(c)
    if revision:
        revision_model = load(Path(revision['archive']) / 'model-receipt.json')
        require(model == revised_model(revision_model, revision),
                'model is not the exact retained source-revision parent')
        model = revision_model
    original = model
    if lineage:
        path = Path(lineage['archive']) / 'model-receipt.json'
        require(digest(path) == lineage['parent_model_receipt_sha256'],
                'archived parent model receipt changed')
        original = load(path)
        expected = dict(original, config_hash=c.config_hash, install_id=lineage['new_install_id'],
                        parent_install_id=lineage['parent_install_id'],
                        parent_config_hash=lineage['parent_config_hash'],
                        parent_model_receipt_sha256=lineage['parent_model_receipt_sha256'],
                        advance_run_id=lineage['run_id'])
        require(model == expected and model['model_id'] == lineage['parent_model_id'],
                'model is not the exact retained advance parent')
        for name, expected_identity in lineage['retained_identities'].items():
            require(file_identity(Path(name)) == expected_identity, f'retained inode changed: {name}')
    copy_run = original.get('run_id', '')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', copy_run), 'invalid original copy run')
    copy_log = (Path(lineage['archived_logs'][str(c.base / 'logs' / copy_run)])
                if lineage else c.base / 'logs' / copy_run)
    copied = load(copy_log / 'rewind-copy-result.json')
    require(copied.get('success') and copied.get('cleanup_complete')
            and copied.get('run_id') == copy_run and copied.get('model_id') == original.get('model_id')
            and copied.get('config_hash') == original.get('config_hash')
            and copied.get('install_id') == original.get('install_id')
            and load(copy_log / 'model-receipt.json') == original,
            'model lacks matching original copy/cleanup evidence')
    require(model_files(c.ckpt) == model.get('files')
            and hashlib.sha256(encoded(model['files'])).hexdigest() == model['model_id'],
            'retained model bytes changed')
    return active_model


def file_identity(path):
    info = path.lstat()
    require(not path.is_symlink() and info.st_uid == os.getuid()
            and (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)),
            f'owned real file/directory required: {path}')
    return {'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid}


def revised_model(parent, marker):
    revision = marker['revision']
    return dict(parent, install_id=marker['new_install_id'], source_revision={
        'run_id': revision['run_id'], 'parent_install_id': revision['parent_install_id'],
        'parent_model_receipt_sha256': revision['parent_model_receipt_sha256'],
        'advance_marker_sha256': revision['advance_marker_sha256']})


def revision_tree(root):
    """Read-only byte/inode inventory; never follow an evidence/runtime symlink."""
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), f'unsafe revision tree: {root}')
    entries = {}
    for path in (root, *sorted(root.rglob('*'))):
        info = path.lstat()
        require(info.st_uid == os.getuid(), f'unowned revision evidence: {path}')
        entry = {'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid,
                 'mode': info.st_mode}
        if stat.S_ISREG(info.st_mode):
            entry.update(bytes=info.st_size, sha256=digest(path))
        elif stat.S_ISLNK(info.st_mode):
            entry['target'] = os.readlink(path)
        else:
            require(stat.S_ISDIR(info.st_mode) or stat.S_ISSOCK(info.st_mode),
                    f'unsupported revision evidence: {path}')
        entries[str(path.relative_to(root))] = entry
    return entries


def legacy_worker_query_failure(module, exc):
    """Recognize only the verified old subprocess's worker-truncation failure.

    Inspect the actual failing frame read-only: the old helper masks its child's
    error with a PG-membership exception. Never replace functions or reroute its
    subprocess, and never authorize an upgrade from a supplied/stale receipt.
    """
    if type(exc) is not RuntimeError:
        return None
    tb = exc.__traceback__
    frames = []
    while tb is not None:
        if tb.tb_frame.f_code is module.ray_cleanup_evidence.__code__:
            frames.append(tb.tb_frame.f_locals)
        tb = tb.tb_next
    if len(frames) != 1:
        return None
    frame = frames[0]
    audit, proc, context = frame.get('audit'), frame.get('proc'), frame.get('c')
    job = frame.get('job')
    if (not isinstance(audit, dict) or set(audit) != {'job_id', 'complete', 'error'}
            or audit.get('complete') is not False or audit.get('job_id') != job
            or not isinstance(job, str) or re.fullmatch('[0-9a-f]{8,64}', job) is None
            or frame.get('read_only') is not True or context is None or proc is None
            or proc.returncode != 1
            or str(exc) != f'previous exact-job placement groups absent from fresh audit: {job}'):
        return None
    command = [str(context.python), '-B', str(Path(module.__file__).resolve()),
               '_ray-audit', job, context.state_address, '-']
    if frame.get('command') != command or proc.args != command:
        return None
    try:
        if json.loads(proc.stdout) != audit:
            return None
        error = audit['error']
        if not isinstance(error, str) or not error.startswith('RayStateApiException(') or not error.endswith(')'):
            return None
        message = ast.literal_eval(error[len('RayStateApiException('):-1])
    except (TypeError, ValueError, SyntaxError):
        return None
    pattern = (r'Failed to retrieve all ([0-9]+) workers from the cluster because they are not reachable '
               r'due to data truncation\. It happens when the returned data is too large \(> 10000\) '
               r'To avoid raising an exception and allow having missing output, '
               r'set `raise_on_missing_output=False`\. ')
    match = re.fullmatch(pattern, message) if isinstance(message, str) else None
    if match is None or int(match[1]) <= 10000:
        return None
    return {'kind': 'verified_global_worker_table_truncation', 'command': command,
            'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr,
            'audit': dict(audit), 'global_worker_count': int(match[1]), 'global_worker_limit': 10000}


def refresh(c):
    """Retire this revised CPU generation; replace only the reviewed helper pair."""
    marker_path = c.base / 'rewind-refresh.json'
    require(source_refresh(c) is None, 'refresh is one-shot; no retry or seal repair')
    require(c.arm != 'seed' and os.environ.get('TTL_REWIND_RESET_UNQUALIFIED', '0') == '0',
            'refresh forbids reseeding and reset')
    require(getattr(c, 'phase_lock_fd', None) is not None, 'refresh requires the existing phase lock')
    require(not os.path.lexists(c.log) and not os.path.lexists(c.generation), 'refresh requires a fresh TTL_RUN')
    source = Path(os.environ.get('TTL_REWIND_SOURCE', str(Path(__file__).resolve().parent))).resolve()
    require(not source.is_relative_to(c.base) and not source.is_relative_to(c.pm),
            'incoming refresh source must be outside owned roots')
    for name in (*SOURCES, 'pm_rewind_native.tar.gz'):
        require((source / name).is_file() and not (source / name).is_symlink(),
                f'missing/unsafe refresh payload: {name}')
    payload = {name: (source / name).read_bytes() for name in SOURCES}
    require(payload['pm_rewind_tpu.py'] == Path(__file__).read_bytes(), 'refresh must execute its incoming helper')
    sealed_path, owner_path = c.base / 'rewind-install.json', c.base / 'rewind-owner.json'
    model_path, qualified_path = c.pm / 'model-receipt.json', c.base / 'rewind-qualified.json'
    advance_path, revision_path = c.base / 'rewind-advance.json', c.base / 'rewind-revise.json'
    receipts = (sealed_path, owner_path, model_path, qualified_path, advance_path, revision_path)
    for path in receipts:
        require(path.is_file() and not path.is_symlink(), f'missing/unsafe refresh parent receipt: {path}')
    sealed, old_owner, model = load(sealed_path), load(owner_path), load(model_path)
    expected = {
        'INSTALL_ID': hashlib.sha256(encoded({'sources': sealed['source_hashes'],
                         'native': sealed['native_bundle_sha256']})).hexdigest(),
        'CONFIG_HASH': sealed['config_hash'], 'MODEL_ID': model['model_id'],
        'SOURCE_SHA256': hashlib.sha256(encoded(sealed['source_hashes'])).hexdigest(),
    }
    require(expected == REFRESH_PARENT, 'refresh is bounded to the exact reviewed revised install')
    for key, value in dict(expected, NATIVE_BUNDLE_SHA256=sealed['native_bundle_sha256']).items():
        require(os.environ.get('TTL_REWIND_EXPECT_OLD_' + key) == value,
                f'explicit matching TTL_REWIND_EXPECT_OLD_{key} required')
    helper = c.src / 'pm_rewind_tpu.py'
    require(helper.is_file() and not helper.is_symlink(), 'old sealed helper missing')
    helper_bytes = helper.read_bytes()
    require(hashlib.sha256(helper_bytes).hexdigest() == sealed['files'].get(str(helper))
            == sealed['source_hashes'].get(helper.name) == old_owner['source_hashes'].get(helper.name),
            'old helper must match pinned parent source map, seal and owner before execution')
    spec = importlib.util.spec_from_file_location('_pm_rewind_refresh_old', helper)
    old_module = importlib.util.module_from_spec(spec)
    exec(compile(helper_bytes, str(helper), 'exec'), old_module.__dict__)
    old = old_module.Config()
    old.phase_lock_fd = c.phase_lock_fd
    require(old.check_install() == sealed and old.install_id == expected['INSTALL_ID']
            and old.spec == c.spec == sealed['config'] == old_owner['config']
            and old.config_hash == c.config_hash == expected['CONFIG_HASH']
            and old_owner['source_hashes'] == sealed['source_hashes']
            and old_owner['bundle_sha256'] == sealed['native_bundle_sha256']
            == digest(source / 'pm_rewind_native.tar.gz'),
            'refresh requires exact config/SDK/native bundle, never reinstall')
    revision = old_module.source_revision(old)
    historical = old_module.advance_lineage(old)
    require(revision and historical and revision['new_install_id'] == old.install_id
            and sealed.get('source_revision') == revision['revision']
            and not sealed.get('source_refresh') and not old_owner.get('source_refresh')
            and not model.get('source_refresh'), 'completed, not previously refreshed revision required')
    require(old_module.require_model_receipt(old) == model, 'old model receipt linkage differs')
    old_module.require_qualification(old)
    for prior in (historical, revision['revision']):
        require(c.run != prior['run_id'] and all(Path(name).name != c.run for name in prior['archived_logs']),
                'refresh run ID was already used by a historical generation')
    new_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    changed = {name for name in payload if new_hashes[name] != sealed['source_hashes'].get(name)}
    require(set(payload) == set(sealed['source_hashes'])
            and changed == {'pm_rewind_tpu.py', 'pm_ray_zygote.py'}
            and sealed['source_hashes']['pm_ray_zygote.py'] == REVISION_RUNTIME_SHA256
            and new_hashes['pm_ray_zygote.py'] == REFRESH_RUNTIME_SHA256,
            'only the reviewed successor runtime and control helper may change')
    for name, sha in sealed['source_hashes'].items():
        require(sealed['files'].get(str(c.src / name)) == sha == digest(c.src / name),
                f'old source differs from seal: {name}')
    require(all(any(Path(name).is_relative_to(root) for root in (c.src, c.native, c.base / 'venv'))
                for name in sealed['files']), 'old seal escaped owned source/runtime roots')
    def closed(who, label):
        require(isinstance(who, dict) and type(who.get('pid')) is int and who['pid'] > 0
                and type(who.get('start_ticks')) is int and who['start_ticks'] > 0
                and not old_module.alive(who), f'live/unknown refresh ownership: {label}')
    old_logs = sorted((c.base / 'logs').iterdir())
    for directory in old_logs:
        require(directory.is_dir() and not directory.is_symlink(), f'unsafe old log: {directory}')
        report = load(directory / 'rewind_measurement.json')
        require(report.get('run_id') == directory.name and report.get('host_rank') == c.rank,
                f'unknown current log identity: {directory}')
        phase = report.get('phase')
        if phase == 'rewind-revise':
            require(report == revision, 'prior revision measurement differs from immutable marker')
        elif phase == 'rewind-qualify':
            require((directory / 'rewind-qualification-owner.json').is_file(),
                    'qualification lacks ownership')
        elif phase == 'rewind-prepare':
            require(report.get('success') is True and report.get('config_hash') == c.config_hash,
                    'refresh requires a completed known prepare')
            for role, image in report['images'].items():
                require(role in (('workers', 'api') if c.rank == 0 else ('workers',))
                        and digest(c.cache / role / 'manifest.json') == image['manifest_sha256'],
                        'prepare does not identify the current production image')
        elif phase == 'rewind-run':
            require(report.get('success') is False and report.get('functional_success') is False
                    and not report.get('ready') and not report.get('cache_qualified')
                    and report.get('config_hash') == c.config_hash,
                    'refresh refuses successful or unknown current serving')
            require((directory / 'rewind-run-owner.json').is_file()
                    or (directory / 'rewind-preflight-result.json').is_file(),
                    'failed serving lacks explicit ownership')
        else:
            require(False, f'unknown current phase for refresh: {directory}: {phase}')
        for path in directory.glob('*owner.json'):
            require(path.name in ('rewind-qualification-owner.json', 'rewind-api-help-owner.json',
                                  'rewind-run-owner.json'), f'unknown current ownership: {path}')
            require(path.is_file() and not path.is_symlink(), f'unsafe current owner: {path}')
            owner = load(path)
            require(owner.get('config_hash') == c.config_hash and owner.get('run_id') == directory.name
                    and owner.get('host_rank') == c.rank, f'unknown current owner identity: {path}')
            closed(owner.get('owner'), path)
            for key in ('probe', 'child', 'controller'):
                if owner.get(key) is not None:
                    closed(owner[key], str(path) + '/' + key)
            for who in owner.get('controllers', {}).values():
                closed(who, str(path) + '/controller')
        serving = directory / 'serve_result.json'
        if os.path.lexists(serving):
            outcome = load(serving)
            require(phase == 'rewind-run' and outcome.get('success') is False
                    and outcome.get('functional_success') is False and not outcome.get('ready')
                    and not outcome.get('cache_qualified'),
                    f'successful/unknown current serving evidence: {serving}')
        preflight = directory / 'rewind-preflight-result.json'
        if preflight.is_file():
            closed(load(preflight).get('preflight_owner'), preflight)
        error = directory / 'rewind-phase-error.json'
        if error.exists():
            require(load(error).get('phase') in ('rewind-qualify', 'rewind-run'),
                    f'unknown current phase error: {error}')
    qualified = load(qualified_path)
    require(load(c.base / 'logs' / qualified['run_id'] / 'rewind-qualification-result.json').get('success') is True,
            'published old qualification lacks its successful result')
    # Execute the hash-verified old implementation unmodified, including its
    # real subprocess. Only its specific global-worker truncation may authorize
    # a query-method upgrade, never a successful old retirement claim.
    try:
        audit, _ = old_module.retirement_checks(old, read_only=True)
    except Exception as exc:
        failure = legacy_worker_query_failure(old_module, exc)
        upgrade = {
            'old_helper': str(helper), 'old_helper_sha256': sealed['source_hashes'][helper.name],
            'old_retirement_complete': False, 'old_retirement_error': repr(exc),
            'old_query_failure': failure, 'incoming_retirement_complete': False,
            'phase_flag': 'TTL_REWIND_EXACT_WORKER_AUDIT',
            'authorized': os.environ.get('TTL_REWIND_EXACT_WORKER_AUDIT') == '1',
        }
        c.retirement_query_upgrade = upgrade
        if failure is None:
            raise
        require(upgrade['authorized'],
                'verified old worker-table truncation requires explicit TTL_REWIND_EXACT_WORKER_AUDIT=1; '
                f'old failure retained: {exc}; {failure["audit"]["error"]}')
        # Use the incoming Config and functions at their real source path.
        # The old seal/config/model/native/seed and every current and historical
        # owner must pass again; this is a full read-only audit, not a Ray-only retry.
        require(c.check_install() == sealed and require_model_receipt(c) == model,
                'incoming retirement audit changed parent authority')
        require_qualification(c)
        audit, _ = retirement_checks(c, read_only=True)
        upgrade.update(incoming_retirement_complete=True,
                       incoming_helper=str(Path(__file__).resolve()),
                       incoming_helper_sha256=new_hashes['pm_rewind_tpu.py'],
                       query_method='GetActorInfo.address.worker_id -> GetWorkerInfo',
                       jobs={key: {'job_id': value.get('job_id'),
                                   'no_ray_action': value.get('no_ray_action', False),
                                   'query_proof': value.get('query_proof')}
                             for key, value in audit['ray_audits'].items()})
        audit['query_method_upgrade'] = upgrade
    else:
        c.retirement_query_upgrade = {
            'old_helper': str(helper), 'old_helper_sha256': sealed['source_hashes'][helper.name],
            'old_retirement_complete': True, 'old_retirement_error': None,
            'incoming_retirement_complete': False, 'authorized': False,
        }
        require(False, 'old retirement unexpectedly succeeded; this bounded query-method upgrade '
                       'requires the actual verified worker-table truncation failure')
    for root in (c.pm / 'qualifications', c.cache,
                 *(Path(name) for name in audit['authorized_roots'] if name not in (str(c.base), str(c.pm)))):
        for path in root.rglob('supervisor.json'):
            require(not path.is_symlink(), f'unsafe current native supervisor: {path}')
            retired_supervisor(load(path), str(path))
    archive = c.base / ('rewind-refresh-archive-' + c.run)
    retired_qualifications = c.pm / ('rewind-refresh-qualifications-' + c.run)
    retired_images = c.pm / ('rewind-refresh-images-' + c.run)
    withdrawn = c.base / ('rewind-refresh-parent-install-' + c.run + '.json')
    for path in (archive, retired_qualifications, retired_images, withdrawn):
        require(not os.path.lexists(path), f'unknown prior refresh evidence: {path}')
    private(c.cache)
    roles = {'workers', 'api'} if c.rank == 0 else {'workers'}
    require({path.name for path in c.cache.iterdir()} == roles, 'exact prepared image children required')
    image_trees = {role: revision_tree(c.cache / role) for role in sorted(roles)}
    log_trees = {path.name: revision_tree(path) for path in old_logs}
    sources = revision_tree(c.src)
    require(all(stat.S_ISREG(entry['mode']) or stat.S_ISDIR(entry['mode'])
                for tree in (sources, *log_trees.values()) for entry in tree.values()),
            'unsafe installed source/log entries')
    qualifications = revision_tree(c.pm / 'qualifications')
    retained_trees = {str(path): revision_tree(path) for path in (
        c.base / 'venv', c.native, c.ckpt, c.pm / 'bank', c.pm / 'snap', c.pm / 't3')}
    retained_ids = {str(path): file_identity(path) for path in (
        c.src, c.cache, c.base / 'venv', c.native, c.ckpt, c.pm / 'bank', c.pm / 'snap', c.pm / 't3',
        *(c.src / name for name in SOURCES if name not in changed))}
    archived_logs = {str(path): str(archive / 'logs' / path.name) for path in old_logs}
    parent_hashes = {path.name: digest(path) for path in receipts}
    refresh_record = {
        'run_id': c.run, 'archive': str(archive), 'changed_sources': sorted(changed),
        'parent_install_id': old.install_id, 'parent_config_hash': old.config_hash,
        'parent_model_id': model['model_id'], 'parent_source_hashes': sealed['source_hashes'],
        'parent_install_receipt_sha256': parent_hashes[sealed_path.name],
        'parent_owner_sha256': parent_hashes[owner_path.name],
        'parent_model_receipt_sha256': parent_hashes[model_path.name],
        'advance_marker_sha256': parent_hashes[advance_path.name],
        'revision_marker_sha256': parent_hashes[revision_path.name],
        'archived_logs': archived_logs, 'retired_qualifications': str(retired_qualifications),
        'retired_images': {'original': str(c.cache), 'archive': str(retired_images),
                          'manifests': {role: digest(c.cache / role / 'manifest.json') for role in sorted(roles)}},
        'retired_extra_roots': {name: value for name, value in audit['authorized_roots'].items()
                                if name not in (str(c.base), str(c.pm))},
        'retained_identities': retained_ids, 'seed_receipts': historical['seed_receipts'],
        'archive_files': {},
    }
    new_id = hashlib.sha256(encoded({'sources': new_hashes, 'native': sealed['native_bundle_sha256']})).hexdigest()
    result = {'phase': 'rewind-refresh', 'run_id': c.run, 'host_rank': c.rank,
              'success': False, 'state': 'validated', 'leases_must_stay_held': True,
              'new_source_hashes': new_hashes, 'new_install_id': new_id, 'new_config_hash': c.config_hash,
              'refresh': refresh_record, 'archive': str(archive), 'parent_install_id': old.install_id,
              'parent_config_hash': old.config_hash, 'withdrawn_install_receipt': str(withdrawn),
              'retirement_audit': audit, 'qualification_required': True, 'prepare_required': True}
    require(old.check_install() == sealed and all(digest(path) == parent_hashes[path.name] for path in receipts),
            'old seal/receipts changed during refresh validation')
    # FIRST persistent mutation. This withdrawal stays forever, even if the
    # marker has not yet been written when cancellation/process death occurs.
    sealed_path.rename(withdrawn)
    try:
        stable_write(marker_path, encoded(result))
        private(archive, create=True)
        def preserve(path, target):
            require(path.is_file() and not path.is_symlink(), f'unsafe archive file: {path}')
            before = digest(path)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open('xb') as output:
                with path.open('rb') as input_file:
                    shutil.copyfileobj(input_file, output)
            require(digest(target) == before == digest(path), f'archive source changed: {path}')
            refresh_record['archive_files'][str(target)] = before
        preserve(withdrawn, archive / sealed_path.name)
        for path in receipts[1:]:
            preserve(path, archive / path.name)
        preserve(c.pm / 'seed-receipt.json', archive / 'seed-receipt.json')
        install_log = c.base / 'rewind-install.log'
        if os.path.lexists(install_log):
            preserve(install_log, archive / install_log.name)
        for name, sha in parent_hashes.items():
            require(refresh_record['archive_files'][str(archive / name)] == sha,
                    f'archived parent receipt changed: {name}')
        for name, entry in sources.items():
            if stat.S_ISREG(entry['mode']):
                preserve(c.src / name, archive / 'rewind-src' / name)
                require(refresh_record['archive_files'][str(archive / 'rewind-src' / name)] == entry['sha256'],
                        f'archived source changed: {name}')
        private(archive / 'logs', create=True)
        for directory in old_logs:
            destination = Path(archived_logs[str(directory)])
            directory.rename(destination)
            require(revision_tree(destination) == log_trees[directory.name], 'archived log bytes/inodes changed')
            for name, entry in log_trees[directory.name].items():
                if 'sha256' in entry:
                    refresh_record['archive_files'][str(destination / name)] = entry['sha256']
        (c.pm / 'qualifications').rename(retired_qualifications)
        private(c.pm / 'qualifications', create=True)
        # Keep the cache root inode; move whole production children, including
        # manifests/build/help/native receipts. Never patch or relabel an image.
        private(retired_images, create=True)
        for role in sorted(roles):
            (c.cache / role).rename(retired_images / role)
        qualified_path.unlink()
        for name, value in (('retained-trees.json', retained_trees), ('qualification-tree.json', qualifications),
                            ('image-trees.json', image_trees), ('log-trees.json', log_trees)):
            evidence = archive / name
            stable_write(evidence, encoded(value))
            refresh_record['archive_files'][str(evidence)] = digest(evidence)
        result['state'] = 'archived'
        dump(marker_path, result)
        for name in sorted(changed):
            target = c.src / name
            pending = target.with_name(target.name + '.refresh-pending')
            stable_write(pending, payload[name], stat.S_IMODE(target.stat().st_mode))
            pending.replace(target)
        for name, tree in retained_trees.items():
            require(revision_tree(Path(name)) == tree, f'retained SDK/native/model/seed tree changed: {name}')
        for name, expected_identity in retained_ids.items():
            require(file_identity(Path(name)) == expected_identity, f'retained inode changed: {name}')
        after_sources = revision_tree(c.src)
        require(set(after_sources) == set(sources)
                and all(after_sources[name] == entry for name, entry in sources.items() if name not in changed),
                'unchanged source bytes/inodes changed')
        require(all(digest(c.src / name) == sha for name, sha in new_hashes.items()), 'new source bytes differ')
        require(not any(c.cache.iterdir()) and not any((c.pm / 'qualifications').iterdir())
                and not os.path.lexists(qualified_path), 'refresh must require new qualification and images')
        require(all(digest(Path(name)) == sha for name, sha in historical['seed_receipts'].items()),
                'seed receipts changed')
        require(digest(advance_path) == refresh_record['advance_marker_sha256']
                and digest(revision_path) == refresh_record['revision_marker_sha256'],
                'historical transition markers changed')
        # Re-audit all preserved ownership at its real archived paths, including
        # successful qualification, API help and failed serving's exact job.
        retired_archive_checks(c, historical, 'advance')
        retired_archive_checks(c, revision['revision'], 'revision')
        retired_archive_checks(c, refresh_record, 'refresh')
        files = dict(sealed['files'])
        files.update({str(c.src / name): new_hashes[name] for name in changed})
        receipt = dict(sealed, source_hashes=new_hashes, files=files,
                       installed_at=time.time(), source_refresh=refresh_record)
        owner = dict(old_owner, source_hashes=new_hashes, source_refresh=refresh_record)
        dump(owner_path, owner)
        dump(model_path, refreshed_model(model, result))
        # Only now publish a new seal; the incomplete marker still blocks every
        # dispatcher until the final commit write. No except path restores it.
        dump(sealed_path, receipt)
        result.update(state='complete', success=True, leases_must_stay_held=False,
                      new_install_receipt_sha256=digest(sealed_path), new_owner_sha256=digest(owner_path),
                      new_model_receipt_sha256=digest(model_path))
        private(c.log, create=True)
        dump(c.log / 'rewind_measurement.json', result)
        dump(marker_path, result)
        return result
    except BaseException as exc:
        result.update(success=False, state='failed', leases_must_stay_held=True,
                      error=f'{type(exc).__name__}: {exc}')
        dump(marker_path, result)
        raise


def revise(c):
    """One-shot, exact-config two-helper revision of an unqualified CPU advance."""
    marker_path = c.base / 'rewind-revise.json'
    require(not os.path.lexists(marker_path)
            and not any(c.base.glob('rewind-revision-parent-install-*.json'))
            and not any(c.base.glob('rewind-revision-archive-*')),
            'source revision is one-shot; prior/missing marker requires manual recovery')
    require(c.arm != 'seed' and os.environ.get('TTL_REWIND_RESET_UNQUALIFIED', '0') == '0',
            'source revision forbids reseeding and reset')
    require(getattr(c, 'phase_lock_fd', None) is not None, 'source revision requires the existing phase lock')
    require(not os.path.lexists(c.log) and not os.path.lexists(c.generation),
            'source revision requires a fresh TTL_RUN')
    source = Path(os.environ.get('TTL_REWIND_SOURCE', str(Path(__file__).resolve().parent))).resolve()
    require(not source.is_relative_to(c.base) and not source.is_relative_to(c.pm),
            'incoming source must be outside owned roots')
    for name in (*SOURCES, 'pm_rewind_native.tar.gz'):
        require((source / name).is_file() and not (source / name).is_symlink(),
                f'missing/unsafe incoming revision payload: {name}')
    payload = {name: (source / name).read_bytes() for name in SOURCES}
    require(payload['pm_rewind_tpu.py'] == Path(__file__).read_bytes(),
            'revision must execute its incoming helper')
    sealed_path = c.base / 'rewind-install.json'
    owner_path = c.base / 'rewind-owner.json'
    model_path = c.pm / 'model-receipt.json'
    advance_path = c.base / 'rewind-advance.json'
    for path in (sealed_path, owner_path, model_path, advance_path):
        require(path.is_file() and not path.is_symlink(), f'missing/unsafe old receipt: {path}')
    sealed, old_owner, model = load(sealed_path), load(owner_path), load(model_path)
    helper = c.src / 'pm_rewind_tpu.py'
    require(helper.is_file() and not helper.is_symlink(), 'old sealed helper missing')
    helper_bytes = helper.read_bytes()
    require(hashlib.sha256(helper_bytes).hexdigest() == sealed['files'].get(str(helper))
            == sealed['source_hashes'].get(helper.name) == old_owner['source_hashes'].get(helper.name),
            'old helper must match both seal and owner before execution')
    expected = {
        'INSTALL_ID': hashlib.sha256(encoded({'sources': sealed['source_hashes'],
                         'native': sealed['native_bundle_sha256']})).hexdigest(),
        'CONFIG_HASH': sealed['config_hash'], 'MODEL_ID': model['model_id'],
        'SOURCE_SHA256': hashlib.sha256(encoded(sealed['source_hashes'])).hexdigest(),
    }
    for key, value in expected.items():
        require(os.environ.get('TTL_REWIND_EXPECT_OLD_' + key) == value,
                f'explicit matching TTL_REWIND_EXPECT_OLD_{key} required')
    spec = importlib.util.spec_from_file_location('_pm_rewind_revision_old', helper)
    old_module = importlib.util.module_from_spec(spec)
    exec(compile(helper_bytes, str(helper), 'exec'), old_module.__dict__)
    old = old_module.Config()
    old.phase_lock_fd = c.phase_lock_fd
    require(old.check_install() == sealed and old.install_id == expected['INSTALL_ID']
            and old.spec == c.spec == sealed['config'] == old_owner['config']
            and old.config_hash == c.config_hash == expected['CONFIG_HASH']
            and old_owner['source_hashes'] == sealed['source_hashes']
            and old_owner['bundle_sha256'] == sealed['native_bundle_sha256']
            == digest(source / 'pm_rewind_native.tar.gz'),
            'revision requires the entire exact config and unchanged native bundle')
    historical = old_module.advance_lineage(old)
    require(historical and historical.get('phase') == 'rewind-advance'
            and historical.get('state') == 'complete' and historical.get('success') is True
            and historical['new_install_id'] == old.install_id
            and not sealed.get('source_revision') and not old_owner.get('source_revision')
            and not model.get('source_revision'), 'completed unrevised CPU advance required')
    require(c.run != historical['run_id']
            and all(Path(path).name != c.run for path in historical['archived_logs']),
            'revision requires a run ID never used by the historical install')
    require(old_module.require_model_receipt(old) == model, 'old retained model receipt differs')
    require(set(payload) == set(sealed['source_hashes']), 'revision source inventory changed')
    new_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    changed = {name for name in payload if new_hashes[name] != sealed['source_hashes'][name]}
    require(changed == {'pm_rewind_tpu.py', 'pm_ray_zygote.py'}
            and new_hashes['pm_ray_zygote.py'] == REVISION_RUNTIME_SHA256,
            'revision requires both helper changes and the reviewed frozen CPU runtime')
    require(not any(os.path.lexists(c.src / (name + '.revision-pending')) for name in changed),
            'unknown pending source revision payload')
    for name, sha in sealed['source_hashes'].items():
        require(sealed['files'].get(str(c.src / name)) == sha == digest(c.src / name),
                f'old source identity differs from seal: {name}')
    require(all(any(Path(name).is_relative_to(root) for root in (c.src, c.native, c.base / 'venv'))
                for name in sealed['files']), 'old seal escaped owned runtime/source roots')
    private(c.cache)
    require(not any(c.cache.iterdir()), 'revision refuses any production image cache contents')
    qualified_path = c.base / 'rewind-qualified.json'
    require(not os.path.lexists(qualified_path), 'revision refuses published qualification or unknown qualification marker')
    old_logs = sorted((c.base / 'logs').iterdir())
    for directory in old_logs:
        require(directory.is_dir() and not directory.is_symlink(), f'unsafe old log: {directory}')
        require(not any(directory.glob('prepare-*')) and not any(directory.glob('verify-*'))
                and not any(directory.glob('*api-help*'))
                and not any((directory / name).exists() for name in (
                    'rewind-run-owner.json', 'rewind-preflight-result.json', 'rewind-copy-owner.json')),
                'revision refuses production prepare, serving, or a new model copy')
        for name in ('rewind_measurement.json', 'rewind-phase-error.json'):
            path = directory / name
            if path.exists():
                require(load(path).get('phase') in ('rewind-advance', 'rewind-qualify', 'rewind-collect'),
                        f'unknown or production phase in post-advance evidence: {path}')
        if directory.name == historical['run_id']:
            require(load(directory / 'rewind_measurement.json') == historical,
                    'completed advance measurement differs from its immutable marker')
        else:
            require((directory / 'rewind-qualification-owner.json').is_file(),
                    f'unknown post-advance log generation: {directory}')
            for name in ('rewind-qualification-result.json', 'rewind-qualification.json',
                         'rewind_measurement.json'):
                require(load(directory / name).get('success') is False,
                        f'revision refuses successful/unknown current qualification: {directory}/{name}')
    # Reuse the SEALED helper's complete read-only ownership audit, including its
    # original archived seed, exact Ray jobs/PGs, supervisors and private temp roots.
    audit, _ = old_module.retirement_checks(old, read_only=True)
    archive = c.base / ('rewind-revision-archive-' + c.run)
    retired_qualifications = c.pm / ('rewind-revision-qualifications-' + c.run)
    withdrawn = c.base / ('rewind-revision-parent-install-' + c.run + '.json')
    require(not os.path.lexists(retired_qualifications), 'revision qualification archive already exists')
    archived_logs = {str(path): str(archive / 'logs' / path.name) for path in old_logs}
    log_trees = {path.name: revision_tree(path) for path in old_logs}
    require(all(stat.S_ISREG(entry['mode']) or stat.S_ISDIR(entry['mode'])
                for tree in log_trees.values() for entry in tree.values()), 'unsafe old log entries')
    qualifications = revision_tree(c.pm / 'qualifications')
    sources = revision_tree(c.src)
    require(all(stat.S_ISREG(entry['mode']) or stat.S_ISDIR(entry['mode'])
                for entry in sources.values()), 'unsafe installed source entries')
    # Hash full retained SDK/native/model/seed state, not just Python seal entries.
    retained_trees = {str(path): revision_tree(path) for path in (
        c.base / 'venv', c.native, c.ckpt, c.pm / 'bank', c.pm / 'snap', c.pm / 't3')}
    retained = [c.src, c.base / 'venv', c.native, c.ckpt, *sorted(c.ckpt.iterdir()),
                *(Path(name) for name in sealed['files'] if Path(name) not in {c.src / n for n in changed})]
    retained_ids = {str(path): file_identity(path) for path in retained}
    seed_hashes = historical['seed_receipts']
    parent_hashes = {path.name: digest(path) for path in (sealed_path, owner_path, model_path, advance_path)}
    revision = {
        'run_id': c.run, 'archive': str(archive), 'changed_sources': sorted(changed),
        'parent_install_id': old.install_id, 'parent_config_hash': old.config_hash,
        'parent_model_id': model['model_id'], 'parent_source_hashes': sealed['source_hashes'],
        'parent_install_receipt_sha256': parent_hashes[sealed_path.name],
        'parent_model_receipt_sha256': parent_hashes[model_path.name],
        'parent_owner_sha256': parent_hashes[owner_path.name],
        'advance_marker_sha256': parent_hashes[advance_path.name],
        'archived_logs': archived_logs, 'retired_qualifications': str(retired_qualifications),
        'retired_extra_roots': {name: value for name, value in audit['authorized_roots'].items()
                                if name not in (str(c.base), str(c.pm))},
        'retained_identities': retained_ids, 'seed_receipts': seed_hashes,
        'archive_files': {},
    }
    new_id = hashlib.sha256(encoded({'sources': new_hashes, 'native': sealed['native_bundle_sha256']})).hexdigest()
    result = {'phase': 'rewind-revise', 'run_id': c.run, 'host_rank': c.rank,
              'success': False, 'state': 'validated', 'leases_must_stay_held': True,
              'new_source_hashes': new_hashes, 'new_install_id': new_id,
              'new_config_hash': c.config_hash, 'revision': revision,
              'archive': str(archive), 'parent_install_id': old.install_id,
              'parent_config_hash': old.config_hash, 'withdrawn_install_receipt': str(withdrawn),
              'retirement_audit': audit, 'qualification_required': True, 'prepare_required': True}
    require(old.check_install() == sealed and all(digest(path) == parent_hashes[path.name]
            for path in (sealed_path, owner_path, model_path, advance_path)),
            'old seal/receipts changed during revision validation')
    # FIRST persistent mutation, before even a log directory or marker is made.
    # Keep this withdrawn receipt permanently: losing the marker cannot retry.
    sealed_path.rename(withdrawn)
    try:
        stable_write(marker_path, encoded(result))
        private(archive, create=True)
        def preserve(path, target):
            require(path.is_file() and not path.is_symlink(), f'unsafe archive file: {path}')
            before = digest(path)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open('xb') as output:
                with path.open('rb') as input_file:
                    shutil.copyfileobj(input_file, output)
            require(digest(target) == before == digest(path), f'archive source changed: {path}')
            revision['archive_files'][str(target)] = before
        preserve(withdrawn, archive / sealed_path.name)
        for path in (owner_path, model_path, advance_path):
            preserve(path, archive / path.name)
        for name, sha in parent_hashes.items():
            require(revision['archive_files'][str(archive / name)] == sha,
                    f'archived parent receipt changed during withdrawal: {name}')
        for name, entry in sources.items():
            if stat.S_ISREG(entry['mode']):
                preserve(c.src / name, archive / 'rewind-src' / name)
                require(revision['archive_files'][str(archive / 'rewind-src' / name)] == entry['sha256'],
                        f'archived old source bytes changed: {name}')
        private(archive / 'logs', create=True)
        for directory in old_logs:
            destination = Path(archived_logs[str(directory)])
            directory.rename(destination)
            require(revision_tree(destination) == log_trees[directory.name], 'archived logs changed')
            for name, entry in log_trees[directory.name].items():
                if 'sha256' in entry:
                    revision['archive_files'][str(destination / name)] = entry['sha256']
        (c.pm / 'qualifications').rename(retired_qualifications)
        require(revision_tree(retired_qualifications) == qualifications, 'qualification evidence changed')
        private(c.pm / 'qualifications', create=True)
        private(c.log, create=True)
        evidence = archive / 'retained-trees.json'
        stable_write(evidence, encoded(retained_trees))
        revision['archive_files'][str(evidence)] = digest(evidence)
        evidence = archive / 'qualification-tree.json'
        stable_write(evidence, encoded(qualifications))
        revision['archive_files'][str(evidence)] = digest(evidence)
        result['state'] = 'archived'
        dump(marker_path, result)
        for name in sorted(changed):
            target = c.src / name
            pending = target.with_name(target.name + '.revision-pending')
            require(not os.path.lexists(pending), f'unknown pending source: {pending}')
            stable_write(pending, payload[name], stat.S_IMODE(target.stat().st_mode))
            pending.replace(target)
        for name, tree in retained_trees.items():
            require(revision_tree(Path(name)) == tree, f'retained runtime/model/seed tree changed: {name}')
        for name, expected_identity in retained_ids.items():
            require(file_identity(Path(name)) == expected_identity, f'retained inode changed: {name}')
        after_sources = revision_tree(c.src)
        require(set(after_sources) == set(sources)
                and all(after_sources[name] == entry for name, entry in sources.items() if name not in changed),
                'unchanged source bytes/inodes changed')
        require(all(digest(c.src / name) == sha for name, sha in new_hashes.items()), 'new source bytes differ')
        require(all(digest(Path(name)) == sha for name, sha in seed_hashes.items()), 'seed receipts changed')
        require(digest(advance_path) == revision['advance_marker_sha256'], 'historical advance marker changed')
        require(not any(c.cache.iterdir()), 'production image cache changed during revision')
        require(revision_tree(retired_qualifications) == qualifications, 'retired qualification evidence changed')
        for name, sha in revision['archive_files'].items():
            require(digest(Path(name)) == sha, f'revision archive changed: {name}')
        files = dict(sealed['files'])
        files.update({str(c.src / name): new_hashes[name] for name in changed})
        receipt = dict(sealed, source_hashes=new_hashes, files=files,
                       installed_at=time.time(), source_revision=revision)
        owner = dict(old_owner, source_hashes=new_hashes, source_revision=revision)
        dump(owner_path, owner)
        dump(sealed_path, receipt)
        dump(model_path, revised_model(model, result))
        result.update(state='complete', success=True, leases_must_stay_held=False,
                      new_install_receipt_sha256=digest(sealed_path), new_owner_sha256=digest(owner_path),
                      new_model_receipt_sha256=digest(model_path))
        dump(c.log / 'rewind_measurement.json', result)
        # The complete marker is the last persistent write. No recovery/repair path.
        dump(marker_path, result)
        return result
    except BaseException as exc:
        result.update(success=False, state='failed', leases_must_stay_held=True,
                      error=f'{type(exc).__name__}: {exc}')
        dump(marker_path, result)
        raise


def advance(c):
    """One-time CPU-source transition; no model copy, native install, or reseed."""
    marker_path = c.base / 'rewind-advance.json'
    require(not os.path.lexists(marker_path), 'rewind-advance is one-time; marker already exists')
    require(c.arm != 'seed' and os.environ.get('TTL_REWIND_RESET_UNQUALIFIED', '0') == '0',
            'advance forbids reseeding and reset')
    require(getattr(c, 'phase_lock_fd', None) is not None, 'advance requires the phase lock')
    require(not os.path.lexists(c.log) and not os.path.lexists(c.generation),
            'advance requires a fresh TTL_RUN')
    source = Path(os.environ.get('TTL_REWIND_SOURCE', str(Path(__file__).resolve().parent))).resolve()
    require(not source.is_relative_to(c.base) and not source.is_relative_to(c.pm),
            'incoming sources must be outside the old owned roots')
    payload = {name: (source / name).read_bytes() for name in SOURCES}
    require(payload['pm_rewind_tpu.py'] == Path(__file__).read_bytes(),
            'advance must execute its incoming helper, not a different source payload')
    sealed_path = c.base / 'rewind-install.json'
    sealed = load(sealed_path)
    old_owner = load(c.base / 'rewind-owner.json')
    helper = c.src / 'pm_rewind_tpu.py'
    require(helper.is_file() and not helper.is_symlink(), 'old sealed helper missing')
    helper_bytes = helper.read_bytes()
    require(hashlib.sha256(helper_bytes).hexdigest() == sealed['files'].get(str(helper))
            == sealed['source_hashes'].get(helper.name) == old_owner['source_hashes'].get(helper.name),
            'old helper is not sealed')
    expected = {
        'INSTALL_ID': hashlib.sha256(encoded({'sources': sealed['source_hashes'],
                         'native': sealed['native_bundle_sha256']})).hexdigest(),
        'CONFIG_HASH': sealed['config_hash'],
        'SOURCE_SHA256': hashlib.sha256(encoded(sealed['source_hashes'])).hexdigest(),
    }
    model = load(c.pm / 'model-receipt.json')
    expected['MODEL_ID'] = model['model_id']
    for key, value in expected.items():
        require(os.environ.get('TTL_REWIND_EXPECT_OLD_' + key) == value,
                f'explicit matching TTL_REWIND_EXPECT_OLD_{key} required')
    spec = importlib.util.spec_from_file_location('_pm_rewind_advance_old', helper)
    old_module = importlib.util.module_from_spec(spec)
    exec(compile(helper_bytes, str(helper), 'exec'), old_module.__dict__)
    old = old_module.Config()
    old.phase_lock_fd = c.phase_lock_fd
    require(old.check_install() == sealed and old.install_id == expected['INSTALL_ID']
            and old.config_hash == expected['CONFIG_HASH']
            and old_owner['source_hashes'] == sealed['source_hashes']
            and old_owner['bundle_sha256'] == sealed['native_bundle_sha256'],
            'old config/source/native seal mismatch')
    # Normalize only the exact CPU-boundary migration back to the old spec.
    # The saved API argv must encode that same old argv, never new serve options.
    compatible = json.loads(encoded(c.spec))
    cpu_additions = (
        ('GCONV_PATH', '/usr/lib/x86_64-linux-gnu/gconv'),
        ('RAY_LD_PRELOAD_ON_WORKERS', '0'),
        ('RAY_CLIENT_MODE', '0'),
        ('PM_SB_PRESEED', '1'),
        ('TOKENIZERS_PARALLELISM', 'false'),
        ('PM_REWIND_SERVE_ARGV', json.dumps(old.spec['argv'], separators=(',', ':'))),
    )
    for key, value in cpu_additions:
        require(compatible['env'].get(key) == value, f'advance requires exact frozen {key}={value}')
        if key not in old.spec['env']:
            compatible['env'].pop(key)
    require(compatible['env'].get('PM_WARM') == '0', 'advance requires frozen PM_WARM=0')
    if old.spec['env'].get('PM_WARM') == '1':
        compatible['env']['PM_WARM'] = '1'
    require(old.spec == compatible,
            'advance permits only the explicit CPU additions and PM_WARM=1->0; all other config must match')
    require(set(payload) == set(sealed['source_hashes']), 'source inventory changed')
    for name, sha in sealed['source_hashes'].items():
        require(sealed['files'].get(str(c.src / name)) == sha == digest(c.src / name),
                f'old source identity differs from its seal: {name}')
    require(all(any(Path(name).is_relative_to(root) for root in (c.src, c.native, c.base / 'venv'))
                for name in sealed['files']), 'old seal contains paths outside owned source/runtime roots')
    new_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    require(new_hashes != sealed['source_hashes'], 'advance requires genuinely new source hashes')
    require_model_receipt(old)
    seed_path = c.pm / 'seed-receipt.json'
    seed = load(seed_path)
    require(seed.get('success') and seed.get('config_hash') == old.config_hash
            and seed.get('model_id') == model['model_id'], 'successful old same-config/model seed required')
    seed_hashes = {str(seed_path): digest(seed_path)}
    # No files or logs are written by this audit, including exact-job/PG queries.
    audit, _ = retirement_checks(old, read_only=True)
    seed_run = seed.get('run_id')
    require(seed_run in audit['completed_runs'], 'seed lacks closed generation ownership')
    seed_result = load(c.base / 'logs' / seed_run / 'rewind_measurement.json')
    require(seed_result.get('success') and seed_result.get('arm') == 'seed'
            and seed_result.get('config_hash') == old.config_hash,
            'old seed receipt lacks a successful sealed-config seed run')
    site = Path(subprocess.check_output([str(c.python), '-B', '-c',
                'import sysconfig;print(sysconfig.get_paths()["purelib"])'], text=True).strip())
    require(site.resolve() == site and site.is_relative_to(c.base / 'venv'),
            'patch target is not the dedicated private venv')
    retained = [c.base / 'venv', c.native, c.ckpt,
                *sorted(p for p in c.native.rglob('*') if not p.is_symlink()),
                *sorted(c.ckpt.iterdir()),
                *(Path(name) for name in sealed['files'] if Path(name).is_relative_to(c.base / 'venv'))]
    retained_ids = {str(path): file_identity(path) for path in retained}
    archive = c.base / ('rewind-advance-archive-' + c.run)
    retired_images = c.pm / ('rewind-retired-images-' + c.run)
    retired_qualifications = c.pm / ('rewind-retired-qualifications-' + c.run)
    require(all(not os.path.lexists(path) for path in (archive, retired_images, retired_qualifications)),
            'unknown transition archive already exists')
    old_logs = sorted((c.base / 'logs').iterdir())
    archived_logs = {}
    log_hashes = {}
    for directory in old_logs:
        require(directory.is_dir() and not directory.is_symlink(), f'unsafe old log: {directory}')
        destination = archive / 'logs' / directory.name
        archived_logs[str(directory)] = str(destination)
        for path in directory.rglob('*'):
            require(not path.is_symlink() and (path.is_file() or path.is_dir()),
                    f'unsafe old log entry: {path}')
            if path.is_file():
                log_hashes[str(destination / path.relative_to(directory))] = digest(path)
    image_files = {}
    image_links = {}
    for path in sorted(c.cache.rglob('*')):
        if path.is_symlink():
            relative = path.relative_to(c.cache)
            target = os.readlink(path)
            require(len(relative.parts) == 3 and relative.parts[0] in ('workers', 'api')
                    and relative.parts[1:] == ('build', 'dmtcp_restart_script.sh')
                    and Path(target).name == target and target.startswith('dmtcp_restart_script_')
                    and target.endswith('.sh') and (path.parent / target).is_file()
                    and not (path.parent / target).is_symlink(),
                    f'unsafe old DMTCP restart alias: {path} -> {target}')
            image_links[str(relative)] = target
            continue
        require(not path.is_symlink() and (path.is_dir() or path.is_file()),
                f'unknown old image evidence: {path}')
        if path.is_file():
            image_files[str(path.relative_to(c.cache))] = digest(path)
    require(image_files and any(name.endswith('manifest.json') for name in image_files),
            'old published image evidence required')
    lineage = {'parent_install_id': old.install_id, 'parent_config_hash': old.config_hash,
               'parent_model_id': model['model_id'],
               'parent_model_receipt_sha256': digest(c.pm / 'model-receipt.json'),
               'parent_install_receipt_sha256': digest(sealed_path),
               'parent_source_hashes': sealed['source_hashes'],
               'archived_logs': archived_logs, 'retired_qualifications': str(retired_qualifications),
               'retired_extra_roots': {name: value for name, value in audit['authorized_roots'].items()
                                       if name not in (str(c.base), str(c.pm))},
               'seed_receipts': seed_hashes, 'retained_identities': retained_ids,
               'archive': str(archive), 'retired_images': {
                   'original': str(c.cache), 'archive': str(retired_images),
                   'files': image_files, 'symlinks': image_links}}
    new_id = hashlib.sha256(encoded({'sources': new_hashes,
                           'native': sealed['native_bundle_sha256']})).hexdigest()
    result = {'phase': 'rewind-advance', 'run_id': c.run, 'host_rank': c.rank,
              'success': False, 'state': 'validated', 'leases_must_stay_held': True,
              'new_source_hashes': new_hashes, 'new_install_id': new_id,
              'new_config_hash': c.config_hash, 'lineage': lineage, **lineage,
              'retirement_audit': audit, 'archive_files': dict(log_hashes)}
    withdrawn = c.base / ('rewind-advance-parent-install-' + c.run + '.json')
    require(not any(c.base.glob('rewind-advance-parent-install-*.json')),
            'prior seal withdrawal requires manual recovery, never automatic retry')
    result['withdrawn_install_receipt'] = str(withdrawn)
    # Recheck the old seal immediately before the FIRST persistent mutation.
    require(old.check_install() == sealed, 'old seal changed during validation')
    # Withdrawing the seal is the first atomic mutation. Even an old helper
    # without marker support must fail while the transition is being installed.
    try:
        sealed_path.rename(withdrawn)
        stable_write(marker_path, encoded(result))
        private(archive, create=True)
        private(c.log, create=True)
        def preserve(path, target):
            require(path.is_file() and not path.is_symlink(), f'unsafe archive file: {path}')
            before = digest(path)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open('xb') as output:
                with path.open('rb') as input_file:
                    shutil.copyfileobj(input_file, output)
            require(digest(target) == before == digest(path), f'archive source changed: {path}')
            result['archive_files'][str(target)] = before
        withdrawn.rename(archive / 'rewind-install.json')
        result['archive_files'][str(archive / 'rewind-install.json')] = lineage['parent_install_receipt_sha256']
        for name in ('rewind-owner.json', 'rewind-qualified.json', 'rewind-install.log'):
            path = c.base / name
            if path.exists():
                preserve(path, archive / name)
        preserve(c.pm / 'model-receipt.json', archive / 'model-receipt.json')
        preserve(seed_path, archive / 'seed-receipt.json')
        for name in sealed['files']:
            path = Path(name)
            if path.is_relative_to(c.base / 'venv') and path.suffix == '.py':
                preserve(path, archive / 'venv-sources' / path.relative_to(c.base / 'venv'))
        c.src.rename(archive / 'rewind-src')
        for name, sha in sealed['files'].items():
            path = Path(name)
            if path.is_relative_to(c.src):
                result['archive_files'][str(archive / 'rewind-src' / path.relative_to(c.src))] = sha
        private(archive / 'logs', create=True)
        for directory in old_logs:
            directory.rename(Path(archived_logs[str(directory)]))
        (c.pm / 'qualifications').rename(retired_qualifications)
        private(c.pm / 'qualifications', create=True)
        c.cache.rename(retired_images)
        for name, target in image_links.items():
            path = retired_images / name
            require(path.is_symlink() and os.readlink(path) == target,
                    f'archived DMTCP restart alias changed: {path}')
        private(c.src, create=True)
        private(c.cache, create=True)
        qualified = c.base / 'rewind-qualified.json'
        if qualified.exists():
            qualified.unlink()
        result['state'] = 'archived'
        dump(marker_path, result)
        for name, data in payload.items():
            stable_write(c.src / name, data)
        stable_write(c.src / 'fresh-entry', (archive / 'rewind-src/fresh-entry').read_bytes(), 0o700)
        for filename, module in (('torchtpu_load_plugin.py', 'ttl_probe'),
                                 ('torchtpu_ray_pin_plugin.py', 'ttl_ray_pin'),
                                 ('pm_torchtpu_plugin.py', 'pm_torchtpu')):
            target = site / (module + '.py')
            require(str(target) in sealed['files'] and not target.is_symlink(), 'unsealed plugin target')
            target.write_bytes(payload[filename])
        result['state'] = 'patching'
        dump(marker_path, result)
        patch_env = dict(c.spec['env'], TORCH_DEVICE_BACKEND_AUTOLOAD='0')
        checked([c.python, '-B', c.src / 'pm_source_patches.py', 'apply', site],
                c.log / 'advance-patches.log', env=patch_env)
        for name, expected_identity in retained_ids.items():
            require(file_identity(Path(name)) == expected_identity, f'retained inode changed: {name}')
        require(model_files(c.ckpt) == model['files'], 'model bytes changed during advance')
        require(all(digest(Path(name)) == expected_hash for name, expected_hash in seed_hashes.items()),
                'advance must never rewrite seed receipts')
        require(all(digest(Path(name)) == sha for name, sha in sealed['files'].items()
                    if Path(name).is_relative_to(c.native)), 'native bytes changed during advance')
        for name, sha in result['archive_files'].items():
            require(digest(Path(name)) == sha, f'archive changed during transition: {name}')
        lineage['archive_files'] = dict(result['archive_files'])
        result.update(lineage)
        owner = dict(old_owner, config=c.spec, source_hashes=new_hashes)
        installed_files = set(sealed['files']) | {str(path) for package in ('vllm', 'vllm_torchtpu', 'torch_tpu')
                                                for path in (site / package).rglob('*.py')}
        receipt = dict(sealed, config_hash=c.config_hash, config=c.spec, source_hashes=new_hashes,
                       files={name: digest(Path(name)) for name in sorted(installed_files)},
                       installed_at=time.time(), lineage=lineage)
        new_model = dict(model, config_hash=c.config_hash, install_id=new_id,
                         parent_install_id=old.install_id, parent_config_hash=old.config_hash,
                         parent_model_receipt_sha256=lineage['parent_model_receipt_sha256'],
                         advance_run_id=c.run)
        dump(c.base / 'rewind-owner.json', owner)
        dump(sealed_path, receipt)
        dump(c.pm / 'model-receipt.json', new_model)
        result.update(state='complete', success=True, leases_must_stay_held=False,
                      new_model_receipt_sha256=digest(c.pm / 'model-receipt.json'),
                      new_install_receipt_sha256=digest(sealed_path),
                      new_file_hashes=receipt['files'], qualification_required=True, prepare_required=True)
        # Publish measurement before the commit marker. A crash at either write
        # leaves the marker incomplete, so no ordinary phase can mask failure.
        dump(c.log / 'rewind_measurement.json', result)
        dump(marker_path, result)
        return result
    except BaseException as exc:
        result.update(success=False, state='failed', leases_must_stay_held=True,
                      error=f'{type(exc).__name__}: {exc}')
        dump(marker_path, result)
        raise


def api_help_qualification(c):
    """Restart the real API image through CLI parsing and its fresh CPU helper."""
    cache = c.cache / 'api'
    run_dir = cache / ('help-' + c.run)
    require(not run_dir.exists() and not run_dir.is_symlink(), 'API help qualification needs a new run ID')
    helper_socket = c.log / 'ah.sock'
    require(len(os.fsencode(helper_socket)) < 104 and not os.path.lexists(helper_socket),
            'fresh short API help socket required')
    late = contexts(c)['api']
    late['PM_ZYGOTE_SOCK'] = str(helper_socket)
    command = [str(c.python), str(c.src / 'pm_rewind.py'), 'run', str(cache),
               '--run-dir', str(run_dir), '--timeout', '180']
    for key, value in sorted(late.items()):
        command += ['--env', f'{key}={value}']
    command += ['--', 'serve', '--help']
    owner_path = c.log / 'rewind-api-help-owner.json'
    owner = {'run_id': c.run, 'host_rank': c.rank, 'config_hash': c.config_hash,
             'owner': identity(), 'controller': None, 'run_dir': str(run_dir)}
    stable_write(owner_path, encoded(owner))
    report = {'success': False, 'cleanup_complete': False, 'run_id': c.run,
              'host_rank': c.rank, 'config_hash': c.config_hash, 'command': command,
              'late_env': late, 'manifest_sha256_before': digest(cache / 'manifest.json'),
              'scope': 'Actual native-restored API CLI --help plus fresh CPU helper startup/teardown; '
                       'no engine-child RPC qualification, no shared-Ray connection, no model load, no TPU serving evidence'}
    proc = None
    try:
        CANCELLATION['deferred'] = True
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            with (c.log / 'api-help-controller.log').open('xb') as output:
                proc = subprocess.Popen([str(c.python), str(c.src / 'pm_rewind_tpu.py'),
                    '_checked-parent-death', str(os.getpid()), *command], env=c.spec['env'], cwd=c.base,
                    stdout=output, stderr=subprocess.STDOUT)
            owner['controller'] = identity(proc.pid)
            dump(owner_path, owner)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
            CANCELLATION['deferred'] = False
            if CANCELLATION['pending']:
                CANCELLATION['pending'] = False
                raise KeyboardInterrupt('cancelled during API help controller registration')
        proc.wait(timeout=210)
        require(proc.returncode == 0, f'API image help continuation failed rc={proc.returncode}')
        text = (run_dir / 'output.log').read_text(errors='replace')
        require(re.search(r'(?im)^usage:\s*vllm serve \[model_tag\] \[options\]', text)
                and '--help=all' in text, 'actual vLLM serve usage/help text missing')
        observed = events(run_dir / 'output.log')
        restored = [event for event in observed if event.get('event') == 'PM_REWIND_RESTORED']
        require(len(restored) == 1 and restored[0].get('real_pids') is True
                and type(restored[0].get('pid')) is int and restored[0]['pid'] > 0,
                'API help lacks a single real-PID native restore receipt')
        captures = [event for event in events(cache / 'build/backend.log')
                    if event.get('event') == 'PM_REWIND_RAY_CAPTURE']
        reused = [event for event in observed if event.get('event') == 'PM_REWIND_CPU_REUSE']
        require(len(captures) == len(reused) == 1 and cpu_capture_valid(captures[0], 'api')
                and cpu_reuse_valid(captures[0], reused[0])
                and reused[0]['pid'] == restored[0]['pid'],
                'API help did not retain real prepared CPU objects through native restore')
        report['cpu_object_proof'] = {'capture': captures[0], 'reuse': reused[0]}
        report.update(cli_help=True, restored=restored[0], events=observed,
                      output_sha256=digest(run_dir / 'output.log'))
        checked([c.python, c.src / 'pm_rewind.py', 'verify', cache], c.log / 'api-help-post-verify.log')
        report['manifest_sha256_after'] = digest(cache / 'manifest.json')
        require(report['manifest_sha256_before'] == report['manifest_sha256_after'],
                'API image manifest changed during help qualification')
        report['image_verified_immutable_after_help'] = True
        report['success'] = True
    except BaseException as exc:
        report.update(success=False, error=f'{type(exc).__name__}: {exc}')
    finally:
        previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    report['cleanup_error'] = 'API help controller remains live; retain ownership anchor and leases'
            report['returncode'] = proc.poll() if proc is not None else None
            for name in ('backend.log', 'output.log'):
                if (run_dir / name).exists():
                    shutil.copyfile(run_dir / name, c.log / ('api-help-' + name))
            receipt = load(run_dir / 'supervisor.json')
            report['supervisor'] = receipt
            retired_supervisor(receipt, 'API help continuation')
            require(receipt.get('owned') and all(item.get('reaped') for item in receipt['owned'].values()),
                    'API help supervisor did not reap all recorded children')
            require(proc is not None and proc.poll() is not None and not alive(owner['controller']),
                    'API help controller still alive')
            require(not os.path.lexists(helper_socket), 'API help CPU helper socket remains after teardown')
            report['cleanup_complete'] = True
        except BaseException as exc:
            report.update(success=False, cleanup_complete=False, cleanup_error=repr(exc))
        finally:
            report['success'] = report['success'] and report['cleanup_complete']
            report['leases_must_stay_held'] = not report['cleanup_complete']
            dump(c.log / 'api-help-result.json', report)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return report


def prepare(c):
    c.check_install()
    require_qualification(c)
    require_model_receipt(c)
    private(c.log, create=True)
    results = {}
    for role in ('workers', 'api') if c.rank == 0 else ('workers',):
        cache = c.cache / role
        command = [c.python, c.src / 'pm_rewind.py', 'prepare', cache,
                   '--factory', f'pm_rewind_ray:prepare_{role}', '--python', c.python,
                   '--dmtcp-bin', c.native / 'bin', '--real-pids',
                   '--timeout', os.environ.get('TTL_REWIND_PREPARE_TIMEOUT', '600')]
        for key, value in sorted(c.spec['env'].items()):
            command += ['--env', f'{key}={value}']
        for key in LATE_API if role == 'api' else LATE_WORKER:
            command += ['--late-env', key]
        if role == 'api':
            for path in sorted(c.ckpt.iterdir()):
                if path.suffix in ('.json', '.model', '.txt', '.jinja'):
                    command += ['--input', str(path)]
        checked(command, c.log / f'prepare-{role}.log', timeout=900)
        checked([c.python, c.src / 'pm_rewind.py', 'verify', cache], c.log / f'verify-{role}.log')
        results[role] = {'manifest_sha256': digest(cache / 'manifest.json'),
                         'capture_events': events(cache / 'build/backend.log'),
                         'supervisor': load(cache / 'build/supervisor.json')}
        captures = [event for event in results[role]['capture_events']
                    if event.get('event') == 'PM_REWIND_RAY_CAPTURE']
        require(len(captures) == 1 and cpu_capture_valid(captures[0], role),
                f'{role} capture lacks its role-specific no-connection CPU prefix')
    help_result = api_help_qualification(c) if c.rank == 0 else None
    return {'success': help_result is None or help_result['success'],
            'phase': 'rewind-prepare', 'images': results, 'api_help_qualification': help_result,
            'leases_must_stay_held': bool(help_result and help_result['leases_must_stay_held']),
            'config_hash': c.config_hash, 'install_id': c.install_id, 'host_rank': c.rank}


def cpu_capture_valid(captured, role):
    from pm_ray_zygote import CPU_PRELOAD, WORKER_CPU_PRELOAD
    expected_modules = CPU_PRELOAD if role == 'api' else WORKER_CPU_PRELOAD
    modules = captured.get('saved_module_ids', {})
    objects = captured.get('saved_cpu_ids', {})
    expected_objects = ({'args', 'engine_args', 'config', 'hf_config', 'engine', 'renderer',
                         'tokenizer', 'input_processor', 'output_processor'} if role == 'api' else set())
    return (captured.get('role') == role
            and type(captured.get('pid')) is int and captured['pid'] > 0
            and captured['pid'] == captured.get('proc_pid') == captured.get('capture_pid')
            and captured.get('proxy') is None and captured.get('tpu_initialized') is False
            and captured.get('ray_imported') is (role == 'api')
            and captured.get('ray_initialized') is False
            and set(modules) == set(expected_modules)
            and all(type(value) is int and value > 0 for value in modules.values())
            and captured.get('current_module_ids') == modules
            and set(objects) == expected_objects
            and all(type(value) is int and value > 0 for value in objects.values())
            and captured.get('current_cpu_ids') == objects
            and (role != 'api' or captured.get('cpu_stage') == 'prepared')
            and isinstance(captured.get('nonce'), str)
            and captured.get('saved_native_bytes') == captured['nonce'] + '00'
            and captured.get('native_bytes') == captured['saved_native_bytes']
            and type(captured.get('saved_native_address')) is int
            and captured['saved_native_address'] > 0
            and captured.get('native_address') == captured['saved_native_address'])


def cpu_reuse_valid(captured, reused):
    return (all(reused.get(key) == captured.get(key) for key in (
                'nonce', 'role', 'capture_pid', 'saved_module_ids', 'saved_cpu_ids',
                'saved_native_address', 'saved_native_bytes'))
            and reused.get('current_module_ids') == captured.get('saved_module_ids')
            and reused.get('current_cpu_ids') == captured.get('saved_cpu_ids')
            and reused.get('native_address') == captured.get('saved_native_address')
            and reused.get('native_bytes') == captured.get('saved_native_bytes')
            and reused.get('cpu_stage') == 'prepared'
            and type(reused.get('pid')) is int and reused['pid'] > 0
            and reused['pid'] == reused.get('proc_pid'))


def events(path):
    found = []
    if not Path(path).exists():
        return found
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            line = line.strip()
            if line.startswith('{'):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if (str(event.get('event', '')).startswith('PM_REWIND')
                        or event.get('event') == 'PM_RAY_ZYGOTE_PROOF'):
                    found.append(event)
            elif line.startswith(('PM_REWIND_RAY_CAPTURE ', 'PM_REWIND_RAY_RESUME ',
                                  'PM_REWIND_RAY_CHILD ', 'PM_REWIND_CPU_REUSE ', 'PM_RAY_ZYGOTE_PROOF ',
                                  'PM_RAY_ZYGOTE_PROXY_IDENTITY ', 'PM_RAY_ZYGOTE_PROXY_EXIT ')):
                prefix, _, body = line.partition(' ')
                try:
                    found.append({'event': prefix, **json.loads(body)})
                except ValueError:
                    require(prefix != 'PM_RAY_ZYGOTE_PROXY_IDENTITY',
                            f'malformed proxy ownership receipt: {path}')
    return found


def snapshot_artifacts(c):
    artifacts = []
    for rank in range(4 * c.rank, 4 * c.rank + 4):
        manifest = c.pm / 'snap' / f'rank{rank}.json'
        binary = c.pm / 'snap' / f'rank{rank}.bin'
        require(manifest.is_file() and binary.is_file() and not manifest.is_symlink()
                and not binary.is_symlink(), f'missing terminal snapshot for local rank {rank}')
        m = load(manifest)
        require(m.get('format') == 2 and m.get('stage') == 'after_load_model'
                and m.get('rank') == rank and m.get('world_size') == 8
                and type(m.get('bytes')) is int and m['bytes'] > 0
                and binary.stat().st_size == m['bytes'] and m.get('tensors'),
                f'invalid terminal snapshot for rank {rank}')
        for tensor in m['tensors']:
            require(type(tensor.get('offset')) is int and type(tensor.get('nbytes')) is int
                    and 0 <= tensor['offset'] <= tensor['offset'] + tensor['nbytes'] <= m['bytes'],
                    f'invalid tensor extent in {manifest}')
        artifacts.append({'rank': rank, 'manifest_sha256': digest(manifest), 'bytes': m['bytes'],
                          'binary_mtime_ns': binary.stat().st_mtime_ns,
                          'binary_sha256': digest(binary)})
    return artifacts


def executable_artifacts(c, *, strict=True):
    """Inventory actual bytes even on a functional run with missing cache artifacts."""
    files = {}
    for name in ('bank', 't3'):
        root = private(c.pm / name)
        for path in sorted(root.rglob('*')):
            require(not path.is_symlink(), f'executable cache symlink refused: {path}')
            if path.is_dir():
                continue
            info = path.stat()
            require(stat.S_ISREG(info.st_mode), f'non-file executable artifact: {path}')
            files[str(path.relative_to(c.pm))] = {'bytes': info.st_size, 'sha256': digest(path)}
    env = c.spec['env']
    missing = []
    def present(ok, message):
        if not ok:
            missing.append(message)
    bank = c.pm / 'bank'
    for rank in range(c.rank * 4, c.rank * 4 + 4):
        if env.get('PM_AOT_PRELOAD') == '1':
            present(any(bank.glob(f'torch_compile_cache/torch_aot_compile/*/rank_{rank}_0/model')),
                    f'missing outer AOT artifact for local rank {rank}')
        if env.get('PM_BANK_PRELOAD') == '1' or env.get('PM_FAST_REPLAY') == '1':
            present(any(bank.glob(f'torch_compile_cache/*/rank_{rank}_0/*/artifact_compile_range_*')),
                    f'missing executable bank for local rank {rank}')
        if env.get('PM_SAMPLER_BANK') == '1':
            present(any((bank / 'pm_sampler_bank' / f'rank_{rank}').glob('*.aot')),
                    f'missing sampler bank for local rank {rank}')
    if env.get('PM_AOT_SLIM') == '1':
        present(any(key.startswith('bank/pm_aot_slim/') for key in files), 'missing slim AOT artifacts')
    present(any(key.startswith('t3/') for key in files), 'configured private Tier3 cache is empty')
    if strict:
        require(not missing, '; '.join(missing))
    return {'files': files, 'sha256': hashlib.sha256(encoded(files)).hexdigest(),
            **({'missing_required_artifacts': missing} if not strict else {})}


def artifact_summary(contract):
    return {'sha256': contract['sha256'], 'files': len(contract['files']),
            'bytes': sum(entry['bytes'] for entry in contract['files'].values())}


def warm_cache_evidence(c, paths, *, seed=False):
    """Keep all cache diagnostics; TPU init and terminal restore remain mandatory."""
    bad = re.compile(
        r'PM_AOT.*hit=False|PM_AOT_SLIM.*\bwrote\b|'
        r'PM_(?:AOT|BANK)_PRELOAD.*(?:skipped|no artifact|does not match)|'
        r'PM_SAMPLER_BANK.*(?:stock path|built=[1-9][0-9]*|stock=[1-9][0-9]*)|'
        r'PM_SNAP.*running the stock loader|Compiling model again due to a load failure|'
        r'\[TpuCompilerAdaptor\].*(?:Saved compiled artifact|Unreadable artifact|Ignoring artifact|'
        r'Could not replay artifact|Failed to save artifact|serialize\(\) failed)')
    required = {}
    env = c.spec['env']
    for rank in range(c.rank * 4, c.rank * 4 + 4):
        tags = {'tpu_init': re.compile(rf'PM_REWIND_TPU_INIT rank={rank}\b')}
        if seed:
            required[rank] = tags
            continue
        tags['snapshot_hit'] = re.compile(rf'PM_SNAP rank={rank}: snapshot present, skipping the checkpoint')
        if env.get('PM_SNAP_ASYNC') == '1':
            tags['snapshot_started'] = re.compile(rf'PM_SNAP_RESTORE_ASYNC rank={rank} started:')
            tags['snapshot_complete'] = re.compile(rf'PM_SNAP_JOIN rank={rank} ')
        else:
            tags['snapshot_complete'] = re.compile(rf'PM_SNAP_RESTORE rank={rank} ')
        if env.get('PM_AOT_PRELOAD') == '1':
            tags['aot'] = re.compile(rf'PM_AOT rank={rank}\b.*(?:preloaded|try_load_aot hit=True)')
        if env.get('PM_AOT_SLIM') == '1':
            tags['slim'] = re.compile(rf'PM_AOT_SLIM rank={rank}\b.*loaded ')
        if env.get('PM_BANK_PRELOAD') == '1':
            tags['bank'] = re.compile(rf'PM_BANK_PRELOAD rank={rank}\b.*artifacts=[1-9][0-9]*')
        if env.get('PM_SAMPLER_BANK') == '1':
            tags['sampler'] = re.compile(rf'PM_SAMPLER_BANK rank={rank}\b.*loaded=[1-9][0-9]* .*built=0 .*stock=0\b')
        required[rank] = tags
    seen = {rank: set() for rank in required}
    fallback = []
    hits = []
    for path in paths:
        if not path.exists():
            continue
        label = str(path.relative_to(c.log)) if path.is_relative_to(c.log) else str(path)
        with path.open(errors='replace') as stream:
            for number, line in enumerate(stream, 1):
                if not seed and bad.search(line):
                    fallback.append({'file': label, 'line': number, 'text': line.rstrip()})
                for rank, tags in required.items():
                    matched = [tag for tag, pattern in tags.items() if pattern.search(line)]
                    seen[rank].update(matched)
                    if matched:
                        hits.append({'file': label, 'line': number,
                                     'rank': rank, 'tags': matched, 'text': line.rstrip()})
    missing = {str(rank): sorted(set(tags) - seen[rank]) for rank, tags in required.items()
               if set(tags) - seen[rank]}
    essential = {'tpu_init', 'snapshot_hit', 'snapshot_started', 'snapshot_complete'}
    missing_functional = {rank: [tag for tag in tags if tag in essential]
                          for rank, tags in missing.items() if essential.intersection(tags)}
    terminal_fallback = [item for item in fallback
                         if re.search(r'PM_SNAP.*running the stock loader', item['text'])]
    return {'complete': not fallback and not missing, 'missing_positive_hits': missing,
            'functional_complete': not missing_functional and not terminal_fallback,
            'missing_functional_hits': missing_functional, 'terminal_fallback': terminal_fallback,
            'positive_hits': hits, 'fallback_count': len(fallback), 'fallback': fallback,
            'scope': ('real deferred TPU init markers' if seed else
                      'per-rank real TPU init, terminal restore and replay diagnostics; executable cache identity checked separately')}


def proxy_cleanup_evidence(c, controllers, *, ownership_closed=False, run_id=None):
    recorded = [event for controller in controllers.values() for event in controller['events']
                if event.get('event') == 'PM_RAY_ZYGOTE_PROXY_IDENTITY']
    identities = {}
    for event in recorded:
        who = event.get('proxy', {})
        require(event.get('run_id') == (run_id or c.run) and type(who.get('pid')) is int and who['pid'] > 0
                and type(who.get('start_ticks')) is int and who['start_ticks'] > 0,
                'invalid generation-scoped proxy identity receipt')
        identities[(who['pid'], who['start_ticks'])] = who
    deadline = time.monotonic() + 30
    while identities and any(alive(who) for who in identities.values()) and time.monotonic() < deadline:
        time.sleep(.02)
    remaining = [who for who in identities.values() if alive(who)]
    return {'complete': ownership_closed and not remaining,
            'measurement_complete': len(identities) >= 4,
            'ownership_closed': ownership_closed,
            'identities': list(identities.values()), 'remaining': remaining,
            'scope': 'closed launch/supervisor ownership plus actual disappearance of every recorded accepted proxy'}


class Marker:
    # Marker sockets are wrappers' resources, never part of a CPU image.
    def __init__(self, c):
        self.c = c
        self.lock = threading.Lock()
        self.state = {'run_id': c.run, 'host_rank': c.rank, 'config_hash': c.config_hash,
                      'install_id': c.install_id, 'model_id': c.model_id,
                      'arm': c.arm, 'cache_mode': c.cache_mode, 'status': 'barrier',
                      'api_launch': 'not_started' if c.rank == 0 else 'remote'}
        marker = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                with marker.lock:
                    body = encoded(dict(marker.state, heartbeat_wall=time.time()))
                self.send_response(200 if self.path == f'/{c.run}/state' else 404)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        self.server = http.server.ThreadingHTTPServer((c.ip, c.mark_port), Handler, bind_and_activate=False)
        self.server.allow_reuse_address = True  # TIME_WAIT reuse, never SO_REUSEPORT/listener sharing
        self.server.server_bind()  # refuse occupied port, never stop another listener
        self.server.server_activate()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def peer_state(c):
    url = f'http://{HOSTS[1-c.rank]}:{c.mark_port}/{c.run}/state'
    try:
        with HTTP.open(url, timeout=0.25) as response:
            value = json.load(response)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None
    require(value.get('run_id') == c.run and value.get('host_rank') == 1-c.rank
            and value.get('config_hash') == c.config_hash and value.get('arm') == c.arm
            and value.get('install_id') == c.install_id and value.get('model_id') == c.model_id
            and value.get('cache_mode') == c.cache_mode,
            'peer generation/configuration mismatch')
    return value


def fresh_entry(argv):
    """Executable adapter for supervise's DMTCP launcher options; never launches DMTCP."""
    p = argparse.ArgumentParser()
    p.add_argument('--join-coordinator', action='store_true')
    for key in ('coord-host', 'coord-port', 'tmpdir'):
        p.add_argument('--' + key)
    p.add_argument('context')
    a = p.parse_args(argv)
    ctx = load(a.context)
    os.chdir(ctx['cwd'])
    factory = getattr(importlib.import_module('pm_rewind_ray'), 'prepare_' + ctx['role'])
    resume = factory()  # genuinely new interpreter; generation/identity absent here
    os.environ.update(ctx['late'])
    return resume(ctx['argv']) or 0


def fresh_controller(context):
    import pm_rewind
    ctx = load(context)
    run_dir = Path(ctx['run_dir'])
    run_dir.mkdir(mode=0o700)
    return pm_rewind.supervise([ctx['entry'], str(context)], ctx['frozen'], run_dir,
                               ctx['bin_dir'], ctx['timeout'])


def contexts(c):
    mode = 'save' if c.arm == 'seed' else 'restore'
    pin = {k: v for k, v in c.spec['env'].items() if k.startswith(('TPU_', 'TORCH_TPU_'))}
    pin.update(PM_SNAP_MODE=mode)
    common = {'TTL_RUN': c.run, 'PM_RAY_ZYGOTE_SOCK': str(c.socket),
              'PM_SNAP_MODE': mode, 'TTL_RAY_WORKER_ENV': json.dumps(pin, sort_keys=True)}
    runtime_env = dict(c.spec['env'], **common, TTL_RAY_WORKER='1')
    runtime = {'py_executable': shlex.join([str(c.python), '-S', str(c.src / 'pm_ray_zygote.py'), 'exec']),
               'env_vars': runtime_env}
    api = dict(common, RAY_ADDRESS=c.address, TTL_RAY_RUNTIME_ENV=json.dumps(runtime, sort_keys=True),
               TTL_RAY_JOBID_FILE=str(c.log / 'ray_job_id'), TPU_WORKER_ID='0', VLLM_HOST_IP=HOSTS[0])
    if c.spec['api_forkserver']:
        api['PM_ZYGOTE_SOCK'] = str(c.generation / 'api.sock')
    return {'workers': common, 'api': api}


def checked_parent_death(expected_parent, command):
    """Exec shim: establish owner death outside the multithreaded host's fork path."""
    expected_parent = int(expected_parent)
    libc = ctypes.CDLL(None, use_errno=True)
    require(os.getppid() == expected_parent
            and libc.prctl(1, signal.SIGTERM, 0, 0, 0) == 0
            and os.getppid() == expected_parent, 'controller owner disappeared before parent-death binding')
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    os.execv(command[0], command)


def start_controller(c, role, late, owner):
    run_dir = c.generation / role
    argv = c.spec['argv'] if role == 'api' else []
    if c.arm == 'restore':
        command = [str(c.python), str(c.src / 'pm_rewind.py'), 'run', str(c.cache / role),
                   '--run-dir', str(run_dir), '--timeout', str(c.timeout + 2400)]
        for key, value in sorted(late.items()):
            command += ['--env', f'{key}={value}']
        command += ['--', *argv]
    else:
        context = c.generation / (role + '-context.json')
        dump(context, {'role': role, 'argv': argv, 'late': late, 'frozen': c.spec['env'],
                       'cwd': str(c.base), 'entry': str(c.src / 'fresh-entry'),
                       'run_dir': str(run_dir), 'bin_dir': str(c.native / 'bin'),
                       'timeout': c.timeout + 2400})
        command = [str(c.python), str(c.src / 'pm_rewind_tpu.py'), '_fresh-controller', str(context)]
    command = [str(c.python), str(c.src / 'pm_rewind_tpu.py'), '_checked-parent-death',
               str(os.getpid()), *command]
    with (c.log / (role + '-controller.log')).open('xb') as output:
        owner['launch'][role] = 'attempted'
        dump(c.log / 'rewind-run-owner.json', owner)
        try:
            proc = subprocess.Popen(command, env=c.spec['env'], cwd=c.base,
                                    stdout=output, stderr=subprocess.STDOUT)
        except OSError:
            owner['launch'][role] = 'spawn_failed'
            dump(c.log / 'rewind-run-owner.json', owner)
            raise
    return proc


def request_validation(c, check):
    with (c.log / 'request-probe.log').open('xb') as output:
        proc = subprocess.Popen([str(c.python), str(c.src / 'torchtpu_req_probe.py'), str(c.port),
                                 str(c.log / 'req_probe.json')], env=c.spec['env'],
                                stdout=output, stderr=subprocess.STDOUT)
        try:
            end = time.monotonic() + 1800
            while proc.poll() is None:
                check()
                require(time.monotonic() < end, 'request probe timed out')
                time.sleep(.02)
            require(proc.returncode == 0, 'request probe exited nonzero')
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
    requests = load(c.log / 'req_probe.json')['requests']
    require([r.get('name') for r in requests] == ['q1_cold', 'q1_warm', 'q2_newshape', 'q2_warm', 'd128'],
            'request probe did not produce exactly the five required records')
    for r in requests:
        require(not r.get('error') and isinstance(r.get('text'), str) and r['text'].strip()
                and type(r.get('completion_tokens')) is int and r['completion_tokens'] > 0
                and type(r.get('prompt_tokens')) is int and r['prompt_tokens'] > 0,
                f'invalid request record: {r.get("name")}')
    body = {'model': 'gemma4', 'messages': [{'role': 'user', 'content':
            'Name the three primary colors in one short sentence.'}], 'max_tokens': 32, 'temperature': 0}
    request = urllib.request.Request(f'http://127.0.0.1:{c.port}/v1/chat/completions',
                                    data=encoded(body), headers={'Content-Type': 'application/json'})
    # A bounded stdlib request runs off-thread so peer/controller failures are still noticed.
    outcome = {}
    def greedy():
        try:
            with HTTP.open(request, timeout=600) as response:
                outcome['value'] = json.load(response)
        except BaseException as exc:
            outcome['error'] = repr(exc)
    thread = threading.Thread(target=greedy, daemon=True)
    thread.start()
    while thread.is_alive():
        check()
        thread.join(.02)
    require('value' in outcome, f'greedy request failed: {outcome}')
    value = outcome['value']
    dump(c.log / 'greedy_check.json', value)
    require(not value.get('error') and value.get('choices')
            and value['choices'][0].get('message', {}).get('content', '').strip()
            and type(value.get('usage', {}).get('completion_tokens')) is int
            and value['usage']['completion_tokens'] > 0, 'invalid greedy completion')
    return {'requests': requests, 'greedy': value}


def ray_audit(job, address, destination, gcs_address):
    """Strict job discovery plus keyed GCS actor->worker proof; no Ray driver."""
    from dataclasses import asdict, is_dataclass
    def rows(values):
        return [asdict(v) if is_dataclass(v) else dict(v) for v in values]
    result = {'job_id': job, 'complete': False}
    try:
        import grpc
        from ray._raylet import ActorID, JobID, NodeID, WorkerID
        from ray.core.generated import gcs_pb2, gcs_service_pb2, gcs_service_pb2_grpc
        from ray.util.state import list_actors, list_placement_groups

        def exact_id(value, kind, label):
            require(isinstance(value, str) and re.fullmatch('[0-9a-f]+', value) is not None,
                    f'missing/invalid {label}')
            identity = kind.from_hex(value)
            require(not identity.is_nil() and identity.hex() == value, f'nil/noncanonical {label}')
            return identity.binary()

        def table(reply, field, label):
            require(reply.HasField('status') and reply.status.code == 0,
                    f'{label} RPC failed: code={reply.status.code}, message={reply.status.message!r}')
            require(reply.HasField(field), f'{label} returned no exact record')
            return getattr(reply, field)

        job_bytes = exact_id(job, JobID, 'job ID')
        endpoint = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9._-]*|\[[0-9a-fA-F:]+\]):([0-9]+)',
                                gcs_address) if isinstance(gcs_address, str) else None
        require(endpoint is not None and 0 < int(endpoint[2]) <= 65535,
                'explicit GCS host:port required; no auto/local/dashboard/Ray Client address')
        actors = rows(list_actors(address=address, filters=[('job_id', '=', job)],
                                 limit=10000, timeout=10, detail=True, raise_on_missing_output=True))
        groups = rows(list_placement_groups(address=address, filters=[('creator_job_id', '=', job)],
                                           limit=10000, timeout=10, detail=True, raise_on_missing_output=True))
        workers = []
        result.update(actors=actors, placement_groups=groups, workers=workers,
                      recorded_pg_ids=[g['placement_group_id'] for g in groups],
                      recorded_worker_ids=[], unbound_actors=[],
                      query_proof={'method': 'GetActorInfo.address.worker_id -> GetWorkerInfo',
                                   'gcs_address': gcs_address, 'state_address': address,
                                   'actor_rpc_count': 0, 'worker_rpc_count': 0,
                                   'unbound_actor_ids': [],
                                   'job_binding': 'ActorTableData.job_id -> address.worker_id; '
                                                  'WorkerTableData has no job_id field'},
                      proxy_evidence='exact GCS actor/job/node/PID/worker identity and worker lifecycle; '
                                     'no global worker list or arbitrary process scan')
        require(len({a['actor_id'] for a in actors}) == len(actors), 'duplicate discovered actor IDs')
        require(len(set(result['recorded_pg_ids'])) == len(groups), 'duplicate discovered PG IDs')
        for group in groups:
            require(group.get('creator_job_id') == job and group.get('state') == 'REMOVED',
                    f'placement group is not exact-job REMOVED: {group}')
        # Both stubs share the explicitly configured GCS endpoint. The dashboard
        # remains only the strict actor/PG discovery source; never infer a GCS port.
        with grpc.insecure_channel(gcs_address) as channel:
            actor_stub = gcs_service_pb2_grpc.ActorInfoGcsServiceStub(channel)
            worker_stub = gcs_service_pb2_grpc.WorkerInfoGcsServiceStub(channel)
            for actor in actors:
                actor_id = actor['actor_id']
                actor_bytes = exact_id(actor_id, ActorID, 'actor ID')
                require(actor.get('job_id') == job and actor.get('state') == 'DEAD',
                        f'discovered actor is not exact-job DEAD: {actor_id}')
                actor_reply = actor_stub.GetActorInfo(
                    gcs_service_pb2.GetActorInfoRequest(actor_id=actor_bytes), timeout=10)
                result['query_proof']['actor_rpc_count'] += 1
                data = table(actor_reply, 'actor_table_data', f'GetActorInfo({actor_id})')
                require(data.actor_id == actor_bytes and data.job_id == job_bytes
                        and data.state == gcs_pb2.ActorTableData.DEAD,
                        f'exact actor ID/job/state mismatch: {actor_id}')
                # State rows discover ownership; exact GCS data supplies execution
                # identity even when optional dashboard projection fields are absent.
                pid = data.pid
                projected_pid = actor.get('pid')
                require(projected_pid is None or (type(projected_pid) is int and projected_pid == pid),
                        f'discovered/exact actor PID mismatch: {actor_id}')
                projected_node = actor.get('node_id')
                require(projected_node in (None, '') or
                        exact_id(projected_node, NodeID, f'discovered actor {actor_id} node ID') == data.node_id,
                        f'discovered/exact actor node mismatch: {actor_id}')
                never_started = (data.HasField('death_cause')
                                 and data.death_cause.HasField('actor_died_error_context')
                                 and data.death_cause.actor_died_error_context.never_started is True)
                unschedulable = (data.HasField('death_cause')
                                 and data.death_cause.HasField('actor_unschedulable_context'))
                if never_started or unschedulable:
                    # Ray may store a pending node/raylet address before granting
                    # a worker lease (gcs.proto node_id and scheduler grant path).
                    # This proves no GCS worker binding, NOT no process attempts:
                    # retirement separately verifies native/proxy PID ownership.
                    require(pid == 0 and not data.HasField('node_id')
                            and data.start_time == 0 and data.max_restarts == 0
                            and data.num_restarts == 0
                            and data.num_restarts_due_to_lineage_reconstruction == 0
                            and data.num_restarts_due_to_node_preemption == 0
                            and not data.preempted and not data.resource_mapping
                            and not data.address.worker_id
                            and actor.get('node_id') in (None, '')
                            and actor.get('worker_id') in (None, ''),
                            f'unbound actor has prior execution/lease/worker binding: {actor_id}')
                    if data.address.node_id:
                        exact_id(data.address.node_id.hex(), NodeID, f'pending actor {actor_id} node ID')
                    if never_started:
                        death = data.death_cause.actor_died_error_context
                        require(death.pid == 0 and death.actor_id in (b'', actor_bytes)
                                and death.node_ip_address == data.address.ip_address,
                                f'never-started death context contradicts exact actor: {actor_id}')
                    result['unbound_actors'].append({
                        'actor_id': actor_id, 'job_id': job, 'state': 'DEAD',
                        'pid': pid, 'start_time': data.start_time, 'node_id': '', 'worker_id': '',
                        'max_restarts': data.max_restarts, 'num_restarts': data.num_restarts,
                        'num_restarts_due_to_lineage_reconstruction': data.num_restarts_due_to_lineage_reconstruction,
                        'num_restarts_due_to_node_preemption': data.num_restarts_due_to_node_preemption,
                        'preempted': data.preempted, 'resource_mapping_count': len(data.resource_mapping),
                        'pending_node_address': {'node_id': data.address.node_id.hex(),
                                                 'ip_address': data.address.ip_address,
                                                 'port': data.address.port},
                        'actor_rpc_status': actor_reply.status.code,
                        'death_context': ('actor_died_error_context.never_started' if never_started
                                          else 'actor_unschedulable_context'),
                        'proof': 'exact DEAD first-attempt actor without recorded GCS worker lease/binding',
                        'native_process_cleanup': 'separate required retirement predicate'})
                    result['query_proof']['unbound_actor_ids'].append(actor_id)
                    continue
                require(pid > 0,
                        f'actor lacks an exact positive PID and exact unbound death proof: {actor_id}')
                node_bytes = exact_id(data.node_id.hex(), NodeID, f'exact actor {actor_id} node ID')
                require(data.HasField('node_id')
                        and data.HasField('address') and data.address.node_id == node_bytes,
                        f'exact actor node mismatch: {actor_id}')
                worker_bytes = exact_id(data.address.worker_id.hex(), WorkerID,
                                        f'actor {actor_id} address.worker_id')
                worker_id = worker_bytes.hex()
                require(not actor.get('worker_id') or actor['worker_id'] == worker_id,
                        f'discovered actor worker ID mismatch: {actor_id}')
                worker_reply = worker_stub.GetWorkerInfo(
                    gcs_service_pb2.GetWorkerInfoRequest(worker_id=worker_bytes), timeout=10)
                result['query_proof']['worker_rpc_count'] += 1
                worker = table(worker_reply, 'worker_table_data', f'GetWorkerInfo({worker_id})')
                require(worker.HasField('worker_address')
                        and worker.worker_address.worker_id == worker_bytes
                        and worker.worker_address.node_id == node_bytes and worker.pid == pid
                        and worker.is_alive is False,
                        f'exact worker ID/node/PID mismatch or still alive: {worker_id}')
                workers.append({'worker_id': worker_id, 'actor_id': actor_id, 'job_id': job,
                                'node_id': node_bytes.hex(), 'pid': pid, 'is_alive': False,
                                'actor_rpc_status': actor_reply.status.code,
                                'worker_rpc_status': worker_reply.status.code})
                result['recorded_worker_ids'].append(worker_id)
        result['complete'] = True
    except Exception as exc:
        result['error'] = repr(exc)
    if destination == '-':
        print(json.dumps(result, sort_keys=True), flush=True)
    else:
        dump(destination, result)
    return 0 if result['complete'] else 1


def collect_job_logs(c, job):
    require(re.fullmatch('[0-9a-f]{8,64}', job), 'invalid recorded Ray job id')
    source = Path(os.environ.get('TTL_RAY_LOG_DIR', '/tmp/ray/session_latest/logs'))
    destination = c.log / 'ray-workers'
    destination.mkdir(mode=0o700, exist_ok=True)
    files = []
    for path in source.glob(f'worker-*-{job}-*'):
        if not path.is_file() or not re.fullmatch(r'worker-[0-9a-f]+-' + re.escape(job) + r'-[0-9]+\.(out|err)', path.name):
            continue
        target = destination / path.name
        shutil.copyfile(path, target)
        files.append(path.name)
    return files


def stop_controllers(c, controllers):
    receipts = {}
    # API first: ordinary shutdown allows Ray to release actors/PG before worker zygotes exit.
    for role in ('api', 'workers'):
        proc = controllers.get(role)
        if proc is None:
            continue
        if proc.poll() is None:
            proc.terminate()  # only Popen-owned ordinary controllers, never restored SID/PID
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass  # Do not SIGKILL an ownership anchor, even if its cleanup is stuck.
        path = c.generation / role / 'supervisor.json'
        receipt = load(path) if path.exists() else None
        observed = [event for name in ('output.log', 'backend.log')
                    for event in events(c.generation / role / name)]
        restored = [e['restore_s'] for e in observed if e.get('event') == 'PM_REWIND_RESTORED'
                    and 'restore_s' in e]
        receipts[role] = {'returncode': proc.poll(), 'supervisor': receipt,
                          'events': observed, 'restore_s': restored[0] if len(restored) == 1 else None,
                          'restore_s_scope': 'CPU image restart only; not HTTP readiness',
                          'complete': proc.poll() is not None and bool(receipt)
                          and receipt.get('cleanup') == 'complete'}
        for name in ('output.log', 'backend.log'):
            output = c.generation / role / name
            if output.exists():
                shutil.copyfile(output, c.log / f'{role}-{name}')
    return receipts


def local_cleanup_evidence(c, owner, previous=None, *, read_only=False):
    """Reconstruct local ownership from the launch journal and final raw receipts."""
    run_id = owner['run_id']
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', run_id), 'invalid owned run ID')
    require(owner.get('config_hash') == c.config_hash and owner.get('host_rank') == c.rank,
            'cleanup ownership/configuration mismatch')
    log = c.base / 'logs' / run_id
    generation = c.base / 'runs' / run_id
    roles = {'workers', 'api'} if c.rank == 0 else {'workers'}
    stages = owner.get('launch', {})
    require(set(stages) == roles and set(owner['controllers']) <= roles,
            'missing or unknown role launch journal; ownership unresolved')
    controllers = {}
    previous = previous or {}
    for role, stage in stages.items():
        who = owner['controllers'].get(role)
        directory = generation / role
        archived = log / f'{role}-supervisor.json'
        if stage in ('not_started', 'spawn_failed'):
            require(who is None and not directory.exists() and not archived.exists(),
                    f'{role}: no-launch journal contradicts controller/supervisor evidence')
            continue
        require(stage in ('attempted', 'started') and who is not None,
                f'{role}: launch attempted without a registered controller identity')
        require(not alive(who), f'{role}: registered controller still alive: {who}')
        receipt_path = directory / 'supervisor.json' if generation.exists() else archived
        require(receipt_path.is_file(), f'{role}: final owned supervisor receipt missing: {receipt_path}')
        receipt = load(receipt_path)
        retired_supervisor(receipt, f'{run_id}/{role}')
        observed = []
        log_identity = {}
        for name in ('output.log', 'backend.log'):
            source = directory / name if generation.exists() else log / f'{role}-{name}'
            require(not source.is_symlink(), f'{role}: unexpected symlink in event ledger: {source}')
            if name == 'backend.log':
                require(source.is_file(), f'{role}: backend event ledger is missing: {source}')
            log_identity[name] = {'sha256': digest(source), 'bytes': source.stat().st_size} if source.exists() else None
            if not generation.exists():
                require(name in receipt.get('event_logs', {}) and receipt['event_logs'][name] == log_identity[name],
                        f'{role}: archived event ledger changed or missing: {source}')
            observed.extend(events(source))
            if not read_only and generation.exists() and source.exists():
                shutil.copyfile(source, log / f'{role}-{name}')
        receipt['event_logs'] = log_identity
        if not read_only:
            dump(archived, receipt)
        restored = [event['restore_s'] for event in observed
                    if event.get('event') == 'PM_REWIND_RESTORED' and 'restore_s' in event]
        entry = dict(previous.get('controllers', {}).get(role, {}))
        entry.update(controller=who, supervisor=receipt, events=observed, complete=True,
                     restore_s=restored[0] if len(restored) == 1 else None,
                     restore_s_scope='CPU image restart only; not HTTP readiness')
        controllers[role] = entry
    proxies = proxy_cleanup_evidence(c, controllers, ownership_closed=True, run_id=run_id)
    known = {(who['pid'], who['start_ticks'])
             for who in (previous.get('proxy_cleanup') or {}).get('identities', [])}
    require(known <= {(who['pid'], who['start_ticks']) for who in proxies['identities']},
            'previously recorded proxy identities disappeared from the ownership ledger')
    return {'controllers': controllers, 'proxy_cleanup': proxies,
            'local_cleanup_complete': proxies['complete'],
            'launch_journal': dict(stages),
            'local_cleanup_reason': ('all registered launches closed and recorded proxies gone'
                                     if proxies['complete'] else f"live recorded proxies: {proxies['remaining']}")}


def ray_cleanup_evidence(c, owner, report, *, read_only=False):
    """No API launch is explicit evidence; any possible API launch needs an exact job."""
    run_id = owner['run_id']
    log = c.base / 'logs' / run_id
    peer = report.get('peer_cleanup') or owner.get('peer_terminal') or {}
    peer_no_api = (peer.get('run_id') == run_id and peer.get('host_rank') == 0
                   and peer.get('config_hash') == c.config_hash
                   and peer.get('status') in ('cleaned', 'released')
                   and peer.get('api_launch') in ('not_started', 'spawn_failed'))
    no_api = (owner['launch'].get('api') in ('not_started', 'spawn_failed') if c.rank == 0
              else owner.get('peer_api_authorized') is False or peer_no_api)
    job_file = log / 'ray_job_id'
    candidates = {job for job in (report.get('job_id'), owner.get('job_id'), peer.get('job_id'),
                  (report.get('ray_cleanup') or {}).get('job_id'),
                  job_file.read_text().strip() if job_file.exists() else None) if job}
    if no_api:
        require(not candidates and not report['proxy_cleanup']['identities'],
                'no-API launch proof contradicts recorded Ray job/proxy activity')
        return {'complete': True, 'no_ray_action': True, 'recorded_pg_ids': [],
                'basis': ('local API never spawned' if c.rank == 0 else
                          'peer API never authorized' if owner.get('peer_api_authorized') is False
                          else 'host0 terminal launch journal proves no API spawned')}
    require(len(candidates) == 1,
            f'API may have connected but exact Ray job ownership is unknown/conflicting: {sorted(candidates)}')
    job = next(iter(candidates))
    require(re.fullmatch('[0-9a-f]{8,64}', job), 'invalid recorded Ray job ID')
    destination = log / f'ray-recheck-{time.monotonic_ns()}.json'
    helper = Path(__file__).resolve() if read_only else c.src / 'pm_rewind_tpu.py'
    command = [str(c.python), '-B', str(helper),
               '_ray-audit', job, c.state_address, '-' if read_only else str(destination),
               c.spec['ray_address']]
    if read_only:
        proc = subprocess.run(command, env=c.spec['env'], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, timeout=150)
        try:
            audit = json.loads(proc.stdout)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f'Ray audit returned invalid evidence: job={job}, rc={proc.returncode}, '
                               f'stdout={proc.stdout!r}, stderr={proc.stderr!r}') from exc
    else:
        with (log / 'ray-audit.log').open('ab') as output:
            proc = subprocess.run(command, env=c.spec['env'], stdout=output,
                                  stderr=subprocess.STDOUT, timeout=150)
        require(destination.is_file(), f'Ray state audit produced no evidence: job={job}, rc={proc.returncode}')
        audit = load(destination)
    require(audit.get('job_id') == job, 'Ray audit job identity mismatch')
    previous = report.get('ray_cleanup') or {}
    retained = previous.get('ownership_inventory') or {}
    required = {
        'placement_groups': set(previous.get('recorded_pg_ids', [])) | set(retained.get('placement_groups', [])),
        'workers': set(previous.get('recorded_worker_ids', [])) | set(retained.get('workers', [])),
        'actors': {a['actor_id'] for a in previous.get('actors', [])} | set(retained.get('actors', [])),
    }
    observed = {
        'placement_groups': set(audit.get('recorded_pg_ids', [])),
        'workers': set(audit.get('recorded_worker_ids', [])),
        'actors': {a['actor_id'] for a in audit.get('actors', [])},
    }
    # Failed cleanup may be persisted by its caller. Carry prior ownership
    # separately, never presenting retained IDs as fresh GCS observations.
    audit['ownership_inventory'] = {key: sorted(required[key] | observed[key]) for key in required}
    if proc.returncode != 0 or audit.get('complete') is not True:
        require(not read_only,
                f'Ray audit failed: job={job}, rc={proc.returncode}, '
                f'error={audit.get("error")!r}, evidence={audit}')
        # Preserve run()'s bounded shutdown poll without certifying incomplete
        # evidence. Retirement callers still require a complete terminal audit.
        audit.update(complete=False, returncode=proc.returncode,
                     no_ray_action=False, evidence_file=str(destination))
        report['job_id'] = job
        return audit
    require(required['placement_groups'] <= observed['placement_groups'],
            f'previous exact-job placement groups absent from fresh audit: {job}')
    require(required['workers'] <= observed['workers'],
            f'previous exact-job workers absent from fresh audit: {job}')
    require(required['actors'] <= observed['actors'],
            f'previous exact-job actors absent from fresh audit: {job}')
    audit.update(complete=proc.returncode == 0 and bool(audit.get('complete')),
                 no_ray_action=False, evidence_file=None if read_only else str(destination))
    report['job_id'] = job
    return audit


def require_qualification(c):
    qualified = load(c.base / 'rewind-qualified.json')
    require(qualified.get('success') and qualified.get('cleanup_complete')
            and qualified.get('config_hash') == c.config_hash and qualified.get('install_id') == c.install_id,
            'successful same-install CPU/native qualification required before model copy or TPU launch')


def model_files(path):
    files = {}
    for entry in sorted(path.iterdir()):
        require(entry.is_file() and not entry.is_symlink(), f'non-regular model payload: {entry}')
        files[entry.name] = {'bytes': entry.stat().st_size, 'sha256': digest(entry)}
    return files


def copy_model(c):
    c.check_install()
    require_qualification(c)
    private(c.ckpt)
    require(not any(c.ckpt.iterdir()) and not (c.pm / 'model-receipt.json').exists(),
            'copy requires an empty owned model directory; partial/successful copies are never reused')
    raw_source = Path(os.environ.get('TTL_SRC', '/models/gemma-4-31b'))
    require(raw_source.is_absolute(), 'TTL_SRC must be an absolute read-only source')
    source = raw_source.resolve(strict=True)
    require(source.is_dir() and not source.is_relative_to(c.pm) and not source.is_relative_to(c.base),
            'model source must be outside owned experiment roots')
    sizes = {}
    for path in sorted(source.iterdir()):
        require(not path.is_symlink(), f'source symlink refused: {path}')
        if path.is_file():
            sizes[path.name] = path.stat().st_size
    indices = [name for name in sizes if name.endswith('.safetensors.index.json')]
    require('config.json' in sizes and len(indices) == 1, 'source needs config.json and one safetensors index')
    index = load(source / indices[0])
    weights = index.get('weight_map')
    require(isinstance(weights, dict) and weights, 'empty/invalid model weight map')
    require(all(isinstance(name, str) and Path(name).name == name and name.endswith('.safetensors')
                and sizes.get(name, 0) > 0 for name in weights.values()), 'index refers to missing/unsafe weight shards')
    source_identity = {name: digest(source / name) for name in ('config.json', indices[0])}
    private(c.log, create=True)
    info = c.ckpt.stat()
    owner_path = c.log / 'rewind-copy-owner.json'
    owner = {'run_id': c.run, 'host_rank': c.rank, 'config_hash': c.config_hash, 'owner': identity(),
             'child': None, 'destination': str(c.ckpt),
             'destination_identity': {'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid},
             'source_read_only': str(source)}
    stable_write(owner_path, encoded(owner))
    command = [str(c.python), str(c.src / 'torchtpu_ckpt_copy.py'), str(source), str(c.ckpt), '--threads', '48']
    result = {'phase': 'rewind-copy', 'run_id': c.run, 'host_rank': c.rank, 'config_hash': c.config_hash,
              'install_id': c.install_id,
              'command': command, 'source': str(source), 'destination': str(c.ckpt),
              'success': False, 'cleanup_complete': False,
              'source_verification': 'helper checks safetensors sizes and first/middle/last 16 MiB samples; wrapper compares full config/index SHA-256',
              'target_integrity': 'full SHA-256 of every copied file; not a claim of full source-weight hashing'}
    proc = None
    try:
        CANCELLATION['deferred'] = True
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            with (c.log / 'checkpoint-copy.log').open('xb') as output:
                proc = subprocess.Popen([str(c.python), str(c.src / 'pm_rewind_tpu.py'),
                    '_checked-parent-death', str(os.getpid()), *command], env=c.spec['env'], cwd=c.base,
                    stdout=output, stderr=subprocess.STDOUT)
            owner['child'] = identity(proc.pid)
            dump(owner_path, owner)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
            CANCELLATION['deferred'] = False
            if CANCELLATION['pending']:
                CANCELLATION['pending'] = False
                raise KeyboardInterrupt('cancelled during checkpoint-copy registration')
        proc.wait(timeout=float(os.environ.get('TTL_REWIND_COPY_TIMEOUT', '3600')))
        require(proc.returncode == 0, f'checkpoint copy failed rc={proc.returncode}')
        summaries = []
        with (c.log / 'checkpoint-copy.log').open() as output:
            for line in output:
                if line.startswith('CKPT_COPY_JSON='):
                    summaries.append(json.loads(line.partition('=')[2]))
        require(len(summaries) == 1 and summaries[0].get('threads') == 48, 'missing copy-helper completion receipt')
        files = model_files(c.ckpt)
        require({name: entry['bytes'] for name, entry in files.items()} == sizes,
                'copied model inventory/sizes differ from source')
        require(all((source / name).stat().st_size == size for name, size in sizes.items()),
                'source size changed during copy')
        require(all(files[name]['sha256'] == expected == digest(source / name)
                    for name, expected in source_identity.items()), 'model config/index changed or copied incorrectly')
        model_id = hashlib.sha256(encoded(files)).hexdigest()
        model = {'success': True, 'config_hash': c.config_hash, 'install_id': c.install_id,
                 'run_id': c.run, 'source': str(source), 'source_identity': source_identity,
                 'files': files, 'model_id': model_id, 'copy_helper': summaries[0]}
        result.update(success=True, model_id=model_id, files=len(files),
                      bytes=sum(entry['bytes'] for entry in files.values()),
                      source_identity=source_identity, copy_helper=summaries[0])
    except BaseException as exc:
        result.update(success=False, error=f'{type(exc).__name__}: {exc}')
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                result['cleanup_error'] = 'copy process remains live; retain leases and owned partial destination'
        result['returncode'] = proc.poll() if proc is not None else None
        result['cleanup_complete'] = proc is None or proc.poll() is not None
        result['success'] = result['success'] and result['cleanup_complete']
        result['leases_must_stay_held'] = not result['cleanup_complete']
        if result['success']:
            dump(c.log / 'model-receipt.json', model)
        dump(c.log / 'rewind-copy-result.json', result)
        if result['success']:
            # Commit authority only after both LOG evidence records exist.
            dump(c.pm / 'model-receipt.json', model)
    return result


def run(c):
    c.run_launch_registered = False
    installed = c.check_install()
    c.cache_mode = os.environ.get('TTL_REWIND_CACHE_MODE', 'strict')
    require(c.cache_mode in ('strict', 'functional'), 'TTL_REWIND_CACHE_MODE must be strict|functional')
    require(c.cache_mode != 'functional' or c.arm in ('restore', 'plain'),
            'functional cache measurement cannot seed or replace seed receipts')
    require(c.spec['env']['PM_SB_PRESEED'] == '1', 'both measurement arms require PM_SB_PRESEED=1')
    private(c.log, create=True)
    require(not (c.log / 'rewind-run-owner.json').exists(), 'TTL_RUN has already been used; choose a new run ID')
    require_qualification(c)
    if installed.get('source_refresh'):
        # Even plain comparison runs must follow a new successful prepare/help;
        # historical receipts live only in archives and cannot satisfy this gate.
        roles = {'workers', 'api'} if c.rank == 0 else {'workers'}
        prepared = []
        for path in (c.base / 'logs').glob('*/rewind_measurement.json'):
            proof = load(path)
            if proof.get('phase') == 'rewind-prepare' and proof.get('success') is True:
                if (proof.get('install_id') == c.install_id and proof.get('config_hash') == c.config_hash
                        and proof.get('host_rank') == c.rank and not proof.get('leases_must_stay_held')
                        and set(proof.get('images', {})) == roles):
                    prepared.append(proof)
        require(any(all((c.cache / role / 'manifest.json').is_file()
                        and digest(c.cache / role / 'manifest.json') == proof['images'][role]['manifest_sha256']
                        for role in roles) for proof in prepared),
                'refreshed installs require successful same-install prepare/help before any serving arm')
    owner = {'run_id': c.run, 'host_rank': c.rank, 'owner': identity(), 'controllers': {},
             'private_paths': [str(c.generation)], 'config_hash': c.config_hash,
             'launch': {role: 'not_started' for role in (('workers', 'api') if c.rank == 0 else ('workers',))},
             'peer_api_authorized': False}
    report = {'phase': 'rewind-run', 'run_id': c.run, 'host_rank': c.rank, 'host': c.ip,
              'arm': c.arm, 'measured': c.arm != 'seed', 'config_hash': c.config_hash,
              'config': c.spec, 'source_hashes': installed['source_hashes'],
              'cache_mode': c.cache_mode, 'functional_success': False, 'cache_qualified': False,
              'comparison_scope': ('functional native-to-HTTP/request measurement; not an all-hit or controlled speedup claim'
                                   if c.cache_mode == 'functional' else 'strict seed-identical executable cache'),
              'host_metadata': {'uname': list(os.uname()), 'uid': os.getuid(),
                                'wrapper_python': sys.version, 'runtime_python': str(c.python),
                                'owner': owner['owner']},
              'ready': False, 'success': False, 'cleanup_complete': False,
              'global_cache': {'path': '/dev/shm/torch_tpu_cache',
                               'existed_before': Path('/dev/shm/torch_tpu_cache').exists(),
                               'harness_action': 'no shared-cache payload reads, writes or deletion',
                               'configured_redirect': str(c.pm / 't3'),
                               'redirect_effectiveness': 'unverified_until_runtime',
                               'evidence': 'vllm env_override forwarding confirmed; native cache consumer behavior not established'}}
    marker = None
    controllers = {}
    peer_last = time.monotonic()
    job = None
    t0 = None
    cache_contract = None
    def monitor():
        nonlocal peer_last, job
        for role, proc in controllers.items():
            require(proc.poll() is None, f'{role} controller exited unexpectedly: {proc.returncode}')
        peer = peer_state(c)
        if peer:
            peer_last = time.monotonic()
            if peer.get('job_id') and owner.get('job_id') != peer['job_id']:
                owner['job_id'] = peer['job_id']
                dump(c.log / 'rewind-run-owner.json', owner)
            if peer.get('status') in ('cleaned', 'released'):
                owner['peer_terminal'] = peer
                dump(c.log / 'rewind-run-owner.json', owner)
            require(peer.get('status') != 'abort'
                    and not (peer.get('status') in ('cleaned', 'released') and not peer.get('success')),
                    f'peer aborted: {peer.get("error", peer.get("status"))}')
            if peer.get('job_id'):
                job = peer['job_id']
        require(time.monotonic() - peer_last < 10, 'peer heartbeat unavailable for 10 seconds')
        return peer
    def preflight():
        nonlocal cache_contract
        model = require_model_receipt(c)
        c.model_id = model['model_id']
        report['model_id'] = c.model_id
        report['model_source_identity'] = model['source_identity']
        if c.arm != 'seed':
            report['terminal_snapshots'] = snapshot_artifacts(c)
            seed_path = c.pm / 'seed-receipt.json'
            seed = load(seed_path)
            lineage = advance_lineage(c)
            inherited_seed = bool(lineage and
                lineage['seed_receipts'].get(str(seed_path)) == digest(seed_path)
                and seed.get('config_hash') == lineage['parent_config_hash']
                and seed.get('model_id') == lineage['parent_model_id'])
            require((seed.get('config_hash') == c.config_hash or inherited_seed) and seed.get('success')
                    and seed.get('model_id') == c.model_id, 'no successful same-config/model or archived parent seed receipt')
            report['seed_receipt'] = {'sha256': digest(seed_path), 'inherited': inherited_seed,
                                      'run_id': seed.get('run_id'), 'config_hash': seed.get('config_hash')}
            require(seed.get('terminal_snapshots') == report['terminal_snapshots'],
                    'terminal artifacts changed since the successful seed; refuse warm fallback/replacement')
            actual = executable_artifacts(c, strict=False)
            cache_contract = {key: actual[key] for key in ('files', 'sha256')}
            report['executable_artifacts_before'] = actual
            report['executable_artifacts_before_summary'] = artifact_summary(actual)
            report['seed_cache_identical'] = seed.get('executable_artifacts') == cache_contract
            dump(c.log / 'rewind-cache-before.json', {
                'cache_mode': c.cache_mode, 'seed_receipt': report['seed_receipt'],
                'seed_cache_identical': report['seed_cache_identical'], 'actual': actual})
            if c.cache_mode == 'strict':
                require(report['seed_cache_identical'] and not actual['missing_required_artifacts'],
                        'bank/Tier3/sampler artifacts changed or are missing since seed')
        else:
            require(not (c.pm / 'seed-receipt.json').exists()
                    and not any((c.pm / 'snap').iterdir()), 'seed refuses existing terminal artifacts')
        for role in ('workers', 'api') if c.rank == 0 else ('workers',):
            if c.arm == 'restore':
                rewind = importlib.import_module('pm_rewind')
                require(Path(rewind.__file__).resolve() == c.src / 'pm_rewind.py',
                        'preflight imported an unpinned image verifier')
                verified_cache, man = rewind.load_cache(c.cache / role, verify=True)
                with (c.log / f'preflight-{role}.log').open('ab') as output:
                    output.write(encoded({'event': 'PM_REWIND_VERIFIED',
                                          'cache': str(verified_cache), 'images': len(man['images'])}))
                require(man['config']['frozen_env'] == c.spec['env'], 'image frozen environment drift')
                report.setdefault('images', {})[role] = {
                    'manifest_sha256': digest(c.cache / role / 'manifest.json'),
                    'capture_events': events(c.cache / role / 'build/backend.log')}
        with socket.socket() as port_check:
            port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port_check.bind(('0.0.0.0', c.port))
        with socket.socket() as port_check:
            port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port_check.bind(('0.0.0.0', c.mark_port))
        require(not c.generation.exists() and not c.generation.is_symlink(), 'generation must be new')
    preflight()
    c.run_launch_registered = True
    stable_write(c.log / 'rewind-run-owner.json', encoded(owner))
    try:
        private(c.generation, create=True)
        marker = Marker(c)
        barrier_end = time.monotonic() + 180
        while True:
            peer = peer_state(c)
            if peer:
                peer_last = time.monotonic()
                require(peer['status'] != 'abort', f'peer aborted: {peer.get("error")}')
                if peer['status'] in (('armed',) if c.rank == 0 else ('barrier', 'start', 'running')):
                    break
            require(time.monotonic() < barrier_end, 'two-host barrier timed out')
            time.sleep(.01)
        if c.rank == 0:
            t0 = time.monotonic()
            report['t0_wall'] = time.time()
            marker.update(status='start', t0_wall=report['t0_wall'])
        else:
            # Host0 cannot start its API until this durable authorization is advertised.
            owner['peer_api_authorized'] = True
            dump(c.log / 'rewind-run-owner.json', owner)
            marker.update(status='armed')
            while peer.get('status') not in ('start', 'running'):
                require(time.monotonic() < barrier_end, 'master start timed out')
                time.sleep(.01)
                peer = monitor() or {}
            report['master_t0_wall'] = peer['t0_wall']
            report['start_observed_wall'] = time.time()
            report['wall_clock_launch_lag_s_not_clock_corrected'] = report['start_observed_wall'] - peer['t0_wall']
        # No wait for worker readiness: API and local worker controller start concurrently.
        late = contexts(c)
        for role in ('workers', 'api') if c.rank == 0 else ('workers',):
            CANCELLATION['deferred'] = True
            blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
            try:
                controllers[role] = start_controller(c, role, late[role], owner)
                owner['controllers'][role] = identity(controllers[role].pid)
                owner['launch'][role] = 'started'
                dump(c.log / 'rewind-run-owner.json', owner)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
                CANCELLATION['deferred'] = False
                if CANCELLATION['pending']:
                    CANCELLATION['pending'] = False
                    raise KeyboardInterrupt('host task cancelled during controller registration')
        marker.update(status='running')
        if c.rank == 0:
            end = t0 + c.timeout
            while True:
                monitor()
                if (c.log / 'ray_job_id').exists():
                    job = (c.log / 'ray_job_id').read_text().strip()
                    if owner.get('job_id') != job:
                        owner['job_id'] = job
                        dump(c.log / 'rewind-run-owner.json', owner)
                    marker.update(job_id=job)
                try:
                    with HTTP.open(f'http://127.0.0.1:{c.port}/health', timeout=.5) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    pass
                require(time.monotonic() < end, 'HTTP readiness timed out')
                time.sleep(.02)
            report['ready_s'] = time.monotonic() - t0
            report['ready'] = True
            report.update(request_validation(c, monitor))
            monitor()
            require(job, 'API did not record an exact Ray job ID')
            report['success'] = True
            marker.update(status='done', success=True, ready_s=report['ready_s'], job_id=job)
        else:
            while True:
                peer = monitor()
                if peer and peer.get('status') in ('done', 'cleaned', 'released'):
                    report.update(ready=peer.get('success', False), success=peer.get('success', False),
                                  master_ready_s=peer.get('ready_s'))
                    break
                require(time.monotonic() < barrier_end + c.timeout + 2400, 'master completion timed out')
                time.sleep(.01)
        if c.arm == 'seed':
            report['terminal_snapshots'] = snapshot_artifacts(c)
    except BaseException as exc:
        report.update(success=False, error=f'{type(exc).__name__}: {exc}')
        if marker:
            marker.update(status='abort', error=report['error'])
    finally:
        # Avoid a second cancellation interrupting ownership-preserving teardown.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            report['controllers'] = stop_controllers(c, controllers)
            local = local_cleanup_evidence(c, owner, report)
            report.update(local)
            local_complete = local['local_cleanup_complete']
            if not report['proxy_cleanup']['measurement_complete']:
                report.update(success=False, evidence_error='fewer than four local accepted proxy identities')
            if c.arm == 'restore':
                for role in ('workers', 'api') if c.rank == 0 else ('workers',):
                    restored = [event for event in report['controllers'].get(role, {}).get('events', [])
                                if event.get('event') == 'PM_REWIND_RESTORED']
                    if not (len(restored) == 1 and restored[0].get('real_pids') is True
                            and type(restored[0].get('pid')) is int and restored[0]['pid'] > 0):
                        report.update(success=False, evidence_error=f'{role} lacks real-PID native restore proof')
            # Ray redirects worker logs only after connection. Keep the original
            # no-connection witness in this owned generation, not only Ray's logs.
            prefix_path = c.generation / 'worker-prefix.jsonl'
            prefix_events = []
            if os.path.lexists(prefix_path):
                require(prefix_path.is_file() and not prefix_path.is_symlink()
                        and prefix_path.stat().st_uid == os.getuid(), 'unsafe worker-prefix witness')
                before = digest(prefix_path)
                saved_prefix = c.log / 'worker-prefix.jsonl'
                shutil.copyfile(prefix_path, saved_prefix)
                require(digest(saved_prefix) == before == digest(prefix_path),
                        'worker-prefix witness changed during collection')
                prefix_events = events(saved_prefix)
                require(len(prefix_events) == len(saved_prefix.read_text().splitlines())
                        and all(event.get('event') == 'PM_RAY_ZYGOTE_PROOF' for event in prefix_events),
                        'malformed worker-prefix witness')
                report['worker_prefix_log'] = {'path': str(saved_prefix), 'sha256': before,
                                              'events': prefix_events}
            else:
                report['worker_prefix_log'] = {'missing': True, 'path': str(prefix_path)}
            if (c.log / 'ray_job_id').exists():
                job = (c.log / 'ray_job_id').read_text().strip()
            if job:
                report['job_id'] = job
                report['ray_logs'] = collect_job_logs(c, job)
                report['worker_proofs'] = prefix_events + [event for filename in report['ray_logs']
                    for event in events(c.log / 'ray-workers' / filename)]
                report['native_proof_scope'] = (
                    'Capture receipts retain native buffer/module witnesses. Worker PM_RAY_ZYGOTE_PROOF '
                    'retains saved nonce/module addresses at bootstrap. A post-resume native-buffer getter '
                    'is not invoked by this inference harness; absent proof is not claimed.')
                capture_events = (report.get('images', {}).get('workers', {}).get('capture_events', [])
                                  if c.arm == 'restore' else report['controllers'].get('workers', {}).get('events', []))
                captures = [event for event in capture_events if event.get('event') == 'PM_REWIND_RAY_CAPTURE']
                proofs = prefix_events
                rank_workers = {rank: set() for rank in range(c.rank * 4, c.rank * 4 + 4)}
                for filename in report['ray_logs']:
                    worker_pid = int(filename.rsplit('-', 1)[1].split('.', 1)[0])
                    text = (c.log / 'ray-workers' / filename).read_text(errors='replace')
                    for rank in rank_workers:
                        if re.search(rf'PM_REWIND_TPU_INIT rank={rank}\b', text):
                            rank_workers[rank].add(worker_pid)
                service_pids = set().union(*rank_workers.values())
                proxy_workers = {event['worker_pid']: event.get('proxy')
                    for event in report['controllers'].get('workers', {}).get('events', [])
                    if event.get('event') == 'PM_RAY_ZYGOTE_PROXY_IDENTITY'
                    and type(event.get('worker_pid')) is int and event.get('run_id') == c.run}
                report['worker_rank_pids'] = {str(rank): sorted(pids) for rank, pids in rank_workers.items()}
                captured = captures[0] if len(captures) == 1 else {}
                valid_capture = cpu_capture_valid(captured, 'workers')
                wanted_modules = {name: hex(address) for name, address in
                                  captured.get('saved_module_ids', {}).items()} if valid_capture else {}
                valid = (valid_capture and all(len(pids) == 1 for pids in rank_workers.values())
                    and len(service_pids) == 4 and service_pids.issubset({event.get('pid') for event in proofs})
                    and all(
                    event.get('nonce') == captured.get('nonce') and event.get('modules') == wanted_modules
                    and event.get('tpu_initialized') is False and event.get('ray_imported') is False
                    and event.get('ray_initialized') is False
                    and type(event.get('pid')) is int and event['pid'] > 0
                    and event['pid'] != captured.get('capture_pid')
                    and event.get('proxy') is not None
                    and event['proxy'] == proxy_workers.get(event['pid'])
                    and event['proxy'] in report['proxy_cleanup']['identities'] for event in proofs))
                report['worker_saved_prefix_verified'] = valid
                report['worker_cpu_capture'] = captured
                if not valid:
                    report.update(success=False, evidence_error='missing or mismatching per-host saved worker prefix proof')
            if c.rank == 0:
                observed = report['controllers'].get('api', {}).get('events', [])
                capture_events = (report.get('images', {}).get('api', {}).get('capture_events', [])
                                  if c.arm == 'restore' else observed)
                captures = [event for event in capture_events if event.get('event') == 'PM_REWIND_RAY_CAPTURE']
                reused = [event for event in observed if event.get('event') == 'PM_REWIND_CPU_REUSE']
                valid = (len(captures) == len(reused) == 1 and cpu_capture_valid(captures[0], 'api')
                         and cpu_reuse_valid(captures[0], reused[0]))
                if c.arm == 'restore':
                    restored = [event for event in observed if event.get('event') == 'PM_REWIND_RESTORED']
                    valid = (valid and len(restored) == 1 and restored[0].get('real_pids') is True
                             and reused[0]['pid'] == restored[0].get('pid')
                             and reused[0]['pid'] != captures[0]['capture_pid'])
                report['api_cpu_object_proof'] = {'capture': captures, 'reuse': reused, 'verified': valid}
                if not valid:
                    report.update(success=False, evidence_error='missing or mismatching API CPU object reuse proof')
            try:
                after = executable_artifacts(c, strict=False)
                report['executable_artifacts_after'] = after
                report['executable_artifacts_after_summary'] = artifact_summary(after)
                paths = [c.log / f'{role}-{name}' for role in report['controllers']
                         for name in ('output.log', 'backend.log')]
                paths += [c.log / 'ray-workers' / name for name in report.get('ray_logs', [])]
                if c.arm == 'seed':
                    cache_contract = {key: after[key] for key in ('files', 'sha256')}
                    evidence = warm_cache_evidence(c, paths, seed=True)
                    report['tpu_init_evidence'] = evidence
                    if not evidence['complete'] or after['missing_required_artifacts']:
                        report.update(success=False, artifact_error='missing deferred TPU init or seed artifacts')
                else:
                    report['terminal_snapshots_after'] = snapshot_artifacts(c)
                    require(report['terminal_snapshots_after'] == report['terminal_snapshots'],
                            'terminal snapshots changed during measured run')
                    require(digest(c.pm / 'seed-receipt.json') == report['seed_receipt']['sha256'],
                            'seed receipt changed during measured run')
                    evidence = warm_cache_evidence(c, paths)
                    evidence['unchanged'] = cache_contract == {key: after[key] for key in ('files', 'sha256')}
                    evidence['seed_identical'] = report['seed_cache_identical']
                    evidence['missing_artifacts_before'] = report['executable_artifacts_before']['missing_required_artifacts']
                    evidence['missing_artifacts_after'] = after['missing_required_artifacts']
                    evidence['complete'] = (evidence['complete'] and evidence['unchanged']
                        and evidence['seed_identical'] and not evidence['missing_artifacts_before']
                        and not evidence['missing_artifacts_after'])
                    report['warm_cache_evidence'] = evidence
                    report['cache_qualified'] = evidence['complete']
                    if not evidence['functional_complete']:
                        report.update(success=False, artifact_error='missing per-rank deferred TPU init or terminal restore evidence')
            except Exception as exc:
                report.update(success=False, artifact_error=repr(exc))
            report['functional_success'] = report['success']
            if c.arm != 'seed' and c.cache_mode == 'strict' and not report['cache_qualified']:
                report.update(success=False, cache_error='warm executable cache changed or replay fell back/missing hit evidence')
            if marker:
                marker.update(status='cleaned', success=report['success'], local_cleanup_complete=local_complete,
                              functional_success=report['functional_success'], cache_qualified=report['cache_qualified'],
                              job_id=job, api_launch=owner['launch'].get('api', 'remote'))
                end = time.monotonic() + 180
                peer = None
                while time.monotonic() < end:
                    peer = peer_state(c)
                    if peer and peer.get('status') in ('cleaned', 'released'):
                        break
                    time.sleep(.01)
                report['peer_cleanup'] = peer
                if peer and peer.get('status') in ('cleaned', 'released'):
                    owner['peer_terminal'] = peer
                    dump(c.log / 'rewind-run-owner.json', owner)
                both = local_complete and bool(peer) and bool(peer.get('local_cleanup_complete'))
                report['cleanup_complete'] = both
                report['success'] = report['success'] and both and bool(peer.get('success'))
                report['functional_success'] = (report['functional_success'] and both
                                                 and bool(peer and peer.get('functional_success')))
                report['cache_qualified'] = report['cache_qualified'] and bool(peer and peer.get('cache_qualified'))
                if c.rank == 0:
                    # No API launch is a positive journal proof, not an empty-job inference.
                    audit_end = time.monotonic() + 60
                    while True:
                        audit = ray_cleanup_evidence(c, owner, report)
                        if audit.get('complete') or time.monotonic() > audit_end:
                            break
                        time.sleep(.25)
                    report['ray_cleanup'] = audit
                    report['cleanup_complete'] = both and bool(audit.get('complete'))
                    report['success'] = report['success'] and report['cleanup_complete']
                report['functional_success'] = report['functional_success'] and report['cleanup_complete']
                if c.rank == 0:
                    marker.update(status='released', cleanup_complete=report['cleanup_complete'],
                                  success=report['success'], ray_cleanup=report.get('ray_cleanup'),
                                  functional_success=report['functional_success'], cache_qualified=report['cache_qualified'])
                    end = time.monotonic() + 15
                    while time.monotonic() < end:
                        peer = peer_state(c)
                        if peer and peer.get('status') == 'released':
                            break
                        time.sleep(.01)
                else:
                    end = time.monotonic() + 240
                    while time.monotonic() < end:
                        peer = peer_state(c)
                        if peer and peer.get('status') == 'released':
                            break
                        time.sleep(.01)
                    if peer and peer.get('status') == 'released':
                        report['peer_cleanup'] = peer
                        owner['peer_terminal'] = peer
                        dump(c.log / 'rewind-run-owner.json', owner)
                    report['ray_cleanup'] = ray_cleanup_evidence(c, owner, report)
                    report['cleanup_complete'] = (local_complete and report['ray_cleanup']['complete']
                        and (both or owner.get('peer_api_authorized') is False))
                    report['success'] = report['success'] and report['cleanup_complete'] and bool(peer and peer.get('success'))
                    report['functional_success'] = (report['functional_success'] and report['cleanup_complete']
                                                     and bool(peer and peer.get('functional_success')))
                    marker.update(status='released', success=report['success'],
                                  cleanup_complete=report['cleanup_complete'],
                                  functional_success=report['functional_success'], cache_qualified=report['cache_qualified'])
                    time.sleep(.1)
            else:
                # Failure before publishing the barrier cannot authorize a peer API.
                report['ray_cleanup'] = ray_cleanup_evidence(c, owner, report)
                report['cleanup_complete'] = local_complete and report['ray_cleanup']['complete']
        except BaseException as exc:
            report.update(success=False, cleanup_complete=False, cleanup_error=repr(exc))
        finally:
            if marker:
                marker.close()
        # Even teardown/log-collection failures must retain actual after-state,
        # never silently substitute the seed bank or label an incomplete run qualified.
        try:
            if 'executable_artifacts_after' not in report:
                report['executable_artifacts_after'] = executable_artifacts(c, strict=False)
            evidence_key = 'tpu_init_evidence' if c.arm == 'seed' else 'warm_cache_evidence'
            if evidence_key not in report:
                paths = [c.generation / role / name for role in ('api', 'workers')
                         for name in ('output.log', 'backend.log')]
                paths += list((c.log / 'ray-workers').glob('*'))
                report[evidence_key] = warm_cache_evidence(c, paths, seed=c.arm == 'seed')
                report[evidence_key]['qualification_incomplete'] = True
                report.update(success=False, functional_success=False, cache_qualified=False)
        except Exception as exc:
            report.update(success=False, functional_success=False, cache_qualified=False,
                          final_cache_evidence_error=repr(exc))
        report['functional_success'] = report['functional_success'] and report['cleanup_complete']
        report['global_cache']['exists_after'] = Path('/dev/shm/torch_tpu_cache').exists()
        report['leases_must_stay_held'] = not report['cleanup_complete']
        if c.arm == 'seed' and report['success']:
            dump(c.pm / 'seed-receipt.json', {'config_hash': c.config_hash, 'success': True,
                 'run_id': c.run, 'model_id': c.model_id, 'terminal_snapshots': report['terminal_snapshots'],
                 'executable_artifacts': cache_contract})
        dump(c.log / 'rewind_measurement.json', report)
        dump(c.log / 'serve_result.json', {'ready': report['ready'], 'success': report['success'],
             'functional_success': report['functional_success'], 'cache_qualified': report['cache_qualified'],
             'cache_mode': c.cache_mode, 'comparison_scope': report['comparison_scope'],
             'ready_s': report.get('ready_s'), 'measured': report['measured']})
    return report


def qualify(c):
    """Real native restore/CPU actor proof in an isolated local Ray, never the shared cluster."""
    c.check_install()
    private(c.log, create=True)
    root = c.pm / 'qualifications' / c.run
    require(not root.exists() and not root.is_symlink(), 'qualification requires a new run/output directory')
    owner_path = c.log / 'rewind-qualification-owner.json'
    record = {'run_id': c.run, 'host_rank': c.rank, 'config_hash': c.config_hash,
              'owner': identity(), 'probe': None, 'output': str(root), 'extra_private_roots': {}}
    stable_write(owner_path, encoded(record))
    command = [str(c.python), str(c.src / 'pm_rewind_ray_probe.py'), '--python', str(c.python),
               '--dmtcp-bin', str(c.native / 'bin'), '--allocator',
               str(c.native / 'allocator/libtcmalloc_minimal.so.4'),
               '--out-dir', str(root), '--timeout', '180']
    env = {key: c.spec['env'][key] for key in ('PATH', 'HOME', 'LANG', 'PYTHONPATH',
                                              'PYTHONUNBUFFERED', 'PYTHONDONTWRITEBYTECODE', 'GCONV_PATH')}
    env.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0', CUDA_VISIBLE_DEVICES='')
    result = {'phase': 'rewind-qualify', 'run_id': c.run, 'host_rank': c.rank,
              'config_hash': c.config_hash, 'command': command, 'success': False,
              'cpu_only': True, 'shared_ray_used': False, 'cleanup_complete': False}
    proc = None
    try:
        CANCELLATION['deferred'] = True
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            with (c.log / 'qualification.log').open('xb') as output:
                proc = subprocess.Popen([str(c.python), str(c.src / 'pm_rewind_tpu.py'),
                    '_checked-parent-death', str(os.getpid()), *command], env=env, cwd=c.base,
                    stdout=output, stderr=subprocess.STDOUT)
            record['probe'] = identity(proc.pid)
            dump(owner_path, record)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
            CANCELLATION['deferred'] = False
            if CANCELLATION['pending']:
                CANCELLATION['pending'] = False
                raise KeyboardInterrupt('cancelled during qualification registration')
        proc.wait(timeout=float(os.environ.get('TTL_REWIND_QUALIFY_TIMEOUT', '3600')))
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                result['cleanup_error'] = 'probe still owns descendants; retain leases, do not kill ownership anchor'
        result['returncode'] = proc.poll() if proc is not None else None
        try:
            proof = load(root / 'report.json')
            dump(c.log / 'rewind-qualification.json', proof)
            evidence = c.log / 'qualification-evidence'
            private(evidence, create=True)
            for path in root.glob('*.log*'):
                if path.is_file() and not path.is_symlink():
                    shutil.copyfile(path, evidence / path.name)
            temp = Path(proof['private_temp']) if proof.get('private_temp') else None
            if temp is not None and not temp.is_relative_to(root):
                require(temp.parent == Path('/tmp') and temp.name.startswith('pmrr-')
                        and (root / 'private-temp').is_symlink()
                        and (root / 'private-temp').resolve() == temp, 'unknown qualification temp ownership')
                private(temp)
                info = temp.stat()
                record['extra_private_roots'][str(temp)] = {
                    'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid}
                dump(owner_path, record)
            cleanup = proof.get('probe_cleanup') or {}
            result['cleanup_complete'] = (proc is not None and proc.poll() is not None
                and cleanup.get('cleanup') == 'complete' and not cleanup.get('errors')
                and not proof.get('cleanup_errors'))
            for label, receipt in proof.get('receipts', {}).items():
                retired_supervisor(receipt, 'qualification/' + label)
            require(not any(alive(item['identity']) for item in proof.get('commands', [])
                            if item.get('identity')), 'qualification command still alive')
            result.update(probe_success=proof.get('success'), checks=proof.get('checks'),
                          receipts=proof.get('receipts'), probe_cleanup=cleanup,
                          report=str(c.log / 'rewind-qualification.json'),
                          extra_private_roots=record['extra_private_roots'])
            result['success'] = (proc.returncode == 0 and proof.get('success') is True
                                 and proof.get('cpu_only') is True and result['cleanup_complete']
                                 and not result.get('error'))
        except Exception as exc:
            result.update(success=False, cleanup_complete=False, evidence_error=repr(exc))
        result['leases_must_stay_held'] = not result['cleanup_complete']
        dump(c.log / 'rewind-qualification-result.json', result)
        if result['success']:
            dump(c.base / 'rewind-qualified.json', {'success': True, 'cleanup_complete': True,
                 'config_hash': c.config_hash, 'install_id': c.install_id, 'run_id': c.run,
                 'report': str(c.log / 'rewind-qualification.json')})
    return result


def cleanup(c):
    c.check_install()
    report_path = c.log / 'rewind_measurement.json'
    report = load(report_path) if report_path.exists() else {}
    owner_path = c.log / 'rewind-run-owner.json'
    if not owner_path.exists():
        proof_path = c.log / 'rewind-preflight-result.json'
        require(proof_path.is_file(), 'missing explicit pre-registration failure receipt')
        report = load(proof_path)
        require(report.get('launch_registration') == 'not_reached'
                and report.get('run_id') == c.run and report.get('host_rank') == c.rank
                and report.get('config_hash') == c.config_hash
                and report.get('preflight_owner') and not alive(report['preflight_owner'])
                and not c.generation.exists() and not c.generation.is_symlink(),
                'missing owner is not no-launch proof; preflight ownership or generation is unresolved')
        return {'phase': 'rewind-cleanup', 'success': True, 'run_id': c.run,
                'cleanup_complete': True, 'leases_must_stay_held': False, 'removed': [],
                'scope': 'recorded preflight failure before generation registration; no launch authorized'}
    owner = load(owner_path)
    require(owner['run_id'] == c.run and owner['config_hash'] == c.config_hash, 'owner record mismatch')
    require(not alive(owner['owner']) and not any(alive(p) for p in owner['controllers'].values()),
            'live owner/controller: refuse cleanup')
    require(owner['private_paths'] == [str(c.generation)], 'unexpected owner path list')
    try:
        report.update(run_id=c.run, host_rank=c.rank)
        report.update(local_cleanup_evidence(c, owner, report))
        require(report['local_cleanup_complete'], report['local_cleanup_reason'])
        report['ray_cleanup'] = ray_cleanup_evidence(c, owner, report)
        require(report['ray_cleanup'].get('complete'),
                f"exact-job Ray cleanup pending: {report['ray_cleanup']}")
        report.update(cleanup_complete=True, leases_must_stay_held=False,
                      cleanup_scope='this host owned generation and exact shared Ray job; run cleanup on BOTH hosts',
                      cleanup_revalidated_wall=time.time())
        report.pop('cleanup_error', None)
    except BaseException as exc:
        report.update(cleanup_complete=False, leases_must_stay_held=True,
                      cleanup_error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        dump(report_path, report)
    if c.generation.exists():
        private(c.generation)
        shutil.rmtree(c.generation)
    return {'phase': 'rewind-cleanup', 'success': True, 'run_id': c.run,
            'cleanup_complete': True, 'leases_must_stay_held': False,
            'scope': report['cleanup_scope'], 'proxy_cleanup': report['proxy_cleanup'],
            'ray_cleanup': report['ray_cleanup'],
            'removed': owner['private_paths'], 'retained': ['logs', 'images', 'PM state', 'model', 'venv'],
            'global_cache': 'never deleted; private state retirement is a separate explicit operation'}


def phase_lock_path(c):
    return c.base.parent / ('.' + c.base.name + '.rewind-phase.lock')


def phase_lock(c, *, create=True):
    """Exclude a new prepare/run while retirement validates and deletes owned roots."""
    private(c.base)
    root = c.base.stat()
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | (os.O_CREAT if create else 0)
    fd = os.open(phase_lock_path(c), flags, 0o600)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and not info.st_mode & 0o077, 'untrusted phase lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = c.base.stat()
        linked = phase_lock_path(c).lstat()
        require((root.st_dev, root.st_ino) == (current.st_dev, current.st_ino)
                and os.fstat(fd).st_nlink == 1
                and (info.st_dev, info.st_ino) == (linked.st_dev, linked.st_ino),
                'installation/phase lock changed while acquiring ownership')
        c.phase_lock_fd = fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def retired_supervisor(receipt, label):
    require(receipt and receipt.get('cleanup') == 'complete',
            f'incomplete supervisor: {label}: cleanup={(receipt or {}).get("cleanup")}, '
            f'reaper={(receipt or {}).get("reaper")}')
    require(isinstance(receipt.get('owned'), dict)
            and all(who.get('reaped') for who in receipt['owned'].values()),
            f'unreaped or unrecorded owned children: {label}')
    recorded = [receipt.get('reaper'), *receipt.get('owned', {}).values()]
    require(all(isinstance(who, dict) and type(who.get('pid')) is int and who['pid'] > 0
                and type(who.get('start_ticks')) is int and who['start_ticks'] > 0 for who in recorded),
            f'missing real process identity: {label}')
    remaining = [who for who in recorded if alive(who)]
    require(not remaining, f'live recorded processes: {label}: {remaining}')


def retired_advance_checks(c):
    """Audit every immutable generation, rebasing paths but never its receipts."""
    lineage = advance_lineage(c)
    if not lineage:
        return {}, {}
    roots, audits = retired_archive_checks(c, lineage, 'advance')
    revision = source_revision(c)
    if revision:
        more_roots, more_audits = retired_archive_checks(c, revision['revision'], 'revision')
        require(all(name not in roots or roots[name] == value for name, value in more_roots.items()),
                'conflicting historical/revision temp-root identities')
        roots.update(more_roots)
        audits.update(more_audits)
    refreshed = source_refresh(c)
    if refreshed:
        more_roots, more_audits = retired_archive_checks(c, refreshed['refresh'], 'refresh')
        require(all(name not in roots or roots[name] == value for name, value in more_roots.items()),
                'conflicting historical/refresh temp-root identities')
        roots.update(more_roots)
        audits.update(more_audits)
    return roots, audits


def retired_archive_checks(c, lineage, kind):
    """Read preserved receipts in their archive without rewriting original evidence."""
    require(kind in ('advance', 'revision', 'refresh'), 'unknown retired archive kind')
    archive = Path(lineage['archive'])
    require(archive.parent == c.base and archive.name == f'rewind-{kind}-archive-' + lineage['run_id'],
            'unknown retired archive root')
    private(archive)
    for name, sha in lineage['archive_files'].items():
        path = Path(name)
        require(path.is_relative_to(archive) and path.is_file() and not path.is_symlink()
                and digest(path) == sha, f'archived parent evidence changed: {path}')
    old = copy.copy(c)
    old.base = archive
    old.config_hash = lineage['parent_config_hash']
    old.spec = load(archive / 'rewind-owner.json')['config']
    old.state_address = old.spec['ray_state_address']
    def gone(who, label):
        require(isinstance(who, dict) and type(who.get('pid')) is int and who['pid'] > 0
                and type(who.get('start_ticks')) is int and who['start_ticks'] > 0
                and not alive(who), f'live/unknown archived process: {label}')
    for path in (archive / 'logs').glob('*/*owner.json'):
        owner = load(path)
        require(owner.get('config_hash') == old.config_hash, f'unknown archived owner: {path}')
        gone(owner.get('owner'), str(path))
        for key in ('probe', 'child', 'controller'):
            if owner.get(key) is not None:
                gone(owner[key], str(path) + '/' + key)
        for who in owner.get('controllers', {}).values():
            gone(who, str(path) + '/controller')
        if kind == 'refresh':
            require(path.name in ('rewind-qualification-owner.json', 'rewind-api-help-owner.json',
                                  'rewind-run-owner.json')
                    and owner.get('host_rank') == c.rank and owner.get('run_id') == path.parent.name,
                    f'unknown retired refresh ownership: {path}')
        if kind == 'revision' or (kind == 'refresh' and path.name == 'rewind-qualification-owner.json'):
            require(path.name == 'rewind-qualification-owner.json'
                    and owner.get('host_rank') == c.rank and owner.get('run_id') == path.parent.name
                    and owner.get('output') == str(c.pm / 'qualifications' / path.parent.name),
                    f'unknown retired revision ownership: {path}')
            result = load(path.parent / 'rewind-qualification-result.json')
            require((kind == 'refresh' or result.get('success') is False) and result.get('cleanup_complete')
                    and not result.get('leases_must_stay_held'),
                    f'retired revision qualification lacks closed ownership: {path}')
            for label, supervisor in (result.get('receipts') or {}).items():
                retired_supervisor(supervisor, str(path) + '/' + label)
            for name, expected in owner.get('extra_private_roots', {}).items():
                alias = Path(lineage['retired_qualifications']) / path.parent.name / 'private-temp'
                require(lineage['retired_extra_roots'].get(name) == expected
                        and alias.is_symlink() and alias.resolve() == Path(name),
                        f'retired qualification private-temp identity changed: {path}')
        if kind == 'refresh' and path.name == 'rewind-api-help-owner.json':
            help_dir = Path(lineage['retired_images']['archive']) / 'api' / ('help-' + path.parent.name)
            require(c.rank == 0 and owner.get('run_dir') == str(c.cache / 'api' / help_dir.name),
                    f'unknown archived API help ownership: {path}')
            result = load(path.parent / 'api-help-result.json')
            require(result.get('cleanup_complete') and not result.get('leases_must_stay_held'),
                    f'archived API help cleanup incomplete: {path}')
            retired_supervisor(load(help_dir / 'supervisor.json'), str(help_dir))
    for path in (archive / 'logs').glob('*/rewind-preflight-result.json'):
        gone(load(path).get('preflight_owner'), str(path))
    for path in (archive / 'logs').glob('*/rewind-qualification.json'):
        proof = load(path)
        for command in proof.get('commands', []):
            if command.get('identity'):
                gone(command['identity'], str(path) + '/command')
        for who in (proof.get('probe_cleanup') or {}).get('owned', {}).values():
            require(who.get('reaped'), f'unreaped archived probe child: {path}')
            gone(who, str(path) + '/probe-child')
        for label, supervisor in (proof.get('receipts') or {}).items():
            retired_supervisor(supervisor, str(path) + '/' + label)
    audits = {}
    for path in (archive / 'logs').glob('*/rewind-run-owner.json'):
        owner = load(path)
        report = load(path.parent / 'rewind_measurement.json')
        if kind == 'refresh':
            cleaned = load(path.parent / 'rewind-cleanup.json')
            require(report.get('success') is False and report.get('functional_success') is False
                    and not report.get('ready') and not report.get('cache_qualified')
                    and cleaned.get('success') and cleaned.get('phase') == 'rewind-cleanup'
                    and cleaned.get('run_id') == owner['run_id']
                    and owner.get('private_paths') == [str(c.base / 'runs' / owner['run_id'])],
                    f'archived refresh run lacks failed-serving/closed-generation proof: {path}')
        report.update(local_cleanup_evidence(old, owner, report, read_only=True))
        require(report['local_cleanup_complete'], f'archived local cleanup incomplete: {path}')
        audit = ray_cleanup_evidence(old, owner, report, read_only=True)
        require(audit.get('complete'), f'archived exact-job/PG audit incomplete: {path}')
        audits[kind + '/' + owner['run_id']] = audit
    extra_roots = lineage['retired_extra_roots']
    evidence_roots = [Path(lineage['retired_qualifications'])]
    if kind == 'advance':
        evidence_roots.append(Path(lineage['retired_images']['archive']))
    else:
        require(evidence_roots[0] == c.pm / (f'rewind-{kind}-qualifications-' + lineage['run_id'])
                and revision_tree(evidence_roots[0]) == load(archive / 'qualification-tree.json'),
                f'retired {kind} qualification bytes/inodes changed')
    if kind == 'refresh':
        image_root = Path(lineage['retired_images']['archive'])
        require(image_root == c.pm / ('rewind-refresh-images-' + lineage['run_id'])
                and lineage['retired_images']['original'] == str(c.cache), 'unknown retired refresh images')
        image_trees = load(archive / 'image-trees.json')
        require(set(image_trees) == ({'workers', 'api'} if c.rank == 0 else {'workers'})
                and {path.name for path in image_root.iterdir()} == set(image_trees),
                'retired refresh image inventory changed')
        for role, tree in image_trees.items():
            require(revision_tree(image_root / role) == tree, f'retired refresh {role} bytes/inodes changed')
            retired_supervisor(load(image_root / role / 'build/supervisor.json'), str(image_root / role))
        evidence_roots.append(image_root)
        log_trees = load(archive / 'log-trees.json')
        require({path.name for path in (archive / 'logs').iterdir()} == set(log_trees)
                and lineage['archived_logs'] == {
                    str(c.base / 'logs' / name): str(archive / 'logs' / name) for name in log_trees},
                'retired refresh log inventory changed')
        for name, tree in log_trees.items():
            require(revision_tree(archive / 'logs' / name) == tree, 'retired refresh log bytes/inodes changed')
        qowners = {path.parent.name for path in (archive / 'logs').glob('*/rewind-qualification-owner.json')}
        require({path.name for path in evidence_roots[0].iterdir()} == qowners,
                'unknown retired refresh qualification directories')
        help_runs = {path.parent.name for path in (archive / 'logs').glob('*/rewind-api-help-owner.json')}
        require({str(path.relative_to(image_root)) for path in image_root.glob('*/help-*')}
                == {'api/help-' + name for name in help_runs}, 'unknown retired refresh help directories')
    for root in evidence_roots:
        require(root.parent == c.pm, 'archived CPU evidence escaped PM root')
        private(root)
    if kind == 'advance':
        image_root = Path(lineage['retired_images']['archive'])
        for name, sha in lineage['retired_images']['files'].items():
            path = image_root / name
            require(path.is_file() and not path.is_symlink() and digest(path) == sha,
                    f'retired image evidence changed: {path}')
        for name, target in lineage['retired_images']['symlinks'].items():
            path = image_root / name
            require(path.is_symlink() and os.readlink(path) == target,
                    f'retired DMTCP restart alias changed: {path}')
    for name, expected in extra_roots.items():
        root = Path(name)
        require(root.parent == Path('/tmp') and root.name.startswith('pmrr-')
                and file_identity(root) == expected, f'archived qualification temp replaced: {root}')
        private(root)
        device = root.stat().st_dev
        require(not os.path.ismount(root), f'archived temp is a mount: {root}')
        for directory, dirs, _ in os.walk(root, followlinks=False):
            for name in dirs:
                child = Path(directory) / name
                require(child.is_symlink() or (child.stat().st_dev == device and not os.path.ismount(child)),
                        f'nested archived temp mount: {child}')
        evidence_roots.append(root)
    for root in evidence_roots:
        for path in root.rglob('supervisor.json'):
            require(not path.is_symlink(), f'unsafe archived supervisor: {path}')
            retired_supervisor(load(path), str(path))
    return dict(extra_roots), audits


def retirement_checks(c, *, read_only=False):
    """Validate active work and all archives; transitions use the read-only path."""
    installed = c.check_install()
    owner = load(c.base / 'rewind-owner.json')
    roots = [c.pm, c.base]
    require(owner.get('retirement_roots') == [str(path) for path in roots],
            'installation does not authorize whole-root retirement')
    require(owner.get('phase_lock') == str(phase_lock_path(c))
            and getattr(c, 'phase_lock_fd', None) is not None, 'missing retirement lock authority')
    require(set(installed.get('retirement_roots', {})) == {str(path) for path in roots},
            'missing root identity receipts')
    for root in roots:
        private(root)
        info = root.stat()
        require(installed['retirement_roots'][str(root)] ==
                {'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid},
                f'owned root was replaced: {root}')
        # A nested mount would escape the installation's deletion authority.
        for directory, dirs, _ in os.walk(root, followlinks=False):
            for name in dirs:
                child = Path(directory) / name
                if not child.is_symlink():
                    require(child.stat().st_dev == info.st_dev and not os.path.ismount(child),
                            f'nested mount refused: {child}')
    completed = []
    parent_extra_roots, audits = retired_advance_checks(c)
    for path in sorted((c.base / 'logs').glob('*/rewind-preflight-result.json')):
        if (path.parent / 'rewind-run-owner.json').exists():
            continue
        failed = load(path)
        generation = c.base / 'runs' / path.parent.name
        require(failed.get('launch_registration') == 'not_reached'
                and failed.get('run_id') == path.parent.name and failed.get('host_rank') == c.rank
                and failed.get('config_hash') == c.config_hash
                and failed.get('preflight_owner') and not alive(failed['preflight_owner'])
                and not generation.exists() and not generation.is_symlink(),
                f'preflight ownership remains unresolved: {path.parent.name}')
    for owner_path in sorted((c.base / 'logs').glob('*/rewind-run-owner.json')):
        run_owner = load(owner_path)
        run_id = owner_path.parent.name
        require(run_owner.get('run_id') == run_id and run_owner.get('host_rank') == c.rank
                and run_owner.get('config_hash') == c.config_hash, f'unknown run ownership: {owner_path}')
        require(not alive(run_owner['owner'])
                and not any(alive(who) for who in run_owner['controllers'].values()),
                f'live generation owner/controller: {run_id}')
        measurement = load(owner_path.parent / 'rewind_measurement.json')
        cleaned = load(owner_path.parent / 'rewind-cleanup.json')
        require(measurement.get('run_id') == run_id
                and cleaned.get('phase') == 'rewind-cleanup' and cleaned.get('run_id') == run_id
                and cleaned.get('success'), f'run lacks generation removal receipt: {run_id}')
        local = local_cleanup_evidence(c, run_owner, measurement, read_only=read_only)
        require(local['local_cleanup_complete'], f'{run_id}: {local["local_cleanup_reason"]}')
        measurement.update(local)
        audit = ray_cleanup_evidence(c, run_owner, measurement, read_only=read_only)
        require(audit.get('complete'), f'fresh Ray cleanup remains pending: {run_id}: {audit}')
        key = audit.get('job_id') or ('no-ray/' + run_id)
        require(key not in audits or audits[key]['recorded_pg_ids'] == audit['recorded_pg_ids'],
                f'conflicting exact-job PG receipts: {key}')
        audits[key] = audit
        completed.append(run_id)
    qualification_roots = set()
    extra_roots = dict(parent_extra_roots)
    for path in sorted((c.base / 'logs').glob('*/rewind-qualification-owner.json')):
        qualified = load(path)
        qrun = path.parent.name
        qroot = c.pm / 'qualifications' / qrun
        require(qualified.get('run_id') == qrun and qualified.get('host_rank') == c.rank
                and qualified.get('config_hash') == c.config_hash
                and qualified.get('output') == str(qroot), 'unknown qualification ownership')
        require(not alive(qualified['owner'])
                and (qualified.get('probe') is None or not alive(qualified['probe'])),
                f'live qualification owner: {qrun}')
        qreport = load(path.parent / 'rewind-qualification-result.json')
        require(qreport.get('cleanup_complete') and not qreport.get('leases_must_stay_held'),
                f'qualification cleanup unproven: {qrun}')
        for label, supervisor in (qreport.get('receipts') or {}).items():
            retired_supervisor(supervisor, f'{qrun}/{label}')
        proof = load(path.parent / 'rewind-qualification.json')
        require(not any(alive(command['identity']) for command in proof.get('commands', [])
                        if command.get('identity')), f'live qualification command: {qrun}')
        for who in (proof.get('probe_cleanup') or {}).get('owned', {}).values():
            require(who.get('reaped') and not alive(who), f'live/unreaped qualification child: {qrun}')
        qualification_roots.add(str(qroot))
        for name, expected in qualified.get('extra_private_roots', {}).items():
            temp = Path(name)
            require(temp.parent == Path('/tmp') and temp.name.startswith('pmrr-')
                    and (qroot / 'private-temp').is_symlink()
                    and (qroot / 'private-temp').resolve() == temp, 'unrecognized qualification temp')
            private(temp)
            info = temp.stat()
            require(expected == {'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid},
                    f'qualification temp root was replaced: {temp}')
            for directory, dirs, _ in os.walk(temp, followlinks=False):
                for name in dirs:
                    child = Path(directory) / name
                    require(child.is_symlink() or
                            (child.stat().st_dev == info.st_dev and not os.path.ismount(child)),
                            f'nested qualification mount refused: {child}')
            extra_roots[str(temp)] = expected
    require({str(path) for path in (c.pm / 'qualifications').iterdir()} == qualification_roots,
            'unknown qualification output directories remain')
    roots = [*(Path(path) for path in sorted(extra_roots)), *roots]
    copies = []
    for path in sorted((c.base / 'logs').glob('*/rewind-copy-owner.json')):
        copied = load(path)
        crun = path.parent.name
        require(copied.get('run_id') == crun and copied.get('host_rank') == c.rank
                and copied.get('config_hash') == c.config_hash
                and copied.get('destination') == str(c.ckpt), 'unknown model-copy ownership')
        require(not alive(copied['owner']) and
                (copied.get('child') is None or not alive(copied['child'])), f'live model copy: {crun}')
        copy_result = load(path.parent / 'rewind-copy-result.json')
        require(copy_result.get('cleanup_complete') and not copy_result.get('leases_must_stay_held'),
                f'copy cleanup unproven: {crun}')
        info = c.ckpt.stat()
        require(copied['destination_identity'] ==
                {'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid},
                'owned copy destination was replaced')
        copies.append(crun)
    require(not any((c.base / 'runs').iterdir()), 'uncleaned or unknown generation directories remain')
    help_runs = set()
    for path in sorted((c.base / 'logs').glob('*/rewind-api-help-owner.json')):
        help_owner = load(path)
        hrun = path.parent.name
        help_dir = c.cache / 'api' / ('help-' + hrun)
        require(help_owner.get('run_id') == hrun and help_owner.get('host_rank') == 0
                and c.rank == 0 and help_owner.get('config_hash') == c.config_hash
                and help_owner.get('run_dir') == str(help_dir), 'unknown API help ownership')
        require(not alive(help_owner['owner']) and
                (help_owner.get('controller') is None or not alive(help_owner['controller'])),
                f'live API help owner/controller: {hrun}')
        help_result = load(path.parent / 'api-help-result.json')
        require(help_result.get('cleanup_complete') and not help_result.get('leases_must_stay_held'),
                f'API help cleanup unproven: {hrun}')
        retired_supervisor(load(help_dir / 'supervisor.json'), f'API help/{hrun}')
        help_runs.add(str(help_dir))
    require({str(path) for path in c.cache.glob('*/help-*')} == help_runs,
            'unknown API help generation directories remain')
    for image in c.cache.iterdir():
        require(image.is_dir() and not image.is_symlink() and image.name in ('workers', 'api'),
                f'unknown image path: {image}')
        retired_supervisor(load(image / 'build/supervisor.json'), str(image))
    require(shutil.rmtree.avoids_symlink_attacks, 'symlink-safe rmtree required')
    receipt = {'phase': 'rewind-retire', 'run_id': c.run, 'host_rank': c.rank,
               'config_hash': c.config_hash, 'source_hashes': installed['source_hashes'],
               'authorized_roots': {**installed['retirement_roots'], **extra_roots},
               'completed_runs': completed, 'qualifications': sorted(qualification_roots), 'copies': copies,
               'ray_audits': audits, 'phase_lock': owner['phase_lock'],
               'global_cache': 'not touched', 'deleted': []}
    if installed.get('source_revision'):
        revision = installed['source_revision']
        receipt['source_revision_run_id'] = revision['run_id']
        receipt['retired_revision_evidence'] = [
            revision['archive'], revision['retired_qualifications'],
            str(c.base / ('rewind-revision-parent-install-' + revision['run_id'] + '.json'))]
    if installed.get('source_refresh'):
        refreshed = installed['source_refresh']
        receipt['source_refresh_run_id'] = refreshed['run_id']
        receipt['retired_refresh_evidence'] = [
            refreshed['archive'], refreshed['retired_qualifications'], refreshed['retired_images']['archive'],
            str(c.base / ('rewind-refresh-parent-install-' + refreshed['run_id'] + '.json'))]
    return receipt, roots


def retire(c):
    """Delete exactly the explicitly owned roots after fresh retirement checks."""
    receipt, roots = retirement_checks(c)
    # The caller must archive stdout before invoking this phase; no local archive
    # survives whole-install retirement. Emit authority before the first deletion.
    print('REWIND_RETIRE_AUTHORIZATION_JSON=' + json.dumps(receipt, sort_keys=True, separators=(',', ':')),
          flush=True)
    try:
        for root in roots:
            shutil.rmtree(root)
            receipt['deleted'].append(str(root))
        # Keep the sibling lock linked for the entire root deletion. An opener
        # delayed until after unlink must reject its stale fd/root identity.
        locked = os.fstat(c.phase_lock_fd)
        linked = phase_lock_path(c).lstat()
        require(not c.base.exists() and (locked.st_dev, locked.st_ino) == (linked.st_dev, linked.st_ino),
                'root/lock replaced during retirement; retain sibling lock')
        phase_lock_path(c).unlink()
        receipt['deleted'].append(str(phase_lock_path(c)))
    except BaseException as exc:
        receipt.update(success=False, leases_must_stay_held=True, error=repr(exc))
        return receipt
    receipt.update(success=True, cleanup_complete=True, leases_must_stay_held=False)
    return receipt


def main():
    os.umask(0o077)
    sys.dont_write_bytecode = True
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if len(sys.argv) > 1 and sys.argv[1] == '_checked-parent-death':
        return checked_parent_death(sys.argv[2], sys.argv[3:])
    if len(sys.argv) > 1 and sys.argv[1] == '_fresh-controller':
        return fresh_controller(sys.argv[2])
    if len(sys.argv) > 1 and sys.argv[1] == '_ray-audit':
        p = argparse.ArgumentParser(prog=sys.argv[0] + ' _ray-audit')
        p.add_argument('job')
        p.add_argument('state_address')
        p.add_argument('destination')
        p.add_argument('gcs_address')
        a = p.parse_args(sys.argv[2:])
        return ray_audit(a.job, a.state_address, a.destination, a.gcs_address)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('rewind-install', 'rewind-advance', 'rewind-revise', 'rewind-refresh',
                                  'rewind-qualify', 'rewind-prepare', 'rewind-copy', 'rewind-run',
                                  'rewind-collect', 'rewind-cleanup', 'rewind-retire'))
    a = p.parse_args()
    def cancel(*_):
        if CANCELLATION['deferred']:
            CANCELLATION['pending'] = True
            return
        raise KeyboardInterrupt('host task cancelled')
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(1, signal.SIGTERM, 0, 0, 0) == 0 and os.getppid() == parent,
            'could not establish parent-death cancellation')
    c = None
    lock_fd = None
    transition_ready = False
    try:
        c = Config()
        transitions = ('rewind-advance', 'rewind-revise', 'rewind-refresh')
        lock_fd = phase_lock(c, create=a.mode not in transitions)
        source_revision(c)
        if a.mode == 'rewind-install' and any(os.path.lexists(path) for path in (
                c.base / 'rewind-owner.json', c.base / 'rewind-install.json', c.src / 'pm_rewind_tpu.py')):
            require(Path(__file__).resolve() == c.src / 'pm_rewind_tpu.py',
                    'incoming install is bootstrap-only; use rewind-advance for an existing installation')
        if a.mode not in transitions:
            advance_lineage(c)
        transition_ready = True
        if a.mode not in ('rewind-install', *transitions):
            require(Path(__file__).resolve() == c.src / 'pm_rewind_tpu.py',
                    'ordinary operations must execute the sealed installed helper')
            c.check_install()
        if a.mode == 'rewind-collect':
            report = load(c.log / 'rewind_measurement.json')
        else:
            report = {'rewind-install': install, 'rewind-advance': advance, 'rewind-revise': revise,
                      'rewind-refresh': refresh,
                      'rewind-qualify': qualify, 'rewind-prepare': prepare, 'rewind-copy': copy_model,
                      'rewind-run': run, 'rewind-cleanup': cleanup, 'rewind-retire': retire}[a.mode](c)
        report.setdefault('run_id', c.run)
        report.setdefault('host_rank', c.rank)
        if a.mode not in ('rewind-collect', 'rewind-retire', *transitions):
            if not c.log.exists():
                if not (c.base / 'logs').exists():
                    private(c.base / 'logs', create=True)
                private(c.log, create=True)
            # An idempotent install must not erase the successful reset receipt
            # embedded in its measurement (or a qualification's existing evidence).
            if not (report.get('unchanged') and (c.log / 'rewind_measurement.json').exists()):
                dump(c.log / ('rewind_measurement.json' if a.mode != 'rewind-cleanup' else 'rewind-cleanup.json'), report)
    except BaseException as exc:
        report = {'phase': a.mode, 'success': False, 'error': f'{type(exc).__name__}: {exc}',
                  'leases_must_stay_held': a.mode in ('rewind-install', 'rewind-advance', 'rewind-revise', 'rewind-refresh',
                      'rewind-prepare', 'rewind-copy', 'rewind-qualify', 'rewind-run', 'rewind-cleanup', 'rewind-retire')}
        if c is not None and hasattr(c, 'retirement_query_upgrade'):
            report['retirement_query_upgrade'] = c.retirement_query_upgrade
        if c is not None and hasattr(c, 'reset_receipt'):
            report.update(reset=c.reset_receipt, reset_receipt_path=c.reset_receipt.get('receipt_path'),
                          leases_must_stay_held=True)
        if (a.mode == 'rewind-run' and c is not None and getattr(c, 'run_launch_registered', None) is False
                and not (c.log / 'rewind-run-owner.json').exists()
                and not c.generation.exists() and not c.generation.is_symlink()):
            report.update(run_id=c.run, host_rank=c.rank, config_hash=c.config_hash,
                          preflight_owner=identity(), launch_registration='not_reached',
                          cleanup_complete=True, leases_must_stay_held=False,
                          cache_mode=getattr(c, 'cache_mode', os.environ.get('TTL_REWIND_CACHE_MODE', 'strict')),
                          functional_success=False, cache_qualified=False, measured=False,
                          cleanup_scope='preflight failed before any launch authorization or generation registration')
        # Rejected transition preconditions must not touch any old log generation.
        if c and transition_ready and c.log.is_dir() and a.mode not in ('rewind-advance', 'rewind-revise', 'rewind-refresh'):
            if report.get('launch_registration') == 'not_reached':
                dump(c.log / 'rewind-preflight-result.json', report)
            dump(c.log / 'rewind-phase-error.json', report)
            if a.mode == 'rewind-run' or not (c.log / 'rewind_measurement.json').exists():
                dump(c.log / 'rewind_measurement.json', report)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
    print('REWIND_RESULT_JSON=' + json.dumps(report, sort_keys=True, separators=(',', ':')), flush=True)
    return 0 if report.get('success') else 1


if __name__ == '__main__':
    sys.exit(main())
