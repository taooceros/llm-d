#!/usr/bin/env python3
"""Native-JAX host<->TPU snapshot bandwidth benchmark (synthetic BF16, no checkpoint I/O).

Payload = the real S-layout TP8 per-chip shard shapes of Gemma-4-31B: 1028 arrays,
7,862,118,616 B/chip, 31,448,474,464 B per 4-chip host (identical to the byte count of
results/restore/ShardedRestore/S-snapshot.json). Shapes come from the measured parameter
catalogue (path + global shape + PartitionSpec); the 'model' axis is divided by TP=8.

Cluster run (rcm_exec only, serialised on the cluster lock, never detached):
  flock /tmp/hetero-cluster.lock .venv/bin/python rcm_exec run scripts/snapshot_bandwidth.py \
      --inventory /tmp/hetero-inventory-only.json --catalogue <parameter_catalogue.json> \
      [--topology auto|host4|tp8] [--phases inspect,d2h,d2h_large,h2d,h2d_large,host_copy] \
      [--output results/restore/FastSnapshot/bandwidth.json]
  The launcher stages the inventory and catalogue as FILES under _snapshot_bandwidth_inputs/
  inside the job working_dir (never large env vars) and removes them when the stage ends.
  Topology 'auto' first binds ONE host's 4 chips alone (single-host subslice env, no
  jax.distributed); if that cannot initialise, it falls back to the two 4-chip hosts of one
  TP8 S instance (jax.distributed, like scripts/ici_bench.py).  Reservations (placement
  group, actors) and every temp file die with the job (finally).

Local CPU smoke (never touches the cluster; reduced bytes are refused on the cluster path):
  .venv/bin/python scripts/snapshot_bandwidth.py --smoke-bytes 32MiB \
      --catalogue <parameter_catalogue.json> --output /tmp/bw_smoke.json

Method guards (every sample): fresh device output (jitted generator, readiness outside the
timer); transfer completion inside the timer; host bytes proven equal to an independently
computed expectation (memcmp vs a tiled numpy pattern) / device bytes proven by an on-device
sum of bits plus an on-device regenerated-pattern mismatch count; host cache state labelled
(cold = no jax ndarray cache; warm = repeat of np.asarray, never used to gate).
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
from pathlib import Path
import shlex
import shutil
import socket
import sys
import threading
import time
import traceback
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = 'snapshot_bandwidth/1'
TP, CHIPS = 8, 4
EXPECTED_CHIP_BYTES = 7_862_118_616
EXPECTED_HOST_BYTES = 31_448_474_464
PERIOD = 127                       # synthetic values are (flat_index + salt) % 127, exact in bf16
BLOCK_ELEMS = PERIOD * 32768       # tiled expectation block (8.3 MB of bf16)
CHUNK_ELEMS = 1 << 21              # host verification/fill chunk (4 MiB)
LARGE_COLUMNS = 8192
DEFAULT_CATALOGUES = (
    ROOT / 'results/restore/FastSnapshot/parameter_catalogue.json',
    ROOT / 'results/ici/ICIBench/parameter_catalogue.json',
    Path('/home/hongtao/hetero-wt/ICIBench/results/ici/ICIBench/parameter_catalogue.json'),
)
DEFAULT_OUTPUT = ROOT / 'results/restore/FastSnapshot/bandwidth.json'
INPUT_DIR = '_snapshot_bandwidth_inputs'
PHASES = ('inspect', 'd2h', 'd2h_large', 'h2d', 'h2d_large', 'host_copy')


# --------------------------------------------------------------------------- utilities
def parse_size(text):
    text = str(text).strip()
    for suffix, scale in (('GiB', 2**30), ('MiB', 2**20), ('KiB', 2**10), ('B', 1)):
        if text.endswith(suffix):
            return int(float(text[:-len(suffix)]) * scale)
    return int(text)


def meminfo():
    rows = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        rows[key] = int(value.split()[0]) * 1024
    return rows


def mem_available():
    return meminfo()['MemAvailable']


def vm_hwm():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) * 1024
    return None


class MemoryWatch:
    """Minimum MemAvailable (host headroom) and peak RSS observed inside a region."""

    def __init__(self, interval=0.2):
        self.interval, self.min_available, self.stop = interval, None, threading.Event()

    def _loop(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def _sample(self):
        value = mem_available()
        self.min_available = value if self.min_available is None else min(self.min_available, value)

    def __enter__(self):
        try:  # reset the kernel's peak-RSS watermark so VmHWM is per region
            Path('/proc/self/clear_refs').write_text('5')
        except OSError:
            pass
        self.start_available = mem_available()
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self._sample()

    def report(self):
        return {'mem_available_start_bytes': self.start_available,
                'mem_available_min_bytes': self.min_available,
                'peak_rss_bytes': vm_hwm()}


def par_map(function, items, workers):
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(min(workers, len(items))) as pool:
        return list(pool.map(function, items))


def stats(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {'n': len(ordered), 'max': ordered[-1], 'median': median, 'min': ordered[0], 'first': values[0]}


def exception_record(exc):
    return {'type': type(exc).__name__, 'message': str(exc)[:2000],
            'traceback': traceback.format_exc()[-4000:]}


# --------------------------------------------------------------------------- payload
def resolve_input(path):
    """Input files are relative to the job working_dir (ROOT); absolute paths pass through."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_catalogue(path=None, raw=None):
    """`raw` = bytes already read by the launcher/driver; else read `path` or a default file."""
    if raw is not None:
        source = f'file:{path}'
    else:
        candidates = [Path(path)] if path else list(DEFAULT_CATALOGUES)
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise FileNotFoundError('parameter catalogue not found; tried ' + ', '.join(map(str, candidates)))
        raw, source = found.read_bytes(), str(found)
    value = json.loads(raw)
    import ast
    arrays = []
    for index, row in enumerate(value['parameters']):
        if row['dtype'] != 'bfloat16':
            raise ValueError(f"{row['path']}: catalogue dtype {row['dtype']} is not bfloat16")
        spec = row['partition_spec']
        # PartitionSpec repr holds only tuple/None/str literals; never eval.
        parts = ast.literal_eval('(' + spec[2:-1] + ')')
        parts = parts if isinstance(parts, tuple) else (parts,)
        shape = list(row['shape'])
        for axis, part in enumerate(parts):
            if part == 'model':
                if shape[axis] % TP:
                    raise ValueError(f"{row['path']}: dim {axis}={shape[axis]} not divisible by TP={TP}")
                shape[axis] //= TP
            elif part is not None:
                raise ValueError(f"{row['path']}: unsupported partition axis {part!r}")
        arrays.append({'index': index, 'path': row['path'], 'global_shape': row['shape'],
                       'partition_spec': spec, 'shape': tuple(shape), 'bytes': math.prod(shape) * 2})
    return {'source': source, 'sha256': hashlib.sha256(raw).hexdigest(), 'arrays': arrays,
            'evidence_file': value.get('evidence_file'), 'run_id': value.get('run_id')}


def smoke_subset(arrays, target_bytes):
    """Deterministic stride subset of the REAL catalogue rows (no invented shapes)."""
    small = [a for a in arrays if a['bytes'] <= max(target_bytes // 4, 2)] or arrays[:1]
    total = sum(a['bytes'] for a in small)
    stride = max(1, math.ceil(total / max(target_bytes, 1)))
    return small[::stride]


def large_workload(nbytes):
    elements = nbytes // 2
    shape = (elements // LARGE_COLUMNS, LARGE_COLUMNS) if elements % LARGE_COLUMNS == 0 else (elements,)
    return [{'index': 0, 'path': f'synthetic_large_{nbytes}', 'shape': shape, 'bytes': nbytes,
             'global_shape': list(shape), 'partition_spec': 'P()'}]


def salt(seed, chip, index):
    return (seed * 13 + index * 31 + chip * 17) % PERIOD


_block = None


def block():
    global _block
    if _block is None:
        import ml_dtypes
        _block = (np.arange(BLOCK_ELEMS, dtype=np.uint32) % PERIOD).astype(ml_dtypes.bfloat16).view(np.uint8)
    return _block


def expected_view(start, count, salt_value):
    offset = (start + salt_value) % PERIOD
    return block()[2 * offset: 2 * (offset + count)]


def fill_u8(destination, elements, salt_value):
    for start in range(0, elements, CHUNK_ELEMS):
        count = min(CHUNK_ELEMS, elements - start)
        destination[2 * start: 2 * (start + count)] = expected_view(start, count, salt_value)


def u8(array):
    return array.reshape(-1).view(np.uint8)


_libc = ctypes.CDLL(None)
_libc.memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_libc.memcmp.restype = ctypes.c_int


def pointer(array):
    return array.__array_interface__['data'][0]


def memeq(a, b):
    return a.nbytes == b.nbytes and _libc.memcmp(pointer(a), pointer(b), a.nbytes) == 0


def trim_heap():
    """Collect and hand freed glibc heap back to the OS so MemAvailable reflects released host arrays."""
    gc.collect()
    try:
        _libc.malloc_trim(0)
    except (AttributeError, OSError):
        pass


def compare_pattern(host, shape, salt_value):
    """None if the LOGICAL (C-order) bytes of `host` equal the expected pattern, else a failure row.

    The runtime may hand back a legal non-C-contiguous ndarray; its logical content is checked via a
    C-contiguous copy (run after the timer, never inside it)."""
    elements = math.prod(shape)
    if tuple(host.shape) != tuple(shape) or host.dtype.itemsize != 2:
        return {'reason': 'shape/dtype', 'shape': list(host.shape), 'dtype': str(host.dtype)}
    flat = u8(host if host.flags.c_contiguous else np.ascontiguousarray(host))
    for start in range(0, elements, CHUNK_ELEMS):
        count = min(CHUNK_ELEMS, elements - start)
        got, want = flat[2 * start: 2 * (start + count)], expected_view(start, count, salt_value)
        if not memeq(got, want):
            diff = np.nonzero(got != want)[0]
            return {'reason': 'bytes differ', 'first_byte': 2 * start + int(diff[0]),
                    'differing_bytes_in_chunk': int(diff.size)}
    return None


def verify_hosts(hosts, workload, seed, workers):
    """Independent logical byte equality of host arrays against the numpy-tiled expectation.

    Layout is reported, not judged: non_c_contiguous counts arrays whose memory layout is not C order."""
    jobs = [(c, i, h) for c, arrays in hosts.items() for i, h in enumerate(arrays)]

    def one(job):
        c, i, h = job
        return (c, i, compare_pattern(h, workload[i]['shape'], salt(seed, c, i)))
    bad = [row for row in par_map(one, jobs, min(workers, 8)) if row[2]]
    odd = [(c, i, h) for c, i, h in jobs if not h.flags.c_contiguous]
    return {'ok': not bad and all(len(a) == len(workload) for a in hosts.values()),
            'arrays': len(jobs), 'bytes': sum(h.nbytes for _, _, h in jobs),
            'non_c_contiguous': len(odd), 'c_contiguous': len(jobs) - len(odd),
            'non_c_examples': [{'chip': c, 'index': i, 'shape': list(h.shape), 'strides': list(h.strides),
                                'f_contiguous': bool(h.flags.f_contiguous)} for c, i, h in odd[:3]],
            'bad': [{'chip': c, 'index': i, **row} for c, i, row in bad[:5]], 'bad_count': len(bad)}


def host_bit_sums(arrays, workers):
    """Sum of bf16 bit patterns (mod 2^32) per array, chunk-parallel, computed on the host."""
    span = 1 << 23
    tasks = []
    for index, array in enumerate(arrays):
        flat = u8(array).view(np.uint16)
        tasks += [(index, flat[s: s + span]) for s in range(0, flat.size, span)]
    totals = [0] * len(arrays)
    for (index, _), part in zip(tasks, par_map(lambda t: int(np.sum(t[1], dtype=np.uint64)), tasks, workers)):
        totals[index] += part
    return [t & 0xFFFFFFFF for t in totals]


def partition(units, workers):
    total = sum(u[2] for u in units) or 1
    target, groups, acc = total / workers, [[]], 0
    for unit in units:
        if len(groups) < workers and acc >= target * len(groups):
            groups.append([])
        groups[-1].append(unit)
        acc += unit[2]
    return groups


# --------------------------------------------------------------------------- digests
def digest_registry():
    """name -> {'fn': callable|None, 'available': bool, ...}; missing libs are recorded, never substituted."""
    registry = {'none': {'fn': None, 'available': True},
                'sha256': {'fn': lambda mv: hashlib.sha256(mv).digest(), 'available': True,
                           'impl': 'hashlib ' + hashlib.sha256.__name__},
                'zlib_crc32': {'fn': lambda mv: zlib.crc32(mv), 'available': True, 'impl': 'zlib.crc32 (stdlib)'}}
    try:
        import xxhash
        registry['xxhash'] = {'fn': lambda mv: xxhash.xxh3_64(mv).intdigest(), 'available': True,
                              'impl': f'xxhash {xxhash.VERSION} xxh3_64'}
    except Exception as exc:
        registry['xxhash'] = {'fn': None, 'available': False, 'error': f'{type(exc).__name__}: {exc}'}
    try:
        import crc32c
        registry['crc32c'] = {'fn': lambda mv: crc32c.crc32c(mv), 'available': True, 'input_copy_required': False,
                              'impl': f"crc32c {importlib.metadata.version('crc32c')}"}
    except Exception as exc:
        try:
            import google_crc32c
            # google_crc32c.value() accepts only bytes (not ndarray/memoryview): the bytes() copy is part of
            # the timed digest cost and is reported via input_copy_required.
            registry['crc32c'] = {'fn': lambda mv: google_crc32c.value(bytes(mv)), 'available': True,
                                  'input_copy_required': True,
                                  'impl': f"google_crc32c {importlib.metadata.version('google-crc32c')} "
                                          f"({google_crc32c.implementation})"}
        except Exception as exc2:
            registry['crc32c'] = {'fn': None, 'available': False,
                                  'error': f'crc32c: {type(exc).__name__}: {exc}; google_crc32c: '
                                           f'{type(exc2).__name__}: {exc2}'}
    return registry


# --------------------------------------------------------------------------- D2H mechanisms
# Each returns (hosts: {chip: [ndarray]}, extra: dict). '_keep' in extra lists device
# objects to delete after verification.
def flat_items(tree, chips):
    return [(c, i, a) for c in chips for i, a in enumerate(tree[c])]


def timed(function, argument):
    start = time.perf_counter()
    value = function(argument)
    return value, time.perf_counter() - start


def per_chip(chips, function):
    with ThreadPoolExecutor(len(chips)) as pool:
        futures = {c: pool.submit(timed, function, c) for c in chips}
        rows = {c: f.result() for c, f in futures.items()}
    return {c: r[0] for c, r in rows.items()}, {'chip_elapsed_s': {str(c): r[1] for c, r in rows.items()}}


def d2h_serial_asarray(rt, tree, chips):
    return {c: [np.asarray(a) for a in tree[c]] for c in chips}, {}


def d2h_async_all_asarray(rt, tree, chips):
    for _, _, a in flat_items(tree, chips):
        a.copy_to_host_async()
    return {c: [np.asarray(a) for a in tree[c]] for c in chips}, {}


def collect_did_copy(arrays):
    values, flags = [], [0, 0]
    for a in arrays:
        value, did = a._single_device_array_to_np_array_did_copy()
        values.append(value)
        flags[bool(did)] += 1
    return values, flags


def d2h_async_did_copy(rt, tree, chips):
    for _, _, a in flat_items(tree, chips):
        a.copy_to_host_async()
    hosts, counts = {}, [0, 0]
    for c in chips:
        hosts[c], flags = collect_did_copy(tree[c])
        counts = [counts[0] + flags[0], counts[1] + flags[1]]
    return hosts, {'did_copy_false': counts[0], 'did_copy_true': counts[1]}


def d2h_device_get_tree(rt, tree, chips):
    got = rt.jax.device_get({c: tree[c] for c in chips})
    return {c: list(got[c]) for c in chips}, {}


def d2h_chip_threads_asarray(rt, tree, chips):
    return per_chip(chips, lambda c: [np.asarray(a) for a in tree[c]])


def d2h_chip_threads_async_asarray(rt, tree, chips):
    def work(c):
        for a in tree[c]:
            a.copy_to_host_async()
        return [np.asarray(a) for a in tree[c]]
    return per_chip(chips, work)


def d2h_chip_threads_did_copy(rt, tree, chips):
    def work(c):
        for a in tree[c]:
            a.copy_to_host_async()
        return collect_did_copy(tree[c])
    rows, extra = per_chip(chips, work)
    flags = [sum(rows[c][1][k] for c in chips) for k in (0, 1)]
    extra.update(did_copy_false=flags[0], did_copy_true=flags[1])
    return {c: rows[c][0] for c in chips}, extra


# Production-route mechanisms: C-order materialization is INSIDE the timer, exactly as the writer does
# (hetero/weight_snapshot.py: np.ascontiguousarray(np.asarray(shard.data)).reshape(shard.data.shape)).
# The raw np.asarray mechanisms above can return legal non-C-contiguous ndarrays and are NOT writer-eligible.
PRODUCTION_ROUTE = 'weight_snapshot.snapshot: np.ascontiguousarray(np.asarray(shard.data))'


def contiguous(host):
    """(C-contiguous array, whether a layout-fixing copy was needed)."""
    return np.ascontiguousarray(host), not host.flags.c_contiguous


def d2h_serial_contiguous(rt, tree, chips):
    out, copies = {}, 0
    for c in chips:
        row = []
        for a in tree[c]:
            host, copied = contiguous(np.asarray(a))
            copies += copied
            row.append(host)
        out[c] = row
    return out, {'layout_copies': copies, 'production_route': PRODUCTION_ROUTE}


def d2h_async_contiguous(rt, tree, chips):
    for _, _, a in flat_items(tree, chips):
        a.copy_to_host_async()
    out, copies = {}, 0
    for c in chips:
        row = []
        for a in tree[c]:
            host, copied = contiguous(np.asarray(a))
            copies += copied
            row.append(host)
        out[c] = row
    return out, {'layout_copies': copies, 'production_route': PRODUCTION_ROUTE}


def d2h_chip_threads_contiguous(rt, tree, chips):
    def work(c):
        row, copies = [], 0
        for a in tree[c]:
            host, copied = contiguous(np.asarray(a))
            copies += copied
            row.append(host)
        return row, copies
    rows, extra = per_chip(chips, work)
    extra.update(layout_copies=sum(r[1] for r in rows.values()), production_route=PRODUCTION_ROUTE)
    return {c: rows[c][0] for c in chips}, extra


def make_pool_contiguous(workers):
    def mechanism(rt, tree, chips):
        items = flat_items(tree, chips)

        def task(item):
            return contiguous(np.asarray(item[2]))
        with ThreadPoolExecutor(workers) as pool:
            values = list(pool.map(task, items))
        out = {c: [] for c in chips}
        for (c, _, _), (host, _) in zip(items, values):
            out[c].append(host)
        return out, {'layout_copies': sum(1 for _, copied in values if copied), 'production_route': PRODUCTION_ROUTE}
    return mechanism


def make_pool_d2h(workers, did_copy):
    def mechanism(rt, tree, chips):
        items = flat_items(tree, chips)

        def task(item):
            a = item[2]
            if not did_copy:
                return np.asarray(a), None
            a.copy_to_host_async()
            return a._single_device_array_to_np_array_did_copy()
        with ThreadPoolExecutor(workers) as pool:
            values = list(pool.map(task, items))
        hosts = {c: [] for c in chips}
        for (c, _, _), (value, _) in zip(items, values):
            hosts[c].append(value)
        extra = {}
        if did_copy:
            copied = sum(1 for _, did in values if did)
            extra = {'did_copy_true': copied, 'did_copy_false': len(values) - copied}
        return hosts, extra
    return mechanism


def d2h_pinned_host(rt, tree, chips):
    """Runtime-owned pinned-host staging (device_put to memory_kind='pinned_host'), then host read."""
    from jax.sharding import SingleDeviceSharding
    staged = {c: [rt.jax.device_put(a, SingleDeviceSharding(rt.chips[c], memory_kind='pinned_host'))
                  for a in tree[c]] for c in chips}
    rt.jax.block_until_ready(staged)
    return {c: [np.asarray(s) for s in staged[c]] for c in chips}, {'_keep': [s for c in chips for s in staged[c]]}


def d2h_mechanisms(pool_sizes):
    table = {'serial_asarray': (d2h_serial_asarray, False, False),
             'async_all_asarray': (d2h_async_all_asarray, False, False),
             'async_did_copy': (d2h_async_did_copy, True, False),
             'device_get_tree': (d2h_device_get_tree, False, False),
             'chip_threads_asarray': (d2h_chip_threads_asarray, False, True),
             'chip_threads_async_asarray': (d2h_chip_threads_async_asarray, False, True),
             'chip_threads_did_copy': (d2h_chip_threads_did_copy, True, True),
             'serial_contiguous': (d2h_serial_contiguous, False, False),
             'async_contiguous': (d2h_async_contiguous, False, False),
             'chip_threads_contiguous': (d2h_chip_threads_contiguous, False, True)}
    for n in pool_sizes:
        table[f'pool{n}_asarray'] = (make_pool_d2h(n, False), False, False)
        table[f'pool{n}_did_copy'] = (make_pool_d2h(n, True), True, False)
        table[f'pool{n}_contiguous'] = (make_pool_contiguous(n), False, False)
    table['pinned_host'] = (d2h_pinned_host, True, False)
    return table      # name -> (function, optional, needs_multiple_chips)


# --------------------------------------------------------------------------- H2D mechanisms
def h2d_put_all_ready_end(rt, host, chips):
    out = {c: [rt.put(h, rt.chips[c]) for h in host[c]] for c in chips}
    rt.jax.block_until_ready(out)
    return out, {}


def h2d_put_each_ready_each(rt, host, chips):
    out = {c: [] for c in chips}
    for c in chips:
        for h in host[c]:
            x = rt.put(h, rt.chips[c])
            x.block_until_ready()
            out[c].append(x)
    return out, {}


def h2d_tree_put_per_chip(rt, host, chips):
    out = {c: list(rt.put(host[c], rt.chips[c])) for c in chips}
    rt.jax.block_until_ready(out)
    return out, {}


def h2d_chip_threads_put(rt, host, chips):
    def work(c):
        arrays = [rt.put(h, rt.chips[c]) for h in host[c]]
        rt.jax.block_until_ready(arrays)
        return arrays
    out, extra = per_chip(chips, work)
    return out, extra


def make_pool_h2d(workers):
    def mechanism(rt, host, chips):
        items = [(c, h) for c in chips for h in host[c]]
        with ThreadPoolExecutor(workers) as pool:
            values = list(pool.map(lambda item: rt.put(item[1], rt.chips[item[0]]), items))
        out = {c: [] for c in chips}
        for (c, _), x in zip(items, values):
            out[c].append(x)
        rt.jax.block_until_ready(out)
        return out, {}
    return mechanism


def h2d_mechanisms(pool_sizes):
    table = {'put_all_ready_end': (h2d_put_all_ready_end, False, False),
             'put_each_ready_each': (h2d_put_each_ready_each, False, False),
             'tree_put_per_chip': (h2d_tree_put_per_chip, True, False),
             'chip_threads_put': (h2d_chip_threads_put, False, True)}
    for n in pool_sizes:
        table[f'pool{n}_put'] = (make_pool_h2d(n), False, False)
    return table


# --------------------------------------------------------------------------- runtime
class Runtime:
    """Per-process (per-host) benchmark state.  jax is imported lazily in initialize()."""

    def __init__(self, cfg, catalogue_raw=None):
        self.cfg = cfg
        self.catalogue_raw = catalogue_raw
        self.seed = 0
        self.kernels = {}
        self.compile_s = 0.0
        self.tmpdir = Path(cfg['tmp_root']) / f"omp_snapshot_bandwidth_{cfg.get('run_id') or 'local'}_{uuid.uuid4().hex[:8]}"
        self.sources = {}
        self.workers = max(1, min(32, len(os.sched_getaffinity(0))))
        self.digests = digest_registry()
        self.d2h = d2h_mechanisms(cfg['pool_sizes'])
        self.h2d = h2d_mechanisms(cfg['pool_sizes'])

    # ---- init
    def initialize(self, expected_global_devices=None):
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp = jax, jnp
        local = sorted(jax.local_devices(), key=lambda d: d.id)
        if len(local) < CHIPS:
            raise RuntimeError(f'need {CHIPS} local devices, found {len(local)}: {local}')
        self.chips = local[:CHIPS]
        if expected_global_devices is not None and len(jax.devices()) != expected_global_devices:
            raise RuntimeError(f'chip binding mismatch: jax.devices()={len(jax.devices())} expected '
                               f'{expected_global_devices}')
        try:
            import inspect as _inspect
            self.may_alias = 'may_alias' in _inspect.signature(jax.device_put).parameters
        except (TypeError, ValueError):
            self.may_alias = False
        catalogue = load_catalogue(self.cfg.get('catalogue'), self.catalogue_raw)
        arrays = catalogue['arrays']
        self.catalogue = {k: v for k, v in catalogue.items() if k != 'arrays'}
        chip_bytes = sum(a['bytes'] for a in arrays)
        self.catalogue.update(arrays=len(arrays), full_chip_bytes=chip_bytes, full_host_bytes=chip_bytes * CHIPS,
                              matches_real_snapshot=(chip_bytes == EXPECTED_CHIP_BYTES
                                                     and chip_bytes * CHIPS == EXPECTED_HOST_BYTES))
        if self.cfg.get('smoke_bytes') is None:
            if not self.catalogue['matches_real_snapshot']:
                raise RuntimeError(f'catalogue bytes {chip_bytes}/chip {chip_bytes * CHIPS}/host do not equal '
                                   f'the real snapshot {EXPECTED_CHIP_BYTES}/{EXPECTED_HOST_BYTES}')
        else:
            arrays = smoke_subset(arrays, self.cfg['smoke_bytes'])
        self.workloads = {'model': arrays}
        for size in self.cfg['large_sizes']:
            self.workloads[f'large_{size}'] = large_workload(size)
        self.tmpdir.mkdir(parents=True, exist_ok=False)
        block()
        return self.devices_row()

    def devices_row(self):
        rows = []
        for d in self.chips:
            rows.append({'id': d.id, 'kind': d.device_kind, 'platform': d.platform,
                         'process_index': d.process_index,
                         'coords': list(getattr(d, 'coords', []) or []),
                         'core_on_chip': getattr(d, 'core_on_chip', None)})
        return rows

    def put(self, value, device):
        if self.cfg.get('cpu_unalias'):
            # CPU smoke only: jax 0.11.2 CPU device_put aliases host numpy/memmap memory even with
            # may_alias=False, so hand it a private copy. Never set on TPU.
            value = self.jax.tree_util.tree_map(lambda h: np.array(h, copy=True), value)
        if self.may_alias:
            return self.jax.device_put(value, device, may_alias=False)
        return self.jax.device_put(value, device)

    # ---- kernels (compiled outside every timer)
    def _pattern(self, shape, salt_value):
        jnp, lax = self.jnp, self.jax.lax
        strides, acc = [], 1
        for n in reversed(shape):
            strides.append(acc)
            acc *= n
        strides.reverse()
        flat = None
        for axis, stride in enumerate(strides):
            term = lax.broadcasted_iota(jnp.uint32, shape, axis) * jnp.uint32(stride)
            flat = term if flat is None else flat + term
        return ((flat + salt_value) % jnp.uint32(PERIOD)).astype(jnp.bfloat16)

    def kernel(self, kind, shape, device):
        key = (kind, tuple(shape), device.id)
        if key in self.kernels:
            return self.kernels[key]
        jax, jnp = self.jax, self.jnp
        from jax.sharding import SingleDeviceSharding
        start = time.perf_counter()
        sharding = SingleDeviceSharding(device)
        scalar = jax.ShapeDtypeStruct((), jnp.uint32)
        if kind == 'gen':
            fn = jax.jit(lambda s: self._pattern(tuple(shape), s), out_shardings=sharding)
            exe = fn.lower(scalar).compile()
        else:
            def check(x, s):
                bits = jax.lax.bitcast_convert_type(x, jnp.uint16)
                want = jax.lax.bitcast_convert_type(self._pattern(x.shape, s), jnp.uint16)
                return (jnp.sum(bits.astype(jnp.uint32), dtype=jnp.uint32),
                        jnp.sum((bits != want).astype(jnp.uint32), dtype=jnp.uint32))
            fn = jax.jit(check)
            exe = fn.lower(jax.ShapeDtypeStruct(tuple(shape), jnp.bfloat16, sharding=sharding), scalar).compile()
        self.compile_s += time.perf_counter() - start
        self.kernels[key] = exe
        return exe

    def precompile(self, workload, chips, kinds):
        before = self.compile_s
        for c in chips:
            for shape in sorted({tuple(a['shape']) for a in workload}):
                for kind in kinds:
                    self.kernel(kind, shape, self.chips[c])
        return self.compile_s - before

    # ---- device payload
    def generate(self, workload, chips, seed):
        tree = {c: [] for c in chips}
        for c in chips:
            for i, a in enumerate(workload):
                tree[c].append(self.kernel('gen', a['shape'], self.chips[c])(np.uint32(salt(seed, c, i))))
        self.jax.block_until_ready(tree)       # readiness outside every timer
        return tree

    def device_check(self, tree, workload, chips, seed, expected_sums):
        """On-device sum of bits and regenerated-pattern mismatch count vs host-computed sums."""
        refs = [(c, i, self.kernel('check', workload[i]['shape'], self.chips[c])(x, np.uint32(salt(seed, c, i))))
                for c in chips for i, x in enumerate(tree[c])]
        got = self.jax.device_get([r[2] for r in refs])
        bad = []
        for (c, i, _), (total, mismatches) in zip(refs, got):
            if int(mismatches) != 0 or (expected_sums is not None and int(total) != expected_sums[c][i]):
                bad.append({'chip': c, 'index': i, 'device_sum': int(total), 'mismatches': int(mismatches),
                            'host_sum': None if expected_sums is None else expected_sums[c][i]})
        return {'ok': not bad, 'arrays': len(refs), 'bad_count': len(bad), 'bad': bad[:5]}

    def release(self, *groups):
        for group in groups:
            if isinstance(group, dict):
                group = [x for v in group.values() for x in v]
            for x in group or []:
                try:
                    if hasattr(x, 'delete'):
                        x.delete()
                except Exception:
                    pass
        gc.collect()

    def hbm_headroom(self, chips, need):
        rows = []
        for c in chips:
            try:
                stats_ = self.chips[c].memory_stats()
            except Exception:
                stats_ = None
            if stats_ and 'bytes_limit' in stats_:
                rows.append({'chip': c, 'free': stats_['bytes_limit'] - stats_.get('bytes_in_use', 0)})
        return rows, all(r['free'] >= need * 1.05 for r in rows)

    def host_ok(self, need):
        free = mem_available()
        return free - need >= self.cfg['headroom_floor_bytes'], free

    # ---- D2H
    def run_d2h(self, op):
        workload, chips = self.workloads[op['workload']], op['scope']
        per_chip_bytes = sum(a['bytes'] for a in workload)
        record = {'bytes_per_chip': per_chip_bytes, 'payload_bytes': per_chip_bytes * len(chips),
                  'arrays_per_chip': len(workload), 'chips': chips, 'samples': []}
        mech = op['mechanism']
        control = mech == 'warm_control'
        function, optional, multi = (d2h_serial_asarray, False, False) if control else self.d2h[mech]
        if multi and len(chips) < 2:
            return {**record, 'status': 'not_applicable', 'reason': 'needs >1 chip'}
        hbm, hbm_ok = self.hbm_headroom(chips, per_chip_bytes)
        # warm_control keeps a second host copy; *_contiguous holds the strided output and its C-order copy together.
        ok, free = self.host_ok(record['payload_bytes'] * (2 if control or mech.endswith('_contiguous') else 1))
        record['hbm_free'], record['host_mem_available_bytes'] = hbm, free
        if not (hbm_ok and ok):
            return {**record, 'status': 'skipped', 'reason': 'insufficient ' + ('HBM' if not hbm_ok else 'host') + ' headroom'}
        record['compile_s'] = self.precompile(workload, chips, ['gen'])
        for rep in range(op['reps']):
            tree = hosts = keep = None
            seed = self.next_seed()
            try:
                tree = self.generate(workload, chips, seed)
                arrays = [a for _, _, a in flat_items(tree, chips)]
                cold = sum(1 for a in arrays if getattr(a, '_npy_value', None) is None)
                gc.collect()
                gc.disable()
                try:
                    with MemoryWatch() as watch:
                        wall = time.time()
                        start = time.perf_counter()
                        hosts, extra = function(self, tree, chips)
                        elapsed = time.perf_counter() - start
                finally:
                    gc.enable()
                keep = extra.pop('_keep', None)
                populated = sum(1 for a in arrays if getattr(a, '_npy_value', None) is not None)
                verify = verify_hosts(hosts, workload, seed, self.workers)
                sample = {'rep': rep, 'seed': seed, 'wall_start': wall, 'elapsed_s': elapsed,
                          'gbps': record['payload_bytes'] / elapsed / 1e9,
                          'host_cache_state': 'cold' if cold == len(arrays) else 'warm',
                          'cold_arrays': cold, 'cache_populated_after': populated, 'arrays': len(arrays),
                          'gating_eligible': cold == len(arrays) and not control,
                          'production_route': bool(extra.get('production_route')),
                          'verify': verify, 'extra': extra, 'mem': watch.report(),
                          # a production-route result must itself be C-contiguous, not just logically equal
                          'verified': verify['ok'] and not (extra.get('production_route') and verify['non_c_contiguous'])}
                record['samples'].append(sample)
                if control:     # warm repeat on the SAME arrays: python ndarray cache hit; never gating
                    warm_start = time.perf_counter()
                    again = {c: [np.asarray(a) for a in tree[c]] for c in chips}
                    warm_elapsed = time.perf_counter() - warm_start
                    warm_verify = verify_hosts(again, workload, seed, self.workers)
                    record['samples'].append({
                        'rep': rep, 'seed': seed, 'elapsed_s': warm_elapsed,
                        'gbps': record['payload_bytes'] / warm_elapsed / 1e9, 'host_cache_state': 'warm',
                        'cold_arrays': 0, 'gating_eligible': False, 'verify': warm_verify,
                        'verified': warm_verify['ok'], 'extra': {'repeat_of': 'cold sample above'}})
                    del again
            except Exception as exc:
                record['error'] = exception_record(exc)
                break
            finally:
                self.release(tree, keep)
                hosts = tree = keep = None
                trim_heap()
        return self.finish(record, optional)

    def next_seed(self):
        self.seed += 1
        return self.seed

    def finish(self, record, optional=False):
        samples = [s for s in record['samples'] if s.get('verified')]
        gate = [s['gbps'] for s in samples if s['gating_eligible']]
        record['summary'] = {'gating_gbps': stats(gate), 'gating_elapsed_s': stats(
            [s['elapsed_s'] for s in samples if s['gating_eligible']]),
            'all_samples_verified': bool(record['samples']) and len(samples) == len(record['samples']),
            'ungated_samples': len([s for s in record['samples'] if not s['gating_eligible']])}
        if record.get('error'):
            record['status'] = 'unavailable' if optional else 'failed'
        elif not record['summary']['all_samples_verified']:
            record['status'] = 'failed'
            record['error'] = {'type': 'VerificationFailure', 'message': 'a sample failed byte verification'}
        else:
            record['status'] = 'ok'
        return record

    # ---- H2D
    def host_source(self, kind, workload_name, chips):
        """Host source for `chips` only; one (kind, workload) retained at a time, built lazily per chip."""
        key = (kind, workload_name)
        if key not in self.sources:
            self.release_sources()
            self.sources[key] = {'host': {}, 'writable': {}, 'sums': {}, 'seed': 10_000, 'files': []}
        entry = self.sources[key]
        workload, seed = self.workloads[workload_name], entry['seed']
        import ml_dtypes
        for c in chips:
            if c in entry['host']:
                continue
            total = sum(a['bytes'] for a in workload)
            if kind == 'mmap':
                path = self.tmpdir / f'h2d-{workload_name}-chip{c}.bin'
                entry['files'].append(path)
                wr = np.memmap(path, dtype=np.uint8, mode='w+', shape=(total,))
                offsets, off = [], 0
                for a in workload:
                    offsets.append(off)
                    off += a['bytes']
                par_map(lambda t: fill_u8(wr[offsets[t]: offsets[t] + workload[t]['bytes']],
                                          workload[t]['bytes'] // 2, salt(seed, c, t)),
                        range(len(workload)), self.workers)
                wr.flush()
                entry['writable'][c] = [wr[offsets[t]: offsets[t] + workload[t]['bytes']] for t in range(len(workload))]
                ro = np.memmap(path, dtype=np.uint8, mode='r', shape=(total,))
                entry['host'][c] = [ro[offsets[t]: offsets[t] + workload[t]['bytes']].view(ml_dtypes.bfloat16)
                                    .reshape(workload[t]['shape']) for t in range(len(workload))]
            else:
                arrays = [np.empty(a['shape'], ml_dtypes.bfloat16) for a in workload]
                par_map(lambda t: fill_u8(u8(arrays[t]), workload[t]['bytes'] // 2, salt(seed, c, t)),
                        range(len(workload)), self.workers)
                entry['host'][c] = arrays
                entry['writable'][c] = [u8(x) for x in arrays]
            entry['sums'][c] = host_bit_sums(entry['host'][c], self.workers)
        return entry

    def release_sources(self):
        for entry in self.sources.values():
            entry['host'].clear()
            entry['writable'].clear()
            for path in entry['files']:
                path.unlink(missing_ok=True)
        self.sources.clear()
        trim_heap()

    def run_h2d(self, op):
        workload, chips = self.workloads[op['workload']], op['scope']
        per_chip_bytes = sum(a['bytes'] for a in workload)
        mech, kind = op['mechanism'], op['source']
        function, optional, multi = self.h2d[mech]
        record = {'bytes_per_chip': per_chip_bytes, 'payload_bytes': per_chip_bytes * len(chips),
                  'arrays_per_chip': len(workload), 'chips': chips, 'source': kind,
                  'device_put_may_alias_false': self.may_alias, 'samples': []}
        if multi and len(chips) < 2:
            return {**record, 'status': 'not_applicable', 'reason': 'needs >1 chip'}
        hbm, hbm_ok = self.hbm_headroom(chips, per_chip_bytes)
        key = (kind, op['workload'])
        if key not in self.sources:
            self.release_sources()          # a different retained source would otherwise count against headroom
        have = self.sources[key]['host'] if key in self.sources else {}
        need = per_chip_bytes * len([c for c in chips if c not in have])    # only the selected, uncached chips
        record['source_chips_cached'] = sorted(have)
        ok, free = self.host_ok(need)
        record['hbm_free'], record['host_mem_available_bytes'] = hbm, free
        if not (hbm_ok and ok):
            return {**record, 'status': 'skipped', 'reason': 'insufficient ' + ('HBM' if not hbm_ok else 'host') + ' headroom'}
        edge = 64
        try:
            source = self.host_source(kind, op['workload'], chips)
            record['compile_s'] = self.precompile(workload, chips, ['gen', 'check'])
            record['source_host_sums_computed'] = True
            for rep in range(op['reps']):
                device = None
                try:
                    gc.collect()
                    gc.disable()
                    try:
                        with MemoryWatch() as watch:
                            wall = time.time()
                            start = time.perf_counter()
                            device, extra = function(self, source['host'], chips)
                            elapsed = time.perf_counter() - start        # includes block_until_ready
                    finally:
                        gc.enable()
                    # No-alias proof: corrupt host edges after readiness, device bytes must stay exact.
                    aliases = 0
                    for c in chips:
                        for i, w in enumerate(source['writable'][c]):
                            n = min(edge, w.nbytes)
                            w[:n] = 0xFF
                            w[w.nbytes - n:] = 0xFF
                        try:
                            aliases += int(device[c][0].unsafe_buffer_pointer() == pointer(source['host'][c][0]))
                        except Exception:
                            pass
                    check = self.device_check(device, workload, chips, source['seed'], source['sums'])
                    for c in chips:      # restore the edges from the expectation
                        for i, w in enumerate(source['writable'][c]):
                            n = min(edge, w.nbytes)
                            elements = workload[i]['bytes'] // 2
                            w[:n] = expected_view(0, n // 2, salt(source['seed'], c, i))
                            w[w.nbytes - n:] = expected_view(elements - n // 2, n // 2, salt(source['seed'], c, i))
                    record['samples'].append({
                        'rep': rep, 'wall_start': wall, 'elapsed_s': elapsed,
                        'gbps': record['payload_bytes'] / elapsed / 1e9, 'gating_eligible': True,
                        'host_cache_state': 'n/a', 'device_check': check,
                        'verify': check, 'verified': check['ok'] and aliases == 0,
                        'host_pointer_aliases_device_buffer': aliases, 'extra': extra, 'mem': watch.report()})
                except Exception as exc:
                    record['error'] = exception_record(exc)
                    break
                finally:
                    self.release(device)
                    device = None
        except Exception as exc:
            record['error'] = exception_record(exc)
        return self.finish(record, optional)

    # ---- host copy / digest matrix
    def run_host_copy(self, op):
        self.release_sources()      # drop any cached H2D host source before building the 31.4 GB pool
        workload, chips = self.workloads['model'], list(range(CHIPS))
        per_chip = sum(a['bytes'] for a in workload)
        record = {'bytes_per_chip': per_chip, 'payload_bytes': per_chip * CHIPS, 'cells': [],
                  'digests': {k: {kk: vv for kk, vv in v.items() if kk != 'fn'} for k, v in self.digests.items()}}
        ok, free = self.host_ok(per_chip * CHIPS * 2)
        hbm, hbm_ok = self.hbm_headroom(chips, per_chip)
        record['host_mem_available_bytes'] = free
        if not (ok and hbm_ok):
            return {**record, 'status': 'skipped', 'reason': 'insufficient host/HBM headroom'}
        tree = hosts = None
        reuse = None
        try:
            self.precompile(workload, chips, ['gen'])
            seed = self.next_seed()
            tree = self.generate(workload, chips, seed)
            # Raw (possibly strided) pool exactly as np.asarray returns it: the source of the writer's timed
            # np.copyto cells.  Verified by logical bytes; layout counts recorded.
            raw, _ = d2h_chip_threads_asarray(self, tree, chips)
            pool_verify = verify_hosts(raw, workload, seed, self.workers)
            record['source_pool_verify'] = {k: v for k, v in pool_verify.items() if k != 'bad'}
            self.release(tree)
            tree = None
            if not pool_verify['ok']:
                raise RuntimeError('source pool failed byte verification')
            # C-order logical copy (outside every timer) = source of the flat-byte comparison cells and the
            # independent reference for destination verification of every cell.
            hosts = {c: [np.ascontiguousarray(h) for h in raw[c]] for c in chips}
            src = {c: [u8(h) for h in hosts[c]] for c in chips}
            units = {c: [] for c in chips}
            for c in chips:
                off = 0
                for i, a in enumerate(workload):
                    units[c].append((c, i, a['bytes'], off))
                    off += a['bytes']
            all_units = [u for c in chips for u in units[c]]
            reference = {}

            import ml_dtypes
            bf16 = np.dtype(ml_dtypes.bfloat16)

            def run_cell(mode, threads, digest_name, rep, route='flat_contiguous_source'):
                digest = self.digests[digest_name]
                cell = {'mode': mode, 'threads': threads, 'digest': digest_name, 'rep': rep, 'route': route}
                if not digest['available']:
                    return {**cell, 'status': 'unavailable', 'error': digest.get('error')}
                ok_mem, _ = self.host_ok(per_chip * CHIPS if mode == 'first_touch' else 0)
                if not ok_mem:
                    return {**cell, 'status': 'skipped', 'reason': 'insufficient host headroom'}
                paths = {c: self.tmpdir / f'dest-{mode}-{digest_name}-{threads}-{rep}-chip{c}.bin' for c in chips}
                dest = {}
                poison_before = None
                try:
                    if mode == 'reused':
                        dest = reuse
                        par_map(lambda c: dest[c].__setitem__(slice(None), 0xFF), chips, len(chips))  # poison
                        poison_before = all(par_map(lambda c: bool((dest[c][::4096] == 0xFF).all()), chips, len(chips)))
                    gc.collect()
                    gc.disable()
                    try:
                        with MemoryWatch() as watch:
                            start = time.perf_counter()
                            if mode == 'first_touch':
                                for c in chips:
                                    with open(paths[c], 'wb') as f:
                                        f.truncate(per_chip)
                                    dest[c] = np.memmap(paths[c], dtype=np.uint8, mode='r+', shape=(per_chip,))
                            groups = partition(all_units, threads)

                            def work(group):
                                out = {}
                                for c, i, n, off in group:
                                    if digest['fn']:
                                        out[(c, i)] = digest['fn'](memoryview(src[c][i]))
                                    if route == 'copyto_strided':     # writer route: raw strided host -> shaped mmap view
                                        np.copyto(np.ndarray(workload[i]['shape'], bf16, buffer=dest[c], offset=off),
                                                  raw[c][i], casting='no')
                                    else:
                                        dest[c][off: off + n] = src[c][i]
                                return out
                            with ThreadPoolExecutor(len(groups)) as pool:
                                digests = {}
                                for part in pool.map(work, groups):
                                    digests.update(part)
                            for c in chips:
                                dest[c].flush()
                            elapsed = time.perf_counter() - start
                    finally:
                        gc.enable()
                    equal = all(par_map(lambda u: memeq(src[u[0]][u[1]], dest[u[0]][u[3]: u[3] + u[2]]),
                                        all_units, self.workers))
                    row = {**cell, 'elapsed_s': elapsed, 'gbps': record['payload_bytes'] / elapsed / 1e9,
                           'digest_input_copy_required': bool(digest.get('input_copy_required')),
                           'dest_equals_source_memcmp': equal, 'mem': watch.report()}
                    if mode == 'reused':
                        row['poison_present_before_copy'] = poison_before
                    if digest['fn']:
                        previous = reference.setdefault(digest_name, digests)
                        row['digests_match_first_run'] = previous == digests
                    row['status'] = 'ok' if equal and poison_before is not False and row.get(
                        'digests_match_first_run', True) else 'failed'
                    return row
                except Exception as exc:
                    return {**cell, 'status': 'failed', 'error': exception_record(exc)}
                finally:
                    if mode == 'first_touch':
                        dest.clear()
                        for p in paths.values():
                            p.unlink(missing_ok=True)

            for mode in ('first_touch', 'reused'):
                if mode == 'reused':
                    reuse = {}
                    for c in chips:
                        path = self.tmpdir / f'reuse-chip{c}.bin'
                        with open(path, 'wb') as f:
                            f.truncate(per_chip)
                        reuse[c] = np.memmap(path, dtype=np.uint8, mode='r+', shape=(per_chip,))
                        reuse[c][:] = 0xFF            # allocate every page up front (pages become 'reused')
                for threads in self.cfg['copy_threads']:
                    for digest_name in self.digests:
                        for rep in range(op['reps']):
                            record['cells'].append(run_cell(mode, threads, digest_name, rep))
                for threads in self.cfg['copy_threads']:
                    for rep in range(op['reps']):
                        record['cells'].append(run_cell(mode, threads, 'none', rep, 'copyto_strided'))
            # restore-side verification: fresh read-only mapping + digest of destination bytes
            record['verify_read'] = []
            # make the reused files hold exactly the source bytes before the restore-side read
            par_map(lambda u: reuse[u[0]].__setitem__(slice(u[3], u[3] + u[2]), src[u[0]][u[1]]),
                    all_units, self.workers)
            for c in chips:
                reuse[c].flush()
            for digest_name, digest in self.digests.items():
                if digest['fn'] is None or not digest['available'] or digest_name not in reference:
                    continue
                for threads in self.cfg['copy_threads']:
                    for rep in range(op['reps']):
                        record['verify_read'].append(self.verify_read(
                            digest_name, digest, threads, rep, chips, units, reference[digest_name], reuse,
                            per_chip, record['payload_bytes']))
        except Exception as exc:
            record['error'] = exception_record(exc)
        finally:
            reuse = None
            hosts = raw = None
            self.release(tree)
            for path in self.tmpdir.glob('dest-*'):
                path.unlink(missing_ok=True)
            for path in self.tmpdir.glob('reuse-*'):
                path.unlink(missing_ok=True)
            trim_heap()
        failed = [c for c in record['cells'] + record.get('verify_read', []) if c['status'] == 'failed']
        measured = [c for c in record['cells'] if c['status'] == 'ok']
        skipped = [c for c in record['cells'] if c['status'] == 'skipped']
        record['summary'] = {'cells_ok': len(measured), 'cells_failed': len(failed), 'cells_skipped': len(skipped),
                             'cells_unavailable': len([c for c in record['cells'] if c['status'] == 'unavailable']),
                             'max_gbps': max([c['gbps'] for c in measured], default=None)}
        record['status'] = 'failed' if failed or record.get('error') else ('skipped' if skipped else 'ok')
        if skipped and record['status'] == 'skipped':
            record['reason'] = 'cells skipped for insufficient host headroom'
        return record

    def verify_read(self, name, digest, threads, rep, chips, units, reference_digests, reuse, per_chip, payload):
        row = {'digest': name, 'threads': threads, 'rep': rep}
        try:
            paths = {c: reuse[c].filename for c in chips}
            all_units = [u for c in chips for u in units[c]]
            gc.collect()
            gc.disable()
            try:
                start = time.perf_counter()
                maps = {c: np.memmap(paths[c], dtype=np.uint8, mode='r', shape=(per_chip,)) for c in chips}
                found = {}

                def work(group):
                    return {(c, i): digest['fn'](memoryview(maps[c][off: off + n])) for c, i, n, off in group}
                with ThreadPoolExecutor(threads) as pool:
                    for part in pool.map(work, partition(all_units, threads)):
                        found.update(part)
                elapsed = time.perf_counter() - start
            finally:
                gc.enable()
            maps.clear()
            row.update(elapsed_s=elapsed, gbps=payload / elapsed / 1e9, digests_match_source=found == reference_digests,
                       digest_input_copy_required=bool(digest.get('input_copy_required')),
                       status='ok' if found == reference_digests else 'failed')
        except Exception as exc:
            row.update(status='failed', error=exception_record(exc))
        return row

    # ---- inspection / evidence
    def run_inspect(self, op):
        import jax
        row = {'host': socket.gethostname(), 'python': sys.version, 'platform': platform.platform(),
               'cpu_count': os.cpu_count(), 'sched_affinity': len(os.sched_getaffinity(0)),
               'meminfo': {k: v for k, v in meminfo().items() if k in
                           ('MemTotal', 'MemAvailable', 'Shmem', 'Cached', 'HugePages_Total', 'Hugepagesize')},
               'tmp_dir': str(self.tmpdir), 'tmp_free_bytes': shutil.disk_usage(self.tmpdir).free,
               'catalogue': self.catalogue, 'devices': self.devices_row(),
               'jax_process_count': jax.process_count(), 'jax_global_devices': len(jax.devices()),
               'workloads': {k: {'arrays': len(v), 'bytes_per_chip': sum(a['bytes'] for a in v),
                                 'unique_shapes': len({tuple(a['shape']) for a in v})}
                             for k, v in self.workloads.items()},
               'effective_env': {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(
                   ('TPU_', 'CLOUD_TPU', 'JAX_', 'XLA_', 'LIBTPU', 'MEGASCALE'))},
               'packages': {}, 'digests': {k: {kk: vv for kk, vv in v.items() if kk != 'fn'}
                                           for k, v in self.digests.items()}}
        for name in ('jax', 'jaxlib', 'libtpu', 'libtpu-nightly', 'numpy', 'ml_dtypes', 'ray', 'xxhash', 'crc32c',
                     'google-crc32c', 'psutil', 'tpu-inference', 'vllm', 'vllm-tpu'):
            try:
                row['packages'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                row['packages'][name] = None
        try:
            row['memory_kinds'] = [str(m.kind) for m in self.chips[0].addressable_memories()]
            row['device_memory_stats'] = self.chips[0].memory_stats()
        except Exception as exc:
            row['memory_kinds_error'] = str(exc)
        row['api'] = self.api_inspection()
        return {'status': 'ok', **row}

    def api_inspection(self):
        jax, jnp = self.jax, self.jnp
        probe = self.jax.device_put(np.zeros((8, 128), np.float32), self.chips[0])
        probe.block_until_ready()
        before = getattr(probe, '_npy_value', 'missing')
        np.asarray(probe)
        after = getattr(probe, '_npy_value', 'missing')
        array_type = type(probe)
        names = ('copy_to_host_async', '_copy_single_device_array_to_host_async', '_npy_value',
                 '_single_device_array_to_np_array_did_copy', 'unsafe_raw_buffer', 'unsafe_buffer_pointer',
                 '__dlpack__', '__buffer__', 'device_buffers')
        caps = {n: hasattr(probe, n) for n in names}
        out = {'array_type': f'{array_type.__module__}.{array_type.__qualname__}', 'capabilities': caps,
               'host_cache_probe': {'npy_value_before_asarray_is_none': before is None,
                                    'npy_value_after_asarray_is_none': after is None,
                                    'meaning': 'asarray populates the python ndarray cache (repeat reads are warm)'
                                               if before is None and after is not None else 'see raw values'},
               'device_put_signature': str(inspect.signature(jax.device_put)),
               'device_put_may_alias_param': self.may_alias,
               'caller_owned_d2h_route': {}}
        try:
            raw = probe.unsafe_raw_buffer()
            out['caller_owned_d2h_route']['unsafe_raw_buffer'] = {
                'type': type(raw).__name__, 'public_members': [m for m in dir(raw) if not m.startswith('_')]}
        except Exception as exc:
            out['caller_owned_d2h_route']['unsafe_raw_buffer'] = {'error': f'{type(exc).__name__}: {exc}'}
        try:
            np.from_dlpack(probe)
            out['caller_owned_d2h_route']['np.from_dlpack'] = 'succeeded (host-visible buffer)'
        except Exception as exc:
            out['caller_owned_d2h_route']['np.from_dlpack'] = f'{type(exc).__name__}: {str(exc)[:200]}'
        try:
            import jaxlib._jax as native
            out['jaxlib_native_names'] = sorted(n for n in dir(native) if any(
                s in n.lower() for s in ('host', 'rawbuffer', 'transfer', 'dlpack')))
        except Exception as exc:
            out['jaxlib_native_names_error'] = str(exc)
        needles = ('copy_to_host_async', '_single_device_array_to_np_array_did_copy', '_npy_value',
                   'batched_copy_to_host_async', 'unsafe_raw_buffer', 'def device_get', 'def device_put',
                   'may_alias', 'pinned_host')
        citations, root = {}, Path(jax.__file__).parent / '_src'
        for name in ('array.py', 'api.py', 'dispatch.py', 'xla_bridge.py'):
            path = root / name
            if not path.is_file():
                continue
            for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
                for needle in needles:
                    if needle in line:
                        hits = citations.setdefault(needle, [])
                        if len(hits) < 8:
                            hits.append({'file': str(path), 'line': number, 'text': line.strip()[:160]})
        out['source_citations'] = citations
        out['jax_version'], out['jax_file'] = jax.__version__, jax.__file__
        probe.delete()
        return out

    def execute(self, op):
        started = time.time()
        try:
            handler = {'inspect': self.run_inspect, 'd2h': self.run_d2h, 'h2d': self.run_h2d,
                       'host_copy': self.run_host_copy}[op['kind']]
            row = handler(op)
        except Exception as exc:
            row = {'status': 'failed', 'error': exception_record(exc)}
        row.update(id=op['id'], host=socket.gethostname(), wall_start=started, wall_s=time.time() - started,
                   mem_available_after_bytes=mem_available())
        return row

    def cleanup(self):
        self.release_sources()
        left = sum(p.stat().st_size for p in self.tmpdir.rglob('*') if p.is_file()) if self.tmpdir.exists() else 0
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        return {'tmp_dir': str(self.tmpdir), 'bytes_before_rmtree': left, 'exists_after': self.tmpdir.exists()}


# --------------------------------------------------------------------------- plan / execution
def csv(text):
    return [t for t in str(text).split(',') if t]


def build_plan(cfg):
    phases = cfg['phases']
    d2h = cfg['d2h_mechanisms'] or list(d2h_mechanisms(cfg['pool_sizes'])) + ['warm_control']
    h2d = cfg['h2d_mechanisms'] or list(h2d_mechanisms(cfg['pool_sizes']))
    symmetry_d2h, symmetry_h2d = ['serial_asarray', 'async_did_copy', 'serial_contiguous'], ['put_all_ready_end']
    large = [f'large_{s}' for s in cfg['large_sizes']]
    reps = cfg['reps']
    plan = []

    def add(kind, workload, scope, mechanism, n, **extra):
        label = f"chips{len(scope)}" if len(scope) > 1 else f'chip{scope[0]}'
        plan.append({'id': f"{kind}/{workload}/{label}/{mechanism}" + (f"/{extra['source']}" if 'source' in extra else ''),
                     'kind': kind, 'workload': workload, 'scope': scope, 'mechanism': mechanism, 'reps': n, **extra})
    if 'inspect' in phases:
        plan.append({'id': 'inspect', 'kind': 'inspect'})
    if 'd2h' in phases:
        for scope in ([0], list(range(CHIPS))):
            for m in d2h:
                add('d2h', 'model', scope, m, reps)
        for chip in (1, 2, 3):
            for m in symmetry_d2h:
                add('d2h', 'model', [chip], m, max(1, min(reps, 2)))
    if 'd2h_large' in phases:
        for w in large:
            for scope in ([0], list(range(CHIPS))):
                for m in [x for x in d2h if x in ('serial_asarray', 'async_did_copy', 'device_get_tree',
                                                  'serial_contiguous', 'async_contiguous', 'chip_threads_contiguous',
                                                  'chip_threads_asarray', 'pinned_host', 'warm_control')]:
                    add('d2h', w, scope, m, reps)
    if 'h2d' in phases:
        for source in cfg['h2d_sources']:
            for scope in ([0], list(range(CHIPS))):
                for m in h2d:
                    add('h2d', 'model', scope, m, reps, source=source)
        for chip in (1, 2, 3):
            for m in symmetry_h2d:
                add('h2d', 'model', [chip], m, max(1, min(reps, 2)), source=cfg['h2d_sources'][0])
    if 'h2d_large' in phases:
        for w in large:
            for scope in ([0], list(range(CHIPS))):
                for m in [x for x in h2d if x in ('put_all_ready_end', 'chip_threads_put')]:
                    add('h2d', w, scope, m, reps, source='anon')
    if 'host_copy' in phases:
        plan.append({'id': 'host_copy/model/chips4', 'kind': 'host_copy', 'reps': cfg['copy_reps']})
    return plan


def run_plan(execute, plan, cfg, result):
    """execute(op, timeout) -> list of per-host rows.  Records every op, honours the budget."""
    deadline = time.time() + cfg['budget_s']
    result['ops'] = []
    for op in plan:
        remaining = deadline - time.time()
        if remaining <= 0:
            result['ops'].append({'id': op['id'], 'status': 'skipped', 'reason': 'budget_exhausted'})
            result['incomplete'] = True
            continue
        timeout = min(cfg['op_timeout_s'], remaining + 300)
        rows = execute(op, timeout)
        result['ops'].append({'id': op['id'], 'op': {k: v for k, v in op.items() if k != 'id'}, 'hosts': rows})
        progress = {'id': op['id'], 'hosts': [
            {'host': r.get('host'), 'status': r.get('status'),
             'payload_bytes': r.get('payload_bytes'), 'summary': r.get('summary')} for r in rows]}
        print('[OMP_EVENT: BW_OP ' + json.dumps(progress, separators=(',', ':')) + ']', flush=True)
    failed = [o['id'] for o in result['ops'] if any(r.get('status') == 'failed' for r in o.get('hosts', []))]
    result['failed_ops'] = failed
    result['unavailable_ops'] = [o['id'] for o in result['ops']
                                 if any(r.get('status') == 'unavailable' for r in o.get('hosts', []))]
    # Required ops that produced no measurement (headroom skip, budget) make the run incomplete, not ok.
    result['skipped_ops'] = [o['id'] for o in result['ops'] if o.get('status') == 'skipped'
                             or any(r.get('status') == 'skipped' for r in o.get('hosts', []))]
    result['incomplete'] = bool(result.get('incomplete') or result['skipped_ops'])
    result['ok'] = not failed and not result['incomplete']
    best = {}
    for o in result['ops']:
        for r in o.get('hosts', []):
            g = ((r.get('summary') or {}).get('gating_gbps') or {})
            if g.get('max') is not None:
                best.setdefault(r['id'], []).append({'host': r['host'], 'max_gbps': g['max'],
                                                     'median_gbps': g['median'], 'payload_bytes': r['payload_bytes']})
    result['throughput_summary'] = best


def config_from_args(args):
    return {'phases': csv(args.phases), 'reps': args.reps, 'copy_reps': args.copy_reps,
            'pool_sizes': [int(x) for x in csv(args.pool_sizes)],
            'copy_threads': [int(x) for x in csv(args.copy_threads)],
            'd2h_mechanisms': csv(args.d2h_mechanisms), 'h2d_mechanisms': csv(args.h2d_mechanisms),
            'h2d_sources': csv(args.h2d_sources),
            'large_sizes': [parse_size(x) for x in csv(args.large_sizes)],
            'smoke_bytes': parse_size(args.smoke_bytes) if args.smoke_bytes else None,
            'headroom_floor_bytes': args.headroom_floor_bytes, 'budget_s': args.budget_s,
            'op_timeout_s': args.op_timeout, 'tmp_root': args.tmp_root,
            'catalogue': str(args.catalogue) if args.catalogue else None}


# --------------------------------------------------------------------------- local CPU smoke
def smoke(args):
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=4').strip()
    cfg = config_from_args(args)
    cfg['cpu_unalias'] = True
    cfg['large_sizes'] = [parse_size(x) for x in csv(args.smoke_large_sizes)]
    result = {'schema': SCHEMA, 'mode': 'local-cpu-smoke', 'ok': False, 'config': cfg, 'saved_at': time.time(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    runtime = Runtime(cfg)
    try:
        result['init'] = runtime.initialize(expected_global_devices=4)
        run_plan(lambda op, timeout: [runtime.execute(op)], build_plan(cfg), cfg, result)
    except Exception as exc:
        result['error'] = exception_record(exc)
    finally:
        result['cleanup'] = runtime.cleanup()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(json.dumps({'ok': result['ok'], 'failed_ops': result.get('failed_ops'),
                      'skipped_ops': result.get('skipped_ops'), 'unavailable_ops': result.get('unavailable_ops'),
                      'output': str(args.output)}))
    if not result['ok']:
        raise SystemExit(1)


def sweep_tmp(tmp_root, run_id):
    """Remove only THIS run's tmpfs directories on one host (runs after the actor process is dead)."""
    import glob
    removed = []
    for path in glob.glob(os.path.join(tmp_root, f'omp_snapshot_bandwidth_{run_id}_*')):
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(path) for f in files
                   if os.path.isfile(os.path.join(d, f)))
        shutil.rmtree(path, ignore_errors=True)
        removed.append({'path': path, 'bytes': size, 'exists_after': os.path.exists(path)})
    return {'host': socket.gethostname(), 'removed': removed}


def make_controller(ray):
    """Per-host actor.  The script source travels as an argument and is exec'd on the worker, so no
    worker-side file layout is assumed."""
    @ray.remote(num_cpus=1, resources={'TPU': 4})
    class Controller:
        def __init__(self, env, rank, count, coordinator, expected_global, source, script_path, cfg, catalogue_raw):
            os.environ.update(env)
            os.environ['TPU_VISIBLE_CHIPS'] = '0,1,2,3'
            os.environ.pop('JAX_COMPILATION_CACHE_DIR', None)
            for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                os.environ.pop(name, None)
            self.rank, self.count, self.coordinator = rank, count, coordinator
            self.expected_global, self.cfg, self.catalogue_raw = expected_global, cfg, catalogue_raw
            import types
            module = types.ModuleType('snapshot_bandwidth')
            module.__file__ = script_path
            sys.modules['snapshot_bandwidth'] = module
            exec(compile(source, script_path, 'exec'), module.__dict__)
            self.bw = module
            self.node_ip = ray.util.get_node_ip_address()

        def initialize(self):
            import jax
            if self.count > 1:
                jax.distributed.initialize(coordinator_address=self.coordinator, num_processes=self.count,
                                           process_id=self.rank, initialization_timeout=180)
            self.runtime = self.bw.Runtime(self.cfg, self.catalogue_raw)
            devices = self.runtime.initialize(expected_global_devices=self.expected_global)
            return {'rank': self.rank, 'node_ip': self.node_ip, 'host': socket.gethostname(), 'devices': devices}

        def execute(self, op):
            row = self.runtime.execute(op)
            row.update(rank=self.rank, node_ip=self.node_ip)
            return row

        def cleanup(self):
            return self.runtime.cleanup() if hasattr(self, 'runtime') else {}
    return Controller


# --------------------------------------------------------------------------- cluster (remote job)
def remote(args):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy
    from hetero.layout import resolve_layout, builtin_layout, LayoutRequest, InstanceRequest
    from hetero.slice_inventory import SliceInventory
    from hetero.evidence import emit_event, emit_measurements
    if args.smoke_bytes:
        raise SystemExit('--smoke-bytes is local CPU only; reduced payloads are refused on the cluster')
    cfg = config_from_args(args)
    cfg['run_id'] = uuid.uuid4().hex[:12]
    if not args.catalogue:
        raise SystemExit('--catalogue (a file under the job working dir) is required with --remote')
    inventory = SliceInventory.from_dict(json.loads(resolve_input(args.inventory).read_text()))
    catalogue_raw = resolve_input(args.catalogue).read_bytes()
    source = Path(__file__).read_text()
    script_path = str(Path(__file__).resolve())
    plan = build_plan(cfg)

    Controller = make_controller(ray)

    ray.init(address='auto', namespace='snapshot-bandwidth')
    result = {'schema': SCHEMA, 'mode': 'cluster', 'ok': False, 'config': cfg, 'topology_attempts': [],
              'script_sha256': hashlib.sha256(source.encode()).hexdigest(), 'script_path': script_path,
              'launched_at': time.time()}
    reservation = {'pg': None, 'actors': [], 'nodes': []}

    def stop():
        info = {'actors_killed': 0, 'cleanup': [], 'placement_group_removed': False}
        for actor in reservation['actors']:
            try:
                info['cleanup'].append(ray.get(actor.cleanup.remote(), timeout=120))
            except Exception as exc:
                info['cleanup'].append({'error': str(exc)[:500]})
        for actor in reservation['actors']:
            try:
                ray.kill(actor, no_restart=True)
                info['actors_killed'] += 1
            except Exception as exc:
                emit_event('BW_KILL_ERROR', error=str(exc))
        # The actor process is dead now, so nothing can race this: free any tmpfs it left behind.
        info['sweep'] = []
        for node in reservation['nodes']:
            try:
                info['sweep'].append(ray.get(ray.remote(num_cpus=0)(sweep_tmp).options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False)).remote(
                        cfg['tmp_root'], cfg['run_id']), timeout=120))
            except Exception as exc:
                info['sweep'].append({'node': node, 'error': str(exc)[:500]})
        if reservation['pg'] is not None:
            try:
                remove_placement_group(reservation['pg'])
                info['placement_group_removed'] = True
            except Exception as exc:
                emit_event('BW_PG_ERROR', error=str(exc))
        reservation['pg'], reservation['actors'], reservation['nodes'] = None, [], []
        return info

    def start(topology):
        if topology == 'host4':
            ip = args.host_ip or resolve_layout(inventory, builtin_layout('S')).instances[0].hosts[0].node_ip
            instance = resolve_layout(inventory, LayoutRequest('BW4', (InstanceRequest('host4', CHIPS, host_ips=(ip,)),))).instances[0]
            expected_global = CHIPS
        else:
            instance = resolve_layout(inventory, builtin_layout('S')).instances[0]
            expected_global = TP
        controller_id = {h.node_ip: h.process_rank for h in instance.hosts}
        reservation['nodes'] = [h.ray_node_id for h in instance.hosts]
        coordinator = f'{instance.coordinator_ip}:12355'
        result['instance'] = instance.to_dict()
        result['topology'] = topology
        reservation['pg'] = placement_group(
            [{'CPU': 1, 'TPU': 4, f'node:{h.node_ip}': .001} for h in instance.hosts],
            strategy='STRICT_SPREAD', name=f'bw-{topology}-{int(time.time())}')
        ray.get(reservation['pg'].ready(), timeout=120)
        emit_event('BW_RESERVED', placement_group=reservation['pg'].id.hex(), topology=topology)
        for i, host in enumerate(instance.hosts):
            reservation['actors'].append(Controller.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=reservation['pg'], placement_group_bundle_index=i),
                resources={'TPU': 4, f'node:{host.node_ip}': .001}).remote(
                    instance.process_env(host.process_rank), controller_id[host.node_ip], len(instance.hosts),
                    coordinator, expected_global, source, script_path, cfg, catalogue_raw))
        inits = ray.get([a.initialize.remote() for a in reservation['actors']], timeout=args.init_timeout)
        emit_event('BW_INITIALIZED', topology=topology, hosts=len(inits))
        return inits

    try:
        order = ['host4', 'tp8'] if args.topology == 'auto' else [args.topology]
        for topology in order:
            try:
                result['init'] = start(topology)
                result['topology_attempts'].append({'topology': topology, 'ok': True})
                break
            except Exception as exc:
                result['topology_attempts'].append({'topology': topology, 'ok': False, **exception_record(exc)})
                result['pre_fallback_cleanup'] = stop()
                time.sleep(15)
        if not reservation['actors']:
            raise RuntimeError('no topology could be initialised: ' + json.dumps(result['topology_attempts'])[:1500])

        def execute(op, timeout):
            return ray.get([a.execute.remote(op) for a in reservation['actors']], timeout=timeout)
        run_plan(execute, plan, cfg, result)
    except Exception as exc:
        result['error'] = exception_record(exc)
    finally:
        result['cleanup'] = stop()
        time.sleep(15)
        result['finished_at'] = time.time()
        emit_measurements(result)


def local(args):
    from omp_rcm import StageRunner, stage
    from omp_rcm.ray_job import RayJobSupervisor
    if args.smoke_bytes:
        raise SystemExit('--smoke-bytes runs in-process on CPU: call scripts/snapshot_bandwidth.py directly')
    source_catalogue = Path(args.catalogue) if args.catalogue else next(
        (p for p in DEFAULT_CATALOGUES if p.is_file()), None)
    if source_catalogue is None or not source_catalogue.is_file():
        raise SystemExit('no parameter catalogue found; pass --catalogue <parameter_catalogue.json>')
    inputs = ROOT / INPUT_DIR

    @stage(name='snapshot_bandwidth', order_idx=0, expected_s=args.timeout)
    def submit(state, timing):
        # Stage inputs as files in the working_dir package; the package is captured at submission.
        inputs.mkdir(parents=True, exist_ok=True)
        try:
            (inputs / 'catalogue.json').write_bytes(source_catalogue.read_bytes())
            (inputs / 'inventory.json').write_bytes(Path(args.inventory).read_bytes())
            supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
            argv = ['python3', '-u', 'scripts/snapshot_bandwidth.py', '--remote', '--topology', args.topology,
                    '--inventory', f'{INPUT_DIR}/inventory.json', '--catalogue', f'{INPUT_DIR}/catalogue.json',
                    '--phases', args.phases, '--reps', str(args.reps), '--copy-reps', str(args.copy_reps),
                    '--pool-sizes', args.pool_sizes, '--copy-threads', args.copy_threads,
                    '--h2d-sources', args.h2d_sources, '--large-sizes', args.large_sizes,
                    '--headroom-floor-bytes', str(args.headroom_floor_bytes), '--budget-s', str(args.budget_s),
                    '--op-timeout', str(args.op_timeout), '--init-timeout', str(args.init_timeout),
                    '--tmp-root', args.tmp_root, '--timeout', str(args.timeout)]
            for flag, value in (('--d2h-mechanisms', args.d2h_mechanisms), ('--h2d-mechanisms', args.h2d_mechanisms),
                                ('--host-ip', args.host_ip)):
                if value:
                    argv += [flag, value]
            submission = supervisor.submit_job(
                entrypoint=shlex.join(argv),
                runtime_env={'working_dir': str(ROOT),
                             'excludes': ['.venv*', '.git', '.jj', 'data', 'results', '**/results/**', 'formalization',
                                          'reports', '__pycache__', '*.pyc'],
                             'env_vars': {'PYTHONUNBUFFERED': '1'}},
                name=f'snapshot-bandwidth-{args.topology}', metadata={'purpose': 'snapshot-host-transfer-bandwidth'})
            state.set('submission_id', submission)
            result = supervisor.supervise_until_completion(
                submission_id=submission, timeout_s=args.timeout, stage_name='snapshot_bandwidth', state_store=state)
            payload = {'submission_id': submission, 'saved_at': time.time(), 'status': result['status'],
                       'measurements': result['measurements'], 'entrypoint': shlex.join(argv)}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + '\n')
            print(f'[OMP_EVENT: BW_SAVED path={args.output}]', flush=True)
        finally:
            shutil.rmtree(inputs, ignore_errors=True)
    runner = StageRunner(silence_timeout_s=args.timeout)
    runner.register(submit)
    if not runner.run():
        raise SystemExit(1)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--remote', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--topology', choices=('auto', 'host4', 'tp8'), default='auto')
    p.add_argument('--host-ip', default=None, help='host4 only: pin this node (default: first host of S small_a)')
    p.add_argument('--phases', default=','.join(PHASES))
    p.add_argument('--reps', type=int, default=3, help='samples per D2H/H2D mechanism')
    p.add_argument('--copy-reps', type=int, default=2)
    p.add_argument('--pool-sizes', default='8,32', help='array-level thread pool sizes')
    p.add_argument('--copy-threads', default='1,4')
    p.add_argument('--d2h-mechanisms', default='', help='comma subset (default all + warm_control)')
    p.add_argument('--h2d-mechanisms', default='')
    p.add_argument('--h2d-sources', default='anon,mmap', help='host source kinds: anon heap | mmap tmpfs file')
    p.add_argument('--large-sizes', default='256MiB,1GiB,4GiB')
    p.add_argument('--headroom-floor-bytes', type=int, default=40_000_000_000)
    p.add_argument('--tmp-root', default='/dev/shm')
    p.add_argument('--budget-s', type=float, default=4500)
    p.add_argument('--op-timeout', type=float, default=1500)
    p.add_argument('--init-timeout', type=float, default=300)
    p.add_argument('--timeout', type=float, default=5400, help='whole RCM/Ray job deadline')
    p.add_argument('--inventory', default='/tmp/hetero-inventory-only.json')
    p.add_argument('--catalogue', type=Path, default=None,
                   help='parameter_catalogue.json (default: first existing of ' +
                        ', '.join(str(c) for c in DEFAULT_CATALOGUES) + ')')
    p.add_argument('--ray-url', default='http://127.0.0.1:8265')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--smoke-bytes', default=None,
                   help='LOCAL CPU ONLY: per-chip payload cap (e.g. 32MiB) using a stride subset of real rows')
    p.add_argument('--smoke-large-sizes', default='1MiB,4MiB')
    return p


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.output is None:
        arguments.output = Path('/tmp/snapshot_bandwidth_smoke.json') if arguments.smoke_bytes else DEFAULT_OUTPUT
    if arguments.smoke_bytes:
        smoke(arguments)
    elif arguments.remote:
        remote(arguments)
    else:
        local(arguments)
