#!/usr/bin/env python3
"""Sliced (parallel ranged-read) copy of a HF checkpoint dir from the GCS FUSE mount to tmpfs.

Each worker thread preads a fixed-size slice of a source file and pwrites it into a
pre-sized destination file, so a lazy FUSE mount (about 0.2 GB/s per stream) is read
with N concurrent streams. Prints a JSON summary line with bytes, seconds and GB/s.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('src')
p.add_argument('dst')
p.add_argument('--threads', type=int, default=48)
p.add_argument('--slice-mib', type=int, default=64)
a = p.parse_args()
SL = a.slice_mib << 20
os.makedirs(a.dst, exist_ok=True)
t0 = time.time()
big, total = [], 0
for name in sorted(os.listdir(a.src)):
    s = os.path.join(a.src, name)
    if not os.path.isfile(s):
        continue
    if name.endswith('.safetensors'):
        big.append((s, os.path.join(a.dst, name), os.path.getsize(s)))
    else:
        shutil.copyfile(s, os.path.join(a.dst, name))
jobs = []
for s, d, n in big:
    fd = os.open(d, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, n)
    os.close(fd)
    total += n
    jobs += [(s, d, off, min(SL, n - off)) for off in range(0, n, SL)]
done = [0]


def work(job):
    s, d, off, ln = job
    sf = os.open(s, os.O_RDONLY)
    df = os.open(d, os.O_WRONLY)
    try:
        got = 0
        while got < ln:
            buf = os.pread(sf, ln - got, off + got)
            if not buf:
                raise IOError(f'short read {s} @ {off + got}')
            os.pwrite(df, buf, off + got)
            got += len(buf)
    finally:
        os.close(sf)
        os.close(df)
    done[0] += ln
    return ln


with cf.ThreadPoolExecutor(a.threads) as ex:
    futs = [ex.submit(work, j) for j in jobs]
    last = time.time()
    for f in cf.as_completed(futs):
        f.result()
        if time.time() - last > 15:
            last = time.time()
            print(f'copy progress {done[0] / 2**30:.1f}/{total / 2**30:.1f} GiB {done[0] / 1e9 / (time.time() - t0):.2f} GB/s', flush=True)
secs = time.time() - t0
# Verify: sizes plus sampled hashes of first/middle/last 16 MiB of each large file.
for s, d, n in big:
    assert os.path.getsize(d) == n, (d, n)
    for off in (0, max(0, n // 2 - (8 << 20)), max(0, n - (16 << 20))):
        hs = hashlib.sha256()
        hd = hashlib.sha256()
        with open(s, 'rb') as f1, open(d, 'rb') as f2:
            f1.seek(off); f2.seek(off)
            hs.update(f1.read(16 << 20)); hd.update(f2.read(16 << 20))
        assert hs.digest() == hd.digest(), f'sample hash mismatch {d} @ {off}'
print('CKPT_COPY_JSON=' + json.dumps({'bytes': total, 'seconds': round(secs, 2), 'GB_per_s': round(total / 1e9 / secs, 3), 'threads': a.threads, 'slice_mib': a.slice_mib}), flush=True)
