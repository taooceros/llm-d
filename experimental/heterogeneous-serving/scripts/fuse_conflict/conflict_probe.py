"""Engine-free probe of the vLLM prefetch x loader conflict on GCS FUSE (one host, bounded range).

Readers (each runs in its own thread, like vLLM's background prefetch thread):
  walk   vLLM's default loader access: safe_open(framework="pt") over the mmap'ed file, then
         get_tensor(name).clone() in key order; restricted to tensors lying inside [0, limit).
  vpf    vLLM's _prefetch_checkpoint: open(path, "rb"), while f.read(16 MiB), over [start, start+len).
  seq    the same buffered sequential reader, used as a second plain reader.

Every variant starts cold: posix_fadvise(DONTNEED) on all files of both mounts, then mincore must
report 0 resident bytes for the ranges the variant reads.  Output: one JSON object per variant on
stdout (per-reader bytes, seconds, and a progress timeline sampled every second).
"""
import argparse
import ctypes
import json
import mmap
import os
import struct
import sys
import threading
import time

BLOCK = 16 * 1024 * 1024
FILE1 = "gemma-4-31b/model-00001-of-00002.safetensors"
FILE2 = "gemma-4-31b/model-00002-of-00002.safetensors"
GIB = 1024 ** 3


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    base = 8 + n
    meta.pop("__metadata__", None)
    return {k: (base + v["data_offsets"][0], base + v["data_offsets"][1]) for k, v in meta.items()}


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def resident(path, start, length):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    page = os.sysconf("SC_PAGE_SIZE")
    start -= start % page
    pages = (length + page - 1) // page
    vec = (ctypes.c_ubyte * pages)()
    with open(path, "rb") as f, mmap.mmap(f.fileno(), length, offset=start, flags=mmap.MAP_PRIVATE,
                                         prot=mmap.PROT_READ | mmap.PROT_WRITE) as m:
        anchor = ctypes.c_char.from_buffer(m)
        try:
            if libc.mincore(ctypes.addressof(anchor), length, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore")
        finally:
            del anchor
    return sum(v & 1 for v in vec) * page


class Reader(threading.Thread):
    def __init__(self, name, fn):
        super().__init__(daemon=True)
        self.label, self.fn = name, fn
        self.done_bytes, self.began, self.ended, self.error = 0, None, None, None

    def run(self):
        self.began = time.time()
        try:
            self.fn(self)
        except Exception as exc:  # recorded, reported in the row
            self.error = repr(exc)
        self.ended = time.time()


def walk_fn(path, limit):
    from safetensors import safe_open
    offsets = header(path)
    names = {k for k, (a, b) in offsets.items() if b <= limit}

    def run(r):
        with safe_open(path, framework="pt") as h:
            for name in h.keys():  # same order as vLLM's safetensors_weights_iterator
                if name not in names:
                    continue
                t = h.get_tensor(name).clone()
                r.done_bytes += t.numel() * t.element_size()
                del t
    return run, sum(offsets[k][1] - offsets[k][0] for k in names)


def buffered_fn(path, start, length, block=BLOCK):
    def run(r):
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(block, left))
                if not chunk:
                    break
                left -= len(chunk)
                r.done_bytes += len(chunk)
    return run


def build(variant, a, b, limit):
    """a, b: mount roots for the first and second reader.  Returns (readers, walk bytes).

    Walk variants parse the safetensors header first, as safe_open does, which leaves the file's first
    few hundred KiB in the page cache.  Plain seq variants read nothing beforehand, so both readers'
    first FUSE reads land at offset 0; their "-head" twins read the header first, like the walk."""
    f1a, f1b, f2b = os.path.join(a, FILE1), os.path.join(b, FILE1), os.path.join(b, FILE2)
    walk, walk_bytes = walk_fn(f1a, limit) if variant.startswith("walk") else (None, 0)
    if variant.endswith("-head"):
        header(f1a)
        variant = variant[: -len("-head")]
    rows = {
        "walk": [("walk", walk)],
        "vpf": [("vpf", buffered_fn(f1b, 0, limit))],
        "walk+vpf": [("walk", walk), ("vpf", buffered_fn(f1b, 0, limit))],
        "walk+vpf-disjoint": [("walk", walk), ("vpf", buffered_fn(f1b, limit, limit))],
        "walk+vpf-file2": [("walk", walk), ("vpf", buffered_fn(f2b, 0, limit))],
        "seq": [("seq1", buffered_fn(f1a, 0, limit))],
        "seq+seq": [("seq1", buffered_fn(f1a, 0, limit)), ("seq2", buffered_fn(f1b, 0, limit))],
        "seq+seq-disjoint": [("seq1", buffered_fn(f1a, 0, limit)), ("seq2", buffered_fn(f1b, limit, limit))],
    }[variant]
    return [Reader(n, fn) for n, fn in rows], walk_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", required=True)
    p.add_argument("--limit-gib", type=float, default=4.0)
    p.add_argument("--mount-a", default="/models")
    p.add_argument("--mount-b", default="/models")
    p.add_argument("--timeout", type=float, default=1800)
    args = p.parse_args()
    limit = int(args.limit_gib * GIB)
    mounts = sorted({args.mount_a, args.mount_b, "/models", "/models-trace"})
    for variant in args.variants.split(","):
        for m in mounts:
            for f in (FILE1, FILE2):
                evict(os.path.join(m, f))
        cold = {m: resident(os.path.join(m, FILE1), 0, 2 * limit) + resident(os.path.join(m, FILE2), 0, limit)
                for m in mounts}
        readers, walk_bytes = build(variant, args.mount_a, args.mount_b, limit)
        t0 = time.time()
        for r in readers:
            r.start()
        timeline = []
        while any(r.is_alive() for r in readers) and time.time() - t0 < args.timeout:
            time.sleep(1.0)
            timeline.append([round(time.time() - t0, 1)] + [r.done_bytes for r in readers])
        row = {
            "variant": variant, "limit_bytes": limit, "walk_target_bytes": walk_bytes,
            "mount_a": args.mount_a, "mount_b": args.mount_b, "resident_before": cold,
            "timed_out": any(r.is_alive() for r in readers),
            "readers": [{"name": r.label, "bytes": r.done_bytes, "error": r.error,
                         "seconds": round((r.ended or time.time()) - r.began, 2),
                         "gbps": round(r.done_bytes / 1e9 / max(1e-9, (r.ended or time.time()) - r.began), 4)}
                        for r in readers],
            "wall_s": round(time.time() - t0, 2), "timeline": timeline,
        }
        print(json.dumps(row), flush=True)
        print(f"# {variant}: " + ", ".join(f"{x['name']} {x['bytes']/1e9:.2f} GB {x['seconds']} s {x['gbps']} GB/s"
                                          for x in row["readers"]) + (" TIMEOUT" if row["timed_out"] else ""),
              file=sys.stderr, flush=True)
        if row["timed_out"]:
            os._exit(3)  # abandoned reader threads would bleed into the next variant


if __name__ == "__main__":
    main()
