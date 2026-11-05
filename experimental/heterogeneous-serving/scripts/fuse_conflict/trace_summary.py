"""Summarize a gcsfuse trace log (gke-gcsfuse-sidecar, log-severity=trace) per file handle.

FUSE ReadFile ops carry (inode, handle, offset, size); GCS range requests carry [start, end) but no
handle.  A range request is opened inside the ReadFile op it serves and starts at that op's offset,
so each request is attributed to the outstanding op with the same offset.

Usage: trace_summary.py LOG[.gz] [--buckets-s 10]
"""
import argparse
import collections
import gzip
import json
import re

OP_IN = re.compile(r"Op (0x[0-9a-f]+)\s+connection\.go:\d+\] <- ReadFile \(inode (\d+), PID \d+, handle (\d+), offset (\d+), (\d+) bytes\)")
OP_OUT = re.compile(r"Op (0x[0-9a-f]+)\s+connection\.go:\d+\] -> ReadFile")
GCS_IN = re.compile(r'gcs: Req\s+(0x[0-9a-f]+): <- ReadWithReadHandle\("(.*?)", \[(\d+), (\d+)\)\)')
MIB = 1 << 20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--buckets-s", type=float, default=10.0)
    args = p.parse_args()
    opener = gzip.open if args.log.endswith(".gz") else open
    outstanding = {}  # op id -> (handle, offset, size)
    fuse = collections.defaultdict(list)  # handle -> [(t, offset, size)]
    gcs = collections.defaultdict(list)  # handle -> [(t, start, end)]
    unattributed = 0
    t0 = None
    with opener(args.log, "rt") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg, ts = rec.get("message", ""), rec["timestamp"]["seconds"] + rec["timestamp"]["nanos"] / 1e9
            t0 = ts if t0 is None else t0
            if m := OP_IN.search(msg):
                op, _inode, handle, off, size = m[1], int(m[2]), int(m[3]), int(m[4]), int(m[5])
                outstanding[op] = (handle, off, size)
                fuse[handle].append((ts - t0, off, size))
            elif m := OP_OUT.search(msg):
                outstanding.pop(m[1], None)
            elif m := GCS_IN.search(msg):
                start, end = int(m[3]), int(m[4])
                owners = [h for h, off, _ in outstanding.values() if off == start]
                if len(owners) != 1:
                    unattributed += 1
                    continue
                gcs[owners[0]].append((ts - t0, start, end))
    out = {"unattributed_gcs_requests": unattributed, "handles": {}}
    for handle in sorted(fuse):
        ops, reqs = fuse[handle], gcs.get(handle, [])
        backward = forward_gt8 = forward_le8 = 0
        prev_end = None
        for _, off, size in ops:
            if prev_end is not None:
                if off < prev_end:
                    backward += 1
                elif off > prev_end + 8 * MIB:
                    forward_gt8 += 1
                elif off > prev_end:
                    forward_le8 += 1
            prev_end = off + size
        sizes = collections.Counter(round((e - s) / MIB, 1) for _, s, e in reqs)
        timeline = collections.defaultdict(lambda: [0, 0])
        for t, s, e in reqs:
            b = int(t // args.buckets_s)
            timeline[b][0] += 1
            timeline[b][1] += e - s
        out["handles"][handle] = {
            "fuse_reads": len(ops), "fuse_bytes": sum(s for _, _, s in ops),
            "fuse_read_sizes": collections.Counter(s for _, _, s in ops).most_common(4),
            "jumps_backward": backward, "jumps_forward_le_8MiB": forward_le8, "jumps_forward_gt_8MiB": forward_gt8,
            "gcs_requests": len(reqs), "gcs_requested_bytes": sum(e - s for _, s, e in reqs),
            "gcs_request_MiB_hist": sizes.most_common(8),
            "per_bucket_requests_and_MiB": {f"{b * args.buckets_s:.0f}s": [v[0], round(v[1] / MIB)]
                                            for b, v in sorted(timeline.items())},
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
