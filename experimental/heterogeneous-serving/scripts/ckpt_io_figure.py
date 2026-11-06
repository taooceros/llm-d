#!/usr/bin/env python3
"""Paper figure: cold whole-checkpoint read throughput per host vs concurrent read streams.

    python3 scripts/ckpt_io_figure.py results/switch/autoscale-probe/ckpt-io-906d2ef.json --svg out.svg

Input: startup_state.py ckpt-io (all target hosts at once, page cache evicted before each variant).
Range reads over GCS FUSE (seq = 1 stream, par-N = N streams) are plotted against the stream count with
the 1-stream rate times N; the default loader's tensor walk (st-lazy, over FUSE; and over the local tmpfs
copy that gcs-32 downloads) and the GCS API download (gcs-32) are drawn as references.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from pathlib import Path

X0, X1, Y0, Y1 = 50.0, 292.0, 28.0, 196.0  # plot box
LO, HI = 0.05, 10.0  # GB/s axis (log)
STREAMS = 64


def x(n):
    return X0 + math.log2(n) / math.log2(STREAMS) * (X1 - X0)


def y(v):
    return Y1 - (math.log10(v) - math.log10(LO)) / (math.log10(HI) - math.log10(LO)) * (Y1 - Y0)


def load(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)["measurements"]


def rates(variant):
    return [r["gbps"] for r in variant["ranks"]]


CONFLICT_ROWS = (  # variant, label, seconds key
    ("st-lazy", "tensor walk alone", "seconds"),
    ("vpf", "vLLM prefetch alone", "seconds"),
    ("st-lazy+vpf", "walk + vLLM prefetch", "seconds"),
    ("st-lazy+par-32", "walk + 32-stream prefetch", "seconds"),
)
BX0, BLO, BHI = 132.0, 10.0, 5000.0


def bx(s):
    return BX0 + math.log10(s / BLO) / math.log10(BHI / BLO) * (X1 - BX0)


def conflict_panel(o, c, top):
    """Panel (b): seconds for the loader's tensor walk to finish with a background reader (log axis)."""
    o.append(f'<text x="4" y="{top + 11}" font-weight="bold" font-size="11">(b) Tensor walk with a background reader</text>')
    rows0 = top + 22
    bottom = rows0 + 20 * len(CONFLICT_ROWS)
    for t in (10, 30, 100, 300, 1000, 3000):
        o.append(f'<line x1="{bx(t):.1f}" y1="{rows0 - 3}" x2="{bx(t):.1f}" y2="{bottom}" stroke="#e6e6e6"/>')
        o.append(f'<text x="{bx(t):.1f}" y="{bottom + 12}" text-anchor="middle" fill="#555">{t}</text>')
    o.append(f'<text x="{(BX0 + X1) / 2}" y="{bottom + 25}" text-anchor="middle">seconds to read the 62.5 GB checkpoint (log)</text>')
    colors = {"st-lazy": "#CC6677", "vpf": "#999999", "st-lazy+vpf": "#882255", "st-lazy+par-32": "#117733"}
    for i, (variant, label, key) in enumerate(CONFLICT_ROWS):
        secs = [r[key] for r in c["variants"][variant]["ranks"]]
        med, yy = statistics.median(secs), rows0 + 20 * i
        o.append(f'<text x="{BX0 - 5}" y="{yy + 10}" text-anchor="end">{label}</text>')
        o.append(f'<rect x="{BX0}" y="{yy}" width="{bx(med) - BX0:.1f}" height="13" fill="{colors[variant]}"/>')
        o.append(f'<line x1="{bx(min(secs)):.1f}" y1="{yy + 6.5}" x2="{bx(max(secs)):.1f}" y2="{yy + 6.5}" stroke="#222" stroke-width="1"/>')
        o.append(f'<text x="{bx(max(secs)) + 4:.1f}" y="{yy + 10}" font-weight="bold">{med:.0f} s</text>')
    return bottom + 30


FD_ROWS = (  # variant, label
    ("ofd-16", "16 threads, fd per read"),
    ("sfd-16", "16 threads, one shared fd"),
    ("sfd-1", "1 thread, one shared fd"),
)
GLO, GHI = 0.001, 10.0


def gx(v):
    return BX0 + math.log10(v / GLO) / math.log10(GHI / GLO) * (X1 - BX0)


def fd_panel(o, f, top):
    """Panel (c): per-host GB/s of 64 MiB preads by descriptor sharing (log axis)."""
    o.append(f'<text x="4" y="{top + 11}" font-weight="bold" font-size="11">(c) Concurrent reads of one file: descriptor sharing</text>')
    rows0 = top + 22
    bottom = rows0 + 20 * len(FD_ROWS)
    for t, s in ((0.001, "0.001"), (0.01, "0.01"), (0.1, "0.1"), (1, "1"), (10, "10")):
        o.append(f'<line x1="{gx(t):.1f}" y1="{rows0 - 3}" x2="{gx(t):.1f}" y2="{bottom}" stroke="#e6e6e6"/>')
        o.append(f'<text x="{gx(t):.1f}" y="{bottom + 12}" text-anchor="middle" fill="#555">{s}</text>')
    o.append(f'<text x="{(BX0 + X1) / 2}" y="{bottom + 25}" text-anchor="middle">GB/s per host, 64 MiB preads (log)</text>')
    colors = {"ofd-16": "#117733", "sfd-16": "#882255", "sfd-1": "#CC6677"}
    for i, (variant, label) in enumerate(FD_ROWS):
        vals = rates(f["variants"][variant])
        med, yy = statistics.median(vals), rows0 + 20 * i
        o.append(f'<text x="{BX0 - 5}" y="{yy + 10}" text-anchor="end">{label}</text>')
        o.append(f'<rect x="{BX0}" y="{yy}" width="{gx(med) - BX0:.1f}" height="13" fill="{colors[variant]}"/>')
        o.append(f'<line x1="{gx(min(vals)):.1f}" y1="{yy + 6.5}" x2="{gx(max(vals)):.1f}" y2="{yy + 6.5}" stroke="#222" stroke-width="1"/>')
        o.append(f'<text x="{gx(max(vals)) + 4:.1f}" y="{yy + 10}" font-weight="bold">{med:.3g}</text>')
    return bottom + 30


def svg(m, c=None, f=None):
    v = m["variants"]
    total = statistics.median(r["bytes"] for r in v["seq"]["ranks"]) / 1e9
    per_stream = statistics.median(rates(v["seq"]))
    walk = rates(v["st-lazy"])
    local = [r["local_touch_bytes"] / r["local_touch_s"] / 1e9 for r in v["gcs-32"]["ranks"]]
    api = rates(v["gcs-32"])
    ranges = [(1, rates(v["seq"]))] + [(int(k[4:]), rates(v[k])) for k in sorted(v, key=lambda k: int(k[4:]) if k.startswith("par-") else 0)
                                        if k.startswith("par-")]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 {{h}}" font-family="Liberation Sans, Arial, '
         f'Helvetica, sans-serif" font-size="10.5" fill="#222">',
         f'<text x="4" y="11" font-weight="bold" font-size="11">(a) Cold checkpoint read over GCS FUSE</text>']
    for n in (1, 2, 4, 8, 16, 32, 64):
        o.append(f'<line x1="{x(n):.1f}" y1="{Y0}" x2="{x(n):.1f}" y2="{Y1}" stroke="#e6e6e6"/>')
        o.append(f'<text x="{x(n):.1f}" y="{Y1 + 13}" text-anchor="middle" fill="#555">{n}</text>')
    for t in (0.1, 0.3, 1, 3, 10):
        o.append(f'<line x1="{X0}" y1="{y(t):.1f}" x2="{X1}" y2="{y(t):.1f}" stroke="#e6e6e6"/>')
        o.append(f'<text x="{X0 - 4}" y="{y(t) + 3.5:.1f}" text-anchor="end" fill="#555">{t:g}</text>')
    for s in (1000, 100, 10):  # seconds to read the checkpoint
        o.append(f'<text x="{X1 + 4}" y="{y(total / s) + 3.5:.1f}" fill="#555">{s} s</text>')
    o.append(f'<rect x="{X0}" y="{Y0}" width="{X1 - X0}" height="{Y1 - Y0}" fill="none" stroke="#888" stroke-width="0.8"/>')
    o.append(f'<text x="{(X0 + X1) / 2}" y="{Y1 + 27}" text-anchor="middle">concurrent read streams per host</text>')
    o.append(f'<text transform="translate(14 {(Y0 + Y1) / 2}) rotate(-90)" text-anchor="middle">GB/s per host</text>')
    o.append(f'<text transform="translate(333 {(Y0 + Y1) / 2}) rotate(90)" text-anchor="middle">'
             f'time to read {total:.1f} GB</text>')
    # 1-stream rate x N
    o.append(f'<line x1="{x(1):.1f}" y1="{y(per_stream):.1f}" x2="{x(STREAMS):.1f}" y2="{y(per_stream * STREAMS):.1f}" '
             'stroke="#999" stroke-width="1" stroke-dasharray="4 3"/>')
    # references
    for vals, color, label in ((local, "#4477AA", "tensor walk, local tmpfs copy"),
                               (walk, "#CC6677", "tensor walk, GCS FUSE")):
        med = statistics.median(vals)
        o.append(f'<rect x="{X0}" y="{y(max(vals)):.1f}" width="{X1 - X0}" height="{max(0.8, y(min(vals)) - y(max(vals))):.1f}" '
                 f'fill="{color}" fill-opacity="0.18"/>')
        o.append(f'<line x1="{X0}" y1="{y(med):.1f}" x2="{X1}" y2="{y(med):.1f}" stroke="{color}" stroke-width="1.6"/>')
        ty = y(med) + 12
        o.append(f'<text x="{X1 - 4}" y="{ty:.1f}" text-anchor="end" fill="{color}" font-weight="bold">'
                 f'{label}: {med:.2f} GB/s ({total / med:.0f} s)</text>')
    # the loader walk crosses the 1-stream line at its equivalent stream count
    eq = statistics.median(walk) / per_stream
    o.append(f'<circle cx="{x(eq):.1f}" cy="{y(statistics.median(walk)):.1f}" r="3.2" fill="#fff" stroke="#CC6677" stroke-width="1.4"/>')
    o.append(f'<text x="{x(eq) + 5:.1f}" y="{y(statistics.median(walk)) - 6:.1f}" fill="#CC6677">≈{eq:.1f} streams</text>')
    # range reads over FUSE
    pts = []
    for n, vals in ranges:
        med = statistics.median(vals)
        pts.append(f'{x(n):.1f},{y(med):.1f}')
        o.append(f'<line x1="{x(n):.1f}" y1="{y(min(vals)):.1f}" x2="{x(n):.1f}" y2="{y(max(vals)):.1f}" stroke="#117733" stroke-width="1.4"/>')
    o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#117733" stroke-width="1.8"/>')
    for p in pts:
        cx, cy = p.split(",")
        o.append(f'<circle cx="{cx}" cy="{cy}" r="3.2" fill="#117733"/>')
    # GCS API download
    med = statistics.median(api)
    o.append(f'<path d="M {x(32):.1f} {y(med) - 4.5:.1f} l 4.5 4.5 l -4.5 4.5 l -4.5 -4.5 z" fill="#AA4499"/>')
    o.append(f'<text x="{x(32) - 7:.1f}" y="{y(med) + 3.5:.1f}" text-anchor="end" fill="#AA4499">GCS API, 32 workers: {med:.2f}</text>')
    # legend
    ly = Y1 + 44
    o.append(f'<line x1="{X0}" y1="{ly}" x2="{X0 + 18}" y2="{ly}" stroke="#117733" stroke-width="1.8"/>'
             f'<circle cx="{X0 + 9}" cy="{ly}" r="3" fill="#117733"/>'
             f'<text x="{X0 + 23}" y="{ly + 3.5}">64 MiB range reads, GCS FUSE (median; host min–max)</text>')
    o.append(f'<line x1="{X0}" y1="{ly + 15}" x2="{X0 + 18}" y2="{ly + 15}" stroke="#999" stroke-dasharray="4 3"/>'
             f'<text x="{X0 + 23}" y="{ly + 18.5}">1-stream rate ({per_stream:.2f} GB/s) × streams</text>')
    height = ly + 26
    if c is not None:
        height = conflict_panel(o, c, ly + 30)
    if f is not None:
        height = fd_panel(o, f, height)
    o.append("</svg>")
    return "\n".join(o).replace("{h}", str(int(height))) + "\n", \
        {"per_stream": per_stream, "walk_streams": eq, "local_walk_gbps": statistics.median(local)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--conflict", type=Path, help="ckpt-io run with st-lazy, vpf, st-lazy+vpf, st-lazy+par-32")
    parser.add_argument("--fd", type=Path, help="ckpt-io run with ofd-16, sfd-16, sfd-1")
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    text, stats = svg(load(args.probe), load(args.conflict) if args.conflict else None,
                      load(args.fd) if args.fd else None)
    args.svg.write_text(text)
    print({k: round(v, 3) for k, v in stats.items()})


if __name__ == "__main__":
    main()
