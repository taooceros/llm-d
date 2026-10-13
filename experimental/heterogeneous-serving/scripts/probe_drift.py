#!/usr/bin/env python3
"""Summarize next-token drift between `snapshot_validate.py probe` artifacts (local only).

Per probe point (prompt index, prefix length) the canonical engine's top-32 table is
used; replicas inside one artifact must be bitwise identical.  Per pair of artifacts:
argmax agreement, max |Δlogprob| over shared tokens with p >= 1e-3 in either table
(median, nearest-rank p90, max) and total variation over the union (missing tokens = 0).
With `--reference`, also the S-vs-H symmetry of each layout's deviation from that
reference (paired sign test; recorded, not gated under full bf16 state).

    scripts/probe_drift.py S=probe-S.json H=probe-H.json --reference fp32=probe-S-float32.json \
        --json drift.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path


def load(path: Path) -> dict:
    measured = json.loads(path.read_text())['measurements']
    if measured.get('action') != 'probe' or measured.get('ok') is not True:
        raise SystemExit(f'{path}: not a successful probe artifact')
    tables: dict = {}
    for row in measured['rows']:
        key = (row['prompt_index'], row['prefix_len'])
        tables.setdefault(key, {})[row['engine']] = {int(k): float(v) for k, v in row['top_logprobs'].items()}
    replicas_identical = all(len({json.dumps(sorted(t.items())) for t in engines.values()}) == 1
                             for engines in tables.values())
    canonical = {key: engines[sorted(engines)[0]] for key, engines in tables.items()}
    return {'path': str(path), 'layout': measured['layout'], 'engine_identity': measured['engine_identity'],
            'replicas_identical': replicas_identical, 'tables': canonical}


def deviation(a: dict, b: dict) -> float:
    tokens = [t for t in a.keys() & b.keys() if max(math.exp(a[t]), math.exp(b[t])) >= 1e-3]
    return max(abs(a[t] - b[t]) for t in tokens)


def total_variation(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(math.exp(a.get(t, -math.inf)) - math.exp(b.get(t, -math.inf)))
                     for t in a.keys() | b.keys())


def nearest_rank(values: list, q: float) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


def compare(a: dict, b: dict) -> dict:
    keys = sorted(a['tables'])
    if keys != sorted(b['tables']):
        raise SystemExit('probe artifacts cover different probe points')
    devs = {k: deviation(a['tables'][k], b['tables'][k]) for k in keys}
    tvs = [total_variation(a['tables'][k], b['tables'][k]) for k in keys]
    argmax_equal = sum(max(a['tables'][k], key=a['tables'][k].get) == max(b['tables'][k], key=b['tables'][k].get)
                       for k in keys)
    worst = max(devs, key=devs.get)
    return {'probes': len(keys), 'argmax_equal': argmax_equal,
            'max_abs_dlogprob': {'median': statistics.median(devs.values()),
                                 'p90': nearest_rank(list(devs.values()), 0.9), 'max': devs[worst],
                                 'max_at': {'prompt_index': worst[0], 'prefix_len': worst[1]}},
            'tv': {'median': statistics.median(tvs), 'max': max(tvs)},
            'per_probe': [{'prompt_index': k[0], 'prefix_len': k[1], 'max_abs_dlogprob': devs[k]} for k in keys]}


def sign_test(x: list, y: list) -> dict:
    wins = sum(a > b for a, b in zip(x, y))
    losses = sum(a < b for a, b in zip(x, y))
    n = wins + losses
    tail = sum(math.comb(n, i) for i in range(0, min(wins, losses) + 1)) / 2 ** n if n else 1.0
    return {'first_larger': wins, 'second_larger': losses, 'p_two_sided': min(1.0, 2 * tail)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('artifacts', nargs='+', help='LABEL=probe.json')
    parser.add_argument('--reference', help='LABEL=probe.json the others are compared against')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    loaded = {label: load(Path(path)) for label, path in (item.split('=', 1) for item in args.artifacts)}
    out = {'artifacts': {k: {key: v[key] for key in ('path', 'layout', 'replicas_identical')}
                         for k, v in loaded.items()}, 'pairs': {}}
    for x, y in itertools.combinations(loaded, 2):
        out['pairs'][f'{x}_vs_{y}'] = compare(loaded[x], loaded[y])
    if args.reference:
        label, path = args.reference.split('=', 1)
        reference = load(Path(path))
        out['artifacts'][label] = {key: reference[key] for key in ('path', 'layout', 'replicas_identical')}
        vs_ref = {x: compare(loaded[x], reference) for x in loaded}
        for x, result in vs_ref.items():
            out['pairs'][f'{x}_vs_{label}'] = result
        for x, y in itertools.combinations(loaded, 2):
            dx = [r['max_abs_dlogprob'] for r in vs_ref[x]['per_probe']]
            dy = [r['max_abs_dlogprob'] for r in vs_ref[y]['per_probe']]
            out[f'symmetry_{x}_vs_{y}_against_{label}'] = {
                **sign_test(dx, dy),
                'abs_median_difference': abs(statistics.median(dx) - statistics.median(dy))}
    for name, pair in out['pairs'].items():
        d = pair['max_abs_dlogprob']
        print(f"{name}: argmax {pair['argmax_equal']}/{pair['probes']} median {d['median']:.3f} "
              f"p90 {d['p90']:.3f} max {d['max']:.3f} TV median {pair['tv']['median']:.3f} max {pair['tv']['max']:.3f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
