#!/usr/bin/env python3
"""Timeline of the PM_SAMPLE_DIR stack samples (pm_torchtpu plugin, diagnostics).

Usage: pm_sample_report.py SAMPLE_DIR T0_EPOCH [--min 0.3] [--depth 2] [--top 25 --window A:B --pid P]

Per process (role from its stack: APIServer / EngineCore / Worker): first and last sample relative to T0,
then segments of consecutive samples that share the same "anchor", the innermost `--depth` frames that live
in vllm, vllm_torchtpu, torch_tpu, transformers or the plugin. Segments shorter than --min s are folded into
"(other)". With --window/--pid, prints the top inclusive frames in that window instead.
"""
import argparse
import collections
import glob
import json
import os

KEEP = ('vllm/', 'vllm_torchtpu/', 'torch_tpu/', 'transformers/', 'pm_torchtpu', 'torch/_dynamo', 'torch/fx',
        'torch/_functorch', 'torch/compiler', 'jax/', 'tokenizers', 'xgrammar')


def load(path):
    frames, samples, head = {}, [], {}
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if 'f' in r:
                frames[r['f']] = r['n']
            elif 't' in r and 's' in r:
                samples.append((r['t'], r['s']))
            else:
                head = r
    return head, frames, samples


def role(frames, samples):
    names = collections.Counter()
    for _, s in samples[:: max(1, len(samples) // 200)]:
        for f in s:
            names[frames.get(f, '')] += 1
    text = ' '.join(names)
    if 'worker_main' in text or 'WorkerProc' in text:
        return 'Worker'
    if 'run_engine_core' in text:
        return 'EngineCore'
    if 'pm_zygote' in text:
        return 'Zygote'
    return 'APIServer'


def anchor(frames, ids, depth):
    keep = [frames.get(f, '?') for f in ids if any(k in frames.get(f, '') for k in KEEP)]
    if not keep:
        return frames.get(ids[0], '?') if ids else '(idle)'
    return ' < '.join(k.rsplit(':', 1)[0] for k in keep[:depth])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir')
    p.add_argument('t0', type=float)
    p.add_argument('--min', type=float, default=0.3)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--top', type=int, default=0)
    p.add_argument('--window', default='')
    p.add_argument('--pid', type=int, default=0)
    a = p.parse_args()
    for path in sorted(glob.glob(os.path.join(a.dir, '*.jsonl'))):
        head, frames, samples = load(path)
        if not samples:
            continue
        pid = int(os.path.basename(path).split('.')[0])
        if a.pid and pid != a.pid:
            continue
        r = role(frames, samples)
        print(f'== {r} pid={pid} samples={len(samples)} from {samples[0][0] - a.t0:.2f} to {samples[-1][0] - a.t0:.2f} s')
        if a.top:
            lo, hi = (float(x) for x in a.window.split(':')) if a.window else (-float('inf'), float('inf'))
            win = [s for t, s in samples if lo <= t - a.t0 <= hi]
            inc = collections.Counter()
            for s in win:
                for k in {frames.get(f, '?').rsplit(':', 1)[0] for f in s}:
                    inc[k] += 1
            for k, n in inc.most_common(a.top):
                print(f'  {n / max(1, len(win)) * 100:5.1f}%  {k}')
            continue
        segs = []
        for t, s in samples:
            k = anchor(frames, s, a.depth)
            if segs and segs[-1][2] == k:
                segs[-1][1] = t
            else:
                segs.append([t, t, k])
        other = 0.0
        for i, (t1, t2, k) in enumerate(segs):
            end = segs[i + 1][0] if i + 1 < len(segs) else t2
            if end - t1 >= a.min:
                if other:
                    print(f'  {"":>7}  {other:5.2f}s  (other)')
                    other = 0.0
                print(f'  {t1 - a.t0:7.2f}  {end - t1:5.2f}s  {k}')
            else:
                other += end - t1


if __name__ == '__main__':
    main()
