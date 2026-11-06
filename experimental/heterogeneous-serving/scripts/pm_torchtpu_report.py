#!/usr/bin/env python3
"""Startup timelines of torchtpu probe serve runs (stock and TPUFlex arms) from their pod logs.

Usage: pm_torchtpu_report.py LOGROOT RUN [RUN ...] [--json OUT]
LOGROOT holds <run>/host.log and <run>/vllm_raw.log (results/torchtpu-load/pod_logs/logs).
Every milestone is in seconds after T0: the launch of `vllm serve`, the gate opening when the workers were
pre-spawned, or the start of the fresh-host arm (before the fetch / checkpoint copy) -- the host.log line
"T0 critical path starts". Worker milestones are taken from TP rank 0.
"""
import json
import pathlib
import re
import sys

MILESTONES = [  # (key, regex on vllm_raw.log, first|last)
    ('zygote_ready', r'PM_ZYGOTE_READY', 'first'),
    ('engine_core_start', r'Initializing a V1 LLM engine', 'first'),
    ('parked', r'PM_PARKED rank=0 ', 'first'),
    ('device_ready', r'(PM_INIT_DEVICE rank=0 |Init worker \| rank=0 )', 'first'),
    ('load_start', r'Worker_TP0 .*Loading model from scratch', 'first'),
    ('weights_done', r'Worker_TP0 .*(Loading weights took|PM_SNAP_RESTORE rank=0 |PM_SNAP_RESTORE_ASYNC rank=0 )',
     'first'),
    ('restore_joined', r'PM_SNAP_JOIN rank=0 ', 'first'),
    ('kv_sized', r'GPU KV cache size', 'first'),
    ('backbone_start', r'Worker_TP0 .*Compiling model backbone', 'first'),
    ('engine_init_done', r'init engine \(profile, create kv cache, warmup model\) took', 'first'),
    ('server_start', r'Starting vLLM server', 'first'),
]
NUMBERS = {
    'weights_s': r'Worker_TP0 .*(?:Loading weights took ([\d.]+) seconds|PM_SNAP_RESTORE rank=0 .* total=([\d.]+)s)',
    'restore_gbps': r'PM_SNAP_RESTORE rank=0 .*\(([\d.]+) GB/s\)',
    'snap_save_s': r'PM_SNAP_SAVE rank=0 .* took=([\d.]+)s',
    'init_engine_s': r'init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s',
    'compilation_s': r'init engine .*\(compilation: ([\d.]+) s\)',
    'kv_tokens': r'GPU KV cache size: ([\d,]+) tokens',
    'kv_bytes': r'PM_KV_BYTES rank=0 bytes=(\d+)',
    'init_device_s': r'PM_INIT_DEVICE rank=0 took=([\d.]+)s',
    'snap_wait_s': r'PM_SNAP_WAIT rank=0 waited=([\d.]+)s',
    'join_wait_s': r'PM_SNAP_JOIN rank=0 .*waited=([\d.]+)s',
    'restore_wall_s': r'PM_SNAP_JOIN rank=0 .*restore_wall=([\d.]+)s',
    'warm_api_s': r'PM_WARM api_server took=([\d.]+)s',
    'warm_engine_s': r'PM_WARM engine_core took=([\d.]+)s',
}


def secs(hms):
    h, m, s = hms.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def stamp(line):
    m = re.match(r'(\d\d:\d\d:\d\d\.\d+) ', line)
    return secs(m.group(1)) if m else None


def run_report(root, run):
    d = root / run
    host = (d / 'host.log').read_text().splitlines()
    vlog = (d / 'vllm_raw.log').read_text(errors='replace').splitlines()
    launch = next(stamp(x) for x in host if 'vllm launched' in x)
    gate = next((stamp(x) for x in host if 'T0 critical path starts' in x),
                next((stamp(x) for x in host if 'GATE_OPEN' in x), None))
    ready = next((stamp(x) for x in host if 'READY (health 200)' in x), None)
    t0 = gate if gate is not None else launch

    def rel(t):
        if t is None:
            return None
        dt = t - t0
        return round(dt + 86400 if dt < -43200 else dt, 2)  # UTC clock wraps at midnight

    t0_line = next((x for x in host if 'T0 critical path starts' in x), '')
    kind = 'launch' if gate is None else ('fresh' if 'fresh host' in t0_line else 'gate')
    out = {'run': run, 'T0': kind,
           'prespawn_s': round(gate - launch, 2) if kind == 'gate' else None,
           'launch_s': rel(launch) if kind == 'fresh' else None,
           'ready_s': rel(ready)}
    htext = '\n'.join(host)
    m = re.search(r'SOURCE_EXITED ([\d.]+) s', htext)
    if m:
        out['source_exit_s'] = float(m.group(1))
    fetch_log = d / 'fetch.log'  # host.log truncates the FETCH_JSON line
    m = re.search(r'FETCH_JSON=(\{.*\})', fetch_log.read_text() if fetch_log.exists() else htext)
    if m:
        try:
            fj = json.loads(m.group(1))
            out['fetch_s'] = fj['seconds']
            out['fetch_gbps'] = fj['GBps']
            out['fetch_events'] = fj['events']
        except ValueError:
            pass
    ck_log = d / 'fetch_ckpt.log'
    if ck_log.exists():
        text = ck_log.read_text()
        line = next((x for x in text.splitlines() if 'FETCH_CKPT_JSON=' in x), None)
        if line:
            cj = json.loads(line.split('FETCH_CKPT_JSON=', 1)[1])
            out['ckpt_fetch_s'] = cj['seconds']
            out['ckpt_fetch_gbps'] = cj['GBps']
            out['ckpt_errors'] = len(cj['errors'])
            # the shards are renamed into place right before this line (fetch-ckpt renames them together)
            out['weights_landed'] = None if cj['errors'] else rel(stamp(line))
    for key, needle in (('ckpt_copied', 'checkpoint copied'), ('model_fetched', 'model dir fetched'),
                        ('ckpt_config_fetched', 'checkpoint config fetched'),
                        ('ckpt_fetched', 'checkpoint fetched')):
        t = next((stamp(x) for x in host if needle in x), None)
        if t is not None:
            out[key] = rel(t)
    for key, rx, which in MILESTONES:
        hits = [stamp(x) for x in vlog if re.search(rx, x)]
        out[key] = rel(hits[0] if which == 'first' else hits[-1]) if hits else None
    text = '\n'.join(vlog)
    for key, rx in NUMBERS.items():
        m = re.search(rx, text)
        if m:
            v = next(g for g in m.groups() if g is not None).replace(',', '')
            out[key] = float(v) if '.' in v else int(v)
    m = re.findall(r'Using cache directory: \S*torch_compile_cache/([0-9a-f]+)/rank_0_0', text)
    out['serve_cache_key'] = m[0] if m else None
    out['backbone_compiled'] = bool(re.search(r'Worker_TP0 .*Compiling FX graph for range \(16, 16\)', text))
    probe = d / 'req_probe.json'
    if probe.exists():
        reqs = json.loads(probe.read_text())['requests']
        out['texts'] = {r['name']: r.get('text') for r in reqs}
        out['ttft_first_s'] = reqs[0].get('ttft_s')
        out['decode_tok_s'] = next((r.get('decode_tok_per_s_after_first') for r in reqs if r['name'] == 'd128'),
                                   None)
    return out


COLS = ['run', 'T0', 'prespawn_s', 'launch_s', 'device_ready', 'load_start', 'weights_done', 'restore_joined',
        'join_wait_s', 'kv_sized', 'backbone_start', 'engine_init_done', 'server_start', 'ready_s', 'fetch_s',
        'kv_tokens', 'serve_cache_key', 'backbone_compiled']


def main():
    args = sys.argv[1:]
    jout = None
    if '--json' in args:
        i = args.index('--json')
        jout = args[i + 1]
        del args[i:i + 2]
    root, runs = pathlib.Path(args[0]), args[1:]
    reps = [run_report(root, r) for r in runs]
    print('| ' + ' | '.join(COLS) + ' |')
    print('|' + '---|' * len(COLS))
    for r in reps:
        print('| ' + ' | '.join('' if r.get(c) is None else str(r.get(c)) for c in COLS) + ' |')
    ref = reps[0].get('texts')
    for r in reps[1:]:
        if ref and r.get('texts'):
            same = {k: r['texts'].get(k) == v for k, v in ref.items()}
            print(f"greedy texts {r['run']} vs {reps[0]['run']}: {'identical' if all(same.values()) else same}")
    if jout:
        pathlib.Path(jout).write_text(json.dumps(reps, indent=1) + '\n')


if __name__ == '__main__':
    main()
