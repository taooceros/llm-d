#!/usr/bin/env python3
"""Cold-start matrix cell `tpu-torch` (vLLM-TorchTPU, TP8 over 2 v5e hosts, Ray executor): per-run timelines and the
cell's matrix_summary.json, from the probe's pulled pod logs.

Usage: torchtpu_matrix_report.py --plan PLAN.json --logroot results/torchtpu-load/pod_logs/logs --out DIR

PLAN.json: {"cell": ..., "hardware": ..., "engine": {...}, "versions": {...}, "image_or_venv": ..., "code_commit": ...,
            "reference_run": RUN, "arms": {ARM: {"runs": [RUN, ...], "status": "ok|dnf|na", "t0": "...",
            "bound_s": S (dnf), "note": "..."}}}
Every time is seconds after the run's T0 (host.log): the `vllm serve` launch, the pre-spawn gate opening
("T0 critical path starts" without fresh host), or the fresh-host T0 (before any delivery). Ready = /health 200; a
run counts only if its greedy check (matrix prompt, temperature 0) returned the reference run's text.
Worker milestones come from TP rank 0 (its Ray worker log, collected per host); every vLLM line carries its own
millisecond timestamp (probe logging config), so the forwarding delay of Ray's log stream does not enter.
Phases (contiguous, they add up to ready):
  download     T0 -> vllm launch (delivery on the critical path before the engine starts; 0 when T0 = launch/gate)
  spawn        launch/gate -> TP0 worker process up (API server, engine core, Ray placement group, actors, imports)
  init_device  -> TP0 `Init worker` (TorchTPU/PJRT device + tpu_dist init across the 8 chips)
  lazy         null: vLLM-TorchTPU has no separate lazy-initialisation phase
  load_model   -> TP0 load_model returned (model init + weights, or the TPUFlex restore issue)
  kv           -> engine core `GPU KV cache size` (KV sizing; pinned bytes, so no profile run)
  compile      -> engine core `init engine ... took` (KV allocation, restore join, backbone/sampler compile or bank
               load, warm-up)
  graphs       null: no CUDA-graph capture on TPU (the compiled buckets are in compile)
  tail         -> /health 200 (API server start-up after the engine)
"""
import argparse
import gzip
import json
import pathlib
import re
import statistics

TS_VLLM = re.compile(r'(?:INFO|WARNING|ERROR|DEBUG) \d\d-\d\d (\d\d:\d\d:\d\d\.\d{3}) ')
TS_HOST = re.compile(r'^(\d\d:\d\d:\d\d\.\d+) ')
ANSI = re.compile(r'\x1b\[[0-9;]*m')  # Ray colours forwarded worker prefixes
PHASES = ['download', 'spawn', 'init_device', 'lazy', 'load_model', 'kv', 'compile', 'graphs', 'tail']


def secs(hms):
    h, m, s = hms.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def vstamp(line):
    """vLLM's own (ms) timestamp of a log line, else the receiving host's prefix."""
    m = TS_VLLM.search(line)
    if m:
        return secs(m.group(1))
    m = TS_HOST.match(line)
    return secs(m.group(1)) if m else None


def read(path):
    p = pathlib.Path(path)
    if not p.exists() and pathlib.Path(f'{p}.gz').exists():  # large logs are committed gzipped
        p = pathlib.Path(f'{p}.gz')
    if not p.exists():
        return ''
    data = p.read_bytes()
    if p.suffix == '.gz':
        data = gzip.decompress(data)
    return ANSI.sub('', data.decode(errors='replace'))


def worker_logs(run_dir, vlog=''):
    """{tp_rank: text} from the collected Ray worker logs of both hosts; fallback (log_to_driver runs): rank 0's
    forwarded lines in the driver log, keyed by its Ray worker pid on the driver host."""
    out = {}
    for d in (run_dir / 'ray_logs', run_dir / 'host1' / 'ray_logs'):
        if not d.is_dir():
            continue
        stems = {}
        for f in sorted(d.glob('worker-*.gz')):  # a worker's .out and .err share the stem
            stems.setdefault(f.name.split('.')[0], []).append(f)
        for files in stems.values():
            text = ''.join(read(f) for f in files)
            # Ray-zygote-forked workers carry no "Worker_TPn pid=" tag: the PM plugin's per-rank line identifies them
            m = re.search(r'Worker_TP(\d+) pid=', text) or re.search(r'PM_INIT_DEVICE rank=(\d+)', text)
            if m:
                out.setdefault(int(m.group(1)), '')
                out[int(m.group(1))] += text
    if 0 not in out and vlog:
        m = re.search(r'\(RayWorkerProc pid=(\d+)\) \(Worker_TP0 pid=', vlog)
        if m:
            tag = f'(RayWorkerProc pid={m.group(1)})'
            out[0] = '\n'.join(x for x in vlog.splitlines() if tag in x)
    return out


def first(text, rx):
    for line in text.splitlines():
        if re.search(rx, line):
            return vstamp(line)
    return None


def run_report(root, run):
    d = root / run
    host = read(d / 'host.log')
    i = host.rfind('[host] serve start run=')  # a retried run id appends to host.log: keep the last attempt only
    if i > 0:
        host = host[host.rfind('\n', 0, i) + 1:]
    vlog = read(d / 'vllm_raw.log')
    hl = host.splitlines()

    def hfirst(needle):
        return next((secs(TS_HOST.match(x).group(1)) for x in hl if needle in x and TS_HOST.match(x)), None)

    launch = hfirst('vllm launched')
    fresh = next((secs(TS_HOST.match(x).group(1)) for x in hl
                  if 'T0 critical path starts (fresh host' in x and TS_HOST.match(x)), None)
    gate = hfirst('GATE_OPEN T0')
    ready = hfirst('READY (health 200)')
    kind = 'fresh' if fresh is not None else ('gate' if gate is not None else 'launch')
    t0 = {'fresh': fresh, 'gate': gate, 'launch': launch}[kind]

    def rel(t):
        if t is None or t0 is None:
            return None
        dt = t - t0
        return round(dt + 86400 if dt < -43200 else dt, 2)

    workers = worker_logs(d, vlog)
    w0 = workers.get(0, '')
    m = {
        'launch': launch,
        'gate': gate,
        'engine_core': first(vlog, r'EngineCore.*Initializing a V1 LLM engine'),
        'pg_ready': first(vlog, r'TTL_RAY_PIN placement_group='),
        'workers_up': first(w0, r'TTL_RAY_PIN pid='),
        'device_ready': first(w0, r'(PM_INIT_DEVICE rank=0 |Init worker \| rank=0 )'),
        'load_start': first(w0, r'Loading model from scratch'),
        'weights_done': first(w0, r'(Loading weights took|PM_SNAP_RESTORE(_ASYNC)? rank=0 )'),
        'load_done': first(w0, r'TTL_PROBE load_model call='),
        'kv_sized': first(vlog, r'GPU KV cache size'),
        'engine_done': first(vlog, r'init engine \(profile, create kv cache, warmup model\) took'),
        'ready': ready,
    }
    out = {'run': run, 'T0': kind, 'milestones': {k: rel(v) for k, v in m.items()}}
    out['prespawn_s'] = round(gate - launch, 2) if kind == 'gate' and gate and launch else None
    start = rel(gate if kind == 'gate' else launch)
    order = [('download', 0.0 if kind != 'fresh' else None, start),
             ('spawn', start, rel(m['workers_up'])), ('init_device', None, rel(m['device_ready'])),
             ('load_model', None, rel(m['load_done'])), ('kv', None, rel(m['kv_sized'])),
             ('compile', None, rel(m['engine_done'])), ('tail', None, rel(ready))]
    phases = {p: None for p in PHASES}
    prev = 0.0
    for name, lo, hi in order:
        lo = prev if lo is None else lo
        if hi is None:
            phases[name] = None
            continue
        phases[name] = round(max(0.0, hi - max(lo, 0.0)), 2) if name != 'download' else round(max(0.0, hi), 2)
        prev = max(prev, hi)
    if kind == 'gate':  # spawn happened before T0
        phases['spawn'] = round(max(0.0, (rel(m['workers_up']) or 0.0)), 2)
    out['phases'] = phases
    out['ready_s'] = rel(ready)
    mm = re.search(r'init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s \(compilation: ([\d.]+) s\)',
                   vlog)
    if mm:
        out['init_engine_s'], out['compilation_s'] = float(mm.group(1)), float(mm.group(2))
    mm = re.search(r'GPU KV cache size: ([\d,]+) tokens', vlog)
    out['kv_tokens'] = int(mm.group(1).replace(',', '')) if mm else None
    mm = re.search(r'Loading weights took ([\d.]+) seconds', w0)
    out['tp0_load_weights_s'] = float(mm.group(1)) if mm else None
    gc = d / 'greedy_check.json'
    try:
        out['greedy_text'] = json.loads(gc.read_text())['choices'][0]['message']['content'] if gc.exists() else None
    except (ValueError, KeyError, IndexError):
        out['greedy_text'] = None
    for name, path in (('fetch', d / 'fetch.log'), ('fetch_host1', d / 'host1' / 'fetch.log')):
        mm = re.search(r'FETCH_JSON=(\{.*\})', read(path))
        if mm:
            try:
                fj = json.loads(mm.group(1))
                out[name] = {'seconds': fj.get('seconds'), 'GBps': fj.get('GBps'), 'bytes': fj.get('bytes')}
            except ValueError:
                pass
    for name, path in (('ckpt_fetch', d / 'fetch_ckpt.log'), ('ckpt_fetch_host1', d / 'host1' / 'fetch_ckpt.log')):
        mm = re.search(r'FETCH_CKPT_JSON=(\{.*\})', read(path))
        if mm:
            try:
                cj = json.loads(mm.group(1))
                out[name] = {'seconds': cj.get('seconds'), 'GBps': cj.get('GBps'), 'errors': len(cj.get('errors', []))}
            except ValueError:
                pass
    res = read(d / 'serve_result.json')
    out['serve_result'] = json.loads(res) if res.strip() else None
    out['watchdog_fired'] = (d / 'mem_watchdog_fired').exists() or (d / 'host1' / 'mem_watchdog_fired').exists()
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--logroot', default='results/torchtpu-load/pod_logs/logs')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    plan = json.loads(pathlib.Path(a.plan).read_text())
    root = pathlib.Path(a.logroot)
    ref = run_report(root, plan['reference_run'])['greedy_text']
    runs, arms, extra = {}, {}, {}
    # extra_arms (same format as arms): non-canonical rows kept with their numbers (e.g. superseded configs)
    for (arm, spec), dest in [((k, v), arms) for k, v in plan['arms'].items()] + \
            [((k, v), extra) for k, v in plan.get('extra_arms', {}).items()]:
        reps = [run_report(root, r) for r in spec.get('runs', [])]
        for r in reps:
            r['greedy_ok'] = r.get('greedy_text') == ref and ref is not None
            runs[r['run']] = r
        status = spec['status']
        ok = [r for r in reps if r['ready_s'] is not None and r['greedy_ok']]
        entry = {'status': status, 'n': len(reps) if status != 'na' else 0}
        if status == 'ok':
            entry['runs'] = [r['ready_s'] for r in reps]
            entry['median'] = med(entry['runs'])
            entry['phases'] = {p: med([r['phases'][p] for r in reps]) for p in PHASES}
            if len(ok) != len(reps):
                entry['status'] = 'ok?'  # flagged for the author: a run without ready or with a greedy mismatch
        elif status == 'dnf':
            entry['runs'] = [None] * len(reps)
            entry['median'] = None
            entry['phases'] = {p: None for p in PHASES}
        else:
            entry['runs'], entry['median'], entry['phases'] = [], None, {p: None for p in PHASES}
        entry['gke'] = None
        entry['note'] = spec.get('note', '')
        entry['t0'] = spec.get('t0')
        entry['run_ids'] = spec.get('runs', [])
        if 'bound_s' in spec:
            entry['bound_s'] = spec['bound_s']
        dest[arm] = entry
    summary = {k: plan[k] for k in ('cell', 'hardware', 'tp', 'hosts', 'engine', 'versions', 'image_or_venv',
                                    'code_commit', 't0')}
    summary['arms'] = arms
    if extra:
        summary['extra_arms'] = extra
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'matrix_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (out / 'runs.json').write_text(json.dumps({'reference_run': plan['reference_run'], 'reference_text': ref,
                                               'runs': runs}, indent=1) + '\n')
    print(f'{"arm":16} {"status":6} {"n":>2} {"median":>8}  runs')
    for arm, e in list(arms.items()) + [(f'[x] {k}', v) for k, v in extra.items()]:
        print(f'{arm[:40]:40} {e["status"]:6} {e["n"]:>2} {str(e["median"]):>8}  {e["runs"]}')
        if e['status'].startswith('ok'):
            print('   ' + ' '.join(f'{p}={v}' for p, v in e['phases'].items()))


if __name__ == '__main__':
    main()
