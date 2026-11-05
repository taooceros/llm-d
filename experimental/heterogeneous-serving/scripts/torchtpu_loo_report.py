#!/usr/bin/env python3
"""Leave-one-out (LOO) ablation of TPUFlex config B in the `tpu-torch` matrix cell (vLLM-TorchTPU 0.29.0, TP8 over
one v5e 2x4 pair, Ray executor): loo_summary.json plus runs.json, from the probe's pulled pod logs.

Usage: torchtpu_loo_report.py --plan PLAN.json --logroot results/torchtpu-load/pod_logs/logs --out DIR

PLAN.json: {"cell", "hardware", "tp", "code_commit", "matrix_ref": {"pm-gcs": S, ...}, "t0", "ready_def",
            "baseline_arm": "pm-gcs", "reference_run": RUN,
            "arms": {ARM: {"removed": str|null, "fallback": str|null, "runs": [RUN, ...], "note": str}},
            "failed_attempts": [{"run", "arm", "cause"}]}
Timelines and phases come from torchtpu_matrix_report.run_report (same T0, ready and phase definitions as the matrix).
A run counts only if it became ready and its greedy check returned the reference run's text. Per run this adds the
steady decode rate (request probe `d128`, tok/s after the first token) and `mechanism_check`: what the logs of all 8
workers, both hosts' host/zygote logs and the driver log show about each TPUFlex mechanism.
"""
import argparse
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from torchtpu_matrix_report import PHASES, read, run_report, worker_logs  # noqa: E402

ARM_KNOBS = ('PM_SAMPLER_BANK', 'PM_SB_PRESEED', 'PM_FAST_REPLAY', 'PM_AOT_PRELOAD', 'PM_BANK_PRELOAD', 'PM_AOT_SLIM',
             'PM_SNAP_MODE', 'PM_OVERLAP_RENDERER', 'PM_WARM', 'PM_SKIP_MM_PROBE', 'PM_RAY_ZYGOTE_SOCK',
             'PM_WEIGHTS_WAIT_FILE', 'TTL_CHIP_BIND')


def ints(rx, text):
    return [int(x) for x in re.findall(rx, text)]


def floats(rx, text):
    return [float(x) for x in re.findall(rx, text)]


def mechanism_check(d):
    """Evidence per mechanism from the logs of run dir d (host rank 0 in d, host rank 1 in d/host1)."""
    vlog = read(d / 'vllm_raw.log')
    workers = worker_logs(d, vlog)
    allw = '\n'.join(workers.get(r, '') for r in sorted(workers))
    hosts = [read(d / 'host.log'), read(d / 'host1' / 'host.log')]
    for i, h in enumerate(hosts):  # a retried run id appends to host.log: keep the last attempt only
        j = h.rfind('[host] serve start run=')
        hosts[i] = h[h.rfind('\n', 0, j) + 1:] if j > 0 else h
    rzl = [read(d / 'ray_zygote.log'), read(d / 'host1' / 'ray_zygote.log')]
    try:
        rt_env = json.loads(read(d / 'ray_runtime_env.json'))['env_vars']
    except (ValueError, KeyError):
        rt_env = {}
    m = {'worker_logs': len(workers)}
    m['knobs_in_worker_env'] = {k: rt_env.get(k) for k in ARM_KNOBS}
    # canonical chip binding
    rd = re.findall(r'TTL_RANK_DEVICE rank=(\d+) local_rank=\d+ chip=\d+ device_id=(\d+) coords=\S+, \S+, \S+ mode=(\w+)',
                    allw)
    m['rank_device'] = {int(r): int(dev) for r, dev, _ in rd}
    m['rank_device_identity'] = len(rd) == 8 and all(int(r) == int(dev) for r, dev, _ in rd)
    m['binding_mode'] = sorted({mode for _, _, mode in rd})
    m['chip_bind_errors'] = len(re.findall(r'TTL_CHIP_BIND.*ERROR', allw))
    # sampler bank (+ preseed)
    sb = re.findall(r'PM_SAMPLER_BANK rank=(\d+) loaded=(\d+) \([\d.]+s\) built=(\d+) \([\d.]+s\) stock=(\d+) '
                    r'salt=(\w+) placement=(\w+)', allw)
    m['sampler_bank'] = {
        'ranks': len(sb), 'loaded': sorted({int(x[1]) for x in sb}), 'built': sorted({int(x[2]) for x in sb}),
        'stock': sorted({int(x[3]) for x in sb}), 'salt': sorted({x[4] for x in sb}),
        'placement': sorted({x[5] for x in sb}),
        'load_s_rank0': (floats(r'PM_SAMPLER_BANK rank=0 loaded=\d+ \(([\d.]+)s\)', allw) or [None])[0],
        'stock_path_lines': allw.count('stock path'), 'placement_mismatch_lines': allw.count('PLACEMENT MISMATCH'),
        'preseed_ranks': len(set(re.findall(r'PM_SB_PRESEED rank=(\d+)', allw))),
    }
    # executable bank: replays (backbone buckets), preloads, slim AOT artifact, fast replay
    reps = {}
    for r in re.findall(r'PM_BANK_REPLAY rank=(\d+) ', allw):
        reps[int(r)] = reps.get(int(r), 0) + 1
    m['exec_bank'] = {
        'replays_per_rank': sorted(set(reps.values())), 'ranks_replaying': len(reps),
        'preload_artifacts': sorted(set(ints(r'PM_BANK_PRELOAD rank=\d+ key=\w+ artifacts=(\d+)', allw))),
        'preload_ranks': len(set(re.findall(r'PM_BANK_PRELOAD rank=(\d+) key=', allw))),
        'preload_s_rank0': (floats(r'PM_BANK_PRELOAD rank=0 key=\w+ artifacts=\d+ took=([\d.]+)s', allw) or [None])[0],
        'aot_preloaded_ranks': len(set(re.findall(r'PM_AOT rank=(\d+) load_compiled_function preloaded', allw))),
        'aot_preload_ranks': len(set(re.findall(r'PM_AOT_PRELOAD rank=(\d+) /', allw))),
        'aot_try_load_hit_ranks': len(set(re.findall(r'PM_AOT rank=(\d+) try_load_aot hit=True', allw))),
        'aot_try_load_miss_ranks': len(set(re.findall(r'PM_AOT rank=(\d+) try_load_aot hit=False', allw))),
        'aot_slim_loaded_ranks': len(set(re.findall(r'PM_AOT_SLIM rank=(\d+) loaded ', allw))),
        'aot_slim_written': len(re.findall(r'PM_AOT_SLIM rank=\d+ wrote ', allw)),
        'preload_mismatch_lines': allw.count('does not match its graph'),
    }
    # weights: snapshot restore (+ join) or the stock loader on the streamed checkpoint
    m['weights'] = {
        'snapshot_ranks': len(set(re.findall(r'PM_SNAP rank=(\d+): snapshot present', allw))),
        'no_snapshot_ranks': len(set(re.findall(r'PM_SNAP rank=(\d+): no snapshot', allw))),
        'restore_join_waited_s': floats(r'PM_SNAP_JOIN rank=\d+ at=\w+ waited=([\d.]+)s', allw),
        'restore_wall_s': floats(r'PM_SNAP_JOIN rank=\d+ at=\w+ waited=[\d.]+s restore_wall=([\d.]+)s', allw),
        'snap_wait_s': floats(r'PM_SNAP_WAIT rank=\d+ waited=([\d.]+)s', allw),
        'stock_loader_s': floats(r'Loading weights took ([\d.]+) seconds', allw),
        'ckpt_wait_s': floats(r'PM_WEIGHTS_WAIT rank=\d+ waited=([\d.]+)s', allw),
    }
    # zygotes
    zr = re.search(r'PM_ZYGOTE_READY took=([\d.]+)s', vlog)
    m['engine_zygote'] = {'ready_s': float(zr.group(1)) if zr else None,
                          'clients': len(re.findall(r'PM_ZYGOTE client pid=', vlog))}
    m['ray_zygote'] = {'forks_per_host': [len(re.findall(r'PM_RAY_ZYGOTE fork pid=', z)) for z in rzl],
                       'disabled': sum('PM_RAY_ZYGOTE_DISABLED' in z for z in rzl),
                       'ready_s': floats(r'PM_RAY_ZYGOTE_READY took=([\d.]+)s', '\n'.join(rzl))}
    m['worker_torch_threads'] = sorted(set(ints(r'TTL_WORKER_CPU rank=\d+ torch_threads=(\d+)', allw)))
    # API-server path: warm-up threads (renderer overlap and mm-probe skip log nothing of their own: env above)
    m['warm'] = {'warm_lines': re.findall(r'PM_WARM (\w+) took=', vlog)}
    # delivery: overlapped (launch right after the model dir) or sequential (whole state first)
    fetch = {}
    for name, h, fl in (('host0', hosts[0], d / 'fetch.log'), ('host1', hosts[1], d / 'host1' / 'fetch.log')):
        mm = re.search(r'FETCH_JSON=(\{.*\})', read(fl))
        fj = json.loads(mm.group(1)) if mm else {}
        fetch[name] = {'sequential': 'PM state fetched (sequential)' in h, 'seconds': fj.get('seconds'),
                       'GBps': fj.get('GBps'), 'bytes': fj.get('bytes'), 'errors': len(fj.get('errors', []) or []),
                       'state_done_s': (fj.get('events') or {}).get('state_done')}
        mm = re.search(r'FETCH_CKPT_JSON=(\{.*\})', read(fl.parent / 'fetch_ckpt.log'))
        if mm:
            cj = json.loads(mm.group(1))
            fetch[name]['ckpt'] = {'seconds': cj.get('seconds'), 'GBps': cj.get('GBps'), 'bytes': cj.get('bytes'),
                                   'errors': len(cj.get('errors', []))}
    m['delivery'] = fetch
    m['collected_worker_logs'] = [ints(r'collected (\d+) worker log files', h)[-1:] for h in hosts]
    m['host_min_mem_gb'] = [round(int(x) / 2**20, 1) if x.strip().isdigit() else None
                            for x in (read(d / 'mem_min_available_kb'), read(d / 'host1' / 'mem_min_available_kb'))]
    m['host_min_disk_gb'] = [round(int(x) / 2**20, 1) if x.strip().isdigit() else None
                             for x in (read(d / 'disk_min_available_kb'), read(d / 'host1' / 'disk_min_available_kb'))]
    return m


def decode_tps(d):
    try:
        reqs = json.loads(read(d / 'req_probe.json'))['requests']
    except (ValueError, KeyError):
        return None
    return next((r.get('decode_tok_per_s_after_first') for r in reqs if r.get('name') == 'd128'), None)


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
    runs, arms = {}, {}
    for arm, spec in plan['arms'].items():
        reps = []
        for rid in spec['runs']:
            r = run_report(root, rid)
            r['greedy_ok'] = ref is not None and r.get('greedy_text') == ref
            r['decode_tok_s'] = decode_tps(root / rid)
            r['mechanism_check'] = mechanism_check(root / rid)
            runs[rid] = r
            reps.append(r)
        ok = [r for r in reps if r['ready_s'] is not None and r['greedy_ok']]
        arms[arm] = {
            'removed': spec.get('removed'), 'fallback': spec.get('fallback'),
            'status': 'ok' if len(ok) == len(reps) == 3 else ('partial' if ok else 'failed'),
            'n': len(ok), 'median': med([r['ready_s'] for r in ok]), 'runs': [r['ready_s'] for r in ok],
            'run_ids': [r['run'] for r in ok],
            'phases': {p: med([r['phases'][p] for r in ok]) for p in PHASES},
            'compilation_s': med([r.get('compilation_s') for r in ok]),
            'decode_tok_s': med([r['decode_tok_s'] for r in ok]),
            'decode_tok_s_runs': [r['decode_tok_s'] for r in ok],
            'delta_vs_full_s': None,
            'mechanism_check': {r['run']: r['mechanism_check'] for r in ok},
            'note': spec.get('note', ''),
        }
    base = arms.get(plan['baseline_arm'], {}).get('median')
    for e in arms.values():
        if base is not None and e['median'] is not None:
            e['delta_vs_full_s'] = round(e['median'] - base, 2)
            e['phase_delta_vs_full_s'] = {
                p: (round(e['phases'][p] - arms[plan['baseline_arm']]['phases'][p], 2)
                    if e['phases'][p] is not None and arms[plan['baseline_arm']]['phases'][p] is not None else None)
                for p in PHASES}
    summary = {k: plan.get(k) for k in ('cell', 'hardware', 'tp', 'hosts', 'engine', 'code_commit', 'matrix_ref', 't0',
                                        'ready_def', 'baseline_arm')}
    summary['arms'] = arms
    summary['failed_attempts'] = plan.get('failed_attempts', [])
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'loo_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (out / 'runs.json').write_text(json.dumps({'reference_run': plan['reference_run'], 'reference_text': ref,
                                               'runs': runs}, indent=1) + '\n')
    print(f'{"arm":20} {"st":7} n {"median":>7} {"delta":>7} {"tok/s":>6}  runs')
    for arm, e in arms.items():
        print(f'{arm:20} {e["status"]:7} {e["n"]} {str(e["median"]):>7} {str(e["delta_vs_full_s"]):>7} '
              f'{str(e["decode_tok_s"]):>6}  {e["runs"]} {e["run_ids"]}')
        print('    ' + ' '.join(f'{p}={v}' for p, v in e['phases'].items() if v is not None)
              + f' compilation={e["compilation_s"]}')


if __name__ == '__main__':
    main()
