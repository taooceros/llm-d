#!/usr/bin/env python3
"""Summarize measured real-engine transitions, retaining blocked gates as null."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def analyze(run_path, output, standalone_path=None):
    run = json.loads(run_path.read_text())
    measured = run['measurements']
    run_id = run['submission_id']
    output.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + '\n')
    for worker in measured['workers']:
        if worker.get('parameter_catalogue') is not None:
            save('parameter_catalogue.json', {'run_id': run_id, 'process': worker['process'],
                'parameters': worker['parameter_catalogue'], 'evidence_file': str(run_path)})
    cache_rows = []
    writers = {}
    for worker in measured['workers']:
        warmups = [p for p in worker['phases'] if p['phase'] == 'compile_or_warm_up_model']
        allocations = [p for p in worker['phases'] if p['phase'] == 'initialize_from_config']
        cache_rows.append({'process': worker['process'], 'warmups': warmups, 'kv_allocations': allocations})
        for event in worker.get('cache_writer_events', []):
            if event['eligible_writer'] and event['backend'] == 'tpu':
                writers.setdefault(event['cache_key'], set()).add(event['process'])
    conflicts = {key: sorted(ranks) for key, ranks in writers.items() if len(ranks) > 1}
    patched = any(worker.get('submesh_cache_writer', False) for worker in measured['workers'])
    save('compile_cache_check.json', {'run_id': run_id, 'evidence_file': str(run_path),
        'policy': 'prototype_lowest_participating_rank' if patched else 'native_global_process_0',
        'per_process': cache_rows, 'eligible_TPU_writer_processes_by_key': {key: sorted(ranks) for key, ranks in writers.items()},
        'competing_writer_keys': conflicts, 'writers_unique': not conflicts if writers else None,
        'writer_events_semantics': 'Eligibility calls, not filesystem-write counters; native minimum-duration/callback/error policies still apply.',
        'native_guard_source': 'JAX 0.10.2 jax/_src/compiler.py:802-805',
        'TPU_assignment_key_source': 'JAX 0.10.2 jax/_src/cache_key.py:129,327'})
    lifetime = []
    for worker in measured['workers']:
        pids = {worker['pid']}
        for transition in measured['transitions']:
            for report in transition.get('workers', []):
                if report['process'] == worker['process']:
                    pids.add(report['pid'])
        lifetime.append({'process': worker['process'], 'pids': sorted(pids),
            'runtime_initializations': sum(p['phase'] == 'runtime_and_runner_init' for p in worker['phases']),
            'initial_checkpoint_loads': sum(p['phase'] == 'initial_checkpoint_load' for p in worker['phases'])})
    save('worker_lifetime_check.json', {'run_id': run_id, 'per_process': lifetime,
        'one_runtime_and_load_per_kept_process': all(len(w['pids']) == 1 and w['runtime_initializations'] == 1 and w['initial_checkpoint_loads'] == 1 for w in lifetime),
        'scope': 'Instrumented worker/runtime lifetime and checkpoint-loader entry; transitions attach the supplied resident model instead of the native builder.'})

    baselines = {'restore_C_s': [221.4, 228.6],
                 'restore_source': 'results/restore/ShardedRestore/breakeven.json',
                 'full_redeploy_C_s': 419,
                 'full_source': 'Measured median 419 s provided in assignment; LoadProfiler results document its loader/initialization components',
                 'comparability': 'previous 32-chip deployment timeline vs one-box drain-to-first-token; not identical endpoints'}
    def timing_report(worker):
        return {key: value for key, value in worker.items()
                if key not in ('parameter_catalogue', 'cache_writer_events')}
    for target, filename in [('2TP8', 'transition_TP16_to_2TP8.json'), ('TP16', 'transition_2TP8_to_TP16.json')]:
        transitions = [t for t in measured['transitions'] if t['target'] == target]
        transitions = [dict(t, **{key: [timing_report(w) for w in t[key]]
            for key in ('workers', 'workers_after_reshard') if key in t}) for t in transitions]
        save(filename, {'run_id': run_id, 'evidence_file': str(run_path), 'target': target,
            'status': 'measured' if transitions and all(t['phases_complete'] for t in transitions) else 'blocked',
            'C_s': [t['C_s'] for t in transitions], 'transitions': transitions,
            'partial_worker_timings': [timing_report(w) for w in measured['workers']], 'failure': measured['failure'],
            'comparison': baselines})
    references = {}
    for tp, layout in [(16, 'H'), (8, 'S')]:
        path = ROOT / f'results/restore/ShardedRestore/{layout}-reference.json'
        ref = json.loads(path.read_text())['measurements']
        references[tp] = {'file': str(path.relative_to(ROOT)), 'sampling': ref['sampling'],
                          'prompts': ref['prompts'], 'outputs': next(iter(ref['outputs'].values())),
                          'manifest_sha256': ref['manifest_sha256']}
    checks = []
    for name, rows in measured['serving'].items():
        tp = 16 if 'TP16' in name else 8
        reference = references[tp]
        for engine, row in enumerate(rows):
            actual = row['outputs']
            expected = reference['outputs']
            mismatches = [i for i in range(max(len(actual), len(expected)))
                          if i >= len(actual) or i >= len(expected) or actual[i] != expected[i]]
            checks.append({'stage': name, 'engine': engine, 'tp': tp,
                           'token_identical': not mismatches, 'mismatching_prompt_indices': mismatches,
                           'reference_file': reference['file']})
    after = [row for row in checks if row['stage'].startswith('switch')]
    save('greedy_check.json', {'run_id': run_id, 'status': 'measured' if after else 'blocked',
        'all_reconfigured_token_identical': all(row['token_identical'] for row in after) if after else None,
        'checks': checks, 'prompt_count': 32, 'sampling': references[16]['sampling'],
        'manifest_sha256': references[16]['manifest_sha256'], 'failure': measured['failure']})
    def rates(serving, selector):
        result = []
        for name, rows in serving.items():
            if selector(name):
                for engine, row in enumerate(rows):
                    seconds = row['end'] - row['first_token']
                    result.append({'stage': name, 'engine': engine,
                        'batch_tokens_per_s': row['tokens_per_s'],
                        'decode_tokens_per_s': (row['output_tokens'] - len(row['outputs'])) / seconds,
                        'decode_seconds': seconds, 'output_tokens': row['output_tokens']})
        return result
    throughput = rates(measured['serving'], lambda name: '2TP8' in name)
    comparison = {'run_id': run_id, 'evidence_file': str(run_path),
        'status': 'partial' if throughput else 'blocked', 'in_runtime_TP8': throughput,
        'standalone_TP8': [], 'failure': measured['failure'],
        'decode_metric': '(output tokens - one first token/request) / wall time from first batch token to batch completion; includes remaining prefill and batch tail, not isolated steady-state kernel throughput',
        'concurrency': 'two TP8 engines serve concurrent batches in the resident runtime; standalone control runs one engine, both using the same synchronous prototype executor',
        'reason': 'No matched two-controller standalone batch supplied.'}
    if standalone_path is not None:
        standalone = json.loads(standalone_path.read_text())
        control = standalone['measurements']
        if control['mode'] != 'standalone_TP8_control' or control['runtime_process_count'] != 2:
            raise ValueError('control must be the matched two-controller standalone TP8 experiment')
        if control['prompts'] != measured['prompts'] or control['sampling'] != measured['sampling']:
            raise ValueError('control prompt/sampling contract differs')
        baseline = rates(control['serving'], lambda name: name == 'initial_TP8')
        comparison.update({'standalone_run_id': standalone['submission_id'],
            'standalone_evidence_file': str(standalone_path), 'standalone_TP8': baseline,
            'control_token_identical': control['ok'], 'control_failure': control['failure']})
        if baseline and throughput and control['ok']:
            rate = baseline[0]['decode_tokens_per_s']
            for row in throughput:
                row['decode_speed_ratio_to_standalone'] = row['decode_tokens_per_s'] / rate
                row['decode_slowdown_percent'] = 100 * (1 - row['decode_speed_ratio_to_standalone'])
            comparison.update({'status': 'measured', 'reason': None,
                'qualification_limit': 'One short fixed batch/control, not a sustained full-slice throughput qualification; no pass threshold imposed.'})
    save('throughput_check.json', comparison)
    return {'run_id': run_id, 'ok': measured['ok'], 'greedy_checks': checks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ici/InPlaceEngine')
    parser.add_argument('--standalone', type=Path, help='matched standalone TP8 control evidence')
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.output, args.standalone), indent=2))
