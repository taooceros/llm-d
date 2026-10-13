#!/usr/bin/env python3
"""Consolidate raw snapshot campaign evidence without treating fallback as restore."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def measurement(root, name):
    value = load(root / name)
    return value.get('measurements', value) if value else None


def phases(root, name):
    measured = measurement(root, name)
    if not measured:
        return {}
    return {instance['descriptor']['instance_id']: instance['initialization']['worker_init_phases']
            for instance in measured['deployment']['instances']}


def durations(rows, restored_only=False):
    out = {}
    for name, workers in rows.items():
        out[name] = []
        for worker in workers:
            status = worker.get('weight_snapshot', {})
            row = {'executor_rank': worker['executor_rank'], 'weight_snapshot': status}
            if not restored_only or status.get('restored') is True:
                row['seconds'] = {key: clock['end'] - clock['start']
                                  for key, clock in worker['phases'].items()}
            out[name].append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/restore/ShardedRestore'))
    parser.add_argument('--table', type=Path, default=Path.home() / 'hetero-research/results/reconfig/throughput_table_measured.json')
    parser.add_argument('--mix', type=Path, default=Path.home() / 'hetero-research/results/reconfig/mixA_model.json')
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    result = {'layouts': {}, 'local_verification': load(root / 'local_verification.json')}
    for layout in ('S', 'H'):
        cold = phases(root, f'{layout}-normal-deploy.json')
        restored = phases(root, f'{layout}-restored-deploy.json')
        reference = measurement(root, f'{layout}-reference.json')
        validation = measurement(root, f'{layout}-restored-validation.json')
        snapshots = measurement(root, f'{layout}-snapshot.json')
        pilot = measurement(root, f'{layout}-restored-pilot.json')
        same = bool(reference and validation and validation.get('ok') and reference.get('outputs') == validation.get('outputs'))
        all_restored = bool(restored) and all(worker.get('weight_snapshot', {}).get('restored') is True
                          for workers in restored.values() for worker in workers)
        row = {'greedy_output_equality': same if validation else None,
               'every_worker_restored': all_restored if restored else None,
               'cold_load_per_worker': durations(cold),
               'restore_per_worker': durations(restored, restored_only=True),
               'snapshot': snapshots,
               'memory_after_snapshot': measurement(root, f'{layout}-after-write-memory.json'),
               'greedy_reference_source': f'{layout}-reference.json',
               'greedy_validation_source': f'{layout}-restored-validation.json',
               'W0_after_restore': pilot.get('result') if pilot and all_restored and same else None}
        if row['W0_after_restore'] and layout == 'S':
            makespan = row['W0_after_restore']['makespan_s']
            row['W0_historical_comparison'] = {'historical_makespan_s': 343.3,
                'historical_repetitions_s': [352.4, 334.2], 'restored_makespan_s': makespan,
                'difference_s': makespan - 343.3, 'ratio': makespan / 343.3}
        unchecked = phases(root, f'{layout}-unchecked-deploy.json')
        if unchecked:
            check = measurement(root, f'{layout}-unchecked-validation.json')
            row['unchecked_variant'] = {'label': 'SHA256 verification disabled; not used for transition costs',
                'per_worker': durations(unchecked, restored_only=True),
                'greedy_output_equality': bool(check and reference and check.get('ok') and check.get('outputs') == reference.get('outputs'))}
        result['layouts'][layout] = row
    result['ok'] = all(row['greedy_output_equality'] is True and row['every_worker_restored'] is True
                       and row['W0_after_restore'] is not None for row in result['layouts'].values())
    result['cleanup'] = measurement(root, 'cleanup-tmpfs.json')
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    transitions = load(root / 'transitions.json')
    if not transitions:
        transitions = {'status': 'not measured', 'reason': 'campaign did not reach both successful restored transitions'}
        (root / 'transitions.json').write_text(json.dumps(transitions, indent=2) + '\n')
    costs = [row['cost_s'] for row in transitions if row.get('cost_s') is not None] if isinstance(transitions, list) else []
    if len(costs) != 2 or not result['ok']:
        (root / 'breakeven.json').write_text(json.dumps({'status': 'not recomputed', 'reason': 'requires both measured transition costs and successful no-fallback greedy validation'}, indent=2) + '\n')
        print(json.dumps({'ok': result['ok'], 'transitions_measured': len(costs)}))
        return
    table_copy = root / 'throughput_table_measured.json'
    table_copy.write_text(args.table.read_text())
    mix_copy = root / 'mixA_model.json'
    mix_copy.write_text(args.mix.read_text())
    table_path = root / 'breakeven_table.json'
    subprocess.run([sys.executable, 'scripts/reconfig_breakeven.py', '--table', str(table_copy),
                    '--cost', str(min(costs)), str(max(costs)), '--json', str(table_path)], check=True)
    mix = load(mix_copy)
    table = load(table_copy)
    # Only H<->S costs were measured. Applying them to M is explicitly a
    # projection; partial overlap scales by the independently measured ratio.
    partial_fraction = mix['C_partial_effective_s'] / mix['C_full_s']
    mix_rows = []
    for cost in sorted(set(costs)):
        for mode, estimated_cost in [('full_redeploy_restore', cost),
                                     ('partial_restore', cost * partial_fraction)]:
            copies = (estimated_cost + mix['static_overlap_forfeited_s']) / mix['gain_per_W0_copy_s']
            mix_rows.append({'mode': mode, 'status': 'ESTIMATE',
                'source_measured_full_restore_C_s': cost, 'effective_cost_s': estimated_cost,
                'assumption': 'M-to-S full cost approximated by measured H/S restore range; partial/full effective-cost ratio held fixed' if mode == 'partial_restore' else 'M-to-S full cost approximated by measured H/S restore range',
                'overlap_forfeited_s': mix['static_overlap_forfeited_s'],
                'gain_per_W0_copy_s': mix['gain_per_W0_copy_s'],
                'breakeven_W0_copies': copies, 'breakeven_output_tokens': copies * table['W0']['output_tokens']})
    report = {'measured_restore_C_range_s': [min(costs), max(costs)],
              'table': load(table_path), 'table_cost_assumption': 'Measured H/S restore cost range applied to every beneficial layout switch; M costs projected, not measured',
              'mix_A': mix_rows, 'partial_effective_cost_fraction': partial_fraction,
              'sources': {'cost': 'transitions.json', 'throughput': table_copy.name, 'mix_terms': mix_copy.name}}
    (root / 'breakeven.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'ok': result['ok'], 'measured_restore_C_range_s': report['measured_restore_C_range_s']}))


if __name__ == '__main__':
    main()
