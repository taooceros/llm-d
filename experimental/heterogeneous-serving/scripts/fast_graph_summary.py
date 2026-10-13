#!/usr/bin/env python3
"""Per-worker start-up clocks for the fast-graph A/B, base vs fast, cold and restore.

Reads <out>/<arm>-deploy.json (worker_init_phases) and <arm>-validation.json
(greedy outputs) for the four arms written by scripts/fast_graph_ab.sh and
writes <out>/summary.json and <out>/summary.md. Every number is a worker
clock difference; no model is applied.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ARMS = {
    'S-cold-base': ('off', False), 'S-cold-fast': ('off', True),
    'S-restore-base': ('read', False), 'S-restore-fast': ('read', True),
}
CLOCKS = ['init_device', 'load_model', 'snapshot_verify', 'snapshot_agree', 'loader_entry', 'graph_build',
          'graph_validate', 'graph_agree', 'snapshot_device_placement', 'graph_update',
          'initialize_from_config', 'precompile_sampling', 'precompile_gather_logprobs', 'precompile_flush',
          'kv_cache_allocate', 'compile_or_warm_up_model']


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def measurement(root, name):
    value = load(root / name)
    return value.get('measurements', value) if value else None


def workers(root, arm):
    measured = measurement(root, f'{arm}-deploy.json')
    if not measured:
        return []
    rows = []
    for instance in measured['deployment']['instances']:
        for worker in instance['initialization']['worker_init_phases']:
            phases = worker['phases']
            seconds = {key: phases[key]["end"] - phases[key]["start"] for key in CLOCKS if key in phases}
            if "snapshot_verify" in phases and "snapshot_device_placement" in phases:
                seconds["gap_sha_end_to_placement"] = (
                    phases["snapshot_device_placement"]["start"] - phases["snapshot_verify"]["end"])
                seconds["gap_placement_end_to_load_end"] = (
                    phases["load_model"]["end"] - phases["snapshot_device_placement"]["end"])
            rows.append({'instance_id': instance['descriptor']['instance_id'],
                         'executor_rank': worker['executor_rank'],
                         'weight_snapshot': worker.get('weight_snapshot', {}), 'seconds': seconds})
    return rows


def median_table(rows):
    keys = sorted({k for r in rows for k in r['seconds']})
    return {k: statistics.median(r["seconds"][k] for r in rows if k in r["seconds"]) for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/restore/FastGraph'))
    parser.add_argument('--reference', type=Path,
                        default=Path.home() / 'hetero-wt/ShardedRestore/results/restore/ShardedRestore/S-reference.json')
    args = parser.parse_args()
    root = args.output
    reference = measurement(args.reference.parent, args.reference.name)
    result = {'arms': {}, 'reference': str(args.reference)}
    for arm, (mode, fast) in ARMS.items():
        rows = workers(root, arm)
        validation = measurement(root, f'{arm}-validation.json')
        deploy = measurement(root, f'{arm}-deploy.json')
        entry = {'mode': mode, 'fast_graph': fast, 'workers': rows,
                 'median': median_table(rows) if rows else None,
                 'fast_graph_reported': sorted({str(r['weight_snapshot'].get('fast_graph')) for r in rows}),
                 'every_worker_restored': all(r['weight_snapshot'].get('restored') is True for r in rows) if rows else None,
                 'greedy_equal_to_reference': bool(validation and validation.get('ok') is True and reference
                     and reference.get('outputs')
                     and all(key in validation and key in reference and validation[key] == reference[key]
                             for key in ('layout', 'manifest_sha256', 'prompts', 'sampling', 'outputs'))),
                 'submission_id': (load(root / f'{arm}-deploy.json') or {}).get('submission_id'),
                 'deploy_ok': None if not deploy else deploy.get('ok')}
        result['arms'][arm] = entry
        entry['worker_evidence_ok'] = (
            len(rows) == 8
            and len({r['instance_id'] for r in rows}) == 4
            and len({(r['instance_id'], r['executor_rank']) for r in rows}) == 8
            and all(r['weight_snapshot'].get('fast_graph') is fast
                    and r['weight_snapshot'].get('mode') == mode
                    and all(clock in r["seconds"] for clock in ("load_model", "graph_build"))
                    and (mode != "read" or r["weight_snapshot"].get("restored") is True) for r in rows))
    deltas = {}
    for base, fast in (('S-cold-base', 'S-cold-fast'), ('S-restore-base', 'S-restore-fast')):
        b, f = result['arms'][base]['median'], result['arms'][fast]['median']
        if b and f:
            deltas[f'{fast} - {base}'] = {k: round(f[k] - b[k], 3) for k in b if k in f}
    result['median_deltas_s'] = deltas
    result['ok'] = all(a['greedy_equal_to_reference'] and a['deploy_ok'] is True
                       and a['worker_evidence_ok'] for a in result['arms'].values())
    (root / 'summary.json').write_text(json.dumps(result, indent=2))
    lines = ['# Fast-graph A/B (layout S, TP8, 4 instances x 2 workers)', '',
             f"Reference greedy outputs: `{args.reference}`", '']
    keys = ['load_model', 'gap_sha_end_to_placement', 'graph_build', 'loader_entry', 'graph_validate', 'graph_update',
            'snapshot_verify', 'snapshot_device_placement', 'initialize_from_config', 'precompile_sampling',
            'precompile_gather_logprobs', 'kv_cache_allocate', 'compile_or_warm_up_model']
    lines.append('| arm | mode | fast | n | restored | greedy=ref | ' + ' | '.join(keys) + ' |')
    lines.append('|' + '---|' * (6 + len(keys)))
    for arm, entry in result['arms'].items():
        med = entry['median'] or {}
        lines.append(f"| {arm} | {entry['mode']} | {entry['fast_graph']} | {len(entry['workers'])} | "
                     f"{entry['every_worker_restored']} | {entry['greedy_equal_to_reference']} | "
                     + " | ".join(f"{med[k]:.3f}" if k in med else "" for k in keys) + " |")
    lines += ['', '## Median deltas (fast - base), seconds', '']
    for name, delta in deltas.items():
        lines.append(f"- {name}: " + ', '.join(f'{k} {v:+.2f}' for k, v in delta.items() if k in keys))
    (root / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'ok': result['ok'], 'median_deltas_s': {k: {kk: vv for kk, vv in v.items() if kk in ('load_model', 'graph_build', 'gap_sha_end_to_placement')} for k, v in deltas.items()}}))
    raise SystemExit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
