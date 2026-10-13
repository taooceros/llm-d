#!/usr/bin/env python3
"""Summarize recorded FastSnapshot evidence without replacing raw measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('results/restore/FastSnapshot'))
    parser.add_argument('--baseline', type=Path, default=Path('/home/hongtao/hetero-wt/ShardedRestore/results/restore/ShardedRestore'))
    args = parser.parse_args()
    out = args.directory
    snapshots = {}
    restores = {}
    for path in sorted(out.glob('*-cold.json')):
        layout = path.name.split('-')[0]
        samples = {}
        baseline_path = args.baseline / f'{layout}-snapshot.json'
        baseline = load(baseline_path)['measurements']['workers'] if baseline_path.is_file() else {}
        for label in ('cold', 'reused'):
            source = out / f'{layout}-{label}.json'
            evidence = load(source)
            if evidence['status'] != 'SUCCEEDED' or evidence['measurements'].get('ok') is not True:
                raise RuntimeError('snapshot sample did not complete successfully')
            rows = []
            for engine, workers in evidence['measurements']['workers'].items():
                for worker in workers:
                    old = next((r for r in baseline.get(engine, [])
                                if r['process_rank'] == worker['process_rank']), None)
                    row = {'engine': engine, **worker,
                           'GBps': worker['bytes'] / worker['snapshot_s'] / 1e9,
                           'baseline_s': old['snapshot_s'] if old else None,
                           'speedup_vs_baseline': old['snapshot_s'] / worker['snapshot_s'] if old else None}
                    rows.append(row)
            samples[label] = {'source': str(source), 'submission_id': evidence['submission_id'],
                              'workers': rows, 'slowest_worker_s': max(r['snapshot_s'] for r in rows),
                              'fastest_worker_s': min(r['snapshot_s'] for r in rows),
                              'bytes_per_host': sorted(set(r['bytes'] for r in rows)),
                              'total_host_bytes': sum(r['bytes'] for r in rows)}
        samples['reused_destination_poison'] = load(out / f'{layout}-poison.json')
        samples['publication_and_sizes'] = load(out / f'{layout}-files.json')
        samples['cold_byte_equality'] = load(out / f'{layout}-source-fingerprint.json')['measurements']['weight_equality']
        samples['cold_file_fingerprint_source'] = str(out / f'{layout}-cold-file-fingerprint.json')
        if not samples['cold_byte_equality']['passed']:
            raise RuntimeError('cold snapshot bytes were not independently verified')
        snapshots[layout] = samples
        source = out / f'{layout}-restored-validation.json'
        evidence = load(source)
        measured = evidence['measurements']
        restored_weight_equality = load(out / f'{layout}-restored-fingerprint.json')['measurements']['weight_equality']
        if evidence['status'] != 'SUCCEEDED' or measured.get('ok') is not True or not measured['greedy_equality']['passed'] or not restored_weight_equality['passed']:
            raise RuntimeError('restore is not qualified by both full-weight and token equality')
        rows = []
        for engine, workers in measured['worker_init_phases'].items():
            for worker in workers:
                phases = worker['phases']
                parts = {name: phase['end'] - phase['start'] for name, phase in phases.items()
                         if name.startswith('snapshot_') or name == 'load_model'}
                rows.append({'engine': engine, 'executor_rank': worker['executor_rank'],
                             'weight_snapshot': worker['weight_snapshot'], 'parts_s': parts})
        release_path = out / f'{layout}-normal-release.json'
        release = load(release_path)
        release_states = {r['state'] for r in release['measurements']['released']}
        producer_released = (release['status'] == 'SUCCEEDED' and release_states == {'REMOVED'}
                             and release['saved_at'] < min(w['phases']['load_model']['start']
                                 for ws in measured['worker_init_phases'].values() for w in ws))
        if not producer_released:
            raise RuntimeError('missing independent-producer-exit restore proof')
        restores[layout] = {'source': str(source), 'submission_id': evidence['submission_id'],
                            'producer_released_before_consumer_deploy': producer_released,
                            'producer_release_source': str(release_path),
                            'workers': rows, 'greedy_equality': measured['greedy_equality'],
                            'weight_equality': restored_weight_equality,
                            'validation_only_fingerprint_sources': [str(out / f'{layout}-{label}-fingerprint.json')
                                                                   for label in ('source', 'restored')],
                            'slowest_snapshot_device_placement_s': max(
                                r['parts_s']['snapshot_device_placement'] for r in rows)}
    if not snapshots:
        raise RuntimeError('no completed cold/reused/restore evidence')
    (out / 'snapshot.json').write_text(json.dumps({
        'schema_version': 1, 'layouts': snapshots,
        'timing_note': 'snapshot_s includes completion and atomic publication. Inspect each worker phase_semantics for overlapping stages. Cold/reused describe destination pages; host-cache and did_copy evidence distinguish source DMA.'}, indent=2) + '\n')
    (out / 'restore.json').write_text(json.dumps({'schema_version': 1, 'layouts': restores,
        'timing_note': 'snapshot_read_tmpfs measures mmap construction, not full physical file reads. Device placement includes grouping, JAX puts, reassembly and completion, not pure DMA. load_model additionally includes CPU model/graph construction; initialization and sampling/gather compilation are separate startup costs. v2 checksum_kind=none performs no byte checksum in the restore timer; validation-only full SHA and W0 token gates are separate.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
