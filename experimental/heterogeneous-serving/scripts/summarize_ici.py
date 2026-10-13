#!/usr/bin/env python3
"""Aggregate saved RCM evidence; latency uses each repetition's slowest controller."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import statistics


def load(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return json.load(stream)

def stats(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def native_compile_summary(rows):
    cold = [r for r in rows if r['cold'] and r.get('implicit_compile_spans') is not None]
    if not cold:
        return None
    phases = {
        'trace_s': '/jax/core/compile/jaxpr_trace_duration',
        'mlir_lower_s': '/jax/core/compile/jaxpr_to_mlir_module_duration',
        'backend_compile_or_cache_restore_s': '/jax/core/compile/backend_compile_duration',
    }
    result = {name: stats([sum(s['elapsed_s'] for s in r['implicit_compile_spans']
                              if s['event'] == event) for r in cold])
              for name, event in phases.items()}
    unions = []
    for r in cold:
        total, end = 0., float('-inf')
        for span in sorted(r['implicit_compile_spans'], key=lambda s: s['start_s']):
            total += max(0., span['end_s'] - max(end, span['start_s']))
            end = max(end, span['end_s'])
        unions.append(total)
    result['host_span_union_s'] = stats(unions)
    result['span_counts'] = stats([len(r['implicit_compile_spans']) for r in cold])
    result['controllers'] = len(cold)
    return result



def summarize(paths):
    output, groups = [], collections.defaultdict(list)
    for path in paths:
        saved = load(path)
        m = saved['measurements']
        if not (m.get('ok') or m.get('reshard_ok')):
            raise ValueError(f'{path} failed: {m.get("error")}')
        for process in m['processes']:
            for row in process['rows']:
                groups[(row['size'], row['route'], row['direction'], saved['submission_id'])].append(
                    dict(row, process_index=process['process_index']))
    for (size, route, direction, submission), rows in groups.items():
        warm = [r for r in rows if not r['cold']]
        repetitions = collections.defaultdict(list)
        for row in warm:
            repetitions[row['repetition']].append(row)
        slowest = [max(r['elapsed_s'] for r in rs) for rs in repetitions.values()]
        bytes_total = rows[0]['logical_bytes_moved']
        boxes = len({r['process_index'] // 4 for r in rows})
        peaks = [d['peak_bytes_in_use'] for r in rows for phase in ('memory_before', 'memory_after')
                 for d in r[phase].values() if d]
        output.append({'size': size, 'route': route, 'direction': direction, 'submission_id': submission,
                       'warm_repetitions': len(repetitions), 'controllers': len({r['process_index'] for r in rows}),
                       'slowest_controller_latency_s': stats(slowest),
                       'all_controller_latency_s': stats([r['elapsed_s'] for r in warm]),
                       'cold_controller_latency_s': stats([r['elapsed_s'] for r in rows if r['cold']]),
                       'logical_bytes_moved_per_box': bytes_total // boxes, 'logical_bytes_moved_total': bytes_total,
                       'effective_total_GB_s': stats([bytes_total / t / 1e9 for t in slowest]),
                       'per_chip_lifetime_HBM_peak_bytes': stats(peaks),
                       'HBM_peak_note': 'device.memory_stats peak is cumulative since runtime start; may include earlier routes and verification',
                       'generation_compile_s_all_controllers': stats([r['generation_compile_s'] for r in rows if r['cold']]),
                       'explicit_route_compile_s_all_controllers': stats([r['route_compile_s'] for r in rows if r['cold']]),
                       'native_cold_compilation': native_compile_summary(rows),
                       'compile_note': 'Generation/check/slice lowering measured separately. Native cold trace/lower/backend spans are host service times; backend includes compile or cache restore. Null native stats mean uninstrumented prior data, not zero. Spans cannot be subtracted to claim pure TPU transfer time. First route trial is cold-labelled; shared runtime caches remain intact.',
                       'verified': all(c['errors'] == 0 for r in rows for c in r['checksums'] or []),
                       'traffic_note': rows[0]['interpretation']})
    return {'sources': [str(p) for p in paths], 'rows': output}


def summarize_other(path, kind):
    saved = load(path)
    m = saved['measurements']
    common = {'source': str(path), 'submission_id': saved['submission_id'], 'ok': m['ok']}
    if kind == 'orders':
        inferred = {
            'a': [(0,0),(1,0),(0,1),(1,1),(1,3),(0,3),(1,2),(0,2)],
            'b': [(2,0),(3,0),(2,1),(3,1),(3,3),(2,3),(3,2),(2,2)],
            'h': [(0,0),(1,0),(2,0),(3,0),(3,1),(2,1),(1,1),(1,2),(2,2),(3,2),(3,3),(2,3),(1,3),(0,3),(0,2),(0,1)]}
        comparisons = {name: all([tuple(d['coords'][:2]) for d in o['orders'][name]] == expected for o in m['orders'])
                       for name, expected in inferred.items()}
        return {**common, 'processes': m['orders'], 'instance': m['instance'], 'coordinator': m['coordinator'],
                'controller_ids': m['controller_ids'], 'matches_feasibility_inferred_order': comparisons,
                'inferred_source': 'PathwaysReshard/feasibility.md lines95-106',
                'native_S_process_groups': [[0,2],[1,3]]}
    processes = m['processes']
    if kind == 'subgroup':
        return {**common, 'processes': processes, 'all_results_correct': all(p['correct'] for p in processes),
                'elapsed_s': stats([p['elapsed_s'] for p in processes]), 'tested_process_subsets': [[0,1],[2,3]],
                'geometry_note': 'Requested controller subsets are horizontal TP8s; native S boxes are vertical {0,2}/{1,3}'}
    if kind == 'mixed':
        if m['mode'] == 'campaign':
            if any('mixed' not in p for p in processes):
                return {**common, 'all_results_correct': False, 'status': 'M_actor_failed',
                        'error': m.get('error'), 'reshard_measurements_preserved': m.get('reshard_ok', False)}
            processes = [p['mixed'] for p in processes]
        if any(p['role'] == 'subset_compile_failed' for p in processes):
            return {**common, 'all_results_correct': False, 'processes': processes,
                    'status': 'independent_subset_compile_failed', 'reshard_measurements_preserved': True}
        if not all(p['correct'] for p in processes):
            return {**common, 'all_results_correct': False, 'processes': processes,
                    'status': 'independent_compute_failed', 'reshard_measurements_preserved': True}
        transfers = [p for p in processes if p['role'] == 'reshard']
        computes = [p for p in processes if p['role'] == 'independent_compute']
        start = max(min(r['start_wall_ns'] for r in p['rows']) for p in transfers)
        end = min(max(r['end_wall_ns'] for r in p['rows']) for p in transfers)
        overlaps = [{'process_index': p['process_index'], 'compute_steps_observed_inside_reshard_window':
                     [e['step'] for e in p['progress_events'] if start <= e['wall_ns'] <= end]}
                    for p in computes]
        return {**common, 'all_results_correct': all(p['correct'] for p in processes), 'processes': processes,
                'overlap_window_wall_ns': [start,end], 'independent_progress_during_reshard': overlaps,
                'concurrent_steps_per_s': stats([p['steps']/p['elapsed_s'] for p in computes]),
                'baseline_steps_per_s': stats([p['baseline_steps']/p['baseline_s'] for p in computes]),
                'clock_note': 'Wall clocks from different hosts; event overlap is evidence of progress, not a synchronized throughput A/B study'}
    raise ValueError(kind)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--kind', choices=('reshard','orders','subgroup','mixed','failures'), default='reshard')
    args = p.parse_args()
    if args.kind == 'failures':
        result = {'failures': []}
        for path in args.inputs:
            saved = load(path)
            m = saved['measurements']
            result['failures'].append({'source': str(path), 'submission_id': saved['submission_id'],
                                      'error': m['error'], 'cleanup': m['cleanup']})
    else:
        result = summarize(args.inputs) if args.kind == 'reshard' else summarize_other(args.inputs[0], args.kind)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
