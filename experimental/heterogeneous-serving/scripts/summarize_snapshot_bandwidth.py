#!/usr/bin/env python3
"""Build the acceptance bandwidth JSON from verified, completed measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def _summarize_pinned(data, model_bytes):
    """Keep verified partial-payload runtime DMA distinct from full-model snapshots."""
    if data.get('status') != 'SUCCEEDED':
        raise RuntimeError('runtime-pinned probe did not succeed')
    probe = data['measurements']['probe']
    cases = probe['cases']
    eligible = [row for row in cases
                if row.get('mechanism') == 'preallocated_pinned_donation'
                and row.get('ok') is True
                and row.get('verified_bytes') == row['bytes']
                and len(row.get('source_cache_none_before', [])) == row['chips']
                and all(row['source_cache_none_before'])
                and len(row.get('destination_reused', [])) == row['chips']
                and all(row['destination_reused'])]
    best = {}
    for chips in (1, 4):
        choices = [row for row in eligible if row['chips'] == chips]
        if not choices:
            raise RuntimeError(f'missing verified runtime-pinned component for {chips} chips')
        best['chip0' if chips == 1 else 'host4'] = max(choices, key=lambda row: row['GBps'])
    return {'submission_id': data['submission_id'],
            'source_sha256': data['measurements']['source_sha256'],
            'best_verified_component_samples': best,
            'verified_pinned_sample_count': len(eligible),
            'total_recorded_samples': len(cases),
            'timing': probe['timing'],
            'scope': 'Fresh source, reused runtime-owned pinned destination; one synthetic large array per chip. This is neither a full 1,028-array-per-chip model transfer nor a persistent snapshot.',
            'full_host_payload_divided_by_partial_payload_component_s': model_bytes / (best['host4']['GBps'] * 1e9),
            'extrapolation_not_measurement': True,
            'excluded_from_component_timer': ['destination allocation and first touch', 'JIT compilation', 'logical NumPy host-view copy', 'persistent-file copy', 'manifest publication', 'byte verification'],
            'interpretation': 'Verified runtime-pinned DMA can exceed the standard collector path substantially. Do not use the standard collector ceiling to claim a hardware-imposed impossibility of a one-second snapshot. Production preserves the safe logical-array path; physical/tiled pointer persistence is not implemented.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('results/restore/FastSnapshot'))
    parser.add_argument('--mmap-probe', type=Path)
    parser.add_argument('--pinned-probe', type=Path)
    parser.add_argument('--checksum-probe', type=Path)
    args = parser.parse_args()
    directory = args.directory
    sources = [directory / name for name in ('bandwidth-first-run.json', 'bandwidth-corrected.json',
                                             'bandwidth-host-copy.json')]
    runs = []
    samples = []
    copies = []
    inspections = []
    for source in sources:
        if not source.is_file():
            continue
        data = load(source)
        measured = data['measurements']
        archive = directory / source.name.replace('.json', '-source.py')
        entry = {'source': str(source), 'submission_id': data['submission_id'],
                 'ray_status': data['status'], 'all_ops_ok': measured['ok'],
                 'failed_ops': measured['failed_ops'], 'skipped_ops': measured['skipped_ops'],
                 'unavailable_ops': measured['unavailable_ops'], 'cleanup': measured['cleanup'],
                 'script_sha256': measured.get('script_sha256'),
                 'script_path': measured.get('script_path'), 'config': measured['config']}
        if archive.is_file():
            entry.update(submitted_source_archive=str(archive),
                         submitted_source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        runs.append(entry)
        for operation in measured['ops']:
            op = operation['op']
            for host in operation['hosts']:
                if op['kind'] == 'inspect':
                    inspections.append({'source': str(source), **host})
                elif op['kind'] == 'host_copy':
                    copies.append({'source': str(source), 'component_not_snapshot': True, **host})
                elif op['kind'] in ('d2h', 'h2d'):
                    for sample in host.get('samples', []):
                        if not sample.get('verified') or not sample.get('gating_eligible'):
                            continue
                        samples.append({'source': str(source), 'submission_id': data['submission_id'],
                                        'op_id': operation['id'], 'direction': op['kind'],
                                        'workload': op['workload'], 'mechanism': op['mechanism'],
                                        'scope': op['scope'], 'node_ip': host['node_ip'],
                                        'payload_bytes': host['payload_bytes'],
                                        'bytes_per_chip': host['bytes_per_chip'],
                                        'arrays_per_chip': host['arrays_per_chip'], **sample})
    if not samples:
        raise RuntimeError('no verified transfer measurements')
    ceilings = {}
    for workload in ('model', 'any_measured_array'):
        ceilings[workload] = {}
        for direction in ('d2h', 'h2d'):
            ceilings[workload][direction] = {}
            for scope in ([0], [1], [2], [3], [0, 1, 2, 3]):
                choices = [s for s in samples if s['direction'] == direction and s['scope'] == scope
                           and (workload != 'model' or s['workload'] == 'model')]
                if not choices:
                    raise RuntimeError(f'missing verified {direction}/{workload}/{scope}')
                best = max(choices, key=lambda s: s['gbps'])
                key = 'host4' if len(scope) == 4 else f'chip{scope[0]}'
                ceilings[workload][direction][key] = best
    host_ceiling = ceilings['any_measured_array']['d2h']['host4']['gbps']
    model_bytes = ceilings['model']['d2h']['host4']['payload_bytes']
    probes = {}
    for name, source in (('caller_owned_mmap', args.mmap_probe), ('preallocated_runtime_pinned', args.pinned_probe),
                         ('on_device_checksum', args.checksum_probe)):
        if source is not None:
            probes[name] = {'source': str(source),
                            'result_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                            'evidence': load(source)}
    pinned = _summarize_pinned(probes['preallocated_runtime_pinned']['evidence'], model_bytes) if 'preallocated_runtime_pinned' in probes else None
    result = {'schema_version': 1, 'runs': runs, 'worker_inspection': inspections,
              'verified_transfer_samples': samples, 'ceilings': ceilings,
              'destination_costs': copies, 'additional_pjrt_probes': probes,
              'runtime_pinned_component': pinned,
              'host_payload_bytes': model_bytes,
              'best_observed_standard_d2h_GBps': host_ceiling,
              'host_payload_over_standard_d2h_ceiling_s': model_bytes / (host_ceiling * 1e9),
              'measurement_scope': 'Best observed tested transfer path, not a vendor PCIe link specification. Fresh source device buffers and all transfer completions are inside the gating methodology; array generation/compile and byte verification are outside transfer clocks.',
              'integrity_tradeoff': 'Production v2 defaults to no byte checksum, retaining manifest/tree/device/file identity. Optional SHA256 is source-derived. Source/file/restored SHA256 and W0 token equality validate the experiment outside performance clocks.',
              'component_note': 'Destination memcpy/checksum and transfer samples are components, not complete snapshots. Integrated snapshot_s additionally includes file allocation, overlap, flush/fsync and atomic manifest publication. Reused destination is poisoned; same live production source may have PJRT internal host-cache warmth.'}
    (directory / 'bandwidth.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
