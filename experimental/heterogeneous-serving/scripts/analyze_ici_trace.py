#!/usr/bin/env python3
"""Identify TPU collective transfers versus host-copy events in JAX JSON traces."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('profile', type=Path)
p.add_argument('--run', type=Path, help='Run evidence associated with the selected profile timestamp')
p.add_argument('--timestamp', help='Exact JAX profile directory timestamp, to exclude earlier runs')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows = []
for path in sorted(a.profile.rglob('*.trace.json.gz')):
    if a.timestamp and a.timestamp not in path.parts:
        continue
    raw = path.read_bytes()
    events = json.loads(gzip.decompress(raw))['traceEvents']
    counts = collections.Counter(e.get('name', '') for e in events)
    collective = [e for e in events if e.get('name') == 'collective-permute-start']
    host_copies = [e for e in events if any(token in e.get('name', '').lower()
                    for token in ('hosttodevice', 'devicetohost', 'host_to_device', 'device_to_host', 'transfertodevice', 'transferfromdevice', 'infeed', 'outfeed'))]
    rows.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                 'events': len(events), 'collective_permute_start_count': len(collective),
                 'collective_permute_done_count': counts['collective-permute-done'],
                 'host_parameter_copy_event_count_by_name': len(host_copies),
                 'collective_example': collective[0] if collective else None,
                 'reshard_execution_count': sum(v for k,v in counts.items() if k.startswith('CommonPjRtLoadedExecutable::Execute (jit__device_put_reshard)'))})
run = None
if a.run:
    opener = gzip.open if a.run.suffix == '.gz' else open
    with opener(a.run, 'rt') as stream:
        run = json.load(stream)
collective_count = sum(r['collective_permute_start_count'] for r in rows)
host_copy_count = sum(r['host_parameter_copy_event_count_by_name'] for r in rows)
a.output.write_text(json.dumps({'traces': rows,
    'tpu_collective_observed': collective_count > 0,
    'host_copy_event_count_by_name': host_copy_count,
    'conclusion': 'Device-side TPU collective-permute observed with no host-copy event names' if collective_count and not host_copy_count else 'Selected profile does not provide the same no-host-copy collective evidence',
    'interpretation': 'Native TPU collective transfer supports ICI/no-host-weight-roundtrip pathway. No physical-link counters or wire-byte measurement; bytes_accessed is HBM accounting, not network traffic.',
    'provenance': {'source': str(a.run) if a.run else None,
                   'submission_id': run['submission_id'] if run else None,
                   'profile_timestamp': a.timestamp,
                   'note': 'Run association is supplied by experiment collection metadata; Ray run IDs are not encoded in JAX traces.'}}, indent=2) + '\n')
