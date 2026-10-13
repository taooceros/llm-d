#!/usr/bin/env python3
"""RCM trace collector: copy worker artifacts and remove this experiment's paths."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omp_rcm import StageRunner, stage

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()


@stage(name='ici_trace_collection', order_idx=0, expected_s=180)
def collect(state, timing):
    m = json.loads(a.run.read_text())['measurements']
    namespace = 'llm-d-optimized-baseline'
    pods = json.loads(subprocess.check_output(['kubectl', '-n', namespace, 'get', 'pods', '-o', 'json'], timeout=60))['items']
    rows = []
    a.output.mkdir(parents=True, exist_ok=True)
    for host in m['instance']['hosts']:
        matches = [x for x in pods if x['spec'].get('nodeName') == host['hostname'] and x['metadata']['labels'].get('ray.io/node-type') == 'worker']
        if len(matches) != 1:
            raise ValueError(f'Expected one existing worker pod on {host["hostname"]}, got {len(matches)}')
        pod = matches[0]['metadata']['name']
        rank = host['process_rank']
        remote = f'/tmp/ici-profile-process{rank}'
        destination = a.output / f'process{rank}'
        copied = subprocess.run(['kubectl', '-n', namespace, 'cp', '-c', 'ray-worker', f'{pod}:{remote}', str(destination)], capture_output=True, text=True, timeout=120)
        row = {'pod': pod, 'path': remote, 'copy_returncode': copied.returncode, 'copy_stderr': copied.stderr}
        if copied.returncode == 0:
            removed = subprocess.run(['kubectl', '-n', namespace, 'exec', pod, '-c', 'ray-worker', '--', 'rm', '-rf', remote], capture_output=True, text=True, timeout=60)
            row['cleanup_returncode'] = removed.returncode
        rows.append(row)
        timing.log(f'{pod} copied={copied.returncode == 0}')
    (a.output / 'collection.json').write_text(json.dumps({'source': str(a.run), 'workers': rows}, indent=2) + '\n')


if __name__ == '__main__':
    runner = StageRunner(silence_timeout_s=300)
    runner.register(collect)
    if not runner.run():
        raise SystemExit(1)
