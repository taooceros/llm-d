#!/usr/bin/env python3
"""Recover/clean the existing key-only submission; never submit or retry work."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omp_rcm import StageRunner, stage
from omp_rcm.ray_job import RayJobSupervisor

JOB = 'raysubmit_iYjGTy27Ui5kiiPh'
OUT = ROOT / 'results/overlap/OverlapHelper/key-only-s-20261002a'


@stage(name='recover_existing_key_only', order_idx=1, expected_s=30)
def recover(state, timing):
    supervisor = RayJobSupervisor(timing_engine=timing)
    supervisor.attach_job(JOB, name='overlap-key-only-recovery')
    status, terminal = supervisor.poll_job_status(JOB)
    before = status
    stopped = False
    if not terminal:
        stopped = supervisor.stop_job(JOB)
        status, terminal = supervisor.poll_job_status(JOB)
    logs = supervisor.poll_job_logs(JOB, stdout_stream=False)
    measurements = supervisor.extract_json_measurements(logs)
    result = {'submission_id': JOB, 'status': status, 'terminal': terminal,
              'measurements': measurements,
              'recovery': {'reason': 'outer 1800-second queue+shell timeout killed original supervisor',
                           'status_before_cleanup': before, 'stop_requested': stopped,
                           'new_cluster_submissions': 0}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'recovered-job.log').write_text(logs)
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'submission_id': JOB, 'status': status, 'terminal': terminal,
                      'measurement_keys': list(measurements), 'stop_requested': stopped}), flush=True)


if __name__ == '__main__':
    runner = StageRunner(silence_timeout_s=30)
    runner.register(recover)
    if not runner.run():
        raise SystemExit(1)
