#!/usr/bin/env bash
# Unfrozen 1-rep pilot: does a looser migration trigger fire, and what does it cost?
# Usage: scripts/run_pilot.sh <layout> <arm>... ; env WORKLOADS, POLICY, RUN_ID, HEAD_POD
set -euo pipefail
cd "$(dirname "$0")/.."
layout=$1; shift
workloads=${WORKLOADS:-W0 W1}
policy=${POLICY:-'{"source_pressure_free_block_fraction":0.25}'}
run_id=${RUN_ID:-pilot-$(date +%Y%m%dT%H%M%S)}
head_pod=${HEAD_POD:-tpu-ray-cluster-vllm-tpu-head-r5tfg}
mkdir -p results/pilot
for workload in $workloads; do
  for arm in "$@"; do
    out=results/pilot/${run_id}_${workload}_${arm}.json
    .venv/bin/python rcm_exec run scripts/hetero_ctl.py run-arm --layout "$layout" \
      --head-pod "$head_pod" --output "$out" --timeout 3000 -- \
      --arm "$arm" --workload "$workload" --run-id "$run_id" \
      --unfrozen-diagnostic --policy "$policy" 2>&1 | grep -E "ARM_RESULT|RUN_FINISHED|Reason" || true
    .venv/bin/python rcm_exec resume --abort >/dev/null 2>&1 || true
  done
done
