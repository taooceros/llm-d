#!/usr/bin/env bash
# Unfrozen 1-rep pilot: does a looser migration trigger fire, and what does it cost?
# Usage: scripts/run_pilot.sh <layout> <arm>... ; env WORKLOADS, POLICY, RUN_ID, HEAD_POD
# EPP_CONFIG=plain|reserve|latebinding sends the arm through the llm-d EPP router
# (scripts/epp_router.sh must have rolled out that config); routing then comes
# from deploy/epp/$EPP_CONFIG.yaml, so POLICY defaults to the empty amendment.
set -euo pipefail
cd "$(dirname "$0")/.."
layout=$1; shift
workloads=${WORKLOADS:-W0 W1}
epp_args=()
if [ -n "${EPP_CONFIG:-}" ]; then
  epp_args=(--epp-config "$EPP_CONFIG")
  policy=${POLICY:-'{}'}
else
  policy=${POLICY:-'{"source_pressure_free_block_fraction":0.25}'}
fi
run_id=${RUN_ID:-pilot-$(date +%Y%m%dT%H%M%S)}
head_pod=${HEAD_POD:-tpu-ray-cluster-vllm-tpu-head-r5tfg}
mkdir -p results/pilot
for workload in $workloads; do
  for arm in "$@"; do
    out=results/pilot/${run_id}_${workload}_${arm}.json
    .venv/bin/python rcm_exec run scripts/hetero_ctl.py run-arm --layout "$layout" \
      --head-pod "$head_pod" --output "$out" --timeout "${TIMEOUT:-3000}" -- \
      --arm "$arm" --workload "$workload" --run-id "$run_id" \
      --unfrozen-diagnostic --policy "$policy" "${epp_args[@]}" --arm-timeout "${TIMEOUT:-3000}" 2>&1 | grep -E "ARM_RESULT|RUN_FINISHED|Reason" || true
    .venv/bin/python rcm_exec resume --abort >/dev/null 2>&1 || true
  done
done
