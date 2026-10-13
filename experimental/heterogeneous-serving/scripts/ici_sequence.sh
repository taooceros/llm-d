#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=/home/hongtao/.kube/hetero-agent.yaml
mode=${1:-orders}
scope=${2:-box}
output=${3:-results/ici/ICIBench/${mode}_${scope}.json}
cleanup() {
  code=$?
  if [[ "$code" == 2 ]]; then .venv/bin/python rcm_exec resume --abort; fi
  printf '%s release ICIBench %s %s exit=%s\n' "$(date -Is)" "$scope" "$mode" "$code" >> /tmp/hetero-cluster.state
}
trap cleanup EXIT
.venv/bin/python rcm_exec status
printf '%s deploy ICIBench %s %s\n' "$(date -Is)" "$scope" "$mode" >> /tmp/hetero-cluster.state
.venv/bin/python rcm_exec run scripts/ici_bench.py --mode "$mode" --scope "$scope" --output "$output" "${@:4}"
if [[ " $* " == *" --profile "* ]]; then
  .venv/bin/python rcm_exec status
  .venv/bin/python rcm_exec run scripts/collect_ici_trace.py "$output" --output results/ici/ICIBench/profile
fi
