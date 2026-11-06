#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=/home/hongtao/.kube/hetero-agent.yaml
cleanup() {
  code=$?
  if [[ "$code" == 2 ]]; then .venv/bin/python rcm_exec resume --abort; fi
  printf '%s release InPlaceEngine exit=%s\n' "$(date -Is)" "$code" >> /tmp/hetero-cluster.state
}
trap cleanup EXIT
.venv/bin/python rcm_exec status
printf '%s deploy InPlaceEngine\n' "$(date -Is)" >> /tmp/hetero-cluster.state
.venv/bin/python rcm_exec run scripts/ici_engine.py "$@"
