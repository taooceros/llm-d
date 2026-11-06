#!/usr/bin/env bash
# Autoscale characterization ablation, queued behind whatever holds the cluster lock:
#   flock /tmp/hetero-cluster.lock scripts/ablation_queue.sh > results/switch/ablation-queue.log 2>&1
# (1) startup_state.py ckpt-io: cold whole-checkpoint reads on large_b's 4 hosts, no engine;
# (2) switch_campaign.sh autoscale with the ablation arms (ARMS below), n = 2 each plus anchors.
set -uo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
export HEAD_POD=${HEAD_POD:?set HEAD_POD} MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
INVENTORY=${INVENTORY:-/tmp/hetero-inventory-only.json}
TAG=$(git rev-parse --short HEAD)
mkdir -p results/switch/autoscale-probe
echo "lock acquired $(date -u +%FT%T)"
.venv/bin/python rcm_exec status 2>&1 | head -n 1
.venv/bin/python rcm_exec run scripts/startup_state.py ckpt-io --inventory "$INVENTORY" \
  --gcs gs://hongtaozhang-tpu-model-cache-us-west1/gemma-4-31b --target H:large_b \
  --output "results/switch/autoscale-probe/ckpt-io-$TAG.json" --timeout 3600
probe=$?
echo "ckpt-io exit=$probe $(date -u +%FT%T)"
if (( probe != 0 )); then .venv/bin/python rcm_exec resume --abort 2>&1 | tail -n 2; fi
sleep 30
ARMS=${ARMS:-"snap snap-xla snap-bank raw-pf ckpt-fuse ckpt-gcs-jax gcs snap snap-xla snap-bank raw-pf ckpt-fuse ckpt-gcs-jax raw raw-jax"} \
  scripts/switch_campaign.sh autoscale
echo "campaign exit=$? $(date -u +%FT%T)"
