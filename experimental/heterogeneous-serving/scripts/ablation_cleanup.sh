#!/usr/bin/env bash
# The autoscale-906d2ef EXIT cleanup ran while the Ray dashboard port-forward was down. Every step failed,
# so this repeats it. It runs under the exclusive cluster lock:
#   flock -n /tmp/hetero-cluster.lock scripts/ablation_cleanup.sh
# Steps: drain + gateway-down + release of H (and Mb/S, as the EXIT trap does); remove the seed scope and the
# four failed arm scopes on every host; delete the uploaded startup state from GCS; tmpfs cleanup.
set -uo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
HEAD_POD=${HEAD_POD:?set HEAD_POD}
OUT=results/switch/autoscale-906d2ef
RCM=(.venv/bin/python rcm_exec)
rc=0
step() {  # step <label> <script> <args...>: one supervised run; abort the daemon on failure
  local label=$1; shift
  printf '{"run":"autoscale-906d2ef","event":"rcm_start:%s","layout":"none","t":%s}\n' "$label" "$(date +%s.%N)" >> "$OUT/timeline.jsonl"
  "${RCM[@]}" run "$@"
  local r=$?
  printf '{"run":"autoscale-906d2ef","event":"rcm_end:%s:exit=%s","layout":"none","t":%s}\n' "$label" "$r" "$(date +%s.%N)" >> "$OUT/timeline.jsonl"
  if (( r != 0 )); then "${RCM[@]}" resume --abort >/dev/null 2>&1 || true; rc=1; fi
  sleep 3  # let the daemon exit before the next run
}
ctl() { step "$1" scripts/hetero_ctl.py "$2" --layout "$3" --head-pod "$HEAD_POD" --output "$OUT/$1.json" --timeout "$4" "${@:5}"; }
"${RCM[@]}" status 2>&1 | head -n 1 | grep -q Idle || { echo "rcm daemon busy"; exit 3; }
ctl recleanup-drain drain H 1800 --force
ctl recleanup-gateway-down gateway-down H 600
for layout in H Mb S; do ctl "recleanup-release-$layout" release "$layout" 600; done
for scope in autoscale-906d2ef-seed autoscale-906d2ef-raw-pf-r11 autoscale-906d2ef-ckpt-fuse-r12 \
             autoscale-906d2ef-ckpt-gcs-jax-r13 autoscale-906d2ef-raw-r14; do
  step "recleanup-scope-$scope" scripts/snapshot_validate.py cleanup --layout H --output "$OUT/recleanup-scope-$scope.json" \
    --run-scope "$scope"
done
step recleanup-gcs-delete scripts/startup_state.py delete \
  --gcs gs://hongtaozhang-tpu-model-cache-us-west1/omp_autoscale/autoscale-906d2ef \
  --output "$OUT/recleanup-gcs-delete.json" --timeout 900
step recleanup-tmpfs scripts/snapshot_validate.py cleanup --layout S --output "$OUT/recleanup-tmpfs.json"
printf 'switch-campaign autoscale-906d2ef recleanup rc=%s %s\n' "$rc" "$(date -Is)" >> /tmp/hetero-cluster.state
echo "recleanup rc=$rc"
exit "$rc"
