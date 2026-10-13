#!/usr/bin/env bash
# Partial M->S reconfiguration: small_a/small_b keep serving phase 2 while `large` is released
# and small_c/small_d start, then join.  One run per invocation, from a fresh M deploy.
# Run: flock /tmp/hetero-cluster.lock scripts/reconfig_partial.sh <workload1|-> <workload2> [tag]
#   workload1 "-" skips phase 1 (smoke).  Routing: the W4 reserve policy, as in reconfig_mix.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
W1=${1:?workload1 or -}; W2=${2:?workload2}; TAG=${3:-run}
HEAD=${HEAD_POD:-tpu-ray-cluster-vllm-tpu-head-r5tfg}
COMMIT=$(git rev-parse --short HEAD)
RCM=".venv/bin/python rcm_exec"
POLICY='{"decode_length_blind":true,"work_conserving_fallback":true,"small_prompt_token_limit":16384,"long_reserve_tokens":36864}'
OUT=results/reconfig
LOG=$OUT/partial-$COMMIT.log
EVENTS=$OUT/partial-$COMMIT.jsonl
RUN_ID=partial-$COMMIT-$TAG
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mark() { printf '{"run":"%s","event":"%s","t":%s}\n' "$RUN_ID" "$1" "$(date +%s.%N)" >> "$EVENTS"; }
state() { echo "reconfig-partial $COMMIT $* $(date)" >> /tmp/hetero-cluster.state; }
daemon_active() { $RCM status 2>&1 | grep -q "^Status: Active Worker Daemon"; }
ctl() {
  local action=$1 layout=$2 out=$3; shift 3
  rm -f "$out"
  if daemon_active; then log "refusing $action: an RCM daemon is active"; return 2; fi
  $RCM run scripts/hetero_ctl.py "$action" --layout "$layout" --head-pod "$HEAD" \
      --output "$out" --timeout 14400 "$@" 2>&1 | grep -E "RUN_FINISHED|TRAP_PAUSED|Reason|TRANSITION|ARM_RESULT" | tee -a "$LOG"
  if daemon_active; then $RCM resume --abort >/dev/null 2>&1; fi
  python3 -c "import json,sys; m=json.load(open('$out'))['measurements']; sys.exit(0 if m.get('ok', True) is not False else 1)" 2>/dev/null
}
teardown() {
  ctl drain "$1" /tmp/pr-drain.json -- --force; ctl gateway-down "$1" /tmp/pr-gwdown.json
  ctl release "$1" /tmp/pr-release.json || true
}

log "partial $RUN_ID: phase1=$W1 phase2=$W2"
for L in S H M; do teardown "$L" >/dev/null; done
mark deploy_start
ctl deploy M $OUT/partialdeploy-$RUN_ID-M.json --inventory /tmp/hetero-inventory-only.json -- \
  --max-model-len 16384 --instance-engine-settings '{"large":{"max_model_len":36864}}' \
  || { log "deploy M failed"; state "deploy-failed M"; exit 1; }
mark deploy_end
state "deployed M"
phase1=(--arm1 M0 --workload1 "$W1"); [ "$W1" = "-" ] && phase1=(--arm1 M0)
mark transition_start
ctl run-transition S $OUT/transition-$RUN_ID.json --inventory /tmp/hetero-inventory-only.json -- \
  --from-layout M --to-layout S "${phase1[@]}" --arm2 S --workload2 "$W2" \
  --policy "$POLICY" --run-id "$RUN_ID" --max-model-len 16384 --arm-timeout 7200
rc=$?
mark transition_end
log "run-transition exit $rc"
teardown S; teardown M; state "released after $RUN_ID"
exit $rc
