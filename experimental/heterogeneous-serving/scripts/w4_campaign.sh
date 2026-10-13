#!/usr/bin/env bash
# W4 long-document + chat mix: heterogeneous M versus homogeneous H, one locked session.
# Run: flock /tmp/hetero-cluster.lock scripts/w4_campaign.sh <reps> [layouts...]
# S (4xTP8) cannot serve W4: every long request exceeds a TP8 engine's KV pool.
# Each layout runs two routing policies: "plain" (BASE_POLICY) and "resv"
# (work-conserving fallback plus a long-request KV reserve on large engines).
set -uo pipefail
cd "$(dirname "$0")/.."
REPS=${1:-1}; shift || true
LAYOUTS=${*:-M H}
HEAD=${HEAD_POD:-tpu-ray-cluster-vllm-tpu-head-r5tfg}
NS=llm-d-optimized-baseline
COMMIT=$(git rev-parse --short HEAD)
RCM=".venv/bin/python rcm_exec"
declare -A POLICIES=(
  [plain]='{"decode_length_blind":true}'
  [resv]='{"decode_length_blind":true,"work_conserving_fallback":true,"small_prompt_token_limit":16384,"long_reserve_tokens":36864}'
)
POLICY_ORDER=${POLICY_ORDER:-resv plain}
export TIMEOUT=7200
LOG=results/pilot/w4-campaign-$COMMIT.log
mkdir -p results/pilot
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
state() { echo "w4-campaign $COMMIT $* $(date)" >> /tmp/hetero-cluster.state; }

daemon_active() { $RCM status 2>&1 | grep -q "^Status: Active Worker Daemon"; }
ctl() {  # ctl <action> <layout> [args...]: one hetero_ctl stage, fail loud
  local action=$1 layout=$2; shift 2
  local out=/tmp/w4-$action-$layout.json
  rm -f "$out"
  if daemon_active; then log "refusing $action: an RCM daemon is active"; $RCM status | tee -a "$LOG"; return 2; fi
  $RCM run scripts/hetero_ctl.py "$action" --layout "$layout" --head-pod "$HEAD" \
      --output "$out" --timeout 5400 "$@" 2>&1 | grep -E "RUN_FINISHED|TRAP_PAUSED|Reason" | tee -a "$LOG"
  if daemon_active; then $RCM resume --abort >/dev/null 2>&1; fi
  python3 -c "import json,sys; m=json.load(open('$out'))['measurements']; sys.exit(0 if m.get('ok', True) is not False else 1)" 2>/dev/null
}
teardown() {
  local layout=$1
  ctl drain "$layout" -- --force; ctl gateway-down "$layout"; ctl release "$layout" || true
}

log "campaign $COMMIT reps=$REPS layouts=$LAYOUTS policies=$POLICY_ORDER"
log "slice before: $(tail -1 /tmp/hetero-cluster.state)"
kubectl -n $NS exec $HEAD -c ray-head -- mkdir -p /tmp/omp_hetero/data/data/workloads \
  /tmp/omp_hetero/data/results/scheduling_headroom_20260915
kubectl -n $NS cp -c ray-head data/workloads/longdoc_govreport.json \
  $HEAD:/tmp/omp_hetero/data/data/workloads/longdoc_govreport.json
kubectl -n $NS cp -c ray-head results/scheduling_headroom_20260915/manifest.json \
  $HEAD:/tmp/omp_hetero/data/results/scheduling_headroom_20260915/manifest.json
if [ ! -s data/manifests/W4.json ]; then
  $RCM run scripts/hetero_ctl.py build-manifest --head-pod "$HEAD" --output /tmp/w4-manifest.json \
      --timeout 1800 -- --workloads W4 2>&1 | grep -E "RUN_FINISHED|Reason" | tee -a "$LOG"
  cp results/hetero/artifacts/manifests/W4.json data/manifests/W4.json || { log "W4 manifest build failed"; exit 1; }
fi
log "W4 manifest $(python3 -c "import json;m=json.load(open('data/manifests/W4.json'));print(m['num_requests'],m['useful_output_tokens'],m['context_max'],m['manifest_sha256'][:12])")"

for layout in $LAYOUTS; do
  case $layout in
    M) arm=M0; deploy_args=(--max-model-len 16384 --instance-engine-settings '{"large":{"max_model_len":36864}}');;
    H) arm=H; deploy_args=(--max-model-len 36864);;
    *) log "layout $layout cannot serve W4"; exit 1;;
  esac
  if [ "${REUSE:-}" = "$layout" ]; then log "reusing deployed $layout"; else
    teardown "$layout"
    log "deploy $layout ${deploy_args[*]}"
    ctl deploy "$layout" --inventory /tmp/hetero-inventory-only.json -- "${deploy_args[@]}" \
      || { log "deploy $layout failed"; state "deploy-failed $layout"; exit 1; }
    state "deployed $layout"
  fi
  for rep in $(seq "${REP_START:-1}" "$((${REP_START:-1} + REPS - 1))"); do
    for name in $POLICY_ORDER; do
      log "run W4 $arm policy=$name rep $rep"
      WORKLOADS=W4 RUN_ID=w4-$COMMIT-$name-r$rep POLICY="${POLICIES[$name]}" HEAD_POD=$HEAD \
        scripts/run_pilot.sh "$layout" "$arm" | tee -a "$LOG"
    done
  done
  teardown "$layout"; state "released $layout"
done
log "campaign done"
