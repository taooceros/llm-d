#!/usr/bin/env bash
# End-to-end multi-phase bulk job: one static layout vs a full-redeploy reconfiguration.
# Run: flock /tmp/hetero-cluster.lock scripts/reconfig_mix.sh <arm>...
#   arm "static:<L>:<COMPOSITE>"          one continuous backlog of COMPOSITE on layout L
#   arm "reconf:<L1>:<W1>:<L2>:<W2>"      W1 on L1, full teardown + deploy of L2, W2 on L2
# Every arm starts from a fresh deploy of its first layout and ends released, so arms
# are independent; the measured job wall time runs from the first dispatch to the last
# finished request (scripts/summarize_reconfig.py).  Boundaries go to
# results/reconfig/mix-<commit>.jsonl.  Routing: the W4 "reserve" policy in every arm
# (on S it reduces to the blind policy: no large instances, no second pool).
set -uo pipefail
cd "$(dirname "$0")/.."
HEAD=${HEAD_POD:-tpu-ray-cluster-vllm-tpu-head-r5tfg}
COMMIT=$(git rev-parse --short HEAD)
RCM=".venv/bin/python rcm_exec"
POLICY='{"decode_length_blind":true,"work_conserving_fallback":true,"small_prompt_token_limit":16384,"long_reserve_tokens":36864}'
MODEL=${MODEL:-/models/gemma-4-31b}
export TIMEOUT=14400
OUT=results/reconfig
LOG=$OUT/mix-$COMMIT.log
EVENTS=$OUT/mix-$COMMIT.jsonl
mkdir -p "$OUT" results/pilot
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mark() { printf '{"arm":"%s","event":"%s","layout":"%s","t":%s}\n' "$1" "$2" "$3" "$(date +%s.%N)" >> "$EVENTS"; }
state() { echo "reconfig-mix $COMMIT $* $(date)" >> /tmp/hetero-cluster.state; }
daemon_active() { $RCM status 2>&1 | grep -q "^Status: Active Worker Daemon"; }
ctl() {
  local action=$1 layout=$2 out=$3; shift 3
  rm -f "$out"
  if daemon_active; then log "refusing $action: an RCM daemon is active"; return 2; fi
  $RCM run scripts/hetero_ctl.py "$action" --layout "$layout" --head-pod "$HEAD" \
      --output "$out" --timeout 5400 "$@" 2>&1 | grep -E "RUN_FINISHED|TRAP_PAUSED|Reason" | tee -a "$LOG"
  if daemon_active; then $RCM resume --abort >/dev/null 2>&1; fi
  python3 -c "import json,sys; m=json.load(open('$out'))['measurements']; sys.exit(0 if m.get('ok', True) is not False else 1)" 2>/dev/null
}
arm_for() { case $1 in M) echo M0;; *) echo "$1";; esac; }
deploy_args() {  # TP16 engines get the long context W4 needs; TP8 engines stay at 16384
  DEPLOY_ARGS=(--model "$MODEL")
  case $1 in
    M) DEPLOY_ARGS+=(--max-model-len 16384 --instance-engine-settings '{"large":{"max_model_len":36864}}');;
    H) DEPLOY_ARGS+=(--max-model-len 36864);;
    S) DEPLOY_ARGS+=(--max-model-len 16384);;
  esac
}
teardown() {
  local tag=$1 layout=$2
  mark "$tag" teardown_start "$layout"
  ctl drain "$layout" /tmp/mix-drain.json -- --force
  ctl gateway-down "$layout" /tmp/mix-gwdown.json
  ctl release "$layout" /tmp/mix-release.json || true
  mark "$tag" released "$layout"
}
deploy() {
  local tag=$1 layout=$2
  mark "$tag" deploy_start "$layout"
  deploy_args "$layout"
  ctl deploy "$layout" "$OUT/mixdeploy-$tag-$layout.json" --inventory /tmp/hetero-inventory-only.json -- "${DEPLOY_ARGS[@]}" \
    || { log "deploy $layout failed"; state "deploy-failed $layout"; exit 1; }
  mark "$tag" deploy_end "$layout"
  state "deployed $layout"
}
run() {
  local tag=$1 layout=$2 workload=$3
  mark "$tag" "run_start:$workload" "$layout"
  WORKLOADS=$workload RUN_ID=mix-$tag POLICY="$POLICY" HEAD_POD=$HEAD \
    scripts/run_pilot.sh "$layout" "$(arm_for "$layout")" | tee -a "$LOG"
  mark "$tag" "run_end:$workload" "$layout"
}

for spec in "$@"; do
  IFS=: read -r kind a b c d <<< "$spec"
  tag=$COMMIT-$(echo "$spec" | tr ':' '_')-$(date +%H%M%S)
  log "arm $spec ($tag)"
  for L in S H M; do teardown "$tag" "$L" >/dev/null; done
  case $kind in
    static) deploy "$tag" "$a"; run "$tag" "$a" "$b"; teardown "$tag" "$a";;
    reconf) deploy "$tag" "$a"; run "$tag" "$a" "$b"
            teardown "$tag" "$a"; deploy "$tag" "$c"; run "$tag" "$c" "$d"; teardown "$tag" "$c";;
    *) log "unknown arm $spec"; exit 1;;
  esac
  state "released after $spec"
done
log "mix done"
