#!/usr/bin/env bash
# Q1 diagnostic (bulk_reconfiguration_plan.md): one-token top-32 probes at fixed prefixes
# (results/bf16kv/probe-points-q1.json) on fp32-KV S, bf16-KV H and bf16-KV S engines, to tell
# whether the bf16 S/H disagreement is benign bf16 sensitivity or a layout-dependent fault.
# Normal deploys only (weight_snapshot=off), pod-local XLA cache; no snapshots are written.
#
#   flock /tmp/hetero-cluster.lock scripts/bf16_probe_diag.sh
#   scripts/bf16_probe_diag.sh --dry-run
#
# Env: HEAD_POD (required unless --dry-run), RUN_ID, OUT, INVENTORY, MODEL, POINTS.
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=0
if [[ "${1:-}" == --dry-run ]]; then DRY=1; shift; fi
(( $# == 0 )) || { echo "usage: $0 [--dry-run]" >&2; exit 2; }
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
if (( ! DRY )); then
  if flock -n /tmp/hetero-cluster.lock true; then
    echo 'Run under: flock /tmp/hetero-cluster.lock scripts/bf16_probe_diag.sh' >&2; exit 1
  fi
  : "${HEAD_POD:?Set HEAD_POD to the current Ray head pod}"
fi
HEAD_POD=${HEAD_POD:-'<head-pod>'}
COMMIT=$(git rev-parse --short HEAD)
RUN_ID=${RUN_ID:-bf16diag-$COMMIT}
OUT=${OUT:-results/bf16kv/$RUN_ID}
INVENTORY=${INVENTORY:-/tmp/hetero-inventory-only.json}
MODEL=${MODEL:-/models/gemma-4-31b}
KEEP_SNAPSHOTS=${KEEP_SNAPSHOTS:-0}
RESERVE=62000000000
RCM=(.venv/bin/python rcm_exec)
POINTS=${POINTS:-results/bf16kv/probe-points-q1.json}
# kv float32 selects the FP32-state profile; bfloat16 is full bf16 state (EngineSettings NUMERIC_PROFILES).
settings() {
  local extra=
  [[ $1 == float32 ]] && extra=',"activation_dtype":"float32","matmul_precision":"highest"'
  echo "{\"weight_snapshot\":\"off\",\"startup_profile\":true,\"kv_cache_dtype\":\"$1\"$extra}"
}
[[ -s data/manifests/W0.json ]] || { echo "missing data/manifests/W0.json" >&2; exit 1; }
(( DRY )) || mkdir -p "$OUT"
owned_run=0
started=0
current=

show() {  # print one command, single-quoting arguments with shell metacharacters
  local arg line=
  for arg in "$@"; do
    if [[ "$arg" =~ [^A-Za-z0-9_./:=,@+-] ]]; then line+="'$arg' "; else line+="$arg "; fi
  done
  echo "${line% }"
}
mark() {
  (( DRY )) || printf '{"run":"%s","event":"%s","layout":"%s","t":%s}\n' \
    "$RUN_ID" "$1" "${current:-none}" "$(date +%s.%N)" >> "$OUT/timeline.jsonl"
}
state() { (( DRY )) || printf 'bf16-provision %s %s %s\n' "$RUN_ID" "$*" "$(date -Is)" >> /tmp/hetero-cluster.state; }
idle() {
  (( DRY )) && return 0
  "${RCM[@]}" status --json > "$OUT/rcm-status.json"
  .venv/bin/python -c 'import json,sys; sys.exit(bool(json.load(open(sys.argv[1]))["alive"]))' "$OUT/rcm-status.json"
}
run() {
  if (( DRY )); then show "${RCM[@]}" run "$@"; return 0; fi
  idle || { echo 'Refusing to touch an active RCM daemon' >&2; return 1; }
  owned_run=1
  local rc=0
  mark "rcm_start:$2"
  "${RCM[@]}" run "$@" || rc=$?
  mark "rcm_end:$2:exit=$rc"
  if (( rc != 0 )); then "${RCM[@]}" resume --abort || true; owned_run=0; return "$rc"; fi
  # RUN_FINISHED can precede daemon exit by a few milliseconds; wait for our own daemon only.
  for attempt in {1..30}; do
    if idle; then owned_run=0; return 0; fi
    sleep 0.2
  done
  echo 'Own completed RCM daemon did not exit; refusing next stage' >&2
  return 1
}
ok() {
  (( DRY )) && return 0
  .venv/bin/python -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["measurements"].get("ok") is False)' "$1"
}
ctl() {  # ctl <action> <layout> <label> <timeout> [hetero_ctl args...]
  local action=$1 layout=$2 label=$3 timeout=$4; shift 4
  run scripts/hetero_ctl.py "$action" --layout "$layout" --head-pod "$HEAD_POD" \
    --output "$OUT/$label.json" --timeout "$timeout" "$@"
  ok "$OUT/$label.json"
}
validate() {
  local action=$1 layout=$2 label=$3; shift 3
  run scripts/snapshot_validate.py "$action" --layout "$layout" --output "$OUT/$label.json" "$@"
}
deploy() {  # deploy <layout> <label> <engine settings> [cli args...]
  local layout=$1 label=$2 settings=$3; shift 3
  current=$layout
  mark "deploy_start:$label"
  ctl deploy "$layout" "$label-deploy" 5400 --inventory "$INVENTORY" -- \
    --max-model-len 16384 --model "$MODEL" --engine-settings "$settings" "$@"
  mark "deploy_end:$label"
  state "deployed $layout $label"
}
teardown() {  # teardown <layout> <label>: release stays mandatory after a failed drain
  local layout=$1 label=$2 rc=0
  mark "teardown_start:$label"
  ctl drain "$layout" "$label-drain" 1800 -- --force || rc=1
  ctl gateway-down "$layout" "$label-gateway-down" 600 || rc=1
  ctl release "$layout" "$label-release" 600 || rc=1
  mark "released:$label"
  state "released $layout $label"
  [[ "$layout" == "$current" ]] && current=
  return "$rc"
}
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  (( DRY )) && exit "$rc"
  set +e
  if (( owned_run )); then "${RCM[@]}" resume --abort; fi
  if (( started )); then
    if [[ -n "$current" ]]; then
      ctl drain "$current" cleanup-drain 1800 -- --force
      ctl gateway-down "$current" cleanup-gateway-down 600
    fi
    for layout in H S; do
      ctl release "$layout" "cleanup-release-$layout" 600 || rc=1
    done
    if (( ! KEEP_SNAPSHOTS )); then validate cleanup S cleanup-tmpfs || rc=1; fi
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

idle
[[ -s "$POINTS" ]] || { echo "missing $POINTS" >&2; exit 1; }
validate preflight S preflight --reserve-bytes "$RESERVE"
started=1
for spec in S:float32 H:bfloat16 S:bfloat16; do
  layout=${spec%%:*}; kv=${spec#*:}
  deploy "$layout" "$layout-$kv" "$(settings "$kv")"
  validate probe "$layout" "probe-$layout-$kv" --probe-points "$POINTS"
  teardown "$layout" "$layout-$kv"
done
