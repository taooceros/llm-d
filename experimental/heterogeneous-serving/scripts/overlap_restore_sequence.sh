#!/usr/bin/env bash
# Invoke under one blocking flock /tmp/hetero-cluster.lock. One layout per run.
set -euo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
if flock -n /tmp/hetero-cluster.lock true; then echo 'Requires external blocking flock' >&2; exit 1; fi
: "${HEAD_POD:?}" "${LAYOUT:?S or H}" "${RUN_ID:?}"
[[ "$LAYOUT" == S || "$LAYOUT" == H ]]
OUT=results/overlap/OverlapCompile/$RUN_ID
mkdir -p "$OUT"
RCM=(.venv/bin/python rcm_exec)
current= started=0 owned=0
warm=/models/jax_cache_overlap_${RUN_ID}_normal
cold=/models/jax_cache_overlap_${RUN_ID}_cold_restore
mark() { printf '{"event":"%s","layout":"%s","t":%s}\n' "$1" "$LAYOUT" "$(date +%s.%N)" >> "$OUT/timeline.jsonl"; }
idle() {
  "${RCM[@]}" status --json > "$OUT/rcm-status.json"
  .venv/bin/python -c 'import json,sys;sys.exit(bool(json.load(open(sys.argv[1]))["alive"]))' "$OUT/rcm-status.json"
}
run() {
  idle || { echo 'Foreign RCM daemon active' >&2; return 1; }
  owned=1
  local rc=0
  "${RCM[@]}" run "$@" || rc=$?
  if ((rc)); then "${RCM[@]}" resume --abort || true; fi
  for attempt in {1..30}; do if idle; then owned=0; break; fi; sleep 0.2; done
  return "$rc"
}
ctl() { local action=$1 output=$2; shift 2; run scripts/hetero_ctl.py "$action" --layout "$LAYOUT" --head-pod "$HEAD_POD" --output "$OUT/$output.json" --timeout 3000 "$@"; }
validate() { local action=$1 label=$2; shift 2; run scripts/snapshot_validate.py "$action" --layout "$LAYOUT" --output "$OUT/$label.json" "$@"; }
release() {
  ctl drain "$1-drain" -- --force
  ctl gateway-down "$1-gateway-down"
  ctl release "$1-release"
  printf 'OverlapCompile released %s run=%s %s\n' "$LAYOUT" "$RUN_ID" "$(date -Is)" >> /tmp/hetero-cluster.state
  current=
}
cleanup() {
  local rc=$?; trap - EXIT INT TERM; set +e
  if ((owned)); then "${RCM[@]}" resume --abort; fi
  if [[ -n "$current" ]]; then
    ctl drain cleanup-drain -- --force
    ctl gateway-down cleanup-gateway-down
    ctl release cleanup-release || rc=1
  fi
  if ((started)); then
    validate cleanup cleanup-tmpfs || rc=1
    for cache in "$warm" "$cold"; do
      run scripts/overlap_storage_probe.py --cleanup-cache --root "$cache" --output "$OUT/cleanup-${cache##*/}.json" || rc=1
    done
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deploy() {
  local mode=$1 cache=$2 label=$3
  local capture_setting=null
  local cache_env='{"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS":"0","JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES":"0","JAX_LOG_COMPILES":"0","JAX_EXPLAIN_CACHE_MISSES":"0"}'
  if [[ "$mode" == off && -n "${BUNDLE_ROOT:-}" ]]; then capture_setting="\"$BUNDLE_ROOT\""; fi
  current=$LAYOUT
  mark "deploy_start:$label"
  ctl deploy "$label-deploy" --inventory "${INVENTORY:-/tmp/hetero-inventory-only.json}" -- \
    --max-model-len 16384 --model /models/gemma-4-31b \
    --engine-settings "{\"weight_snapshot\":\"$mode\",\"startup_profile\":true,\"compilation_cache_dir\":\"$cache\",\"startup_compile_bundle\":$capture_setting,\"startup_device_trace\":${DEVICE_TRACE:-false},\"worker_env\":$cache_env}"
  mark "deploy_end:$label"
  .venv/bin/python scripts/analyze_overlap.py --check-deploy "$OUT/$label-deploy.json" > "$OUT/$label-controls.json"
  printf 'OverlapCompile deployed %s mode=%s run=%s %s\n' "$LAYOUT" "$mode" "$RUN_ID" "$(date -Is)" >> /tmp/hetero-cluster.state
}
idle
run scripts/overlap_source_probe.py submit --skip-aot --output "$OUT/installed-source.json" --timeout 300
validate preflight preflight --reserve-bytes 62000000000
started=1
# Populate only this experiment's separate cache via normal prior deploy.
deploy off "$warm" normal
validate validate reference
validate snapshot snapshot
release normal
deploy read "$warm" warm-restore
validate check warm-restored-check --require-restored
validate validate warm-validation --require-restored --reference "$OUT/reference.json"
mark teardown_start
release warm
mark released
deploy read "$cold" cold-restore
validate check cold-restored-check --require-restored
validate validate cold-validation --require-restored --reference "$OUT/reference.json"
release cold
