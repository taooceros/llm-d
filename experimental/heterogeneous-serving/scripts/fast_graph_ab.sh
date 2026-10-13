#!/usr/bin/env bash
# Fast-graph A/B on layout S: normal loader and snapshot restore, each with
# fast_graph off and on. Greedy outputs are checked against the ShardedRestore
# reference on every arm. Run only as:
#   HEAD_POD=<head> flock /tmp/hetero-cluster.lock scripts/fast_graph_ab.sh
# This script never acquires a second lock or cancels a foreign RCM job.
set -euo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
if flock -n /tmp/hetero-cluster.lock true; then
  echo 'Run under blocking flock /tmp/hetero-cluster.lock' >&2; exit 1
fi
: "${HEAD_POD:?Set HEAD_POD to the current Ray head pod}"
OUT=${OUT:-results/restore/FastGraph}
INVENTORY=${INVENTORY:-/tmp/hetero-inventory-only.json}
REF=${REF:?set REF to a bf16-KV reference produced under the current engine settings}
[[ -f "$REF" ]] || { echo "missing reference $REF" >&2; exit 1; }
RCM=(.venv/bin/python rcm_exec)
mkdir -p "$OUT"
TIMELINE=$OUT/timeline.jsonl
owned_run=0
started=0
current=
mark() { printf '{"event":"%s","layout":"%s","t":%s}\n' "$1" "$2" "$(date +%s.%N)" >> "$TIMELINE"; }
idle() {
  "${RCM[@]}" status --json > "$OUT/rcm-status.json"
  .venv/bin/python -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(bool(s["alive"]))' "$OUT/rcm-status.json"
}
run() {
  idle || { echo 'Refusing to touch an active RCM daemon' >&2; return 1; }
  owned_run=1
  local rc=0
  mark "rcm_start:$2" "${current:-none}"
  "${RCM[@]}" run "$@" || rc=$?
  mark "rcm_end:$2:exit=$rc" "${current:-none}"
  if (( rc != 0 )); then
    "${RCM[@]}" resume --abort || true
  fi
  if (( rc == 0 )); then
    for attempt in {1..30}; do
      if idle; then owned_run=0; return 0; fi
      sleep 0.2
    done
    echo 'Own successful RCM daemon did not exit; refusing next stage' >&2
    return 1
  fi
  owned_run=0
  return "$rc"
}
ctl() {
  local action=$1 layout=$2 output=$3; shift 3
  run scripts/hetero_ctl.py "$action" --layout "$layout" --head-pod "$HEAD_POD" \
    --output "$output" --timeout 5400 "$@"
  .venv/bin/python -c 'import json,sys; m=json.load(open(sys.argv[1]))["measurements"]; sys.exit(m.get("ok") is False)' "$output"
}
validate() {
  local action=$1 layout=$2 label=$3; shift 3
  run scripts/snapshot_validate.py "$action" --layout "$layout" --output "$OUT/$label.json" "$@"
}
teardown() {
  local layout=$1 label=$2
  mark teardown_start "$layout"
  ctl drain "$layout" "$OUT/$label-drain.json" -- --force
  mark drained "$layout"
  ctl gateway-down "$layout" "$OUT/$label-gateway-down.json"
  mark gateway_down "$layout"
  ctl release "$layout" "$OUT/$label-release.json"
  mark released "$layout"
  printf 'FastGraph released %s %s\n' "$layout" "$(date -Is)" >> /tmp/hetero-cluster.state
  current=
}
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if (( owned_run )); then "${RCM[@]}" resume --abort; fi
  if (( started )); then
    if [[ -n "$current" ]]; then
      ctl drain "$current" "$OUT/cleanup-drain.json" -- --force
      ctl gateway-down "$current" "$OUT/cleanup-gateway-down.json"
      ctl release "$current" "$OUT/cleanup-release.json" || rc=1
    fi
    validate cleanup S cleanup-tmpfs || rc=1
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deploy() {
  local layout=$1 mode=$2 fast=$3 label=$4
  current=$layout
  mark "deploy_start:$label" "$layout"
  ctl deploy "$layout" "$OUT/$label-deploy.json" --inventory "$INVENTORY" -- \
    --max-model-len 16384 --model "${MODEL:-/models/gemma-4-31b}" \
    --engine-settings "{\"weight_snapshot\":\"$mode\",\"weight_snapshot_verify\":true,\"fast_graph\":$fast}"
  mark "deploy_end:$label" "$layout"
  printf 'FastGraph deployed %s mode=%s fast_graph=%s %s\n' "$layout" "$mode" "$fast" "$(date -Is)" >> /tmp/hetero-cluster.state
}
arm() {
  local mode=$1 fast=$2 label=$3; shift 3
  deploy S "$mode" "$fast" "$label"
  validate validate S "$label-validation" --reference "$REF" "$@"
  if [[ "$label" == S-cold-base ]]; then
    # Writes the snapshot the two restore arms read; the write is outside every clock.
    validate snapshot S S-snapshot
    validate memory S S-after-write-memory
  fi
  teardown S "$label"
}
idle
validate preflight S initial-memory --reserve-bytes 62000000000
started=1
arm off false S-cold-base
arm off true S-cold-fast
arm read false S-restore-base --require-restored
arm read true S-restore-fast --require-restored
.venv/bin/python scripts/fast_graph_summary.py --output "$OUT" --reference "$REF"
# EXIT trap removes /dev/shm/omp_weights snapshots on all TPU hosts.
