#!/usr/bin/env bash
# Run under: flock /tmp/hetero-cluster.lock scripts/fast_snapshot_sequence.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
if flock -n /tmp/hetero-cluster.lock true; then
  echo 'Requires externally held /tmp/hetero-cluster.lock' >&2; exit 1
fi
: "${HEAD_POD:?Set HEAD_POD to the current Ray head pod}"
OUT=${OUT:-results/restore/FastSnapshot}
INVENTORY=${INVENTORY:-/tmp/hetero-inventory-only.json}
REFERENCE=${REFERENCE:-/home/hongtao/hetero-wt/ShardedRestore/results/restore/ShardedRestore}
RCM=(.venv/bin/python rcm_exec)
mkdir -p "$OUT"
owned_run=0
started=0
current=
idle() {
  "${RCM[@]}" status --json > "$OUT/rcm-status.json"
  .venv/bin/python -c 'import json,sys; sys.exit(bool(json.load(open(sys.argv[1]))["alive"]))' "$OUT/rcm-status.json"
}
run() {
  idle || { echo 'Refusing active foreign RCM daemon' >&2; return 1; }
  owned_run=1
  local rc=0
  "${RCM[@]}" run "$@" || rc=$?
  if (( rc != 0 )); then "${RCM[@]}" resume --abort || true; fi
  if (( rc == 0 )); then
    for attempt in {1..30}; do
      if idle; then owned_run=0; return 0; fi
      sleep 0.2
    done
    echo 'Own completed RCM daemon did not exit' >&2
    return 1
  fi
  owned_run=0
  return "$rc"
}
ctl() {
  local action=$1 layout=$2 label=$3; shift 3
  run scripts/hetero_ctl.py "$action" --layout "$layout" --head-pod "$HEAD_POD" \
    --output "$OUT/$label.json" --timeout 5400 "$@"
  .venv/bin/python -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["measurements"].get("ok") is False)' "$OUT/$label.json"
}
validate() {
  local action=$1 layout=$2 label=$3; shift 3
  run scripts/snapshot_validate.py "$action" --layout "$layout" --output "$OUT/$label.json" "$@"
}
release() {
  local label=$1
  ctl drain "$current" "$label-drain" -- --force
  ctl gateway-down "$current" "$label-gateway-down"
  ctl release "$current" "$label-release"
  printf 'fast-snapshot released %s %s\n' "$current" "$(date -Is)" >> /tmp/hetero-cluster.state
  current=
}
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if (( owned_run )); then "${RCM[@]}" resume --abort; fi
  if (( started )); then
    if [[ -n "$current" ]]; then
      ctl drain "$current" cleanup-drain -- --force
      ctl gateway-down "$current" cleanup-gateway-down
      ctl release "$current" cleanup-release || rc=1
    fi
    validate cleanup S cleanup-tmpfs || rc=1
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
idle
validate preflight S preflight --reserve-bytes 62000000000
started=1
validate cleanup S initial-cleanup
if [[ ${BENCH_ONLY:-0} == 1 ]]; then
  run scripts/snapshot_bandwidth.py --inventory "$INVENTORY" \
    --catalogue "${CATALOGUE:-/home/hongtao/hetero-wt/ICIBench/results/ici/ICIBench/parameter_catalogue.json}" \
    --topology auto --reps 1 --copy-reps 1 --pool-sizes 8 \
    --d2h-mechanisms serial_asarray,async_all_asarray,device_get_tree,chip_threads_asarray,chip_threads_did_copy,async_did_copy,pinned_host,serial_contiguous,async_contiguous,chip_threads_contiguous,pool8_contiguous \
    --h2d-mechanisms put_all_ready_end,tree_put_per_chip,chip_threads_put \
    --budget-s 2700 --timeout 3300 --output "${BANDWIDTH_OUTPUT:-$OUT/bandwidth.json}"
  .venv/bin/python -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["measurements"]["ok"])' "${BANDWIDTH_OUTPUT:-$OUT/bandwidth.json}"
  exit 0
fi
if [[ ${HOST_COPY_PRELUDE:-1} == 1 ]]; then
  run scripts/snapshot_bandwidth.py --inventory "$INVENTORY" \
    --catalogue "${CATALOGUE:-/home/hongtao/hetero-wt/ICIBench/results/ici/ICIBench/parameter_catalogue.json}" \
    --topology auto --phases inspect,host_copy --reps 1 --copy-reps 1 \
    --budget-s 700 --timeout 900 --output "$OUT/bandwidth-host-copy.json"
  .venv/bin/python -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["measurements"]["ok"])' "$OUT/bandwidth-host-copy.json"
fi
for layout in ${LAYOUTS:-S}; do
  current=$layout
  ctl deploy "$layout" "$layout-normal-deploy" --inventory "$INVENTORY" -- \
    --max-model-len 16384 --model "${MODEL:-/models/gemma-4-31b}" \
    --engine-settings '{"weight_snapshot":"off"}'
  printf 'fast-snapshot deployed %s mode=off %s\n' "$layout" "$(date -Is)" >> /tmp/hetero-cluster.state
  validate snapshot "$layout" "$layout-cold"
  validate file-fingerprint "$layout" "$layout-cold-file-fingerprint"
  validate poison "$layout" "$layout-poison"
  validate snapshot "$layout" "$layout-reused"
  validate files "$layout" "$layout-files"
  validate fingerprint "$layout" "$layout-source-fingerprint" --reference "$OUT/$layout-cold-file-fingerprint.json"
  release "$layout-normal"
  current=$layout
  ctl deploy "$layout" "$layout-restored-deploy" --inventory "$INVENTORY" -- \
    --max-model-len 16384 --model "${MODEL:-/models/gemma-4-31b}" \
    --engine-settings '{"weight_snapshot":"read"}'
  printf 'fast-snapshot deployed %s mode=read %s\n' "$layout" "$(date -Is)" >> /tmp/hetero-cluster.state
  validate fingerprint "$layout" "$layout-restored-fingerprint" --reference "$OUT/$layout-source-fingerprint.json"
  validate validate "$layout" "$layout-restored-validation" --require-restored --reference "$REFERENCE/$layout-reference.json"
  release "$layout-restored"
done
.venv/bin/python scripts/summarize_fast_snapshot.py --directory "$OUT"
# EXIT cleanup verifies every host-local /dev/shm/omp_weights directory was removed.
