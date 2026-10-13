#!/usr/bin/env bash
# Run only as: flock /tmp/hetero-cluster.lock scripts/reconfig_snapshot.sh
# This script never acquires a second lock or cancels a foreign RCM job.
set -euo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
if flock -n /tmp/hetero-cluster.lock true; then
  echo 'Run under blocking flock /tmp/hetero-cluster.lock' >&2; exit 1
fi
: "${HEAD_POD:?Set HEAD_POD to the current Ray head pod}"
OUT=${OUT:-results/restore/ShardedRestore}
INVENTORY=${INVENTORY:-/tmp/hetero-inventory-only.json}
ORDER=${ORDER:-W0}
[[ "$ORDER" == W0 ]] || { echo 'Snapshot experiment requires ORDER=W0' >&2; exit 1; }
RCM=(.venv/bin/python rcm_exec)
mkdir -p "$OUT"
TIMELINE=$OUT/timeline.jsonl
owned_run=0
started=0
current=
serial=0
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
    # The pre-run idle check plus the externally held lock establishes ownership.
    "${RCM[@]}" resume --abort || true
  fi
  if (( rc == 0 )); then
    # RUN_FINISHED can precede daemon exit by a few milliseconds. Wait only
    # for this script's successful daemon, never a foreign active owner.
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
  printf 'sharded-snapshot released %s %s\n' "$layout" "$(date -Is)" >> /tmp/hetero-cluster.state
  current=
}
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if (( owned_run )); then "${RCM[@]}" resume --abort; fi
  if (( started )); then
    if [[ -n "$current" ]]; then
      # Release remains mandatory even if drain or gateway shutdown fails.
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
  local layout=$1 mode=$2 label=$3 verify=${4:-true}
  current=$layout
  mark "deploy_start:$label" "$layout"
  ctl deploy "$layout" "$OUT/$label-deploy.json" --inventory "$INVENTORY" -- \
    --max-model-len 16384 --model "${MODEL:-/models/gemma-4-31b}" \
    --engine-settings "{\"weight_snapshot\":\"$mode\",\"weight_snapshot_verify\":$verify}"
  mark "deploy_end:$label" "$layout"
  printf 'sharded-snapshot deployed %s mode=%s %s\n' "$layout" "$mode" "$(date -Is)" >> /tmp/hetero-cluster.state
}
pilot() {
  local layout=$1 label=$2
  mark "run_start:W0:$label" "$layout"
  ctl run-arm "$layout" "$OUT/$label-pilot.json" -- --arm "$layout" --workload W0 \
    --run-id "rc-snapshot-$label-$layout-s1" --unfrozen-diagnostic \
    --policy '{"decode_length_blind":true}' --arm-timeout 7200
  mark "run_end:W0:$label" "$layout"
}
unchecked_restore() {
  local layout=$1
  local material
  material=$(.venv/bin/python -c 'import json,sys; m=json.load(open(sys.argv[1]))["measurements"]; p=[r["phases"]["snapshot_verify"] for rows in m["worker_init_phases"].values() for r in rows]; print(int(max(v["end"]-v["start"] for v in p)>=15))' "$OUT/$layout-restored-validation.json")
  if [[ "$material" == 1 ]]; then
    deploy "$layout" read "$layout-unchecked" false
    validate validate "$layout" "$layout-unchecked-validation" --require-restored --reference "$OUT/$layout-reference.json"
    teardown "$layout" "$layout-unchecked"
  fi
}
# Refuse active ownership before installing cleanup responsibility.
idle
if [[ "${RESUME_AFTER_S:-0}" == 1 ]]; then
  validate preflight S resume-memory --reserve-bytes 62000000000
  started=1
  validate files S snapshot-files-before-resume
  unchecked_restore S
  layouts=H
else
  validate preflight S initial-memory --reserve-bytes 62000000000
  started=1
  layouts="S H"
fi
for layout in $layouts; do
  validate memory "$layout" "$layout-before-write-memory" --reserve-bytes 62000000000
  deploy "$layout" off "$layout-normal"
  validate validate "$layout" "$layout-reference"
  validate snapshot "$layout" "$layout-snapshot"
  validate memory "$layout" "$layout-after-write-memory"
  teardown "$layout" "$layout-normal"
  deploy "$layout" read "$layout-restored"
  validate validate "$layout" "$layout-restored-validation" --require-restored --reference "$OUT/$layout-reference.json"
  pilot "$layout" "$layout-restored"
  teardown "$layout" "$layout-restored"
  unchecked_restore "$layout"
done
# Snapshot-backed full redeployment transitions, with W0 only and blind routing.
for layout in S H S; do
  serial=$((serial + 1))
  label="transition-$serial-$layout"
  if [[ -n "$current" ]]; then teardown "$current" "$label-previous"; fi
  deploy "$layout" read "$label"
  validate check "$layout" "$label-worker-restore" --require-restored
  pilot "$layout" "$label"
done
teardown "$current" final
.venv/bin/python scripts/snapshot_validate.py analyze --output "$OUT"
# EXIT trap removes all /dev/shm/omp_weights snapshots on TPU hosts.
