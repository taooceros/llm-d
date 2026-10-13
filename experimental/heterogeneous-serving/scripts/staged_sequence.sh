#!/usr/bin/env bash
# E2: staged S -> Mb -> H reconfiguration of the 32-chip v5e slice (Gemma-4 31B) that never
# stops serving, where every engine that starts restores its weights with FastSnapshot
# (weight_snapshot=read, weight_snapshot_verify=false) and builds its graph with FastGraph
# (fast_graph=true).
#
#   flock /tmp/hetero-cluster.lock scripts/staged_sequence.sh      # one blocking flock owns it all
#   scripts/staged_sequence.sh --dry-run                           # print the commands, touch nothing
#
#   prep      normal (weight_snapshot=off) deploys that publish the snapshots below and record a
#             32-prompt greedy reference (snapshot_validate.py validate), then teardown
#   deploy S  4xTP8, snapshot restore     (check --require-restored + greedy token equality)
#   S -> Mb   run-transition: phase 1 W0 on S; small_c/small_d released, large_b joins
#             while phase 2 W5 runs on the kept small_a/small_b (check Mb --require-restored)
#   Mb -> H   run-transition: small_a/small_b released, large_a joins while phase 2 W5 runs
#             on the kept large_b, finishing on H (check H --require-restored)
#   teardown  drain / gateway-down / release (H, then any Mb/S leftovers); EXIT removes tmpfs
#
# Snapshot prerequisites.  A restore reads /dev/shm/omp_weights/<layout>/<instance>/rank<r>
# (hetero/tpu_worker.py:_snapshot_location), keyed by the layout the engine is DEPLOYED in, and
# refuses (falling back to the normal loader) unless model, TP, dtype, hosts and chips match.
# Kept engines never restart, so only the engines that start need a snapshot:
#   S/small_a..small_d  S deploy        deploy S off;  snapshot_validate.py snapshot --layout S
#   Mb/large_b          S->Mb joiner    deploy Mb off; snapshot_validate.py snapshot --layout Mb --instances large_b
#   H/large_a           Mb->H joiner    deploy H off;  snapshot_validate.py snapshot --layout H --instances large_a
# hetero_ctl.py has no snapshot command: snapshots come from scripts/snapshot_validate.py
# snapshot (HeteroEngine.snapshot_weights on a ready normal deploy, as in reconfig_snapshot.sh
# and fast_snapshot_sequence.sh) or from deploying with weight_snapshot=write.  An S or H
# snapshot of large_b cannot serve Mb: same chips, different <layout> directory.
# Per host the prerequisites hold 62 GB (one TP8 shard) + 31 GB (one TP16 shard), the same as
# reconfig_snapshot.sh's concurrent S+H snapshots.  PREP=0 reuses snapshots already published
# (KEEP_SNAPSHOTS=1 on the previous run skips the EXIT tmpfs cleanup).
#
# Env: HEAD_POD (required unless --dry-run), RUN_ID, OUT, INVENTORY, MODEL, POLICY_NAME
# (resv = W4 reserve policy as in reconfig_partial.sh, blind = decode-length-blind only),
# PREP (1), KEEP_SNAPSHOTS (0), TRANSITION_TIMEOUT (14400 s per run-transition job),
# REFERENCE ($OUT; directory with prep-<layout>-reference.json when PREP=0).
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=0
if [[ "${1:-}" == --dry-run ]]; then DRY=1; shift; fi
(( $# == 0 )) || { echo "usage: $0 [--dry-run]" >&2; exit 2; }
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
if (( ! DRY )); then
  if flock -n /tmp/hetero-cluster.lock true; then
    echo 'Run under: flock /tmp/hetero-cluster.lock scripts/staged_sequence.sh' >&2; exit 1
  fi
  : "${HEAD_POD:?Set HEAD_POD to the current Ray head pod}"
fi
HEAD_POD=${HEAD_POD:-'<head-pod>'}
COMMIT=$(git rev-parse --short HEAD)
RUN_ID=${RUN_ID:-staged-$COMMIT}
OUT=${OUT:-results/staged/StagedFastRestore/$RUN_ID}
INVENTORY=${INVENTORY:-/tmp/hetero-inventory-only.json}
MODEL=${MODEL:-/models/gemma-4-31b}
PREP=${PREP:-1}
KEEP_SNAPSHOTS=${KEEP_SNAPSHOTS:-0}
RESERVE=62000000000
REFERENCE=${REFERENCE:-$OUT}  # directory holding prep-<layout>-reference.json (PREP=0 reruns)
RCM=(.venv/bin/python rcm_exec)
# Persistent compile cache: vLLM's pod-local on-disk XLA cache (/root/.cache/vllm/xla_cache, the
# default when compilation_cache_dir is unset), as in the FastSnapshot/FastGraph campaigns; the prep
# normal deploys warm it on the same hosts the timed engines use.  Not /models/jax_cache: on that
# GCS-fuse mount, staged-60ba81c's large_b joiner logged Errno 116/5 cache-entry writes and then
# hung in execute_model ~100 s into phase 2 until its RPC timed out 31 min later (gateway 500/503).
# startup_profile records per-phase worker clocks.
FAST='{"weight_snapshot":"read","weight_snapshot_verify":false,"fast_graph":true,"startup_profile":true}'
NORMAL='{"weight_snapshot":"off","startup_profile":true}'
case ${POLICY_NAME:-resv} in
  resv) POLICY='{"decode_length_blind":true,"work_conserving_fallback":true,"small_prompt_token_limit":16384,"long_reserve_tokens":36864}';;
  blind) POLICY='{"decode_length_blind":true}';;
  *) echo "POLICY_NAME must be resv or blind" >&2; exit 2;;
esac
# W4/W5 long-context TP16 engines; the TP8 engines (and kept-engine matching) stay at 16384.
LONG_MB='{"large_b":{"max_model_len":36864}}'
LONG_H='{"large_a":{"max_model_len":36864},"large_b":{"max_model_len":36864}}'
for manifest in W0 W5; do
  [[ -s data/manifests/$manifest.json ]] || { echo "missing data/manifests/$manifest.json" >&2; exit 1; }
done
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
state() { (( DRY )) || printf 'staged-fast-restore %s %s %s\n' "$RUN_ID" "$*" "$(date -Is)" >> /tmp/hetero-cluster.state; }
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
    # Engines adopted across layouts keep their original placement-group names (S--small_*,
    # Mb--large_b), so every staged layout is released, not only the current one.
    for layout in H Mb S; do
      ctl release "$layout" "cleanup-release-$layout" 600 || rc=1
    done
    if (( ! KEEP_SNAPSHOTS )); then validate cleanup S cleanup-tmpfs || rc=1; fi
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

prep_snapshot() {  # prep_snapshot <layout> <instances|-> [deploy cli args...]
  local layout=$1 instances=${2#-}; shift 2
  validate memory "$layout" "prep-$layout-memory" --reserve-bytes "$RESERVE"
  deploy "$layout" "prep-$layout-normal" "$NORMAL" "$@"
  validate snapshot "$layout" "prep-$layout-snapshot" ${instances:+--instances "$instances"}
  # Greedy reference from the normal loader (same per-instance max_model_len as the restored
  # engines) for the token-equality checks after each restore.
  validate validate "$layout" "prep-$layout-reference"
  teardown "$layout" "prep-$layout-normal"
}
restored() {  # restored <layout>: every worker restored, greedy tokens equal the normal loader
  validate check "$1" "$1-restore-check" --require-restored
  validate validate "$1" "$1-restored-validation" --require-restored --reference "$REFERENCE/prep-$1-reference.json"
}
transition() {  # transition <from> <to> <arm1> <workload1|-> <arm2> <workload2> <instance settings>
  local from=$1 to=$2 arm1=$3 w1=$4 arm2=$5 w2=$6 instance_settings=$7
  local phase1=(--arm1 "$arm1")
  [[ "$w1" == - ]] || phase1+=(--workload1 "$w1")
  mark "transition_start:$from-$to"
  ctl run-transition "$to" "transition-$from-$to" "${TRANSITION_TIMEOUT:-14400}" --inventory "$INVENTORY" -- \
    --from-layout "$from" --to-layout "$to" "${phase1[@]}" --arm2 "$arm2" --workload2 "$w2" \
    --policy "$POLICY" --run-id "$RUN_ID-$from-$to" --max-model-len 16384 --model "$MODEL" \
    --engine-settings "$FAST" --instance-engine-settings "$instance_settings" --arm-timeout 7200
  current=$to
  mark "transition_end:$from-$to"
  state "transitioned $from->$to"
}

idle
validate preflight S preflight --reserve-bytes "$RESERVE"
started=1
if (( PREP )); then
  validate cleanup S prep-initial-cleanup
  prep_snapshot H large_a --instance-engine-settings "$LONG_H"
  prep_snapshot Mb large_b --instance-engine-settings "$LONG_MB"
  prep_snapshot S -
  validate files S prep-snapshot-files
fi
deploy S S-restore "$FAST"
restored S
transition S Mb S W0 Mb0 W5 "$LONG_MB"
restored Mb
transition Mb H Mb0 - H W5 "$LONG_H"
restored H
teardown H final
for layout in Mb S; do teardown "$layout" "final-leftover-$layout" || true; done
# EXIT: release any leftover placement groups and remove /dev/shm/omp_weights on every host.
