#!/usr/bin/env bash
# Cold provisioning of S (4xTP8) and H (2xTP16) on the 32-chip v5e slice (Gemma-4 31B) under
# full bf16 state (bf16 activations: activation_dtype unset; bf16 KV; default matmul precision).
# Q1 ran it under fp32 activations + bf16 KV; R2 of docs bulk_reconfiguration_plan.md reruns it.
# Each layout is deployed cold (weight_snapshot=off), its FastSnapshot weight snapshot is
# published, a 32-prompt greedy reference is recorded, and every engine probes the fixed probe
# points (top-32 next-token tables).  The XLA cache is pod-local (compilation_cache_dir unset).
# Each layout is then restored from its snapshot (weight_snapshot=read, fast_graph=true) and must
# equal its own reference token for token (restore equality, hard gate; replicas must also be
# identical).  Cross-layout S-vs-H drift is recorded (validate --agreement-reference, and
# scripts/probe_drift.py over the probe artifacts), not gated.  Any failed stage stops the
# script and the EXIT trap tears the cluster down.
#
#   flock /tmp/hetero-cluster.lock scripts/bf16_provision.sh      # one blocking flock owns it all
#   scripts/bf16_provision.sh --dry-run                           # print the commands, touch nothing
#
# Env: HEAD_POD (required unless --dry-run), RUN_ID, OUT, INVENTORY, MODEL, POINTS (probe points),
# FP32_PROBE (fp32-state probe artifact for the drift summary), KEEP_SNAPSHOTS (0: EXIT removes
# /dev/shm/omp_weights on every host), RESUME_FROM=<stage label> + RESUME_LAYOUT=<deployed layout>
# (continue an interrupted run in the same RUN_ID/OUT; earlier stages are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=0
if [[ "${1:-}" == --dry-run ]]; then DRY=1; shift; fi
(( $# == 0 )) || { echo "usage: $0 [--dry-run]" >&2; exit 2; }
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
if (( ! DRY )); then
  if flock -n /tmp/hetero-cluster.lock true; then
    echo 'Run under: flock /tmp/hetero-cluster.lock scripts/bf16_provision.sh' >&2; exit 1
  fi
  : "${HEAD_POD:?Set HEAD_POD to the current Ray head pod}"
fi
HEAD_POD=${HEAD_POD:-'<head-pod>'}
COMMIT=$(git rev-parse --short HEAD)
RUN_ID=${RUN_ID:-bf16prov-$COMMIT}
OUT=${OUT:-results/bf16kv/$RUN_ID}
INVENTORY=${INVENTORY:-/tmp/hetero-inventory-only.json}
MODEL=${MODEL:-/models/gemma-4-31b}
KEEP_SNAPSHOTS=${KEEP_SNAPSHOTS:-0}
RESERVE=62000000000
RCM=(.venv/bin/python rcm_exec)
# Pod-local XLA cache (compilation_cache_dir unset); startup_profile records per-phase clocks.
# Full bf16 state: activation_dtype unset, KV dtype and matmul precision named explicitly.
BF16='"kv_cache_dtype":"bfloat16","matmul_precision":"default"'
NORMAL="{\"weight_snapshot\":\"off\",\"startup_profile\":true,$BF16}"
FAST="{\"weight_snapshot\":\"read\",\"weight_snapshot_verify\":false,\"fast_graph\":true,\"startup_profile\":true,$BF16}"
POINTS=${POINTS:-results/bf16kv/probe-points-q1.json}
FP32_PROBE=${FP32_PROBE:-results/bf16kv/bf16diag-d7b1d91/probe-S-float32.json}
[[ -s "$POINTS" ]] || { echo "missing $POINTS" >&2; exit 1; }
[[ -s data/manifests/W0.json ]] || { echo "missing data/manifests/W0.json" >&2; exit 1; }
(( DRY )) || mkdir -p "$OUT"
owned_run=0
started=0
current=
resumed=0
RESUME_FROM=${RESUME_FROM:-}
RESUME_LAYOUT=${RESUME_LAYOUT:-}

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
idle() {  # text status: `status --json` embeds Ray job log tails (tens of MB, truncated on a full disk)
  (( DRY )) && return 0
  "${RCM[@]}" status > "$OUT/rcm-status.txt" 2>&1 || return 1
  grep -q '^Status: Idle' "$OUT/rcm-status.txt"
}
stage() {  # stage <label> <command...>: with RESUME_FROM, skip every stage before that label
  local label=$1; shift
  if [[ -n "$RESUME_FROM" ]] && (( ! resumed )) && [[ "$label" != "$RESUME_FROM" ]]; then
    echo "resume: skipping $label"; return 0
  fi
  resumed=1
  "$@"
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
if [[ -n "$RESUME_FROM" ]]; then
  # Continue an interrupted run in the same OUT; RESUME_LAYOUT names the layout still deployed.
  started=1; current=$RESUME_LAYOUT
  mark "resume_from:$RESUME_FROM"
fi
stage preflight validate preflight S preflight --reserve-bytes "$RESERVE"
started=1
stage prep-initial-cleanup validate cleanup S prep-initial-cleanup
# S: cold normal deploy, snapshot, reference, probes
stage prep-S-memory validate memory S prep-S-memory --reserve-bytes "$RESERVE"
stage prep-S-normal deploy S prep-S-normal "$NORMAL"
stage prep-S-snapshot validate snapshot S prep-S-snapshot
stage prep-S-reference validate validate S prep-S-reference
stage probe-S validate probe S probe-S --probe-points "$POINTS"
stage teardown-S-normal teardown S prep-S-normal
# H: cold normal deploy, snapshot, reference (+ recorded H-side drift vs S), probes
stage prep-H-memory validate memory H prep-H-memory --reserve-bytes "$RESERVE"
stage prep-H-normal deploy H prep-H-normal "$NORMAL"
stage prep-H-snapshot validate snapshot H prep-H-snapshot
stage prep-H-reference validate validate H prep-H-reference --agreement-reference "$OUT/prep-S-reference.json"
stage probe-H validate probe H probe-H --probe-points "$POINTS"
stage teardown-H-normal teardown H prep-H-normal
stage prep-snapshot-files validate files S prep-snapshot-files
# S restore: equality vs S reference (+ recorded S-side drift vs H)
stage S-restore deploy S S-restore "$FAST"
stage S-restore-check validate check S S-restore-check --require-restored
stage S-restored-validation validate validate S S-restored-validation --require-restored --reference "$OUT/prep-S-reference.json" --agreement-reference "$OUT/prep-H-reference.json"
stage teardown-S-restore teardown S S-restore
# H restore: equality vs H reference
stage H-restore deploy H H-restore "$FAST"
stage H-restore-check validate check H H-restore-check --require-restored
stage H-restored-validation validate validate H H-restored-validation --require-restored --reference "$OUT/prep-H-reference.json"
stage teardown-final teardown H final
(( resumed )) || { echo "RESUME_FROM=$RESUME_FROM matches no stage" >&2; exit 2; }

if (( ! DRY )); then
.venv/bin/python - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
pools = {}
for path in sorted(glob.glob(os.path.join(out, "*-deploy.json"))):
    label = os.path.basename(path)[:-len("-deploy.json")]
    try:
        instances = json.load(open(path))["measurements"]["deployment"]["instances"]
    except (KeyError, TypeError, ValueError):
        continue
    if isinstance(instances, dict):
        instances = [dict(v, instance_id=k) if isinstance(v, dict) else v for k, v in instances.items()]
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        init = inst.get("initialization") or {}
        kv = init.get("kv_capacity")
        iid = inst.get("instance_id")
        pools.setdefault(label, {})[iid] = kv
        clocks = inst.get("stage_clocks") or init.get("stage_clocks")
        kvs = kv if isinstance(kv, dict) else {}
        print(f"{label} {iid} block_size={kvs.get('block_size')} total_blocks={kvs.get('total_blocks')} "
              f"usable_tokens={kvs.get('usable_tokens')} engine_init_s={inst.get('engine_init_s', init.get('engine_init_s'))} "
              f"stage_clocks={clocks}")
json.dump(pools, open(os.path.join(out, "kv_pools.json"), "w"), indent=2)
PY
.venv/bin/python scripts/probe_drift.py S="$OUT/probe-S.json" H="$OUT/probe-H.json" \
  --reference fp32="$FP32_PROBE" --json "$OUT/drift.json"
fi
