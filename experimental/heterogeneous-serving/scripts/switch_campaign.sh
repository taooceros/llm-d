#!/usr/bin/env bash
# Production in-process layout switch campaigns (hetero run-switch), Gemma-4 31B on the 32-chip v5e slice.
#
#   flock /tmp/hetero-cluster.lock scripts/switch_campaign.sh full      # P2: S -> H, W0 then W6
#   flock /tmp/hetero-cluster.lock scripts/switch_campaign.sh partial   # P3: S -> Mb, W0 then W0+W6 mix
#   scripts/switch_campaign.sh <full|partial> --dry-run                 # print the commands, touch nothing
#
# full    4xTP8 S serves W0 (phase 1); at T0 the in-process controller drains S, deploys H (2xTP16,
#         max_model_len 131072, snapshot restore) and phase 2 (W6) starts the moment the new gateway routes.
#         After each rep the harness returns H -> S exactly like reconfig_measure.sh / the published
#         harness (drain H --force, gateway-down, release, deploy S, check, run_pilot W0) so the
#         harness H -> S cost C can be computed from the same slice state (scripts/switch_summary.py).
# partial small_a/small_b stay; small_c/small_d are drained; large_b (TP16, max_model_len 131072) joins
#         while phase 2 (W0,W6: W0-class on the kept pair, W6-class on the joiner) runs.
#
# Every deploy restores weights with FastSnapshot + FastGraph (FAST).  PREP publishes the snapshots
# and the greedy references; KEEP_SNAPSHOTS=1 skips the EXIT tmpfs cleanup so PREP=0 can reuse them.
# Disk floors before every stage that deploys or runs: local / >= 10 GB and /dev/shm >= 20 GB free.
#
# Env: HEAD_POD (required unless --dry-run), REPS (3), PREP (1), KEEP_SNAPSHOTS (0), MAX_ATTEMPTS (2),
# OUT (results/switch/<mode>-<commit>), POLICY, INVENTORY, MODEL, SWITCH_TIMEOUT (5400 s per run-switch job),
# ARM_TIMEOUT (900 s per run-switch stream),
# REFERENCE ($OUT; directory with prep-<layout>-reference.json when PREP=0).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-}
case $MODE in full|partial) shift;; *) echo "usage: $0 <full|partial> [--dry-run]" >&2; exit 2;; esac
DRY=0
if [[ "${1:-}" == --dry-run ]]; then DRY=1; shift; fi
(( $# == 0 )) || { echo "usage: $0 <full|partial> [--dry-run]" >&2; exit 2; }
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
if (( ! DRY )); then
  if flock -n /tmp/hetero-cluster.lock true; then
    echo "Run under: flock /tmp/hetero-cluster.lock scripts/switch_campaign.sh $MODE" >&2; exit 1
  fi
  : "${HEAD_POD:?Set HEAD_POD to the current Ray head pod}"
fi
HEAD_POD=${HEAD_POD:-'<head-pod>'}
export HEAD_POD
NS=llm-d-optimized-baseline
COMMIT=$(git rev-parse --short HEAD)
RUN_BASE=$MODE-$COMMIT
OUT=${OUT:-results/switch/$RUN_BASE}
INVENTORY=${INVENTORY:-/tmp/hetero-inventory-only.json}
MODEL=${MODEL:-/models/gemma-4-31b}
REPS=${REPS:-3}
PREP=${PREP:-1}
KEEP_SNAPSHOTS=${KEEP_SNAPSHOTS:-0}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
SWITCH_TIMEOUT=${SWITCH_TIMEOUT:-5400}
# Per-stream client deadline inside run-switch: phase 2 takes ~4-10 min, so a hung joiner (E2) is cut
# off after ARM_TIMEOUT instead of holding the slice for 2 h.
ARM_TIMEOUT=${ARM_TIMEOUT:-900}
RESERVE=62000000000
REFERENCE=${REFERENCE:-$OUT}
RCM=(.venv/bin/python rcm_exec)
POLICY=${POLICY:-'{"decode_length_blind":true,"large_fresh_traffic_share":0.0,"reserve_large_headroom_blocks":0}'}
# As staged_sequence.sh: pod-local XLA cache (no GCS-fuse), FastSnapshot restore, FastGraph, startup profile.
FAST='{"weight_snapshot":"read","weight_snapshot_verify":false,"fast_graph":true,"startup_profile":true}'
NORMAL='{"weight_snapshot":"off","startup_profile":true}'
LARGE_B_LONG='{"large_b":{"max_model_len":131072}}'
for manifest in W0 W6; do
  [[ -s data/manifests/$manifest.json ]] || { echo "missing data/manifests/$manifest.json (scripts/w6_campaign.sh builds W6)" >&2; exit 1; }
done
[[ -s data/workloads/longctx_govreport.json ]] || { echo "missing data/workloads/longctx_govreport.json" >&2; exit 1; }
(( DRY )) || mkdir -p "$OUT"
started=0
owned_run=0
current=
failures=0

show() {  # print one command, single-quoting arguments with shell metacharacters
  local arg line=
  for arg in "$@"; do
    if [[ "$arg" =~ [^A-Za-z0-9_./:=,@+-] ]]; then line+="'$arg' "; else line+="$arg "; fi
  done
  echo "${line% }"
}
mark() {  # mark <event>: wall-clock epoch seconds into $OUT/timeline.jsonl
  (( DRY )) || printf '{"run":"%s","event":"%s","layout":"%s","t":%s}\n' \
    "$RUN_BASE" "$1" "${current:-none}" "$(date +%s.%N)" >> "$OUT/timeline.jsonl"
}
state() { (( DRY )) || printf 'switch-campaign %s %s %s\n' "$RUN_BASE" "$*" "$(date -Is)" >> /tmp/hetero-cluster.state; }
disk_floor() {  # disk_floor <stage>: local / >= 10 GB and /dev/shm >= 20 GB free, else stop (EXIT cleanup runs)
  if (( DRY )); then echo "# disk floor before $1: / >= 10 GB, /dev/shm >= 20 GB free"; return 0; fi
  local root_kb shm_kb
  root_kb=$(df -Pk / | awk 'NR==2{print $4}'); shm_kb=$(df -Pk /dev/shm | awk 'NR==2{print $4}')
  if (( root_kb < 10*1024*1024 || shm_kb < 20*1024*1024 )); then
    echo "disk floor refused before $1: / free ${root_kb} KB (<10 GB) or /dev/shm free ${shm_kb} KB (<20 GB)" >&2
    state "disk-floor-stop $1"
    exit 2
  fi
}
idle() {  # text status: `status --json` embeds tens of MB of job logs (see bf16_provision.sh @ a104ebb)
  (( DRY )) && return 0
  local s
  s=$("${RCM[@]}" status 2>&1 || true)
  ! grep -q "^Status: Active Worker Daemon" <<<"$s"
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
  local wait_i
  for wait_i in {1..30}; do
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
switch_complete() {  # complete, exact, accounted, segments reconciled, and every new worker restored from the snapshot
  (( DRY )) && return 0
  .venv/bin/python -c 'import json,sys; r=json.load(open(sys.argv[1]))["measurements"]["result"]; sys.exit(0 if r.get("completed") is True and (r.get("restored") or {}).get("all_restored") is True else 1)' "$1"
}
diagnose() {  # diagnose <layout> <rid>: evidence of a failed run-switch before anything is torn down
  local layout=$1 rid=$2
  mark "diagnose_start:$rid"
  ctl gateway-status "$layout" "diag-$rid-gateway-status" 300 || true
  ctl inspect "$layout" "diag-$rid-inspect" 300 || true
  if (( DRY )); then
    show kubectl -n "$NS" cp -c ray-head "$HEAD_POD:/tmp/omp_hetero/artifacts/$rid" "$OUT/diag-$rid-artifacts"
    show "${RCM[@]}" logs --tail 600 --level WARN,ERROR
  else
    kubectl -n "$NS" cp -c ray-head "$HEAD_POD:/tmp/omp_hetero/artifacts/$rid" "$OUT/diag-$rid-artifacts" || true
    "${RCM[@]}" logs --tail 600 --level WARN,ERROR > "$OUT/diag-$rid-rcm-logs.txt" 2>&1 || true
  fi
  mark "diagnose_end:$rid"
}
ctl() {  # ctl <action> <layout> <label> <timeout> [hetero_ctl args...]
  local action=$1 layout=$2 label=$3 timeout=$4; shift 4
  (( DRY )) || rm -f "$OUT/$label.json"  # never judge this run by a previous attempt's output
  run scripts/hetero_ctl.py "$action" --layout "$layout" --head-pod "$HEAD_POD" \
    --output "$OUT/$label.json" --timeout "$timeout" "$@" || return $?
  ok "$OUT/$label.json"
}
validate() {
  local action=$1 layout=$2 label=$3; shift 3
  run scripts/snapshot_validate.py "$action" --layout "$layout" --output "$OUT/$label.json" "$@"
}
deploy() {  # deploy <layout> <label> <engine settings> [--max-model-len N (default 16384)] [cli args...]
  local layout=$1 label=$2 settings=$3; shift 3
  local maxlen=16384 extra=()
  while (( $# )); do
    case $1 in --max-model-len) maxlen=$2; shift 2;; *) extra+=("$1"); shift;; esac
  done
  disk_floor "deploy $layout $label"
  current=$layout
  mark "deploy_start:$label"
  ctl deploy "$layout" "$label-deploy" 5400 --inventory "$INVENTORY" -- \
    --max-model-len "$maxlen" --model "$MODEL" --engine-settings "$settings" "${extra[@]}" || return $?
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
    # Adopted kept engines keep their original placement-group names (S--small_*), so every layout is released.
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

prep_snapshot() {  # prep_snapshot <layout> <instances|-> [deploy cli args, e.g. --max-model-len N...]
  local layout=$1 instances=${2#-}; shift 2
  disk_floor "prep $layout"
  validate memory "$layout" "prep-$layout-memory" --reserve-bytes "$RESERVE"
  deploy "$layout" "prep-$layout-normal" "$NORMAL" "$@"
  validate snapshot "$layout" "prep-$layout-snapshot" ${instances:+--instances "$instances"}
  # Greedy reference from the normal loader for the token-equality checks after each restore.
  validate validate "$layout" "prep-$layout-reference"
  teardown "$layout" "prep-$layout-normal"
}
restored() {  # restored <layout> [check-only]: every worker restored (+ greedy tokens equal the normal loader)
  validate check "$1" "$1-restore-check" --require-restored
  [[ "${2:-}" == check-only ]] || validate validate "$1" "$1-restored-validation" --require-restored --reference "$REFERENCE/prep-$1-reference.json"
}
stage_catalogue() {  # the longctx catalogue W6 prompts are built from, staged on the head as w6_campaign.sh does
  if (( DRY )); then
    show kubectl -n "$NS" exec "$HEAD_POD" -c ray-head -- mkdir -p /tmp/omp_hetero/data/data/workloads
    show kubectl -n "$NS" cp -c ray-head data/workloads/longctx_govreport.json "$HEAD_POD:/tmp/omp_hetero/data/data/workloads/longctx_govreport.json"
    return 0
  fi
  local dst=/tmp/omp_hetero/data/data/workloads/longctx_govreport.json want got try
  want=$(sha256sum data/workloads/longctx_govreport.json | cut -d' ' -f1)
  kubectl -n "$NS" exec "$HEAD_POD" -c ray-head -- mkdir -p /tmp/omp_hetero/data/data/workloads
  # Skip when the head already holds the same bytes; retry a flaky copy and verify (a cut-off
  # kubectl cp leaves a truncated file behind).
  for try in 1 2 3; do
    got=$(kubectl -n "$NS" exec "$HEAD_POD" -c ray-head -- sha256sum "$dst" 2>/dev/null | cut -d' ' -f1 || true)
    [[ "$got" == "$want" ]] && return 0
    kubectl -n "$NS" cp -c ray-head data/workloads/longctx_govreport.json "$HEAD_POD:$dst" || sleep 5
  done
  got=$(kubectl -n "$NS" exec "$HEAD_POD" -c ray-head -- sha256sum "$dst" 2>/dev/null | cut -d' ' -f1 || true)
  [[ "$got" == "$want" ]] || { echo "longctx catalogue staging failed (sha $got != $want)" >&2; return 1; }
}
record_run() {  # record_run <rep> <attempt> <run_id> <ok>
  (( DRY )) || printf '{"mode":"%s","rep":%s,"attempt":%s,"run_id":"%s","ok":%s}\n' \
    "$MODE" "$1" "$2" "$3" "$4" >> "$OUT/runs.jsonl"
}
run_id_for() { local id=$RUN_BASE-r$1; (( $2 > 1 )) && id+=-a$2; echo "$id"; }
write_campaign() {
  (( DRY )) || printf '{"mode":"%s","commit":"%s","run_base":"%s","reps":%s,"policy":%s}\n' \
    "$MODE" "$COMMIT" "$RUN_BASE" "$REPS" "$POLICY" > "$OUT/campaign.json"
}

# ---------------------------------------------------------------- full (P2): S -> H
return_to_source() {  # harness H -> S exactly like reconfig_measure.sh: drain/gateway-down/release H, deploy S, check, W0 pilot
  local r=$1 rc=0 label=ret-r$1 pilot_out
  mark "teardown_start:$label"
  ctl drain H "$label-drain" 1800 -- --force || rc=1
  mark "drained:$label"
  ctl gateway-down H "$label-gateway-down" 600 || rc=1
  mark "gateway_down:$label"
  ctl release H "$label-release" 600 || rc=1
  mark "released:$label"
  current=
  (( rc == 0 )) || return 1
  deploy S "$label" "$FAST" --max-model-len 16384 || return 1
  mark "check_start:$label"
  validate check S "$label-check" --require-restored || return 1
  mark "check_end:$label"
  disk_floor "return pilot $label"
  mark "pilot_start:$label"
  # scripts/run_pilot.sh <layout> <arm>: env WORKLOADS RUN_ID POLICY HEAD_POD; writes results/pilot/<RUN_ID>_<W>_<arm>.json
  if (( DRY )); then
    echo "WORKLOADS=W0 RUN_ID=$RUN_BASE-ret-r$r POLICY='$POLICY' HEAD_POD=$HEAD_POD TIMEOUT=7200 scripts/run_pilot.sh S S"
  else
    idle || return 1
    WORKLOADS=W0 RUN_ID=$RUN_BASE-ret-r$r POLICY=$POLICY HEAD_POD=$HEAD_POD TIMEOUT=7200 scripts/run_pilot.sh S S \
      | tee -a "$OUT/return-r$r.log" || true
    idle || return 1
  fi
  mark "pilot_end:$label"
  pilot_out=results/pilot/$RUN_BASE-ret-r${r}_W0_S.json
  if (( ! DRY )); then
    .venv/bin/python -c 'import json,sys; r=json.load(open(sys.argv[1]))["measurements"]["result"]; sys.exit(0 if r.get("all_requests_complete") is True else 1)' "$pilot_out" || return 1
  fi
}
campaign_full() {
  validate preflight S preflight --reserve-bytes "$RESERVE"
  started=1
  stage_catalogue
  if (( PREP )); then
    validate cleanup S prep-initial-cleanup
    prep_snapshot S -
    prep_snapshot H - --max-model-len 131072
    validate files S prep-snapshot-files
    deploy H H-restore "$FAST" --max-model-len 131072
    restored H
    # Same-loader static reference: W6 on the FAST-restored H (the published 206.2 s came from a
    # normal-loader H); switch_summary.py can recompute the ramp against it.
    disk_floor "static W6 on restored H"
    if (( DRY )); then
      echo "WORKLOADS=W6 RUN_ID=$RUN_BASE-static-W6-H POLICY='$POLICY' HEAD_POD=$HEAD_POD TIMEOUT=7200 scripts/run_pilot.sh H H"
    else
      idle || return 1
      WORKLOADS=W6 RUN_ID=$RUN_BASE-static-W6-H POLICY=$POLICY HEAD_POD=$HEAD_POD TIMEOUT=7200 scripts/run_pilot.sh H H \
        | tee -a "$OUT/static-W6-H.log" || true
      idle || return 1
    fi
    teardown H H-restore
  fi
  deploy S S-restore "$FAST" --max-model-len 16384
  restored S
  local r=1 attempt=1 rid
  while (( r <= REPS )); do
    rid=$(run_id_for "$r" "$attempt")
    disk_floor "run-switch $rid"
    mark "switch_start:r$r"
    if ctl run-switch H "switch-r$r" "$SWITCH_TIMEOUT" --inventory "$INVENTORY" -- \
        --from-layout S --switch-mode full --phase1 W0 --phase2 W6 --static-target-s 206.2 \
        --run-id "$rid" --policy "$POLICY" --engine-settings "$FAST" --max-model-len 131072 \
        --model "$MODEL" --arm-timeout "$ARM_TIMEOUT" && switch_complete "$OUT/switch-r$r.json"; then
      mark "switch_end:r$r"
      current=H
      record_run "$r" "$attempt" "$rid" true
      state "switched S->H $rid"
      if ! return_to_source "$r"; then
        record_run "$r" "$attempt" "$RUN_BASE-ret-r$r" false
        state "return-failed r$r"; echo "harness return H->S failed in rep $r; stopping" >&2; return 1
      fi
      r=$((r + 1)); attempt=1
    else
      mark "switch_end:r$r:failed"
      (( DRY )) || cp -f "$OUT/switch-r$r.json" "$OUT/switch-r$r-a$attempt-failed.json" 2>/dev/null || true
      record_run "$r" "$attempt" "$rid" false
      diagnose H "$rid"
      failures=$((failures + 1))
      state "switch-failed r$r attempt $attempt"
      (( failures < MAX_ATTEMPTS )) || { echo "run-switch failed $failures times; stopping" >&2; return 1; }
      # Fail closed: nothing is retried in place.  Release everything, rebuild the S start state, new attempt id.
      teardown H "switch-r$r-failed-H" || true
      for layout in Mb S; do teardown "$layout" "switch-r$r-failed-$layout" || true; done
      deploy S "S-restore-a$((attempt + 1))" "$FAST" --max-model-len 16384
      restored S check-only
      attempt=$((attempt + 1))
    fi
  done
  teardown S final
}

# ---------------------------------------------------------------- partial (P3): S -> Mb
campaign_partial() {
  validate preflight S preflight --reserve-bytes "$RESERVE"
  started=1
  stage_catalogue
  if (( PREP )); then
    validate cleanup S prep-initial-cleanup
    prep_snapshot S -
    prep_snapshot Mb large_b --instance-engine-settings "$LARGE_B_LONG"
    validate files S prep-snapshot-files
  fi
  local r=1 attempt=1 rid
  while (( r <= REPS )); do
    rid=$(run_id_for "$r" "$attempt")
    deploy S "S-restore-r$r-a$attempt" "$FAST" --max-model-len 16384
    if (( r == 1 && attempt == 1 )); then restored S; else restored S check-only; fi
    disk_floor "run-switch $rid"
    mark "switch_start:r$r"
    if ctl run-switch Mb "switch-r$r" "$SWITCH_TIMEOUT" --inventory "$INVENTORY" -- \
        --from-layout S --switch-mode partial --phase1 W0 --phase2 W0,W6 --static-target-s 409.2 \
        --run-id "$rid" --policy "$POLICY" --engine-settings "$FAST" \
        --instance-engine-settings "$LARGE_B_LONG" --max-model-len 16384 \
        --model "$MODEL" --arm-timeout "$ARM_TIMEOUT" && switch_complete "$OUT/switch-r$r.json"; then
      mark "switch_end:r$r"
      current=Mb
      record_run "$r" "$attempt" "$rid" true
      state "switched S->Mb $rid"
      validate check Mb "Mb-restore-check-r$r" --require-restored
      teardown Mb "switch-r$r-Mb"
      teardown S "switch-r$r-leftover-S" || true  # kept engines keep their S placement-group names
      r=$((r + 1)); attempt=1
    else
      mark "switch_end:r$r:failed"
      (( DRY )) || cp -f "$OUT/switch-r$r.json" "$OUT/switch-r$r-a$attempt-failed.json" 2>/dev/null || true
      record_run "$r" "$attempt" "$rid" false
      diagnose Mb "$rid"
      failures=$((failures + 1))
      state "switch-failed r$r attempt $attempt"
      for layout in Mb S; do teardown "$layout" "switch-r$r-failed-$layout" || true; done
      (( failures < MAX_ATTEMPTS )) || { echo "run-switch failed $failures times; stopping" >&2; return 1; }
      attempt=$((attempt + 1))
    fi
  done
}

idle
disk_floor preflight
write_campaign
"campaign_$MODE"
# EXIT: release any leftover placement groups; remove /dev/shm/omp_weights on every host unless KEEP_SNAPSHOTS=1.
