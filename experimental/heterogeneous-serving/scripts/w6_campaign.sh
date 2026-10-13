#!/usr/bin/env bash
# W6 long-context (>52.9k-token prompts) GovReport workload: homogeneous H versus heterogeneous M.
# Run: flock /tmp/hetero-cluster.lock scripts/w6_campaign.sh <reps> [layouts...]
# H (2xTP16, max_model_len 131072) runs W6 alone. M runs MIX_W6_W0 (W6 then W0): the single
# "plain" policy has large_fresh_traffic_share 0, so W0 prompts (context <= 16384) go to the
# TP8 "small*" instances and W6 prompts (> 16384, fit only large's 131072) go to the TP16 "large".
# Routing evidence: hetero/gateway/policy.py select_initial (eligible = max_model_len >= context;
# empty fits_small -> large pool) and hetero/experiment.py _instance_roles (small*/large* names).
set -uo pipefail
cd "$(dirname "$0")/.."
REPS=${1:-1}; shift || true
LAYOUTS=${*:-H M}
HEAD=${HEAD_POD:-tpu-ray-cluster-vllm-tpu-head-gxwsv}
NS=llm-d-optimized-baseline
COMMIT=$(git rev-parse --short HEAD)
RCM=".venv/bin/python rcm_exec"
declare -A POLICIES=(
# reserve_large_headroom_blocks 0: the default 64-block headroom on every large* instance (H's two
# TP16s included) would, at the 131072-context block size, refuse the longest W6 prompts.
  [plain]='{"decode_length_blind":true,"large_fresh_traffic_share":0.0,"reserve_large_headroom_blocks":0}'
)
POLICY_ORDER=${POLICY_ORDER:-plain}
export TIMEOUT=7200
SCRATCH=/dev/shm/hetero-scratch
mkdir -p "$SCRATCH" results/pilot
LOG=results/pilot/w6-campaign-$COMMIT.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
state() { echo "w6-campaign $COMMIT $* $(date)" >> /tmp/hetero-cluster.state; }

preflight() {  # refuse when local disk is too tight: / >= 4 GB, /dev/shm >= 20 GB free
  local root_kb shm_kb
  root_kb=$(df -Pk / | awk 'NR==2{print $4}'); shm_kb=$(df -Pk /dev/shm | awk 'NR==2{print $4}')
  if [ "$root_kb" -lt $((4*1024*1024)) ] || [ "$shm_kb" -lt $((20*1024*1024)) ]; then
    log "preflight refused: / free ${root_kb}KB (<4GB) or /dev/shm free ${shm_kb}KB (<20GB)"; exit 2
  fi
}
gz_big() {  # gz_big <min_bytes> <files...>: gzip existing JSON files at least min_bytes
  local min=$1; shift
  for f in "$@"; do
    [ -f "$f" ] && [ "$(stat -c %s "$f")" -ge "$min" ] && gzip -f "$f"
  done
}
compress_run() {  # compress_run <run_id>: big JSON written by this script's run/deploy stage
  gz_big 1048576 results/hetero/artifacts/"$1"/*.json results/pilot/"$1"_*.json
}
compress_startup() {  # per-worker startup_profiles JSON written by this layout's runs
  local f
  for f in results/hetero/artifacts/w6-$COMMIT-*/*.json results/pilot/w6-$COMMIT-*.json; do
    [ -f "$f" ] && grep -q '"startup_profiles"' "$f" && gzip -f "$f"
  done
}

daemon_active() { $RCM status 2>&1 | grep -q "^Status: Active Worker Daemon"; }
ctl() {  # ctl <action> <layout> [args...]: one hetero_ctl stage, fail loud
  local action=$1 layout=$2; shift 2
  local out=$SCRATCH/w6-$action-$layout.json
  rm -f "$out"
  if daemon_active; then log "refusing $action: an RCM daemon is active"; $RCM status | tee -a "$LOG"; return 2; fi
  $RCM run scripts/hetero_ctl.py "$action" --layout "$layout" --head-pod "$HEAD" \
      --output "$out" --timeout 5400 "$@" 2>&1 | grep -E "RUN_FINISHED|TRAP_PAUSED|Reason" | tee -a "$LOG"
  if daemon_active; then $RCM resume --abort >/dev/null 2>&1; fi
  python3 -c "import json,sys; m=json.load(open('$out'))['measurements']; sys.exit(0 if m.get('ok', True) is not False else 1)" 2>/dev/null
}
keep() {  # keep <action> <layout> [suffix]: gzip a stage output into results/pilot as evidence
  local src=$SCRATCH/w6-$1-$2.json
  [ -f "$src" ] && gzip -c "$src" > "results/pilot/w6-$COMMIT-$1-$2${3:-}.json.gz"
}
teardown() {
  local layout=$1 tag=${2:-}
  ctl drain "$layout" -- --force; keep drain "$layout" "$tag"
  ctl gateway-down "$layout"; keep gateway-down "$layout" "$tag"
  ctl release "$layout" || true; keep release "$layout" "$tag"
}

preflight
log "campaign $COMMIT reps=$REPS layouts=$LAYOUTS policies=$POLICY_ORDER"
log "slice before: $(tail -1 /tmp/hetero-cluster.state)"
kubectl -n $NS exec $HEAD -c ray-head -- mkdir -p /tmp/omp_hetero/data/data/workloads \
  /tmp/omp_hetero/data/results/scheduling_headroom_20260915
kubectl -n $NS cp -c ray-head data/workloads/longctx_govreport.json \
  $HEAD:/tmp/omp_hetero/data/data/workloads/longctx_govreport.json
# The MIX's W0 part is expanded from the frozen data/manifests/W0.json shipped with the source tree;
# the historical manifest is staged as w4_campaign does, for any on-head W0 rebuild.
kubectl -n $NS cp -c ray-head results/scheduling_headroom_20260915/manifest.json \
  $HEAD:/tmp/omp_hetero/data/results/scheduling_headroom_20260915/manifest.json
if [ ! -s data/manifests/W6.json ]; then
  $RCM run scripts/hetero_ctl.py build-manifest --head-pod "$HEAD" --output "$SCRATCH/w6-manifest.json" \
      --timeout 1800 -- --workloads W6 2>&1 | grep -E "RUN_FINISHED|Reason" | tee -a "$LOG"
  cp results/hetero/artifacts/manifests/W6.json data/manifests/W6.json || { log "W6 manifest build failed"; exit 1; }
fi
log "W6 manifest $(python3 -c "import json;m=json.load(open('data/manifests/W6.json'));print(m['num_requests'],m['useful_output_tokens'],m['context_max'],m['manifest_sha256'][:12])")"

for layout in $LAYOUTS; do
  preflight
  case $layout in
    M) arm=M0; workloads=MIX_W6_W0
       deploy_args=(--max-model-len 16384 --instance-engine-settings '{"large":{"max_model_len":131072}}');;
    H) arm=H; workloads=W6; deploy_args=(--max-model-len 131072);;
    *) log "layout $layout is not part of the W6 campaign (H, M only)"; exit 1;;
  esac
  if [ "${REUSE:-}" = "$layout" ]; then log "reusing deployed $layout"; else
    teardown "$layout" -pre
    log "deploy $layout ${deploy_args[*]}"
    ctl deploy "$layout" --inventory /tmp/hetero-inventory-only.json -- "${deploy_args[@]}"
    rc=$?; keep deploy "$layout"; rm -f "$SCRATCH"/w6-deploy-"$layout".json
    [ $rc -eq 0 ] || { log "deploy $layout failed"; state "deploy-failed $layout"; exit 1; }
    state "deployed $layout"
  fi
  for rep in $(seq "${REP_START:-1}" "$((${REP_START:-1} + REPS - 1))"); do
    for name in $POLICY_ORDER; do
      preflight
      log "run $workloads $arm policy=$name rep $rep"
      WORKLOADS=$workloads RUN_ID=w6-$COMMIT-$name-r$rep POLICY="${POLICIES[$name]}" HEAD_POD=$HEAD \
        scripts/run_pilot.sh "$layout" "$arm" | tee -a "$LOG"
      compress_run "w6-$COMMIT-$name-r$rep"
      out=results/pilot/w6-$COMMIT-$name-r${rep}_${workloads}_${arm}.json
      if ! python3 -c "import json,sys; r=json.load(open('$out'))['measurements']['result']; sys.exit(0 if r.get('all_requests_complete') is True else 1)" 2>/dev/null; then
        log "run $workloads $arm rep $rep failed or incomplete ($out); stopping $layout"
        teardown "$layout" -final; state "run-failed $layout"; exit 1
      fi
    done
  done
  compress_startup
  teardown "$layout" -final; state "released $layout"
done
log "campaign done"
