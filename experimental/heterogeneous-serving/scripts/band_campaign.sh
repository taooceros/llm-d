#!/usr/bin/env bash
# Band campaign: one workload (default W7, mid-length 16k-48k GovReport prompts) on homogeneous layouts
# S (4xTP8) and H (2xTP16), at per-layout max_model_len (S 51200, H 131072; env MML_S, MML_H).
# Run: flock /tmp/hetero-cluster.lock scripts/band_campaign.sh [--dry-run] <reps> <layout...>
# Env: WORKLOAD (default W7), TAG (default lowercase WORKLOAD; run id <TAG>-<commit>-plain-r<N>),
#      MML_S, MML_H, REUSE, REP_START, HEAD_POD, POLICY_ORDER.
# --dry-run prints the cluster commands (rcm_exec, kubectl, run_pilot.sh) instead of running them.
# arm = layout name. The "plain" policy has large_fresh_traffic_share 0 and no large headroom reserve
# (the default 64-block reserve would refuse the longest prompts at 512-token blocks on H).
set -uo pipefail
cd "$(dirname "$0")/.."
DRY=0; ARGS=()
for a in "$@"; do if [ "$a" = "--dry-run" ]; then DRY=1; else ARGS+=("$a"); fi; done
set -- "${ARGS[@]}"
REPS=${1:-1}; shift || true
LAYOUTS=${*:-S H}
WORKLOAD=${WORKLOAD:-W7}
TAG=${TAG:-$(echo "$WORKLOAD" | tr '[:upper:]' '[:lower:]')}
MML_S=${MML_S:-51200}; MML_H=${MML_H:-131072}
HEAD=${HEAD_POD:-tpu-ray-cluster-vllm-tpu-head-gxwsv}
NS=llm-d-optimized-baseline
COMMIT=$(git rev-parse --short HEAD)
RCM=".venv/bin/python rcm_exec"
if [ $DRY -eq 1 ]; then
  RCM="echo DRY-RUN rcm_exec"
  kubectl() { echo "DRY-RUN kubectl $*"; }
fi
declare -A POLICIES=(
# reserve_large_headroom_blocks 0: the default 64-block headroom on every large* instance (H's two
# TP16s included) would, at the 131072-context block size, refuse the longest W6 prompts.
  [plain]='{"decode_length_blind":true,"large_fresh_traffic_share":0.0,"reserve_large_headroom_blocks":0}'
)
POLICY_ORDER=${POLICY_ORDER:-plain}
export TIMEOUT=7200
SCRATCH=/dev/shm/hetero-scratch
mkdir -p "$SCRATCH" results/pilot
LOG=results/pilot/$TAG-campaign-$COMMIT.log
if [ $DRY -eq 1 ]; then
  LOG=/dev/null; SCRATCH=$(mktemp -d); trap 'rm -rf "$SCRATCH"' EXIT
  log() { echo "[dry-run] $*"; }
  state() { echo "[dry-run] state $TAG $COMMIT $*"; }
else
  log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
  state() { echo "$TAG-campaign $COMMIT $* $(date)" >> /tmp/hetero-cluster.state; }
fi

preflight() {  # refuse when local disk is too tight: / >= 10 GB, /dev/shm >= 20 GB free
  local root_kb shm_kb
  root_kb=$(df -Pk / | awk 'NR==2{print $4}'); shm_kb=$(df -Pk /dev/shm | awk 'NR==2{print $4}')
  if [ "$root_kb" -lt $((10*1024*1024)) ] || [ "$shm_kb" -lt $((20*1024*1024)) ]; then
    log "preflight refused: / free ${root_kb}KB (<10GB) or /dev/shm free ${shm_kb}KB (<20GB)"; exit 2
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
  for f in results/hetero/artifacts/$TAG-$COMMIT-*/*.json results/pilot/$TAG-$COMMIT-*.json; do
    [ -f "$f" ] && grep -q '"startup_profiles"' "$f" && gzip -f "$f"
  done
}

daemon_active() { [ $DRY -eq 1 ] && return 1; $RCM status 2>&1 | grep -q "^Status: Active Worker Daemon"; }
ctl() {  # ctl <action> <layout> [args...]: one hetero_ctl stage, fail loud
  local action=$1 layout=$2; shift 2
  local out=$SCRATCH/$TAG-$action-$layout.json
  rm -f "$out"
  if [ $DRY -eq 1 ]; then
    echo "DRY-RUN rcm_exec run scripts/hetero_ctl.py $action --layout $layout --head-pod $HEAD --output $out --timeout 5400 $*"
    return 0
  fi
  if daemon_active; then log "refusing $action: an RCM daemon is active"; $RCM status | tee -a "$LOG"; return 2; fi
  $RCM run scripts/hetero_ctl.py "$action" --layout "$layout" --head-pod "$HEAD" \
      --output "$out" --timeout 5400 "$@" 2>&1 | grep -E "RUN_FINISHED|TRAP_PAUSED|Reason" | tee -a "$LOG"
  if daemon_active; then $RCM resume --abort >/dev/null 2>&1; fi
  [ $DRY -eq 1 ] && return 0
  python3 -c "import json,sys; m=json.load(open('$out'))['measurements']; sys.exit(0 if m.get('ok', True) is not False else 1)" 2>/dev/null
}
keep() {  # keep <action> <layout> [suffix]: gzip a stage output into results/pilot as evidence
  local src=$SCRATCH/$TAG-$1-$2.json
  [ -f "$src" ] && gzip -c "$src" > "results/pilot/$TAG-$COMMIT-$1-$2${3:-}.json.gz"
}
teardown() {
  local layout=$1 tag=${2:-}
  ctl drain "$layout" -- --force; keep drain "$layout" "$tag"
  ctl gateway-down "$layout"; keep gateway-down "$layout" "$tag"
  ctl release "$layout" || true; keep release "$layout" "$tag"
}

preflight
log "campaign $COMMIT reps=$REPS layouts=$LAYOUTS policies=$POLICY_ORDER"
[ $DRY -eq 1 ] || log "slice before: $(tail -1 /tmp/hetero-cluster.state)"
kubectl -n $NS exec $HEAD -c ray-head -- mkdir -p /tmp/omp_hetero/data/data/workloads \
  /tmp/omp_hetero/data/results/scheduling_headroom_20260915
for cat in longctx midctx; do  # both band catalogues are staged; the manifest builder reads the one it needs
  # Skip when the head already holds the same bytes; retry and verify (a cut-off kubectl cp leaves a
  # truncated file behind, as in P2 attempt 1).
  src=data/workloads/${cat}_govreport.json dst=/tmp/omp_hetero/data/data/workloads/${cat}_govreport.json
  want=$(sha256sum "$src" | cut -d' ' -f1)
  for try in 1 2 3; do
    got=$(kubectl -n $NS exec $HEAD -c ray-head -- sha256sum "$dst" 2>/dev/null | cut -d' ' -f1)
    [ "$got" = "$want" ] && break
    kubectl -n $NS cp -c ray-head "$src" "$HEAD:$dst" || sleep 5
  done
  [ $DRY -eq 1 ] || [ "$(kubectl -n $NS exec $HEAD -c ray-head -- sha256sum "$dst" 2>/dev/null | cut -d' ' -f1)" = "$want" ] \
    || { log "staging $cat failed (sha mismatch)"; exit 1; }
done
# W0 (and any on-head rebuild) reads the historical manifest shipped as in w4/w6_campaign.
kubectl -n $NS cp -c ray-head results/scheduling_headroom_20260915/manifest.json \
  $HEAD:/tmp/omp_hetero/data/results/scheduling_headroom_20260915/manifest.json
if [ ! -s data/manifests/$WORKLOAD.json ]; then
  $RCM run scripts/hetero_ctl.py build-manifest --head-pod "$HEAD" --output "$SCRATCH/$TAG-manifest.json" \
      --timeout 1800 -- --workloads $WORKLOAD 2>&1 | grep -E "RUN_FINISHED|Reason" | tee -a "$LOG"
  if [ $DRY -eq 0 ]; then
    cp results/hetero/artifacts/manifests/$WORKLOAD.json data/manifests/$WORKLOAD.json || { log "$WORKLOAD manifest build failed"; exit 1; }
  fi
fi
[ $DRY -eq 1 ] || log "$WORKLOAD manifest $(python3 -c "import json;m=json.load(open('data/manifests/$WORKLOAD.json'));print(m['num_requests'],m['useful_output_tokens'],m['context_max'],m['manifest_sha256'][:12])")"

for layout in $LAYOUTS; do
  preflight
  case $layout in
    S) arm=S; workloads=$WORKLOAD; deploy_args=(--max-model-len "$MML_S");;
    H) arm=H; workloads=$WORKLOAD; deploy_args=(--max-model-len "$MML_H");;
    *) log "layout $layout is not part of the band campaign (S, H only)"; exit 1;;
  esac
  if [ "${REUSE:-}" = "$layout" ]; then log "reusing deployed $layout"; else
    teardown "$layout" -pre
    log "deploy $layout ${deploy_args[*]}"
    ctl deploy "$layout" --inventory /tmp/hetero-inventory-only.json -- "${deploy_args[@]}"
    rc=$?; keep deploy "$layout"; rm -f "$SCRATCH"/$TAG-deploy-"$layout".json
    [ $rc -eq 0 ] || { log "deploy $layout failed"; state "deploy-failed $layout"; exit 1; }
    state "deployed $layout"
  fi
  for rep in $(seq "${REP_START:-1}" "$((${REP_START:-1} + REPS - 1))"); do
    for name in $POLICY_ORDER; do
      preflight
      log "run $workloads $arm policy=$name rep $rep"
      [ $DRY -eq 1 ] && echo "DRY-RUN WORKLOADS=$workloads RUN_ID=$TAG-$COMMIT-$name-r$rep POLICY='${POLICIES[$name]}' HEAD_POD=$HEAD scripts/run_pilot.sh $layout $arm" && continue
      WORKLOADS=$workloads RUN_ID=$TAG-$COMMIT-$name-r$rep POLICY="${POLICIES[$name]}" HEAD_POD=$HEAD \
        scripts/run_pilot.sh "$layout" "$arm" | tee -a "$LOG"
      [ $DRY -eq 1 ] && continue
      compress_run "$TAG-$COMMIT-$name-r$rep"
      out=results/pilot/$TAG-$COMMIT-$name-r${rep}_${workloads}_${arm}.json
      if ! python3 -c "import json,sys; r=json.load(open('$out'))['measurements']['result']; sys.exit(0 if r.get('all_requests_complete') is True else 1)" 2>/dev/null; then
        log "run $workloads $arm rep $rep failed or incomplete ($out); stopping $layout"
        teardown "$layout" -final; state "run-failed $layout"; exit 1
      fi
    done
  done
  [ $DRY -eq 1 ] || compress_startup
  teardown "$layout" -final; state "released $layout"
done
log "campaign done"
