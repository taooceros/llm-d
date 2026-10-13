#!/usr/bin/env bash
# W3 heterogeneous-vs-homogeneous campaign, decode-length-blind, one locked session.
# Run: flock /tmp/hetero-cluster.lock scripts/w3_campaign.sh <reps> [layouts...]
# Every arm uses the same policy amendment; only layout and migration differ.
set -uo pipefail
cd "$(dirname "$0")/.."
REPS=${1:-1}; shift || true
LAYOUTS=${*:-M S H}
HEAD=tpu-ray-cluster-vllm-tpu-head-r5tfg
NS=llm-d-optimized-baseline
COMMIT=$(git rev-parse --short HEAD)
RCM=".venv/bin/python rcm_exec"
POLICY='{"decode_length_blind":true}'
export TIMEOUT=7200
LOG=results/pilot/w3-campaign-$COMMIT.log
mkdir -p results/pilot
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
state() { echo "w3-campaign $COMMIT $* $(date)" >> /tmp/hetero-cluster.state; }

daemon_active() { $RCM status 2>&1 | grep -q "^Status: Active Worker Daemon"; }
ctl() {  # ctl <action> <layout> [args...]: one hetero_ctl stage, fail loud
  local action=$1 layout=$2; shift 2
  local out=/tmp/w3-$action-$layout.json
  rm -f "$out"
  if daemon_active; then log "refusing $action: an RCM daemon is active"; $RCM status | tee -a "$LOG"; return 2; fi
  $RCM run scripts/hetero_ctl.py "$action" --layout "$layout" --head-pod "$HEAD" \
      --output "$out" --timeout 3600 "$@" 2>&1 | grep -E "RUN_FINISHED|TRAP_PAUSED|Reason" | tee -a "$LOG"
  # A trapped stage leaves a paused daemon; abort only our own stage.
  if daemon_active; then $RCM resume --abort >/dev/null 2>&1; fi
  python3 -c "import json,sys; m=json.load(open('$out'))['measurements']; sys.exit(0 if m.get('ok', True) is not False else 1)" 2>/dev/null
}
teardown() {
  local layout=$1
  ctl drain "$layout" -- --force; ctl gateway-down "$layout"; ctl release "$layout" || true
}

log "campaign $COMMIT reps=$REPS layouts=$LAYOUTS policy=$POLICY"
last=$(tail -1 /tmp/hetero-cluster.state); log "slice before: $last"
# REUSE_M=1: M is already deployed from this runtime by a previous attempt.
[ "${REUSE_M:-0}" = 1 ] || teardown M   # release removes every owned group

# Data: stage the reasoning catalogue on the head, build the W3 manifest there.
kubectl -n $NS exec $HEAD -c ray-head -- mkdir -p /tmp/omp_hetero/data/data/workloads
kubectl -n $NS cp -c ray-head data/workloads/reasoning_r1.json $HEAD:/tmp/omp_hetero/data/data/workloads/reasoning_r1.json
if [ ! -s data/manifests/W3.json ]; then
  $RCM run scripts/hetero_ctl.py build-manifest --head-pod "$HEAD" --output /tmp/w3-manifest.json \
      --timeout 1800 -- --workloads W3 2>&1 | grep -E "RUN_FINISHED|Reason" | tee -a "$LOG"
  cp results/hetero/artifacts/manifests/W3.json data/manifests/W3.json || { log "W3 manifest build failed"; exit 1; }
fi
log "W3 manifest $(python3 -c "import json;m=json.load(open('data/manifests/W3.json'));print(m['num_requests'],m['useful_output_tokens'],m['manifest_sha256'][:12])")"

for layout in $LAYOUTS; do
  case $layout in M) arms="M0 MR MK-H";; S) arms="S";; H) arms="H";; esac
  if [ "$layout" = M ] && [ "${REUSE_M:-0}" = 1 ]; then log "reusing deployed M"; else
  log "deploy $layout (max_model_len 16384)"
  ctl deploy "$layout" --inventory /tmp/hetero-inventory-only.json -- --max-model-len 16384 \
    || { log "deploy $layout failed"; state "deploy-failed $layout"; exit 1; }
  state "deployed $layout"; fi
  if [ "$layout" = M ]; then
    ctl qualify-continuation M --protocol results/hetero/continuation_qualification_protocol.json \
      || { log "continuation qualification failed; stopping before timed runs"; exit 1; }
    log "qualification: $(python3 -c "import json;q=json.load(open('/tmp/w3-qualify-continuation-M.json'))['measurements']['qualification'];print(q['passed'],q.get('errors'))")"
  fi
  for rep in $(seq 1 "$REPS"); do
    for arm in $arms; do
      log "run W3 $arm rep $rep"
      WORKLOADS=W3 RUN_ID=w3-$COMMIT-r$rep POLICY="$POLICY" scripts/run_pilot.sh "$layout" "$arm" | tee -a "$LOG"
    done
  done
  teardown "$layout"; state "released $layout"
done
log "campaign done"
