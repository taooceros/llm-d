#!/usr/bin/env bash
# Reconfiguration study, measurement phase: per-layout throughput on W0 and W5, and an
# instrumented full transition (teardown + deploy + first run) into every layout.
# Run: flock /tmp/hetero-cluster.lock scripts/reconfig_measure.sh [layouts...]   (default: S H M)
# Every phase boundary is appended to results/reconfig/timeline-<commit>.jsonl as
# {"event", "layout", "t"} with wall-clock epoch seconds; deploy outputs (with per-worker
# start-up phase clocks) are kept under results/reconfig/deploy-<commit>-<layout>.json.
# Routing is the blind BASE_POLICY ("plain") in every arm.
set -uo pipefail
cd "$(dirname "$0")/.."
LAYOUTS=${*:-S H M}
HEAD=${HEAD_POD:-tpu-ray-cluster-vllm-tpu-head-r5tfg}
NS=llm-d-optimized-baseline
COMMIT=$(git rev-parse --short HEAD)
RCM=".venv/bin/python rcm_exec"
POLICY='{"decode_length_blind":true}'
ORDER=${ORDER:-"W0 W5 W5 W0"}   # reps alternate order within one deployment
export TIMEOUT=7200
OUT=results/reconfig
LOG=$OUT/measure-$COMMIT.log
TIMELINE=$OUT/timeline-$COMMIT.jsonl
mkdir -p "$OUT" results/pilot
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mark() { printf '{"event":"%s","layout":"%s","t":%s}\n' "$1" "$2" "$(date +%s.%N)" >> "$TIMELINE"; }
state() { echo "reconfig-measure $COMMIT $* $(date)" >> /tmp/hetero-cluster.state; }

daemon_active() { $RCM status 2>&1 | grep -q "^Status: Active Worker Daemon"; }
ctl() {  # ctl <action> <layout> <out> [args...]: one hetero_ctl stage, fail loud
  local action=$1 layout=$2 out=$3; shift 3
  rm -f "$out"
  if daemon_active; then log "refusing $action: an RCM daemon is active"; $RCM status | tee -a "$LOG"; return 2; fi
  $RCM run scripts/hetero_ctl.py "$action" --layout "$layout" --head-pod "$HEAD" \
      --output "$out" --timeout 5400 "$@" 2>&1 | grep -E "RUN_FINISHED|TRAP_PAUSED|Reason" | tee -a "$LOG"
  if daemon_active; then $RCM resume --abort >/dev/null 2>&1; fi
  python3 -c "import json,sys; m=json.load(open('$out'))['measurements']; sys.exit(0 if m.get('ok', True) is not False else 1)" 2>/dev/null
}
teardown() {
  local layout=$1
  mark teardown_start "$layout"
  ctl drain "$layout" /tmp/rc-drain-$layout.json -- --force
  mark drained "$layout"
  ctl gateway-down "$layout" /tmp/rc-gwdown-$layout.json
  mark gateway_down "$layout"
  ctl release "$layout" /tmp/rc-release-$layout.json || true
  mark released "$layout"
}
arm_for() { case $1 in M) echo M0;; *) echo "$1";; esac; }

log "measure $COMMIT layouts=$LAYOUTS order=$ORDER policy=$POLICY"
log "slice before: $(tail -1 /tmp/hetero-cluster.state)"
kubectl -n $NS exec $HEAD -c ray-head -- mkdir -p /tmp/omp_hetero/data/data/workloads
kubectl -n $NS cp -c ray-head data/workloads/translation_ted.json \
  $HEAD:/tmp/omp_hetero/data/data/workloads/translation_ted.json
if [ ! -s data/manifests/W5.json ]; then
  $RCM run scripts/hetero_ctl.py build-manifest --head-pod "$HEAD" --output /tmp/w5-manifest.json \
      --timeout 1800 -- --workloads W5 2>&1 | grep -E "RUN_FINISHED|Reason" | tee -a "$LOG"
  cp results/hetero/artifacts/manifests/W5.json data/manifests/W5.json || { log "W5 manifest build failed"; exit 1; }
fi
for w in W0 W5; do
  log "$w manifest $(python3 -c "import json;m=json.load(open('data/manifests/$w.json'));print(m['num_requests'],m['useful_output_tokens'],m['manifest_sha256'][:12])")"
done

prev=${PREV_LAYOUT:-}
for layout in $LAYOUTS; do
  arm=$(arm_for "$layout")
  # Tear down whatever this campaign deployed last (or PREV_LAYOUT), then this layout's
  # leftovers; teardown of a layout that is not deployed is a fast no-op.
  [ -n "$prev" ] && teardown "$prev"
  [ "$prev" != "$layout" ] && teardown "$layout"
  log "deploy $layout"
  mark deploy_start "$layout"
  ctl deploy "$layout" $OUT/deploy-$COMMIT-$layout.json --inventory /tmp/hetero-inventory-only.json \
      -- --max-model-len 16384 --model "${MODEL:-/models/gemma-4-31b}" \
      || { log "deploy $layout failed"; state "deploy-failed $layout"; exit 1; }
  mark deploy_end "$layout"
  state "deployed $layout"
  rep=0
  for w in $ORDER; do
    rep=$((rep + 1))
    log "run $w $arm slot $rep"
    mark "run_start:$w:$rep" "$layout"
    WORKLOADS=$w RUN_ID=rc-$COMMIT-$layout-s$rep POLICY="$POLICY" HEAD_POD=$HEAD \
      scripts/run_pilot.sh "$layout" "$arm" | tee -a "$LOG"
    mark "run_end:$w:$rep" "$layout"
  done
  prev=$layout
done
teardown "$prev"; state "released $prev"
log "measure done"
