#!/usr/bin/env bash
# One blocking flock owns this entire matched cold/warm/overlap sequence.
set -euo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
if flock -n /tmp/hetero-cluster.lock true; then echo 'Requires external blocking flock' >&2; exit 1; fi
: "${HEAD_POD:?}" "${RUN_ID:?}" "${A:?}" "${B:?}"
[[ "$A$B" == SH || "$A$B" == HS ]]
: "${TREATMENT:?baseline or overlap}"
[[ "$TREATMENT" == baseline || "$TREATMENT" == overlap ]]
OUT=results/overlap/OverlapCompile/$RUN_ID
mkdir -p "$OUT"
RCM=(.venv/bin/python rcm_exec)
current= started=0 owned=0
bundle=/models/omp_overlap_${RUN_ID}_bundle
warm=/models/jax_cache_overlap_${RUN_ID}_normal
cold=/models/jax_cache_overlap_${RUN_ID}_cold
overlap=/models/jax_cache_overlap_${RUN_ID}_background
[[ "$B" == H ]] && topology=v5e:4x4 || topology=v5e:2x4
idle() {
  "${RCM[@]}" status --json > "$OUT/rcm-status.json"
  .venv/bin/python -c 'import json,sys;sys.exit(bool(json.load(open(sys.argv[1]))["alive"]))' "$OUT/rcm-status.json"
}
run() {
  idle || { echo 'Foreign RCM daemon active' >&2; return 1; }
  owned=1
  local rc=0
  "${RCM[@]}" run "$@" || rc=$?
  if ((rc)); then "${RCM[@]}" resume --abort || true; fi
  for attempt in {1..30}; do if idle; then owned=0; break; fi; sleep .2; done
  return "$rc"
}
mark() { printf '{"event":"%s","layout":"%s","t":%s}\n' "$1" "${current:-none}" "$(date +%s.%N)" >> "$OUT/timeline.jsonl"; }
ctl() { local action=$1 layout=$2 label=$3; shift 3; run scripts/hetero_ctl.py "$action" --layout "$layout" --head-pod "$HEAD_POD" --output "$OUT/$label.json" --timeout 3000 "$@"; }
validate() { local action=$1 layout=$2 label=$3; shift 3; run scripts/snapshot_validate.py "$action" --layout "$layout" --output "$OUT/$label.json" "$@"; }
release() {
  local label=$1
  mark "teardown_start:$label"
  ctl drain "$current" "$label-drain" -- --force
  ctl gateway-down "$current" "$label-gateway-down"
  ctl release "$current" "$label-release"
  printf 'OverlapCompile released %s run=%s %s\n' "$current" "$RUN_ID" "$(date -Is)" >> /tmp/hetero-cluster.state
  mark "released:$label"
  current=
}
cleanup() {
  local rc=$?; trap - EXIT INT TERM; set +e
  if ((owned)); then "${RCM[@]}" resume --abort; fi
  if [[ -n "$current" ]]; then
    ctl drain "$current" cleanup-drain -- --force
    ctl gateway-down "$current" cleanup-gateway-down
    ctl release "$current" cleanup-release || rc=1
  fi
  if ((started)); then
    validate cleanup "$A" cleanup-tmpfs || rc=1
    for cache in "$warm" "$cold" "$overlap"; do
      run scripts/overlap_storage_probe.py --cleanup-cache --root "$cache" --output "$OUT/cleanup-${cache##*/}.json" || rc=1
    done
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deploy() {
  local layout=$1 mode=$2 cache=$3 label=$4 capture=${5:-null}
  local cache_env='{"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS":"0","JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES":"0","JAX_LOG_COMPILES":"0","JAX_EXPLAIN_CACHE_MISSES":"0"}'
  current=$layout
  mark "deploy_start:$label"
  ctl deploy "$layout" "$label-deploy" --inventory "${INVENTORY:-/tmp/hetero-inventory-only.json}" -- \
    --max-model-len 16384 --model /models/gemma-4-31b \
    --engine-settings "{\"weight_snapshot\":\"$mode\",\"startup_profile\":true,\"compilation_cache_dir\":\"$cache\",\"startup_compile_bundle\":$capture,\"worker_env\":$cache_env}"
  mark "deploy_end:$label"
  .venv/bin/python scripts/analyze_overlap.py --check-deploy "$OUT/$label-deploy.json" > "$OUT/$label-controls.json"
  printf 'OverlapCompile deployed %s mode=%s run=%s %s\n' "$layout" "$mode" "$RUN_ID" "$(date -Is)" >> /tmp/hetero-cluster.state
}
pilot() {
  local label=$1 background=${2:-0}
  mark "pilot_start:$label"
  local extra=()
  if ((background)); then extra+=(--background-precompile --bundle-root "$bundle" --cache-dir "$overlap" --topology "$topology"); fi
  run scripts/overlap_pilot.py --layout "$current" --run-id "overlap-$RUN_ID-$label" \
    --head-pod "$HEAD_POD" --output "$OUT/$label-pilot.json" --timeout 2400 "${extra[@]}"
  mark "pilot_end:$label"
}
idle
validate preflight "$A" preflight --reserve-bytes 62000000000
started=1
# Explicitly provision target graphs/snapshots before any measured transitions.
# Their complete normal-deploy duration stays in provisioning evidence.
deploy "$B" off "$warm" target-provision "\"$bundle\""
validate validate "$B" target-reference
validate snapshot "$B" target-snapshot
release target-provision
# Identical current-layout config/traffic for all three switch treatments.
deploy "$A" off /models/jax_cache source-provision
validate snapshot "$A" source-snapshot
if [[ "$TREATMENT" == baseline ]]; then
pilot source-cold-control
release source-cold
deploy "$B" read "$cold" target-cold
validate check "$B" target-cold-check --require-restored
pilot target-cold
validate validate "$B" target-cold-validation --require-restored --reference "$OUT/target-reference.json"
release target-cold
deploy "$A" read /models/jax_cache source-warm
validate check "$A" source-warm-check --require-restored
pilot source-warm-control
release source-warm
deploy "$B" read "$warm" target-warm
validate check "$B" target-warm-check --require-restored
pilot target-warm
validate validate "$B" target-warm-validation --require-restored --reference "$OUT/target-reference.json"
release target-warm
else
pilot source-undisturbed
pilot source-overlap 1
release source-overlap
deploy "$B" read "$overlap" target-overlap
validate check "$B" target-overlap-check --require-restored
pilot target-overlap
validate validate "$B" target-overlap-validation --require-restored --reference "$OUT/target-reference.json"
release target-overlap
fi
.venv/bin/python - "$OUT" "$RUN_ID" "$A" "$B" "$TREATMENT" <<'PY'
import json,sys,pathlib
out,run,a,b,treatment=sys.argv[1:]
labels=(["source-cold-control","target-cold","source-warm-control","target-warm"] if treatment=="baseline"
        else ["source-undisturbed","source-overlap","target-overlap"])
deployments=({"source-cold-control":"source-provision", "target-cold":"target-cold",
              "source-warm-control":"source-warm", "target-warm":"target-warm"} if treatment=="baseline"
             else {"source-undisturbed":"source-provision", "source-overlap":"source-provision",
                   "target-overlap":"target-overlap"})
campaign={'source_layout':a,'target_layout':b,'run_id':run,'pilots':{label:f'overlap-{run}-{label}' for label in labels},
          "deployments":{label:f"{deploy}-deploy.json" for label,deploy in deployments.items()},
          'transitions':[{'source':f'source-{t}-control' if t!='overlap' else 'source-overlap',
                          "target":f"target-{t}","treatment":t,"from":a,"to":b}
                         for t in (["cold","warm"] if treatment=="baseline" else ["overlap"])]}
pathlib.Path(out,'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n')
PY
