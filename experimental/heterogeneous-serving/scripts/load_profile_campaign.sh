#!/usr/bin/env bash
# Run with: flock /tmp/hetero-cluster.lock bash scripts/load_profile_campaign.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
export OMP_PROFILE_LOAD=1
HEAD=${HEAD_POD:-tpu-ray-cluster-vllm-tpu-head-r5tfg}
OUT=results/restore/LoadProfiler
mkdir -p "$OUT"
RCM=(.venv/bin/python rcm_exec)
owned=0
current=
ctl() {
  local action=$1 layout=$2 output=$3; shift 3
  local status
  status=$("${RCM[@]}" status)
  local remote_timeout=60 client_timeout=90
  if [[ "$action" == deploy ]]; then remote_timeout=1200; client_timeout=1500; fi
  printf '%s\n' "$status"
  if [[ "$status" == *"Active Worker Daemon"* ]]; then
    echo 'Refusing to interfere with an active daemon' >&2; return 1
  fi
  owned=1
  timeout --kill-after=10s "$client_timeout" "${RCM[@]}" run scripts/hetero_ctl.py "$action" --layout "$layout" \
    --head-pod "$HEAD" --output "$output" --timeout "$remote_timeout" "$@"
  # RUN_FINISHED can precede daemon process exit. Wait only for our own
  # successful stage to settle, under the lock, before checking the next run.
  for attempt in {1..30}; do
    status=$("${RCM[@]}" status)
    if [[ "$status" != *"Active Worker Daemon"* ]]; then owned=0; return 0; fi
    sleep 0.2
  done
  echo "Our completed daemon did not settle within the bounded grace period" >&2
  return 1
}
teardown() {
  local layout=$1
  ctl drain "$layout" "$OUT/drain_$layout.json" -- --force
  ctl gateway-down "$layout" "$OUT/gateway_down_$layout.json"
  ctl release "$layout" "$OUT/release_$layout.json"
  printf 'LoadProfiler released %s %s\n' "$layout" "$(date --iso-8601=seconds)" >> /tmp/hetero-cluster.state
}
cleanup() {
  local code=$?
  trap - EXIT
  if ((owned)); then "${RCM[@]}" resume --abort || true; owned=0; fi
  if [[ -n "$current" ]]; then teardown "$current" || code=1; fi
  exit "$code"
}
trap cleanup EXIT
# Check tool availability without modifying the image.
kubectl -n llm-d-optimized-baseline exec "$HEAD" -c ray-head -- \
  sh -c 'command -v py-spy || true' | tee "$OUT/py_spy_availability.txt"
for layout in S H M; do teardown "$layout"; done
PODS=$(kubectl -n llm-d-optimized-baseline get pods -o name | grep tpu-worker-group-worker)
for pod in $PODS; do
  kubectl -n llm-d-optimized-baseline exec "$pod" -c ray-worker -- python3 -c \
    'import json,pathlib,socket,os; m=dict((line.split(":")[0],line.split(":")[1].strip()) for line in pathlib.Path("/proc/meminfo").read_text().splitlines()); s=os.statvfs("/dev/shm"); print(json.dumps({"host":socket.gethostname(),"mem_available_bytes":int(m["MemAvailable"].split()[0])*1024,"tmpfs_free_bytes":s.f_bavail*s.f_frsize}))' \
    >> "$OUT/host_memory_before_deploy.jsonl"
done
for layout in S H; do
  current=$layout
  ctl deploy "$layout" "$OUT/load_profile_$layout.json" --inventory /tmp/hetero-inventory-only.json \
    -- --max-model-len 16384 --model /models/gemma-4-31b
  printf 'LoadProfiler deployed %s %s\n' "$layout" "$(date --iso-8601=seconds)" >> /tmp/hetero-cluster.state
  teardown "$layout"
  current=
done
