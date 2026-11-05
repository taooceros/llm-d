#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
OUT=results/overlap/OverlapHelper/key-only-s-20261002a
mkdir -p "$OUT"
exec 9>/tmp/hetero-cluster.lock
flock 9
RCM=(.venv/bin/python rcm_exec)
owned=0
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if ((owned)); then
    "${RCM[@]}" status --json > "$OUT/rcm-status-cleanup.json"
    if .venv/bin/python -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["alive"])' "$OUT/rcm-status-cleanup.json"; then
      "${RCM[@]}" resume --abort > "$OUT/rcm-abort.log" 2>&1
    fi
  fi
  "${RCM[@]}" status --json > "$OUT/rcm-status-after.json"
  flock -u 9
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${RCM[@]}" status --json > "$OUT/rcm-status-before.json"
.venv/bin/python -c 'import json,sys;sys.exit(bool(json.load(open(sys.argv[1]))["alive"]))' "$OUT/rcm-status-before.json"
if [[ -e "$OUT/result.json" ]]; then
  echo 'Refusing to repeat an already-recorded key-only job' >&2
  exit 1
fi
owned=1
"${RCM[@]}" run scripts/overlap_key_only.py submit --output "$OUT/result.json"
