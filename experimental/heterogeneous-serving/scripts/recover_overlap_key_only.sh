#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec 9>/tmp/hetero-cluster.lock
flock -n 9
OUT=results/overlap/OverlapHelper/key-only-s-20261002a
.venv/bin/python rcm_exec status --json > "$OUT/rcm-status-before-recovery.json"
.venv/bin/python -c 'import json,sys;sys.exit(bool(json.load(open(sys.argv[1]))["alive"]))' "$OUT/rcm-status-before-recovery.json"
.venv/bin/python rcm_exec run scripts/recover_overlap_key_only.py
.venv/bin/python rcm_exec status --json > "$OUT/rcm-status-after-recovery.json"
