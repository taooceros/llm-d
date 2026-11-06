#!/usr/bin/env bash
# Checkpoint read probe (startup_state.py ckpt-io) on one H instance's 4 hosts, sharing the cluster:
# holds /tmp/hetero-cluster.lock shared plus /tmp/hetero-leases/<ip>.lock of each host (as
# switch_campaign.sh AS_PART), with its own RCM dir.  No engine, no placement group.
#   PART=large_a VARIANTS=st-lazy,vpf scripts/ckpt_io_partition.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PART=${PART:-large_a}
VARIANTS=${VARIANTS:?set VARIANTS}
INVENTORY=${INVENTORY:-/tmp/hetero-inventory-only.json}
TAG=${TAG:-$(git rev-parse --short HEAD)}
OUT=results/switch/autoscale-probe/ckpt-io-$TAG-${PART}.json
LOCK=/tmp/hetero-cluster.lock LEASE_DIR=/tmp/hetero-leases
export OMP_RCM_DIR=/tmp/omp_rcm-ckpt-io-$TAG
hosts=$(.venv/bin/python - "$INVENTORY" "$PART" <<'EOF'
import json, sys
from hetero.layout import builtin_layout, resolve_layout
from hetero.slice_inventory import SliceInventory
data = json.load(open(sys.argv[1]))
inventory = SliceInventory.from_dict(data.get("measurements", {}).get("inventory", data))
instance = next(i for i in resolve_layout(inventory, builtin_layout("H")).instances if i.instance_id == sys.argv[2])
print(",".join(instance.host_ips))
EOF
) || exit 1
mapfile -t ips < <(tr , '\n' <<<"$hosts" | sort -V)
mkdir -p "$LEASE_DIR" "$(dirname "$OUT")"
ready=$(mktemp -u /tmp/hetero-lease.XXXXXX)
chain=(flock -o -s -w 21600 "$LOCK")
for ip in "${ips[@]}"; do chain+=(flock -o -w 21600 "$LEASE_DIR/$ip.lock"); done
"${chain[@]}" sh -c ': > "$1"; while kill -0 "$2" 2>/dev/null; do sleep 5; done' hetero-lease "$ready" "$$" &
holder=$!
until [[ -e $ready ]]; do kill -0 "$holder" 2>/dev/null || { echo "lease not acquired" >&2; exit 3; }; sleep 1; done
rm -f "$ready"
info=$(printf '{"run":"ckpt-io-%s","instance":"H:%s","hosts":"%s","worktree":"%s","pid":%s,"rcm_dir":"%s","since":"%s"}' \
  "$TAG" "$PART" "$hosts" "$PWD" "$$" "$OMP_RCM_DIR" "$(date -Is)")
for ip in "${ips[@]}"; do printf '%s\n' "$info" > "$LEASE_DIR/$ip.json"; done
printf 'ckpt-io-%s partition-lease H:%s %s %s\n' "$TAG" "$PART" "$hosts" "$(date -Is)" >> /tmp/hetero-cluster.state
cleanup() {
  for ip in "${ips[@]}"; do rm -f "$LEASE_DIR/$ip.json"; done
  pkill -f -- "hetero-lease $ready" 2>/dev/null || true
  printf 'ckpt-io-%s partition-lease-released H:%s %s\n' "$TAG" "$PART" "$(date -Is)" >> /tmp/hetero-cluster.state
}
trap cleanup EXIT
echo "lease H:$PART ($hosts) $(date -u +%T)"
.venv/bin/python rcm_exec run scripts/startup_state.py ckpt-io --inventory "$INVENTORY" \
  --gcs gs://hongtaozhang-tpu-model-cache-us-west1/gemma-4-31b --target "H:$PART" --variants "$VARIANTS" \
  --output "$OUT" --timeout 5400
rc=$?
(( rc == 0 )) || .venv/bin/python rcm_exec resume --abort 2>&1 | tail -n 1
echo "ckpt-io exit=$rc $(date -u +%T) -> $OUT"
exit "$rc"
