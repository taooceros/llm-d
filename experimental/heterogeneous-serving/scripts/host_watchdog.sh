#!/usr/bin/env bash
# Host memory + disk watchdog for risky boots (the bounded Run:ai retry of the TP8 matrix).
#
#   scripts/host_watchdog.sh <instance> <out.jsonl> <ip> [<ip>...]
#
# Every 2 s, for each TPU worker pod on the given host IPs: /proc/meminfo (MemTotal, MemAvailable) and
# `df -Pk /` of the node's root/overlay filesystem (where vLLM's local caches land).  If MemAvailable
# falls below 25% of MemTotal, or the root filesystem has less than 25% free, it stops the RUNNING
# `hetero.cli deploy` Ray job(s) (run it only under the exclusive cluster lock) and kills the instance's engine actor and placement groups (layout S, instance <instance>,
# including -tpuhold) through the head pod, then writes <out.jsonl>.breach and exits 3.  Runs until
# killed or until <out.jsonl>.stop exists.  Read-only otherwise.
set -uo pipefail
instance=$1 out=$2; shift 2
ips=("$@")
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/hetero-agent.yaml}
NS=llm-d-optimized-baseline
HEAD=${HEAD_POD:?HEAD_POD}
RAY=${RAY_URL:-http://127.0.0.1:8265}
declare -A pod
for ip in "${ips[@]}"; do
  pod[$ip]=$(kubectl -n $NS get pods -o jsonpath='{range .items[*]}{.metadata.name} {.status.podIP}{"\n"}{end}' \
    | awk -v ip="$ip" '$2 == ip && /worker/ {print $1}')
  [[ -n ${pod[$ip]} ]] || { echo "no worker pod on $ip" >&2; exit 2; }
done
sample() {  # sample <ip>: one JSON row
  local ip=$1 raw
  raw=$(kubectl -n $NS exec "${pod[$ip]}" -c ray-worker -- sh -c \
    'awk "/^MemTotal:|^MemAvailable:/{print \$2}" /proc/meminfo; df -Pk / | awk "NR==2{print \$2, \$4}"' 2>/dev/null) || return 1
  read -r -d '' total avail fs_total fs_free <<<"$(tr '\n' ' ' <<<"$raw")"
  printf '{"t":%s,"ip":"%s","mem_total_kb":%s,"mem_available_kb":%s,"root_total_kb":%s,"root_free_kb":%s}\n' \
    "$(date +%s.%N)" "$ip" "$total" "$avail" "$fs_total" "$fs_free"
}
breach() {  # breach <row>: cancel the boot now
  echo "$1" > "$out.breach"
  for id in $(curl -s -m 10 "$RAY/api/jobs/" | python3 -c 'import json,sys; print(" ".join(j["submission_id"] for j in json.load(sys.stdin) if j.get("status") in ("RUNNING","PENDING") and j.get("submission_id") and "hetero.cli deploy" in (j.get("entrypoint") or "")))'); do
    curl -s -m 10 -X POST "$RAY/api/jobs/$id/stop" >> "$out.breach"; echo " stopped $id" >> "$out.breach"
  done
  kubectl -n $NS exec "$HEAD" -c ray-head -- python3 - "$instance" >> "$out.breach" 2>&1 <<'EOF'
import sys, ray
from ray.util import placement_group_table, remove_placement_group
from ray.util.placement_group import PlacementGroup
from ray._raylet import PlacementGroupID
inst = sys.argv[1]
ray.init(address="auto", namespace="omp_hetero", log_to_driver=False)
try:
    ray.kill(ray.get_actor(f"omp-hetero-engine--S--{inst}", namespace="omp_hetero"), no_restart=True)
    print("killed engine actor", inst)
except Exception as exc:
    print("engine actor:", type(exc).__name__)
for pg_id, info in placement_group_table().items():
    if info.get("name") in (f"omp-hetero--S--{inst}", f"omp-hetero--S--{inst}-tpuhold") and info.get("state") != "REMOVED":
        remove_placement_group(PlacementGroup(PlacementGroupID.from_hex(pg_id)))
        print("removed pg", info["name"])
EOF
  exit 3
}
while [[ ! -e $out.stop ]]; do
  for ip in "${ips[@]}"; do
    row=$(sample "$ip") || continue
    echo "$row" >> "$out"
    python3 -c 'import json,sys; r=json.loads(sys.argv[1]); sys.exit(0 if r["mem_available_kb"] >= 0.25*r["mem_total_kb"] and r["root_free_kb"] >= 0.25*r["root_total_kb"] else 1)' "$row" \
      || breach "$row"
  done
  sleep 2
done
