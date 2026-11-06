#!/usr/bin/env bash
# Runs ON a TPU worker pod (inside a Ray task pinned to one host). Phases: setup | serve | fusebench | cleanup.
# Everything lives under $BASE (overlay disk) except the checkpoint copy and the TPUFlex state under /dev/shm.
# Ray mode (TTL_RAY_HOSTS=ip0,ip1,... in host-rank order, TTL_HOST_RANK=r): one engine across those hosts, launched
# like vLLM-JAX: a single `vllm serve` driver on host rank 0 with TPU_MULTIHOST_BACKEND=ray and the Ray executor,
# whose workers are Ray actors pinned to the hosts (torchtpu_ray_pin_plugin.py). Every host runs this script for
# the same phase; for serve, rank 0 runs the driver and the other ranks prepare their host (wipes, per-host fetch,
# gate relay) and wait. Hosts synchronise over tiny HTTP marker servers (LOG/marks, port TTL_MARK_PORT).
set -uo pipefail
PHASE=${1:?phase}
BASE=${TTL_BASE:-/root/torchtpu_load}
VENV=$BASE/venv
UVTOOL=$BASE/uvtool
CKPT=${TTL_CKPT:-/dev/shm/torchtpu_load_gemma4_31b}
CKPT_TMPFS=${TTL_CKPT_TMPFS:-/dev/shm/torchtpu_load_gemma4_31b}  # the tmpfs copy, whatever CKPT serves from
SRC=${TTL_SRC:-/models/gemma-4-31b}
RUN=${TTL_RUN:?run id}
LOG=$BASE/logs/$RUN
mkdir -p "$LOG"
VCACHE=${TTL_VLLM_CACHE:-$BASE/vllm_cache}  # VLLM_CACHE_ROOT (TPUFlex arms: the executable bank in /dev/shm)
PMROOT=/dev/shm/pm_torchtpu                   # TPUFlex state: bank/, snap/, t3/
ts() { while IFS= read -r l; do printf '%s %s\n' "$(date -u +%H:%M:%S.%3N)" "$l"; done; }
say() { printf '%s [host] %s\n' "$(date -u +%H:%M:%S.%3N)" "$*" | tee -a "$LOG/host.log"; }

RAY_MODE=0; HRANK=0; NH=1; MYIP=127.0.0.1; HOST0=127.0.0.1; PEERS=()
if [ -n "${TTL_RAY_HOSTS:-}" ]; then
  RAY_MODE=1
  IFS=, read -r -a HOSTS <<< "$TTL_RAY_HOSTS"
  NH=${#HOSTS[@]}
  HRANK=${TTL_HOST_RANK:?TTL_HOST_RANK with TTL_RAY_HOSTS}
  MYIP=${HOSTS[$HRANK]}
  HOST0=${HOSTS[0]}
  PEERS=("${HOSTS[@]:1}")
  # One 2x(2*NH) v5e slice of NH hosts (2x2 chips each), described to libtpu the way a dedicated v5litepod-(4*NH)
  # node pool would be; the pod's own TPU_* environment describes the whole shared 4x8 slice.
  TPU_ENV="TPU_SKIP_MDS_QUERY=true TPU_ACCELERATOR_TYPE=${TTL_ACCEL:-v5litepod-$((4 * NH))} TPU_TOPOLOGY=${TTL_TOPO:-2x$((2 * NH))} TPU_TOPOLOGY_WRAP=false,false,false TPU_HOST_BOUNDS=1,$NH,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_WORKER_HOSTNAMES=$TTL_RAY_HOSTS TORCH_TPU_TOPOLOGY=2,$((2 * NH)),1 TPU_MULTIHOST_BACKEND=ray"
  TPU_ENV_HOST="TPU_WORKER_ID=$HRANK VLLM_HOST_IP=$MYIP"
else
  TPU_ENV="TPU_SKIP_MDS_QUERY=true TPU_ACCELERATOR_TYPE=${TTL_ACCEL:-v5litepod-4} TPU_TOPOLOGY=${TTL_TOPO:-2x2} TPU_TOPOLOGY_WRAP=false,false,false TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_MULTIHOST_BACKEND=mp"
  TPU_ENV_HOST="TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost VLLM_HOST_IP=127.0.0.1"
fi
PMGCS=${TTL_PM_GCS:-}
[ "$RAY_MODE" = 1 ] && [ -n "$PMGCS" ] && PMGCS="$PMGCS/host$HRANK"  # Ray mode: per-host state (this host's ranks)
# pm-peer: TTL_PM_PEER=http://src0:P,http://src1:P (source host ranks); host rank r fetches from source rank r
if [ -n "${TTL_PM_PEER:-}" ]; then
  IFS=, read -r -a _PEERSRC <<< "$TTL_PM_PEER"
  PMGCS=${_PEERSRC[$HRANK]:?TTL_PM_PEER has no entry for host rank $HRANK}
fi
MPORT=${TTL_MARK_PORT:-18470}
# A mark's content is the run id, so a marker server left over from an earlier run (killed without its EXIT trap,
# e.g. t8z2 -> t8z3: its server kept port 18470 and served stale t0/gate marks) can never satisfy a wait.
mark() { echo "$RUN" > "$LOG/marks/$1"; }
has_mark() { [ "$(curl -sf -m 2 "http://$1:$MPORT/$2" 2>/dev/null)" = "$RUN" ]; }
await_mark() {  # await_mark HOST NAME TIMEOUT_S: poll HOST's marker server until this run's NAME exists
  local end=$(( $(date +%s) + $3 ))
  until has_mark "$1" "$2"; do
    [ "$(date +%s)" -ge "$end" ] && { say "timeout waiting for mark $2 from $1"; return 1; }
    sleep 0.02
  done
}
kv_json() {  # kv_json K=V ... -> compact JSON object
  python3.12 -c 'import json, sys; print(json.dumps(dict(a.split("=", 1) for a in sys.argv[1:]), separators=(",", ":")))' "$@"
}
collect_ray_logs() {  # collect_ray_logs JOBID: this host's worker stdout/stderr files of that Ray job, gzipped
  local jid=$1 n=0 f
  [ -n "$jid" ] || { say "no Ray job id: worker logs not collected"; return 0; }
  mkdir -p "$LOG/ray_logs"
  for f in /tmp/ray/session_latest/logs/worker-*-"$jid"-*.out /tmp/ray/session_latest/logs/worker-*-"$jid"-*.err; do
    [ -s "$f" ] || continue
    gzip -c "$f" > "$LOG/ray_logs/$(basename "$f").gz"; n=$((n + 1))
  done
  say "collected $n worker log files of Ray job $jid"
}
MEMW=""
mem_watchdog() {  # mem_watchdog CMD...: run CMD once if MemAvailable or the overlay's free space drops below a floor
  # Shared cluster: a node under memory or disk pressure gets its pod evicted, and KubeRay recreates every TPU pod.
  local floor_kb=$(( ${TTL_MEM_FLOOR_GB:-12} * 1024 * 1024 )) dfloor_kb=$(( ${TTL_DISK_FLOOR_GB:-15} * 1024 * 1024 ))
  local avail disk min=999999999999 dmin=999999999999
  while sleep 1; do
    avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    disk=$(df --output=avail -k / | awk 'NR == 2 {print $1}')
    if [ "$avail" -lt "$min" ]; then min=$avail; echo "$min" > "$LOG/mem_min_available_kb"; fi
    if [ "$disk" -lt "$dmin" ]; then dmin=$disk; echo "$dmin" > "$LOG/disk_min_available_kb"; fi
    if [ "$avail" -lt "$floor_kb" ] || [ "$disk" -lt "$dfloor_kb" ]; then
      say "WATCHDOG: MemAvailable $((avail / 1024)) MiB (floor ${TTL_MEM_FLOOR_GB:-12} GiB), overlay free $((disk / 1024)) MiB (floor ${TTL_DISK_FLOOR_GB:-15} GiB); stopping this host's engine processes"
      "$@"; return
    fi
  done
}
kill_my_ray_workers() { local p; p=$(my_ray_pids); [ -n "$p" ] && kill -KILL $p 2>/dev/null; return 0; }
MARKSRV=""
my_ray_pids() {  # this probe's Ray worker actors: raylet children whose runtime env carries TTL_RAY_WORKER=1
  python3.12 -c '
import os
for p in os.listdir("/proc"):
    try:
        if p.isdigit() and b"\0TTL_RAY_WORKER=1\0" in b"\0" + open(f"/proc/{p}/environ", "rb").read():
            print(p)
    except OSError:
        pass'
}
wait_ray_workers() {  # Ray mode: this host's worker actors exit with the driver (Ray kills a dead driver's actors)
  [ "$RAY_MODE" = 1 ] || return 0
  local end=$(( $(date +%s) + ${TTL_WORKER_EXIT_S:-60} )) pids
  while pids=$(my_ray_pids) && [ -n "$pids" ]; do
    if [ "$(date +%s)" -ge "$end" ]; then
      say "Ray workers still alive ${TTL_WORKER_EXIT_S:-60} s after the driver: $(echo $pids) -> SIGKILL"
      kill -KILL $pids 2>/dev/null; sleep 2; break
    fi
    sleep 0.5
  done
  say "no Ray worker processes of this probe left on this host"
}
trap '[ -n "$MARKSRV" ] && kill "$MARKSRV" 2>/dev/null; [ -n "$MEMW" ] && kill "$MEMW" 2>/dev/null' EXIT

run_isolated() {  # run "$@" in an env stripped of the image's JAX/vLLM/Ray/TPU-multihost state
  # shellcheck disable=SC2086
  env -i PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin" HOME=/root LANG=C.UTF-8 \
    HF_HOME="$BASE/hf" VLLM_CACHE_ROOT="$VCACHE" XDG_CACHE_HOME="$BASE/xdg" \
    $TPU_ENV $TPU_ENV_HOST ${TTL_EXTRA_ENV:-} "$@"
}
# Chip enumeration (local index -> slice device_id/coords; torchtpu_chipmap.py), probed while the chips are free and
# cached on the pod at setup (Ray mode: every host of the pair at once, in the pair's 2x4 torch_tpu world, because a
# host's chip orientation depends on the slice it is in). The canonical chip binding (TTL_CHIP_BIND=canonical,
# torchtpu_ray_pin_plugin.py) reads it and refuses a map probed for another host list / host rank.
CHIPMAP=$BASE/chipmap.json
chipmap_probe() {
  local py=/usr/local/bin/python3.12 rc args=()
  [ -x "$VENV/bin/python" ] && py=$VENV/bin/python
  [ "$RAY_MODE" = 1 ] && args=(--hosts "$TTL_RAY_HOSTS" --host-rank "$HRANK")
  # single-host rounds are opt-in: their localhost slice builder floods the shared cluster's log stream (chipmap.py)
  [ "${TTL_CHIPMAP_LOCAL:-0}" = 1 ] && args+=(--local-rounds)
  mkdir -p "$BASE"
  say "chipmap probe ($py ${args[*]})"
  env -i PATH="$(dirname "$py"):/usr/local/bin:/usr/bin:/bin" HOME=/root LANG=C.UTF-8 \
    "$py" "$(dirname "$0")/torchtpu_chipmap.py" "$CHIPMAP.tmp" "${args[@]}" 2>&1 | ts | tee -a "$LOG/host.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" = 0 ]; then mv "$CHIPMAP.tmp" "$CHIPMAP"; cp "$CHIPMAP" "$LOG/chipmap.json"; else say "chipmap probe failed rc=$rc"; fi
  return "$rc"
}
chipmap_valid() {  # the cache exists and was probed for this host list and host rank
  [ -s "$CHIPMAP" ] && python3.12 -c 'import json, sys
c = json.load(open(sys.argv[1])).get("context") or {}
sys.exit(0 if c.get("hosts") == sys.argv[2].split(",") and c.get("host_rank") == int(sys.argv[3]) else 1)' \
    "$CHIPMAP" "$TTL_RAY_HOSTS" "$HRANK"
}

case $PHASE in
setup)
  say "setup start run=$RUN host=$(hostname) ray_mode=$RAY_MODE host_rank=$HRANK ip=$MYIP"
  df -h / /dev/shm | tee -a "$LOG/host.log"
  ls -la /dev/shm "$BASE" 2>&1 | tee -a "$LOG/host.log"
  if [ ! -x "$UVTOOL/bin/uv" ]; then
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/root python3.12 -m venv "$UVTOOL" 2>&1 | ts
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/root "$UVTOOL/bin/pip" install -q uv 2>&1 | ts
  fi
  UV=$UVTOOL/bin/uv
  VENV_FROM_TAR=0
  if [ -n "${TTL_VENV_TARBALL:-}" ] && [ ! -x "$VENV/bin/python" ]; then  # the published setup venv (venvupload), same path
    say "venv from $TTL_VENV_TARBALL"
    t0=$SECONDS
    python3.12 - "$TTL_VENV_TARBALL" <<'PY' | tar -C / -xzf -
import json, shutil, sys, urllib.parse, urllib.request
b, _, n = sys.argv[1].removeprefix('gs://').partition('/')
md = 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token'
tok = json.loads(urllib.request.urlopen(urllib.request.Request(md, headers={'Metadata-Flavor': 'Google'}), timeout=10).read())
url = f'https://storage.googleapis.com/storage/v1/b/{b}/o/{urllib.parse.quote(n, safe="")}?alt=media'
req = urllib.request.Request(url, headers={'Authorization': f'Bearer {tok["access_token"]}'})
shutil.copyfileobj(urllib.request.urlopen(req, timeout=60), sys.stdout.buffer, 16 << 20)
PY
    ps=("${PIPESTATUS[@]}")
    say "venv extracted rc=${ps[0]}/${ps[1]} seconds=$((SECONDS - t0))"
    [ "${ps[0]}" = 0 ] && [ "${ps[1]}" = 0 ] && [ -x "$VENV/bin/python" ] || exit 1
    VENV_FROM_TAR=1
  fi
  if [ "$VENV_FROM_TAR" != 1 ]; then
  if [ ! -x "$VENV/bin/python" ]; then
    say "creating venv (no system site packages)"
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/root "$UV" venv --python /usr/local/bin/python3.12 --no-python-downloads "$VENV" 2>&1 | ts
  fi
  RUNAI_PKG=()
  [ "${TTL_RUNAI:-0}" = 1 ] && RUNAI_PKG=("runai-model-streamer[gcs]>=0.15.7,<0.17")  # vllm[runai] floor, vllm-torchtpu[benchmarking] cap
  # Ray mode: the venv's ray must be the cluster's (the driver joins the running cluster and the raylets start the
  # worker actors with this venv's interpreter); vllm-torchtpu 0.29.0 requires ray[default] unpinned (2.59.0 resolved).
  [ -n "${TTL_RAY_VERSION:-}" ] && RUNAI_PKG+=("ray[default]==$TTL_RAY_VERSION")
  say "uv pip install vllm-torchtpu==0.29.0 ${RUNAI_PKG[*]:-} (UV_TORCH_BACKEND=cpu)"
  t0=$SECONDS
  env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/root UV_TORCH_BACKEND=cpu UV_CACHE_DIR="$BASE/uvcache" \
    "$UV" pip install --python "$VENV/bin/python" ${TTL_PIP_EXTRA:-} "vllm-torchtpu==0.29.0" ${RUNAI_PKG[@]+"${RUNAI_PKG[@]}"} py-spy 2>&1 | ts | tee "$LOG/install.log"
  rc=${PIPESTATUS[0]}
  say "install rc=$rc seconds=$((SECONDS - t0))"
  [ "$rc" = 0 ] || exit 1
  fi
  "$UV" pip freeze --python "$VENV/bin/python" > "$LOG/pip_freeze.txt"
  grep -i -E '^(vllm|torch|jax|libtpu|tpu|transformers|numpy|ray|runai|google-cloud)' "$LOG/pip_freeze.txt" | tee -a "$LOG/host.log"
  du -sh "$VENV" | tee -a "$LOG/host.log"
  if [ "$RAY_MODE" = 1 ]; then chipmap_probe || exit 1; fi  # the pair's chip map (all hosts of the pair at once)
  if [ -n "${TTL_GCS_URI:-}" ]; then
    say "GCS access check $TTL_GCS_URI"
    python3.12 "$(dirname "$0")/torchtpu_gcs_tools.py" auth "$TTL_GCS_URI" 2>&1 | ts | tee -a "$LOG/host.log"
    [ "${PIPESTATUS[0]}" = 0 ] || exit 1
  fi
  if [ "${TTL_SKIP_COPY:-0}" != 1 ]; then
    if [ ! -f "$CKPT/.copy_done" ]; then
      say "copying checkpoint $SRC -> $CKPT"
      python3.12 "$(dirname "$0")/torchtpu_ckpt_copy.py" "$SRC" "$CKPT" --threads "${TTL_COPY_THREADS:-48}" 2>&1 | ts | tee "$LOG/ckpt_copy.log"
      [ "${PIPESTATUS[0]}" = 0 ] || exit 1
      touch "$CKPT/.copy_done"
    fi
    du -sh "$CKPT" | tee -a "$LOG/host.log"
  fi
  df -h /dev/shm | tee -a "$LOG/host.log"
  say "setup done"
  ;;

serve)
  TP=${TTL_TP:-$((4 * NH))}
  PORT=${TTL_PORT:-8000}
  [ "$RAY_MODE" = 1 ] && PORT=${TTL_PORT:-8011}  # not the pod's readiness port: this engine must not look like a cluster server
  say "serve start run=$RUN tp=$TP ray_mode=$RAY_MODE host_rank=$HRANK/$NH ip=$MYIP extra='${TTL_VLLM_EXTRA:-}'"
  if [ ! -x "$VENV/bin/python" ] || [ ! -x "$VENV/bin/vllm" ]; then  # pod recreated since setup: pod-local state is gone
    say "FATAL: no venv at $VENV on $MYIP (run the setup phase on this host first)"; exit 3
  fi
  run_isolated bash -c 'env | sort | grep -E "^(TPU|TORCH_TPU|VLLM|JAX|PATH|HOME)"' > "$LOG/serve_env.txt"
  "$VENV/bin/python" -c 'import vllm, vllm_torchtpu, torch, jax, libtpu; print(vllm.__file__, torch.__version__, jax.__version__)' 2>&1 | tee -a "$LOG/host.log"
  free -g | tee -a "$LOG/host.log"
  if [ "${TTL_PROBE_PLUGIN:-1}" = 1 ]; then
    SPK=$("$VENV/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
    cp "$(dirname "$0")/torchtpu_load_plugin.py" "$SPK/ttl_probe.py"
    mkdir -p "$SPK/ttl_probe-0.0.0.dist-info"
    printf 'Metadata-Version: 2.1\nName: ttl-probe\nVersion: 0.0.0\n' > "$SPK/ttl_probe-0.0.0.dist-info/METADATA"
    printf '[vllm.general_plugins]\nttl_probe = ttl_probe:register\n' > "$SPK/ttl_probe-0.0.0.dist-info/entry_points.txt"
    say "installed measurement-only plugin ttl_probe (sync cross-check of load_weights)"
  fi
  SPK=$("$VENV/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
  if [ "$RAY_MODE" = 1 ]; then  # placement only: pin the Ray executor's bundles to TTL_RAY_HOSTS (every host's venv)
    cp "$(dirname "$0")/torchtpu_ray_pin_plugin.py" "$SPK/ttl_ray_pin.py"
    mkdir -p "$SPK/ttl_ray_pin-0.0.0.dist-info"
    printf 'Metadata-Version: 2.1\nName: ttl-ray-pin\nVersion: 0.0.0\n' > "$SPK/ttl_ray_pin-0.0.0.dist-info/METADATA"
    printf '[vllm.general_plugins]\nttl_ray_pin = ttl_ray_pin:register\n' > "$SPK/ttl_ray_pin-0.0.0.dist-info/entry_points.txt"
    say "installed placement plugin ttl_ray_pin (hosts $TTL_RAY_HOSTS)"
    mkdir -p "$LOG/marks"; rm -f "$LOG/marks"/*
    # a marker server of an earlier run that died without its EXIT trap (Ray job stopped/killed) still holds the port
    pkill -f -- "$BASE/logs/[^ ]*/marks" 2>/dev/null && { say "killed a stale marker server"; sleep 0.5; }
    pkill -f -- "pm_ray_zygote.py serve" 2>/dev/null && say "killed a stale Ray worker zygote"
    python3.12 -c 'import functools, http.server, sys
class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):  # polled every 20 ms: no access log
        pass
http.server.ThreadingHTTPServer((sys.argv[1], int(sys.argv[2])), functools.partial(H, directory=sys.argv[3])).serve_forever()' \
      "$MYIP" "$MPORT" "$LOG/marks" > "$LOG/marks_http.err" 2>&1 &
    MARKSRV=$!
    sleep 0.3; kill -0 "$MARKSRV" 2>/dev/null || { say "marker server failed to start"; cat "$LOG/marks_http.err"; exit 1; }
    mark up
    await_mark "$MYIP" up 20 || { say "marker server did not start"; cat "$LOG/marks_http.err"; exit 1; }
    # The worker logs are collected from the raylet's log files afterwards (ttl_ray_pin: no log_to_driver), so every
    # vLLM line carries its own millisecond timestamp: logging format only, vLLM's default plus %(msecs).
    LOGCFG=$BASE/vllm_logging_ms.json
    cat > "$LOGCFG" <<'JSON'
{"version": 1, "disable_existing_loggers": false,
 "formatters": {"vllm": {"class": "vllm.logging_utils.NewLineFormatter", "datefmt": "%m-%d %H:%M:%S",
                         "format": "%(levelname)s %(asctime)s.%(msecs)03d [%(fileinfo)s:%(lineno)d] %(message)s"}},
 "handlers": {"vllm": {"class": "logging.StreamHandler", "formatter": "vllm", "level": "INFO", "stream": "ext://sys.stdout"}},
 "loggers": {"vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": false}}}
JSON
    env | grep -i -E '^(TPU|TORCH_TPU|XLA|JAX|LIBTPU|GLOG|ABSL|VLLM|ENABLE_PJRT|PJRT)' | sort > "$LOG/ambient_env.txt"
    # Canonical chip binding: local rank k takes the k-th local chip in slice-coordinate order (needs the pair's
    # chipmap, probed at setup or by the chipmap phase: both hosts at once, so it cannot be redone here alone)
    if [ -n "${TTL_CHIP_BIND:-}" ]; then
      chipmap_valid || { say "FATAL: TTL_CHIP_BIND=$TTL_CHIP_BIND but no chipmap for hosts $TTL_RAY_HOSTS rank $HRANK at $CHIPMAP: run --phase chipmap on this pair"; exit 3; }
      cp "$CHIPMAP" "$LOG/chipmap.json"
      say "chip binding $TTL_CHIP_BIND: $(python3.12 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("stock", d["stock_device_id"], "canonical", d["canonical"], "->", d["canonical_device_id"])' "$CHIPMAP")"
    fi
  else
    rm -rf "$SPK/ttl_ray_pin.py" "$SPK/ttl_ray_pin-0.0.0.dist-info"
  fi
  if [ "${TTL_PM:-0}" = 1 ]; then  # TPUFlex plugin; inert unless PM_* variables reach vLLM (TTL_EXTRA_ENV / TTL_GATE)
    cp "$(dirname "$0")/pm_torchtpu_plugin.py" "$SPK/pm_torchtpu.py"
    mkdir -p "$SPK/pm_torchtpu-0.0.0.dist-info"
    printf 'Metadata-Version: 2.1\nName: pm-torchtpu\nVersion: 0.0.0\n' > "$SPK/pm_torchtpu-0.0.0.dist-info/METADATA"
    printf '[vllm.general_plugins]\npm_torchtpu = pm_torchtpu:register\n' > "$SPK/pm_torchtpu-0.0.0.dist-info/entry_points.txt"
    say "installed TPUFlex plugin pm_torchtpu"
    # vLLM source patches, each gated by its own PM_* variable (inert for stock arms); idempotent
    python3.12 "$(dirname "$0")/pm_source_patches.py" apply "$SPK" 2>&1 | ts | tee -a "$LOG/host.log"
    [ "${PIPESTATUS[0]}" = 0 ] || exit 1
  else
    rm -rf "$SPK/pm_torchtpu.py" "$SPK/pm_torchtpu-0.0.0.dist-info"
  fi
  if [ "${TTL_PM_FRESH:-0}" = 1 ]; then
    say "fresh TPUFlex state: $(du -sh "$PMROOT"/* 2>/dev/null | tr '\n' ' ')"
    rm -rf "$PMROOT"
  fi
  mkdir -p "$PMROOT/bank" "$PMROOT/snap" "$PMROOT/t3"
  GATE_ENV=""
  [ "${TTL_GATE:-0}" = 1 ] && GATE_ENV="PM_GATE_FILE=$LOG/pm_gate" && rm -f "$LOG/pm_gate"
  TOOLS="$(dirname "$0")/torchtpu_gcs_tools.py"
  if [ -n "${TTL_EVICT_DIR:-}" ]; then
    for d in $TTL_EVICT_DIR; do
      say "evicting page cache of $d"
      python3.12 "$TOOLS" evict "$d" 2>&1 | tail -3 | ts | tee -a "$LOG/host.log"
    done
  fi
  if [ "${TTL_FRESH_CACHE:-0}" = 1 ]; then  # cold start: no compile/torch.compile/asset cache from earlier runs of this probe
    SINCE=${TTL_SINCE:?TTL_SINCE bounds which /dev/shm/torch_tpu_cache is mine}
    say "fresh caches: $(du -sh "$VCACHE" "$BASE/xdg" /dev/shm/torch_tpu_cache 2>/dev/null | tr '\n' ' ')"
    rm -rf "$BASE/vllm_cache" "$VCACHE" "$BASE/xdg"
    find /dev/shm -maxdepth 1 -name torch_tpu_cache -newermt "$SINCE" -exec rm -rf {} + 2>/dev/null
    [ -e /dev/shm/torch_tpu_cache ] && say "WARNING: /dev/shm/torch_tpu_cache older than $SINCE left in place"
  fi
  if [ -n "${TTL_FRESH_HOST:-}" ]; then  # fresh host (autoscale): no local weights, snapshot, bank or compile cache
    SINCE=${TTL_SINCE:?TTL_SINCE bounds which /dev/shm/torch_tpu_cache is mine}
    # pm | stock deliver the checkpoint (or TPUFlex state) into tmpfs; nodeliver serves from where CKPT points
    # (gcsfuse /models for stock-lazy, gs:// for Run:ai) and delivers at most its compile cache (TTL_FRESH_STATE=1).
    case "$TTL_FRESH_HOST" in
      pm|stock) case "$CKPT" in /dev/shm/*) ;; *) say "ERROR: fresh-host $TTL_FRESH_HOST needs a tmpfs CKPT path"; exit 1;; esac ;;
      nodeliver) ;;
      *) say "ERROR: unknown TTL_FRESH_HOST=$TTL_FRESH_HOST"; exit 1;;
    esac
    case "$CKPT_TMPFS" in /dev/shm/*) ;; *) say "ERROR: TTL_CKPT_TMPFS must be under /dev/shm"; exit 1;; esac
    say "fresh host ($TTL_FRESH_HOST): wiping $(du -sh "$PMROOT" "$VCACHE" "$BASE/vllm_cache" "$BASE/xdg" "$BASE/hf" /dev/shm/torch_tpu_cache "$CKPT_TMPFS" 2>/dev/null | tr '\n' ' ')"
    rm -rf "$PMROOT" "$VCACHE" "$BASE/vllm_cache" "$BASE/xdg" "$BASE/hf" "$CKPT_TMPFS"
    find /dev/shm -maxdepth 1 -name torch_tpu_cache -newermt "$SINCE" -exec rm -rf {} + 2>/dev/null
    [ -e /dev/shm/torch_tpu_cache ] && say "WARNING: /dev/shm/torch_tpu_cache older than $SINCE left in place"
    mkdir -p "$PMROOT/snap"
    # The checkpoint must come from GCS, not the page cache. The venv stands in for the container image, which a
    # freshly pulled node has just written (page-cache warm); TTL_FRESH_EVICT_VENV=1 measures a cold-disk image.
    EV=""; [ "$TTL_FRESH_HOST" != pm ] && EV="$SRC"; [ "${TTL_FRESH_EVICT_VENV:-0}" = 1 ] && EV="$EV $VENV"
    for d in $EV; do
      say "evicting page cache of $d"
      python3.12 "$TOOLS" evict "$d" 2>&1 | tail -3 | ts | tee -a "$LOG/host.log"
    done
    free -g | tee -a "$LOG/host.log"
  fi
  rm -f "$LOG/net.stop"
  python3.12 "$TOOLS" netmon "$LOG/net.tsv" "$LOG/net.stop" &
  NETMON=$!
  mem_kill() {
    [ -s "$LOG/sid" ] && pkill -KILL -s "$(cat "$LOG/sid")" 2>/dev/null
    kill_my_ray_workers; pkill -f "$(dirname "$0")/pm_artifacts.py" 2>/dev/null
    touch "$LOG/mem_watchdog_fired"
  }
  mem_watchdog mem_kill &
  MEMW=$!
  if [ -n "${TTL_PM_PEER:-}" ]; then  # the source's server must be up before T0 (its startup is not this boot's cost)
    pdl=$(( $(date +%s) + ${TTL_PM_PEER_WAIT_S:-900} ))
    until curl -sf -m 2 "$PMGCS/health" > /dev/null; do
      [ "$(date +%s)" -ge "$pdl" ] && { say "pm-peer source $PMGCS not up"; exit 1; }
      sleep 0.5
    done
    say "pm-peer source $PMGCS up"
  fi
  sleep 5  # idle RX baseline before launch
  if [ "$RAY_MODE" = 1 ]; then  # every host prepared (wipes, evictions, baseline) before the common T0
    if [ "$HRANK" = 0 ]; then
      for h in "${PEERS[@]}"; do await_mark "$h" prep_done 900 || exit 1; done
      mark t0
    else
      mark prep_done
      await_mark "$HOST0" t0 1800 || exit 1
    fi
  fi
  T0_FRESH=""
  FETCH=""
  host_done() {  # host_done NAME: a peer reports, host rank 0 waits for every peer's report
    if [ "$HRANK" != 0 ]; then mark "$1"; return 0; fi
    for h in "${PEERS[@]}"; do await_mark "$h" "$1" "${TTL_READY_TIMEOUT:-2400}" || exit 1; done
    [ ${#PEERS[@]} -gt 0 ] && say "peers report $1"
    return 0
  }
  if [ -n "${TTL_FRESH_HOST:-}" ]; then
    T0_FRESH=$(date +%s.%N)
    say "T0 critical path starts (fresh host: $TTL_FRESH_HOST)"
    start_fetch() {  # pm_artifacts.py fetch in the background, Run:ai streamer from this host's TTL_PM_GCS
      env RUNAI_STREAMER_CONCURRENCY="${TTL_FETCH_CONC:-32}" python3.12 "$(dirname "$0")/pm_artifacts.py" fetch "$@" \
        --src "${PMGCS:?TTL_PM_GCS}" --lib "$SPK/runai_model_streamer/libstreamer/lib/libstreamer.so" \
        --model-dir "$CKPT_TMPFS" --bank-dir "$VCACHE" --xla-dir /dev/shm/torch_tpu_cache --snap-dir "$PMROOT/snap" \
        > >(ts > "$LOG/fetch.log") 2>&1 &
      FETCH=$!
    }
    if [ "$TTL_FRESH_HOST" = pm ] && [ "${TTL_PM_WEIGHTS:-snap}" = ckpt ]; then
      # Ablation (loo-snapshot): the TPUFlex state without the weight snapshot (bank + XLA cache: --state-only)
      # and the plain HF checkpoint for the stock loader, with the same Run:ai streamer (fetch-ckpt, as stock-ckpt-cc).
      # Both stream while vLLM starts; the stock load_weights waits for the checkpoint's completion marker
      # (plugin: PM_WEIGHTS_WAIT_FILE=$PMROOT/.weights_done), the workers' gate for .state_done as usual.
      (
        start_fetch --state-only
        env RUNAI_STREAMER_CONCURRENCY="${TTL_FETCH_CONC:-32}" python3.12 "$(dirname "$0")/pm_artifacts.py" fetch-ckpt \
          --src "${TTL_CKPT_GCS:?TTL_CKPT_GCS}" --lib "$SPK/runai_model_streamer/libstreamer/lib/libstreamer.so" \
          --dst "$CKPT" --marker-dir "$PMROOT" --parts "${TTL_CKPT_PARTS:-2}" > >(ts > "$LOG/fetch_ckpt.log") 2>&1 &
        CK=$!
        wait "$FETCH"; a=$?; wait "$CK"; b=$?
        exit $(( a != 0 || b != 0 ))
      ) &
      FETCH=$!
      until [ -e "$PMROOT/.config_done" ]; do
        kill -0 "$FETCH" 2>/dev/null || { say "fetches exited before the checkpoint config landed"; cat "$LOG/fetch.log" "$LOG/fetch_ckpt.log"; exit 1; }
        sleep 0.02
      done
      say "checkpoint config fetched; state and checkpoint weights streaming in the background"
    elif [ "$TTL_FRESH_HOST" = pm ]; then  # TPUFlex: fetch the startup state while vLLM starts up
      start_fetch
      until [ -e "$PMROOT/.model_done" ]; do
        kill -0 "$FETCH" 2>/dev/null || { say "fetch exited before the model dir landed"; cat "$LOG/fetch.log"; exit 1; }
        sleep 0.02
      done
      say "model dir fetched"
      if [ "${TTL_PM_SEQUENTIAL:-0}" = 1 ]; then  # ablation (loo-overlap): the whole state lands on every host first
        wait "$FETCH"; rc=$?; FETCH=""
        say "PM state fetched (sequential) rc=$rc $(grep -o 'FETCH_JSON=.*' "$LOG/fetch.log" | cut -c1-200)"
        [ "$rc" = 0 ] || exit 1
        host_done state_done
      fi
    elif [ "$TTL_FRESH_HOST" = nodeliver ]; then  # weights are read where CKPT points, by the engine itself
      if [ "${TTL_FRESH_STATE:-0}" = 1 ]; then  # its own compile cache from GCS, complete before launch
        start_fetch --state-only
        wait "$FETCH"; rc=$?; FETCH=""
        say "compile cache fetched rc=$rc $(grep -o 'FETCH_JSON=.*' "$LOG/fetch.log" | cut -c1-200)"
        [ "$rc" = 0 ] || exit 1
        host_done state_done
      fi
    elif [ -n "${TTL_CKPT_GCS:-}" ]; then  # stock with the same Run:ai streamer for the plain checkpoint
      # TTL_FRESH_STATE=1: its own compile cache (VLLM_CACHE_ROOT + XLA cache) from GCS, complete before launch;
      # fetched first and alone (f16: its tar extraction took 21.7 s next to the weight stream, ~4 s alone).
      # TTL_CKPT_OVERLAP=1: launch once config/tokenizer and that cache are there; the weights keep streaming
      # (stock vLLM opens them in load_weights, ~40 s later; fetch-ckpt renames the shards when all landed).
      if [ "${TTL_FRESH_STATE:-0}" = 1 ]; then
        start_fetch --state-only
        wait "$FETCH"; rc=$?; FETCH=""
        say "compile cache fetched rc=$rc"
        [ "$rc" = 0 ] || exit 1
      fi
      env RUNAI_STREAMER_CONCURRENCY="${TTL_FETCH_CONC:-32}" python3.12 "$(dirname "$0")/pm_artifacts.py" fetch-ckpt \
        --src "$TTL_CKPT_GCS" --lib "$SPK/runai_model_streamer/libstreamer/lib/libstreamer.so" --dst "$CKPT" \
        --marker-dir "$PMROOT" --parts "${TTL_CKPT_PARTS:-2}" > >(ts > "$LOG/fetch_ckpt.log") 2>&1 &
      CKFETCH=$!
      until [ -e "$PMROOT/.config_done" ]; do
        kill -0 "$CKFETCH" 2>/dev/null || { say "fetch-ckpt exited early"; cat "$LOG/fetch_ckpt.log"; exit 1; }
        sleep 0.02
      done
      if [ "${TTL_CKPT_OVERLAP:-0}" = 1 ]; then
        [ "$RAY_MODE" = 1 ] && { say "ERROR: TTL_CKPT_OVERLAP is single-host only"; exit 1; }
        FETCH=$CKFETCH
        say "checkpoint config fetched; weights streaming in the background"
      else
        wait "$CKFETCH" || { say "fetch-ckpt failed"; exit 1; }
        say "checkpoint fetched $(grep -o 'FETCH_CKPT_JSON=.*' "$LOG/fetch_ckpt.log" | cut -c1-300)"
        host_done ckpt_done
      fi
    else  # stock: copy the checkpoint from the GCS FUSE mount to tmpfs (sliced, 48 streams), then launch
      # TTL_FRESH_STATE=1: stock vLLM whose own compile cache (VLLM_CACHE_ROOT + XLA cache) was published to GCS;
      # fetched next to the checkpoint copy, complete before launch.
      [ "${TTL_FRESH_STATE:-0}" = 1 ] && start_fetch --state-only
      python3.12 "$(dirname "$0")/torchtpu_ckpt_copy.py" "$SRC" "$CKPT" --threads "${TTL_COPY_THREADS:-48}" 2>&1 | ts | tee "$LOG/ckpt_copy.log"
      [ "${PIPESTATUS[0]}" = 0 ] || { say "checkpoint copy failed"; exit 1; }
      say "checkpoint copied"
      if [ -n "$FETCH" ]; then
        wait "$FETCH"; rc=$?; FETCH=""
        say "compile cache fetched rc=$rc $(grep -o 'FETCH_JSON=.*' "$LOG/fetch.log" | cut -c1-200)"
        [ "$rc" = 0 ] || exit 1
      fi
      host_done ckpt_done
    fi
  fi
  RZPID=""
  if [ "$RAY_MODE" = 1 ]; then
    # The driver joins the running Ray cluster (ray.init in the engine core, ttl_ray_pin). Its worker actors are
    # started by each host's raylet with this venv's interpreter and this runtime env; every value that must reach
    # a worker before vLLM's own env propagation (which never overrides the raylet's pod env) goes in here.
    # TTL_RAY_ZYGOTE=1 (TPUFlex): that interpreter is pm_ray_zygote.py's launcher, and the worker processes are
    # forked from this host's pre-imported zygote, started here (at T0 / launch) with the same environment.
    RZYG_ENV=""; PYEXE="$VENV/bin/python"
    if [ "${TTL_RAY_ZYGOTE:-0}" = 1 ]; then
      cp "$(dirname "$0")/pm_ray_zygote.py" "$BASE/pm_ray_zygote.py"
      RZYG_ENV="PM_RAY_ZYGOTE_SOCK=/tmp/pm_ray_zygote_$RUN.sock"
      PYEXE="$VENV/bin/python -S $BASE/pm_ray_zygote.py exec"
    fi
    # shellcheck disable=SC2086
    WORKER_ENV_JSON=$(kv_json $TPU_ENV)
    # shellcheck disable=SC2206
    RT_KV=(PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin" HOME=/root LANG=C.UTF-8 HF_HOME="$BASE/hf"
      VLLM_CACHE_ROOT="$VCACHE" XDG_CACHE_HOME="$BASE/xdg" VLLM_LOGGING_CONFIG_PATH="$LOGCFG" $TPU_ENV
      TTL_RAY_HOSTS="$TTL_RAY_HOSTS" TTL_RAY_WORKER=1 TTL_RUN="$RUN"
      TTL_RAY_WORKER_UNSET="${TTL_RAY_WORKER_UNSET:-ENABLE_PJRT_COMPATIBILITY,TPU_NAME,JAX_PLATFORMS}"
      TTL_RAY_WORKER_ENV="$WORKER_ENV_JSON" TTL_CHIP_BIND="${TTL_CHIP_BIND:-}" TTL_CHIPMAP="$CHIPMAP" ${TTL_EXTRA_ENV:-} $GATE_ENV $RZYG_ENV)
    RUNTIME_JSON=$(python3.12 -c 'import json, sys
print(json.dumps({"py_executable": sys.argv[1], "env_vars": dict(a.split("=", 1) for a in sys.argv[2:])}, separators=(",", ":")))' \
      "$PYEXE" "${RT_KV[@]}")
    echo "$RUNTIME_JSON" > "$LOG/ray_runtime_env.json"
    if [ -n "$RZYG_ENV" ]; then
      env -i "${RT_KV[@]}" "$VENV/bin/python" "$BASE/pm_ray_zygote.py" serve > >(ts > "$LOG/ray_zygote.log") 2>&1 &
      RZPID=$!
      say "Ray worker zygote started pid=$RZPID"
    fi
  fi
  if [ "$RAY_MODE" = 1 ] && [ "${TTL_PYSPY_RAY:-0}" = 1 ]; then  # debug: periodic py-spy dumps of this host's Ray workers
    mkdir -p "$LOG/pyspy"
    ( while [ ! -e "$LOG/net.stop" ]; do
        sleep "${TTL_PYSPY_EVERY:-30}"
        for pid in $(my_ray_pids); do
          timeout 20 "$VENV/bin/py-spy" dump --pid "$pid" > "$LOG/pyspy/$(date -u +%H%M%S)_ray_${pid}.txt" 2>&1
        done
      done ) &
  fi
  if [ "$RAY_MODE" = 1 ] && [ -n "${TTL_STACK_SIG_S:-}" ]; then  # debug: SIGUSR2 -> faulthandler stacks (PM_STACK_DUMP_SIG)
    ( while [ ! -e "$LOG/net.stop" ]; do
        sleep "$TTL_STACK_SIG_S"
        for f in "$LOG"/pm_stacks_rank*_pid*.txt; do [ -e "$f" ] && kill -USR2 "$(basename "${f##*_pid}" .txt)" 2>/dev/null; done
      done ) &
  fi
  if [ "$RAY_MODE" = 1 ] && [ "$HRANK" != 0 ]; then
    # Peer host: this host's Ray workers are started by its raylet when the driver's executor asks for them. Relay
    # the pre-spawn gate to this host's workers, then wait for the driver host to finish.
    if [ "${TTL_GATE:-0}" = 1 ]; then
      local_tp=$(( TP / NH ))
      until [ "$(ls "$LOG"/pm_gate.parked.* 2>/dev/null | wc -l)" -ge "$local_tp" ]; do
        has_mark "$HOST0" done && break
        sleep 0.05
      done
      mark parked
      say "PARKED $(ls "$LOG"/pm_gate.parked.* 2>/dev/null | wc -l) local workers"
      until has_mark "$HOST0" gate; do
        has_mark "$HOST0" done && break
        sleep 0.02
      done
      touch "$LOG/pm_gate"; say "GATE_RELAYED"
    fi
    await_mark "$HOST0" done $(( ${TTL_READY_TIMEOUT:-2400} + 900 ))
    say "driver host done"
    [ -n "$FETCH" ] && { wait "$FETCH"; say "fetch rc=$? $(grep -h -o -E 'FETCH(_CKPT)?_JSON=.*' "$LOG/fetch.log" "$LOG/fetch_ckpt.log" 2>/dev/null | cut -c1-400)"; }
    touch "$LOG/net.stop"; wait "$NETMON" 2>/dev/null; kill "$MEMW" 2>/dev/null
    [ -n "$RZPID" ] && kill "$RZPID" 2>/dev/null
    wait_ray_workers
    JID=$(curl -sf -m 5 "http://$HOST0:$MPORT/jobid"); [ "${JID%% *}" = "$RUN" ] || JID="$RUN "
    collect_ray_logs "${JID#* }"
    for d in ${TTL_EVICT_DIR:-}; do python3.12 "$TOOLS" resident "$d" 2>&1 | tail -2 | ts | tee -a "$LOG/host.log"; done
    mark peer_done
    sleep 5  # the driver host polls every 20 ms
    kill "$MARKSRV" 2>/dev/null
    say "serve (peer) done"
    exit 0
  fi
  # TPUFlex zygote (TTL_ZYGOTE=1): a pre-imported fork server started with the API server, in the same session
  # and environment; the engine core and the TP workers are forked from it (plugin: PM_ZYGOTE_SOCK).
  ZYG_ENV=""
  [ "${TTL_ZYGOTE:-0}" = 1 ] && ZYG_ENV="PM_ZYGOTE_SOCK=/tmp/pm_zygote_$RUN.sock"
  SRC_SID=""
  src_alive() { [ -n "$SRC_SID" ] && ps -o stat= -s "$SRC_SID" 2>/dev/null | grep -qv '^Z'; }
  if [ "${TTL_SOURCE:-0}" = 1 ]; then  # handoff: a source engine serves on the chips while the target pre-spawns
    SPORT=${TTL_SRC_PORT:-8001}
    # shellcheck disable=SC2086
    setsid bash -c "exec env -i PATH='$VENV/bin:/usr/local/bin:/usr/bin:/bin' HOME=/root LANG=C.UTF-8 HF_HOME='$BASE/hf' VLLM_CACHE_ROOT='$VCACHE' XDG_CACHE_HOME='$BASE/xdg' TPU_SKIP_MDS_QUERY=true TPU_ACCELERATOR_TYPE=${TTL_ACCEL:-v5litepod-4} TPU_TOPOLOGY=${TTL_TOPO:-2x2} TPU_TOPOLOGY_WRAP=false,false,false TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost TPU_MULTIHOST_BACKEND=mp VLLM_HOST_IP=127.0.0.1 ${TTL_SRC_EXTRA_ENV:-} '$VENV/bin/vllm' serve '$CKPT' --served-model-name gemma4 --tensor-parallel-size $TP --max-model-len ${TTL_MAXLEN:-2048} --max-num-seqs ${TTL_MAXSEQS:-8} --port $SPORT ${TTL_SRC_VLLM_EXTRA:-}" > >(ts > "$LOG/source_vllm_raw.log") 2>&1 &
    SRC_SID=$!
    say "source engine launched sid=$SRC_SID port=$SPORT"
    src_deadline=$(( $(date +%s) + ${TTL_READY_TIMEOUT:-2400} ))
    until curl -sf -m 3 "http://127.0.0.1:$SPORT/health" > /dev/null; do
      if ! src_alive || [ "$(date +%s)" -ge "$src_deadline" ]; then
        say "source engine failed to become ready"; pkill -KILL -s "$SRC_SID" 2>/dev/null; exit 1
      fi
      sleep 0.5
    done
    curl -s -m 120 "http://127.0.0.1:$SPORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"gemma4","messages":[{"role":"user","content":"What is the capital of France? Answer in one sentence."}],"temperature":0,"max_tokens":16}' \
      > "$LOG/source_greedy.json"
    say "SOURCE_READY serving on the chips: $(python3.12 -c 'import json,sys; print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"])' "$LOG/source_greedy.json" 2>&1 | head -1)"
  fi
  DRV_EXTRA=""; EXEC_FLAGS=""
  if [ "$RAY_MODE" = 1 ]; then
    DRV_EXTRA="RAY_ADDRESS=${TTL_RAY_ADDRESS:?TTL_RAY_ADDRESS} RAY_DEDUP_LOGS=0 TTL_RAY_HOSTS=$TTL_RAY_HOSTS TTL_RAY_RUNTIME_ENV='$RUNTIME_JSON' TTL_RAY_WORKER_ENV='$WORKER_ENV_JSON' TTL_RAY_JOBID_FILE=$LOG/ray_job_id VLLM_LOGGING_CONFIG_PATH=$LOGCFG"
    EXEC_FLAGS="--distributed-executor-backend ray"
  fi
  VENV_ENV="PATH='$VENV/bin:/usr/local/bin:/usr/bin:/bin' HOME=/root LANG=C.UTF-8 HF_HOME='$BASE/hf' VLLM_CACHE_ROOT='$VCACHE' XDG_CACHE_HOME='$BASE/xdg' $TPU_ENV $TPU_ENV_HOST ${TTL_EXTRA_ENV:-} $GATE_ENV $ZYG_ENV $DRV_EXTRA"
  ZYG_CMD=""
  [ -n "$ZYG_ENV" ] && ZYG_CMD="env -i $VENV_ENV '$VENV/bin/python' '$(dirname "$0")/pm_zygote.py' &"
  # shellcheck disable=SC2086
  setsid bash -c "$ZYG_CMD exec env -i $VENV_ENV '$VENV/bin/vllm' serve '$CKPT' --served-model-name gemma4 --tensor-parallel-size $TP $EXEC_FLAGS --max-model-len ${TTL_MAXLEN:-2048} --max-num-seqs ${TTL_MAXSEQS:-8} --port $PORT ${TTL_VLLM_EXTRA:-}" > >(ts > "$LOG/vllm_raw.log") 2>&1 &
  SID=$!
  T_LAUNCH=$(date +%s.%N)
  T0=${T0_FRESH:-$T_LAUNCH}  # start of the critical path: launch, gate opening (pre-spawned) or fresh-host T0
  say "vllm launched sid=$SID"
  echo $SID > "$LOG/sid"
  stop_server() {
    say "stopping server session $SID"
    pkill -TERM -s "$SID" 2>/dev/null; [ -n "$SRC_SID" ] && pkill -TERM -s "$SRC_SID" 2>/dev/null
    sleep 10; pkill -KILL -s "$SID" 2>/dev/null; [ -n "$SRC_SID" ] && pkill -KILL -s "$SRC_SID" 2>/dev/null
    [ -n "$FETCH" ] && kill "$FETCH" 2>/dev/null
    touch "$LOG/net.stop"
    if [ "$RAY_MODE" = 1 ]; then
      [ -n "$RZPID" ] && kill "$RZPID" 2>/dev/null
      wait_ray_workers
      [ -f "$LOG/ray_job_id" ] && echo "$RUN $(tr -d '\n' < "$LOG/ray_job_id")" > "$LOG/marks/jobid"
      collect_ray_logs "$(cat "$LOG/ray_job_id" 2>/dev/null)"
      mark done
    fi
    return 0
  }
  trap 'stop_server; [ -n "$MARKSRV" ] && kill "$MARKSRV" 2>/dev/null; [ -n "$MEMW" ] && kill "$MEMW" 2>/dev/null' EXIT TERM INT
  mkdir -p "$LOG/pyspy"
  [ "${TTL_PYSPY:-0}" = 1 ] && (
    while kill -0 $SID 2>/dev/null; do
      sleep "${TTL_PYSPY_EVERY:-30}"
      for pid in $(pgrep -s "$SID"); do
        n=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-60 | tr -c 'A-Za-z0-9_.:-' '_')
        "$VENV/bin/py-spy" dump --pid "$pid" > "$LOG/pyspy/$(date -u +%H%M%S)_${pid}_${n}.txt" 2>&1 || true
      done
    done
  ) &
  WATCH=$!
  # Stream the engine log to stdout while waiting for readiness.
  tail -n +1 -F "$LOG/vllm_raw.log" 2>/dev/null &
  TAILP=$!
  if [ -n "$GATE_ENV" ]; then
    park_deadline=$(( $(date +%s) + ${TTL_PARK_TIMEOUT:-600} ))
    parked() {  # Ray mode: this host's marker files plus every peer's report; else the workers' log lines
      if [ "$RAY_MODE" = 1 ]; then
        [ "$(ls "$LOG"/pm_gate.parked.* 2>/dev/null | wc -l)" -ge $(( TP / NH )) ] || return 1
        for h in "${PEERS[@]}"; do has_mark "$h" parked || return 1; done
        return 0
      fi
      [ "$(grep -c 'PM_PARKED' "$LOG/vllm_raw.log" 2>/dev/null)" -ge "$TP" ]
    }
    until parked; do
      if ! pgrep -s "$SID" > /dev/null; then say "vllm session exited before all workers parked"; break; fi
      if [ "$(date +%s)" -ge "$park_deadline" ]; then say "park timeout"; break; fi
      sleep 0.05
    done
    say "PARKED (all $TP workers) $(awk -v a="$T_LAUNCH" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}') s after launch; settling ${TTL_GATE_SETTLE:-5}s"
    sleep "${TTL_GATE_SETTLE:-5}"
    if [ -n "$SRC_SID" ]; then  # the source must still serve normally while the target is parked next to it
      python3.12 "$(dirname "$0")/torchtpu_req_probe.py" "$SPORT" "$LOG/source_req_probe_parked.json" 2>&1 | ts | tee -a "$LOG/host.log"
      say "SOURCE_PROBED while target parked"
    fi
    T0=$(date +%s.%N)
    say "T0 critical path starts"
    if [ -n "$SRC_SID" ]; then  # T0 = stop the source; open the gate once its processes have released the chips
      pkill -TERM -s "$SRC_SID" 2>/dev/null
      kill_at=$(( $(date +%s) + 60 ))
      while src_alive; do
        [ "$(date +%s)" -ge "$kill_at" ] && { say "source still alive 60 s after SIGTERM; SIGKILL"; pkill -KILL -s "$SRC_SID"; }
        sleep 0.05
      done
      say "SOURCE_EXITED $(awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}') s after SIGTERM"
    fi
    touch "$LOG/pm_gate"
    [ "$RAY_MODE" = 1 ] && mark gate
    say "GATE_OPEN T0 (pre-spawn took $(awk -v a="$T_LAUNCH" -v b="$T0" 'BEGIN{printf "%.1f", b-a}') s incl. settle)"
  fi
  deadline=$(( $(date +%s) + ${TTL_READY_TIMEOUT:-2400} ))
  READY=0
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! pgrep -s "$SID" > /dev/null; then say "vllm session exited before ready"; break; fi
    if curl -sf -m 3 "http://127.0.0.1:$PORT/health" > /dev/null; then
      READY=1; T_READY=$(date +%s.%N)
      say "READY (health 200) $(awk -v a="$T_LAUNCH" -v z="$T0" -v b="$T_READY" 'BEGIN{printf "from_launch=%.2f from_T0=%.2f", b-a, b-z}')"
      break
    fi
    sleep "${TTL_POLL:-0.25}"
  done
  if [ "$READY" = 1 ] && [ "${TTL_GREEDY_CHECK:-0}" = 1 ]; then
    # The matrix's output check (same request as the CUDA campaign's pod_run.sh): greedy, compared offline to the
    # reference run's text.
    curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"gemma4","messages":[{"role":"user","content":"Name the three primary colors in one short sentence."}],"max_tokens":32,"temperature":0}' \
      > "$LOG/greedy_check.json"
    T_CHECK=$(date +%s.%N)
    say "GREEDY_CHECK $(awk -v z="$T0" -v b="$T_CHECK" 'BEGIN{printf "from_T0=%.2f", b-z}') $(python3.12 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"]))' "$LOG/greedy_check.json" 2>&1 | head -1)"
  fi
  if [ "$READY" = 1 ] && [ "${TTL_REQ_PROBE:-0}" = 1 ]; then
    python3.12 "$(dirname "$0")/torchtpu_req_probe.py" "$PORT" "$LOG/req_probe.json" 2>&1 | ts | tee -a "$LOG/host.log"
  elif [ "$READY" = 1 ] && [ "${TTL_GREEDY_CHECK:-0}" != 1 ]; then
    for i in 1 2; do
      curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"gemma4","messages":[{"role":"user","content":"What is the capital of France? Answer in one sentence."}],"temperature":0,"max_tokens":32}' \
        | tee "$LOG/greedy_$i.json" ; echo
    done
  fi
  [ "$READY" = 1 ] || say "NOT_READY within the ready timeout ${TTL_READY_TIMEOUT:-2400} s (or the engine exited)"
  sleep 3; kill $TAILP 2>/dev/null
  kill $WATCH 2>/dev/null
  stop_server
  trap - EXIT TERM INT
  touch "$LOG/net.stop"; wait "$NETMON" 2>/dev/null; kill "$MEMW" 2>/dev/null
  if [ -n "$FETCH" ]; then
    wait "$FETCH"; say "fetch rc=$? $(grep -h -o -E 'FETCH(_CKPT)?_JSON=.*' "$LOG/fetch.log" "$LOG/fetch_ckpt.log" 2>/dev/null | cut -c1-400)"
  fi
  for d in ${TTL_EVICT_DIR:-}; do python3.12 "$TOOLS" resident "$d" 2>&1 | tail -2 | ts | tee -a "$LOG/host.log"; done
  echo "{\"ready\": $READY, \"ready_timeout_s\": ${TTL_READY_TIMEOUT:-2400}}" > "$LOG/serve_result.json"
  if [ "$RAY_MODE" = 1 ]; then  # keep this marker server up until every peer has seen "done"
    for h in "${PEERS[@]}"; do await_mark "$h" peer_done 300; done
    kill "$MARKSRV" 2>/dev/null
  fi
  say "serve done ready=$READY"
  ;;

chipmap)  # the per-host chip enumeration probe alone (chips must be free); refreshes the setup cache
  chipmap_probe
  exit $?
  ;;

pmupload)  # pack this host's TPUFlex startup state (weight-less model dir, bank, XLA cache, snapshot) to GCS
  PACK=/dev/shm/pm_pack
  rm -rf "$PACK"
  "$VENV/bin/python" "$(dirname "$0")/pm_artifacts.py" pack --model-dir "$CKPT" --bank-dir "$VCACHE" \
    --xla-dir /dev/shm/torch_tpu_cache --snap-dir "$PMROOT/snap" --out "$PACK" 2>&1 | ts | tee -a "$LOG/host.log"
  [ "${PIPESTATUS[0]}" = 0 ] || exit 1
  "$VENV/bin/python" "$(dirname "$0")/pm_artifacts.py" upload --pack "$PACK" --dst "${PMGCS:?TTL_PM_GCS}" \
    --workers "${TTL_UPLOAD_WORKERS:-32}" 2>&1 | ts | tee -a "$LOG/host.log"
  rc=${PIPESTATUS[0]}
  rm -rf "$PACK"
  exit "$rc"
  ;;

pmserve)  # pm-peer source: pack this host's TPUFlex state and serve it over HTTP (Range) to a target pair
  PACK=/dev/shm/pm_pack
  rm -rf "$PACK"
  say "pmserve start ip=$MYIP rank=$HRANK"
  "$VENV/bin/python" "$(dirname "$0")/pm_artifacts.py" pack --model-dir "$CKPT" --bank-dir "$VCACHE" \
    --xla-dir /dev/shm/torch_tpu_cache --snap-dir "$PMROOT/snap" --out "$PACK" 2>&1 | ts | tee -a "$LOG/host.log"
  [ "${PIPESTATUS[0]}" = 0 ] || exit 1
  pkill -f -- "pm_artifacts.py serve" 2>/dev/null && say "killed a stale pm-peer server"
  python3.12 "$(dirname "$0")/pm_artifacts.py" serve --pack "$PACK" --bind "$MYIP" --port "${TTL_PM_SERVE_PORT:-18471}" \
    --idle "${TTL_PM_SERVE_IDLE_S:-60}" --max-s "${TTL_PM_SERVE_MAX_S:-1800}" 2>&1 | ts | tee -a "$LOG/host.log"
  rc=${PIPESTATUS[0]}
  rm -rf "$PACK"
  exit "$rc"
  ;;

venvupload)  # the setup venv as a tarball (same path /root/torchtpu_load/venv) for e2e fresh pools; rank 0 only
  [ "$HRANK" = 0 ] || exit 0
  [ -x "$VENV/bin/python" ] || { say "FATAL: no venv"; exit 3; }
  T=/dev/shm/ttl_venv.tar.gz
  tar -C / -czf "$T" "${VENV#/}" && say "packed $(du -h "$T" | cut -f1)"
  "$VENV/bin/python" -c 'import sys
from google.cloud import storage
b, _, n = sys.argv[2].removeprefix("gs://").partition("/")
storage.Client().bucket(b).blob(n).upload_from_filename(sys.argv[1], timeout=1800)
print("uploaded", sys.argv[2])' "$T" "${TTL_VENV_GCS:?TTL_VENV_GCS}" 2>&1 | ts | tee -a "$LOG/host.log"
  rc=${PIPESTATUS[0]}
  rm -f "$T"
  exit "$rc"
  ;;

pmdelete)
  "$VENV/bin/python" "$(dirname "$0")/pm_artifacts.py" delete --dst "${PMGCS:?TTL_PM_GCS}" 2>&1 | ts | tee -a "$LOG/host.log"
  exit "${PIPESTATUS[0]}"
  ;;

fusebench)
  say "fusebench start run=$RUN host=$(hostname)"
  python3.12 "$(dirname "$0")/torchtpu_fuse_bench.py" ${TTL_FB_FILE:-} 2>&1 | ts
  rc=${PIPESTATUS[0]}
  rm -rf "$LOG"; rmdir "$BASE/logs" "$BASE" 2>/dev/null
  exit "$rc"
  ;;

h2dbench)  # host-to-device bandwidth ceiling (JAX/PJRT, all local chips, one process); needs the setup venv
  say "h2dbench start run=$RUN host=$(hostname)"
  env -i PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin" HOME=/root TPU_SKIP_MDS_QUERY=true \
    TPU_ACCELERATOR_TYPE=${TTL_ACCEL:-v5litepod-4} TPU_TOPOLOGY=${TTL_TOPO:-2x2} TPU_TOPOLOGY_WRAP=false,false,false \
    TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost \
    "$VENV/bin/python" "$(dirname "$0")/torchtpu_h2d_bench.py" ${TTL_H2D_GB:-4} 2>&1 | ts | tee "$LOG/h2d.log"
  exit "${PIPESTATUS[0]}"
  ;;

cleanup)
  say "cleanup"
  SINCE=${TTL_SINCE:?TTL_SINCE (e.g. "2026-10-22 18:20") bounds which leftovers are mine}
  pkill -KILL -s "$(cat "$BASE"/logs/*/sid 2>/dev/null | tr '\n' ' ' | awk '{print $NF}')" 2>/dev/null
  pkill -KILL -f "$VENV/bin" 2>/dev/null
  pids=$(my_ray_pids); [ -n "$pids" ] && { echo "killing leftover Ray workers of this probe: $(echo $pids)"; kill -KILL $pids; }
  pkill -f "ThreadingHTTPServer" 2>/dev/null; pkill -f "pm_artifacts.py serve" 2>/dev/null
  find /tmp -maxdepth 1 -name 'pm_*zygote_*.sock' -newermt "$SINCE" -delete 2>/dev/null
  # Leftovers of the vLLM/torch_tpu/libtpu processes of this probe (all created after $SINCE; nobody else ran on this host).
  find /dev/shm -maxdepth 1 -name torch_tpu_cache -newermt "$SINCE" -exec rm -rf {} + 2>/dev/null
  find /dev/shm/torch_tpu_cache -type f ! -newermt "$SINCE" 2>/dev/null | head -1 | grep -q . && echo "WARNING: older files in torch_tpu_cache"
  find /tmp -maxdepth 1 \( -name 'tpu_diag_trigger_*.socket' -o -name 'vllm_dist_*' \) -newermt "$SINCE" -delete 2>/dev/null
  find /tmp -maxdepth 1 -type s -name '*-*-*-*-*' -newermt "$SINCE" -delete 2>/dev/null
  find /tmp/tpu_logs -maxdepth 1 -type f -name 'tpu_driver.*.log.*' -newermt "$SINCE" -delete 2>/dev/null
  find /dev/shm -maxdepth 1 -type f -name 'sem.mp-*' -newermt "$SINCE" -delete 2>/dev/null
  find /tmp /tmp/tpu_logs -maxdepth 1 -xtype l -newermt "$SINCE" -print -delete 2>/dev/null
  case "$CKPT" in /dev/shm/*) rm -rf "$CKPT";; esac  # never the gcsfuse mount or a gs:// URI
  rm -rf /dev/shm/pm_pack
  rm -rf "$PMROOT"  # TPUFlex bank / snapshot / Tier-3 cache
  rm -rf "$BASE/venv" "$BASE/uvtool" "$BASE/uvcache" "$BASE/hf" "$BASE/vllm_cache" "$BASE/xdg" "$BASE/logs" \
    "$CHIPMAP" "$CHIPMAP.tmp" "$BASE/pm_ray_zygote.py" "$BASE/vllm_logging_ms.json"; rmdir "$BASE" 2>/dev/null
  [ -e "$BASE" ] && echo "left in $BASE: $(ls -A "$BASE" | tr '\n' ' ')"
  ls -la /dev/shm | grep -v psm_
  ls /tmp | grep -c -E '^[0-9a-f]{8}-' 
  df -h /dev/shm /
  exit 0
  ;;
*) echo "unknown phase"; exit 2;;
esac
