#!/usr/bin/env bash
# Warm-start experiment: stage the Gemma-4 31B checkpoint into host RAM (/dev/shm tmpfs) on
# every TPU worker host, so engine start-up loads weights from local memory instead of the
# gcsfuse /models mount.  Run under the cluster lock, outside any engine start-up.
# Usage: scripts/reconfig_warm_stage.sh stage|remove|status
# Each host holds one 62.5 GB copy (tmpfs counts against host RAM; 188 GB per host).
set -euo pipefail
NS=llm-d-optimized-baseline
SRC=/models/gemma-4-31b
DST=/dev/shm/gemma-4-31b
PODS=$(kubectl -n $NS get pods -o name | grep tpu-worker-group-worker)
case ${1:?stage|remove|status} in
  stage)
    for p in $PODS; do
      # Copy the checkpoint files only (no .cache/.eval_results), then verify sizes.
      kubectl -n $NS exec "$p" -c ray-worker -- sh -c "
        set -e; mkdir -p $DST
        start=\$(date +%s.%N)
        for f in \$(ls $SRC | grep -v '^\\.'); do cp $SRC/\$f $DST/\$f; done
        end=\$(date +%s.%N)
        for f in \$(ls $SRC | grep -v '^\\.'); do
          [ \$(stat -c %s $SRC/\$f) = \$(stat -c %s $DST/\$f) ] || { echo size-mismatch \$f; exit 1; }
        done
        echo \$(hostname) staged \$(du -sb $DST | cut -f1) bytes in \$(echo \"\$end - \$start\" | bc) s" &
    done
    wait;;
  remove)
    for p in $PODS; do kubectl -n $NS exec "$p" -c ray-worker -- rm -rf $DST & done; wait;;
  status)
    for p in $PODS; do kubectl -n $NS exec "$p" -c ray-worker -- sh -c "echo \$(hostname) \$(du -sh $DST 2>/dev/null | cut -f1) \$(df -h /dev/shm | tail -1)"; done;;
esac
