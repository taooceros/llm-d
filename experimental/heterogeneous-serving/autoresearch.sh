#!/bin/bash
set -euo pipefail

# ==============================================================================
# Autoresearch Canonical Benchmark Harness: Scheduling Policy Evaluation
# Accelerator Target: Cloud TPU v5e (TP=8, KV Capacity: 120,000 tokens)
# Workload: ShareGPT Scale-256 Multi-Batch Trace (4 cohorts of 64 reqs)
# Contract:
#   1. Batch window M=64 requests.
#   2. Reorder only within individual batches of 64 requests.
#   3. Zero preemptions (preemption immunity).
#   4. Earlier batches strictly take priority across batch boundaries.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv-3.12/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

WORKLOAD="${WORKLOAD:-data/workloads/sharegpt_sampled_500.json}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CAPACITY="${CAPACITY:-120000}"
POLICY="${POLICY:-active}"

"$PYTHON_BIN" benchmarks/run_policy_benchmark.py \
    --workload "$WORKLOAD" \
    --batch-size "$BATCH_SIZE" \
    --capacity "$CAPACITY" \
    --policy "$POLICY"
