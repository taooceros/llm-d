# Plan: vLLM TPU Inference with RayDistributedExecutorV2

This document outlines the setup plan and architecture for deploying a **TP8 (Tensor Parallel 8)** inference pipeline on Google Cloud TPUs using vLLM's `RayDistributedExecutorV2` with a custom-allocated placement group.

---

## 1. Scenario Context
* **Hardware**: Two `2x4` TPU node pools (4 TPU VM hosts in total, each hosting 4 TPU chips, yielding 16 chips total).
* **Goal**: Run **TP8** inference (requires exactly 2 TPU hosts / 8 chips).
* **Target Engine**: vLLM V1 with `RayDistributedExecutorV2` backend.
* **Selected Setup**: Decoupled / Wrapper Script setup. This ensures that the engine only allocates resources for exactly 2 TPU nodes (8 chips) out of the 16 available chips in the Ray cluster, avoiding resource bloat and device allocation crashes.

---

## 2. Script Description: `vllm_tpu_wrapper.py`

The wrapper script [`vllm_tpu_wrapper.py`](file:///usr/local/google/home/hongtaozhang/git/admission-control-vllm/vllm_tpu_wrapper.py) manages the Ray placement group creation and server launching.

### Key Features:
1. **Automated Host Calculation**: Intercepts the `--tensor-parallel-size` (and `--pipeline-parallel-size`) arguments to calculate the exact number of TPU hosts needed (e.g. for TP8 on hosts with 4 chips each, it allocates 2 hosts).
2. **Ray Placement Group Pre-allocation**: Before vLLM initializes, the script requests a Ray Placement Group matching the host requirements using the `PACK` strategy. This packs the TPU VM hosts close to each other.
3. **Environment Setup**: Automatically sets required environment variables internally:
   * `TPU_MULTIHOST_BACKEND="ray"`
   * `VLLM_USE_RAY_V2_EXECUTOR_BACKEND="1"`
4. **API Server Actor Encapsulation**: Runs the vLLM OpenAI API Server inside a remote Ray Actor on the first bundle of the placement group. Because the server runs inside the actor, vLLM automatically inherits the custom placement group and avoids allocating the remaining nodes in the cluster.

---

## 3. Implementation Steps

1. **Launch the Wrapper Server**:
   Execute the wrapper script from your workspace. Pass the standard arguments you would normally pass to `vllm serve`:
   ```bash
   ./vllm_tpu_wrapper.py \
       --model "/path/to/your/model" \
       --tensor-parallel-size 8 \
       --pipeline-parallel-size 1 \
       --host 0.0.0.0 \
       --port 8000
   ```

2. **Access the Server**:
   Once initialized, the OpenAI-compatible API server will be available at:
   `http://<host-ip>:8000/v1`

---

## 4. Verification and Debugging

To ensure JAX and Ray are utilizing the nodes correctly:

1. **Verify Ray Resource Allocation**:
   Check if exactly 2 TPU nodes (8 TPU resources) are allocated, leaving the other 2 nodes free.
   ```bash
   ray status
   ray list placement-groups
   ```

2. **Check vLLM Logs**:
   Look for the following log markers confirming the bypassed cluster initialization:
   * `"Using existing placement group..."`
   * `"RayDistributedExecutorV2 | nodes_with_device=2"`
   * `"RayDistributedExecutorV2 | placement_group_specs=[{'TPU': 4.0, ...}, {'TPU': 4.0}]"`

3. **Verify JAX Mesh Initialization**:
   Look at the worker stdout logs (available in Ray's `/tmp/ray/session_latest/logs/` or dashboard). Ensure they show:
   * JAX devices mapping: `8 local/global devices` detected (not 16).
   * No "remote device allocation" or device index errors.
