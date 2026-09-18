# Handoff Note: Offline vLLM Batch Inference on $2 	imes 4$ TPU Topology ($	ext{TP}=8$)

## 1. Overview & Objective
This handoff document provides instructions for executing **offline vLLM batch inference** across a **2-host (8-chip) TPU subslice** configured as a **$2 	imes 4$ / $4 	imes 2$ TPU ICI Mesh Topology** with Tensor Parallelism degree $	ext{TP}=8$.

---

## 2. Hardware Topology & Host Pairing

On a 4-host Cloud TPU v5e/v6e slice (`4x4` torus / 16 chips total):
- **4 physical worker hosts** (4 TPU chips per host = 16 chips total).
- The hosts form a $2 	imes 2$ grid:
  - **Worker 0** (`192.168.102.18`): Grid $(0, 0)$, Chips $X \in [0..1], Y \in [0..1]$
  - **Worker 1** (`192.168.102.19`): Grid $(1, 0)$, Chips $X \in [2..3], Y \in [0..1]$
  - **Worker 2** (`192.168.102.17`): Grid $(0, 1)$, Chips $X \in [0..1], Y \in [2..3]$
  - **Worker 3** (`192.168.102.16`): Grid $(1, 1)$, Chips $X \in [2..3], Y \in [2..3]$

### **Valid Connected 2-Host Pairs (Manhattan Distance = 1):**
| Pair Option | Worker IDs | Node IPs | Mesh Subslice Orientation |
| :--- | :--- | :--- | :--- |
| **Option 1** | **Worker 0 + Worker 1** | `192.168.102.18` + `192.168.102.19` | **Dimension X ($4 	imes 2$ sub-mesh)** |
| **Option 2** | **Worker 0 + Worker 2** | `192.168.102.18` + `192.168.102.17` | **Dimension Y ($2 	imes 4$ sub-mesh)** |
| **Option 3** | **Worker 1 + Worker 3** | `192.168.102.19` + `192.168.102.16` | **Dimension Y ($2 	imes 4$ sub-mesh)** |
| **Option 4** | **Worker 2 + Worker 3** | `192.168.102.17` + `192.168.102.16` | **Dimension X ($4 	imes 2$ sub-mesh)** |

*(Note: Diagonal pairs `(0, 3)` or `(1, 2)` cannot form contiguous ICI sub-meshes).*

---

## 3. Tooling & Setup

1. **Topology Discovery Module (`pkg_util/tpu_topology.py`)**:
   - Discovers Ray worker nodes, computes grid coordinates, and automatically picks a connected 2-host subslice.
   - Verify topology mapping via CLI:
     ```bash
     python3 pkg_util/tpu_topology.py
     ```

2. **Offline Inference Script (`demos/run_vllm_subslice_offline.py`)**:
   - Sets required multihost environment variables (`TPU_MULTIHOST_BACKEND=ray`, `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`).
   - Automatically selects a connected 2-host pair.
   - Allocates a Ray Placement Group with `STRICT_SPREAD` across the two host IPs.
   - Initializes the vLLM engine with $	ext{TP}=8$ and runs batch inference.

---

## 4. Execution Steps

### **Step 1: Set Up Port Forwarding (if running outside the cluster)**
Forward Ray GCS and client ports from the TPU Ray head pod:
```bash
kubectl port-forward -n llm-d-optimized-baseline pod/tpu-ray-cluster-vllm-tpu-head-vr462 6379:6379 10001:10001 8265:8265 &
```

### **Step 2: Run Offline vLLM Batch Inference**
Execute the offline inference runner:
```bash
python3 demos/run_vllm_subslice_offline.py
```

---

## 5. Key Verification Parameters

- **Target Model:** `google/gemma-2-9b` (or target Gemma / LLaMA model)
- **Tensor Parallelism Degree:** $	ext{TP}=8$
- **Total Active TPU Chips:** 8 TPU v5e chips (2 hosts $	imes$ 4 chips)
- **Placement Group Strategy:** `STRICT_SPREAD` with bundle specs:
  ```python
  [
      {"TPU": 4, "node:<HOST_A_IP>": 1},
      {"TPU": 4, "node:<HOST_B_IP>": 1},
  ]
  ```
