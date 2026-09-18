# Physical Performance and Memory Bandwidth Analysis of TPU PagedAttention Across Concurrency Scaling and Topologies

## 1. Overview and Core Motivation

In distributed large language model (LLM) serving systems such as vLLM, inference execution is partitioned into two distinct physical phases:
1. **Prefill Phase (Prompt Processing)**: Dominated by dense GEMM matrix multiplications, representing a compute-bound workload with high arithmetic intensity.
2. **Decode Phase (Token Generation)**: Each active request contributes only 1 Query token per step, but must scan its entire accumulated historical Key-Value (KV) cache. The operational intensity of this step is typically around $\approx 1.0\text{ FLOP / Byte}$, making it an extreme memory-bandwidth-bound workload.

**PagedAttention** allocates KV cache into fixed-size physical pages (e.g., `page_size=16`), eliminating external memory fragmentation. However, **concurrent batch size ($B$) exerts a dual-edged effect on PagedAttention hardware execution**:
- **Memory Bus Underutilization at Low Concurrency**: When $B$ is small, the volume of KV data read per step is insufficient to leverage the high burst-access efficiency of TPU HBM, leading to low Memory Bandwidth Utilization (MBU).
- **Memory Bus Saturation and Latency Scaling at High Concurrency**: As $B$ increases, HBM read bandwidth saturates. Beyond the saturation point, per-step latency degrades linearly with $B \times L$, increasing Time Per Output Token (TPOT).
- **Non-Contiguous Memory Access and Page Table Overhead**: Large batches with interleaved, non-contiguous physical page indices introduce memory gather overhead and potential HBM channel access contention on Cloud TPU v5e.

This study systematically characterizes the physical hardware latency, generation throughput, and effective Memory Bandwidth Utilization (MBU) of the PagedAttention kernel on Google Cloud TPU v5e clusters using the **Gemma-4 31B** architecture. We evaluate batch sizes $B \in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]$ across context lengths $L \in [256, 512, 1024, 2048, 4096, 8192]$, quantifying the impact of Grouped Query Attention (GQA) head sharding across **$2\times 4$ (TP=8)** and **$4\times 4$ (TP=16)** physical interconnect topologies.

---

## 2. Hardware Topologies and GQA Head Sharding Model

### 2.1 Physical Cluster Environment
- **Hardware Platform**: Google Cloud TPU v5e (TPU-v5litepod-16, 4 physical hosts, 16 TPU v5e chips).
- **Per-Chip Hardware Specifications**:
  - HBM Capacity: 16 GB
  - Peak HBM Memory Bandwidth: $819\text{ GB/s}$
  - On-Chip Vector Memory (VMEM): 16 MB
  - On-Chip Scalar Memory (SMEM): 1.00 MB (1,048,576 Bytes)
  - Peak Matrix Compute (BF16): 197 TFLOPS

### 2.2 Gemma-4 31B GQA Architecture & Tensor Parallelism Sharding
Gemma-4 31B utilizes **Grouped Query Attention (GQA)** to reduce KV cache memory footprints:
- Hidden Dimension: $H = 5376$
- Query Heads: $N_q = 32$
- Key/Value Heads: $N_{kv} = 16$ (Group Size = 2 Query heads per KV head)
- Head Dimension: $d = 128$
- Layer Count: $N_{\text{layers}} = 48$

Under different Tensor Parallelism (TP) configurations, the number of KV heads assigned to each TPU chip differs substantially:

| Topology | TP Degree | Query Heads / Chip | KV Heads / Chip | KV Cache Bytes / Token / Chip |
| :--- | :--- | :--- | :--- | :--- |
| **$2\times 4$ (Subslice)** | $\text{TP}=8$ | $32 / 8 = \mathbf{4}$ heads | $16 / 8 = \mathbf{2}$ heads | $2 \times 2 \times 128 \times 2 = 1,024\text{ Bytes}$ |
| **$4\times 4$ (Full Pod)** | $\text{TP}=16$ | $32 / 16 = \mathbf{2}$ heads | $16 / 16 = \mathbf{1}$ head | $1 \times 2 \times 128 \times 2 = 512\text{ Bytes}$ |

> **Key Architectural Insight**: In the $4\times 4$ (TP=16) topology, each TPU chip hosts **only 1 individual KV head**. At small batch sizes ($B=1$), the memory stream per step is narrow, underutilizing HBM burst access. Consequently, $4\times 4$ exhibits higher sensitivity to batch size aggregation than $2\times 4$.

---

## 3. Mathematical Formulations and Evaluation Metrics

### 3.1 Single-Step PagedAttention Data Volume
In a single decode step for batch size $B$ and average context length $L$, the model must read all historical Key and Value cache tensors (BF16 format, 2 bytes per scalar):

$$\text{Bytes}_{\text{KV, global}}(B, L) = 2 \times B \times L \times N_{kv} \times d \times 2 = 4 \cdot B \cdot L \cdot N_{kv} \cdot d$$

Substituting Gemma-4 31B architectural constants ($N_{kv}=16, d=128$):
$$\text{Bytes}_{\text{KV, global}}(B, L) = 4 \cdot B \cdot L \cdot 16 \cdot 128 = 8,192 \cdot B \cdot L \quad (\text{Bytes})$$

Under Tensor Parallelism $\text{TP}$, the KV heads are uniformly partitioned across devices. **The per-chip HBM read volume per transformer layer per step** is:
$$\text{Bytes}_{\text{per\_chip}}(B, L) = \frac{8,192 \cdot B \cdot L}{\text{TP}} \quad (\text{Bytes})$$

For the full 48-layer model ($N_{\text{layers}}=48$), total per-chip HBM traffic per step is $48 \times \text{Bytes}_{\text{per\_chip}}(B, L)$.

### 3.2 Effective Memory Bandwidth
Let $T_{\text{attn}}(B, L)$ (in seconds) denote the physically measured execution latency of a single PagedAttention step. The effective memory read bandwidth achieved on hardware is:

$$\text{BW}_{\text{eff}}(B, L) = \frac{\text{Bytes}_{\text{per\_chip}}(B, L)}{T_{\text{attn}}(B, L)} \quad (\text{GB/s})$$

### 3.3 Memory Bandwidth Utilization (MBU)
MBU quantifies how close the kernel operates to the hardware HBM interface limit:

$$\text{MBU}(B, L) = \frac{\text{BW}_{\text{eff}}(B, L)}{\text{BW}_{\text{peak}}} \times 100\%$$

For TPU v5e, $\text{BW}_{\text{peak}} = 819\text{ GB/s}$. In non-contiguous block-table gather workloads, achieving $50\% \sim 60\%$ MBU indicates near-complete saturation of physical memory channels.

### 3.4 Generation Throughput & Per-Token Latency (TPOT)
- **Cluster Decode Throughput**:
  $$\text{Throughput}(B, L) = \frac{B}{T_{\text{step}}(B, L)} \quad (\text{tokens / second})$$
- **Time Per Output Token (TPOT)**:
  $$\text{TPOT}(B, L) = T_{\text{step}}(B, L) \quad (\text{ms / token})$$

---

## 4. Experimental Setup and Execution Protocol

### 4.1 Target Kernel
- Benchmarks execute the native optimized Cloud TPU kernel in vLLM:
  `tpu_inference.kernels.ragged_paged_attention.v3.kernel.ragged_paged_attention`
- Paired with `get_kv_cache_shape` to construct realistic physical HBM page pools (`page_size=16`, BF16).

### 4.2 Parameter Sweep Space
1. **Concurrent Batch Size ($B$)**:
   $$B \in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]$$
2. **Historical Context Length ($L$)**:
   $$L \in [256, 512, 1024, 2048, 4096, 8192]$$
3. **Hardware Topologies**:
   - $2\times 4$ (TP=8, spanning 2 physical hosts)
   - $4\times 4$ (TP=16, spanning 4 physical hosts)

### 4.3 Output Metrics
- P50, P90, and mean step latency (ms).
- Equivalent full-model (48-layer) decode step latency (ms).
- Effective memory read bandwidth ($\text{GB/s}$) and MBU (%).
- Aggregate cluster decode throughput ($\text{tok/s}$) and TPOT ($\text{ms/tok}$).
- Memory bus saturation thresholds and compiler resource exhaustion boundaries.

---

## 5. Reverse Control Mechanism (RCM) Execution Protocol

1. **Execution Entry**: Workloads are orchestrated via the Reverse Control Mechanism:
   ```bash
   python3 -m omp_rcm run benchmarks/run_paged_attention_matrix_empirical.py
   ```
2. **Placement Group Node Pinning**: Bundles are bound to verified physical TPU host IPs using `strategy="STRICT_SPREAD"`.
3. **Structured Telemetry Protocol**:
   - Stage progression: `[OMP_EVENT: STAGE_ENTERED stage=...]`
   - Cell measurement: `[OMP_EVENT: BATCH_ATTENTION_MEASURED batch=... length=... mbu=...]`
   - Milestone completion: `[OMP_EVENT: BENCHMARK_SUCCESS json=results/paged_attention_matrix_empirical.json]`

---

## 6. Physical Hardware Ground Truth & Architectural Insights

All metrics are gathered from physical Cloud TPU v5e cluster executions (`results/paged_attention_batch_size.json` and `results/paged_attention_matrix_empirical.json`).

### 6.1 Batch Size Scaling Baseline (Fixed Context Length $L = 1024$)

| Batch Size ($B$) | $2\times 4$ (TP=8) Layer (ms) | $2\times 4$ MBU % | $4\times 4$ (TP=16) Layer (ms) | $4\times 4$ MBU % | Speedup ($4\times 4$ vs $2\times 4$) | $4\times 4$ 48-Layer Step (ms) | $4\times 4$ Decode Throughput (tok/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 0.3599 ms | 0.4% | 0.3672 ms | 0.2% | 0.980x | 17.63 ms | 56.7 |
| **2** | 0.3280 ms | 0.8% | 0.3475 ms | 0.4% | 0.944x | 16.68 ms | 119.9 |
| **4** | 0.3395 ms | 1.5% | 0.3243 ms | 0.8% | 1.047x | 15.57 ms | 257.0 |
| **8** | 0.3413 ms | 3.0% | 0.3180 ms | 1.6% | 1.073x | 15.26 ms | 524.2 |
| **16** | 0.3839 ms | 5.3% | 0.3496 ms | 2.9% | 1.098x | 16.78 ms | 953.5 |
| **32** | 0.3687 ms | 11.1% | 0.3622 ms | 5.7% | 1.018x | 17.39 ms | 1,840.6 |
| **64** | 0.4390 ms | 18.7% | 0.4259 ms | 9.6% | 1.031x | 20.44 ms | 3,130.8 |
| **128** | 0.5701 ms | 28.8% | 0.4782 ms | 17.1% | 1.192x | 22.95 ms | 5,576.4 |
| **192** | 0.6742 ms | 36.5% | 0.5763 ms | 21.3% | 1.170x | 27.66 ms | 6,940.8 |
| **256** | 0.7733 ms | 42.4% | 0.6486 ms | 25.3% | 1.192x | 31.13 ms | 8,222.3 |
| **384** | 0.9577 ms | 51.3% | 0.7875 ms | 31.2% | **1.216x** | 37.80 ms | **10,159.1** |
| **512** | 1.1298 ms | **58.0%** | 0.9242 ms | 35.5% | **1.222x** | 44.36 ms | **11,541.9** |

### 6.2 Context Length Scaling Baseline (Fixed Batch Size $B = 32$)

| Context Length ($L$) | $2\times 4$ Layer (ms) | $2\times 4$ 48-Layer (ms) | $4\times 4$ Layer (ms) | $4\times 4$ 48-Layer (ms) | Speedup ($4\times 4$ vs $2\times 4$) | $4\times 4$ Effective BW |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **256** | 0.3626 ms | 17.40 ms | 0.3703 ms | 17.77 ms | 0.979x | 20.3 GB/s (2.5% MBU) |
| **512** | 0.3603 ms | 17.29 ms | 0.3444 ms | 16.53 ms | 1.046x | 43.6 GB/s (5.3% MBU) |
| **1024** | 0.3764 ms | 18.07 ms | 0.3681 ms | 17.67 ms | 1.023x | 82.2 GB/s (10.0% MBU) |
| **2048** | 0.4239 ms | 20.35 ms | 0.4083 ms | 19.60 ms | 1.038x | 147.2 GB/s (18.0% MBU) |
| **4096** | 0.5636 ms | 27.05 ms | 0.4736 ms | 22.73 ms | **1.190x** | 141.7 GB/s (17.3% MBU) |

### 6.3 Core Physical Discoveries and Scheduler Guidance

1. **Free Throughput Zone ($B \le 32$)**:
   In the range $B \in [1, 32]$, single-step latency remains almost flat between $0.32 \sim 0.38\text{ ms}$ per layer ($15 \sim 18\text{ ms}$ for the 48-layer model), with MBU below 11%. Batching requests up to 32 imposes virtually zero penalty on TPOT, while cluster decode throughput scales from $56.7\text{ tok/s}$ to **$1,840.6\text{ tok/s}$** (a $32\times$ throughput increase).
   - **Scheduling Policy**: Lightly loaded serving instances should aggressively batch requests up to $B=32$ without concern for latency degradation.
2. **Bandwidth Saturation Wall ($B \ge 64$)**:
   Beyond $B=64$, per-step latency begins a clear upward inflection (rising from 0.36 ms to 0.55 ms at $B=128$, a 52% latency penalty). On $2\times 4$, per-chip effective memory bandwidth reaches $245.2\text{ GB/s}$ (30% MBU).
   - **Scheduling Policy**: Interactive workloads with strict TPOT SLA targets should cap decode batch size at 64. Offline batch inference workloads can push batch sizes to 128 or higher to maximize throughput.
3. **GQA Fragmentation Penalty at Low Concurrency**:
   At $B \in [1, 2]$, $4\times 4$ (TP=16) is $2\% \sim 5\%$ slower than $2\times 4$ (TP=8). Gemma-4 31B possesses 16 KV heads; under TP=16, each TPU chip holds only 1 KV head, resulting in inefficient HBM burst transfers when $B$ is minimal.
   - **Scheduling Policy**: Single-stream low-concurrency workloads should be deployed on smaller TP topologies ($2\times 4$ or single-host).
4. **Long-Context Dividend on $4\times 4$**:
   At $L=4096$, the $4\times 4$ full-model step latency is 22.73 ms compared to 27.05 ms on $2\times 4$, saving **4.32 ms per token** ($1.19\times$ speedup). Distributing the KV cache across 16 chips delays HBM bandwidth saturation under long context lengths.
5. **Ultra-High Concurrency ($B \ge 256$) Bandwidth Relief on $4\times 4$**:
   At $B=512$, $2\times 4$ reaches **$475.2\text{ GB/s}$ per chip (58.0% MBU)**, approaching the physical limit of non-contiguous gather operations, causing per-layer latency to degrade to 1.13 ms. By halving per-chip KV memory traffic, $4\times 4$ operates at 35.5% MBU and 0.92 ms per layer ($1.222\times$ speedup), exceeding **11,540 tok/s** aggregate decode throughput.

---

### 6.4 Expanded 60-Cell Empirical Matrix and SMEM Hardware Boundary

The expanded matrix spans $B \in [1, 8, 16, 32, 64, 128, 256, 512, 768, 1024]$ across $L \in [256, 512, 1024, 2048, 4096, 8192]$ on physical hardware.

#### 1. Full-Model 48-Layer Step Latency Matrix (TPOT in ms: $2\times 4$ / $4\times 4$)
```
Context (L)  | B=1         | B=8         | B=16        | B=32        | B=64        | B=128       | B=256       | B=512       | B=768       | B=1024     
--------------------------------------------------------------------------------------------------------------------------------------------------
L=8192       | 17.8/19.5   | 22.9/22.1   | 30.1/23.5   | 42.7/30.7   | 61.3/42.4   | 105.9/66.5  | 191.7/113.8 | SMEM_OOM    | SMEM_OOM    | SMEM_OOM   
L=4096       | 17.3/15.6   | 19.9/18.1   | 22.2/19.6   | 28.4/23.3   | 40.1/29.8   | 55.6/42.4   | 97.4/63.9   | 174.0/109.2 | 249.9/158.9 | SMEM_OOM   
L=2048       | 16.7/16.2   | 18.1/19.1   | 19.1/17.2   | 21.4/19.2   | 25.8/21.5   | 35.9/28.4   | 52.9/38.9   | 89.4/59.6   | 125.0/82.0  | 161.1/104.8
L=1024       | 16.2/16.6   | 17.4/17.0   | 17.2/18.5   | 19.0/18.4   | 22.2/20.6   | 26.1/23.1   | 35.5/30.3   | 52.9/44.3   | 70.9/56.3   | 89.3/69.2  
L=512        | 16.7/17.1   | 18.8/17.4   | 16.8/18.1   | 17.6/19.3   | 19.4/19.6   | 23.3/21.5   | 28.0/26.8   | 38.4/36.6   | 51.4/44.6   | 61.8/52.9  
L=256        | 18.8/18.7   | 18.3/18.3   | 18.2/17.7   | 18.9/17.4   | 19.7/19.4   | 22.5/20.1   | 27.0/26.8   | 37.4/35.7   | 43.0/41.6   | 51.9/49.6  
```

#### 2. Cluster Aggregate Decode Throughput Matrix (tokens/sec: $2\times 4$ / $4\times 4$)
```
Context (L)  | B=1         | B=8         | B=16        | B=32        | B=64        | B=128       | B=256       | B=512        | B=768        | B=1024      
---------------------------------------------------------------------------------------------------------------------------------------------------
L=8192       | 56/51       | 349/361     | 531/680     | 748/1041    | 1043/1509   | 1208/1924   | 1335/2249   | SMEM_OOM     | SMEM_OOM     | SMEM_OOM    
L=4096       | 57/64       | 402/441     | 719/817     | 1128/1372   | 1596/2149   | 2300/3020   | 2627/4007   | 2941/4691    | 3072/4833    | SMEM_OOM    
L=2048       | 59/61       | 443/419     | 839/929     | 1495/1668   | 2478/2978   | 3561/4508   | 4838/6587   | 5726/8591    | 6144/9359    | 6356/9775   
L=1024       | 61/60       | 460/471     | 931/864     | 1687/1740   | 2881/3106   | 4899/5552   | 7204/8453   | 9681/11559   | 10838/13642  | 11467/14801 
L=512        | 60/58       | 426/460     | 954/885     | 1813/1656   | 3289/3263   | 5488/5941   | 9138/9539   | 13333/13975  | 14955/17228  | 16570/19344 
L=256        | 53/53       | 437/437     | 877/905     | 1691/1838   | 3253/3308   | 5680/6356   | 9487/9562   | 13695/14355  | 17852/18469  | 19723/20629 
```

#### 3. Architectural Boundary Discovery: TPU v5e Scalar Memory (SMEM) 1.0 MB Limit
- **Not an HBM Capacity Issue**: At $L=8192, B=512$, per-chip KV allocation requires only $\approx 2.15\text{ GB}$, well within the 16 GB HBM capacity.
- **SMEM (Scalar Memory) Hardware Ceiling**:
  Each TPU v5e core contains **1.00 MB (1,048,576 Bytes)** of on-chip Scalar Memory (SMEM). In the Pallas PagedAttention kernel implementation (`RPAd`), the full physical page index table `page_indices` (4 bytes per int32 entry) is prefetched into SMEM to service non-contiguous page lookups without scalar bus stalls.
  For batch size $B$ and context length $L$ with page size 16:
  $$\text{Bytes}_{\text{SMEM}} \approx B \times \frac{L}{16} \times 4 = \frac{B \times L}{4}\text{ Bytes}$$
  When $\text{Bytes}_{\text{SMEM}}$ plus kernel scalar descriptor overhead exceeds 1.0 MB (occurring at $B \times L > 3.6 \times 10^6\text{ tokens}$), the XLA compiler fails with:
  `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space smem. Used 1.01M of 1.00M smem.`
- **Admission Control Invariant**:
  Admission control policies and batch schedulers must enforce a hard token budget constraint:
  $$\sum_{i=1}^B L_i \le 3.5 \times 10^6\text{ tokens}$$
  to guarantee execution remains within the physical 1.0 MB SMEM boundary.

### 6.5 Empirical Matrix Heatmap Visualization

The complete 60-cell empirical matrix heatmap is rendered at:
![PagedAttention Empirical Heatmap](../../visuals/paged_attention_matrix_heatmap.svg)
- **Panel A**: Full-model step latency (TPOT ms/token) heatmap across $B$ and $L$ with $4\times 4$ vs $2\times 4$ speedup multipliers and red-hatched `SMEM OOM` cells.
- **Panel B**: Aggregate cluster decode throughput (tokens/s) reaching peak **$20,629\text{ tok/s}$** at $(L=256, B=1024)$.
