# Cloud TPU v5e XLA Compiler & HLO Graph Profiling Report

## 1. Executive Hardware & Compiler Overview
- **Target Architecture**: Cloud TPU v5e (8 chips, 2x4 ICI mesh, TP=8) across hosts `192.168.102.18` and `192.168.102.19`
- **Model**: Gemma-4 31B BF16 (61.4 GB weights hot in HBM)
- **JAX Compilation Cache**: `/models/jax_cache` (275 persistent HLO module executables)
- **Vector Memory (Vmem) Allocation**: 112.0 MB per core scratchpad

---

## 2. Deepsea Compiler Pass Latency Breakdown
| Compiler Stage | Duration (ms) | % of End-to-End | Purpose |
| :--- | :---: | :---: | :--- |
| **HLO Passes** | 7.24 ms | 29.7% | Graph optimization, SPMD partitioning, sharding propagation |
| **Backend Passes (MSA)** | 10.36 ms | 42.4% | Memory Space Assignment (Vmem scratchpad) & instruction scheduling |
| **Code Generation** | 4.54 ms | 18.6% | Deepsea native TPU machine code emission |
| **Total End-to-End** | **24.41 ms** | **100.0%** | Single static bucket JIT compilation |

---

## 3. Compiled HLO Module Taxonomy
| HLO Module Category | Compiled Instances | Role in Serving Pipeline |
| :--- | :---: | :--- |
| `jit_run_model` | 158 | Top-level model execution harness across all bucket shapes |
| `jit__uniform` | 22 | XLA lowered kernel execution primitive |
| `jit_dynamic_slice` | 14 | Dynamic KV-cache paged memory indexing |
| `jit__layer_forward` | 11 | Single transformer block forward pass with tensor sharding |
| `jit__swiglu_and_allreduce_impl` | 9 | Feed-forward network (SwiGLU) fused with cross-host AllReduce |
| `jit_structured_decode_fn` | 9 | Single-token autoregressive decode step per batch slot |
| `jit_convert_element_type` | 8 | XLA lowered kernel execution primitive |
| `jit__layer_stack_step` | 7 | Full 30-layer stack decode execution pipeline |
| `jit__prefill_layer_stack_forward` | 5 | Chunked prompt prefill execution over batched tokens |
| `jit__threefry_seed` | 5 | XLA lowered kernel execution primitive |

---

## 4. Kernel Fusion & Collective Communication Mix
- **Kernel Fusion Ratio**: **66.9%** of identified operations are fused, eliminating HBM memory roundtrips for intermediate activation tensors.
- **Cross-Host AllReduce**: 1,086 collective communication instructions operate across the 2x4 ICI mesh links (`192.168.102.18` $\leftrightarrow$ `192.168.102.19`).
- **MXU Matrix Operations**: 806 `dot_general` instructions drive physical matrix multiplication hardware.

---

## 5. Architectural Implications for Admission Control & Oracle Scheduling
1. **Zero-Recompilation Guarantee**:
   - Because vLLM pre-compiles static shapes for $\mathcal{B} = \{16, 32, 64, 128, 256\}$ in `/models/jax_cache`, all 275 HLO modules are hot in disk/HBM cache.
   - Dynamic batch admission within static shapes incurs **0 JIT compilation pauses**.
2. **Continuous Pipelining vs. Recompilation Bubbles**:
   - Continuous slotted admission (`run_policy_f_continuous_slotted_pipeline`) maintains steady-state execution strictly within pre-compiled HLO shapes.
   - Cohort draining incurs an under-utilization penalty where the compiler must transition into lower-throughput BS=16 execution graphs.
3. **Cross-Host Link Efficiency**:
   - Fusing AllReduce directly into SwiGLU and QKV (`jit__swiglu_and_allreduce_impl`, `jit__qkv_and_allreduce_impl`) overlaps cross-host ICI communication latency directly with MXU compute, preventing network stall bubbles.
