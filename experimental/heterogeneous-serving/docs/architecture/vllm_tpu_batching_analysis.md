# Technical Deep-Dive: Request Batching and Admission Control in vLLM v0, vLLM v1, and vLLM-TPU

This report presents a technical breakdown of how **vLLM v0**, **vLLM v1**, and **vLLM-TPU**
evaluate inference requests and determine whether to admit them into an execution batch.

---

## 1. Executive Summary

| Dimension | vLLM v0 Scheduler | vLLM v1 Scheduler | vLLM-TPU / TPU Inference |
| :--- | :--- | :--- | :--- |
| **Architectural Model** | Discrete state machine<br>(`waiting`, `running`, `swapped`) | Continuous token-level<br>scheduler loop | Multi-rank IPC orchestrator<br>wrapping v1 Scheduler per rank |
| **Phase Handling** | Phase-separated (`_schedule_prefills`,<br>`_schedule_running`) or chunked prefill | Unified token scheduling loop<br>(no distinct prefill/decode phase) | DP routing + multi-rank lockstep<br>batching with host lookahead |
| **Budgeting Metric** | `SchedulingBudget`<br>(`max_num_batched_tokens`, `max_num_seqs`) | Unified token budget<br>(`max_num_scheduled_tokens`, `max_num_running_reqs`) | DP-aware token/request load<br>balancing & static tensor padding |
| **Memory Allocation** | `BlockSpaceManager`<br>(`can_allocate`, `can_append_slots`, `can_swap_in`) | `KVCacheManager`<br>(`allocate_slots`, `get_computed_blocks`, watermarks) | Hybrid Attention/Mamba page padding<br>& compact Mamba slot allocation |
| **Preemption** | Re-computation (`WAITING`) or<br>Swap-out to host CPU (`SWAPPED`) | Re-computation (`PREEMPTED` queue)<br>with priority or FCFS policy | Rank-level preemption with<br>AsyncScheduler pre-compensation |
| **Hardware Alignment** | Dynamic batch sizes<br>optimized for GPU execution | Dynamic batch sizes<br>with spec-decode padding | Static shape bucketing<br>(`MIN_NUM_SEQS = 8`), AOT JAX |

---

## 2. vLLM v0 Scheduler Architecture

### 2.1 Queue State Machine
In vLLM v0
([`vllm/core/scheduler.py`](file:///usr/local/google/home/hongtaozhang/git/admission-control-vllm/google3/experimental/users/ranlihao/vllm/core/scheduler.py#L426)),
requests are represented as `SequenceGroup` objects distributed across three queues:
1. `self.waiting`: Deque of newly arrived requests waiting for initial prefill allocation.
2. `self.running`: Deque of active requests currently undergoing prefill or continuous decode steps.
3. `self.swapped`: Deque of requests whose KV cache blocks have been swapped out to host CPU memory due to GPU memory pressure.

### 2.2 Decision Logic & Scheduling Methods
The main entrypoint `Scheduler.schedule()` calls `_schedule()`, which dispatches to
`_schedule_default()` or `_schedule_chunked_prefill()`.

#### A. Default Scheduler (`_schedule_default`)
In `_schedule_default()`:
1. **Running Queue First (`_schedule_running`)**:
   - Iterates through `self.running` requests.
   - Checks if GPU KV cache can append slots for next token via `block_manager.can_append_slots(seq_group)`.
   - Checks token and sequence limits via `SchedulingBudget.can_schedule(num_new_tokens, num_new_seqs)`.
   - If memory allocation fails: invokes `_preempt()` or `_swap_out()`. Preempted requests are removed from `running`.
2. **Swapped Queue Second (`_schedule_swapped`)**:
   - If no requests were preempted from `running`, checks requests in `self.swapped`.
   - Verifies `block_manager.can_swap_in(seq_group)`.
   - If allocation succeeds, blocks are swapped in from CPU to GPU.
3. **Waiting Queue Third (`_schedule_prefills`)**:
   - If no swapped requests were processed and no running requests preempted, evaluates `self.waiting`.
   - Checks `block_manager.can_allocate(seq_group)`, verifying free GPU blocks $\ge$ prompt block count.
   - Evaluates budget: $\text{num_batched_tokens} + \text{prompt_len} \le \text{max_num_batched_tokens}$.

#### B. Chunked Prefill Scheduler (`_schedule_chunked_prefill`)
In `_schedule_chunked_prefill()`:
- Unifies prefill and decode token allocation into a single pass.
- Long prompt prefills are broken into chunks up to `max_num_batched_tokens`.
- Prefill chunks and decode steps co-exist in the same batch up to the `SchedulingBudget` limit.

---

## 3. vLLM v1 Scheduler Architecture

### 3.1 Unified Continuous Token Scheduling Loop
vLLM v1 eliminates the rigid distinction between prefill and decode phases. In
`Scheduler.schedule()`
([`vllm/v1/core/sched/scheduler.py`](file:///usr/local/google/home/hongtaozhang/git/admission-control-vllm/vllm/vllm/v1/core/sched/scheduler.py#L433)),
every request is tracked by its progress towards completion:

$$\text{num_new_tokens} = \text{num_tokens_with_spec} + \text{num_output_placeholders} -
\text{num_computed_tokens}$$

### 3.2 Key Budgeting Parameters & Evaluation Sequence
The scheduling step operates under unified token and request limits:
- `max_num_scheduled_tokens`: Maximum total tokens scheduled in a single engine step.
- `max_num_running_reqs`: Maximum active requests admitted.
- `long_prefill_token_threshold`: Max chunk size for long prefill requests.

#### Step 1: Schedule `running` Requests
1. Calculates `num_new_tokens` for each running request.
2. Caps `num_new_tokens` by `token_budget` and `long_prefill_token_threshold`.
3. Calls `self.kv_cache_manager.allocate_slots(request, num_new_tokens)`.
4. If `allocate_slots` returns `None` (insufficient free blocks):
   - Triggers preemption loop (`_preempt_request`).
   - Evicts lowest priority request (under `PRIORITY` policy) or last request (under `FCFS` policy).

#### Step 2: Schedule `waiting` Requests
1. Checks capacity: $\text{len}(running) + \text{num_waiting_for_streaming} < \text{max_num_running_reqs}$.
2. Evaluates prefix caching hits via `kv_cache_manager.get_computed_blocks(request)`.
3. Computes remaining tokens to schedule: $\text{num_new_tokens} = \text{request.num_tokens} - \text{num_computed_tokens}$.
4. Invokes `allocate_slots` with `watermark_blocks` protection.

---

## 4. vLLM-TPU / TPU Inference Architecture

### 4.1 Data-Parallel Multi-Rank Orchestrator (`DPScheduler`)
`DPScheduler`
([`tpu_inference/core/sched/dp_scheduler.py`](file:///usr/local/google/home/hongtaozhang/git/admission-control-vllm/tpu-inference/tpu_inference/core/sched/dp_scheduler.py#L56))
acts as a distributed control plane over multiple TPU data-parallel ranks. Each DP rank runs an
isolated vLLM v1 `Scheduler` instance inside a dedicated worker process
(`_scheduler_worker_process`) communicating via IPC sockets.

### 4.2 Multi-Tier Request Routing (`_find_best_rank_for_request`)
When a request arrives, `DPScheduler.add_request()` calls `_find_best_rank_for_request()` to select
the target DP rank:

1. **Tier 1: Prefix Cache Hit Probing**:
   - Sends `PROBE_COMPUTED_BLOCKS` command to all DP ranks.
   - If any rank has existing prefix cache blocks, routes to the rank with the highest hit count.
2. **Tier 2: Load Balancing Tuple Sort**:
   - Queries routing metrics from all ranks via `_get_rank_routing_state()`:
     - `pending[rank]`: Pending prefill token count.
     - `inflight[rank]`: Total active requests (`running + waiting`).
     - `min_remaining[rank]`: Smallest remaining output tokens among running requests.
   - Selects rank minimizing the lexicographical tuple:
     $$\text{best_rank} = \arg\min_{r} \Big( \text{pending}[r],\, \text{inflight}[r],\,
\text{min_remaining}[r] \Big)$$

### 4.3 Continue Decode Host-Side Patching
On TPU, multi-step on-device decoding ("Continue Decode") executes up to $N =
\text{max_decode_steps}$ iterations per host step. To prevent host-side scheduler out-of-sync
errors, `patch_vllm_scheduler_for_continue_decode()` applies three monkey-patches:
1. **KV Cache Lookahead Allocation**: Sets `num_lookahead_tokens = max_decode_steps - 1` in `Scheduler.__init__`, ensuring `KVCacheManager` reserves enough KV blocks for all $N$ tokens ahead of time.
2. **Host Progress Reconciliation**: Advances `num_computed_tokens` by $N - 1$ extra tokens when model outputs return.
3. **Async Placeholder Compensation**: Pre-compensates `num_output_placeholders` by adding $(N-1)$ before subtracting $N$, preventing negative underflow.

### 4.4 Static Bucketing, Tensor Padding, & AOT JAX Compilation
TPUs require fixed tensor shapes to avoid costly XLA re-compilations at runtime:
1. **Sequence Bucketing (`MIN_NUM_SEQS = 8`)**:
   $$\text{padded_reqs} = \begin{cases} 8 & \text{if } x \le 8 \\ 2^{\lceil \log_2 x \rceil} &
\text{if } x > 8 \end{cases}$$
2. **Token Padding (`get_token_paddings`)**:
   - Generates power-of-2 or linear gap padding steps up to `max_token_size`.
3. **AOT Compilation (`CompilationManager`)**:
   - Pre-compiles JAX model graphs for all valid combinations of `(padded_num_reqs, padded_token_len)` before inference starts.

### 4.5 Hybrid Attention & Mamba KV Cache Allocation
For hybrid Attention + Mamba models:
- **Uniform Page Size Padding**: `update_mamba_page_size_padded` standardizes byte sizes across Attention and Mamba groups so vLLM calculates uniform block counts.
- **Compact Mamba Allocation**: Mamba recurrent state is allocated based on active request slots (`_mamba_num_blocks`) rather than full attention sequence history, conserving TPU HBM.

---

## 5. Comprehensive Comparative Summary Matrix

| Component / Feature | vLLM v0 Scheduler | vLLM v1 Scheduler | vLLM-TPU Inference |
| :--- | :--- | :--- | :--- |
| **Primary Code Path** | `google3/experimental/users/`<br>`ranlihao/vllm/core/scheduler.py` | `vllm/v1/core/sched/scheduler.py` | `tpu_inference/core/sched/dp_scheduler.py` |
| **Execution Loop** | Discrete `_schedule()` phases<br>(`running` $\rightarrow$ `swapped` $\rightarrow$ `prefills`) | Single unified `schedule()` token<br>allocation loop | `DPScheduler` IPC dispatcher<br>$\rightarrow$ per-rank v1 `schedule()` |
| **Token Budgeting** | `SchedulingBudget` (`max_num_batched_tokens`) | `max_num_scheduled_tokens`<br>(unified prefill/decode) | Ranked DP balancing +<br>static shape bucketing |
| **Request Limit** | `max_num_seqs` | `max_num_running_reqs` | `max_num_running_reqs` padded<br>to `MIN_NUM_SEQS = 8` |
| **Memory Manager** | `BlockSpaceManager` | `KVCacheManager` | TPU `KVCacheManager`<br>(Hybrid Mamba compact sizing) |
| **Prefix Caching** | Optional block-level hashing | Integrated hit check<br>(`get_computed_blocks`) | Multi-rank IPC query<br>(`PROBE_COMPUTED_BLOCKS`) |
| **Multi-Step Execution** | 1 step per forward pass | 1 step per forward pass | Continue Decode ($N$ steps<br>on TPU via lookahead slots) |
| **Preemption Mechanism** | Swap-out to CPU swap or recompute | Evict to `PREEMPTED` queue<br>(Priority/FCFS) | Per-rank preemption with<br>placeholder pre-compensation |
| **Tensor Padding** | Variable dynamic batch shapes | Variable dynamic batch shapes | Mandatory static padding<br>(`get_token_paddings`, AOT JAX) |


---

## 📎 Linked Documents & Notebook Files

- 📓 **Interactive Jupyter Notebook (`.ipynb`)**: [vllm_tpu_batching_analysis.ipynb](file:///usr/local/google/home/hongtaozhang/git/admission-control-vllm/vllm_tpu_batching_analysis.ipynb)
- 📄 **HTML Version**: [vllm_tpu_batching_analysis_doc.html](./vllm_tpu_batching_analysis_doc.html)
- 🚀 **Open directly in Google Colab**: [Upload to Google Colab](https://colab.research.google.com/#create=true)

---