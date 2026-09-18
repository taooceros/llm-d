# Admission Control & Clairvoyant Oracle Scheduling for vLLM on Cloud TPU

A production-grade Python framework for **clairvoyant oracle scheduling**, **predictive admission control**, and **preemption dynamics simulation** for high-throughput LLM serving with vLLM on **Cloud TPU v5e** clusters.

## Static heterogeneous serving testbed

The execution contract is [HETEROGENEOUS_LLMD_GATEWAY_GOAL.md](HETEROGENEOUS_LLMD_GATEWAY_GOAL.md).
This testbed uses physical TPU v5e execution, `/models/gemma-4-31b`, real
ShareGPT conversations, and disabled prefix caching. Historical simulator
results elsewhere in this repository are not evidence for this evaluation.

On the existing Ray cluster, reuse its recorded physical inventory only while
the Ray node generation matches. `plan` and `preflight` never allocate devices.
`deploy` reconciles explicit node-pinned placement groups, native engines,
logical-instance discovery, and the gateway; identical reapplication attaches
to existing actors rather than replacing their weights.

```bash
./rcm_exec run scripts/hetero_ctl.py preflight --layout M --output /tmp/hetero-preflight.json
./rcm_exec run -d scripts/hetero_ctl.py deploy --layout M --output /tmp/hetero-deploy.json --timeout 5400
./rcm_exec wait --timeout 600
./rcm_exec status
```

The serving configuration now names `kv_cache_dtype="float32"`,
`activation_dtype="float32"`, `matmul_precision="highest"`, the exact native
`backend_revision`, and the serving-package `runtime_revision`; parameter storage
remains `dtype="bfloat16"` by default. These fields participate in attachment
checks and migration build identity. The supported state profile is FP32/highest;
drain incompatible live engines rather than silently reconfiguring them.

`preflight` uses accelerator-free, node-pinned checks of the native sources on
the selected hosts and Ray head. `deploy` applies the hash-pinned
`hetero/native_backend_patch.json` before creating engines: native float32 cache
schema support and Gemma4 embedding/norm activation selection at construction.
Unknown source contents are refused untouched. Replacement is atomic per file,
not across nodes; any partial failure prevents engine startup until every
target verifies ready. Already-patched files are not rewritten. There is no
runtime schema rebuild, function transplant, or compiler-pass override.

M is TP8 + TP8 + TP16; H is two TP16; S is four TP8. A layout change is an
administrative operation after draining, never a change within a workload.
The installed Envoy calls the actual ExtProc v3 gRPC service (service port
9002, target 8201). Its `ImmediateResponse` is assembled by the same gateway
that owns placement and migration. This is not an HTTP-only proxy or an EPP
decision followed by a second scheduler. No direct-backend fallback is used.

```bash
./rcm_exec run scripts/hetero_ctl.py gateway-smoke --layout M --output /tmp/hetero-smoke.json
./rcm_exec run -d scripts/hetero_ctl.py qualify-lifecycle --layout M \
  --output /tmp/hetero-lifecycle.json --timeout 1800 -- \
  --restart-instance small_b --prompts-per-instance 24 --decode-tokens 32
./rcm_exec wait --timeout 600
```

Scoped maintenance fences target admissions, withdraws its discovery entry,
requires native drain/free-page acknowledgment, and promotes only a newer
ready generation. The discovery watcher does not silently rebind engines
during workloads. Peers retain their actors, groups and bindings.
During deployment, both maintenance promotion and ordinary reconciliation retry
discovery while kubelet readiness catches up, under one discovery deadline.
An unready replacement remains excluded; timeout fails the command rather than
advertising partial readiness. Reapplying deployment attaches to resident engines
without reloading their weights.

The retained scoped-restart proof is
`results/hetero/20260916/gateway_scoped_restart_M.json`: eleven invariants
passed across five actual ExtProc batches (120 requests, 3,840 output tokens).
`results/hetero/20260916/integrated_transport_correctness.json` records real
gateway handoffs from both small instances: 14 recompute and six host-KV
migrations. Each diagnostic batch delivered exactly 196,608 useful tokens.
Host-KV moved 15,099,494,400 logical payload bytes with no recompute fallback;
aggregate network traffic was not measured. Other-instance native progress
was observed during every handoff window. These are correctness diagnostics,
not matched-history cost or end-to-end performance comparisons.

**Numerical qualification:** the original BF16-state cohort in
`results/hetero/20260916/continuation_fidelity_M_dispatch_repaired.json` fails
the unchanged cross-TP logprob bound despite exact suffixes and lossless host
transport. Its39 performance runs remain a separate cohort.

The minimal FP32-state diagnostic now passes the complete unchanged gate:
`results/hetero/20260918/minimal_fp32_state_continuation.json`. All15 numerical
comparisons pass (maximum difference0.009636878967285156, complete top32 overlap,
finite values), alongside twelve exact suffixes, two abort resumes and cleanup.
This uses BF16 weights, FP32 activation/KV state, JAX highest, original native
arithmetic and default compiler passes.

The import-fixed permanent implementation is fully qualified under
`results/hetero/evaluation_fp32_state_20260918_v2_source_frozen.json`
(cohort `0f5e4c095b0e…`, source commit `78529499f6da6b`). Its seven raw
receipts are pinned by `results/hetero/20260918/permanent_fp32_v2_qualification_bundle.json`.
Continuation passes all unchanged numerical checks, twelve suffixes, two abort
resumes and cleanup; 104,107 iteration rows retire one-to-one. Integrated Envoy
qualification commits 15 recompute and eight host-KV handoffs. All 18 matched
mechanism trials and nine pairs pass. The earlier first FP32 freeze and its
pre-execution missing-import failure remain preserved, not counted as timing.
The original BF16-state freeze still does not authorize migration.

Local `--protocol`, `--qualification-report`, `--inventory`, and
`--layout-file` inputs are staged byte-exactly with submission hashes. For
mechanism cost, the qualification input is the raw continuation artifact
`report.json`, not the outer control-plane result envelope. After any instance
restart, old generation-specific qualification is not reusable.

Monitor through `rcm_exec`, not Ray job polling. A diagnostic trap preserves
resident state; inspect its ownership before retrying or cancelling. Never
resume a source until the destination abort is acknowledged, and never infer
correct continuation from token counts alone.

### Measurement cohort and report

Every timing run is bound to one source freeze. `scripts/freeze_predeclaration.py
--source-freeze` hashes the committed tree and authorizes arms; `run-arm`
refuses an unauthorized arm, a protocol whose body hash differs, or loaded
source that differs from the frozen tree. Editing a frozen file invalidates the
freeze: commit, re-freeze, and redeploy the engines so loaded code equals frozen
code. Runs from a superseded freeze are quarantined, never compared.

The historical BF16-state cohort is `6ffbc8d3fe92…` (commit `d7dd01a232f8cc`): 12
instance-efficiency runs (H/S on E_SHORT/E_LONG) and 27 end-to-end runs
(H/S/M0 on W0/W1/W2), three paired repetitions each, layout order alternated
so no arm occupies a fixed position. MR and MK-H stay machine-blocked.

The separate FP32-state campaign completed all 102 scheduled arm runs and six
balanced repetitions per declared contrast: 41,608,170 useful output tokens and
2,746,690 retained iteration rows. Every run has exact-generation startup
attribution. The executed plan and driver source are retained at
`results/hetero/campaign-0f5e4c095b0e/execution_plan.json`. Do not append duplicate
trials to a completed cohort; a new experiment needs a new declaration.

Regenerate its report without changing the original39 results:

```bash
python3 scripts/analyze_efficiency.py \
  --protocol results/hetero/evaluation_fp32_state_20260918_v2_source_frozen.json \
  --output results/hetero/reports/fp32_state_20260918_v2/efficiency_analysis.json
python3 scripts/render_report.py \
  --protocol results/hetero/evaluation_fp32_state_20260918_v2_source_frozen.json \
  --output-dir results/hetero/reports/fp32_state_20260918_v2
```

`--protocol` is staged byte-exactly; pass it on every timed run. Runtime uploads
exclude `results/` except the seven hash-pinned qualification wrappers required
by a qualified freeze. Arm evidence is read from
`results/hetero/artifacts/<run-id>/`, which the measurement itself writes; the
`--output` file is only a convenience copy. Do not pipe `rcm_exec` into `head`:
the resulting SIGPIPE can kill the client before it writes that copy.

Historical BF16-state outcome: four TP8 instances beat two TP16 instances on W0, W2 and both
efficiency regions, and lose on W1, so no uniform layout is universally best.
The static TP8+TP8+TP16 mix without migration beats neither on any workload, so
the instance-mix hypothesis is not supported. Migration, transport and system
benefit remain unmeasured in that historical cohort. New FP32-state comparisons
must use their own qualified freeze and report directory. Reports derive expected
pairs from the selected declaration, expose missing runs, and never borrow
historical efficiency or mechanism measurements. Initialization is joined only
to a measured cold deployment of the exact run generations/settings; reused
startup plus serving is labeled as noncontiguous measured components.

**FP32-state findings:** S reduces makespan versus H by a paired median 29.74%
on E_SHORT and 4.50% on E_LONG (the latter below the declared 5% usefulness
threshold). The mixed MR policy is slower than the strongest uniform reference
by 15.42% on W0, 3.63% on W1 and 61.12% on W2. Its 24 enabled runs make zero
migration proposals: the measured active-source free-block minimum is 13/102,
above the frozen 0.12 pressure threshold. This is not evidence that executing
migration cannot help; W1 did not exercise the intended policy condition. No
thresholds were retuned. Host-KV is slower than recompute at all three matched
mechanism checkpoints; the unexercised end-to-end KV arm cannot establish a
transport benefit. Full paired ranges, clock limitations and raw evidence are
in `results/hetero/reports/fp32_state_20260918_v2/REPORT.md`.

The final H/S/M inspections show zero registered/live engines and zero
created/pending owned placement groups. The gateway actor is stopped and its
pool has no labelled endpoints; the existing Ray cluster and Kubernetes
gateway/proxy resources remain. Receipt:
`results/hetero/20260918/permanent_fp32_v2_final_cleanup.json`. For teardown,
drain the layout **while the gateway is alive**, then stop the gateway, so
discovery-label reconciliation runs. The campaign used the reverse order and
its two residual H labels were removed explicitly by retired generation.

### Supported fault envelope

`hetero/gateway/state.py` is the authoritative ownership journal and defines what
the handoff protocol survives. Recoverable: coordinator process loss with the
same journal and lock path and unchanged live engine generations; delayed or
duplicated prepare, commit, release and cancel RPCs; lost acknowledgements. A
torn final append cannot have authorized an RPC and is discarded; complete
malformed records fail closed; write or fsync errors latch uncertainty rather
than permit a rollback.

Not recoverable automatically, by design: loss of the journal directory or its
node, a missing journal beside its lock marker, journal corruption, or replaced
engine generations. Each refuses new traffic and requires an operator to drain
and retire the old fabric before bootstrapping a new ledger. The default /tmp
journal path survives process loss only, not machine replacement. Interrupted
HTTP responses are never reconstructed; cancellation or an idle-proven reset is
required first. The flock is a local writer guard, not a distributed lease, and
native tensors never enter the journal.


---

## 1. Package Architecture

The codebase is organized into a modular, production-grade Python package (`admission_control`):

```
admission-control-vllm/
├── admission_control/               # First-class Python library
│   ├── core/                        # Data models & hardware constraints
│   │   ├── models.py                # Request, HardwareSpec, SchedulePlan, TokenProfile
│   │   └── constants.py             # Physical TPU HBM capacities, static XLA buckets, latencies
│   ├── algorithms/                  # Scheduling algorithms & priority policies
│   │   ├── solver.py                # Asymmetric staggering, 2D water-filling, schedule simulation
│   │   └── policies.py              # Binned LPT, Pure LPT, Density, SPT, Windowed batch slicing
│   ├── simulation/                  # Discrete-event scheduler simulator
│   │   ├── simulator.py             # Step-by-step vLLM v1 scheduler preemption simulator
│   │   └── block_manager.py         # PagedAttention physical block allocator model
│   ├── tpu/                         # Cloud TPU hardware integration
│   │   ├── topology.py              # TPUTopologyDiscovery, ConnectedPair, host mesh coordinates
│   │   └── placement.py             # Explicit node-pinned Ray placement groups (STRICT_SPREAD)
│   ├── serving/                     # Real accelerator execution engine
│   │   ├── worker.py                # TPUPreemptionBenchmarkWorker (hot in-HBM persistent actor)
│   │   ├── orchestrator.py          # Cluster runner & comparative policy coordinator
│   │   └── client.py                # Ray Job Submission SDK client with structured log streaming
│   ├── workloads/                   # Workload ingestion & synthesis
│   │   ├── loader.py                # Empirical trace loader, validator, and registry
│   │   └── generator.py             # Empirical quantile & distribution workload synthesizer
│   └── cli.py                       # Unified command-line interface (`admission-control`)
├── omp_rcm/                         # Reverse Control Mechanism (RCM) SDK
│   ├── core.py                      # StageRunner, StateStore, WatchdogTimer, DiagnosticTrap
│   └── cli.py                       # `rcm_exec` CLI tool
├── data/                            # Canonical workload repository
│   └── workloads/                   # Canonical conversational traces, stress workloads, ShareGPT
├── docs/                            # Documentation & analytical assets
│   ├── architecture/                # TPU scaling plans, precompile analysis, decoupling reports
│   └── visuals/                     # Publication-grade SVG performance diagrams
├── experiments/                     # Experiment runners & backward compatibility shims
│   └── tp8_preemption_benchmark/    # Preemption benchmark scripts, plots, and HTML reports
├── tests/                           # Unit test suite
│   └── test_admission_control.py    # Unit tests for core models, solvers, simulation, and topology
├── autoresearch.sh                  # Autonomous TPU benchmark harness entrypoint
└── pyproject.toml                   # Standard Python packaging configuration
```

---

## 2. Theoretical & Mathematical Foundations

### A. The Physical Memory Cliff on TP=8
On **Cloud TPU v5e** (16 GB HBM per chip, 8 chips in a $2\times 4$ ICI mesh, TP=8), Gemma-4 31B BF16 weights consume **$61.4\text{ GB}$** in HBM, leaving **$\approx 46.8\text{ GB}$** for the physical KV cache pool ($\approx 120,000\text{ tokens}$).

When $N=64$ requests with prompt length $P=256$ and decode limit $D=2048$ arrive concurrently:
- Initial prompt allocation: $64 \times 256 = \mathbf{16,384\text{ tokens}}$ ($13.6\%$ of KV memory).
- Peak unconstrained demand: $64 \times (256 + 2048) = \mathbf{147,456\text{ tokens}} > 120,000\text{ tokens}$.
- Default greedy vLLM admission triggers an eviction cliff at step:
  $$t_{\text{cliff}} = \frac{120,000 - (64 \times 256)}{64} = \mathbf{1,620\text{ steps}}$$
  evicting 16 requests and discarding **$30,016\text{ tokens}$** of recomputation waste.

### B. Memory Deficit Conservation & Asymmetric Staggering
The clairvoyant oracle enforces the deficit conservation invariant across staggered cohorts $(K_1, K_2)$ at offset $t_{\text{offset}}$:
$$K_2 \cdot t_{\text{offset}} \ge K(P + D) - C = \Delta_{\text{mem}}$$
Minimizing makespan $C_{\max} = D + \lceil \Delta_{\text{mem}} / K_2 \rceil$ by maximizing Cohort 2 size ($K_1=16, K_2=48$ @ $t=572$) reduces makespan from $2,906$ steps to **$2,620\text{ steps}$** with **$0\text{ preemptions}$** and **$100\%\text{ goodput efficiency}$**.

### C. 16-Token Binned LPT Priority
Rather than unconstrained knapsack packing that delays long decodes, the **16-Token Binned LPT** priority rule:
$$\text{priority}(i) = \left( -(D_i // 16),\; P_i,\; -D_i \right)$$
pairs short prompts ($P=479$) with long decodes ($D=496$) within the $2,048$-token colocated prefill chunk budget, ensuring decoding begins in Step 1 while strictly bounding decode completion variance to within $16$ steps.

### D. Bounded Window Slicing ($M=64$)
To prevent starvation in long-running workloads, requests are partitioned into sequential windows of $M=64$ requests:
- **Intra-Batch Reordering**: Binned LPT priority is applied strictly *within* each batch of $M=64$ requests (maximum positional displacement $\le 63$).
- **Inter-Batch Overlap**: As requests in batch $k$ finish, slots are continuously backfilled by batch $k+1$, sustaining full $C=64$ stream saturation without inter-wave drain bubbles.

---

## 3. Command-Line Interface (`admission-control`)

The unified CLI provides instant access to solvers, simulators, and cluster topology:

### Solve an Optimal Clairvoyant Schedule
```bash
admission-control solve \
  --workload data/workloads/sharegpt_sampled_500.json \
  --capacity 120000 \
  --compare
```

### Run vLLM Scheduler Preemption Simulation
```bash
admission-control simulate \
  --workload data/workloads/sharegpt_sampled_500.json \
  --capacity 120000
```

### Inspect Cloud TPU Host Placement on Ray
```bash
admission-control topology
```

### Profile Workload Token Length Distributions
```bash
admission-control profile \
  --workload data/workloads/sharegpt_sampled_500.json
```

---

## 4. Running Benchmarks & Tests

### Run Unit Tests
```bash
# Run admission_control test suite
.venv-3.12/bin/python -m unittest discover tests -v

# Run RCM Reverse Control test suite
.venv-3.12/bin/python -m unittest reverse_control_lab/tests/test_suite.py -v
```

### Run Live Autoresearch Benchmark on TPU v5e (TP=8)
```bash
bash autoresearch.sh
```

---

## 5. Python API Usage

```python
from admission_control import (
    HardwareSpec,
    Request,
    solve_schedule,
    VLLMSchedulerSimulator,
    load_workload_requests,
)

# 1. Load production workload
requests = load_workload_requests("data/workloads/sharegpt_sampled_500.json")

# 2. Configure Cloud TPU v5e hardware specifications
hw = HardwareSpec(capacity_tokens=120_000, max_batch_size=128)

# 3. Solve clairvoyant optimal execution plan
plan = solve_schedule(requests, hw, priority="binned_lpt")
print(f"Makespan: {plan.makespan_steps} steps | Expected Goodput: {plan.goodput_tok_s:.1f} tok/s")

# 4. Simulate discrete-event vLLM scheduler preemption dynamics
sim = VLLMSchedulerSimulator(capacity_tokens=120_000)
results = sim.simulate_all_policies([r.to_dict() for r in requests])
for res in results:
    print(f"{res['policy_name']}: {res['total_preemptions']} preemptions, {res['goodput_tok_per_s']:.1f} tok/s")
```
