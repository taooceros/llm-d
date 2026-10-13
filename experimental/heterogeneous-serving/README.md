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

### Explicit host-RAM weight snapshots

`EngineSettings.weight_snapshot` defaults to `"off"`. For a ready, idle engine,
its `snapshot_weights()` actor RPC publishes each worker's final runner-state
arrays under `/dev/shm/omp_weights/<layout>/<instance>/rank<process-rank>/`.
Alternatively, `"weight_snapshot": "write"` snapshots immediately after the
normal load and precision checks. Neither mode changes the serving step path.

Redeploy with `--engine-settings '{"weight_snapshot":"read"}'` (or the same
field in `--instance-engine-settings`) to construct the abstract native nnx
model and restore final shards before the runner captures its execution
closures. No checkpoint weight iterator or random-weight allocation runs on
the successful restore path. By default every shard is checksummed before placement;
model identity, TP size, weight dtype/quantization, backend revision, physical
host/chip assignment, ordered JAX mesh, parameter paths/shapes/dtypes and shard
indices must agree. A worker refusal is voted across the instance before all
workers fall back to the normal loader; `worker_init_phases()` exposes the
refusal reason and `weight_snapshot.restored` so fallback timings cannot be
mistaken for restore timings.

The model fingerprint hashes configuration contents plus checkpoint file
sizes and modification times, **not all checkpoint contents**: checkpoints
must remain immutable (rewriting bytes while preserving size and mtime is
outside this identity contract). Snapshot bytes themselves use SHA-256.
Quantized/draft abstract graphs currently refuse and use the normal loader.
`max_model_len` is deliberately absent from weight compatibility.
For an explicit checksum-cost A/B only, `"weight_snapshot_verify": false`
skips byte hashes but retains all compatibility and shard-shape checks.
Worker evidence records `verified_checksums`; this is not equivalent to a
verified restore. The campaign tries this variant only when SHA verification
takes at least 15 seconds and still requires exact greedy-output equality.
`snapshot_verify` includes reading/faulting mapped tmpfs bytes; mapping setup
and device placement have separate phase clocks.

Run the S/TP8 and H/TP16 validation campaign from an isolated worktree with
`HEAD_POD` and `KUBECONFIG` set:

```bash
flock /tmp/hetero-cluster.lock scripts/reconfig_snapshot.sh
.venv/bin/python scripts/summarize_snapshot.py --root results/restore/ShardedRestore
XLA_FLAGS=--xla_force_host_platform_device_count=8 .venv/bin/python -m pytest -q tests/test_weight_snapshot.py
```

### Opt-in next-layout compile overlap experiments

`startup_profile` defaults to `false`. When enabled it records JAX monitoring
events, actual persistent-cache hits/misses and cache keys, per-bucket AOT and
synchronized warm-up intervals, and KV allocation intervals in
`worker_init_phases()[*].startup_profiles`. Parallel intervals are unioned into
exclusive wall-clock categories; they must not be summed as serial costs.
Cache-hit lookup/decompression and executable deserialize/load are separate.
On a miss JAX exposes indivisible `compile_and_load`, so this category is not
proof of pure host compilation. Warm-up/allocation spans include host dispatch
and synchronization, with nested compilation subtracted. All hooks are removed
before serving; disabled engines do not install them.
`startup_device_trace` also defaults to `false`. Dedicated split runs can enable
it to capture actual TPU `XLA Modules` device lanes, retain every module span
used for timing, and separate device-module execution from host compile/load
clocks. Nested XLA-op lanes and host events are not counted as device module
time. Missing device lanes/phase annotations yield `null`, not guessed zero.
Profiler setup/export/parse/cleanup costs are recorded outside the phase clock
and still belong in deployment elapsed time. Matched switch-C and serving-impact
runs leave this diagnostic flag off. Private local trace directories are removed
after their timing evidence is retained in the per-worker profile.

The published warm-up dependency (`8057ab8`, `2a3dd37`) supplies `worker_env`
and direct JAX configuration after its import-time environment snapshot. Cache
settings are verified after device initialization and before startup phases;
an unexpected backend override fails instead of being silently reapplied.
`worker_env` is trusted deployment configuration, recorded in evidence: do not
put secrets in it or expose it to inference clients. The explicit
`compilation_cache_dir` setting takes precedence over cache paths in this map.

`startup_compile_bundle` defaults to `null`. Setting it to a unique experiment
directory captures process-0 startup modules, exact compiler options, ordered
devices and original cache keys. This is **prior target graph provisioning**,
not free first-ever next-layout discovery. Its normal-deploy cost must remain
in the experiment evidence; an already populated normal cache is the warm
control. No cache key is rewritten to force a hit.

`scripts/overlap_pilot.py --background-precompile` (off by default) replays those
modules using an abstract TPU topology in a CPU-only worker process beside
the existing blind W0 pilot. It never invokes the executable or initializes
the real TPU backend. The pilot joins the complete background task before
returning, and its result retains compile start/end and process CPU time.
Any work extending beyond the last source request belongs in switch cost.
Replica exports are deduplicated by their observed JAX executable key. Primary
cold, warm, and background-precompile arms all use coverage-matched zero cache
write thresholds (seconds and bytes), with compile/miss logging disabled. This
allows subsecond KV/allocation and warm-up executables to persist without
confounding cache coverage with overlap scheduling. Effective settings must
agree in every worker phase and final worker report; environment strings alone
are not proof. Replay retains its actual effective JAX options too. Target keyed
deserialize successes and monitoring hits, not precompile key equality alone,
are retained in the switch evidence. A default-one-second warm cache is not the
primary matched control.
Device-only startup traces explicitly set Python tracer level 0 and host tracer
level 2 to avoid the measured event-volume/time perturbation; trace setup and
export remain separate clocks. Attribution consolidation marks runs diagnostic
unless all eight unique worker profiles, effective 16k/512/2048 configuration,
requested cache directory and coverage-matched options, and (when tracing)
controlled profiler options and phase/device lanes agree.
Historical uncontrolled traces remain diagnostic rather than becoming baselines.
Both sequence scripts run the read-only `analyze_overlap.py --check-deploy` gate
after each deploy and retain its verdict. Switch consolidation also requires
complete W0 requests, exact greedy-output equality against the provisioned
target reference, and normal cleanup evidence before replacing unmeasured
switch/break-even reports. Output validation runs after the target W0 pilot.
The existing disjoint protocol warmup still precedes measured W0: C therefore
ends at the first **measured W0** dispatch, not first-ever target service. The
gap charges that warmup without useful-W0 credit; this is conservative,
harness-inclusive overhead rather than the minimum production outage.

Before submission, `scripts/overlap_pilot.py --layout S --run-id input-smoke
--check-inputs` loads the real catalogue and all five W0/warmup manifests,
checks referenced source IDs and token budgets, and reports their hashes.
Pilot jobs ship the small manifests but use the existing staged catalogue at
`--data-dir` (default `/tmp/omp_hetero/data`), never a duplicate large catalogue
inside the Ray upload. Local and remote actual input hashes must agree; remote
catalogue drift is rejected before W0. Historical manifest catalogue hashes
remain explicit and may differ in these unfrozen pilots: manifests are not
rewritten, and historical rates are not same-workload measurements. Controls
and treatment require identical actual staged inputs for current-rate models.

Run each sequence inside one blocking lock, with a unique `RUN_ID`, current
`HEAD_POD`, and `KUBECONFIG`:

```bash
LAYOUT=S RUN_ID=s-split DEVICE_TRACE=true BUNDLE_ROOT=/models/omp_overlap_s_bundle \
  flock /tmp/hetero-cluster.lock scripts/overlap_restore_sequence.sh
A=S B=H RUN_ID=s-to-h-control TREATMENT=baseline \
  flock /tmp/hetero-cluster.lock scripts/overlap_switch_sequence.sh
A=S B=H RUN_ID=s-to-h TREATMENT=overlap \
  flock /tmp/hetero-cluster.lock scripts/overlap_switch_sequence.sh
.venv/bin/python scripts/analyze_overlap.py \
  --campaign results/overlap/OverlapCompile/s-to-h/campaign.json
```
Repeat `--campaign` for each measured direction/treatment to consolidate them.
Break-even rows use only that switch's observed direction and retain both run
IDs; no cost is borrowed for unmeasured directions or a four-phase cycle.
The cost ledger retains raw gap-plus-ramp C, precompile wall/CPU work, and
matched source-W0 slowdown. Positive observed slowdown is charged; a negative
noisy delta is not credited. Hidden precompile work is not added again, and its
late join is already inside C. First-use estimates additionally charge the full
target-provision start-to-release interval, including graph/bundle creation,
reference generation, and weight snapshot preparation. Reused-provisioning
estimates are explicitly conditional on that prior work already being paid.
Matched-current W0 models use observed middle-20–80% median 10-second output
rates only when workload, routing knobs, serving configuration and actual
runtime fingerprints match, with no replay counted as useful output. Only the
small/large instance-binding lists are excluded from routing-knob comparison:
those bindings are the layout treatment and remain in the raw evidence. These are
steady-rate projections, not observed crossover experiments. Reused historical
throughput inputs remain explicitly labeled cross-revision model inputs with
runtime equivalence unestablished.

The scripts release their own engines, placement groups, tmpfs snapshots and
uniquely named compile caches before unlocking. Cache cleanup accepts only
non-symlink `/models/jax_cache_overlap_*` paths; `/models/jax_cache` is untouched.
Raw worker timings, complete client
endpoints, serving impact and break-even evidence belong under
`results/overlap/OverlapCompile/`; cache-file existence alone is not a hit.

The campaign requires 40 GB of RAM and tmpfs headroom after staging, refuses
pre-existing engines, retains machine-readable evidence under
`results/restore/ShardedRestore/`, and releases its deployments and removes
all snapshot copies before relinquishing the cluster lock.
The summary retains cold-vs-restore timings, snapshot bytes and cost, exact
greedy validation, restored W0 makespan, measured transition costs and a
break-even table. Mixed-workload M-to-S and partial-restore projections are
explicitly estimates: only full H/S restore transitions are measured, and the
partial projection preserves the historical partial/full effective-cost ratio.

### Scoped fast graph construction

`EngineSettings.fast_graph` (default `false`) applies a load-scoped JAX 0.10
workaround to both normal checkpoint loading and snapshot restore. It removes
the expensive tracer-origin formatting from the `Tracer.sharding`
`AttributeError` swallowed by Flax during abstract graph construction; it does
not skip graph validation, change weights, or alter device execution. The
original property is restored even when loading raises. Worker snapshot
status records the requested flag; both loaders record the abstract
`nnx.eval_shape` call as `graph_build`, separately from total `load_model`.
The override is process-global for `load_model`, not thread-local; this does
not establish safety for concurrent initializers in the same process.

The authorized S A/B is `HEAD_POD=<current-head> flock /tmp/hetero-cluster.lock
bash scripts/fast_graph_ab.sh`. Run only after higher-priority cluster
validation has released the slice. Its four arms compare normal/restore with
the flag off/on, check greedy outputs against ShardedRestore, and retain
per-worker clocks in `results/restore/FastGraph/summary.json`. The summary
fails closed on missing workers, flag evidence, restored status, or reference
agreement. CPU synthetic graph timings are not TPU startup measurements.

The completed S/TP8 four-arm run used frozen runtime source `df46a72`
([evidence](results/restore/FastGraph/summary.md),
[provenance](results/restore/FastGraph/source.json)). Medians across eight
workers, in seconds:

| Loader | Graph: baseline / fast | Total load: baseline / fast |
|---|---:|---:|
| Normal checkpoint | 14.739 / 1.790 | 234.675 / 222.587 |
| Verified snapshot restore | 14.631 / 1.767 | 39.436 / 26.455 |

All four arms matched the reference exactly: 128 greedy outputs and 8,140
tokens per arm. Both restore arms reported verified restoration on all eight
workers. Every arm removed its four actors and placement groups; final cleanup
reported zero snapshot bytes on all eight hosts. The separate CPU synthetic
graph smoke measured 9.760 / 0.571 seconds; it was not a TPU measurement.

This is one fixed-order A/B, not a throughput result or a cold-filesystem-cache
experiment. Snapshots were written after the normal baseline, so the normal
fast arm also had resident snapshot files. Total-load differences are observed
deltas, not perfectly isolated causal estimates. Worker clocks run in parallel
and must not be summed into deployment wall-time savings.
The reducer retains raw worker durations and medians in JSON and rounds only
the final displayed table; it does not round worker observations before aggregation.

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

### llm-d EPP ingress (unfrozen pilots)

`deploy/k8s/omp-epp-router.yaml` is a second router in which Envoy asks the
forked llm-d EPP (llm-d-router branch `research/heterogeneous-tpu-20260918`,
`hetero-placement` and late-binding plugins) for a destination and forwards the
request itself (`ORIGINAL_DST` on `x-gateway-destination-endpoint`). The
Python gateway stays deployed for engine bindings, pod labels
(`omp.llm-d.ai/role`, `/tp`, `/max-model-len`) and iteration capture, but is
off the request path, so migration is off. Arm configs mirror the W4 policies:
`deploy/epp/plain.yaml`, `reserve.yaml` (fallback plus a 36,864-token long
reserve) and `latebinding.yaml` (one EPP queue, longest prompt first, dynamic
long reserve). Engines serve the EPP contract directly: `/metrics` exports
`vllm:kv_cache_usage_perc` and `omp:kv_{free,total}_blocks`/`omp:kv_block_size`,
and `/v1/completions` accepts `prompt` as one token-id array.

```bash
OMP_EPP_IMAGE=<registry>/llm-d-router-epp:<tag> scripts/epp_router.sh latebinding
EPP_CONFIG=latebinding WORKLOADS=W4 scripts/run_pilot.sh M M0
```

`run-arm --epp-config` refuses unless every EPP pod is annotated with that
file's SHA-256, and accounts each request from the engine response: one
segment, exact budget, and the Envoy upstream matching the serving instance.
Without late binding the EPP refuses a request that fits nowhere (429); the
client then polls at the Python gateway's `admission_poll_s`, as the Python
admission loop does, and records `capacity_retries`/`retry_request_bytes` per
request (each retry resends the whole body). `admission_wait_s` is that
client-clock polling plus Envoy's own `x-omp-epp-queue-ms`. Before warmup and
before the timed run the harness resets hetero-placement's quota counters and
projection (`POST :9095/hetero/reset`, waiting out live leases) and checks
`/hetero/state` holds no reservation after the run. An engine whose client
disconnects aborts the generation. EPP runs are unfrozen pilots, never cohort
evidence.

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
