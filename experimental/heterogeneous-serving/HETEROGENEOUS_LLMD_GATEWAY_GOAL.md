# Goal: static heterogeneous bulk inference behind llm-d, with request migration

## Assignment and scope

Build and evaluate a bulk inference testbed on the **existing Ray TPU cluster**, using **llm-d as the common gateway and central cluster-level scheduler**, a **fixed mix of small and large serving instances**, and **request migration between those instances**. Initial placement and every migration request must pass through this gateway scheduling authority. Determine whether migration can preserve the useful throughput of densely batched small instances as some requests grow too memory-intensive, after charging all migration and control costs.

This is not simply prefill/decode separation. Each instance can execute both prefill and decode. The smaller instances are intended to carry dense bulk batches; the larger instance provides another placement option for long-context requests. Prefill/decode disaggregation and KV offloading are sources of reusable mechanisms, not the prescribed architecture.

This is a goal prompt for subsequent execution. Revising it does not authorize launching experiments, redeploying the gateway, changing running engines, or recreating the Ray cluster. Changes to vLLM, its TPU backend, and llm-d are in scope when required to implement the real behavior. Migration performance and ICI availability are unqualified; a negative result is valid. ICI KV transfer is an optional extension, not a deployment, evaluation or completion prerequisite.

**Static configuration means:** choose instance count, chip assignment, tensor parallelism, runtime/communicator arrangement, engine limits, routing rules, migration thresholds and transport mode before a run; freeze them until that workload drains. Requests may move and admission may respond to observed state under those fixed rules. No online mesh resizing, replica scaling, weight relocation, role switching or parameter search.

## Objective and research questions

Minimize **end-to-end dispatch-to-drained makespan at asserted-equal useful emitted tokens** for a fixed, deep conversational backlog. Report useful output tokens/s and makespan reduction separately. Count only newly generated useful output, never replayed history, transferred KV, or duplicate delivery.

Answer three separate questions:

1. In which measured batch/context regimes do smaller instances offer better useful throughput per allocated chip? Treat that advantage as the hypothesis to exploit, not a universal hardware law.
2. On the same static heterogeneous layout, does migrating growing requests beat leaving them on their original instances?
3. For the same migration policy, is rebuilding destination KV from token history or transferring existing KV over a qualified host-staged path better for total bulk makespan? Optional extension: can ICI improve this comparison without coupling otherwise independent engine execution?

TTFT, TPOT, request latency and fairness are diagnostics, not competing interactive SLAs. High scheduled batch size, fewer preemptions and more freed blocks do not themselves establish a throughput improvement. Native preemption remains permitted; do not promise zero preemption.

The timed interval starts before the first workload dispatch decision and ends after every request's final output is accounted for. Include queueing, routing, proxy work, admission, pause/retirement, transfer or recomputation, destination waiting, ownership reconciliation and online bookkeeping. Report initialization separately and also initialization-plus-serving time. Do not add overlapping stage durations to invent a makespan decomposition.

Predeclare a common useful-progress interval with abundant fresh backlog in all compared arms; report its throughput and terminal drain separately. If the manifest is too shallow at cluster scale, retain the whole-workload result but do not claim sustained-bulk improvement.

## Evidence to reuse, and what is not established

| Source | Grounded observation | Limitation for this goal |
|---|---|---|
| Prior cluster inspection in this session | Ray cluster `tpu-ray-cluster-vllm-tpu` in namespace `llm-d-optimized-baseline`; eight TPU workers reported 32 TPU resources, and Kubernetes node labels reported `4x8` | Inventory is not proof of current availability, valid disjoint subslices, or cross-runtime ICI access. Preserve running workloads. |
| Prior gateway inspection in this session | Envoy plus endpoint picker image `ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.9.0`; an `InferencePool` using pod-label selection | Does not establish migration support, a credit gate, or direct discovery of arbitrary Ray actors. |
| `results/scheduling_headroom_20260915/REPORT.md`, matched A/B/C section | The retained TP16 engine had 566 usable KV blocks, native async queue capacity 2; arrival cap128 beat the tested adaptive admission arm on W-SMALL | Configuration-specific capacity and policy evidence, not a universal TP16 capacity or multi-instance optimum. |
| `results/scheduling_headroom_20260915/native_probe_final.json` | Installed scheduler was `vllm.v1.core.sched.async_scheduler.AsyncScheduler`; measured MRO includes `vllm.v1.core.sched.scheduler.Scheduler` | Reinspect the actual build before changing it; the TPU runner and migration APIs have not been qualified. |
| `benchmarks/common/model_config.py` | Repository Gemma declaration has 48 layers, 16 KV heads and head dimension 128 | A declaration is not the mounted model configuration or its physical sharding/layout. Derive bytes, strides and rank mapping from the actual runtime. |
| `results/decode_allocation_20260913/` and `results/scheduling_tracks_20260914/` | Existing batch/context and bulk scheduling evidence to audit before collecting more | Reuse only within its documented workload, topology and measurement domain. |

Remove the earlier draft's timing estimates and causal claims from the execution contract. There is no qualified recompute/host-transfer/ICI timing comparison or crossover length here. Head count alone does not prove matrix tiling efficiency; topology labels do not prove collective routes. Do not extrapolate a long-prompt throughput measurement to a different context or infer usable KV capacity by dividing nominal weight size by chip count.

## Static instance layout and resource ownership

First candidate, subject to topology and runtime qualification:

| Instance | Chips | Hosts at four chips per host | Serving role |
|---|---:|---:|---|
| Small A | 8 | 2 | Prefill and decode; dense-batch candidate |
| Small B | 8 | 2 | Prefill and decode; dense-batch candidate |
| Large | 16 | 4 | Prefill and decode; long-request destination |
| Total | 32 | 8 | All allocated hardware charged, including idle capacity |

These are logical names, not assignments to worker IDs 0–7. Use `TPUTopologyDiscovery` via `pkg_util.tpu_topology`, validate physical coordinates and disjoint contiguous selections, and derive coordinator/rank order and runtime bounds from that evidence. A Ray placement group reserves resources; it does not by itself create a valid TPU communication domain.

Use explicit node-pinned bundles of `{"TPU": 4.0, "node:<discovered-ip>": 0.001}` with `STRICT_SPREAD`, await readiness with a bounded timeout, and bind each worker to its intended bundle. Validate device visibility, rank ordering, CPU needs and exclusive accelerator ownership. Do not rely on unpinned auto-placement or infer a submesh from sorted IPs.

Reuse the existing cluster and any appropriate retained engines. Before new allocation, reconcile placement-group ownership and clean only demonstrably stale, owned resources. A `CREATED` or `PENDING` state alone does not establish staleness. Never run blanket placement-group cleanup on this shared cluster. Between matched arms, change topology only after the previous workload drains and only when the experiment requires it; never reslice within an arm.

Measure actual usable KV blocks and bytes, block/cache-group layout, engine context limits and supported concurrency on each instance. The historical 566-block observation is a reference, not a constant to enforce on new engines. A larger TP setting does not automatically raise the model's logical context limit.

## Phase 1: contiguous subslice allocation and TPU runtime bring-up

**This is a dedicated engineering phase and a hard gate before migration implementation or comparative timing.** Its deliverable is a reproducible launcher and evidence that the fixed TP8 + TP8 + TP16 engines work together on the existing Ray cluster. Placement-group readiness, a successful import, or a single isolated TP8 run is insufficient.

### Reuse existing initialization work, but qualify its assumptions

- `admission_control/tpu/topology.py` discovers hosts and derives coordinates from worker IDs and bounds; probe failures/defaults and ID-derived geometry must not become physical contiguity proof. The initialization path in `benchmarks/run_measurement_audit.py` explicitly requires observed chip coordinates for TP8 and checks a contiguous rectangle. Reuse that evidence-driven approach for every chosen subslice.
- `benchmarks/run_measurement_audit.py` already sets a subslice environment before constructing `vllm.LLM`, passes an explicit placement group to the executor, and collects actual global/local/selected devices. Its current non-TP8 selection takes all discovered hosts, so it cannot be assumed to select a four-host TP16 subset correctly on this eight-host cluster. Extend the existing path to consume the complete validated multi-instance layout.
- `admission_control/tpu/measurement_worker.py` preserves the original `TPU_WORKER_ID`, derives an instance-local process rank from the ordered `TPU_PROCESS_ADDRESSES`, checks coordinator consistency, and sets local task/worker IDs before parent device initialization. Reuse and audit this rank translation rather than forwarding physical slice-wide IDs unchanged.
- `admission_control/tpu/placement.py` currently calls broad stale-placement cleanup inside `create_node_pinned_placement_group`. Correct that ownership behavior before reusing it for multiple live instances; creating Small B must not remove Small A's placement group. Reuse shared helpers only after they satisfy the scoped lifecycle contract.

### Physical topology and executable placement plan

1. Inventory the parent slice identity, physical chip coordinates, host IPs, Ray node IDs, original TPU worker IDs, resource ownership and current cluster generation. Reuse a saved coordinate proof only if it is tied to the same hardware allocation. Missing or ambiguous geometry is a preflight failure, not a reason to guess from worker/IP order. Any necessary device discovery runs under RCM with explicit owned resources; do not initialize an extra TPU runtime on chips held by unrelated work.
2. Produce one layout mapping every selected physical chip/host to exactly one logical instance and its deterministic process/rank order. Prove each selected rectangle is supported and contiguous, instances are disjoint, and the total allocation matches the plan. Preserve physical identity separately from instance-local numbering.
3. Materialize explicit node-pinned placement groups and bind actual vLLM child workers to the intended bundles. Await every group with bounded timeouts and inspect actual worker placement. Reserve the required CPU resources as well as TPU resources. Do not rely on the driver actor's placement to imply executor placement.

### Per-instance environment and rank contract

Generate a per-process environment manifest from the physical layout and installed runtime contract. The following variables occur in the existing launchers; they are inputs to audit, not a universal copy-paste recipe:

| Concern | Existing variables/settings to inspect | Qualification requirement |
|---|---|---|
| Execution-domain shape | `TPU_TOPOLOGY`, `TPU_HOST_BOUNDS`, `TPU_PROCESS_BOUNDS` | Derive the instance-local topology, orientation and host/process grid; do not leak the parent `4x8` bounds into each small instance. |
| Local chip layout | `TPU_CHIPS_PER_HOST_BOUNDS`, `TPU_CHIPS_PER_PROCESS_BOUNDS`, `TPU_CHIPS_PER_HOST` where consumed | Match actual chips and process ownership; host count, chip count and TP rank count are not interchangeable. |
| Rendezvous | `TPU_PROCESS_ADDRESSES`, `TPU_PROCESS_PORT`, and coordinator settings actually used by the backend | Use a consistent ordered member list for each runtime domain, reachable worker-to-worker addresses and non-conflicting address/port ownership. Same ports on disjoint hosts need not conflict; never assume separate logical instances imply separate sockets. |
| Rank remapping | `CLOUD_TPU_TASK_ID`, `TPU_WORKER_ID`, executor rank and process index | Map physical worker identity to instance-local process ranks; coordinator/rank zero and all workers must agree. Keep original IDs in the evidence. |
| Backend/device initialization | `TPU_MULTIHOST_BACKEND`, `JAX_PLATFORMS`, `VLLM_ENABLE_V1_MULTIPROCESSING`, and device-visibility settings actually consumed | Inspect the installed TPU backend/PJRT path; set the required values before device/runtime initialization, including initialization triggered by imports. Verify real device ownership rather than trusting environment strings. |
| Compilation/runtime provenance | `JAX_COMPILATION_CACHE_DIR`, vLLM/TPU backend/JAX/jaxlib/libtpu versions and relevant flags | Record effective values and cache/compilation policy; do not treat a working cache or inherited flag as proof of correct topology. |

The current audit uses host/process bounds `1,2,1` for its observed TP8 orientation, `2,2,1` for TP16, and local chip bounds `2,2,1`. These are source examples, not values to hardcode regardless of physical orientation or runtime process model. Record the actual derivation and validation for this deployment.

Explicitly propagate the environment to every engine process and every nested Ray/vLLM worker. Setting variables only on the Ray head or parent actor is insufficient. Audit runtime-env propagation and any backend environment allowlist; compare intended and effective values inside each real worker. Do not mutate shared head/worker-pod environment or an already initialized JAX/PJRT runtime to retarget a different subslice. Use separately owned runtime processes and preserve unrelated actors.

For each worker, record instance ID, physical node/chip coordinates, original TPU worker ID, instance-local process index, executor rank, coordinator address, placement-group/bundle identity, global/local device inventory and selected engine devices. For independent runtimes, each TP8/TP16 engine must see its qualified 8/16-chip execution domain. If a shared runtime is deliberately chosen, record its broader visibility and prove disjoint engine execution subsets instead; do not misreport a 32-device global view as an independent 8-device runtime.

### Bring-up sequence, failure handling and exit gate

- Validate metadata/layout/environment consistency before loading model weights. Reject overlapping assignments, missing/duplicate process ranks, inconsistent bounds or missing rendezvous members with actionable diagnostics. Bound placement, rendezvous, device initialization and engine startup stages; use RCM structured events and traps to distinguish scheduling waits from collective initialization hangs.
- Bring up each intended execution domain, then keep all three engines resident together. Load real `/models/gemma-4-31b` weights and run short real ShareGPT prefill and decode on **all three concurrently**, with prefix caching disabled and per-iteration evidence. Verify every instance returns correct request accounting and makes progress while the others run. A local device listing or CPU mock does not satisfy this gate.
- Verify actual physical device membership and runtime rank mapping against the plan, exclusive ownership, native asynchronous execution and measured usable KV capacity. Check for unintended full-slice rendezvous, cross-instance environment leakage and collective waits coupling unrelated engines. This phase proves independent serving, not cross-instance ICI transfer.
- Demonstrate that reapplying the same layout attaches to the same owned actors/groups without creating duplicates, and that a failed partial initialization cleans up only its owned new resources. In qualification, exercise a scoped restart of one owned instance after draining it and show that peers remain available; this is a lifecycle check outside timed workloads, not dynamic reconfiguration during a workload.
- Apply the same topology/environment qualification to the homogeneous H and S layouts before they are benchmarked. A validated mixed layout does not automatically validate the all-small or all-large layout. If optional ICI work changes the communicator arrangement, requalify it outside a workload with matched comparison scope; do not require that work for the initial testbed.

**Exit artifacts:** physical topology evidence; resolved instance/host/chip/rank map; intended and effective per-worker environment manifests; version/image/configuration hashes; placement ownership records; stage clocks; concurrent real-serving results and capacity/cleanup evidence. No migration phase starts until this gate passes or the precise unavailable prerequisite is reported. Export only relevant sanitized environment values, not a dump of credentials or the entire process environment.

## Phase 2: repeatable deployment and Ray serving-instance discovery

**Deployment simplicity is a deliverable, not a manual setup note.** After Phase 1, a documented configuration-driven operation must attach to the existing Ray cluster, reconcile the owned serving instances, connect them to llm-d and report readiness. This phase qualifies ordinary gateway serving and discovery before request-migration experiments.

### Identify serving instances, not just Kubernetes or Ray nodes

The existing llm-d pool uses Kubernetes label-based discovery. Keep that integration where useful, but introduce a stable mapping for Ray-managed serving instances: a multihost TP16 engine is one schedulable serving endpoint, not four independent backends, and a Ray worker pod/node being alive does not imply a model endpoint is ready.

Use a minimal serving-instance registry populated by the launcher and actual serving processes; reuse existing Ray named-actor/control state where practical instead of introducing an unrelated database. The registry describes endpoints and capabilities; the gateway remains the sole cluster-level scheduling authority. Required records:

| Descriptor | Required meaning |
|---|---|
| Logical instance identity | Deployment/layout ID, stable instance ID (such as Small A), current instance generation and configuration digest |
| Ray ownership | Ray cluster and namespace, coordinator/worker actor identities, placement-group ID, and the Phase 1 host/chip/rank map |
| Reachable serving endpoint | Actual serving/control addresses and ports reachable from the gateway, plus the implemented metadata/metrics path; never infer them from `ray.nodes()` alone |
| Serving capability | Model/build identity, TP size, context limit, measured KV capacity, supported continuation/transport modes and current readiness |
| Lifecycle | Starting, ready, draining or unavailable state, health/lease freshness and last accepted registration generation |

Preserve stable instance IDs across owned restarts while changing the instance generation and endpoint binding. Instance generation is distinct from a request's migration ownership epoch; stale registrations must not resurrect a replaced endpoint or accept its old control messages. Register as routable only after the engine, model, runtime membership and serving endpoint are ready. Withdraw from new initial/migration selections on stale health or draining, without silently reassigning already-owned requests.

The existing `admission_control/serving/orchestrator.py` uses a single fixed persistent actor name and namespace. Extend the established launcher/lifecycle pattern to distinguish all configured instances and verify their configuration on attachment; a successful `ping()` alone is not proof that the existing actor matches the requested model, TP size or host assignment.

### Bridge the registry into the installed llm-d discovery contract

Inspect the actual endpoint-picker/`InferencePool` version and choose one supported discovery path. For the currently observed pod-selector contract, the conservative integration is a pod-backed serving adapter (or existing equivalent proxy) representing each logical Ray instance; the gateway selects labeled adapter endpoints and each adapter resolves the current registered instance. Do not label every TPU worker pod as a model replica. If the installed stack already supports a registered external-endpoint provider, qualify that simpler direct path instead of adding unnecessary adapters. Record the chosen path and charge any proxy/adapter overhead equally in comparisons.

Do not assume that creating an `EndpointSlice` makes an arbitrary Ray actor discoverable to a pod-selector `InferencePool`, or invent CRD fields. The bridge must map the EPP-selected backend back to the stable instance ID and current generation used by the central scheduler, reservations and migration coordinator. Discovery must never introduce a second independent routing decision. Kubernetes labels may remain the gateway-facing representation; Ray registration supplies the actual engine identity and lifecycle.

Verify gateway-to-endpoint network reachability, DNS/address scope, port ownership and the necessary namespace-scoped permissions. Dynamic actor addresses must be updated by the integration, not pasted into Envoy YAML by hand. Keep registry updates and readiness observation in the deployed control service; agent monitoring remains through RCM, not bespoke Ray job polling.

### Operator workflow and acceptance

- **One desired configuration:** existing Ray cluster connection and namespace, model/image revisions, logical instances and topology constraints, fixed serving/scheduling settings, gateway namespace/discovery mode and resource ownership. Generate the resolved physical layout, per-process environment and gateway bindings from this configuration. Do not ask operators to edit IP lists or export per-worker TPU variables manually.
- **Plan/preflight:** provide a non-mutating operation that reports required resources, proposed mappings, missing permissions/model/data/network prerequisites and intended changes. Resolve physical assignments from qualified topology evidence rather than guessing values to make the plan look complete.
- **Deploy/reconcile:** one documented command under the existing RCM submission workflow creates or attaches to owned instances, applies the chosen gateway integration and waits for explicit readiness with bounded stage timeouts. Repeating it with the same configuration reuses the same actors, placement groups and registrations. A conflicting live configuration produces an actionable refusal instead of reconfiguring active workloads. No new Ray cluster is created.
- **Inspect and drain/teardown:** provide documented commands showing desired versus observed instance state, resolved placement/environment, endpoint registration, health and last failure; drain before removing only the deployment's owned resources. Startup failures must identify whether placement, TPU rendezvous, model loading, serving reachability or gateway registration failed. Preserve unrelated jobs, namespaces, labels and placement groups.
- **Prove usability:** from the declared prerequisites, execute the deploy command, route short real ShareGPT work through llm-d to every logical instance, and verify that the mixed layout exposes three serving instances rather than eight worker-node replicas. Reapply unchanged configuration and show that no duplicate resources appear. Outside timed workloads, demonstrate that a drained instance restart/address change updates registration, removes the stale endpoint and does not interrupt peers. Qualify equivalent discovery for H and S using the same workflow.

**Exit artifacts:** desired configuration and resolved manifests, runnable/documented deploy/inspect/drain entry points, instance-registry schema and live descriptor snapshots, gateway bindings, readiness/failure evidence and successful gateway-routed real inference. Configuration rendering or Kubernetes pod readiness alone does not satisfy this phase.

## Architecture and responsibilities

```text
Fixed bulk manifest / backlog
             |
 llm-d gateway serving layer <--- migration proposals from engines
 (central scheduler + EPP integration,
  bounded dispatch, migration coordinator,
  stable request/response owner)
        /          |          \
   Small A      Small B      Large
   TP8 P+D      TP8 P+D      TP16 P+D
        \__________|__________/
 Gateway-authorized continuation migration
   replay request through gateway OR
 direct engine-to-engine KV data transfer

Existing Ray cluster: fixed placement and engine lifetimes
RCM: workload submission, event streaming and supervision
```

### llm-d as the central scheduling authority

- Route both fresh requests and migration/continuation requests through the llm-d gateway scheduling layer. It owns cluster-level destination selection, admission, destination reservations, ownership commits and request-to-engine assignments. Source engines may report migration candidates and local pressure, but must not independently select a destination or submit continuation requests directly to another engine.
- Extend the installed EPP/serving integration so initial placement and migration use one authoritative eligibility, selection and admission pipeline, with phase-specific constraints. Existing endpoint scorers rank candidates; they do not already implement continuation, reserve memory or throttle a saturated single destination. Do not add a competing scheduler in the proxy, source engines or Ray launcher.
- The bounded dispatcher, migration coordinator and stable response owner are parts of the gateway serving layer. They may be separate processes, but share the same authoritative ownership/reservation state and do not make independent routing decisions. A gateway decision authorizes destination preparation and continuation; native vLLM schedulers still own per-iteration batching, sampling and physical allocation. Ray owns placement and worker lifetimes, not per-request migration policy.
- A stable request/response owner retains request identity, result assembly, committed token offsets and cancellation across engine changes. Bulk completion is sufficient; where streaming is exposed, retain the client response and consume source/destination output by committed epoch and token offset. Changing a header or route entry cannot retarget an already-open upstream response.
- Reuse an existing suitable serving proxy/connector if available. Modify llm-d integration as needed; do not invent plugin names or assume ExtProc already implements continuation or migration.
- Use the serving-instance registry and the qualified gateway discovery bridge from Phase 2. Keep stable instance ID, generation, model readiness, draining and endpoint removal consistent across EPP selection, reservations and migration ownership. A Ray node address is not an actor serving endpoint, and worker-pod labels alone do not identify a multihost serving instance.

### Migration requests traverse the gateway; KV payloads need not

1. The source reports a migration candidate to the gateway under the existing logical request ID and current ownership epoch, with observed progress and source pressure. This is a scheduling event, not a new user inference request; it must not reset prompt/output semantics, generation limits or accounting.
2. The gateway validates the current owner and policy, selects an eligible destination using cluster-wide state, and obtains an authoritative destination reservation before instructing source quiescence. A stale or duplicate proposal is reconciled by transaction/epoch rather than scheduled twice. The gateway may defer or decline a migration while preserving source progress.
3. After the request-specific safe snapshot, the gateway forwards the authorized continuation to the chosen destination. For recompute, this includes token history and continuation state. For KV migration, it includes continuation metadata and the authorized transfer descriptor; the source and destination may move KV bytes directly over qualified host transport, or ICI if that optional path is qualified. The gateway need not relay KV tensors or hold device buffers.
4. Preparation acknowledgments, commit/abort decisions, ownership updates and cancellation return through the gateway coordinator, following the shared safe-handoff protocol below. Engines may not turn an authorized tensor transfer into an independently scheduled request. The stable response owner keeps one logical result across the migration.

This is centralized request scheduling with a direct KV data path, not centralized tensor copying or centralized execution of every decode step. If gateway scheduling is unavailable, do not admit uncoordinated migrations or elect a source engine as a replacement scheduler; already-owned requests may continue under their existing authorization. The instance layout and scheduling policy remain fixed throughout the workload.

### Static routing and bounded migration policy

- Prefer small instances for eligible requests, but do not force every request through them. Route a prompt that cannot fit a small instance directly to a qualified large instance. Define a fixed work-conserving rule for fresh traffic on Large so half the cluster is not accidentally left idle awaiting outliers; preserve migration headroom and apply the same rule to no-migration controls.
- Begin with one bounded, deterministic small-to-large migration policy. Use observed prompt/generated lengths, actual allocation, source pressure, scheduled batch/context state and destination eligibility. Fix thresholds, per-request migration limit and concurrency limits before timing; no bidirectional ping-pong or online policy search.
- Long length identifies a candidate, not an unconditional command to migrate. Destination backlog, remaining legal generation budget, bytes to move/rebuild and source capacity relief all matter. A request near completion can be cheaper to finish locally. Do not claim profitability from an unvalidated cost formula.
- Online decisions may read public request parameters and observations, not hidden realized output targets or future completion lengths. Keep any oracle experiment explicitly separate.
- Keep a bounded backlog and migration ledger in the gateway dispatcher/coordinator. Reserve destination capacity atomically before pausing source work; simultaneous proposals from Small A and B must not consume the same capacity. Reservations must participate in the destination's authoritative native allocation path, not just a sampled utilization counter.
- Account separately for native resident allocations, outstanding incoming reservations, transfer/rebuild staging and a declared allowance for resident decode growth. Reservation-to-allocation conversion consumes each credit once. Future completion is not free capacity. Prompt-only reservations are insufficient for ongoing decode growth.
- Bound host/device staging and in-flight migrations. If a destination is unavailable or full, defer migration while preserving source progress and throttling fresh admissions as necessary. Native allocation remains authoritative. Qualify progress when no migration can be admitted; do not deadlock waiting for a release that requires a paused request to run.

## Two migration mechanisms, one continuation contract

### Recompute-only migration: rebuild KV, do not regenerate output

Conceptually the destination replays original prompt token IDs `P` followed by the already-generated token IDs `O[:k]` to rebuild its KV cache, then samples the next output. The generated prefix is teacher-forced history, not sampled again or emitted again. No KV tensor transfer or cross-TP KV resharding is needed; token IDs and continuation metadata still cross the network.

This must restore a **continuation request**, not submit `[P, O[:k]]` blindly as a new user prompt. Preserve:

- raw token IDs, original prompt/output boundary, model/tokenizer/template identity and any required position metadata; no text round-trip or chat-template reapplication;
- original sampling parameters, actual RNG state/counter position rather than just its seed, penalty history and applicable logit-processor state;
- original generation limit and remaining budget, emitted-token count, EOS/minimum-length/stop-sequence state, structured-output state where enabled, and output/logprob accounting;
- computed positions, generated-but-not-yet-computed tokens, and committed output offset separately. The final sampled token need not yet have KV. Audit the precise native replay boundary to avoid skipping or computing it twice.

Rebuild KV in the destination's native layout while keeping the original logical prompt/output semantics. KV reconstruction may use native chunked prefill; it must not consume extra RNG draws, reset penalties, extend generation limits, or count replay tokens as useful new output. It adds real compute and can interfere with Large's resident requests.

**Implementation order:** qualify recompute-only continuation first as the lower-dependency migration mechanism. This is an engineering choice, not a prediction that it is faster. Host-staged KV-transfer qualification remains in scope; ICI qualification is optional.

### KV-state migration: offload/disaggregation mechanisms plus ownership transfer

Export the source's valid computed KV and continuation state, translate its actual physical layout, import into native destination allocations, and continue without rebuilding the transferred positions. This resembles offload/reload, but the destination is another engine and may use different sharding. It resembles a P/D handoff, but migration may occur after many decode steps and must preserve their sampling/output state.

Inspect all actual cache groups, dtypes, scales, valid token ranges, padding, positions, block sizes and shard mappings. Do not assume a simple two-heads-to-one-head split, that host count equals TP rank count, or that physical block IDs are portable. Do not transfer uninitialized block tails as valid state. Unsupported layouts must be reported, not silently recomputed and counted as successful KV transfers.

### Host-staged KV transport and optional ICI

The core transport comparison is recompute versus qualified host-staged KV transfer: device-to-host, host transport and destination import. The host-staged path needs real implementation and measurement. Ray object-store use does not imply zero-copy TPU-to-TPU transport. Neither host transport nor recompute is a guaranteed fallback. Freeze the selected mechanism per measured arm, record any fallback explicitly, and never relabel recomputation as successful KV transfer or host transport as ICI.

**ICI is an optional acceleration path, not a gate for deployment, core experiments or completion.** Cross-subslice ICI KV transfer may be possible, but sharing a physical TPU slice or a Ray cluster does not establish access between independently initialized serving runtimes. Pursue a straightforward supported path if available. If qualification or implementation requires substantial backend/communicator redesign or compromises fixed topology and independent engine progress, record the reason and defer it to future work. Proving ICI impossible is not required.

If ICI is pursued, inspect the actual TPU backend/PJRT/XLA APIs, device ownership and communication scope, and qualify transfer between the fixed execution groups. Choose any supported runtime/communicator arrangement before the workload and prove that unrelated instances retain independent stepping. No runtime mesh reconfiguration or global engine synchronization solely to make migration work. Measure actual ICI transfers before reporting an ICI arm; unsupported, unattempted or deferred ICI has no performance result.

## Safe handoff shared by both mechanisms

The following are protocol states/invariants, not claims that matching APIs already exist:

1. **Reserve:** gateway coordinator validates the migration proposal, selects the destination, admits a bounded migration against destination capacity and records request ID, source/destination, transaction ID and ownership epoch.
2. **Quiesce the request:** stop issuing new work for this request only. Retire all its issued native work and account for its outputs; preserve unrelated native asynchronous stepping. A scheduler call returning is not a retirement barrier. The historical engine had two batches in flight.
3. **Snapshot:** retain the complete continuation state and committed delivery offset. Keep source execution fenced while the snapshot is used. Requests finishing or cancelling during quiescence must terminate cleanly rather than migrate as unfinished work.
4. **Prepare:** destination validates the snapshot and allocation, then rebuilds KV or imports it into a prepared continuation. KV replay/import work may run, but the destination must not sample or emit continuation tokens before ownership commit. Confirm transfer reads and writes have completed where applicable.
5. **Commit ownership:** use one authoritative, recoverable ownership record with a monotonic epoch and an atomic source-to-destination transition. Record the decision before authorizing destination generation and source release. Make commands idempotent by transaction/epoch; source scheduling, destination activation and proxy output all enforce the owner/epoch, not merely output filtering after duplicate execution.
6. **Release:** only after request-specific retirement, completed transfer reads if any, destination preparation acknowledgment and committed ownership may native source cleanup release KV. Recompute avoids KV reads, but not quiescence or ownership safety. Keep the source KV until this conservative commit point in the first implementation; early release with a separately recoverable replay log is outside this experiment.
7. **Reconcile uncertainty:** a timeout is not an abort. Consult the authoritative commit decision; an already-committed move cannot be rolled back because a commit message or acknowledgment was delayed. If no commit occurred, fence/abort the prepared destination before resuming source. If the decision cannot be established, pause safely and report an unresolved failure rather than run both owners. Handle duplicate/delayed prepare, commit, release and cancel messages. On coordinator restart, recover ownership before resuming either engine.

The proxy must reconcile all source outputs through the handoff offset before advancing ownership for later output, so stale-epoch filtering cannot discard an undelivered valid prefix. Deliver each logical output token and final completion once. Cancellation releases source/destination reservations and buffers without reviving the request. Do not promise uninterrupted service through arbitrary process loss; document the supported fault envelope and surface failures rather than truncating or fabricating success.

## Source changes and reuse

Start from installed source and existing connector/offload/recompute mechanisms; extend those rather than implement a second scheduler or allocator.

- Saved probe: `vllm.v1.core.sched.async_scheduler.AsyncScheduler`, source `/workspace/vllm/vllm/v1/core/sched/async_scheduler.py`; MRO includes `vllm.v1.core.sched.scheduler.Scheduler` and `SchedulerInterface`.
- Existing harness: `benchmarks/kv_admission.py` imports `vllm.v1.request.Request` and `RequestStatus`, and tracks native `num_in_flight_tokens`, `num_output_placeholders`, allocations and issue/retirement identities. Reuse its bookkeeping lessons, not a claim that its admission hook already implements migration.
- Saved probe records `EngineCore.step_with_batch_queue` and `KVCacheManager` allocation/free methods. Inspect their actual owning classes, deferred-free rules, connector contracts and callsites before editing.
- Discover the actual TPU worker/model-runner class and loaded module in the serving build, including any `tpu_inference` implementation. Do not prescribe a guessed `TPUModelRunner` path or use legacy V0 `SequenceGroup`/`BlockManager` seams.
- Modify llm-d/proxy integration for endpoint eligibility, stable request ownership and continuation delivery as needed. Audit the versioned discovery API and existing proxy behavior before choosing an integration seam.
- Record source paths, revisions, image digests, configuration and model identity. A local Skaffold file does not establish that referenced source trees exist or that hot-syncing a live engine is safe. Preserve native async stepping and sampling; do not replace the outer stepping loop with a synchronous migration loop.

## Research evaluation: establish whether the system is useful

The research deliverable is an empirical answer, not just a working gateway or migration protocol. Separate engineering correctness, mechanism cost and end-to-end benefit. A valid negative finding completes an experiment; a working handoff alone does not establish usefulness.

### Controlled arms and causal comparisons

Use matched total allocated hardware, model, workload, gateway/proxy path and instrumentation. Every arm has a static configuration throughout its workload.

**Homogeneous means uniform serving-instance size:** all instances use the same TP setting, even though every arm here uses the same TPU v5e hardware family. Compare the heterogeneous system against both uniform layouts below, not against a single undersized server. All baseline requests still pass through llm-d's central scheduler with capacity-aware initial placement; disabling migration must not mean disabling ordinary load balancing. Keep the same serving/proxy path and comparable CPU/network allocation, and disclose any differences.

| Arm | Static layout | Behavior and purpose |
|---|---|---|
| H | Two TP16 instances, 32 chips | Required homogeneous large-instance baseline; no migration |
| S | Four TP8 instances, 32 chips, if qualified | Homogeneous all-small baseline; no migration; run whenever feasible |
| M0 | Two TP8 plus one TP16, 32 chips | Central gateway scheduling, no migration; same-layout control |
| MR | Same layout and initial routing policy as M0 | Recompute-only migration |
| MK-H | Same layout, routing and migration policy as MR | Qualified host-staged KV transfer |
| MK-I | Same layout, routing and migration policy as MR | Optional: qualified ICI KV transfer; may be deferred if difficult; never relabel a host fallback as ICI |

**Primary system comparison:** the frozen migration treatment versus H and S, reporting both results and the strongest tested homogeneous reference. The comparison against M0 is a separate migration ablation, not a replacement for homogeneous baselines.

Contrasts (core unless explicitly optional):

- **S versus H:** can smaller instances deliver better high-batch bulk throughput per allocated chip? Reuse existing batch/context evidence first; run only a few predeclared real-serving regimes needed to resolve the claimed advantage and its memory boundary. Head count, nominal bandwidth and batch size are not substitutes for measured throughput.
- **M0 versus H and S:** what does static heterogeneity plus gateway placement buy without migration? Compare against the strongest tested and qualified homogeneous reference, not only the easiest baseline to beat.
- **MR or a qualified KV arm versus M0:** does migration itself help on the identical resource mix? Keep the initial routing algorithm, eligibility rules, native engine limits and proxy path unchanged; only migration is enabled. Actual later routing/load may diverge as a consequence of migration and must be recorded.
- **MR versus MK-H:** is preserving and transferring KV worth its cost relative to recomputing it?
- **Optional, only if MK-I is qualified: MK-I versus MK-H, with MR retained as a reference.** What does ICI change in both handoff cost and whole-workload makespan? A faster data copy alone is not proof of a more useful serving system.

Qualify S by checking model fit and four disjoint valid TP8 placements; if feasible, it is part of the baseline set, not optional because H is easier to beat. Record an infeasible or unavailable reference explicitly and limit the claim. Report performance relative to each measured homogeneous layout rather than claiming an optimum over untested layouts. The core migration comparison can proceed with MR while host-staged KV work is qualified. MK-I is optional: if deferred, report its status and reason without treating its absence as an incomplete core evaluation.

Historical `cap128` is the starting reference for the prior TP16/W-SMALL configuration, not a proven optimal cap for every new instance mix. Use existing evidence and a bounded, disclosed development budget to select reasonable per-instance settings. Give baselines comparable tuning effort; freeze settings before evaluation. Never give only the treatment privileged future-length knowledge. Charge all allocated chips even if Large is underfilled. M0 must use the same legitimate fresh-traffic policy for Large, not an intentionally idle destination.

### Three focused experiment groups

1. **Instance efficiency and capacity boundary.** Establish the supported dense-batch operating region for TP8 and TP16 using real model execution and real conversational contexts. Report actual scheduled batch, padded shapes, context distribution, usable KV capacity, preemptions and useful tokens/s/chip. Compare equal-chip serving layouts rather than confusing one faster large instance with better aggregate efficiency. Reuse qualified measurements; no broad batch-size/topology sweep.
2. **Migration mechanism cost under load.** Compare recompute and each qualified KV path at matched source/destination configurations, request-history checkpoints and a fixed real background workload. Choose a small set of context checkpoints spanning the expected migration regime before timing. Measure proposal-to-destination-continuation time, source-release time, destination queue/reservation delay, rebuild/transfer work, bytes moved, control CPU and useful progress of unrelated resident requests. Report source/target load and repetitions; do not price stages by summing overlapping durations. Use teacher-forced real histories where needed to compare the same checkpoint, without regenerating or counting the prefix as new useful output. Isolated handoff timings explain mechanisms; they do not replace the bulk comparison.
3. **End-to-end usefulness and scope.** Run the controlled arms on the primary deep backlog through llm-d, then compare the frozen selected treatment against M0 and both qualified homogeneous baselines on narrowly chosen held-out workloads. Report which uniform layout is strongest in each workload rather than selecting only the weaker comparator. Do not run every parameter against every workload. A one-at-a-time threshold or destination-headroom sensitivity check is justified only to answer a specific unresolved fragility or bottleneck question; it is not an open-ended search for a winner.

### Workloads and information boundaries

Reuse the saved B1/W-SMALL manifest first: 1,763 requests and 721,595 useful output tokens in the historical experiment. Validate dataset provenance, unique prompt construction, token IDs, context feasibility and hash before reuse. Resolve ShareGPT data from the existing manifest or mounted `data/sharegpt_sampled_500.json` / `ShareGPT_V3_unfiltered_cleaned_split.json`; do not assume a local dataset file exists. Use fresh unique prompts within iterations and replay the same frozen manifest across matched arms, with request/KV state reset between arms.

Predeclare a compact workload set:

- **W0: primary mixed backlog.** The saved manifest is the first comparison, not a newly chosen favorable draw. Confirm that it remains deep relative to the combined cluster capacity.
- **W1: held-out long-context/growing-request workload.** Use disjoint real conversations, with declared prompt/output-budget distributions and enough observed growth to exercise the intended memory pressure. This asks whether any W0 benefit survives outside the development cohort.
- **W2: low-migration control workload.** Use disjoint compact real conversations where the frozen policy is expected to migrate little. This exposes gateway/controller overhead and harmful unnecessary migrations when there is little capacity benefit to obtain.

Declare selection rules and workload hashes before viewing treatment results. Keep useful-token budgets identical across arms within a workload; different workloads need not have equal totals. If controlled output lengths are imposed for comparability, record that workload law, preserve real prompt provenance and keep realized future targets out of online scheduling inputs. Do not substitute random token sequences. Report realized migration eligibility and coverage, including if W1 fails to trigger the intended condition; do not silently relabel or select a different workload after seeing a winner.

If W0 is too shallow for a common sustained interval at 32 chips, report its full makespan as such. Before treatment timing, define and freeze a deeper manifest from additional unique real conversations if a sustained-bulk claim requires it, preserving W0 as a separately identified result. Never duplicate cached prompts or change manifests within a matched comparison.

### Measurement and repeatability contract

Set `enable_prefix_caching=False` on every engine. Retaining a migrating request's own KV is not cross-request prefix caching. Use real weights at `/models/gemma-4-31b`, real vLLM serving and physical TPU execution. No simulators, analytical timing substitutes, synthetic token loops or random-tensor performance stand-ins.

Before timing, save a predeclaration containing hypotheses, exact contrasts, manifests, source/image/configuration hashes, routing and migration settings, selected transport, initialization/warmup policy, common sustained-progress boundaries, repetition count, arm order, exclusion rules and analysis method. Separate development/tuning observations from held-out evaluation. No code changes or threshold tuning between matched arms.

Use independently reset whole-workload runs as the replication unit. Collect at least three paired repetitions for decisive end-to-end comparisons, balancing AB/BA order; predeclare any additional narrow confirmation rather than repeatedly running until a favorable result appears. Report every run and paired effect, along with the chosen uncertainty summary and its small-sample limitations. Thousands of per-iteration rows are not thousands of independent experimental repetitions. Do not import a historical noise range as the uncertainty of this cluster experiment.

Primary results are end-to-end makespan T and useful throughput U/T for the same useful output U. Relative to baseline B and treatment X, report makespan reduction as `1 - T_X/T_B` and throughput increase as `T_B/T_X - 1`; do not interchange those percentages. Include all gateway scheduling, reservations, quiescence, replay/transfer, commit/reconciliation and output collection inside the measured interval. Record initialization separately, plus initialization-and-serving time. Define warmup consistently and keep its prompts out of the timed manifest.

Record every individual prefill/decode iteration: timestamps, phase/mixed work, useful/replayed tokens, context and batch/padded shapes where available, latency, throughput and TPOT where defined. Use real-prompt `max_tokens=1` prefill qualification as required; it is not the generation limit for bulk requests. Do not subtract an independent prefill run to invent pure decode time. Capture per-instance backlog, resident allocations, incoming reservations, actual source release, destination activation, migration proposal/approval/commit/decline counts, bytes, fallback coverage, preemptions and recomputation. Label host measurements separately from physical device timings, and keep instrumentation identical across arms.

Record centrally scheduled migration overhead and backlog at the gateway, so a gateway bottleneck cannot be hidden behind backend tokens/s. All headline runs must traverse the actual llm-d scheduling path for initial placement and migration. A direct-backend measurement may diagnose overhead, but cannot stand in for the system result. Report failed requests, timeouts, retries and unfinished work. Never shorten the denominator, discard slow requests or count replayed/duplicate tokens to obtain a speedup; incomplete arms are not valid winner evidence.

Assert per-request committed-prefix preservation, original generation budget/stop handling, no missing/duplicate delivery, and equal useful-token totals for fixed-output timing arms. Cross-TP floating-point execution may change future samples even with restored RNG; do not promise bitwise-identical future output solely from a shared seed. Qualify same-configuration continuation equivalence and cross-TP logits/output behavior with predeclared numerical criteria. Report divergences explicitly; equal output counts are not proof of exact trajectory equivalence. Unsupported continuation features must block those requests rather than reset their semantics.

### Usefulness and falsification criteria

Predeclare a minimum practically useful improvement from the project's deployment/complexity tradeoff before viewing timing results, and report the measured effect and uncertainty rather than promising a particular speedup. A single favorable run is preliminary evidence, not a usefulness claim.

- **Instance-mix benefit:** M0 beats the qualified homogeneous reference. This supports static heterogeneity/placement, not migration.
- **Incremental migration benefit:** MR or a qualified KV arm beats M0 after all online costs. Larger small-instance batches or earlier memory release without lower makespan do not satisfy this criterion.
- **System benefit:** the frozen migration treatment beats the strongest tested and qualified equal-resource homogeneous baseline with repeatable, practically meaningful improvement; report its comparison with both H and S wherever qualified. Beating M0 but losing to a uniform layout demonstrates a useful mechanism within the mixed layout, not an overall system advantage. If only a subset of workloads benefits, state that operating region and regressions rather than claiming a universal improvement.
- **Sustained benefit:** the common deep-backlog interval improves, not only terminal drain. If no common interval exists or only drain improves, narrow the claim accordingly.
- **Transport benefit:** any measured KV arm, including optional MK-I, must perform genuine transfers with reported coverage and improve end-to-end behavior relative to its qualified comparator; faster isolated transfer is only a mechanism result.
- **Negative or unresolved result:** no small-instance efficiency advantage, no actual migrations, destination bottleneck, harmful recomputation, gateway overhead, fallback domination, correctness failure, or an effect unresolved by run variability must remain visible. Stop the unsupported claim rather than automatically broadening the search. No gain is a valid research conclusion.

The report must distinguish a correct implementation, a working mechanism, a workload-specific speedup and a demonstrated system advantage. These are different levels of evidence.

## Bounded execution sequence and acceptance gates

1. **Complete Phase 1: contiguous subslice allocation and TPU runtime bring-up.** Implement the reproducible layout/environment launcher and satisfy its hard exit gate above: physical contiguity, correct local ranks and effective worker environment, actual device membership, all three real engines serving concurrently, scoped lifecycle behavior and recorded capacities. Do not replace this phase with a placement-group-ready check or a brief inventory. Qualify each homogeneous baseline layout before its timing runs.
2. **Complete Phase 2: repeatable deployment and Ray serving-instance discovery.** Deliver the configuration-driven deploy/reconcile, inspect and scoped drain workflow; qualify stable instance registration and the installed llm-d discovery bridge. Demonstrate real gateway serving for every logical instance, idempotent redeployment and stale-endpoint removal before migration work.
3. **Qualify continuation and ownership with recompute.** Implement the minimum native hooks and serving integration; use a short real ShareGPT batch that actually exercises migration. Trace each source proposal through gateway destination selection, reservation, continuation dispatch and ownership commit; verify that no engine-to-engine request-routing bypass exists. Exercise simultaneous proposals from both small instances, stale/duplicate proposals and a destination refusal. Cover generated-but-uncomputed tokens, remaining budget, penalties/stops/RNG where enabled, cancellation, in-flight retirement, concurrent destination reservations, lost acknowledgments and delayed commits. A successful new-request API call is not continuation proof.
4. **Qualify host-staged KV transfer; optionally qualify ICI.** Reuse the same handoff contract. Export real generated KV, validate layout conversion and destination import, and demonstrate resumed generation with unrelated instances still progressing. Record the actual transport and software boundary. Apply the optional ICI feasibility rule above; deferring ICI must not delay the core evaluation.
5. **Run the predeclared research comparisons after core qualification.** Execute the instance-efficiency, migration-cost and end-to-end groups above, including S whenever feasible. The headline comparison is the frozen treatment versus H and S. H/S versus M0 isolates the static layout/routing contribution; M0 versus MR isolates recompute migration; MR versus MK-H compares recomputation with host-staged KV transfer. Add MK-I versus MK-H only if the optional ICI path is qualified with actual transfers. Keep the gateway path and matched resources, collect paired repetitions, and evaluate the frozen selected treatment against M0 and both qualified homogeneous baselines on W1/W2 without retuning. Record unsupported arms, optional deferrals and unresolved outcomes rather than replacing core comparisons with easier ones.
6. **Conclude and restore owned state.** Report gain, regression or unresolved difference without a promised winner. Verify every engine's request count, native queue and allocator return to its measured idle state. Retain agreed long-lived weights/placement groups; remove only owned temporary resources and throwaway scripts, and update affected implementation documentation.

Submit distributed execution through `rcm_exec` / `RayJobSupervisor` with structured `[OMP_EVENT: ...]` streaming. Monitor with RCM status/wait/logs/ray-status, not agent-written Ray job polling. Follow trap/fix/resume without discarding accelerator state unnecessarily. Use Jujutsu for repository operations. No broad test suites or unrelated cleanup.

## Deliverables and definition of done

Deliver the reproducible deployment entry points and desired/resolved configuration artifacts; physical topology proof, resolved rank map and intended/effective worker environment manifests; concurrent TPU bring-up and scoped lifecycle evidence; Ray serving-instance registry records and qualified llm-d discovery bindings; source/connector and gateway changes; gateway scheduling/ownership traces for initial placement and every migration; continuation/transport qualification evidence; experiment predeclarations and frozen workload/source provenance; machine-readable per-run, per-request and per-iteration results; and a concise research report. The report must include equal-resource makespan/throughput tables with individual paired runs and uncertainty, sustained/drain results, small-instance batch/capacity evidence, migration-cost and gateway-overhead breakdowns, held-out behavior, and explicit answers to the three research questions and usefulness criteria. Use measured plots/tables, not predicted performance curves.

Completion requires the dedicated contiguous-subslice/runtime qualification and repeatable deployment/discovery phases to pass, a working end-to-end bulk path through llm-d, demonstrated migration on the fixed layout, measured same-layout no-migration and equal-resource homogeneous comparisons, repeatability evidence for headline claims, and verified resource cleanup. Report held-out results for the frozen policy and every feasibility limitation. Keep failed, incomplete and fallback-dominated runs visible and exclude them from winner claims. An unsupported continuation mode may remain an explicitly evidenced external blocker, not a fictional fallback success. Optional ICI may be deferred without blocking completion; report its status and reason rather than claiming a transfer result. A valid negative empirical outcome can complete the research evaluation; a working testbed without comparative measurements cannot. A positive usefulness claim requires the predeclared performance and correctness criteria, not scaffolding, transfer latency alone or an untested protocol.
