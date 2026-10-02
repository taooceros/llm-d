# In-place Gemma-4 31B reconfiguration: feasibility and local proof

Date: 2026-10-02. Branch: `research/pathways-reshard`. No TPU computation, deployment, package change, Kubernetes resource creation, or benchmark was performed. The only cluster reads were pod metadata and the requested import/version-only worker exec. See `runtime_versions.json` for the exact command and observed versions. Existing campaign JSONs were read locally.

## Decision

**No-go as a switch inside today's independent per-instance Ray runtimes. Conditional go for a dedicated persistent multi-controller JAX runtime; Pathways is a larger alternative, not the first experiment.** ICI resharding is supported, and the local prototype proves value-preserving TP8-to-TP16-to-TP8 mechanics. What is absent is a live engine/runtime ownership and mesh-rebuild protocol. `JAX_PLATFORMS=proxy` is not that protocol.

| Approach | Decision and blocker | Engineering effort [ESTIMATE], one experienced engineer |
|---|---|---|
| Existing engines, change their TPU environment/TP in place | No-go: each runtime discovers only its instance; live buffers cannot be handed between independent PJRT clients by `device_put`. Destroy/reinitialize loses the proposed resident-weights advantage. | Not a viable implementation without changing runtime ownership |
| Persistent eight-controller JAX runtime, engines as disjoint submeshes | Best conditional go: JAX explicitly permits independent computations in disjoint process subsets. Needs one global runtime, shared distributed array handles, collective transition rendezvous, replacement runner/scheduler state. | 1–3 days for isolated ICI proof; 2–4 weeks for end-to-end safe prototype; 4–8 weeks including transition/failure qualification |
| One Pathways client owning multiple in-process engine runners | Conditional go: one client sees all devices and can assign submeshes. Requires new backend deployment, matched images, mesh selection, removal of Ray worker runtimes, and per-engine config isolation. | 2–5 days for backend/ICI proof once approvals and compatible artifacts exist; 3–6 weeks for engine integration; 6–10 weeks including qualification |
| Several independent engine clients on a conventional single Pathways proxy | No-go: conventional cloud troubleshooting docs explicitly permit one client session. More Ray clients do not give submesh isolation. | Do not pursue this shape |
| Experimental Shared Pathways Service, separate proxy per client | Not established for this use case: multi-client support exists, but advertised requests allocate TPU **instances/slices**, not arbitrary subboxes of one 32-chip slice; cross-client resident-array transfer is not demonstrated. | 1–2 weeks [ESTIMATE] just to establish fractional-slice allocation and array interoperability; no justified total integration estimate until those blockers clear |
| Host-sharded snapshot/restore | Lower integration risk: compatible with today's engine lifecycle and does not require shared-runtime scheduling. A successful snapshot is measured below; restore load timing and LoadProfiler results remain pending. | See those assignments; not estimated independently here |

Effort ranges are judgement, not measurements. They assume no vendor/runtime bug and do not include approval lead time.

## What the Pathways flag actually does

Paths below are relative to `~/heterogeneous-serving-tpu` unless prefixed `hetero/` (assembled workspace). The repo's current Gemma backend is native Flax/NNX BF16 parameters with FP32 activation and KV state, enforced in `hetero/tpu_worker.py:131–139,158–232`.

* `vllm/vllm/envs.py:1476–1479`: `VLLM_TPU_USING_PATHWAYS` is inferred from `proxy` in `JAX_PLATFORMS`; setting a same-named variable alone does not select Pathways.
* `tpu-inference/tpu_inference/__init__.py:42–67`: imports `pathwaysutils`, initializes it before JAX operations, and eagerly resolves vLLM's platform. This is global process initialization, not engine-scoped.
* `platforms/tpu_platform.py:158–160`: device-name probing reports `TPU v6 lite` on Pathways to avoid early multiprocess IFRT access. This string is **not evidence that the hardware must be v6e**; it is also a v5e validation risk.
* `platforms/tpu_platform.py:236–274`: native multiprocess DP is incompatible with Pathways (`TPU_MULTIPROCESS_DP=1` rejected); `VLLM_ENABLE_V1_MULTIPROCESSING` must be zero. Single-process SPMD DP is the fallback, not four independent H/S/M schedulers.
* `platforms/tpu_platform.py:315–346`: `TPU_MULTIHOST_BACKEND=ray` still selects Ray executors; Pathways is not a special new executor here. A Pathways owner should use a uniprocess executor with remote devices, not reuse today's per-host Ray worker construction.
* `executors/ray_distributed_executor.py:96–119,245–282`: creates a compiled DAG and separate Ray worker actors. V2 and multiprocess executors are alternate process transports, not a weight-sharing service. Today's `hetero/engine_actor.py:312–362` expressly binds each instance to a Ray placement group and worker class.
* `models/common/model_loader.py:276–280` rewrites dummy loading to `pathways_dummy`. `pathways_dummy_loader.py:71–123,148–169` generates random arrays directly on the remote TPU mesh and postprocesses them; it is **not a real-checkpoint fast loader**, and cannot establish restored model correctness.
* `models/vllm/vllm_model_wrapper.py:185–207`, `layers/vllm/quantization/unquantized.py:103–106`, and `layers/vllm/process_weights/cleanup_sharding.py:97–100` contain explicit remote/dummy tensor handling. Ordinary non-dummy loading is not disabled by the flag.
* `runner/kv_cache_manager.py:1421–1464` already implements inter-runner KV transfer: Pathways uses `pathwaysutils.experimental.reshard`, native JAX uses `device_put`, followed by `block_until_ready`. It demonstrates a mesh-transfer primitive, not live TP reconfiguration.
* `platforms/tpu_platform.py:216–218` returns false from `can_update_inplace`; that hook is not a promise of `reconfigure(new_mesh)`.

### Multiple engines on one backend: distinguish a proxy from a shared service

[Google's troubleshooting guide](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/troubleshooting-pathways#debugging_common_errors) says a Pathways cluster maintains a session with one client at a time. Hence today's independent engine actor/worker processes cannot all connect to **one conventional proxy**.

There is newer, explicitly experimental [Shared Pathways Service](https://github.com/AI-Hypercomputer/pathways-utils/blob/main/pathwaysutils/experimental/shared_pathways_service/README.md): each client gets a **dedicated proxy pod**, clients request a TPU type/topology and number of instances, and clients wait when matching resources are occupied. This prevents an overbroad claim that Pathways never supports multiple clients. It does **not** establish that four TP8 engines can allocate subboxes inside our single `tpuv5e:4x8` slice, or that array handles can cross those client sessions.

One client can see all remote devices as addressable (`process_index()==0`, `local_devices()==devices()`), according to the [porting guide](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/porting-jax-workloads#process_index). Multiple logical runners on disjoint meshes **within that same process/client** are plausible [INFERENCE], not qualified vLLM functionality. `worker/tpu_worker.py:377–411` supports `sharding_strategy.device_indexes`; under Pathways the local device list is global, but absent indexes every engine selects the first N devices and overlaps. A mesh-aware owner must explicitly partition devices and serialize context-dependent initialization. Several `LLMEngine` instances sharing vLLM global config/platform/parallel state need qualification, not an assumption of thread safety.

## What a real Pathways deployment on this cluster requires

Observed packages: local JAX/JAXLIB **0.11.2**; worker JAX/JAXLIB **0.10.2**, libtpu **0.0.43**, importable `pathwaysutils`; pod image `vllm/vllm-tpu:v0.26.0` with recorded digest (`runtime_versions.json`). Merely importing the package says nothing about a live proxy or compatible server.

Primary requirements:

1. A Pathways resource-manager/head and IFRT proxy on CPU, plus eight Pathways workers owning the four TPU devices on each of our eight v5e hosts. Images: `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:<matching-tag>` for RM and workers, and `.../proxy_server:<matching-tag>` for proxy. Client image needs the matching JAX/JAXLIB and pathwaysutils. `JAX_PLATFORMS=proxy`, `JAX_BACKEND_TARGET=grpc://<proxy>:<port>`, initialization before JAX use, and proxy-side compiler flags are required.
2. Version matching is explicit in [JAX versioning](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/porting-jax-workloads#jax_versioning): server/proxy artifacts have `jax-<version>` tags. Do **not** use the tutorial's `jax-0.5.3` images against our 0.10.2 worker/client or local 0.11.2. Availability/access to a matching 0.10.2 or 0.11.2 artifact was not established; that is a prerequisite, not a reason to silently downgrade this Gemma fork. The newer inference guide mixes `latest` backend images with a `jax-0.8.0` JetStream client; pin a verified matched release, not those mutable examples.
3. [Current cluster setup](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster) recommends GKE **1.32.2-gke.1475000+**, a dedicated **n2-standard-64** CPU node (Toolkit currently provisions n4-standard-64), and **JobSet v0.8.0**. These are documented reference requirements; installed controller/GKE versions and spare CPU capacity were not queried. The namespace-scoped credential is not assumed able to install cluster-scoped CRDs/controllers.
4. [Batch jobs](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/batch-workload) use direct JobSets; the old [PathwaysJob operator](https://github.com/google/pathways-job) is deprecated in their favor. [Serving example](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-multihost-tpu-jetstream) uses **LeaderWorkerSet v0.6.1** (an older inference page uses v0.4.0). JobSet and LWS are alternative lifecycle wrappers, not both obligatorily needed for one experiment.
5. v5e is supported by an exact-shape [primary v5e 4x8 JobSet fixture](https://github.com/AI-Hypercomputer/pathways-utils/blob/main/pathwaysutils/test/experimental/gke/testdata/model_lite_tpuv5e_4x8.yaml): `--instance_type=tpuv5e:4x8`, eight workers, `google.com/tpu:4`, and selectors `tpu-v5-lite-podslice`, `4x8`. This fixture is evidence of intended support, not a benchmark; its example has two slices, so adapt to **one**. No need to change TPU generation or topology merely to test Pathways.
6. Headless DNS/services and network reachability among RM/proxy/workers, scratch GCS location, image-pull access, and workload identity permissions. The [porting guide](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/porting-jax-workloads#compilation_cache) requires bucket metadata/update and object list/create/get permissions for persistent compilation cache. A private scratch prefix/bucket must be approved.
7. **Pathways disables normal JAX persistent compilation caching** during `initialize()` and uses server-side GCS cache (or nonpersistent memory if scratch omitted). `/models/jax_cache` is therefore not automatically reusable under Pathways. Native multi-controller JAX can retain that cache, but compiler keys depend on device assignment/sharding, so TP8 and TP16 need distinct warm executables. [JAX cache documentation](https://docs.jax.dev/en/latest/persistent_compilation_cache.html) is the primary reference.

The existing Ray pods reserve all TPU Kubernetes devices. Replacing their libtpu owners with Pathways workers requires an explicitly approved reservation/redeployment plan; simply adding a second TPU-owning pod to occupied hosts will not work. Preview/access status is described in the [Pathways-utils README](https://github.com/AI-Hypercomputer/pathways-utils); access must be established separately.

## Persistent multi-controller JAX: independent submeshes are supported

The maintained [JAX multi-controller guide](https://docs.jax.dev/en/latest/501/multiprocess.html#running-on-a-subset-of-devices) explicitly states that subset meshes are useful for **different computations concurrently on different devices**, and that only processes participating in a sharding must execute its computations in the same order. This is not a requirement that all eight hosts execute every TP8 decode step. Each TP8's two controllers can run that engine's same collective sequence, while another disjoint pair runs another sequence. A shared global distributed runtime must still be initialized on all eight hosts before backend/device access.

For transitions, [cross-process device_put](https://docs.jax.dev/en/latest/501/multiprocess.html#transferring-data-across-processes-with-jax-device-put) uses ICI when available and must be issued on **every process participating in source OR destination**. All four controllers of a TP16 box therefore rendezvous for TP8-pair merges/splits. Controllers without addressable shards can keep a global array handle; they cannot simply lack the source object. Other disjoint boxes may continue serving if the transition does not include them. This is supported JAX semantics, not a measured vLLM integration result.

Today's `hetero/layout.py:121–150` configures a separate instance-local TPU rendezvous, and `hetero/tpu_worker.py:122–150,248–299` initializes/checks only that execution domain. `reconfigure` cannot resize that rendezvous by changing environment strings after initialization. Instead, retain one runtime spanning all 32 chips, and route execution into selected submeshes with distributed array handle ownership maintained across all controllers.

## Concrete mesh-bound objects and required cutover sequence

1. **Fence admission and drain to zero live requests, retire async work.** Use the engine actor's step/cache fences (`hetero/engine_actor.py:690–713`) and verify in-flight batches/async sampling have retired. Merely checking a request queue is insufficient. No mid-request KV migration is assumed in this estimate. Preserve external queue/routing but invalidate old block allocations and prefix-cache state.
2. **Free old KV and input/cache references before weight expansion.** `runner/tpu_runner.py:849–850,1268–1305` owns `kv_caches`, layer-to-cache indices, `KVCacheConfig`, device metadata buffer, and manager lifecycle. `runner/kv_cache_manager.py:719–755` constructs cache/group metadata; `runner/kv_cache.py:56–73,152–170` derives cache shape and `NamedSharding` from the mesh. At TP8, parameters alone consume about 7.8GB/chip; retained TP16 source plus fresh TP8 destination consumes about 11.7GB/chip [ESTIMATE], so keeping the old KV is dangerous on 16GB HBM. Layerwise buffer lifetime/donation needs a measured HBM budget.
3. **Construct destination sharding config and devices; reshard parameter leaves.** `layers/common/sharding.py:193–223,318–319` binds device count/indexes, TP/DP/CP strategy. `worker/tpu_worker.py:377–411,463–465` binds device selection and `TPUModelRunner`. `runner/tpu_runner.py:808–818,870–981` binds `devices`, `dp_size`, and mesh. Do not mutate global runtime topology or reopen libtpu.
4. **Rebuild model graph/static closures and parameter metadata without checkpoint loading.** `models/jax/gemma4.py:297,306–315,1048–1058` stores meshes and chooses KV projection specs based on TP; `models/jax/gemma4_mm.py:151,521,665–674` stores vision/model meshes if MM is retained. `models/common/model_loader.py:355–391,408–437` captures mesh-specific KV/hidden/logits/embed shardings, `graphdef`, and state treedef in JIT closures. `runner/tpu_runner.py:1194–1222` stores `state`, `state_leaves`, `model`, model/logits/embed functions, and sharded sampling RNG. Updating only `runner.mesh` would leave all of these stale. K/V spec can change if KV-head divisibility crosses TP8/TP16; infer each destination spec from a freshly abstract-constructed graph, not a universal same-axis rule.
5. **Rebuild scheduler/cache allocator and attention/input metadata.** `vllm/vllm/v1/engine/core.py:139–161` builds scheduler from `KVCacheConfig`, block size and hash block size. `runner/kv_cache_manager.py:686–717` can replace `InputBatch` and reconnect persistent batch manager. Attention duplicate-head count/cache groups/physical pages must be recomputed for new TP. Speculative/structured/MM managers (`runner/tpu_runner.py:825–841`) and request/input state cannot keep references to old mesh buffers. Disable optional features only by an explicitly limited experiment contract, not as a hidden compatibility fix.
6. **Compile/warm new execution before publishing.** `runner/compilation_manager.py:164–195,217–240` lowers/compiles/warms under runner mesh; `runner/decode_loop.py:100–101,379–380,487–488` passes static mesh into decode JIT. `runner/tpu_runner.py:1307–1329` captures/executes under mesh. Existing cache can amortize repeated **same assignment/config** launches; zero warm-up is not established. Rebuild executor/DAG bindings and engine capacity records, then atomically publish gateway/registry generation and reopen admission. On any failed rebuild remain withdrawn; do not resume with half-replaced mesh state.

The natural design is a transition transaction owned outside the decode hot path. No `__getattribute__` interception or per-step reconfiguration polling is needed.

## CPU prototype and real layout orders

`scripts/reshard_prototype.py` runs only with 16 CPU devices. It creates BF16 Gemma-like parameter schemas using real specs: gate/up `P(None,"model")`; down/embedding `P("model",None)`; Q `P(None,"model",None)`; global four-head K fallback `P(None,None,"model")`; attention output `P("model",None,None)`; fused local QKV `P(None,"model")`; norms/PLE and an optional default untied head replicated. Sources: `models/jax/gemma4.py:75–90,313–340,393–398,616–626,798–802` and `layers/jax/linear.py:147–170,231–267`. This is a reduced Gemma-like tree, not a byte-for-byte 31B parameter catalogue; optional PLE/untied-head branches need not be present in 31B.

Actual configuration subsequently arrived from ShardedRestore's successful preflight `raysubmit_m6ad4ZL824GfQuh6` (`initial-memory.json`, captured subset in `restore_comparison_partial.json`): hidden5376, intermediate21504,60layers, vocab262144,32query heads,16local KV heads,4global KV heads, local head_dim256/global512, global `attention_k_eq_v=true`, tied embeddings, no PLE or MoE, no KV-shared layers/double-wide MLP. Every sixth layer is full attention. Thus the prototype includes the correct active fused-local-QKV and global-K head_dim TP specs, but its optional replicated PLE/untied-head examples are not actual31B leaves. Local16KVheads are divisible by both TP8 andTP16; global4heads use head_dim fallback at both. No TP-dependent KV spec branch flip is expected for this checkpoint [INFERENCE from captured config + constructors], though rebuilding graph/metadata is still required.

CPU devices 0–7 hold S.small_a; 8–15 hold S.small_b, in native TP8 logical order. Each holds the **same full global model**, not complementary halves. The H mesh spans their union in native TP16 order. Two implementations are value-checked:

* Replica-aware merge assembles destination device shards from resident source-shard slices using JIT slicing and `device_put`, then builds a global array with `make_array_from_single_device_arrays`. No host parameter round trip occurs before equality verification.
* A DP2×TP8 representation of the identical source replicas is resharded with `jit(out_shardings=TP16)`. **Direct JIT with native S/H assignment orders fails with `ValueError: Received incompatible devices for jitted computation`**, despite the equal device sets. The exact observed error is retained in JSON. Explicit `device_put` to an ordered DP2×TP8 mesh matching the H assignment resolves this, after which JIT changes the replication/sharding. That generic reordered route may move more bytes than the optimized logical ledger.

H→S uses `device_put` for each destination TP8 mesh, with identity JIT output sharding and equality against the original complete tree. Both restored replicas are verified. On CPU/JAX0.11.2, equality passed. Default tree: **194,816 logical bytes**, **188,416 TP-sharded bytes**; exact minimum missing destination payload is **117,760 bytes S→H**, **306,176 H→S**. CPU compile-plus-execution times are recorded but are **not ICI measurements**. Device-list compiler warnings occurred during the reordered JIT route; they were not suppressed.

The prototype calls `resolve_layout` on the real inventory, preserving physical chip boxes and host order. It replays JAX's v5e mesh utility with relative device coordinates (native runtime coordinates checked by `hetero/tpu_worker.py:248–299`). [JAX 0.10.2 mesh source](https://github.com/jax-ml/jax/blob/jax-v0.10.2/jax/_src/mesh_utils.py#L120-L164) has the same TP8 and TP16 ring handlers used by local 0.11.2; no TPU ordering was newly observed. These order results are therefore **[INFERENCE] from code + measured inventory**, clearly labeled in JSON. A shared-runtime implementation must intentionally preserve these assignments: calling the utility on absolute global coordinates for a non-origin box may choose a different fallback ordering.

Actual H.large_a box is x=0..3,y=0..3; S.small_a x=0..1,y=0..3; S.small_b x=2..3,y=0..3. The other 16-chip group is translated by y+4. Thus the small meshes are **vertical halves**, not top/bottom row blocks. JSON includes every ordered coordinate and host IP for H/S/M.

For the first box (z=0 omitted), inferred native orders are:

```
S.a: (0,0) (1,0) (0,1) (1,1) (1,3) (0,3) (1,2) (0,2)
S.b: (2,0) (3,0) (2,1) (3,1) (3,3) (2,3) (3,2) (2,2)
H.a: (0,0) (1,0) (2,0) (3,0) (3,1) (2,1) (1,1) (1,2)
     (2,2) (3,2) (3,3) (2,3) (1,3) (0,3) (0,2) (0,1)
```

**S→H is not zero-communication in these orders.** For a sharded weight of size W, H rank i needs TP16 chunk i, while S rank r holds TP16 chunks 2r and 2r+1. Only six of sixteen H devices already contain their desired chunk; ten require a permutation transfer, giving **5W/8** minimum inter-device payload per 16-chip box. Reverse expansion needs **13W/8**: six chips need W/16 and ten need W/8. Norms/replicated leaves remain local.

Zero-transfer merge could instead use H order `(S.a[0],S.b[0],S.a[1],S.b[1],...)`. That interleaves equal-ranked replica holders and retains the needed half shard everywhere. Simply concatenating the two TP8 lists is not sufficient. Changing TP16 order for this property may hurt serving collective locality and changes compile-cache keys; it is a separate design/benchmark, not today's order. Even zero-ICI does not mean zero local HBM copying or zero rebuild time.

### Transition byte estimates (decimal GB, W=62.5GB)

These are **[ESTIMATE] extrapolations**, not measured wire traffic or exact 31B loaded-buffer sizes. All checkpoint bytes are approximated as TP-sharded BF16; actual replicated parameters, tied aliases, packed/quantized leaves, metadata, changing specs, and multi-hop routing require a real parameter catalogue/trace. Full per-device ledger is in `reshard_prototype.json`.

| Transition | Changed 16-chip boxes | Minimum missing destination payload [ESTIMATE] | Per changed chip received [ESTIMATE] | Largest received/chip [ESTIMATE] |
|---|---:|---:|---|---:|
| S→H | 2 merges | 78.125GB | six × 0; ten × 3.90625GB per box | 3.90625GB |
| H→S | 2 splits | 203.125GB | six × 3.90625; ten × 7.8125GB per box | 7.8125GB |
| S→M | 1 merge, other box kept | 39.0625GB | as merge; 16 kept chips receive 0 | 3.90625GB |
| M→H | 1 merge, other box kept | 39.0625GB | as merge; 16 kept chips receive 0 | 3.90625GB |
| H→M | 1 split, other box kept | 101.5625GB | as split; 16 kept chips receive 0 | 7.8125GB |
| M→S | 1 split, other box kept | 101.5625GB | as split; 16 kept chips receive 0 | 7.8125GB |

A single-source `device_put(S.a,H)` cannot exploit all locally retained data on S.b; the table assumes replica-aware selection or an equivalent replicated global-array operation. Do not present that ledger as the traffic of the simple generic route.

## Component-only cost C and uncertainty

[TPU v5e primary specification](https://docs.cloud.google.com/tpu/docs/v5e#system_architecture): **400GB/s bidirectional ICI per chip**, four ports, 2D torus. Use **200GB/s one-way** [INFERENCE from bidirectional convention], not 400GB/s in each direction, and not the full-pod bisection bandwidth for our subslice. This is a vendor peak specification, **not measured effective throughput**.

Component model [ESTIMATE]:

`C_components = T_reshard + 26s KV initialization + (54..64)s warm-up + ~10s gateway = (90..100)s + T_reshard`.

Only the inherited KV/warm-up/gateway durations are measured (recorded source clocks in `measured_components.json`). Using those durations in a new persistent-runtime design is itself an assumption. The ~54s endpoint came from the H long-context arm (`max_model_len=36864`); the S checkpoint warm-start arms at 16384 warm in about64s. No fictitious new measurement at 16384 is claimed.

Receive-side ideal lower bound `max_received_per_chip / 200GB/s` is 0.01953s for merges and 0.03906s for splits. Sensitivity at assumed 10–100% efficiency gives:

| Direction family | T_transfer receive-side sensitivity [ESTIMATE] | C_components [ESTIMATE] |
|---|---:|---:|
| S→H, S→M, M→H | 0.0195–0.1953s | 90.02–100.20s |
| H→S, H→M, M→S | 0.0391–0.3906s | 90.04–100.39s |

Those are **not statistical confidence intervals or guaranteed end-to-end bounds**. A practical component-only headline is **about95s ±5s**, plus unmeasured transfer/rendezvous overhead. Source egress, collective choice, hops, bisection contention, slicing/copy cost, per-tensor launches and cache misses can make the transfer take longer than the receive-side sensitivity. For a conservative diagnostic, serializing the entire full-slice payload at 200GB/s would take 0.391s for S→H and 1.016s for H→S [ESTIMATE]; even that does not bound framework launch/runtime delays. If only 1% peak is sustained, the receive-side terms become about1.95s/3.91s [ESTIMATE]. **The true C has no measured finite upper error bar**: drain time, runner/scheduler rebuild, distributed coordination and cold compilation are unknown additive terms. No invented rebuild latency is included.

For comparison, measured full redeployment C is407–445s (median419s), and partial M→S effective cost is about186s (campaign `reconfiguration_breakeven.typ` and `reconfiguration_evidence/e2e.json`). A hypothetical95s components-only switch is substantially below those values, but cannot be advertised as achieved. ShardedRestore's successful S snapshot `raysubmit_nKBkDxKXY3p8V4WG` reports **31,448,474,464 bytes/host**, **1,028 arrays**, and **51.5446s** for small_a/rank0 (`S-snapshot.json`, captured subset in `restore_comparison_partial.json`). This is snapshot/export time, **not restore/load time**. The host payload includes buffers and replicated content, so dividing by four gives about7.862GB/chip [CALCULATED], not an exact unique global parameter size. Restore load timings and LoadProfiler final output remain pending; no restore-C comparison is fabricated. The default recommendation remains measure restore first, then pursue shared-runtime reconfiguration only if its extra gains justify its failure/ownership complexity.

## Approvals required before real experimentation

Not performed, and outside existing H/S/M serving-run approval:

* A synthetic resident-buffer ICI microbenchmark owning a persistent global/subslice JAX runtime (and temporary Ray actors/placement groups), followed by deterministic cleanup. Existing approval is for serving deployment/runs, not silently a new synthetic TPU job.
* Switching device ownership from eight Ray workers to Pathways workers; recreating/reconfiguring workloads or TPU reservations to avoid two libtpu owners.
* New RM/proxy/worker pods, Services, JobSet/LWS objects, service accounts/RBAC/network exposure, and any installation/upgrades of CRDs/controllers.
* New server/proxy/client images, package changes, or JAX/JAXLIB version migration; explicit matched artifact validation first.
* GCS scratch/cache creation and its lifecycle/IAM changes; dedicated CPU capacity/node-pool creation if existing capacity is insufficient. No TPU node-pool/topology change is presumed necessary.
* Experimental Shared Pathways Service would additionally create **one proxy pod per client**, require its cleanup privileges, and need a separately approved allocation/interop test.

### Smallest useful on-cluster ICI microbenchmark proposal

Prefer **native JAX, one TP16 box/four existing hosts**, no model weights, no Pathways/Kubernetes/image changes. Through `rcm_exec`, use one blocking cluster flock around reservation, four controller launches, measurements, and release; check status before run and append deploy/release ownership records. With approval, temporarily reserve that box while the other 16 chips remain unused or owned by a disjoint serving run according to the integration owner's isolation policy. The first experiment need not exercise all32 chips.

1. Initialize one four-controller JAX runtime on the selected TP16 box. Record JAX/libtpu version, real `mesh.devices` order and physical coordinates. Create two TP8 submeshes in the existing native orders, with one replicated synthetic BF16 256MiB global parameter per engine, formed directly on-device from deterministic indices.
2. Prove independent subgroup execution: controllers of TP8a execute a different arithmetic/collective count from TP8b, then rendezvous all four before reshard. This isolates subgroup scheduling semantics without vLLM.
3. Reshard to TP16 and back using cross-process `device_put` (or same-assignment replicated-array JIT) issued on every source/destination controller. `block_until_ready` outside timing to ensure warm initialization; time each transfer with readiness synchronization. Keep compile/lower/create and checksum times separate. Verify per-device shard values/hash on every controller without downloading full buffers.
4. After one cold round, measure ten warm rounds [proposed repetitions, not evidence], then repeat with 1GiB global payload. Even two TP8 copies plus destinations stay well below HBM capacity; no KV needed. Report median and min/max wall time across controllers, logical missing bytes, achieved effective bandwidth, and HBM peak. Trace one round to confirm ICI rather than host-staged copies and count collectives/hops; profile setup requires approval if it writes external artifacts.
5. Include native-order merge, reverse split, and deliberately nested zero-ICI order as a control. If operations are near launch-noise at256MiB, increase payload within a measured HBM budget before drawing bandwidth conclusions. Full model-scale validation and eight-controller runtime integration are subsequent approved experiments, not claimed by this minimal proof.
6. Always destroy owned runtime actors/placement group and release the lock; no resident buffers or deployment remain. All commands must originate from an isolated worktree and go through RCM. This report deliberately does not pretend an absent cluster benchmark launcher has been implemented.

## Reproduction and evidence

From the committed isolated worktree:

```sh
XLA_FLAGS=--xla_force_host_platform_device_count=16 JAX_PLATFORMS=cpu \
  .venv/bin/python scripts/reshard_prototype.py \
  --inventory /tmp/hetero-inventory-only.json \
  --output results/restore/PathwaysReshard/reshard_prototype.json
.venv/bin/python -m pytest -q tests/test_reshard_prototype.py
```

The committed JSON embeds the original inventory and SHA256. To reproduce after `/tmp` changes, write `json.load(open('results/restore/PathwaysReshard/reshard_prototype.json'))['inventory']` to a local temporary JSON file and pass that file with `--inventory`. This requires no cluster access. Tests create a deterministic synthetic inventory with the same4×8 physical geometry and run the prototype in a fresh CPU subprocess at width16/layers1, verifying full value equality plus replica-byte balance and kept-half invariants.

Exact saved-inventory reproduction (local data extraction only):

```sh
.venv/bin/python -c 'import json,pathlib; p=pathlib.Path("results/restore/PathwaysReshard/reshard_prototype.json"); pathlib.Path("/tmp/pathways-reshard-saved-inventory.json").write_text(json.dumps(json.loads(p.read_text())["inventory"]))'
XLA_FLAGS=--xla_force_host_platform_device_count=16 JAX_PLATFORMS=cpu \
  .venv/bin/python scripts/reshard_prototype.py \
  --inventory /tmp/pathways-reshard-saved-inventory.json \
  --output /tmp/pathways-reshard-reproduced.json
```

Final scoped verification: `.venv/bin/python -m pytest -q tests/test_reshard_prototype.py` → **1 passed in12.14s**. The smoke invocation above produced both equality flags true and the committed JSON. Its replica-aware merge / reordered-JIT merge / reverse split CPU times were19.1766s /0.3350s /0.04356s, including compilation/dispatch and **not transferable to TPU performance**.

Files: `reshard_prototype.json` (local value proof, exact tiny byte ledger, full orders, six transition estimates, observed direct-JIT error, inventory); `runtime_versions.json` (read-only package/image evidence); `measured_components.json` (existing campaign clocks). No project-wide test/lint/build was run; integration owner owns that validation. CPU-only proof cannot validate actual TPU ICI speed, remote-Pathways support, subgroup failure semantics, or correct vLLM serving after mesh replacement.
