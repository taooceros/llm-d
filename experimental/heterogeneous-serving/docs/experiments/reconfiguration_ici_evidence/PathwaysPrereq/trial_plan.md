# Pathways prerequisite decision and unapplied trial plan

Date: 2026-10-02. Branch: `ici/pathways-prereq`. Evidence run ID: `readonly-pathways-prereq-20261002`.

## Decision

**No-go for the reference Pathways trial using only existing namespace resources/capacity. Conditional go after explicit approvals below.** The TPU shape and matching public images are available; the missing pieces are orchestration and CPU head capacity, not a demonstrated TPU incompatibility.

This is not a proof that Pathways fundamentally requires a cluster-scoped operator: an operator-free manually wired backend or an adaptation to the installed LeaderWorkerSet controller might work [INFERENCE]. Putting RM/proxy CPU workloads on TPU hosts could avoid adding a CPU pool [INFERENCE]. Those are unqualified alternative deployments, not a verified namespace-only trial. Merely setting `JAX_PLATFORMS=proxy` cannot turn the existing Ray workers into Pathways workers.

**Nothing was applied, scaled, deployed, pulled onto nodes, initialized on TPU, or benchmarked.** All cluster commands were read-only metadata/auth checks or import/source-only Python execs. `prereqs.json` contains each exact command, full stdout/stderr, exit code, and raw-output path. No `rcm_exec run` was performed.

## Observed prerequisites

| Check | Observed result | Evidence check IDs in prereqs.json |
|---|---|---|
| GKE control plane and nodes | `v1.36.4-gke.1391000` | 0, 5, 7, 8 |
| JobSet | CRD NotFound using default credentials; no jobset-system workloads and no JobSet controller in all-namespace deployment inventory | 9, 13, 23 |
| LeaderWorkerSet | CRD Established; controller `registry.k8s.io/lws/lws:v0.4.0`, 2/2 available | 9, 12 |
| CPU | One `e2-standard-32` node; 31.850 allocatable CPUs, 21.288 CPU requests, hence 10.562 unrequested CPUs [INFERENCE from observed accounting]; allocatable memory 122195648 Ki, requested memory 40512916352 bytes | 8, 15 |
| Recommended/reference head | Upstream template requests 16 CPU/100G proxy + 8 CPU/32G RM; cannot fit current CPU node concurrently with its existing workload. No `n2-standard-64` node exists. | raw/pathways_template.yaml.j2:62–65,101–104; 8,15 |
| TPU | Eight Ready `ct5lp-hightpu-4t` nodes, pool `tpu-v5e-4x8`, accelerator `tpu-v5-lite-podslice`, topology `4x8`, four `google.com/tpu` devices per host; NoSchedule TPU taint | 7,8 |
| Device ownership | Eight Ray worker pods each request/limit all four TPU devices on their host. Releasing Ray actors does NOT release the Kubernetes device allocation. | 3,14 |
| Ray scale semantics | Worker group `tpu-worker-group`: replicas/min/max=1, `numOfHosts=8`; scale one **group** to zero, not eight replicas. Helm-managed RayCluster; prevent Helm/GitOps reconciliation during trial. | 14 |
| Namespace token | nodes/CRDs reads Forbidden; namespace pod/exec works; patch RayCluster and create JobSet/LWS denied | 1,2,4,16–18 |
| Namespace scheduling policy | Default-credential read returns no NetworkPolicies, ResourceQuotas or LimitRanges | 26 (token refusal:22) |
| Worker packages | JAX/JAXLIB 0.10.2, libtpu 0.0.43, pathwaysutils 0.1.11 | 19 |
| Experimental reshard | `from pathwaysutils.experimental import reshard; reshard.reshard` exists. Signature `(x, sharding, *, donate=False, may_alias=None, cache_resharding_plans=False)` | 21 |
| Shared Pathways Service | Package files exist, but importing `isc_pathways` fails because `portpicker` is absent. It is NOT runnable as shipped. No package installation attempted. | 19,20 |

The initial `hasattr(pathwaysutils.experimental, 'reshard') == False` in check 19 is **not** absence of support: the empty package initializer does not eagerly import that submodule. Check 21 uses the actual fork import and establishes support. Shared-service failure is separate from conventional single-client Pathways, which is the proposed trial; no Shared Service dependency upgrade is needed for this trial.

### Image selection

Registry listing found **65 tags matching JAX 0.10.2 in each repository**, including the matching-date pair below. **No JAX 0.11.2 tag** was returned in either complete listing. Use the cluster's 0.10.2 client, not the workstation's 0.11.2 venv. Availability in registry does not prove nodes can pull or that the binaries work on this slice.

* RM and workers: `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20261001-jax_0.10.2@sha256:3b2c87dccc93266d84e7fcf5be84fcbcf97d7f837b4e8a0f318eb52cef4c0836`.
* Proxy: `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20261001-jax_0.10.2@sha256:99abf3bb5d536659fa50dfbecdf53c4e3dfd1a803406e560a2ce845e7289f401`.
* Single CPU controller/client: existing Ray head container `vllm/vllm-tpu:v0.26.0`, with a fresh, separate process. Verify its package versions/imports match the worker before trial; do not reuse a live JAX client or change packages in an existing container.

Commands: `gcloud artifacts docker tags list us-docker.pkg.dev/cloud-tpu-v2-images/pathways/{server,proxy_server} --format=json`; exact separate invocations and all tags/digests are checks 10–11. No image was pulled onto the cluster.

## Verified effects of the fork Pathways flag

Source paths relative to `~/heterogeneous-serving-tpu`; see `raw/fork_source_excerpts.txt` for captured source lines.

* `vllm/vllm/envs.py:1476–1479`: flag is inferred from `proxy` in `JAX_PLATFORMS`; a same-named environment variable is not the switch.
* `tpu-inference/tpu_inference/__init__.py:42–67`: initializes pathwaysutils before JAX operations; process-global, not engine-local.
* `tpu_inference/platforms/tpu_platform.py:236–263`: Pathways makes native multiprocess DP incompatible; requesting `TPU_MULTIPROCESS_DP=1` errors. Use single-process SPMD DP instead.
* `platforms/tpu_platform.py:271–274`: requires `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
* `platforms/tpu_platform.py:158–160`: bypasses physical device probing and returns the string `TPU v6 lite`. That is a workaround, not a restriction to v6e hardware.
* `models/vllm/vllm_model_wrapper.py:185–207`: dummy mode skips CPU parameter allocation and disables the CPU default-device context. Real checkpoint loading is not disabled.
* `models/common/model_loader.py:276–280`: only dummy loading is rewritten to `pathways_dummy`.
* `layers/vllm/quantization/unquantized.py:103–120` and `layers/vllm/process_weights/cleanup_sharding.py:97–115`: replace local tensor conversion with remote/dummy-aware handling; not a real-weight fast-restore implementation.
* `utils.py:125–126`: replaces ordinary per-device memory accounting with Pathways accounting.
* `runner/kv_cache_manager.py:1447–1464`: uses the experimental **submodule's** `reshard` then blocks; native mode uses `jax.device_put`. This proves an available primitive, not model reconfiguration.
* `platforms/tpu_platform.py:315–346`: Ray backend executor selection is still Ray. The Pathways flag does not remove independent runtimes or implement cross-client buffer handoff. `can_update_inplace` is false at :216–218.

For this isolated array-transfer trial use JAX directly, not vLLM scheduler/engine integration. One controller owns the entire remote device inventory; do not start four separate proxy clients for S.

## Approval boundary and exact objects

### Required external approvals for the reference route

1. Install a vendor-supported JobSet CRD/controller/webhook release compatible with GKE 1.36. This involves cluster-scoped CRD, ClusterRole/Binding and webhook resources plus controller workloads outside this namespace. **Absent today; no installation attempted.** Select and pin the operator release with the cluster administrator before installation; do not run a generic XPK cluster-create command against this existing cluster.
2. Add one `n2-standard-64` CPU node pool in `us-west1-c`, or approve another capacity arrangement that accommodates 24 CPU and 132 decimal GB for RM/proxy. This is a node-pool/billing change. Existing Ray head/client remains on the old CPU node. The draft selects `node.kubernetes.io/instance-type: n2-standard-64`, so it intentionally cannot schedule on current nodes.
3. Verify existing-bucket scratch permissions **without stopping serving**. Draft scratch prefix is `gs://hongtaozhang-tpu-model-cache-us-west1/pathways-artifacts/ici-pathways-trial`. The default service account has no Workload Identity annotation (check 24); existing read access through GCSFuse does not establish server scratch write access. Approve writes to this isolated prefix. If an IAM grant or new bucket is required, obtain separate approval before doing it; this study did not read/write objects or change IAM. Never delete model/cache prefixes during cleanup.

### Namespaced approvals

* Create `JobSet/ici-pathways-trial` using `draft_manifests/pathways-jobset.yaml`. It creates one CPU head pod (proxy + RM), eight indexed TPU worker pods through Jobs, and the JobSet headless Service/DNS. All are in `llm-d-optimized-baseline`. No extra LoadBalancer, ingress, PathwaysJob CRD, Shared Service, or new GCS bucket is proposed.
* Pull the two pinned Pathways images on nodes; this is a new backend image deployment, not a modification of existing vLLM images.
* Temporarily patch `RayCluster/tpu-ray-cluster-vllm-tpu` with `draft_manifests/ray-scale-down.json`, then restore with `ray-restore.json`. The token cannot patch this CR; an authorized operator must execute, or grant narrowly scoped namespaced rights. Do not delete worker pods directly: KubeRay recreates them.
* Run one fresh JAX 0.10.2/pathwaysutils 0.1.11 process in the existing CPU Ray head as the single client, with `JAX_PLATFORMS=proxy`, `JAX_BACKEND_TARGET=grpc://ici-pathways-trial-pathways-head-0-0.ici-pathways-trial.llm-d-optimized-baseline.svc.cluster.local:29000`. Check the actual coordinator DNS emitted by JobSet before connecting. Use the existing CPU image unchanged. A conventional proxy permits only one client session.
* Approve a complete serving outage and fresh weight/compile reload on rollback. Coordinate with the running research owner first; do not drain or abort another agent's experiment.

Drafts are adapted from XPK commit `c3c57e7ab85bc68cc8abb972f23689e1bc423d2f`, `src/xpk/templates/pathways_workload_create.yaml.j2`. Headless mode keeps RM/proxy as normal containers. Removed Kueue queue/priority dependencies, selected the existing TPU pool, set one `tpuv5e:4x8` instance and eight four-chip hosts, pinned matching images, and kept host networking plus coordinator/topology environment. XPK's type mapping explicitly maps v5litepod to tpuv5e (`raw/xpk_system.py:825–833`; instance identifier construction `raw/xpk_workload.py:292–295`). The manifests are **offline-validated YAML proposals, not server-admitted or runtime-qualified deployments**. JobSet release/schema and runtime smoke are approval-time gates.

## Trial sequence and rollback (commands below are proposals, NOT executed)

Wrap the entire future sequence in one blocking `flock /tmp/hetero-cluster.lock <approved-sequence-script>`, no outer timeout. Install/provision approved infrastructure before this lock/outage. Set a trap that deletes only this trial JobSet, waits for TPU worker termination, restores the saved Ray worker-group fields, and removes only trial scratch artifacts. Append deploy/release lines to `/tmp/hetero-cluster.state`.

1. Check `.venv/bin/python rcm_exec status` and ownership; require idle/no unrelated deployed actors. Capture current RayCluster YAML and its worker-group replicas/min/max/numOfHosts; suspend any Helm/GitOps rollout for this release during the trial. Record serving layout and output correctness baseline. Drain/release **owned** serving engines and placement groups using the normal supervised workflow after approval. Keep the Ray head alive.
2. Authorized operator: `kubectl -n llm-d-optimized-baseline patch raycluster tpu-ray-cluster-vllm-tpu --type=json --patch-file=results/ici/PathwaysPrereq/draft_manifests/ray-scale-down.json`. The patch tests the group name, lowers minReplicas and replicas from one to zero; maxReplicas remains one. Wait until all eight Ray worker pods have disappeared and no pod reserves `google.com/tpu` on the slice. Do not overlap Pathways workers with Ray device allocations.
3. Apply only the reviewed JobSet draft after approvals. Wait for all eight workers and head to be Running; inspect logs, discover coordinator DNS, and require the single client to see exactly 32 devices, all addressable, on the expected v5e topology. Abort startup after a predeclared 10-minute deadline [ESTIMATE operational limit], do not repeatedly churn pods or change packages.
4. Run the minimal benchmark below in one fresh client process, writing machine-readable timing/value/geometry evidence locally or to the isolated trial prefix. No model-engine traffic is routed to this backend.
5. Disconnect the client and delete only `JobSet/ici-pathways-trial` with cascading deletion. Wait for its TPU pods/Jobs and service to disappear and device allocations to clear. Clean only the isolated scratch prefix after evidence is copied. Do not change TPU node labels, taints, pools, or Ray templates.
6. Restore saved worker-group values; today's proposed patch is `kubectl -n llm-d-optimized-baseline patch raycluster tpu-ray-cluster-vllm-tpu --type=json --patch-file=results/ici/PathwaysPrereq/draft_manifests/ray-restore.json`. If original values changed before approval, restore the fresh snapshot instead. Wait for eight new worker pods/Ray nodes. Readiness may be 1/2 without a serving engine; use Ray inventory rather than an incorrect all-containers-Ready condition.
7. Reconnect dashboard port-forward if needed. Through RCM, inventory → preflight → deploy the saved serving layout → deterministic greedy correctness smoke and gateway health check. New worker pod names invalidate previous inventory. Serving hot patches live in pod writable layers and **will be lost** when pods are recreated; normal deployment must reapply the pinned research forks. Restore experiment readiness, not just Kubernetes pod count.

**Rollback on any failure:** preserve logs/evidence, stop only the trial client, delete only trial-owned JobSet/resources, wait for all trial TPU processes/pods to exit, restore saved Ray scale fields, inventory and redeploy previous serving layout, verify greedy tokens and endpoint health. Never resurrect Ray workers before releasing Pathways device allocations. If the approved trap cannot clear devices or restore Ray, retain the cluster lock and escalate to the operator; never release a lock while resident trial state remains. Leave newly provisioned CPU infrastructure/operator installed unless the user separately approves removal; deleting a CRD can destroy unrelated resources.

### Downtime estimate

**[ESTIMATE] Reserve a 30–45 minute outage window**, with startup capped at 10 minutes and the actual reshard experiment capped at 10 minutes. Ray pod recreation/image cache, cleanup and full reloading can add another 5–15 minutes [ESTIMATE]; rollback can dominate. Existing measured engine reconfiguration 407–445 seconds (median 419) is context from LoadProfiler, **not a measurement of Kubernetes worker scale-down/scale-up or Pathways**. Provisioning CPU/operator approvals happen before outage and are excluded. No Pathways startup or reshard latency was measured here.

## Smallest useful experiment

Own all 32 devices with one client but first exercise one physical 4x4 box (16 chips) while the other box has no benchmark arrays. Use physical `(coords, core_on_chip)` ordering recorded in evidence, not proxy enumeration order. TP16 spans all 16 chips; the two TP8 meshes split x=0..1 and x=2..3 over y=0..3. Mirror the second box later for whole-slice H↔S qualification; initial one-box measurement is not whole-slice reconfiguration C.

Generate a deterministic BF16 **Gemma-4 31B weight-shaped pytree**, using the real config/parameter shapes from the existing native model/snapshot metadata, including tied embeddings exactly once, attention (local/global) and MLP sharding. Do not use only one square GEMM or a toy vector and call it model-scale. Construct arrays on devices, not a full host checkpoint, and report exact unique leaf byte count. Reference shape/size context is feasibility.md: 60 layers, hidden 5376, intermediate 21504, vocabulary 262144, local/global KV heads 16/4. Approximate 62.8 GB unique BF16 weights and 7.8/3.9 GB per chip at TP8/TP16 are **[ESTIMATE]** until the trial records actual pytree bytes.

1. Create independent but value-identical TP8 copies A/B; fully block initialization and on-device fingerprints before timing. No KV cache or serving compilation in this experiment.
2. S→H: compare `jax.device_put` with `pathwaysutils.experimental.reshard.reshard(..., donate=False)` and then `donate=True` on fresh equivalent sources. Select and declare which TP8 copy supplies each leaf; other replica leaves become obsolete only after success. A merge of two identical model replicas is not concatenation along a parameter axis.
3. H→S: reshard to each TP8 target, creating two value-identical model copies. Measure both target transfers and the completed transition. Donate only after the last consumer of a source; blindly donating the first fan-out leg invalidates the second. Preserve logical values and tied leaves. Run bounded leaf batches if peak HBM needs it, report batch size and actual memory peaks; do not silently shrink weight volume. At 16GB/chip, retaining source + both destinations can exhaust HBM.
4. Block the entire output pytree inside each timed region using `jax.block_until_ready`. Record first transition separately, then at least five steady-state round trips; median/min/max per direction, both-leg total H→S, exact bytes, device geometry, donation/caching settings, host RSS and available HBM statistics. Avoid timing only asynchronous dispatch.
5. Verify shape/sharding invariants and two independent value fingerprints computed on devices outside timing. Include coordinate-dependent values in a small diagnostic tensor to catch replica/order errors; do not treat merely nonempty output as correctness. Compare deterministic checksums per real-shaped leaf and across both TP8 replicas; if diagnosing a mismatch, full small-tensor host comparisons are acceptable outside the benchmark.

This measures a model-scale **resident array transfer primitive**, not serving reconfiguration C. It cannot yet claim any fraction of the measured 419-second C removed: vLLM/Flax runner mesh rebuilding, KV lifecycle, kernel recompilation, scheduling and failure recovery remain separate integration work. Greedy-token equality with real Gemma weights belongs to the subsequent real-engine trial; synthetic BF16 checksums do not establish model correctness.

## Reproduction and validation

To repeat this read-only study, execute each `checks[].command` in `prereqs.json` from this worktree, preserving command stdout/stderr/exit code. Cluster-scoped commands deliberately use `env -u KUBECONFIG`; all privileged fallback uses were read-only. Worker exec probes import utilities and inspect source, never call `jax.devices`, `device_put`, `initialize`, or start service deployment. The command-output pair for check 25 is a negative lookup of one possible PathwaysJob CRD spelling; the decision does not rely on that spelling, only the complete API/deployment inventory and JobSet's exact CRD lookup.

Offline validation parses JSON/YAML, checks check-output exit codes, verifies eight × four requested chips, pins matched image tags/digests and guards the Ray group name; it does not contact the apiserver. Actual operator schema admission, image pull, scratch IAM, proxy session, physical topology and runtime reshard are explicitly unverified until an approved trial.

Sources captured locally: official interactive/batch workload pages (`raw/interactive.html`, `raw/batch.html`), XPK pinned generator/template/type mappings, Shared Pathways Service README. Context7 lookup returned unrelated Pathway ETL libraries; official Google documentation and pinned upstream sources were used instead. No claim is based on those unrelated lookup results.
