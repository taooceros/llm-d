# Compilation-overlap investigation: handoff

## Read this first

**The campaign is closed, not paused. Do not restart its old supervisors.** The attempted background compiler failed during TPU topology initialization while the source layout was serving. There is **no accepted live-overlap speedup, treated target-cache reuse, overlap switch cost, or new crossover**. The source workload completed. The separate cold/warm H→S controls remain qualified as observations, not as proof of overlap.

This handoff describes the next investigation; it does not authorize a new TPU run or implement a fix. The companion [failure report](compilation_overlap_failure.typ) explains the result. The existing [reconfiguration report](reconfiguration_ici.typ) provides the broader snapshot, graph, cache and ICI context.

## Goal and boundaries

Test whether preparing the next layout's executables on host CPUs can overlap execution of the current layout, without a second process acquiring the serving TPUs. Model: Gemma-4 31B on a 32-chip TPU v5e slice. Layouts: H = 2×TP16; S = 4×TP8; M = TP16 + 2×TP8. The failed live attempt served H while preparing S.

Do not equate these distinct proofs:

1. Captured executable keys match a replay.
2. Replay writes actual persistent-cache files with the expected payloads.
3. Replay succeeds while an existing process serves on the same host's TPUs.
4. The subsequently deployed target reads and successfully deserializes those entries.
5. The complete transition is cheaper after provisioning, serving interference, warm-up and ramp costs are accounted for.

The isolated replay established (1) and (2). It did not establish (3)–(5).

## Exact failure and code entrypoints

Worktree: `~/hetero-wt/OverlapCompile`, branch `overlap/restore-precompile`.

- Source/launcher revision: `9009fc7da5f03a57077ffdaa485d63a54d9190ef`. This is **not** proof of the loaded backend revision; the campaign recorded `UNFROZEN` / `loaded_runtime: null`.
- Final immutable artifact revision: `c22954164fcac8e1a2db32e8c59424af2e19eb9e`.
- Job: `raysubmit_69VBfS2EwyGPMt5B`.
- Background PID 572460 ran on `192.168.102.38`; serving PID 570736 was `large_a`, rank 0, on that same host. These are historical identities, not PIDs to signal now.

Observed call chain, with line numbers referring to the retained source:

```text
scripts/overlap_pilot.py:72       compile_task
scripts/overlap_replay.py:109     replay
jax.experimental.topologies.get_topology_desc(platform="tpu", ...)
jax._src.xla_bridge.make_pjrt_tpu_topology
xla_client.initialize_pjrt_plugin("tpu")
JaxRuntimeError: ABORTED: The TPU is already in use by process
with pid 570736. Not attempting to load libtpu.so in this process.
```

This happened **before the replay compile/profile loop**, not during graph compilation or target execution. Host-side compilation does not imply device-independent initialization.

`overlap_replay.py:82` sets `JAX_PLATFORMS=cpu`; checks at lines 101/143 inspect `xla_bridge._backends`. The explicit TPU-topology API nevertheless initializes the plugin. `overlap_pilot.py:96–99` already uses a CPU-only runtime environment including `TPU_SKIP_MDS_QUERY=1`, one Ray CPU reservation, and node affinity to a TPU worker host. Reapplying these flags is not a fix. A one-CPU Ray reservation also does not prove compiler threads are CPU-pinned or limited to one core.

A second, independent harness defect prolonged the failure:

```python
# Historical /tmp/overlap_physical_cache_gate.py:33
result = evidence['measurements']['pilot']['during_run']['result']
```

No background `result` existed. The gate raised `KeyError` before `StageRunner`, leaving daemon 4133323 as a zombie under stuck client 4133322. Terminating only that owned client triggered the existing EXIT cleanup. Dependent S→H arms were not submitted. The historical gate is retained in `final-operational-scripts-and-logs.tar.gz`; do not depend on a surviving `/tmp` copy or relaunch it.

## What can still be cited

| Evidence | Accepted scope | Not established |
|---|---|---|
| H→S cold C = 458.296556 s; warm C = 225.260380 s | Single cold-then-warm pilot observations; completed workloads and output equality | Causal effect of background compilation, repeated estimate, production outage |
| Warm startup: 2,088 keyed reads/successful deserializations and listener hits over 16 worker-phases; zero instrumented misses | Actual ordinary warm-start reuse | Reuse of an overlap-produced target cache |
| Isolated S/H replay: 261 matching executable keys and physical files per layout | Persistent output; file hashes/bytes agree when inspected across eight hosts | Eight independent caches, coexistence with serving, target consumption |
| Failed source W0: 1,763 requests / 721,595 output tokens; 490.7819 s control versus 502.8775 s failed attempt | Source workload completed | Successful-compile interference or a causal 12.0956 s slowdown |
| Snapshot write: 50–52 → 14–15 s per worker | Faster tested path with SHA-256 removed from the timed path | Equal-integrity A/B, one-second complete snapshot, zero-copy qualification |
| FastGraph restore graph: 14.631 → 1.766 s; SHA-on `load_model`: 39.436 → 26.454 s | Independently qualified graph optimization with output checks | Combined snapshot/cache/overlap transition saving |
| Persist small compilations: warm deploy 383.9 → 275.4 s | One startup comparison with effective cache settings checked | Full transition C or a new break-even threshold |

C is the gap from the last source completion to the **first measured target W0 dispatch**, plus the positive output-equivalent ramp deficit over the first 120 s. The steady rate is the median of 10 s bins in the middle 60% of target W0. Standard pre-W0 harness warm-up is included in the gap; post-W0 equality and cache inventory are excluded. Cold: 451.409829 + 6.886727 s. Warm: 218.716232 + 6.544148 s.

Prior target provisioning took 920.802339 s and included loading, references, snapshot creation and release. It is a separate upfront cost, not free work or pure compilation. Do not add/subtract arbitrary parts of it to create an overlap saving. Historical and current catalogues differ; the reducer suppresses six historical crossover candidates. Missing original S/H trace markers also prevent host/device attribution; missing device time is **unknown**, not zero.

## Evidence map

Raw root, relative to `~/hetero-wt/OverlapCompile`:

```text
results/overlap/OverlapCompile/
  controlled-hs-overlap-20261002b/
    failure-observed.json
    background-topology-error.log
    physical-gate-startup-error.log
    physical-gate-zombie-process-tree.log
  overlap_switch.json                 # observed controls; null overlap claims
  serving_impact.json                 # failed-attempt difference is not interference proof
  warmup_split.json / kv_split.json   # retain attribution limitations
  breakeven.json / startup_projection.json
  final-cluster-cleanup-proof.json / final-local-closure.json
  final-artifact-manifest.json / final-git-paths.txt
  final-operational-scripts-and-logs.tar.gz
  pre-sh-owned-bundle-graphs.tar.gz
```

The raw graph archive is 67,423,350 bytes, SHA-256:

```text
b6c3539ea40deeb66dfd5865d99d26e095b01552dc1ca56ea35a72c4129161b2
```

It contains 14,094 files and 4,698 descriptors, with verified MLIR/options hashes and no duplicate-key payload variants. This is a **post-run archive integrity check**, not evidence that runtime replay enforced those hashes.

Compact published evidence is adjacent to this handoff:

- [`reconfiguration_ici_evidence/OverlapFollowup/`](reconfiguration_ici_evidence/OverlapFollowup/): failure logs, final reports, generated tables and `manifest.json` with source hashes/compression boundaries.
- [`reconfiguration_ici_evidence/FollowupAudit/`](reconfiguration_ici_evidence/FollowupAudit/): `followup_final_overlap_claims.json` (audit `b6b5a24`) and `followup_artifact_seal_claims.json` (audit `2ea7783`), plus isolated replay, warm/cold, graph, restore and threshold audits.
- The immutable-artifact audit verified 317 declared Git payloads, 318 paths including the manifest, and 25 gzip rehydrations. The large archive is retained in Git, not external-only storage.

## Worktree and branch map

Preserve feature isolation. These are handoff pins, not assurances about future branch HEADs; none of the feature branches is being merged by this documentation change.

| Worktree under `~/hetero-wt/` | Branch | Pin | Purpose |
|---|---|---|---|
| `OverlapCompile` | `overlap/restore-precompile` | `c229541` | Sealed failed campaign and replay source |
| `AuditFollowup` | `audit/reconfig-followup` | `2ea7783` | Independent acceptance, cleanup and seal audits |
| `ResultsFable` | `helper/results-analysis` | `39d09ca` | Gated table generator |
| `FastSnapshot` | `restore/fast-snapshot` | `1f2dadf` | Snapshot/restore path only |
| `FastGraph` | `restore/fast-graph` | `1409b24` | Graph fix; audited runtime `df46a72`, earlier result `f317e57` |
| `WarmupSetFable` | `helper/warmup-set` | `0ebdf22` | Effective cache-threshold experiment |

## Safe local reproduction

These commands do not submit cluster jobs or modify sealed evidence:

```sh
WT="$HOME/hetero-wt/OverlapCompile"
SEAL=c22954164fcac8e1a2db32e8c59424af2e19eb9e
git -C "$WT" show "$SEAL:results/overlap/OverlapCompile/overlap_switch.json"
git -C "$WT" show "$SEAL:results/overlap/OverlapCompile/controlled-hs-overlap-20261002b/background-topology-error.log"

cd "$HOME/hetero-wt/ResultsFable"
.venv/bin/python -m pytest -q tests/test_reconfig_results_tables.py
.venv/bin/python scripts/reconfig_results_tables.py \
  --out /tmp/overlap-handoff-tables.md \
  --json /tmp/overlap-handoff-tables.json
```

The generator uses the existing evidence worktrees, including `ShardedRestore` and `SnapshotHelper`; keep them available. Inspect its input options before relocating anything. The recorded qualification at `39d09ca` was 21 tests plus 8 subtests. The report is built locally with:

```sh
# From the directory containing this handoff and the companion report:
typst compile --features html --format html compilation_overlap_failure.typ /tmp/compilation-overlap-failure.html
typst compile compilation_overlap_failure.typ /tmp/compilation-overlap-failure.pdf
```

## Next work, in order

### 1. Make missing-result failure terminate cleanly

In a new implementation worktree/branch, preserve the sealed failure and repair the gate contract: unsuccessful/missing background results must produce an explicit failed outcome, skip dependent target work, and enter bounded owned cleanup. Add a local regression using the retained failure shape. Do not repair this by inventing an empty successful result, ignoring cache failures, or loosening the success gate. Verify that exceptions before `StageRunner` cannot leave a client indefinitely waiting on a dead daemon.

### 2. Prove coexistence before another full benchmark

First inspect the **installed worker JAX/jaxlib/libtpu versions and actual topology API path**. Local source inspection alone is not worker proof. Candidate approaches are a genuinely topology-only compiler setup that does not acquire local devices, or compilation coordinated inside an already-owning runtime. Both are unvalidated alternatives, not established fixes.

The smallest next cluster gate, once authorized, is an owned serving process plus one real captured TPU executable compiled through the intended target-topology path on the same worker host. A CPU-backend toy, an idle-slice compile, environment flags, or a cache-key calculation alone does not pass it. Require:

- Source serving remains healthy and completes output checks while preparation runs.
- Background setup and compilation complete without trying to become a second local TPU owner; preserve setup/plugin logs and ownership evidence.
- Actual target keys, payload identity and persistent-cache output are recorded. Use a fresh process per replay target: JAX 0.10.2 previously latched the first cache directory in a process.
- Failure terminates within a bounded interval and releases only owned resources.

If this gate fails, stop and report the new blocker. Do not launch full target deployments merely to discover the same setup failure again.

### 3. Only then run matched measurements

Use cold, coverage-matched warm, and overlap arms on the same verified loaded runtime, catalogue, manifest, engine settings and warm-up protocol. Verify minimum compile time and entry size are effectively `0/0` in **every worker and the replay**, rather than trusting launcher configuration. Establish actual target cache reads and successful deserializations before claiming reuse; key equality or physical files alone are insufficient.

Preserve ESTALE/cache-write failures, directory initialization, rank/writer behavior, payload hashes and CPU placement. The previous warm evidence retained 80 ESTALE warning records; do not silently repair one arm's cache. Measure host CPU contention and serving throughput/latency under **successful** background compilation. Record provisioning and fanout separately and charge them when required on the critical path. Do not mix diagnostic tracing into only one timed arm.

Keep post-W0 correctness/inventory work outside C, keep required pre-W0 warm-up inside the gap, and never insert proof analysis between target deployment and measured dispatch. Report both timing endpoints and raw gaps. Randomize/repeat matched runs before generalizing a single contrast. Recompute crossover only with eligible current workload rates; never substitute historical rates from a different catalogue.

## Recorded closure and cluster safety

The final receipts recorded 32/32 TPU chips available; no live placement groups or actors among 296 retained experiment groups; all 24 owned cache/bundle paths absent on all eight hosts; zero snapshot bytes; all 13 owned local processes gone; RCM neither alive nor zombie; and successful cluster-lock acquisition/release. The pre-existing shared registry and inventory job's own supervisor were not leaks. No dependent S→H arm remained queued.

This is the **recorded campaign closure**, not a fresh statement of present cluster availability. A new experiment requires current ownership/inventory checks and authorization for mutations. Use the assembled workspace and `rcm_exec` for TPU work, serialize under `/tmp/hetero-cluster.lock`, preserve trap/cleanup receipts, and never abort or replace foreign jobs. Do not reanimate historical `/tmp` orchestration or treat a service's old “ready” label as proof of progress.
