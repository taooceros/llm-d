# Bulk-inference reconfiguration: plan and decision log

This file is the single source of truth for this study's scope, decisions, experiment queue and results. Chat transcripts and `local://` scratch files are not. Update it in the same change as any decision or result.

## Goal

The workload is bulk (offline) inference of Gemma-4 31B on the 32-chip TPU v5e slice (8 hosts × 4 chips). A job runs in one-way phases: many small requests, then a batch of long-context requests whose context exceeds what one TP8 instance can hold (about 52.9k tokens under full bf16), so a TP16 instance is mandatory for them. The original job model (small → middle → large) was replaced on 2026-10-05: under full bf16, S wins every class of that model (R3).

1. Measure how fast a layout switch can be made.
2. Find the phase sizes above which switching beats a static layout.

Layouts:

- S = 4×TP8
- H = 2×TP16
- Mb = TP16 + 2×TP8, the staged S→H intermediate

## Non-goals

- Latency SLOs and online serving.
- Model or engine-setting changes between jobs.
- Live background compilation (see the decision log).

## Decision log

| Date | Decision | Basis |
|---|---|---|
| 2026-10-05 | Same model and engine settings across jobs. The persistent compile cache stays warm, and per-layout weight snapshots are kept in host RAM. A layout's first-ever deploy is provisioning, not per-job cost. | User |
| 2026-10-05 | Pause compile-overlap work. Cold compile is a provisioning cost under the same-model assumption. | Warm cache: 2,088 keyed hits and 0 misses (`compilation_overlap_handoff.md`) |
| 2026-10-05 | Reject TP32 as the large-request layout. | KV pool 78,592 tokens, only +8.9% over TP16. W4 makespan 671.5 s vs H 347.1 s. Per-step time equals one TP16 engine. Evidence: `~/hetero-wt/ReconfigLevers` @ `4ac0bd9`, `results/tp32/`. |
| 2026-10-05 | Use a bf16 KV cache for **all** experiments. Weights stay bf16; activations stay fp32 unless the backend forces otherwise, in which case stop and report. The large-request class runs on TP16 with bf16 KV (about 145k tokens per instance; to be re-measured). | User |
| 2026-10-05 | Compile caches are pod-local, never `/models/jax_cache`. | GCS-fuse cache writes failed (Errno 116/5) just before the joining TP16 engine hung in E2 (`StagedFastRestore` @ `fdc9f7e`, `results/staged/e2_summary.json`) |
| 2026-10-05 | KV dtype is always named explicitly (`bfloat16` default, `float32` selectable); `auto` is refused by `EngineSettings` and the worker. | With `auto` the native `quantize_kv` cast is skipped, so fp32 K/V would meet a bf16 cache in RPA (`gemma4.py:409-412`, `kernel.py:1406`). |
| 2026-10-05 | bf16 correctness gate = exact within a layout + cross-layout tie gate (definition under Results/Q0). fp32 references are refused for bf16 runs (`engine_identity` key). | Measured TP8/TP16 bf16-KV next-token perturbation up to 0.92 nats (`results/hetero/20260918/noncache_activation_continuation.json`), so exact S-vs-H equality is not a valid bf16 gate. |
| 2026-10-05 | Q2 reuses the published harnesses unchanged (`reconfig_measure.sh S H`: plain deploy, max_model_len 16384, blind policy, W0 W5 W5 W0; `w4_campaign.sh 2 H` with `POLICY_ORDER=plain`, max_model_len 36864). The only configuration change vs the published fp32 rows is the KV dtype default. | Keeps the cost model comparable to `throughput_table_measured.json`. |
| 2026-10-05 | Native patch adds a widening cast in the RPA v3 kernel: when `q_scale` is unset and the loaded K/V block dtype differs from q, cast K and V to the query dtype (fp32) before both matmuls. Activations, matmul precision and KV storage stay as decided. | Q1 attempt 1: every S worker failed to compile `RPAd-p_64-bq_1_1-bkv_8192_8192-sw_1024`: "Mosaic … Bad rhs type: 128, 128" (fp32 q × bf16 K). The alternative, casting q to bf16, would change activations. `research/bf16-kv` @ `a48034a`, backend revision `9d8ff441…`. |
| 2026-10-05 | The pre-registered cross-layout tie gate is withdrawn (it failed in Q1; see Results). It asserts S/H near-equality that bf16 KV cannot provide in this stack. Replacement bf16 gate, adopted **after** seeing the failure: (1) within-layout bitwise equality across replicas and snapshot restores (hard); (2) layout symmetry on the 44-probe set: paired sign test of per-probe max\|Δlogprob\| vs fp32, S against H, p ≥ 0.05 and \|median_S − median_H\| ≤ 0.1 nat (hard; a single-layout fault would fail it); (3) argmax equal to fp32 on every probe where the fp32 top-1/top-2 margin > 0.5 nat (hard). The bf16-vs-fp32 deviation itself is recorded, not gated. Proceed to Q2: rates use `ignore_eos` fixed output lengths (`hetero/experiment.py:99,666`), so token identity cannot change them. | Q1 diagnostic `bf16diag-d7b1d91`; advisor fable-5-1 (twice). Whether bf16 KV's accuracy cost is acceptable is a user decision. |
| 2026-10-05 | **Full bf16 state for all experiments:** bf16 activations, bf16 KV, default matmul precision. This supersedes the fp32-activation clause above, and the RPA widening patch (`a48034a`) is withdrawn. Live migration stays in scope. Its numerical gate becomes relative: the migrated continuation must stay within the drift between a fresh source run and a fresh destination run. The absolute 0.25-nat bound no longer applies. | User decision, after the top-level decomposition of the 2026-09-16 bf16 continuation failure (`~/hetero-research/results/hetero/20260916/continuation_fidelity_M_dispatch_repaired.json`). With no migration at all, a fresh TP8 run and a fresh TP16 run already differ by 0.72–1.375 nats (argmax equal; top-32 overlap 31/32 in all 3 cases). Migrated-recompute output equals the fresh output exactly; host-KV import gives 0.53–1.25 nats. Same-TP migrated suffixes match exactly (12/12), and host-KV vs recompute suffixes give identical tokens (32/32). So the drift comes from bf16 arithmetic under a different TP split, not from migration code. Under fp32 state it is 0.0001–0.0096 nats (`20260918/minimal_fp32_state_continuation.json`). Q2 also showed the fp32-q × bf16-KV profile is 1.6–2.1× slower per step. |
| 2026-10-05 | **Job model changed to small → long-context** (user option (a)). The long phase uses requests that only fit a TP16 instance (> 52.9k tokens). New queue L0–L2; R4/Q3–Q5 paused until L2. | R3: under full bf16, S beats H on W0 (190.8 vs 306.4 s), W5 (754.3 vs 774.4 s) and W4 (147.7 vs 187.4 s), so an S→H switch in the old job model never pays. |
| 2026-10-05 | **Resume compile/execution overlap** (user). Decision-log row 2 ("pause compile-overlap") is reversed: the switch path is not actually warm. Decomposition of the P2 H restore deploy (`~/hetero-wt/Bf16Kv/results/switch/full-7a3aaeb/H-restore-deploy.json.gz`, worker 0) shows effective `min_compile_time_secs = 1.0`. Persistent-cache hits are 2/77 executables in `initialize_from_config` and 11/184 in `compile_or_warm_up_model`, so every start backend-compiles about 248 small executables (≈47 s summed, ≈43 s exclusive wall). Trace + lower add ≈30 s on H and ≈47 s on S, all host-side. Engine process start adds 21.5 s. The device-bound remainder (init_device, weight placement, executable load, warm-up execute, KV allocation) is ≈14 s. Queue O0–O3. | Top-level analysis of P2 startup profiles. The `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` env exists only in the old overlap scripts, not in the bf16 / run-switch deploy path. |
| 2026-10-07 | Overlap design (O1): persistent-cache thresholds 0/0 on every instance; target engines pre-spawned in CPU-only bundles during the source phase and parked before the first device touch; and an executable bank. Deviceless background compilation is rejected for the switch path. | Under the same-model rule, a deviceless compile only rebuilds what the warm cache holds, and it needs a libtpu lock bypass. JAX trace/lower caches are process-local, so only serialized executables spare the target its trace and lower. Advisor (fable-5-1): accept with changes, all folded into the O2 gate, which passed. |
| 2026-10-07 | **Explore a unified runtime per 4-host block** (user). One 16-chip JAX runtime, with the small instances running as 8-chip submeshes inside it, so the source process can compile the larger target layout in the background without a second TPU owner. Start with synthetic gates U0–U2; vLLM integration only after they pass. | Today each instance initializes its own sub-slice runtime (`hetero/layout.py:137-151` sets per-instance `TPU_TOPOLOGY`/`TPU_HOST_BOUNDS`/`TPU_PROCESS_ADDRESSES`), so a TP8 process cannot see the TP16 chips. `CompileExecutionSmoke` (`1b8149e`) already ran a 4-host 16-chip runtime that compiled an 8-chip target while executing; the lower (origin) half passed, and loading on the upper half aborted twice (cause unknown). Direct in-process jit on the upper half was never tested. |
| 2026-10-10 | **Unified runtime (U0–U3): the mechanism works, vLLM integration is not recommended now** (organizer recommendation; the decision is the user's). In one 16-chip runtime, two 8-chip host-row halves run independently: a 30 s stall inside one half's collective leaves the other at 0.97–0.99× its rate. All 4 processes compile the TP16 set in the background (11.8 s; serving −12/−13%). The in-HBM switch takes 12 ms 2×TP8→TP16 and 48 ms back, with 0 compiles and exact checksums. U1/U2 use host rows because libtpu 0.0.43 cannot compile ICI collectives on the origin x-half. | U1/U2/U1d results, deviceless reproduction (`research/unified-runtime` @ `945a152`, evidence `7f0a118`). U3: ≈1.5–3k LOC (vLLM per-process singletons, per-host Ray executor, our process-wide votes and bank), one failure domain per block, and C 27.5 → ≈ 9–10 s [INF] for ≈ +0.3 pp median job gain. |
| 2026-10-12 | **Provisioning folds into a layout's first deploy; there is no separate pass** (user). The first switch into a layout loads from the checkpoint, snapshots the weights and writes the executable bank in that same deploy, before serving. Later switches restore. Queue F0–F3, which includes a host-side checkpoint pre-read before T0. | O3 accounting ✔ (`OverlapSwitch/results/switch/full-c762edb-ov`): the measured 27.5 s excludes provisioning, which ran as a separate pass: normal deploy 255.4 s + snapshot 7.2 s/worker (rcm wall 13.9 s) + restore deploy with bank write 88.2 s, ≈ 351 s. Only the bank needs the 88 s restore deploy; P2/O0 need only normal deploy + snapshot. O3 saved 105 s versus P2: 43.0 s of recompiles removed (O0), 32.5 s of Ray + process start overlapped (pre-spawn), and 33.5 s of trace+lower+cache read moved into provisioning and replaced by a 3.6 s bank load. A first switch with pre-spawn and inline snapshot + bank is estimated at ≈ 237 s [INF]: normal-deploy engine init 216.5 s ✔ + drain/handoff/gateway ≈ 7.5 s + ramp ≈ 3.5 s + snapshot/bank 9.6 s. The separate pass gives 351 + 27.5 = 378 s. The 2026-10-02 snapshot analysis already recommended snapshotting each layout right after its first deploy. |
| 2026-10-12 | F-series settings (organizer): the cold first-switch target uses the normal loader **with FastGraph** (`fast_graph=true`, as every restore since P2), inline snapshot right after `load_model` and bank write after `compile_or_warm_up_model`. The F1 normal deploy is also the H persistent-cache warm-up: the worker pods were recreated with the 2026-10-09 Ray restart, and `/root/.cache/vllm/xla_cache` and `/root/.cache/omp_exec_bank` no longer exist on any of the 8 pods. C_first is therefore defined in the same-model regime: XLA cache warm for H (filled only by F1's H deploy; F2's S prep fills only S entries), checkpoint already read once by these hosts. The truly first cold compile is reported separately as F1's compile/warm-up addend. | FastGraph only shortens the abstract-graph trace (O3 graph_build 14.7 s normal vs 2.1 s restore ✔); it changes nothing executed on the device (`README` "Scoped fast graph construction"). Read-only `ls /root/.cache` on all 8 worker pods, 2026-10-12: only `pip`, `virtualenv` ✔. Advisor (fable-5-1) review of F2. |
| 2026-10-12 | **C_first is reported with the persistent compile cache warm** (user). A from-scratch deploy, i.e. after a pod restart with no XLA cache and a cold checkpoint read, is reported separately; F1 attempt 1 measured it at 861.5 s ✔. Local disk: the user freed space themselves (37 GB free on `/` at resume) and the 10 GB floor stays. `/tmp/omp_rcm` is now a symlink to `/dev/shm/omp_rcm`, made while RCM was Idle, so RCM telemetry no longer lands on `/`. | User answers after F1 attempt 1 stopped at the disk floor. In the hours before the stop, about 24 GB of new HuggingFace blobs (not from this study) were written to `/`. |

## Verified facts so far (fp32 KV; re-measure under bf16 where marked)

- Warm H→S, all timed from the source's last finished request (top-level recheck from raw timestamps):
  - Published gap 218.716 s; C = 225.260 s.
  - Without harness: about 160.5 s.
  - Engine path:
    - Process start before TPU use: 21.1 s.
    - `load_model`: 40.7 s, of which the SHA check is 20.2 s and graph build about 14.3 s.
    - `initialize_from_config`: 4.2 s.
    - Compile/warm-up: 50.2 s. Of that, JAX trace is 15.8 s and lower is 20.4 s, both sequential on the main thread.
  - Controller overhead: 32.5 s. Harness: 64.8 s.
  - The persistent cache cannot skip trace or lower, and `NUM_PRECOMPILE_WORKERS` only parallelizes `compile()`.
  - **Re-measured under full bf16 in P2:** production S→H C 132.2 s; the published-harness H→S return 273.7 s.
- FastSnapshot + FastGraph on S: `load_model` 5.78–5.96 s, greedy equality on 4 engines. A joining TP16 fast restore took 86.3 s. **Re-measured under full bf16 in P2/P3:** TP16 `load_model` 5.0 s; a joining TP16 is ready 126.7 s after T0, and its engine init is 90.4 s.
- Partial M→S: the measured serving gap was 9.3 s; effective C about 186 s versus 413 s for a full switch (`reconfiguration_breakeven.typ`).
- Cost model: `~/hetero-wt/ReconfigLevers/scripts/bulk_cost_model.py`. It reproduces the published break-even rows and the switching arms of the mixed jobs. Static mixed-queue rates must be measured, not derived. It is calibrated with fp32 rates; **recalibrate under bf16.**

## Process rules

- Code lives in `~/hetero-research` (local `main`, no remote) and its `~/hetero-wt/*` worktrees. Every cluster run names its exact commit.
- Before each cluster item, write its spec into this file: hypothesis, arms, commit, `rcm_exec`/`hetero_ctl` steps, endpoints, stop criteria and expected duration.
- Timed comparison items (Q3 onward) also need an advisor review of the spec before they run.
- Freeze code before timed arms. Never hot-patch during a timed arm.
- At most 2 fix attempts per blocker; then stop and record it.
- AGENTS.md cluster rules apply:
  - `rcm_exec` only; check `status` first.
  - Serialize on `/tmp/hetero-cluster.lock`.
  - Use `KUBECONFIG=$HOME/.kube/hetero-agent.yaml`.
  - Release owned resources and record cleanup receipts.
- Results go into the Results section with ✔ (verified from the artifact) or [INF], plus their artifact paths.

## Experiment queue

Q0–Q2 are done; they ran under the withdrawn fp32-activation, bf16-KV profile (Results). Everything below runs under full bf16 state.

| ID | Kind | Item | Exit criterion |
|---|---|---|---|
| R0 | local | On `research/bf16-kv`: full bf16 state, i.e. `activation_dtype` unset (model dtype bf16), `kv_cache_dtype=bfloat16`, default matmul precision; revert the widening patch; `engine_identity` and freeze checks follow. Gates: replicas within a layout and snapshot restores must match exactly (hard). Cross-layout drift is recorded, not gated. | Unit tests pass; commit; gate text updated here |
| R1 | local | Find and fix the gateway failure at about 255 requests in flight (W4 on H, Q2). Root cause from code and config, plus a local reproduction where possible. | Cause named with file:line; fix committed with a test |
| R2 | cluster | Full-bf16 provisioning for S and H (cold): record each instance's KV pool, generate per-layout references, check restore equality on **both** S and H, and record S-vs-H drift. | KV pools ✔; restore equality ✔ on S and H; or a blocker |
| R3 | cluster | Full-bf16 rates with the Q2 harness: W0 and W5 on S and H; W4 on H and on S (TP8 now fits W4's 36,864 context). Recalibrate the cost model. | Rates ✔; break-even phase sizes for C = 160.5 / 80 / 45 s |
| R4 | cluster | Requalify migration under full bf16, across at least 12 cases (prompt lengths 256–4096 and both directions). Hard checks cover only the migration implementation: same-TP suffixes must match exactly, and host-KV vs recompute suffixes must give identical tokens. Cross-TP drift is recorded, not gated (user, 2026-10-05: bf16 rounding drift doesn't matter). | Implementation checks ✔ |
| Q3 | cluster | Warm full S→H switch with fast restore and a pod-local cache; record every segment as in the verified breakdown above. | Production C ✔ |
| Q4 | cluster | Staged S→Mb→H (rerun of E2). | Effective C and join tail ✔, or a blocker |
| Q5 | cluster | Restrict precompile buckets to the shapes the bulk phases use (target: the 36 s trace+lower). Must show zero serving recompiles with `VLLM_XLA_CHECK_RECOMPILATION=1`. | Compile/warm-up time ✔ |

R4 and Q5 stay paused. Q3 and Q4 are superseded by P2 and P3 below (user approval 2026-10-05).

### Long-context job model (user option (a), 2026-10-05)

| ID | Kind | Item | Exit criterion |
|---|---|---|---|
| L0 | local | Build a long-context workload W6 from real data, following the `hetero-workload-campaign` conventions: prompts above 52.9k tokens (so they cannot fit a TP8 pool), with prompt + output ≤ the TP16 limit chosen in L1 (TP16 pool ≈145k tokens; the model supports 262,144 positions). Never truncate; drop what doesn't fit. Extend `bulk_cost_model.py` to the new job (k_s W0 manifests, then k_l W6 manifests). Arms: static H; static M (long requests on the TP16 while small requests run concurrently on the 2×TP8); S→H full switch; S→M partial switch. Static S is infeasible because long requests don't fit. Switch costs: the existing regimes (full ≈160.5 s production warm, staged ≈186 s effective, projected 80 / 45 s). | Manifest built and dry-checked; cost model extended; validation still reproduces the published rows |
| L1 | cluster | Full-bf16 rates: W6 on H, and a concurrent W0 + W6 mix on M (W6 routed to the TP16, W0 to the TP8s). TP16 `max_model_len` is set to fit W6; record the TP16 KV pool and per-request admission at that setting. 2 reps each. | Rates ✔, all requests complete |
| L2 | local | Break-even for the long-context job across the switch-cost regimes; identify which arm wins in each region of (k_s, k_l). | Region map ✔; says whether Q3/Q4 are worth running |

### Switch costs and mid-length band (user approval 2026-10-05, after L2)

Order: P0 and P1 are local and run in parallel; P2–P5 are cluster items and run serially. Full bf16 state, pod-local compile cache, fast restore (FastSnapshot + FastGraph), warm persistent cache. Before any timed arm, the switch harness must record every segment in the "Verified facts" breakdown: source last completion, drain, gateway, release, deploy start, engine-init phases, endpoint/gateway ready, first measured target dispatch, ramp.

| ID | Kind | Item | Exit criterion |
|---|---|---|---|
| P0 | local | Build W7, a real-data mid-length band: prompts 16k–48k tokens, so both a TP8 (52.9k pool) and a TP16 can hold them, with real output lengths; never truncate. The GovReport band below 52.9k is the first candidate; justify any other source. Size it so a manifest takes several minutes on S. Also W6x: W6 repeated for longer long phases (user approved repeats; label repeated documents). Extend `bulk_cost_model.py` with W7 arms. | Manifests built on the head; cost model validation still passes |
| P1 | local | Production switch harness without the benchmark harness: one in-process sequence (drain → release → deploy → gateway ready) with no evidence dump, restore-check job or pilot startup on the critical path. Dispatch comes from an already-running client that sends the target-phase workload as soon as the gateway routes. Gaps are measured from the client's last source completion to the first target dispatch and the first target token. | Local tests; a dry-run of the sequence logic |
| P2 | cluster | Full S→H warm switch (old Q3): 3 reps, S running W0 → H running W6. Record the production C with its segments, plus the harness-equivalent C for comparison with 225.26 s. | C ✔ with segments; output checks pass |
| P3 | cluster | Partial S→Mb switch (old Q4, re-scoped): 3 reps. Two TP8s keep serving W0 while the other two release and a TP16 joins; then W6 goes to the TP16. Measure the serving gap, the throughput of the kept TP8s during the join, the join tail (joiner first-served to steady), and the effective C. | Effective C ✔; join tail ✔; or a blocker |
| P4 | cluster | W0 on H at max_model_len 131,072 (TP16 with 512-token blocks), 2 reps, to pin static H and k_s*. | Rates ✔ |
| P5 | cluster | W7 on S and on H, 2 reps each; recompute the region map with W7 and the measured C values from P2/P3, replacing the projected regimes. | Rates ✔; updated region map |

### Compile/execution overlap (user, 2026-10-05, after P0–P5)

Target: move every host-side preparation step of the target layout (process start, trace, lower, backend compile) off the critical path, either by overlapping it with source execution or by caching, so that C approaches the device-bound remainder plus controller time. Full bf16 state; P1 `run-switch` harness; P2/P3 endpoints and segment accounting.

| ID | Kind | Item | Exit criterion |
|---|---|---|---|
| O0 | local + cluster | Effective cache threshold 0/0 on the deploy/run-switch path, verified in every worker's `compile_cache` report and the startup profiles. Re-run the P2 full S→H switch (3 reps) and the P3 partial S→Mb switch (3 reps). Expect near-total cache hits; report hits/misses per phase and the new C with segments. | C ✔ with segments; cache hits ≥ 95% of executables in both phases |
| O1 | local | Feasibility of host-only target preparation in a process that does **not** own the TPUs, on the installed worker JAX/jaxlib/libtpu (read versions without importing jax on workers). For each step, say whether it can run before release and what consumes it: (a) pre-spawned engine process up to the first device touch; (b) trace + lower of the target layout's executables for the target mesh without device handles (AbstractMesh / `lowering_platforms` / exported StableHLO), and how the target engine reuses the result (skips its own trace/lower; persistent-cache key equality); (c) serialized compiled executables (`jax.experimental.serialize_executable` or the PJRT cache payload) keyed by our own identity, so the target skips trace/lower/compile and only deserializes + loads; (d) the libtpu single-owner lock: where it triggers and what the deviceless topology path needs. Get one advisor review (fable-5-1 or astra-6) of the chosen design. | Design with evidence per step (file:line); local prototype on CPU where possible, with tests |
| O2 | cluster | Coexistence gate (from `compilation_overlap_handoff.md`): S serves W0 while the O1 mechanism prepares one real H executable set on the same hosts. Pass conditions: the source completes with output checks, measured throughput impact, no second TPU owner, and the target consumes the prepared artifacts after release (skipped trace/lower or cache hit, verified in profiles). Failure ends within a bounded time and releases only owned resources. | Gate verdict ✔; on failure, a recorded blocker, then stop |
| O3 | cluster | Integrated overlap switch: full S→H and partial S→Mb with the O1 mechanism running during the last part of the source phase, 3 reps each. Measure C with segments, source slowdown during preparation, and the region-map update at the new C. | C ✔; source impact ✔; region map ✔ |

### Unified runtime with in-process submeshes (user, 2026-10-07, after O0–O3)

Synthetic only (no vLLM). One 4-host block (2×2 hosts, 4×4 chips) initialized as a single JAX runtime, the same shape as `CompileExecutionSmoke`. Reuse its harness and evidence conventions (`~/hetero-wt/CompileExecutionSmoke`, `hetero/compile_execution_smoke.py`, `scripts/compile_execution_smoke.py`).

| ID | Kind | Item | Exit criterion |
|---|---|---|---|
| U0 | local | (1) Root-cause attempt for the upper-half abort from the retained smoke logs (libtpu/PjRt messages, device assignment, coordinates). (2) Confirm from `hetero/layout.py` packing that S, H, M and Mb each fit two 4-host blocks, i.e. no instance spans blocks. (3) Write U1/U2 as a new mode of the smoke harness, with CPU tests (multi-device CPU emulation). | Cause hypothesis with evidence; packing table; harness committed with tests |
| U1 | cluster | **Submesh independence:** in one 16-chip runtime, the lower and upper 8-chip submeshes each run a jitted program driven only by their own 2 processes, concurrently, with direct jit (no serialization). Variants: (a) both halves stepping; (b) one half idle or stalled for 30 s while the other keeps stepping; (c) each half compiles a new program while the other executes. | Both halves run with correct outputs; no cross-half blocking in (b); per-half step rates recorded; or the upper-half failure reproduced with libtpu logs, then stop |
| U2 | cluster | **In-runtime switch:** both halves serving a model-shaped program (TP8-sharded weights, ≥ 1 GB per chip); all 4 processes compile the 16-chip target program in a background thread; then stop both halves, reshard the weights to the 16-chip mesh with `device_put` (no host round trip), and execute the target. Measure: serving slowdown during the background compile (compared with `CompileExecutionSmoke`'s −19%), compile time, switch time (last half step → first target step) split into stop / reshard / first execution, output correctness. Also the reverse (16 → 2×8). | Switch time ✔ with segments; serving impact ✔; correctness ✔ |
| U3 | local | Only if U1 and U2 pass: inventory of the tpu-inference/vLLM assumptions that a submesh engine would break (`jax.devices()`, `process_count`, global barriers, distributed init, worker lifecycle), with file:line, an estimate of the integration size, and the failure-isolation implication (a dead process takes down the whole block). | Inventory and go/no-go recommendation |

### First switch without a separate provisioning pass (user, 2026-10-12, after U0–U3)

Target: measure a layout's first-ever switch when provisioning is folded into it, then shrink that cost. The same-model rule holds: the persistent compile cache is warm, and the target layout's snapshot and bank do not exist yet.

- Base: `overlap/prespawn-bank` @ `bbbeb4f` (code `d510c36`), in a new worktree `~/hetero-wt/FirstSwitch` on branch `research/first-switch`.
- Settings: full bf16; the P1 `run-switch` harness; O3 settings and endpoints.
- How a "first switch" is emulated without wiping warm state: each rep uses a fresh snapshot directory and a fresh bank namespace, which the run creates and then deletes itself. `/root/.cache/vllm/xla_cache` and the existing banks are never touched.

| ID | Kind | Item | Exit criterion |
|---|---|---|---|
| F0 | local | (a) A normal (checkpoint) deploy can write the executable bank right after `compile_or_warm_up_model`. It can also publish the weight snapshot before the endpoint registers: right after `load_model` if possible, otherwise after compile while the engine is idle. Both are engine settings, off by default. Show that a later restore deploy accepts that bank: the bank identity must not depend on the loader path, and if it does, fix it and explain why the fix is safe. Show that the snapshot passes the existing manifest and generation checks. (b) `run-switch` cold mode: the target pre-spawns, then loads from the checkpoint with (a) enabled. Includes the per-run snapshot directory and bank namespace, and their cleanup. (c) Add to the segment accounting: inline snapshot, bank write, and the normal loader's `load_model` sub-phases (checkpoint read vs device placement vs graph build) as far as the loader exposes them. CPU tests. | Tests pass; commit; F1/F2 spec written here |
| F1 | cluster | Gate: H normal deploy with inline snapshot + bank write → teardown → H restore deploy that reads both. Pass conditions: restored on every worker; bank used unanimously with 0 backend compiles and 0 misses or rejections; greedy output equal to the H reference. Record the time added by the snapshot and by the bank write. | Gate ✔ with times, or a recorded blocker, then stop |
| F2 | cluster | Cold first switch S→H, 3 reps. S runs W0; the target is pre-spawned, released at T0, loads from the checkpoint with inline snapshot + bank write, then serves W6. Then run one warm S→H switch that uses what the last cold rep wrote (expect O3's 27.5 s). Report C_first with segments and compare it with the [INF] 237 s. Report the amortized per-switch cost for N = 1, 3, 10 from measured values. | C_first ✔ with segments; warm follow-up C ✔ |
| F3 | local + cluster | Host-side checkpoint pre-read before T0. (a) Local: from the normal loader code (and `~/hetero-wt/LoadProfiler` if useful), split `load_model`'s 170 s into host-only work and device work. Design how the parked target, before `jax.devices()`, can read and shard the TP16 weights on the host; preferably it writes the TP16 snapshot files itself, so the target then takes the existing restore path. Budget host RAM and `/dev/shm` with the S snapshot present. Get an advisor review of the design. (b) Cluster coexistence gate: S serves W0 while the parked target does the pre-read. Measure the S throughput impact (same windows as O2: 2,661 / 4,070 tok/s), host memory, and the pre-read time. (c) Only if (b) passes and the mechanism works within ≤ 2 fix attempts: the cold first switch with pre-read, 2 reps. | (a) design + numbers; (b) gate verdict ✔ with source impact; (c) C_first ✔; otherwise a recorded blocker |

Order: F0 → F1 → F2 → F3. F3a may run alongside F0, since both are local. Stop at a failed gate. The organizer returns after F3 or at the first unresolved blocker.

Open, not blocking: consolidate the surviving branches and give `~/hetero-research` a remote.

## Results

### Q0 — bf16 KV default (local) — done

- Code: `research/bf16-kv` @ `7a06030` (`~/hetero-wt/Bf16Kv`), on `fdc9f7e`.
  - `EngineSettings.kv_cache_dtype` defaults to `bfloat16`; choices `bfloat16`, `float32` (`hetero/engine_actor.py:48,75`). The worker refuses anything else (`hetero/tpu_worker.py:239-241`). The worker/experiment/freeze dtype checks were already parametric in `kv_cache_dtype` and are unchanged.
  - New `hetero/agreement.py`; `scripts/snapshot_validate.py validate` records `engine_identity` (`model_build`, includes KV dtype), `tp_sizes`, replica equality, and with `--agreement-reference` the cross-layout gate. `scripts/fast_graph_ab.sh` no longer defaults to the fp32 reference. `scripts/bf16_provision.sh` is the Q1 driver.
- Tests ✔: `.venv/bin/python -m pytest tests -q -p no:cacheprovider --ignore tests/test_restore_tail_worker.py` → 228 passed, 8 failed + 8 errors. The 16 failures are the pre-existing baseline at `fdc9f7e` (same node ids; missing `results/hetero/*protocol*.json` files in the worktree). New tests: `tests/test_agreement.py`, 4 in `tests/test_snapshot_validation.py`.
- Backend support ✔ (code reading, no cluster): the deployed native patch only adds the `float32` CacheDType literal and fp32 activation plumbing; `bfloat16` is a native CacheDType. A non-`auto` cache dtype casts only K and V (`gemma4.py:409-412,483-489`, scale 1.0, `q_scale` stays None). Activations stay fp32 (`gemma4.py:558-604,986-991`). The RPA v3 kernel keeps fp32 q and output and promotes the bf16 KV operand in both matmuls (`kernel.py:447-456,530,1671-1672`). No activation or matmul-precision change is needed.
  - CPU check ✔: pure-JAX reference RPA, fp32 q, head_dim 256 and 512: bf16 vs fp32 KV relative L2 difference 5.7–6.1e-3.
  - Risk for Q1: fp32-q × bf16-KV is not covered by upstream kernel tests; first deploy may expose lowering issues.
- Precedent "TP16 bf16 KV 145k": `~/hetero-research/results/hetero/20260916/e2e_deploy_H_r2.json`, 567 blocks × 256, 566 usable = 144,896 tokens per TP16 ✔. It ran with `cache_dtype=auto` → bf16 and bf16 activations [INF: predates fp32-state commit `fcfc09974`]. Against the fp32 pools at identical block size/gpu_memory_utilization (TP16 283 blocks, TP8 103 blocks) the ratio is 2.00 ✔, so the 2× is the KV dtype alone. Prediction [INF]: TP16 ≈ 144.6–144.9k, TP8 ≈ 52.5–52.7k usable tokens.

**bf16 correctness gate (Q0 definition, `hetero/agreement.py`).** References: 32 W0 prompts (`data/manifests/W0.json`, sha `8a65e939…`), greedy, max_tokens 64, `ignore_eos=false`, generated fresh per layout under bf16 KV and keyed by `engine_identity`.

1. Within a layout (exact): all replicas of the same TP size give identical tokens; a restored deploy equals its own layout's bf16 reference token for token (`validate --reference`, refuses references without a matching `engine_identity`).
2. Cross-layout (S vs H), run from each side with `validate --agreement-reference <other>`. For every prompt where the two greedy outputs first differ at index d, the canonical engine is probed at `prompt + own[:d]` (`next_token_diagnostic`, top-32 logprobs). Pass needs all of:
   - tie: both layouts' tokens at d are in the top-32 and `max_lp − min(lp_own, lp_other) ≤ 1.0` nat, all finite;
   - at least 16 of 32 prompts identical;
   - at most 2 prompts diverge at d < 4.
   - Basis: fp32 S/H references agreed on 32/32 prompts; measured TP8/TP16 next-token perturbation under bf16 KV is ≤ 0.92 nats with the same argmax (`noncache_activation_continuation.json`). A wrong KV dtype, packing or corrupt page breaks every attention read: early divergence on most prompts, multi-nat gaps or tokens outside the top-32 [INF]. The 16/32 and d<4 counts are a loose bug screen, not calibrated; Q1 records the observed numbers.

### Q1 — spec (bf16 cold provisioning, S and H)

- Hypothesis: bf16 KV deploys on S and H without backend changes; per-instance pool ≈ 2× fp32 (TP8 ≈ 52.5k, TP16 ≈ 144.6k usable tokens); restore equality holds; S/H pass the cross-layout gate.
- Commit: `research/bf16-kv` @ `7a06030`. Engine settings: bf16 weights, `kv_cache_dtype=bfloat16`, fp32 activations, `highest`, max_model_len 16384, pod-local XLA cache (`compilation_cache_dir` unset), `startup_profile`.
- Steps (`HEAD_POD=<head> flock /tmp/hetero-cluster.lock scripts/bf16_provision.sh`, every stage one `rcm_exec run`, output `results/bf16kv/bf16prov-7a06030/`):
  1. `rcm_exec status` Idle; inventory if the Ray session changed; stage data into the head pod.
  2. preflight, tmpfs cleanup.
  3. S cold normal deploy (`weight_snapshot=off`) → `snapshot` → `validate` (S reference) → teardown (drain --force, gateway-down, release).
  4. H cold normal deploy → `snapshot` → `validate --agreement-reference S-ref` (H side of the gate) → teardown.
  5. S FastSnapshot+FastGraph restore deploy → `check --require-restored` → `validate --reference S-ref --agreement-reference H-ref` → teardown.
  6. H restore deploy → `check --require-restored` → `validate --reference H-ref` → teardown; EXIT trap removes `/dev/shm/omp_weights` on all hosts.
- Endpoints: KV pool per instance (`kv_capacity` in each `*-deploy.json`, `kv_pools.json`); cold vs restore engine-init and worker phases; gate verdicts and observed divergence counts.
- Stop criteria: any compile/lowering error, replica mismatch, restore fallback, or gate failure → stop, record, at most 2 fix attempts. Expected duration ≈ 75–90 min (two cold deploys ≈ 15–20 min each, two restores ≈ 5 min each, validations, teardowns).

Q1 attempt 1 (`7a06030`, run `bf16prov-7a06030`, 20:09–20:22Z): the S cold deploy failed in compile/warm-up after ~12 min. Mosaic rejected the mixed fp32-q × bf16-K matmul in RPA v3 on all S workers ✔ (`~/hetero-wt/Bf16Kv/results/bf16kv/bf16prov-7a06030/deploy-failure-excerpt.txt`). Cleanup ✔: drain, gateway-down, release S and H (all REMOVED), tmpfs cleanup ok, RCM Idle. Fix attempt 1: `a48034a` (decision log). Rerun as `bf16prov-a48034a` with the same steps.

Q1 attempt 2 (`a48034a`, run `bf16prov-a48034a`): S cold deploy succeeded (566 s; compile/warm-up 254–256 s), snapshot and S reference ✔. The teardown drain was then refused by the gateway: "administrative transition requires idle responses…". Cause ✔: the S gateway recovered E2's ownership ledger `/tmp/omp-hetero-gateway-S.json` on the head pod, with 32 W5 requests still `owned` from the hung E2 run. Cleanup ran (release, tmpfs) ✔. Fix: ledger + lock moved aside (copy at `results/bf16kv/stale-gateway-ledger-S-e2.json` and `/tmp/omp-hetero-gateway-S.json.stale-e2-20261005` on the head pod).

Q1 attempt 3 (run `bf16prov-a48034a-r2`, artifacts `~/hetero-wt/Bf16Kv/results/bf16kv/bf16prov-a48034a-r2/`):

| Item | S (4×TP8) | H (2×TP16) |
|---|---|---|
| KV pool per instance, bf16 ✔ | 827 × 64 = 52,928 (52,864 usable) | 2,270 × 64 = 145,280 (145,216 usable) |
| fp32 pool (published) | 26,432 (413 × 64) | 72,192 |
| Ratio | 2.00 | 2.01 |
| Cold normal deploy ✔ | 566 s (attempt 2; compile 254–256 s) | 568 s (compile 262 s) |
| Warm normal deploy ✔ | 378 s (compile 71–73 s) | — |
| FastSnapshot+FastGraph restore deploy ✔ | 148 s (engine init 130 s) | not reached |
| Replica equality ✔ | 4/4 identical | 2/2 identical |
| Restore equality vs own bf16 reference ✔ | 32/32 × 4 engines | not reached |

- Cross-layout gate: H side PASS (28/32 exact; divergences at generated index 1, 1, 9, 21; tie gaps 0.001, 0.045, 0.346, 0.086). **S side FAIL** ✔ (`S-restored-validation.json`): tie gaps 0.0, 0.041, **1.553**, **9.114** nats. At prompt 31 with the same 276-token prefix, S gives the H token −9.3 nats while H gives it −1.11 (`diagnostics` in both files). The run stopped there; EXIT cleanup ✔ (drain, gateway-down, release S/H, tmpfs).
- Offline comparison with the fp32 references (`results/restore/ShardedRestore/{S,H}-reference.json`): fp32 S vs fp32 H 32/32 exact; bf16 vs fp32 S 26/32 (first divergence at 0, 0, 1, 3, 48, 50); bf16 vs fp32 H 25/32 (0, 0, 1, 3, 9, 48, 50); bf16 S vs bf16 H 28/32 ✔.
- Advisor (fable-5-1): the 8-nat single-forward S/H gap is ~200× the prior TP8/TP16 bf16-KV measurement and points to a layout-dependent fault in the bf16 KV path (most likely the packed-K/V load or mask in the patched RPA kernel, TP16 side suspected) [INF]. Do not measure throughput before it is resolved.

**Q1 diagnostic — spec.** Hypothesis: one of the bf16 layouts matches fp32 within ~0.3 nat on fixed-prefix probes and the other does not (fault), or neither does (genuine sensitivity). Commit `research/bf16-kv` @ `d7b1d91`; driver `scripts/bf16_probe_diag.sh` (flock, one `rcm_exec run` per stage): fp32-KV S normal deploy → `snapshot_validate.py probe` → teardown; bf16 H → probe → teardown; bf16 S → probe → teardown. 44 probe points (`results/bf16kv/probe-points-q1.json`): the 4 divergence prefixes, every prompt at an empty prefix, and prompts 0–7 at a 32-token prefix; every engine probes every point (one-token top-32). Endpoint: per point, max |Δlogprob| over shared top-32 and argmax agreement for bf16-S vs fp32, bf16-H vs fp32 and bf16-S vs bf16-H. Stop criteria as Q1. About 35 min.

Q1 diagnostic result (run `bf16diag-d7b1d91`, `~/hetero-wt/Bf16Kv/results/bf16kv/bf16diag-d7b1d91/probe-{S-float32,H-bfloat16,S-bfloat16}.json`), 44 probes, max\|Δlogprob\| over tokens with p ≥ 1e-3 in either table:

| Pair | argmax equal | median | p90 | max | TV median / max |
|---|---|---|---|---|---|
| bf16 S vs fp32 ✔ | 41/44 (35/35 where fp32 margin > 0.5) | 0.109 | 1.24 | 2.97 (p21, d0) | 0.010 / 0.298 |
| bf16 H vs fp32 ✔ | 41/44 (35/35) | 0.125 | 1.06 | 8.53 (p31, d21) | 0.010 / 0.617 |
| bf16 S vs bf16 H ✔ | 42/44 | 0.097 | 1.35 | 8.16 | — |

- Replicas bitwise identical in all three deploys ✔. Sign test of per-probe deviation, S vs H: 21 vs 23, p = 0.88; \|median diff\| 0.016 ✔. The large outliers sit on different probes for S and H, and both layouts flip argmax vs fp32 where fp32 is near a tie (p27, p31 at d0).
- Verdict: symmetric bf16-KV sensitivity, no single-layout fault ✔ under the replacement gate (decision log). CPU emulation of the patched kernel on per-shard S/H shapes was bitwise S≡H (Mosaic not exercised) [INF: the S/H difference comes from TP8/TP16 fp32 differences before the bf16 cast, e.g. the global layers' head_dim-sharded k_norm/RoPE].
- Accuracy cost of bf16 KV vs fp32 KV ✔: 64-token greedy sequences differ on 6–7 of 32 prompts per layout; per-probe TV up to 0.30 (S) and 0.62 (H).
- Not done in Q1: H snapshot-restore equality (run stopped before it). It moves to Q3's prep.
- fp32-KV S deploy in the diagnostic: 525 s (cold fp32 compile 217 s); bf16 H warm deploy 385 s (compile 80 s); bf16 S warm 377 s (73 s).

### Q2 — spec (bf16 per-layout rates)

- Hypothesis: bf16 KV doubles per-instance pools (✔ Q1), so concurrency-bound phases speed up. Biggest effect expected on W4 and W5 on H [INF].
- Commit `research/bf16-kv` @ `d7b1d91` (bf16 KV default; native backend `9d8ff441…`). Harness unchanged from the published rows: `reconfig_measure.sh S H` (plain deploy, max_model_len 16384, blind policy `{"decode_length_blind":true}`, `run-arm --unfrozen-diagnostic`, ORDER `W0 W5 W5 W0` → run ids `rc-d7b1d91-{S,H}-s{1..4}`), then `POLICY_ORDER=plain w4_campaign.sh 2 H` (max_model_len 36864 → `w4-d7b1d91-plain-r{1,2}`). One `flock /tmp/hetero-cluster.lock` session; every stage is one `rcm_exec run`; HEAD_POD `tpu-ray-cluster-vllm-tpu-head-gxwsv`; data already staged in the head pod (sha-checked).
- Endpoints: makespan per arm (`results/pilot/*.json`), mean of 2 reps; completeness (all requests, useful tokens = manifest). Then `bf16_rate_table.py` → `bulk_cost_model.py --table` recalibration (validation stays on the published fp32 table) and break-even phase sizes for C = 160.5 / 80 / 45 s.
- Stop criteria: any arm incomplete or failed → stop that layout, record, at most 2 fix attempts. Expected duration ≈ 2.5–3 h.

### Q2 — results (bf16 KV, `research/bf16-kv` @ `591e3fe`, same harness as the published rows)

Note: the run ids carry `591e3fe` (the evidence commit on top of `d7b1d91`; no code change). Artifacts `~/hetero-wt/Bf16Kv/results/pilot/rc-591e3fe-{S,H}-s{1..4}_*.json`; driver log `results/bf16kv/q2-driver.log`. All W0/W5 arms complete with exact token accounting ✔.

| Workload | Layout | bf16 makespan (reps) ✔ | fp32 published | Change |
|---|---|---|---|---|
| W0 (small) | S | 455.4 s (450.7, 460.0) | 343.3 s (S W0 in E2: 337.6) | +33% |
| W0 | H | 779.8 s (779.4, 780.3) | 497.2 s | +57% |
| W5 (middle) | S | 1166.4 s (1168.6, 1164.3) | 1450.7 s | −20% |
| W5 | H | 1469.1 s (1483.3, 1454.8) | 1372.4 s | +7% |
| W4 (large) | H | **blocked**: 2 reps incomplete | 347.1 s | — |

- Mechanism ✔ (per-engine iteration stats in the same artifacts vs `~/hetero-research/results/pilot/rc-bab4f9e-*`): bf16 KV doubles the admitted batch where KV was the bound (W5: S max scheduled 8 → 14–16, H 20–21 → 41) and removes preemptions in W0 (S 1,483 → 0, H 526 → 0). But per-iteration time rises sharply: W0 on S p50 0.077–0.079 → 0.123–0.155 s at similar batch (63–78 vs 69–76); W5 on H 0.055–0.057 → 0.118–0.120 s; W5 on S 0.040–0.042 → 0.066–0.068 s. Only W5 on S gains net. [INF: the in-kernel bf16→fp32 widening of every loaded K/V block (decision log) and the doubled decode KV tile for packed bf16 make attention slower than with fp32 KV; not profiled.]
- W4 on H blocker ✔ (`results/bf16kv/q2-w4-failed/`, `results/pilot/w4-campaign-591e3fe.log`): rep 1 lost 36/904 requests (router 503, all within 0.2 s at t = 37 s); rep 2 lost 268 (router 500, within 1.3 s at t = 167 s, then every remaining request). Both bursts began with exactly 255 client requests in flight; failures return in 3–8 ms and never reach gateway admission (no trace rows). [INF: ext_proc/gateway saturation under the larger admitted concurrency; `hetero/gateway/extproc.py:176-183` documents the same 500/upstream "-" signature.] The completed part already took 472 s and 362 s, so bf16 W4 on H is not faster than fp32's 347.1 s [INF: a lower bound only]. No fix attempted; the gateway actor logs were empty.
- Deploy for W4 (H, max_model_len 36864, bf16 KV) ✔: 567 blocks × 256 per TP16 (144,896 usable tokens), vs 72,192 under fp32 (`results/pilot/w4-deploy-H-bf16.json`).

**Cost model recalibration** (`~/hetero-wt/ReconfigLevers` @ `055c1b9`: `scripts/bf16_rate_table.py` → `results/bulk_cost_model/throughput_table_bf16.json`; `bulk_cost_model.py --table … --json results/bulk_cost_model/model_bf16.json`). Published-row validation still runs on the fp32 table and reproduces all 10 rows (max \|Δk\| 0.009) ✔.

| C regime | k* W0, switch to S for the small phase (O = S tail 32.6 s) | k* W5, switch S→H for the middle phase |
|---|---|---|
| 160.5 s (warm production) | 0.60 manifests (fp32: 1.25) | none: H is 302.7 s/manifest slower than S (fp32: 2.47) |
| 80 s (projected levers) | 0.35 (fp32: 0.73) | none |
| 45 s (no trace) | 0.24 | none |

- One W0 manifest = 721,595 output tokens, so under bf16 a small phase of ≥ ~0.43M tokens (C = 160.5 s) pays for leaving H. Under bf16 there is no middle-phase reason to go to H at all. The large phase still needs a W4 rate (blocked) [INF: S may now hold W4's 36,864-token requests, since a bf16 TP8 pool is 52,864 tokens; untested].

**Cleanup receipt (after Q2)** ✔: RCM Idle; `snapshot_validate.py preflight` ok (it refuses registered engines or non-REMOVED placement groups), `snapshot_directory_bytes` 0 on 8/8 hosts, `files` finds no snapshot manifests (`results/bf16kv/q2-cleanup/`); H gateway down (`final-gateway-down-H.json`, removed=true); H placement groups REMOVED (`final-release-H.json`); `inspect` registry [] observed []. Head-pod leftovers: stale E2 S ledger copy `/tmp/omp-hetero-gateway-S.json.stale-e2-20261005`; failed W4 artifacts in `/tmp/omp_hetero/artifacts/w4-591e3fe-plain-r{1,2}` (copied locally).

**Decisions for the user**

1. bf16 KV with fp32 activations is slower per step in this kernel. Of W0/W5 on S/H, it helps only W5 on S. Options: (a) keep fp32 KV for rate experiments; (b) allow bf16 activations in attention (q cast to bf16, the kernel's native bf16 path; changes activations); (c) profile and optimize the widened kernel first.
2. The bf16 accuracy cost vs fp32 KV (6–7/32 greedy sequences differ; per-probe TV up to 0.6) is it acceptable? The cross-layout gate was replaced after it failed (decision log).
3. W4 under bf16 needs the gateway saturation at ~255 in-flight requests fixed before it can be measured; and S becomes a W4 candidate.

### R0 — full bf16 state (local) — done

- Code: `research/bf16-kv` @ `69a705c` (`~/hetero-wt/Bf16Kv`), on `4c652ca`.
  - `EngineSettings` defaults: `dtype=bfloat16`, `activation_dtype` unset (activations follow the bf16 model dtype; the native Gemma4 patch keeps `activation_dtype=None`), `kv_cache_dtype=bfloat16`, `matmul_precision=default`. Only two numeric profiles are accepted (`NUMERIC_PROFILES`, `hetero/engine_actor.py`): full bf16 `(unset, bfloat16, default)` and FP32-state `(float32, float32, highest)` for fp32 references. Mixed q/KV dtypes are refused, since the RPA v3 kernel has no mixed-dtype matmul path once the widening is gone.
  - Widening patch withdrawn: the native patch entry is now `rpa_v3_kernel_pristine` (original sha = the a48034a widened kernel `b97aa3be…`, patched sha = the pristine upstream kernel `bcd21fda…`). Deploy preparation therefore *restores* the pristine kernel on any host still carrying the widening, and the worker/experiment/freeze checks require the pristine sha on every worker. New `BACKEND_REVISION`.
  - Worker (`hetero/tpu_worker.py`), qualification (`hetero/experiment.py`) and freeze (`scripts/freeze_predeclaration.py`) checks compare the *effective* activation dtype (`native_backend.effective_activation_dtype`: override, else model dtype). Without this, `jnp.dtype(None)` would read as float64. `engine_identity` (`model_build`) carries `activation_dtype: null` and `matmul_precision: default`, so Q1/Q2 references (fp32 activations) are refused for full-bf16 runs.
  - `snapshot_validate.py validate --agreement-reference` records S-vs-H drift (`agreement.gated=false`, divergence indices, tie gaps) and no longer fails the stage. New `scripts/probe_drift.py` summarizes probe artifacts. On the Q1 diagnostic files it reproduces the Q1 table exactly (medians, nearest-rank p90, max, TV, sign test 21/23, p = 0.88) ✔.
  - Harness: `scripts/bf16_provision.sh` is the R2 driver (adds per-layout probes and the drift summary). `scripts/w4_campaign.sh` accepts `S` (max_model_len 36864). `reconfig_measure.sh` and `w4_campaign.sh` pass no engine settings, so they now run full bf16 by default.
- Tests ✔: `.venv/bin/python -m pytest tests -q -p no:cacheprovider --ignore tests/test_restore_tail_worker.py` → 252 passed, with the same 8 failed + 8 errors as the `4c652ca` baseline (missing `results/hetero/*protocol*`/fp32 continuation files, plus one restore_graph assertion). The ignored module needs `flax`. New: `tests/test_full_bf16_state.py` (profiles, defaults, effective dtype, no widening in the patch, and preparation restoring a widened copy of the local pristine kernel to `bcd21fda…` in a temp root), `tests/test_probe_drift.py`, and drift-recording/identity tests in `tests/test_snapshot_validation.py`.

**Full-bf16 correctness gate (replaces the Q0 gate and the post-Q1 replacement gate):**

1. Hard: within a layout, all replicas of a TP size give token-identical greedy outputs (32 W0 prompts, 64 tokens). A snapshot restore must equal its own layout's full-bf16 reference token for token (`validate --reference`, matching `engine_identity`).
2. Recorded, not gated: S-vs-H drift, measured as exact prompts, first-divergence indices and tie gaps (`validate --agreement-reference`); per-probe max\|Δlogprob\|, argmax agreement and TV on the 44 Q1 probe points (`probe_drift.py`), S vs H and each vs the fp32-state S probe (`bf16diag-d7b1d91/probe-S-float32.json`).

### R2 — spec (full-bf16 cold provisioning, S and H)

- Hypothesis: full bf16 compiles with the stock RPA kernel on S and H. Per-instance KV pools equal the Q1 bf16 pools (KV dtype unchanged; activation memory may shift them slightly) [INF]. Restore equality holds on both layouts. S/H drift is of the order measured in Q1 (median ≈ 0.1 nat; outliers up to several nats on near-ties) [INF].
- Commit `research/bf16-kv` @ `69a705c`. Engine settings `{"weight_snapshot":"off"|"read", "startup_profile":true, "kv_cache_dtype":"bfloat16", "matmul_precision":"default"}` (activation unset); restore adds `weight_snapshot_verify:false, fast_graph:true`; max_model_len 16384; pod-local XLA cache. HEAD_POD `tpu-ray-cluster-vllm-tpu-head-gxwsv`; Ray session `session_2026-10-03_14-58-37` unchanged since the last inventory, so `/tmp/hetero-inventory-only.json` stays valid.
- Steps: `KUBECONFIG=$HOME/.kube/hetero-agent.yaml HEAD_POD=… flock /tmp/hetero-cluster.lock scripts/bf16_provision.sh` (every stage is one `rcm_exec run`; run id `bf16prov-69a705c`, output `results/bf16kv/bf16prov-69a705c/`):
  1. `rcm_exec status` Idle; preflight; tmpfs cleanup.
  2. S cold normal deploy (the first deploy re-patches the native files and restores the pristine kernel) → `snapshot` → `validate` (S reference) → `probe` (44 points) → teardown.
  3. H cold normal deploy → `snapshot` → `validate --agreement-reference S-ref` (drift recorded) → `probe` → teardown.
  4. S restore deploy → `check --require-restored` → `validate --reference S-ref --agreement-reference H-ref` → teardown.
  5. H restore deploy → `check --require-restored` → `validate --reference H-ref` → teardown. The EXIT trap releases S/H and removes `/dev/shm/omp_weights` on all hosts.
  6. Locally: `kv_pools.json`, `drift.json` (`probe_drift.py`).
- Endpoints: KV pool per instance; cold vs restore deploy time and compile/warm-up; replica equality; restore equality on S and H; drift summary.
- Stop criteria: compile/lowering error, replica mismatch, restore fallback or restore inequality → stop, record, at most 2 fix attempts. Expected duration ≈ 80–100 min (two cold deploys ≈ 10 min each, two restores ≈ 3–5 min each, references, probes, teardowns).

### R1 — gateway failure at ~255 in flight (local) — root cause not pinned; diagnostics committed

- Code: `research/gateway-255` @ `0395e78` (`~/hetero-wt/Gateway255`), cherry-picked onto `research/bf16-kv` as `ed99e08`.
- "255 in flight" is not a limit ✔: it is the client's closed-loop steady state. `ADMITTED_CAP = 256` (`hetero/experiment.py:64`) feeds `asyncio.Semaphore(256)` and `httpx.Limits(max_connections=256)` (`experiment.py:221-222`), so every request starts with 255 others in flight. In r2, 309 of the 380 successful starts before the burst also saw exactly 255, and 256 concurrent requests were served at t = 0.14 s in both reps.
- Other candidate limits were checked and excluded ✔:
  - The gateway semaphore (`core.py:29`, 256) waits and never fails, and one permit was free at onset (gateway in-flight = 255).
  - The Envoy `omp_gateway` circuit breakers are set to 40,000 (`deploy/k8s/omp-router.yaml:149-153`).
  - The gRPC ExtProc server sets no `maximum_concurrent_rpcs` and advertises no MAX_CONCURRENT_STREAMS.
  - Engine actors use `max_concurrency=1024`.
- Local reproduction ✔ (did not reproduce the failure). Setup: Envoy 1.33.2 with the deployed `omp-router.yaml`, the real `ExtProcServer` and `build_app` over a sleeping fake core, and the real `_dispatch_http` client sending the real W4 bodies. It served 904/904 requests at 256 concurrency and 1,808/1,808 at 512, with `streams_failed` = 0.
- Failure signature (Q2 artifacts) ✔:
  - The refusals were a global, transient refusal of *new* requests only. In-flight requests all completed, and the failed requests left no gateway record or trace row.
  - r1: 36 × 503, starting at 36.77–37.02 s, recovering by itself within ~0.25 s.
  - r2: 268 × 500, starting at 166.7–168.0 s. Each failure returned in ~9 ms, so the closed loop drained the whole remaining queue; recovery was never observed.
  - Not retained: the gateway actor log, Envoy access logs and response bodies. A gateway 503 and an ExtProc-abort 500 look identical in Envoy's log.
- Ranked candidates [INF]:
  - r1 (503): a discovery health-lease expiry (`hetero/gateway/service.py:176-178`), with the lease set at `service.py:349-352` and renewed by the 5 s `_discovery_loop`. That loop shares `administration_lock` with the migration loop (`service.py:620-640`).
  - r2 (500): an ExtProc stream abort (`extproc.py:160-172`) or a non-Runtime dispatch exception (`service.py:111-117`).
  - The two reps may have different causes.
- Fix committed: diagnostics only, no limit or behaviour change.
  - Failed client records and summary rows now keep `http_status`, `response_protocol` (the gateway marker header, absent on Envoy local replies), headers and the first 1 KiB of the body.
  - The gateway counts ExtProc aborts (`extproc_aborts` → summary `gateway_extproc_aborts`).
  - The discovery 503 now names its reason (lease expired vs reconcile error).
  - Tests: `tests/test_ingress_failure_signature.py`, 4 tests. Three fail on `4c652ca`; the 512-stream ExtProc guard passes on both. Full suite at `ed99e08`: 256 passed, same 16 baseline failures ✔.
- Exit criterion not met: there is no cause with file:line and no fix. The next W4 run with `ed99e08` (R3) records which hop answers.

### R2 — results (full-bf16 cold provisioning, run `bf16prov-69a705c`, `research/bf16-kv` @ `69a705c`)

Artifacts: `~/hetero-wt/Bf16Kv/results/bf16kv/bf16prov-69a705c/` (committed `1e3ba13`); driver logs `results/bf16kv/r2-driver-69a705c{,-resume,-resume2}.log`. Native backend revision `81164462…`; NATIVE_BACKEND_READY on the first deploy, so the pristine-kernel restore applied ✔.

| Item | S (4×TP8) | H (2×TP16) |
|---|---|---|
| KV pool per instance ✔ | 827 × 64 = 52,928 (52,864 usable), all 4 | 2,270 × 64 = 145,280 (145,216 usable), both |
| vs Q1 (fp32 act, bf16 KV) | identical | identical |
| Cold normal deploy ✔ (hetero_ctl elapsed / RCM stage) | 486.3 s / 511.7 s; engine init 458–462 s; compile/warm-up 173.5–176.2 s | 515.9 s / 564.8 s; init 499 s; compile/warm-up 210.6–211.1 s |
| FastSnapshot+FastGraph restore deploy ✔ | 147.6 s / 184.6 s; init 128.7–130.4 s; compile/warm-up 71.6–72.2 s | 154.9 s / 176.5 s; init 137.6–138.1 s; compile/warm-up 80.4–80.6 s |
| Restored on every worker (`check --require-restored`) ✔ | 8/8 | 8/8 |
| Replica equality ✔ | 4/4 identical (cold and restore) | 2/2 identical (cold and restore) |
| Restore equality vs own full-bf16 reference ✔ | 32/32 × 4 engines | 32/32 × 2 engines |

- Cold compile/warm-up under full bf16 is shorter than under fp32-act/bf16-KV (S 174 s vs 254–256 s; H 211 s vs 262 s) ✔. The pod-local caches held only fp32-activation entries, so these were cold compiles [INF: no cache-hit counters read].
- S-vs-H drift (recorded, not gated):
  - Greedy, 32 W0 prompts × 64 tokens: 19/32 identical; first divergence at 0, 0, 0, 0, 1, 1, 2, 3, 5, 9, 15, 38, 46 ✔.
  - Tie gaps at the divergence point: up to 4.19 nats from the H side and 10.31 from the S side (`agreement` in `prep-H-reference.json`, `S-restored-validation.json`) ✔.
  - Probes (`drift.json`, 44 Q1 points), versus the Q1 rows (fp32 activations, bf16 KV):

| Pair | argmax equal | median max\|Δlp\| | p90 | max | TV median / max | Q1 median / max |
|---|---|---|---|---|---|---|
| full-bf16 S vs H ✔ | 39/44 | 0.513 | 3.07 | 5.37 (p10, d0) | 0.040 / 0.826 | 0.097 / 8.16 |
| full-bf16 S vs fp32 S ✔ | 38/44 | 0.515 | 2.96 | 8.09 (p31, d21) | 0.043 / 0.585 | 0.109 / 2.97 |
| full-bf16 H vs fp32 S ✔ | 39/44 | 0.551 | 3.27 | 5.39 (p10, d0) | 0.041 / 0.822 | 0.125 / 8.53 |

- Symmetry ✔: S vs H deviation from fp32, sign test 21/23, p = 0.88; \|median diff\| 0.036 nat. No single-layout fault.
- Accuracy cost vs the fp32-state references (64-token greedy) ✔: S 18/32 identical, H 21/32 (Q1 bf16-KV only: 26 and 25). Full bf16 moves the median next-token deviation from ~0.1 to ~0.5 nat.
- Process incident ✔: the first driver attempt stopped after `prep-S-reference`. Its idle check parsed `rcm_exec status --json`, which embeds the last Ray job's log tail (≈ 33 MB line) and was cut off at 72 MiB while `/` was full (2.5 GB RCM telemetry). The EXIT cleanup refused to run for the same reason, which left S deployed. Then the dashboard port-forward died, which stranded the first resume. Fixes:
  - RCM telemetry gzipped to `/tmp/omp_rcm-telemetry-upto-20261006T0150.jsonl.gz` and truncated; go-build and uv caches pruned.
  - `bf16_provision.sh` @ `a104ebb`: text idle check plus `RESUME_FROM`/`RESUME_LAYOUT`.
  - Port-forward restarted as a managed service.
  - The run resumed at `probe-S` with S still deployed (`inspect`: 4/4 ready). No `hetero/` code changed, so the runtime revision and `engine_identity` are unchanged.
- Cleanup receipt (R2) ✔: final drain, gateway-down and release of H; S released after its restore; cleanup release of S and H; tmpfs cleanup (`cleanup-tmpfs.json`); RCM Idle.

### R3 — spec (full-bf16 rates with the Q2 harness)

- Hypothesis [INF]: without the fp32-q widening, per-iteration time falls back to or below the fp32 rows, while the bf16 pools keep the 2× admitted batch. W0 on S and H should then be no slower than fp32 (343.3 / 497.2 s), and W5 should beat fp32 on both layouts. W4 should complete on H, and on S (TP8 pool 52,864 ≥ 36,864).
- Commit `research/bf16-kv` @ `ed99e08` (full bf16 + R1 diagnostics); native backend `81164462…`. Harness unchanged from Q2: plain deploy, no engine-settings override (so full-bf16 defaults), blind policy `{"decode_length_blind":true}`, `run-arm --unfrozen-diagnostic`.
- Steps, in one `flock /tmp/hetero-cluster.lock` session (`KUBECONFIG=$HOME/.kube/hetero-agent.yaml HEAD_POD=tpu-ray-cluster-vllm-tpu-head-gxwsv`; every stage is one `rcm_exec run`; driver log `results/bf16kv/r3-driver.log`):
  1. `scripts/reconfig_measure.sh S H` (max_model_len 16384, ORDER `W0 W5 W5 W0`) → `rc-ed99e08-{S,H}-s{1..4}`.
  2. `POLICY_ORDER=plain scripts/w4_campaign.sh 2 H S` (max_model_len 36864) → `w4-ed99e08-plain-r{1,2}` on H, then on S.
  3. Locally: `ReconfigLevers` @ `a8c2b7e` `scripts/bf16_rate_table.py --rc-commit ed99e08 --w4-commit ed99e08 --label full-bf16 --out results/bulk_cost_model/throughput_table_fullbf16.json` → `bulk_cost_model.py --table … --json results/bulk_cost_model/model_fullbf16.json`.
- Endpoints:
  - Makespan per arm, mean of 2 reps; completeness (all requests, exact useful tokens).
  - Per-iteration p50 and admitted batch vs Q2 and the fp32 rows.
  - For any W4 failure, the answering hop from the R1 fields.
  - Break-even phase sizes for C = 160.5 / 80 / 45 s.
- Stop criteria: an incomplete or failed W0/W5 arm stops that layout (record; at most 2 fix attempts). W4 failures are recorded with their signature, with no hot-patching. Expected duration ≈ 2.5–3.5 h.

### R3 — results (full bf16, `research/bf16-kv` @ `ed99e08`, Q2 harness unchanged)

Artifacts: `~/hetero-wt/Bf16Kv/results/pilot/{rc,w4}-ed99e08-*.json` (committed `eac20be`). Logs: `results/bf16kv/r3-driver.log`, `results/reconfig/measure-ed99e08.log`, `results/pilot/w4-campaign-ed99e08.log`. Every arm is complete, with exact token accounting (W0 1,763 req / 721,595 tok; W5 250 / 982,899; W4 904 / 387,975), 0 failed requests and 0 ExtProc aborts ✔.

| Workload | Layout | full bf16 (reps) ✔ | fp32 published | Q2 fp32-act/bf16-KV ✔ | full bf16 vs fp32 |
|---|---|---|---|---|---|
| W0 (small) | S | **190.8 s** (192.1, 189.5) | 343.3 s | 455.4 s | −44% |
| W0 | H | **306.4 s** (307.2, 305.6) | 497.2 s | 779.8 s | −38% |
| W5 (middle) | S | **754.3 s** (743.7, 764.8) | 1450.7 s | 1166.4 s | −48% |
| W5 | H | **774.4 s** (772.6, 776.3) | 1372.4 s | 1469.1 s | −44% |
| W4 (large) | H | **187.4 s** (190.0, 184.9) | 347.1 s | blocked (2 reps incomplete) | −46% |
| W4 | S | **147.7 s** (148.3, 147.0) | — (fp32 TP8 pool < 36,864) | not run | — |

- Mechanism ✔ (per-engine iteration stats in the same artifacts, compared with `rc-591e3fe-*` and `~/hetero-research/results/pilot/rc-bab4f9e-*`). Per-iteration p50 roughly halves vs Q2 at the same batch:
  - W0 on H: 0.20–0.21 → 0.072–0.100 s.
  - W5 on H: 0.118–0.122 → 0.065–0.066 s.
  - W5 on S: 0.066–0.069 → 0.045 s.
  - W0 on S: 0.085–0.232 → 0.054–0.090 s.
  - The bf16 pools keep the doubled admitted batch (W5 max scheduled: S 15–16 vs fp32 8; H 39–43 vs fp32 20–21). The 2× batch now costs little per step (W5 on H 0.065 s vs fp32 0.055 s), so throughput nearly doubles.
  - W5 still preempts: S 236–238, H 208–211 per rep, similar to Q2 and fp32. W0 has none.
- S is now the fastest static layout for every class ✔: W0 by 115.6 s/manifest, W5 by 20.1 s, W4 by 39.7 s. The W5 margin is about one rep spread (S reps differ by 21 s) [INF: not significant at n = 2].
- W4 deploys at max_model_len 36864 (block 256) ✔: H 567 blocks = 144,896 usable per TP16; S 206 blocks = 52,480 usable per TP8 (`results/pilot/w4-deploy-{H,S}-fullbf16.json`). Both were cold compiles for the new context length: deploy 483.6 s (H) and 463.9 s (S). Warm plain deploys at 16384: S 375.4 s, H 383.0 s.
- R1 follow-up ✔: the ~255-in-flight failure did not recur in any of the 4 W4 reps (H 2, S 2) or the 8 W0/W5 arms. `failed_requests` is empty and `gateway_extproc_aborts.count` = 0 everywhere, so no answering hop was observed. [INF: the faster full-bf16 engines may simply avoid the trigger. Its cause stays open, and the R1 diagnostics remain in place.]

**Cost-model recalibration** ✔. Code: `~/hetero-wt/ReconfigLevers` @ `a8c2b7e` (`bf16_rate_table.py --label full-bf16` now also reads W4 on S) and `84318d0` (results). Outputs: `results/bulk_cost_model/throughput_table_fullbf16.json` and `model_fullbf16.json` (`--kv-measured` TP8 52,864 / TP16 145,216). Published-row validation still runs on the fp32 table and reproduces all 10 rows (max \|Δk\| 0.009 manifests); the mix residuals are unchanged.

| C regime (O = S drain tail 32.6 s) | k* W0: switch H→S for the small phase | k* W5: switch S→H for the middle phase | k* W4: switch to H for the large phase |
|---|---|---|---|
| 160.5 s (warm production) | **1.67 manifests** ≈ 1.21M output tokens (fp32 1.25; Q2 0.60) | none: H 20.1 s/manifest slower | none: H 39.7 s/manifest slower |
| 80 s (projected levers) | **0.97** ≈ 0.70M tokens (fp32 0.73; Q2 0.35) | none | none |
| 45 s (no trace) | **0.67** ≈ 0.48M tokens (Q2 0.24) | none | none |

- With O = 0, the W0 break-even falls to 1.39 / 0.69 / 0.39 manifests.
- Example job, 4 W0 → 2 W5 → 1 W4 [derived from the table]:
  - static S: 4·190.8 + 2·754.3 + 147.7 − 2·32.6 (the model's same-layout overlap, as in its static H) = **2,354.3 s**;
  - static H: 2,896.6 s;
  - S→H at C = 160.5: 2,659.9 s (`model_fullbf16.json`; the example table has no static-S column because W4 on S was not measurable before).
- Under full bf16 the bulk job should stay on S. A layout switch only pays when the job *starts* on H (or another layout) and the small phase holds ≥ 0.7–1.7 W0 manifests.

**Cleanup receipt (after R3)** ✔ (`results/bf16kv/r3-cleanup/`):

- RCM Idle.
- `snapshot_validate.py preflight` ok; `snapshot_directory_bytes` 0 on 8/8 hosts.
- `inspect` for S and H: registry [], observed []; all owned placement groups REMOVED (S 52, H 20 historical PGs).
- Final teardowns: W4 campaign drain, gateway-down (removed=true) and release for H and S; reconfig_measure gateway-down and release for H.
- R2 tmpfs cleanup ✔ (`bf16prov-69a705c/cleanup-tmpfs.json`).
- Head-pod leftovers, unchanged from Q2: the stale E2 S ledger copy and the failed Q2 W4 artifacts.
- Local: RCM telemetry rotated (see R2).
- The dashboard port-forward runs as a managed local service.

**Decisions for the user (after R0–R3)**

1. Accept full bf16 as the rate basis. Under full bf16 every rate improves 38–48% over fp32, and S wins every class. The accuracy cost vs the fp32-state references: 64-token greedy sequences differ on 14/32 (S) and 11/32 (H) prompts; median per-probe deviation ≈ 0.5 nat.
2. Given that S dominates W0, W5 and W4, should the study's job model change? For example: jobs that start on H, or larger large-request contexts that do not fit a TP8 pool (> 52,480 tokens). Otherwise the one-way S→H switch has no positive break-even.
3. R1: the gateway failure is unexplained and did not recur under full bf16. Options: keep the diagnostics and move on (recommended), or spend a targeted reproduction under the slower Q2 profile.
4. Disk: `/` on this host is at ~99%. RCM telemetry grows by GBs per campaign, and each provisioning run writes about 1.4 GB of JSON (per-worker `startup_profiles`).

### L0 — W6 long-context workload and long-job cost model (local) — done

**Source choice.** GovReport (`ccdv/govreport-summarization`, all four shards: train 0/1, validation, test; cached in `/tmp/longwl/govreport-*.parquet`), same prompt template as W4 (`Summarize the following government report.` + report + `Summary:`), decode length = Gemma-token length of the real human summary, run with `ignore_eos`. Rejected:

- PG-19: books have no reference output, so the decode length would be synthetic; most books also exceed the 145k-token TP16 pool.
- LongBench-v2: multiple-choice, so the real output is about one token (a prefill-only phase), and many contexts exceed the TP16 pool.

GovReport keeps the W4 precedent (real documents, real output lengths) and needs no new instruction format.

**Catalogue** `longctx` ✔ (`scripts/build_longctx_catalogue.py`, `data/workloads/longctx_govreport.json`, sha256 `d1ba9a9b…`). Filter: prompt > 52,864 tokens (does not fit a full-bf16 TP8 pool), prompt + decode ≤ 131,072, 64 ≤ decode ≤ 2,048, deduped by report hash, never truncated. Of all GovReport reports, 65 have prompts above 52,864 tokens; 63 pass. Prompt 52,917–127,370 tokens (median 62,853); decode 77–1,534 (median 677); context 53,295–128,342. The character pre-filter (≥ 150k chars) loses nothing: the largest prompt under 150k chars is 37,749 tokens.

**Manifests** ✔ (`research/bf16-kv` @ `c1d80df` + `cee352a`, `~/hetero-wt/Bf16Kv`; `load_manifest` round-trips both):

| Manifest | Requests | Output tokens | Prompt tokens | Context p50 / max | sha256 |
|---|---|---|---|---|---|
| W6 (40 of the 63, seed 20261006, `request_class` long) | 40 | 26,951 | 2,750,436 | 64,338 / 128,342 | `3ae550fb…` |
| MIX_W6_W0 = W6 ×1 then W0 ×1 (rows labelled by `phase_workload`) | 1,803 | 748,546 | 3,180,294 | 502 / 128,342 | `ebb23284…` |

- Size: 40 requests so that one W6 manifest on H takes several minutes [INF: about 5–10 min, from ~69k-token prefills and at most two requests resident per TP16; to be measured in L1].
- TP16 `max_model_len` for W6 = 131,072 (≥ largest context 128,342; below the 144,896–145,216-token TP16 pool).
- Routing on M ✔ (code + unit test `tests/test_w6_manifest.py`): W6 contexts exceed the TP8 `max_model_len` 16,384, so only the TP16 is eligible (`hetero/gateway/policy.py:158,189-190`); with `large_fresh_traffic_share` 0 and no fallback, W0 goes only to the TP8s (`policy.py:184,196`).
- Admission fix ✔: the default `reserve_large_headroom_blocks` = 64 applies to every `large*` instance, H's two TP16s included. At 512-token blocks it would refuse a 127,370-token prompt. The W6 policy sets it to 0 (test covers 256/512/1,024-token blocks).
- Tests ✔: full suite at `cee352a` 262 passed, same 16 baseline failures.

**Cost model** ✔ (`~/hetero-wt/ReconfigLevers` @ `76f446d`, `ea2cbf1`): `bulk_cost_model.py --long-table` adds `long_job` (k_s W0 manifests, then k_l W6 manifests; all requests known at t = 0). Rates: a0S, a0H (R3), a0M = W0-class finish on M's 2×TP8, t6H = W6 on H, t6M = W6-class finish on M's TP16 (both from MIX_W6_W0 on M). Convention as in the published break-even rows: switching adds C; a static layout keeps the phase overlap O (32.6 s; O = 0 also reported).

- static H = k_s·a0H + k_l·t6H − O
- static M = max(k_s·a0M, k_l·t6M) [INF: linear in k]
- S→H = k_s·a0S + C + k_l·t6H
- S→M: 2 TP8 become the TP16 at τ (serving W6 from τ + C) while 2 TP8 keep serving W0. Optimum τ* = max(0, (k_s − (C + k_l·t6M)/a0M)·a0S); makespan = k_s·a0S + (C + k_l·t6M)(1 − a0S/a0M) if k_s·a0M > C + k_l·t6M, else C + k_l·t6M. S→M beats static M iff k_s·a0M > C + k_l·t6M.
- Static S is infeasible. Break-even S→H vs static H: k_s* = (C + O)/(a0H − a0S), independent of k_l.
- `scripts/long_rate_table.py` builds the table from the L1 pilot files and joins records with the composite to get per-class finish times; tested on the fp32 `MIX_W4_W0x4` M run (7,956/7,956 records labelled; W0 finish = run makespan 1,757.2 s).
- Published-row validation unchanged: 10 rows, max \|Δk\| 0.009 ✔. Tests: 17 new pass; suite failures identical to `84318d0` (15).

### L1 — spec (full-bf16 W6 rates on H and M)

- Hypothesis [INF]: W6 on H takes about 5–10 min per manifest, bounded by long prefills and at most two resident requests per TP16. On M, W6 on the single TP16 takes about twice as long as on H, and W0 on the 2×TP8 about 2 × 190.8 s.
- Commit `research/bf16-kv` @ `cee352a` (full-bf16 defaults; native backend `81164462…`). Policy `{"decode_length_blind":true,"large_fresh_traffic_share":0.0,"reserve_large_headroom_blocks":0}`, `run-arm --unfrozen-diagnostic`.
- Deploys: H `--max-model-len 131072`; M `--max-model-len 16384 --instance-engine-settings '{"large":{"max_model_len":131072}}'`. This is the first deploy at 131,072, so it is a cold compile [INF: ~8–10 min each].
- Steps: `KUBECONFIG=$HOME/.kube/hetero-agent.yaml HEAD_POD=tpu-ray-cluster-vllm-tpu-head-gxwsv flock /tmp/hetero-cluster.lock scripts/w6_campaign.sh 2 H M`. Every stage is one `rcm_exec run`; Ray session `session_2026-10-03_14-58-37` unchanged, so `/tmp/hetero-inventory-only.json` is valid. The script:
  1. Checks disk floors (`/` ≥ 4 GB, `/dev/shm` ≥ 20 GB) before each layout and run, and stages the `longctx` catalogue on the head.
  2. Per layout: pre-teardown → deploy → 2 reps (`w6-cee352a-plain-r{1,2}`; H runs W6 arm H, M runs MIX_W6_W0 arm M0) → final teardown (drain, gateway-down, release). Deploy and teardown outputs are kept gzipped in `results/pilot/`; large run JSON is gzipped after each stage.
  3. Locally: `long_rate_table.py --commit cee352a --deploy-json …` → `bulk_cost_model.py --long-table …` (L2).
- Endpoints: per-arm makespan (H) and per-class finish times (M), mean of 2 reps; completeness (all requests, exact tokens: W6 40 / 26,951; MIX 1,803 / 748,546); TP16 KV pool and block size at 131,072; per-request admission on the TP16 (max running/scheduled, preemptions, iteration p50).
- Stop criteria: deploy or compile failure, or an incomplete run → stop that layout and record it; at most 2 fix attempts; no hot-patching. Expected duration ≈ 1.5–2 h.

L1 attempt 1 (`cee352a`, 05:29–05:46Z): both deploys succeeded, but all 4 runs failed within ~20 s in `run_arm` ✔ (`hetero/experiment.py:710`, `AutoTokenizer.from_pretrained('/tmp/longwl/gemma')`). The locally frozen W6 manifest carried the local tokenizer path as `model`. The campaign script did not stop on failed runs (it continued to M); no timed run was produced. Teardowns ran (`results/pilot/w6-cee352a-{drain,gateway-down,release}-{H,M}-{pre,final}.json.gz`).

- KV pools at `max_model_len` 131,072 ✔ (`results/pilot/w6-cee352a-deploy-{H,M}.json.gz`): every TP16 has 283 blocks × 512 = 144,896 tokens (H both; M's `large`). M's TP8s at 16,384: 827 × 64 = 52,928. The 512-token block confirms the headroom fix: 64 blocks would have held back 32,768 tokens. H deploy 486.1 s (hetero_ctl elapsed), M 385.7 s RCM stage.
- Fix 1: W6 rebuilt on the head (`hetero_ctl build-manifest`, model `/models/gemma-4-31b`): the same `manifest_sha256` `3ae550fb…` and catalogue sha as the local build, so the request set is unchanged ✔; MIX_W6_W0 re-written (`ebb23284…`, unchanged). The script now stops a layout on a failed or incomplete run (`5b94430`). Rerun as `w6-5b94430-plain-r{1,2}`.

### L1 — results (full bf16, `research/bf16-kv` @ `5b94430`, runs `w6-5b94430-plain-r{1,2}`)

Artifacts: `~/hetero-wt/Bf16Kv/results/pilot/w6-5b94430-*` and `results/hetero/artifacts/w6-5b94430-plain-r{1,2}/` (committed `95f9e3c`); log `results/pilot/w6-campaign-5b94430.log`. All 4 runs complete with exact token accounting (W6 40 req / 26,951 tok; MIX 1,803 / 748,546), 0 failed requests, 0 migrations ✔. Rate table: `~/hetero-wt/ReconfigLevers` @ `1b3bb23`, `results/bulk_cost_model/throughput_table_long_fullbf16.json` (+ `.runs.json`).

| Arm | Quantity | Mean (reps) ✔ |
|---|---|---|
| W6 on H (2×TP16 @ 131,072) | makespan | **206.2 s** (204.6, 207.9) |
| MIX_W6_W0 on M | run makespan | 409.3 s (404.6, 414.0) |
| ↳ W6 class on the TP16 | finish | **409.2 s** (404.5, 413.9); p50 191.5 / 200.6 s |
| ↳ W0 class on the 2×TP8 | finish | **352.7 s** (355.9, 349.6); p50 144.7 / 144.5 s |

- TP16 at `max_model_len` 131,072 ✔ (`w6-5b94430-deploy-{H,M}.json.gz`): 283 blocks × 512 = 144,896 tokens on all three TP16s (H `large_a/b`, M `large`). TP8 at 16,384: 827 × 64. Warm deploys: H 361.3 s, M 377.9 s (hetero_ctl elapsed).
- Admission ✔: every TP16 held at most 2 W6 requests (max and p50 running/scheduled = 2 on H and on M), 0 preemptions, iteration p50 0.028 s. A third request never fits: W6 context p50 is 64,338 tokens against a 144,896-token pool.
- W6 on M's single TP16 takes 1.98× W6 on H ✔, so the TP16 rate is the same in both layouts. W0 on M's 2×TP8 takes 352.7 s, i.e. 1.85× W0 on S (190.8 s), with 578 / 582 preemptions per TP8 per rep (max running 116–118); W0 on S had none in R3.
- Hypothesis check: W6 on H was faster than predicted (3.4 min, not 5–10 min).

### L2 — long-context job region map (local)

`bulk_cost_model.py --table throughput_table_fullbf16.json --long-table throughput_table_long_fullbf16.json --json results/bulk_cost_model/model_long_fullbf16.json` (`1b3bb23`). Published-row validation 10 rows, max \|Δk\| 0.009 ✔. Inputs: a0S 190.8, a0H 306.4 (R3), a0M 352.7, t6H 206.2, t6M 409.2 s per manifest; O = 32.6 s. Grid k_s 0–8 × k_l 0.25–8, step 0.25. An independent recomputation of the four arm formulas reproduces the printed map boundaries ✔.

Winning arm by region (C = 160.5 s; the order of the four regions is the same for every C, and a lower C widens S→H):

| Region | Winner | Boundary |
|---|---|---|
| Little small work, W6-heavy | static H | k_s < k_s* and k_s < 0.663·k_l + 0.106 (where H beats a W6-bound M) |
| W6-heavy, k_s above k_s* | S→H | k_s > k_s* = (C + O)/(a0H − a0S) and k_s < (203.0·k_l − C)/190.8 (beats W6-bound static M) |
| Balanced: k_s·352.7 ≈ k_l·409.2 | static M | between the S→H and S→M lines |
| W0-heavy | S→M | k_s·a0M > C + k_l·t6M, i.e. k_s > (C + 409.2·k_l)/352.7 |

| C | k_s* S→H vs H (O = 32.6 / 0) | S→M vs M: k_s > | Share of grid won (H / M / S→H / S→M) | Gain of the winning switch vs best static, median / max |
|---|---|---|---|---|
| 160.5 (full, warm production) | 1.67 / 1.39 | 0.455 + 1.160·k_l | 0.19 / 0.17 / 0.27 / 0.37 | 10.0% / 33.3% |
| 186 (staged effective) | 1.89 / 1.61 | 0.527 + 1.160·k_l | 0.21 / 0.19 / 0.24 / 0.36 | 9.7% / 32.8% |
| 80 (projected levers) | 0.97 / 0.69 | 0.227 + 1.160·k_l | 0.12 / 0.11 / 0.38 / 0.39 | 11.1% / 34.8% |
| 45 (projected, no trace) | 0.67 / 0.39 | 0.128 + 1.160·k_l | 0.09 / 0.08 / 0.43 / 0.40 | 11.9% / 35.5% |

- Winner ranges in k_s at C = 160.5 (O = 32.6):
  - k_l = 1: H 0–0.77, M 0.77–1.62, S→M > 1.62.
  - k_l = 2: H 0–1.43, M 1.43–2.78, S→M > 2.78.
  - k_l = 4: H 0–1.67, S→H 1.67–3.41, M 3.41–5.10, S→M > 5.10.
  - k_l = 8: H 0–1.67, S→H 1.67–7.67, M > 7.67.
- Example points at C = 160.5, in seconds (H / M / S→H / S→M):
  - (k_s, k_l) = (4, 2): 1,605 / 1,411 / 1,336 / **1,213**.
  - (8, 2): 2,831 / 2,822 / 2,099 / **1,976**.
  - (2, 4): 1,405 / 1,637 / **1,367** / 1,797.
  - (4, 4): 2,018 / **1,637** / 1,748 / 1,797.
- Static S is infeasible: W6 does not fit a TP8.
- **Are Q3/Q4 worth running? Yes.** The switching arms win 60–83% of the grid, with a median gain of about 10% and up to 35% over the best static layout. Their regions depend directly on C.
  - Q3 (warm full S→H) prices the S→H region.
  - Q4's first step S→Mb is exactly the S→M switch (Mb = TP16 + 2×TP8 = M), which wins the largest region. That step's effective C is now borrowed from the fp32-era partial M→S measurement (186 s).
  - Recommendation: re-scope Q4 to S→Mb (effective C with the two kept TP8s serving, plus the join tail) before the full staged S→Mb→H.
- Caveats [INF]:
  - a0H is the R3 W0-on-H rate at `max_model_len` 16,384. A static H that also serves W6 runs at 131,072 with 512-token blocks, and its W0 rate there is unmeasured. A slower a0H shrinks the static-H region and lowers k_s*.
  - Static M and S→M scale linearly from one manifest of each class.
  - S→M assumes the kept TP8s run at M's W0 rate during the switch and that the TP16 serves from τ + C.
  - All C values are prior or projected costs; none is measured under full bf16.

**Cleanup receipt (after L1)** ✔:

- RCM Idle.
- Final drain, gateway-down and release for H and M all ok (`results/pilot/w6-5b94430-{drain,gateway-down,release}-{H,M}-final.json.gz`). Placement groups REMOVED: H 24/24, M 6/6.
- Attempt-1 teardowns also ok (`w6-cee352a-*-final`). No snapshots or tmpfs weights were used.
- Cluster state line: `w6-campaign 5b94430 released M`.
- Head pod: the `longctx` catalogue is staged at `/tmp/omp_hetero/data/data/workloads/longctx_govreport.json` (21 MB, kept for reruns). Leftovers from Q2 are unchanged.
- Local: `/dev/shm/hetero-scratch` removed. `/` has 62 GB free (freed outside this item).

**Decisions for the user (after L0–L2)**

1. Run Q3 (full S→H) and a re-scoped Q4 (S→Mb = S→M partial switch, effective C and join tail) under full bf16. Both decide regions that the model says are won by switching.
2. Optional, about 20 min: measure W0 on H at `max_model_len` 131,072 (2 reps). This pins static H and k_s* for jobs that keep H for both phases.
3. W6 is 40 of 63 qualifying GovReport documents. 63 is the whole real supply above 52.9k tokens with a real output length, so a larger long phase needs repeats (as W0x4 does) or another source.

### P0 — W7 mid-length band, W6 repeats, W7 cost-model arms (local) — done

**Source.** GovReport again (same four shards, same template and real-summary decode length as W4/W6, `ignore_eos`). The band below 52.9k is the first candidate in the plan, and it is large: 2,304 deduplicated reports have 16,384 ≤ prompt ≤ 49,152 tokens, prompt + decode ≤ 51,200 and 64 ≤ decode ≤ 2,048, so no second source is needed. The 51,200 context cap keeps every request inside a TP8 deployed at `max_model_len` 51,200 (bf16 TP8 pool 52.9k).

- Catalogue `midctx` ✔ (`scripts/build_midctx_catalogue.py`, `data/workloads/midctx_govreport.json`, sha256 `1222a9b3…`, 27.9 MB): a seeded sample (`Random(20261007)`) of 240 of the 2,304 eligible reports, never truncated. The 45,000-character pre-filter is lossless: below 45k characters the largest prompt is 12,300 tokens; the minimum chars/token at ≥ 16,384 tokens is 3.03.
- Manifest **W7** ✔ (120 of the 240, seed 20261007, `request_class` mid), built on the head (`hetero_ctl build-manifest`, model `/models/gemma-4-31b`) with the same `manifest_sha256` `db5e59a8…` as the local dry build: 120 requests, 79,531 output tokens, 2,769,014 prompt tokens; prompt 16,409–42,605 (median 21,070); decode 166–1,436 (median 663); context p50 21,765, max 43,071. The real band is skewed to its low end. Sizing [INF, fitted step model of the R3 iterations]: about 4 min per manifest on S and on H.
- **W6x2 / W6x4** ✔ (composites of W6 ×2 / ×4; every row now carries `repeat`, ids `W6xN-p0r<k>-…`): 80 req / 53,902 tok (`ce3598ec…`), 160 / 107,804 (`3df1d186…`). Prefix caching is off, so a repeat costs the same as the first copy. Existing composites keep their frozen shas (W0x4, MIX_W4_W0x4, MIX_W4_W5, MIX_W6_W0 `ebb23284…`) ✔.
- `scripts/band_campaign.sh` (P4/P5 driver, from `w6_campaign.sh`): `WORKLOAD`, per-layout `max_model_len` S 51,200 / H 131,072, W6 policy, disk floors `/` ≥ 10 GB and `/dev/shm` ≥ 20 GB before every layout and run, stops a layout on an incomplete run.
- Code: `research/bf16-kv` @ `1ba74e7`. Tests ✔: suite 271 passed, the same 16 baseline failures; new `tests/test_w7_manifest.py`.
- Cost model ✔ (`~/hetero-wt/ReconfigLevers` @ `e1045eb`): `mid_job` = k_s W0, then k_m W7, then optionally k_l W6. Arms: static S = k_s·a0S + k_m·t7S − O (k_l = 0 only); static H = k_s·a0H + k_m·t7H + k_l·t6H − O; S→H before W7 = k_s·a0S + C + k_m·t7H + k_l·t6H; S→H before W6 = k_s·a0S + k_m·t7S + C + k_l·t6H. Break-even k_m* = (C + O)/(t7S − t7H) (none if t7S ≤ t7H). New flags: `--c-full`/`--c-partial` (S→H arms use the measured full C, S→M the measured partial C), `--a0h-override` (W0 on H at 131,072), `long_rate_table.py --w7-*-runs/--w0h131k-runs`. Default output is byte-identical to the old code on the L2 inputs ✔; published-row validation unchanged (10 rows, max \|Δk\| 0.009) ✔; tests 203 passed, the same 15 baseline failures.

### P1 — production switch harness (local) — done

Code: `research/bf16-kv` @ `6c6cf36` (+ `ad94de5`, `7bae5a3`: driver idle check and timeouts; `d16fc32`: advisor fixes, see P2 spec). `hetero/switch.py`, CLI `hetero.cli run-switch`, one Ray job per switch via `hetero_ctl.py run-switch`.

- **Sequence** (in-process, one job): pre-flight (gateway idle on the source, manifests loaded, request bodies built, policy set; no warm-up requests) → phase 1 dispatched by an asyncio client that keeps one httpx connection pool open for the whole run → **T0 = the client's last phase-1 completion** wakes a controller thread:
  - full: `drain_deployment(source, remove=True)` (drain, stop, shutdown, kill, PG release per instance) → `gateway_down` → `deploy(target)` (FastSnapshot + FastGraph) → `gateway_up(target, policy)`;
  - partial: drain the released engines → adopt the kept ones → `retarget_layout(target, joiners)` → capture → `deploy(joiners)` → `join_instances`. These steps are the helpers extracted from `run_transition`, with identical order and checks.
- **Dispatch gates:** each phase-2 stream starts when the controller signals and a light `OmpGateway.routing_status()` confirms the required instances are bound under the target layout (0.25 s poll fallback). In partial mode, rows that fit the kept TP8 `max_model_len` go to a kept stream (opens after retarget) and the rest to a joiner stream (opens after join). The caps add up to 256: joiner = its row count (40 for W6), kept = 216, as in the L1 MIX run.
- **Off the critical path:** no evidence dump, restore-check job, pilot start-up or warm-up between T0 and the first target dispatch. Iteration capture is armed just before the first target dispatch: a recorded harness segment `capture_arm`, subtracted in `C_production_ex_capture_s`. Records, iterations, the gateway journal, deploy and drain reports are written after phase 2.
- **Clocks:** contiguous segments from T0: controller_wake, drain_release (per-instance step timings added to `drain_deployment`), gateway_down, deploy_controller_pre, engine_process_start, engine_init (per-phase median/max over workers: init_device, load_model with snapshot sub-phases, determine_available_memory, initialize_from_config, compile_or_warm_up), endpoint_register, gateway_up, gate_observe, capture_arm, dispatch. They must sum to `gap_dispatch_s` within 0.05 s, or the run is marked incomplete.
- **Endpoints:** gap to the first target dispatch and to the first target token (engine-host clock); target makespan; ramp as makespan minus the static reference (`--static-target-s`) and as the published iteration-bin deficit; `C_production_s` = gap + ramp. Partial mode adds kept_gap, join delay, kept-pair tokens/s during and after the join, joiner bins, `C_eff_s` = (phase-2 end − T0) − static finish, and per-class finish times.
- Tests ✔: `tests/test_switch.py`, 23 tests. They check ordering, that no dispatch happens before its gate, segment reconciliation, the 216/40 caps with joiner rows only after the join, fail-closed abort (report written, streams cancelled, capture disarmed, never retried), unchanged `run_transition` behaviour, drain timings and the `routing_status` shape. Full suite 294 passed, the same 16 baseline failures. `scripts/switch_dry_run.py` runs both modes end to end on fakes; ordering asserts pass and segments reconcile (sum = gap).
- Drivers: `scripts/switch_campaign.sh full|partial` (from `staged_sequence.sh`: prep snapshots + greedy references, restore checks, fail-closed teardown, disk floors, `MAX_ATTEMPTS=2`, `ARM_TIMEOUT` 900 s) and `scripts/switch_summary.py` (offline recomputation and tables).

### P2 — spec (full S→H warm switch, production C)

- Hypothesis [INF]: production C = drain/release of 4 idle TP8s (tens of s) + H restore deploy (R2: 154.9 s hetero_ctl elapsed, engine init 137.6 s, compile/warm-up 80.5 s) + gateway up (~5 s) + W6 ramp ≈ 170–210 s. The harness return H→S on the published harness ≈ 230–280 s, compared with the fp32-era 225.26 s.
- Commit `research/bf16-kv` @ `d16fc32`, frozen for the timed runs (no hot-patching). Full-bf16 defaults; FAST = `{"weight_snapshot":"read","weight_snapshot_verify":false,"fast_graph":true,"startup_profile":true}`, pod-local XLA cache; S at `max_model_len` 16,384, H at 131,072; policy `{"decode_length_blind":true,"large_fresh_traffic_share":0.0,"reserve_large_headroom_blocks":0}`.
- Advisor review (opus-5-5, before the runs). P2: no blocker. P3: one blocker, now fixed. The fixes, all in `d16fc32`:
  - Hang evidence: `finalize` writes client records, the gateway journal and traces before `stop_capture` (which raised in E2) and records its error instead of failing. A failed run dumps its partial records, journal, traces and routing status, and still returns its report, so `hetero_ctl` fetches the artifacts. The driver then saves `gateway-status`, `inspect`, a copy of the head-side artifact directory and RCM WARN/ERROR logs before any teardown.
  - Gates and checks: the kept gate also waits for `capturing`; `switch_complete` also requires `restored.all_restored`.
  - Driver: `ARM_TIMEOUT` 900 s; a `local` loop variable in `run()`.
  - New measurement: a same-loader static reference, W6 on the FAST-restored prep H (`full-d16fc32-static-W6-H`), because the 206.2 s came from a normal-loader H.
  - Reporting rules from the review: the makespan ramp is the headline and the iteration-bin deficit is a diagnostic, since it reads 20.5/22.0 s on the static L1 W6 runs. The harness comparison reports the switch tail, each teardown job, the deploy job, the extra check job and pilot start, C with and without the check job, and the loader (FAST vs the published SHA-checked load) and dtype differences. P3 also reports C_kept = (kept-class finish − T0) − 352.7 and C_eff from the W6-class finish.
- Steps: `KUBECONFIG=$HOME/.kube/hetero-agent.yaml HEAD_POD=tpu-ray-cluster-vllm-tpu-head-gxwsv REPS=3 flock /tmp/hetero-cluster.lock scripts/switch_campaign.sh full`. Every stage is one `rcm_exec run`; output `results/switch/full-d16fc32/`. Ray session unchanged, so `/tmp/hetero-inventory-only.json` stays valid.
  1. Prep: preflight; tmpfs cleanup; S normal deploy → snapshot → greedy reference → teardown; H normal deploy @131,072 → snapshot → reference → teardown; H restore deploy → `check --require-restored` + `validate --reference` → static W6 pilot → teardown; S restore deploy → check + validate.
  2. Per rep r = 1..3: `run-switch` full (phase 1 W0 on S → T0 → controller → phase 2 W6 on H, `--static-target-s 206.2`, run id `full-d16fc32-r<r>`). Then the harness return H→S on the published harness shape (drain H `--force`, gateway-down, release, S restore deploy, `check --require-restored`, `run_pilot.sh` W0 on S, run id `full-d16fc32-ret-r<r>`), with timeline marks. The return leaves S deployed for the next rep. After rep 3: teardown S; the EXIT trap releases everything and removes tmpfs snapshots.
  3. Locally: `scripts/switch_summary.py results/switch/full-d16fc32`.
- Endpoints (mean of 3 reps and each rep):
  - **Production C** = (first W6 dispatch − T0) + (W6 makespan − 206.2 s), with every segment. Also reported: C with the iteration-bin ramp deficit, C without `capture_arm`, and the gap to the first W6 token.
  - **Harness-equivalent C ✔**: the measured H→S return on the published harness = (first pilot dispatch − last W6 completion) + the published ramp deficit of the W0 pilot. Same direction and harness as the fp32-era 225.26 s, split into teardown jobs, deploy job, check job and pilot start-up.
  - **S→H harness-equivalent [INF]** = production S→H C + the harness and controller segments measured in the return.
  - Output checks: restored on every worker (deploy report), exact token accounting on both phases, 0 failed requests; greedy restore equality in prep for both S and H.
- Stop criteria: any deploy, restore or compile failure, incomplete phase or failed segment reconciliation stops the rep. The driver fails closed (teardown, rebuild S, new attempt id), at most 2 failed attempts in total, then record and move on to P3. Expected duration ≈ 1.5 h (prep ≈ 30 min; 3 × (run-switch ≈ 10 min + return ≈ 10 min)).

P2 attempt 1 (`d16fc32`, 09:00Z): the driver stopped before the first deploy ✔ (`results/switch/p2-driver-d16fc32.log`). Its `kubectl cp` of the `longctx` catalogue hit an API-server i/o timeout and left a truncated file on the head (sha `77c7781f…` ≠ `d1ba9a9b…`). Only preflight, release no-ops and tmpfs cleanup had run; cleanup ✔. Fix attempt 1: `29e7b0f` (driver only) skips the copy when the head's sha already matches, else retries up to 3 times and verifies the sha. The rerun uses run ids `full-29e7b0f-*`; no `hetero/` code changed.

P2 attempt 2 (`29e7b0f`, 09:12–10:03Z, `results/switch/full-29e7b0f/`). Prep succeeded ✔:
- snapshots, references, and the H and S restore checks + validation, with restore equality on both;
- same-loader static W6 on the restored H: **202.2 s** (`results/pilot/full-29e7b0f-static-W6-H_W6_H.json`, complete).

Both run-switch attempts of rep 1 then served every request with exact tokens and a full restore, but the harness flagged them incomplete ✔ (`switch-r1-a{1,2}-failed.json`), so the driver stopped after 2 and cleaned up ✔ (all PGs REMOVED, tmpfs cleaned, RCM Idle):
- Cause ✔ (harness false negative): phase-1 accounting in full mode uses the client records, because the source gateway's journal goes with the source gateway. Their per-segment HTTP summary carries token counts but no token ids, so `_check_accounting` reported `no_duplicate_delivery = false`. The first-token clock had the same gap: it lives only in the gateway journal.
- Offline recheck of both runs ✔: phase 1 1,763/1,763 single-segment requests, delivered = accounted = segment tokens, 721,595 tokens; phase 2 complete; restored 8/8.
- Production C 136.5 s and 132.5 s; gap to first W6 dispatch 131.7 / 130.7 s; to first W6 token 136.8 / 135.6 s.
- Segments (r1 / a2): drain+release 3.5 / 2.6, gateway down 0.2, deploy controller 10.7 / 10.8, engine process start 21.6 / 21.5, engine init 90.9 / 90.8, endpoint register 1.8, gateway up 3.1 / 3.2, gate 0.004, capture arm 0.007 s.

These two runs are kept as supporting evidence; the harness return did not run for them. Fix attempt 2: `7a3aaeb` (analysis code in `finalize` only):
- phase 1 uses a client-authority check (one segment; delivered = accounted = segment tokens);
- in partial mode phase 1 uses the gateway journal;
- engine first-token clocks are copied from the journal onto the client records;
- regression test added; suite 295 passed, the same 16 baseline failures.

The rerun uses run ids `full-7a3aaeb-*` with the same steps.

### P2 — results (full S→H warm switch, `research/bf16-kv` @ `7a3aaeb`, runs `full-7a3aaeb-r{1,2,3}`)

Artifacts: `~/hetero-wt/Bf16Kv/results/switch/full-7a3aaeb/` (driver outputs, `timeline.jsonl`, `runs.jsonl`) and `results/hetero/artifacts/full-7a3aaeb-{r1,r2,r3,ret-r1,ret-r2,ret-r3,static-W6-H}/`; summary `scripts/switch_summary.py results/switch/full-7a3aaeb`; driver log `results/switch/p2-driver-7a3aaeb.log`.

Run health ✔:
- every rep is complete, with exact accounting in both phases (W0 1,763 req / 721,595 tok on S; W6 40 / 26,951 on H) and 0 failed requests;
- every new worker was restored from the snapshot (8/8);
- in all reps the gate opened on the controller event (no poll fallback);
- the offline recomputation equals the in-process segments (max diff 0.00 s);
- prep restore checks and greedy restore equality on S and H passed (`H-restored-validation.json`, `S-restored-validation.json`);
- every harness-return pilot is complete.

**Production C, full S→H (in-process, no harness on the critical path)** ✔:

| | r1 | r2 | r3 | mean |
|---|---|---|---|---|
| T0 → first W6 dispatch | 130.68 | 130.52 | 130.72 | **130.64 s** |
| T0 → first W6 token (engine clock) | 135.67 | 134.80 | 135.29 | 135.26 s |
| W6 makespan on the new H | 208.12 | 211.46 | 203.76 | 207.78 s |
| ramp = makespan − 206.2 (L1 static, normal loader) | 1.92 | 5.26 | −2.44 | 1.58 s |
| **C_production = gap + ramp** | 132.60 | 135.78 | 128.27 | **132.22 s** |
| C vs same-loader static W6 (FAST-restored H: 202.2 / 204.4 s, mean 203.3) | 135.5 | 138.7 | 131.2 | 135.1 s |
| C with the iteration-bin deficit (diagnostic; biased high for W6, advisor) | 143.34 | 144.20 | 144.96 | 144.17 s |

The two attempt-2 runs (`full-29e7b0f-r1{,-a2}`, rechecked offline) give C = 136.5 and 132.5 s. All five switches fall within 128.3–136.5 s.

Segments (mean of 3, s after T0, contiguous; they sum to the gap within 0.01 s) ✔:

| Segment | Class | s | Notes |
|---|---|---|---|
| controller wake | controller | 0.01 | |
| drain + release of 4 idle TP8s | engine | 1.92 | maintenance, drain ack, stop, shutdown, kill, PG release |
| gateway down | controller | 0.17 | |
| deploy: Ray backend, placement, actor create | controller | 11.01 | stage clocks backend 2.0, placement 0.1, actor 0.5; the rest is ray/inventory/native-backend preparation |
| engine process start before `init_device` | engine | 21.50 | fp32-era 21.1 |
| engine init (`init_device` → last engine ready) | engine | 90.73 | medians: init_device 4.1, **load_model 5.0** (snapshot read 0.44, device placement 0.86, other 3.5; fp32-era 40.7), **initialize_from_config 26.0** (fp32-era 4.2), **compile/warm-up 54.6** (fp32-era 50.2) |
| endpoint register | engine | 2.16 | |
| gateway up (create, bind, start) | controller | 3.13 | |
| gate, capture arm, dispatch | controller / harness | 0.02 | |
| **gap** | | **130.64** | engine 116.3, controller 14.3, harness 0.01 |

**Harness-equivalent C, return H→S on the published harness** ✔ (same direction and harness shape as the fp32-era 225.26 s):

| | r1 | r2 | r3 | mean |
|---|---|---|---|---|
| gap: last W6 completion → first pilot dispatch | 279.05 | 278.43 | 255.87 | 271.12 s |
| ramp deficit of the W0 pilot (published method) | 2.50 | 2.49 | 2.60 | 2.53 s |
| **C_harness** | 281.55 | 280.92 | 258.48 | **273.65 s** |
| C_harness without the restore-check job | 257.1 | 256.7 | 233.6 | 249.2 s |

Mean split:
- switch tail (run-switch finalize + artifact fetch, equivalent to the published "source pilot tail") 14.4 s;
- teardown jobs: drain 8.3, gateway-down 7.7, release 6.1 s;
- S restore deploy job 189.4 s: remote 149.0 s, of which engine init ≈ 108 s with compile/warm-up 72.0 s and initialize_from_config 25.7 s; the remaining 40.5 s is kubectl apply/rollout, job submit and the evidence stream (r3 23 s, r1/r2 49 s);
- restore-check job 24.5 s;
- pilot start → first dispatch 20.6 s (warm-up etc.).

Comparability notes:
- FAST restore here, against the published SHA-checked load (40.7 s);
- full bf16 here, against fp32;
- one extra check job (the published shape had a "controls check + check job" segment as well);
- head-pod clock for T0 and dispatch; controller-machine marks for the split.

- **S→H harness-equivalent [INF]**: production C + the harness/controller overheads measured in the return:
  - switch tail 14.4;
  - teardown jobs minus their real work, 20.1;
  - deploy-job wall minus remote, 40.5;
  - check 24.5;
  - pilot start 20.6.
  - Total ≈ **252 s**, against 132 s production: about 120 s (48%) of a harness-measured C is harness and controller shape.
- Hypothesis check: production C (132 s) is below the predicted 170–210 s. load_model fell from 40.7 to 5.0 s (FastSnapshot + FastGraph), and the in-process controller removes about 120 s of harness and controller shape. The harness return (274 s) is inside the predicted 230–280 s. It is slower than the fp32-era 225 s because initialize_from_config (+21 s) and S compile/warm-up (+22 s vs 50.2) grew under full bf16 [INF: not profiled], plus the extra check job.
- Static W6 on the FAST-restored H ✔: 202.2 s (`full-29e7b0f-static-W6-H`) and 204.4 s (`full-7a3aaeb-static-W6-H`), against 206.2 s on the normal loader (L1).

**Cleanup receipt (P2)** ✔: final teardown S (drain, gateway-down, release); the EXIT cleanup released H, Mb and S, and no owned PG is left non-REMOVED; tmpfs cleanup ok (`results/switch/full-7a3aaeb/cleanup-*.json`); RCM Idle; state line `switch-campaign full-7a3aaeb released S final`. Attempt 1 and 2 cleanups ✔ as recorded above.

### P3 — spec (partial S→Mb, effective C, join tail)

- Hypothesis [INF]:
  - kept gap ≈ drain of two idle TP8s + adopt + retarget + capture ≈ 10–25 s (fp32-era 9.3 s);
  - TP16 joiner restore ≈ 140–160 s, join ≈ 5–10 s;
  - kept-pair W0 rate during the join ≈ the M pair rate, 721,595/352.7 = 2,046 tok/s (half of S = 1,891 tok/s);
  - C_eff ≈ join delay + joiner ramp ≈ 160–200 s, against the 186 s (fp32-era) that L2 borrowed.
- Commit `7a3aaeb` (`d16fc32` + the two P2 fixes above), frozen. Mb = small_a, small_b (TP8 @16,384, kept) + large_b (TP16 @131,072, FAST restore from an Mb snapshot). Same policy.
- Steps: `… REPS=3 flock /tmp/hetero-cluster.lock scripts/switch_campaign.sh partial` (output `results/switch/partial-7a3aaeb/`).
  1. Prep: S normal → snapshot → reference → teardown; Mb normal with `large_b` @131,072 → snapshot of `large_b` → reference → teardown.
  2. Per rep: S restore deploy → check (+ validate on rep 1) → `run-switch` partial (phase 1 W0 on S; phase 2 = W0 on the kept pair + W6 on the joiner, `--static-target-s 409.2`, the L1 W6-class finish on M) → `check Mb --require-restored` → teardown Mb and leftover S PGs.
- Endpoints:
  - serving gap = first kept dispatch − T0, with segments;
  - kept-pair tok/s during the join and after it, vs 2,046;
  - join delay = first W6 dispatch − T0, plus deploy/join and first-W6-token times;
  - join tail = the joiner's per-10-s bins from first served to steady, and W6-class finish − first W6 dispatch − 409.2;
  - **effective C** = (phase-2 end − T0) − 409.2 s (the S→M arm's C when W6-bound, which holds here because 352.7 < C + 409.2);
  - per-class finish times.
- Stop criteria: the E2 joiner hang (joiner engine stops making progress, RPC timeouts, gateway 500/503). If it recurs: capture diagnostics (the driver's `diagnose`: RCM logs, `inspect`, gateway status, the run-switch artifacts and report from the head), tear down, and stop P3 after at most 2 attempts. `ARM_TIMEOUT` 900 s caps a hung stream. Other failures: same as P2. Expected duration ≈ 1.3 h.

### P3 — results (partial S→Mb, `research/bf16-kv` @ `7a3aaeb`, runs `partial-7a3aaeb-r{1,2,3}`)

Artifacts: `~/hetero-wt/Bf16Kv/results/switch/partial-7a3aaeb/` and `results/hetero/artifacts/partial-7a3aaeb-r{1,2,3}/`; driver log `results/switch/p3-driver-7a3aaeb.log`; summary `scripts/switch_summary.py results/switch/partial-7a3aaeb`.

Run health ✔:
- all 3 reps succeeded on the first attempt;
- exact accounting in both phases (phase 1 W0 1,763/721,595 against the gateway journal; phase 2 W0 1,763/721,595 + W6 40/26,951), 0 failed requests;
- the joiner `large_b` was restored on every worker, and `check Mb --require-restored` passed after each rep;
- every gate opened on the controller event;
- the offline recomputation equals the in-process values.
- **The E2 joiner hang did not recur** ✔: no stalled joiner, no RPC timeout, no gateway 500/503. Each joiner served its 40 W6 requests at the static-M rate.

| ✔ (s after T0 = last phase-1 completion) | r1 | r2 | r3 | mean |
|---|---|---|---|---|
| **serving gap of the kept pair** (first kept dispatch) | 1.93 | 2.27 | 2.05 | **2.08** (fp32-era 9.3) |
| ↳ drain + release of small_c/small_d, adopt, retarget, capture arm | 1.36 / 0.02 / 0.50 / 0.03 | 1.61 / 0.01 / 0.59 / 0.03 | 1.49 / 0.01 / 0.50 / 0.03 | 1.49 / 0.01 / 0.53 / 0.03 |
| joiner deploy ready | 126.96 | 126.83 | 126.31 | 126.70 |
| join returned = first W6 dispatch | 127.92 | 129.33 | 127.99 | **128.41** |
| first W6 token (engine clock) | 132.42 | 134.57 | 132.43 | 133.14 |
| kept-pair W0 tok/s during the join (first kept dispatch → join) | 2,502 | 2,510 | 2,554 | **2,522** |
| kept-pair W0 tok/s after the join | 1,863 | 1,844 | 1,917 | 1,875 |
| kept-class (W0) finish | 346.1 | 347.7 | 336.6 | 343.4 |
| C_kept = kept finish − 352.7 (static-M W0-class finish) | −6.6 | −5.0 | −16.1 | **−9.2** |
| W6-class finish | 541.1 | 532.8 | 540.4 | 538.1 |
| W6 on the joiner, first dispatch → finish (static M: 409.2) | 413.2 | 403.5 | 412.5 | 409.7 |
| **effective C = W6-class finish − 409.2** | 131.90 | 123.61 | 131.25 | **128.92** |

- Join tail ✔: none measurable. From its first dispatch the joiner finishes W6 in 409.7 s, against 409.2 s on a static M (−5.7 to +4.0 s per rep). Its per-10-s token bins sit at the steady W6 level (median ≈ 69 tok/s) from the second bin on; the first bin is a W6 prefill.
- Kept pair: 2,522 tok/s during the join is 0.67 of a full S on W0 (721,595/190.8 = 3,782 tok/s), the same fraction as fp32-era mix A, and 23% above the static-M pair rate (2,046). The phase-2 W0 stream had no TP16 competitor for capacity or concurrency before the join [INF]. The W0 class therefore finishes 9 s *before* the static-M W0-class time: the kept side costs nothing.
- The joiner's engine anatomy equals the full switch's H: controller pre-engine 10.9 s, process start 21.8 s, engine init 90.4 s. Medians: load_model 5.0, initialize_from_config 26.1, compile/warm-up 54.7 s. Effective C (128.9 s) ≈ production full C (132.2 s): both are the time to start a FAST-restored TP16.
- Against the S→M arm in L2: C_partial = 128.9 s replaces the borrowed 186 s, and the arm's assumption that the kept pair runs from τ with no gap holds (2.1 s gap, C_kept ≤ 0).

**Cleanup receipt (P3)** ✔:
- per rep: `check Mb`, teardown Mb (drain, gateway-down, release) and the leftover S PGs;
- EXIT cleanup: release H/Mb/S with no non-REMOVED owned PG (`results/switch/partial-7a3aaeb/cleanup-*.json`) and tmpfs cleanup ok;
- RCM Idle; state line `switch-campaign partial-7a3aaeb released S switch-r3-leftover-S`.

### P4 — spec (W0 on H at max_model_len 131,072)

- Hypothesis [INF]: at 131,072 the TP16 KV block is 512 tokens (283 blocks, 144,896 tokens). Each W0 request (context median ≈ 400 tokens) then holds at least one 512-token block, versus about 7 × 64 at 16,384, so W0 on H runs slower than R3's 306.4 s. Predicted 306–400 s per manifest.
- Commit: `research/bf16-kv` at the HEAD when P4 starts (code = `7a3aaeb`; later commits are evidence only), run ids `w0h131k-<sha>-plain-r{1,2}`. Policy = the W6 policy (`{"decode_length_blind":true,"large_fresh_traffic_share":0.0,"reserve_large_headroom_blocks":0}`), which a static H job serving W0 and then W6 would run; R3's W0-on-H used the base reserve of 16 blocks (1,024 tokens at 64-token blocks). Plain deploy (normal loader, warm cache), as in R3/L1.
- Steps: `KUBECONFIG=… HEAD_POD=tpu-ray-cluster-vllm-tpu-head-gxwsv WORKLOAD=W0 TAG=w0h131k flock /tmp/hetero-cluster.lock scripts/band_campaign.sh 2 H` (pre-teardown → deploy H `--max-model-len 131072` → 2 × `run_pilot.sh H H` W0 → final teardown; disk floors before each stage). Then locally: `long_rate_table.py --w0h131k-runs …` → a0H_131k.
- Endpoints: makespan per rep (mean of 2); completeness (1,763 req / 721,595 tok); KV pool and block size; max running, iteration p50 and preemptions per TP16, compared with R3's W0 on H.
- Stop criteria: deploy failure or an incomplete run stops it; at most 2 fix attempts; no hot-patching. Expected ≈ 25 min.

### P4 — results (W0 on H at max_model_len 131,072, runs `w0h131k-1edf1c8-plain-r{1,2}`)

Commit `1edf1c8`: code = `7a3aaeb`, plus the P2/P3 evidence commit and a `band_campaign.sh` staging fix (verified sha copy with retries). Artifacts: `~/hetero-wt/Bf16Kv/results/pilot/w0h131k-1edf1c8-*` (deploy, teardowns, campaign log `w0h131k-campaign-1edf1c8.log`) and `results/hetero/artifacts/w0h131k-1edf1c8-plain-r{1,2}/`.

| ✔ | r1 | r2 | mean | R3 (H @16,384, base policy) |
|---|---|---|---|---|
| W0 makespan on H | 270.57 | 265.04 | **267.80 s** | 306.4 s (307.2, 305.6) |
| scheduled requests p50 (large_a / large_b) | 89 / 66 | 66 / 52 | | 52 / 47, 39 / 60 |
| iteration latency p50 (s) | 0.115 / 0.092 | 0.092 / 0.073 | | 0.094 / 0.081, 0.072 / 0.100 |
| preemptions | 0 | 0 | | 0 |

- Completeness ✔: 1,763 req / 721,595 tok in both reps, 0 failed requests. KV pool ✔: 282 blocks × 512 = 144,384 usable tokens per TP16 (L1: 283 × 512). Warm deploy 360.2 s (hetero_ctl elapsed).
- **Hypothesis falsified**: W0 on H at 131,072 is 12.6% *faster* than R3's 306.4 s, not slower. The run admits a larger batch (scheduled p50 52–89 vs 39–60) with no preemptions and similar step latency. The two runs differ in two ways: `max_model_len`/block size (512 vs 64 tokens) and policy (the W6 policy has `reserve_large_headroom_blocks` 0; R3 used the base reserve of 16 blocks on H's large instances). Which one raises the batch is not isolated [INF].
- For the cost model: a static H that serves W0 and then W6 runs at exactly this setting, so a0H_131k = 267.8 s is the input for that arm (`--a0h-override 267.8`). It narrows the S advantage on W0 from 115.6 to 77.0 s per manifest.
- Cleanup receipt (P4) ✔: final drain, gateway-down (removed=true) and release of H, with no non-REMOVED owned PG (`results/pilot/w0h131k-1edf1c8-{drain,gateway-down,release}-H-final.json.gz`); RCM Idle.

### P5 — spec (W7 on S and H; region maps at measured C)

- Hypothesis [INF]: W7 (context median 21.8k): a TP8 pool (52.9k) holds 2 such requests and a TP16 (144.9k) about 6. The fitted step model predicts about 4 min per manifest on both, with H/S ≈ 0.9–1.0. So a W0 → W7 job switches S→H only if t7S − t7H is large compared with C.
- Commit as P4; run ids `w7-<sha>-plain-r{1,2}`. S deploys at `max_model_len` 51,200: the first deploy at this length is a cold compile [INF ≈ 8 min]. H deploys at 131,072. Same W6 policy (S has no large instances).
- Steps: `… WORKLOAD=W7 flock /tmp/hetero-cluster.lock scripts/band_campaign.sh 2 S H`. Then locally (`ReconfigLevers`):
  1. `long_rate_table.py --w7-s-runs … --w7-h-runs … --w0h131k-runs …` → mid table;
  2. `bulk_cost_model.py --table throughput_table_fullbf16.json --long-table throughput_table_long_fullbf16.json --mid-table <mid> --c-full <P2 production C> --c-partial <P3 effective C> --a0h-override <P4> --json results/bulk_cost_model/model_switchcost_fullbf16.json` → long-job and mid-job region maps at the measured C (O = 32.6 s and 0).
- Endpoints: t7S, t7H (mean of 2), completeness (120 req / 79,531 tok), per-instance admission (max running, preemptions, iteration p50); region maps (winner per region, break-evens k_s*, k_m*, S→M boundary) at the measured C, compared with L2's projected regimes.
- Stop criteria: as P4. Expected ≈ 50 min.

### P5 — results (W7 on S and H, runs `w7-1edf1c8-plain-r{1,2}`; region maps at measured C)

Commit `1edf1c8` (as P4). Artifacts: `~/hetero-wt/Bf16Kv/results/pilot/w7-1edf1c8-*` and `results/hetero/artifacts/w7-1edf1c8-plain-r{1,2}/` (evidence commit `e54dcb6`, which also holds the head-built W7 manifest). Model: `~/hetero-wt/ReconfigLevers` @ `d4f324b`, `results/bulk_cost_model/{throughput_table_mid_fullbf16.json, model_switchcost_fullbf16.json, model_switchcost_fullbf16.txt}`.

| W7 ✔ (120 req / 79,531 tok, complete, 0 failed) | r1 | r2 | mean | admission per instance |
|---|---|---|---|---|
| S (4×TP8 @51,200; 205 × 256 = 52,480 tokens per TP8) | 249.17 | 247.54 | **248.4 s** | scheduled p50 2 (max 2–3), 5 / 14 preemptions per run, step p50 0.032 s |
| H (2×TP16 @131,072; 282 × 512 = 144,384 per TP16) | 203.36 | 206.23 | **204.8 s** | scheduled p50 6 (max 7), 0 / 1 preemptions, step p50 0.033 s |

- H is 17.6% faster than S on W7 (H/S = 0.82; predicted 0.9–1.0 [INF]). At the same step time H keeps 6 requests resident per TP16, against 2 per TP8 on S. Mid-length context is the first workload in this study where H beats S on rate. Deploys: S @51,200 was a cold compile, 466.4 s (compile/warm-up median 155.4 s); H warm, 361.6 s.
- Cleanup receipt (P5) ✔: final drain, gateway-down and release for S and for H, with no non-REMOVED owned PG (`results/pilot/w7-1edf1c8-{drain,gateway-down,release}-{S,H}-final.json.gz`); RCM Idle.

**Region maps at the measured C** ✔:
- Inputs: C_full = 132.2 s (P2), C_partial = 128.9 s (P3), a0H = 267.8 s (P4, static H at 131,072), t7S = 248.4 s, t7H = 204.8 s (P5); a0S 190.8, a0M 352.7, t6H 206.2, t6M 409.2 (R3/L1); O = 32.6 s.
- Published-row validation: 10 rows, max \|Δk\| 0.009 ✔. Recomputing the L2 shares from the same arm functions reproduces L2's C = 160.5 / 186 rows exactly ✔.

*Long-context job (k_s W0, then k_l W6), grid k_s 0–8 × k_l 0.25–8:*

| | static H | static M | S→H | S→M | switching total | gain of winning switch vs best static, median / max |
|---|---|---|---|---|---|---|
| L2 (C = 160.5 projected, a0H 306.4) | 0.19 | 0.17 | 0.27 | 0.37 | 0.64 | 10.0% / 33.3% |
| **measured** (C 132.2 / 128.9, a0H 267.8) | **0.24** | **0.14** | **0.24** | **0.38** | **0.62** | **8.3% / 24.5%** |
| measured, O = 0 | 0.19 | 0.15 | 0.29 | 0.38 | 0.67 | 8.3% / 25.6% |
| measured C, with R3's a0H 306.4 (sensitivity) | 0.17 | 0.15 | 0.31 | 0.38 | 0.69 | 10.4% / 33.9% |

- Winner per region (same order as L2):
  - static H: k_s < k_s* = (C + O)/(a0H − a0S) = **2.14** (1.72 at O = 0; L2 1.67);
  - S→H: k_s > 2.14, while the job is W6-heavy;
  - static M: the balanced band;
  - S→M: k_s > (C_partial + 409.2·k_l)/352.7 = **0.365 + 1.160·k_l** (L2 at 160.5: 0.455 + 1.160·k_l).
- k_s ranges by k_l:
  - k_l = 1: H 0–0.88, M 0.88–1.53, S→M > 1.53;
  - k_l = 2: H 0–1.64, M 1.64–2.69, S→M > 2.69;
  - k_l = 4: H 0–2.14, S→H 2.14–3.56, M 3.56–5.01, S→M > 5.01;
  - k_l = 8: H 0–2.14, S→H 2.14–7.82, M > 7.82.
- Example points (H / M / S→H / S→M, s):
  - (4, 2): 1,451 / 1,411 / 1,308 / **1,198**;
  - (8, 2): 2,522 / 2,822 / 2,071 / **1,961**;
  - (2, 4): **1,328** / 1,637 / 1,339 / 1,766;
  - (4, 4): 1,863 / **1,637** / 1,720 / 1,766.
- What changed vs L2:
  - The measured switch costs are about 30 s below the projected 160.5 s. The partial C is 57 s below the borrowed 186 s, which widens S→M (0.37 → 0.38).
  - A faster static H (a0H 267.8 instead of 306.4) moves k_s* from 1.67 to 2.14 and wins back cells from S→H and M.
  - Net: switching still wins 62% of the grid, but the best gain falls from 33% to 25%.

*Mid-band job (k_s W0, then k_m W7, then k_l W6), grid k_s, k_m 0–8; S→H only (no M arm: W7 on M is unmeasured):*

| k_l | static S | static H | S→H before W7 | gain of S→H vs best static, median / max |
|---|---|---|---|---|
| 0 | 0.42 | 0.21 | 0.38 | 3.3% / 6.5% |
| 1 | infeasible | 0.27 | 0.73 | 9.7% / 19.5% |
| 2 | infeasible | 0.27 | 0.73 | 8.9% / 17.9% |

- Break-evens:
  - S→H before W7 beats static S (k_l = 0) for k_m > k_m* = (C + O)/(t7S − t7H) = **3.78 W7 manifests** (3.03 at O = 0);
  - it beats static H for k_s > 2.14, for any k_m and k_l;
  - switching before W7 always beats switching before W6, by k_m·43.6 s, because t7S > t7H.
- Winner per region:
  - k_l = 0:
    - static S when k_m < 3.78 and k_s·77.0 > k_m·43.6 (W0-heavy);
    - static H when k_s < 2.14 and k_s·77.0 < k_m·43.6 (W7-heavy, little W0);
    - S→H before W7 when k_s > 2.14 and k_m > 3.78.
  - k_l ≥ 1: static H below k_s = 2.14, S→H above.
- Examples (k_s, k_m, k_l → S / H / S→H, s): (2, 4, 0) 1,343 / **1,322** / 1,333; (4, 8, 0) 2,718 / 2,677 / **2,534**; (4, 2, 1) — / 1,654 / **1,511**.
- Caveats [INF]:
  - a0S is R3's W0 on S at 16,384; S running W0 + W7 would be deployed at 51,200, which is unmeasured, and P4 shows such a setting change can move a rate by 13%.
  - The mid job has no M or S→M arm, so its k_l ≥ 1 shares are against static H only.
  - All arms scale linearly per manifest.
  - C_full was measured S@16,384 → H@131,072; switching from an S at 51,200 needs the S snapshot at that length, untested.

**Final cleanup receipt (P0–P5)** ✔:
- RCM Idle;
- every campaign's final drain, gateway-down and release succeeded, with no non-REMOVED owned PG (P2 `results/switch/full-7a3aaeb/`, P3 `results/switch/partial-7a3aaeb/`, P4/P5 `results/pilot/*-1edf1c8-*-final.json.gz`);
- TPU-host tmpfs snapshots removed by the P2 and P3 EXIT cleanups (`cleanup-tmpfs.json`); P4/P5 used none;
- state line `w7-campaign 1edf1c8 released H`;
- local `/dev/shm/hetero-scratch` removed; `/` has 61 GB free.
- Head pod:
  - `longctx` (21 MB) and `midctx` (28 MB, sha `1222a9b3…`) catalogues are staged at `/tmp/omp_hetero/data/data/workloads/` (kept for reruns); the truncated copy from P2 attempt 1 was overwritten and verified;
  - run-switch artifacts sit under `/tmp/omp_hetero/artifacts/{full,partial}-*` (copied locally);
  - Q2 leftovers are unchanged.

**Decisions for the user (after P0–P5)**

1. Use the measured costs as the study's C:
   - full S→H 132 s (production, in-process; a published-style harness adds about 120 s);
   - partial S→Mb 129 s effective with a 2 s kept-pair gap.
   The projected 160.5 / 186 s regimes can be retired. The next lever is the engine path: process start 21.5 s, initialize_from_config 26 s (it was 4.2 s in the fp32 era, unexplained), compile/warm-up 55 s. Q5 (bucket restriction) and an initialize_from_config profile target exactly these. Both are outside this queue.
2. W0 on H at 131,072 is 12.6% faster than at 16,384 (block size and policy are confounded). Should one short run isolate the cause (H @131,072 with the base 16-block reserve)? It moves k_s* (2.14 vs 1.67).
3. W7 makes H the faster layout for mid-length contexts: S→H pays after 3.8 W7 manifests (W0 → W7 jobs, k_l = 0). Should W7 on M (W7 on the TP16 alongside W0 on the TP8s) and W0 on S at 51,200 be measured, to complete the mid-job arms?
4. R4 and Q5 remain paused, as instructed.

### O0 — local change (done) and spec (cluster)

Local ✔ (`research/bf16-kv` @ `dbc5513`):
- Cause of the 1.0 s threshold ✔: the run-switch / `switch_campaign.sh` path never set the thresholds. `FAST` carries no `worker_env` (`scripts/switch_campaign.sh:59`), so `configure_compile_cache` (`hetero/compile_cache.py:4-19`, called at `hetero/tpu_worker.py:251`) received neither variable and JAX 0.10.2 kept its default `min_compile_time_secs = 1.0`. The 0/0 env existed only in the old overlap drivers (`scripts/overlap_restore_sequence.sh:60`). The cache dir is vLLM's pod-local `/root/.cache/vllm/xla_cache` (`tpu_inference/runner/compilation_manager.py:64-67`).
- Fix: `hetero/layout.py` puts `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` and `…_MIN_ENTRY_SIZE_BYTES=0` into every instance's intended runtime env (`PERSISTENT_CACHE_THRESHOLD_ENV`), so every deploy (normal, restore, run-switch target, partial joiner) applies them in each worker before `init_device` and `verify_compile_cache` fails the start if anything overrides them. JAX 0.10.2 only overrides an entry-size of 0 with `default_min_cache_entry_size()`, which returns 0 (`jax/_src/compilation_cache.py:230-232,262-264`); a write is skipped only for `process_id != 0` (0 on every worker, recorded) or host callbacks (`jax/_src/compiler.py:802-821`). The layout digest changes, so prep rebuilds snapshots and references as usual.
- Test: `tests/test_compile_cache.py::test_every_layout_process_applies_zero_thresholds` (every builtin layout and instance). Suite 296 passed, the same 16 baseline failures.
- `scripts/cache_hit_summary.py REPORT…`: per worker and startup phase, persistent-cache lookups that hit or missed, the exclusive-wall split, and the thresholds read from both the worker's `compile_cache` report and each startup profile. On the P2/P3 reports it reproduces the decomposition ✔: run-switch H targets `initialize_from_config` 48/1,848 hits (2.6%, i.e. 2/77 per worker), `compile_or_warm_up_model` 264/4,416 (6.0%, 11/184); P3 joiners the same; S restore deploy 0/77 and 11/184.

Cluster spec:
- Hypothesis [INF]: with 0/0, after the prep deploys have written the small entries, every target start hits ≥ 95% of lookups in both phases. `backend_compile_and_load` (23.6 + 19.5 s on H) becomes executable loads (a few s), so engine init falls from 90.7 to ≈ 50 s and C_full from 132.2 to ≈ 90–95 s; C_eff (partial) from 128.9 to ≈ 88–92 s. Trace + lower (≈ 29 s on H) and process start (21.5 s) are unchanged.
- Commit `dbc5513`, frozen for the timed arms. Everything else as P2/P3 (same FAST settings, policies, `max_model_len`, statics 206.2 / 409.2 s).
- Steps: `KUBECONFIG=$HOME/.kube/hetero-agent.yaml HEAD_POD=tpu-ray-cluster-vllm-tpu-head-gxwsv REPS=3 flock /tmp/hetero-cluster.lock scripts/switch_campaign.sh full` (run ids `full-dbc5513-*`), then the same with `partial` (`partial-dbc5513-*`). Then locally `switch_summary.py` on both, and `cache_hit_summary.py` on every run-switch `deploy_report.json.gz`, the return and prep deploys.
- Endpoints: as P2 (production C with segments, harness-equivalent C) and P3 (kept gap, join delay, C_eff, join tail); plus hits/misses per phase (all workers and per worker) and the thresholds of every worker report and profile.
- Stop criteria: as P2/P3 (≤ 2 failed attempts per campaign; fail-closed teardown). A worker whose report or profile shows a threshold other than 0/0 means the spec did not apply: stop that campaign and record it. A hit rate below 95% is a result, not a stop: finish the reps and list the missing modules.
- Disk floors before each stage as in P2 (`/` ≥ 10 GB, `/dev/shm` ≥ 20 GB). Worker pod caches at start: 806 MB, 118–119 entries, 75 GB free on `/` (read-only `du`/`df`). Expected ≈ 1.5 h + 1.3 h.

O0 full attempt 1 (`dbc5513`, 2026-10-06 21:2x–21:49Z, `results/switch/full-dbc5513-interrupted/`): prep and the H restore check + validation succeeded ✔. The driver was then killed by SIGTERM from the controller session (not by the cluster) at the start of the static W6 pilot, before its EXIT cleanup, so restored H stayed deployed. Manual cleanup with the driver's own commands at 2026-10-07 ✔: drain H `--force`, gateway-down (removed), release H/Mb/S (all PGs REMOVED), tmpfs cleanup ok (`cleanup-*.json`); RCM Idle. Early O0 signal from this attempt ✔ (`cache_hit_summary.py`; every worker report and profile 0/0):

| deploy (8 workers) | `initialize_from_config` hits | wall (median) | `compile_or_warm_up_model` hits | wall (median) | of which trace / lower / backend compile / executable load |
|---|---|---|---|---|---|
| prep S normal (first 0/0 start: writes) | 0/616 | 26.7 s | 104/1,472 | 75.2 s | 17.9 / 28.0 / 19.5 / 2.9 |
| prep H normal (first 0/0 start: writes) | 16/616 | 27.1 s | 104/1,472 | 58.1 s | 14.0 / 15.0 / 19.5 / 2.7 |
| **H restore (next start)** | **616/616 (100%)** | **2.6 s** (P2: 25.9) | **1,472/1,472 (100%)** | **34.9 s** (P2: 54.6) | 13.7 / 14.7 / **0.0** / 2.9 |

The rerun is detached from the controller session (`setsid`; `/dev/shm/hetero-scratch/o0/run_o0.sh`: full, then partial, each under the cluster lock), with the same commit, run ids and steps.

### O0 — results (`research/bf16-kv` @ `dbc5513`, runs `full-dbc5513-*`, `partial-dbc5513-*`)

Artifacts (evidence commit `146edc0`): `~/hetero-wt/Bf16Kv/results/switch/{full,partial}-dbc5513/` (driver outputs, `switch_summary.{md,json}`) and `results/hetero/artifacts/{full,partial}-dbc5513-*/`; driver logs `results/switch/o0-{full,partial}-driver-dbc5513.log`. Both campaigns exited 0 with no failed attempt. In every rep: exact accounting in both phases, 0 failed requests, all workers restored, gates opened on the controller event, offline = in-process segments (max diff 0.00 s), prep restore equality on S, H (and Mb).

**Cache ✔**: every worker's `compile_cache` report and every startup profile reads 0/0, on all switch targets, joiners, harness returns and prep deploys.

| hits / lookups (all workers) | `initialize_from_config` | `compile_or_warm_up_model` | P2/P3 (1.0 s threshold) |
|---|---|---|---|
| full: H targets r1–r3 (24 workers) | 1,848/1,848 (**100%**) | 4,416/4,416 (**100%**) | 2.6% / 6.0% |
| partial: TP16 joiners r1–r3 (12 workers) | 924/924 (**100%**) | 2,208/2,208 (**100%**) | 2.6% / 6.0% |
| harness returns: S r1–r3 (24 workers) | 1,848/1,848 (100%) | 4,416/4,416 (100%) | 0% / 6.0% |

H target medians, O0 (P2): `initialize_from_config` 2.6 s (25.9), `compile_or_warm_up_model` 35.0 s (54.6) = trace 13.7 + lower 14.9 + backend compile **0.0** (19.5) + lookup 0.7 + executable load 2.9 + warm-up 2.0. S return `compile_or_warm_up_model` 52.2 s (72.0) = trace 18.1 + lower 27.4 + executable load 3.0. What remains is exactly trace + lower.

**Full S→H, production C ✔**:

| | r1 | r2 | r3 | mean | P2 |
|---|---|---|---|---|---|
| T0 → first W6 dispatch | 87.32 | 87.97 | 86.43 | **87.24 s** | 130.64 |
| T0 → first W6 token | 92.57 | 92.70 | 92.04 | 92.44 s | 135.26 |
| ramp (makespan − 206.2) | −0.28 | 1.74 | 6.29 | 2.58 s | 1.58 |
| **C_production** | 87.04 | 89.71 | 92.72 | **89.82 s** | 132.22 |
| C, iteration-bin deficit (diagnostic) | 109.25 | 104.90 | 99.41 | 104.52 s | 144.17 |

Segments (mean, s): controller wake 0.01, drain + release 1.95, gateway down 0.17, deploy controller pre 11.07, engine process start 21.56, **engine init 47.23** (P2 90.73; init_device 3.6, load_model 4.8, initialize_from_config 2.6, compile/warm-up 35.2), endpoint register 2.02, gateway up 3.23, gate/capture/dispatch 0.01. Sum = gap.

Harness-equivalent return H→S ✔: C_harness **222.3 s** (214.9 / 226.0 / 225.9; P2 273.7), of which the S deploy job is 136.2 s (P2 189.4).

**Partial S→Mb ✔**:

| | r1 | r2 | r3 | mean | P3 |
|---|---|---|---|---|---|
| kept-pair serving gap | 1.92 | 2.03 | 2.65 | 2.20 s | 2.08 |
| join delay (first W6 dispatch) | 84.90 | 84.34 | 88.36 | **85.87 s** | 128.41 |
| first W6 token | 89.27 | 89.42 | 93.09 | 90.59 s | 133.14 |
| kept-pair W0 tok/s during the join | 2,557 | 2,567 | 2,567 | 2,564 | 2,522 |
| kept-class finish | 339.5 | 342.9 | 361.4 | 347.9 s | 343.4 |
| W6-class finish | 500.9 | 492.1 | 496.4 | 496.5 s | 538.1 |
| **effective C = W6-class finish − 409.2** | 91.66 | 82.89 | 87.21 | **87.25 s** | 128.92 |

Exit criterion met ✔: C with segments and ≥ 95% cache hits in both phases. The hypothesis (C_full ≈ 90–95, C_eff ≈ 88–92 s) holds. The remaining host-side critical path on the target is controller pre 11.1 + process start 21.6 + trace/lower 28.6 ≈ 61 s of the 87 s gap, which is what O1 targets. Phase-1 W0 on S takes 191.6 / 193.9 / 192.8 s here (P2: 198.4 / 196.0 / 190.6); this is the source baseline for O2/O3.

**Cleanup receipt (O0)** ✔: both EXIT cleanups released H/Mb/S, with every owned PG REMOVED and tmpfs cleanup ok (`cleanup-*.json` in both directories); RCM Idle; state line `switch-campaign partial-dbc5513 released S switch-r3-leftover-S`. Worker pod caches grew with the 0/0 entries (not removed: they are the warm cache).

### O1 — host-only target preparation: feasibility and design (local) — done

Installed worker versions ✔ (pip metadata via read-only `kubectl exec`, no jax import): Python 3.12.13, jax 0.10.2, jaxlib 0.10.2, libtpu 0.0.43, ray 2.56.1, vllm 0.23.1rc1.dev1474+g0ba2aa35a.tpu (`/workspace/vllm`), tpu_inference `/workspace/tpu_inference` @ `8d31f0866` (image `vllm/vllm-tpu:v0.26.0`). An identical CPU venv (same wheels) and a read-only copy of the installed vllm/tpu_inference sources were used for all evidence; `compilation_manager.py`, `tpu_runner.py` and `tpu_worker.py` match our fork by md5 (only `gemma4.py` is patched at deploy). Line numbers: installed JAX (`jax/_src/…`) and installed tpu_inference/vllm. Prototypes, tests and strace logs: `~/hetero-wt/OverlapSwitch/results/overlap/o1/o1-prototypes.tar.gz` (three opus-5-5 research agents; tests rerun by the organizer: 5 + 4 passed ✔).

| Step | Can it run before release? | Evidence | Verdict |
|---|---|---|---|
| (a) engine process up to the first device touch | **Yes**, in CPU-only Ray bundles | First device touch is `jax.devices()` in `TPUWorker.init_device` (`tpu_inference/worker/tpu_worker.py:410`); `import libtpu` only sets a path, tpu_info reads env and `/dev` globs, the driver's `libtpu.sdk` import maps no `libtpu.so` (prototype). A TPU-bundle PG stays pending while the source holds `TPU:4`; a bundle without the TPU key is accepted; the unmodified executor's `resources={"TPU":0}` is rejected by Ray 2.56.1, so the executor needs a subclass. Attribution of the 32.5 s ✔ (P2 r1 logs): controller pre 11.0 = backend prep 2.1 + PG 0.1 + actor 0.5 + **driver vLLM import 8.2**; process start 21.5 = engine config 1.5 + tokenizer/InputProcessor ≈ 7.4 [INF split] + executor/actor 1.7 + **worker imports 10.4** + prelude 0.6. Idle parked RSS ≈ 1 GB per worker, 1.8 GB driver (hosts ≈ 164 GB free). Gate prototype: release → device 55 ms; premature release waits for the source to exit (device 56 ms after kill). | **Use** (removes ≈ 32.5 s) |
| (b) trace + lower for the target mesh without devices | Possible (deviceless `v5e:4x4` / `v5e:2x4` topology: 16 devices in 4 virtual processes, ids and coords equal H's; AbstractMesh export), but **not reusable by the target** | All JAX trace/lower caches are in-process and keyed by Python objects: `_infer_params_cached` (`pjit.py:593-598`), `_pjit_lower` (`pjit.py:1247-1271`), `MeshComputation._executable` (`pxla.py:1236-1244`), `_cached_compilation` (`pxla.py:1499-1519`). The persistent-cache key hashes the lowered module (`cache_key.py:101-107`), so a fresh process always traces and lowers before a lookup (prototype: 6/6 hits, still 7 ms trace + 35 ms lower). `jax.export` reuse is slower than native trace+lower (0.12–0.33 vs 0.05 s) and compiles under another key; `flatbuffers` is not installed on workers. | **Reject** |
| (c) serialized compiled executables keyed by our identity | Produced at provisioning on real devices; loaded after release | `serialize_executable` round-trip in two fresh processes (8-device CPU mesh, donate + static args): `deserialize_and_load` 3.2 ms vs 73 ms first jit call; no trace/lower/compile events; C++ fast path (103.0 vs 102.3 µs per call); shape/sharding mismatches raise `TypeError`/`ValueError` before execution. run_model-shaped model (48 layers, 8 buckets): bank load 1.65 s vs 8.8 s native first calls, equal outputs. Bankable share of the P2 H start: trace 99.9%, lower 98.5%, compile/load 94.1% (≈ 250 entries per instance shape: run_model 8, sample 28, substitute 144, select 41, logits 7, structured 7, gather 7, prompt 8); the rest is ≈ 3 s of eager glue, covered by O0. Serving call sites to wrap: `tpu_runner.py:1630,1688,1693,1698,1701,1994,2006,2009,2015,2055,1376,2575`; the skip goes into `CompilationManager._run_compilation` (`compilation_manager.py:126-196`, incl. compile_only callers). Deviceless TPU executables serialize (142 KB) but loading one needs a real client — untested. | **Use** (removes trace + lower ≈ 29 s H / 46 s S, plus residual compile) |
| (d) libtpu single-owner lock | Not needed by the chosen design | The lock is a POSIX `fcntl(F_SETLK, F_WRLCK)` on `/tmp/libtpu_lockfile`, taken inside `PJRT_Plugin_Initialize` (strace between plugin-load and plugin-initialized markers), not at dlopen; the deviceless topology path calls it too, which is the prior "already in use" failure. `ALLOW_MULTIPLE_LIBTPU_LOAD=1` (string in `libtpu.so` 0.0.43) skips the lockfile entirely; with `JAX_PLATFORMS=cpu` + `TPU_SKIP_MDS_QUERY=1` a simulated v5e host showed no lockfile, `/dev/vfio` open or VFIO ioctl. Without `JAX_PLATFORMS=cpu`, `Lowered.compile()` builds a real client and opens `/dev/vfio/{0..3}` (`compiler.py:73-92`, `xla_bridge.py:198-219`) — a second owner. The deviceless key matched across hosts, but it also depends on `XLA_FLAGS`/`LIBTPU_INIT_ARGS` tokens and the cache-dir string (`cache_key.py:342-365`, `compiler.py:265-281`). | Deviceless background compile **rejected for the switch path**: under the same-model rule it only rebuilds what the warm cache already holds, and it is the one design that needs a lock bypass |

**Chosen design**: O0 (cache 0/0) + (a) pre-spawn + (c) executable bank.
- Pre-spawn: during the source phase the target driver and workers start in CPU-only bundles and each worker parks in `HeteroTPUWorker.init_device` before `super().init_device`, on a ReleaseGate. The controller releases each host after the source drain and PG release; the worker then waits until `/tmp/libtpu_lockfile` carries no foreign lock and no other pid holds `/dev/vfio` fds (bounded).
- Bank: a provisioning restore deploy with `exec_bank="write"` serializes every precompiled bucket per worker (pod-local, identity-keyed manifest). A target with `exec_bank="read"` loads it after `initialize_from_config`, replaces the runner's jitted callables with a dispatcher (bucket → `Compiled`; a miss or rejection falls back to the jit and is recorded), and skips lower/compile for banked buckets.
- Predicted C [INF] ≈ drain 2 + gateway 0.2 + gate + init_device 4 + load_model 5 + initialize_from_config (after O0) + bank load + warm-up 2 + endpoint/gateway 5.3 ≈ 25–35 s, against 132 s.

**Advisor review** (fable-5-1): **accept with changes**. Ranked risks:
1. A premature device touch in a parked worker. Any lockfile or vfio access before the gate is a hard fail.
2. Handoff latency (source actor exit → last vfio fd closed → target `jax.devices()`) is unmeasured and could dominate.
3. A bank hit/miss divergence across hosts would hang a collective. Use the bank only if all ranks agree.
4. `Compiled` fidelity (donation, placement, out shardings) and the per-executable TPU load time × 250.
5. Identity gaps. Add `LIBTPU_INIT_ARGS`, `XLA_FLAGS`, JAX config, vllm/tpu_inference versions, dtypes, process index and host order.
6. `initialize_from_config` (26 s) must be decomposed; O0 measures it.

Required gate proofs, folded into O2:
- parked workers show no lockfile or vfio access while the source serves, and the source is unaffected;
- the measured handoff timeline;
- banked executables give bitwise-identical output to native jit on the live mesh (within-layout greedy equality), with a measured load time;
- a forced single-host bank miss makes every rank fall back, with no hang.

Implementation ✔ (local; `~/hetero-wt/OverlapSwitch`, branch `overlap/prespawn-bank`, from `dbc5513`; two opus-5-5 implementers, reviewed by the organizer and a reviewer agent):
- `eac4391` executable bank (`hetero/exec_bank.py`). `BankedFunction` replaces each precompiled jit at its call sites: the `run_model` closure cell, `compute_logits`, the sample/logprob module globals, select, substitute and structured decode. Its bucket key is the static values plus the shape/dtype of the per-bucket arguments; weights and KV are left out of the key but `Compiled` still validates them. `CompilationManager._run_compilation` is wrapped so that banked buckets skip lower/compile while warm-up still runs through the bank. Write mode captures every AOT `Compiled`; the store is pod-local per rank with an atomic rename and a sha256 manifest.
  - Identity: model, KV, scheduler shape, `num_blocks`, mesh and host order, topology, versions, `platform_version`, `XLA_FLAGS`/`LIBTPU_INIT_ARGS`, JAX config, revisions, digests of the runner sources.
  - Unanimity: rank 0 collects every rank's vote over Ray `internal_kv` (120 s bound) and every rank follows its verdict, so ranks never mix banked and jit paths.
  - Evidence: `omp_init_phases()["exec_bank"]` with per-function hits/misses/rejections. CPU tests: 11.
- `37e1317` pre-spawn (`hetero/prespawn.py`, split `deployment.deploy`, `switch.py`):
  - CPU-only placement groups; a `/proc/self` check before parking and again when released (a mapped `libtpu.so`, a vfio fd or a lockfile fd raises); the ReleaseGate actor; a bounded wait until the libtpu lock and the vfio fds are free; TPU-hold groups; registration only at commit; fail-closed abort.
  - New reconciled segments `prespawn_wait`, `prespawn_release`, `device_handoff`, plus a source-delivery window before, during and after the spawn.
  - `switch_campaign.sh` gains `OVERLAP=1` and `GATE=1`; the `OVERLAP=0` dry-run output is byte-identical to before. CPU tests: 30.
- `2dd7d2e` (organizer): under `OVERLAP`, the bank-read target's greedy tokens are checked against the native normal-loader reference after phase 2, outside C, as a hard check. Full mode checks H; partial checks Mb; the sabotage deploy is checked too.
- `47b5c67` (organizer, fixing the reviewer's blocker ✔). The installed vLLM `get_bundles_sorted_by_node` drops bundles without a TPU key (`vllm/v1/executor/ray_utils.py:429-432`), so a CPU-only group would have no worker ranks (`ray_executor_v2.py:330`). The pre-spawn executor now swaps in an unfiltered version with the same driver-first order, only inside its own `_init_executor`. Tests: 32 pre-spawn tests; the full suite has 339 passed, with the same 8 failed and 8 errors as the baseline.
- Reviewer should-fix items, not fixed (they do not arise in the campaign path) [INF]:
  - TPU-hold groups are named `<instance>-tpuhold`, so `drain` and `release --instances` miss them. The campaign only releases whole layouts, which removes them.
  - A spawn still blocked when phase 1 fails is not aborted by the switch. The driver's EXIT `release H/Mb/S` removes its groups and actors.

### O2 — spec (same-host coexistence gate)

- Hypothesis [INF]: parked target workers (imports, engine config, tokenizer, Ray actors; RSS ≈ 1–2 GB per host) coexist with a serving S without touching the TPU. After release the target takes the devices within about 1 s of the source exit. It loads ≈ 250 banked executables per rank in a few seconds, runs no trace or lower for them, and its greedy outputs equal the native reference. A forced single-rank bank refusal makes every rank fall back with no hang. Source W0 slows by less than 5% while workers spawn.
- Commit `overlap/prespawn-bank` @ `46ba270`, frozen. It is `47b5c67` plus a driver-only change: the restore `check --require-restored` is kept before each greedy validation, because `validate` records `require_restored` but does not enforce it remotely. Run from `~/hetero-wt/OverlapSwitch` (`.venv`/`data` symlinked to `Bf16Kv`). Settings as O0, plus: bank write on the prep H restore deploy; pre-spawn + bank read on the run-switch target, with `--prespawn-at-s 0`.
- Steps: `KUBECONFIG=… HEAD_POD=tpu-ray-cluster-vllm-tpu-head-gxwsv GATE=1 flock /tmp/hetero-cluster.lock scripts/switch_campaign.sh full` (run ids `full-46ba270-ov-*`, output `results/switch/full-46ba270-ov/`), detached as in O0:
  1. Prep as P2/O0: S and H normal → snapshot → reference → teardown; H restore with `exec_bank=write` → check + validate → static W6 → teardown.
  2. Sabotage deploy: H restore, `exec_bank=read`, `exec_bank_sabotage_rank=1` → check → validate against the H reference → teardown.
  3. S restore → check + validate.
  4. One overlapped run-switch (S W0 → H W6, pre-spawn + bank read) → greedy validation of H → the harness return to S → final teardown.
- Pass conditions, all hard:
  - (i) **Source health**: phase 1 complete with exact accounting (1,763 req / 721,595 tok, 0 failed).
  - (ii) **Throughput impact measured**: tokens/s during and after the spawn vs before it and vs O0 phase 1 (reported, not gated).
  - (iii) **No second TPU owner**: every pre-spawned worker's self-checks are clean at park and at release, no foreign lock/vfio holder appears before release, and every `parked_at` ≤ `released_at`.
  - (iv) **Target consumption**: on every target worker the bank unanimity verdict is `use`, entries are loaded, AOT buckets are skipped, rejections = 0 and no serving misses on banked functions. Startup profiles show trace/lower only for unbanked glue. Greedy validation equals the reference.
  - (v) **Sabotage**: every rank reports `use = false` with the same reason, the deploy finishes, and validation passes.
  - (vi) **Bounded failure**: any failure ends within the driver's timeouts (park 900 s, device-free 120 s, `ARM_TIMEOUT` 900 s, `SWITCH_TIMEOUT` 5400 s) and the EXIT cleanup releases only H/Mb/S groups of this campaign.
- Endpoints: the handoff timeline per worker (release → lock free → vfio free → devices ready); bank load s; C with the new segments (indicative, n = 1).
- Stop: a failed condition is a blocker. Allow at most 2 fix attempts (new commit, new run ids, re-freeze); then record and stop O2/O3. Expected ≈ 1.5 h.

O2 attempt 1 (`46ba270`, 2026-10-07 07:4x–08:35Z, `~/hetero-wt/OverlapSwitch/results/switch/full-46ba270-ov/`, driver log `results/switch/o2-gate-driver-46ba270.log`) — **failed at target warm-up; blocker found and fixed**:
- What passed ✔:
  - Prep. The bank-write H restore wrote **187 entries per rank on all 8 workers, 0 skipped**.
  - Sabotage (v): all 8 ranks reported `use = false` with reason `ranks refused: {1: 'sabotaged …'}`; the deploy finished; check and greedy validation passed (32 prompts × 2 engines, token for token).
  - Pre-spawn during phase 1: spawn 0–2.7 s, **all 8 hosts parked at 34.1 s**, phase 1 complete with T0 at 193.5 s (O0 phase 1: 191.6–193.9 s).
  - Release 1.6–2.6 s after T0. `init_device`, `load_model`, KV allocation, bank load and **unanimity `use`** all completed on the target.
- Failure ✔ (Ray worker log on `192.168.102.47`, `worker-cd06…-311880.err`): the first banked `run_model` warm-up call aborted in PjRt with `INVALID_ARGUMENT: PjRtBuffer's memory space is addressed by device TPU_5(process=0,(1,1,0,0)) vs sharding is on device TPU_4(process=0,(0,1,0,0))` (stack: `exec_bank.py:358` → `stages.py:890` → `pxla.py:395`).
- Cause ✔: `jax.experimental.serialize_executable.deserialize_and_load` loads the XLA executable onto `backend.devices()` in id order when `execution_devices` is not given (`serialize_executable.py:63-64,116-119`). The unpickled shardings keep the mesh order, which on a TP16 sub-slice is ids `[0, 1, 4, 5, 2, 3, …]`. The CPU tests used an id-ordered mesh, so they could not see this. Reproduced on CPU with a permuted 8-device mesh (`OverlapSwitch/results/overlap/o2/perm_probe.py`): loaded local devices `[0..7]` vs compiled `[0,1,4,5,2,3,6,7]`, the same abort; with `execution_devices` in banked order, outputs are equal.
- Rerun after the driver's fail-closed rebuild of S, same commit: `large_a` (hosts .47–.51) aborted the same way. `large_b`'s rank on `192.168.102.54` failed earlier, in `init_device`, with `open(/dev/vfio/0): 16: Device or resource busy` (the next attempt's failure, below). The driver stopped after 2 attempts.
- Bounded failure (vi) ✔: each failure surfaced within 27 s of T0. The driver tore down H/Mb/S, and the EXIT cleanup released H/Mb/S, including both `-tpuhold` groups (all REMOVED). tmpfs cleanup ok; RCM Idle.
- **Fix attempt 1** (`19adba0`): each entry stores its executable's device assignment (`_unloaded_executable.device_list` ids) and is loaded with `execution_devices` in that order. After loading, the XLA executable's `local_devices()` order is checked against the banked order, and a mismatch is a refusal. The test mesh now uses the TPU-like permuted order: red before the fix, green after (11 passed on jax 0.11.2; 9 on the 0.10.2 venv); suite 339 passed with the baseline failures.

O2 attempt 2 (code `19adba0`; HEAD `0b4841d` = `19adba0` + the attempt-1 evidence commit, so run ids are `full-0b4841d-ov-*`; 2026-10-07 08:4x–09:25Z; driver log `results/switch/o2-gate-driver-0b4841d.log`) — **failed at the device handoff; blocker found, fix attempt 2**:
- Prep, bank write and sabotage passed as before. Pre-spawn parked all 8 hosts at 34.3–34.5 s of phase 1, and phase 1 completed (T0 at 190.8 / 187.3 s).
- Failure ✔: in both run-switch attempts, `large_b`'s worker on `192.168.102.54` failed in `jax.devices()` (`tpu_inference/worker/tpu_worker.py:410`) with `TPU initialization failed: open(/dev/vfio/0): 16: Device or resource busy`, 1.0 s and 17.8 s after its gate opened. The gate's device-free wait had returned: /proc showed no lockfile lock and no `/dev/vfio` fd held by another process.
- Cause [INF, consistent with the evidence]: the killed source worker closes its fds early in process exit, but the VFIO group stays busy until the kernel's release path (device reset) finishes. A /proc scan cannot see that window. On the normal path it is hidden behind ≈ 32 s of target process start.
- Bounded failure (vi) ✔: the same teardown and EXIT cleanup, all groups REMOVED, RCM Idle.
- **Fix attempt 2 (last allowed)** `d510c36`: the device-free wait also requires every `/dev/vfio/<N>` group to open. It probes with `open(O_RDWR)` and closes at once; it is safe because a VFIO group admits one opener and the probe runs only after the lock and /proc checks are clear. `EBUSY` keeps waiting (bounded at 120 s), and the gate record gains `groups_openable_at` and `busy_polls`. Tests: 34 pre-spawn tests; suite 341 passed with the baseline failures. The rerun uses run ids `full-d510c36-ov-*`. If it fails, O2 stops with a recorded blocker.

### O2 — results (`overlap/prespawn-bank` @ `d510c36`, run `full-d510c36-ov-r1`) — **gate PASSED**

Artifacts: evidence commit after `d510c36` on `overlap/prespawn-bank`, under `~/hetero-wt/OverlapSwitch/results/switch/full-d510c36-ov/` (driver outputs, `switch_summary.md`, `overlap_check.json`) and `results/hetero/artifacts/full-d510c36-ov-*`; driver log `results/switch/o2-gate-driver-d510c36.log`; checker `scripts/overlap_check.py`. The campaign exited 0 on its first attempt.

| Condition | Verdict | Evidence ✔ |
|---|---|---|
| (i) source health | **pass** | Phase 1 complete with exact accounting (1,763 req / 721,595 tok, 0 failed). Phase 1 lasted 189.2 s (O0: 191.6 / 193.9 / 192.8). |
| (ii) source throughput impact | **none measurable** (n = 1) | Spawn window, first 34.3 s of phase 1: 2,661 tok/s (O0 same window: 2,639 / 2,665 / 2,667). Parked, 34.3 s → T0: 4,070 tok/s (O0: 4,013 / 3,949 / 3,975). |
| (iii) no second TPU owner | **pass** | 8/8 workers parked 155 s before release. Self-checks were clean at park and at release: no `libtpu.so` mapped, no vfio or lockfile fd. No foreign lock or /proc vfio holder after release. On the 4 `large_b` hosts every `/dev/vfio/0–3` was still EBUSY for 0.42–0.65 s after release; the fix waited that out (the attempt-2 failure mode). Release → devices ready: median 4.32 s, max 4.55 s, of which `init_device` takes 4.08. |
| (iv) target consumption | **pass** | 8/8 workers: unanimity `use`, 187 entries loaded in 3.41–3.57 s, all 199 AOT buckets skipped. `compile_or_warm_up_model` 1.52 s (O0 35.0): lower 0.11, trace 0, backend compile 0. After phase 2 + validation, banked calls hit with **0 misses and 0 rejections** (`run_model` 65,956 hits; logits, sample, select ≈ 65.9k each). Greedy validation equals the native normal-loader reference token for token (32 prompts × 2 engines). |
| (v) sabotage fallback | **pass** | Repeated in this run: 8/8 ranks `use = false` with `ranks refused: {1: 'sabotaged …'}`; the deploy finished; check and greedy validation passed. |
| (vi) bounded failure / cleanup | **pass** | The two earlier failures ended within 27 s of T0 with owned-only cleanup. This run's final teardown and EXIT cleanup ✔: H/Mb/S all REMOVED including `-tpuhold`, tmpfs ok, RCM Idle, state `released S final`. |

Indicative overlapped full S→H, n = 1 ✔:
- Production C **23.32 s** (gap to first W6 dispatch 24.11 s, first token 29.64 s, ramp −0.79 s); with the iteration-bin deficit 38.4 s.
- Segments: drain + release 1.91, gateway down 0.18, pre-spawn release 0.21, device handoff 0.65, **engine init 17.02**, endpoint register 1.16, gateway up 2.96. Sum = gap.
- Engine init split (rank 0): init_device 4.08, load_model 4.85, initialize_from_config 2.6, bank load 3.45, compile/warm-up 1.52.
- The harness return C (277.7 s) is not comparable to O0: the post-switch greedy validation now sits in its switch tail (62.3 s vs 14.4 s).

### O3 — spec (integrated overlap switch, full and partial, 3 reps each)

- Hypothesis [INF]:
  - full S→H C ≈ 23–26 s, partial S→Mb effective C ≈ 22–28 s, kept-pair gap ≈ 2 s;
  - no measurable phase-1 slowdown;
  - the partial joiner's bank (written by an extra Mb restore deploy, `large_b` only) is used on 4/4 workers.
- Code = `d510c36` (frozen). The run ids carry the HEAD at launch, i.e. the O2 evidence commit; `hetero/` is unchanged, so `RUNTIME_REVISION` and the bank identity are those of `d510c36`. Run from `~/hetero-wt/OverlapSwitch`, detached, under the cluster lock.
- Steps:
  - `OVERLAP=1 REPS=3 scripts/switch_campaign.sh full`: prep with bank write on the H restore → 3 × (S restore → run-switch with pre-spawn at phase-1 start + bank read → H check + greedy validation → harness return).
  - Then `OVERLAP=1 REPS=3 … partial`: prep plus the Mb bank deploy → 3 × (S restore → run-switch partial with `large_b` pre-spawned on the released TP8 hosts + bank read → Mb check + greedy validation → teardown).
  - Locally: `switch_summary.py` and `overlap_check.py` on every rep.
  - Region maps: `bulk_cost_model.py` (`ReconfigLevers`) at the new means, with every other input as in P5.
- Endpoints:
  - full: production C with segments (mean of 3 and per rep);
  - partial: kept gap, join delay, C_eff, join tail, kept-pair tok/s;
  - source impact: phase-1 tok/s during and after the spawn vs O0;
  - every O2 check per rep;
  - long-job and mid-job region-map shares and break-evens vs P5 and O0.
- Stop criteria: as P2/P3 (≤ 2 failed attempts per campaign, fail-closed). A failed O2 check in any rep is a blocker: at most 2 fix attempts, then record and stop. Expected ≈ 1.6 h + 1.6 h.

### O3 — results (code `d510c36`, runs `full-c762edb-ov-r{1,2,3}`, `partial-c762edb-ov-r{1,2,3}`)

Artifacts: `~/hetero-wt/OverlapSwitch` (branch `overlap/prespawn-bank`, evidence commit `bbbeb4f`), under `results/switch/{full,partial}-c762edb-ov/` (`switch_summary.md`, `overlap_check.json`) and `results/hetero/artifacts/{full,partial}-c762edb-ov-*`; driver logs `results/switch/o3-{full,partial}-driver-c762edb.log`. Both campaigns exited 0, with every rep passing on its first attempt.

Every rep ✔:
- complete with exact accounting in both phases, 0 failed requests, all restored, segments reconciled;
- every O2 check: parked before T0 and before release, self-checks clean, bank `use` on every target worker (8/8 full, 4/4 joiner), backend compile 0 in compile/warm-up;
- post-switch greedy validation equal to the native reference, with **0 bank misses and 0 rejections** over all serving calls.

The EBUSY wait fired in 5 of 6 reps, on the four `large_b` hosts.

**Full S→H, production C ✔**:

| | r1 | r2 | r3 | mean | O0 | P2 |
|---|---|---|---|---|---|---|
| T0 → first W6 dispatch | 23.96 | 24.23 | 23.75 | **23.98 s** | 87.24 | 130.64 |
| T0 → first W6 token | 28.28 | 29.50 | 29.16 | 28.98 s | 92.44 | 135.26 |
| ramp (makespan − 206.2) | 5.92 | 3.12 | 1.55 | 3.53 s | 2.58 | 1.58 |
| **C_production** | 29.88 | 27.35 | 25.30 | **27.51 s** | 89.82 | 132.22 |
| C, iteration-bin deficit (diagnostic) | 48.86 | 42.25 | 40.69 | 43.93 s | 104.52 | 144.17 |

Segments (mean, s): drain + release 2.16, gateway down 0.17, pre-spawn release 0.08, device handoff 0.60, **engine init 16.74** (init_device ≈ 4.1, load_model ≈ 4.9, initialize_from_config ≈ 2.6, bank load 3.4–3.5, compile/warm-up ≈ 1.5), endpoint register 1.16, gateway up 3.05. The removed segments are controller pre (11.1), process start (21.6) and trace + lower (28.6).

**Partial S→Mb ✔**:

| | r1 | r2 | r3 | mean | O0 | P3 |
|---|---|---|---|---|---|---|
| kept-pair gap | 2.29 | 1.99 | 2.11 | 2.13 s | 2.20 | 2.08 |
| join delay (first W6 dispatch) | 20.47 | 20.58 | 20.86 | **20.64 s** | 85.87 | 128.41 |
| first W6 token | 25.22 | 25.13 | 25.53 | 25.29 s | 90.59 | 133.14 |
| kept-pair W0 tok/s during / after the join | 2,410 / 2,061 | 2,440 / 2,099 | 2,423 / 2,099 | 2,424 / 2,086 | 2,564 / 1,937 | 2,522 / 1,875 |
| kept-class finish (static M 352.7) | 349.3 | 342.7 | 343.1 | 345.0 s | 347.9 | 343.4 |
| W6-class finish | 427.8 | 424.5 | 431.8 | 428.0 s | 496.5 | 538.1 |
| **effective C = W6-class finish − 409.2** | 18.56 | 15.34 | 22.61 | **18.84 s** | 87.25 | 128.92 |

Join tail: none measurable. W6 on the joiner, first dispatch → finish, takes 407.4 s against 409.2 s on a static M.

**Source impact ✔** (phase-1 W0 on S while the target pre-spawns at phase-1 start):

| | spawn window (first ≈ 34 s), tok/s | parked → T0, tok/s | phase 1 length |
|---|---|---|---|
| O3 full r1–r3 | 2,661 / 2,661 / 2,673 | 3,932 / 4,024 / 4,030 | 194.6 / 191.3 / 190.7 s |
| O3 partial r1–r3 | 2,662 / 2,671 / 2,677 | 4,095 / 3,968 / 4,113 | 188.4 / 193.0 / 187.7 s |
| O0 (no pre-spawn), same windows | 2,639 / 2,665 / 2,667 | 4,013 / 3,949 / 3,975 | 191.6 / 193.9 / 192.8 s |

No slowdown is measurable: every overlapped rep is within the O0 spread or above it. The spawn window coincides with the W0 ramp-up, so its absolute rate is low in both arms.

**Region maps at the overlapped C** ✔ (`~/hetero-wt/ReconfigLevers` @ `377f11a`, `results/bulk_cost_model/model_overlap_fullbf16.{json,txt}`; O0 maps in `model_cache00_fullbf16.*`). Every other input is as in P5 (a0H 267.8 at 131,072; O = 32.6 s). The P5 map recomputes identically apart from provenance strings ✔.

*Long-context job (k_s W0, then k_l W6), grid k_s 0–8 × k_l 0.25–8, O = 32.6 s:*

| C full / partial | static H | static M | S→H | S→M | switching | gain median / max | k_s* (S→H vs H) | S→M boundary k_s at k_l = 1 / 2 / 4 |
|---|---|---|---|---|---|---|---|---|
| P5: 132.2 / 128.9 | 0.24 | 0.14 | 0.24 | 0.38 | 0.62 | 8.3% / 24.5% | 2.14 | 1.53 / 2.69 / 5.01 |
| O0: 89.8 / 87.3 | 0.20 | 0.11 | 0.30 | 0.39 | 0.69 | 8.7% / 25.4% | 1.59 | 1.41 / 2.57 / 4.89 |
| **O3: 27.5 / 18.8** | **0.12** | **0.06** | **0.41** | **0.41** | **0.82** | **9.9% / 26.8%** | **0.78** | 1.21 / 2.37 / 4.69 |
| O3, O = 0 | 0.06 | 0.06 | 0.47 | 0.41 | 0.88 | 10.4% / 27.9% | 0.36 | same |

- O3 winners by k_l:
  - k_l = 1: H 0–0.78, S→H 0.78–0.92, M 0.92–1.21, S→M > 1.21;
  - k_l = 2: H 0–0.78, S→H 0.78–1.98, M 1.98–2.37, S→M > 2.37;
  - k_l = 4: H 0–0.78, S→H 0.78–4.11, M 4.11–4.69, S→M > 4.69;
  - k_l = 8: H 0–0.78, S→H > 0.78.
- Mid-band job: S→H before W7 beats static S (k_l = 0) for k_m > **1.38** W7 manifests (O0: 2.81; P5: 3.78). S→H beats static H for k_s > 0.78 at any k_m and k_l.
- Caveats [INF]: the P5 rate caveats still hold (a0S at 16,384; no M arm in the mid job; linear scaling). The O = 32.6 s S-tail term now exceeds C itself, so the O = 0 row bounds what a tail-aware dispatcher could add.

**What remains on the critical path** (full, 24.0 s gap):
- device-bound engine init ≈ 16.7 s: init_device 4.1, weight restore 4.9, initialize_from_config 2.6 (KV allocation + the sampling/logprob precompile through the cache), bank load 3.4, warm-up 1.5;
- gateway up 3.1, drain + release 2.2, endpoint register 1.2, handoff 0.6 (of which ≤ 0.65 s is waiting for VFIO EBUSY).

**Cleanup receipt (O0–O3)** ✔:
- every campaign's EXIT cleanup released H/Mb/S with all owned PGs REMOVED, `-tpuhold` groups included (`results/switch/*/cleanup-*.json`), tmpfs cleanup ok;
- the O0 attempt-1 manual cleanup is recorded above;
- RCM Idle; last state line `switch-campaign partial-c762edb-ov released S switch-r3-leftover-S`;
- worker pods: the pod-local XLA caches gained the 0/0 entries and `/root/.cache/omp_exec_bank/` holds the provisioned banks (warm-state assets, kept like the cache);
- local `/dev/shm/hetero-scratch` removed after archiving its probes into `OverlapSwitch/results/overlap/`.

**Decisions for the user (after O0–O3)**

1. Adopt the overlapped costs as the study's C: full S→H **27.5 s**, partial S→Mb **18.8 s** effective.
   - Prerequisites: pre-spawn at phase-1 start, a bank provisioned per target shape on its hosts, and the cache at 0/0.
   - The bank is keyed to the source revision, so every code change needs one bank-writing restore deploy per target shape [INF: a few minutes each, as a warm restore deploy].
   - Should `overlap/prespawn-bank` be merged into `research/bf16-kv` as the default switch path?
2. The remaining 24 s is mostly device-bound. The next levers would be weight restore (4.9 s), bank load (3.4 s for 187 entries per rank, loaded serially), init_device (4.1 s) and gateway up (3.1 s). Pursue them, or stop here? At C ≈ 27 s, the S-tail term O = 32.6 s is now the larger switching overhead.
3. Two reviewer should-fix items are not fixed, because they cannot arise in this campaign path: the TPU-hold groups under scoped `drain`/`release --instances`, and the late-spawn abort. Fix them before any non-campaign use.

### U0 — local (done)

**(1) Upper-half abort: cause hypothesis.** Evidence ✔, from `~/hetero-wt/CompileExecutionSmoke/results/smoke/tpu-{mechanism,load-options}-full-job.log.gz` and the matching `.json`:
- Both upper-half runs (`2402ee2`, then `1df3b8e` with the original compile options passed at load) died at the same native CHECK: `F tpu_topology.cc:66 Check failed: ValidateIndexForBounds(index, bounds)`. The stack is `PjRtCApiClient::LoadSerializedExecutable → pjrt::PJRT_Executable_DeserializeAndLoad → xla::TpuClient::LoadSerializedExecutable → LoadInternal → GetAddressableDevices → tpu::TpuTopology::IdForLogicalDevice → Linearize`. That is the libtpu load of a *serialized* executable, before any execution.
- Only the two target participants aborted: ranks 1 and 3 (chips x ∈ {2, 3}, ids 2,3,6,7 and 10,11,14,15; target ids `[2,3,6,7,10,11,14,15]`). Ranks 0 and 2 do not hold target chips; they returned `participates: false` and their reports.
- The same path, client and options loaded and ran the lower half (ids `[0,1,4,5,8,9,12,13]`, x ∈ {0, 1}) correctly, twice (`tpu-origin-half`, `tpu-origin-long`: load 0.95–1.04 s, max error ≤ 3.9e-6).
- Hypothesis [INF; narrowed by U1d, see below]: when libtpu loads a serialized executable whose devices are a strict sub-box of the runtime topology, `GetAddressableDevices` linearizes the chips' absolute coordinates against the bounds of the executable's own device box (2×4 for a TP8 half). The origin half's coordinates happen to lie inside [0,2)×[0,4); the upper half's x = 2, 3 do not, and the CHECK aborts the process. This is an offset bug or unsupported case on the deserialize path, not a property of the upper chips. Which chips abort, and that only loading aborts, both fit; the libtpu source is not available to confirm it.
- Consequences [INF]: (i) U1 uses direct jit on the live client (no serialization). Whether the live compile path shares `LoadInternal` is unknown; U1 decides. (ii) The JAX persistent cache and the O-series executable bank both load through deserialize-and-load. In a unified runtime they would abort for every non-origin submesh. *Superseded by U1d: on the y-halves both halves, including the non-origin one, load live-compiled executables correctly; the abort is specific to the smoke's x-upper compile-only executable.* Probe **U1d** (below) tests this directly with a live-compiled executable on each half. (iii) Today's engines are unaffected: each instance owns a sub-slice runtime whose chips start at the origin.

**(2) Packing ✔.** `hetero/layout.py` (`resolve_layout`: first-fit, most-square host rectangles) on the 8-host inventory (host grid 2×4, 2×2 chips per host). Host cells are (x, y). Block A is host rows y ∈ {0, 1} (= H `large_a`); block B is rows y ∈ {2, 3} (= H `large_b`).

| Layout | Instance | TP | Host cells | Chips (x, y) | Block |
|---|---|---|---|---|---|
| H | large_a | 16 | (0,0) (1,0) (0,1) (1,1) | 0–3, 0–3 | A |
| H | large_b | 16 | (0,2) (1,2) (0,3) (1,3) | 0–3, 4–7 | B |
| S | small_a / small_b | 8 | (0,0)(0,1) / (1,0)(1,1) | 0–1 / 2–3, 0–3 | A lower / A upper |
| S | small_c / small_d | 8 | (0,2)(0,3) / (1,2)(1,3) | 0–1 / 2–3, 4–7 | B lower / B upper |
| M | small_a, small_b; large | 8, 8; 16 | as S small_a/b; as H large_b | | A, A; B |
| Mb | small_a, small_b; large_b | 8, 8; 16 | as M | | A, A; B |

No instance spans two blocks. Every TP8 is an x-half of a block. In the block runtime's rank order (row-major, x fastest), the lower half (small_a) is ranks {0, 2} and the upper half (small_b) is ranks {1, 3}. These are the halves the harness uses, and the smoke's coordinate split. Placement uses host-grid positions, not IPs, so it survives the 2026-10-09 Ray restart (U1 re-runs the inventory).

**(3) Harness ✔.** `research/unified-runtime` @ `b6e8cf1` (`~/hetero-wt/UnifiedRuntime`), branched from `CompileExecutionSmoke` `1b8149e`. It was not rebased onto `OverlapSwitch` `bbbeb4f`: the harness uses only H's first instance, and `bbbeb4f`'s `layout.py` adds only the cache-threshold env and the Mb entry. New mode of `scripts/compile_execution_smoke.py`: `--experiment u1|u2|u1d`, plus `--unified '<json overrides>'`. The logic is in `hetero/unified_runtime.py`.
- Ordering contract: all device work in a process runs on one command thread, in driver-submitted order. The two processes of a half get identical command sequences, and 16-chip work goes to all four, so multi-process programs launch in the same order everywhere and the halves share no barrier. The driver keeps each half's queue 3 batches deep (≈ 0.1 s per batch), and the halves are fed independently. Compiles run on separate threads and never execute.
- Model program: a TP-sharded MLP stack, unrolled, with bf16 weights and f32 accumulation. Per layer: up projection column-sharded, down projection row-sharded with an all-reduce, residual scale 0.25 (contractive, so TP8 vs TP16 and device vs host differ by summation order only). Weights are an integer hash of the global index, identical in both halves and on the host. An exact position-weighted uint32 checksum of the bf16 bits is sharding-independent, so any moved or lost element changes it.
- Switch: both halves' local weight shards are wrapped zero-copy into one 16-device "two replicas" array (mesh (8, 2), row k = (lower[k], upper[k])), then `jax.device_put` to the TP16 sharding, then the first target step. In JAX 0.10.2 a same-device-set `device_put` of a non-addressable array is `jit(_device_put_reshard, out_shardings=…)` (`jax/_src/dispatch.py:327-358, 462-466`). The harness AOT-compiles exactly that program in the background, which warms the cache. On the CPU fleet the switch then runs 0 backend compiles; without the warm-up it runs 2 per rank. Target order `local` puts TP16 shard 2k on lower[k] and 2k+1 on upper[k], so every TP16 shard is a slice of the TP8 shard already on that chip. `natural` (device-id order) and `aot` (a precompiled identity reshard) are alternative arms.
- Tests: `tests/test_unified_runtime.py`, 13 pass on jax 0.10.2 (the worker version; `~/hetero-wt/FastGraph/.venv-helper`) and on 0.11.2 (~37 s). The fleet tests run the real U1/U2/U1d protocols in one `jax.distributed` runtime of 4 local processes × 4 CPU devices (gloo). Each half's programs are launched only by its own two processes, with unequal step counts per half. They check: stall blocking only its own half; both switch directions; checksum equality; bitwise-equal outputs after the round trip; verdict rejection of a moved element, a changed reverse output, an unblocked stall and a slowed active half. The existing overlap mode is unchanged: CPU smoke passes on 0.11.2. On 0.10.2 it fails before and after this change, because 0.10.2 has no CPU compile-only compiler factory.

### U1 — spec (submesh independence; + probe U1d)

- Hypothesis [INF]: direct jit on the upper half works, because the U0 abort is on the deserialize path. The halves step independently: a_both/a0 ≥ 0.9 per half. A stalled half blocks only its own processes. Unknown: whether libtpu's launch barrier (`--xla_tpu_use_enhanced_launch_barrier=true` in the worker argv) involves non-participating hosts or times out during a 30 s stall.
- Code: `research/unified-runtime` @ `3b445ea`, frozen (no hot-patching). It is `b6e8cf1` plus the pre-run review fixes (opus-5-5 review): native-compile overlap windows with evidence thresholds, the TP16 bucket set, arm (e), cross-half tolerance (bitwise equality recorded as data only), driver-clock offsets with per-rank monotonic switch splits, a failure on any compile inside a prepared switch, and a lock-protected report snapshot. 14 CPU tests pass. Block: H `large_a` (4 hosts, one 16-chip runtime; env from `instance.process_env`, compilation cache disabled). Halves: lower = ranks {0, 2}, upper = ranks {1, 3}.
- Program: TP8 MLP, 8 layers, d 1024, ff 4096, batch 32 (host reference computed). (c) program: 384 layers, d 256, ff 512. Batches ≈ 0.1 s, queue depth 3; control 10 s; stall 30 s.
- Arms, in order, in one job: (a0) each half alone 10 s; (a) both 20 s; (b1) one half idle 30 s while the other steps (both directions); (c) each half compiles the (c) program in a background thread while the other half steps, then both adopt and run it; (e) U2 calibration: 10 s baseline, then all 4 processes compile the U2 TP16 target set in a background thread (64 layers, d 4096, ff 12288; batch buckets 8–256, plus the warm `device_put` reshard programs) while both halves keep stepping the small program, then 10 s post; (b2) one half stalled *inside* a step: its leader sleeps 30 s while its partner has launched the step and waits in the collective, the other half keeps stepping (both directions). Output checks after each arm.
- Pass: in every check round, the two ranks of a half are bitwise equal; host-reference error is ≤ 0.05 (≤ 0.1 for the 384-layer program); the two halves differ by ≤ the same tolerance (cross-half bitwise equality is recorded, not required, because each half is a separately compiled program); every background compile succeeds. In each (b2): the active half's rate is ≥ 0.8 × its (a) rate, its max step ≤ 1 s, and the stalled partner's step lasted ≥ 27 s (the stall really blocked in the collective). Recorded: per-half rates for every arm, a_both/a0, active-during-compile ratios, and (e) per-half rates inside each rank's native-compile intervals against its baseline. (e) is the dispatch-bound counterpart of the smoke's −19% and sets the U2 compile window; it is data only.
- Steps (one `flock /tmp/hetero-cluster.lock` session each, `KUBECONFIG=$HOME/.kube/hetero-agent.yaml`, RCM Idle and 32 TPU available first; disk floors / ≥ 10 GB, /dev/shm ≥ 20 GB):
  1. Inventory ✔ (Ray restarted 2026-10-09, session `session_2026-10-09_03-03-28`). Run from the worktree: `~/hetero-research`'s upload exceeds Ray's 100 MB request limit (HTTP 413; its untracked `data/` is 205 MB), so the first attempt trapped and was aborted with no job started. `.venv/bin/python rcm_exec run scripts/hetero_ctl.py inventory --head-pod tpu-ray-cluster-vllm-tpu-head-5mkgz --output /tmp/hetero-inventory.json --timeout 900` → SUCCESS in 44.8 s; extracted to `/tmp/hetero-inventory-only.json` (copy: `results/unified/inventory-2026-10-10.json`). Block A is now `.61, .56, .57, .63` (ranks 0–3). The packing table recomputes identically on it.
  2. From the worktree: `.venv/bin/python rcm_exec run scripts/compile_execution_smoke.py --experiment u1 --inventory /tmp/hetero-inventory-only.json --output results/unified/u1-3b445ea.json --timeout 2400`.
- Expected ≈ 8–12 min (runtime init ≈ 1 min; the (e) compile time is unknown). Stop: a native abort or hang on the upper half ends U1; record the libtpu log and do not run U2. At most 2 fix attempts per blocker.
- Evidence: `results/unified/` (result JSON, full job log, RCM status before/after, gzipped). Cleanup receipt: actors DEAD, placement group REMOVED, 32 TPU available.
- **U1d** (organizer addition; runs last, after U2, because an abort ends its runtime): `--experiment u1d`. Each half serializes its live-compiled TP8 executable with `jax.experimental.serialize_executable` and loads it back on its own 8 devices, which is the persistent-cache/bank path. Lower half first (control), then upper. Prediction from U0 [INF]: the lower half loads and matches bitwise, and the upper half aborts at `ValidateIndexForBounds`.

**U1 attempt 1 (`3b445ea`, job `raysubmit_WjLyv7qFvVrava57`) — blocker: the TP8 program does not compile on the lower x-half.** The job ran 11.6 s. Runtime init was fine (4 hosts, 16 devices, jax 0.10.2 / libtpu 0.0.43). Ranks 0 and 2 then failed the *direct-jit compile* of the small TP8 program on x-lower (ids `[0,1,4,5,8,9,12,13]`) with a Python exception, not an abort: `INTERNAL: RET_CHECK failure (platforms/xla/service/jellyfish/lowering/net_util.cc:2304) device_id < kMaxDeviceCount Unexpected device_id 8 [0:4] in replica group (0,1,2,3,4,5,6,7) :: target has 8 device_id`, raised while lowering the SPMD all-reduce. The driver stopped there. The upper ranks were still in setup, so their outcome was not observed. Cleanup ✔: 4 actors DEAD, PG REMOVED, 32 TPU free. Evidence: `results/unified/u1-3b445ea.json{,.gz}`, `u1-3b445ea-driver.log`.

**Cause, reproduced deviceless ✔.** libtpu 0.0.43 runs its TPU compiler on a v5e 4x4 *compile-only* topology with no TPU attached (`scripts/unified_runtime_deviceless_probe.py`, venv jax/jaxlib 0.10.2 + libtpu 0.0.43; `results/unified/deviceless-probe-libtpu0.0.43.json`). It gives the identical RET_CHECK for every program with ICI collectives whose device box contains the origin chip and whose ids are not dense 0..N−1:
- fails: x-lower TP8 (every device order tried: id, column, ring, reversed, 2-D (4,2) mesh); host-0 2×2 TP4 `[0,1,4,5]`;
- compiles: x-upper TP8, y-lower TP8 (`0..7`), y-upper TP8 (`8..15`), the pairs `[0,1]` and `[4,5]`, every TP16 order, and the 2×TP8 ↔ TP16 reshard programs;
- the same x-upper program without collectives also compiles.

So libtpu 0.0.43 has two complementary sub-box defects: compiling collectives fails on an origin-anchored, non-dense box (x-lower), and deserialize-and-load aborts on a non-origin box (x-upper, U0). In the 4×4 block, the S-layout TP8 boxes (host columns) cannot both run TP programs in one runtime. Host rows (y-halves) can both compile.

**Fix attempt 1 (of 2), decision (organizer):** halve the block along y (host rows) for U1/U2: lower = ids 0–7 (ranks 0, 1), upper = ids 8–15 (ranks 2, 3). TP16 local order `[0,8,1,9,…]`. A row half is still an 8-chip ICI box (4×2 instead of 2×4). A unified-runtime layout would pack its TP8s as host rows instead of `resolve_layout`'s columns; this is a layout change, not an engine change. Code `research/unified-runtime` @ `945a152` = `3b445ea` + `half_split` (default `y`; `x` reproduces attempt 1) + the deviceless probe; 14 CPU tests pass. Run otherwise as specified: `--output results/unified/u1-945a152.json --timeout 2400`. The deviceless compile times give U2's background window: TP16 bucket set 8–256 ≈ 9.9 s plus warm reshards 0.5 s on this host [INF for TPU hosts]; (c) program ≈ 3 s. If attempt 2 fails on a new blocker, record it and stop U1.

### U1 — results (attempt 2, `945a152`, y-halves; job `raysubmit_ZHYwsXEkwSnEQb8N`) — **PASSED**

Artifacts: `~/hetero-wt/UnifiedRuntime/results/unified/u1-945a152.json{,.gz}` and the driver log. The job ran 246.6 s. One 16-chip runtime on block A: rank 0 r3qx and rank 1 64b6 hold the lower half, ids 0–7; rank 2 kk2b and rank 3 v48p hold the upper half, ids 8–15. Every program is direct jit with the persistent cache off, each half dispatched only by its own 2 processes. Verdict `passed`, no failures ✔.

Per-half rates (small TP8 program, 8 layers; step ≈ 0.46 ms, dispatch-bound; steps/s):

| arm | lower | upper | ratio vs (a) lower / upper |
|---|---|---|---|
| (a0) alone | 2,070 | 2,009 | (a)/(a0) 1.05 / 1.08 |
| (a) both concurrently | 2,179 | 2,162 | — |
| (b1) other half idle 30 s | 2,202 | 2,119 | 1.01 / 0.98 |
| (c) other half compiles the 384-layer program (3.3 s) | 2,151 | 2,132 | 0.99 / 0.99 |
| (b2) other half stalled 30 s inside a step | 2,105 | 2,145 | 0.97 / 0.99 |
| (e) all 4 processes compile the U2 TP16 set (native-compile intervals) | 1,682 | 1,637 | −22.2% / −23.3% vs its 10 s baseline |

- (b2) ✔: the stalled half's waiting partner sat 29.9997 s (upper) and 29.9994 s (lower) in its blocked collective, then both halves finished all 5 steps with correct output. During each stall the other half's max step was 4.5 / 4.3 ms. No libtpu launch-barrier timeout and no cross-half blocking.
- Correctness ✔: in all 8 check rounds the ranks agree within each half. Host-reference error is 0.0014 for the small program and 0.032 for the 384-layer (c) program (bound 0.1). The two halves were also bitwise equal in every round (data only).
- (e) ✔ (U2 calibration, data): each rank compiled the TP16 set (64 layers × 4096 × 12288; buckets 8–256 plus 4 warm reshard programs) in 12.0–12.3 s; the native intervals total 11.5–11.8 s (10 per rank). Inside them each half ran ≈ 19k fully contained steps. p50 step barely moved (0.454 → 0.463 ms), but the mean step rose 0.46 → 0.59 ms. That is the same order as CompileExecutionSmoke's −19% (rank 0 only, 0.34 ms steps, a 32 s compile-only call), and here every serving process compiles.
- Not run: the x-halves (the S-layout columns), which are blocked by the libtpu compile defect (attempt 1). Cleanup ✔: actors DEAD, PG REMOVED, 32 TPU free.

### U2 — spec (in-runtime switch 2×TP8 ↔ TP16, y-halves)

- Hypothesis [INF]:
  - TP8 model step ≈ 2–4 ms: 1.61 GB/chip at ~0.8 TB/s, plus 64 all-reduces over 8 chips. The TP16 step is about half that.
  - The background TP16 compile takes ≈ 12 s (U1 (e)). Serving slows less than U1 (e)'s −22%, because model steps are device-bound.
  - Forward reshard (local order: every TP16 shard is a slice of the TP8 shard already on its chip) ≈ an HBM copy of 0.8 GB/chip, i.e. milliseconds.
  - Reverse reshard is a pairwise ICI exchange of 0.8 GB/chip (tens of ms).
  - Switch, last half step → first target step, well under 1 s in both directions, with 0 compiles on the critical path.
- Code `945a152`, frozen. Model: 64 layers, d 4096, ff 12288, batch 32; bf16 weights = 1.61 GB/chip at TP8, 0.81 GB/chip at TP16. Each half initializes its own weights on device. Background job on all 4 processes: the TP16 forward for buckets 8–256, plus the 4 warm `device_put` reshard programs. The switch runs `jax.device_put` on the zero-copy 2-replica view (forward) or on the TP16 arrays (reverse); the reverse reuses the TP8 executables still loaded since setup.
- Phases: baseline 10 s → background compile (both halves serving) → post 10 s → switch forward (queued behind live batches) → TP16 serves 10 s → switch reverse → 2×TP8 serve 10 s. Output checks after each step.
- Arms:
  - **run 1** (primary): target order `local`, warm `device_put`;
  - **run 2** (diagnostic): `--unified '{"target_order":"natural"}'`. The TP16 shards then sit in device-id order, so the reshard moves data over ICI, and `device_put` compiles its reorder/reshard program at switch time.
- Pass, per `verdict_u2`:
  - weights ≥ 1 GB/chip at TP8;
  - each half internally bitwise-equal at every check; halves within 0.05 of each other; TP16 first output within 0.05 of the TP8 output;
  - exact position-weighted checksums equal at setup, after forward and after reverse;
  - 2×TP8 outputs after the reverse bitwise equal to setup;
  - run 1: 0 backend compiles inside either switch.
  - Overlap evidence: per rank, ≥ 5 s of native compile and ≥ 20 fully contained serving steps; otherwise `inconclusive`.
- Recorded:
  - serving rate change, call window and native window, vs the −19% / U1 (e) −22%;
  - compile time per bucket and in total;
  - switch time and stop / reshard / first-execution split, on the driver-corrected clock and per-rank monotonic;
  - TP16 step rate; HBM in use after setup, after forward and after reverse.
- Steps (lock, status, floors as in U1): from the worktree, `.venv/bin/python rcm_exec run scripts/compile_execution_smoke.py --experiment u2 --inventory /tmp/hetero-inventory-only.json --output results/unified/u2-945a152-local.json --timeout 2400`; then run 2 with `--output results/unified/u2-945a152-natural.json`.
- Expected ≈ 5 min each. Stop: a failure is recorded with its logs; at most 2 fix attempts.

### U2 — results (`945a152`, y-halves) — **PASSED** (run 1 primary, run 2 diagnostic)

Artifacts: `~/hetero-wt/UnifiedRuntime/results/unified/u2-945a152-{local,natural}.json{,.gz}` and the driver logs. The jobs are `raysubmit_VXETgsN2qweKNDXr` (run 1, 70.1 s) and run 2 (71.4 s). Both have verdict `passed`, no failures, and no overlap-evidence problems ✔.

Model: 64 layers, d 4096, ff 12288, bf16; **1.61 GB/chip at TP8** ✔ (`1,610,612,736` B). HBM in use per chip: 1.62–1.63 GB after setup, 2.44–2.45 GB after the forward switch (TP8 + TP16 copies), 1.64 GB after the reverse.

**Serving and compile** (run 1; per half, mean of its two ranks):

| | lower | upper |
|---|---|---|
| TP8 baseline | 269.0 steps/s (3.72 ms; p50 3.71 ms) | 268.4 (3.73 ms) |
| inside the native-compile intervals (11.2–11.3 s per rank, 10 intervals, ≈ 2,630 contained steps per rank) | 234.0 (**−13.0%**; p50 3.77 ms, mean 4.27 ms) | 236.8 (**−11.8%**) |
| whole compile-thread call (11.8–11.9 s, trace+lower included) | −16.0% | −14.5% |
| post-compile control | 268.2 | 267.0 |
| after the round trip (2×TP8 again) | 268.9 | 268.4 |

- The slowdown is smaller than U1 (e) (−22%, dispatch-bound 0.46 ms steps) and CompileExecutionSmoke (−19%) ✔. It shows as rare long steps (the mean moves, the p50 barely does).
- Background compile on every rank ✔: 11.8–11.9 s per call. TP16 forward buckets 8/16/32/64/128/256 took 1.7/1.0/1.1/1.7/2.2/3.4–3.5 s, and the 4 warm reshard programs 0.12–0.17 s each.
- TP16 after the switch: 342 steps/s (2.92 ms per step) vs 269 for each TP8.

**Switch** (driver-corrected cross-host clock, offset uncertainty ≈ RTT/2 ≈ 0.5 ms; per-rank monotonic in brackets) ✔:

| | switch (last source step → first target step on every rank) | stop | reshard | first execution | compiles on the switch |
|---|---|---|---|---|---|
| run 1 forward 2×TP8 → TP16 (`local` order) | **12.0 ms** [11.96–14.60] | 0.08 | 6.1 [6.1–8.75] | 5.8 [5.7–5.8] | **0** |
| run 1 reverse TP16 → 2×TP8 | **48.0 ms** [47.6–48.0] | 0.08 | 43.2 [42.7–43.2] | 4.7 [4.6–4.9] | **0** |
| run 2 forward (`natural` order) | 285 ms [284–316] | 0.09 | 278 (of which compile ≤ 246) | 7.0 | 2 per rank |
| run 2 reverse (`natural`) | 343 ms | 0.09 | 339 (of which compile ≤ 287) | 4.3 | 2 per rank |

- In run 1 the halves stopped 2.6 ms apart (forward), so the earlier half was down 14.6 ms. In the local order the forward reshard is an on-chip slice copy (6 ms). The reverse rebuilds each TP8 replica pairwise, chip k ↔ chip k+8, 0.8 GB per chip in ≈ 39–43 ms.
- Run 2 shows what the background warm-up buys. With a non-local order, `device_put` compiles its reorder/reshard programs on the critical path (≈ 0.25–0.29 s), and moving the data adds ≈ 30–50 ms.
- Correctness ✔ (both runs):
  - the exact position-weighted checksums are identical at setup, after forward and after reverse (`(2382864628, 1995569434)` on all 4 ranks);
  - each half's ranks agree bitwise at every check, and the two halves also agree bitwise (data);
  - the first TP16 output is within 0.0067 (run 2: 0.0057) of the TP8 output;
  - the 2×TP8 outputs after the reverse are bitwise equal to setup on all ranks.
- The reverse reuses the TP8 executables loaded at setup, so it has no compile by design (limitation recorded in the result). Cleanup ✔ (both): 4 actors DEAD, PG REMOVED, 32 TPU free.

**U1d amendment (before running):** the default `half_split` is now `y`, so the probe tests the y-halves. Prediction from U0 [INF]: the lower half (ids 0–7, origin box) loads and matches bitwise; the upper half (ids 8–15, non-origin box) aborts at `ValidateIndexForBounds`. Command: `--experiment u1d --output results/unified/u1d-945a152.json --timeout 1200`. An abort ends the job; the lower probe result and pre-probe reports are already saved by then.

### U1d — results (`945a152`, y-halves, job `raysubmit_KXBTsGxaAXZeX4JN`, 11.2 s) — **PASSED; the U0 load prediction is refuted for y-halves**

`results/unified/u1d-945a152.json{,.gz}`. Each half serialized its live-compiled TP8 executable (`jax.experimental.serialize_executable`; lower 0.78 MB, upper 1.25 MB) and deserialize-and-loaded it on its own 8 devices. Load took 4.1–4.2 ms on all 4 ranks, and the outputs are bitwise equal to the directly compiled program on all 4 ✔. The **upper y-half (ids 8–15, a non-origin box) loads fine**, so U0's general form ("any non-origin sub-box aborts on load") is wrong.

The CompileExecutionSmoke abort stays specific to its case: an **x**-upper executable (ids 2,3,6,7,…) produced by the *compile-only* client. Which of the two factors matters is not isolated [INF], and it is moot for this design: the x-halves are already blocked by the compile defect. For the y-halves both the direct-jit path and the deserialize path (persistent cache, executable bank) work. Cleanup ✔: actors DEAD, PG REMOVED, 32 TPU free.

### U3 — tpu-inference/vLLM assumption inventory and go/no-go (local) — done

Gate: U1 ✔ and U2 ✔ (both on host-row halves). The inventory comes from a read-only scout over the forks: `tpu-inference` @ `ac4497925`, whose `tpu_worker.py`, `tpu_runner.py` and `compilation_manager.py` match the installed `8d31f0866` by md5 (O1); vLLM; our glue in `~/hetero-wt/OverlapSwitch/hetero`. The organizer re-read the rows marked ✔; the rest are the scout's reading.

The engine model a unified runtime needs: two TP8 engines on disjoint halves (2 processes each) inside one 4-process runtime, plus a TP16 engine in the **same 4 processes** that loads/compiles in the background, then takes over the resharded weights.

| Assumption | Where | What breaks | Size |
|---|---|---|---|
| Engine devices = first N global devices | `tpu_inference/worker/tpu_worker.py:409-411` `self.devices = jax.devices()[:total_devices]` ✔ | the second TP8 (upper half) would pick devices 0–7, the other engine's | S: inject `devices=` (`TPUWorker.__init__`) |
| `device_indexes` path resolves only `jax.local_devices()` | `tpu_worker.py:381-406` | a cross-host half or TP16 cannot be named | M |
| Mesh built from `self.devices` | `tpu_runner.py:881-981`, `utils.py:297-322` | works for subsets; non-contiguous sets fall back to a logical-order mesh | S (verify) |
| Multi-host formed from per-instance libtpu env (`TPU_TOPOLOGY/HOST_BOUNDS/PROCESS_ADDRESSES/VISIBLE_CHIPS`), read once per OS process; tpu-inference never calls `jax.distributed.initialize` | `tpu_worker.py:296-371`; ours `hetero/layout.py:128-148`, `engine_actor.py:29-34` | one process cannot hold a TP8-instance env and a block env. Needs one block bootstrap (`jax.distributed.initialize`, as the harness does) with engines that skip init | L |
| Executor = one Ray worker actor per TPU host, PG over all hosts, rank forced to 0 when PP = 1 | `executors/ray_distributed_executor.py:170-329`, `:436-437` ✔ | there is no place for a TP8 engine on 2 of 4 hosts plus a TP16 engine in the same actors | L: an in-process multi-engine host |
| Two engines in one process: vLLM `set_current_vllm_config` scoping, `parallel_state` init, tpu-inference `_PP` global | `tpu_worker.py:426` ✔, `:413-443`; `distributed/jax_parallel_state.py:62-63` ✔ | process-wide singletons; a background TP16 load/compile can read or overwrite the TP8 engine's state [INF on the vLLM internals] | M–L |
| `jax.clear_caches()` after weight load | `tpu_runner.py:1248` ✔ | wipes the serving TP8's jit caches when the TP16 loads, so the TP8 recompiles at serve time | S |
| Process-global compile-cache config and the recompilation guard | `compilation_manager.py:66-73` ✔ | shared by both engines; the TP16's background compile can trip the TP8's ForbidCompile guard [INF] | M |
| Our module-global executable-bank patches; bank keyed by `process_index`/`process_count` | `hetero/exec_bank.py:17-26, 130-189, 404-429, 544-593` | patches swap callables in shared runner modules; rank dirs and unanimity count 4 processes, not the engine's 2 | M–L |
| `process_allgather` votes over all processes; device-evidence asserts `len(jax.devices()) == tp_size` | `hetero/tpu_worker.py:421, 454` ✔; `:565-640` | an allgather called by one engine deadlocks (the other half never calls it); the asserts fail in a 16-device runtime | M |
| HBM sizing: `limit × util − bytes_in_use` over the engine's devices | `tpu_worker.py:506-564` ✔, `utils.py:123-152` | `bytes_in_use` includes the other engine's weights and KV, so KV blocks shrink or the check raises | M: explicit per-engine budgets |
| Topology order id from global anchors; transfer ports keyed by rank | `distributed/utils.py:60-66, 100-140` | rank comes out 0..3 instead of 0..1; ports collide with two engines per process | S |
| Executable deserialization (bank, persistent cache) | `exec_bank.py:404-420`; `compile_overlap.py:157` | **works on both host-row halves (U1d)**; it aborted only for the smoke's x-upper compile-only executable | none on y-halves |

- **Failure isolation** ✔ (`jax/_src/distributed.py` 0.10.2): process 0 hosts the coordination service. `heartbeat_timeout_seconds=100` and `shutdown_timeout_seconds=300` (`:73-74`), and `jax_enable_recoverability` defaults to False (`:38-45`). A dead or OOM-killed process is detected after ≤ 100 s, and without recoverability the failure propagates to every task [INF: the propagation is in the C++ service]. If process 0 dies, the service dies with it. The 4-host block becomes **one failure domain**: a crash in one TP8 engine takes down the other TP8. Today each instance is its own libtpu runtime and fails alone. U1 (b2) showed that a *stalled* half does not block the other, but a *dead* process is a different case (not tested).
- **Layout constraint** (U1 attempt 1): libtpu 0.0.43 cannot compile ICI collectives on the origin-anchored x-half. Unified-runtime TP8s must therefore be host rows (`[(0,0),(1,0)]`, `[(0,1),(1,1)]`), not `resolve_layout`'s columns. That is a layout change, and it moves S/M/Mb instance placement.
- **Integration size** [INF]: ≈ 1.5–3k LOC over ~8–12 files.
  - Ours: block bootstrap plus an in-process multi-engine host replacing the per-host Ray executor (0.6–1k LOC); the votes, evidence and exec-bank identity (0.4–0.6k); the layout env and row packing (0.1k).
  - tpu-inference fork: device selection, HBM budget, `clear_caches`/guards, `_PP`/topology id (≈ 0.2k).
  - vLLM: per-engine config and `parallel_state` isolation (unknown size).
  - A whole new class of failure modes (shared process, shared HBM, shared failure domain) would need its own qualification campaign.
- **Expected payoff** [INF from the O3 segments]:
  - U2 replaces O3's device-bound engine init, ≈ 14.5 s, with ≈ 0.05 s of in-HBM reshard and no compile: init_device 4.1, weight restore 4.9, bank load 3.4, warm-up 1.5 and handoff 0.6 all leave the critical path.
  - What remains is drain/release 2.2 + gateway 3.2 + endpoint 1.2 + KV allocation (`initialize_from_config` 2.6) → full S→H **C ≈ 9–10 s instead of 27.5 s**.
  - The cost is a −12% to −13% source slowdown during the background preparation.
  - In the O3 region maps, that C change is worth ≈ +0.3 pp median job gain: O0→O3, 62 s of C bought +1.2 pp. That is less than the effect of the O = 32.6 s S-tail term.

**Recommendation: NO-GO for vLLM integration now** (the mechanism itself is GO; it passed U1/U2).
- It saves about 17 s per switch on a 27.5 s C, which moves the bulk job's region map by tenths of a percent.
- It costs an L-sized integration that touches vLLM singletons and the Ray executor model, a host-row layout change forced by a libtpu defect, and a block-wide failure domain.
- Revisit if the workload needs many switches per job or sub-second switching (online re-sharding, where 12–48 ms matters), or if the remaining 24 s critical path is otherwise exhausted.
- Cheaper levers inside the current per-instance design first: weight restore 4.9 s and bank load 3.4 s (O3 user decision 2).

**Cleanup receipt (U0–U3)** ✔:
- 6 Ray jobs ran in session `session_2026-10-09_03-03-28`: inventory, U1 ×2, U2 ×2 and U1d. All are terminal, and every unified job's EXIT receipt shows its 4 actors DEAD, its PG REMOVED and 32 TPU available.
- After the last job: RCM Idle, Ray `0.0/32.0 TPU`, `0.0/904.0 CPU` in use.
- The first inventory attempt trapped before submitting (HTTP 413, upload of `~/hetero-research` > 100 MB because of its untracked 205 MB `data/`) and was cleared with `rcm_exec resume --abort`. Receipt: `results/unified/inventory-attempt1-413-trap.log.gz`.
- Worker pods untouched: no `kubectl exec`; every unified job ran with the JAX compilation cache disabled, so `/root/.cache/vllm/xla_cache` and `/root/.cache/omp_exec_bank` were not written.
- `/tmp/hetero-inventory{,-only}.json` now hold the fresh 2026-10-10 inventory. It is valid for the current Ray session and was left in place for later runs.
- Local `/dev/shm/hetero-scratch` (1.3 GB: probe scripts, a libtpu 0.0.43 CPU venv, logs) removed. Every result it held is committed under `~/hetero-wt/UnifiedRuntime/results/unified/` (`research/unified-runtime` @ `7f0a118` and its receipt commit).
- The dashboard port-forward started for this work has been stopped.

**Decisions for the user (after U0–U3)**
1. Accept the no-go on vLLM integration of the unified runtime (keep the O3 per-instance design), or fund it (≈ 1.5–3k LOC, host-row TP8 layout, one failure domain per block)?
2. If it is ever pursued: accept host-row TP8 packing given the libtpu 0.0.43 x-half compile defect, and report that defect upstream (deviceless repro: `scripts/unified_runtime_deviceless_probe.py`)?
3. Optional follow-ups, none run: (a) a dead-process (not stalled) isolation test in a unified block, measuring how long the other half survives with `jax_enable_recoverability` off/on; (b) isolate whether the smoke's x-upper load abort comes from the x offset or from the compile-only client.

### F0 — local (done)

Worktree `~/hetero-wt/FirstSwitch`, branch `research/first-switch`, from `bbbeb4f`; `.venv` and `data` are symlinked to `Bf16Kv` as in `OverlapSwitch`. Code **`11bd0fb`** (organizer).

(a) **Inline provisioning in a normal deploy** ✔:
- The snapshot already existed as an engine setting: `weight_snapshot="write"` snapshots every worker right after the normal `load_model` and the precision check (`hetero/tpu_worker.py` `load_model`, clock `snapshot_write`), before KV allocation, compile and the endpoint. That is "right after `load_model`". It is safe because the runner state is assigned once at load (`tpu_runner.py:1194`, tpu_inference @ `8d31f0866`), weights are never donated (only KV and inputs: `decode_loop.py:117`, `kv_cache_manager.py:1362`, `tpu_runner.py:535,571`), and nothing writes `runner.state` later. So the snapshot equals the served state.
- The bank write needed no code either: `exec_bank="write"` captures during `initialize_from_config` and writes after `compile_or_warm_up_model` on any loader path.
- New setting: `weight_snapshot_root` (default `/dev/shm/omp_weights`). Together with `exec_bank_dir` it gives a per-run namespace.
- **Bank identity vs loader path.** `exec_bank.engine_identity` reads only `tp_size` and `runtime_env` from the instance payload (`exec_bank.py:126-189`), so the loader mode, the roots, the nonce, the snapshot generation and the pre-spawn gate cannot enter the digest. The other inputs are taken after the load: `model_config.dtype`, which `get_flax_model` restores to the original torch dtype on both paths (`model_loader.py:322-354`; the restore path differs only in `_get_nnx_model`); `num_blocks`, 283 for both the O3 H normal deploy and the restore deploy ✔ (`prep-H-normal-deploy`, `H-restore-deploy`); the mesh; KV shapes; versions. No fix was needed. CPU proof: a digest test across normal+write and restore+read payloads. In fresh processes, a bank written by one process is used by a second process whose weights come from the first process's snapshot: unanimous `use`, 0 trace/lower/compile events for banked functions, 0 misses, 0 rejections, outputs equal (`tests/test_exec_bank.py::test_bank_from_a_checkpoint_load_serves_snapshot_restored_weights`). F1 confirms this on the TPUs.
- **Manifest and generation.** The inline snapshot is a v2 manifest whose generation is the actor's per-instance `weight_snapshot_generation`, shared by every rank, so the restore's generation vote (`tpu_worker.py:449-457`) passes. `prepare` accepts it, and restore is exact under a run-scope root (`tests/test_first_switch.py::test_scope_snapshot_passes_manifest_and_restore_checks`).
- **Run scopes** (`hetero/run_scope.py`): snapshot `/dev/shm/omp_weights_runs/<scope>`, bank `/root/.cache/omp_exec_bank_runs/<scope>`. `scripts/snapshot_validate.py {files,cleanup,memory} --run-scope <scope>` checks or removes exactly these two directories on every TPU host, never the default snapshot root, the warm banks or the XLA cache (tests cover a neighbouring scope and both default roots).
- `switch_campaign.sh` gains the modes `first-gate` (F1) and `first` (F2); their dry-runs are listed in the specs below. `switch.provisioned_flags` checks per worker that a snapshot was published with one generation per instance and a bank was written (entries > 0, no write error, no skipped entry, one identity digest per instance). It is in every `run-switch` report.
- (c) **Segments.** `deploy_anatomy` now ends engine init at the bank write (`exec_bank_write` follows `compile_or_warm_up_model`). It adds `graph_build`, `weight_load` (`DefaultModelLoader.load_weights`), `snapshot_write`, `exec_bank_load` and `exec_bank_write`, and, for the normal loader only, `checkpoint_iter` (time in the checkpoint iterator's `next()`), `host_convert` (`jax_array_from_reshaped_torch`) and `shard_place` (`assign_and_shard_param`), patched in `weight_utils` and `unquantized` for this load only. `load_model_unattributed` = load − graph build − weight load. Semantics, from the F3a code walk: the safetensors tensors are lazy mmaps, so `checkpoint_iter` measures file opens (≈ 0.2 s [INF from LoadProfiler]), and the checkpoint bytes are read as page faults inside `host_convert`. `switch_summary.py` handles `first` campaigns: means over the cold reps, the warm rep beside them, start-up phase medians, and the amortized per-switch cost (C_first + (N − 1)·C_warm)/N for N = 1, 3, 10. `overlap_check.py` checks a cold rep's provisioning instead of restore and bank use; `--deploy` checks a bank-read restore deploy (F1).
- Tests ✔: `tests/test_first_switch.py` (11) and the new exec-bank case pass. Full suite: 350 passed, 3 skipped, 8 failed + 8 errors. The failures and errors are the pre-existing baseline set (missing `results/hetero/*protocol*.json` fixtures and `test_restore_graph`). `tests/test_restore_tail_worker.py` needs flax and was skipped, as before.
- Not done locally: a real-model run of the normal loader. No local venv has flax, torch and tpu_inference (F3a). F1 is the device proof.

### F1 — spec (inline snapshot + bank write gate, H)

- Hypothesis [INF]:
  - H normal deploy with `{"weight_snapshot":"write","fast_graph":true,"startup_profile":true,"exec_bank":"write"}` plus scope `first-gate-<commit>`: every worker publishes a ≈ 15.8 GB snapshot in ≈ 7 s (O3 explicit RPC: 7.2 s ✔) and writes ≈ 187 bank entries in ≈ 2.4 s.
  - Its compile/warm-up is cold. The XLA cache is gone (decision log, 2026-10-12), so this deploy also refills it.
  - The following H restore deploy (`FAST` + `exec_bank=read` from the same scope) is restored on 8/8 workers, with a unanimous `use` and 0 backend compiles in compile/warm-up. Its greedy outputs equal the reference taken from the normal deploy, with 0 misses and 0 rejections over the 2 × 32 validation generations.
- Code: `research/first-switch` @ `11bd0fb`, frozen. Run ids and OUT come from the HEAD at launch (`first-gate-11bd0fb`).
- Steps, from `~/hetero-wt/FirstSwitch` with `KUBECONFIG=$HOME/.kube/hetero-agent.yaml`, after checking that RCM is Idle and the lock is free:
  1. Inventory (the Ray session dates from 2026-10-09): `.venv/bin/python rcm_exec run scripts/hetero_ctl.py inventory --head-pod tpu-ray-cluster-vllm-tpu-head-5mkgz --output /tmp/hetero-inventory.json --timeout 900`, then extract `/tmp/hetero-inventory-only.json`.
  2. `HEAD_POD=tpu-ray-cluster-vllm-tpu-head-5mkgz flock /tmp/hetero-cluster.lock scripts/switch_campaign.sh first-gate`, run detached, with driver log `results/switch/f1-gate-driver-11bd0fb.log`.
  3. The command sequence: preflight → `memory H` → `deploy H H-first-normal` (scope settings, max_model_len 131,072) → `files H --run-scope` → `validate H prep-H-reference` → teardown → `deploy H H-first-restore` (`FAST` + `exec_bank=read` + scope) → `check --require-restored` → `validate --require-restored --reference prep-H-reference.json` → `check` again (serving bank counters) → teardown → `cleanup --run-scope` → EXIT cleanup (release H/Mb/S, remaining scopes, default tmpfs root).
- Pass conditions, all hard:
  - (i) `H-first-normal-deploy` succeeds; `provisioned_flags` is true on 8/8 workers; `H-first-scope-files` shows a v2 manifest of the expected size and a bank manifest on all 8 hosts.
  - (ii) The restore deploy is restored on 8/8, with a unanimous `use` on 8/8, entries loaded, and `backend_compile_and_load` < 0.05 s in compile/warm-up (`overlap_check.py --deploy`).
  - (iii) The restored greedy outputs equal the normal-loader reference (32 prompts × 2 engines), and replicas are equal.
  - (iv) The post-validation check shows 0 misses and 0 rejections on every banked function.
- Recorded: `snapshot_write` and `exec_bank_write` per worker (median, max); the cold normal deploy's phases, including the new `load_model` sub-phases; the restore deploy's phases. Offline: the reference's outputs compared with O3's `prep-H-reference.json.gz` outputs (code revision differs, so the `engine_identity` key differs; data only).
- Stop: a failed condition is a blocker. At most 2 fix attempts (new commit, new run base); otherwise stop and do not run F2. Expected ≈ 30–40 min (a cold H compile ≈ 10–15 min [INF]).
- Cleanup: the scope is removed by the campaign (receipt `H-first-scope-cleanup.json`); release receipts for H/Mb/S. The default tmpfs root is emptied, as in every campaign.

### F2 — spec (cold first switch S→H, 3 reps + 1 warm; timed)

- Hypothesis [INF]. C_first ≈ 205–235 s. Gap ≈ drain + gateway + release + handoff 3.0 ✔(O3) + engine init + endpoint/gateway 4.2 ✔(O3); ramp ≈ 3.5. The engine-init estimate (O3's 170.2 s load ran with an unknown page-cache state, so this is indicative only):
  - init_device 4.1 + `load_model` ≈ 157 (O3 normal 170.2 ✔, minus ≈ 12.6 of graph build saved by FastGraph) + snapshot 7.2 + memory/KV 2.6 + compile/warm-up ≈ 35–38 (warm cache; trace + lower + cache reads, O0/O3 normal ✔) + bank write 2.4 ≈ 208–211.
  - That is below the [INF] 237 s, by the FastGraph saving and because process start is overlapped by the pre-spawn.
  - The warm follow-up equals O3: C ≈ 27.5 s.
  - Source impact during the pre-spawn: none (O2/O3).
- Code: frozen at the F1-passing commit; F1 and F2 must use the same `hetero/`, because `REFERENCE_H` carries the code revision in `engine_identity`. If F1 needed a fix, F2 runs at that fixed commit with `REFERENCE_H` empty, and the campaign then makes its own H normal-loader reference.
- Steps: `HEAD_POD=tpu-ray-cluster-vllm-tpu-head-5mkgz REFERENCE_H=results/switch/first-gate-<commit> flock /tmp/hetero-cluster.lock scripts/switch_campaign.sh first`, run detached, with driver log `results/switch/f2-first-driver-<commit>.log`. The run base is `first-<commit>-ov`.
  1. Prep: preflight → stage the W6 catalogue → default tmpfs cleanup → S normal → S snapshot → S reference → teardown → `files S` → S restore + validation.
  2. Cold reps r1–r3: run-switch S→H, full, W0 → W6, `--static-target-s 206.2`, `--prespawn-at-s 0`. Target settings: `COLD_PRESPAWN` + scope `<rid>`, i.e. normal loader + FastGraph + inline snapshot + bank write + pre-spawn. Then: `cold_complete` (completed, accounted, reconciled, `provisioned.all_provisioned`) → H greedy validation against `REFERENCE_H` → `files H --run-scope <rid>` → teardown H → remove the scope (r1, r2 only) → S restore + check.
  3. Warm: run-switch with `FAST_PRESPAWN` and the r3 scope (restore + bank read) → `switch_complete` (all restored) → validation against `REFERENCE_H` → teardown → remove the r3 scope.
  4. Locally: `switch_summary.py`, `overlap_check.py` for every rep (with the validation files), and `analyze_overlap`-style source windows.
- Endpoints:
  - C_production = gap to the first W6 dispatch + ramp (makespan − 206.2), per rep and as the mean of the cold reps, with the contiguous segments;
  - engine-init split: init_device, graph_build, weight_load (checkpoint_iter / host_convert / shard_place), the rest of load_model, snapshot_write, determine_available_memory, initialize_from_config, compile_or_warm_up_model, exec_bank_write;
  - C_first vs the [INF] 237 s; warm C vs O3's 27.5 s;
  - amortized per-switch cost for N = 1, 3, 10, from measured C_first and C_warm: (C_first + (N − 1)·C_warm)/N, next to the separate-pass accounting (351 + 27.5·N)/N from the O3 decomposition (decision log, 2026-10-12);
  - source tok/s in the O2 windows (2,661 / 4,070).
- Pass/stop:
  - Every cold rep is complete with exact accounting (W0 1,763 req / 721,595 tok; W6 complete), segments reconcile, the pre-spawn checks are clean (parked before T0 and release; self-checks clean), it is provisioned on 8/8, its greedy output equals the reference, and its scope files are complete.
  - The warm rep is restored on 8/8 with the bank used on 8/8, 0 backend compile and greedy equal.
  - A failed rep follows the existing fail-closed loop: release, scope removal, S rebuild, a new attempt id, at most 2 failed attempts. A failed check is a blocker (≤ 2 fix attempts), and F3 does not start.
- Expected ≈ 75 min: prep 15, each cold rep ≈ 15 (phase 1 190 s + C ≈ 220 s + W6 206 s + validation + teardown + S restore), warm ≈ 10.
- Cleanup receipts: per-rep scope cleanup JSONs, `cleanup-release-{H,Mb,S}.json`, `cleanup-tmpfs.json`, RCM Idle, lock free.
- **Advisor review (fable-5-1): accept with changes, all adopted** in `research/first-switch` @ **`ea16e24`**. That commit changes scripts only; `hetero/` is byte-identical to `11bd0fb` (`git diff --stat 11bd0fb ea16e24 -- hetero` is empty), so `RUNTIME_REVISION`, the bank identity and F1's reference `engine_identity` carry over. F2 runs at `ea16e24`:
  1. Before every cold rep and the warm rep, a read-only per-host record of checkpoint page-cache residency (`snapshot_validate.py pagecache`: mincore over `/models/gemma-4-31b/*.safetensors`, which reads no bytes) plus meminfo. `host_convert` is reported per rep, and the reps are not averaged blindly if the regime drifts.
  2. Same-model regime, as in the decision row above. Per-rep trace/lower/backend compile in compile/warm-up; r1 is reported separately if its backend compile is not ≈ 0.
  3. C_first is compared with both the 237 s [INF] (no FastGraph) and the 208–211 s [INF] (FastGraph), with the measured graph_build.
  4. Amortization columns: this campaign's warm C (n = 1), O3's 27.51 s (n = 3), and the separate pass (255.4 + 7.2 + 88.2 = 350.8 s of deploy walls, + 27.51·N)/N.
  5. The warm rep gets a post-validation `check --require-restored`, for its serving bank counters.

### F3a — normal-loader split and pre-read design (local) — done; advisor-reviewed

Analysis (opus-5-5 research agent; transcript `agent://OrganizerFirstSwitch.F3aLoaderSplit`). Sources: tpu_inference @ `8d31f0866` (TI), vllm @ `0ba2aa35a` (VL), jax 0.10.2 (J), O3 artifacts, and LoadProfiler (`~/hetero-wt/LoadProfiler`, `results/restore/LoadProfiler/summary.json`; a separately profiled 240 s TP16 load, so not a decomposition of the 170 s run):

| Step | file:line | host / device | s |
|---|---|---|---|
| get_model → get_flax_model, dtype/quant config | TI `models/common/model_loader.py:614-643, 316-348` | host | 0.06 ✔ (O3 phases) |
| graph build `nnx.eval_shape(model_class(...))` | `model_loader.py:270-271` | host trace; needs a Mesh object, no device data | 14.25–14.99 ✔ (O3); 2.1 with FastGraph ✔ |
| checkpoint iterator `safe_open`/`get_tensor` (lazy mmaps) | VL `default_loader.py:287-298, 427`; `weight_utils.py:861-907, 969-974` | host | 0.18 (LP) |
| torch reshape/permute + `t2j` on the JAX CPU backend; checkpoint page faults land here | TI `models/jax/utils/weight_utils.py:742-766`; `utils.py:78-95` | host | 127.6 (LP) |
| qkv/gate_up concat + reorder | TI `layers/jax/quantization/unquantized.py:144-167` | host | 5.5 (LP) |
| local shard selection, `make_array_from_callback` | TI `layers/common/utils.py:154-160` | host | 6.0 (LP) |
| load-time CPU compiles / cache lookups | J `compiler.py:387` | host | LP 76.8; ≈ 5–10 in O3 [INF] |
| post-load `jax.clear_caches` | `weight_utils.py:1005, 1073` | host | 4.25 (LP) |
| H2D `device_put` | `utils.py:128-168` → J `pxla.py:221` | **device** | 1.64 (LP; 15.82 GB/rank) |
| `create_jit_model` + `nnx.split` | `model_loader.py:153-177, 305-307, 366` | **device** | ≈ 2.8 [INF] |
| **load_model total** | `hetero/tpu_worker.py:344` | — | **170.05 ✔ (O3)**; host ≈ 165, device ≈ 4–5 [INF] |

- Deviceless facts:
  - `jax.devices('cpu')` initializes every platform, including the TPU (J `xla_bridge.py:799-860, 952`), and a plugin cannot be unloaded. So the parked worker must not run JAX compute, and a pre-read must run in a **separate CPU-only process**.
  - The snapshot manifest needs only payload- or file-derived context plus the mesh device order. Stub v5e devices through `mesh_utils.create_device_mesh` reproduce the recorded H mesh ids `[0,1,2,3,7,6,5,9,10,11,15,14,13,12,8,4]` ✔ (O3 bank identity).
  - Specs and shard indices depend on mesh position only, so a 16-CPU-device `(1,16)` `('data','model')` mesh reproduces them.
- Designs, ranked:
  - **A**: a CPU-only subprocess started by the parked worker writes this rank's TP16 snapshot in the `weight_snapshot` v2 format, so the released target takes the existing restore path. Saving if the pre-read finishes before T0: up to ≈ 165 s of `load_model` + 7.2 s of snapshot write [INF]. ≈ 480 LOC + 250 LOC of tests.
  - **C**: a numpy shard-first writer; an optimization of A, high risk.
  - **B**: page-cache warm only. Saving unknown; ≈ 0 if the cache is already warm.
  - **D**: A inside the parked process; rejected, because backend init is irreversible.
- Budget per host ✔ (O3 `preflight.json`, `prep-*-memory.json`, `prep-snapshot-files.json`): idle MemAvailable ≈ 195 GB, /dev/shm free 191 GB. The S snapshot is 31.45 GB/host and the H snapshot 15.82 GB/host. With S and H snapshots present: MemAvailable 147.5 GB, tmpfs free 143.9 GB. Peak non-reclaimable with S serving and a pre-read child ≈ 76 GB [INF], against the 40 GB floor.
- **Advisor review (astra-6): accept with changes.** Gate a single-process, *streaming* A, after a cheap attribution pre-step:
  1. Prove isolation before coexistence. `tpu_inference/__init__.py` has an engine-first import hook (tpu_raiden: absent from the worker image ✔, pip metadata 2026-10-12), so `JAX_PLATFORMS=cpu` alone is not proof. Record the child's TPU-relevant opens and mappings, not just a final sample.
  2. Streaming: intercept `assign_and_shard_param` (TI `weight_utils.py:769-790`) and write only this rank's shards immediately; one process, bounded threads, no layer parallelism or C rewrite in this gate. Account for retained merged tensors (`unquantized.py:144-167`). The 12 GB child peak is unproven, so measure it.
  3. Independent exactness: validate every path, spec, index and device, and every shard hash, against a **device-produced** snapshot (tiny model locally; the full model on all 8 hosts outside the timed coexistence). Generation agreement is per instance. A refusal or fallback fails the gate.
  4. Neither a 145 s preparation time nor a 165 s saving is established; the bank write remains.
  5. First read F1/F2's `host_convert` / `shard_place` and the page-cache records; run a separate sequential-read comparison only if they leave I/O vs conversion unresolved.
  6. Frozen acceptance: done before T0, < 5% S degradation, memory floor, exact accounting, restored and greedy equality, bounded child cleanup, ≤ 2 fixes.
- Decision (organizer): implement the advisor's minimal streaming A in a new worktree **`~/hetero-wt/FirstSwitchPreread`** (branch `research/first-switch-preread`, from `ea16e24`), in parallel with F1/F2, which run frozen from `FirstSwitch`. F3b's spec is written after the F2 results and the attribution pre-step.

### F1 — results, attempt 1 (`11bd0fb`, run `first-gate-11bd0fb`) — **incomplete: stopped by the local disk floor before the restore deploy** (not a code blocker)

Artifacts: `~/hetero-wt/FirstSwitch/results/switch/first-gate-11bd0fb/` (evidence commit `9e0df49`), driver log `results/switch/f1-gate-driver-11bd0fb.log.gz`, inventory `results/switch/inventory-2026-10-12.json` (same 8 hosts `.56–.63` as 2026-10-10 ✔).
- Preflight and `memory H` passed. The H normal deploy with inline snapshot + bank write **succeeded** (hetero_ctl 861.5 s; engine init 850.7 s) ✔:
  - `provisioned_flags`: snapshot published and bank written on **8/8** workers, one generation and one identity digest per instance ✔.
  - **Time added: `snapshot_write` median 7.10 s (max 7.39 s; 15,824,457,568 B/rank); `exec_bank_write` median 2.57 s (max 2.62 s; 187 entries/rank, 0 skipped)** ✔.
  - `H-first-scope-files`: one v2 manifest (15.82 GB) and one bank manifest (187 entries, ≈ 378 MB) on all 8 hosts ✔.
  - The greedy reference `prep-H-reference.json` was produced from this deploy ✔.
- Truly-first (cold) regime ✔ (median over 8 workers). These pods had no XLA cache, and the checkpoint was cold for this deploy:
  - init_device 3.16 s; **load_model 571.2 s** (graph_build 1.82, weight_load 565.1, rest 4.25); snapshot 7.10; initialize_from_config 30.2; **compile_or_warm_up 189.8 s** (backend compile 148.2, trace 14.1, lower 14.8, warm-up 9.8); bank write 2.57.
  - Inside weight_load: checkpoint iterator 14.8 s and shard placement 44.4 s. host_convert sums to 780 s, more than its 565 s window. [INF] The two hooks overlap: unquantized's fused path calls both wrappers, so its time is counted twice. The split is valid only as an upper bound and needs a non-nested hook before F2 (see blockers).
  - The cold load (571 s vs O3's 170 s) shows that the checkpoint read regime dominates load_model. This is the attribution pre-step the F3 advisor asked for [INF: page-cache state was not recorded in this attempt; `pagecache` exists from `ea16e24`].
- Stop ✔: before `deploy H H-first-restore`, `disk_floor` found local `/` at 9.1 GB (< 10 GB). The causes are other activity on the controller host (`~/.cache/huggingface`, `~/tts`, rust toolchains, all outside this run) and RCM's 5.6 GB `/tmp/omp_rcm/telemetry.jsonl`. The campaign stopped fail-closed. A rerun of the whole gate is needed: the restore half (restored 8/8, bank used, 0 backend compile, greedy equal, 0 misses/rejections) is **not yet shown**.
- Cleanup receipts ✔: EXIT trap released H/Mb/S (all owned PGs REMOVED, `cleanup-release-*.json`); the scope was removed on all 8 hosts (`cleanup-scope-first-gate-11bd0fb.json`: snapshot 15.83 GB + bank 378 MB per host existed and are absent afterwards); default tmpfs root cleaned (`cleanup-tmpfs.json`). Afterwards RCM was Idle and the lock free. Side effect: the H XLA cache entries now exist on the pods (warm state, kept).
- Process note: during attempt 1, `ea16e24` (scripts only) was committed in the running worktree. Later jobs of this attempt uploaded `snapshot_validate.py` with the added, unused `pagecache` action; no action used by F1 changed. `switch_campaign.sh` was restored in place to `11bd0fb`'s bytes, so the running bash was unaffected. From now on, edits go only to other worktrees while a campaign runs.
- Housekeeping: `/tmp/omp_rcm/telemetry.jsonl` was gzipped to `/tmp/omp_rcm/telemetry-until-20261012T0550.jsonl.gz` and truncated (RCM idle).

**Status at organizer return (budget stop, 2026-10-12):** F0 done; F1 needs a rerun (attempt 2); F2 not run; F3a done; F3b/F3c not run. Pre-read implementation (design A) in progress on `research/first-switch-preread` (`~/hetero-wt/FirstSwitchPreread`, from `ea16e24`) by an opus implementer; its commits are unreviewed. Next steps:
1. Fix the nested `host_convert` accounting (time only the outermost call).
2. Get local `/` above 10 GB.
3. Rerun F1 at the fixed commit (`REFERENCE_H` from that run).
4. F2 per spec.
5. F3b spec after the F2 data.
