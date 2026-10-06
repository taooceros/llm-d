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
  - **Re-measure under bf16.**
- FastSnapshot + FastGraph on S: `load_model` 5.78–5.96 s, greedy equality on 4 engines. A joining TP16 fast restore took 86.3 s. **Re-measure under bf16.**
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
