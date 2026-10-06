# Bulk-inference reconfiguration: plan and decision log

This file is the single source of truth for this study's scope, decisions, experiment queue and results. Chat transcripts and `local://` scratch files are not. Update it in the same change as any decision or result.

## Goal

The workload is bulk (offline) inference of Gemma-4 31B on the 32-chip TPU v5e slice (8 hosts × 4 chips). A job runs in one-way phases: many small requests, then many middle requests, then a final batch of large requests.

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
| R4 | cluster | Requalify migration under full bf16 with the relative gate, across at least 12 cases (prompt lengths 256–4096 and both directions). Same-TP suffixes must match exactly, and host-KV vs recompute suffixes must give identical tokens. | Gate verdict ✔ |
| Q3 | cluster | Warm full S→H switch with fast restore and a pod-local cache; record every segment as in the verified breakdown above. | Production C ✔ |
| Q4 | cluster | Staged S→Mb→H (rerun of E2). | Effective C and join tail ✔, or a blocker |
| Q5 | cluster | Restrict precompile buckets to the shapes the bulk phases use (target: the 36 s trace+lower). Must show zero serving recompiles with `VLLM_XLA_CHECK_RECOMPILATION=1`. | Compile/warm-up time ✔ |

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
