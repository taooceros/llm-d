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

## Experiment queue (bf16 KV)

| ID | Kind | Item | Exit criterion |
|---|---|---|---|
| Q0 | local | Make bf16 KV the default on branch `research/bf16-kv` (worktree `~/hetero-wt/Bf16Kv`, based on `StagedFastRestore` @ `fdc9f7e`). Check backend support, update the worker/experiment dtype checks, the gates and the tests. Define the bf16 correctness gate: new per-layout greedy references plus a cross-layout agreement threshold. | Unit tests pass; commit; gate defined here |
| Q1 | cluster | bf16 provisioning and smoke for S and H. First deploys are cold because bf16 changes the cache keys. Record the KV pool per instance, generate the bf16 references and check cross-layout agreement. | KV pool recorded; gate passes; or a blocker is recorded |
| Q2 | cluster | Re-measure bf16 per-layout rates: W0 (small) on S and H, W5 (middle) on S and H, W4 (large) on H. Recalibrate the cost model. | Rates ✔; break-even phase sizes under bf16 |
| Q3 | cluster | Warm full S→H switch with fast restore and a pod-local cache. Record every segment as in the verified breakdown above. | Production C ✔ |
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
