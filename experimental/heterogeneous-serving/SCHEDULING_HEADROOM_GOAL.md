# Goal: improve sustained bulk inference beyond the best static concurrency policy

## Assignment and status

Investigate whether native-state-aware admission and adaptive active concurrency can improve real bulk inference beyond the best already measured static configuration. Reuse existing batch-size measurements; do not rediscover that larger batches can be faster. Implement one bounded controller, qualify its execution contract, and measure one focused comparison. A speedup, regression or unresolved difference is an acceptable honest outcome; success is not predetermined.

This is a goal prompt for subsequent execution. Writing it does not launch jobs, claim the controller exists, or reactivate the abandoned full Phase-3 matrix. It supersedes the earlier cap64-only, W-LARGE-first experiment prescription. `local/design_offline_planner.md` retains the native transition and information-boundary contract; this prompt governs workload selection, concurrency scope and measurement sequence.

## Objective: bulk inference, not interactive serving

Use a fixed, deep backlog larger than KV capacity, with sustained memory pressure and enough request heterogeneity to offer admission choices. Optimize **measured dispatch-to-drained makespan at asserted-equal useful emitted tokens**, equivalently useful-output throughput for that fixed workload.

- TTFT, per-request latency, fairness and short-request responsiveness are not competing objectives.
- Prefer improvements that persist while backlog remains plentiful. A shallow workload's terminal-drain improvement is not evidence of better sustained bulk scheduling.
- Preemptions, mean batch, padded lanes, host allocation, recomputed tokens and model scores are diagnostics, not objectives. Native preemption is permitted when its throughput tradeoff is profitable.
- All online control, observation, state maintenance and fallback work stays inside makespan. Report startup separately and also report startup-plus-serving time; do not hide a large startup behind the serving metric.
- State makespan reduction and throughput increase separately. Do not promise a 5%, 10% or 20% gain from earlier speculative discussion.

## Existing evidence to reuse first

Primary bulk evidence: `results/scheduling_tracks_20260914/REPORT.md`, especially the bulk bucket comparison and its underlying artifacts. The B1/W-SMALL manifest has 1,763 requests, 721,595 useful output tokens and SHA-256 `5f30f9fa9ff6eb097618448abd434843b7c04986187561da9419bfd80013936a`.

| Historical arm | Makespan (s) | Useful tokens/s | Mean scheduled batch | Preemptions |
|---|---:|---:|---:|---:|
| Arrival, cap64 | 371.5 | 1942 | 55.2 | 0 |
| Arrival, cap128 | 340.8 | 2117 | 89.4 | 0 |
| Arrival, cap256 | 357.8 | 2017 | 108.9 | 873 |
| Arrival, uncapped | 361.1 | 1999 | 109.8 | 865 |

These rounded historical results already show approximately 8.3% less makespan from cap64 to cap128, followed by approximately 5.0% more makespan from cap128 to cap256. Cap128 is the best tested static operating point in this cohort, not a proven optimum. Arrival cap128 also beat the tested short-decode-first ordering variants. Do not substitute an inferior ordering or cap as the primary comparator.

Controlled decode evidence: `results/decode_allocation_20260913/envelope.csv`, its report, provenance and held-out measurements. For TP16, prompt length 1024 and forced decode length 256, measured decode rates were 139.403921, 128.373958 and 153.910945 tokens/s/chip at active batches 64, 65 and 113 respectively. This demonstrates a padding boundary and recovery as the larger bucket fills; it does not predict mixed-workload makespan.

Check provenance before reuse: topology, model/build, context distribution, prompt source, execution mode, instrumentation and measured quantity. Historical controlled uniform/synthetic-prompt decode measurements are not real-ShareGPT mixed-workload evidence or isolated device timings. Preserve those qualifications. Never turn decode-only coefficients, scheduled-token ratios or padded-lane counts into an alleged measured wall-time cost.

Other relevant evidence:

- `results/policy_headroom_20260914/REPORT.md`: conservative reservation sometimes helped, sometimes substantially hurt; fewer preemptions was not sufficient.
- `results/20260915_planner/QUICK_RESULTS.md` and `P3_native_pair_results.json`: W-LARGE native LPT cap64 took 299.619 s versus 307.864 s for native planner order. These are a different workload/cohort, not alternatives to the W-SMALL baseline above.
- Existing failed and fallback-dominated planner/F3 runs remain evidence. Do not repeat their broad matrix or discard their negative findings.

Produce a compact evidence table distinguishing what is measured, what is inferred, and what remains missing. Use source artifacts rather than treating an old report's causal interpretation as established fact.

## Investigation question and hypothesis

Can the controller retain the batching benefits of temporarily higher concurrency while avoiding enough subsequent recomputation, poor batch composition or admission stalls to beat the best static policy after its own overhead?

Begin with B1/W-SMALL's saved manifest, not a newly selected favorable draw. Use the existing cap128-to-cap256 region as the first operating range to investigate; lower effective concurrency is allowed when the controller defers admission. Do not rebuild the engine or change `max_num_seqs` to adjust the admission policy.

Distinguish three quantities throughout:

1. Backlog depth: requests available for service.
2. Admitted, unfinished population: requests submitted to or retained by the engine, including waiting/preempted requests.
3. Actual scheduled batch: requests/tokens receiving service in a particular native batch, and their actual padded shapes.

A larger admitted population does not guarantee a larger scheduled batch. Oversubscription here permits admission beyond a conservative/static request cap or endpoint reservation; it never permits physically allocating beyond the native KV pool. KV offload, migration, replica/topology changes and custom eviction are outside this first experiment.

## Non-negotiable architecture

**Do not assign to or replace `core.step_fn`, `EngineCore.step`, `EngineCore.step_with_batch_queue`, or the outer engine stepping method.** Keep the installed native asynchronous execution, queue capacity, sampling, output retirement and free rules unchanged. Record and verify that identity/configuration at installation, during execution and after cleanup. No forced synchronization, ready-first cutover or batch drain to simplify prediction.

Intercept only an audited fresh waiting-admission boundary, after native resident scheduling has established remaining token budget and provisional allocations. The precise installed source location must be qualified before intervention. First keep baseline arrival priorities and native resident traversal/victim selection unchanged; change admission timing/effective concurrency only. Candidate-order optimization is not bundled into the first comparison.

- Use shared audited native token/chunk/allocation arithmetic for immediate previews and commit. Do not implement a second approximate scheduler.
- Track observed generated tokens, computed cursor, issued-but-unobserved work, actual allocations, request phase, queue membership, provisional work and issue/retirement identities separately.
- Preview from the actual native program point, not an earlier snapshot before resident allocations.
- Return a bounded admission/defer action with state version, prerequisites and expected immediate structural delta; reject stale actions before committing.
- Native code owns allocation, refunds, request transitions and output processing. No post-filtering `SchedulerOutput` and no hand-adjusting counters to make a prediction appear correct.
- Spend actual free capacity only. Predicted completion is not a returned block; already allocated in-flight blocks must not be counted again as unallocated debt.
- Deferring fresh admissions must not prevent recomputation/resumption, resident service or output retirement. An empty active set with a feasible backlog must make progress.
- Unsupported source/configuration and deterministic preview mismatches invalidate optimized intervention. Restore baseline progress, preserve the failure and make no certificate claim.

Native preemption remains legal. The old synchronous release-credit proof, ideal calendar area bound and a finite-horizon forecast do not establish a native asynchronous zero-preemption guarantee or a wall-time bound.

## Forecasts, costs and information boundaries

Reuse existing `FutureModel` and measured batch/context information rather than starting a long timetable search. Forecasts must include residual work beyond their horizon; postponing a growth event or recomputation past that horizon cannot make it free.

Start with Track C to test the mechanism under exact target information. Remaining useful work is target minus **observed** generated tokens, not scheduler calls or the computed cursor. Track pending work separately to avoid duplicate issue.

Preserve Track O's legal boundary even if only C gets the first timing comparison. O may receive the public workload law, prompt features, actual output/allocation observations and permitted issued/retired-work observations; reject targets, seeds, nested payloads and subclass carriers. Inputs remain immutable copies. An adapter must not leak hidden targets through exact counterfactual completion/feasibility answers. Use the same action mechanics for C/O; do not claim an oracle-gap result from C versus an oracle-free static baseline.

Reuse measured decode curves within their supported domain. Mixed prefill, recomputation, context composition and overlapping issue/retirement require separate qualifications. Do not sum overlapping batch durations as makespan, equate token waste with time waste, subtract an independent prefill run to invent pure decode performance, or invent missing coefficients. If the evidence does not support a proposed score, state that before implementation and choose a bounded, explicitly testable hypothesis rather than presenting an unvalidated predictor as ground truth.

Keep the controller small and event-driven. Proposed limits: 5 s startup, 1 ms per admission decision including snapshot/preview, and total online controller CPU no more than 1% of matched baseline makespan. Charge maintenance and unsuccessful proposals. Use explicit baseline fallback on exhaustion and record both decision coverage and budget violations. Fallback-dominated timing is not evidence that the new scheduling idea was evaluated successfully.

## Bounded execution sequence

1. **Reuse evidence and select one hypothesis.** Locate saved manifests, native source and relevant per-step artifacts. State the expected structural effect and what result would falsify it. No new broad batch-size or workload sweep.
2. **Implement and shadow-check one admission seam.** Run targeted arithmetic/native-boundary checks, then a short real-serving shadow run with decisions non-intervening. Compare deterministic token/allocation deltas at matching boundaries. Exercise actual block growth, in-flight output retirement, stale-action rejection, recomputation eligibility and no-progress handling. CPU fixtures prove mechanics only, never performance.
3. **Run one frozen-source bulk comparison after conformance passes.** Use the exact saved B1/W-SMALL manifest, same engine configuration and trace settings for:
   - A: native arrival, static admission cap128 — primary best-known reference.
   - B: native arrival, static admission cap256 — larger-concurrency control.
   - C: native arrival with the state-aware controller, maximum admitted population256 — adaptive timing/concurrency treatment.
   Keep the native engine maximum and physical KV capacity identical for all arms. Maximum256 is a policy population limit, not a promised scheduled batch. Historical times motivate these arms but do not substitute for matched current-source controls. Assert identical manifests and 721,595 useful emitted tokens; report startup and all online costs. No separate full default/cap64/LPT/eviction matrix.
4. **Attribute the result conservatively.** C versus A asks whether adaptive admission beats the best existing static configuration; C versus B controls for simply allowing more concurrency. Actual batch/context distributions, padding, issued work, preemptions and recomputation explain candidate mechanisms but do not independently price wall-time components.
5. **Stop or confirm narrowly.** A clear regression, no genuine intervention coverage or failed conformance ends this hypothesis with a report. If a promising or close result needs repeatability evidence, repeat the relevant matched pair with reversed order and no source/configuration edits. Do not borrow a historical repeat range as today's noise floor. No automatic second workload, backlog-depth series, long search or parameter sweep. Any additional experiment needs a specific unresolved question, not a desire to obtain a positive result.

For sustained-bulk interpretation, predeclare a common useful-progress interval from existing baseline traces in which fresh backlog remains abundant, and report useful tokens per wall second there as well as full makespan. Use identical progress boundaries, not policy-specific favorable time windows. Verify backlog adequacy for every arm; if no common sustained interval exists, retain the whole-workload result but do not call it sustained-bulk evidence. Report terminal drain separately and do not redefine windows after seeing the winner.

## Hardware, provenance and cleanup

- All performance evidence must use physical TPU v5e/TP16, real `vllm.LLM`, mounted `/models/gemma-4-31b`, and real ShareGPT prompts. No offline policy simulator, mock execution, synthetic GEMM shortcut or estimated timing presented as a benchmark.
- Reuse actor `measurement_audit_20260914_tracks_tp16` in namespace `omp_measurement_audit`, run-id `20260914_tracks`. Do not reload weights or recreate its placement group. If unavailable, report the prerequisite rather than silently launching a cold replacement.
- Submit through `scripts/submit_measurement_audit.py` under RCM. Recovery first, one arm per plan step, full plan paths relative to `/workspace`. Monitor through RCM status/wait/logs, not bespoke Ray API polling. Follow the trap/fix/resume protocol.
- Any genuinely required new distributed initialization still needs topology discovery and explicit node-pinned placement; do not remove the retained actor's active group as stale.
- Keep prefix caching disabled, `warmup=False`, `manifest_per_repeat=False`, and real prompt provenance. Record source/module/native build/configuration hashes and ordered manifests. No source changes between timing arms.
- Retain per-iteration prefill/decode and useful-output evidence, actual worker shapes where available, and matched instrumentation. Do not pretend trace collection is free or host allocation counters are physical HBM telemetry. Do not launch a profiler unless an identified unanswered question requires it; use on-actor summaries rather than ingesting raw traces.
- Restore admission hooks/queues without losing or duplicating requests. Verify actual engine quiescence, an empty batch queue and all 566 usable blocks returned after excluding the null block. Stopping a Ray submission alone does not prove a detached actor call stopped.
- Use Jujutsu, not raw git. Preserve failed and fallback-dominated artifacts. After verification, remove owned throwaway scripts and update relevant implementation documentation; do not add broad tests or cleanup unrelated work.

## Deliverables and completion criteria

Deliver a concise report plus machine-readable evidence containing:

1. The existing-measurement table, selected hypothesis, reused curve domain and remaining uncertainties.
2. The implemented admission contract, explicit unsupported cases, native-step identity proof and shadow/conformance results.
3. Matched A/B/C results: full makespan, useful throughput, startup, total online overhead, actual intervention/fallback coverage, scheduled/padded batch distributions, recomputation/preemptions, host allocation, sustained interval and terminal drain. Any invalid/incomplete arm remains visible and excluded from winner claims.
4. An explicit answer to **whether adaptive concurrency/admission improves on the best static policy**, not merely on cap64, uncapped admission or the old synchronous planner. Qualify repeatability and keep C-only and workload-specific scope explicit.
5. Verified allocator/engine cleanup with weights retained, all affected callers/docs updated, and preserved provenance.

Completion requires a real qualified outcome or an explicitly evidenced external blocker after reachable work is finished. Correct mechanics without timing is not a demonstrated throughput improvement. No gain is a valid conclusion; inventing a gain or expanding the investigation indefinitely is not.
