# Goal: when does reconfiguring the TPU slice pay for itself in bulk inference?

## Assignment and status

Quantify how much a bulk-inference job gains by changing the serving layout of the 32-chip TPU v5e
slice between workload phases (H = 2×TP16, S = 4×TP8, M = TP16 + 2×TP8), and find the job size
above which that gain exceeds the measured cost of reconfiguring. Reuse existing per-layout
measurements and the calibrated simulator for planning; the conclusion must rest on measured
cluster runs. "Reconfiguration never pays at realistic sizes" is an acceptable honest outcome.

## Objective: bulk inference, not interactive serving

A job is a fixed backlog: an ordered sequence of phases, each a known manifest (W0, W1, W3, W4, W5)
with a known number of useful output tokens. Optimize **total wall time from first dispatch to
last phase drained, including every reconfiguration**, at asserted-equal useful emitted tokens.
Compare against the best *single* static layout for the whole job, chosen with hindsight. A layout
that cannot serve a phase (S on W4) is infeasible for that phase, not merely slow.

## Existing evidence to reuse first

Per-layout makespans (seconds; lower is better):

| Workload | Output tokens | H | S | M | Source |
|---|---|---|---|---|---|
| W0 chat | 721,595 | 516 | 330 | 500 | simulated, blind (measured M0 401.6 s, `pilot-m-20260929`) |
| W1 chat, mid outputs | 663,746 | 589 | 473 | 563 | simulated, blind (measured M0 length-aware 506.5 s) |
| W3 reasoning (P≈279, D≈5.9k) | 1,079,823 | 1284 | 1354 | 1382 | measured `w3-c7edd5a-r1`, blind |
| W4 long-doc + chat | 387,975 | 347.1 | infeasible | 311.9 | measured `w4-*`, M with reserve policy |
| W5 TED translation (P≈3.5k, D≈4k) | 983k (250-request sample) | 1374 | 1508 | 1374 | simulated, blind |

- Simulator: `scripts/simulate_length_aware.py` (step model fitted on W3, validated on W3 blind to
  within 1–2% makespan). It is not validated on W0, W1 or W5, and it overestimates W0 on M by about
  25% (500 s simulated vs 401.6 s measured), so treat those rows as planning estimates and measure
  any row the conclusion depends on.
- Reconfiguration cost measured so far: `deploy` 358–370 s (W4 campaign, `/tmp/w4-deploy-*.json`);
  full teardown + deploy in the W3 campaign 510–555 s; `drain`, `gateway-down`, `release` together
  about 3 s. During a full redeploy all 32 chips produce nothing.
- W4 report: `llm-d/experimental/heterogeneous-serving/docs/experiments/w4_longdoc_heterogeneity.typ`.

## Investigation question and hypothesis

With per-phase throughputs r_L and a reconfiguration cost C (seconds of lost slice time), switching
from layout A to B for a phase of N output tokens pays when

    N / r_A − N / r_B > C,   i.e.   N > C · r_A · r_B / (r_B − r_A).

Hypothesis: the per-phase gain is 10–35% where phases differ in kind (chat → S; KV-bound long
context → H; TP16-only prompts → M), so with C ≈ 6–9 min the break-even phase size is on the order
of 10⁶ output tokens — about one to three W3-sized phases. Falsified if the measured C (including
compile/warm-up to steady throughput) or the measured per-phase gaps put break-even beyond ~10⁷
tokens, or if the best static layout is within noise of the reconfigured schedule for every mix.

## Non-negotiable measurement rules

- Measure C end to end for each transition you use (H→S, S→H, H→M, M→H, S→M, M→S): from last
  request of the old phase drained to the new layout serving at steady throughput. Split it into
  drain, release, placement-group creation, weight load, XLA compile/warm-up and gateway rollout.
  Report the time-to-steady-throughput, not only `deploy` elapsed.
- Every performance claim uses real `hetero_ctl` runs on the TPU slice through `rcm_exec`, real
  manifests and exact output budgets (`ignore_eos`). The simulator may choose which runs to do; it
  is never the evidence.
- Same model, same chips, same routing policy family across compared arms. State the policy per
  phase (blind vs reserve vs length-aware) and do not credit a routing change to reconfiguration.
- Cluster rules in `AGENTS.md` apply: one experiment at a time, `status` before `run`, and **every
  deploy, drain, release and timed run needs the user's approval first**.

## Design space to evaluate

1. **Full redeploy** between phases (what exists today).
2. **Partial reconfiguration**: keep the engines shared by both layouts serving while only the
   changed half is redeployed (H↔M keeps one TP16; S↔M keeps two TP8). Today `deploy` recreates
   every instance because instance names differ between layouts (`large` vs `large_a`); making the
   reconcile loop keep a matching engine is in scope if it is needed for the measurement. Measure
   how much of C this hides and how much throughput the surviving half delivers meanwhile.
3. **Warm start**: whether weights or compile caches can be kept across a redeploy to cut C
   (persistent XLA cache, host-resident weights). Measure; do not assume.

Out of scope unless the above shows a clear need: changing the slice size or requesting more chips,
live migration of running requests across the transition (continuation qualification still fails).

## Bounded execution sequence

1. **Plan from evidence.** Build the throughput table and compute break-even N for every useful
   transition with C = 360 s and C = 555 s. Pick at most two job mixes that the model says benefit
   (for example W0-like chat followed by W3-like reasoning; W4 followed by W5) and one that should not.
2. **Measure C.** One instrumented transition per direction you rely on, with a stage-by-stage
   timeline and the time to steady throughput from iteration logs.
3. **Close the throughput gaps you rely on.** Measure any row in the table that is only simulated and
   that the chosen mixes depend on (at least W5 on H and M, and W0/W1 on S and H if chat phases are
   used). Two repetitions per arm.
4. **Run the comparison.** For each chosen mix: best static layout end to end vs the reconfigured
   schedule end to end, including real transitions, two repetitions each, order alternated.
5. **Fit and report break-even.** Using measured C and r_L, report break-even phase sizes with
   their uncertainty, and check the model against the measured end-to-end results.

## Deliverables and completion criteria

A concise Typst report (HTML-first, same conventions as the W4 report, pushed to the fork branch
`research/heterogeneous-tpu-20260918` with a gistd link) plus machine-readable evidence containing:

- the measured reconfiguration timeline per transition, with a breakdown and time to steady throughput;
- the per-phase throughput table, marking each value measured or simulated;
- end-to-end results for each mix: static vs reconfigured wall time, per-phase time, lost time in
  transitions, repetitions and spread;
- break-even output tokens (and equivalent requests) per transition, with the formula's inputs;
- whether partial reconfiguration or warm start changes the answer, with measurements;
- threats to validity and the commands, commit ids and run ids needed to reproduce.

Done when every number used in the break-even conclusion is either measured on the cluster or
explicitly labelled as a simulator estimate that the conclusion does not depend on.
