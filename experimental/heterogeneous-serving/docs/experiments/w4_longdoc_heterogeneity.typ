// W4: heterogeneous vs homogeneous TPU layouts on a long-document + chat mix.
// Compile: typst compile w4_longdoc_heterogeneity.typ
#set document(title: "Heterogeneous TPU serving on a long-document + chat mix (W4)", author: "hetero-research")
#set page(paper: "us-letter", margin: (x: 2.2cm, y: 2cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show table: set text(size: 9pt)
#show table.cell.where(y: 0): strong

#let good = rgb("#1a7f37")
#let bad = rgb("#cf222e")
#let note(body) = block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%, body)

// Horizontal bar chart of makespans; each row: (label, seconds, colour).
#let bars(rows, scale: 0.9pt) = grid(
  columns: (auto, 1fr), column-gutter: 8pt, row-gutter: 5pt, align: (right + horizon, left + horizon),
  ..rows.map(((label, value, colour)) => (
    text(size: 9pt, label),
    box(width: value * scale, height: 11pt, fill: colour, inset: (x: 4pt),
        align(right + horizon, text(size: 8pt, fill: white, str(value) + " s"))),
  )).flatten(),
)

#align(center)[
  #text(size: 16pt, weight: "bold")[Heterogeneous TPU serving on a long-document + chat mix (W4)]
  #v(2pt)
  Gemma-4 31B on TPU v5e 4×8 (32 chips) · pilot runs of 2026-09-30 · branch `research/heterogeneous-tpu-20260918`
]

= Summary

On a bulk backlog of 880 ShareGPT chat requests mixed with 24 real government reports whose context
is too long for a TP8 engine, the heterogeneous layout *M* (TP8 + TP8 + TP16) finished
#text(fill: good)[*10.1% sooner*] than the best homogeneous layout *H* (2 × TP16): a mean makespan of
311.9 s against 347.1 s (7.9% and 12.4% in the two repetitions). Mean useful throughput rose from
1118 to 1245 tokens/s on the same 32 chips.

The gain depends on the router. Under the pre-existing routing policy, M was
#text(fill: bad)[*30% slower*] than H (452.2 s): the router sends half of all new requests to the
TP16 engine and waits for it when it is full, so the TP8 engines ran out of work at about 210 s and
the long documents were starved. Two policy changes fix this: spill to whichever pool has room
(_work-conserving fallback_), and keep one maximum-length request's worth of KV on each TP16 engine
that short requests may not use (_long-request reserve_).

The 4 × TP8 layout *S*, which beat H on every earlier chat workload, cannot serve W4 at all.

#note[
  *Status.* These are unfrozen pilots with 2 repetitions per main arm and 1 for the diagnostic arms.
  They show the effect exists and roughly how large it is; they are not a timed, pre-declared
  comparison.
]

= Setup

== Hardware and layouts

All arms use the same 32 TPU v5e chips (a 4×8 slice) and the same Gemma-4 31B checkpoint
(`/models/gemma-4-31b`), served by vLLM + tpu-inference behind the Python Envoy ExtProc gateway.
The router never sees the output length (`decode_length_blind`); it sees prompt length and each
engine's free KV blocks.

#table(
  columns: (auto, auto, auto, auto, 1fr),
  align: (left, left, right, right, left),
  table.header[Layout][Engines][max_model_len][KV per engine][Notes],
  [H], [2 × TP16], [36,864], [282 × 256 = 72,192 tok], [homogeneous baseline],
  [M], [TP8 + TP8 + TP16], [16,384 / 36,864], [TP8: 412 × 64 = 26,368 tok \ TP16: 72,192 tok],
    [per-instance `max_model_len` (new `--instance-engine-settings`)],
  [S], [4 × TP8], [16,384], [26,368 tok], [infeasible: every long document exceeds a TP8 engine's whole KV pool],
)

At `max_model_len` 36,864 the TP16 engine allocates 256-token KV blocks, against 64-token blocks at
16,384. Any KV threshold the router applies must therefore be expressed in tokens, not blocks
(see @sec-validity).

== Workload W4

W4 is a closed backlog: all 904 requests are queued at $t = 0$ and the gateway dispatches them in
arrival order as KV capacity allows. Every request runs with `ignore_eos` and `max_tokens` fixed to
its reference output length, so every arm produces exactly 387,975 useful output tokens.

#table(
  columns: (auto, auto, auto, auto, 1fr),
  align: (left, right, right, right, left),
  table.header[Class][Requests][Prompt tok \ min / med / max][Output tok \ med / max][Source],
  [short], [880], [27 / 122.5 / 1,269], [246.5 / 4,096],
    [seeded sample of the historical W0 (prompt, output) demands bound to real ShareGPT conversations],
  [long], [24], [26,506 / 30,184.5 / 35,637], [636.5 / 981],
    [GovReport reports (`ccdv/govreport-summarization`); output = length of the real human summary],
)

Long documents need 27,069–36,511 tokens of context, all above the TP8 pool of 26,368 tokens.
Long request $k$ sits at a seeded position inside the $k$-th of 24 equal slices of the arrival
order (indices 30, 70, 107, …, 876), so long work arrives throughout the backlog. In total the
workload is 948,975 prompt tokens, 727,303 of them (77%) from the 24 long documents. Manifest
SHA-256 prefix `e380318ae33a`.

== Routing policies

Both policies share the base settings (length-blind, migration off, 16 blocks of headroom on large
engines). They differ in three fields:

#table(
  columns: (auto, auto, auto, 1fr),
  align: (left, center, center, left),
  table.header[Field][plain][reserve][Effect],
  [`work_conserving_fallback`], [off], [on],
    [if the preferred pool has no admissible room, try the other pool instead of waiting],
  [`long_reserve_tokens`], [0], [36,864],
    [short requests may not use the last 36,864 KV tokens on a large engine; long requests may],
  [`small_prompt_token_limit`], [4,096], [16,384], [prompts above this count as long for the reserve],
)

In M, both policies send 50% of new requests to the TP16 engine first
(`large_fresh_traffic_share` 0.5); prompts that cannot fit a TP8 engine always go to TP16. In H,
the share is irrelevant and only the reserve takes effect.

= Results

All 8 runs completed 904/904 requests with exact output budgets and one-to-one iteration capture.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, left, right, right, right, right),
  table.header[Layout][Policy][Makespan rep 1][Makespan rep 2][Throughput (mean)][vs best H],
  [H], [plain], [347.2 s], [346.9 s], [1118 tok/s], [baseline],
  [H], [reserve], [370.0 s], [372.6 s], [1045 tok/s], text(fill: bad)[+7.0%],
  [*M*], [*reserve*], [*304.1 s*], [*319.6 s*], [*1245 tok/s*], text(fill: good)[*−10.1%*],
  [M], [large-dedicated#super[†]], [332.9 s], [–], [1166 tok/s], text(fill: good)[−4.1%],
  [M], [plain], [452.2 s], [–], [858 tok/s], text(fill: bad)[+30.3%],
)
#text(size: 8.5pt)[† An earlier version of the reserve was counted in blocks; with 256-token blocks it
exceeded the TP16 pool, so no short request could use the TP16 engine. Kept as a diagnostic arm.]

#figure(
  bars((
    ("M reserve (mean)", 311.9, good),
    ("M large-dedicated", 332.9, good.lighten(35%)),
    ("H plain (mean)", 347.1, luma(90)),
    ("H reserve (mean)", 371.3, luma(140)),
    ("M plain", 452.2, bad),
  )),
  caption: [Makespan, lower is better. Same 32 chips and the same 387,975 output tokens in every arm.],
)

== Where the time goes

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, left),
  table.header[Run][Short p50 / last finish][Long p50 / last finish][Makespan][Preemptions][Short requests on TP16],
  [H plain r1], [113.9 / 278.9 s], [297.9 / 347.2 s], [347.2 s], [18], [880 (all)],
  [H plain r2], [118.3 / 289.5 s], [295.1 / 346.8 s], [346.9 s], [14], [880 (all)],
  [H reserve r1], [214.2 / 370.0 s], [148.1 / 296.7 s], [370.0 s], [0], [880 (all)],
  [H reserve r2], [207.8 / 372.6 s], [149.2 / 292.0 s], [372.6 s], [0], [880 (all)],
  [M reserve r1], [126.9 / 304.0 s], [148.2 / 258.3 s], [304.1 s], [930], [56],
  [M reserve r2], [124.8 / 319.5 s], [147.3 / 252.1 s], [319.6 s], [913], [53],
  [M large-dedicated], [140.1 / 332.8 s], [141.6 / 238.6 s], [332.9 s], [931], [0],
  [M plain], [111.4 / 274.1 s], [347.6 / 452.2 s], [452.2 s], [10], [439],
)

*H plain: long documents finish last.* Short requests take any free KV as soon as it appears, so a
30k-token document waits until enough KV is free at once. The median long document finishes at
about 296 s, against about 116 s for the median short request, and the last long documents set the
makespan.

*H reserve: fixing the order costs capacity.* Holding 36,864 tokens (51% of each engine's pool)
free for long documents pulls their median finish forward to about 149 s, but short requests now
share half the KV and the median short finish moves to about 211 s. Makespan gets 7% worse. On a
homogeneous layout, the reserve only trades one class's delay for the other's.

*M reserve: each engine type does the work it suits.* The TP16 engine prefills all long documents
(about 762k prompt tokens) plus about 55 short requests, and goes idle at 280–287 s. The two TP8
engines serve about 825 short requests. The long documents finish by 252–258 s, about 90 s earlier
than in H plain, and the short requests finish at nearly the same median as in H.

*M plain: the router idles two thirds of the slice.* Half of new requests are owed to the TP16
engine; when it is full, the router waits for it even though the TP8 engines have room. The TP16
engine ends up with 439 short requests plus all 24 long documents and runs until 452 s, while the
TP8 engines stop at 204 s and 218 s.

== Per-engine work (M reserve r1 vs H plain r1)

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right),
  table.header[Engine][Chips][Decode tokens][Prefill tokens][Last step][Preemptions (replayed tok)],
  [M large (TP16)], [16], [40,760], [763,200], [287.1 s], [0],
  [M small_a (TP8)], [8], [167,733], [194,389], [296.3 s], [477 (103,847)],
  [M small_b (TP8)], [8], [179,482], [196,696], [303.8 s], [453 (91,966)],
  [H large_a (TP16)], [16], [198,445], [444,723], [338.8 s], [0],
  [H large_b (TP16)], [16], [189,530], [516,579], [347.0 s], [18 (11,708)],
)

Per chip, each TP8 engine in M decoded about 69–74 tokens/s (both reserve runs) against about
34–37 tokens/s per chip on an H engine. The H engines also carry about half of the long-document
prefill each, so this ratio is an upper bound on the TP8 advantage, not a clean per-chip
efficiency measurement.

= Threats to validity <sec-validity>

- *Small sample.* Two repetitions of each main arm; the M reserve runs differ by 15.5 s
  (304.1 vs 319.6 s). The observed gain is 7.9–12.4%; the true range could be wider.
- *Unfrozen code.* The policy and manifest code changed during the campaign (block → token reserve).
  No run was hot-patched mid-arm, but the runs are not bound to a source freeze.
- *Run order.* H ran reserve then plain in rep 1 and plain then reserve in rep 2, on one deployment
  per repetition. Each M reserve run had its own fresh deployment; the M large-dedicated and M plain
  runs shared one. No warm-state effect is visible (H plain r1 and r2 differ by 0.3 s), but order
  was not randomised.
- *Makespan only.* This is a bulk-backlog measurement. Per-request latency and SLO attainment were
  not measured and would favour different policies.
- *No S baseline.* S cannot serve W4 at all, so the comparison is M against H only. On
  short-only workloads S was the best layout in earlier cohorts.
- *Workload shape.* The benefit requires a mix in which a minority of requests needs a large engine
  and the majority can run on small ones. The share of long work (24 of 904 requests, 77% of prompt
  tokens) was chosen, not swept.

= Remaining headroom and next steps

+ *Tail balance.* In both M reserve runs the TP16 engine goes idle 17–40 s before the last TP8 engine,
  while the TP8 engines preempt and replay about 900 times (≈ 200k replayed tokens). Sending the last
  short requests, or migrating late decodes, to the idle TP16 engine could close that gap.
  Migration is blocked: continuation qualification still fails with "native decision remains
  unresolved".
+ *Admission on TP8.* The TP8 engines admit more than their KV can hold, which is where the
  preemptions come from. A tighter admission threshold on small engines may cut the replay cost.
+ *Sweep the mix.* Vary the long-document share (e.g. 1%, 3%, 10% of requests) to find where M
  stops beating H.
+ *Timed cohort.* Freeze the source and repeat M reserve and H plain with ≥ 5 repetitions in
  randomised order.

= Reproduction

Code lives in the assembled workspace `~/hetero-research` at commits `8ff2585` (W4 builder,
per-instance engine settings, fallback and reserve), `45af2f1` (reserve counted in tokens) and
`67a17c9` (campaign rep offsets).

```sh
# long-document catalogue (122 GovReport records)
scripts/build_longdoc_catalogue.py govreport-train-0.parquet tokenizer.json \
    data/workloads/longdoc_govreport.json
# deploy + run under the shared-cluster lock (M, then H)
flock /tmp/hetero-cluster.lock scripts/w4_campaign.sh 1 M H
# tables in this report
scripts/summarize_w4.py w4-45af2f1-plain-r1 w4-45af2f1-plain-r2 w4-45af2f1-resv-r1 \
    w4-45af2f1-resv-r2 w4-67a17c9-resv-r2 w4-8ff2585-plain-r1 w4-8ff2585-resv-r1
```

Deploy settings: M uses `--max-model-len 16384 --instance-engine-settings '{"large":{"max_model_len":36864}}'`;
H uses `--max-model-len 36864`. Results: `results/pilot/w4-*_W4_*.json`; per-request records,
iterations and gateway traces: `results/hetero/artifacts/w4-*/`.

#table(
  columns: (auto, auto, auto),
  align: left,
  table.header[Run id][Arms][Policy],
  [`w4-45af2f1-plain-r1`, `-r2`], [H], [plain],
  [`w4-45af2f1-resv-r1`], [H, M0], [reserve],
  [`w4-45af2f1-resv-r2`], [H], [reserve],
  [`w4-67a17c9-resv-r2`], [M0], [reserve],
  [`w4-8ff2585-plain-r1`], [M0], [plain],
  [`w4-8ff2585-resv-r1`], [M0], [large-dedicated (block reserve)],
)
