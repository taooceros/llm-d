// W4: heterogeneous vs homogeneous TPU layouts on a long-document + chat mix.
// Written for Typst HTML export first; the PDF is a secondary target.
//   HTML: typst compile --features html --format html w4_longdoc_heterogeneity.typ
//   PDF:  typst compile w4_longdoc_heterogeneity.typ
// Styling that HTML export drops (text fill, block fill, box widths) goes
// through the helpers below, which emit inline CSS in HTML and native
// Typst styling in PDF.
#let title = "Heterogeneous TPU serving on a long-document + chat mix (W4)"
#set document(title: title, author: "hetero-research")
#set heading(numbering: "1.")
#show table.cell.where(y: 0): strong

#let css = ```
body { max-width: 60rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.55;
       font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #1f2328; }
h1, h2, h3 { line-height: 1.25; margin-top: 1.8em; }
h2 { border-bottom: 1px solid #d0d7de; padding-bottom: .25em; }
table { border-collapse: collapse; margin: 1em 0; font-size: .92em; display: block; overflow-x: auto; }
th, td { border: 1px solid #d0d7de; padding: .35em .7em; vertical-align: top; }
th { background: #f6f8fa; text-align: left; }
tr:nth-child(even) td { background: #fbfcfd; }
code { background: #eff1f3; padding: .1em .3em; border-radius: 4px; font-size: .9em; }
pre { background: #f6f8fa; padding: .8em 1em; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }
figure { margin: 1.5em 0; }
/* html.frame sets an inline em size; scale drawn figures to the column instead. */
svg.typst-frame { width: 100% !important; height: auto !important; }
figcaption { font-size: .9em; color: #59636e; margin-top: .5em; }
.subtitle { color: #59636e; margin-top: -.5em; }
.note { background: #f6f8fa; border-left: 4px solid #0969da; padding: .6em 1em; margin: 1em 0; }
.footnote { font-size: .88em; color: #59636e; }
.bars { display: grid; grid-template-columns: max-content 1fr; gap: .4em .8em; align-items: center; }
.bar { color: #fff; font-size: .82em; text-align: right; padding: .15em .5em; border-radius: 3px;
       white-space: nowrap; box-sizing: border-box; }
```.text

// `target` exists only when the html feature is enabled; without it this is a PDF build.
#let is-html() = "target" in dictionary(std) and std.target() == "html"

#show: doc => context if is-html() {
  html.elem("style", css)
  doc
} else {
  set page(paper: "us-letter", margin: (x: 2.2cm, y: 2cm), numbering: "1")
  set text(size: 10.5pt)
  set par(justify: true)
  show table: set text(size: 9pt)
  doc
}

#let good = rgb("#1a7f37")
#let bad = rgb("#cf222e")
#let tp16 = rgb("#0969da")
#let orange = rgb("#e16f24")
#let div(class, body) = html.elem("div", attrs: (class: class), body)
// Drawn figures: inline SVG in HTML, native layout in PDF.
#let diagram(body) = context {
  let drawn = block(width: 16cm, align(left, { set par(justify: false); body }))
  if is-html() { html.frame(drawn) } else { drawn }
}
#let node(body, stroke: luma(140), fill: white) = block(stroke: stroke, fill: fill, inset: 6pt,
  radius: 3pt, width: 100%, text(size: 8.5pt, body))
#let colored(colour, body) = context if is-html() {
  html.elem("span", attrs: (style: "color:" + colour.to-hex()), body)
} else { text(fill: colour, body) }
#let note(body) = context if is-html() { div("note", body) } else {
  block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%, body)
}
#let footnote-text(body) = context if is-html() { div("footnote", body) } else {
  text(size: 8.5pt, body)
}

// Horizontal bar chart of makespans; each row: (label, seconds, colour).
#let bars(rows) = context if is-html() {
  let longest = calc.max(..rows.map(r => r.at(1)))
  div("bars", rows.map(((label, value, colour)) => (
    html.elem("span", label),
    html.elem("div", attrs: (class: "bar", style: "width:" + str(calc.round(value / longest * 100, digits: 1))
      + "%;background:" + colour.to-hex()), str(value) + " s"),
  )).flatten().join())
} else {
  grid(
    columns: (auto, 1fr), column-gutter: 8pt, row-gutter: 5pt, align: (right + horizon, left + horizon),
    ..rows.map(((label, value, colour)) => (
      text(size: 9pt, label),
      box(width: value * 0.9pt, height: 11pt, fill: colour, inset: (x: 4pt),
          align(right + horizon, text(size: 8pt, fill: white, str(value) + " s"))),
    )).flatten(),
  )
}

#context if is-html() {
  html.elem("h1", title)
  html.elem("p", attrs: (class: "subtitle"))[Gemma-4 31B on TPU v5e 4×8 (32 chips) · pilot runs of
    2026-09-30 · branch `research/heterogeneous-tpu-20260918`]
} else {
  align(center)[
    #text(size: 16pt, weight: "bold", title)
    #v(2pt)
    Gemma-4 31B on TPU v5e 4×8 (32 chips) · pilot runs of 2026-09-30 · branch `research/heterogeneous-tpu-20260918`
  ]
}

= Summary

On a bulk backlog of 880 ShareGPT chat requests mixed with 24 real government reports whose context
is too long for a TP8 engine, the heterogeneous layout *M* (TP8 + TP8 + TP16) finished
#colored(good)[*10.1% sooner*] than the best homogeneous layout *H* (2 × TP16): a mean makespan of
311.9 s against 347.1 s (7.9% and 12.4% in the two repetitions). Mean useful throughput rose from
1118 to 1245 tokens/s on the same 32 chips.

The gain depends on the router. Under the pre-existing routing policy, M was
#colored(bad)[*30% slower*] than H (452.2 s): the router sends half of all new requests to the
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

W4 is a closed backlog: all 904 requests are queued at time 0 and the gateway dispatches them in
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
Long request _k_ sits at a seeded position inside the _k_-th of 24 equal slices of the arrival
order (indices 30, 70, 107, …, 876), so long work arrives throughout the backlog. In total the
workload is 948,975 prompt tokens, 727,303 of them (77%) from the 24 long documents. Manifest
SHA-256 prefix `e380318ae33a`.

== Routing policies <sec-policies>

Every run uses the same router; the two policies differ in three settings. Both are blind to output
length: the router knows each request's prompt length and every engine's free KV blocks, and
charges a new request its prompt plus one KV block.

#table(
  columns: (auto, auto, auto, 1fr),
  align: (left, center, center, left),
  table.header[Field][plain][reserve][Effect],
  [`work_conserving_fallback`], [off], [on],
    [if the preferred pool has no admissible room, try the other pool instead of waiting],
  [`long_reserve_tokens`], [0], [36,864],
    [short requests may not use the last 36,864 KV tokens on a TP16 engine; long requests may],
  [`small_prompt_token_limit`], [4,096], [16,384], [prompts above this count as long for the reserve],
)

The name "reserve" covers both changes. @fig-flow shows where they act in the routing decision,
@fig-kv shows the admission check the reserve changes, and @fig-quota and @fig-starve show what
each change fixes.

#figure(
  diagram({
    let arrow(label) = align(center, text(size: 8pt, fill: luma(90))[#label])
    let side(label) = align(center + horizon, text(size: 8pt, fill: luma(90))[#label])
    grid(
      columns: (7.4cm, 1.3cm, 6.3cm), row-gutter: 3pt, align: left + horizon,
      node[*Request arrives.* Router knows prompt length _P_; output length is hidden.], [], [],
      arrow[↓], [], [],
      node[Charge _P_ + 1 KV block.], [], [],
      arrow[↓], [], [],
      node[Does it fit a TP8 engine (_P_ + 1 block ≤ 16,384)?], side[no →],
        node(fill: tp16.lighten(85%))[TP16 pool only. Every long document takes this path.],
      arrow[yes ↓], [], [],
      node[Is TP16 owed this request? True while TP16 has taken fewer than 50% of all dispatched
        requests (M only; H has no TP8 pool).], side[yes →], node[Try TP16 first, then TP8.],
      arrow[no ↓ #h(4pt) try TP8 first, then TP16], [], [],
      node[Does any engine in the first pool pass the admission check (@fig-kv)?], side[yes →],
        node[Dispatch to the least-loaded engine that passed.],
      arrow[no ↓], [], [],
      grid.cell(colspan: 3, grid(columns: (1fr, 1fr), column-gutter: 8pt,
        node(stroke: bad, fill: bad.lighten(90%))[*plain:* wait one poll (still holding a dispatch
          slot), then retry from the top.],
        node(stroke: good, fill: good.lighten(90%))[*reserve:* try the second pool; wait only if it
          has no room either.],
      )),
    )
  }),
  caption: [Routing decision for one new request, as implemented in `select_initial`. The two
    policies differ only in the last step and in the admission check.],
) <fig-flow>

#figure(
  diagram({
    // One TP16 engine: 282 blocks of 256 tokens. Snapshot: 90 blocks in use, 192 free.
    let u = 0.053cm
    let seg(n, fill, label, light: false) = box(width: n * u, height: 0.9cm, fill: fill,
      stroke: 0.5pt + white, align(center + horizon,
        text(size: 7.5pt, fill: if light { black } else { white }, label)))
    let span(start, n, fill, label) = pad(left: start * u, stack(dir: ttb, spacing: 2pt,
      box(width: n * u, height: 5pt, fill: fill), text(size: 8pt, label)))
    set text(size: 8.5pt)
    stack(dir: ttb, spacing: 6pt,
      text(weight: "bold")[One TP16 engine: 282 blocks × 256 tokens (snapshot with 192 blocks free)],
      stack(dir: ltr,
        seg(90, luma(150), [in use: 90]),
        seg(32, white, [32], light: true),
        seg(144, orange, [long-request reserve: 144 blocks = 36,864 tok]),
        seg(16, luma(60), [16]),
      ),
      span(90, 32, good, [short request may use: 192 − 144 − 16 = *32 blocks*]),
      span(90, 176, tp16, [long request may use: 192 − 16 = *176 blocks*]),
      v(8pt),
      [The reserve is a threshold on the free-block count, not a fixed set of blocks. A 30,000-token
        document needs ⌈30,256 / 256⌉ = 119 blocks and is admitted (119 ≤ 176). A 500-token chat
        request needs 3 blocks and is admitted while at least 3 of the 32 remain. The dark 16
        blocks are headroom that applies to every request on a TP16 engine in both policies. TP8
        engines (412 blocks × 64 tokens) have neither headroom nor reserve.],
    )
  }),
  caption: [Admission check on a TP16 engine under the reserve policy. Under plain, the orange
    region does not exist and both request types may use 176 blocks.],
) <fig-kv>

#figure(
  diagram({
    let cell(fill, label, dark: true) = box(width: 0.95cm, height: 0.75cm, fill: fill,
      stroke: 1pt + white, radius: 2pt, align(center + horizon,
        text(size: 7pt, fill: if dark { white } else { black }, label)))
    let d16(l) = cell(tp16, l)
    let d8(l) = cell(good, l)
    let w(l) = cell(luma(215), l, dark: false)
    let row(label, ..cells) = grid(columns: (2.6cm, auto), align: left + horizon,
      text(size: 8.5pt, label), stack(dir: ltr, ..cells.pos()))
    set text(size: 8.5pt)
    stack(dir: ttb, spacing: 8pt,
      [Arrival order →. Scenario: the TP16 engine has no room for a short request.],
      row([*plain*], d16[r1], d8[r2], w[r3 ⧗], w[r4 ⧗], w[r5 ⧗], w[r6 ⧗], w[r7 ⧗], w[r8 ⧗], w[r9 ⧗],
        w[r10 ⧗], w[r11 ⧗]),
      pad(left: 2.6cm)[r3 is owed to TP16 and waits. The quota counter only moves on dispatch, so r4,
        r5, … are owed to TP16 too and wait behind it. TP8 gets one request per request TP16
        accepts, however idle it is.],
      row([*reserve*], d8[r1 ↪], d8[r2 ↪], d8[r3 ↪], d8[r4 ↪], d16[doc], d8[r5 ↪], d8[r6 ↪],
        d8[r7 ↪], d8[r8 ↪], d8[r9 ↪], d8[r10 ↪]),
      pad(left: 2.6cm)[Each short request is owed to TP16, finds no room outside the reserve, and
        falls back (↪) to TP8. A long document ("doc") may use the reserve and goes to TP16.],
      grid(columns: 6, column-gutter: 6pt, align: horizon,
        cell(tp16, []), [to TP16], cell(good, []), [to TP8], cell(luma(215), [⧗], dark: false),
        [waiting]),
    )
  }),
  caption: [What work-conserving fallback fixes in M (schematic). Measured in the M runs: under
    plain, 439 of 880 short requests went to TP16 (dispatch alternated almost exactly) and the TP8
    engines went idle at 204 s and 218 s. Under reserve, 803 short requests were owed to TP16 and
    fell back to TP8; only 56 ran on TP16.],
) <fig-quota>

#figure(
  diagram({
    let u = 0.053cm
    let seg(n, fill, label, light: false) = box(width: n * u, height: 0.8cm, fill: fill,
      stroke: 0.5pt + white, align(center + horizon,
        text(size: 7.5pt, fill: if light { black } else { white }, label)))
    let row(label, body, note) = grid(columns: (2.2cm, auto), row-gutter: 3pt, align: left + horizon,
      text(size: 8.5pt, label), body, [], text(size: 8pt, note))
    set text(size: 8.5pt)
    stack(dir: ttb, spacing: 10pt,
      [One H engine (TP16, 282 blocks) while a 30,000-token document (119 blocks) is waiting.],
      row([*plain*], stack(dir: ltr, seg(254, luma(150), [short requests: 254 blocks]),
        seg(12, white, [12], light: true), seg(16, luma(60), [])),
        [Short requests finish a few blocks at a time and a waiting short request takes the freed
          blocks at once, so 119 free blocks rarely appear together. Measured: median document
          finished at 297.9 s, median short request at 113.9 s.]),
      row([*reserve*], stack(dir: ltr, seg(122, luma(150), [short: 122 blocks]),
        seg(144, orange, [kept free: 144 blocks]), seg(16, luma(60), [])),
        [Short requests stop at 160 free blocks, so a document can start as soon as it arrives.
          Measured: median document 148.1 s, but short requests now share about half the KV and
          their median moved to 214.2 s.]),
    )
  }),
  caption: [What the long-request reserve fixes, and what it costs, on the homogeneous layout
    (schematic occupancy; measured times from H r1). H has no second pool, so only the reserve
    acts there.],
) <fig-starve>

In short: the reserve makes room for long documents by taking room from short requests. On H that
is a trade (makespan +7%); on M the short requests it displaces have somewhere else to go, the two
TP8 engines, and fallback sends them there.

= Results

All 8 runs completed 904/904 requests with exact output budgets and one-to-one iteration capture.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, left, right, right, right, right),
  table.header[Layout][Policy][Makespan rep 1][Makespan rep 2][Throughput (mean)][vs best H],
  [H], [plain], [347.2 s], [346.9 s], [1118 tok/s], [baseline],
  [H], [reserve], [370.0 s], [372.6 s], [1045 tok/s], colored(bad)[+7.0%],
  [*M*], [*reserve*], [*304.1 s*], [*319.6 s*], [*1245 tok/s*], colored(good)[*−10.1%*],
  [M], [large-dedicated#super[†]], [332.9 s], [–], [1166 tok/s], colored(good)[−4.1%],
  [M], [plain], [452.2 s], [–], [858 tok/s], colored(bad)[+30.3%],
)
#footnote-text[† An earlier version of the reserve was counted in blocks; with 256-token blocks it
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
