// Reconfiguration break-even for bulk inference on one TPU v5e slice.
// Written for Typst HTML export first; the PDF is a secondary target.
//   HTML: typst compile --features html --format html reconfiguration_breakeven.typ
//   PDF:  typst compile reconfiguration_breakeven.typ
// Same helpers and conventions as w4_longdoc_heterogeneity.typ.
#let title = "When does reconfiguring the TPU slice pay for itself in bulk inference?"
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
figure img { width: 100%; height: auto; }
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
// HTML export drops equations; render each one as an inline SVG frame instead.
#show math.equation: it => context if is-html() {
  if it.block { html.elem("div", attrs: (style: "text-align:center;margin:.8em 0"), html.frame(it)) }
  else { box(html.frame(it)) }
} else { it }

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
#let div(class, body) = html.elem("div", attrs: (class: class), body)
// Hand-written SVG kept inline so the report stays one self-contained file.
// Font list: browsers pick a system sans; Typst's PDF build falls back to its bundled fonts.
#let svg-fig(source) = image(bytes(source.text), format: "svg", width: 100%)
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
  html.elem("p", attrs: (class: "subtitle"))[Gemma-4 31B on TPU v5e 4×8 (32 chips) · runs of
    2026-10-01 · branch `research/heterogeneous-tpu-20260918`]
} else {
  align(center)[
    #text(size: 16pt, weight: "bold", title)
    #v(2pt)
    Gemma-4 31B on TPU v5e 4×8 (32 chips) · runs of 2026-10-01 · branch `research/heterogeneous-tpu-20260918`
  ]
}

= Summary

Switching the slice between layouts H (2×TP16), S (4×TP8) and M (TP16 + 2×TP8) costs
*about 440 s of lost slice time* per transition with today's full redeploy (measured: 445 s
S→H, 436 s H→M). Two thirds of that is loading the 62.5 GB checkpoint from the gcsfuse
`/models` mount (236–238 s per engine); XLA compile is only 64 s because the compilation cache
is already persistent.

The per-phase gains reconfiguration can buy are small next to that cost. The largest measured
gap is chat (W0): S finishes in 343 s against 497 s on H and 468 s on M. On the long-context
workloads the best layout beats the worst by only 6–7% (W3, W5). With the measured cost, a
switch pays only once the phase it serves is *2.0M output tokens or more* (W0 on H→S, 2.7× a W0
phase), and 4.6–16M tokens for every other transition. At the size of a single manifest (0.4–1.1M
tokens), no switch pays.

#note[
  *Status.* Measured so far: per-layout throughput on W0 and W5 (2 repetitions each) and two
  instrumented full transitions. End-to-end mixed jobs, partial reconfiguration and warm start
  are prepared (code in `~/hetero-research`) but not yet run on the cluster; their sections below
  state predictions only, labelled as such.
]

= Setup

All runs use the same 32 chips, the same Gemma-4 31B checkpoint (`/models/gemma-4-31b`) and the
same vLLM + tpu-inference stack behind the Python gateway. Every request runs with `ignore_eos`
and `max_tokens` fixed to its reference output length, so a workload's useful output is identical
in every arm. Routing is output-length-blind.

#table(
  columns: (auto, auto, auto, auto, 1fr),
  align: (left, right, right, right, left),
  table.header[Workload][Requests][Output tokens][Prompt tokens][Content],
  [W0 chat], [1,763], [721,595], [429,858], [historical ShareGPT (prompt, output) demands],
  [W3 reasoning], [160], [1,079,823], [48,465], [R1 reasoning traces (median output 5.9k)],
  [W4 long-doc + chat], [904], [387,975], [948,975], [24 GovReport documents that only fit TP16, plus 880 chats],
  [W5 translation], [250], [982,899], [856,779], [whole TED talks translated to French/German (median prompt 3.5k, output 4.0k)],
)

W5 is new in this study: prompts are full English TED transcripts and the output length is the
length of the talk's real published translation (IWSLT `ted_talks_iwslt`, WIT3 2015-05-30).

= Cost of one reconfiguration <sec-cost>

A transition is measured from the last request of the old phase finishing to the new layout
reaching steady throughput. The campaign script (`scripts/reconfig_measure.sh`) stamps every
boundary; each engine worker records wall-clock start and end of its start-up phases
(`HeteroTPUWorker`), and the first run on the new layout gives the ramp.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 230" font-family="Helvetica, Arial, 'DejaVu Sans', sans-serif" font-size="12" fill="#1f2328">
<text x="20" y="20" font-weight="bold">S → H full reconfiguration: last request of phase 1 drained (0 s) → new layout at steady throughput (444 s)</text>
<rect x="20.0" y="34" width="11.7" height="40" fill="#afb8c1" stroke="#fff" stroke-width="1"/>
<rect x="31.7" y="34" width="32.1" height="40" fill="#8c959f" stroke="#fff" stroke-width="1"/>
<rect x="63.7" y="34" width="29.2" height="40" fill="#d0d7de" stroke="#fff" stroke-width="1"/>
<rect x="92.9" y="34" width="35.5" height="40" fill="#54aeff" stroke="#fff" stroke-width="1"/>
<rect x="128.4" y="34" width="5.8" height="40" fill="#0969da" stroke="#fff" stroke-width="1"/>
<rect x="134.2" y="34" width="385.3" height="40" fill="#e16f24" stroke="#fff" stroke-width="1"/>
<text x="326.9" y="58" text-anchor="middle" fill="#fff" font-size="11.5">load weights (gcsfuse) 238 s</text>
<rect x="519.5" y="34" width="42.4" height="40" fill="#bf8700" stroke="#fff" stroke-width="1"/>
<rect x="561.9" y="34" width="104.8" height="40" fill="#8250df" stroke="#fff" stroke-width="1"/>
<text x="614.4" y="58" text-anchor="middle" fill="#fff" font-size="11.5">compile + warm-up 65 s</text>
<rect x="666.8" y="34" width="34.7" height="40" fill="#6e7781" stroke="#fff" stroke-width="1"/>
<rect x="701.4" y="34" width="23.5" height="40" fill="#d0d7de" stroke="#fff" stroke-width="1"/>
<rect x="724.9" y="34" width="15.1" height="40" fill="#1a7f37" stroke="#fff" stroke-width="1"/>
<line x1="25.8" y1="74" x2="25.8" y2="86" stroke="#8c959f" stroke-width="0.8"/>
<text x="25.8" y="96" text-anchor="start" font-size="11">results 7 s</text>
<line x1="47.7" y1="74" x2="47.7" y2="102" stroke="#8c959f" stroke-width="0.8"/>
<text x="47.7" y="112" text-anchor="start" font-size="11">teardown 20 s</text>
<line x1="78.3" y1="74" x2="78.3" y2="118" stroke="#8c959f" stroke-width="0.8"/>
<text x="78.3" y="128" text-anchor="start" font-size="11">job start 18 s</text>
<line x1="110.6" y1="74" x2="110.6" y2="86" stroke="#8c959f" stroke-width="0.8"/>
<text x="110.6" y="96" text-anchor="start" font-size="11">Ray workers 22 s</text>
<line x1="131.3" y1="74" x2="131.3" y2="102" stroke="#8c959f" stroke-width="0.8"/>
<text x="131.3" y="112" text-anchor="start" font-size="11">TPU init 4 s</text>
<line x1="540.7" y1="74" x2="540.7" y2="118" stroke="#8c959f" stroke-width="0.8"/>
<text x="540.7" y="128" text-anchor="end" font-size="11">KV alloc 26 s</text>
<line x1="684.1" y1="74" x2="684.1" y2="86" stroke="#8c959f" stroke-width="0.8"/>
<text x="684.1" y="96" text-anchor="end" font-size="11">endpoints, registry, gateway 21 s</text>
<line x1="713.2" y1="74" x2="713.2" y2="102" stroke="#8c959f" stroke-width="0.8"/>
<text x="713.2" y="112" text-anchor="end" font-size="11">1st dispatch 14 s</text>
<line x1="732.5" y1="74" x2="732.5" y2="118" stroke="#8c959f" stroke-width="0.8"/>
<text x="732.5" y="128" text-anchor="end" font-size="11">ramp 9 s</text>
<line x1="20.0" y1="150" x2="20.0" y2="156" stroke="#59636e"/><text x="20.0" y="170" text-anchor="middle" font-size="10.5" fill="#59636e">0</text>
<line x1="101.0" y1="150" x2="101.0" y2="156" stroke="#59636e"/><text x="101.0" y="170" text-anchor="middle" font-size="10.5" fill="#59636e">50</text>
<line x1="182.0" y1="150" x2="182.0" y2="156" stroke="#59636e"/><text x="182.0" y="170" text-anchor="middle" font-size="10.5" fill="#59636e">100</text>
<line x1="263.0" y1="150" x2="263.0" y2="156" stroke="#59636e"/><text x="263.0" y="170" text-anchor="middle" font-size="10.5" fill="#59636e">150</text>
<line x1="344.0" y1="150" x2="344.0" y2="156" stroke="#59636e"/><text x="344.0" y="170" text-anchor="middle" font-size="10.5" fill="#59636e">200</text>
<line x1="425.0" y1="150" x2="425.0" y2="156" stroke="#59636e"/><text x="425.0" y="170" text-anchor="middle" font-size="10.5" fill="#59636e">250</text>
<line x1="506.0" y1="150" x2="506.0" y2="156" stroke="#59636e"/><text x="506.0" y="170" text-anchor="middle" font-size="10.5" fill="#59636e">300</text>
<line x1="587.1" y1="150" x2="587.1" y2="156" stroke="#59636e"/><text x="587.1" y="170" text-anchor="middle" font-size="10.5" fill="#59636e">350</text>
<line x1="668.1" y1="150" x2="668.1" y2="156" stroke="#59636e"/><text x="668.1" y="170" text-anchor="middle" font-size="10.5" fill="#59636e">400</text>
<line x1="20" y1="150" x2="740" y2="150" stroke="#59636e"/>
<text x="740" y="186" text-anchor="end" font-size="11" fill="#59636e">seconds since phase 1 drained</text>
<path d="M92.9,200 v6 h573.9 v-6" fill="none" stroke="#1f2328"/><text x="379.8" y="222" text-anchor="middle" font-size="11.5">engine start-up 354 s (slowest engine; all engines start in parallel)</text>
</svg>
```),
  caption: [Measured S→H transition, one wall clock. Engines start in parallel; the bar shows the
    slowest one. Ramp = output the first 120 s lost against the run's median rate, in seconds of
    steady output.],
) <fig-timeline>

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right, right),
  table.header[Transition][Teardown#super[a]][Deploy][of which load weights][compile + warm-up][1st dispatch][Ramp][Cost C],
  [S→H], [27.0 s], [376.2 s], [237.8 s], [64.7 s], [14.5 s], [9.3 s], [*445.0 s*],
  [H→M], [24.0 s], [374.8 s], [236.7 s], [63.9 s], [14.1 s], [4.5 s], [*435.5 s*],
  [(cold) →S], [–], [378.3 s], [233.6 s], [63.4 s], [14.6 s], [5.9 s], [–],
)
#footnote-text[a. From the last finished request: 7 s of result writing by the finished run, then
drain, gateway-down and release, each a separate cluster job (about 6 s each, almost all job
submission). The remaining gap to the deploy (about 18 s) is the deploy job's own start-up.]

Inside the deploy, the slowest engine spends 22 s starting Ray worker processes, 4 s initialising
TPU devices, *236–238 s loading weights*, 26 s profiling and allocating the KV cache and 64 s
compiling and warming up; registry, endpoints and gateway roll-out add about 21 s. The three
deploys agree to within 4 s, so C is not a sampling artefact. Of the 445 s, about 360 s is engine
start-up that any redeploy pays; about 70 s is orchestration overhead (job submissions, result
writing) that a purpose-built controller could remove.

= Throughput per layout

#table(
  columns: (auto, auto, auto, auto, 1fr),
  align: (left, right, right, right, left),
  table.header[Workload][H][S][M][Source],
  [W0], [497.2 s \ (500.0, 494.4)], [*343.3 s* \ (352.4, 334.2)], [467.7 s \ (482.5, 452.9)],
    [measured, `rc-bab4f9e-*`, blind policy; simulator said 516 / 330 / 500],
  [W3], [*1284 s*], [1354 s], [1382 s], [measured `w3-c7edd5a-r1`, 1 repetition],
  [W4], [347.1 s \ (347.2, 346.9)], [infeasible], [*311.9 s* \ (304.1, 319.6)],
    [measured W4 campaign; M with the reserve policy, H plain],
  [W5], [*1372.4 s* \ (1375.1, 1369.7)], [1450.7 s \ (1455.4, 1446.1)], [1409.6 s \ (1395.3, 1423.9)],
    [measured, `rc-bab4f9e-*`, blind; simulator said 1374 / 1508 / 1374],
)

Every value is a makespan for the whole manifest; per-repetition values in parentheses. The
simulator was within 5% everywhere except W0 on M (+7%) and W5 on S (+4%), and it ranked the
layouts correctly on both new workloads.

= Break-even phase size

Switching from layout A to B for a phase pays when the time it saves exceeds C:

$ N / r_A - N / r_B > C quad <==> quad N > C dot r_A r_B / (r_B - r_A) $

With makespans $T$ for one manifest of $N_0$ tokens, the break-even multiple of that manifest is
simply $k = C / (T_A - T_B)$. The table uses $C = 420$ s (midpoint) and gives the range over both
repetitions of each layout and $C in [395, 445]$ s.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, left, right, right, right, right),
  table.header[Phase][Switch][Saved per manifest][Break-even manifests][Break-even output tokens][≈ requests],
  [W0], [H → S], [153.9 s], [2.7 (2.4–3.1)], [*2.0M* (1.7–2.3M)], [4,800],
  [W0], [M → S], [124.4 s], [3.4 (2.7–4.4)], [2.4M (1.9–3.2M)], [6,000],
  [W3], [M → H], [98.0 s], [4.3 (4.0–4.5)], [4.6M (4.4–4.9M)], [690],
  [W4], [H → M], [35.2 s], [11.9 (9.2–16.3)], [4.6M (3.6–6.3M)], [10,800],
  [W5], [S → H], [78.3 s], [5.4 (4.6–6.3)], [5.3M (4.5–6.2M)], [1,340],
  [W3], [S → H], [70.0 s], [6.0 (5.6–6.4)], [6.5M (6.1–6.9M)], [960],
  [W5], [S → M], [41.2 s], [10.2 (6.6–20)], [10.0M (6.5–19.7M)], [2,550],
  [W0], [H → M], [29.5 s], [14.2 (8.4–37)], [10.3M (6.1–27M)], [25,100],
  [W5], [M → H], [37.2 s], [11.3 (7.3–22)], [11.1M (7.2–21.7M)], [2,820],
  [W3], [M → S], [28.0 s], [15.0 (14.1–15.9)], [16.2M (15.2–17.2M)], [2,400],
)

The hypothesis predicted 10–35% per-phase gains and break-even near 10⁶ tokens. Only chat
reaches that gain (31% S over H); the long-context phases differ by 3–7%. Break-even is therefore
2–5M tokens for the useful transitions, above the predicted 10⁶ but below the 10⁷ falsification
line, and the W3/W5 M→S and H↔M switches sit at or beyond 10⁷.

A whole job only gains if *every* switch pays: a job that changes layout once must make the
gain on its second phase exceed C, while the best static layout already captures whichever phase
dominates. With today's C that requires multi-hour phases (a 2M-token chat phase is about 16 min
on S; a 5M-token W5 phase is about 2 hours on H).

= End-to-end mixed jobs (prepared, not yet run)

Chosen from the break-even table (`results/reconfig/breakeven_ranges.json`):

#table(
  columns: (auto, auto, auto, 1fr),
  align: (left, left, left, left),
  table.header[Mix][Phases][Arms][Prediction from measured phases (not a result)],
  [A (should benefit)], [W4, then W0 ×4 (3.27M tokens)], [static M; M → S full; M → S partial],
    [static M ≈ 2,183 s#super[b]; full redeploy ≈ 2,125 s (−2.7%); partial ≈ 1,900 s if the kept TP8 pair serves at half S's rate during the boot],
  [C (control)], [W4, then W5], [static M; M → H full],
    [static M ≈ 1,722 s; M → H ≈ 2,124 s (+23%): one W5 phase saves 37 s against C ≈ 440 s],
)
#footnote-text[b. Uses W0 on M from the blind runs (467.7 s); in the mix M runs with long-context
TP16 settings and the reserve policy, which the static arm will measure directly.]

Mix A's composite manifest repeats the frozen W0 manifest four times under new request ids
(prefix caching is off, so repeats cost the same as the first pass). Runner:
`scripts/reconfig_mix.sh static:M:MIX_W4_W0x4 reconf:M:W4:S:W0x4`, two repetitions per arm in
alternating order.

= Partial reconfiguration and warm start (prepared, not yet run)

*Partial.* M and S share their two TP8 engines chip for chip, and M's TP16 engine sits on the same
hosts as H's second engine. A partial M→S switch keeps `small_a`/`small_b` serving phase 2 while
the TP16 engine is released and `small_c`/`small_d` start, then joins them to the gateway. It
cannot shorten the 355 s start-up (engines already start in parallel); it only keeps half the
slice busy during it. If the kept pair runs at half of S's rate, the effective cost drops to about
$355 dot 0.5 + 30 approx 210$ s, which would halve every break-even size for S↔M (W0 M→S: about
1.2M tokens). S↔H shares no engine, so it gets nothing. Status: driver under implementation
(`hetero/transition.py`), not measured.

*Warm start.* The compile cache is already persistent (`/models/jax_cache`), so the remaining
lever is weights. Each host has 175 GB of free tmpfs; staging the checkpoint into `/dev/shm`
(`scripts/reconfig_warm_stage.sh`) and deploying with `--model /dev/shm/gemma-4-31b` would replace
the gcsfuse read. If load time fell from 237 s to the order of 30 s, C would drop to about 230 s
with today's orchestration and about 160 s without it, roughly halving all break-even sizes.
Status: not measured.

= Threats to validity

- *Two repetitions.* W0 on M varies by 30 s between repetitions (482.5 vs 452.9 s); break-even
  ranges above include that spread. W3 values are single runs from an earlier campaign.
- *Orchestration overhead is in C.* About 70 s of the 440 s is job submission and result
  writing. A controller that calls the deploy API directly would pay about 370 s.
- *Routing is fixed per arm*, length-blind. Length-aware scheduling would shift the per-layout
  gaps (simulated 3–13%) and therefore break-even; it is not credited to reconfiguration here.
- *Manifest size.* Real bulk jobs are larger than one manifest; break-even is reported as a
  multiple so it can be applied to any phase size with the same request mix.
- *Unfrozen pilots.* None of these runs is bound to a source freeze.

= Reproduction

Workspace `~/hetero-research`: commits `22e8b6f`, `bab4f9e` (W5 workload, worker phase clocks,
campaign), `d83fdc9` (timeline analysis), `9c4e1b9` (composite manifests, mix runner, warm-start
staging).

```sh
# W5 catalogue (TED talks; Gemma tokenizer)
scripts/build_translation_catalogue.py <xml-20150616 dir> tokenizer.json data/workloads/translation_ted.json
# per-layout throughput + instrumented transitions (S, then H, then M), under the cluster lock
flock /tmp/hetero-cluster.lock scripts/reconfig_measure.sh S H M
# transition costs and break-even
scripts/reconfig_timeline.py bab4f9e --json results/reconfig/timeline-bab4f9e.json
scripts/reconfig_breakeven.py --table results/reconfig/throughput_table_measured.json --cost 395 440
```

Evidence: `results/reconfig/` (timeline JSONL, deploy outputs with per-worker phase clocks,
throughput tables, break-even JSON) and `results/hetero/artifacts/rc-bab4f9e-*` (per-request
records and per-step iterations).
