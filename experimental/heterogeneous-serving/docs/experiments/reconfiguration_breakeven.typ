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
    2026-10-01/02 · branch `research/heterogeneous-tpu-20260918`]
} else {
  align(center)[
    #text(size: 16pt, weight: "bold", title)
    #v(2pt)
    Gemma-4 31B on TPU v5e 4×8 (32 chips) · runs of 2026-10-01/02 · branch `research/heterogeneous-tpu-20260918`
  ]
}

= Summary

- *A full reconfiguration costs 407–445 s of lost slice time* (six measured transitions, median
  419 s). The engines' start-up is 355 s of it, and 237 s of that is loading weights.
  Staging the checkpoint in host RAM changes nothing (load 234–239 s from tmpfs vs 234–237 s from
  gcsfuse): the time is spent processing weights on the host, not reading them. The XLA cache is
  already persistent, so compile is only 64 s.
- *Per-phase gains are small next to that.* Only chat shows a large gap (S finishes W0 31% sooner
  than H); on the long-context workloads the best layout beats the worst by 3–7%. With a full
  redeploy, a switch pays only for phases of *2.0M output tokens or more* (W0, H→S), and
  4.6–16M tokens for every other transition. At single-manifest size (0.4–1.1M tokens) nothing pays.
- *Measured end to end*, a 3.27M-token job (W4 long documents, then four W0 chat backlogs) took
  1,755 s on a static M layout, *1,962 s (+11.8%) with a full M→S redeploy* and
  #colored(good)[*1,723 s (−1.8%) with a partial M→S reconfiguration*], which keeps the two shared
  TP8 engines serving while the other half restarts. The control job (W4 then W5, M→H) lost 24%,
  as predicted.
- *Partial reconfiguration halves the cost*: effective C falls from 413 s to 186 s because the
  kept TP8 pair delivers 67% of the full slice's chat throughput during the 363 s restart. That
  moves the break-even for the W4→chat job from 6.4 to 3.7 W0 backlogs (4.7M → 2.7M chat tokens).
  It only exists where layouts share engines chip for chip (S↔M, H↔M), and it needed one bug fix
  after two of four cluster attempts failed (@sec-validity).

The hypothesis (gains of 10–35%, break-even near 10⁶ tokens) is partly falsified: break-even is
2–5M tokens for the useful transitions, under the 10⁷ falsification line, and reconfiguration
beat the best static layout only with partial reconfiguration and only by 1.8%.

= Setup

All runs use the same 32 chips, the same Gemma-4 31B checkpoint (`/models/gemma-4-31b`) and the
same vLLM + tpu-inference stack behind the Python gateway. Every request runs with `ignore_eos`
and `max_tokens` fixed to its reference output length, so useful output is identical in every arm
of a comparison. Routing is output-length-blind.

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
Multi-phase jobs are composite manifests (`MIX_W4_W0x4`, `MIX_W4_W5`, `W0x4`) built from the frozen
parts; W0×4 repeats the W0 prompts under new request ids, and prefix caching is off, so a repeat
costs as much as the first pass.

= Cost of one reconfiguration <sec-cost>

A transition is measured from the last request of the old phase finishing to the new layout
serving at steady throughput. The campaign scripts stamp every boundary, each engine worker
records the wall-clock start and end of its start-up phases (`HeteroTPUWorker`), and the first run
on the new layout gives the ramp.

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
  caption: [Measured S→H transition, on one wall clock. Engines start in parallel; the bar shows
    the slowest one. Ramp = output the first 120 s lost against the run's median rate, in seconds
    of steady output.],
) <fig-timeline>

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right, right),
  table.header[Transition][Teardown#super[a]][Deploy][of which load weights][compile + warm-up][1st dispatch][Ramp][Cost C],
  [S→H], [27.0 s], [376.2 s], [237.8 s], [64.7 s], [14.5 s], [9.3 s], [*445.0 s*],
  [H→M], [24.0 s], [374.8 s], [236.7 s], [63.9 s], [14.1 s], [4.5 s], [*435.5 s*],
  [M→S (mix A, rep 1 / 2)], [25.4 / 24.0 s], [372.6 / 374.3 s], [237.2 / 236.6 s], [62.2 / 62.3 s], [15.6 / 14.6 s], [5.6 / 6.2 s], [*419.2 / 419.1 s*],
  [M→H (control, rep 1 / 2)], [25.9 / 26.1 s], [362.7 / 364.7 s], [237.4 / 237.9 s], [54.5 / 54.4 s], [14.3 / 14.7 s], [4.6 / 4.7 s], [*407.5 / 410.2 s*],
  [(cold) →S], [–], [378.3 s], [233.6 s], [63.4 s], [14.6 s], [5.9 s], [–],
)
#footnote-text[a. From the last finished request: 7 s of result writing by the finished run, then
drain, gateway-down and release, each a separate cluster job (about 6 s each, almost all job
submission). The remaining gap before the deploy (about 18 s) is the deploy job's own start-up.
In the mix runs the deploy column includes the deploy job's start-up. The end-to-end job times in
@sec-e2e already contain the ramp, so they use the gap without it (412.9–413.6 s for M→S).]

Inside the deploy, the slowest engine spends 22 s starting Ray worker processes, 4 s initialising
TPU devices, *236–238 s loading weights*, 26 s profiling and allocating the KV cache and 64 s
compiling and warming up; registry, endpoints and gateway roll-out add about 21 s. All 16 deploys
took 363–380 s. About 360 s of C is engine start-up that any redeploy pays; about 50–80 s is
orchestration overhead (job submissions, result writing) that a purpose-built controller could
remove.

*Warm start does not help.* Copying the 62.5 GB checkpoint into `/dev/shm` on all eight hosts
takes 25 s per host (about 2.5 GB/s), so the gcsfuse mount is not the bottleneck. Deploying S from
the tmpfs copy gave the same load time as from gcsfuse:

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  table.header[Weights from][load_model (8 workers)][compile + warm-up][Engine init][W0 makespan after deploy],
  [gcsfuse `/models`], [233.6–236.9 s], [63.4–64.0 s], [349.7–351.8 s], [352.4 s],
  [tmpfs `/dev/shm`], [234.3–238.8 s], [61.3–64.0 s], [347.8–352.0 s], [340.6 s],
)

The 210 s left after reading the files is host-side weight processing and transfer to the TPUs
[INFERENCE: dtype handling and sharding in the tpu-inference loader; not profiled]. Cutting it
would need the weights to stay resident on the devices, which a layout change by definition
does not allow for the engines that change shape.

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

Every value is the makespan of one whole manifest, with per-repetition values in parentheses.
The simulator was within 5% everywhere except W0 on M (+7%) and W5 on S (+4%), and it ranked the
layouts correctly on both new workloads. Every number in this report is measured; the simulator
only chose which runs to do.

= Break-even phase size

Switching from layout A to B for a phase pays when the time it saves exceeds C:

$ N / r_A - N / r_B > C quad <==> quad N > C dot r_A r_B / (r_B - r_A) $

With makespans $T$ for one manifest of $N_0$ tokens, the break-even multiple of that manifest is
$k = C / (T_A - T_B)$. The table uses $C = 420$ s (midpoint) and gives the range over both
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

The table is per phase. A whole job pays only if every switch does, and the static layout keeps
one more advantage the formula ignores: in one continuous backlog the next phase starts while the
previous one drains, whereas a reconfiguration must wait for the drain. In mix A that overlap was
worth 115 s (@sec-e2e).

= End-to-end mixed jobs <sec-e2e>

Two jobs, chosen from the break-even table before running: mix A (W4, then W0×4: 3.27M tokens) was
the one most likely to benefit; the control (W4, then W5: 1.37M tokens) should not. W4 needs a
TP16 engine, so S cannot run either job statically. Every arm starts from a fresh deploy of its
first layout and ends released; arms alternated order across the two repetitions. Routing is the
W4 "reserve" policy in every phase (on S it reduces to the blind policy, since S has no TP16
engine).

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, left, right, right, right, right),
  table.header[Job][Arm][Rep 1][Rep 2][Mean][vs static M],
  [Mix A], [static M], [1,752.6 s], [1,757.1 s], [1,754.9 s], [baseline],
  [Mix A], [M → S full redeploy], [1,966.3 s], [1,957.0 s], [1,961.7 s], colored(bad)[+11.8%],
  [Mix A], [*M → S partial*], [1,715.8 s], [1,730.2 s], [*1,723.0 s*], colored(good)[*−1.8%*],
  [Control], [static M], [1,771.7 s], [1,818.5 s], [1,795.1 s], [baseline],
  [Control], [M → H full redeploy], [2,243.7 s], [2,222.5 s], [2,233.1 s], colored(bad)[+24.4%],
)

#figure(
  bars((
    ("A: M → S partial", 1723.0, good),
    ("A: static M", 1754.9, luma(90)),
    ("A: M → S full", 1961.7, bad),
    ("Control: static M", 1795.1, luma(140)),
    ("Control: M → H full", 2233.1, bad.lighten(30%)),
  )),
  caption: [Job wall time from first dispatch to last finished request, including transitions.
    Mean of two repetitions.],
)

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  table.header[Mix A arm (means)][W4 phase][Transition gap][W0×4 phase][Job],
  [static M (one backlog)], [overlaps#super[c]], [–], [1,558.8 s (from 196 s)], [1,754.9 s],
  [M → S full], [316.7 s], [413.2 s], [1,231.7 s], [1,961.7 s],
  [M → S partial], [305.6 s], [9.3 s], [1,408.1 s], [1,723.0 s],
  [static S, W0×4 alone (reference)], [–], [–], [1,229.4 s], [–],
)
#footnote-text[c. In one backlog the phases overlap: W0 requests start at about 196 s while
long documents are still running; the last W4 request finished at 752 s and 1,326 s in the two
repetitions.]

*Why full redeploy loses mix A.* S serves a W0 backlog at 307.7 s per copy against 389.7 s for M
inside the mixed backlog, so the four copies save 328 s. The redeploy costs 413 s, and switching
also forfeits the 115 s of overlap the static backlog gets. Fitting those measured terms, full
redeploy breaks even at *6.4 W0 copies (4.7M chat tokens)* after a W4 phase.

*Why partial wins it.* The transition gap shrinks to 9 s (drain, re-label, retarget). During the
363 s while `small_c`/`small_d` start, `small_a`/`small_b` serve 1,344 and 1,358 requests (561k and
566k tokens), 67% of the full slice's chat rate on half the chips. The W0×4 phase then takes 177 s
longer than on a fully started S, so the effective cost is $9 + 177 approx 186$ s. Break-even
moves to *3.7 W0 copies (2.7M chat tokens)*; at four copies partial wins by 32 s.

*Control.* The W5 phase ran 43–104 s faster on H than inside the static M backlog (37 s in the
blind single-manifest runs), against C ≈ 404 s, so M→H loses by 438 s.
Part of that loss is policy: on H, the reserve policy holds 36,864 KV tokens per engine for long
prompts that W5 does not have, and W5 took 1,504–1,523 s on H instead of 1,372 s under the blind
policy. Even at 1,372 s the switch would lose about 290 s.

*Model check.* Before the runs, the per-manifest numbers predicted static M ≈ 2,183 s and full
redeploy ≈ 2,125 s for mix A. The full-redeploy arm came close (1,962 s), but static M was far
better (1,755 s): with the reserve policy and work-conserving fallback, M serves chat at 1,852
tok/s inside the backlog, against 1,543 tok/s in the blind single-manifest runs used for the
prediction. The prediction got the ordering of the full-redeploy arm wrong by assuming a weaker
static baseline; the break-even values above are refitted on the measured arms.

= Partial reconfiguration

M and S share their two TP8 engines chip for chip, and M's TP16 engine sits on the same hosts, in
the same order, as H's second engine. The new `run-transition` command (`hetero/transition.py`)
keeps the shared engines loaded: it adopts them into the target layout in place, releases only the
engines that change, fences the joiners out of discovery, starts them in the background while
phase 2 runs on the kept engines, and joins them to the gateway once they are up.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, left),
  table.header[Run][Drain][Joiner start-up][Join][Phase-2 requests before join][Outcome],
  [smoke (W0 only)], [1.1 s], [361.7 s], [0.7 s], [1,312 (547k tok)], [completed],
  [mix A rep 1], [1.0 s], [363.7 s], [0.7 s], [1,344 (561k tok)], [completed],
  [mix A rep 2], [1.2 s], [364.7 s], [failed], [–], colored(bad)[join failed; 5,669 requests got 503],
  [mix A rep 2b], [1.1 s], [364.2 s], [failed], [–], colored(bad)[same failure],
  [mix A rep 2c (after fix)], [1.0 s], [362.7 s], [9.4 s], [1,358 (566k tok)], [completed],
)

The two failures had one cause. A joiner's coordinator pod reported `Ready=False` for a few seconds
after its engine was up; the join labelled it anyway, discovery saw an unready labelled endpoint
and failed closed, and the gateway answered 503 for about 40 s, which failed the whole remaining
backlog. The fix (commit `2d003a4`) waits for the joiner pods to be Ready before labelling them
(outside the administration lock, so discovery keeps its health lease) and withdraws the labels if
a join still fails; rep 2c's 9.4 s join is that wait. A regression test reproduces the unready
pod.

Partial reconfiguration does not apply to S↔H, which share no engine, and H↔M would also need the
kept TP16 engine renamed (`large_b` → `large`), which is not implemented.

= Threats to validity <sec-validity>

- *Two repetitions.* Spreads: W0 on M 30 s, static M control 47 s, everything else under 25 s.
  The partial arm's second repetition is the run after the fix, so partial has two successful
  repetitions out of four attempts; the failed attempts are reported, not averaged in.
- *In-sample break-even for mix A.* The 6.4 / 3.7 copy break-evens are fitted on the same arms
  they describe. The per-manifest table is independent of the mix runs.
- *Policy confound in the control.* The reserve policy, chosen for W4, slows W5 on H by about
  10%; the control would lose even without it.
- *Orchestration overhead is in C.* About 50–80 s of the 407–445 s is job submission and result
  writing; a controller calling the deploy API directly would pay about 370 s, which would move
  the full-redeploy break-evens down by about 10%, not change their order of magnitude.
- *Repeated prompts.* W0×4 reuses W0's prompts; prefix caching is off, so the compute is the same,
  but a real chat backlog of that size would have more varied lengths.
- *Unfrozen pilots.* None of these runs is bound to a source freeze, and the partial driver
  (unit-tested, 164 tests pass) changed once during the campaign.

= Reproduction

Workspace `~/hetero-research`. Commits: `22e8b6f`/`bab4f9e` (W5 workload, worker phase clocks,
measurement campaign), `9c4e1b9` (composite manifests, mix runner, warm-start staging),
`4fcfb97`/`297ce98` (partial reconfiguration, `run-transition`), `2d003a4` (join readiness fix),
`583f2da` (end-to-end summariser), `fe5325d` (namespace-scoped cluster credentials).

```sh
# per-layout throughput + instrumented full transitions (S, then H, then M)
flock /tmp/hetero-cluster.lock scripts/reconfig_measure.sh S H M
# warm start: stage weights in /dev/shm, deploy S from it, remove the copies
scripts/reconfig_warm_stage.sh stage && MODEL=/dev/shm/gemma-4-31b ORDER=W0 scripts/reconfig_measure.sh S
scripts/reconfig_warm_stage.sh remove
# end-to-end arms
scripts/reconfig_mix.sh static:M:MIX_W4_W0x4 reconf:M:W4:S:W0x4 static:S:W0x4 \
  static:M:MIX_W4_W5 reconf:M:W4:H:W5
scripts/reconfig_partial.sh W4 W0x4 mixA-r1
# analysis
scripts/reconfig_timeline.py bab4f9e --json results/reconfig/timeline-bab4f9e.json
scripts/reconfig_breakeven.py --table results/reconfig/throughput_table_measured.json --cost 395 440
scripts/summarize_reconfig.py <run dirs> --json results/reconfig/e2e.json
```

Run ids (`results/hetero/artifacts/`): `rc-bab4f9e-{S,H,M}-s{1..4}` (throughput),
`rc-73e3679-S-s1` (warm start), `mix-b2458d2-*`, `mix-583f2da-*`, `mix-fe5325d-*`, `mix-985594e-*`
(end-to-end), `partial-73e3679-smoke`, `partial-b2458d2-mixA-r1`, `partial-2d003a4-mixA-r2c`
(completed), `partial-fe5325d-mixA-r2`, `partial-985594e-mixA-r2b` (failed joins).

Machine-readable evidence is committed next to this report in `reconfiguration_evidence/`:
throughput table, break-even tables, transition timelines, warm-start phase clocks, end-to-end
per-phase results (`e2e.json`), the mix-A model fit and every transition report.
