// Faster reconfiguration: pre-sharded weight restore and in-place resharding over ICI.
// Written for Typst HTML export first; the PDF is a secondary target.
//   HTML: typst compile --features html --format html reconfiguration_ici.typ
//   PDF:  typst compile reconfiguration_ici.typ
// Same helpers and conventions as w4_longdoc_heterogeneity.typ.
#let title = "Cutting reconfiguration cost: weight snapshots and in-place resharding over ICI"
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
    2026-10-02 · follow-up to the reconfiguration break-even report · branch
    `research/heterogeneous-tpu-20260918`]
} else {
  align(center)[
    #text(size: 16pt, weight: "bold", title)
    #v(2pt)
    Gemma-4 31B on TPU v5e 4×8 (32 chips) · runs of 2026-10-02 · follow-up to the reconfiguration
    break-even report
  ]
}

= Summary

Switching the slice between layouts (H = 2×TP16, S = 4×TP8, M = TP16 + 2×TP8) cost a median
419 s in the break-even study. That made a switch pay only for phases of about 2M output tokens
or more. This study attacks the largest part of that cost, getting weights into a new engine, in
two ways.

- *The 237 s weight load is not file I/O.* Iterating the checkpoint files takes 0.2 s and the
  final copy to the chips 2–3 s. Most of the rest is host-side work: about 128 s converting
  tensors, about 12 s in other host operations and about 17 s unattributed, plus 75 s compiling or
  loading kernels from the cache.
  [INFERENCE: page faults from reading the memory-mapped files land in the conversion time, so
  some I/O may hide there.]
- *Pre-sharded snapshot restore* saves each engine's final on-chip weight shards to host RAM and
  places them back directly, skipping vLLM's loader. Weight loading drops from 237 s to 28–40 s
  (about 20 s without checksum verification, measured on S/TP8 only; most of it model building,
  not data movement).
  Greedy outputs stay token-identical, and a measured full-slice switch costs
  #colored(good)[*221–229 s*], against 417–427 s for the same transitions with a full redeploy
  (407–445 s raw).
  It works with today's deploy path.
- *Moving weights over ICI is fast*: 0.41 s per 16-chip box and 0.75–1.08 s for the whole slice,
  for the full 62.5 GB of Gemma-4 31B-shaped weights.
- *In-place reconfiguration of a real engine works on one box.* Four persistent processes share
  one JAX runtime and switch TP16 → 2×TP8 → TP16 twice without ever reloading the checkpoint, with
  token-identical outputs. A warm switch costs #colored(good)[*77–101 s*]. Moving the weights is 4 s
  of it; rebuilding the engines is the rest.
- *Not ready for the full slice or for M.* When one box computed while the other resharded, the
  TPU runtime aborted. That blocks keeping half the slice serving during a switch. Pathways, the
  other route to in-place resharding, can't run here without changes outside the namespace.
- *Separate startup follow-ups improved measured components.* Cold TP8 snapshot writes take
  14–15 s rather than 50–52 s, but the new timing excludes SHA-256. A graph-construction fix
  reduces checksum-enabled restore `load_model` from 39.44 s to 26.45 s. Persisting small
  compilations reduces observed warm deployment from 383.9 s to 275.4 s. These are separate
  experiments, not additive savings or a measured combined switch.
- *Compilation overlap was blocked in the live-serving test.* Offline replay produced valid
  persistent cache files with the slice idle. With serving active, topology discovery tried
  to initialize the TPU plugin and failed because another process owned the devices. No
  overlap switch cost, successful-compile serving slowdown, or new crossover is established.

#note[
  *Status.* Every number is measured on the cluster unless labelled [ESTIMATE] or [INFERENCE].
  Full-slice C (full redeploy and restore) is the time from the last finished request of the old
  layout to the first request dispatched on the new one, plus the ramp deficit: output lost in the
  first 120 s against the run's median rate, in seconds of steady output. The in-place results are
  from one 16-chip box, measured as wall time from drain to the first token on every new engine,
  without gateway, ramp or the second box. The two are not directly comparable. None of this code
  is merged into `main`.
]

#figure(
  bars((
    ("Full redeploy (median)", 419.0, bad),
    ("In place, first switch to 2×TP8", 241.9, bad.lighten(40%)),
    ("Restore H → S", 228.6, luma(110)),
    ("Restore S → H", 221.4, luma(110)),
    ("In place, warm switch to 2×TP8", 100.6, good),
    ("In place, 2×TP8 → TP16", 77.4, good),
  )),
  caption: [Reconfiguration cost C. Full redeploy and restore: full 32-chip slice, last finished
    request → first dispatch on the new layout, plus the ramp deficit (seconds of lost steady
    output). In place: one 16-chip box, wall time from drain to the first token on every new
    engine.],
)

= The approaches at a glance <sec-glance>

A layout switch has to give each new engine its weights, a model graph, a KV cache, compiled code
and a place in the gateway. The approaches differ in which of these survive the switch.
@fig-paths shows how the weights travel; @fig-survive shows what each approach keeps.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 420" font-family="Helvetica, Arial, 'DejaVu Sans', sans-serif" font-size="12" fill="#1f2328">
<defs><marker id="ah59636e" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#59636e"/></marker><marker id="ah1a7f37" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#1a7f37"/></marker></defs>
<text x="10" y="20" font-weight="bold">How the new engines get their weights (TP8 timings, measured)</text>
<text x="10" y="92" font-weight="bold" font-size="12.5">Full redeploy</text>
<text x="10" y="217" font-weight="bold" font-size="12.5">Snapshot restore</text>
<text x="10" y="342" font-weight="bold" font-size="12.5">In place over ICI</text>
<rect x="170" y="60" width="150" height="56" rx="6" fill="#fff1e5" stroke="#e16f24" stroke-width="1.3"/>
<text x="245.0" y="77.0" text-anchor="middle" font-size="11.5" fill="#1f2328" font-weight="bold">Checkpoint</text>
<text x="245.0" y="92.0" text-anchor="middle" font-size="10.5" fill="#1f2328">/models (gcsfuse)</text>
<text x="245.0" y="107.0" text-anchor="middle" font-size="10.5" fill="#1f2328">62.5 GB safetensors</text>
<rect x="410" y="60" width="150" height="56" rx="6" fill="#fbe3cf" stroke="#e16f24" stroke-width="1.3"/>
<text x="485.0" y="77.0" text-anchor="middle" font-size="11.5" fill="#1f2328" font-weight="bold">Host CPU</text>
<text x="485.0" y="92.0" text-anchor="middle" font-size="10.5" fill="#1f2328">host work 157 s</text>
<text x="485.0" y="107.0" text-anchor="middle" font-size="10.5" fill="#1f2328">compile/cache-load 75 s</text>
<rect x="650" y="60" width="150" height="56" rx="6" fill="#ddf4ff" stroke="#0969da" stroke-width="1.3"/>
<text x="725.0" y="84.5" text-anchor="middle" font-size="11.5" fill="#1f2328" font-weight="bold">New engine HBM</text>
<text x="725.0" y="99.5" text-anchor="middle" font-size="10.5" fill="#1f2328">7.8 GB per chip</text>
<line x1="320" y1="88" x2="408" y2="88" stroke="#59636e" stroke-width="1.6" marker-end="url(#ah59636e)"/>
<text x="365.0" y="80" text-anchor="middle" font-size="10.5" fill="#59636e">iterate files 0.2 s</text>
<line x1="560" y1="88" x2="648" y2="88" stroke="#59636e" stroke-width="1.6" marker-end="url(#ah59636e)"/>
<text x="605.0" y="67" text-anchor="middle" font-size="10.5" fill="#59636e">copy to chips</text>
<text x="605.0" y="80" text-anchor="middle" font-size="10.5" fill="#59636e">3 s</text>
<text x="170" y="136" font-size="10.5" fill="#59636e">Old processes are killed; new ones start (Ray workers 22 s), then load_model 237 s, KV 26 s, compile 64 s.</text>
<rect x="170" y="185" width="150" height="56" rx="6" fill="#ddf4ff" stroke="#0969da" stroke-width="1.3"/>
<text x="245.0" y="209.5" text-anchor="middle" font-size="11.5" fill="#1f2328" font-weight="bold">Old engine HBM</text>
<text x="245.0" y="224.5" text-anchor="middle" font-size="10.5" fill="#1f2328">final shards</text>
<rect x="410" y="185" width="150" height="56" rx="6" fill="#dafbe1" stroke="#1b7c83" stroke-width="1.3"/>
<text x="485.0" y="202.0" text-anchor="middle" font-size="11.5" fill="#1f2328" font-weight="bold">Host RAM</text>
<text x="485.0" y="217.0" text-anchor="middle" font-size="10.5" fill="#1f2328">/dev/shm per host</text>
<text x="485.0" y="232.0" text-anchor="middle" font-size="10.5" fill="#1f2328">15.8–31.4 GB</text>
<rect x="650" y="185" width="150" height="56" rx="6" fill="#ddf4ff" stroke="#0969da" stroke-width="1.3"/>
<text x="725.0" y="209.5" text-anchor="middle" font-size="11.5" fill="#1f2328" font-weight="bold">New engine HBM</text>
<text x="725.0" y="224.5" text-anchor="middle" font-size="10.5" fill="#1f2328">same layout as saved</text>
<line x1="320" y1="213" x2="408" y2="213" stroke="#59636e" stroke-width="1.6" marker-end="url(#ah59636e)"/>
<text x="365.0" y="192" text-anchor="middle" font-size="10.5" fill="#59636e">snapshot once</text>
<text x="365.0" y="205" text-anchor="middle" font-size="10.5" fill="#59636e">26–52 s (idle)</text>
<line x1="560" y1="213" x2="648" y2="213" stroke="#59636e" stroke-width="1.6" marker-end="url(#ah59636e)"/>
<text x="605.0" y="179.0" text-anchor="middle" font-size="10.5" fill="#59636e">read 0.8 s</text>
<text x="605.0" y="192.0" text-anchor="middle" font-size="10.5" fill="#59636e">verify 20 s</text>
<text x="605.0" y="205.0" text-anchor="middle" font-size="10.5" fill="#59636e">place 1.5 s</text>
<text x="170" y="261" font-size="10.5" fill="#59636e">Skips conversion and load-time JIT: load_model 40 s (TP8) / 28 s (TP16); ≈20 s without the checksum. Processes still restart.</text>
<rect x="170" y="310" width="150" height="56" rx="6" fill="#ddf4ff" stroke="#0969da" stroke-width="1.3"/>
<text x="245.0" y="334.5" text-anchor="middle" font-size="11.5" fill="#1f2328" font-weight="bold">HBM, 2 × TP8</text>
<text x="245.0" y="349.5" text-anchor="middle" font-size="10.5" fill="#1f2328">weights stay resident</text>
<rect x="650" y="310" width="150" height="56" rx="6" fill="#efe5ff" stroke="#8250df" stroke-width="1.3"/>
<text x="725.0" y="334.5" text-anchor="middle" font-size="11.5" fill="#1f2328" font-weight="bold">HBM, 1 × TP16</text>
<text x="725.0" y="349.5" text-anchor="middle" font-size="10.5" fill="#1f2328">same chips, new mesh</text>
<line x1="320" y1="338" x2="648" y2="338" stroke="#1a7f37" stroke-width="3" marker-end="url(#ah1a7f37)"/>
<text x="485.0" y="317" text-anchor="middle" font-size="10.5" fill="#1a7f37">ICI collective-permute, chip to chip, no host copy</text>
<text x="485.0" y="330" text-anchor="middle" font-size="10.5" fill="#1a7f37">0.41 s per 16-chip box · 0.75–1.08 s for the slice</text>
<text x="170" y="386" font-size="10.5" fill="#59636e">Processes, JAX runtime and weights survive; vLLM engines are rebuilt around them (≈97 s of the 101 s warm switch).</text>
<line x1="10" y1="160" x2="810" y2="160" stroke="#d0d7de"/><line x1="10" y1="285" x2="810" y2="285" stroke="#d0d7de"/>
</svg>
```),
  caption: [Weight path for each approach. Full redeploy reads the checkpoint and converts it on the
    host; restore copies final shards back from host RAM; in-place keeps them on the chips and
    moves them over ICI.],
) <fig-paths>

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 534" font-family="Helvetica, Arial, 'DejaVu Sans', sans-serif" font-size="11" fill="#1f2328">
<text x="10" y="20" font-weight="bold" font-size="12">What survives a layout switch</text>
<text x="275.0" y="40" text-anchor="middle" font-weight="bold" font-size="12">Full redeploy</text>
<text x="425.0" y="40" text-anchor="middle" font-weight="bold" font-size="12">Partial (S↔M)</text>
<text x="575.0" y="40" text-anchor="middle" font-weight="bold" font-size="12">Snapshot restore</text>
<text x="725.0" y="40" text-anchor="middle" font-weight="bold" font-size="12">In place (ICI)</text>
<text x="190" y="70.5" text-anchor="end" font-size="11" fill="#1f2328" font-weight="bold">Worker processes,</text>
<text x="190" y="83.5" text-anchor="end" font-size="11" fill="#1f2328" font-weight="bold">JAX runtime</text>
<rect x="203" y="53" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="275.0" y="70.5" text-anchor="middle" font-size="11" fill="#cf222e">recreated</text>
<text x="275.0" y="83.5" text-anchor="middle" font-size="11" fill="#cf222e">(Ray workers 22 s)</text>
<rect x="353" y="53" width="144" height="40" rx="5" fill="#fff8c5" stroke="#9a6700" stroke-width="1"/>
<text x="425.0" y="70.5" text-anchor="middle" font-size="11" fill="#9a6700">kept on shared</text>
<text x="425.0" y="83.5" text-anchor="middle" font-size="11" fill="#9a6700">engines only</text>
<rect x="503" y="53" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="575.0" y="70.5" text-anchor="middle" font-size="11" fill="#cf222e">recreated</text>
<text x="575.0" y="83.5" text-anchor="middle" font-size="11" fill="#cf222e">(Ray workers 22 s)</text>
<rect x="653" y="53" width="144" height="40" rx="5" fill="#dafbe1" stroke="#1a7f37" stroke-width="1"/>
<text x="725.0" y="77.0" text-anchor="middle" font-size="11" fill="#1a7f37">kept</text>
<text x="190" y="123.0" text-anchor="end" font-size="11" fill="#1f2328" font-weight="bold">Weights in HBM</text>
<rect x="203" y="99" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="275.0" y="116.5" text-anchor="middle" font-size="11" fill="#cf222e">reloaded from</text>
<text x="275.0" y="129.5" text-anchor="middle" font-size="11" fill="#cf222e">checkpoint, 237 s</text>
<rect x="353" y="99" width="144" height="40" rx="5" fill="#fff8c5" stroke="#9a6700" stroke-width="1"/>
<text x="425.0" y="116.5" text-anchor="middle" font-size="11" fill="#9a6700">kept on shared;</text>
<text x="425.0" y="129.5" text-anchor="middle" font-size="11" fill="#9a6700">reloaded on others</text>
<rect x="503" y="99" width="144" height="40" rx="5" fill="#fff8c5" stroke="#9a6700" stroke-width="1"/>
<text x="575.0" y="116.5" text-anchor="middle" font-size="11" fill="#9a6700">from /dev/shm</text>
<text x="575.0" y="129.5" text-anchor="middle" font-size="11" fill="#9a6700">20–40 s</text>
<rect x="653" y="99" width="144" height="40" rx="5" fill="#dafbe1" stroke="#1a7f37" stroke-width="1"/>
<text x="725.0" y="116.5" text-anchor="middle" font-size="11" fill="#1a7f37">kept; resharded</text>
<text x="725.0" y="129.5" text-anchor="middle" font-size="11" fill="#1a7f37">over ICI, 4 s</text>
<text x="190" y="169.0" text-anchor="end" font-size="11" fill="#1f2328" font-weight="bold">Model graph</text>
<rect x="203" y="145" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="275.0" y="162.5" text-anchor="middle" font-size="11" fill="#cf222e">built inside</text>
<text x="275.0" y="175.5" text-anchor="middle" font-size="11" fill="#cf222e">load_model</text>
<rect x="353" y="145" width="144" height="40" rx="5" fill="#fff8c5" stroke="#9a6700" stroke-width="1"/>
<text x="425.0" y="169.0" text-anchor="middle" font-size="11" fill="#9a6700">kept on shared</text>
<rect x="503" y="145" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="575.0" y="162.5" text-anchor="middle" font-size="11" fill="#cf222e">built inside</text>
<text x="575.0" y="175.5" text-anchor="middle" font-size="11" fill="#cf222e">restore</text>
<rect x="653" y="145" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="725.0" y="169.0" text-anchor="middle" font-size="11" fill="#cf222e">rebuilt, 15 s</text>
<text x="190" y="215.0" text-anchor="end" font-size="11" fill="#1f2328" font-weight="bold">KV cache</text>
<rect x="203" y="191" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="275.0" y="215.0" text-anchor="middle" font-size="11" fill="#cf222e">allocated, 26 s</text>
<rect x="353" y="191" width="144" height="40" rx="5" fill="#fff8c5" stroke="#9a6700" stroke-width="1"/>
<text x="425.0" y="215.0" text-anchor="middle" font-size="11" fill="#9a6700">kept on shared</text>
<rect x="503" y="191" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="575.0" y="215.0" text-anchor="middle" font-size="11" fill="#cf222e">allocated, 26 s</text>
<rect x="653" y="191" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="725.0" y="208.5" text-anchor="middle" font-size="11" fill="#cf222e">allocated,</text>
<text x="725.0" y="221.5" text-anchor="middle" font-size="11" fill="#cf222e">2 × 12 s in turn</text>
<text x="190" y="261.0" text-anchor="end" font-size="11" fill="#1f2328" font-weight="bold">Compiled code</text>
<rect x="203" y="237" width="144" height="40" rx="5" fill="#fff8c5" stroke="#9a6700" stroke-width="1"/>
<text x="275.0" y="254.5" text-anchor="middle" font-size="11" fill="#9a6700">disk cache</text>
<text x="275.0" y="267.5" text-anchor="middle" font-size="11" fill="#9a6700">54–64 s</text>
<rect x="353" y="237" width="144" height="40" rx="5" fill="#fff8c5" stroke="#9a6700" stroke-width="1"/>
<text x="425.0" y="261.0" text-anchor="middle" font-size="11" fill="#9a6700">kept on shared</text>
<rect x="503" y="237" width="144" height="40" rx="5" fill="#fff8c5" stroke="#9a6700" stroke-width="1"/>
<text x="575.0" y="261.0" text-anchor="middle" font-size="11" fill="#9a6700">disk cache, 63 s</text>
<rect x="653" y="237" width="144" height="40" rx="5" fill="#fff8c5" stroke="#9a6700" stroke-width="1"/>
<text x="725.0" y="254.5" text-anchor="middle" font-size="11" fill="#9a6700">disk cache, 33 s</text>
<text x="725.0" y="267.5" text-anchor="middle" font-size="11" fill="#9a6700">(needs writer patch)</text>
<text x="190" y="300.5" text-anchor="end" font-size="11" fill="#1f2328" font-weight="bold">vLLM engine core,</text>
<text x="190" y="313.5" text-anchor="end" font-size="11" fill="#1f2328" font-weight="bold">scheduler</text>
<rect x="203" y="283" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="275.0" y="307.0" text-anchor="middle" font-size="11" fill="#cf222e">recreated</text>
<rect x="353" y="283" width="144" height="40" rx="5" fill="#fff8c5" stroke="#9a6700" stroke-width="1"/>
<text x="425.0" y="307.0" text-anchor="middle" font-size="11" fill="#9a6700">kept on shared</text>
<rect x="503" y="283" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="575.0" y="307.0" text-anchor="middle" font-size="11" fill="#cf222e">recreated</text>
<rect x="653" y="283" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="725.0" y="307.0" text-anchor="middle" font-size="11" fill="#cf222e">recreated, ≈20 s</text>
<text x="190" y="353.0" text-anchor="end" font-size="11" fill="#1f2328" font-weight="bold">Gateway</text>
<rect x="203" y="329" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="275.0" y="346.5" text-anchor="middle" font-size="11" fill="#cf222e">torn down,</text>
<text x="275.0" y="359.5" text-anchor="middle" font-size="11" fill="#cf222e">rolled out 21 s</text>
<rect x="353" y="329" width="144" height="40" rx="5" fill="#dafbe1" stroke="#1a7f37" stroke-width="1"/>
<text x="425.0" y="346.5" text-anchor="middle" font-size="11" fill="#1a7f37">kept; retarget</text>
<text x="425.0" y="359.5" text-anchor="middle" font-size="11" fill="#1a7f37">+ join, 9 s</text>
<rect x="503" y="329" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="575.0" y="353.0" text-anchor="middle" font-size="11" fill="#cf222e">rolled out, 22 s</text>
<rect x="653" y="329" width="144" height="40" rx="5" fill="#f6f8fa" stroke="#8c959f" stroke-width="1"/>
<text x="725.0" y="346.5" text-anchor="middle" font-size="11" fill="#59636e">not integrated</text>
<text x="725.0" y="359.5" text-anchor="middle" font-size="11" fill="#59636e">(prototype)</text>
<text x="190" y="392.5" text-anchor="end" font-size="11" fill="#1f2328" font-weight="bold">Serving during</text>
<text x="190" y="405.5" text-anchor="end" font-size="11" fill="#1f2328" font-weight="bold">the switch</text>
<rect x="203" y="375" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="275.0" y="399.0" text-anchor="middle" font-size="11" fill="#cf222e">none</text>
<rect x="353" y="375" width="144" height="40" rx="5" fill="#dafbe1" stroke="#1a7f37" stroke-width="1"/>
<text x="425.0" y="392.5" text-anchor="middle" font-size="11" fill="#1a7f37">shared engines,</text>
<text x="425.0" y="405.5" text-anchor="middle" font-size="11" fill="#1a7f37">67% of full rate</text>
<rect x="503" y="375" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="575.0" y="399.0" text-anchor="middle" font-size="11" fill="#cf222e">none</text>
<rect x="653" y="375" width="144" height="40" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1"/>
<text x="725.0" y="392.5" text-anchor="middle" font-size="11" fill="#cf222e">none: overlap</text>
<text x="725.0" y="405.5" text-anchor="middle" font-size="11" fill="#cf222e">aborts the runtime</text>
<line x1="200" y1="424" x2="800" y2="424" stroke="#1f2328" stroke-width="1"/>
<text x="190" y="450.0" text-anchor="end" font-size="12" fill="#1f2328" font-weight="bold">Measured cost C</text>
<text x="275.0" y="450.0" text-anchor="middle" font-size="12" fill="#1f2328" font-weight="bold">407–445 s</text>
<text x="425.0" y="443.5" text-anchor="middle" font-size="12" fill="#1f2328" font-weight="bold">186 s effective</text>
<text x="425.0" y="456.5" text-anchor="middle" font-size="12" fill="#1f2328" font-weight="bold">(mix A)</text>
<text x="575.0" y="450.0" text-anchor="middle" font-size="12" fill="#1f2328" font-weight="bold">221–229 s</text>
<text x="725.0" y="443.5" text-anchor="middle" font-size="12" fill="#1f2328" font-weight="bold">77–101 s</text>
<text x="725.0" y="456.5" text-anchor="middle" font-size="12" fill="#1f2328" font-weight="bold">(one box, warm)</text>
<rect x="200" y="470" width="12" height="12" rx="2" fill="#dafbe1" stroke="#1a7f37"/><text x="217" y="480" font-size="10.5" fill="#424a53">kept or better</text>
<rect x="350" y="470" width="12" height="12" rx="2" fill="#fff8c5" stroke="#9a6700"/><text x="367" y="480" font-size="10.5" fill="#424a53">partly kept / fast path</text>
<rect x="500" y="470" width="12" height="12" rx="2" fill="#ffebe9" stroke="#cf222e"/><text x="517" y="480" font-size="10.5" fill="#424a53">rebuilt or lost</text>
<rect x="650" y="470" width="12" height="12" rx="2" fill="#f6f8fa" stroke="#8c959f"/><text x="667" y="480" font-size="10.5" fill="#424a53">not available</text>
</svg>
```),
  caption: [State that survives a switch, and what rebuilding the rest costs. Partial
    reconfiguration is from the break-even study; it keeps the two TP8 engines that M and S share
    and pays full cost on the other half. Times measured; in-place on one box.],
) <fig-survive>

= Where weight loading spends its time <sec-load>

The `LoadProfiler` agent added load-time-only instrumentation (off by default) to the engine
worker and profiled all 8 workers of one S deploy and one H deploy. The load path for this model
is tpu-inference's native Flax loader (`get_flax_model`), not the torch wrapper. The profiler's own
overhead was not measured, so the split below is approximate.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 307" font-family="Helvetica, Arial, 'DejaVu Sans', sans-serif" font-size="12" fill="#1f2328">
<text x="10" y="18" font-weight="bold">Where load_model goes, TP8 engine, median of 8 workers (measured)</text>
<text x="180" y="49" text-anchor="end" font-weight="bold">Checkpoint loader</text>
<text x="180" y="63" text-anchor="end" font-size="10.5" fill="#59636e">normal deploy</text>
<rect x="190.0" y="36" width="0.8" height="30" fill="#afb8c1" stroke="#fff" stroke-width="1"/>
<rect x="190.4" y="36" width="351.7" height="30" fill="#e16f24" stroke="#fff" stroke-width="1"/>
<text x="366.3" y="55" text-anchor="middle" fill="#fff" font-size="11">157</text>
<rect x="542.1" y="36" width="167.8" height="30" fill="#8250df" stroke="#fff" stroke-width="1"/>
<text x="626.0" y="55" text-anchor="middle" fill="#fff" font-size="11">75</text>
<rect x="709.9" y="36" width="6.7" height="30" fill="#0969da" stroke="#fff" stroke-width="1"/>
<rect x="716.6" y="36" width="9.2" height="30" fill="#8c959f" stroke="#fff" stroke-width="1"/>
<text x="731.8" y="55" font-weight="bold">239 s</text>
<rect x="190.0" y="72" width="9" height="9" fill="#afb8c1"/><text x="202.0" y="81" font-size="10.5" fill="#424a53">file iteration 0.2 s</text>
<rect x="336.0" y="72" width="9" height="9" fill="#e16f24"/><text x="348.0" y="81" font-size="10.5" fill="#424a53">host work (convert + other) 157 s</text>
<rect x="560.0" y="72" width="9" height="9" fill="#8250df"/><text x="572.0" y="81" font-size="10.5" fill="#424a53">compile / cache-load 75 s</text>
<rect x="190.0" y="87" width="9" height="9" fill="#0969da"/><text x="202.0" y="96" font-size="10.5" fill="#424a53">host→device 3.0 s</text>
<rect x="318.0" y="87" width="9" height="9" fill="#8c959f"/><text x="330.0" y="96" font-size="10.5" fill="#424a53">post-load 4.1 s</text>
<text x="180" y="131" text-anchor="end" font-weight="bold">Snapshot restore</text>
<text x="180" y="145" text-anchor="end" font-size="10.5" fill="#59636e">SHA-256 verified</text>
<rect x="190.0" y="118" width="0.8" height="30" fill="#afb8c1" stroke="#fff" stroke-width="1"/>
<rect x="190.2" y="118" width="1.7" height="30" fill="#54aeff" stroke="#fff" stroke-width="1"/>
<rect x="191.9" y="118" width="45.7" height="30" fill="#bf8700" stroke="#fff" stroke-width="1"/>
<text x="214.8" y="137" text-anchor="middle" fill="#fff" font-size="11">20</text>
<rect x="237.6" y="118" width="3.4" height="30" fill="#0969da" stroke="#fff" stroke-width="1"/>
<rect x="241.0" y="118" width="38.1" height="30" fill="#8c959f" stroke="#fff" stroke-width="1"/>
<text x="260.0" y="137" text-anchor="middle" fill="#fff" font-size="11">17</text>
<text x="285.0" y="137" font-weight="bold">40 s</text>
<rect x="190.0" y="154" width="9" height="9" fill="#afb8c1"/><text x="202.0" y="163" font-size="10.5" fill="#424a53">manifest 0.1 s</text>
<rect x="300.0" y="154" width="9" height="9" fill="#54aeff"/><text x="312.0" y="163" font-size="10.5" fill="#424a53">tmpfs read 0.8 s</text>
<rect x="422.0" y="154" width="9" height="9" fill="#bf8700"/><text x="434.0" y="163" font-size="10.5" fill="#424a53">SHA-256 verify 20 s</text>
<rect x="562.0" y="154" width="9" height="9" fill="#0969da"/><text x="574.0" y="163" font-size="10.5" fill="#424a53">placement 1.5 s</text>
<rect x="190.0" y="169" width="9" height="9" fill="#8c959f"/><text x="202.0" y="178" font-size="10.5" fill="#424a53">model build + other 17 s</text>
<text x="180" y="213" text-anchor="end" font-weight="bold">Snapshot restore</text>
<text x="180" y="227" text-anchor="end" font-size="10.5" fill="#59636e">unverified</text>
<rect x="190.0" y="200" width="44.1" height="30" fill="#8c959f" stroke="#fff" stroke-width="1"/>
<text x="212.1" y="219" text-anchor="middle" fill="#fff" font-size="11">20</text>
<text x="240.1" y="219" font-weight="bold">20 s</text>
<rect x="190.0" y="236" width="9" height="9" fill="#8c959f"/><text x="202.0" y="245" font-size="10.5" fill="#424a53">all restore phases 20 s</text>
<line x1="190" y1="267" x2="750" y2="267" stroke="#59636e"/>
<line x1="190.0" y1="267" x2="190.0" y2="272" stroke="#59636e"/><text x="190.0" y="284" text-anchor="middle" font-size="10" fill="#59636e">0</text>
<line x1="302.0" y1="267" x2="302.0" y2="272" stroke="#59636e"/><text x="302.0" y="284" text-anchor="middle" font-size="10" fill="#59636e">50</text>
<line x1="414.0" y1="267" x2="414.0" y2="272" stroke="#59636e"/><text x="414.0" y="284" text-anchor="middle" font-size="10" fill="#59636e">100</text>
<line x1="526.0" y1="267" x2="526.0" y2="272" stroke="#59636e"/><text x="526.0" y="284" text-anchor="middle" font-size="10" fill="#59636e">150</text>
<line x1="638.0" y1="267" x2="638.0" y2="272" stroke="#59636e"/><text x="638.0" y="284" text-anchor="middle" font-size="10" fill="#59636e">200</text>
<line x1="750.0" y1="267" x2="750.0" y2="272" stroke="#59636e"/><text x="750.0" y="284" text-anchor="middle" font-size="10" fill="#59636e">250</text>
<text x="750" y="299" text-anchor="end" font-size="10.5" fill="#59636e">seconds</text>
</svg>
```),
  caption: [Weight loading for one TP8 engine, normal loader against snapshot restore. Medians over
    8 workers (raysubmit_WhmszEvQNQQRbC8D for the loader; ShardedRestore validation for restore).],
) <fig-load>

#table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  table.header[load_model part (median of 8 workers)][S / TP8][H / TP16],
  [Total `load_model`], [239.1 s], [240.0 s],
  [Iterating safetensors files#super[b]], [0.2 s], [0.2 s],
  [Host work (conversion ≈128 s, other ops ≈12 s, unattributed ≈17 s)], [157.0 s], [156.5 s],
  [Compiling or loading kernels from cache during load], [74.9 s], [76.8 s],
  [Host → device], [3.0 s], [2.0 s],
  [Post-load processing], [4.1 s], [4.3 s],
  [Final shard bytes per host], [31.4 GB], [15.8 GB],
)
#footnote-text[b. Time spent in the file iterator only. The files are memory-mapped, so page faults
on first access are counted in host work, not here.]

- Halving the bytes per chip (TP8 → TP16) leaves the load time unchanged, so final transfer size
  doesn't drive it.
- One function, `jax_array_from_reshaped_torch`, takes 128 s over 1,188 tensor conversions. The
  slowest are attention `o_proj` weights at about 4 s each [INFERENCE: their reshape misses the
  fast bfloat16 path and falls back to a slow conversion].
- The loader makes 3,365 compile-or-cache calls and clears its caches 333 times, once per module
  [INFERENCE: the clearing prevents reuse of kernels with the same shape].
- The copies of final shards to the chips, measured per array, ran at 9.7–11.9 GB/s: 2.6 s (TP8)
  and 1.6 s (TP16) of put time per worker. That is a per-array rate, not a measured floor for a
  whole restore.

= Approach A: pre-sharded snapshot restore <sec-restore>

*Mechanism.* After an engine has loaded and is idle, an RPC writes each worker's final per-device
parameter arrays to `/dev/shm/omp_weights/<layout>/<instance>/rank<k>/`. A manifest records each
array's path, shape, dtype, sharding, mesh device order, checksum and the settings that shape the
weights. On deploy, `weight_snapshot: read` builds the model graph without reading the checkpoint
and places each shard directly on its device. Any manifest mismatch (TP size, device order,
model, missing shard or bad checksum) falls back to the normal loader and logs why. Branch
`restore/sharded-snapshot`, commit `cc0f44f`; off by default; 8 local tests.

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  table.header[Engine start-up (8 workers, min–max)][S normal][S restore][H normal][H restore],
  [`load_model`], [234.3–236.9 s], [*39.4–40.5 s*], [236.7–239.2 s], [*28.3–28.8 s*],
  [of which SHA-256 verify], [–], [20.2–20.4 s], [–], [10.1–10.3 s],
  [of which tmpfs read], [–], [0.7–0.8 s], [–], [0.7–0.8 s],
  [of which device placement], [–], [1.4–2.5 s], [–], [0.8–0.9 s],
  [`load_model`, verification off], [–], [19.1–20.3 s], [–], [not run],
  [Compile + warm-up], [62.2–64.0 s], [62.1–62.5 s], [63.1–64.5 s], [63.2–63.4 s],
)

*Correctness.* For 32 fixed W0 prompts (greedy, 64 tokens), every restored engine of S and H gave
output tokens identical to a normally loaded engine, and every worker took the restore path. A W0
run after restore took 343.7 s on S (343.3 s before) and 491.0 s on H (497.2 s).

*Snapshot cost.* Writing a snapshot takes 50–52 s per TP8 worker and 26 s per TP16 worker, while
the engine is idle. The validation harness snapshotted the engines one after another, so the whole
S layout took 210.4 s and H 57.9 s. That time is not in the switch costs below, which assume the
target layout's snapshot already exists; a job that has to snapshot the old layout right before
switching away pays it on the critical path. Snapshots occupy 31.4 GB per host for S and 15.8 GB
for H. Both together fit in the roughly 175 GB of free tmpfs per host.

*Measured switch cost with restore* (full slice, same method as the break-even study):

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right, right),
  table.header[Switch][Teardown][Deploy][of which restore][KV alloc][Compile + warm-up][1st dispatch + ramp][Cost C],
  [S → H], [27.4 s], [165.0 s], [28.4 s], [25.8 s], [63.3 s], [29.0 s#super[c]], [*221.4 s*],
  [H → S], [26.3 s], [175.4 s], [39.8 s], [25.7 s], [62.6 s], [27.0 s#super[c]], [*228.6 s*],
  [S → H, full redeploy (before)#super[a]], [45.0 s], [376.2 s], [237.8 s], [26.2 s], [64.7 s], [23.8 s], [445.0 s \ (427.0 s like-for-like)],
)
#footnote-text[a. From the break-even study. That campaign also tore down the not-yet-deployed
target layout before deploying it, which added 18.0 s that the restore runs didn't pay (18.1 s
for H → M). Without it the full-redeploy costs are 427.0 s (S → H) and 417.5 s (H → M).
c. Includes a 6–7 s pre-serving diagnostic check that the restore runs added and the baseline
didn't, so restore's C is overstated by that much.]

Compile and warm-up (63 s), KV allocation (26 s) and orchestration (teardown, Ray workers, job
submission, gateway: about 90 s) now dominate. Restore removes about 206 s on the like-for-like
S → H comparison (427.0 → 221.4 s), nearly all of it from the weight load.

== Follow-up: faster snapshots, independently restored <sec-fast-snapshot>

FastSnapshot was qualified on S (four TP8 engines), not H. Each worker writes 31.45 GB.
These are worker wall times, not full-layout transition costs:

#table(
  columns: (1fr, auto),
  table.header[Measured component][Range across eight workers],
  [Cold snapshot, including publication], [14.00–14.92 s],
  [Repeated snapshot into reused files], [3.56–3.65 s],
  [Restore device placement and completion], [1.41–1.48 s],
  [Complete restore `load_model`], [18.94–19.29 s],
)

The repeated write can benefit from runtime host caching; it is not a fresh device-to-host
snapshot rate. Timed manifests use `checksum_kind=none`: strong fingerprints were checked
outside the timed path. Cold files, live source, and restored weights matched across 32,896
shards (251.588 GB across eight workers). All 128 prompt comparisons and 8,140 output tokens
matched the pinned reference. The producer actors and placement groups were removed before
consumer initialization; all eight worker PIDs changed on the same hosts. Both generations
were released and all snapshot directories cleaned. An earlier attempt failed in the
validation harness with a missing `layout` field; its artifacts are retained, not counted
as a successful restore.

*Hardware versus collector.* A separate four-chip, 1 GiB-per-chip probe measured runtime-owned
pinned-host donation at 56.54 and 58.29 GB/s (median 57.42). Its timer includes output readiness
but excludes a subsequent copied host-array acquisition; the ordinary collector, at about
11.8 GB/s, includes that acquisition. These are not interchangeable snapshot measurements.
Dividing 31.45 GB by the best pinned rate gives 0.54 s: an extrapolated transfer component,
not a measured full-model snapshot. The tested caller-mmap route did not eliminate the copy.

*Cheap, weaker integrity check.* A synthetic four-chip probe at all 4,112 real shard shapes
computed per-shard sum/XOR of BF16 bits. Enqueue through ready outputs took 0.266–0.273 s;
including host readback took 0.733–0.735 s. Compilation and source generation were excluded.
This reduction is position-insensitive and has demonstrated collisions: it is not a SHA-256
replacement, nor an integrated snapshot checksum measurement.

*No new switch-cost claim.* These results do not measure a new C or break-even point. The
transition table above remains the earlier ShardedRestore campaign. Compilation-overlap key
checks match all 261 distinct captured S executables, but key equality proves neither target
cache hits nor serving-performance benefits. The initial cache-threshold A/B retained
threshold 1.0 in every worker and is invalid, not a measured null effect.

Evidence: `reconfiguration_ici_evidence/FollowupAudit/` contains independent source-hashed
verdicts (commits `bae2c76`, `913fff5`, `7f46e9a`). Snapshot data path `1d591d8`, corrected
validation harness `a5ebfdc`; helper branches `helper/snapshot-route` and `helper/overlap`.
Further cache/overlap measurements are not included until qualified.

*Checksum-matched comparison.* The 50–52 s to 14–15 s snapshot comparison changes both the
copy implementation and checksum policy. It is not a 3.5× gain at equal integrity guarantees.
The earlier checksum-disabled restore already took 19.1–20.3 s, close to FastSnapshot
restore at 18.94–19.29 s. The approximately 20 s SHA cost was directly measured on restore;
its contribution to the old snapshot writer was inferred, not separately measured.

== Follow-up: removing graph-construction overhead <sec-fast-graph>

A separate S-only four-arm experiment tested an opt-in workaround for expensive diagnostic
formatting in JAX tracer sharding access during abstract model construction. It temporarily
replaces the process-global `Tracer.sharding` property during `load_model`, restoring the
original in `finally`. It does not alter the inference step or device executable. The patch
is not thread-local; concurrent-initializer safety has not been established.

#table(
  columns: (1fr, auto, auto),
  table.header[Component, median across eight workers][Baseline][Fast graph],
  [Normal-loader graph construction], [14.739 s], [1.790 s],
  [Normal-loader complete `load_model`], [234.675 s], [222.587 s],
  [Restore graph construction], [14.631 s], [1.767 s],
  [Restore complete `load_model`], [39.436 s], [26.455 s],
  [Restore SHA-256 verification], [20.280 s], [20.292 s],
)

Both restore arms used the original v1 snapshot path with full SHA-256 verification, not
FastSnapshot with checksums disabled. All four arms matched 128 reference outputs / 8,140
tokens each. Frozen source identities matched and each arm was released before the next;
final snapshot cleanup left zero bytes on all eight hosts.

This is one fixed-order sequence, not eight independent repetitions. Normal loading does
not imply cold filesystem or compile caches; snapshots were resident for the normal-fast
arm but not the normal baseline. Total-load differences are observations with these
confounds, and concurrent worker savings must not be summed. No combined FastSnapshot,
cache-threshold, full-transition C, H qualification, or inference-throughput gain is claimed.

Evidence: runtime `df46a72` on `restore/fast-graph`, results `f317e57`, independent audit
`46c1461` in `reconfiguration_ici_evidence/FollowupAudit/followup_graph_claims.json`.

== Follow-up: persist the small compilations <sec-cache-threshold>

The corrected S experiment compared a persistent-cache minimum compile time of 1 s with
0 s; both used entry-size threshold zero. Applying environment variables after JAX import
was insufficient, so the worker now updates JAX configuration explicitly before constructing
the runner and verifies the effective settings. The original failed contrast remains excluded.

Each setting had one first deployment and one reuse deployment. In the reuse comparison:

#table(
  columns: (1fr, auto, auto),
  table.header[Observed warm-start metric][Threshold 1 s][Threshold 0 s],
  [Harness deployment wall time], [383.867 s], [275.387 s],
  [Remote controller elapsed], [377.860 s], [269.314 s],
  [`load_model`, per-engine max-rank range], [236.50–240.05 s], [171.76–174.45 s],
  [Initialize + compile/warm-up, same scope], [96.29–98.41 s], [54.13–54.81 s],
  [Cache entries per worker], [12], [344],
  [Startup persistent-hit events], [96], [29,008],
  [Startup persistent-miss events], [28,912], [0],
)

Independent replay verified all 32 effective-worker configurations across the four arms,
matching source maps and all 64 matched worker logs, plus release and cleanup. The observed
deployment difference is 108.479 s. These are startup-only observations: no workload or
token-correctness test, full reconfiguration C, or break-even measurement was performed.
Fixed order and correlated workers preclude treating the worker spread as independent trials.

Hit counts are events, including CPU loader work, not distinct executables: the union of
distinct hit keys across workers was 12 versus 381. Zero misses refers only to captured
startup logs. JAX messages labelled “Finished XLA compilation” also follow cache hits; their
durations are not pure compiler time. This result is separate from FastGraph and FastSnapshot;
their savings cannot simply be added.

Evidence: `helper/warmup-set`, runtime `2a3dd37`, results `0ebdf22`, run
`ct-20261002b-configfix`; independent audit `87f83d3` in
`reconfiguration_ici_evidence/FollowupAudit/followup_threshold_claims.json`.

== Compilation overlap: blocked with live serving <sec-overlap-followup>

Offline replay compiled 261 distinct captured executable groups for each of S and H, with
actual keys matching their captures. This proves CPU-hosted compilation and key agreement,
not target cache reuse. The sequential S/H preflight ran in one process; its requested H
cache directory was absent at cleanup. A source-matched JAX 0.10.2 reproduction showed why
changing configuration is insufficient: an initialized process-global cache retains its
original directory. A fresh process uses the new directory correctly. Remote H file placement
was not independently observed, so it is not inferred from that reproduction.

An isolated-process retry removed that cache-directory ambiguity. It produced 261 physical
cache files per layout, with SHA-256/byte inventories identical across eight hosts: 149.289 MB
for S and 86.753 MB for H. That standalone qualification took 533.848 s including inventories,
outside C; both cache roots were subsequently removed. It proved persistence with the slice
idle, not compatibility with a live serving runtime.

The later H-to-S baseline completed a closed cold-switch interval. It is explicitly an
*unfrozen pilot*, with the runtime cohort check bypassed, not frozen/cohort headline evidence.
Both layouts completed 1,763 W0 requests and 721,595 output tokens. Their measured makespans
were 497.533 s (H) and 332.000 s (S), with identical actual input hashes. The actual catalogue
differs from historical provenance; these rates must not be combined with historical
crossover inputs. Independent post-W0 greedy outputs matched the target reference.

#table(
  columns: (1fr, auto, auto),
  table.header[Closed H-to-S pilot accounting][Cold target cache][Warm target cache],
  [Last source completion to first measured target dispatch], [451.410 s], [218.716 s],
  [First-120-s positive output-deficit equivalent], [6.887 s], [6.544 s],
  [Sum: descriptive harness-inclusive C], [458.297 s], [225.260 s],
  [Prior target provisioning: deploy, reference, snapshot, release], [920.802 s], [920.802 s],
  [C plus that target provisioning], [1,379.099 s], [1,146.063 s],
)

The gap charges source artifact export, teardown, target deployment and controls/restore
checks, plus pilot setup and internal warmup. Its endpoint is the first *measured client
dispatch*, not the first-ever serving request. The ramp term is an output-deficit estimate,
not another literal no-serving interval. Post-W0 equality, later cache inventories and process
holds are outside this closed interval. The last row adds the stated target provisioning only,
not the initial source deployment or useful source workload.

“Cold” names the initially empty target cache, not an all-miss startup: replicas share it and
can hit each other’s entries. The subsequent warm baseline passed the startup reuse check:
all 2,088 reads across 16 phase-worker profiles contained a successful same-key deserialization,
with zero misses/backend compilations in those two phases. Unkeyed listener events corroborate
the count, not the key identity. These profiles precede the standard W0 warmup. Both warm-arm
W0 runs completed 1,763 requests / 721,595 tokens; makespans were 491.710 s (H) and 333.837 s (S).
Post-W0 greedy equality matched all 128 outputs / 8,140 tokens.

A read-only inventory after the final W0 found all 295 observed startup keys, byte-identical
across eight hosts, in each of the two baseline cache roots. That inventory is not itself
proof of pre-start availability; the actual startup reads above provide the reuse evidence.
The 28.841 s process hold and 24.850 s inventory were outside C. All five baseline generations
were drained/released, caches removed, tmpfs snapshots cleaned, and paused processes resumed.
This does not imply later authorized overlap work had ended.

The 80 retained ESTALE warning records include 35 Ray repeat-compressed lines; they are not
80 uniquely identified failed writes. Passing later coverage does not erase those warnings.
These cold/warm observations are a fixed-order, single unfrozen-pilot sequence, not an overlap
result or independent repeated trials. Device/host execution attribution remains unsupported
because the S/H traces lacked required phase markers.

*Live-serving attempt failed before compilation.* During H serving,
`topologies.get_topology_desc(platform="tpu", ...)` initialized the TPU plugin and aborted because
the serving process already owned the TPU devices. The background task returned no replay
result. The source workload still completed, but no treated target was deployed and no overlap
C was measured. Its 502.878 s makespan versus the 490.782 s control is an incidental failed-attempt
difference, not measured successful-compile interference.

The subsequent physical-cache check assumed a replay result existed, raised `KeyError` before
the supervised stage, and left the client waiting on a zombie daemon. Only that owned stuck
client was terminated to trigger existing cleanup. Dependent S-to-H cold/warm/overlap runs
were not submitted. This is a blocker of the tested topology-initialization path, not proof
that all CPU-hosted precompilation is impossible. The earlier empty-slice preflight did not
test coexistence with active serving.

No warm-versus-overlap saving or new break-even conclusion is established. The final reducer
retains only the two observed baseline C values; failed/unmeasured transition costs remain
null. Six historical direction-matched crossover candidates are suppressed because workload
and runtime eligibility are not established.

*Archive and closure.* The independent audit matched the serving-owner PID in the failure
to the deployed engine on the same host. It streamed and verified the 67,423,350-byte bundle
archive: 14,094 files and 4,698 descriptors, all MLIR/options hashes matching, with no duplicate-key
payload variants. This qualifies the retained payloads after the run; it does not retroactively
add runtime hash enforcement or make the failed overlap successful.

Final inventory found all 32 TPU chips available, no live placement groups or actors in the
296 retained experiment groups, zero snapshot bytes on all eight hosts, all 24 owned paths
absent on every host, and all 13 owned local processes gone. RCM was neither alive nor zombie,
and the blocking cluster lock was acquired and released. The pre-existing shared registry
and the inventory job’s own supervisor were not experiment leaks. No dependent S-to-H arm
was submitted.

Evidence: launcher/worktree revision `9009fc7` (not a loaded-backend verification), baseline
`controlled-hs-baseline-20261002b`, independent audits
`6530457` (replay), `9be7555` (cache path), `93670de` (cold interval), `ee0d5e8`
(warm interval and startup reuse), `6837f56` (isolated physical persistence), and `b6b5a24`
(final failure, archive and cleanup), archived under `reconfiguration_ici_evidence/FollowupAudit/`.
The compact final reports and generated tables are under
`reconfiguration_ici_evidence/OverlapFollowup/`; `manifest.json` records source hashes and
compression boundaries. Table generator `helper/results-analysis` commit `39d09ca` passed
21 tests plus 8 subtests; no measured overlap benefit or historical crossover is emitted.
The full raw campaign is sealed on `overlap/restore-precompile` at `c229541`. Independent
artifact audit `2ea7783` verified all 317 declared Git payloads and all 25 gzip rehydrations,
including the committed 67.4 MB graph archive. Its receipt is also in `FollowupAudit/`.

= Approach B: in-place resharding over ICI <sec-ici>

Today every engine is its own JAX runtime with its chips fixed at start-up, and two runtimes can't
exchange device buffers. In-place resharding therefore needs one persistent JAX runtime that spans
every engine of a box. Engines become groups of processes on sub-meshes of that runtime.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 330" font-family="Helvetica, Arial, 'DejaVu Sans', sans-serif" font-size="12" fill="#1f2328">
<defs><marker id="ah59636e" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#59636e"/></marker><marker id="ah1a7f37" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#1a7f37"/></marker></defs>
<text x="10" y="20" font-weight="bold">One 16-chip box (4 hosts × 4 chips): S halves vs one H engine</text>
<text x="141" y="44" text-anchor="middle" font-weight="bold">S: small_a | small_b</text>
<text x="141" y="59" text-anchor="middle" font-size="10.5" fill="#59636e">each TP8 engine = 2 hosts (x = 0–1, x = 2–3)</text>
<rect x="60" y="70" width="38" height="38" rx="4" fill="#0969da"/>
<text x="79.0" y="93.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="103" y="70" width="38" height="38" rx="4" fill="#0969da"/>
<text x="122.0" y="93.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="146" y="70" width="38" height="38" rx="4" fill="#1a7f37"/>
<text x="165.0" y="93.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="189" y="70" width="38" height="38" rx="4" fill="#1a7f37"/>
<text x="208.0" y="93.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="60" y="113" width="38" height="38" rx="4" fill="#0969da"/>
<text x="79.0" y="136.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="103" y="113" width="38" height="38" rx="4" fill="#0969da"/>
<text x="122.0" y="136.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="146" y="113" width="38" height="38" rx="4" fill="#1a7f37"/>
<text x="165.0" y="136.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="189" y="113" width="38" height="38" rx="4" fill="#1a7f37"/>
<text x="208.0" y="136.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="60" y="156" width="38" height="38" rx="4" fill="#0969da"/>
<text x="79.0" y="179.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="103" y="156" width="38" height="38" rx="4" fill="#0969da"/>
<text x="122.0" y="179.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="146" y="156" width="38" height="38" rx="4" fill="#1a7f37"/>
<text x="165.0" y="179.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="189" y="156" width="38" height="38" rx="4" fill="#1a7f37"/>
<text x="208.0" y="179.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="60" y="199" width="38" height="38" rx="4" fill="#0969da"/>
<text x="79.0" y="222.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="103" y="199" width="38" height="38" rx="4" fill="#0969da"/>
<text x="122.0" y="222.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="146" y="199" width="38" height="38" rx="4" fill="#1a7f37"/>
<text x="165.0" y="222.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="189" y="199" width="38" height="38" rx="4" fill="#1a7f37"/>
<text x="208.0" y="222.0" text-anchor="middle" font-size="10.5" fill="#fff">1/8</text>
<rect x="57" y="67" width="87" height="87" rx="6" fill="none" stroke="#1f2328" stroke-width="1.2" stroke-dasharray="4 3"/>
<rect x="57" y="153" width="87" height="87" rx="6" fill="none" stroke="#1f2328" stroke-width="1.2" stroke-dasharray="4 3"/>
<rect x="143" y="67" width="87" height="87" rx="6" fill="none" stroke="#1f2328" stroke-width="1.2" stroke-dasharray="4 3"/>
<rect x="143" y="153" width="87" height="87" rx="6" fill="none" stroke="#1f2328" stroke-width="1.2" stroke-dasharray="4 3"/>
<text x="681" y="44" text-anchor="middle" font-weight="bold">H: large_a</text>
<text x="681" y="59" text-anchor="middle" font-size="10.5" fill="#59636e">one TP16 engine = all 4 hosts</text>
<rect x="600" y="70" width="38" height="38" rx="4" fill="#8250df"/>
<text x="619.0" y="93.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="643" y="70" width="38" height="38" rx="4" fill="#8250df"/>
<text x="662.0" y="93.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="686" y="70" width="38" height="38" rx="4" fill="#8250df"/>
<text x="705.0" y="93.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="729" y="70" width="38" height="38" rx="4" fill="#8250df"/>
<text x="748.0" y="93.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="600" y="113" width="38" height="38" rx="4" fill="#8250df"/>
<text x="619.0" y="136.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="643" y="113" width="38" height="38" rx="4" fill="#8250df"/>
<text x="662.0" y="136.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="686" y="113" width="38" height="38" rx="4" fill="#8250df"/>
<text x="705.0" y="136.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="729" y="113" width="38" height="38" rx="4" fill="#8250df"/>
<text x="748.0" y="136.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="600" y="156" width="38" height="38" rx="4" fill="#8250df"/>
<text x="619.0" y="179.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="643" y="156" width="38" height="38" rx="4" fill="#8250df"/>
<text x="662.0" y="179.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="686" y="156" width="38" height="38" rx="4" fill="#8250df"/>
<text x="705.0" y="179.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="729" y="156" width="38" height="38" rx="4" fill="#8250df"/>
<text x="748.0" y="179.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="600" y="199" width="38" height="38" rx="4" fill="#8250df"/>
<text x="619.0" y="222.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="643" y="199" width="38" height="38" rx="4" fill="#8250df"/>
<text x="662.0" y="222.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="686" y="199" width="38" height="38" rx="4" fill="#8250df"/>
<text x="705.0" y="222.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="729" y="199" width="38" height="38" rx="4" fill="#8250df"/>
<text x="748.0" y="222.0" text-anchor="middle" font-size="10.5" fill="#fff">1/16</text>
<rect x="597" y="67" width="87" height="87" rx="6" fill="none" stroke="#1f2328" stroke-width="1.2" stroke-dasharray="4 3"/>
<rect x="597" y="153" width="87" height="87" rx="6" fill="none" stroke="#1f2328" stroke-width="1.2" stroke-dasharray="4 3"/>
<rect x="683" y="67" width="87" height="87" rx="6" fill="none" stroke="#1f2328" stroke-width="1.2" stroke-dasharray="4 3"/>
<rect x="683" y="153" width="87" height="87" rx="6" fill="none" stroke="#1f2328" stroke-width="1.2" stroke-dasharray="4 3"/>
<line x1="260" y1="125" x2="588" y2="125" stroke="#1a7f37" stroke-width="2.5" marker-end="url(#ah1a7f37)"/>
<text x="425.0" y="104" text-anchor="middle" font-size="10.5" fill="#1a7f37">S → H: 0.41 s native device order</text>
<text x="425.0" y="117" text-anchor="middle" font-size="10.5" fill="#1a7f37">0.23 s interleaved order (no data moved)</text>
<line x1="590" y1="175" x2="262" y2="175" stroke="#1a7f37" stroke-width="2.5" marker-end="url(#ah1a7f37)"/>
<text x="425" y="195" text-anchor="middle" font-size="10.5" fill="#1a7f37">H → S: 0.42 s (each chip receives the half it lacks)</text>
<text x="60" y="270" font-size="10.5" fill="#59636e">Numbers in chips: share of the model's weights each chip holds (TP8: 7.8 GB, TP16: 3.9 GB). Dashed squares: hosts. The other box</text>
<text x="60" y="285" font-size="10.5" fill="#59636e">(S small_c/d, H large_b = M large) is identical. In S → H every chip already holds a superset of its TP16 share; with the native</text>
<text x="60" y="300" font-size="10.5" fill="#59636e">TP16 device order the shares sit on other chips (5/8 of the weights move); an interleaved order moves nothing. Times: ICIBench, median of 10.</text>
</svg>
```),
  caption: [Chip geometry of one box. The two layouts use the same chips, so switching only
    regroups chips into a different mesh and moves each chip's share of the weights over ICI.],
) <fig-geometry>

== The transfer itself (ICIBench)

The `ICIBench` agent started one multi-controller JAX runtime per 16-chip box (4 hosts × 4
chips), then one over the whole slice, inside Ray actors with the layouts' TPU environments.
Weights were synthetic BF16 arrays generated on device with the real Gemma-4 31B parameter shapes
and partition specs: 1,028 arrays, 62.5 GB, 7.86 GB per chip at TP8 and 3.96 GB at TP16.

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, right, right),
  table.header[Scope][Route][S → H (2×TP8 → TP16)][H → S (TP16 → 2×TP8)],
  [16-chip box], [`jax.device_put`], [*0.41 s* (0.39–0.55)], [*0.42 s* (0.40–0.44)],
  [16-chip box], [assemble from local shards], [0.62 s (0.58–0.67)], [0.62 s (0.56–0.68)],
  [16-chip box], [interleaved TP16 device order], [0.23 s (0.21–0.29)], [0.41 s (0.37–0.44)],
  [32-chip slice], [`jax.device_put`], [*0.75 s* (0.74–0.76)], [*1.08 s* (1.07–1.08)],
  [32-chip slice], [assemble from local shards], [0.73 s (0.69–0.75)], [1.28 s (1.23–1.33)],
  [32-chip slice], [interleaved TP16 device order], [0.23 s (0.21–0.30)], [0.44 s (0.39–0.52)],
)
#footnote-text[Median (min–max) of 10 warm repetitions, each timed by its slowest process. The
first, cold repetition took 3.3 s (S → H) and 3.7 s (H → S) on the slice because of compilation.
Every shard checksum matched.]

- *Bandwidth.* On the slice, `device_put` moved 78 GB (S → H) and 203 GB (H → S) of logical data,
  104 and 189 GB/s in total. A profile of one 16-chip box (`device_put`, S → H, first repetition)
  shows 1,768 TPU collective-permute operations per process and no host copies, so that transfer
  went over ICI and not through host memory. The other routes and the 32-chip runs were not
  profiled.
- *Memory.* `device_put` peaks at 11.8 GB of the 16 GB per chip. Assembling from local shards
  peaks at 15.7 GB, too close to the limit for a real engine that also needs KV cache.
- *Device order matters.* With an interleaved TP16 device order, S → H needs no data movement
  (only local slicing), which is why it takes 0.23 s.
- *Runtime constraints found:*
  - JAX ranks must keep the physical TPU process identity. Remapping them broke compilation.
  - `device_put` can't change the device set and the shard layout at once. H → S therefore goes
    through a global 16-chip replica mesh, then restricts to TP8 handles.
  - In a 32-chip runtime the TPU compiler rejects reshards whose devices cover only 16 chips, so
    slice measurements use all-chip meshes.
- *Independent groups inside one runtime work* when they don't overlap in time with global work:
  processes {0,1} and {2,3} ran different computations with different numbers of collectives
  concurrently and both were correct.
- *But one box can't keep serving while the other reshards.* In that test the runtime aborted with
  "unexpected leader in the launch group with a different launch id", and libtpu killed the
  process (`SLICE_FAILURE_SW_INJECT_ERROR`). Host-side barriers before the overlap didn't help;
  the global reshard and the box-local step still raced. The agents haven't determined whether a
  scheduling fix exists.

== A real engine, switched in place (InPlaceEngine)

The `InPlaceEngine` agent built a prototype driver (not the production deploy path) around one
box:
- Four persistent Ray actors, one per host, form one JAX runtime and keep their processes and
  their on-chip weights for the whole run.
- vLLM engine cores, schedulers, runners, KV caches and compiled functions are torn down and
  rebuilt for each layout. A custom executor attaches the new engine cores to the existing
  actors.
- As TP16, all four processes form one engine; as 2×TP8, processes {0,1} and {2,3} each form one.

The run loads the checkpoint once (263–267 s in the shared runtime), then switches TP16 → 2×TP8 →
TP16 → 2×TP8 → TP16. It serves the same 32 greedy W0 prompts after each switch.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 367" font-family="Helvetica, Arial, 'DejaVu Sans', sans-serif" font-size="12" fill="#1f2328">
<text x="10" y="18" font-weight="bold">Reconfiguration cost C, broken down (measured)</text>
<text x="180" y="49" text-anchor="end" font-weight="bold">Full redeploy</text>
<text x="180" y="63" text-anchor="end" font-size="10.5" fill="#59636e">S→H, full slice</text>
<rect x="190.0" y="36" width="56.0" height="30" fill="#8c959f" stroke="#fff" stroke-width="1"/>
<text x="218.0" y="55" text-anchor="middle" fill="#fff" font-size="11">45</text>
<rect x="246.0" y="36" width="27.3" height="30" fill="#54aeff" stroke="#fff" stroke-width="1"/>
<text x="259.6" y="55" text-anchor="middle" fill="#fff" font-size="11">22</text>
<rect x="273.3" y="36" width="4.5" height="30" fill="#0969da" stroke="#fff" stroke-width="1"/>
<rect x="277.7" y="36" width="295.9" height="30" fill="#e16f24" stroke="#fff" stroke-width="1"/>
<text x="425.7" y="55" text-anchor="middle" fill="#fff" font-size="11">238</text>
<rect x="573.7" y="36" width="32.6" height="30" fill="#bf8700" stroke="#fff" stroke-width="1"/>
<text x="590.0" y="55" text-anchor="middle" fill="#fff" font-size="11">26</text>
<rect x="606.3" y="36" width="80.5" height="30" fill="#8250df" stroke="#fff" stroke-width="1"/>
<text x="646.5" y="55" text-anchor="middle" fill="#fff" font-size="11">65</text>
<rect x="686.8" y="36" width="26.6" height="30" fill="#6e7781" stroke="#fff" stroke-width="1"/>
<text x="700.1" y="55" text-anchor="middle" fill="#fff" font-size="11">21</text>
<rect x="713.4" y="36" width="18.0" height="30" fill="#afb8c1" stroke="#fff" stroke-width="1"/>
<rect x="731.5" y="36" width="11.6" height="30" fill="#1a7f37" stroke="#fff" stroke-width="1"/>
<text x="749.0" y="55" font-weight="bold">444 s</text>
<rect x="190.0" y="72" width="9" height="9" fill="#8c959f"/><text x="202.0" y="81" font-size="10.5" fill="#424a53">teardown + job start 45 s</text>
<rect x="366.0" y="72" width="9" height="9" fill="#54aeff"/><text x="378.0" y="81" font-size="10.5" fill="#424a53">Ray workers 22 s</text>
<rect x="488.0" y="72" width="9" height="9" fill="#0969da"/><text x="500.0" y="81" font-size="10.5" fill="#424a53">TPU init 3.6 s</text>
<rect x="598.0" y="72" width="9" height="9" fill="#e16f24"/><text x="610.0" y="81" font-size="10.5" fill="#424a53">load weights 238 s</text>
<rect x="190.0" y="87" width="9" height="9" fill="#bf8700"/><text x="202.0" y="96" font-size="10.5" fill="#424a53">KV alloc 26 s</text>
<rect x="294.0" y="87" width="9" height="9" fill="#8250df"/><text x="306.0" y="96" font-size="10.5" fill="#424a53">compile + warm-up 65 s</text>
<rect x="452.0" y="87" width="9" height="9" fill="#6e7781"/><text x="464.0" y="96" font-size="10.5" fill="#424a53">gateway 21 s</text>
<rect x="550.0" y="87" width="9" height="9" fill="#afb8c1"/><text x="562.0" y="96" font-size="10.5" fill="#424a53">1st dispatch 14 s</text>
<rect x="190.0" y="102" width="9" height="9" fill="#1a7f37"/><text x="202.0" y="111" font-size="10.5" fill="#424a53">ramp 9.3 s</text>
<text x="180" y="146" text-anchor="end" font-weight="bold">Snapshot restore</text>
<text x="180" y="160" text-anchor="end" font-size="10.5" fill="#59636e">S→H, full slice</text>
<rect x="190.0" y="133" width="34.1" height="30" fill="#8c959f" stroke="#fff" stroke-width="1"/>
<text x="207.0" y="152" text-anchor="middle" fill="#fff" font-size="11">27</text>
<rect x="224.1" y="133" width="26.9" height="30" fill="#54aeff" stroke="#fff" stroke-width="1"/>
<text x="237.5" y="152" text-anchor="middle" fill="#fff" font-size="11">22</text>
<rect x="251.0" y="133" width="4.9" height="30" fill="#0969da" stroke="#fff" stroke-width="1"/>
<rect x="255.8" y="133" width="35.3" height="30" fill="#e16f24" stroke="#fff" stroke-width="1"/>
<text x="273.5" y="152" text-anchor="middle" fill="#fff" font-size="11">28</text>
<rect x="291.2" y="133" width="32.1" height="30" fill="#bf8700" stroke="#fff" stroke-width="1"/>
<text x="307.2" y="152" text-anchor="middle" fill="#fff" font-size="11">26</text>
<rect x="323.3" y="133" width="78.8" height="30" fill="#8250df" stroke="#fff" stroke-width="1"/>
<text x="362.7" y="152" text-anchor="middle" fill="#fff" font-size="11">63</text>
<rect x="402.1" y="133" width="27.4" height="30" fill="#6e7781" stroke="#fff" stroke-width="1"/>
<text x="415.7" y="152" text-anchor="middle" fill="#fff" font-size="11">22</text>
<rect x="429.4" y="133" width="26.0" height="30" fill="#afb8c1" stroke="#fff" stroke-width="1"/>
<text x="442.4" y="152" text-anchor="middle" fill="#fff" font-size="11">21</text>
<rect x="455.4" y="133" width="10.1" height="30" fill="#1a7f37" stroke="#fff" stroke-width="1"/>
<text x="471.5" y="152" font-weight="bold">221 s</text>
<rect x="190.0" y="169" width="9" height="9" fill="#8c959f"/><text x="202.0" y="178" font-size="10.5" fill="#424a53">teardown 27 s</text>
<rect x="294.0" y="169" width="9" height="9" fill="#54aeff"/><text x="306.0" y="178" font-size="10.5" fill="#424a53">Ray workers 22 s</text>
<rect x="416.0" y="169" width="9" height="9" fill="#0969da"/><text x="428.0" y="178" font-size="10.5" fill="#424a53">TPU init 3.9 s</text>
<rect x="526.0" y="169" width="9" height="9" fill="#e16f24"/><text x="538.0" y="178" font-size="10.5" fill="#424a53">restore weights 28 s</text>
<rect x="190.0" y="184" width="9" height="9" fill="#bf8700"/><text x="202.0" y="193" font-size="10.5" fill="#424a53">KV alloc 26 s</text>
<rect x="294.0" y="184" width="9" height="9" fill="#8250df"/><text x="306.0" y="193" font-size="10.5" fill="#424a53">compile + warm-up 63 s</text>
<rect x="452.0" y="184" width="9" height="9" fill="#6e7781"/><text x="464.0" y="193" font-size="10.5" fill="#424a53">deploy/gateway 22 s</text>
<rect x="592.0" y="184" width="9" height="9" fill="#afb8c1"/><text x="604.0" y="193" font-size="10.5" fill="#424a53">1st dispatch 21 s</text>
<rect x="190.0" y="199" width="9" height="9" fill="#1a7f37"/><text x="202.0" y="208" font-size="10.5" fill="#424a53">ramp 8.1 s</text>
<text x="180" y="243" text-anchor="end" font-weight="bold">In place (ICI)</text>
<text x="180" y="257" text-anchor="end" font-size="10.5" fill="#59636e">TP16→2×TP8, one box, warm</text>
<rect x="190.0" y="230" width="1.9" height="30" fill="#8c959f" stroke="#fff" stroke-width="1"/>
<rect x="191.9" y="230" width="0.8" height="30" fill="#bf3989" stroke="#fff" stroke-width="1"/>
<rect x="192.6" y="230" width="19.2" height="30" fill="#1b7c83" stroke="#fff" stroke-width="1"/>
<rect x="211.8" y="230" width="4.9" height="30" fill="#e16f24" stroke="#fff" stroke-width="1"/>
<rect x="216.6" y="230" width="1.0" height="30" fill="#0969da" stroke="#fff" stroke-width="1"/>
<rect x="217.6" y="230" width="30.4" height="30" fill="#bf8700" stroke="#fff" stroke-width="1"/>
<text x="232.8" y="249" text-anchor="middle" fill="#fff" font-size="11">24</text>
<rect x="248.0" y="230" width="41.6" height="30" fill="#8250df" stroke="#fff" stroke-width="1"/>
<text x="268.8" y="249" text-anchor="middle" fill="#fff" font-size="11">33</text>
<rect x="289.6" y="230" width="25.6" height="30" fill="#6e7781" stroke="#fff" stroke-width="1"/>
<text x="302.4" y="249" text-anchor="middle" fill="#fff" font-size="11">21</text>
<text x="321.2" y="249" font-weight="bold">101 s</text>
<rect x="190.0" y="266" width="9" height="9" fill="#8c959f"/><text x="202.0" y="275" font-size="10.5" fill="#424a53">free KV + runner 1.5 s</text>
<rect x="348.0" y="266" width="9" height="9" fill="#bf3989"/><text x="360.0" y="275" font-size="10.5" fill="#424a53">repack 0.6 s</text>
<rect x="446.0" y="266" width="9" height="9" fill="#1b7c83"/><text x="458.0" y="275" font-size="10.5" fill="#424a53">graph rebuild 15 s</text>
<rect x="580.0" y="266" width="9" height="9" fill="#e16f24"/><text x="592.0" y="275" font-size="10.5" fill="#424a53">reshard 3.9 s</text>
<rect x="190.0" y="281" width="9" height="9" fill="#0969da"/><text x="202.0" y="290" font-size="10.5" fill="#424a53">functions 0.8 s</text>
<rect x="306.0" y="281" width="9" height="9" fill="#bf8700"/><text x="318.0" y="290" font-size="10.5" fill="#424a53">KV alloc, 2 engines in turn 24 s</text>
<rect x="524.0" y="281" width="9" height="9" fill="#8250df"/><text x="536.0" y="290" font-size="10.5" fill="#424a53">warm-up 33 s</text>
<rect x="190.0" y="296" width="9" height="9" fill="#6e7781"/><text x="202.0" y="305" font-size="10.5" fill="#424a53">engine core, binding, 1st token 21 s</text>
<line x1="190" y1="327" x2="750" y2="327" stroke="#59636e"/>
<line x1="190.0" y1="327" x2="190.0" y2="332" stroke="#59636e"/><text x="190.0" y="344" text-anchor="middle" font-size="10" fill="#59636e">0</text>
<line x1="252.2" y1="327" x2="252.2" y2="332" stroke="#59636e"/><text x="252.2" y="344" text-anchor="middle" font-size="10" fill="#59636e">50</text>
<line x1="314.4" y1="327" x2="314.4" y2="332" stroke="#59636e"/><text x="314.4" y="344" text-anchor="middle" font-size="10" fill="#59636e">100</text>
<line x1="376.7" y1="327" x2="376.7" y2="332" stroke="#59636e"/><text x="376.7" y="344" text-anchor="middle" font-size="10" fill="#59636e">150</text>
<line x1="438.9" y1="327" x2="438.9" y2="332" stroke="#59636e"/><text x="438.9" y="344" text-anchor="middle" font-size="10" fill="#59636e">200</text>
<line x1="501.1" y1="327" x2="501.1" y2="332" stroke="#59636e"/><text x="501.1" y="344" text-anchor="middle" font-size="10" fill="#59636e">250</text>
<line x1="563.3" y1="327" x2="563.3" y2="332" stroke="#59636e"/><text x="563.3" y="344" text-anchor="middle" font-size="10" fill="#59636e">300</text>
<line x1="625.6" y1="327" x2="625.6" y2="332" stroke="#59636e"/><text x="625.6" y="344" text-anchor="middle" font-size="10" fill="#59636e">350</text>
<line x1="687.8" y1="327" x2="687.8" y2="332" stroke="#59636e"/><text x="687.8" y="344" text-anchor="middle" font-size="10" fill="#59636e">400</text>
<line x1="750.0" y1="327" x2="750.0" y2="332" stroke="#59636e"/><text x="750.0" y="344" text-anchor="middle" font-size="10" fill="#59636e">450</text>
<text x="750" y="359" text-anchor="end" font-size="10.5" fill="#59636e">seconds</text>
</svg>
```),
  caption: [Where the switch cost goes, measured. Top two rows: full 32-chip slice, S → H. Bottom
    row: one box, TP16 → 2×TP8, second switch with the compile-cache fix, in a smaller engine
    configuration (`max_model_len` 4096). "Engine core, binding, 1st token" is the untimed
    remainder.],
) <fig-cost>

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  table.header[Switch (one box)][Without cache fix: 1st][2nd][With cache fix: 1st][2nd],
  [TP16 → 2×TP8], [237.0 s], [229.5 s], [241.9 s], [*100.6 s*],
  [2×TP8 → TP16], [76.5 s], [75.9 s], [77.4 s], [*76.8 s*],
)

*Correctness.*
- Every serving batch was token-identical to normally loaded engines of the same TP size: 7
  batches of 32 prompts in each of the two arms, references from `ShardedRestore`.
- Each of the four processes kept its PID and ran exactly one runtime initialisation and one
  checkpoint load.

*What the warm 100.6 s contains* (warm switch to 2×TP8, cache fix on; the two TP8 engines are
built one after the other):

#table(
  columns: (auto, auto),
  align: (left, right),
  table.header[Step][Time on the critical path],
  [Drain], [0.002 s],
  [Free KV cache and runner], [1.4–1.5 s],
  [Repack fused gate/up and QKV weights on device], [0.6 s],
  [Rebuild the model graph for the new mesh], [15.0–15.4 s],
  [Reshard weights (`device_put` + sub-mesh handles)], [3.4–3.9 s],
  [Rebuild model functions], [0.4–0.8 s],
  [KV cache allocation: engine A, then engine B (12.2 s each)], [24.4 s],
  [Compile + warm-up (both engines together)], [33.4 s],
  [Not timed separately: engine-core construction, executor attachment, first tokens], [about 20 s],
)

*Not a like-for-like configuration.* The prototype ran with `max_model_len` 4096, 32 sequences
and 1,024 batched tokens per step, against 16,384, 512 and 2,048 in the deployed engines. Its
shorter warm-up (33 s against 63 s) and KV allocation (12 s against 26 s per engine) therefore
come partly from the smaller configuration, not only from keeping the weights resident. The weight
steps (repack, reshard) don't depend on these settings.

*Three problems the prototype had to solve:*
+ *Fused weights depend on the TP size.* The fused gate/up and QKV weights store per-projection
  slices interleaved by TP rank, so a plain transfer scrambles them. The `Helper` agent found this
  before any results were produced; the prototype now unpacks and repacks those weights on the
  chips during the switch (0.6 s).
+ *Only global process 0 writes the compile cache.* JAX 0.10.2 lets only process 0 write
  compiled code (`compiler.py:802–805`), so the TP8 engine on processes 2–3 never saved its
  kernels and compiled from scratch on every switch (159–164 s). An opt-in prototype patch lets
  the lowest process of each engine write instead. That makes the second switch to 2×TP8 take
  101 s instead of 230 s. The 451 cache keys had no competing writers.
+ *Smaller fixes.* The checkpoint loader assumed one process (it took global CPU device 0 on
  every host). Ray's job setup crashed on a 147 KB environment-variable payload; inputs now ship
  as files. vLLM's executor contract needed `non_block` support and a health check.

*Serving speed after a switch.* A TP8 engine inside the shared runtime decoded a fixed batch at
370.7 and 373.5 tokens/s after a warm switch, 1.1% and 0.3% below a standalone TP8 engine on the
same hosts (374.7 tokens/s). Right after a cold switch it was 3.5–5.4% slower. Both in-runtime
engines ran at once, the standalone engine alone.

*Limits of the prototype.*
- One box only, no gateway, `max_model_len` 4096.
- Its TP8 engines cover horizontal 4×2 chip rectangles, while the normal S layout uses vertical
  2×4 ones.
- No failure recovery: a crash in any process takes down every engine in the runtime.

= Pathways (read-only check) <sec-pathways>

Pathways would let one controller own all 32 chips and reshard with `jax.device_put` or
`pathwaysutils` reshard, without a shared multi-controller runtime. The `PathwaysPrereq` agent
checked what a trial would take, without changing the cluster.

- *Available:*
  - The TPU pool is the right shape (eight `ct5lp-hightpu-4t` hosts, 4×8).
  - Public Pathways images exist for JAX 0.10.2.
  - The worker image's `pathwaysutils` 0.1.11 has the experimental reshard.
- *Missing:*
  - The JobSet controller isn't installed; that is a cluster-scoped change.
  - The Pathways head needs about 24 CPU and 132 GB, but the CPU node has about 10.5 CPU free.
  - Scaling the Ray TPU workers down and up needs rights on the RayCluster that the namespace
    token doesn't have.
  - A trial needs a 30–45 min serving outage [ESTIMATE].
- *Verdict:* no-go with namespace-only resources; conditional go after approvals for the JobSet
  install, a CPU node pool and scratch storage. Draft manifests, unapplied, and a trial plan are on
  branch `ici/pathways-prereq`.

= What it means for break-even <sec-breakeven>

Using the measured per-phase gains from the break-even study (W0: S finishes 153.9 s sooner than H
per 721,595-token manifest; mix A: 82 s gained per W0 copy after a W4 phase, with 115 s of phase
overlap forfeited by switching). A switch into S splits TP16 engines into TP8, so each row uses the
cost measured in that direction (H → S or TP16 → 2×TP8); the opposite direction serves W0 more
slowly and has no W0 break-even.

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, left),
  table.header[Switch mechanism][Cost C used][W0 H → S break-even][Mix A break-even (W0 copies)][Status],
  [Full redeploy], [419 s (median)], [2.0M tokens (2.7 manifests)], [6.4], [measured],
  [Partial (keep shared TP8 engines)], [186 s effective], [–], [3.7], [measured],
  [Snapshot restore], [228.6 s (H → S)], [*1.07M tokens* (1.49)], [4.2], [C measured; break-even computed; mix A applies H → S cost to M → S],
  [Partial + restore], [100–103 s effective], [–], [2.6–2.7], [ESTIMATE],
  [In place, first switch], [237.0–241.9 s (TP16 → 2×TP8)], [1.11–1.13M tokens (1.54–1.57)], [–], [C measured on one box],
  [In place, warm], [100.6 s (TP16 → 2×TP8)], [*0.47M tokens* (0.65)], [2.6], [ESTIMATE: one-box C, no gateway, ramp or second box],
)

- *Restore halves break-even with no change to the serving architecture.* A chat phase of about
  1.5 W0 manifests (1.07M output tokens) now pays for H → S, against 2.7 before.
- *A warm in-place switch would more than halve it again* (0.65 manifests), but only once the
  shared-runtime abort is solved and the gateway and full slice are integrated, which will add to
  its 100.6 s. Its first switch is no better than restore. The faster 77 s direction (2×TP8 →
  TP16) only helps phases that run better on H.
- *Neither removes the remaining fixed cost.* Compile and warm-up (33–63 s), KV allocation
  (12–26 s per engine) and orchestration dominate both. In the in-place prototype, building the two
  TP8 engines one after the other (24 s of KV allocation), rebuilding the model graph (15 s) and
  engine construction (about 20 s) are the next targets. Moving the weights is already 4 s.

= Threats to validity <sec-validity>

- *Different measurement spans.* Restore and full redeploy are measured over the whole slice, as
  the gap from the last finished request to the first new dispatch plus the ramp deficit (seconds
  of lost steady output), gateway included. In-place is one box, wall time from drain to the first
  token on every engine, without gateway, ramp or the second box. The in-place break-even rows apply
  the one-box cost to the slice.
- *Repetitions.* Restore transitions: one per direction. In-place: two switches in each direction
  per arm, two arms. ICI transfer: 10 warm repetitions per route.
- *Prototype patches.* The compile-cache fix wraps a JAX internal and refuses other JAX versions.
  The in-place executor is a synchronous Ray RPC executor, not the production Ray DAG path.
- *Synthetic transfer weights.* ICIBench used real shapes with synthetic values; correctness with
  real weights comes from InPlaceEngine's greedy checks.
- *Snapshots live in host RAM* and vanish if a host restarts; restore then falls back to the
  normal loader.
- *Unfrozen code; nothing merged.* All runs are pilots on unmerged branches.

= Reproduction

Each agent worked in its own git worktree of `~/hetero-research`, under `~/hetero-wt/`.
Machine-readable evidence from every branch is copied next to this report in
`reconfiguration_ici_evidence/`.

#table(
  columns: (auto, auto, auto, 1fr),
  align: (left, left, left, left),
  table.header[Agent][Branch][Commit][What it holds],
  [LoadProfiler], [`restore/load-profiler`], [`884ee70`], [load-time instrumentation, profile campaign and analysis],
  [ShardedRestore], [`restore/sharded-snapshot`], [`cc0f44f`], [snapshot/restore, validation and transition campaign],
  [PathwaysReshard], [`research/pathways-reshard`], [`aeb0de4`], [feasibility study, CPU reshard prototype],
  [ICIBench], [`ici/benchmark`], [`aa6ab2e`], [multi-controller launcher, transfer benchmark, trace analysis],
  [InPlaceEngine], [`ici/in-place-engine`], [`e21156a`], [persistent-runtime engine prototype, cache-writer patch],
  [PathwaysPrereq], [`ici/pathways-prereq`], [`bd4a580`], [prerequisite checks, draft manifests, trial plan],
)

```sh
export KUBECONFIG=~/.kube/hetero-agent.yaml
# load profile (LoadProfiler worktree)
flock /tmp/hetero-cluster.lock bash scripts/load_profile_campaign.sh
# ICI transfer, one box and whole slice (ICIBench worktree)
flock /tmp/hetero-cluster.lock scripts/ici_sequence.sh reshard box out.json --sizes 256MiB,1GiB,model --profile
flock /tmp/hetero-cluster.lock scripts/ici_sequence.sh reshard slice out.json --sizes model
# in-place engine, with the compile-cache fix (InPlaceEngine worktree)
flock /tmp/hetero-cluster.lock scripts/ici_engine_sequence.sh --submesh-cache-writer \
  --run-id inplace-patched-UNIQUE --output results/ici/InPlaceEngine/run.json
.venv/bin/python scripts/ici_engine_analyze.py results/ici/InPlaceEngine/run.json \
  --standalone results/ici/InPlaceEngine/run_standalone8_corrected.json
```

Run ids: load profile `raysubmit_WhmszEvQNQQRbC8D` (S), `raysubmit_2nBVMqqZSjXNP12X` (H); ICI box
`raysubmit_AxSaeccWYsesZFLf`, slice `raysubmit_VtvrR5PzNZiWgkkG`, groups
`raysubmit_FphtKh5zYWCXvj6A`; in-place `raysubmit_eZ37RTqNQBVq2r5T` (no cache fix),
`raysubmit_6Kx3zaVPJJwS7UvW` (cache fix), standalone TP8 `raysubmit_dSDKwhfgqiejg4qN`.
