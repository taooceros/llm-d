// PhaseMorph: Sub-30-Second TPU Mesh Reconfiguration for Heterogeneous Bulk LLM Serving
// MLSys (mlsys.sty / icml.sty) two-column layout for PDF/paged view; single-column academic HTML for ?g-output=html.
//   HTML: typst compile --features html --format html phasemorph_mlsys_paper.typ /tmp/phasemorph.html
//   PDF:  typst compile phasemorph_mlsys_paper.typ /tmp/phasemorph.pdf

#let title = "PhaseMorph: Sub-30-Second TPU Mesh Reconfiguration for Heterogeneous Bulk LLM Serving"
#let running-title = title
#let author = "Hongtao Zhang"
#let affiliation = [Heterogeneous & Reconfigurable TPU Serving Study, #link("https://github.com/taooceros/llm-d")[github.com/taooceros/llm-d]]
#let testbed = [Cloud TPU v5e $4 times 8$ slice (32 chips), Gemma-4 31B (BF16)]
#let notice = [Preprint formatted in the MLSys style; not peer reviewed. Copyright 2026 by the author(s).]

#set document(title: title, author: author)
#set heading(numbering: "1.1.")
#set figure.caption(separator: [. ])
#show figure.where(kind: table): set figure.caption(position: top)
#show table.cell.where(y: 0): strong
#show math.equation: set text(font: ("TeX Gyre Termes Math", "New Computer Modern Math"))

#let css = ```
body {
  max-width: 46rem;
  margin: 2.5rem auto 4rem;
  padding: 0 1.25rem;
  font-family: "Times New Roman", "TeX Gyre Termes", "Nimbus Roman", Times, "Libertinus Serif", Georgia, serif;
  font-size: 17px;
  line-height: 1.5;
  color: #111;
  text-align: justify;
  hyphens: auto;
}
header.mlsys-front { margin-bottom: 0.5rem; }
h1.mlsys-title {
  margin: 0;
  padding: 0.8rem 0.5rem;
  border-top: 3px solid #000;
  border-bottom: 1px solid #000;
  font-size: 1.55rem;
  line-height: 1.25;
  font-weight: 700;
  text-align: center;
}
.mlsys-authors { text-align: center; font-weight: 700; margin: 1.1rem 0 0.15rem; }
.mlsys-affil { text-align: center; font-size: 0.9em; font-style: italic; }
.mlsys-notice {
  text-align: center;
  font-size: 0.78em;
  font-style: italic;
  color: #333;
  margin: 0.7rem auto 0;
  padding-top: 0.35rem;
  border-top: 0.5px solid #000;
  max-width: 34rem;
}
section.mlsys-abstract { margin: 1.6rem 2.5rem 2rem; font-size: 0.94em; }
.mlsys-abstract-title { text-align: center; font-weight: 700; font-size: 1.12em; margin-bottom: 0.35rem; }
h2, h3, h4, h5 { font-family: inherit; font-weight: 700; line-height: 1.3; text-align: left; color: #000; }
h2 { font-size: 1.2rem; margin: 1.9em 0 0.6em; }
h3 { font-size: 1.04rem; margin: 1.4em 0 0.5em; }
h4 { font-size: 1rem; font-style: italic; margin: 1.1em 0 0.4em; }
p { margin: 0 0 0.75em; }
a { color: #0b4f9c; text-decoration: none; }
li { margin-bottom: 0.3em; }
table {
  border-collapse: collapse;
  display: block;
  width: max-content;
  max-width: 100%;
  overflow-x: auto;
  margin: 0.5em auto;
  font-size: 0.84em;
  line-height: 1.35;
  text-align: left;
}
th, td { border: none; background: none; padding: 0.28em 0.6em; vertical-align: top; }
th { font-weight: 700; }
thead tr:first-child > * { border-top: 1.5px solid #000; }
thead tr:last-child > * { border-bottom: 0.75px solid #000; }
tbody tr:last-child > * { border-bottom: 1.5px solid #000; }
code {
  font-family: ui-monospace, "DejaVu Sans Mono", Menlo, Consolas, monospace;
  font-size: 0.82em;
  background: none;
  padding: 0;
}
figure { margin: 1.7em 0; text-align: center; }
figure img, figure svg.typst-frame { width: 100% !important; height: auto !important; }
figcaption { font-size: 0.9em; text-align: justify; margin: 0.5em 0; }
section[role="doc-bibliography"] ul { padding-left: 0; font-size: 0.9em; }
section[role="doc-bibliography"] li { margin-bottom: 0.35em; padding-left: 2em; text-indent: -2em; }
```.text

#let is-html() = "target" in dictionary(std) and std.target() == "html"

#show math.equation: it => context if is-html() {
  if it.block { html.elem("div", attrs: (style: "text-align:center;margin:.7em 0"), html.frame(it)) }
  else { box(html.frame(it)) }
} else { it }

#let svg-fig(source) = image(bytes(source.text), format: "svg", width: 100%)
#let wide-fig(body, caption: none) = figure(body, caption: caption, placement: top, scope: "parent")

// Booktabs-style table: no vertical rules; 1.0pt top rule, 0.5pt header rule, 1.0pt bottom rule.
#let booktabs(columns: auto, align: auto, header: (), ..rows) = table(
  columns: columns,
  align: align,
  stroke: none,
  inset: (x: 4pt, y: 2.6pt),
  table.hline(stroke: 1pt),
  table.header(..header),
  table.hline(stroke: 0.5pt),
  ..rows,
  table.hline(stroke: 1pt),
)

#let abstract-content = [
  Bulk LLM inference on multi-host accelerator slices must commit to a tensor-parallel (TP) layout, yet the best layout changes as a backlog moves between short- and long-context phases. On a 32-chip Cloud TPU v5e slice serving Gemma-4 31B in BF16, four 8-chip instances deliver 28.8%–37.7% higher short-context throughput than two 16-chip instances, while 16-chip instances hold $2.75 times$ more KV cache, finish mid-context documents 17.6% faster, and are the only feasible layout beyond 52.9k tokens of context. A static heterogeneous partition wins only a narrow 6% band of mixed workloads, and live cross-TP KV migration loses to recomputation, so reshaping the slice between workload phases is the natural remedy. Unfortunately, redeploying vLLM engines across the slice takes 350–419 s. We show that this cost is dominated neither by disk I/O nor by device initialization, but by host-CPU tensor layout conversion, sharding JIT compilation, eviction of sub-second compilations from the persistent cache, process-local JAX tracing, and worker spawning.

  We present _PhaseMorph_, a reconfiguration system for vLLM on Cloud TPU that materializes device-ready serving state ahead of time—pre-sharded weight snapshots, shape-only model graphs, and a device-order-preserving bank of serialized XLA executables—and prepares target workers in clean-room CPU-only processes that take over the TPU through a VFIO-aware handoff. PhaseMorph reduces warm full-slice reconfiguration by 95%, from 419.0 s to 21.1–27.5 s, and partial reconfiguration to 18.8 s, with no measurable foreground slowdown; a background CPU pre-read further cuts the cold first switch by 40% (222.6 s $arrow.r$ 134.1 s). An in-runtime submesh prototype bounds the hardware cost of a layout change at 12.0 ms, and process-isolated switching already captures 99.7% of the achievable bulk makespan gain. Across a two-phase workload space, phase-aware switching beats the best static layout on 82%–88% of jobs, with median and maximum makespan reductions of 9.9% and 26.8%.
]

#show: doc => context if is-html() {
  html.elem("style", css)
  html.elem("header", attrs: (class: "mlsys-front"), {
    html.elem("h1", attrs: (class: "mlsys-title"), title)
    html.elem("div", attrs: (class: "mlsys-authors"), author)
    html.elem("div", attrs: (class: "mlsys-affil"), affiliation)
    html.elem("div", attrs: (class: "mlsys-notice"))[#notice Testbed: #testbed.]
  })
  html.elem("section", attrs: (class: "mlsys-abstract"), {
    html.elem("div", attrs: (class: "mlsys-abstract-title"), [Abstract])
    abstract-content
  })
  doc
} else {
  set page(
    paper: "us-letter",
    margin: (x: 0.875in, top: 1in, bottom: 1in),
    columns: 2,
    header-ascent: 25%,
    header: context if here().page() > 1 {
      set par(justify: false)
      stack(
        spacing: 3pt,
        align(center, text(size: 8.5pt, weight: "bold", running-title)),
        line(length: 100%, stroke: 0.4pt),
      )
    },
    footer: context align(center, text(size: 9pt, counter(page).display("1"))),
  )
  set columns(gutter: 0.25in)
  set text(font: ("TeX Gyre Termes", "Nimbus Roman", "Libertinus Serif"), size: 10pt)
  set par(justify: true, leading: 0.5em, spacing: 0.85em, first-line-indent: 0pt)
  set enum(indent: 0.3em, body-indent: 0.45em)
  set list(indent: 0.3em, body-indent: 0.45em)
  show heading: set text(size: 10pt, weight: "bold")
  show heading.where(level: 1): set text(size: 11pt)
  show heading.where(level: 1): set block(above: 1.4em, below: 0.75em)
  show heading.where(level: 2): set block(above: 1.15em, below: 0.6em)
  show heading.where(level: 3): set block(above: 1em, below: 0.55em)
  show figure: set block(spacing: 1.1em)
  show figure.caption: set text(size: 9pt)
  show figure.caption: set align(left)
  show figure.caption: set par(justify: true)
  set figure(placement: auto)
  show table: set text(size: 8pt, hyphenate: false)
  show table: set par(justify: false)
  show heading: set text(hyphenate: false)
  show heading: set par(justify: false)
  show bibliography: set text(size: 8.5pt)
  show bibliography: set par(spacing: 0.55em)

  // Title banner spanning both columns: 2pt top rule, 14pt bold title, 1pt bottom rule, author block.
  place(top + center, scope: "parent", float: true, clearance: 0.3in, block(width: 100%, {
    set par(justify: false)
    line(length: 100%, stroke: 2pt)
    v(0.14in)
    align(center, text(size: 14pt, weight: "bold", title))
    v(0.12in)
    line(length: 100%, stroke: 1pt)
    v(0.18in)
    align(center)[#text(size: 10pt, weight: "bold", author)#super[1]]
    v(1pt)
    align(center, text(size: 9pt, style: "italic", affiliation))
  }))

  // Bottom-of-first-column affiliation and preprint notice (mlsys.sty footnote block).
  place(bottom + left, float: true, clearance: 0.9em, block(width: 100%, {
    set par(justify: true, leading: 0.4em, spacing: 0.4em)
    set text(size: 7.5pt)
    line(length: 35%, stroke: 0.5pt)
    v(1pt)
    [#super[1]#affiliation. Correspondence to: #author.]
    parbreak()
    emph[#notice Testbed: #testbed.]
  }))

  // MLSys abstract: centered bold heading, 9pt text, inset margins.
  {
    align(center, text(size: 11pt, weight: "bold")[Abstract])
    v(-0.25em)
    pad(x: 0.2in, {
      set text(size: 9pt)
      set par(spacing: 0.7em)
      abstract-content
    })
    v(0.4em)
  }
  doc
}

= Introduction <sec-intro>

Large language models are now applied to bulk and offline jobs as routinely as to interactive chat: synthetic data generation, long-document summarization, repository-wide code analysis, and model evaluation all submit deep request backlogs whose composition shifts from one phase to the next. Unlike interactive serving, which is governed by per-request time-to-first-token and time-per-output-token targets, these jobs care about a single quantity—the _makespan_ from first dispatch until the backlog is drained—or equivalently the sustained useful output tokens per second of a fixed accelerator allocation. Static provisioning serves such jobs poorly: a deployment sized for the longest contexts wastes compute on short requests, and a deployment tuned for short requests runs out of KV capacity on long ones. Modern serving stacks therefore lean on elasticity, both by autoscaling the number of instances @fu2024serverlessllm @zhang2025blitzscale @sun2024llumnix and by _parallelism hot-switching_, which reconfigures the parallelism strategy to match the current workload @wu2024loongserve @chen2025gyges.

On tightly coupled accelerator pods such as Google Cloud TPU v5e @googlev5e, the choice of tensor-parallel (TP) degree is particularly consequential. On a 32-chip v5e slice serving Gemma-4 31B @gemma2024 in BF16, partitioning the slice into four 8-chip instances (`S`: $4 times$`TP8`) drains a short-context chat manifest 28.8%–37.7% faster than two 16-chip instances (`H`: $2 times$`TP16`), because smaller rings need fewer AllReduce hops and keep the matrix units' contractions deep. Memory pulls in the opposite direction: after the 61.4 GB of weights, a `TP8` instance retains only 52,864 tokens of usable KV cache, whereas a `TP16` instance retains 145,216 ($2.75 times$). `H` therefore finishes mid-context documents 17.6% faster than `S` and is the only feasible layout once contexts exceed 52.9k tokens. A static mixed layout (`M`: `TP16` + $2 times$`TP8`) is optimal in only 6% of mixed workloads, and migrating live requests across TP degrees is slower than recomputing them (@sec-static). The natural remedy is to _reshape the slice between workload phases_: serve short phases on `S`, then switch to `M` or `H` when long contexts arrive.

Both elasticity and hot-switching, however, are bottlenecked by cold-start latency. Redeploying vLLM @kwon2023vllm engines across the slice takes 350–419 s, and at that cost a job must contain more than 4.2 million short-phase output tokens before switching pays off at all (@sec-eval).

At first glance, this problem seems amenable to simple caching and RAM staging: stage the checkpoint in host memory, keep a warm persistent XLA @openxla compilation cache, and restart the engine. However, a compiled TPU serving engine is tightly coupled to its host process, to its device mesh, and to the compiled-program state it accumulates while warming up. First, TPU weight loading is not bound by storage: of the 237 s spent in `load_model`, file reads take 0.2 s and host-to-device DMA 2.8 s, while host-CPU tensor layout conversion takes 128.4 s, per-parameter sharding JIT compilation 75.3 s, and Flax @heek2023flax graph construction (`nnx.eval_shape`) 14.6 s. Staging the checkpoint in `/dev/shm` consequently yields no speedup. Second, even with a warm persistent cache, worker initialization takes more than 80 s: JAX @bradbury2018jax silently refuses to cache programs that compile in under one second, so 248 small helper programs per worker are recompiled on every start (42.4 s), and JAX's Python tracing and StableHLO lowering (28.6–45.5 s) are process-local and cannot be cached at all. Third, the obvious way to hide this work—preparing the target engine while the source layout is still serving—collides with the TPU runtime: any second process that so much as queries the TPU topology aborts on PJRT's `/tmp/libtpu_lockfile`, and freshly started workers race the kernel's VFIO device reset (`EBUSY`). Keeping weights resident inside a single multi-host runtime instead hits a `libtpu` sub-torus collective routing bug (`net_util.cc:2304`) and merges hosts into one failure domain.

We present _PhaseMorph_, a reconfiguration system for vLLM on Cloud TPU that removes these overheads from the critical path of a layout change. Realizing PhaseMorph poses three challenges. _First, how can a new engine obtain device-ready weights without repeating host-side conversion?_ PhaseMorph materializes each worker's post-conversion, per-chip weight shards as a snapshot in host shared memory and rebuilds the model graph from shapes alone; for the first switch into a layout, a background CPU process produces the snapshot while the source layout is still serving. _Second, how can compiled programs survive a process boundary when tracing state is process-local?_ PhaseMorph removes the persistent cache's compile-time threshold and serializes every ahead-of-time compiled executable, together with its exact device order, into an executable bank that new workers load without tracing. _Third, how can target workers be prepared concurrently with a live source engine that owns the TPU?_ PhaseMorph pre-spawns target workers in clean-room CPU-only processes that verifiably hold no TPU handle, parks them on a release gate, and hands the devices over through a barrier that waits out the kernel's VFIO reset.

#wide-fig(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 314" font-family="Helvetica, Arial, sans-serif" font-size="11.5" fill="#1f2328">
  <text x="12" y="16" font-size="12.5" font-weight="bold">Critical-Path Reconfiguration Cost C per System Variant (Measured on 32-Chip TPU v5e)</text>
  <line x1="215" y1="36" x2="800" y2="36" stroke="#d0d7de" stroke-width="1"/>
  <text x="215" y="32" text-anchor="middle" font-size="10" fill="#59636e">0 s</text>
  <text x="349" y="32" text-anchor="middle" font-size="10" fill="#59636e">100 s</text>
  <text x="483" y="32" text-anchor="middle" font-size="10" fill="#59636e">200 s</text>
  <text x="617" y="32" text-anchor="middle" font-size="10" fill="#59636e">300 s</text>
  <text x="751" y="32" text-anchor="middle" font-size="10" fill="#59636e">400 s</text>
  <!-- Row 1: Vanilla vLLM redeploy -->
  <text x="205" y="56" text-anchor="end" font-weight="bold">Vanilla vLLM Redeploy</text>
  <text x="205" y="69" text-anchor="end" font-size="10" fill="#59636e">External job + checkpoint load</text>
  <rect x="215" y="42" width="87" height="28" fill="#8c959f" stroke="#fff"/>
  <text x="258" y="60" text-anchor="middle" fill="#fff" font-size="10">Harness 65s</text>
  <rect x="302" y="42" width="30" height="28" fill="#0969da" stroke="#fff"/>
  <rect x="332" y="42" width="318" height="28" fill="#cf222e" stroke="#fff"/>
  <text x="491" y="60" text-anchor="middle" fill="#fff" font-size="10.5" font-weight="bold">load_model: Host Convert (128s) + Load JIT (75s) = 237s</text>
  <rect x="650" y="42" width="35" height="28" fill="#d4a72c" stroke="#fff"/>
  <rect x="685" y="42" width="91" height="28" fill="#8250df" stroke="#fff"/>
  <text x="730" y="60" text-anchor="middle" fill="#fff" font-size="10">Trace/Compile 68s</text>
  <text x="783" y="60" font-weight="bold" fill="#cf222e">419.0 s</text>

  <!-- Row 2: PhaseMorph-Snap -->
  <text x="205" y="98" text-anchor="end" font-weight="bold">PhaseMorph-Snap</text>
  <text x="205" y="111" text-anchor="end" font-size="10" fill="#59636e">/dev/shm shards + FastGraph</text>
  <rect x="215" y="84" width="20" height="28" fill="#8c959f" stroke="#fff"/>
  <rect x="235" y="84" width="29" height="28" fill="#0969da" stroke="#fff"/>
  <text x="249" y="102" text-anchor="middle" fill="#fff" font-size="9.5">22s</text>
  <rect x="264" y="84" width="7" height="28" fill="#1a7f37" stroke="#fff"/>
  <rect x="271" y="84" width="35" height="28" fill="#d4a72c" stroke="#fff"/>
  <text x="288" y="102" text-anchor="middle" fill="#fff" font-size="9.5">26s</text>
  <rect x="306" y="84" width="86" height="28" fill="#8250df" stroke="#fff"/>
  <text x="349" y="102" text-anchor="middle" fill="#fff" font-size="10">Trace + Warmup 55s</text>
  <text x="399" y="102" font-weight="bold" fill="#1f2328">132.2 s (-68%)</text>

  <!-- Row 3: PhaseMorph-Cache -->
  <text x="205" y="140" text-anchor="end" font-weight="bold">PhaseMorph-Cache</text>
  <text x="205" y="153" text-anchor="end" font-size="10" fill="#59636e">+ zero-threshold XLA cache</text>
  <rect x="215" y="126" width="21" height="28" fill="#8c959f" stroke="#fff"/>
  <rect x="236" y="126" width="29" height="28" fill="#0969da" stroke="#fff"/>
  <text x="250" y="144" text-anchor="middle" fill="#fff" font-size="9.5">22s</text>
  <rect x="265" y="126" width="10" height="28" fill="#1a7f37" stroke="#fff"/>
  <rect x="275" y="126" width="60" height="28" fill="#8250df" stroke="#fff"/>
  <text x="305" y="144" text-anchor="middle" fill="#fff" font-size="9.5">Host Trace 35s</text>
  <text x="343" y="144" font-weight="bold" fill="#1f2328">89.8 s (-79%)</text>

  <!-- Row 4: PhaseMorph (full system) -->
  <text x="205" y="182" text-anchor="end" font-weight="bold">PhaseMorph (Full)</text>
  <text x="205" y="195" text-anchor="end" font-size="10" fill="#59636e">+ pre-spawn + executable bank</text>
  <rect x="215" y="168" width="9" height="28" fill="#8c959f" stroke="#fff"/>
  <rect x="224" y="168" width="10" height="28" fill="#1a7f37" stroke="#fff"/>
  <rect x="234" y="168" width="10" height="28" fill="#2da44e" stroke="#fff"/>
  <text x="252" y="186" font-weight="bold" fill="#1a7f37">21.1–27.5 s full / 18.8 s partial (-95%)</text>

  <!-- Row 5: PhaseMorph-PreRead (cold first switch) -->
  <text x="205" y="224" text-anchor="end" font-weight="bold">PhaseMorph-PreRead</text>
  <text x="205" y="237" text-anchor="end" font-size="10" fill="#59636e">Cold 1st switch, CPU pre-read child</text>
  <rect x="215" y="210" width="9" height="28" fill="#8c959f" stroke="#fff"/>
  <rect x="224" y="210" width="114" height="28" fill="#e16f24" stroke="#fff"/>
  <text x="281" y="228" text-anchor="middle" fill="#fff" font-size="9.5">Wait 80s + 5s restore</text>
  <rect x="338" y="210" width="56" height="28" fill="#8250df" stroke="#fff"/>
  <text x="366" y="228" text-anchor="middle" fill="#fff" font-size="9.5">38s</text>
  <text x="402" y="228" font-weight="bold" fill="#1f2328">134.1 s (vs 222.6 s no pre-read, -40%)</text>

  <!-- Row 6: PhaseMorph-Submesh (unified runtime prototype) -->
  <text x="205" y="266" text-anchor="end" font-weight="bold">PhaseMorph-Submesh</text>
  <text x="205" y="279" text-anchor="end" font-size="10" fill="#59636e">In-HBM submesh slice over ICI</text>
  <rect x="215" y="252" width="3" height="28" fill="#1a7f37"/>
  <text x="225" y="270" font-weight="bold" fill="#1a7f37">0.012 s (12.0 ms fwd) / 0.048 s (48.0 ms rev) — Hardware Lower Bound</text>

  <!-- Legend -->
  <rect x="215" y="292" width="10" height="10" fill="#8c959f"/><text x="229" y="301" font-size="10">Control/Handoff</text>
  <rect x="315" y="292" width="10" height="10" fill="#0969da"/><text x="329" y="301" font-size="10">Process Spawn (22s)</text>
  <rect x="435" y="292" width="10" height="10" fill="#cf222e"/><text x="449" y="301" font-size="10">Host Weight Convert</text>
  <rect x="560" y="292" width="10" height="10" fill="#d4a72c"/><text x="574" y="301" font-size="10">Sub-1s Compiles</text>
  <rect x="665" y="292" width="10" height="10" fill="#8250df"/><text x="679" y="301" font-size="10">Host Trace/Lower</text>
  <rect x="775" y="292" width="10" height="10" fill="#1a7f37"/><text x="789" y="301" font-size="10">Fast Path</text>
</svg>
```),
  caption: [[Reconfiguration Latency Reduction] Critical-path reconfiguration cost $C$ of an `S → H` full-slice switch (unless noted) as PhaseMorph's mechanisms are added. Removing software-stack overheads compresses switching from 419.0 s (vanilla vLLM redeploy) to 21.1–27.5 s (18.8 s partial), the cold first switch from 222.6 s to 134.1 s, and in-runtime submesh switching to 12.0 ms.],
) <fig-waterfall>

We implement PhaseMorph across vLLM, `tpu-inference` @tpuinference, Ray @moritz2018ray, and the `llm-d` @llmd router, and evaluate it on a 32-chip TPU v5e slice. PhaseMorph reduces warm full-slice reconfiguration by 95%, from 419.0 s to 21.1–27.5 s, and partial reconfiguration to 18.8 s, without measurable slowdown of the source layout; the background pre-read cuts the cold first switch by 40% to 134.1 s. A unified-runtime prototype bounds the in-HBM hardware cost of a layout change at 12.0 ms. With these costs, phase-aware switching beats the best static layout on 82%–88% of a two-phase workload space. In summary, we make the following contributions:

- We characterize the topology–capacity dilemma of LLM serving on multi-host TPU slices and decompose the cost of changing the serving layout, showing that 95% of it is avoidable software-stack overhead rather than hardware initialization or data movement.
- We design PhaseMorph, which materializes device-ready weights, graphs, and executables ahead of time and prepares target workers in clean-room CPU processes, cutting reconfiguration from minutes to seconds while keeping engines process-isolated.
- We evaluate PhaseMorph end to end on Cloud TPU v5e, quantify its break-even job sizes across a two-phase workload space, and compare it against an in-runtime submesh prototype that establishes the hardware lower bound.


= Background and Motivation <sec-motivation>

== Parallelism Trade-Offs on Cloud TPU Slices <sec-tradeoffs>

Our testbed is an 8-host, 32-chip Cloud TPU v5e slice @googlev5e arranged as a $4 times 8$ 2D torus with wrap-around inter-chip interconnect (ICI) links, the same torus-of-chips organization introduced with earlier TPU generations @jouppi2023tpuv4. Each host drives four chips, each with 16 GB of HBM2e at 819 GB/s, 197 TFLOPS of BF16 peak compute across four $128 times 128$ matrix units (MXUs), and 1.00 MB of on-chip scalar memory (SMEM) per TensorCore. We serve Gemma-4 31B @gemma2024 in native BF16: 48 layers, hidden size 4,096, FFN intermediate size 21,504, 32 query heads and 16 KV heads (grouped-query attention with ratio $2:1$), head dimension 128, and a 61.4 GB parameter footprint.

_Compute favors small meshes._ Raising the TP degree across hosts lowers per-chip utilization through three mechanisms. Each Transformer layer issues two AllReduce collectives (after the attention output projection and after the FFN down projection), or 96 synchronizations per step. A ring of $N$ chips needs $2(N-1)$ routing steps, so moving from `TP=4` (one host, 6 steps) to `TP=8` (two hosts, 14 steps) and `TP=16` (four hosts, 30 steps) introduces an inter-host collective floor of $approx 20" ms"$ at small sequence buckets ($S lt.eq 1024$). TP also shortens the contraction dimension of the reduce-side projections: the MXUs need contraction depths $K gt.double 128$ to amortize systolic pipeline fill and drain, but the per-chip depth $K = d_"model" \/ "TP"$ falls from 512 at `TP=8` to 256 at `TP=16`, only two $128 times 128$ tiles. Finally, the FFN intermediate size shards cleanly at `TP=8` ($21504 \/ 8 = 2688 = 21 times 128$) but not at `TP=16` ($1344 = 10.5 times 128$), so XLA zero-pads every FFN weight matrix to 1,408 columns and wastes 4.5% of FFN FLOPs. @tab-topology quantifies the net effect: on 16 chips, four `TP4` instances deliver $2.22 times$ the aggregate throughput of one `TP16` instance.

#figure(
  booktabs(
    columns: 6,
    align: (left, center, right, right, right, right),
    header: ([Topology], [Chips / hosts], [Layer (ms)], [Tok/s per inst.], [MFU], [16-chip Tok/s]),
    [$4 times$`TP4`], [4 / 1], [25.19], [13,551], [68.16%], [54,205],
    [$2 times$`TP8`], [8 / 2], [16.85], [20,261], [50.95%], [40,521],
    [$1 times$`TP16`], [16 / 4], [13.99], [24,394], [30.67%], [24,394],
  ),
  caption: [Measured Gemma-4 31B scaling across 16 TPU v5e chips (data parallelism across instances) at the 16,384-token compilation bucket: per-layer time, throughput per instance, and aggregate throughput of the 16 chips. Smaller TP degrees achieve up to $2.22 times$ higher aggregate throughput.],
) <tab-topology>

_Memory favors large meshes._ Model weights are fixed across a TP group, so a larger group leaves disproportionately more HBM for the KV cache. On `TP=8` (128 GB of HBM), weights plus XLA scratch space leave 52,864 usable KV tokens (826 blocks of 64 tokens). On `TP=16` (256 GB), per-chip weight shards halve from 7.68 GB to 3.84 GB and usable capacity grows to 145,216 tokens, $2.75 times$ per instance or $1.37 times$ per chip. Decode bandwidth follows the same trend. In a 60-cell sweep of the Pallas `ragged_paged_attention` kernel, a TPU descendant of PagedAttention @kwon2023vllm, decode step latency stays flat at 15–18 ms for batch sizes $B lt.eq 32$ (memory-bandwidth utilization below 11%) and hits the HBM bandwidth wall at $B gt.eq 64$. At $B in {1, 2}$, `TP=16` is 2%–5% slower than `TP=8` because the 16 KV heads shard to a single head per chip (512 B per token per chip), which underuses the HBM burst length; for contexts $L gt.eq 4096$ or $B gt.eq 128$, however, `TP=16` halves per-chip KV traffic and is $1.19"–"1.22 times$ faster.

_The SMEM page-table ceiling._ The attention kernel prefetches the entire `int32` page table into the chip's 1.00 MB SMEM:
$ "Bytes"_"SMEM" (B, L_"max") = B times ceil(L_"max" / P) times 4 "bytes". $
With page size $P = 16$, any batch with $sum_(i=1)^B L_i > 3.5 times 10^6$ tokens (e.g., $B = 512$, $L = 8192$) fails to compile with `RESOURCE_EXHAUSTED: Used 1.01M of 1.00M smem`, even when its KV cache occupies only 2.15 GB of HBM. Serving long contexts therefore requires larger blocks (64 or 512 tokens) together with `TP=16` sharding.

_Neither extreme suffices._ @tab-workloads shows the resulting regime inversion on the full slice. `S` wins every short- and medium-context workload, by up to 28.8% over `H`; `H` wins the mid-context `W7` by 17.6% because it keeps 6 requests resident per instance versus 2 on `S`; and on the long-context `W6`, `S` cannot admit a single request. Scaling further to one `TP32` instance does not help: because the model has only 16 KV heads, `TP=32` replicates KV heads across chip pairs, adding only 8.9% usable KV capacity over `TP=16` while nearly halving slice throughput (671.5 s versus 347.1 s on `W4`).

#wide-fig(
  booktabs(
    columns: 8,
    align: (left, right, right, left, right, right, right, left),
    header: ([Workload], [Reqs], [Out Toks], [Context Range], [`S` ($4 times$`TP8`) (s)], [`H` ($2 times$`TP16`) (s)], [`M` (`TP16`+$2 times$`TP8`) (s)], [Static Winner]),
    [`W0` (ShareGPT chat)], [1,763], [721,595], [$lt.eq 16"k"$ (median 502)], [*190.8*], [267.8 (at 131k) / 306.4 (at 16k)], [352.7 ($2 times$`TP8`)], [*`S`* (−28.8% vs. `H`)],
    [`W4` (24 long + 880 chat)], [904], [387,975], [$27.1"k–"36.5"k"$], [*147.7*], [187.4], [—], [*`S`* (−21.2% vs. `H`)],
    [`W5` (TED translation)], [250], [982,899], [median $P$ 3.5k, $D$ 4.0k], [*754.3*], [774.4], [—], [*`S`* (−2.6% vs. `H`)],
    [`W7` (GovReport mid)], [120], [79,531], [$16.4"k–"43.1"k"$ (median 21.8k)], [248.4], [*204.8*], [—], [*`H`* (−17.6% vs. `S`)],
    [`W6` (GovReport long)], [40], [26,951], [$53.3"k–"128.3"k"$ (median 64.3k)], [Infeasible (>52.9k)], [*206.2*], [409.2 ($1 times$`TP16`)], [*`H`* (`S` infeasible)],
  ),
  caption: [Measured full-BF16 makespans of one workload manifest on the 32-chip TPU v5e slice. `S` dominates short- and medium-context workloads (`W0`, `W4`, `W5`), whereas `H` wins by 17.6% on mid-context `W7` (6 resident requests per `TP16` instance versus 2 per `TP8`) and is mandatory on long-context `W6`. For `M`, the column reports the sub-layout that serves the workload.],
) <tab-workloads>

== Limitations of Static Heterogeneity and Live Request Migration <sec-static>

Before changing layouts at runtime, we ask whether a fixed heterogeneous layout, `M` = `TP16` + $2 times$`TP8`, or live cross-TP request migration can bridge the short/long divide.

_Short-request pollution._ Consider the mixed workload `W4` (880 short chats and 24 long GovReport documents of 27.1k–36.5k tokens) in a capacity-constrained regime in which `TP8` instances hold 26,368 KV tokens, too few for the long documents, and `TP16` instances hold 72,192. Under a plain load-balancing router, `M` finishes in 452.2 s, 30.3% slower than homogeneous `H` (347.1 s). Early short chats spill onto the `TP16` instance and occupy its KV blocks, starving the 24 long documents, while the two `TP8` engines drain their short queues by $t = 210" s"$ and then sit idle for the remaining 242 s.

_Reservation-aware routing._ We eliminate this head-of-line blocking, both in our ExtProc gateway and in the `llm-d` @llmd router's endpoint picker, by combining work-conserving bidirectional spill with a KV reservation floor for long requests. A short request may spill onto `TP16` only if $"FreeKV"_"TP16" - "Cost"(r_"short") gt.eq 36,864$ tokens, which preserves one full-length document slot whenever long documents remain unserved. With reservation, `M` finishes in 311.9 s (314.7 s and 309.0 s across repetitions, and 317.8 s through the `llm-d` endpoint picker's late-binding policy), 10.1% faster than `H` (@tab-w4).

#figure(
  booktabs(
    columns: 4,
    align: (left, right, right, right),
    header: ([Layout and routing policy (`W4`)], [Mean (s)], [Tok/s], [vs. `H`]),
    [`H` ($2 times$`TP16`), plain], [347.1], [1,118], [baseline],
    [`TP32` ($1 times$`TP32`), plain], [671.5], [578], [+93.5%],
    [`M` (`TP16`+$2 times$`TP8`), plain], [452.2], [858], [+30.3%],
    [`M`, reserve + fallback (ExtProc)], [*311.9*], [*1,245*], [*−10.1%*],
    [`M`, `llm-d` late binding], [*317.8*], [*1,221*], [*−8.4%*],
  ),
  caption: [Measured `W4` makespan across static layouts and routing policies. KV capacity reservation turns `M` from a 30.3% regression into a 10.1% improvement over `H`.],
) <tab-w4>

_A narrow phase-balance band._ Static `M` wins only when its short and long queues drain at the same time. Under full BF16, its two `TP8` instances drain `W0` at 352.7 s per manifest, and its `TP16` instance drains `W6` at 409.2 s per manifest. Because `TP8` cannot execute `W6`, a job with $k_s$ short and $k_l$ long manifests finishes at $max(352.7 k_s, 409.2 k_l)$: whenever one phase dominates, half of the slice idles, and `M` is optimal in only 6% of the $(k_s, k_l)$ space (@sec-eval).

_Migration loses to recomputation._ Rather than fixing the layout, one could migrate requests from `TP8` to `TP16` instances as their contexts grow, in the spirit of Llumnix @sun2024llumnix. Across a 102-run evaluation, enabling live migration degraded makespan by 3.6% to 61.1% on three of our workloads. At matched mechanism checkpoints, staging a request's 15.1 GB KV state through host DRAM was slower than re-dispatching its token history for parallel re-prefill on the MXUs, which costs $approx 98" ms"$ per 1,000 tokens. Migration across TP degrees also breaks numerical reproducibility: summing partial dot products over 8 versus 16 shards in 96 AllReduces per step changes BF16 rounding, with a median drift of 0.097 nats across 44 fixed-prefix probes and near-tie outliers of up to 8.16 nats. Serving each request entirely within one phase layout avoids this drift.

These limits motivate changing the layout of the _whole slice_ between phases instead of moving requests between fixed meshes—provided the change itself is cheap.

== Anatomy of the TPU Reconfiguration Bottleneck and Existing Pitfalls <sec-anatomy>

We define the *reconfiguration cost* $C$ as the wall-clock time lost between the completion of the last source-phase request ($T_0$) and steady-state serving of the target phase: the dispatch gap until the first target-phase request is dispatched plus the ramp deficit relative to the target layout's steady-state rate. Redeploying vanilla vLLM engines to switch `S ↔ H` costs a median $C = 419.0" s"$ (412.9 s gap and 6.1 s ramp). As in prior cold-start studies @liu2026foundry @zeng2025medusa, we split this cost into environment initialization and worker initialization.

_Environment initialization._ This model-agnostic part spawns Ray worker processes and imports Python, vLLM, and JAX. It costs 32.7 s per switch, plus $approx 120" s"$ of harness overhead when the switch is driven by external shell-level job submissions. On GPUs, prior work removes this cost with pre-warmed workers that have finished process creation and library import but do not yet hold accelerator resources @liu2026foundry @zeng2025medusa. On TPUs this is harder than it appears, because importing the serving stack tends to initialize the TPU runtime, which may have only one owner per host.

_Worker initialization._ This part builds model- and layout-specific state: device initialization, weight loading, KV cache allocation, and compilation. Weight loading dominates the baseline at 237.0 s, or 67% of the 355 s engine initialization. Instrumenting the loader shows that disk iteration and file reads take 0.16 s and host-to-device DMA 2.76 s, whereas host-CPU tensor transposes, reshapes, and dtype conversions take 128.4 s, load-time JIT compilation of parameter shardings 75.3 s, and Flax `nnx.eval_shape` graph construction 14.6 s. Even once weight loading is fast, KV cache initialization takes 25.9 s and compilation and warm-up 54.6 s with a warm persistent XLA cache, whose hit rates are only 2.6% (48 of 1,848 programs) and 6.0% (264 of 4,416).

_Pitfall 1: RAM staging and peer transfer._ GPU systems accelerate weight loading with faster storage paths, host caching, or RDMA transfer from peers @fu2024serverlessllm @zhang2025blitzscale. These remedies target the wrong cost on TPUs: staging the 61.4 GB checkpoint in `/dev/shm` tmpfs leaves `load_model` unchanged (237.6 s), because the time goes to host-side conversion and sharding compilation, not to bytes in flight.

_Pitfall 2: Background compilation on a live host._ The persistent cache cannot remove JAX's Python tracing and StableHLO lowering (28.6 s on `H` and 45.5 s on `S`), which run on the main thread before any cache lookup because JAX's trace cache is process-local. A natural workaround is to compile the target layout in a second process on each host while the source layout serves. However, a process launched with `JAX_PLATFORMS=cpu` that requests TPU topology descriptors to compile for the target immediately aborts with `ABORTED: The TPU is already in use` on `/tmp/libtpu_lockfile`, because querying the topology unconditionally initializes the PJRT TPU plugin.

_Pitfall 3: In-place rebuild within one runtime._ Keeping all weights resident and re-slicing them inside a single multi-host JAX runtime, as Pathways-style @barham2022pathways and Alpa-style @zheng2022alpa systems map programs onto device submeshes, avoids reloading entirely. In practice, `libtpu` aborts when compiling collectives for some sub-boxes of the torus (@sec-design-submesh), vLLM's process-global singletons would need extensive refactoring, and all participating hosts become one failure domain.

PhaseMorph is designed around these observations: it removes host-side conversion and compilation from the critical path by materializing their results, and it overlaps environment initialization with source-phase serving without ever sharing a TPU between two owners.

= PhaseMorph Overview <sec-overview>

PhaseMorph operates in two stages that mirror the save/load split of recent cold-start systems @liu2026foundry: a PREPARE stage that runs offline or in the background while the source layout serves, and an online HANDOFF/RESTORE stage on the critical path after $T_0$ (@fig-architecture).

#wide-fig(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 235" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#1f2328">
  <text x="12" y="15" font-size="12" font-weight="bold">PhaseMorph Overlap Timeline: Clean-Room CPU Pre-Spawn, Pre-Read Child &amp; Post-T0 Fast Handoff</text>

  <!-- Time axis -->
  <line x1="145" y1="36" x2="840" y2="36" stroke="#8c959f" stroke-width="1"/>
  <line x1="510" y1="30" x2="510" y2="215" stroke="#cf222e" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="515" y="28" text-anchor="start" font-size="10" font-weight="bold" fill="#cf222e">T0: Phase 1 Drain Complete</text>
  <text x="320" y="31" text-anchor="middle" font-size="9.5" fill="#59636e">← Phase 1 Serving Window (Zero Source Slowdown) →</text>
  <text x="720" y="28" text-anchor="middle" font-size="9.5" fill="#1a7f37">← Critical Path C = 21.1–27.5 s →</text>
  <!-- Lane 1: TPU Hardware / Source S -->
  <text x="138" y="62" text-anchor="end" font-weight="bold">Source S Workers</text>
  <text x="138" y="75" text-anchor="end" font-size="9.5" fill="#59636e">(Owns /dev/vfio/0..3)</text>
  <rect x="145" y="45" width="365" height="34" rx="4" fill="#dafbe1" stroke="#1a7f37"/>
  <text x="327" y="66" text-anchor="middle" fill="#1a7f37" font-weight="bold">S (4×TP8) Serving W0 @ 100% Rate (3,978–4,070 tok/s)</text>
  <rect x="510" y="45" width="28" height="34" rx="3" fill="#ffebe9" stroke="#cf222e"/>
  <text x="524" y="66" text-anchor="middle" font-size="9.5" fill="#cf222e">2.0s</text>

  <!-- Lane 2: Target H Pre-Spawned Worker -->
  <text x="138" y="114" text-anchor="end" font-weight="bold">Target H Workers</text>
  <text x="138" y="127" text-anchor="end" font-size="9.5" fill="#59636e">(clean-room pre-spawn)</text>
  <rect x="175" y="97" width="180" height="34" rx="4" fill="#ddf4ff" stroke="#0969da"/>
  <text x="265" y="118" text-anchor="middle" fill="#0969da">CPU Spawn + Imports (34.3s)</text>
  <rect x="355" y="97" width="155" height="34" rx="4" fill="#f6f8fa" stroke="#8c959f" stroke-dasharray="3,2"/>
  <text x="432" y="113" text-anchor="middle" font-size="10" fill="#59636e">Parked on ReleaseGate</text>
  <text x="432" y="125" text-anchor="middle" font-size="9.5" fill="#59636e">(0 libtpu / 0 vfio fds)</text>

  <!-- Post-T0 Critical Path on Target H -->
  <rect x="538" y="97" width="32" height="34" rx="3" fill="#fff8c5" stroke="#9a6700"/>
  <text x="554" y="112" text-anchor="middle" font-size="9" fill="#9a6700">VFIO</text>
  <text x="554" y="123" text-anchor="middle" font-size="9" fill="#9a6700">0.8s</text>
  <rect x="570" y="97" width="48" height="34" rx="3" fill="#dafbe1" stroke="#1a7f37"/>
  <text x="594" y="112" text-anchor="middle" font-size="9.5" fill="#1a7f37">init_dev</text>
  <text x="594" y="123" text-anchor="middle" font-size="9.5" fill="#1a7f37">3.9s</text>
  <rect x="618" y="97" width="58" height="34" rx="3" fill="#dafbe1" stroke="#1a7f37"/>
  <text x="647" y="112" text-anchor="middle" font-size="9.5" fill="#1a7f37">FastRestore</text>
  <text x="647" y="123" text-anchor="middle" font-size="9.5" fill="#1a7f37">4.9s</text>
  <rect x="676" y="97" width="34" height="34" rx="3" fill="#dafbe1" stroke="#1a7f37"/>
  <text x="693" y="112" text-anchor="middle" font-size="9.5" fill="#1a7f37">KV</text>
  <text x="693" y="123" text-anchor="middle" font-size="9.5" fill="#1a7f37">2.6s</text>
  <rect x="710" y="97" width="65" height="34" rx="3" fill="#dafbe1" stroke="#1a7f37"/>
  <text x="742" y="112" text-anchor="middle" font-size="9.5" fill="#1a7f37" font-weight="bold">ExecBank</text>
  <text x="742" y="123" text-anchor="middle" font-size="9.5" fill="#1a7f37">5.0s (0 trace)</text>
  <rect x="775" y="97" width="42" height="34" rx="3" fill="#dafbe1" stroke="#1a7f37"/>
  <text x="796" y="112" text-anchor="middle" font-size="9.5" fill="#1a7f37">Route</text>
  <text x="796" y="123" text-anchor="middle" font-size="9.5" fill="#1a7f37">3.1s</text>

  <!-- Lane 3: Optional Cold Pre-Read Subprocess -->
  <text x="138" y="168" text-anchor="end" font-weight="bold">CPU Pre-Read Child</text>
  <text x="138" y="181" text-anchor="end" font-size="9.5" fill="#59636e">(PreRead: cold 1st switch)</text>
  <rect x="195" y="151" width="315" height="34" rx="4" fill="#fff8c5" stroke="#9a6700"/>
  <text x="352" y="166" text-anchor="middle" fill="#9a6700" font-weight="bold">JAX_PLATFORMS=cpu (nice 10, 16 fake host devs)</text>
  <text x="352" y="179" text-anchor="middle" font-size="9" fill="#9a6700">Streams load_weights → 32,896 shards to /dev/shm (absorbs 165s before T0)</text>
  <rect x="510" y="151" width="95" height="34" rx="3" fill="#fff8c5" stroke="#9a6700" stroke-dasharray="3,2"/>
  <text x="557" y="172" text-anchor="middle" font-size="9.5" fill="#9a6700">Residual 79.6s</text>
</svg>
```),
  caption: [PhaseMorph workflow for a full `S → H` switch. During the source phase (PREPARE), target `H` workers spawn in CPU-only bundles, verify that they hold no `libtpu` or `/dev/vfio` handle, and park on a release gate; for a cold first switch, a CPU child additionally materializes the weight snapshot. After $T_0$ (HANDOFF/RESTORE), workers wait out the kernel's VFIO reset, restore pre-sharded weights from `/dev/shm` (4.9 s), and load 187 serialized executables from the executable bank (3.45 s).],
) <fig-architecture>

_PREPARE stage._ The first deployment of a layout writes two artifacts as a by-product of normal initialization: a per-worker snapshot of post-conversion weight shards in host shared memory (7.53 s) and a bank of serialized, ahead-of-time compiled executables (2.26 s). When a switch is planned, PhaseMorph spawns the target layout's workers in CPU-only processes while the source layout keeps serving; the workers import the serving stack, build tokenizers, and park on a release gate without touching the TPU. If the target layout has never been deployed, each parked worker also launches a low-priority CPU child that produces the weight snapshot in the background.

_HANDOFF/RESTORE stage._ At $T_0$, the in-process switch controller drains and releases the source engines (about 2 s), and the parked workers proceed: they wait until the kernel has finished resetting the VFIO devices, initialize the TPU, place pre-sharded weights from shared memory onto the chips, allocate the KV cache, load all executables from the bank, and register with the router. Python tracing and backend compilation are skipped entirely on this path.

_Phase-aware switching planner._ PhaseMorph targets two-phase bulk jobs consisting of $k_s$ short-context manifests (`W0`) followed by $k_l$ long-context manifests (`W6`). A calibrated cost model chooses among four strategies, using the per-manifest makespans of @tab-workloads, the `S` tail-drain overhead $O = 32.6" s"$, and the measured switch costs $C_"full"$ and $C_"part"$:

$ T_"H" &= 267.8 k_s + 206.2 k_l, \
  T_"M" &= max(352.7 k_s, 409.2 k_l), \
  T_"S→H" &= 190.8 k_s + O + C_"full" + 206.2 k_l, \
  T_"S→M" &= 190.8 k_s + min_(f in [0,1]) max(352.7(1-f) k_s, thin B), \
  "where" B &= O + C_"part" + 409.2 k_l. $

In the partial switch, two `TP8` instances keep serving the short phase while the other half of the slice is rebuilt as one `TP16` instance; $f$ is the fraction of the short phase served before the switch. @app-cost-model derives the resulting break-even job sizes.

= Design <sec-design>

== Pre-Sharded Weight Materialization and Shape-Only Graph Construction <sec-design-weights>

=== Sharded HBM Layout Materialization

Because weight loading on TPU is dominated by host-side conversion and sharding compilation, PhaseMorph persists the _result_ of loading rather than its input. After a worker has converted and sharded its parameters, it serializes its per-chip HBM shards directly to host shared memory, keyed by layout (30.7 GB per host for `TP8` and 15.4 GB per host for `TP16`). On later switches into that layout, workers skip checkpoint parsing and host conversion entirely: they map the raw shard buffers and assemble global arrays with `jax.make_array_from_single_device_arrays`, which needs no per-parameter sharding compilation. The snapshot is written inline during the first deployment of a layout, so no separate 350.8 s provisioning pass is required.

=== Shape-Only Flax Graph Construction

Building the Flax model graph with `nnx.eval_shape` repeats abstract evaluation and checks that are redundant once the parameters come from a verified snapshot. PhaseMorph instead constructs the graph shape-only, skipping these checks and reducing graph construction from 14.6 s to 1.8 s. Together with the snapshot, this cuts `load_model` from 237.0 s to 5.0 s, a $47 times$ speedup.

=== Background Host Checkpoint Pre-Read for Cold Switches

The first switch into a layout has no snapshot yet. Even with inline provisioning, the cold switch costs $C_"first" = 222.63" s"$, of which 158.9 s is `load_model` and 130.7 s of that is host tensor conversion. PhaseMorph moves this conversion into the source phase. Each parked target worker launches a child process at `nice 10` with `JAX_PLATFORMS=cpu`, 16 fake host devices, and all `TPU_*` environment variables stripped, so that it cannot initialize the TPU runtime. The child runs the normal `load_weights` path on the CPU backend and streams all 32,896 shards to shared memory in the snapshot format. Every shard matches the TPU-produced reference snapshot bitwise (SHA-256), so the target engine consumes the child's output through the same restore path as a warm switch.

== AOT Executable Materialization and Zero-Threshold Caching <sec-design-exec>

=== Eliminating Sub-Second Compilation Eviction

JAX 0.10.2 writes a program to the persistent compilation cache only if it took at least `jax_persistent_cache_min_compile_time_secs` (1.0 s by default) to compile. Each worker therefore silently discards $approx 248$ small helper, sampler, and KV-allocation programs that compile in 50–800 ms each but cost 42.4 s in aggregate on every restart. PhaseMorph sets both the minimum compile time and the minimum entry size to zero in every instance's runtime environment. Persistent cache hits rise to 100% (1,848 of 1,848 and 4,416 of 4,416 programs, with no backend recompilations), KV cache initialization falls from 25.9 s to 2.6 s, and compilation and warm-up from 54.6 s to 35.2 s.

=== Device-Order-Preserving Executable Bank and Unanimity Barrier

The remaining 35 s of warm-up is Python tracing and lowering, which no persistent cache can skip because tracing state lives in the process. PhaseMorph instead serializes the compiled artifacts themselves. After a layout's first warm-up, every rank serializes all 187 ahead-of-time compiled executables with `jax.experimental.serialize_executable` into an executable bank in shared memory (1.07 GB). A deserialized executable is only valid on the device assignment it was compiled for, and that assignment is not the natural device order: on `TP16` the execution order is a permutation such as `[0,1,4,5,2,3,...]`. The bank therefore records each executable's exact `execution_devices` permutation and restores it on load. A new worker reloads all 187 executables with `deserialize_and_load` in 3.45 s, performing no Python tracing or lowering.

Partial use of a bank would leave ranks executing different programs inside one collective. Before using the bank, all ranks of an instance therefore agree through a rank-0 unanimity barrier: if any rank refuses its entries, for example because validation fails, every rank falls back to normal compilation and the deployment still completes.

== Clean-Room Worker Pre-Spawning and VFIO Device Handoff <sec-design-prespawn>

=== Pre-Spawning Without PJRT Lockfile Contention

To overlap the 34.3 s of process spawning, imports, and tokenizer construction with source-phase serving, PhaseMorph creates the target workers in CPU-only Ray bundles (`{"CPU": 1.0, "node:<ip>": 0.001}`) while the source engines still own the TPUs. The difficulty is that a worker must not touch the TPU runtime before the source has released it. Each pre-spawned worker therefore audits itself: it inspects `/proc/self/maps` and `/proc/self/fd` to assert that no `libtpu.so` is mapped and no `/dev/vfio/*` or lockfile descriptor is open, and then parks on an asynchronous release gate inside `TPUWorker.init_device()`. Only when the controller opens the gate does the worker initialize its TPU backend.

=== Asynchronous VFIO Kernel-Reset Barrier

Releasing the gate as soon as the source processes exit is not sufficient. After the source workers exit, the Linux kernel takes 0.42–0.65 s to reset the `/dev/vfio/0..3` devices, and a worker that calls `jax.devices()` within this window fails with `EBUSY`. Released workers therefore poll `os.open("/dev/vfio/<group>", os.O_RDWR)` every 20 ms and initialize the TPU only after every device group opens successfully. The gate and VFIO wait together cost 0.84–0.86 s on the critical path, down from 11.1–12.2 s when workers are spawned after $T_0$.

== In-Runtime Submesh Reconfiguration versus Process Isolation <sec-design-submesh>

To establish the hardware lower bound of a layout change—weights stay in HBM and no process restarts—we built _PhaseMorph-Submesh_, a unified runtime that initializes one JAX domain across four hosts (a $4 times 4$ torus) and switches between two 8-chip `TP8` submeshes and one 16-chip `TP16` mesh.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 205" font-family="Helvetica, Arial, sans-serif" font-size="10.5" fill="#1f2328">
  <!-- Left: x-split fails -->
  <text x="105" y="16" text-anchor="middle" font-weight="bold" fill="#cf222e">Column Split (x-halves): FAILS</text>
  <text x="105" y="29" text-anchor="middle" font-size="9.5" fill="#59636e">x-lower IDs [0,1,4,5,8,9,12,13] non-dense</text>
  <rect x="22" y="38" width="80" height="128" rx="5" fill="#ffebe9" stroke="#cf222e" stroke-width="1.5"/>
  <rect x="108" y="38" width="80" height="128" rx="5" fill="#f6f8fa" stroke="#8c959f" stroke-width="1"/>
  <text x="62" y="58" text-anchor="middle" font-weight="bold" fill="#cf222e">0, 1</text>
  <text x="62" y="88" text-anchor="middle" font-weight="bold" fill="#cf222e">4, 5</text>
  <text x="62" y="118" text-anchor="middle" font-weight="bold" fill="#cf222e">8, 9</text>
  <text x="62" y="148" text-anchor="middle" font-weight="bold" fill="#cf222e">12, 13</text>
  <text x="148" y="58" text-anchor="middle" fill="#59636e">2, 3</text>
  <text x="148" y="88" text-anchor="middle" fill="#59636e">6, 7</text>
  <text x="148" y="118" text-anchor="middle" fill="#59636e">10, 11</text>
  <text x="148" y="148" text-anchor="middle" fill="#59636e">14, 15</text>
  <text x="105" y="182" text-anchor="middle" font-size="9.5" fill="#cf222e">libtpu net_util.cc:2304 RET_CHECK</text>
  <text x="105" y="194" text-anchor="middle" font-size="9.5" fill="#cf222e">device_id &lt; kMaxDeviceCount (8 vs 8)</text>

  <!-- Right: y-split passes -->
  <text x="315" y="16" text-anchor="middle" font-weight="bold" fill="#1a7f37">Host-Row Split (y-halves): PASSES</text>
  <text x="315" y="29" text-anchor="middle" font-size="9.5" fill="#59636e">y-lower [0..7] &amp; y-upper [8..15] dense</text>
  <rect x="232" y="38" width="166" height="61" rx="5" fill="#dafbe1" stroke="#1a7f37" stroke-width="1.5"/>
  <rect x="232" y="105" width="166" height="61" rx="5" fill="#ddf4ff" stroke="#0969da" stroke-width="1.5"/>
  <text x="315" y="62" text-anchor="middle" font-weight="bold" fill="#1a7f37">y-lower (Hosts 0–1): IDs 0, 1, 2, 3</text>
  <text x="315" y="84" text-anchor="middle" font-weight="bold" fill="#1a7f37">IDs 4, 5, 6, 7 (Origin + Dense)</text>
  <text x="315" y="129" text-anchor="middle" font-weight="bold" fill="#0969da">y-upper (Hosts 2–3): IDs 8, 9, 10, 11</text>
  <text x="315" y="151" text-anchor="middle" font-weight="bold" fill="#0969da">IDs 12, 13, 14, 15 (Independent TP8)</text>
  <text x="315" y="182" text-anchor="middle" font-size="9.5" fill="#1a7f37" font-weight="bold">2×TP8 → TP16 in 12.0 ms (0 compiles)</text>
  <text x="315" y="194" text-anchor="middle" font-size="9.5" fill="#1a7f37">TP16 → 2×TP8 in 48.0 ms (ICI swap)</text>
</svg>
```),
  caption: [Submesh partitioning of a 16-chip ($4 times 4$) torus under `libtpu 0.0.43`. Splitting by columns (`x`) aborts in `net_util.cc:2304` because the origin half has non-contiguous device IDs; splitting by host rows (`y`: IDs `0..7` and `8..15`) yields independent submeshes and 12.0 ms in-HBM switching.],
) <fig-submesh>

_Bypassing the sub-box routing bug._ Our first partition split the $4 times 4$ grid along columns, giving the origin half device IDs `[0,1,4,5,8,9,12,13]`. Compiling any collective for this submesh fails inside `libtpu` (`net_util.cc:2304`, `RET_CHECK device_id < kMaxDeviceCount (8 vs. 8)`), because the library assumes that a sub-box anchored at the origin has dense device IDs `0..N-1`. Rotating the split to host rows (IDs `0..7` and `8..15`) satisfies this invariant, both for live `jax.jit` compilation and for `deserialize_and_load` of banked executables (@fig-submesh).

_Isolation and switching._ With host-row halves, the two `TP8` submeshes step concurrently at $1.05"–"1.08 times$ their solo step time, and stalling one half in a 30 s barrier leaves the other at 97%–99% of its throughput. Switching `2×TP8 → TP16` with nested local device order takes 12.04 ms (0.08 ms to stop, 6.14 ms to re-slice the weights on chip, and 5.82 ms for the first step), with no compilation; the reverse switch over ICI takes 48.02 ms.

_Why PhaseMorph keeps process isolation._ Retrofitting unified submeshes into vLLM would require refactoring roughly 1.5k–3k lines around process-global singletons (the TPU worker, `jax.clear_caches()`, and Ray executor bindings) and would merge four hosts into one failure domain. Yet it would only trim the warm switch cost from 21.1–27.5 s to $approx 9"–"10" s"$, improving the median bulk makespan gain by 0.3 percentage points. Process-isolated switching already captures 99.7% of the achievable gain (@sec-eval-submesh), so PhaseMorph uses it in production and treats the unified runtime as a lower bound.

= Implementation <sec-impl>

PhaseMorph is implemented as a set of extensions to vLLM's TPU backend (`tpu-inference` @tpuinference), a small patch to vLLM's Ray executor, an in-process switch controller, and routing policies for both our ExtProc gateway and the `llm-d` router @llmd.

_Weight path._ We instrument the TPU weight loader with a load profiler that attributes wall time to file reads, host tensor operations, sharding compilation, and device transfers per tensor; this profiler produced the breakdown in @sec-anatomy. The snapshot module writes and verifies per-worker shard manifests in `/dev/shm` and restores them with `jax.make_array_from_single_device_arrays`; a companion module constructs the Flax graph from recorded shapes. The pre-read module launches the CPU child, sanitizes its environment, and hands the finished snapshot to the parked worker.

_Compilation path._ The layout module sets the zero compile-time and entry-size thresholds of the persistent cache in every instance's runtime environment. The executable-bank module serializes compiled executables per rank together with their device permutations, implements the rank-0 unanimity barrier, and falls back to normal compilation if any rank refuses. We validate banked engines by comparing their greedy outputs token for token against engines loaded through the native path.

_Worker lifecycle._ The pre-spawn module creates CPU-only Ray placement-group bundles; we patch vLLM's Ray executor utilities so that such bundles are retained for TPU workers. The release gate and VFIO barrier live in the TPU worker's device initialization. The switch controller runs in-process and executes each transition as a fixed sequence—drain, release, deploy, and gateway update—replacing the shell-level job submissions of the baseline. A partial switch keeps two `TP8` engines registered while the other half of the slice is rebuilt, so their traffic pauses only briefly.

_Routing and planning._ Work-conserving spill and the long-request KV reservation floor are implemented as a gateway policy and, for `llm-d`, as endpoint-picker configurations with reservation and late binding. The switching planner is a calibrated bulk cost model that we validated against all 10 end-to-end multi-phase runs (@sec-eval-e2e).

= Evaluation <sec-eval>

We evaluate PhaseMorph along four dimensions: (1) reduction of reconfiguration latency relative to vanilla vLLM redeployment; (2) preservation of source-phase serving throughput while the target is prepared; (3) end-to-end makespan and break-even job sizes across a two-phase workload space; and (4) comparison with in-runtime submesh switching.

== Experimental Setup <sec-eval-setup>

_Testbed and model._ All experiments run on the 32-chip Cloud TPU v5e slice described in @sec-tradeoffs (8 hosts, $4 times 8$ torus) and serve Gemma-4 31B in full BF16 with vLLM and `tpu-inference` on JAX 0.10.2; the submesh prototype uses `libtpu 0.0.43`. Weight snapshots and executable banks reside in each host's `/dev/shm`.

_Workloads._ We use the manifests of @tab-workloads: ShareGPT chat (`W0`), a mix of long documents and chats (`W4`), TED translation (`W5`), and mid- and long-context GovReport summarization (`W7`, `W6`). Requests replay real traces with their recorded output lengths.

_Baselines and variants._ We compare _Vanilla vLLM Redeploy_, which tears down the source engines and deploys the target layout through external job submission, against cumulative PhaseMorph variants: _PhaseMorph-Snap_ (pre-sharded weight snapshots, shape-only graphs, and the in-process controller), _PhaseMorph-Cache_ (adds zero-threshold caching), and _PhaseMorph_ (adds clean-room pre-spawning and the executable bank). _PhaseMorph-PreRead_ adds the background CPU pre-read for cold first switches, and _PhaseMorph-Submesh_ is the unified-runtime prototype. For end-to-end comparisons we also include the static layouts `H` and `M` with reservation-aware routing.

_Metrics._ We report the reconfiguration cost $C$ defined in @sec-anatomy; for partial switches, the effective cost $C_"eff"$ is the delay of the long-phase class relative to a static `M` that serves it from the start. We also report source-phase throughput during preparation and the makespan of complete jobs.

== Reconfiguration Latency Reduction <sec-eval-latency>

_Warm full switch._ @tab-segments decomposes the cost of a warm `S → H` switch. PhaseMorph-Snap removes most of `load_model` (5.0 s) and, by replacing external job submission with the in-process controller, about 120 s of harness overhead, reducing $C$ from 419.0 s to 132.2 s. PhaseMorph-Cache restores the 248 evicted small programs, cutting KV initialization to 2.6 s and compilation and warm-up to 35.2 s, for $C = 89.8" s"$. The full system then removes the remaining host-CPU work from the critical path: process spawning and imports (21.6 s) move into the source phase, the gate and VFIO wait shrink from 11–12 s to 0.86 s, and compilation and warm-up drop to 5.0 s because executables come from the bank. The dispatch gap falls to 23.98 s and $C$ to 27.51 s. A second measurement campaign, in which snapshots and banks were provisioned inline by the first deployment, measured $C = 21.14" s"$; its ramp term is negative because the new engines finished their first target-phase requests slightly ahead of the steady-state reference.

#figure(
  booktabs(
    columns: 6,
    align: (left, right, right, right, right, right),
    header: ([Segment (s)], [Snap], [Cache], [Full A], [Full B], [Partial]),
    [Pre-spawn (hidden)], [0.00], [0.00], [(34.30)], [(34.10)], [(32.39)],
    [Drain + release], [3.75], [3.49], [1.98], [1.95], [1.85],
    [Gate + VFIO wait], [11.11], [12.16], [0.86], [0.84], [0.84],
    [Spawn + imports], [21.55], [21.56], [*0.00*], [*0.00*], [*0.00*],
    [`init_device`], [3.55], [3.63], [3.92], [3.90], [3.74],
    [`load_model`], [5.01], [4.82], [4.91], [4.88], [4.93],
    [`init_from_config`], [25.92], [2.63], [2.57], [2.56], [2.59],
    [`compile_warmup`], [54.61], [35.15], [*4.98*], [*4.95*], [*5.05*],
    [Gateway ready], [4.21], [3.76], [3.09], [3.05], [3.21],
    [*Dispatch gap*], [*130.61*], [*87.20*], [*23.98*], [*23.61*], [*20.63*],
    [Ramp], [+1.63], [+2.64], [+3.53], [−2.47], [−1.79],
    [*Cost $C$*], [*132.24*], [*89.84*], [*27.51*], [*21.14*], [*18.84*],
  ),
  caption: [[Reconfiguration Latency Reduction] Segment breakdown of warm reconfiguration cost for PhaseMorph-Snap, PhaseMorph-Cache, and PhaseMorph. Full A and Full B are full `S → H` switches from two measurement campaigns (B with inline provisioning); Partial is an `S → M` switch, whose cost is $C_"eff"$. Parenthesized pre-spawn times overlap the source phase.],
) <tab-segments>

_Warm partial switch._ A partial `S → M` switch rebuilds only half of the slice. Its effective cost falls from 186.0 s with vanilla redeployment to 128.9 s with PhaseMorph-Snap, 87.3 s with PhaseMorph-Cache, and 18.84 s with PhaseMorph. Throughout the switch, the two kept `TP8` instances pause for only about 2.1 s, and the first long request is dispatched to the new `TP16` instance 20.6 s after $T_0$. Once running, the new instance serves `W6` at the static-`M` rate (407.4 s from first dispatch to finish versus 409.2 s), so the switch leaves no join tail.

_Cold first switch and amortization._ Without pre-read, inline provisioning makes the first switch into `H` cost 222.63 s, avoiding a separate 350.8 s provisioning pass. PhaseMorph-PreRead converts weights during the source phase; the child absorbs about 165 s of conversion before $T_0$ and finishes about 79.6 s after it, so the cold switch costs 134.10 s, 88.5 s (40%) less. Amortized over 1, 3, and 10 switches, the per-switch cost is 134.1, 58.8, and 32.4 s, compared with 222.6, 88.3, and 41.3 s without pre-read and 378.3, 144.4, and 62.6 s with a separate provisioning pass.

== Foreground Serving Throughput Preservation <sec-eval-foreground>

PhaseMorph's preparation runs on the same hosts as the source layout, so it must not slow the source phase. @tab-foreground compares the throughput of `S` on `W0` with and without concurrent preparation. With pre-spawning, every repetition of both full and partial switches lies within or above the spread of runs without preparation, and the source phase lasts 187.7–194.6 s versus 191.6–193.9 s. The pre-read child, which runs at low priority on the host CPUs, likewise leaves `S` throughput at or above the cold-switch control.

#figure(
  booktabs(
    columns: 4,
    align: (left, right, right, right),
    header: ([Preparation], [Early (tok/s)], [Late (tok/s)], [Phase (s)]),
    [None (Cache)], [2,639–2,667], [3,949–4,013], [191.6–193.9],
    [Pre-spawn, full], [2,661–2,673], [3,932–4,030], [190.7–194.6],
    [Pre-spawn, partial], [2,662–2,677], [3,968–4,113], [187.7–193.0],
    [None (cold control)], [2,707], [3,978], [—],
    [Pre-read child], [2,679–2,712], [4,051–4,054], [—],
  ),
  caption: [[Foreground Throughput Preservation] Source-phase `S` ($4 times$`TP8`) throughput on `W0` while the target layout is prepared (ranges over three repetitions, two for pre-read). "None (Cache)" runs PhaseMorph-Cache without pre-spawning; "None (cold control)" is the cold first switch without a pre-read child. For pre-spawn rows, the early window is the first $approx 34" s"$ of the phase, during which workers spawn and `W0` ramps up, and the late window spans from parking until $T_0$; for pre-read rows, the windows are during and after the CPU child.],
) <tab-foreground>

Correctness is preserved as well. Banked executables served every call of the target phase with no misses or rejections (65,956 model-step calls), greedy outputs of banked engines matched those of natively loaded engines token for token on 32 prompts per engine, and deliberately invalidating one rank's bank caused all ranks to fall back to normal compilation without failing the deployment.

== End-to-End Multi-Phase Makespan and Break-Even Analysis <sec-eval-e2e>

We evaluate switching across the two-phase job space $(k_s, k_l) in [0, 8] times [0.25, 8]$ with the cost model of @sec-overview, using a static `H` rate of $a_"0H" = 267.8" s"$ per `W0` manifest (at a 131k-token context limit). The model reproduces all 10 end-to-end multi-phase runs within $max |Delta k| lt.eq 0.009$ manifests.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 265" font-family="Helvetica, Arial, sans-serif" font-size="10.5" fill="#1f2328">
  <text x="215" y="16" text-anchor="middle" font-weight="bold" font-size="11.5">Winning Strategy per (k_l, k_s) at PhaseMorph Switch Cost</text>
  <!-- Plot area: x from 55 to 395 (k_l: 0 to 8, 42.5 px/unit), y from 225 down to 33 (k_s: 0 to 8, 24 px/unit) -->
  <!-- Region 4: Partial S -> M (top left, k_s > 0.053 + 1.160 k_l) -->
  <polygon points="55,206 168,142 365,33 55,33" fill="#ddf4ff"/>
  <!-- Region 3: Static M (narrow band between k_s = 1.064 k_l - 0.144 and k_s = 1.160 k_l + 0.053) -->
  <polygon points="168,142 365,33 395,33 395,45 192,162" fill="#fff8c5"/>
  <!-- Region 2: Full S -> H (right of Static M, above k_s = 0.78 -> y = 206.3) -->
  <polygon points="55,206 168,142 192,162 395,45 395,206" fill="#dafbe1"/>
  <!-- Region 1: Static H (bottom strip k_s < 0.78 -> y in [206.3, 225]) -->
  <rect x="55" y="206.3" width="340" height="18.7" fill="#ffebe9"/>

  <!-- Boundary lines -->
  <line x1="55" y1="206.3" x2="395" y2="206.3" stroke="#1a7f37" stroke-width="1.8"/>
  <line x1="168" y1="142" x2="365" y2="33" stroke="#0969da" stroke-width="1.5"/>
  <line x1="192" y1="162" x2="395" y2="45" stroke="#9a6700" stroke-width="1.5"/>

  <!-- Break-even lines of earlier variants (dashed) -->
  <line x1="55" y1="173.6" x2="395" y2="173.6" stroke="#cf222e" stroke-width="1.2" stroke-dasharray="5,3"/>
  <text x="392" y="170" text-anchor="end" font-size="9" fill="#cf222e">PhaseMorph-Snap k_s* = 2.14 (C = 132.2 s)</text>
  <line x1="55" y1="186.8" x2="395" y2="186.8" stroke="#8250df" stroke-width="1.2" stroke-dasharray="3,3"/>
  <text x="392" y="184" text-anchor="end" font-size="9" fill="#8250df">PhaseMorph-Cache k_s* = 1.59 (C = 89.8 s)</text>
  <text x="392" y="203" text-anchor="end" font-size="9.5" font-weight="bold" fill="#1a7f37">PhaseMorph k_s* = 0.78 (C = 27.5 s)</text>

  <!-- Region labels -->
  <text x="145" y="88" text-anchor="middle" font-weight="bold" fill="#0969da">Partial Switch S → M</text>
  <text x="145" y="102" text-anchor="middle" font-size="9.5" fill="#0969da">(41% of grid: short-heavy)</text>
  <text x="254" y="113" text-anchor="middle" font-size="9" font-weight="bold" fill="#9a6700" transform="rotate(-29 254 110)">Static M (6%)</text>
  <text x="305" y="138" text-anchor="middle" font-weight="bold" fill="#1a7f37">Full Switch S → H</text>
  <text x="305" y="152" text-anchor="middle" font-size="9.5" fill="#1a7f37">(41% of grid: long-heavy)</text>
  <text x="225" y="219" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#cf222e">Static H (12% of grid: k_s &lt; 0.78 W0 manifests)</text>

  <!-- Axes -->
  <rect x="55" y="33" width="340" height="192" fill="none" stroke="#1f2328" stroke-width="1.2"/>
  <text x="50" y="225" text-anchor="end" font-size="9.5">0</text>
  <text x="50" y="177" text-anchor="end" font-size="9.5">2</text>
  <text x="50" y="129" text-anchor="end" font-size="9.5">4</text>
  <text x="50" y="81" text-anchor="end" font-size="9.5">6</text>
  <text x="50" y="37" text-anchor="end" font-size="9.5">8</text>
  <text x="18" y="130" text-anchor="middle" font-weight="bold" transform="rotate(-90 18 130)">Short Phase Size k_s (W0 Manifests)</text>

  <text x="55" y="240" text-anchor="middle" font-size="9.5">0</text>
  <text x="140" y="240" text-anchor="middle" font-size="9.5">2</text>
  <text x="225" y="240" text-anchor="middle" font-size="9.5">4</text>
  <text x="310" y="240" text-anchor="middle" font-size="9.5">6</text>
  <text x="395" y="240" text-anchor="middle" font-size="9.5">8</text>
  <text x="225" y="257" text-anchor="middle" font-weight="bold">Long Phase Size k_l (W6 Manifests)</text>
</svg>
```),
  caption: [[Phase-Space Break-Even Coverage] Best strategy across $(k_l, k_s) in [0.25, 8] times [0, 8]$ (`W6` and `W0` manifests) at PhaseMorph's switch costs ($C_"full" = 27.5" s"$, $C_"part" = 18.8" s"$). Switching wins 82% of the grid (41% full `S → H`, 41% partial `S → M`) and lowers the short-phase break-even threshold $k_s^*$ from 2.14 (PhaseMorph-Snap) to 0.78 `W0` manifests (0.36 if $O = 0$).],
) <fig-regions>

#wide-fig(
  booktabs(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    header: ([System variant], [Full $C$ (s)], [Partial $C_"eff"$ (s)], [Grid won], [Median / max gain], [`W0→W6` $k_s^*$], [`W0→W7` $k_m^*$]),
    [Vanilla vLLM], [419.0], [186.0], [29%], [4.1% / 18.2%], [5.86 (4.23M)], [10.35 (823k)],
    [PhaseMorph-Snap], [132.2], [128.9], [62% (24% `H`, 38% `M`)], [8.3% / 24.5%], [2.14 (1.54M)], [3.78 (301k)],
    [PhaseMorph-Cache], [89.8], [87.3], [69% (30% `H`, 39% `M`)], [8.7% / 25.4%], [1.59 (1.15M)], [2.81 (223k)],
    [*PhaseMorph*], [*27.5*], [*18.8*], [*82%* (41% `H`, 41% `M`)], [*9.9% / 26.8%*], [*0.78 (563k)*], [*1.38 (110k)*],
    [*PhaseMorph, $O = 0$*], [*27.5*], [*18.8*], [*88%* (47% `H`, 41% `M`)], [*10.4% / 27.9%*], [*0.36 (260k)*], [*0.63 (50k)*],
  ),
  caption: [[End-to-End Makespan and Break-Even] Effect of the reconfiguration cost on the share of the two-phase job space won by switching (split into full `S → H` and partial `S → M`), the median and maximum makespan reduction over the best static layout, and the minimum phase sizes $k_s^*$ and $k_m^*$ (in manifests, with output tokens in parentheses) at which switching breaks even. "Vanilla vLLM" is vanilla redeployment; "$O = 0$" assumes tail-aware dispatch.],
) <tab-breakeven>

_Coverage and gains._ As @fig-regions and @tab-breakeven show, vanilla redeployment makes switching worthwhile on only 29% of the job space. PhaseMorph raises this share to 82%, split evenly between full and partial switches, with median and maximum makespan reductions of 9.9% and 26.8% over the best static layout. Static `M` remains best only in the 6% band where its two halves finish together, and static `H` only when the short phase is too small to amortize a switch.

_Break-even job sizes._ Compressing $C$ from 419.0 s to 27.5 s lowers the short-phase break-even threshold $k_s^* = (C + O) \/ (a_"0H" - a_"0S")$ by $7.5 times$, from 5.86 `W0` manifests (4.23M output tokens) to 0.78 manifests (563k output tokens, or 149 s of `S` serving) with conservative tail drain, and to 0.36 manifests (260k tokens) when dispatch is tail-aware ($O = 0$). For mid-context `W0 → W7` jobs, in which `S` remains feasible in both phases, PhaseMorph lowers the `W7` break-even threshold $k_m^*$ from 3.78 to 1.38 manifests (110k output tokens).

== Comparison with In-Runtime Submesh Switching <sec-eval-submesh>

PhaseMorph-Submesh switches `2×TP8 → TP16` in 12.04 ms and back in 48.02 ms, three orders of magnitude faster than PhaseMorph's process-isolated switch. The submeshes are also well isolated: they run concurrently at $1.05"–"1.08 times$ their solo step time, and with 1.61 GB per chip of `TP8` weights, compiling all `TP16` buckets in the background (11.82 s) slows foreground `TP8` serving by only 11.8%–13.0%. For bulk jobs, however, this speed matters little. Integrating unified submeshes into vLLM would lower the warm switch cost only to $approx 9"–"10" s"$ and would improve the median makespan gain by 0.3 percentage points. Process-isolated PhaseMorph therefore already captures 99.7% of the achievable bulk makespan gain while keeping each instance its own failure domain.

= Related Work <sec-related>

== LLM Cold-Start and Weight Loading Optimization

Serverless and autoscaled LLM serving has motivated a line of work on cold-start latency. ServerlessLLM @fu2024serverlessllm exploits multi-tier local checkpoint storage and loading-optimized formats, and BlitzScale @zhang2025blitzscale loads parameters over the compute network from host caches and peers so that new instances can serve while still loading. PipeSwitch @bai2020pipeswitch pipelines model transmission with execution to switch deep learning tasks on a GPU in milliseconds. Medusa @zeng2025medusa and Foundry @liu2026foundry target the remaining initialization work on GPUs, materializing CUDA graph topology and, in Foundry's case, the execution context offline so that graphs need not be recaptured online. PhaseMorph shares the materialization principle but applies it to TPU-specific bottlenecks: on TPU, loading is dominated by host-side layout conversion and sharding compilation rather than data transfer, and compiled state consists of XLA executables bound to device permutations, together with a single-owner device runtime that complicates pre-warming.

== Dynamic Parallelism Reconfiguration and Disaggregation

DistServe @zhong2024distserve and Splitwise @patel2024splitwise disaggregate prefill and decode onto separately provisioned instances, and Mooncake @qin2025mooncake builds a KV-cache-centric disaggregated architecture. Llumnix @sun2024llumnix migrates live requests and their KV caches across instances to balance load. LoongServe @wu2024loongserve adjusts the degree of sequence parallelism at iteration granularity with elastic sequence parallelism, and Gyges @chen2025gyges transforms the tensor parallelism of running instances to match request context lengths. These systems target interactive latency on GPUs, mostly between homogeneous instances. PhaseMorph targets bulk makespan on TPU slices, and our measurements show that on this platform it is cheaper to reconfigure the mesh at phase boundaries than to migrate KV state across TP degrees.

== Accelerator Submesh Runtimes and Compilation Caching

Pathways @barham2022pathways schedules computations onto TPU device islands through a single-controller runtime, and Alpa @zheng2022alpa partitions device clusters into submeshes to place pipeline stages. TPU v4 @jouppi2023tpuv4 introduced optically reconfigurable topologies at slice granularity. These systems fix the device mapping for the lifetime of a program or job, whereas PhaseMorph re-partitions a serving slice between workload phases. Our submesh prototype shows that in-runtime re-partitioning is fast once a `libtpu` device-ID constraint is respected, while our process-isolated design shows that persistent XLA caching @openxla is insufficient on its own: compile-time thresholds and process-local tracing must also be addressed.

= Conclusion <sec-conclusion>

On multi-host Cloud TPU slices, no static tensor-parallel layout simultaneously maximizes short-context throughput and long-context KV capacity, so bulk LLM jobs benefit from changing the slice layout between phases. We showed that the multi-minute cost of such a change is an artifact of the software stack rather than of the hardware. PhaseMorph materializes device-ready weights, graphs, and executables ahead of time and prepares target workers in clean-room CPU processes, reducing warm full-slice reconfiguration from 419.0 s to 21.1–27.5 s, partial reconfiguration to 18.8 s, and the cold first switch to 134.1 s, while an in-runtime prototype bounds the hardware cost at 12.0 ms. At these costs, phase-aware switching is the best strategy for 82%–88% of two-phase bulk workloads.

#let refs = ```bib
@inproceedings{kwon2023vllm,
  title = {Efficient Memory Management for Large Language Model Serving with {PagedAttention}},
  author = {Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E. and Zhang, Hao and Stoica, Ion},
  booktitle = {Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)},
  year = {2023}
}
@inproceedings{zhong2024distserve,
  title = {{DistServe}: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving},
  author = {Zhong, Yinmin and Liu, Shengyu and Chen, Junda and Hu, Jianbo and Zhu, Yibo and Liu, Xuanzhe and Jin, Xin and Zhang, Hao},
  booktitle = {18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year = {2024}
}
@inproceedings{patel2024splitwise,
  title = {Splitwise: Efficient Generative {LLM} Inference Using Phase Splitting},
  author = {Patel, Pratyush and Choukse, Esha and Zhang, Chaojie and Shah, Aashaka and Goiri, {\'I}{\~n}igo and Maleki, Saeed and Bianchini, Ricardo},
  booktitle = {51st ACM/IEEE Annual International Symposium on Computer Architecture (ISCA)},
  year = {2024}
}
@inproceedings{qin2025mooncake,
  title = {Mooncake: Trading More Storage for Less Computation --- A {KVCache}-centric Architecture for Serving {LLM} Chatbot},
  author = {Qin, Ruoyu and Li, Zheming and He, Weiran and Cui, Jialei and Ren, Feng and Zhang, Mingxing and Wu, Yongwei and Zheng, Weimin and Xu, Xinran},
  booktitle = {23rd USENIX Conference on File and Storage Technologies (FAST)},
  year = {2025}
}
@inproceedings{sun2024llumnix,
  title = {Llumnix: Dynamic Scheduling for Large Language Model Serving},
  author = {Sun, Biao and Huang, Ziming and Zhao, Hanyu and Xiao, Wencong and Zhang, Xinyi and Li, Yong and Lin, Wei},
  booktitle = {18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year = {2024}
}
@inproceedings{fu2024serverlessllm,
  title = {{ServerlessLLM}: Low-Latency Serverless Inference for Large Language Models},
  author = {Fu, Yao and Xue, Leyang and Huang, Yeqi and Brabete, Andrei-Octavian and Ustiugov, Dmitrii and Patel, Yuvraj and Mai, Luo},
  booktitle = {18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year = {2024}
}
@inproceedings{bai2020pipeswitch,
  title = {{PipeSwitch}: Fast Pipelined Context Switching for Deep Learning Applications},
  author = {Bai, Zhihao and Zhang, Zhen and Zhu, Yibo and Jin, Xin},
  booktitle = {14th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year = {2020}
}
@inproceedings{barham2022pathways,
  title = {Pathways: Asynchronous Distributed Dataflow for {ML}},
  author = {Barham, Paul and Chowdhery, Aakanksha and Dean, Jeff and Ghemawat, Sanjay and Hand, Steven and Hurt, Dan and Isard, Michael and Lim, Hyeontaek and Pang, Ruoming and Roy, Sudip and others},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys)},
  year = {2022}
}
@inproceedings{zheng2022alpa,
  title = {Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning},
  author = {Zheng, Lianmin and Li, Zhuohan and Zhang, Hao and Zhuang, Yonghao and Chen, Zhifeng and Huang, Yanping and Wang, Yida and Xu, Yuanzhong and Zhuo, Danyang and Xing, Eric P. and Gonzalez, Joseph E. and Stoica, Ion},
  booktitle = {16th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year = {2022}
}
@inproceedings{jouppi2023tpuv4,
  title = {{TPU} v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings},
  author = {Jouppi, Norm and Kurian, George and Li, Sheng and Ma, Peter and Nagarajan, Rahul and Nai, Lifeng and Patil, Nishant and Subramanian, Suvinay and Swing, Andy and Towles, Brian and others},
  booktitle = {Proceedings of the 50th Annual International Symposium on Computer Architecture (ISCA)},
  year = {2023}
}
@misc{googlev5e,
  title = {Cloud {TPU} v5e},
  author = {{Google Cloud}},
  howpublished = {Cloud TPU documentation},
  url = {https://cloud.google.com/tpu/docs/v5e},
  year = {2023}
}
@article{gemma2024,
  title = {Gemma: Open Models Based on {Gemini} Research and Technology},
  author = {{Gemma Team}},
  journal = {arXiv preprint arXiv:2403.08295},
  year = {2024}
}
@article{liu2026foundry,
  title = {Foundry: Template-Based {CUDA} Graph Context Materialization for Fast {LLM} Serving Cold Start},
  author = {Liu, Xueshen and Wu, Yongji and Yao, Yuncheng and Zhuo, Danyang and Stoica, Ion and Mao, Z. Morley},
  journal = {arXiv preprint arXiv:2604.06664},
  year = {2026}
}
@inproceedings{zeng2025medusa,
  title = {Medusa: Accelerating Serverless {LLM} Inference with Materialization},
  author = {Zeng, Shaoxun and Xie, Minhui and Gao, Shiwei and Chen, Youmin and Lu, Youyou},
  booktitle = {Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)},
  year = {2025}
}
@inproceedings{zhang2025blitzscale,
  title = {{BlitzScale}: Fast and Live Large Model Autoscaling with {O(1)} Host Caching},
  author = {Zhang, Dingyan and Wang, Haotian and Liu, Yang and Wei, Xingda and Shan, Yizhou and Chen, Rong and Chen, Haibo},
  booktitle = {19th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year = {2025}
}
@inproceedings{wu2024loongserve,
  title = {{LoongServe}: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism},
  author = {Wu, Bingyang and Liu, Shengyu and Zhong, Yinmin and Sun, Peng and Liu, Xuanzhe and Jin, Xin},
  booktitle = {Proceedings of the 30th ACM Symposium on Operating Systems Principles (SOSP)},
  year = {2024}
}
@article{chen2025gyges,
  title = {Gyges: Dynamic Cross-Instance Parallelism Transformation for Efficient {LLM} Inference},
  author = {Chen, Haoyu and Li, Xue and Qian, Kun and Guan, Yu and Zhao, Jin and Wang, Xin},
  journal = {arXiv preprint arXiv:2509.19729},
  year = {2025}
}
@misc{bradbury2018jax,
  title = {{JAX}: Composable Transformations of {Python}+{NumPy} Programs},
  author = {Bradbury, James and Frostig, Roy and Hawkins, Peter and Johnson, Matthew James and Leary, Chris and Maclaurin, Dougal and Necula, George and Paszke, Adam and VanderPlas, Jake and Wanderman-Milne, Skye and Zhang, Qiao},
  url = {https://github.com/jax-ml/jax},
  year = {2018}
}
@inproceedings{moritz2018ray,
  title = {Ray: A Distributed Framework for Emerging {AI} Applications},
  author = {Moritz, Philipp and Nishihara, Robert and Wang, Stephanie and Tumanov, Alexey and Liaw, Richard and Liang, Eric and Elibol, Melih and Yang, Zongheng and Paul, William and Jordan, Michael I. and Stoica, Ion},
  booktitle = {13th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year = {2018}
}
@misc{openxla,
  title = {{XLA}: Optimizing Compiler for Machine Learning},
  author = {{OpenXLA Project}},
  url = {https://openxla.org/xla},
  year = {2024}
}
@misc{heek2023flax,
  title = {Flax: A Neural Network Library and Ecosystem for {JAX}},
  author = {Heek, Jonathan and Levskaya, Anselm and Oliver, Avital and Ritter, Marvin and Rondepierre, Bertrand and Steiner, Andreas and van Zee, Marc},
  url = {https://github.com/google/flax},
  year = {2023}
}
@misc{llmd,
  title = {llm-d: Kubernetes-Native Distributed Inference at Scale},
  author = {{llm-d Project}},
  url = {https://github.com/llm-d/llm-d},
  year = {2025}
}
@misc{tpuinference,
  title = {tpu-inference: {TPU} Backend for {vLLM}},
  author = {{vLLM Project}},
  url = {https://github.com/vllm-project/tpu-inference},
  year = {2025}
}
```.text

#bibliography(bytes(refs), title: "References", style: "springer-lecture-notes-in-computer-science")

#counter(heading).update(0)
#set heading(numbering: "A.1.", supplement: [Appendix])

= Bulk Cost Model and Break-Even Derivation <app-cost-model>

Let $a_"0S" = 190.8" s"$ and $a_"0H" = 267.8" s"$ be the per-manifest makespans of `W0` on `S` and on `H` (at a 131k-token limit), $a_"6H" = 206.2" s"$ that of `W6` on `H`, and $O = 32.6" s"$ the tail-drain overhead of the last `S` manifest before a switch. For a `W0 → W6` job, `S` cannot serve the long phase, so the relevant static competitor is `H`. Comparing the full-switch and static-`H` makespans of @sec-overview,
$ T_"S→H" < T_"H" &<==> 190.8 k_s + O + C_"full" < 267.8 k_s \
  &<==> k_s > k_s^* = (C_"full" + O) / (a_"0H" - a_"0S"). $
The long-phase terms cancel, so the break-even threshold depends only on the short phase. With $C_"full" = 27.5" s"$, $k_s^* = 60.1 \/ 77.0 = 0.78$ manifests, or 563k output tokens at 721,595 tokens per manifest; with tail-aware dispatch ($O = 0$), $k_s^* = 0.36$.

For a `W0 → W7` job, `S` is feasible in both phases, so the competitor is static `S`, and switching to `H` for the mid-context phase saves $a_"7S" - a_"7H" = 248.4 - 204.8 = 43.6" s"$ per `W7` manifest. The mid-phase break-even threshold is therefore
$ k_m^* = (C_"full" + O) / (a_"7S" - a_"7H"), $
which gives 1.38 `W7` manifests (110k output tokens at 79,531 tokens per manifest) for PhaseMorph and 10.35 manifests for vanilla redeployment. The coverage and gain columns of @tab-breakeven evaluate the four strategy equations of @sec-overview on a grid over $(k_s, k_l) in [0, 8] times [0.25, 8]$ and report, at each point, the best strategy and its reduction relative to the best static layout.

= Reconfiguration Segment Definitions <app-segments>

The segments of @tab-segments partition the dispatch gap of a switch. _Drain + release_ ends when the source engines have finished their last requests and exited. _Gate + VFIO wait_ covers opening the release gate and waiting until every VFIO device group can be opened. _Spawn + imports_ covers creating the target worker processes and importing the serving stack; with pre-spawning it is hidden in the source phase and reported in parentheses. `init_device` initializes the TPU backend, `load_model` places weights on the devices, `init_from_config` allocates the KV cache, and `compile_warmup` compiles or loads all executables and runs warm-up steps. _Gateway ready_ ends when the router dispatches the first target-phase request. The ramp term is the deficit of the first target-phase interval relative to the target layout's steady-state rate, and is negative when the new engines run ahead of it.
