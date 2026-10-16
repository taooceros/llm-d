// PhaseMorph: Sub-30-Second Elastic TPU Mesh Reconfiguration and Autoscaling for LLM Serving
// MLSys (mlsys.sty / icml.sty) two-column layout for PDF/paged view; single-column academic HTML for ?g-output=html.
//   HTML: typst compile --features html --format html phasemorph_mlsys_paper.typ /tmp/phasemorph.html
//   PDF:  typst compile phasemorph_mlsys_paper.typ /tmp/phasemorph.pdf

#let title = "PhaseMorph: Sub-30-Second Elastic Mesh Reconfiguration and Autoscaling for LLM Serving on Cloud TPU"
#let running-title = "PhaseMorph: Elastic Mesh Reconfiguration and Autoscaling for LLM Serving on Cloud TPU"
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
  Elastic LLM serving requires both _horizontal instance autoscaling_ and _intra-slice parallelism reconfiguration_ as request contexts shift between short and long regimes. Whereas single-node GPU servers with large per-device HBM and NVSwitch crossbars typically operate at a fixed tensor-parallel (TP) degree during fault-free service, multi-host Cloud TPU slices make workload-driven TP elasticity necessary. On a 32-chip Cloud TPU v5e slice ($4 times 8$ torus, 16 GB HBM/chip) serving Gemma-4 31B in BF16, four `TP8` instances (`S`) reduce short-context makespan by 28.8%–37.7% over two `TP16` instances (`H`) via shorter ring collectives and deeper systolic contractions, whereas `TP16` instances provide $2.75 times$ more usable KV capacity, complete mid-context workloads 17.6% faster, and are required beyond 52.9k tokens. Because static heterogeneous partitioning is optimal in only 6% of two-phase workloads and live cross-TP KV migration is slower than recomputation, dynamically re-partitioning the slice is preferable—yet redeploying vLLM engines incurs a 350–419 s outage.

  We show that 95% of this latency stems not from storage I/O or device bring-up, but from software-stack overheads around ahead-of-time XLA compilation: host tensor layout conversion, sharding compilation, threshold eviction of sub-second persistent-cache entries, process-local JAX tracing, and exclusive PJRT device locks. Conversely, JAX/XLA's functional execution model permits state materialization without the pointer-relocation and symbol-resolution machinery required by imperative GPU graph-capture systems. We present _PhaseMorph_, an elastic serving system for vLLM on Cloud TPU that materializes pre-sharded weight snapshots, shape-only graphs, and a device-order-preserving bank of serialized PJRT executables, while pre-initializing workers in TPU-decoupled CPU processes with a VFIO-gated handoff. PhaseMorph reduces warm full-slice reconfiguration by 93%–95% (419.0 s to 21.1–27.5 s), partial reconfiguration to 18.8 s, and cold first switches by 40% (222.6 s to 134.1 s) via background CPU pre-reading, outperforming static layouts on 82%–88% of two-phase workloads (9.9% median, 26.8% max makespan reduction).
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

The widespread deployment of large language models (LLMs) across both interactive applications and batch processing pipelines—including synthetic data generation, long-document summarization, repository-scale code analysis, and model evaluation—has made _elastic serving_ a central systems requirement @kwon2023vllm @fu2024serverlessllm @zhang2025blitzscale. Workload demand fluctuates in both request arrival rate and sequence-length distribution, shifting between compute-bound short-context bursts and memory-capacity-bound long-context phases. Accommodating these shifts efficiently requires two forms of elasticity: _horizontal instance autoscaling_, which provisions or decommissions serving replicas as aggregate load varies @fu2024serverlessllm @zeng2025medusa @liu2026foundry, and _parallelism reconfiguration_, which adapts the tensor- or sequence-parallel degree of running instances to match active request characteristics or available hardware @wu2024loongserve @chen2025gyges @miao2024spotserve @xu2025failsafe.

While elasticity on GPU clusters has focused primarily on replica-level serverless scaling @fu2024serverlessllm @zeng2025medusa or fault recovery under device preemption and failure @miao2024spotserve @xu2025failsafe, multi-host Cloud TPU slices @googlev5e @jouppi2023tpuv4 make _workload-driven tensor-parallel (TP) reconfiguration_ essential even during fault-free operation. Unlike single-node GPU servers that combine large per-device memory with all-to-all NVSwitch crossbars (e.g., 640 GB aggregate HBM and 900 GB/s single-hop bandwidth on 8$times$H100), Cloud TPU v5e pods pair 16 GB of HBM2e per chip and $128 times 128$ systolic matrix units (MXUs) with a multi-host 2D torus Inter-Chip Interconnect (ICI). On a 32-chip ($4 times 8$) v5e slice serving Gemma-4 31B @gemma2024 in BF16 (61.4 GB parameters), this architecture creates a fundamental tension between compute utilization and KV-cache capacity. Partitioning the slice into four 8-chip instances (`S`: $4 times$`TP8`) reduces short-context chat (`W0`) makespan by 28.8%–37.7% compared to two 16-chip instances (`H`: $2 times$`TP16`), because smaller rings halve AllReduce hop counts ($14$ vs. $30$) and maintain deeper per-chip MXU contractions ($K = 512$ vs. $256$, yielding 50.95% vs. 30.67% MFU). Conversely, each `TP8` instance retains only 52,864 tokens of usable KV cache after model weights, whereas `TP16` halves per-chip parameter shards and expands usable KV capacity by $2.75 times$ to 145,216 tokens; `H` therefore completes mid-context document workloads (`W7`) 17.6% faster than `S` and is the only feasible layout once contexts exceed 52.9k tokens (`W6`). Because static heterogeneous partitioning (`M`: `TP16` + $2 times$`TP8`) is optimal in only 6% of two-phase workloads and live cross-TP KV migration is slower than MXU prefill recomputation (@sec-static), dynamically re-partitioning the TPU slice across workload phases is necessary to minimize makespan.

Despite the benefits of elastic TPU serving, scaling out a serving instance or reconfiguring a slice between `S`, `M`, and `H` is bottlenecked by a severe _cold-start problem_. In vanilla vLLM @kwon2023vllm on Cloud TPU, redeploying engines across a 32-chip slice incurs a 350–419 s outage (@fig-waterfall), requiring a short-context phase of more than 4.2 million output tokens before a single transition breaks even (@sec-eval). Existing work accelerates GPU cold starts and autoscaling by optimizing storage-to-device weight transfer @fu2024serverlessllm @zhang2025blitzscale or reusing warm container environments @bai2020pipeswitch. However, we find that on Cloud TPU—where execution relies exclusively on ahead-of-time (AOT) XLA-compiled @openxla SPMD binaries across 187 shape buckets rather than eager kernel launches—storage reads (0.2 s) and host-to-device DMA (2.0–3.0 s) account for only 1.2% of model loading, so staging checkpoints in `/dev/shm` yields no speedup. Instead, three software-stack stages dominate the reconfiguration critical path: (1) _host-side weight layout transformation and sharding compilation_ (237.0 s, comprising $approx 128" s"$ of CPU tensor transpose/reshape, $75"–"77" s"$ of per-parameter sharding JIT, and 14.6 s of Flax @heek2023flax abstract graph evaluation), (2) _repeated AOT compilation and process-local JAX tracing_ (exceeding 80 s even with a warm persistent XLA cache, because JAX @bradbury2018jax evicts 248 sub-second helper compilations per worker by default and re-executes Python tracing and StableHLO lowering in every new process), and (3) _serialized worker initialization and device handoff_ (32.7 s of process startup plus 11–12 s of device-release contention). Simply omitting compilation buckets or retaining weights inside a single multi-host runtime either degrades serving throughput or fails due to `libtpu` sub-torus routing restrictions (@sec-design-submesh).

To mitigate this cold-start latency without sacrificing steady-state throughput or process fault isolation, we propose _ahead-of-time serving-state materialization_ and _TPU-decoupled worker preparation_. Rather than performing tensor layout conversion, abstract graph evaluation, Python tracing, and helper compilation online during a transition, our approach materializes device-ready weight shards, model graph metadata, and compiled SPMD executables ahead of time and restores them on the critical path. While conceptually appealing, realizing state materialization and background worker preparation on Cloud TPU raises three technical challenges:
1. _Preserving compiled state across processes and device permutations:_ Unlike imperative GPU CUDA graphs—which embed non-deterministic virtual pointers from `cudaMalloc` and unexported library symbol addresses, forcing systems such as Medusa @zeng2025medusa and Foundry @liu2026foundry to replay allocation sequences and run warmup triggering kernels—compiled TPU programs are multi-host SPMD binaries bound to specific torus device assignments (e.g., non-contiguous permutations such as `[0,1,4,5,2,3,...]` on `TP16`) and preceded by process-local Python tracing.
2. _Eliminating host conversion on cold first-time transitions:_ Pre-sharded weight snapshots eliminate `load_model` overhead on warm transitions, but the first transition into an unprovisioned target layout still faces 130.7 s of host-CPU tensor conversion that cannot use the TPU while the source layout is serving.
3. _Overlapping worker initialization under single-owner device locks:_ Spawning target workers in the background while the source layout serves risks aborting on PJRT's exclusive lockfile (`/tmp/libtpu_lockfile`) if any library import queries TPU topology, and freshly unblocked workers race the Linux kernel's asynchronous `/dev/vfio` device reset (`EBUSY`) upon source termination.
#wide-fig(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 326" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<defs><pattern id="wr" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="4" height="4" fill="#fff"/><line x1="0" y1="0" x2="0" y2="4" stroke="#444" stroke-width="1.4"/></pattern></defs>
<line x1="172" y1="22" x2="172" y2="258" stroke="#e2e2e2" stroke-width="1"/>
<text x="172" y="18" text-anchor="middle" fill="#666">0 s</text>
<line x1="262.5" y1="22" x2="262.5" y2="258" stroke="#e2e2e2" stroke-width="1"/>
<text x="262.5" y="18" text-anchor="middle" fill="#666">100 s</text>
<line x1="353" y1="22" x2="353" y2="258" stroke="#e2e2e2" stroke-width="1"/>
<text x="353" y="18" text-anchor="middle" fill="#666">200 s</text>
<line x1="443.4" y1="22" x2="443.4" y2="258" stroke="#e2e2e2" stroke-width="1"/>
<text x="443.4" y="18" text-anchor="middle" fill="#666">300 s</text>
<line x1="533.9" y1="22" x2="533.9" y2="258" stroke="#e2e2e2" stroke-width="1"/>
<text x="533.9" y="18" text-anchor="middle" fill="#666">400 s</text>
<text x="164" y="34" text-anchor="end" font-size="11.5" font-weight="bold">Vanilla vLLM redeploy</text>
<text x="164" y="47" text-anchor="end" fill="#666">rep. run; median of 6 = 419.0 s</text>
<rect x="172" y="26" width="23" height="18" fill="#8c8c8c" stroke="#fff" stroke-width="0.6"/>
<rect x="195" y="26" width="66.2" height="18" fill="#e4e4e4" stroke="#fff" stroke-width="0.6"/>
<text x="228.1" y="39" text-anchor="middle">73.2</text>
<rect x="261.2" y="26" width="214.6" height="18" fill="#CC6677" stroke="#fff" stroke-width="0.6"/>
<text x="368.5" y="39" text-anchor="middle" fill="#fff">237.2</text>
<rect x="475.8" y="26" width="56.3" height="18" fill="#AA4499" stroke="#fff" stroke-width="0.6"/>
<text x="504" y="39" text-anchor="middle" fill="#fff">62.2</text>
<rect x="532.1" y="26" width="14.1" height="18" fill="#5f5f5f" stroke="#fff" stroke-width="0.6"/>
<rect x="546.2" y="26" width="5.1" height="18" fill="url(#wr)" stroke="#444" stroke-width="0.5"/>
<text x="556.3" y="39" font-weight="bold">419.2 s</text>
<text x="164" y="64" text-anchor="end" font-size="11.5" font-weight="bold">PhaseMorph-Snap</text>
<text x="164" y="77" text-anchor="end" fill="#666">snapshot + FastGraph</text>
<rect x="172" y="56" width="3.4" height="18" fill="#8c8c8c" stroke="#fff" stroke-width="0.6"/>
<rect x="175.4" y="56" width="10.1" height="18" fill="#c8c8c8" stroke="#fff" stroke-width="0.6"/>
<rect x="185.4" y="56" width="19.5" height="18" fill="#88CCEE" stroke="#fff" stroke-width="0.6"/>
<rect x="204.9" y="56" width="3.2" height="18" fill="#DDCC77" stroke="#fff" stroke-width="0.6"/>
<rect x="208.2" y="56" width="4.5" height="18" fill="#CC6677" stroke="#fff" stroke-width="0.6"/>
<rect x="212.7" y="56" width="23.5" height="18" fill="#999933" stroke="#fff" stroke-width="0.6"/>
<rect x="236.1" y="56" width="49.4" height="18" fill="#AA4499" stroke="#fff" stroke-width="0.6"/>
<text x="260.8" y="69" text-anchor="middle" fill="#fff">54.6</text>
<rect x="285.5" y="56" width="4.6" height="18" fill="#5f5f5f" stroke="#fff" stroke-width="0.6"/>
<rect x="290.2" y="56" width="1.5" height="18" fill="url(#wr)" stroke="#444" stroke-width="0.5"/>
<text x="296.6" y="69" font-weight="bold">132.2 s</text>
<text x="164" y="94" text-anchor="end" font-size="11.5" font-weight="bold">PhaseMorph-Cache</text>
<text x="164" y="107" text-anchor="end" fill="#666">+ zero-threshold cache</text>
<rect x="172" y="86" width="3.2" height="18" fill="#8c8c8c" stroke="#fff" stroke-width="0.6"/>
<rect x="175.2" y="86" width="11" height="18" fill="#c8c8c8" stroke="#fff" stroke-width="0.6"/>
<rect x="186.2" y="86" width="19.5" height="18" fill="#88CCEE" stroke="#fff" stroke-width="0.6"/>
<rect x="205.7" y="86" width="3.3" height="18" fill="#DDCC77" stroke="#fff" stroke-width="0.6"/>
<rect x="209" y="86" width="4.4" height="18" fill="#CC6677" stroke="#fff" stroke-width="0.6"/>
<rect x="213.3" y="86" width="2.4" height="18" fill="#999933" stroke="#fff" stroke-width="0.6"/>
<rect x="215.7" y="86" width="31.8" height="18" fill="#AA4499" stroke="#fff" stroke-width="0.6"/>
<text x="231.6" y="99" text-anchor="middle" fill="#fff">35.1</text>
<rect x="247.5" y="86" width="3.4" height="18" fill="#5f5f5f" stroke="#fff" stroke-width="0.6"/>
<rect x="250.9" y="86" width="2.4" height="18" fill="url(#wr)" stroke="#444" stroke-width="0.5"/>
<text x="258.3" y="99" font-weight="bold">89.8 s</text>
<text x="164" y="124" text-anchor="end" font-size="11.5" font-weight="bold">PhaseMorph, full S→H</text>
<text x="164" y="137" text-anchor="end" fill="#666">+ pre-spawn + exec. bank</text>
<rect x="172" y="116" width="1.8" height="18" fill="#8c8c8c" stroke="#fff" stroke-width="0.6"/>
<rect x="173.8" y="116" width="0.8" height="18" fill="#c8c8c8" stroke="#fff" stroke-width="0.6"/>
<rect x="174.6" y="116" width="3.5" height="18" fill="#DDCC77" stroke="#fff" stroke-width="0.6"/>
<rect x="178.1" y="116" width="4.4" height="18" fill="#CC6677" stroke="#fff" stroke-width="0.6"/>
<rect x="182.6" y="116" width="2.3" height="18" fill="#999933" stroke="#fff" stroke-width="0.6"/>
<rect x="184.9" y="116" width="4.5" height="18" fill="#AA4499" stroke="#fff" stroke-width="0.6"/>
<rect x="189.4" y="116" width="4.3" height="18" fill="#5f5f5f" stroke="#fff" stroke-width="0.6"/>
<rect x="193.7" y="116" width="3.2" height="18" fill="url(#wr)" stroke="#444" stroke-width="0.5"/>
<text x="201.9" y="129" font-weight="bold">27.5 s (21.1 s in run B)</text>
<text x="164" y="154" text-anchor="end" font-size="11.5" font-weight="bold">PhaseMorph, partial S→M</text>
<text x="164" y="167" text-anchor="end" fill="#666">C<tspan dy="2.6" font-size="8.8">eff</tspan><tspan dy="-2.6">; two TP8 keep serving</tspan></text>
<rect x="172" y="146" width="1.7" height="18" fill="#8c8c8c" stroke="#fff" stroke-width="0.6"/>
<rect x="173.7" y="146" width="0.8" height="18" fill="#c8c8c8" stroke="#fff" stroke-width="0.6"/>
<rect x="174.4" y="146" width="3.4" height="18" fill="#DDCC77" stroke="#fff" stroke-width="0.6"/>
<rect x="177.8" y="146" width="4.5" height="18" fill="#CC6677" stroke="#fff" stroke-width="0.6"/>
<rect x="182.3" y="146" width="2.3" height="18" fill="#999933" stroke="#fff" stroke-width="0.6"/>
<rect x="184.6" y="146" width="4.6" height="18" fill="#AA4499" stroke="#fff" stroke-width="0.6"/>
<rect x="189.2" y="146" width="3.3" height="18" fill="#5f5f5f" stroke="#fff" stroke-width="0.6"/>
<line x1="189" y1="143" x2="189" y2="167" stroke="#000" stroke-width="1.6"/>
<text x="197.5" y="159" font-weight="bold">C<tspan dy="2.6" font-size="8.8">eff</tspan><tspan dy="-2.6"> 18.8 s</tspan></text>
<text x="164" y="184" text-anchor="end" font-size="11.5" font-weight="bold">Cold first switch</text>
<text x="164" y="197" text-anchor="end" fill="#666">inline snapshot, no pre-read</text>
<rect x="172" y="176" width="2.6" height="18" fill="#8c8c8c" stroke="#fff" stroke-width="0.6"/>
<rect x="174.6" y="176" width="3.5" height="18" fill="#DDCC77" stroke="#fff" stroke-width="0.6"/>
<rect x="178.2" y="176" width="143.8" height="18" fill="#CC6677" stroke="#fff" stroke-width="0.6"/>
<text x="250.1" y="189" text-anchor="middle" fill="#fff">158.9</text>
<rect x="322" y="176" width="6.8" height="18" fill="#882255" stroke="#fff" stroke-width="0.6"/>
<rect x="328.8" y="176" width="2.4" height="18" fill="#999933" stroke="#fff" stroke-width="0.6"/>
<rect x="331.2" y="176" width="32.4" height="18" fill="#AA4499" stroke="#fff" stroke-width="0.6"/>
<text x="347.4" y="189" text-anchor="middle" fill="#fff">35.8</text>
<rect x="363.5" y="176" width="2" height="18" fill="#882255" stroke="#fff" stroke-width="0.6"/>
<rect x="365.6" y="176" width="5" height="18" fill="#5f5f5f" stroke="#fff" stroke-width="0.6"/>
<rect x="370.6" y="176" width="2.9" height="18" fill="url(#wr)" stroke="#444" stroke-width="0.5"/>
<text x="378.4" y="189" font-weight="bold">222.6 s</text>
<text x="164" y="214" text-anchor="end" font-size="11.5" font-weight="bold">PhaseMorph-PreRead</text>
<text x="164" y="227" text-anchor="end" fill="#666">cold first switch</text>
<rect x="172" y="206" width="2.5" height="18" fill="#8c8c8c" stroke="#fff" stroke-width="0.6"/>
<rect x="174.5" y="206" width="3.7" height="18" fill="#DDCC77" stroke="#fff" stroke-width="0.6"/>
<rect x="178.2" y="206" width="72" height="18" fill="#ebbfc7" stroke="#fff" stroke-width="0.6"/>
<text x="214.2" y="219" text-anchor="middle">79.6</text>
<rect x="250.2" y="206" width="5.2" height="18" fill="#CC6677" stroke="#fff" stroke-width="0.6"/>
<rect x="255.4" y="206" width="2.3" height="18" fill="#999933" stroke="#fff" stroke-width="0.6"/>
<rect x="257.7" y="206" width="31.9" height="18" fill="#AA4499" stroke="#fff" stroke-width="0.6"/>
<text x="273.7" y="219" text-anchor="middle" fill="#fff">35.3</text>
<rect x="289.6" y="206" width="2.1" height="18" fill="#882255" stroke="#fff" stroke-width="0.6"/>
<rect x="291.7" y="206" width="4.4" height="18" fill="#5f5f5f" stroke="#fff" stroke-width="0.6"/>
<line x1="293.3" y1="203" x2="293.3" y2="227" stroke="#000" stroke-width="1.6"/>
<text x="301" y="219" font-weight="bold">134.1 s</text>
<text x="164" y="244" text-anchor="end" font-size="11.5" font-weight="bold">PhaseMorph-Submesh</text>
<text x="164" y="257" text-anchor="end" fill="#666">synthetic model, 4×4 runtime</text>
<rect x="172" y="236" width="1.5" height="18" fill="#000"/>
<text x="178" y="249">12.0 ms (2×TP8 → TP16) / 48.0 ms back; 1.61 GB/chip weights, no vLLM engine</text>
<rect x="8" y="273" width="10" height="10" fill="#8c8c8c" stroke="#555" stroke-width="0.5"/>
<text x="22" y="282">drain + release</text>
<rect x="166" y="273" width="10" height="10" fill="#c8c8c8" stroke="#555" stroke-width="0.5"/>
<text x="180" y="282">release gate + VFIO</text>
<rect x="324" y="273" width="10" height="10" fill="#88CCEE" stroke="#555" stroke-width="0.5"/>
<text x="338" y="282">spawn + imports</text>
<rect x="482" y="273" width="10" height="10" fill="#DDCC77" stroke="#555" stroke-width="0.5"/>
<text x="496" y="282">init_device</text>
<rect x="8" y="288" width="10" height="10" fill="#e4e4e4" stroke="#555" stroke-width="0.5"/>
<text x="22" y="297">other deploy work</text>
<rect x="166" y="288" width="10" height="10" fill="#CC6677" stroke="#555" stroke-width="0.5"/>
<text x="180" y="297">load_model</text>
<rect x="324" y="288" width="10" height="10" fill="#ebbfc7" stroke="#555" stroke-width="0.5"/>
<text x="338" y="297">pre-read wait</text>
<rect x="482" y="288" width="10" height="10" fill="#999933" stroke="#555" stroke-width="0.5"/>
<text x="496" y="297">KV cache init</text>
<rect x="8" y="303" width="10" height="10" fill="#AA4499" stroke="#555" stroke-width="0.5"/>
<text x="22" y="312">compile / warm-up</text>
<rect x="166" y="303" width="10" height="10" fill="#882255" stroke="#555" stroke-width="0.5"/>
<text x="180" y="312">snapshot + bank write</text>
<rect x="324" y="303" width="10" height="10" fill="#5f5f5f" stroke="#555" stroke-width="0.5"/>
<text x="338" y="312">endpoint + gateway</text>
<rect x="482" y="303" width="10" height="10" fill="url(#wr)" stroke="#555" stroke-width="0.5"/>
<text x="496" y="312">ramp; tick = C if ramp &lt; 0</text>
</svg>
```),
  caption: [[Reconfiguration Latency Reduction] Critical-path reconfiguration cost $C$ as PhaseMorph's mechanisms are added, drawn to scale from the segments of @tab-segments. Rows 2–5 are warm switches (full `S → H` unless noted; for the partial `S → M` switch the tick marks $C_"eff"$); rows 6–7 are cold first switches into `H`. The vanilla bar is one representative redeploy (`M → S`, 419.2 s; the median of six is 419.0 s), whose "other deploy work" covers worker start-up, device and KV-cache initialization, and registration. Hatched: ramp deficit; a black tick marks $C$ when the ramp is negative. The last row is not comparable to the others: PhaseMorph-Submesh switches a synthetic model with 1.61 GB/chip of weights inside one 16-chip ($4 times 4$) runtime, in 12.0 ms.],
) <fig-waterfall>
To address the first challenge, we observe that XLA's PJRT `LoadedExecutable` objects are hermetic, address-independent binaries parameterized by abstract buffer handles; by disabling sub-second persistent-cache eviction and serializing all 187 AOT executables per rank alongside their exact `execution_devices` permutation into a shared-memory _Executable Bank_, new workers restore all compiled state in 3.45 s without Python tracing, backend compilation, or pointer relocation (@sec-design-exec). To address the second challenge, we exploit XLA's backend-agnostic `NamedSharding` semantics to stream and shard weights concurrently in a low-priority CPU child process configured with 16 virtual host devices (`JAX_PLATFORMS=cpu`) during the source phase, producing 32,896 shards bitwise-identical to TPU-generated snapshots alongside shape-only Flax graph construction (@sec-design-weights). To address the third challenge, we leverage JAX's deferred device binding to pre-spawn target workers in CPU-only Ray bundles verified via `/proc/self` to hold no `libtpu` or `/dev/vfio` descriptors, paired with a release gate and VFIO-polling barrier that synchronize device handoff (@sec-design-prespawn).

We build _PhaseMorph_ across vLLM, `tpu-inference` @tpuinference, Ray @moritz2018ray, and the `llm-d` @llmd router, and evaluate it on a 32-chip Cloud TPU v5e slice serving Gemma-4 31B across five workloads. PhaseMorph reduces warm full-slice reconfiguration and scale-out latency by 93%–95% (from 419.0 s to 21.1–27.5 s) and partial reconfiguration latency to 18.8 s without degrading foreground throughput; for cold first-time transitions, background CPU pre-reading reduces transition latency by 40% (from 222.6 s to 134.1 s). Across a two-phase workload space, PhaseMorph lowers the break-even phase size by $7.5 times$ and outperforms the best static layout on 82%–88% of configurations (9.9% median and 26.8% maximum makespan reduction), capturing $approx 97%$ of the makespan benefit of a 12.0 ms 16-chip unified-runtime submesh prototype while preserving per-instance fault isolation. In summary, we make the following contributions:

- We characterize the throughput–capacity trade-off across tensor-parallel degrees on multi-host Cloud TPU slices and provide a quantitative breakdown of the 350–419 s cold-start and reconfiguration bottleneck in vLLM-on-TPU.
- We design PhaseMorph, introducing sharding-parameterized weight and shape-only graph materialization, a device-order-preserving PJRT Executable Bank with zero-threshold caching, background virtual-device CPU pre-reading, and TPU-decoupled worker pre-spawning with VFIO-gated handoff.
- We evaluate PhaseMorph across warm, partial, and cold transitions on a 32-chip TPU v5e slice, analyze its break-even thresholds and multi-phase coverage, and characterize the trade-offs between process-isolated reconfiguration and in-runtime submesh switching on $4 times 4$ and $4 times 8$ TPU topologies.
= Background and Motivation <sec-motivation>

== Parallelism Trade-Offs on Cloud TPU Slices <sec-tradeoffs>

Our testbed is an 8-host, 32-chip Cloud TPU v5e slice @googlev5e arranged as a $4 times 8$ 2D torus with wrap-around inter-chip interconnect (ICI) links, the same torus-of-chips organization introduced with earlier TPU generations @jouppi2023tpuv4. Each host drives four chips, each with 16 GB of HBM2e at 819 GB/s, 197 TFLOPS of BF16 peak compute across four $128 times 128$ matrix units (MXUs), and 1.00 MB of on-chip scalar memory (SMEM) per TensorCore. We serve Gemma-4 31B @gemma2024 in native BF16: 48 layers, hidden size 4,096, FFN intermediate size 21,504, 32 query heads and 16 KV heads (grouped-query attention with ratio $2:1$), head dimension 128, and a 61.4 GB parameter footprint.

@fig-layouts shows how the three layouts studied here partition the slice.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 238" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<text x="63.5" y="14" text-anchor="middle" font-size="13" font-weight="bold" fill="#0072B2">S</text>
<text x="63.5" y="28" text-anchor="middle">4×TP8</text>
<path d="M23.5 41.5h38v38h-38zM65.5 41.5h38v38h-38zM23.5 83.5h38v38h-38zM65.5 83.5h38v38h-38zM23.5 125.5h38v38h-38zM65.5 125.5h38v38h-38zM23.5 167.5h38v38h-38zM65.5 167.5h38v38h-38z" fill="#c6dcef" stroke="#9a9a9a" stroke-width="0.6"/>
<path d="M26 44h15v15h-15zM44 44h15v15h-15zM68 44h15v15h-15zM86 44h15v15h-15zM26 62h15v15h-15zM44 62h15v15h-15zM68 62h15v15h-15zM86 62h15v15h-15zM26 86h15v15h-15zM44 86h15v15h-15zM68 86h15v15h-15zM86 86h15v15h-15zM26 104h15v15h-15zM44 104h15v15h-15zM68 104h15v15h-15zM86 104h15v15h-15zM26 128h15v15h-15zM44 128h15v15h-15zM68 128h15v15h-15zM86 128h15v15h-15zM26 146h15v15h-15zM44 146h15v15h-15zM68 146h15v15h-15zM86 146h15v15h-15zM26 170h15v15h-15zM44 170h15v15h-15zM68 170h15v15h-15zM86 170h15v15h-15zM26 188h15v15h-15zM44 188h15v15h-15zM68 188h15v15h-15zM86 188h15v15h-15z" fill="#fff" stroke="#777" stroke-width="0.6"/>
<rect x="21.4" y="39.4" width="42.2" height="84.2" fill="none" stroke="#0072B2" stroke-width="1.6" rx="4"/>
<rect x="29.3" y="73.5" width="26.4" height="15" fill="#fff" stroke="#0072B2" stroke-width="1" rx="3"/>
<text x="42.5" y="85.5" text-anchor="middle" font-weight="bold" fill="#0072B2">TP8</text>
<rect x="63.4" y="39.4" width="42.2" height="84.2" fill="none" stroke="#0072B2" stroke-width="1.6" rx="4"/>
<rect x="71.3" y="73.5" width="26.4" height="15" fill="#fff" stroke="#0072B2" stroke-width="1" rx="3"/>
<text x="84.5" y="85.5" text-anchor="middle" font-weight="bold" fill="#0072B2">TP8</text>
<rect x="21.4" y="123.4" width="42.2" height="84.2" fill="none" stroke="#0072B2" stroke-width="1.6" rx="4"/>
<rect x="29.3" y="157.5" width="26.4" height="15" fill="#fff" stroke="#0072B2" stroke-width="1" rx="3"/>
<text x="42.5" y="169.5" text-anchor="middle" font-weight="bold" fill="#0072B2">TP8</text>
<rect x="63.4" y="123.4" width="42.2" height="84.2" fill="none" stroke="#0072B2" stroke-width="1.6" rx="4"/>
<rect x="71.3" y="157.5" width="26.4" height="15" fill="#fff" stroke="#0072B2" stroke-width="1" rx="3"/>
<text x="84.5" y="169.5" text-anchor="middle" font-weight="bold" fill="#0072B2">TP8</text>
<text x="167.5" y="14" text-anchor="middle" font-size="13" font-weight="bold" fill="#D55E00">H</text>
<text x="167.5" y="28" text-anchor="middle">2×TP16</text>
<path d="M127.5 41.5h38v38h-38zM169.5 41.5h38v38h-38zM127.5 83.5h38v38h-38zM169.5 83.5h38v38h-38zM127.5 125.5h38v38h-38zM169.5 125.5h38v38h-38zM127.5 167.5h38v38h-38zM169.5 167.5h38v38h-38z" fill="#f6d3bd" stroke="#9a9a9a" stroke-width="0.6"/>
<path d="M130 44h15v15h-15zM148 44h15v15h-15zM172 44h15v15h-15zM190 44h15v15h-15zM130 62h15v15h-15zM148 62h15v15h-15zM172 62h15v15h-15zM190 62h15v15h-15zM130 86h15v15h-15zM148 86h15v15h-15zM172 86h15v15h-15zM190 86h15v15h-15zM130 104h15v15h-15zM148 104h15v15h-15zM172 104h15v15h-15zM190 104h15v15h-15zM130 128h15v15h-15zM148 128h15v15h-15zM172 128h15v15h-15zM190 128h15v15h-15zM130 146h15v15h-15zM148 146h15v15h-15zM172 146h15v15h-15zM190 146h15v15h-15zM130 170h15v15h-15zM148 170h15v15h-15zM172 170h15v15h-15zM190 170h15v15h-15zM130 188h15v15h-15zM148 188h15v15h-15zM172 188h15v15h-15zM190 188h15v15h-15z" fill="#fff" stroke="#777" stroke-width="0.6"/>
<rect x="125.4" y="39.4" width="84.2" height="84.2" fill="none" stroke="#D55E00" stroke-width="1.6" rx="4"/>
<rect x="150.9" y="73.5" width="33.2" height="15" fill="#fff" stroke="#D55E00" stroke-width="1" rx="3"/>
<text x="167.5" y="85.5" text-anchor="middle" font-weight="bold" fill="#D55E00">TP16</text>
<rect x="125.4" y="123.4" width="84.2" height="84.2" fill="none" stroke="#D55E00" stroke-width="1.6" rx="4"/>
<rect x="150.9" y="157.5" width="33.2" height="15" fill="#fff" stroke="#D55E00" stroke-width="1" rx="3"/>
<text x="167.5" y="169.5" text-anchor="middle" font-weight="bold" fill="#D55E00">TP16</text>
<text x="271.5" y="14" text-anchor="middle" font-size="13" font-weight="bold" fill="#009E73">M</text>
<text x="271.5" y="28" text-anchor="middle">TP16 + 2×TP8</text>
<path d="M231.5 41.5h38v38h-38zM273.5 41.5h38v38h-38zM231.5 83.5h38v38h-38zM273.5 83.5h38v38h-38zM231.5 125.5h38v38h-38zM273.5 125.5h38v38h-38zM231.5 167.5h38v38h-38zM273.5 167.5h38v38h-38z" fill="#c2e6d9" stroke="#9a9a9a" stroke-width="0.6"/>
<path d="M234 44h15v15h-15zM252 44h15v15h-15zM276 44h15v15h-15zM294 44h15v15h-15zM234 62h15v15h-15zM252 62h15v15h-15zM276 62h15v15h-15zM294 62h15v15h-15zM234 86h15v15h-15zM252 86h15v15h-15zM276 86h15v15h-15zM294 86h15v15h-15zM234 104h15v15h-15zM252 104h15v15h-15zM276 104h15v15h-15zM294 104h15v15h-15zM234 128h15v15h-15zM252 128h15v15h-15zM276 128h15v15h-15zM294 128h15v15h-15zM234 146h15v15h-15zM252 146h15v15h-15zM276 146h15v15h-15zM294 146h15v15h-15zM234 170h15v15h-15zM252 170h15v15h-15zM276 170h15v15h-15zM294 170h15v15h-15zM234 188h15v15h-15zM252 188h15v15h-15zM276 188h15v15h-15zM294 188h15v15h-15z" fill="#fff" stroke="#777" stroke-width="0.6"/>
<rect x="229.4" y="39.4" width="42.2" height="84.2" fill="none" stroke="#009E73" stroke-width="1.6" rx="4"/>
<rect x="237.3" y="73.5" width="26.4" height="15" fill="#fff" stroke="#009E73" stroke-width="1" rx="3"/>
<text x="250.5" y="85.5" text-anchor="middle" font-weight="bold" fill="#009E73">TP8</text>
<rect x="271.4" y="39.4" width="42.2" height="84.2" fill="none" stroke="#009E73" stroke-width="1.6" rx="4"/>
<rect x="279.3" y="73.5" width="26.4" height="15" fill="#fff" stroke="#009E73" stroke-width="1" rx="3"/>
<text x="292.5" y="85.5" text-anchor="middle" font-weight="bold" fill="#009E73">TP8</text>
<rect x="229.4" y="123.4" width="84.2" height="84.2" fill="none" stroke="#009E73" stroke-width="1.6" rx="4"/>
<rect x="254.9" y="157.5" width="33.2" height="15" fill="#fff" stroke="#009E73" stroke-width="1" rx="3"/>
<text x="271.5" y="169.5" text-anchor="middle" font-weight="bold" fill="#009E73">TP16</text>
<text x="12" y="81.5" text-anchor="middle" fill="#666" transform="rotate(-90 12 81.5)">block A</text>
<text x="12" y="165.5" text-anchor="middle" fill="#666" transform="rotate(-90 12 165.5)">block B</text>
<rect x="26" y="222" width="14" height="10" fill="#c6dcef" stroke="#9a9a9a" stroke-width="0.6"/>
<text x="44" y="231">host (2×2 chips)</text>
<rect x="146" y="222" width="10" height="10" fill="#fff" stroke="#777" stroke-width="0.6"/>
<text x="160" y="231">v5e chip, 16 GB HBM</text>
</svg>
```),
  caption: [The three serving layouts on the $4 times 8$ v5e slice (eight hosts of $2 times 2$ chips; ICI links omitted). Instances follow host boundaries: each `TP8` spans two hosts and each `TP16` four. `M` keeps block A (top four hosts) as $2 times$`TP8` and serves block B as one `TP16`.],
) <fig-layouts>

_Compute favors small meshes._ Raising the TP degree across hosts lowers per-chip utilization through three mechanisms. Each Transformer layer issues two AllReduce collectives (after the attention output projection and after the FFN down projection), or 96 synchronizations per step. A ring of $N$ chips needs $2(N-1)$ routing steps, so moving from `TP=4` (one host, 6 steps) to `TP=8` (two hosts, 14 steps) and `TP=16` (four hosts, 30 steps) introduces an inter-host collective floor of $approx 20" ms"$ at small sequence buckets ($S lt.eq 1024$). TP also shortens the contraction dimension of the reduce-side projections: the MXUs need contraction depths $K gt.double 128$ to amortize systolic pipeline fill and drain, but the per-chip depth falls from $K = 4096 \/ 8 = 512$ (`TP=8`) to $256$ (`TP=16`, only two $128 times 128$ tiles) on the attention output projection and from $2688$ to $1344$ on the FFN down projection. Finally, the FFN intermediate size shards cleanly at `TP=8` ($21504 \/ 8 = 2688 = 21 times 128$) but not at `TP=16` ($1344 = 10.5 times 128$), so XLA zero-pads every FFN weight matrix to 1,408 columns and wastes 4.5% of FFN FLOPs. @tab-topology quantifies the net effect in a per-layer prefill benchmark at the 16,384-token compilation bucket: scaling from one `TP16` instance to four single-host `TP4` instances raises per-chip MFU from 30.67% to 68.16% ($2.22 times$ aggregate throughput), though in full-model end-to-end serving `TP4` (64 GB HBM) cannot hold both the 61.4 GB weights and a practical KV cache, making `TP8` the smallest feasible serving unit.
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

@fig-kv compares these capacities with the contexts of the mid- and long-context workloads.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 196" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<rect x="78" y="30" width="45.5" height="120" fill="#ececec"/>
<rect x="141" y="30" width="128.1" height="120" fill="#ececec"/>
<line x1="87.1" y1="30" x2="87.1" y2="150" stroke="#999" stroke-width="1" stroke-dasharray="2,2"/>
<line x1="159.8" y1="30" x2="159.8" y2="150" stroke="#999" stroke-width="1" stroke-dasharray="2,2"/>
<text x="100.8" y="24" text-anchor="middle" font-weight="bold">W7 contexts</text>
<text x="205" y="24" text-anchor="middle" font-weight="bold">W6 contexts</text>
<rect x="50" y="46" width="90.2" height="20" fill="#0072B2"/>
<text x="44" y="60" text-anchor="end" font-size="11.5" font-weight="bold">TP8</text>
<text x="135.2" y="60" text-anchor="end" font-weight="bold" fill="#fff">52,864</text>
<text x="52" y="81">W7: 2 resident · W6: none fits</text>
<rect x="50" y="100" width="247.8" height="20" fill="#D55E00"/>
<text x="44" y="114" text-anchor="end" font-size="11.5" font-weight="bold">TP16</text>
<text x="292.8" y="114" text-anchor="end" font-weight="bold" fill="#fff">145,216</text>
<text x="52" y="135">W7: 6 resident · W6: 2 resident</text>
<line x1="50" y1="150" x2="306" y2="150" stroke="#222" stroke-width="1"/>
<line x1="50" y1="150" x2="50" y2="153" stroke="#222" stroke-width="1"/>
<text x="50" y="165" text-anchor="middle">0k</text>
<line x1="92.7" y1="150" x2="92.7" y2="153" stroke="#222" stroke-width="1"/>
<text x="92.7" y="165" text-anchor="middle">25k</text>
<line x1="135.3" y1="150" x2="135.3" y2="153" stroke="#222" stroke-width="1"/>
<text x="135.3" y="165" text-anchor="middle">50k</text>
<line x1="178" y1="150" x2="178" y2="153" stroke="#222" stroke-width="1"/>
<text x="178" y="165" text-anchor="middle">75k</text>
<line x1="220.7" y1="150" x2="220.7" y2="153" stroke="#222" stroke-width="1"/>
<text x="220.7" y="165" text-anchor="middle">100k</text>
<line x1="263.3" y1="150" x2="263.3" y2="153" stroke="#222" stroke-width="1"/>
<text x="263.3" y="165" text-anchor="middle">125k</text>
<line x1="306" y1="150" x2="306" y2="153" stroke="#222" stroke-width="1"/>
<text x="306" y="165" text-anchor="middle">150k</text>
<text x="178" y="180" text-anchor="middle" font-size="11.5" font-weight="bold">Usable KV tokens per instance (BF16)</text>
<text x="90.1" y="42" fill="#666">p50</text>
<text x="162.8" y="42" fill="#666">p50</text>
</svg>
```),
  caption: [Usable BF16 KV-cache capacity per instance versus workload contexts (gray: context range; dashed: median). A `TP8` instance holds 52,864 tokens, enough for two resident `W7` requests but for no `W6` request; a `TP16` instance holds 145,216 tokens ($2.75 times$), enough for six `W7` or two `W6` requests.],
) <fig-kv>

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

@fig-makespan plots the makespans of @tab-workloads with their repetitions.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 276" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<line x1="76" y1="30" x2="76" y2="239" stroke="#e2e2e2" stroke-width="1"/>
<line x1="126.5" y1="30" x2="126.5" y2="239" stroke="#e2e2e2" stroke-width="1"/>
<line x1="177" y1="30" x2="177" y2="239" stroke="#e2e2e2" stroke-width="1"/>
<line x1="227.5" y1="30" x2="227.5" y2="239" stroke="#e2e2e2" stroke-width="1"/>
<line x1="278" y1="30" x2="278" y2="239" stroke="#e2e2e2" stroke-width="1"/>
<rect x="16" y="5" width="10" height="10" fill="#0072B2" stroke="#555" stroke-width="0.5"/>
<text x="30" y="14">S (4×TP8)</text>
<rect x="108" y="5" width="10" height="10" fill="#D55E00" stroke="#555" stroke-width="0.5"/>
<text x="122" y="14">H (2×TP16)</text>
<rect x="200" y="5" width="10" height="10" fill="#009E73" stroke="#555" stroke-width="0.5"/>
<text x="214" y="14">M sub-layout</text>
<rect x="76" y="34" width="48.2" height="10" fill="#0072B2"/>
<line x1="123.8" y1="39" x2="124.5" y2="39" stroke="#000" stroke-width="1"/>
<line x1="123.8" y1="36" x2="123.8" y2="42" stroke="#000" stroke-width="1"/>
<line x1="124.5" y1="36" x2="124.5" y2="42" stroke="#000" stroke-width="1"/>
<text x="127.5" y="43" font-weight="bold">190.8</text>
<rect x="76" y="47" width="77.4" height="10" fill="#D55E00"/>
<line x1="153.2" y1="52" x2="153.6" y2="52" stroke="#000" stroke-width="1"/>
<line x1="153.2" y1="49" x2="153.2" y2="55" stroke="#000" stroke-width="1"/>
<line x1="153.6" y1="49" x2="153.6" y2="55" stroke="#000" stroke-width="1"/>
<text x="156.6" y="56">306.4 (16k)</text>
<rect x="76" y="60" width="67.6" height="10" fill="#D55E00"/>
<line x1="142.9" y1="65" x2="144.3" y2="65" stroke="#000" stroke-width="1"/>
<line x1="142.9" y1="62" x2="142.9" y2="68" stroke="#000" stroke-width="1"/>
<line x1="144.3" y1="62" x2="144.3" y2="68" stroke="#000" stroke-width="1"/>
<text x="147.3" y="69">267.8 (131k)</text>
<rect x="76" y="73" width="89.1" height="10" fill="#009E73"/>
<line x1="164.3" y1="78" x2="165.9" y2="78" stroke="#000" stroke-width="1"/>
<line x1="164.3" y1="75" x2="164.3" y2="81" stroke="#000" stroke-width="1"/>
<line x1="165.9" y1="75" x2="165.9" y2="81" stroke="#000" stroke-width="1"/>
<text x="168.9" y="82">352.7 (2×TP8)</text>
<text x="70" y="58.5" text-anchor="end" font-size="11.5" font-weight="bold">W0</text>
<text x="70" y="71.5" text-anchor="end" fill="#666">chat</text>
<rect x="76" y="95" width="37.3" height="10" fill="#0072B2"/>
<line x1="113.1" y1="100" x2="113.4" y2="100" stroke="#000" stroke-width="1"/>
<line x1="113.1" y1="97" x2="113.1" y2="103" stroke="#000" stroke-width="1"/>
<line x1="113.4" y1="97" x2="113.4" y2="103" stroke="#000" stroke-width="1"/>
<text x="116.4" y="104" font-weight="bold">147.7</text>
<rect x="76" y="108" width="47.3" height="10" fill="#D55E00"/>
<line x1="122.7" y1="113" x2="124" y2="113" stroke="#000" stroke-width="1"/>
<line x1="122.7" y1="110" x2="122.7" y2="116" stroke="#000" stroke-width="1"/>
<line x1="124" y1="110" x2="124" y2="116" stroke="#000" stroke-width="1"/>
<text x="127" y="117">187.4</text>
<text x="70" y="106.5" text-anchor="end" font-size="11.5" font-weight="bold">W4</text>
<text x="70" y="119.5" text-anchor="end" fill="#666">long + chat</text>
<rect x="76" y="130" width="190.5" height="10" fill="#0072B2"/>
<line x1="263.8" y1="135" x2="269.1" y2="135" stroke="#000" stroke-width="1"/>
<line x1="263.8" y1="132" x2="263.8" y2="138" stroke="#000" stroke-width="1"/>
<line x1="269.1" y1="132" x2="269.1" y2="138" stroke="#000" stroke-width="1"/>
<text x="272.1" y="139" font-weight="bold">754.3</text>
<rect x="76" y="143" width="195.5" height="10" fill="#D55E00"/>
<line x1="271.1" y1="148" x2="272" y2="148" stroke="#000" stroke-width="1"/>
<line x1="271.1" y1="145" x2="271.1" y2="151" stroke="#000" stroke-width="1"/>
<line x1="272" y1="145" x2="272" y2="151" stroke="#000" stroke-width="1"/>
<text x="275" y="152">774.4</text>
<text x="70" y="141.5" text-anchor="end" font-size="11.5" font-weight="bold">W5</text>
<text x="70" y="154.5" text-anchor="end" fill="#666">translation</text>
<rect x="76" y="165" width="62.7" height="10" fill="#0072B2"/>
<line x1="138.5" y1="170" x2="138.9" y2="170" stroke="#000" stroke-width="1"/>
<line x1="138.5" y1="167" x2="138.5" y2="173" stroke="#000" stroke-width="1"/>
<line x1="138.9" y1="167" x2="138.9" y2="173" stroke="#000" stroke-width="1"/>
<text x="141.9" y="174">248.4</text>
<rect x="76" y="178" width="51.7" height="10" fill="#D55E00"/>
<line x1="127.4" y1="183" x2="128.1" y2="183" stroke="#000" stroke-width="1"/>
<line x1="127.4" y1="180" x2="127.4" y2="186" stroke="#000" stroke-width="1"/>
<line x1="128.1" y1="180" x2="128.1" y2="186" stroke="#000" stroke-width="1"/>
<text x="131.1" y="187" font-weight="bold">204.8</text>
<text x="70" y="176.5" text-anchor="end" font-size="11.5" font-weight="bold">W7</text>
<text x="70" y="189.5" text-anchor="end" fill="#666">mid docs</text>
<text x="80" y="209" font-style="italic" fill="#0072B2">infeasible: context &gt; TP8 KV pool</text>
<rect x="76" y="213" width="52.1" height="10" fill="#D55E00"/>
<line x1="127.7" y1="218" x2="128.5" y2="218" stroke="#000" stroke-width="1"/>
<line x1="127.7" y1="215" x2="127.7" y2="221" stroke="#000" stroke-width="1"/>
<line x1="128.5" y1="215" x2="128.5" y2="221" stroke="#000" stroke-width="1"/>
<text x="131.5" y="222" font-weight="bold">206.2</text>
<rect x="76" y="226" width="103.3" height="10" fill="#009E73"/>
<line x1="178.1" y1="231" x2="180.5" y2="231" stroke="#000" stroke-width="1"/>
<line x1="178.1" y1="228" x2="178.1" y2="234" stroke="#000" stroke-width="1"/>
<line x1="180.5" y1="228" x2="180.5" y2="234" stroke="#000" stroke-width="1"/>
<text x="183.5" y="235">409.2 (TP16)</text>
<text x="70" y="218" text-anchor="end" font-size="11.5" font-weight="bold">W6</text>
<text x="70" y="231" text-anchor="end" fill="#666">long docs</text>
<line x1="76" y1="241" x2="278" y2="241" stroke="#222" stroke-width="1"/>
<line x1="76" y1="30" x2="76" y2="241" stroke="#222" stroke-width="1"/>
<line x1="76" y1="241" x2="76" y2="244" stroke="#222" stroke-width="1"/>
<text x="76" y="255" text-anchor="middle">0</text>
<line x1="126.5" y1="241" x2="126.5" y2="244" stroke="#222" stroke-width="1"/>
<text x="126.5" y="255" text-anchor="middle">200</text>
<line x1="177" y1="241" x2="177" y2="244" stroke="#222" stroke-width="1"/>
<text x="177" y="255" text-anchor="middle">400</text>
<line x1="227.5" y1="241" x2="227.5" y2="244" stroke="#222" stroke-width="1"/>
<text x="227.5" y="255" text-anchor="middle">600</text>
<line x1="278" y1="241" x2="278" y2="244" stroke="#222" stroke-width="1"/>
<text x="278" y="255" text-anchor="middle">800</text>
<text x="177" y="269" text-anchor="middle" font-size="11.5" font-weight="bold">Makespan of one manifest (s)</text>
</svg>
```),
  caption: [Makespan of one manifest per layout (bars: mean of two repetitions; whiskers: both repetitions; bold: faster full-slice layout). `S` cannot run `W6`. The two `H` bars for `W0` use 16k and 131k context limits; `M` bars show the sub-layout that serves the workload ($2 times$`TP8` for `W0`, `TP16` for `W6`).],
) <fig-makespan>

== Limitations of Static Heterogeneity and Live Request Migration <sec-static>

Before changing layouts at runtime, we ask whether a fixed heterogeneous layout, `M` = `TP16` + $2 times$`TP8`, or live cross-TP request migration can bridge the short/long divide.

_Short-request pollution._ Consider the mixed workload `W4` (880 short chats and 24 long GovReport documents of 27.1k–36.5k tokens) in a capacity-constrained FP32-KV regime in which `TP8` instances hold 26,368 KV tokens, too few for the long documents, and `TP16` instances hold 72,192 (as opposed to @tab-workloads's full-BF16 regime where `TP8` holds 52,864 tokens and fits `W4` directly). Under a plain load-balancing router, `M` finishes in 452.2 s, 30.3% slower than homogeneous `H` (347.1 s). Early short chats spill onto the `TP16` instance and occupy its KV blocks, starving the 24 long documents, while the two `TP8` engines drain their short queues by $t = 210" s"$ and then sit idle for the remaining 242 s.
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

_Environment initialization._ This model-agnostic phase spawns Ray worker processes and imports the Python, vLLM, and JAX runtime stack, requiring 32.7 s per transition (plus $approx 120" s"$ of orchestration overhead when transitions are invoked via external job submissions). While GPU cold-start systems hide this latency using pre-warmed worker pools that import libraries prior to GPU binding @liu2026foundry @zeng2025medusa, applying this technique on Cloud TPU is complicated by PJRT's single-owner device lock, which is acquired whenever the serving stack queries device topology during initialization.

_Worker initialization._ This phase constructs model- and layout-specific state: device initialization, weight loading, KV-cache allocation, and XLA compilation. Weight loading accounts for 237.0 s, or 67% of the 355 s engine startup time. Per-tensor profiling across all eight workers reveals that file iteration and storage reads consume 0.16–0.18 s and host-to-device DMA consumes 2.0–3.0 s, whereas host-CPU tensor transposes, reshapes, and data-type conversions require 127.6–128.4 s, load-time JIT compilation of parameter shardings requires 74.9–76.8 s, and Flax `nnx.eval_shape` graph construction requires 14.6 s. Even when weight loading is bypassed, KV-cache initialization requires 25.9 s and compilation and warmup require 54.6 s under a warm persistent XLA cache, which achieves hit rates of only 2.6% (48 of 1,848 executables) and 6.0% (264 of 4,416).
@fig-load breaks down one worker's `load_model` call.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 206" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<line x1="8" y1="10" x2="8" y2="140" stroke="#e2e2e2" stroke-width="1"/>
<line x1="60" y1="10" x2="60" y2="140" stroke="#e2e2e2" stroke-width="1"/>
<line x1="112" y1="10" x2="112" y2="140" stroke="#e2e2e2" stroke-width="1"/>
<line x1="164" y1="10" x2="164" y2="140" stroke="#e2e2e2" stroke-width="1"/>
<line x1="216" y1="10" x2="216" y2="140" stroke="#e2e2e2" stroke-width="1"/>
<line x1="268" y1="10" x2="268" y2="140" stroke="#e2e2e2" stroke-width="1"/>
<text x="8" y="18" font-size="11.5" font-weight="bold">Checkpoint loader, TP8 (S)</text>
<rect x="8" y="22" width="133.1" height="18" fill="#CC6677" stroke="#fff" stroke-width="0.6"/>
<text x="74.5" y="35" text-anchor="middle" fill="#fff">128.0</text>
<rect x="141.1" y="22" width="12" height="18" fill="#dc96a2" stroke="#fff" stroke-width="0.6"/>
<rect x="153.1" y="22" width="18.1" height="18" fill="#efc6cd" stroke="#fff" stroke-width="0.6"/>
<rect x="171.2" y="22" width="77.9" height="18" fill="#AA4499" stroke="#fff" stroke-width="0.6"/>
<text x="210.2" y="35" text-anchor="middle" fill="#fff">74.9</text>
<rect x="249.1" y="22" width="3.1" height="18" fill="#444444" stroke="#fff" stroke-width="0.6"/>
<rect x="252.2" y="22" width="4.5" height="18" fill="#b5b5b5" stroke="#fff" stroke-width="0.6"/>
<text x="260.7" y="35" font-weight="bold">239.1 s</text>
<text x="8" y="60" font-size="11.5" font-weight="bold">Checkpoint loader, TP16 (H)</text>
<rect x="8" y="64" width="132.7" height="18" fill="#CC6677" stroke="#fff" stroke-width="0.6"/>
<text x="74.3" y="77" text-anchor="middle" fill="#fff">127.6</text>
<rect x="140.7" y="64" width="12" height="18" fill="#dc96a2" stroke="#fff" stroke-width="0.6"/>
<rect x="152.6" y="64" width="18.2" height="18" fill="#efc6cd" stroke="#fff" stroke-width="0.6"/>
<rect x="170.8" y="64" width="79.9" height="18" fill="#AA4499" stroke="#fff" stroke-width="0.6"/>
<text x="210.7" y="77" text-anchor="middle" fill="#fff">76.8</text>
<rect x="250.7" y="64" width="2.1" height="18" fill="#444444" stroke="#fff" stroke-width="0.6"/>
<rect x="252.7" y="64" width="4.6" height="18" fill="#b5b5b5" stroke="#fff" stroke-width="0.6"/>
<text x="261.3" y="77" font-weight="bold">240.0 s</text>
<text x="8" y="102" font-size="11.5" font-weight="bold">PhaseMorph restore (snapshot + FastGraph)</text>
<rect x="8" y="106" width="5.2" height="18" fill="#332288" stroke="#fff" stroke-width="0.6"/>
<text x="17.2" y="119" font-weight="bold">5.0 s</text>
<line x1="8" y1="140" x2="268" y2="140" stroke="#222" stroke-width="1"/>
<line x1="8" y1="140" x2="8" y2="143" stroke="#222" stroke-width="1"/>
<text x="8" y="155" text-anchor="middle">0</text>
<line x1="60" y1="140" x2="60" y2="143" stroke="#222" stroke-width="1"/>
<text x="60" y="155" text-anchor="middle">50</text>
<line x1="112" y1="140" x2="112" y2="143" stroke="#222" stroke-width="1"/>
<text x="112" y="155" text-anchor="middle">100</text>
<line x1="164" y1="140" x2="164" y2="143" stroke="#222" stroke-width="1"/>
<text x="164" y="155" text-anchor="middle">150</text>
<line x1="216" y1="140" x2="216" y2="143" stroke="#222" stroke-width="1"/>
<text x="216" y="155" text-anchor="middle">200</text>
<line x1="268" y1="140" x2="268" y2="143" stroke="#222" stroke-width="1"/>
<text x="268" y="155" text-anchor="middle">250</text>
<text x="272" y="155" fill="#666" dx="6">s</text>
<rect x="8" y="165" width="10" height="10" fill="#CC6677" stroke="#555" stroke-width="0.5"/>
<text x="22" y="174">host tensor convert</text>
<rect x="164" y="165" width="10" height="10" fill="#dc96a2" stroke="#555" stroke-width="0.5"/>
<text x="178" y="174">other host ops</text>
<rect x="8" y="180" width="10" height="10" fill="#efc6cd" stroke="#555" stroke-width="0.5"/>
<text x="22" y="189">graph build + unattributed</text>
<rect x="164" y="180" width="10" height="10" fill="#AA4499" stroke="#555" stroke-width="0.5"/>
<text x="178" y="189">load-time JIT / cache</text>
<rect x="8" y="195" width="10" height="10" fill="#444444" stroke="#555" stroke-width="0.5"/>
<text x="22" y="204">host → device</text>
<rect x="164" y="195" width="10" height="10" fill="#b5b5b5" stroke="#555" stroke-width="0.5"/>
<text x="178" y="204">post-load + file reads</text>
</svg>
```),
  caption: [Per-worker `load_model` time of the stock checkpoint loader (medians of eight profiled workers per layout) versus PhaseMorph's snapshot restore with shape-only graph construction (PhaseMorph-Snap in @tab-segments). Host tensor conversion and load-time sharding JIT dominate; file reads (0.18 s) and host-to-device transfer (2.0–3.0 s) are negligible. Profiled loads take 239–240 s; unprofiled loads take 234–239 s.],
) <fig-load>

_Pitfall 1: Host-RAM staging and peer transfer._ Prior GPU systems accelerate checkpoint loading via local NVMe/DRAM tiering or peer-to-peer RDMA multicast @fu2024serverlessllm @zhang2025blitzscale. On Cloud TPU, staging the 61.4 GB checkpoint in `/dev/shm` tmpfs yields no measurable speedup (`load_model` remains 237.6 s), because latency is dominated by host-CPU layout transformation and sharding compilation rather than storage bandwidth.

_Pitfall 2: Concurrent background compilation on an active host._ Persistent compilation caching does not eliminate JAX's Python tracing and StableHLO lowering (28.6 s on `H` and 45.5 s on `S`), which execute on the main thread prior to cache-key generation because JAX's trace cache is process-local. Compiling the target layout in a separate background process with `JAX_PLATFORMS=cpu` and `get_topology_desc(platform="tpu")` immediately fails with `ABORTED: The TPU is already in use` on `/tmp/libtpu_lockfile`, because querying TPU topology descriptors unconditionally initializes the PJRT TPU plugin.

_Pitfall 3: In-place submesh slicing within a single runtime._ Retaining weights in HBM and re-slicing them inside a single multi-host JAX runtime—analogous to submesh execution in Pathways @barham2022pathways and Alpa @zheng2022alpa—avoids process restarts altogether. However, `libtpu` fails to compile collectives on non-contiguous or chip-0-anchored sub-tori (@sec-design-submesh), vLLM's process-global singletons require intrusive refactoring, and coupling all hosts into a single runtime expands the failure domain to the entire slice.

PhaseMorph addresses these limitations by materializing post-conversion weight shards and compiled executables across process boundaries, and by overlapping worker initialization with foreground serving without acquiring TPU resources prematurely.

= PhaseMorph Overview <sec-overview>

PhaseMorph operates in two stages that mirror the offline/online split of recent cold-start systems @liu2026foundry @zeng2025medusa: a PREPARE stage that executes offline or concurrently in the background while the source layout serves, and an online HANDOFF/RESTORE stage on the critical path after $T_0$ (@fig-architecture).

#wide-fig(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 236" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<defs><pattern id="ar" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="4" height="4" fill="#fff"/><line x1="0" y1="0" x2="0" y2="4" stroke="#444" stroke-width="1.4"/></pattern></defs>
<text x="267" y="12" text-anchor="middle" font-size="11.5" font-weight="bold">PREPARE (during the source phase)</text>
<text x="514" y="12" text-anchor="middle" font-size="11.5" font-weight="bold">HANDOFF / RESTORE (after T<tspan dy="2.6" font-size="9.2">0</tspan><tspan dy="-2.6">)</tspan></text>
<line x1="150" y1="34" x2="384" y2="34" stroke="#666" stroke-width="1"/>
<line x1="400" y1="34" x2="628" y2="34" stroke="#666" stroke-width="1"/>
<line x1="150" y1="31" x2="150" y2="34" stroke="#666" stroke-width="1"/>
<text x="150" y="28" text-anchor="middle" fill="#666">-200</text>
<line x1="208.5" y1="31" x2="208.5" y2="34" stroke="#666" stroke-width="1"/>
<text x="208.5" y="28" text-anchor="middle" fill="#666">-150</text>
<line x1="267" y1="31" x2="267" y2="34" stroke="#666" stroke-width="1"/>
<text x="267" y="28" text-anchor="middle" fill="#666">-100</text>
<line x1="325.5" y1="31" x2="325.5" y2="34" stroke="#666" stroke-width="1"/>
<text x="325.5" y="28" text-anchor="middle" fill="#666">-50</text>
<line x1="384" y1="31" x2="384" y2="34" stroke="#666" stroke-width="1"/>
<text x="384" y="28" text-anchor="middle" fill="#666">0</text>
<line x1="400" y1="31" x2="400" y2="34" stroke="#666" stroke-width="1"/>
<text x="400" y="28" text-anchor="middle" fill="#666">0</text>
<line x1="438" y1="31" x2="438" y2="34" stroke="#666" stroke-width="1"/>
<text x="438" y="28" text-anchor="middle" fill="#666">5</text>
<line x1="476" y1="31" x2="476" y2="34" stroke="#666" stroke-width="1"/>
<text x="476" y="28" text-anchor="middle" fill="#666">10</text>
<line x1="514" y1="31" x2="514" y2="34" stroke="#666" stroke-width="1"/>
<text x="514" y="28" text-anchor="middle" fill="#666">15</text>
<line x1="552" y1="31" x2="552" y2="34" stroke="#666" stroke-width="1"/>
<text x="552" y="28" text-anchor="middle" fill="#666">20</text>
<line x1="590" y1="31" x2="590" y2="34" stroke="#666" stroke-width="1"/>
<text x="590" y="28" text-anchor="middle" fill="#666">25</text>
<line x1="628" y1="31" x2="628" y2="34" stroke="#666" stroke-width="1"/>
<text x="628" y="28" text-anchor="middle" fill="#666">30</text>
<text x="628" y="46" text-anchor="end" fill="#666">s after T<tspan dy="2.6" font-size="8.8">0</tspan><tspan dy="-2.6"> </tspan></text>
<text x="150" y="46" fill="#666">s before T<tspan dy="2.6" font-size="8.8">0</tspan><tspan dy="-2.6"> </tspan></text>
<line x1="387" y1="38" x2="393" y2="30" stroke="#666" stroke-width="1"/>
<line x1="391" y1="38" x2="397" y2="30" stroke="#666" stroke-width="1"/>
<text x="142" y="65" text-anchor="end" font-size="11.5" font-weight="bold">Source S (4×TP8)</text>
<text x="142" y="78" text-anchor="end" fill="#666">owns the TPUs</text>
<text x="142" y="111" text-anchor="end" font-size="11.5" font-weight="bold">Target H workers</text>
<text x="142" y="124" text-anchor="end" fill="#666">CPU-only Ray bundles</text>
<text x="142" y="157" text-anchor="end" font-size="11.5" font-weight="bold">Controller + gateway</text>
<text x="142" y="170" text-anchor="end" fill="#666">in-process</text>
<line x1="384" y1="36" x2="384" y2="180" stroke="#000" stroke-width="1" stroke-dasharray="3,2"/>
<line x1="400" y1="36" x2="400" y2="180" stroke="#000" stroke-width="1" stroke-dasharray="3,2"/>
<rect x="159.1" y="52" width="224.9" height="30" fill="#0072B2"/>
<text x="271.6" y="65" text-anchor="middle" fill="#fff">S serves W0 (phase 1, 190.7–194.6 s)</text>
<text x="271.6" y="78" text-anchor="middle" fill="#fff">no measurable slowdown</text>
<rect x="400" y="52" width="15" height="30" fill="#8c8c8c" stroke="#fff" stroke-width="0.6"/>
<text x="419" y="71">drain + release 2.0 s</text>
<rect x="159.1" y="98" width="40.7" height="30" fill="#88CCEE" stroke="#fff" stroke-width="0.6"/>
<text x="179.5" y="111" text-anchor="middle">spawn</text>
<text x="179.5" y="124" text-anchor="middle">≈34 s</text>
<rect x="199.8" y="99" width="184.2" height="28" fill="#fafafa" stroke="#888" stroke-width="0.8" stroke-dasharray="3,2"/>
<text x="291.9" y="111" text-anchor="middle">parked on release gate</text>
<text x="291.9" y="124" text-anchor="middle" fill="#666">no libtpu, no /dev/vfio handle</text>
<rect x="400" y="99" width="15" height="28" fill="#fafafa" stroke="#888" stroke-width="0.8" stroke-dasharray="3,2"/>
<rect x="415" y="98" width="6.5" height="30" fill="#c8c8c8" stroke="#fff" stroke-width="0.6"/>
<rect x="421.6" y="98" width="29.8" height="30" fill="#DDCC77" stroke="#fff" stroke-width="0.6"/>
<text x="436.5" y="111" text-anchor="middle">dev</text>
<text x="436.5" y="124" text-anchor="middle">3.9</text>
<rect x="451.4" y="98" width="37.3" height="30" fill="#CC6677" stroke="#fff" stroke-width="0.6"/>
<text x="470" y="111" text-anchor="middle" fill="#fff">load</text>
<text x="470" y="124" text-anchor="middle" fill="#fff">4.9</text>
<rect x="488.7" y="98" width="19.5" height="30" fill="#999933" stroke="#fff" stroke-width="0.6"/>
<text x="498.5" y="111" text-anchor="middle" fill="#fff">KV</text>
<text x="498.5" y="124" text-anchor="middle" fill="#fff">2.6</text>
<rect x="508.2" y="98" width="37.8" height="30" fill="#AA4499" stroke="#fff" stroke-width="0.6"/>
<text x="527.1" y="111" text-anchor="middle" fill="#fff">bank</text>
<text x="527.1" y="124" text-anchor="middle" fill="#fff">5.0</text>
<text x="550.1" y="111">0 traces,</text>
<text x="550.1" y="124">0 compiles</text>
<rect x="415" y="144" width="2" height="30" fill="#8c8c8c"/>
<text x="421" y="163">release gate</text>
<rect x="546.1" y="144" width="36.2" height="30" fill="#5f5f5f" stroke="#fff" stroke-width="0.6"/>
<text x="564.2" y="163" text-anchor="middle" fill="#fff">4.8</text>
<rect x="582.2" y="144" width="26.8" height="30" fill="url(#ar)" stroke="#444" stroke-width="0.5"/>
<line x1="582.2" y1="140" x2="582.2" y2="178" stroke="#000" stroke-width="1.2"/>
<line x1="609.1" y1="140" x2="609.1" y2="178" stroke="#000" stroke-width="1.6"/>
<text x="628" y="189" text-anchor="end" font-weight="bold">first W6 dispatch 24.0 s;  C = 27.5 s</text>
<rect x="8" y="205" width="10" height="10" fill="#0072B2" stroke="#555" stroke-width="0.5"/>
<text x="22" y="214">S serving W0</text>
<rect x="134" y="205" width="10" height="10" fill="#88CCEE" stroke="#555" stroke-width="0.5"/>
<text x="148" y="214">spawn + imports</text>
<rect x="260" y="205" width="10" height="10" fill="#8c8c8c" stroke="#555" stroke-width="0.5"/>
<text x="274" y="214">drain / control</text>
<rect x="386" y="205" width="10" height="10" fill="#c8c8c8" stroke="#555" stroke-width="0.5"/>
<text x="400" y="214">VFIO handoff</text>
<rect x="512" y="205" width="10" height="10" fill="#DDCC77" stroke="#555" stroke-width="0.5"/>
<text x="526" y="214">init_device</text>
<rect x="8" y="220" width="10" height="10" fill="#CC6677" stroke="#555" stroke-width="0.5"/>
<text x="22" y="229">restore weights</text>
<rect x="134" y="220" width="10" height="10" fill="#999933" stroke="#555" stroke-width="0.5"/>
<text x="148" y="229">KV cache init</text>
<rect x="260" y="220" width="10" height="10" fill="#AA4499" stroke="#555" stroke-width="0.5"/>
<text x="274" y="229">bank load + warm-up</text>
<rect x="386" y="220" width="10" height="10" fill="#5f5f5f" stroke="#555" stroke-width="0.5"/>
<text x="400" y="229">endpoint + gateway</text>
<rect x="512" y="220" width="10" height="10" fill="url(#ar)" stroke="#555" stroke-width="0.5"/>
<text x="526" y="229">ramp</text>
</svg>
```),
  caption: [PhaseMorph workflow for a warm full `S → H` switch, to scale (left: seconds before $T_0$, means of three runs; right: seconds after $T_0$, Full A column of @tab-segments). During PREPARE, target `H` workers spawn in CPU-only bundles, verify that they hold no `libtpu` or `/dev/vfio` handle, and park on a release gate. After $T_0$ (HANDOFF/RESTORE), the source drains and releases its chips, the workers await completion of the kernel's VFIO reset, restore pre-sharded weights from `/dev/shm` (4.9 s), and load 187 serialized executables from the executable bank (3.45 s of the 5.0 s warm-up). The remaining 4.8 s before the first `W6` dispatch is endpoint registration and gateway roll-out. Cold first switches add a CPU pre-read child (@fig-coldswitch).],
) <fig-architecture>

_PREPARE stage._ During the initial deployment of a layout, each worker persists two artifacts inline: a shared-memory snapshot of post-conversion per-chip weight shards (7.53 s) and a bank of serialized AOT-compiled PJRT executables (2.26 s). When a transition is scheduled, PhaseMorph spawns target workers in CPU-only placement bundles while the source layout continues serving; these workers import the software stack, initialize tokenizers, verify that no TPU handles are held, and block on a release gate. For cold transitions into an unprovisioned layout, each parked worker also spawns a low-priority CPU child process that materializes the target weight snapshot in the background.

_HANDOFF/RESTORE stage._ At $T_0$, the in-process switch controller drains and terminates the source engines ($approx 2" s"$), then opens the release gate. The unblocked target workers wait for the Linux kernel to complete `/dev/vfio` device reset, initialize the PJRT TPU backend, map pre-sharded weights from `/dev/shm` into HBM, allocate the KV cache, deserialize all executables from the bank, and register with the routing gateway—bypassing Python tracing and XLA backend compilation entirely.

_Phase-aware switching planner._ PhaseMorph targets two-phase bulk jobs consisting of $k_s$ short-context manifests (`W0`) followed by $k_l$ long-context manifests (`W6`). A calibrated cost model selects among four execution strategies using the per-manifest makespans of @tab-workloads, the `S` tail-drain overhead $O = 32.6" s"$, and the measured transition costs $C_"full"$ and $C_"part"$:

$ T_"H" &= 267.8 k_s + 206.2 k_l, \
  T_"M" &= max(352.7 k_s, 409.2 k_l), \
  T_"S→H" &= 190.8 k_s + O + C_"full" + 206.2 k_l, \
  T_"S→M" &= min_(f in [0,1]) [190.8 f k_s + max(352.7(1-f) k_s, thin B)], \
  "where" B &= C_"part" + 409.2 k_l. $

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

== TPU-Decoupled Worker Pre-Spawning and VFIO Device Handoff <sec-design-prespawn>
=== Pre-Spawning Without PJRT Lockfile Contention

To overlap the 34.3 s of process spawning, imports, and tokenizer construction with source-phase serving, PhaseMorph creates the target workers in CPU-only Ray bundles (`{"CPU": 1.0, "node:<ip>": 0.001}`) while the source engines still own the TPUs. The difficulty is that a worker must not touch the TPU runtime before the source has released it. Each pre-spawned worker therefore audits itself: it inspects `/proc/self/maps` and `/proc/self/fd` to assert that no `libtpu.so` is mapped and no `/dev/vfio/*` or lockfile descriptor is open, and then parks on an asynchronous release gate inside `TPUWorker.init_device()`. Only when the controller opens the gate does the worker initialize its TPU backend.

=== Asynchronous VFIO Kernel-Reset Barrier

Releasing the gate as soon as the source processes exit is not sufficient. After the source workers exit, the Linux kernel takes 0.42–0.65 s to reset the `/dev/vfio/0..3` devices, and a worker that calls `jax.devices()` within this window fails with `EBUSY`. Released workers therefore poll `os.open("/dev/vfio/<group>", os.O_RDWR)` every 20 ms and initialize the TPU only after every device group opens successfully. The gate and VFIO wait together cost 0.84–0.86 s on the critical path, down from 11.1–12.2 s when workers are spawned after $T_0$.

== In-Runtime Submesh Reconfiguration versus Process Isolation <sec-design-submesh>

To establish the hardware lower bound of a layout change—weights stay in HBM and no process restarts—we built _PhaseMorph-Submesh_, a unified runtime that initializes one JAX domain across four hosts (a $4 times 4$ torus) and switches between two 8-chip `TP8` submeshes and one 16-chip `TP16` mesh.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 196" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<defs><pattern id="sx" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="#fbe3e1"/><line x1="0" y1="0" x2="0" y2="5" stroke="#b3261e" stroke-width="1.2"/></pattern></defs>
<rect x="14" y="38" width="25" height="25" fill="url(#sx)" stroke="#777" stroke-width="0.6"/>
<text x="26.5" y="54.5" text-anchor="middle">0</text>
<rect x="42" y="38" width="25" height="25" fill="url(#sx)" stroke="#777" stroke-width="0.6"/>
<text x="54.5" y="54.5" text-anchor="middle">1</text>
<rect x="70" y="38" width="25" height="25" fill="#eeeeee" stroke="#777" stroke-width="0.6"/>
<text x="82.5" y="54.5" text-anchor="middle">2</text>
<rect x="98" y="38" width="25" height="25" fill="#eeeeee" stroke="#777" stroke-width="0.6"/>
<text x="110.5" y="54.5" text-anchor="middle">3</text>
<rect x="14" y="66" width="25" height="25" fill="url(#sx)" stroke="#777" stroke-width="0.6"/>
<text x="26.5" y="82.5" text-anchor="middle">4</text>
<rect x="42" y="66" width="25" height="25" fill="url(#sx)" stroke="#777" stroke-width="0.6"/>
<text x="54.5" y="82.5" text-anchor="middle">5</text>
<rect x="70" y="66" width="25" height="25" fill="#eeeeee" stroke="#777" stroke-width="0.6"/>
<text x="82.5" y="82.5" text-anchor="middle">6</text>
<rect x="98" y="66" width="25" height="25" fill="#eeeeee" stroke="#777" stroke-width="0.6"/>
<text x="110.5" y="82.5" text-anchor="middle">7</text>
<rect x="14" y="94" width="25" height="25" fill="url(#sx)" stroke="#777" stroke-width="0.6"/>
<text x="26.5" y="110.5" text-anchor="middle">8</text>
<rect x="42" y="94" width="25" height="25" fill="url(#sx)" stroke="#777" stroke-width="0.6"/>
<text x="54.5" y="110.5" text-anchor="middle">9</text>
<rect x="70" y="94" width="25" height="25" fill="#eeeeee" stroke="#777" stroke-width="0.6"/>
<text x="82.5" y="110.5" text-anchor="middle">10</text>
<rect x="98" y="94" width="25" height="25" fill="#eeeeee" stroke="#777" stroke-width="0.6"/>
<text x="110.5" y="110.5" text-anchor="middle">11</text>
<rect x="14" y="122" width="25" height="25" fill="url(#sx)" stroke="#777" stroke-width="0.6"/>
<text x="26.5" y="138.5" text-anchor="middle">12</text>
<rect x="42" y="122" width="25" height="25" fill="url(#sx)" stroke="#777" stroke-width="0.6"/>
<text x="54.5" y="138.5" text-anchor="middle">13</text>
<rect x="70" y="122" width="25" height="25" fill="#eeeeee" stroke="#777" stroke-width="0.6"/>
<text x="82.5" y="138.5" text-anchor="middle">14</text>
<rect x="98" y="122" width="25" height="25" fill="#eeeeee" stroke="#777" stroke-width="0.6"/>
<text x="110.5" y="138.5" text-anchor="middle">15</text>
<rect x="12.5" y="36.5" width="56" height="56" fill="none" stroke="#222" stroke-width="0.9" rx="3" stroke-dasharray="3,2"/>
<rect x="68.5" y="36.5" width="56" height="56" fill="none" stroke="#222" stroke-width="0.9" rx="3" stroke-dasharray="3,2"/>
<rect x="12.5" y="92.5" width="56" height="56" fill="none" stroke="#222" stroke-width="0.9" rx="3" stroke-dasharray="3,2"/>
<rect x="68.5" y="92.5" width="56" height="56" fill="none" stroke="#222" stroke-width="0.9" rx="3" stroke-dasharray="3,2"/>
<line x1="68.5" y1="34" x2="68.5" y2="151" stroke="#000" stroke-width="2"/>
<text x="62.5" y="14" text-anchor="middle" font-size="12" font-weight="bold">Column split</text>
<path d="M109.5,5 L119.5,15 M109.5,15 L119.5,5" fill="none" stroke="#b3261e" stroke-width="2" stroke-linecap="round"/>
<text x="68.5" y="28" text-anchor="middle" fill="#666">IDs 0,1,4,5,8,9,12,13</text>
<text x="68.5" y="163" text-anchor="middle">libtpu RET_CHECK</text>
<text x="68.5" y="177" text-anchor="middle">net_util.cc:2304</text>
<rect x="194" y="38" width="25" height="25" fill="#d4d4d4" stroke="#777" stroke-width="0.6"/>
<text x="206.5" y="54.5" text-anchor="middle">0</text>
<rect x="222" y="38" width="25" height="25" fill="#d4d4d4" stroke="#777" stroke-width="0.6"/>
<text x="234.5" y="54.5" text-anchor="middle">1</text>
<rect x="250" y="38" width="25" height="25" fill="#d4d4d4" stroke="#777" stroke-width="0.6"/>
<text x="262.5" y="54.5" text-anchor="middle">2</text>
<rect x="278" y="38" width="25" height="25" fill="#d4d4d4" stroke="#777" stroke-width="0.6"/>
<text x="290.5" y="54.5" text-anchor="middle">3</text>
<rect x="194" y="66" width="25" height="25" fill="#d4d4d4" stroke="#777" stroke-width="0.6"/>
<text x="206.5" y="82.5" text-anchor="middle">4</text>
<rect x="222" y="66" width="25" height="25" fill="#d4d4d4" stroke="#777" stroke-width="0.6"/>
<text x="234.5" y="82.5" text-anchor="middle">5</text>
<rect x="250" y="66" width="25" height="25" fill="#d4d4d4" stroke="#777" stroke-width="0.6"/>
<text x="262.5" y="82.5" text-anchor="middle">6</text>
<rect x="278" y="66" width="25" height="25" fill="#d4d4d4" stroke="#777" stroke-width="0.6"/>
<text x="290.5" y="82.5" text-anchor="middle">7</text>
<rect x="194" y="94" width="25" height="25" fill="#f4f4f4" stroke="#777" stroke-width="0.6"/>
<text x="206.5" y="110.5" text-anchor="middle">8</text>
<rect x="222" y="94" width="25" height="25" fill="#f4f4f4" stroke="#777" stroke-width="0.6"/>
<text x="234.5" y="110.5" text-anchor="middle">9</text>
<rect x="250" y="94" width="25" height="25" fill="#f4f4f4" stroke="#777" stroke-width="0.6"/>
<text x="262.5" y="110.5" text-anchor="middle">10</text>
<rect x="278" y="94" width="25" height="25" fill="#f4f4f4" stroke="#777" stroke-width="0.6"/>
<text x="290.5" y="110.5" text-anchor="middle">11</text>
<rect x="194" y="122" width="25" height="25" fill="#f4f4f4" stroke="#777" stroke-width="0.6"/>
<text x="206.5" y="138.5" text-anchor="middle">12</text>
<rect x="222" y="122" width="25" height="25" fill="#f4f4f4" stroke="#777" stroke-width="0.6"/>
<text x="234.5" y="138.5" text-anchor="middle">13</text>
<rect x="250" y="122" width="25" height="25" fill="#f4f4f4" stroke="#777" stroke-width="0.6"/>
<text x="262.5" y="138.5" text-anchor="middle">14</text>
<rect x="278" y="122" width="25" height="25" fill="#f4f4f4" stroke="#777" stroke-width="0.6"/>
<text x="290.5" y="138.5" text-anchor="middle">15</text>
<rect x="192.5" y="36.5" width="56" height="56" fill="none" stroke="#222" stroke-width="0.9" rx="3" stroke-dasharray="3,2"/>
<rect x="248.5" y="36.5" width="56" height="56" fill="none" stroke="#222" stroke-width="0.9" rx="3" stroke-dasharray="3,2"/>
<rect x="192.5" y="92.5" width="56" height="56" fill="none" stroke="#222" stroke-width="0.9" rx="3" stroke-dasharray="3,2"/>
<rect x="248.5" y="92.5" width="56" height="56" fill="none" stroke="#222" stroke-width="0.9" rx="3" stroke-dasharray="3,2"/>
<line x1="190" y1="92.5" x2="307" y2="92.5" stroke="#000" stroke-width="2"/>
<text x="242.5" y="14" text-anchor="middle" font-size="12" font-weight="bold">Host-row split</text>
<path d="M296.5,10 L299.8,13.2 L306.5,6.5" fill="none" stroke="#1b7a3a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<text x="248.5" y="28" text-anchor="middle" fill="#666">IDs 0–7 | 8–15</text>
<text x="248.5" y="163" text-anchor="middle">2×TP8 → TP16: 12.0 ms</text>
<text x="248.5" y="177" text-anchor="middle">TP16 → 2×TP8: 48.0 ms</text>
</svg>
```),
  caption: [Submesh partitioning of a 16-chip ($4 times 4$) runtime under `libtpu 0.0.43` (device ID $= x + 4y$; dashed boxes are hosts of $2 times 2$ chips). Splitting by columns aborts in `net_util.cc:2304` because the origin half has non-dense device IDs; splitting by host rows (IDs `0..7` and `8..15`) compiles and switches $2 times$`TP8` → `TP16` in 12.0 ms (48.0 ms back) on a synthetic model with 1.61 GB/chip of weights.],
) <fig-submesh>

_Bypassing the sub-box routing bug._ Our first partition split the $4 times 4$ grid along columns, giving the origin half device IDs `[0,1,4,5,8,9,12,13]`. Compiling any collective for this submesh fails inside `libtpu` (`net_util.cc:2304`, `RET_CHECK device_id < kMaxDeviceCount (8 vs. 8)`), because the library assumes that a sub-box anchored at the origin has dense device IDs `0..N-1`. Rotating the split to host rows (IDs `0..7` and `8..15`) satisfies this invariant, both for live `jax.jit` compilation and for `deserialize_and_load` of banked executables (@fig-submesh).

_Isolation and switching._ With host-row halves, the two `TP8` submeshes step concurrently at $1.05"–"1.08 times$ their solo step time, and stalling one half in a 30 s barrier leaves the other at 97%–99% of its throughput. Switching `2×TP8 → TP16` with nested local device order takes 12.04 ms (0.08 ms to stop, 6.14 ms to re-slice the weights on chip, and 5.82 ms for the first step), with no compilation; the reverse switch over ICI takes 48.02 ms.

On the full $4 times 8$ slice, in-runtime submeshes fail outright: no layout has all of its instances compile in one 32-chip runtime (@fig-fullmesh).

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 182" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<defs><pattern id="fx" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="4" height="4" fill="#fbe3e1"/><line x1="0" y1="0" x2="0" y2="4" stroke="#b3261e" stroke-width="1.2"/></pattern></defs>
<text x="33.2" y="13" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#0072B2">S</text>
<text x="33.2" y="27" text-anchor="middle">4×TP8</text>
<rect x="8.5" y="38.5" width="25.5" height="49.5" fill="url(#fx)" rx="2"/>
<rect x="32.5" y="38.5" width="25.5" height="49.5" fill="#c6dcef" rx="2"/>
<rect x="8.5" y="86.5" width="25.5" height="49.5" fill="#c6dcef" rx="2"/>
<rect x="32.5" y="86.5" width="25.5" height="49.5" fill="#c6dcef" rx="2"/>
<path d="M10 40h10.5v10.5h-10.5zM22 40h10.5v10.5h-10.5zM34 40h10.5v10.5h-10.5zM46 40h10.5v10.5h-10.5zM10 52h10.5v10.5h-10.5zM22 52h10.5v10.5h-10.5zM34 52h10.5v10.5h-10.5zM46 52h10.5v10.5h-10.5zM10 64h10.5v10.5h-10.5zM22 64h10.5v10.5h-10.5zM34 64h10.5v10.5h-10.5zM46 64h10.5v10.5h-10.5zM10 76h10.5v10.5h-10.5zM22 76h10.5v10.5h-10.5zM34 76h10.5v10.5h-10.5zM46 76h10.5v10.5h-10.5zM10 88h10.5v10.5h-10.5zM22 88h10.5v10.5h-10.5zM34 88h10.5v10.5h-10.5zM46 88h10.5v10.5h-10.5zM10 100h10.5v10.5h-10.5zM22 100h10.5v10.5h-10.5zM34 100h10.5v10.5h-10.5zM46 100h10.5v10.5h-10.5zM10 112h10.5v10.5h-10.5zM22 112h10.5v10.5h-10.5zM34 112h10.5v10.5h-10.5zM46 112h10.5v10.5h-10.5zM10 124h10.5v10.5h-10.5zM22 124h10.5v10.5h-10.5zM34 124h10.5v10.5h-10.5zM46 124h10.5v10.5h-10.5z" fill="none" stroke="#8a8a8a" stroke-width="0.5"/>
<rect x="8.5" y="38.5" width="25.5" height="49.5" fill="none" stroke="#b3261e" stroke-width="1.2" rx="2"/>
<path d="M17.2,59.2 L25.2,67.2 M17.2,67.2 L25.2,59.2" fill="none" stroke="#b3261e" stroke-width="2" stroke-linecap="round"/>
<rect x="32.5" y="38.5" width="25.5" height="49.5" fill="none" stroke="#333" stroke-width="1.2" rx="2"/>
<path d="M40.8,63.2 L43.8,66.2 L49.8,60.1" fill="none" stroke="#1b7a3a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="8.5" y="86.5" width="25.5" height="49.5" fill="none" stroke="#333" stroke-width="1.2" rx="2"/>
<path d="M16.8,111.2 L19.8,114.2 L25.8,108.1" fill="none" stroke="#1b7a3a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="32.5" y="86.5" width="25.5" height="49.5" fill="none" stroke="#333" stroke-width="1.2" rx="2"/>
<path d="M40.8,111.2 L43.8,114.2 L49.8,108.1" fill="none" stroke="#1b7a3a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="15.2" cy="45.2" r="2.4" fill="#000"/>
<text x="33.2" y="149.5" text-anchor="middle" font-weight="bold" fill="#b3261e">1 of 4 fails</text>
<text x="96.2" y="13" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#D55E00">H</text>
<text x="96.2" y="27" text-anchor="middle">2×TP16</text>
<rect x="71.5" y="38.5" width="49.5" height="49.5" fill="url(#fx)" rx="2"/>
<rect x="71.5" y="86.5" width="49.5" height="49.5" fill="#f6d3bd" rx="2"/>
<path d="M73 40h10.5v10.5h-10.5zM85 40h10.5v10.5h-10.5zM97 40h10.5v10.5h-10.5zM109 40h10.5v10.5h-10.5zM73 52h10.5v10.5h-10.5zM85 52h10.5v10.5h-10.5zM97 52h10.5v10.5h-10.5zM109 52h10.5v10.5h-10.5zM73 64h10.5v10.5h-10.5zM85 64h10.5v10.5h-10.5zM97 64h10.5v10.5h-10.5zM109 64h10.5v10.5h-10.5zM73 76h10.5v10.5h-10.5zM85 76h10.5v10.5h-10.5zM97 76h10.5v10.5h-10.5zM109 76h10.5v10.5h-10.5zM73 88h10.5v10.5h-10.5zM85 88h10.5v10.5h-10.5zM97 88h10.5v10.5h-10.5zM109 88h10.5v10.5h-10.5zM73 100h10.5v10.5h-10.5zM85 100h10.5v10.5h-10.5zM97 100h10.5v10.5h-10.5zM109 100h10.5v10.5h-10.5zM73 112h10.5v10.5h-10.5zM85 112h10.5v10.5h-10.5zM97 112h10.5v10.5h-10.5zM109 112h10.5v10.5h-10.5zM73 124h10.5v10.5h-10.5zM85 124h10.5v10.5h-10.5zM97 124h10.5v10.5h-10.5zM109 124h10.5v10.5h-10.5z" fill="none" stroke="#8a8a8a" stroke-width="0.5"/>
<rect x="71.5" y="38.5" width="49.5" height="49.5" fill="none" stroke="#b3261e" stroke-width="1.2" rx="2"/>
<path d="M92.2,59.2 L100.2,67.2 M92.2,67.2 L100.2,59.2" fill="none" stroke="#b3261e" stroke-width="2" stroke-linecap="round"/>
<rect x="71.5" y="86.5" width="49.5" height="49.5" fill="none" stroke="#333" stroke-width="1.2" rx="2"/>
<path d="M91.8,111.2 L94.8,114.2 L100.8,108.1" fill="none" stroke="#1b7a3a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="78.2" cy="45.2" r="2.4" fill="#000"/>
<text x="96.2" y="149.5" text-anchor="middle" font-weight="bold" fill="#b3261e">1 of 2 fails</text>
<text x="159.2" y="13" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#009E73">M</text>
<text x="159.2" y="27" text-anchor="middle">TP16+2×TP8</text>
<rect x="134.5" y="38.5" width="25.5" height="49.5" fill="url(#fx)" rx="2"/>
<rect x="158.5" y="38.5" width="25.5" height="49.5" fill="#c2e6d9" rx="2"/>
<rect x="134.5" y="86.5" width="49.5" height="49.5" fill="#c2e6d9" rx="2"/>
<path d="M136 40h10.5v10.5h-10.5zM148 40h10.5v10.5h-10.5zM160 40h10.5v10.5h-10.5zM172 40h10.5v10.5h-10.5zM136 52h10.5v10.5h-10.5zM148 52h10.5v10.5h-10.5zM160 52h10.5v10.5h-10.5zM172 52h10.5v10.5h-10.5zM136 64h10.5v10.5h-10.5zM148 64h10.5v10.5h-10.5zM160 64h10.5v10.5h-10.5zM172 64h10.5v10.5h-10.5zM136 76h10.5v10.5h-10.5zM148 76h10.5v10.5h-10.5zM160 76h10.5v10.5h-10.5zM172 76h10.5v10.5h-10.5zM136 88h10.5v10.5h-10.5zM148 88h10.5v10.5h-10.5zM160 88h10.5v10.5h-10.5zM172 88h10.5v10.5h-10.5zM136 100h10.5v10.5h-10.5zM148 100h10.5v10.5h-10.5zM160 100h10.5v10.5h-10.5zM172 100h10.5v10.5h-10.5zM136 112h10.5v10.5h-10.5zM148 112h10.5v10.5h-10.5zM160 112h10.5v10.5h-10.5zM172 112h10.5v10.5h-10.5zM136 124h10.5v10.5h-10.5zM148 124h10.5v10.5h-10.5zM160 124h10.5v10.5h-10.5zM172 124h10.5v10.5h-10.5z" fill="none" stroke="#8a8a8a" stroke-width="0.5"/>
<rect x="134.5" y="38.5" width="25.5" height="49.5" fill="none" stroke="#b3261e" stroke-width="1.2" rx="2"/>
<path d="M143.2,59.2 L151.2,67.2 M143.2,67.2 L151.2,59.2" fill="none" stroke="#b3261e" stroke-width="2" stroke-linecap="round"/>
<rect x="158.5" y="38.5" width="25.5" height="49.5" fill="none" stroke="#333" stroke-width="1.2" rx="2"/>
<path d="M166.8,63.2 L169.8,66.2 L175.8,60.1" fill="none" stroke="#1b7a3a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="134.5" y="86.5" width="49.5" height="49.5" fill="none" stroke="#333" stroke-width="1.2" rx="2"/>
<path d="M154.8,111.2 L157.8,114.2 L163.8,108.1" fill="none" stroke="#1b7a3a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="141.2" cy="45.2" r="2.4" fill="#000"/>
<text x="159.2" y="149.5" text-anchor="middle" font-weight="bold" fill="#b3261e">1 of 3 fails</text>
<text x="222.2" y="13" text-anchor="middle" font-size="11.5" font-weight="bold">TP32 /</text>
<text x="222.2" y="27" text-anchor="middle">lockstep</text>
<rect x="197.5" y="38.5" width="49.5" height="97.5" fill="#e3e3e3" rx="2"/>
<path d="M199 40h10.5v10.5h-10.5zM211 40h10.5v10.5h-10.5zM223 40h10.5v10.5h-10.5zM235 40h10.5v10.5h-10.5zM199 52h10.5v10.5h-10.5zM211 52h10.5v10.5h-10.5zM223 52h10.5v10.5h-10.5zM235 52h10.5v10.5h-10.5zM199 64h10.5v10.5h-10.5zM211 64h10.5v10.5h-10.5zM223 64h10.5v10.5h-10.5zM235 64h10.5v10.5h-10.5zM199 76h10.5v10.5h-10.5zM211 76h10.5v10.5h-10.5zM223 76h10.5v10.5h-10.5zM235 76h10.5v10.5h-10.5zM199 88h10.5v10.5h-10.5zM211 88h10.5v10.5h-10.5zM223 88h10.5v10.5h-10.5zM235 88h10.5v10.5h-10.5zM199 100h10.5v10.5h-10.5zM211 100h10.5v10.5h-10.5zM223 100h10.5v10.5h-10.5zM235 100h10.5v10.5h-10.5zM199 112h10.5v10.5h-10.5zM211 112h10.5v10.5h-10.5zM223 112h10.5v10.5h-10.5zM235 112h10.5v10.5h-10.5zM199 124h10.5v10.5h-10.5zM211 124h10.5v10.5h-10.5zM223 124h10.5v10.5h-10.5zM235 124h10.5v10.5h-10.5z" fill="none" stroke="#8a8a8a" stroke-width="0.5"/>
<rect x="197.5" y="38.5" width="49.5" height="97.5" fill="none" stroke="#333" stroke-width="1.2" rx="2"/>
<path d="M217.8,87.2 L220.8,90.2 L226.8,84.1" fill="none" stroke="#1b7a3a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="204.2" cy="45.2" r="2.4" fill="#000"/>
<text x="222.2" y="149.5" text-anchor="middle" font-weight="bold" fill="#1b7a3a">compiles</text>
<text x="285.2" y="13" text-anchor="middle" font-size="11.5" font-weight="bold">Per-block</text>
<text x="285.2" y="27" text-anchor="middle">4×4 runtime</text>
<rect x="260.5" y="38.5" width="49.5" height="25.5" fill="#e3e3e3" rx="2"/>
<rect x="260.5" y="62.5" width="49.5" height="25.5" fill="#e3e3e3" rx="2"/>
<rect x="260.5" y="86.5" width="49.5" height="25.5" fill="#e3e3e3" rx="2"/>
<rect x="260.5" y="110.5" width="49.5" height="25.5" fill="#e3e3e3" rx="2"/>
<path d="M262 40h10.5v10.5h-10.5zM274 40h10.5v10.5h-10.5zM286 40h10.5v10.5h-10.5zM298 40h10.5v10.5h-10.5zM262 52h10.5v10.5h-10.5zM274 52h10.5v10.5h-10.5zM286 52h10.5v10.5h-10.5zM298 52h10.5v10.5h-10.5zM262 64h10.5v10.5h-10.5zM274 64h10.5v10.5h-10.5zM286 64h10.5v10.5h-10.5zM298 64h10.5v10.5h-10.5zM262 76h10.5v10.5h-10.5zM274 76h10.5v10.5h-10.5zM286 76h10.5v10.5h-10.5zM298 76h10.5v10.5h-10.5zM262 88h10.5v10.5h-10.5zM274 88h10.5v10.5h-10.5zM286 88h10.5v10.5h-10.5zM298 88h10.5v10.5h-10.5zM262 100h10.5v10.5h-10.5zM274 100h10.5v10.5h-10.5zM286 100h10.5v10.5h-10.5zM298 100h10.5v10.5h-10.5zM262 112h10.5v10.5h-10.5zM274 112h10.5v10.5h-10.5zM286 112h10.5v10.5h-10.5zM298 112h10.5v10.5h-10.5zM262 124h10.5v10.5h-10.5zM274 124h10.5v10.5h-10.5zM286 124h10.5v10.5h-10.5zM298 124h10.5v10.5h-10.5z" fill="none" stroke="#8a8a8a" stroke-width="0.5"/>
<rect x="260.5" y="38.5" width="49.5" height="25.5" fill="none" stroke="#333" stroke-width="1.2" rx="2"/>
<path d="M280.8,51.2 L283.8,54.2 L289.8,48.1" fill="none" stroke="#1b7a3a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="260.5" y="62.5" width="49.5" height="25.5" fill="none" stroke="#333" stroke-width="1.2" rx="2"/>
<path d="M280.8,75.2 L283.8,78.2 L289.8,72.1" fill="none" stroke="#1b7a3a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="260.5" y="86.5" width="49.5" height="25.5" fill="none" stroke="#333" stroke-width="1.2" rx="2"/>
<path d="M280.8,99.2 L283.8,102.2 L289.8,96.1" fill="none" stroke="#1b7a3a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="260.5" y="110.5" width="49.5" height="25.5" fill="none" stroke="#333" stroke-width="1.2" rx="2"/>
<path d="M280.8,123.2 L283.8,126.2 L289.8,120.1" fill="none" stroke="#1b7a3a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<line x1="259" y1="86.8" x2="311.5" y2="86.8" stroke="#000" stroke-width="2.2"/>
<circle cx="267.2" cy="45.2" r="2.4" fill="#000"/>
<text x="285.2" y="149.5" text-anchor="middle" font-weight="bold" fill="#1b7a3a">all compile</text>
<circle cx="14" cy="172" r="2.4" fill="#000"/>
<text x="21" y="176">chip 0 (origin)</text>
<rect x="108" y="167" width="11" height="9" fill="url(#fx)" stroke="#b3261e" stroke-width="1"/>
<text x="123" y="176">proper subset with chip 0: fails</text>
</svg>
```),
  caption: [Deviceless compile matrix on the $4 times 8$ v5e topology (`libtpu` 0.0.43, 0.0.43.2, and 0.0.48). In a 32-chip runtime, `libtpu` enters a limited-ICI-routing mode in which every multi-chip program over a proper subset containing chip 0 fails to compile, so `S`, `H`, and `M` each have one instance that cannot run on its own. Programs over all 32 chips (`TP32`, lockstep DP4$times$`TP8` and DP2$times$`TP16`) compile, as do host-row instances inside per-block 16-chip ($4 times 4$) runtimes, where this mode is off.],
) <fig-fullmesh>

_Process isolation vs. unified submeshes._ Integrating unified submeshes into vLLM would require refactoring approximately 1.5k–3k lines of code around process-global state (`TPUWorker`, `jax.clear_caches()`, and Ray executor bindings), couples four hosts into a single fault domain, and is precluded on full $4 times 8$ v5e slices by `libtpu`'s limited-ICI-routing mode. Moreover, even if supported across 32 chips, reducing warm reconfiguration latency from 21.1–27.5 s to $approx 9"–"10" s"$ would increase the median bulk makespan improvement by only 0.3 percentage points (from 9.9% to 10.2%, indicating that process-isolated reconfiguration already realizes $approx 97%$ of the potential gain; @sec-eval-submesh). PhaseMorph therefore adopts process-isolated switching for production deployments and uses the 16-chip unified runtime as an analytical baseline.
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

_Baselines and variants._ We compare _Vanilla vLLM Redeploy_, which tears down the source engines and deploys the target layout through external job submission, against cumulative PhaseMorph variants: _PhaseMorph-Snap_ (pre-sharded weight snapshots, shape-only graphs, and the in-process controller), _PhaseMorph-Cache_ (adds zero-threshold caching), and _PhaseMorph_ (adds TPU-decoupled pre-spawning and the executable bank). _PhaseMorph-PreRead_ adds the background CPU pre-read for cold first switches, and _PhaseMorph-Submesh_ is the unified-runtime prototype. For end-to-end comparisons we also include the static layouts `H` and `M` with reservation-aware routing.
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

@fig-segments stacks the same segments on one time axis.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 262" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<defs><pattern id="sr" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="4" height="4" fill="#fff"/><line x1="0" y1="0" x2="0" y2="4" stroke="#444" stroke-width="1.4"/></pattern><pattern id="sp" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="4" height="4" fill="#d5edf7"/><line x1="0" y1="0" x2="0" y2="4" stroke="#3b8fb5" stroke-width="1.2"/></pattern></defs>
<line x1="35" y1="233.5" x2="198" y2="233.5" stroke="#e2e2e2" stroke-width="1"/>
<text x="32" y="237.5" text-anchor="end">-30</text>
<line x1="35" y1="196" x2="198" y2="196" stroke="#222" stroke-width="1"/>
<text x="32" y="200" text-anchor="end">0</text>
<line x1="35" y1="158.5" x2="198" y2="158.5" stroke="#e2e2e2" stroke-width="1"/>
<text x="32" y="162.5" text-anchor="end">30</text>
<line x1="35" y1="121" x2="198" y2="121" stroke="#e2e2e2" stroke-width="1"/>
<text x="32" y="125" text-anchor="end">60</text>
<line x1="35" y1="83.5" x2="198" y2="83.5" stroke="#e2e2e2" stroke-width="1"/>
<text x="32" y="87.5" text-anchor="end">90</text>
<line x1="35" y1="46" x2="198" y2="46" stroke="#e2e2e2" stroke-width="1"/>
<text x="32" y="50" text-anchor="end">120</text>
<text x="10" y="127.2" text-anchor="middle" font-size="11.5" font-weight="bold" transform="rotate(-90 10 127.2)">Seconds after T<tspan dy="2.6" font-size="9.2">0</tspan><tspan dy="-2.6"> </tspan></text>
<rect x="44" y="191.3" width="22" height="4.7" fill="#8c8c8c" stroke="#fff" stroke-width="0.4"/>
<rect x="44" y="177.4" width="22" height="13.9" fill="#c8c8c8" stroke="#fff" stroke-width="0.4"/>
<rect x="44" y="150.5" width="22" height="26.9" fill="#88CCEE" stroke="#fff" stroke-width="0.4"/>
<rect x="44" y="146.1" width="22" height="4.4" fill="#DDCC77" stroke="#fff" stroke-width="0.4"/>
<rect x="44" y="139.8" width="22" height="6.3" fill="#CC6677" stroke="#fff" stroke-width="0.4"/>
<rect x="44" y="107.4" width="22" height="32.4" fill="#999933" stroke="#fff" stroke-width="0.4"/>
<rect x="44" y="39.1" width="22" height="68.3" fill="#AA4499" stroke="#fff" stroke-width="0.4"/>
<rect x="44" y="32.7" width="22" height="6.4" fill="#5f5f5f" stroke="#fff" stroke-width="0.4"/>
<rect x="44" y="30.7" width="22" height="2" fill="url(#sr)" stroke="#444" stroke-width="0.4"/>
<text x="55" y="26.7" text-anchor="middle" font-weight="bold">132.2</text>
<text x="55" y="254.5" text-anchor="middle">Snap</text>
<rect x="76" y="191.6" width="22" height="4.4" fill="#8c8c8c" stroke="#fff" stroke-width="0.4"/>
<rect x="76" y="176.4" width="22" height="15.2" fill="#c8c8c8" stroke="#fff" stroke-width="0.4"/>
<rect x="76" y="149.5" width="22" height="26.9" fill="#88CCEE" stroke="#fff" stroke-width="0.4"/>
<rect x="76" y="144.9" width="22" height="4.5" fill="#DDCC77" stroke="#fff" stroke-width="0.4"/>
<rect x="76" y="138.9" width="22" height="6" fill="#CC6677" stroke="#fff" stroke-width="0.4"/>
<rect x="76" y="135.6" width="22" height="3.3" fill="#999933" stroke="#fff" stroke-width="0.4"/>
<rect x="76" y="91.7" width="22" height="43.9" fill="#AA4499" stroke="#fff" stroke-width="0.4"/>
<rect x="76" y="87" width="22" height="4.7" fill="#5f5f5f" stroke="#fff" stroke-width="0.4"/>
<rect x="76" y="83.7" width="22" height="3.3" fill="url(#sr)" stroke="#444" stroke-width="0.4"/>
<text x="87" y="79.7" text-anchor="middle" font-weight="bold">89.8</text>
<text x="87" y="254.5" text-anchor="middle">Cache</text>
<rect x="108" y="196" width="22" height="42.9" fill="url(#sp)" stroke="#3b8fb5" stroke-width="0.6"/>
<rect x="108" y="193.5" width="22" height="2.5" fill="#8c8c8c" stroke="#fff" stroke-width="0.4"/>
<rect x="108" y="192.4" width="22" height="1.1" fill="#c8c8c8" stroke="#fff" stroke-width="0.4"/>
<rect x="108" y="187.6" width="22" height="4.9" fill="#DDCC77" stroke="#fff" stroke-width="0.4"/>
<rect x="108" y="181.4" width="22" height="6.1" fill="#CC6677" stroke="#fff" stroke-width="0.4"/>
<rect x="108" y="178.2" width="22" height="3.2" fill="#999933" stroke="#fff" stroke-width="0.4"/>
<rect x="108" y="172" width="22" height="6.2" fill="#AA4499" stroke="#fff" stroke-width="0.4"/>
<rect x="108" y="166" width="22" height="5.9" fill="#5f5f5f" stroke="#fff" stroke-width="0.4"/>
<rect x="108" y="161.6" width="22" height="4.4" fill="url(#sr)" stroke="#444" stroke-width="0.4"/>
<text x="119" y="157.6" text-anchor="middle" font-weight="bold">27.5</text>
<text x="119" y="254.5" text-anchor="middle">Full A</text>
<rect x="140" y="196" width="22" height="42.6" fill="url(#sp)" stroke="#3b8fb5" stroke-width="0.6"/>
<rect x="140" y="193.6" width="22" height="2.4" fill="#8c8c8c" stroke="#fff" stroke-width="0.4"/>
<rect x="140" y="192.5" width="22" height="1.1" fill="#c8c8c8" stroke="#fff" stroke-width="0.4"/>
<rect x="140" y="187.6" width="22" height="4.9" fill="#DDCC77" stroke="#fff" stroke-width="0.4"/>
<rect x="140" y="181.5" width="22" height="6.1" fill="#CC6677" stroke="#fff" stroke-width="0.4"/>
<rect x="140" y="178.3" width="22" height="3.2" fill="#999933" stroke="#fff" stroke-width="0.4"/>
<rect x="140" y="172.2" width="22" height="6.2" fill="#AA4499" stroke="#fff" stroke-width="0.4"/>
<rect x="140" y="166.5" width="22" height="5.7" fill="#5f5f5f" stroke="#fff" stroke-width="0.4"/>
<line x1="137" y1="169.6" x2="165" y2="169.6" stroke="#000" stroke-width="1.6"/>
<text x="151" y="162.5" text-anchor="middle" font-weight="bold">21.1</text>
<text x="151" y="254.5" text-anchor="middle">Full B</text>
<rect x="172" y="196" width="22" height="40.5" fill="url(#sp)" stroke="#3b8fb5" stroke-width="0.6"/>
<rect x="172" y="193.7" width="22" height="2.3" fill="#8c8c8c" stroke="#fff" stroke-width="0.4"/>
<rect x="172" y="192.6" width="22" height="1.1" fill="#c8c8c8" stroke="#fff" stroke-width="0.4"/>
<rect x="172" y="188" width="22" height="4.7" fill="#DDCC77" stroke="#fff" stroke-width="0.4"/>
<rect x="172" y="181.8" width="22" height="6.2" fill="#CC6677" stroke="#fff" stroke-width="0.4"/>
<rect x="172" y="178.6" width="22" height="3.2" fill="#999933" stroke="#fff" stroke-width="0.4"/>
<rect x="172" y="172.2" width="22" height="6.3" fill="#AA4499" stroke="#fff" stroke-width="0.4"/>
<rect x="172" y="167.7" width="22" height="4.5" fill="#5f5f5f" stroke="#fff" stroke-width="0.4"/>
<line x1="169" y1="172.4" x2="197" y2="172.4" stroke="#000" stroke-width="1.6"/>
<text x="183" y="166.2" text-anchor="middle" font-weight="bold">18.8</text>
<text x="183" y="254.5" text-anchor="middle">Partial</text>
<text x="42" y="213.5" fill="#2f6f8f">pre-spawn</text>
<text x="42" y="226.5" fill="#2f6f8f">hidden →</text>
<rect x="206" y="21" width="10" height="10" fill="#8c8c8c" stroke="#555" stroke-width="0.5"/>
<text x="220" y="30">drain + release</text>
<rect x="206" y="36" width="10" height="10" fill="#c8c8c8" stroke="#555" stroke-width="0.5"/>
<text x="220" y="45">gate + VFIO</text>
<rect x="206" y="51" width="10" height="10" fill="#88CCEE" stroke="#555" stroke-width="0.5"/>
<text x="220" y="60">spawn + imports</text>
<rect x="206" y="66" width="10" height="10" fill="#DDCC77" stroke="#555" stroke-width="0.5"/>
<text x="220" y="75">init_device</text>
<rect x="206" y="81" width="10" height="10" fill="#CC6677" stroke="#555" stroke-width="0.5"/>
<text x="220" y="90">load_model</text>
<rect x="206" y="96" width="10" height="10" fill="#999933" stroke="#555" stroke-width="0.5"/>
<text x="220" y="105">KV init</text>
<rect x="206" y="111" width="10" height="10" fill="#AA4499" stroke="#555" stroke-width="0.5"/>
<text x="220" y="120">compile/warm-up</text>
<rect x="206" y="126" width="10" height="10" fill="#5f5f5f" stroke="#555" stroke-width="0.5"/>
<text x="220" y="135">gateway + other</text>
<rect x="206" y="141" width="10" height="10" fill="url(#sr)" stroke="#555" stroke-width="0.5"/>
<text x="220" y="150">ramp (+)</text>
<rect x="206" y="156" width="10" height="10" fill="url(#sp)" stroke="#555" stroke-width="0.5"/>
<text x="220" y="165">hidden pre-spawn</text>
<line x1="206" y1="186" x2="218" y2="186" stroke="#000" stroke-width="1.6"/>
<text x="222" y="190">C if ramp &lt; 0</text>
</svg>
```),
  caption: [Segments of @tab-segments stacked on one axis (seconds after $T_0$). Bars below zero are pre-spawn time hidden in the source phase. The hatched cap is the ramp deficit; when the ramp is negative, a black tick marks $C$ below the end of the dispatch gap.],
) <fig-segments>

_Warm partial switch._ A partial `S → M` switch rebuilds only half of the slice. Its effective cost falls from 186.0 s with vanilla redeployment to 128.9 s with PhaseMorph-Snap, 87.3 s with PhaseMorph-Cache, and 18.84 s with PhaseMorph. Throughout the switch, the two kept `TP8` instances pause for only about 2.1 s, and the first long request is dispatched to the new `TP16` instance 20.6 s after $T_0$. Once running, the new instance serves `W6` at the static-`M` rate (407.4 s from first dispatch to finish versus 409.2 s), so the switch leaves no join tail.

_Cold first switch and amortization._ Without pre-read, inline provisioning makes the first switch into `H` cost 222.63 s, avoiding a separate 350.8 s provisioning pass. PhaseMorph-PreRead converts weights during the source phase; the child starts when the target workers park (about 156 s before $T_0$, absorbing roughly two-thirds of the 242 s CPU conversion) and finishes about 86 s after $T_0$ (79.6 s of worker wait), so the cold switch costs 134.10 s, 88.5 s (40%) less. Amortized over 1, 3, and 10 switches, the per-switch cost is 134.1, 58.8, and 32.4 s, compared with 222.6, 88.3, and 41.3 s without pre-read and 378.3, 144.4, and 62.6 s with a separate provisioning pass.
@fig-coldswitch places both cold switches on one time axis.

#wide-fig(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 270" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<defs><pattern id="cr" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="4" height="4" fill="#fff"/><line x1="0" y1="0" x2="0" y2="4" stroke="#444" stroke-width="1.4"/></pattern><pattern id="cc" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="#f7e1e5"/><line x1="0" y1="0" x2="0" y2="5" stroke="#CC6677" stroke-width="1.2"/></pattern></defs>
<line x1="150" y1="26" x2="616" y2="26" stroke="#666" stroke-width="1"/>
<line x1="150" y1="23" x2="150" y2="26" stroke="#666" stroke-width="1"/>
<text x="150" y="20" text-anchor="middle" fill="#666">-200</text>
<line x1="204.2" y1="23" x2="204.2" y2="26" stroke="#666" stroke-width="1"/>
<text x="204.2" y="20" text-anchor="middle" fill="#666">-150</text>
<line x1="258.4" y1="23" x2="258.4" y2="26" stroke="#666" stroke-width="1"/>
<text x="258.4" y="20" text-anchor="middle" fill="#666">-100</text>
<line x1="312.6" y1="23" x2="312.6" y2="26" stroke="#666" stroke-width="1"/>
<text x="312.6" y="20" text-anchor="middle" fill="#666">-50</text>
<line x1="366.7" y1="23" x2="366.7" y2="26" stroke="#666" stroke-width="1"/>
<text x="366.7" y="20" text-anchor="middle" fill="#666">0</text>
<line x1="420.9" y1="23" x2="420.9" y2="26" stroke="#666" stroke-width="1"/>
<text x="420.9" y="20" text-anchor="middle" fill="#666">50</text>
<line x1="475.1" y1="23" x2="475.1" y2="26" stroke="#666" stroke-width="1"/>
<text x="475.1" y="20" text-anchor="middle" fill="#666">100</text>
<line x1="529.3" y1="23" x2="529.3" y2="26" stroke="#666" stroke-width="1"/>
<text x="529.3" y="20" text-anchor="middle" fill="#666">150</text>
<line x1="583.5" y1="23" x2="583.5" y2="26" stroke="#666" stroke-width="1"/>
<text x="583.5" y="20" text-anchor="middle" fill="#666">200</text>
<text x="616" y="38" text-anchor="end" fill="#666">s relative to T<tspan dy="2.6" font-size="8.8">0</tspan><tspan dy="-2.6"> </tspan></text>
<line x1="366.7" y1="26" x2="366.7" y2="206" stroke="#000" stroke-width="1" stroke-dasharray="3,2"/>
<text x="8" y="42" font-size="11.5" font-weight="bold">Cold, inline snapshot</text>
<text x="142" y="66" text-anchor="end" font-weight="bold">Source S</text>
<rect x="158.5" y="52" width="208.2" height="20" fill="#0072B2"/>
<text x="262.6" y="66" text-anchor="middle" fill="#fff">serves W0</text>
<text x="142" y="92" text-anchor="end" font-weight="bold">Target H</text>
<rect x="158.5" y="78" width="36.4" height="20" fill="#88CCEE" stroke="#fff" stroke-width="0.6"/>
<rect x="194.9" y="79" width="171.8" height="18" fill="#fafafa" stroke="#888" stroke-width="0.8" stroke-dasharray="3,2"/>
<text x="280.8" y="92" text-anchor="middle" fill="#666">parked (CPU only)</text>
<rect x="366.7" y="78" width="3.1" height="20" fill="#8c8c8c" stroke="#fff" stroke-width="0.5"/>
<rect x="369.9" y="78" width="4.2" height="20" fill="#DDCC77" stroke="#fff" stroke-width="0.5"/>
<rect x="374.1" y="78" width="172.2" height="20" fill="#CC6677" stroke="#fff" stroke-width="0.5"/>
<text x="460.2" y="92" text-anchor="middle" fill="#fff">load_model 158.9</text>
<rect x="546.3" y="78" width="8.2" height="20" fill="#882255" stroke="#fff" stroke-width="0.5"/>
<rect x="554.5" y="78" width="2.9" height="20" fill="#999933" stroke="#fff" stroke-width="0.5"/>
<rect x="557.4" y="78" width="38.8" height="20" fill="#AA4499" stroke="#fff" stroke-width="0.5"/>
<text x="576.8" y="92" text-anchor="middle" fill="#fff">35.8</text>
<rect x="596.1" y="78" width="2.4" height="20" fill="#882255" stroke="#fff" stroke-width="0.5"/>
<rect x="598.6" y="78" width="6" height="20" fill="#5f5f5f" stroke="#fff" stroke-width="0.5"/>
<rect x="604.6" y="78" width="3.4" height="20" fill="url(#cr)" stroke="#444" stroke-width="0.5"/>
<line x1="608" y1="75" x2="608" y2="101" stroke="#000" stroke-width="1.8"/>
<text x="608" y="72" text-anchor="end" font-weight="bold">C<tspan dy="2.6" font-size="8.8">first</tspan><tspan dy="-2.6"> = 222.6 s</tspan></text>
<text x="8" y="114" font-size="11.5" font-weight="bold">PhaseMorph-PreRead</text>
<text x="142" y="138" text-anchor="end" font-weight="bold">Source S</text>
<rect x="161.4" y="124" width="205.3" height="20" fill="#0072B2"/>
<text x="264.1" y="138" text-anchor="middle" fill="#fff">serves W0 (no measurable slowdown)</text>
<text x="142" y="164" text-anchor="end" font-weight="bold">Pre-read child</text>
<rect x="198.2" y="150" width="262" height="20" fill="url(#cc)" stroke="#CC6677" stroke-width="1"/>
<text x="329.2" y="164" text-anchor="middle">CPU convert → snapshot in /dev/shm</text>
<text x="142" y="190" text-anchor="end" font-weight="bold">Target H</text>
<rect x="161.4" y="176" width="36.9" height="20" fill="#88CCEE" stroke="#fff" stroke-width="0.6"/>
<rect x="198.3" y="177" width="168.4" height="18" fill="#fafafa" stroke="#888" stroke-width="0.8" stroke-dasharray="3,2"/>
<text x="282.5" y="190" text-anchor="middle" fill="#666">parked (CPU only)</text>
<rect x="366.7" y="176" width="3" height="20" fill="#8c8c8c" stroke="#fff" stroke-width="0.5"/>
<rect x="369.8" y="176" width="4.4" height="20" fill="#DDCC77" stroke="#fff" stroke-width="0.5"/>
<rect x="374.2" y="176" width="86.2" height="20" fill="#ebbfc7" stroke="#fff" stroke-width="0.5"/>
<text x="417.3" y="190" text-anchor="middle">wait 79.6</text>
<rect x="460.4" y="176" width="6.2" height="20" fill="#CC6677" stroke="#fff" stroke-width="0.5"/>
<rect x="466.6" y="176" width="2.8" height="20" fill="#999933" stroke="#fff" stroke-width="0.5"/>
<rect x="469.4" y="176" width="38.2" height="20" fill="#AA4499" stroke="#fff" stroke-width="0.5"/>
<text x="488.5" y="190" text-anchor="middle" fill="#fff">35.3</text>
<rect x="507.6" y="176" width="2.5" height="20" fill="#882255" stroke="#fff" stroke-width="0.5"/>
<rect x="510.1" y="176" width="5.2" height="20" fill="#5f5f5f" stroke="#fff" stroke-width="0.5"/>
<line x1="512.1" y1="173" x2="512.1" y2="199" stroke="#000" stroke-width="1.8"/>
<text x="512.1" y="170" text-anchor="middle" font-weight="bold">C<tspan dy="2.6" font-size="8.8">first</tspan><tspan dy="-2.6"> = 134.1 s</tspan></text>
<line x1="460.2" y1="170" x2="460.2" y2="176" stroke="#CC6677" stroke-width="1.2"/>
<rect x="8" y="215" width="10" height="10" fill="#0072B2" stroke="#555" stroke-width="0.5"/>
<text x="22" y="224">S serving</text>
<rect x="166" y="215" width="10" height="10" fill="#88CCEE" stroke="#555" stroke-width="0.5"/>
<text x="180" y="224">spawn + imports</text>
<rect x="324" y="215" width="10" height="10" fill="#8c8c8c" stroke="#555" stroke-width="0.5"/>
<text x="338" y="224">drain / handoff</text>
<rect x="482" y="215" width="10" height="10" fill="#DDCC77" stroke="#555" stroke-width="0.5"/>
<text x="496" y="224">init_device</text>
<rect x="8" y="230" width="10" height="10" fill="#CC6677" stroke="#555" stroke-width="0.5"/>
<text x="22" y="239">load_model</text>
<rect x="166" y="230" width="10" height="10" fill="#ebbfc7" stroke="#555" stroke-width="0.5"/>
<text x="180" y="239">pre-read wait</text>
<rect x="324" y="230" width="10" height="10" fill="#882255" stroke="#555" stroke-width="0.5"/>
<text x="338" y="239">snapshot / bank write</text>
<rect x="482" y="230" width="10" height="10" fill="#999933" stroke="#555" stroke-width="0.5"/>
<text x="496" y="239">KV cache init</text>
<rect x="8" y="245" width="10" height="10" fill="#AA4499" stroke="#555" stroke-width="0.5"/>
<text x="22" y="254">compile / warm-up</text>
<rect x="166" y="245" width="10" height="10" fill="#5f5f5f" stroke="#555" stroke-width="0.5"/>
<text x="180" y="254">endpoint + gateway</text>
<rect x="324" y="245" width="10" height="10" fill="url(#cc)" stroke="#555" stroke-width="0.5"/>
<text x="338" y="254">pre-read child</text>
<rect x="482" y="245" width="10" height="10" fill="url(#cr)" stroke="#555" stroke-width="0.5"/>
<text x="496" y="254">ramp</text>
</svg>
```),
  caption: [Cold first switch into `H` with inline snapshot provisioning (top) and with PhaseMorph-PreRead (bottom), to scale. Left of $T_0$: means of the phase-start and parking timestamps (three and two runs); right of $T_0$: mean segments. The pre-read child starts when the workers park, about 156 s before $T_0$, and finishes about 86 s after it. The target worker waits 79.6 s for the child, so `load_model` time on the critical path falls from 158.9 s to 85.3 s and $C_"first"$ from 222.6 s to 134.1 s.],
) <fig-coldswitch>

== Foreground Serving Throughput Preservation <sec-eval-foreground>

PhaseMorph's preparation runs on the same hosts as the source layout, so it must not slow the source phase. @tab-foreground compares the throughput of `S` on `W0` with and without concurrent preparation. With pre-spawning, repetitions of both full and partial switches overlap the control spread (within 0.4% of the control minimum in the late window and at or above it in the early window), and the source phase lasts 187.7–194.6 s versus 191.6–193.9 s. The pre-read child, which runs at low priority (`nice 10`) on the host CPUs, likewise stays within 1.0% of the cold-switch control in the early window and slightly above it thereafter.
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

@fig-source plots every repetition of @tab-foreground.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 182" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<text x="159" y="14" text-anchor="middle" font-weight="bold">Early window</text>
<text x="159" y="28" text-anchor="middle" fill="#666">S tok/s</text>
<rect x="112" y="32" width="94" height="108" fill="none" stroke="#222" stroke-width="0.8"/>
<line x1="137.6" y1="32" x2="137.6" y2="140" stroke="#e2e2e2" stroke-width="1"/>
<text x="137.6" y="154" text-anchor="middle">2,650</text>
<line x1="180.4" y1="32" x2="180.4" y2="140" stroke="#e2e2e2" stroke-width="1"/>
<text x="180.4" y="154" text-anchor="middle">2,700</text>
<line x1="128.2" y1="44" x2="152.2" y2="44" stroke="#0072B2" stroke-width="1.2"/>
<circle cx="128.2" cy="44" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<circle cx="150.5" cy="44" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<circle cx="152.2" cy="44" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<line x1="147" y1="64" x2="157.3" y2="64" stroke="#0072B2" stroke-width="1.2"/>
<circle cx="147.0" cy="64" r="3.3" fill="#0072B2"/>
<circle cx="147.0" cy="64" r="3.3" fill="#0072B2"/>
<circle cx="157.3" cy="64" r="3.3" fill="#0072B2"/>
<line x1="147.9" y1="84" x2="160.7" y2="84" stroke="#0072B2" stroke-width="1.2"/>
<circle cx="147.9" cy="84" r="3.3" fill="#0072B2"/>
<circle cx="155.6" cy="84" r="3.3" fill="#0072B2"/>
<circle cx="160.7" cy="84" r="3.3" fill="#0072B2"/>
<line x1="184.6" y1="104" x2="189.8" y2="104" stroke="#0072B2" stroke-width="1.2"/>
<circle cx="184.6" cy="104" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<circle cx="189.8" cy="104" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<circle cx="184.6" cy="104" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<line x1="162.4" y1="124" x2="190.6" y2="124" stroke="#0072B2" stroke-width="1.2"/>
<circle cx="162.4" cy="124" r="3.3" fill="#0072B2"/>
<circle cx="190.6" cy="124" r="3.3" fill="#0072B2"/>
<line x1="112" y1="94" x2="206" y2="94" stroke="#999" stroke-width="0.8" stroke-dasharray="3,2"/>
<text x="267" y="14" text-anchor="middle" font-weight="bold">Late window</text>
<text x="267" y="28" text-anchor="middle" fill="#666">S tok/s</text>
<rect x="220" y="32" width="94" height="108" fill="none" stroke="#222" stroke-width="0.8"/>
<line x1="239.6" y1="32" x2="239.6" y2="140" stroke="#e2e2e2" stroke-width="1"/>
<text x="239.6" y="154" text-anchor="middle">3,950</text>
<line x1="278.8" y1="32" x2="278.8" y2="140" stroke="#e2e2e2" stroke-width="1"/>
<text x="278.8" y="154" text-anchor="middle">4,050</text>
<line x1="239.2" y1="44" x2="264.3" y2="44" stroke="#0072B2" stroke-width="1.2"/>
<circle cx="264.3" cy="44" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<circle cx="239.2" cy="44" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<circle cx="249.4" cy="44" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<line x1="232.5" y1="64" x2="270.9" y2="64" stroke="#0072B2" stroke-width="1.2"/>
<circle cx="232.5" cy="64" r="3.3" fill="#0072B2"/>
<circle cx="268.6" cy="64" r="3.3" fill="#0072B2"/>
<circle cx="270.9" cy="64" r="3.3" fill="#0072B2"/>
<line x1="246.6" y1="84" x2="303.4" y2="84" stroke="#0072B2" stroke-width="1.2"/>
<circle cx="296.4" cy="84" r="3.3" fill="#0072B2"/>
<circle cx="246.6" cy="84" r="3.3" fill="#0072B2"/>
<circle cx="303.4" cy="84" r="3.3" fill="#0072B2"/>
<line x1="231.4" y1="104" x2="267.4" y2="104" stroke="#0072B2" stroke-width="1.2"/>
<circle cx="253.3" cy="104" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<circle cx="267.4" cy="104" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<circle cx="231.4" cy="104" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<line x1="279.1" y1="124" x2="280.3" y2="124" stroke="#0072B2" stroke-width="1.2"/>
<circle cx="279.1" cy="124" r="3.3" fill="#0072B2"/>
<circle cx="280.3" cy="124" r="3.3" fill="#0072B2"/>
<line x1="220" y1="94" x2="314" y2="94" stroke="#999" stroke-width="0.8" stroke-dasharray="3,2"/>
<text x="104" y="48" text-anchor="end">No prep (Cache)</text>
<text x="104" y="68" text-anchor="end">Pre-spawn, full</text>
<text x="104" y="88" text-anchor="end">Pre-spawn, partial</text>
<text x="104" y="108" text-anchor="end">No child (cold)</text>
<text x="104" y="128" text-anchor="end">Pre-read child</text>
<circle cx="18" cy="170" r="3.1" fill="#fff" stroke="#0072B2" stroke-width="1.3"/>
<text x="25" y="174">control</text>
<circle cx="82" cy="170" r="3.3" fill="#0072B2"/>
<text x="89" y="174">with preparation (one dot per rep)</text>
</svg>
```),
  caption: [Source-phase `S` throughput on `W0`, one dot per repetition of @tab-foreground (open: no preparation; filled: with preparation; rows below the dashed line come from the cold-switch campaign; windows as defined in @tab-foreground). Prepared runs overlap the control spreads; the largest shortfall below a control's minimum is 1.0% (pre-read child, early window).],
) <fig-source>

Correctness is preserved as well. Banked executables served every call of the target phase with no misses or rejections (65,956 model-step calls), greedy outputs of banked engines matched those of natively loaded engines token for token on 32 prompts per engine, and deliberately invalidating one rank's bank caused all ranks to fall back to normal compilation without failing the deployment.

== End-to-End Multi-Phase Makespan and Break-Even Analysis <sec-eval-e2e>

We evaluate switching across the two-phase job space $(k_s, k_l) in [0, 8] times [0.25, 8]$ with the cost model of @sec-overview, using a static `H` rate of $a_"0H" = 267.8" s"$ per `W0` manifest (at a 131k-token context limit). The model reproduces all 10 end-to-end multi-phase runs within $max |Delta k| lt.eq 0.009$ manifests.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 300" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<polygon points="46,252 312,252 312,228.8 67.5,228.8 48.6,241 47.7,238.2 46,237.8" fill="#f3c3a2" stroke="#f3c3a2" stroke-width="0.6"/>
<polygon points="46,237.8 49.4,238.3 271.7,14.8 271.7,14 46,14" fill="#c6dcef" stroke="#c6dcef" stroke-width="0.6"/>
<polygon points="48.6,241 67.5,228.6 300,14.1 300,14 272.5,14 49.4,238.3 48.6,239.9" fill="#b5e0cf" stroke="#b5e0cf" stroke-width="0.6"/>
<polygon points="48.6,239.9 48.6,238.4" fill="#f3c3a2" stroke="#f3c3a2" stroke-width="0.6"/>
<polygon points="67.5,228.8 312,228.8 312,14 300,14.1 67.5,228.6" fill="#e2c6e8" stroke="#e2c6e8" stroke-width="0.6"/>
<path d="M46.0,237.8 L47.7,238.2 L48.6,241.0 L67.5,228.8 L312.0,228.8" fill="none" stroke="#333" stroke-width="0.9"/>
<path d="M46.0,237.8 L49.4,238.3 L271.7,14.8" fill="none" stroke="#333" stroke-width="0.9"/>
<path d="M48.6,241.0 L67.5,228.6 L300.0,14.1" fill="none" stroke="#333" stroke-width="0.9"/>
<path d="M48.6,239.9 L49.4,238.3 L272.5,14.0 L300.0,14.0" fill="none" stroke="#333" stroke-width="0.9"/>
<path d="M48.6,239.9" fill="none" stroke="#333" stroke-width="0.9"/>
<path d="M48.6,238.4" fill="none" stroke="#333" stroke-width="0.9"/>
<path d="M67.5,228.8 L312.0,228.8" fill="none" stroke="#333" stroke-width="0.9"/>
<path d="M67.5,228.6 L300.0,14.1 L312.0,14.0" fill="none" stroke="#333" stroke-width="0.9"/>
<line x1="46" y1="192.5" x2="312" y2="192.5" stroke="#fff" stroke-width="0.6" stroke-opacity="0.7"/>
<line x1="46" y1="133" x2="312" y2="133" stroke="#fff" stroke-width="0.6" stroke-opacity="0.7"/>
<line x1="46" y1="73.5" x2="312" y2="73.5" stroke="#fff" stroke-width="0.6" stroke-opacity="0.7"/>
<rect x="46" y="14" width="266" height="238" fill="none" stroke="#222" stroke-width="1"/>
<line x1="43" y1="252" x2="46" y2="252" stroke="#222" stroke-width="1"/>
<text x="41" y="256" text-anchor="end">0</text>
<line x1="43" y1="192.5" x2="46" y2="192.5" stroke="#222" stroke-width="1"/>
<text x="41" y="196.5" text-anchor="end">2</text>
<line x1="43" y1="133" x2="46" y2="133" stroke="#222" stroke-width="1"/>
<text x="41" y="137" text-anchor="end">4</text>
<line x1="43" y1="73.5" x2="46" y2="73.5" stroke="#222" stroke-width="1"/>
<text x="41" y="77.5" text-anchor="end">6</text>
<line x1="43" y1="14" x2="46" y2="14" stroke="#222" stroke-width="1"/>
<text x="41" y="18" text-anchor="end">8</text>
<line x1="46" y1="252" x2="46" y2="255" stroke="#222" stroke-width="1"/>
<text x="46" y="267" text-anchor="middle">0.25</text>
<line x1="106.1" y1="252" x2="106.1" y2="255" stroke="#222" stroke-width="1"/>
<text x="106.1" y="267" text-anchor="middle">2</text>
<line x1="174.7" y1="252" x2="174.7" y2="255" stroke="#222" stroke-width="1"/>
<text x="174.7" y="267" text-anchor="middle">4</text>
<line x1="243.4" y1="252" x2="243.4" y2="255" stroke="#222" stroke-width="1"/>
<text x="243.4" y="267" text-anchor="middle">6</text>
<line x1="312" y1="252" x2="312" y2="255" stroke="#222" stroke-width="1"/>
<text x="312" y="267" text-anchor="middle">8</text>
<text x="179" y="282" text-anchor="middle" font-size="11.5" font-weight="bold">Long phase k<tspan dy="2.6" font-size="9.2">l</tspan><tspan dy="-2.6"> (W6 manifests)</tspan></text>
<text x="13" y="133" text-anchor="middle" font-size="11.5" font-weight="bold" transform="rotate(-90 13 133.0)">Short phase k<tspan dy="2.6" font-size="9.2">s</tspan><tspan dy="-2.6"> (W0 manifests)</tspan></text>
<text x="102.6" y="64.6" text-anchor="middle" font-size="12" font-weight="bold">Partial S→M</text>
<text x="102.6" y="78.6" text-anchor="middle">41% of grid</text>
<text x="239.9" y="174.6" text-anchor="middle" font-size="12" font-weight="bold">Full S→H</text>
<text x="239.9" y="188.6" text-anchor="middle">41% of grid</text>
<text x="214.2" y="85.2" text-anchor="middle" font-weight="bold" transform="rotate(-44.0 214.2 81.2)">static M 6%</text>
<text x="195.3" y="244.4" text-anchor="middle" font-weight="bold">static H 12%  (k<tspan dy="2.6" font-size="8.8">s</tspan><tspan dy="-2.6"> &lt; k</tspan><tspan dy="2.6" font-size="8.8">s</tspan><tspan dy="-2.6">* = 0.78)</tspan></text>
<circle cx="71.7" cy="222.2" r="4" fill="#fff" stroke="#000" stroke-width="1.6"/>
<line x1="70.7" y1="218.2" x2="68.3" y2="149.4" stroke="#000" stroke-width="0.8"/>
<text x="49.4" y="122.6">measured O3 runs</text>
<text x="49.4" y="135.6">k<tspan dy="2.6" font-size="8.8">s</tspan><tspan dy="-2.6"> = k</tspan><tspan dy="2.6" font-size="8.8">l</tspan><tspan dy="-2.6"> = 1</tspan></text>
</svg>
```),
  caption: [[Phase-Space Break-Even Coverage] Best strategy across $(k_l, k_s) in [0.25, 8] times [0, 8]$ (`W6` and `W0` manifests), computed by the bulk cost model at PhaseMorph's measured costs ($C_"full" = 27.5" s"$, $C_"part" = 18.8" s"$, $O = 32.6" s"$); percentages are shares of the model's 0.25-step grid. Switching wins 82% of the grid (41% full `S → H`, 41% partial `S → M`); static `H` wins only below $k_s^* = 0.78$, and static `M` only in a 6% band. The circle marks $k_s = k_l = 1$, the job size of the measured switch runs.],
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

@fig-breakeven shows how $k_s^*$ scales with the switch cost.

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 240" font-family="Liberation Sans, Arial, Helvetica, sans-serif" font-size="11" fill="#222">
<line x1="44" y1="196" x2="272" y2="196" stroke="#e2e2e2" stroke-width="1"/>
<text x="39" y="200" text-anchor="end">0</text>
<line x1="44" y1="168" x2="272" y2="168" stroke="#e2e2e2" stroke-width="1"/>
<text x="39" y="172" text-anchor="end">1</text>
<line x1="44" y1="140" x2="272" y2="140" stroke="#e2e2e2" stroke-width="1"/>
<text x="39" y="144" text-anchor="end">2</text>
<line x1="44" y1="112" x2="272" y2="112" stroke="#e2e2e2" stroke-width="1"/>
<text x="39" y="116" text-anchor="end">3</text>
<line x1="44" y1="84" x2="272" y2="84" stroke="#e2e2e2" stroke-width="1"/>
<text x="39" y="88" text-anchor="end">4</text>
<line x1="44" y1="56" x2="272" y2="56" stroke="#e2e2e2" stroke-width="1"/>
<text x="39" y="60" text-anchor="end">5</text>
<line x1="44" y1="28" x2="272" y2="28" stroke="#e2e2e2" stroke-width="1"/>
<text x="39" y="32" text-anchor="end">6</text>
<line x1="44" y1="196" x2="44" y2="199" stroke="#222" stroke-width="1"/>
<text x="44" y="211" text-anchor="middle">0</text>
<line x1="94.7" y1="196" x2="94.7" y2="199" stroke="#222" stroke-width="1"/>
<text x="94.7" y="211" text-anchor="middle">100</text>
<line x1="145.3" y1="196" x2="145.3" y2="199" stroke="#222" stroke-width="1"/>
<text x="145.3" y="211" text-anchor="middle">200</text>
<line x1="196" y1="196" x2="196" y2="199" stroke="#222" stroke-width="1"/>
<text x="196" y="211" text-anchor="middle">300</text>
<line x1="246.7" y1="196" x2="246.7" y2="199" stroke="#222" stroke-width="1"/>
<text x="246.7" y="211" text-anchor="middle">400</text>
<line x1="272" y1="196" x2="275" y2="196" stroke="#222" stroke-width="1"/>
<text x="278" y="200">0</text>
<line x1="272" y1="157.2" x2="275" y2="157.2" stroke="#222" stroke-width="1"/>
<text x="278" y="161.2">1M</text>
<line x1="272" y1="118.4" x2="275" y2="118.4" stroke="#222" stroke-width="1"/>
<text x="278" y="122.4">2M</text>
<line x1="272" y1="79.6" x2="275" y2="79.6" stroke="#222" stroke-width="1"/>
<text x="278" y="83.6">3M</text>
<line x1="272" y1="40.8" x2="275" y2="40.8" stroke="#222" stroke-width="1"/>
<text x="278" y="44.8">4M</text>
<rect x="44" y="14" width="228" height="182" fill="none" stroke="#222"/>
<text x="158" y="226" text-anchor="middle" font-size="11.5" font-weight="bold">Reconfiguration cost C (s)</text>
<text x="11" y="105" text-anchor="middle" font-size="11.5" font-weight="bold" transform="rotate(-90 11 105.0)">Break-even k<tspan dy="2.6" font-size="9.2">s</tspan><tspan dy="-2.6">* (W0 manifests)</tspan></text>
<text x="312" y="105" text-anchor="middle" font-size="11.5" font-weight="bold" transform="rotate(90 312 105.0)">Short-phase output tokens</text>
<line x1="44" y1="184.1" x2="272" y2="20.5" stroke="#222" stroke-width="1.3"/>
<line x1="44" y1="196" x2="272" y2="32.4" stroke="#222" stroke-width="1.3" stroke-dasharray="4,3"/>
<text x="152.9" y="96.6" text-anchor="end">O = 32.6 s</text>
<text x="226.4" y="79.1">O = 0</text>
<circle cx="256.3" cy="31.8" r="3.4" fill="#b3261e"/>
<circle cx="111.0" cy="136.1" r="3.4" fill="#555"/>
<circle cx="89.5" cy="151.5" r="3.4" fill="#555"/>
<circle cx="57.9" cy="174.1" r="3.4" fill="#000"/>
<circle cx="54.7" cy="176.5" r="3.4" fill="#000"/>
<text x="249.3" y="35.8" text-anchor="end" font-weight="bold" fill="#b3261e">vanilla 5.86</text>
<text x="119" y="151.1" font-weight="bold">Snap 2.14</text>
<text x="97.5" y="166.5" font-weight="bold">Cache 1.59</text>
<circle cx="57.9" cy="186.0" r="3.4" fill="#fff" stroke="#000" stroke-width="1.3"/>
<circle cx="256.3" cy="43.6" r="3.4" fill="#fff" stroke="#b3261e" stroke-width="1.3"/>
<line x1="56.9" y1="170.2" x2="54.1" y2="71.6" stroke="#000" stroke-width="0.8"/>
<text x="46" y="47.6" font-weight="bold">PhaseMorph</text>
<text x="46" y="60.6">0.70–0.78 (C 21.1–27.5 s)</text>
<line x1="60.9" y1="187" x2="99.7" y2="180.6" stroke="#000" stroke-width="0.8"/>
<text x="102.7" y="184.6">0.36 with O = 0</text>
</svg>
```),
  caption: [Short-phase break-even size $k_s^* = (C + O) \/ (a_"0H" - a_"0S")$, with $a_"0H" - a_"0S" = 77.0" s"$ per `W0` manifest (right axis: output tokens at 721,595 per manifest). Solid line: conservative tail drain ($O = 32.6" s"$); dashed line: tail-aware dispatch ($O = 0$). Points are the measured costs of @tab-segments and @tab-breakeven; PhaseMorph spans $C = 21.1"–"27.5" s"$.],
) <fig-breakeven>

== Comparison with In-Runtime Submesh Switching <sec-eval-submesh>

On the 16-chip ($4 times 4$) synthetic benchmark (1.61 GB/chip of `TP8` weights), PhaseMorph-Submesh transitions from `2×TP8` to `TP16` in 12.04 ms and reverses the transition in 48.02 ms, three orders of magnitude faster than process-isolated reconfiguration. The host-row submeshes also exhibit strong performance isolation: concurrent execution stays within $1.05"–"1.08 times$ of solo step latency, and background compilation of all `TP16` buckets (11.82 s) reduces foreground `TP8` throughput by only 11.8%–13.0%. Nevertheless, for multi-phase batch workloads on v5e slices, process-isolated reconfiguration remains the preferred design: `libtpu`'s limited-ICI-routing mode prevents per-instance submeshes from compiling within a single 32-chip ($4 times 8$) v5e runtime (@fig-fullmesh), and reducing warm $C$ from 21.1–27.5 s to $approx 9"–"10" s"$ improves median makespan reduction by only 0.3 percentage points (9.9% $arrow.r$ 10.2%). Process-isolated PhaseMorph thus captures $approx 97%$ of the achievable makespan improvement while preserving per-instance fault isolation.

= Related Work <sec-related>

== Serverless Cold-Start, Autoscaling, and State Materialization

Reducing instance startup latency has been widely studied in serverless and autoscaled GPU inference. ServerlessLLM @fu2024serverlessllm optimizes checkpoint loading through multi-tier local storage and pipelined DMA, BlitzScale @zhang2025blitzscale multicasts model parameters over RDMA from host caches and active peers during scale-out, and PipeSwitch @bai2020pipeswitch pipelines layer-granularity weight transfer with inference execution. On Cloud TPU, our `LoadProfiler` characterization demonstrates that storage and transfer optimizations alone are ineffective: file reads (0.18 s) and host-to-device DMA (2.0–3.0 s) constitute only 1.2% of `load_model` latency (237.0 s), whereas 98.8% is consumed by host-CPU tensor layout transformation ($approx 128" s"$), per-parameter sharding compilation ($75"–"77" s"$), and Flax abstract graph evaluation (14.6 s). PhaseMorph eliminates these bottlenecks by materializing post-conversion per-chip HBM shards and shape-only model graphs, supplemented by a 16-virtual-device CPU pre-read process for cold deployments.

Our compilation-path optimizations are most closely related to Medusa @zeng2025medusa and Foundry @liu2026foundry, which accelerate single-GPU serverless cold starts (reducing loading latency from 2.85 s to 1.67 s on 0.5B–14B models in Medusa) by materializing vLLM's KV-cache memory profile and captured CUDA graphs offline. Because CUDA graph nodes embed non-deterministic virtual memory pointers returned by `cudaMalloc` as well as randomized kernel entry addresses from closed-source libraries (e.g., internal cuBLAS symbols omitted from dynamic symbol tables), Medusa must intercept and replay the exact buffer allocation sequence via indirect index tables and execute first-layer "triggering kernels" to force driver module loading prior to pointer patching. On multi-host TPU slices, cold-start compilation overhead is two orders of magnitude larger (exceeding 80 s even with a warm persistent XLA cache, and 350–419 s end-to-end) because TPUs lack an eager execution mode and must trace, lower, and compile 187 SPMD bucket executables. Conversely, JAX/XLA's functional execution model permits a cleaner materialization abstraction: PJRT `LoadedExecutable` binaries accept buffer handles as invocation arguments rather than embedding raw virtual addresses, and all kernels are hermetically compiled by XLA. Once PhaseMorph disables sub-second persistent-cache eviction (retaining 248 auxiliary executables per worker) and records the exact `execution_devices` torus permutation, the Executable Bank restores all 187 multi-host executables across processes in 3.45 s without allocation replay or symbol-table inspection, while TPU-decoupled CPU pre-spawning overlaps worker initialization with active serving without contending for `/tmp/libtpu_lockfile`.

== Elastic and Fault-Tolerant Parallelism Reconfiguration

A complementary body of work adapts model parallelism dynamically in response to resource preemption, hardware faults, or workload shifts. SpotServe @miao2024spotserve reconfigures tensor and pipeline parallelism across preemptible GPU spot instances. FailSafe @xu2025failsafe maintains tensor-parallel serving on an 8$times$H100 node when 1–3 GPUs fail, combining cyclic KV-cache placement and hybrid TP/DP attention for non-power-of-two world sizes (e.g., `TP7`) with _Lightning Recovery_, which restores the lost 1/8th KV partition from proactive host-DRAM backups over PCIe (530 ms vs. 22 s for prefill recomputation) and reloads only missing parameter slices using FFN reduction commutativity and NVLink peer exchange (120 ms).

PhaseMorph addresses a complementary elasticity requirement that arises during fault-free operation on Cloud TPU slices: given 16 GB of HBM per chip and 2D torus ICI links, request context length alone induces a tension between `TP8` (28.8%–37.7% lower makespan on short contexts) and `TP16` ($2.75 times$ higher usable KV capacity, required for long contexts). Furthermore, the trade-off between KV state transfer and recomputation reverses between single-node NVLink GPUs and multi-host TPUs: when reconfiguring a 32-chip slice between `4×TP8` and `2×TP16`, staging 15.1 GB of KV state through host DRAM increases makespan by 3.6%–61.1% relative to parallel MXU re-prefill ($approx 98" ms"$ per 1,000 tokens) and introduces BF16 reduction-tree rounding drift (median 0.097 nats). PhaseMorph therefore re-partitions meshes at phase boundaries—or drains retained `TP8` instances in place during partial `S → M` transitions—while reducing multi-host engine transition latency from minutes to seconds.

DistServe @zhong2024distserve, Splitwise @patel2024splitwise, and Mooncake @qin2025mooncake disaggregate prefill and decode across dedicated instance pools, whereas Llumnix @sun2024llumnix migrates live requests across homogeneous instances. LoongServe @wu2024loongserve elastically scales sequence parallelism at iteration granularity within a unified GPU runtime, and Gyges @chen2025gyges transforms TP degrees across instances under interactive latency SLOs. PhaseMorph instead targets batch makespan on multi-host TPU slices and demonstrates that static heterogeneous partitions (`M`), even when paired with reservation-aware `llm-d` routing, are optimal in only 6% of the two-phase workload space.

== Accelerator Submesh Runtimes and Compilation Caching

Pathways @barham2022pathways coordinates computations across TPU islands via a single-controller dataflow runtime, Alpa @zheng2022alpa partitions device clusters into submeshes for pipeline-parallel stages, and TPU v4 @jouppi2023tpuv4 reconfigures optical circuit switches at job boundaries. Rather than fixing device assignments for the duration of a job, PhaseMorph re-partitions a serving slice across workload phases and characterizes both the 12.0 ms in-runtime submesh switching latency on $4 times 4$ topologies and the `libtpu` limited-ICI-routing restriction on $4 times 8$ v5e slices.

= Conclusion <sec-conclusion>

On multi-host Cloud TPU slices, no single static tensor-parallel partition simultaneously optimizes short-context compute utilization and long-context KV-cache capacity: smaller meshes (`TP8`) maximize MFU over the 2D torus and systolic arrays, whereas larger meshes (`TP16`) provide $2.75 times$ greater usable HBM capacity for long sequences. We have shown that the multi-minute latency of TPU instance autoscaling and mesh reconfiguration is an artifact of software-stack initialization rather than hardware bring-up. By leveraging JAX/XLA's functional executable representation and deferred device binding to materialize pre-sharded weights, shape-only model graphs, and device-order-preserving executable banks alongside TPU-decoupled worker pre-spawning, PhaseMorph reduces warm full-slice reconfiguration latency from 419.0 s to 21.1–27.5 s, partial reconfiguration to 18.8 s, and cold first-time transitions to 134.1 s, outperforming static partitioning across 82%–88% of two-phase batch workloads.
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
@article{xu2025failsafe,
  title = {{FailSafe}: High-performance Resilient Serving},
  author = {Xu, Ziyi and Xie, Zhiqiang and Gandhi, Swapnil and Kozyrakis, Christos},
  journal = {arXiv preprint arXiv:2511.14116},
  year = {2025}
}
@inproceedings{miao2024spotserve,
  title = {{SpotServe}: Serving Generative Large Language Models on Preemptible Instances},
  author = {Miao, Xupeng and Shi, Chunan and Duan, Jiangfei and Zhang, Xiaoxi and Lin, Bin and Cui, Bin and Jia, Zhihao},
  booktitle = {Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)},
  year = {2024}
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
