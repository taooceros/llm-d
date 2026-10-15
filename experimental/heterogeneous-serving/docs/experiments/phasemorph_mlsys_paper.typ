// PhaseMorph: Sub-30-Second TPU Mesh Reconfiguration for Heterogeneous Bulk LLM Serving
// Formatted as a two-column MLSys conference paper for PDF/paged view and responsive HTML for ?g-output=html.
//   HTML: typst compile --features html --format html phasemorph_mlsys_paper.typ /tmp/phasemorph.html
//   PDF:  typst compile phasemorph_mlsys_paper.typ /tmp/phasemorph.pdf

#let title = "PhaseMorph: Sub-30-Second TPU Mesh Reconfiguration for Heterogeneous Bulk LLM Serving"
#let subtitle = "MLSys Paper Draft · Cloud TPU v5e 4×8 (32 Chips) · Gemma-4 31B (BF16)"
#set document(title: title, author: "Hongtao Zhang")
#set heading(numbering: "1.")
#show table.cell.where(y: 0): strong

#let css = ```
body {
  max-width: 62rem;
  margin: 2.2rem auto;
  padding: 0 1.25rem;
  line-height: 1.58;
  font-family: "Libertinus Serif", Georgia, Cambria, "Times New Roman", Times, serif;
  color: #1f2328;
}
h1, h2, h3, h4 {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.25;
  margin-top: 1.6em;
}
h1 { font-size: 1.75rem; text-align: center; margin-bottom: 0.3em; }
h2 { font-size: 1.25rem; border-bottom: 1px solid #d0d7de; padding-bottom: 0.2em; }
h3 { font-size: 1.06rem; }
.paper-meta {
  text-align: center;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: #59636e;
  margin-bottom: 1.4rem;
  font-size: 0.95rem;
}
.abstract-box {
  background: #f6f8fa;
  border-top: 2px solid #1f2328;
  border-bottom: 2px solid #1f2328;
  padding: 0.9rem 1.2rem;
  margin: 1.2rem 0 1.8rem 0;
  font-size: 0.95rem;
}
.abstract-title {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-weight: 700;
  text-align: center;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.85rem;
  margin-bottom: 0.4rem;
}
table {
  border-collapse: collapse;
  margin: 1em auto;
  font-size: 0.88em;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  display: block;
  overflow-x: auto;
}
th, td { border: 1px solid #d0d7de; padding: 0.35em 0.65em; vertical-align: top; }
th { background: #f6f8fa; text-align: left; }
tr:nth-child(even) td { background: #fbfcfd; }
code {
  background: #eff1f3;
  padding: 0.1em 0.3em;
  border-radius: 3px;
  font-size: 0.88em;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
pre {
  background: #f6f8fa;
  padding: 0.75em 1em;
  border-radius: 5px;
  overflow-x: auto;
  font-size: 0.85em;
}
pre code { background: none; padding: 0; }
figure { margin: 1.4em 0; text-align: center; }
figure img, svg.typst-frame { width: 100% !important; height: auto !important; }
figcaption {
  font-size: 0.88em;
  color: #424a53;
  margin-top: 0.45em;
  text-align: left;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.note {
  background: #f6f8fa;
  border-left: 4px solid #0969da;
  padding: 0.55em 0.9em;
  margin: 0.9em 0;
  font-size: 0.92em;
}
```.text

#let is-html() = "target" in dictionary(std) and std.target() == "html"

#show math.equation: it => context if is-html() {
  if it.block { html.elem("div", attrs: (style: "text-align:center;margin:.7em 0"), html.frame(it)) }
  else { box(html.frame(it)) }
} else { it }

#let svg-fig(source) = image(bytes(source.text), format: "svg", width: 100%)
#let wide-fig(body, caption: none) = figure(body, caption: caption, placement: top, scope: "parent")

#let abstract-content = [
  Serving large language models (LLMs) on multi-host Cloud TPU slices exposes a fundamental hardware tension between *compute utilization (MFU)* and *usable HBM KV capacity*. On a 32-chip Cloud TPU v5e slice (`4×8` 2D torus across 8 hosts) serving Gemma-4 31B in native BF16, partitioning the slice into four 8-chip instances (`S: 4×TP8`) yields *28.8%–37.7% higher throughput* on short-context chat workloads (`W0`) than two 16-chip instances (`H: 2×TP16`), owing to fewer Ring AllReduce synchronization hops (14 vs. 30) and deeper MXU systolic contractions. Conversely, each `TP8` instance retains only `52,864` tokens of usable KV cache after model weights: on mid-context document workloads (`W7`, `16.4k–43.1k` tokens), `H` outperforms `S` by *17.6%* due to $3 times$ higher resident batch concurrency and halved per-chip KV bandwidth traffic, while on long-context workloads (`W6`, `53.3k–128.3k` tokens), `S` is physically infeasible.

  Neither static heterogeneous partitioning (`M: TP16 + 2×TP8`) nor live cross-TP KV migration resolves this tension across general multi-phase backlogs: static mixed layouts suffer from phase-imbalance stragglers outside a narrow *6%* workload band, and host-staged KV transfer is slower than parallel MXU prefix recomputation. Dynamic mesh reconfiguration—reshaping the TPU slice between `S`, `M`, and `H` as the active workload shifts—is the natural remedy, but naive engine redeployment incurs a prohibitive *350–419 s* outage.

  We present *PhaseMorph*, a reconfiguration system for vLLM on Cloud TPU that eliminates *95%* of this software-stack overhead. By combining (1) `/dev/shm` pre-sharded weight snapshots and shape-only graph construction (`FastRestore`, cutting `load_model` from 237.0 s to 5.0 s), (2) zero-threshold persistent XLA compilation caching (100% cache hits across 4,416 executables, saving 42.4 s), and (3) *clean-room CPU worker pre-spawning* paired with a *serialized executable bank* (bypassing `/tmp/libtpu_lockfile`, kernel VFIO reset races, and 28.6 s of in-process JAX tracing), PhaseMorph compresses warm full-slice `S → H` reconfiguration to *21.1–27.5 s* and partial `S → M` reconfiguration to *18.8 s* with zero foreground slowdown. For cold first switches without pre-existing snapshots, a background CPU pre-read subprocess cuts switching latency by *40%* (222.6 s $arrow.r$ 134.1 s). Finally, we build a unified 16-chip JAX submesh prototype that achieves *12.0 ms* in-HBM ICI resharding after bypassing a `libtpu` sub-torus routing bug, and show why process-isolated 21 s switching captures *99.7%* of achievable bulk makespan gains while preserving multi-host fault isolation. Across a $(k_s, k_l)$ short/long workload grid, PhaseMorph wins *82%–88%* of the job space with median and maximum makespan reductions of *9.9%* and *26.8%* over the best static layout.
]

#show: doc => context if is-html() {
  html.elem("style", css)
  html.elem("h1", title)
  html.elem("div", attrs: (class: "paper-meta"))[
    Hongtao Zhang · Heterogeneous & Reconfigurable TPU Serving Study (`taooceros/llm-d`) · #subtitle
  ]
  html.elem("div", attrs: (class: "abstract-box"), {
    html.elem("div", attrs: (class: "abstract-title"))[Abstract]
    abstract-content
  })
  doc
} else {
  set page(
    paper: "us-letter",
    margin: (x: 1.65cm, y: 1.85cm),
    columns: 2,
    numbering: "1",
  )
  set par(justify: true, leading: 0.55em)
  set text(size: 9.2pt)
  show heading: set text(size: 10.2pt)
  show heading.where(level: 1): set block(above: 1.1em, below: 0.55em)
  show heading.where(level: 2): set block(above: 0.85em, below: 0.45em)
  show table: set text(size: 7.8pt)
  place(
    top + center,
    scope: "parent",
    float: true,
    block(width: 100%, below: 1.1em)[
      #align(center)[
        #text(size: 14.5pt, weight: "bold", title)
        #v(4pt)
        #text(size: 9.5pt, weight: "medium")[Hongtao Zhang · Cloud TPU Heterogeneous Serving Systems Study]
        #v(2pt)
        #text(size: 8.5pt, fill: luma(90))[#subtitle]
      ]
      #v(6pt)
      #block(
        fill: luma(246),
        stroke: (top: 0.8pt + black, bottom: 0.8pt + black),
        inset: (x: 10pt, y: 7pt),
        width: 100%,
      )[
        #align(center)[#text(size: 8.8pt, weight: "bold")[ABSTRACT]]
        #v(2pt)
        #text(size: 8.5pt)[#abstract-content]
      ]
    ],
  )
  doc
}

= Introduction <sec-intro>

Bulk and offline LLM inference—spanning synthetic data generation, long-document summarization, repository-wide code analysis, and model evaluation—operates over deep, heterogeneous request backlogs. Unlike interactive serving governed by per-request tail latency (TTFT and TPOT), bulk inference minimizes *end-to-end dispatch-to-drained makespan* (equivalently, maximizing sustained useful output tokens per second across a fixed accelerator allocation).

On tightly coupled accelerator pods such as Google Cloud TPU v5e, operators must choose a tensor-parallel (TP) degree that partitions a multi-host slice into serving instances. Our characterization of Gemma-4 31B (`61.4 GB` BF16 parameters) on a 32-chip TPU v5e slice (`8` hosts $times$ `4` chips/host, `16 GB` HBM per chip) reveals that *no single static partition can efficiently serve a backlog containing both short and long contexts*:

1. *Small meshes (`S: 4×TP8`) dominate short-context throughput:* At sequence lengths $lt.eq 16"k"$ (`W0` ShareGPT chat), four `TP=8` instances (`2×4` sub-torus each) drain a 1,763-request manifest (`721,595` output tokens) in *190.8 s*, beating two `TP=16` instances (`H: 2×TP16`, 267.8–306.4 s) by *28.8%–37.7%*. Doubling TP from 8 to 16 doubles Ring AllReduce synchronization hops ($2(N-1): 14 arrow.r 30$), halves the MXU systolic contraction depth ($K = 4096 / "TP": 512 arrow.r 256$), and introduces a 4.5% FFN tile-padding bubble (`1344 → 1408`).
2. *Large meshes (`H: 2×TP16`) are required for mid-to-long contexts:* On an 8-chip `TP=8` instance (`128 GB` aggregate HBM), `61.4 GB` of weights plus XLA state leave only *52,864 tokens* of usable KV cache (`826` blocks $times$ `64` tokens). On a 16-chip `TP=16` instance (`256 GB` HBM), weight shards halve per chip and usable KV capacity nearly triples to *145,216 tokens* ($2.75 times$). Consequently, on mid-length document workloads (`W7`, `16.4k–43.1k` context), `H` fits 6 concurrent requests per instance versus 2 on `S` and finishes *17.6% faster* (204.8 s vs. 248.4 s). On long-document workloads (`W6`, `53.3k–128.3k` context), `S` cannot admit a single request.
3. *Scaling to `TP=32` (`1×TP32`) is counterproductive:* Because Gemma-4 31B has 16 KV heads, `TP=32` replicates KV heads across chip pairs, increasing usable KV capacity by only *+8.9%* over `TP=16` while nearly halving slice throughput (671.5 s vs. 347.1 s on `W4`).

A natural alternative is a *static heterogeneous layout* (`M: 1×TP16 + 2×TP8`) behind a cluster router (`llm-d`), optionally paired with live request migration. We evaluate both in @sec-static and find strict structural limits. First, without explicit KV capacity reservation on the `TP16` instance, short requests flood the large pool and head-of-line block long documents, making `M` *30.3% slower* than homogeneous `H`. Even after introducing a *work-conserving fallback + long-reservation* routing policy that makes `M` a *10.1% winner* on a matched mixed workload (`W4`), static `M` remains optimal in only *6%* of the general $(k_s, k_l)$ backlog space—whenever short and long phases do not drain simultaneously, half the slice sits idle. Second, migrating growing requests across TP degrees at runtime underperforms simple token-history recomputation due to multi-gigabyte host-staged KV payloads and cross-TP reduction rounding drift.

Instead of moving stateful requests across fixed meshes, *PhaseMorph reconfigures the TPU meshes to match the active workload phase*—starting in `S` to drain short/medium backlogs at peak MFU, then transitioning into `M` (partial switch, keeping two `TP8` engines live) or `H` (full switch) for long-context phases. Historically, mesh reconfiguration has been dismissed as impractical on TPUs because restarting vLLM engines across a slice takes *350–419 s* ($approx 7$ minutes), requiring over $2.0"M"$ output tokens just to break even.

We show that *95% of this reconfiguration delay is avoidable software-stack overhead* rather than physical hardware initialization or weight transfer (@fig-waterfall). Through a sequence of profiling-driven interventions across vLLM, `tpu-inference`, JAX, and Ray, PhaseMorph reduces warm full-slice reconfiguration to *21.1–27.5 s* (partial reconfiguration to *18.8 s*), cold first-switch reconfiguration to *134.1 s*, and in-runtime ICI resharding to *12.0 ms*.

#wide-fig(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 314" font-family="Helvetica, Arial, sans-serif" font-size="11.5" fill="#1f2328">
  <text x="12" y="16" font-size="12.5" font-weight="bold">Critical-Path Reconfiguration Cost C Across Engineering Cuts (Measured on 32-Chip TPU v5e)</text>
  <line x1="215" y1="36" x2="800" y2="36" stroke="#d0d7de" stroke-width="1"/>
  <text x="215" y="32" text-anchor="middle" font-size="10" fill="#59636e">0 s</text>
  <text x="349" y="32" text-anchor="middle" font-size="10" fill="#59636e">100 s</text>
  <text x="483" y="32" text-anchor="middle" font-size="10" fill="#59636e">200 s</text>
  <text x="617" y="32" text-anchor="middle" font-size="10" fill="#59636e">300 s</text>
  <text x="751" y="32" text-anchor="middle" font-size="10" fill="#59636e">400 s</text>
  <!-- Row 1: Baseline -->
  <text x="205" y="56" text-anchor="end" font-weight="bold">1. Baseline Full Redeploy</text>
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

  <!-- Row 2: P2 FastRestore -->
  <text x="205" y="98" text-anchor="end" font-weight="bold">2. Cut 1: FastRestore (P2)</text>
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

  <!-- Row 3: O0 Cache 0/0 -->
  <text x="205" y="140" text-anchor="end" font-weight="bold">3. Cut 2: 0/0 XLA Cache (O0)</text>
  <text x="205" y="153" text-anchor="end" font-size="10" fill="#59636e">100% cache hits (4,416/4,416)</text>
  <rect x="215" y="126" width="21" height="28" fill="#8c959f" stroke="#fff"/>
  <rect x="236" y="126" width="29" height="28" fill="#0969da" stroke="#fff"/>
  <text x="250" y="144" text-anchor="middle" fill="#fff" font-size="9.5">22s</text>
  <rect x="265" y="126" width="10" height="28" fill="#1a7f37" stroke="#fff"/>
  <rect x="275" y="126" width="60" height="28" fill="#8250df" stroke="#fff"/>
  <text x="305" y="144" text-anchor="middle" fill="#fff" font-size="9.5">Host Trace 35s</text>
  <text x="343" y="144" font-weight="bold" fill="#1f2328">89.8 s (-79%)</text>

  <!-- Row 4: O3 / F2 Pre-spawn + Bank -->
  <text x="205" y="182" text-anchor="end" font-weight="bold">4. Cut 3: Pre-Spawn + Bank</text>
  <text x="205" y="195" text-anchor="end" font-size="10" fill="#59636e">O3 warm (27.5s) / F2 warm (21.1s)</text>
  <rect x="215" y="168" width="9" height="28" fill="#8c959f" stroke="#fff"/>
  <rect x="224" y="168" width="10" height="28" fill="#1a7f37" stroke="#fff"/>
  <rect x="234" y="168" width="10" height="28" fill="#2da44e" stroke="#fff"/>
  <text x="252" y="186" font-weight="bold" fill="#1a7f37">21.1–27.5 s full / 18.8 s partial (-95%)</text>

  <!-- Row 5: F3c Cold First Switch -->
  <text x="205" y="224" text-anchor="end" font-weight="bold">5. Cold 1st Switch + Pre-Read</text>
  <text x="205" y="237" text-anchor="end" font-size="10" fill="#59636e">F3c CPU child converts during S</text>
  <rect x="215" y="210" width="9" height="28" fill="#8c959f" stroke="#fff"/>
  <rect x="224" y="210" width="114" height="28" fill="#e16f24" stroke="#fff"/>
  <text x="281" y="228" text-anchor="middle" fill="#fff" font-size="9.5">Wait 80s + 5s restore</text>
  <rect x="338" y="210" width="56" height="28" fill="#8250df" stroke="#fff"/>
  <text x="366" y="228" text-anchor="middle" fill="#fff" font-size="9.5">38s</text>
  <text x="402" y="228" font-weight="bold" fill="#1f2328">134.1 s (vs 222.6 s no pre-read, -40%)</text>

  <!-- Row 6: U2 Unified Runtime -->
  <text x="205" y="266" text-anchor="end" font-weight="bold">6. Unified Runtime (U2)</text>
  <text x="205" y="279" text-anchor="end" font-size="10" fill="#59636e">In-HBM submesh slice / ICI</text>
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
  caption: [Waterfall decomposition of TPU mesh reconfiguration cost $C$ (`S → H` full-slice switch unless noted) across successive engineering cuts. Stripping software-stack overheads compresses production vLLM switching from 419.0 s to 21.1–27.5 s (18.8 s partial), cold first-switch provisioning from 222.6 s to 134.1 s, and in-runtime submesh switching to 12.0 ms.],
) <fig-waterfall>

= Background & Motivation: The Topology–Capacity Dilemma <sec-motivation>

== Hardware & Model Target
All experiments execute on an 8-host, 32-chip *Cloud TPU v5e* slice configured as a $4 times 8$ 2D torus with wrap-around Inter-Chip Interconnect (ICI) links. Each host manages 4 TPU v5e chips (`16 GB` HBM2e at `819 GB/s` bandwidth, `197 TFLOPS` BF16 peak compute across four $128 times 128$ MXUs, and `1.00 MB` of on-chip Scalar Memory `SMEM` per TensorCore). We serve *Gemma-4 31B* in native BF16 (`48` layers, `hidden_size=4096`, `ffn_intermediate=21504`, `32` query heads, `16` KV heads with GQA $2:1$, `head_dim=128`, total BF16 parameter footprint `61.4 GB`).

== Why Small Meshes (`TP=8`) Win on Short/Medium Compute
Increasing tensor parallelism across TPU v5e hosts degrades per-chip compute utilization (MFU) through three hardware mechanisms:

1. *Ring AllReduce Hop Latency:* Each Transformer layer executes two AllReduce collectives (post-attention O-Proj and post-FFN Down-Proj), totaling 96 synchronizations per step. On a ring of $N$ chips, synchronization requires $2(N-1)$ routing steps: moving from `TP=4` (1 host, 6 steps) to `TP=8` (2 hosts, 14 steps) and `TP=16` (4 hosts, 30 steps) introduces a $approx 20" ms"$ inter-host collective floor at small sequence buckets ($S lt.eq 1024$).
2. *MXU Contraction Depth Halving:* TPU v5e MXUs require contraction depths $K gt.double 128$ to amortize systolic pipeline fill/drain. For O-Proj and Down-Proj, the per-chip reduction dimension $K = d_"model" / "TP"$ drops from 512 (`TP=8`) to 256 (`TP=16`, only two $128 times 128$ tiles deep).
3. *FFN Tile Padding Bubble:* Gemma-4 31B's intermediate size $I = 21,504$ shards cleanly by 128 at `TP=8` ($21504 / 8 = 2688 = 21 times 128$), but at `TP=16` yields $1344 = 10.5 times 128$, forcing XLA to zero-pad every FFN weight matrix to 1408 ($+4.5%$ wasted FLOPs).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, right, right, right, right),
    [Topology (16-Chip Basis)], [Chips/Inst], [16k Layer (ms)], [Inst Tok/s], [MFU (%)], [16-Chip Tok/s],
    [`4× TP4 (DP=4)`], [4 (1 host)], [25.19], [13,551], [68.16%], [54,205],
    [`2× TP8 (DP=2)`], [8 (2 hosts)], [16.85], [20,261], [50.95%], [40,521],
    [`1× TP16 (DP=1)`], [16 (4 hosts)], [13.99], [24,394], [30.67%], [24,394],
  ),
  caption: [Measured Gemma-4 31B scaling across 16 TPU v5e chips at the 16,384-token compilation bucket. Smaller TP degrees achieve up to $2.22 times$ higher aggregate throughput per chip.],
) <tab-topology>

== Why Large Meshes (`TP=16`) Win on Long Contexts
Despite lower short-context MFU (@tab-topology), `TP=16` becomes superior—and eventually mandatory—as context length grows:

1. *Super-Linear Usable KV Capacity Scaling:* Model weights (`61.4 GB`) are fixed across the TP group. On `TP=8` (`128 GB` HBM), weights plus XLA scratchpad leave *52,864 usable KV tokens* (`826` blocks $times$ `64` tokens). On `TP=16` (`256 GB` HBM), per-chip weight shards halve from `7.68 GB` to `3.84 GB`, expanding usable KV capacity to *145,216 tokens* ($2.75 times$ per instance, or $1.37 times$ per chip).
2. *Decode KV Bandwidth & GQA Sharding:* In our 60-cell sweep of Pallas `ragged_paged_attention` v3 (`RPAd`), decode step latency is flat at 15–18 ms for batch sizes $B lt.eq 32$ ($"MBU" < 11%$), then hits the HBM bandwidth wall at $B gt.eq 64$. At $B in {1, 2}$, `TP=16` is 2%–5% slower than `TP=8` because 16 KV heads shard to a single KV head per chip (`512 B/tok/chip`), underutilizing HBM burst length. However, for $L gt.eq 4096$ or $B gt.eq 128$, `TP=16` halves per-chip KV traffic and beats `TP=8` by *$1.19"–"1.22 times$*.
3. *The `1.0 MB SMEM` Hardware Boundary:* Pallas `RPAd` prefetches the entire `int32` page table (`page_indices`) into TPU v5e's `1.00 MB` Scalar Memory:
   $ "Bytes"_"SMEM"(B, L_"max") = B times ceil(L_"max" / P) times 4 "bytes" $
   For page size $P = 16$, whenever $sum_(i=1)^B L_i > 3.5 times 10^6$ tokens (e.g., $B=512, L=8192$), XLA compilation aborts with `RESOURCE_EXHAUSTED: Used 1.01M of 1.00M smem` even when HBM KV usage is only `2.15 GB`. Larger block sizes (`64` or `512` tokens) and `TP=16` sharding are required to serve long contexts (`W6`, `W7`), producing the regime inversion in @tab-workloads.

#wide-fig(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, left, right, right, right, left),
    [Workload], [Reqs], [Out Toks], [Context Range], [`S (4×TP8)` (s)], [`H (2×TP16)` (s)], [`M (TP16+2×TP8)` (s)], [Static Winner],
    [`W0` (ShareGPT Chat)], [1,763], [721,595], [$lt.eq 16"k"$ (med 502)], [*190.8*], [267.8 (at 131k) / 306.4 (at 16k)], [352.7 (2×TP8)], [*S* (`-28.8%` vs. H)],
    [`W4` (24 Long + 880 Chat)], [904], [387,975], [$27.1"k–"36.5"k"$], [*147.7*], [187.4], [—], [*S* (`-21.2%` vs. H)],
    [`W5` (TED Translation)], [250], [982,899], [med $P 3.5"k", D 4.0"k"$], [*754.3*], [774.4], [—], [*S* (`-2.6%` vs. H)],
    [`W7` (GovReport Mid)], [120], [79,531], [$16.4"k–"43.1"k"$ (med 21.8k)], [248.4], [*204.8*], [—], [*H* (`-17.6%` vs. S)],
    [`W6` (GovReport Long)], [40], [26,951], [$53.3"k–"128.3"k"$ (med 64.3k)], [Infeasible (>52.9k)], [*206.2*], [409.2 (1×TP16)], [*H* (`S` infeasible)],
  ),
  caption: [Measured Full-BF16 makespans across the 32-chip TPU v5e slice (`R3`, `L1`, `P4`, `P5`). `S` dominates short/medium workloads (`W0`, `W4`, `W5`), whereas `H` wins by 17.6% on mid-length `W7` (6 resident requests/TP16 vs. 2/TP8) and is mandatory on long-context `W6`.],
) <tab-workloads>

= Limits of Static Heterogeneity & Live Migration <sec-static>

Before reconfiguring meshes at runtime, we evaluate whether a *fixed heterogeneous layout (`M: 1×TP16 + 2×TP8`)* or *live cross-TP request migration* can bridge the short/long divide.

== Static Heterogeneity Requires Reservation-Aware Routing
Consider the mixed workload `W4` (880 short chats + 24 long GovReport documents of `27.1k–36.5k` tokens) in a capacity-constrained regime where `TP8` instances hold `26,368` KV tokens (cannot admit the 24 long documents) and `TP16` instances hold `72,192` tokens (`w4_longdoc_heterogeneity.typ`).

Under the default `plain` router policy (`work_conserving_fallback=off`, `long_reserve_tokens=0`), `M` finishes in *452.2 s*—*30.3% slower* than homogeneous `H` (347.1 s). Early short chats spill onto `TP16` and consume its KV blocks, starving the 24 long documents while the two `TP8` engines drain their short queues at $t = 210" s"$ and sit idle for the remaining 242 s.

We resolve this head-of-line blocking in both our `ExtProc` gateway (`hetero/gateway/policy.py`) and the upstream `llm-d-router` Endpoint Picker (`deploy/epp/reserve.yaml`, `latebinding.yaml`) by pairing *work-conserving bidirectional spill* with a *long-request KV reservation floor* (`long_reserve_tokens = 36,864`): a short request may spill onto `TP16` only if $"FreeKV"_"TP16" - "Cost"(r_"short") gt.eq 36,864$, preserving one full-length document slot whenever long documents remain unserved. With reservation enabled, `M` finishes in *311.9 s* (314.7 s and 309.0 s across reps; 317.8 s via `llm-d` EPP `latebinding`), beating `H` by *10.1%* (@tab-w4).

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    [Layout & Gateway Policy (`W4`)], [Mean (s)], [Tok/s], [vs. `H`],
    [`H (2×TP16)`, `plain`], [347.1], [1,118], [Baseline],
    [`T32 (1×TP32)`, `plain`], [671.5], [578], [`+93.5%`],
    [`M (TP16+2×TP8)`, `plain`], [452.2], [858], [`+30.3%`],
    [`M`, `reserve + fallback` (ExtProc)], [*311.9*], [*1,245*], [*`-10.1%`*],
    [`M`, `llm-d` EPP `latebinding`], [*317.8*], [*1,221*], [*`-8.4%`*],
  ),
  caption: [Measured `W4` makespan across static layouts and routing policies. Capacity reservation turns `M` from a 30.3% regression into a 10.1% improvement.],
) <tab-w4>

== Why Static `M` and Live Migration Fall Short
Despite winning when short and long queues drain simultaneously, static `M` and live migration have three fundamental limits:
1. *Narrow Phase-Balance Band:* Under Full BF16, `M`'s two `TP8` instances drain `W0` at $352.7" s/manifest"$, while its single `TP16` drains `W6` at $409.2" s/manifest"$. Because `TP8` cannot execute `W6` ($>52.9"k"$ tokens), `M`'s makespan is $max(352.7 k_s, 409.2 k_l)$, which is optimal in only *6%* of the $(k_s, k_l)$ phase space (@sec-eval).
2. *Host-KV Transfer vs. Recomputation:* Across a 102-run migration evaluation (`0f5e4c095b0e`), enabling live `TP8 → TP16` migration degraded makespan by $+3.6%$ to $+61.1%$ on `W0–W2`. At matched mechanism checkpoints, staging a request's `15.1 GB` KV state through host DRAM (`Host-KV`) was slower than re-dispatching the token history for parallel MXU re-prefill ($approx 98" ms per 1,000 tokens"$).
3. *Cross-TP Reduction Non-Associativity:* Summing partial dot products across 8 vs. 16 shards in 96 AllReduce operations per step introduces BF16 rounding drift (median $0.097" nats"$ across 44 fixed-prefix probes, with near-tie outliers up to $8.16" nats"$), whereas serving each request within a uniform phase layout guarantees bitwise replica determinism.

= Deconstructing & Compressing Reconfiguration ($419" s" arrow.r 21.1" s"$) <sec-cuts>

We define the *production reconfiguration cost $C$* as the wall-clock time lost between the final source-phase request completion (`T0`) and steady-state target-phase serving (`dispatch_gap + ramp_deficit`). In our baseline deployment (`reconfiguration_breakeven.typ`), switching `S ↔ H` costs a median of *$C = 419.0" s"$* (412.9 s gap + 6.1 s ramp). We apply three architectural cuts to eliminate 95% of this cost.

== Cut 1: Disproving the Disk-I/O Myth via `FastSnapshot` and `FastGraph` ($419.0" s" arrow.r 132.2" s"$)
In the baseline 355 s engine initialization, `load_model` consumes *237.0 s* (67%). Staging the `61.4 GB` safetensors checkpoint into `/dev/shm` tmpfs yielded *zero speedup* (237.6 s). Instrumentation in `LoadProfiler` (`tpu_inference/models/jax/utils/weight_utils.py`) revealed why:
- *Disk iteration & file read:* 0.16 s; *H2D DMA copy:* 2.76 s.
- *Host-CPU tensor transpose, reshape & dtype conversion:* *128.4 s*.
- *Load-time JIT compilation of parameter shardings:* *75.3 s*.
- *Flax NNX graph construction (`nnx.eval_shape`):* *14.6 s*.

*Mechanism (`hetero/weight_snapshot.py`, `hetero/restore_graph.py`):*
1. *Pre-Sharded `/dev/shm` Snapshots (`FastSnapshot`):* Each worker serializes its post-conversion per-chip HBM shards directly to `/dev/shm/omp_weights/<layout>/` (`30.7 GB`/host for `TP8`, `15.4 GB`/host for `TP16`). On subsequent switches, workers bypass safetensors parsing and host conversion, calling `jax.make_array_from_single_device_arrays` on raw `/dev/shm` buffers.
2. *Shape-Only Graph Construction (`FastGraph`):* Bypassing redundant `nnx.eval_shape` checks shrinks model graph construction from $14.6" s" arrow.r bold(1.8" s")$, cutting total `load_model` from $237.0" s" arrow.r bold(5.0" s")$ ($47 times$ speedup).
3. *In-Process Switch Controller (`hetero/switch.py`):* Replacing external shell-level Ray job submissions with an in-process controller (`drain → release → deploy → gateway_up`) removes $approx 120" s"$ of harness overhead, bringing full `S → H` $C$ to *132.2 s* (`P2`) and partial `S → Mb` $C_"eff"$ to *128.9 s* (`P3`, with only a *2.08 s* pause on the two kept `TP8` instances).

== Cut 2: Eliminating Sub-Second Persistent Cache Misses ($132.2" s" arrow.r 89.8" s"$)
Even with a warm persistent XLA cache, `P2` spent *25.9 s* in `initialize_from_config` and *54.6 s* in `compile_or_warm_up_model`. Per-worker compile logs revealed that persistent cache hit rates were only *2.6%* (`48/1,848`) and *6.0%* (`264/4,416`).

*Root Cause & Fix (`O0`, commit `dbc5513`):* JAX 0.10.2 defaults to `jax_persistent_cache_min_compile_time_secs = 1.0`, silently discarding $approx 248$ small helper, sampler, and KV-allocation programs per worker that each compile in 50–800 ms but cumulatively cost *42.4 s* on every restart. Setting `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` and `MIN_ENTRY_SIZE_BYTES=0` in `hetero/layout.py` raised persistent cache hits to *100.0%* (`1,848/1,848` and `4,416/4,416` with 0 backend recompiles), dropping `initialize_from_config` from $25.9" s" arrow.r bold(2.6" s")$ and `compile_or_warm_up_model` from $54.6" s" arrow.r bold(35.2" s")$, reducing full `S → H` $C$ to *89.8 s* (`87.3 s` partial).

== Cut 3: Clean-Room CPU Pre-Spawning & Serialized Executable Banks ($89.8" s" arrow.r 21.1"–"27.5" s"$)
After `O0`, two host-CPU bottlenecks accounted for 68% of the remaining 89.8 s switch cost: (1) Ray worker spawn + Python/vLLM imports (*32.7 s*), and (2) in-process JAX Python tracing and StableHLO lowering (*28.6 s* on `H`, *45.5 s* on `S`), which runs on the main thread before persistent-cache lookup because JAX's `_pjit_lower` trace cache is strictly process-local.

*Why Naive Background Compilation Failed (`compilation_overlap_failure.typ`):* Launching a second process on each host with `JAX_PLATFORMS=cpu` and `get_topology_desc(platform="tpu")` immediately crashed with `ABORTED: The TPU is already in use ... /tmp/libtpu_lockfile`, because querying TPU topology descriptors unconditionally initializes the PJRT TPU plugin.

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
  <text x="138" y="127" text-anchor="end" font-size="9.5" fill="#59636e">(hetero/prespawn.py)</text>
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

  <!-- Lane 3: Optional Cold Pre-Read Subprocess (F3) -->
  <text x="138" y="168" text-anchor="end" font-weight="bold">CPU Pre-Read Child</text>
  <text x="138" y="181" text-anchor="end" font-size="9.5" fill="#59636e">(F3 cold 1st switch only)</text>
  <rect x="195" y="151" width="315" height="34" rx="4" fill="#fff8c5" stroke="#9a6700"/>
  <text x="352" y="166" text-anchor="middle" fill="#9a6700" font-weight="bold">JAX_PLATFORMS=cpu (nice 10, 16 fake host devs)</text>
  <text x="352" y="179" text-anchor="middle" font-size="9" fill="#9a6700">Streams load_weights → 32,896 shards to /dev/shm (absorbs 165s before T0)</text>
  <rect x="510" y="151" width="95" height="34" rx="3" fill="#fff8c5" stroke="#9a6700" stroke-dasharray="3,2"/>
  <text x="557" y="172" text-anchor="middle" font-size="9.5" fill="#9a6700">Residual 79.6s</text>
</svg>
```),
  caption: [PhaseMorph overlap architecture (@sec-cuts). Target `H` workers spawn in CPU-only bundles during Phase 1, verify zero `libtpu`/`/dev/vfio` handles via `/proc/self`, and park on `ReleaseGate`. After `T0`, workers poll `/dev/vfio/<N>` to clear kernel reset `EBUSY`, DMA pre-sharded weights from `/dev/shm` (`4.9 s`), and deserialize 187 AOT executables from the Executable Bank (`3.45 s`).],
) <fig-architecture>

*Our Solution (`O1–O3`, `hetero/prespawn.py`, `hetero/exec_bank.py`, commit `d510c36`, @fig-architecture):*
1. *Clean-Room CPU Pre-Spawning & Kernel VFIO Readiness Guard:* While `S` serves Phase 1, the controller spawns target `H` workers in CPU-only Ray bundles (`{"CPU": 1.0, "node:<ip>": 0.001}`, patching `vllm/v1/executor/ray_utils.py` so CPU-only bundles are retained). Each worker imports vLLM/JAX, builds its tokenizer (`34.3 s`), audits `/proc/self/maps` and `/proc/self/fd` to assert zero `libtpu.so`, `/dev/vfio/*`, or lockfile handles, and parks on an async `ReleaseGate` inside `TPUWorker.init_device()`. At `T0`, after source workers exit, the Linux kernel takes *0.42–0.65 s* to reset `/dev/vfio/0..3`; unparked workers poll `os.open("/dev/vfio/<group>", os.O_RDWR)` every 20 ms before calling `jax.devices()`, preventing `EBUSY` crashes (`O2`).
2. *Serialized Executable Bank (`hetero/exec_bank.py`):* PhaseMorph serializes all 187 precompiled AOT `Compiled` objects per rank via `jax.experimental.serialize_executable` alongside their exact `execution_devices` permutation (`[0,1,4,5,2,3,...]` on `TP16`) to `/dev/shm` (`1.07 GB`). After `ReleaseGate` handoff, workers reload all 187 executables via `deserialize_and_load` in *3.45 s* (`0.0 s` Python trace/lower, `0` misses across 65,956 calls) guarded by a rank-0 unanimity barrier.

#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    [Switch Segment (s)], [`P2`], [`O0`], [`O3` Full], [`F2` Warm], [`O3` Part],
    [Pre-spawn (hidden)], [0.00], [0.00], [(34.30)], [(34.10)], [(32.39)],
    [1. Drain + release], [3.75], [3.49], [1.98], [1.95], [1.85],
    [2. Gate + VFIO wait], [11.11], [12.16], [0.86], [0.84], [0.84],
    [3. Spawn + imports], [21.55], [21.56], [*0.00*], [*0.00*], [*0.00*],
    [4. `init_device`], [3.55], [3.63], [3.92], [3.90], [3.74],
    [5. `load_model`], [5.01], [4.82], [4.91], [4.88], [4.93],
    [6. `init_from_config`], [25.92], [2.63], [2.57], [2.56], [2.59],
    [7. `compile_warmup`], [54.61], [35.15], [*4.98*], [*4.95*], [*5.05*],
    [8. Gateway ready], [4.21], [3.76], [3.09], [3.05], [3.21],
    [*Dispatch Gap*], [*130.61*], [*87.20*], [*23.98*], [*23.61*], [*20.63*],
    [Phase-2 Ramp], [+1.63], [+2.64], [+3.53], [-2.47], [-1.79],
    [*Production $C$ (s)*], [*132.24*], [*89.84*], [*27.51*], [*21.14*], [*18.84*],
  ),
  caption: [Measured segment breakdown of warm reconfiguration cost $C$ across cuts (`P2`, `O0`, `O3`, `F2`). Pre-spawning and the Executable Bank compress full `S → H` $C$ to 21.14–27.51 s and partial `S → Mb` $C_"eff"$ to 18.84 s.],
) <tab-segments>

== Cold First Switch: Inline Provisioning & Background CPU Pre-Read ($222.6" s" arrow.r 134.1" s"$)
When switching to `H` for the first time (warm XLA cache, no pre-existing snapshot or bank):
1. *Inline Provisioning (`F0–F2`, commit `dcc3276`):* Folding `snapshot_write` (7.53 s) and `exec_bank_write` (2.26 s) into the first target deploy eliminates a separate 350.8 s provisioning pass, yielding $C_"first" = 222.63" s"$ (dominated by 158.9 s `load_model`, of which 130.7 s is host tensor conversion).
2. *Background CPU Checkpoint Pre-Read (`F3`, `hetero/preread.py`, commit `b497642`):* During Phase 1, each parked `H` worker spawns a `nice 10` child process (`JAX_PLATFORMS=cpu`, 16 fake host devices, all `TPU_*` env vars stripped) that streams `load_weights` and writes all *32,896 shards* in `weight_snapshot` v2 format to `/dev/shm` before `T0`. All 32,896 shards match the TPU reference snapshot bitwise (`sha256`), `S` foreground throughput is untouched (`2,701 / 4,013 tok/s`), and $C_"first"$ drops by *40% (88.5 s)* from 222.63 s to *134.10 s*, cutting 1/3/10-switch amortized cost to *134.1 / 58.8 / 32.4 s* (vs. 378.3 / 144.4 / 62.6 s with a separate pass).

= Hardware Limit: In-Runtime Submesh Switching in $12" ms"$ <sec-unified>

To establish the physical lower bound when weights stay in HBM and processes never restart, we built a *Unified 16-Chip Runtime* (`~/hetero-wt/UnifiedRuntime`, `U0–U3`) that initializes one JAX domain across 4 hosts ($4 times 4$ torus) and switches between two 8-chip `TP=8` submeshes and one 16-chip `TP=16` mesh.

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
  caption: [16-chip ($4 times 4$) submesh partitioning in `libtpu 0.0.43` (`U0–U2`). Splitting by columns (`x`) aborts in `net_util.cc:2304` because `x-lower` contains origin device 0 with non-contiguous IDs. Splitting by host rows (`y`: `0..7` and `8..15`) enables full submesh independence and 12.0 ms in-HBM switching.],
) <fig-submesh>

1. *Bypassing `libtpu 0.0.43` Sub-Box Routing Bug (`U0/U1`, @fig-submesh):* Splitting the $4 times 4$ grid along columns (`x-lower` IDs `[0,1,4,5,8,9,12,13]`) crashes in `tpu/compiler/service/Au/net_util.cc:2304` (`RET_CHECK device_id < kMaxDeviceCount (8 vs. 8)`) whenever a collective is compiled, because `libtpu` assumes any origin-anchored sub-box has dense device IDs `0..N-1`. Rotating the split to *host rows (`y`: `0..7` and `8..15`)* satisfies this invariant for both live `jax.jit` and `deserialize_and_load` (`U1d`).
2. *Submesh Isolation & $12.0" ms"$ Switch (`U1/U2`):* Both `y`-halves step concurrently at $1.05"–"1.08 times$ their solo rate; stalling `y-upper` in a 30 s barrier leaves `y-lower` at 97%–99% throughput (`max step 4.50 ms`). With `1.61 GB/chip` of `TP8` weights, background compilation of all `TP16` buckets takes 11.82 s while slowing foreground `TP8` serving by only *11.8%–13.0%*. Switching `2×TP8 → TP16` in nested `local` device order takes *12.04 ms* (0.08 ms stop + 6.14 ms on-chip slice + 5.82 ms first step); reverse `TP16 → 2×TP8` over ICI takes *48.02 ms*.
3. *Systems Trade-off (`U3`):* Retrofitting unified submeshes into vLLM requires refactoring ~1.5–3k LOC of process-global singletons (`tpu_worker.py`, `jax.clear_caches()`, Ray executor bindings) and merges 4 hosts into a single failure domain, while only trimming warm $C$ from $21.1"–"27.5" s" arrow.r approx 9"–"10" s"$ ($+0.3" pp"$ median bulk gain). PhaseMorph therefore uses process-isolated `O3`/`F2` switching in production.

= Phase-Space Evaluation & Break-Even Map <sec-eval>

We evaluate PhaseMorph across the two-phase workload grid $(k_s " W0 short manifests", k_l " W6 long manifests") in [0, 8] times [0.25, 8]$ using our calibrated bulk cost model (`scripts/bulk_cost_model.py`, validated against all 10 end-to-end multi-phase runs within $max |Delta k| lt.eq 0.009$, with `S` tail-drain $O = 32.6" s"$ and static `H` @ `131k` rate $a_"0H" = 267.8" s"$):
- *Static `H`:* $T_"H" = 267.8 k_s + 206.2 k_l$
- *Static `M`:* $T_"M" = max(352.7 k_s, 409.2 k_l)$
- *Full Switch `S → H`:* $T_"S→H" = 190.8 k_s + O + C_"full" + 206.2 k_l$
- *Partial Switch `S → M`:* $T_"S→M" = 190.8 k_s + min_(f in [0,1]) max(352.7(1-f)k_s, O + C_"part" + 409.2 k_l)$

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 265" font-family="Helvetica, Arial, sans-serif" font-size="10.5" fill="#1f2328">
  <text x="215" y="16" text-anchor="middle" font-weight="bold" font-size="11.5">Winning Strategy Regions in (k_l W6, k_s W0) Space at O3</text>
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

  <!-- Historical break-even lines (dashed) -->
  <line x1="55" y1="173.6" x2="395" y2="173.6" stroke="#cf222e" stroke-width="1.2" stroke-dasharray="5,3"/>
  <text x="392" y="170" text-anchor="end" font-size="9" fill="#cf222e">P5 Break-Even k_s* = 2.14 (C=132.2s)</text>
  <line x1="55" y1="186.8" x2="395" y2="186.8" stroke="#8250df" stroke-width="1.2" stroke-dasharray="3,3"/>
  <text x="392" y="184" text-anchor="end" font-size="9" fill="#8250df">O0 k_s* = 1.59 (C=89.8s)</text>
  <text x="392" y="203" text-anchor="end" font-size="9.5" font-weight="bold" fill="#1a7f37">O3 k_s* = 0.78 (C=27.5s)</text>

  <!-- Region labels -->
  <text x="145" y="88" text-anchor="middle" font-weight="bold" fill="#0969da">Partial Switch S → M</text>
  <text x="145" y="102" text-anchor="middle" font-size="9.5" fill="#0969da">(41% of grid: short-heavy)</text>
  <text x="330" y="64" text-anchor="middle" font-size="9" font-weight="bold" fill="#9a6700">Static M (6%)</text>
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
  caption: [Phase-space winning map across $(k_l " W6", k_s " W0") in [0.25, 8] times [0, 8]$ at `O3` switch costs ($C_"full"=27.5" s"$, $C_"part"=18.8" s"$). Dynamic switching wins 82% of the grid (41% Full `S → H`, 41% Partial `S → M`), lowering the short-phase break-even threshold $k_s^*$ from 2.14 (`P5`) to 0.78 `W0` manifests (0.36 if $O=0$).],
) <fig-regions>

#wide-fig(
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    [Reconfiguration Regime], [Full $C$ (s)], [Partial $C_"eff"$ (s)], [Grid Won by Switch], [Median / Max Gain], [`W0→W6` $k_s^*$ (Manifests / Toks)], [`W0→W7` $k_m^*$ (Manifests / Toks)],
    [1. Baseline Redeploy], [419.0], [186.0], [29%], [4.1% / 18.2%], [5.86 (`4.23M` tok)], [10.35 (`823k` tok)],
    [2. Cut 1 (`P5`: FastRestore)], [132.2], [128.9], [62% (`24%` H, `38%` M)], [8.3% / 24.5%], [2.14 (`1.54M` tok)], [3.78 (`301k` tok)],
    [3. Cut 2 (`O0`: 0/0 Cache)], [89.8], [87.3], [69% (`30%` H, `39%` M)], [8.7% / 25.4%], [1.59 (`1.15M` tok)], [2.81 (`223k` tok)],
    [*4. Cut 3 (`O3`: Pre-Spawn + Bank)*], [*27.5*], [*18.8*], [*82%* (`41%` H, `41%` M)], [*9.9% / 26.8%*], [*0.78 (`563k` tok)*], [*1.38 (`110k` tok)*],
    [*5. Cut 3 (`O3`) + Tail-Aware ($O=0$)*], [*27.5*], [*18.8*], [*88%* (`47%` H, `41%` M)], [*10.4% / 27.9%*], [*0.36 (`260k` tok)*], [*0.63 (`50k` tok)*],
  ),
  caption: [Impact of reconfiguration cost $C$ on phase-space coverage, makespan reduction over the best static layout, and minimum phase sizes ($k_s^*$, $k_m^*$) required to break even.],
) <tab-breakeven>

As shown in @fig-regions and @tab-breakeven, compressing $C$ from 419.0 s to 27.5 s lowers the short-phase break-even threshold $k_s^* = (C + O) / (a_"0H" - a_"0S")$ by *$7.5 times$*—from 5.86 `W0` manifests ($4.23"M"$ output tokens) down to *0.78 manifests ($563"k"$ output tokens, or 149 s of `S` serving)* with conservative tail drain, and *0.36 manifests ($260"k"$ tokens)* with tail-aware dispatch. For mid-length `W0 → W7` jobs (where `S` is feasible on both phases), `O3` reduces the `W7` break-even threshold $k_m^*$ from 3.78 to *1.38 `W7` manifests* ($110"k"$ output tokens).

= Related Work & Conclusion <sec-conclusion>

*Related Work:* DistServe, Splitwise, Mooncake, and Llumnix disaggregate prefill/decode or migrate live KV caches across homogeneous GPU instances for interactive SLAs. PhaseMorph targets *bulk makespan* across *heterogeneous TP degrees* on TPU slices, showing that phase-level mesh reconfiguration outperforms cross-TP KV migration. ServerlessLLM and PipeSwitch accelerate GPU checkpoint loading; we show via `LoadProfiler` that 98.8% of TPU `load_model` latency is host layout conversion, sharding JIT, and `nnx.eval_shape`, which `FastSnapshot`, `FastGraph`, and `preread` eliminate.

*Conclusion:* On multi-host Cloud TPU slices, no static tensor-parallel partition can simultaneously maximize short-context MFU and long-context KV capacity. *PhaseMorph* proves that the multi-minute barrier to runtime TPU mesh reconfiguration is an artifact of software-stack overheads. By compressing warm full-slice switching from 419.0 s to *21.1–27.5 s*, partial switching to *18.8 s*, cold first-switch provisioning to *134.1 s*, and demonstrating a *12.0 ms* in-runtime hardware bound, PhaseMorph makes dynamic TPU mesh reconfiguration the optimal strategy across *82%–88%* of heterogeneous bulk LLM workloads.

= References <sec-refs>

1. W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," *SOSP*, 2023.
2. Y. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," *OSDI*, 2024.
3. P. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," *ISCA*, 2024.
4. B. Sun et al., "Llumnix: Dynamic Scheduling for Large Language Model Serving," *OSDI*, 2024.
5. Y. Fu et al., "ServerlessLLM: Low-Latency Serverless Inference for Large Language Models," *OSDI*, 2024.
6. P. Barham et al., "Pathways: Asynchronous Distributed Dataflow for ML," *MLSys*, 2022.
7. N. P. Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings," *ISCA*, 2023.
