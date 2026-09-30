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
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 500" font-family="Helvetica, Arial, 'DejaVu Sans', 'Libertinus Serif', sans-serif" font-size="13" fill="#1f2328">
  <defs>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#59636e"/>
    </marker>
  </defs>
  <g stroke="#8c959f" stroke-width="1.2">
    <rect x="40" y="10" width="380" height="46" rx="6" fill="#ffffff"/>
    <rect x="40" y="84" width="380" height="34" rx="6" fill="#ffffff"/>
    <rect x="40" y="146" width="380" height="46" rx="6" fill="#f6f8fa"/>
    <rect x="40" y="226" width="380" height="46" rx="6" fill="#f6f8fa"/>
    <rect x="40" y="316" width="380" height="46" rx="6" fill="#f6f8fa"/>
    <rect x="500" y="146" width="240" height="46" rx="6" fill="#ddf4ff" stroke="#0969da"/>
    <rect x="500" y="226" width="240" height="46" rx="6" fill="#ffffff"/>
    <rect x="500" y="316" width="240" height="46" rx="6" fill="#ffffff"/>
    <rect x="40" y="410" width="330" height="70" rx="6" fill="#ffebe9" stroke="#cf222e" stroke-width="1.6"/>
    <rect x="390" y="410" width="350" height="70" rx="6" fill="#dafbe1" stroke="#1a7f37" stroke-width="1.6"/>
  </g>
  <g stroke="#59636e" stroke-width="1.4" fill="none" marker-end="url(#a)">
    <line x1="230" y1="56" x2="230" y2="82"/>
    <line x1="230" y1="118" x2="230" y2="144"/>
    <line x1="230" y1="192" x2="230" y2="224"/>
    <line x1="230" y1="272" x2="230" y2="314"/>
    <line x1="420" y1="169" x2="498" y2="169"/>
    <line x1="420" y1="249" x2="498" y2="249"/>
    <line x1="420" y1="339" x2="498" y2="339"/>
    <path d="M620,272 V304 H330 V314"/>
    <path d="M230,362 V386 H160 V408"/>
    <path d="M230,386 H565 V408"/>
    <path d="M40,445 H18 V33 H38" stroke-dasharray="5 4"/>
  </g>
  <g font-size="12" fill="#59636e">
    <text x="459" y="163" text-anchor="middle">no</text>
    <text x="238" y="213">yes</text>
    <text x="459" y="243" text-anchor="middle">yes</text>
    <text x="238" y="291">no: TP8 first, then TP16</text>
    <text x="459" y="333" text-anchor="middle">yes</text>
    <text x="238" y="380">no</text>
    <text x="12" y="240" transform="rotate(-90 12 240)" text-anchor="middle">plain: retry</text>
  </g>
  <text x="56" y="30" font-weight="bold">Request arrives</text>
  <text x="56" y="48" font-size="12" fill="#59636e">prompt length P is known · output length is hidden</text>
  <text x="56" y="106">Charge P + 1 KV block</text>
  <text x="56" y="166" font-weight="bold">Does it fit a TP8 engine?</text>
  <text x="56" y="184" font-size="12" fill="#59636e">P + 1 block ≤ 16,384 tokens</text>
  <text x="56" y="246" font-weight="bold">Is TP16 owed this request?</text>
  <text x="56" y="264" font-size="12" fill="#59636e">TP16 has taken &lt; 50% of dispatched requests (M only)</text>
  <text x="56" y="336" font-weight="bold">Does the first pool have an engine with room?</text>
  <text x="56" y="354" font-size="12" fill="#59636e">admission check, Figure 2</text>
  <text x="516" y="166" font-weight="bold">TP16 pool only</text>
  <text x="516" y="184" font-size="12" fill="#59636e">all 24 long documents</text>
  <text x="516" y="254">Try TP16 first, then TP8</text>
  <text x="516" y="336">Dispatch to the least-loaded</text>
  <text x="516" y="353">engine that passed</text>
  <text x="56" y="432" font-weight="bold" fill="#cf222e">plain</text>
  <text x="56" y="451">wait one poll, holding a dispatch slot,</text>
  <text x="56" y="469">then retry from the top</text>
  <text x="406" y="432" font-weight="bold" fill="#1a7f37">reserve (work-conserving fallback)</text>
  <text x="406" y="451">try the second pool;</text>
  <text x="406" y="469">wait only if it has no room either</text>
</svg>
```),
  caption: [Routing decision for one new request, as implemented in `select_initial`. Requests
    that only fit TP16 go through the same room check with TP16 as their only pool. The two
    policies differ only in the last step (bottom) and in the admission check (@fig-kv).],
) <fig-flow>

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 270" font-family="Helvetica, Arial, 'DejaVu Sans', 'Libertinus Serif', sans-serif" font-size="13" fill="#1f2328">
  <!-- 282 blocks over 720 px: 2.553 px per block. Snapshot: 90 used, 192 free. -->
  <text x="20" y="22" font-weight="bold">One TP16 engine · 282 blocks × 256 tokens · snapshot with 192 blocks free</text>
  <g stroke="#ffffff" stroke-width="1.5">
    <rect x="20" y="36" width="230" height="46" fill="#8c959f"/>
    <rect x="250" y="36" width="82" height="46" fill="#ffffff" stroke="#8c959f"/>
    <rect x="332" y="36" width="368" height="46" fill="#e16f24"/>
    <rect x="700" y="36" width="40" height="46" fill="#424a53"/>
  </g>
  <g text-anchor="middle" font-size="12">
    <text x="135" y="64" fill="#ffffff">in use: 90 blocks</text>
    <text x="291" y="64">32</text>
    <text x="516" y="58" fill="#ffffff" font-weight="bold">long-request reserve: 144 blocks</text>
    <text x="516" y="74" fill="#ffffff">= 36,864 tokens, the TP16 max_model_len</text>
    <text x="720" y="64" fill="#ffffff">16</text>
  </g>
  <line x1="332" y1="30" x2="332" y2="88" stroke="#1f2328" stroke-width="1.2" stroke-dasharray="4 3"/>
  <text x="336" y="98" font-size="11" fill="#59636e">threshold on the free count, not fixed blocks</text>
  <rect x="250" y="110" width="82" height="9" rx="2" fill="#1a7f37"/>
  <text x="250" y="137">short request may use 192 − 144 − 16 = <tspan font-weight="bold">32 blocks</tspan></text>
  <rect x="250" y="152" width="450" height="9" rx="2" fill="#0969da"/>
  <text x="250" y="179">long request (prompt &gt; 16,384) may use 192 − 16 = <tspan font-weight="bold">176 blocks</tspan></text>
  <g font-size="12.5">
    <text x="20" y="212">30,000-token document: ⌈30,256 / 256⌉ = 119 blocks ≤ 176 → admitted</text>
    <text x="20" y="232">500-token chat request: 3 blocks ≤ 32 → admitted</text>
    <text x="20" y="252" fill="#59636e">Dark: 16-block headroom, every request, both policies. TP8 engines (412 × 64 tokens): no headroom, no reserve.</text>
  </g>
</svg>
```),
  caption: [Admission check on a TP16 engine under the reserve policy. Under plain the orange
    region does not exist, so both request types may use 176 blocks.],
) <fig-kv>

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 290" font-family="Helvetica, Arial, 'DejaVu Sans', 'Libertinus Serif', sans-serif" font-size="13" fill="#1f2328">
  <!-- Cells 50 × 34, pitch 54, first cell at x = 110. -->
  <text x="110" y="20" fill="#59636e">arrival order →   (scenario: TP16 has no room for another short request)</text>
  <text x="20" y="57" font-weight="bold">plain</text>
  <g stroke="#ffffff" stroke-width="1">
    <rect x="110" y="36" width="50" height="34" rx="3" fill="#0969da"/>
    <rect x="164" y="36" width="50" height="34" rx="3" fill="#1a7f37"/>
    <rect x="218" y="36" width="50" height="34" rx="3" fill="#d0d7de"/>
    <rect x="272" y="36" width="50" height="34" rx="3" fill="#d0d7de"/>
    <rect x="326" y="36" width="50" height="34" rx="3" fill="#d0d7de"/>
    <rect x="380" y="36" width="50" height="34" rx="3" fill="#d0d7de"/>
    <rect x="434" y="36" width="50" height="34" rx="3" fill="#d0d7de"/>
    <rect x="488" y="36" width="50" height="34" rx="3" fill="#d0d7de"/>
    <rect x="542" y="36" width="50" height="34" rx="3" fill="#d0d7de"/>
    <rect x="596" y="36" width="50" height="34" rx="3" fill="#d0d7de"/>
    <rect x="650" y="36" width="50" height="34" rx="3" fill="#d0d7de"/>
  </g>
  <g text-anchor="middle" font-size="12">
    <text x="135" y="58" fill="#ffffff">r1</text>
    <text x="189" y="58" fill="#ffffff">r2</text>
    <text x="243" y="58">r3</text>
    <text x="297" y="58">r4</text>
    <text x="351" y="58">r5</text>
    <text x="405" y="58">r6</text>
    <text x="459" y="58">r7</text>
    <text x="513" y="58">r8</text>
    <text x="567" y="58">r9</text>
    <text x="621" y="58">r10</text>
    <text x="675" y="58">r11</text>
  </g>
  <path d="M222,76 v6 h474 v-6" fill="none" stroke="#cf222e" stroke-width="1.4"/>
  <text x="110" y="102" font-size="12.5">r3 is owed to TP16 and waits. The quota counter moves only on dispatch,</text>
  <text x="110" y="119" font-size="12.5">so r4, r5, … are owed to TP16 too: TP8 gets one request per request TP16 accepts.</text>
  <text x="20" y="163" font-weight="bold">reserve</text>
  <g stroke="#ffffff" stroke-width="1">
    <rect x="110" y="142" width="50" height="34" rx="3" fill="#1a7f37"/>
    <rect x="164" y="142" width="50" height="34" rx="3" fill="#1a7f37"/>
    <rect x="218" y="142" width="50" height="34" rx="3" fill="#1a7f37"/>
    <rect x="272" y="142" width="50" height="34" rx="3" fill="#1a7f37"/>
    <rect x="326" y="142" width="50" height="34" rx="3" fill="#0969da"/>
    <rect x="380" y="142" width="50" height="34" rx="3" fill="#1a7f37"/>
    <rect x="434" y="142" width="50" height="34" rx="3" fill="#1a7f37"/>
    <rect x="488" y="142" width="50" height="34" rx="3" fill="#1a7f37"/>
    <rect x="542" y="142" width="50" height="34" rx="3" fill="#1a7f37"/>
    <rect x="596" y="142" width="50" height="34" rx="3" fill="#1a7f37"/>
    <rect x="650" y="142" width="50" height="34" rx="3" fill="#1a7f37"/>
  </g>
  <g text-anchor="middle" font-size="12" fill="#ffffff">
    <text x="135" y="164">r1 ↪</text>
    <text x="189" y="164">r2 ↪</text>
    <text x="243" y="164">r3 ↪</text>
    <text x="297" y="164">r4 ↪</text>
    <text x="351" y="164">doc</text>
    <text x="405" y="164">r5 ↪</text>
    <text x="459" y="164">r6 ↪</text>
    <text x="513" y="164">r7 ↪</text>
    <text x="567" y="164">r8 ↪</text>
    <text x="621" y="164">r9 ↪</text>
    <text x="675" y="164">r10 ↪</text>
  </g>
  <text x="110" y="202" font-size="12.5">Each short request is owed to TP16, finds no room outside the reserve and falls back (↪) to TP8.</text>
  <text x="110" y="219" font-size="12.5">A long document ("doc") may use the reserve, so it goes to TP16.</text>
  <g font-size="12">
    <rect x="110" y="250" width="22" height="18" rx="3" fill="#0969da"/>
    <text x="140" y="264">dispatched to TP16</text>
    <rect x="280" y="250" width="22" height="18" rx="3" fill="#1a7f37"/>
    <text x="310" y="264">dispatched to TP8</text>
    <rect x="440" y="250" width="22" height="18" rx="3" fill="#d0d7de"/>
    <text x="470" y="264">waiting (holds a dispatch slot)</text>
  </g>
</svg>
```),
  caption: [What work-conserving fallback fixes in M (schematic). Measured in the M runs: under
    plain, 439 of 880 short requests went to TP16, close to strict alternation, and the TP8 engines
    went idle at 204 s and 218 s. Under reserve, 803 short requests were owed to TP16 and fell back
    to TP8; only 56 ran on TP16.],
) <fig-quota>

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 240" font-family="Helvetica, Arial, 'DejaVu Sans', 'Libertinus Serif', sans-serif" font-size="13" fill="#1f2328">
  <!-- 282 blocks over 630 px starting at x = 110: 2.234 px per block. -->
  <text x="20" y="20" font-weight="bold">One H engine (TP16, 282 blocks) while a 30,000-token document (119 blocks) waits</text>
  <text x="20" y="60" font-weight="bold">plain</text>
  <g stroke="#ffffff" stroke-width="1.5">
    <rect x="110" y="36" width="567" height="38" fill="#8c959f"/>
    <rect x="677" y="36" width="27" height="38" fill="#ffffff" stroke="#8c959f"/>
    <rect x="704" y="36" width="36" height="38" fill="#424a53"/>
  </g>
  <text x="393" y="60" text-anchor="middle" font-size="12" fill="#ffffff">short requests: 254 blocks</text>
  <text x="690" y="60" text-anchor="middle" font-size="12">12</text>
  <text x="110" y="94" font-size="12.5">Freed blocks go straight to the next waiting short request, so 119 free blocks rarely line up.</text>
  <text x="110" y="111" font-size="12.5" fill="#59636e">Measured (H r1): median document 297.9 s · median short request 113.9 s</text>
  <text x="20" y="156" font-weight="bold">reserve</text>
  <g stroke="#ffffff" stroke-width="1.5">
    <rect x="110" y="132" width="273" height="38" fill="#8c959f"/>
    <rect x="383" y="132" width="321" height="38" fill="#e16f24"/>
    <rect x="704" y="132" width="36" height="38" fill="#424a53"/>
  </g>
  <text x="246" y="156" text-anchor="middle" font-size="12" fill="#ffffff">short requests: 122 blocks</text>
  <text x="543" y="156" text-anchor="middle" font-size="12" fill="#ffffff">kept free for long requests: 144 blocks</text>
  <text x="110" y="190" font-size="12.5">Short requests stop at 160 free blocks, so the document starts on arrival,</text>
  <text x="110" y="207" font-size="12.5">but short requests now share about half the KV.</text>
  <text x="110" y="224" font-size="12.5" fill="#59636e">Measured (H r1): median document 148.1 s · median short request 214.2 s</text>
</svg>
```),
  caption: [What the long-request reserve fixes, and what it costs, on the homogeneous layout
    (schematic occupancy, measured times). H has no second pool, so only the reserve acts there.],
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
