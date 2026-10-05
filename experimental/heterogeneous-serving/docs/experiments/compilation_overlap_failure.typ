// Self-contained, package-free report. Evidence links point to the sealed published package.
// typst compile --features html --format html compilation_overlap_failure.typ /tmp/compilation-overlap-failure.html
// typst compile compilation_overlap_failure.typ /tmp/compilation-overlap-failure.pdf
#let title = "Why compilation overlap failed"
#set document(title: title, author: "hetero-research")
#set heading(numbering: "1.")
#show table.cell.where(y: 0): strong
#let is-html() = "target" in dictionary(std) and std.target() == "html"
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
.note { background: #f6f8fa; border-left: 4px solid #cf222e; padding: .6em 1em; margin: 1em 0; }
```.text
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
#let note(body) = context if is-html() {
  html.elem("div", attrs: (class: "note"), body)
} else { block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%, body) }
#let svg-fig(source) = image(bytes(source.text), format: "svg", width: 100%)
#let evidence-base = "https://github.com/taooceros/llm-d/blob/3630e63/experimental/heterogeneous-serving/docs/experiments/reconfiguration_ici_evidence/"
#let evidence(path, body) = link(evidence-base + path, body)

#context if is-html() { html.elem("h1", title) } else {
  align(center, text(size: 20pt, weight: "bold", title))
}

Gemma-4 31B on a 32-chip TPU v5e slice · closed compilation-overlap campaign

#note[
  *Result: the tested background path failed before replay compilation.* Obtaining the
  target TPU topology initialized the TPU plugin, which rejected the process because
  the live serving process already owned the devices. The source workload completed,
  but there was no treated target deployment, overlap switch cost or new crossover.
  This is a failure of the tested setup path, not proof that CPU-side precompilation
  can never overlap TPU execution.
]

= Question and intended experiment

The goal was to prepare the next layout's executables on host CPUs while the current
layout continued serving, then reuse the persistent compilation cache after switching.
H uses two TP16 engines; S uses four TP8 engines. The failed attempt served H while
preparing S. M, the mixed TP16 plus two TP8 layout, was not the target of this attempt.

The intended comparison separated a cold target, an ordinary warm target, and a target
precompiled during source serving. It also needed to measure serving interference and
charge any upfront capture, provisioning and cache-publication work. Only the cold/warm
H-to-S baseline completed. Dependent S-to-H arms were not submitted after the blocker.

The central assumption was too strong: *compilation running on host CPUs does not imply
that compiler setup can avoid initializing the TPU runtime.*

= What actually failed

The live job was `raysubmit_69VBfS2EwyGPMt5B`. Its background process, PID 572460, ran on
`192.168.102.38`, where serving PID 570736 already owned the TPU devices as `large_a`,
rank 0. The independently checked call chain was:

```text
scripts/overlap_pilot.py:72       compile_task
scripts/overlap_replay.py:109     replay
jax.experimental.topologies.get_topology_desc(platform="tpu", ...)
jax._src.xla_bridge.make_pjrt_tpu_topology
xla_client.initialize_pjrt_plugin("tpu")
JaxRuntimeError: ABORTED: The TPU is already in use by process
with pid 570736. Not attempting to load libtpu.so in this process.
```

#figure(
  svg-fig(```svg
<svg xmlns="http://www.w3.org/2000/svg" width="900" height="218" viewBox="0 0 900 218">
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#59636e"/></marker></defs>
  <g font-family="Arial, Helvetica, DejaVu Sans, sans-serif" font-size="17" fill="#1f2328">
    <rect x="8" y="12" width="274" height="86" rx="6" fill="#f6f8fa" stroke="#8c959f"/>
    <text x="24" y="40" font-weight="bold">Background process</text>
    <text x="24" y="65">CPU backend selected</text>
    <text x="24" y="86" font-size="15">No replay compilation reached</text>
    <path d="M283 55 H309" stroke="#59636e" stroke-width="2" marker-end="url(#a)"/>
    <rect x="312" y="12" width="274" height="86" rx="6" fill="#fff8c5" stroke="#9a6700"/>
    <text x="328" y="40" font-weight="bold">Explicit TPU topology call</text>
    <text x="328" y="65">Initializes the TPU plugin</text>
    <text x="328" y="86" font-size="15">CPU selection is not isolation</text>
    <path d="M587 55 H613" stroke="#59636e" stroke-width="2" marker-end="url(#a)"/>
    <rect x="616" y="12" width="276" height="86" rx="6" fill="#ffebe9" stroke="#cf222e"/>
    <text x="632" y="40" font-weight="bold">Ownership rejection</text>
    <text x="632" y="65">TPU already in use</text>
    <text x="632" y="86" font-size="15">No background result</text>
    <rect x="312" y="142" width="580" height="60" rx="6" fill="#dafbe1" stroke="#1a7f37"/>
    <text x="328" y="166" font-weight="bold">Existing serving process: PID 570736</text>
    <text x="328" y="190">Same host; owns TPUs; source W0 still completes</text>
    <path d="M754 141 V101" stroke="#59636e" stroke-width="2" marker-end="url(#a)"/>
  </g>
</svg>
```),
  caption: [Observed setup path, schematic rather than a timing diagram. The failure
  precedes the replay compile/profile loop; it is not a graph-compilation failure.],
) <fig-failure>

The code already set `JAX_PLATFORMS=cpu` and checked whether a TPU backend was present
in `xla_bridge._backends`. The remote task also used `TPU_SKIP_MDS_QUERY=1`, one Ray CPU
reservation, and affinity to a TPU worker host. These measures did not prevent the
explicit topology API from initializing the plugin. Reapplying those flags would not
address the observed failure. A Ray CPU reservation alone also does not demonstrate
compiler-thread affinity or a one-core limit.

Evidence: #evidence("OverlapFollowup/background-topology-error.log", [raw topology stack]),
#evidence("OverlapFollowup/failure-observed.json", [failed-attempt record]), and
#evidence("FollowupAudit/followup_final_overlap_claims.json", [independent ownership audit]).

== Why the earlier preflight passed

The isolated replay ran on an idle slice. With no serving process owning the devices,
that environment could not expose the ownership conflict. Fresh processes for S and H
produced 261 matching executable keys and 261 physical cache files per layout, with
matching bytes and hashes when inspected across eight hosts. This proved physical
persistence, not live coexistence or consumption by an overlap-treated target.

An earlier sequential replay had also exposed JAX 0.10.2's process-global cache-directory
latching: changing the requested directory in one process could leave writes going to
the first directory. Isolating each target replay in its own process resolved that
qualification issue. It did not resolve the separate live-serving ownership restriction.

The test-order mistake was to qualify idle-slice persistence before running the smallest
same-host live-serving coexistence gate. The longer campaign then encountered a setup
failure that a narrow prerequisite test could have exposed first.

== Why the supervisor appeared stuck

After the background failure, a physical-cache gate assumed a successful result existed:

```python
# Historical overlap_physical_cache_gate.py:33
evidence['measurements']['pilot']['during_run']['result']
```

The missing `result` raised `KeyError` before `StageRunner`. Daemon 4133323 became a
zombie under stuck client 4133322. Only the owned stuck client was terminated, allowing
the existing EXIT cleanup to run. This was a second defect in failure handling, not slow
TPU compilation. The historical wrapper and logs are retained; restarting it is not a
recovery plan. See the #evidence("OverlapFollowup/physical-gate-startup-error.log", [gate error log]).

= What was measured, and what was not

#table(
  columns: (1.05fr, 1fr, 1.7fr),
  align: (left, left, left),
  table.header([Observation], [Value], [Accepted interpretation]),
  [Cold H-to-S baseline C], [458.296556 s], [Observed pilot switch accounting],
  [Warm H-to-S baseline C], [225.260380 s], [Observed warm control, not an overlap arm],
  [Warm startup cache reuse], [2,088 keyed hits], [Successful reads/deserializations and listener hits over 16 worker-phases],
  [Source W0 in failed attempt], [502.8775 s], [Completed 1,763 requests and 721,595 output tokens],
  [Source control W0], [490.7819 s], [The 12.0956 s difference is incidental, not measured successful-compile interference],
  [Overlap target / C / saving], [Not measured], [No treated target deployment or target-reuse proof; no accepted overlap saving],
)

The warm control had zero cache misses in the instrumented phases. Its post-W0 checks
matched 128 prompts and 8,140 output tokens. Those checks were outside C. Retained ESTALE
cache warnings were not erased or silently repaired. The post-run inventory exposed
295 keys across eight hosts; that does not imply eight independent cache copies.

== Timing boundary

For these pilots, *C = source-to-target dispatch gap + output-equivalent positive ramp deficit*.
The gap starts at the last source completion and ends at the first *measured target W0
dispatch*. The ramp term covers the first 120 s, using the median of 10 s output-rate bins
in the middle 60% of target W0 as the steady-rate reference.

#table(
  columns: (1fr, auto, auto, auto),
  align: (left, right, right, right),
  table.header([H-to-S arm], [Gap (s)], [Ramp (s)], [C (s)]),
  [Cold], [451.409829], [6.886727], [458.296556],
  [Warm], [218.716232], [6.544148], [225.260380],
)

Normal pre-W0 harness warm-up is charged in the gap. Post-W0 output equality and cache
inventory are not. This is harness-inclusive pilot accounting, not a minimum production
outage or a time-to-first-ever-token measurement.

Prior target provisioning took 920.802339 s, including loading, reference generation,
snapshot creation and release. It is reported separately: it was neither free work nor
pure compilation. Cold C plus that provisioning is 1,379.098895 s. A future comparison
must state whether provisioning is amortized or lies on the measured critical path.

== Limits on the comparison

These are single observations in fixed cold-then-warm order. The launcher/worktree was
identified as `9009fc7`, but loaded-backend equivalence was not established:
`UNFROZEN` and `loaded_runtime: null` remain in the evidence. Neither a causal estimate
nor a production saving follows from subtracting the two baseline values.

Historical and current workload catalogues differ. The reducer therefore suppresses
six historical crossover candidates rather than borrowing incompatible throughput
rates. Missing original S/H phase markers also prevent a defensible device-versus-host
attribution. Unknown device work is not zero device work.

Sources: #evidence("OverlapFollowup/overlap_switch.json", [switch reduction]),
#evidence("OverlapFollowup/serving_impact.json", [serving-impact scope]),
#evidence("OverlapFollowup/breakeven.json", [crossover rejection]), and
#evidence("FollowupAudit/followup_warm_transition_claims.json", [warm-control audit]).

= Separate improvements remain valid

The failed overlap attempt does not invalidate independently qualified component work.
These improvements belong to different worktrees and have not been measured as a
combined end-to-end transition.

#table(
  columns: (1fr, 1fr, 1.5fr),
  align: (left, left, left),
  table.header([Component], [Observed change], [Qualification]),
  [Cold TP8 snapshot write], [50–52 → 14–15 s per worker], [Old path includes SHA-256; new timed path excludes it. Not an equal-checksum A/B.],
  [Restore graph construction], [14.631 → 1.766 s], [Separate graph fix, with output checks],
  [Checksum-enabled restore `load_model`], [39.436 → 26.454 s], [SHA-256 remains enabled in both graph-comparison arms],
  [Persist small compilations], [Warm deploy 383.9 → 275.4 s], [One startup comparison with effective settings checked, not full C],
)

There is no qualified one-second complete snapshot or Pallas/direct caller-owned
zero-copy snapshot in this campaign. Synthetic pinned-memory bandwidth and weak
checksum probes are separate from complete snapshot correctness and persistence.

Sources: #evidence("FollowupAudit/followup_restore_claims.json", [restore audit]),
#evidence("FollowupAudit/followup_graph_claims.json", [graph audit]), and
#evidence("FollowupAudit/followup_threshold_claims.json", [cache-threshold audit]).

= What should happen before another campaign

The next work is a narrow prerequisite investigation, not another full benchmark.

+ *Repair failure handling locally.* An unsuccessful or missing background result must
  fail closed, skip dependent target work and enter bounded owned cleanup. Preserve the
  original failure record and add a regression for the absent-result path before any
  further cluster use.
+ *Qualify setup beside a live owner.* With authorization, test one real captured TPU
  executable through the intended topology/compilation path while an owned serving
  process runs on the same host. Require serving health and output checks, successful
  compilation without acquiring the serving devices, actual cache output, and bounded
  cleanup on failure. CPU-only toys, idle-slice runs and environment flags are insufficient.
+ *Only then measure the full comparison.* Match loaded runtime, catalogue, manifest,
  effective cache coverage, engine settings and warm-up protocol. Require actual target
  reads/deserializations, and measure serving interference under successful compilation.
  Charge provisioning and preserve cache errors. Repeat/order-balance observations before
  generalizing or computing an eligible crossover.

A genuinely topology-only setup or coordination inside the existing owning runtime are
possible investigation directions. Neither is a demonstrated fix. The code-level entrypoints,
branch pins, local reproduction commands and acceptance gates are in
`compilation_overlap_handoff.md`. No new TPU experiment was run to write these documents.

= Provenance and closure

The raw campaign is sealed at `c22954164fcac8e1a2db32e8c59424af2e19eb9e` on
`overlap/restore-precompile` in `~/hetero-wt/OverlapCompile`. The raw artifact root is
`results/overlap/OverlapCompile/`. The source/launcher revision is
`9009fc7da5f03a57077ffdaa485d63a54d9190ef`; it is not loaded-runtime proof.

The archive `pre-sh-owned-bundle-graphs.tar.gz` is 67,423,350 bytes and contains
14,094 files and 4,698 descriptors. The audit matched all MLIR/options hashes and found
no duplicate-key payload variants. Its SHA-256 is:

```text
b6c3539ea40deeb66dfd5865d99d26e095b01552dc1ca56ea35a72c4129161b2
```

Audit `b6b5a24` qualified the failure evidence, archive and closure. The later immutable
artifact audit `2ea7783` verified 317 declared Git payloads, 318 paths including the
manifest, and all 25 gzip rehydrations. The raw graph archive is committed, not held
only in external storage. Post-run integrity does not retroactively establish runtime
hash enforcement or make the failed experiment successful.

The *recorded closure*, not a continuous assertion of current availability, showed:

- All 32 TPU chips available; no live groups or actors among 296 retained experiment groups.
- All 24 owned cache/bundle paths absent on all eight hosts; zero snapshot bytes.
- All 13 owned local processes gone; RCM neither alive nor zombie; cluster lock acquired
  and released; no dependent S-to-H arm queued.

The pre-existing shared registry and the inventory job's own supervisor were not leaks.
A subsequent run needs fresh ownership/inventory checks, authorization, the cluster lock
and `rcm_exec`. The closed campaign has no supervisor to resume.

The adjacent `reconfiguration_ici_evidence/OverlapFollowup/` package contains compact
reports, logs, tables and a hash manifest. `FollowupAudit/` contains the independent
receipts. The gated table generator is `helper/results-analysis` at `39d09ca`; its
recorded qualification was 21 tests plus 8 subtests. It emits no accepted overlap
performance comparison and no historical crossover for this campaign.

Sources: #evidence("FollowupAudit/followup_final_overlap_claims.json", [failure/cleanup audit]),
#evidence("FollowupAudit/followup_artifact_seal_claims.json", [immutable-artifact audit]),
#evidence("OverlapFollowup/manifest.json", [compact evidence manifest]), and
#evidence("OverlapFollowup/current_tables.md", [generated acceptance tables]).
