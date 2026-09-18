# Goal Prompt — RCM Agent-Ergonomics Revision

> Paste the block below into `/goal`, or invoke `/goal` and point it at this file.
> It is written to be self-contained for an agent with **no prior conversation context**.

---

## Objective

Revise the OMP Reverse Control Mechanism runtime in
`/usr/local/google/home/hongtaozhang/git/admission-control-vllm-clean/omp_rcm/`
so that an LLM agent driving it does **minimal work** per diagnostic trap.

**Success metric.** The trap → fix → resume cycle drops from ~7 tool calls with unbounded
token ingestion to **3 tool calls with bounded output**:

```
rcm_exec run x.py   →  (exit 2, self-contained trap banner)
edit the file
rcm_exec resume     →  (exit 2 → next trap, or exit 0 → done)
```

Do not stop until Tier 1 and Tier 2 are complete, verified, and committed. Tier 3 is
in scope if time permits; if you skip it, leave a written handoff note.

## Verified baseline — do not re-derive

These were confirmed by inspection of the current tree. Trust them, but re-check any line
number that has drifted.

| Finding | Location |
|---|---|
| `stream_daemon_output` writes **every byte** of daemon stdout to the caller. Used by `cmd_run`, `cmd_reload`, `cmd_patch`, `cmd_steer`. A vLLM run emits tens of thousands of lines before the first trap. | `omp_rcm/cli.py:159-219` |
| `cmd_wait` already implements a correct events-only filter, but it is duplicated and divergent from the above. | `omp_rcm/cli.py:456-464` |
| The trap banner's `Location:` scans the traceback in reverse and takes the **deepest** frame — for vLLM/Ray faults that is inside `site-packages`, i.e. not fixable code. | `omp_rcm/cli.py:109-121` |
| Trap payload has no source context, no frame locals, no recurrence count. | `omp_rcm/core.py:477-502` |
| `state_sample` serializes every `StateStore` key with no count cap. | `omp_rcm/core.py:496-500` |
| `trap_timeout_s` defaults to `None` → a trap blocks **forever**, holding the TPU placement group. | `omp_rcm/core.py:818, 1286-1292` |
| `RETRY` restarts the stage function from line 1, repeating weight loads and placement-group construction. | `omp_rcm/core.py:1330-1341` |
| `rcm_exec run` hard-fails when a stale daemon exists, costing an extra `abort` round-trip. | `omp_rcm/cli.py:224-235` |
| `cmd_wait` returns `0` for "no daemon running", indistinguishable from success. | `omp_rcm/cli.py:415-416` |
| `rcm_exec reload` and `rcm_exec patch` appear **only in AGENTS.md** — zero callers in `scripts/`, `benchmarks/`, `admission_control/`, or tests. Tests drive `HOT_PATCH`/`LIVE_PATCH` through the programmatic steering dict instead. | audit; `tests/test_suite.py:281`, `demos/demo_gemma4_workload.py:199` |
| `/tmp/omp_rcm` and `/tmp/omp_rcm/traps` are mode **750**; `/tmp` is world-traversable. | `stat` |

## Decisions already made — implement, do not relitigate

1. **`rcm_exec run` stays blocking by default.** The agent harness auto-backgrounds long
   commands and wakes reactively on exit, so blocking costs zero polling calls. This is
   only viable once output is capped — treat A1/A2 as hard prerequisites, not parallel work.
2. **Capture frame locals**, with: name-based redaction `(?i)(key|token|secret|password|credential|auth)`;
   size-based elision (`len(repr(v)) > 200` → a type/shape summary, with a fast path for
   numpy/torch/jax arrays); and `chmod 700` on `BASE_RCM_DIR`. No per-stage opt-out.
3. **Fold `reload` and `patch` into `resume`** as thin aliases. Leave the `HOT_PATCH` /
   `LIVE_PATCH` wire protocol completely untouched — the tests depend on it.
4. **Memoization invalidation is explicit only.** `ctx.once(key, fn, deps=(...))`, default
   never-invalidate, plus `rcm_exec resume --fresh` as the clear-all hammer. Do not infer a
   dependency graph.

---

## Tier 1 — Output discipline + self-sufficient trap payload

Strictly additive. Touches `cli.py` and the trap writer only. No engine behavior change.

- **A1** `stream_daemon_output` defaults to the events-only filter when `RCM_AGENT=1` or
  stdout is not a TTY. `--stream` opts into the full firehose.
- **A2** Cap output per invocation (~300 lines / 24 KB) with a
  `… N lines suppressed — rcm_exec logs --tail 50` footer. Emit a one-line heartbeat every
  ~5 min (stage + clocks) so a multi-hour backgrounded run is distinguishable from a hang.
- **A3** Extract one shared `EVENT_MARKERS` constant; delete the duplicated filter logic.
- **A4** Cap `state_sample` (suggest 40 keys, 200 chars each) and apply the Decision-2
  redaction to it.
- **B1** Blame-frame selection: deepest frame under the repo root and **not** matching
  `(site-packages|dist-packages|/\.venv|/usr/lib/python)`; fall back to the deepest frame.
- **B2** Embed a ±15-line source window around the blame line with a `>` gutter marker,
  read via `linecache` at trap time.
- **B3** Frame locals for the blame frame and its immediate parent, under Decision 2.
- **B4** Condensed first-party-only traceback in the banner; full traceback stays in
  `traps/latest.json`.
- **F1** Documented exit codes: `0` complete · `1` failed · `2` trapped · `3` no daemon ·
  `124` timeout. Fixes the `wait`-returns-0 ambiguity.

**Acceptance.** Given a stage that raises inside a vendored library called from first-party
code, the banner names the *first-party* frame, shows its source and locals, and a fresh
agent can write the fix without any additional file read.

## Tier 2 — One resume verb + safety rails

- **C1** `rcm_exec resume`: snapshot `{module: mtime}` for loaded first-party modules at
  trap entry; on resume, diff, `HOT_PATCH` each dirty module, `RETRY`, then wait. The agent
  never names a file or picks an op.
- **C2** `resume --skip` / `--abort` / `--fresh`. `reload` and `patch` become aliases.
  `steer` remains the raw escape hatch.
- **C3** If `resume` finds no dirty modules and no explicit op, **warn loudly** — this is the
  common agent mistake of editing a file the daemon never imported.
- **B5** Recurrence tracking: `sha1(stage, exc_type, blame_file:line)` counted in
  `traps/history.json`, reset on a fresh `run`. Banner shows the occurrence ordinal. At ≥3,
  the directive changes to "stop patching, inspect state, consider MUTATE_CONFIG or SKIP_STAGE".
- **B6** Exactly one fully-resolved, copy-pasteable `NEXT:` command in the banner. No placeholders.
- **E1** Default `trap_timeout_s` to 1800s → auto-`ABORT`, release the placement group, emit
  `RUN_FINISHED status=ABORTED reason=TRAP_TIMEOUT`.
- **E2** `rcm_exec run --replace`, and auto-reap a dead-but-pidfile'd daemon.
- **F2** `--json` on `status`/`wait`/`logs`, plus `rcm_exec trap [--json]` to re-read the
  latest trap without re-waiting (needed after context compaction).

**Acceptance.** A full trap → edit → `rcm_exec resume` → next-trap cycle completes in three
tool calls with no file path or op named by the agent.

## Tier 3 — Cheap retries + prompt slimming

- **D1** `ctx.once(key, fn, deps=())` memoized in `StateStore` so retries skip completed
  sub-steps.
- **D2** Banner lists satisfied `once` keys as "will NOT re-run on resume".
- **E3** Heartbeat timestamp in `live_status.json`; `status` distinguishes running from zombie.
- **F3** Freeze and document the `[OMP_EVENT: <NAME> k=v …]` grammar in one place; replace
  the scattered string literals that depend on it.
- **G** Rewrite the RCM section of `AGENTS.md` down to ~5 lines, since the banner is now
  self-describing.

## Separately: a config smell to fix

`scripts/submit_measurement_audit.py:62` sets `expected_s = sum(step timeouts)` ≈ 7200s, so
the SLA arm of `check_normality` (`core.py:437-446`) fires at 2× ≈ 4 hours and the warning at
2.5 hours — both effectively dead. `expected_s` is being used as a worst-case budget where the
detector wants a realistic estimate. Fix it or file it, but call it out explicitly; it is
independent of the rest of this work.

---

## Verification

**No TPU hardware is required.** The existing suites under `reverse_control_lab/tests/`
(`test_suite.py`, `test_blocking_trap_proof.py`, `test_watchdog_hang_proof.py`) use synthetic
stages with `expected_s≈0.2` and exercise the trap, watchdog, hot-patch, and steering paths.

Per repo policy, run **only** tests covering what you changed — no full sweeps, no
coverage-padding tests. Add a focused test per tier:

- Tier 1: a stage raising from a vendored frame; assert the banner names the first-party
  frame and includes the source window and locals.
- Tier 1: assert output stays under the cap for a stage that prints 10k lines.
- Tier 2: modify a module on disk mid-trap; assert `resume` detects exactly it, hot-patches,
  and the retry executes the new code.
- Tier 2: assert `trap_timeout_s` auto-aborts.
- Tier 3: assert a `once` key is not recomputed across a retry.

Also do a real end-to-end smoke run of `reverse_control_lab/demos/demo_four_stage_pipeline.py`
under `rcm_exec run` and confirm the observed cycle is genuinely 3 calls.

## Non-goals

- Do not change the `HOT_PATCH` / `LIVE_PATCH` / `MUTATE_CONFIG` / `SKIP_STAGE` / `ABORT`
  wire protocol.
- Do not touch `ray_job.py` supervision logic beyond what `--json` output requires.
- Do not add simulation or mock execution paths of any kind.
- Do not restructure `StageRunner.run`'s stage loop beyond what D1 requires.

## Conventions

- **Version control is `jj`, never raw `git`.** Use `jj status`, `jj diff`, `jj log`,
  `jj commit -m`, `jj describe -m`.
- Commit each tier separately with a descriptive message.
- Tag any CL description with `TAG=agy` and `CONV=4e67af42-e5de-444c-a438-a1538fdc407c`,
  with tag lines at the very bottom.
- Preserve all existing comments and docstrings unrelated to your changes.
- Read `AGENTS.md` at the repo root before starting.

## Definition of done

1. Tiers 1 and 2 implemented, with focused tests passing.
2. A demonstrated 3-call trap → fix → resume cycle on a real `rcm_exec run`.
3. `AGENTS.md` updated to match the new surface.
4. Work committed via `jj`, one commit per tier.
5. A short report covering: measured before/after tool-call count and output size, anything
   deferred, and any decision above you had to deviate from — with the reason.
