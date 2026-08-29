# The imperfect-play safety margin

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

Every cap-fitting band in this repo used to be placed against one walk of the memory-dependent tool
world: the ORACLE walk, which plays the freshest workflow token every step and calls `finish` the
step after the workflow completes
([cap-fitting boundary surface](crossover-geometry.md#cap-fitting-boundary-surface)). That walk is
the SHORTEST transcript that finishes, so the peak it reports is the perfect-play peak -- and a real
controller is not bound by it. It re-sends a stale token, or keeps calling `advance` after the
workflow says stop (the failure the task prompt shouts about precisely because models do it), and
every wasted step appends one more transcript entry that every later prompt then carries. A guard
chosen just above the perfect-play peak can therefore still overflow on the run, and an overflowed
cap arm measures overflow rescue rather than cost.

The gap is bounded, because the STEP BUDGET bounds it. An episode runs at most `max_steps` steps, so
at most `max_steps - 1` transcript entries can stand behind its last prompt; perfect play uses
`depth + 1` of those steps, and whatever remains is what an imperfect controller can spend.
`src/llb/bench/memory/worst_case_probe.py` walks the same deterministic world with a
controller that spends ALL of it, which prices the gap exactly with no model and no GPU.

## Which imperfect walk is the worst one

The tool world decides, rather than the probe assuming. A wasted step appends whatever observation
the world returns for it, so the largest transcript comes from the wasted call with the largest
observation: `advance` past the end of the workflow returns the workflow-complete notice, which is
longer than the wrong-token line a stale token returns. Playing the chain perfectly and THEN
stalling on `advance` therefore keeps every workflow observation and makes every wasted entry the
largest one available, so `stalling_controller` is the worst case within the family of controllers
that only drive the workflow tool.

That family is the boundary of the claim, and a deliberate one: a controller free to call
`write_file` with arbitrary content grows a prompt without limit, so there is no worst case to state
over all controllers. The margin prices imperfect play of the TASK, never adversarial use of the
sandbox.

## The margin, measured

`cap_peak_margin` walks both controllers over one held geometry and reports both peaks plus the
difference. On the committed boundary-surface geometry (seven memory tasks, `pad_chars=1200`, an
800-char observation cap at head share 0.6, `max_steps_margin=4`):

| depth | perfect-play peak | worst-case peak | margin | ratio | perfect-play steps | budgeted extra steps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 8374 | 8827 | +453 | 1.054x | 7 | 3 |
| 10 | 11926 | 12379 | +453 | 1.038x | 11 | 3 |

The margin is the same 453 chars at both depths because it is three wasted steps at 151 chars of
capped transcript entry each -- the workflow-complete notice does not grow with depth, while the
peak it is measured against does, which is why the RATIO shrinks as the transcript gets longer. A
geometry whose budget leaves no room for a wasted step is refused (`MIN_BUDGETED_EXTRA_STEPS`)
rather than publishing a zero margin as a margin: there the worst case IS perfect play, and
certifying a guard against it would state a safety property the run cannot have.

**+453 is a rate, not a constant.** It looks like a constant only because every cap-fitting study
holds `max_steps_margin` at 4. Read at three step budgets on the same depth-10 geometry, the margin
is one price per wasted step and nothing else:

| `max_steps_margin` | budgeted extra steps | margin | chars per extra step |
| ---: | ---: | ---: | ---: |
| 4 | 3 | +453 | 151 |
| 6 | 5 | +755 | 151 |
| 8 | 7 | +1057 | 151 |

So a study that widens its step budget to give a slow controller more room widens the head-room its
guards must carry by exactly the same arithmetic; `margin_scaling` in
`src/llb/bench/memory/two_fold/reading.py` re-reads it per design rather than quoting the
number, and CI asserts the rate is one value across the read budgets.

## What design validation does with it

`validate_surface_cells` certifies every predeclared guard against the margin-narrowed band
(`imperfect_play_guard_band` in `src/llb/bench/memory/fold_step/ladder.py`). The two bounds
take DIFFERENT peaks, because the conservative side of each is a different walk:

- **cap must fit for the controller that actually runs**, so the lower bound is the WORST-CASE peak.
  A guard between the two peaks is cap-fitting for a perfect controller and for nobody else.
- **compact must still fire**, and it fires SOONER on a longer transcript, so the upper bound keeps
  the PERFECT-PLAY peak. Raising it to the worst-case peak would admit a guard whose trigger only
  the wasted steps reach. That is not a hypothetical: at depth 6 and `compact_share=0.5`, a
  17,000-char guard puts the trigger at 8500 -- above the 8374-char perfect-play peak and below the
  8827-char worst-case one -- so the oracle arm makes NO summary call while the stalling walk folds
  once. A cell placed there measures compaction only when the controller misbehaves, which is the
  opposite of a held-fixed activation floor.

So the margin narrows the band from below and never widens it from above. Every cell of all three
committed cap-fitting studies -- the boundary surface, the trigger collapse, and the fold-step
crossover -- clears the narrowed band unchanged, so no published geometry moves and no run is
invalidated by the stricter check. The refusal names the margin it applied, so a future design that
places a guard just above the perfect-play peak is told why that is not enough.

## What the run bundles say the model actually spent

The probe prices what the budget ALLOWS; `src/llb/bench/memory/extra_steps.py` reads what a
served model DID. Every compact-versus-cap bundle persists one case row per episode carrying its
`n_steps`, so the extra steps beyond perfect play are already on disk and need no run to recover.
`analyze_surface` attaches them per cell (`observed_extra_steps` per policy arm, plus
`observed_within_budgeted_margin`), and the rendered report prints the margin per depth beside the
largest extra step count any readable bundle at that depth recorded.

The reading is deliberately tolerant of a missing bundle. A published cell's manifest path is
absolute and belongs to the host that ran it, so a fresh clone, a moved `DATA_DIR`, or a CI box
holds the analysis without the bundles behind it. An unreadable bundle reports itself as unread with
a named reason, and `margin_is_covered` answers `None` rather than `True` -- "no arm could be read"
is not the same answer as "the observed steps fit". Extra steps are kept as measured rather than
floored at zero, because an episode the guard refused ends BEFORE the oracle walk does and clamping
would hide a run whose episodes never reached the geometry at all.

## The bound-invariance verdict, stated for the worst case

The summarize-input bound audit
([published values](published-values.md#published-crossovers-under-the-shipped-cap)) certifies a
published cell as bound-invariant by replaying it under both bounds and comparing prompts byte for
byte. Read under the oracle alone, that verdict covers the shortest transcript that finishes: a
longer real transcript could reach a summarize-input cap the oracle transcript never touched.

`agentic_memory_cap_audit` therefore states every verdict TWICE, over both walks. Each row carries
`worst_case_verdict` beside `verdict`, plus the worst-case elision columns and the first divergent
step, and `audit_summary` rolls up `n_worst_case_bound_invariant` /
`n_worst_case_bound_sensitive` plus `worst_case_only_sensitive` -- the cells the oracle walk calls
invariant and the longest admissible transcript does not. What still needs a GPU is decided by the
oracle verdict, because that is the walk the published numbers were measured on; the worst-case
count is what says how far the invariance CLAIM reaches.

On the committed surface the two verdicts agree cell for cell: five cells bit-identical under both
bounds and `surface-d10-g23000` bound-sensitive at 302 elided chars, under perfect play and under
imperfect play alike, with `worst_case_only_sensitive` empty. The mechanism is visible in the
geometry rather than lucky, and the next section is why: a cap-fitting guard puts the compact
trigger inside the prefix the two walks SHARE, and a cap-fitting cell folds exactly once, so the
one fold offers the summarizer the same bytes under both walks.

Worst-case replay is defined only for the memory-chain task builder and refuses any other
(`worst_case_replay_controller`): the pipeline shapes end when their planted files run out, so
stalling one is a different task rather than a longer walk of the same one.

## Why a cap-fitting cell folds exactly once

Every cap-fitting cell ever measured here folds ONCE per episode, and that is a property of the band
rather than of the cells that happened to be picked. Trigger hysteresis raises the trigger to the
FULL guard after the first summary (`episode_prompt.step_prompt`), so a second fold needs the
post-fold prompt to cross the whole guard. A fold replaces at least one transcript entry with a
strictly shorter summary line, so the post-fold prompt sits BELOW the walk's own peak -- and a
cap-fitting guard sits above that peak by construction, imperfect-play margin included. The
inequality has no room in it, and spending the whole step budget does not change it: the stalling
walk raises the peak, but the guard is above the raised peak too.

So the agreement above is structural. It is also a LIMIT: the published verdicts are all statements
about one-fold transcripts, and a repeatedly folding cell is necessarily outside the cap-fitting
band, below the cap peak, where the `observation_cap` arm overflows and no cost delta exists to
publish. CI asserts both halves -- every committed surface cell folds exactly once under both walks,
and the fixture below is refused if any of its cells clears its own cap peak. The same structure
limits the ROUTING rule as well as the invariance verdict, and
[the trigger rule is a one-fold rule](crossover-geometry.md#the-trigger-rule-is-a-one-fold-rule)
measures what the guard costs once an episode folds again.

## The regime the invariance verdict does NOT cover

`samples/benchmarks/agentic_compact_two_fold_geometry_design.json` is the committed fixture for the
repeatedly folding regime. Like the interaction fixture it publishes no number and is deliberately
absent from `AUDITED_DESIGN_PATHS`; what it carries is the validity limit on a statement made
elsewhere. Two depth-10 cells at `compact_share=0.8`, both far below the 11,926-char cap peak:

| cell | guard | guard / cap peak | oracle folds | worst-case folds | oracle verdict | worst-case verdict |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `twofold-d10-g7000` | 7000 | 0.59x | 2 | 3 | `bound_invariant` | **`bound_sensitive`** |
| `twofold-d10-g6500` | 6500 | 0.55x | 3 | 3 | `bound_invariant` | `bound_invariant` |

Verdict: **the invariance holds for one fold only**. On `twofold-d10-g7000` perfect play folds twice
and offers the summarizer 2646 then 5327 chars, which both bounds clear, so the oracle walk calls
the cell bit-identical under the retired and the shipped bound. The stalling walk spends its whole
step budget, grows the post-fold transcript far enough for a THIRD fold, and that fold offers 5625
chars -- 25 of which the `trigger` bound elides and the `window` bound does not. The two bounds stop
sending the same prompts at model call 15, on a transcript the oracle never produces.

The mechanism is sharper than "a longer transcript elides more": the per-fold margin
(`fold_input_margin_chars`) is **zero at every fold the two walks share**. Imperfect play does not
grow the folds the oracle already made -- it adds a later one, and that extra fold is where the
bounds separate. The control cell is what keeps this a finding about the geometry rather than about
the worst-case pass: it folds three times under both walks with byte-identical offered transcripts
and comes out invariant under both.

What this does NOT do is move any published number. Every published cell is cap-fitting, therefore
one-fold, therefore covered by the agreement above; the fixture states where a FUTURE verdict would
need re-reading -- a study that measures compact below the cap peak, or any change to compaction
hysteresis that lets a cap-fitting cell fold twice. The fixture's own declarations are predeclared
and checked (`declaration_drift`), so a runtime change that moves the regime reads as
`the_declared_geometry_no_longer_measures_what_it_declares` rather than quietly becoming the new
expectation.

## Completion through repeated folds

The geometry now has a real-model outcome reading. The committed design
`samples/benchmarks/agentic_compact_repeated_fold_completion_design.json` runs the two depth-10
fixture cells beside the cap-fitting `surface-d10-g14000` one-fold control. It runs `compact` alone:
the repeated-fold guards remain below the 11,926-char cap peak, so an `observation_cap` arm there
would measure overflow rescue rather than the completion cost of another summary hop. Every cell
uses the same two memory tasks, seed 727, `compact_share=0.8`, the shipped `window` summarize-input
bound, and an 8192-token served window.

The outcome is grouped by the fold count each REAL episode measured, not the oracle declaration.
The one-fold control is also an eligibility gate: every control case must complete and measure
exactly one fold before any later cell runs. A model that never reaches the control's fold cannot
anchor a decay claim. This refusal was necessary in practice: the first MamayLM 12B pilot completed
0/2 control cases and measured zero folds in both, so that aggregate is retained as a rejected
pilot and is not completion evidence (lookup key: run
`agent-context-policy-repeated-fold-completion-cost`, run id `1376371810ac`).

CUDA host evidence (2026-08-11, RTX PRO 3000 Blackwell Laptop GPU, 12 GiB): `qwen3:14b` on Ollama
at 22.64 tok/s. The control passed 2/2 with one fold per case, and all six shipped-policy cases
completed:

| cell | guard | oracle folds | measured folds | typed-marker completion | summary-only completion |
| --- | ---: | ---: | --- | ---: | ---: |
| `onefold-d10-g14000` | 14000 | 1 | `1, 1` | 2/2 | 2/2 |
| `twofold-d10-g7000` | 7000 | 2 | `3, 2` | 2/2 | 2/2 |
| `twofold-d10-g6500` | 6500 | 3 | `3, 3` | 2/2 | 2/2 |

Grouped by what actually happened, completion was 2/2 at one fold, 1/1 at two folds, and 3/3 at
three folds. Verdict: **completion is stable through the measured three-fold regime**. The
operator-facing fold-count limit therefore needs no value tighter than three on this model and task
set. The evidence does not authorize extrapolation past three folds or to an unqualified model.

The mechanism reading is causal rather than an inspection of prose. Each cell also runs an
evidence-only `preserve_memory_markers=false` arm: `fold_memory_markers` no longer copies the typed
`[memory: ...]` record into the running summary, leaving only what the model writes. That ablation
also completed 6/6, with no paired loss, so **the model-written summary was sufficient** here; the
typed marker was not required for any measured completion. The shipped default remains marker
preservation enabled -- the ablation shows the marker is not load-bearing on THIS task set, which is
a reason to keep it cheap, not a reason to drop it. What would overturn the stability verdict: a
cell that measures four or more folds, or a model that qualifies on the control and then loses a
case at two or three folds; neither exists in this run, which is why the claim stops at three.
Lookup key: run `agent-context-policy-repeated-fold-completion-cost`, run id `9ce2d99c7a89`.

Both limits that reading names -- one model family, two cases -- are lifted by the replication in
the next section, which is where the operator-facing fold-count rule now comes from.

```bash
make bench-agentic-context-compact-repeated-fold
```

## The fold-count rule, replicated on a second family

Two cases on one model is the shape of evidence that cannot separate "three folds are safe" from
"these two codes are easy", so the rule above was a ceiling result until something moved both
limits at once. `samples/benchmarks/agentic_compact_repeated_fold_replication_design.json` is that
design. It holds the three cells, the seed, the marker ablation, and the one-fold eligibility gate
EXACTLY as the completion design declares them -- the shared cell contract is checked by that
design's own validator, so a geometry drift fails in one place rather than two -- and changes only
what the claim rests on: twelve predeclared memory cases instead of two, a candidate roster the
gate qualifies families from, a predeclared floor of four paired cases per measured fold group, and
a per-family guard fit on the middle rung.

Three properties make it a replication rather than a second run:

- **Identical cases, one digest.** Every fold cell and both marker arms inside a family walk the
  same task set at the same seed, and both families reproduced the same `task_set_digest`. A fold
  group therefore differs from the one-fold control in FOLD COUNT and in nothing else.
- **Paired, not marginal.** Every higher-fold case is paired against the SAME task's one-fold
  outcome, so the reading is a per-task win/loss ledger carrying an interval, not two fractions
  that happen to be equal. Equal marginal rates are what the first run had; they say nothing about
  whether the same tasks were the ones that survived.
- **The rung is held equal, not the guard.** A ladder that fixes one character guard for every
  family reads whatever fold count each family happens to land on, and a family that lands
  elsewhere leaves a rung empty. What has to be equal across families is the MEASURED FOLD COUNT;
  the guard is only what each family needs in order to stand on it.

### Why the middle rung is fitted per family

Holding one guard fixed left a hole. Under the shared `twofold-d10-g7000` guard, `qwen3:14b` put 11
of 12 cases on two folds and `gemma4:e4b` put ONE, against a predeclared floor of four -- so the
middle rung of the ladder carried evidence on one family only, and the rule stood on the rungs
either side of an unpowered one.

The fix is to fit that one cell's guard to the family about to run it, from a measurement the run
already makes:

- **The measurement is free and it is the model's own.** `ContextTelemetry.summary_output_chars`
  records what the summarizer WROTE at each fold, taken before the typed facts are prepended, so it
  is the span the model chose rather than the span the geometry offered. The one-fold control folds
  exactly once per case, so its arm alone carries one fold length per case before any later cell
  runs. Running the control first is what makes the fit cost no extra episode.
- **The fit itself has no model in it.** `fold_length_controller` replays the same deterministic
  oracle walk the geometry probes use, with a summarizer that writes exactly the measured number of
  characters, over a predeclared band of candidate guards. What changes per family is the walk's
  input, not its determinism.
- **The band is predeclared and refused three ways.** It must contain the declared guard (so a
  family the shared constant suits reproduces the published geometry), stay above the deeper cell's
  6500 guard (so the fit cannot hand one cell the other's regime), and stay below the 11,926-char
  cap peak (where the cell is cap-fitting and folds once by construction). The committed band is
  6600 to 11500 in steps of 100.
- **Ties go to the declared guard, then to the middle of the widest run.** A family the shared
  constant already suits keeps the published geometry and only the family it does not suit moves --
  a fit that shuffled every guard would invalidate the comparison it exists to enable. Among the
  rest the centre of the widest contiguous run of equally good guards wins, which is the guard
  furthest from the length at which one more case folds again.

Only the middle cell is fitted. The cap-fitting one-fold control anchors the ladder and is never
fitted; the three-fold cell runs at its declared 6500.

### What ran, and what it measured

CUDA host evidence (2026-08-29, RTX 4060 Ti 16 GB): `qwen3:14b` at 22.45 tok/s and `gemma4:e4b` at
53.98 tok/s, Ollama `num_ctx=8192`, seed 727, 144 episodes in about 36 minutes. Both families
passed the one-fold control 12/12 with exactly one measured fold on every case, so both are
qualified and the gate consumed neither of the roster's fallbacks.

The fit moved both families to the same guard, and both measured what it predicted:

| family | measured fold length (median, range) | declared guard | fitted guard | predicted 2-fold cases | measured |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen3:14b` | 274 (263-304) | 7000 | 7900 | 12 | 12 |
| `gemma4:e4b` | 255 (237-369) | 7000 | 7900 | 12 | 12 |

Grouped by the fold count each REAL episode measured, under the shipped marker-preserving policy:

| family | measured folds | completed | 95% interval | pairs vs one fold | one-fold wins |
| --- | ---: | ---: | --- | ---: | ---: |
| `qwen3:14b` | 1 | 12/12 | [0.758, 1.000] | reference | - |
| `qwen3:14b` | 2 | 12/12 | [0.758, 1.000] | 12 | 0 |
| `qwen3:14b` | 3 | 12/12 | [0.758, 1.000] | 12 | 0 |
| `gemma4:e4b` | 1 | 12/12 | [0.758, 1.000] | reference | - |
| `gemma4:e4b` | 2 | 12/12 | [0.758, 1.000] | 12 | 0 |
| `gemma4:e4b` | 3 | 12/12 | [0.758, 1.000] | 12 | 0 |

Every cell landed all twelve of its cases on ONE fold count, on both families and in both marker
arms, so each rung is a clean group of twelve rather than a spread across two.

Verdict: **the three-fold rule extends across both qualified families, and every rung of the ladder
now carries it.** Not one of the 48 paired higher-fold cases completes at one fold and fails at two
or three. `ladder_fully_powered` is true and `underpowered_ladder_rungs` is empty: no fold count on
either family sits below the four-case floor, so the limit of three no longer rests on the rungs
either side of an unpowered one. The intervals are what the claim is worth: 12/12 is
`[0.758, 1.000]`, so this bounds a completion floor around 0.76 at three folds on both families,
not a proof of perfection. Nothing here authorizes a fourth fold.

### What the fit is worth, and where it is not calibrated

Two things the run says that the fitted numbers alone would hide:

- **The families are not far apart in summary length, so verbosity was not the whole story.**
  `gemma4:e4b`'s median fold length is 255 characters against `qwen3:14b`'s 274 -- shorter, not
  longer -- though its spread is wider (237-369 against 263-304). The earlier reading of the empty
  rung as "this family writes longer summaries" is not what the measurement shows; what the shared
  7000 guard sat on was a boundary where a case's fold count flips on a few dozen characters, and
  the wider-spread family was the one it flipped.
- **The probe ranks guards; it does not predict fold counts in absolute terms.** At the DECLARED
  7000 guard the probe predicts 0 of `qwen3:14b`'s 12 cases on the two-fold rung, where the real
  shared-guard run measured 11. The oracle walk models the post-fold prompt but not a family's step
  verbosity, so its per-guard count can be wrong by a whole fold. What it did correctly here was
  order the candidates and pick 7900, and the run then measured 12 of 12 on both families --
  `prediction_held` is the field that keeps that checkable, and a fit whose prediction fails is
  visible as a number rather than as a surprise in the fold table.

### The mechanism reading, and where the two families disagree

The first single-family run's marker ablation found the typed `[memory: ...]` record was not
load-bearing, which was a statement about one model. It does not transfer. On `qwen3:14b` it holds
-- `model_written_summary_sufficient`, zero marker wins over all 36 paired cases. On `gemma4:e4b`
it fails: `typed_memory_marker_required`, with marker preservation winning 1 of the 12 paired
three-fold cases and losing none. That single discordant case is not a powered claim on its own,
but the direction has now repeated: the earlier shared-guard run on the same family measured 2
marker wins of 23 three-fold cases, also with zero losses, for 3 one-sided discordant cases across
two runs. So the shipped default is not merely cheap-to-keep as the single-family reading
concluded -- on a second qualified family it is the difference between completing and losing the
code at three folds, and dropping it would have cost real completion that one model family could
not have shown.

What would overturn this: a cell that measures four or more folds; a third qualified family that
loses a paired case at two or three folds; a family whose fitted guard cannot reach the target rung
inside the declared band, which the run names
(`no_guard_in_the_declared_band_reaches_the_evidence_floor`) rather than smoothing; or a marker win
in the other direction on either family. Lookup key: run
`agent-context-policy-repeated-fold-completion-replication`, run id `8006514fb3f2`.

```bash
make bench-agentic-context-compact-repeated-fold-replication
```

The command is the same CLI entry point as the single-family study -- it dispatches on the design's
`study_kind` -- so there is one code path to keep correct and two committed designs that select the
reading.

## Implementation map

| What | Where |
| --- | --- |
| Stalling controller, worst-case sequence / fold-input probe, `cap_peak_margin` | `src/llb/bench/memory/worst_case_probe.py` |
| Controller seam on the perfect-play probes | `src/llb/bench/memory/boundary/probe.py` |
| `imperfect_play_guard_band`, `guard_is_cap_fitting_under_imperfect_play` | `src/llb/bench/memory/fold_step/ladder.py` |
| The validation change and `depth_cap_peak_margin` | `src/llb/bench/memory/boundary/surface_cells.py` |
| Per-depth margin and per-cell observed steps in the analysis | `src/llb/bench/memory/boundary/surface.py` |
| Margin lines in the rendered surface | `src/llb/bench/memory/boundary/surface_report.py` |
| Observed extra steps read out of the run bundles | `src/llb/bench/memory/extra_steps.py` |
| Worst-case bound-invariance verdict and its roll-up | `src/llb/bench/memory/cap_audit.py` |
| Controller seam on the audit replay | `src/llb/bench/policy_change/replay.py`, `src/llb/bench/policy_change/audit.py`, `src/llb/bench/policy_change/tasks.py` |
| Two-fold fixture contract and geometry probe | `src/llb/bench/memory/two_fold/fixture.py`, `samples/benchmarks/agentic_compact_two_fold_geometry_design.json` |
| Two-fold audit rows, margin scaling, and the validity reading | `src/llb/bench/memory/two_fold/reading.py` |
| Repeated-fold completion design and compact-only runner | `src/llb/bench/memory/repeated_fold/design.py`, `src/llb/bench/memory/repeated_fold/completion.py` |
| Completion/mechanism readings, report, and command | `src/llb/bench/memory/repeated_fold/reading.py`, `src/llb/bench/memory/repeated_fold/report.py`, `src/llb/cli/bench/memory/repeated_fold.py` |
| Replication design contract and roster | `src/llb/bench/memory/repeated_fold/replication_design.py`, `samples/benchmarks/agentic_compact_repeated_fold_replication_design.json` |
| Per-fold paired uncertainty, evidence floor, and the cross-family reading | `src/llb/bench/memory/repeated_fold/replication_reading.py` |
| Two-family runner, aggregate, and roster driving | `src/llb/bench/memory/repeated_fold/replication.py`, `src/llb/bench/memory/repeated_fold/replication_report.py`, `src/llb/cli/bench/memory/repeated_fold_replication.py` |
| Measured fold length telemetry and the fold-length probe | `src/llb/bench/agentic/context.py`, `src/llb/bench/agentic/context_summary.py`, `src/llb/bench/memory/boundary/probe.py` |
| Per-family two-fold guard fit, its predeclared band, and the runner's guard seam | `src/llb/bench/memory/repeated_fold/guard_fit.py`, `src/llb/bench/memory/repeated_fold/completion.py` |
| Whether every rung of every family's ladder carries evidence | `src/llb/bench/memory/repeated_fold/ladder_coverage.py` |
| Tests | `tests/llb/bench/memory/test_agentic_memory_worst_case_probe.py`, `tests/llb/bench/memory/test_agentic_memory_two_fold_geometry.py`, `tests/llb/bench/memory/test_agentic_memory_repeated_fold_completion.py`, `tests/llb/bench/memory/test_agentic_memory_repeated_fold_replication.py`, `tests/llb/bench/memory/test_agentic_memory_repeated_fold_guard_fit.py` |

The geometry, readings, gate, persistence, and marker ablation use deterministic fakes in `make ci`;
the completion values above come from the named CUDA run.
