# Completion through repeated compact folds

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

Every published cap-fitting cell folds exactly once, which is a property of the band rather than of
the cells that were picked
([the imperfect-play safety margin](imperfect-play-margin.md#why-a-cap-fitting-cell-folds-exactly-once)).
So every statement the cap-fitting studies make is a statement about a ONE-FOLD transcript, and the
question this page answers is the one they cannot: what does an episode lose when the summarizer
folds again, and again, on the same task. The geometry it runs on is the repeatedly folding fixture
[the invariance verdict does not cover](imperfect-play-margin.md#the-regime-the-invariance-verdict-does-not-cover).

## The first reading: one family, two cases

The committed design
`samples/benchmarks/agentic_compact_repeated_fold_completion_design.json` runs the two depth-10
fixture cells beside the cap-fitting `surface-d10-g14000` one-fold control. It runs `compact` alone:
the repeated-fold guards remain below the 11,926-char depth-10 cap peak
([the margin, measured](imperfect-play-margin.md#the-margin-measured)), so an `observation_cap` arm there
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
- **The run order is derived, not declared.** The design declares its cells as a ladder, in
  fold-count order, because that is how a reader sees the rungs. The RUN has a constraint the
  declaration cannot express -- every cell the fit measures against must already have run -- so
  `fitted_cell_order` puts the cap-fitting control first, the never-fitted cells next, and the
  fitted cell last. The runner refuses any order that does not lead with the control, because that
  cell is also the eligibility gate.
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

CUDA host evidence (2026-08-29, RTX 4060 Ti 16 GB): `qwen3:14b` at 21.01 tok/s and `gemma4:e4b` at
50.02 tok/s, Ollama `num_ctx=8192`, seed 727, 144 episodes in about 40 minutes, run in the order
control -> three-fold -> fitted. Both families passed the one-fold control 12/12 with exactly one
measured fold on every case, so both are qualified and the gate consumed neither of the roster's
fallbacks. An earlier run of the same design in the declared cell order, before the fitted cell was
moved behind the three-fold one, measured the identical fold groups and the identical marker
ablation -- so the reordering below costs the outcome readings nothing.

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
- **The probe ranks guards; its absolute per-guard count is only as good as its slack.** The
  earlier shared-guard run measured 11 of `qwen3:14b`'s 12 cases on the two-fold rung at the
  DECLARED 7000 guard, and 1 of `gemma4:e4b`'s. A flat replay of the control's own fold length
  predicts 0 there on both. What it did correctly was order the candidates and pick 7900, and the
  run then measured 12 of 12 on both families. The three sections below are what turns that from
  luck into a number, and what says how far the number reaches: the replay's step model is
  measured rather than assumed, its fold length is replayed at the span the fold actually offered,
  and the distance between a guard's prediction and the point where it flips is reported per
  guard -- which is also what says when a per-guard count must be stated as an interval rather
  than a number.

### The step half of the replayed walk, measured rather than assumed

A fold count is decided by two things: how much of the guard the running summary already spends,
and how fast the rest of it is spent as the transcript grows. The fit measures the first (the fold
length) and used to ASSUME the second, by answering every non-summary step with the oracle's own
minimal call. A family whose steps appended more than that would grow its context at a rate the
probe never saw, and its per-guard count would be wrong for a reason no field recorded.

`ContextTelemetry.step_entry_chars` closes that. It records what each step appended to every later
prompt -- the rendered `- advance({...}) ->` span plus its trailing separator, observation
excluded, because the observation is the WORLD's contribution and the deterministic walk reproduces
it exactly. `fold_length_controller` takes the measured length beside the measured fold length and
pads its replayed call to it (never below it: the oracle's call is the shortest one that plays this
world), so the transcript the probe walks grows at the family's own measured rate. The padding
rides on an argument the workflow tool ignores, so the padded walk plays the identical world.

**The measurement came back at the oracle's own length, on both families.** Every one of the 120
control steps `qwen3:14b` and `gemma4:e4b` each walked rendered at exactly 36 characters, which is
what the oracle walk renders at over this geometry (`oracle_step_entry_chars`), so
`step_length_reading` is `the_family_steps_render_at_the_oracle_walk_length` and the fit padded
nothing. The mechanism this was built to catch is structurally absent here rather than merely
small: only the tool NAME and ARGUMENTS survive into a transcript entry, so a model's raw output
length -- its reasoning, its preamble, whatever it wrote around the call -- never reaches a later
prompt at all. A family can only grow the transcript faster than the oracle by calling a tool with
bigger arguments, and on this task the only useful call carries one workflow token.

That is a negative result, and it is worth the field it cost: the replayed growth rate is now
something the run measured on the family that ran, so a future family that DOES call wider is
priced automatically instead of silently mis-predicted, and the residual calibration error below
cannot be attributed to this half of the walk.

### The span the fold length was measured at

The step half of the walk is the family's own, and the fold length half was not. The control folds
ONCE, at the last step, over the whole ten-entry transcript; the cell the length is replayed at
folds a three-entry span and then a four-entry one. A summarizer offered less writes less, so a
length carried across unchanged is a length measured against the wrong offer -- and the error is
guard-dependent, because a tighter guard folds sooner, over fewer entries, and is wrong by more.

The second measurement that fixes it costs no episode either. The ladder already runs a
never-fitted cell at a fixed 6500 guard whose folds cover much shorter spans, so running it BEFORE
the fitted cell leaves the run holding two measured (offered span, written length) points instead
of one. `summary_fold_input_chars` was already the span each fold offered and `summary_output_chars`
already the length written against it; what changed is the order the cells run in and a replay that
reads both.

Two points give a SLOPE and nothing more, and `fold_span.py` refuses to be more than that:

- **Level per case, slope per family.** Each case keeps its own control-measured fold length as its
  level, so the fit still predicts a case COUNT rather than collapsing twelve cases into one
  replayed episode; the slope -- one number for the family -- is what moves that level to the span
  each replayed fold actually offers (`summary_offered_chars` recovers it from the summarize
  prompt, and equals the telemetry's own span whenever the window bound elides nothing, which it
  does not on any of these cells).
- **Never extrapolated.** Outside the two measured spans the nearer measured span stands. Two
  points are a slope, not a curve, and a fold length at a span no cell of the run ever offered is
  not a measurement.
- **One span is refused a slope rather than given a zero one.** A run that measured a single span
  keeps the flat replay it always had, named `only_one_fold_span_was_measured_so_the_fold_length_is_replayed_flat`.
  The design contract refuses a second source that names the fitted cell, folds no more often than
  it, or runs after it.

### What the second span was worth, and what it was not

| family | control fold | the 6500 cell's folds | slope | fitted cell's spans | it wrote | flat replay off by | span replay off by |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| `qwen3:14b` | 11802 chars offered, 274 written | 4185 offered, 212 written | +0.0081 | 3954-6872 | 244 | -30 | **+11** |
| `gemma4:e4b` | 11802 chars offered, 255 written | 4253 offered, 284 written | -0.0038 | 3954-6893 | 253 | -2 | **-22** |

Both error columns are matched PER FOLD -- each of the fitted cell's 24 folds scored against the
length the replay gives at that fold's own span, median of the errors -- because a two-fold episode
folds a short span and then a long one, and a single median over both lands on whichever cluster
holds the middle value.

**On the family it was diagnosed on it works, and at the declared guard it nearly closes.** On
`qwen3:14b` the replayed length moves from 30 characters too long to 11 too short, and the predicted
count at the DECLARED 7000 guard moves from 0 of 12 to **10 of 12**, against the 11 the earlier
shared-guard run measured. The remaining gap is one case, and it is exactly accounted for: the
declared guard's `fold_count_margin_chars` is 10 characters, and the replay is 11 out. **The guard
where the count still diverges is 7000, and it diverges by one character more than that guard's
margin can absorb.**

**It does not generalize, and the run says so rather than averaging it away.** `gemma4:e4b`'s two
points slope the OTHER way: it writes 284 characters when offered 4253 and 255 when offered 11802,
so on that family the correction makes the replayed length worse -- from 2 characters out to 22 --
and the declared guard still predicts 0 against the 1 the shared-guard run measured. The aggregate
states this as `span_slope_reading`:
`the_qualified_families_disagree_on_the_sign_of_the_span_correction`, with both slopes named. Two
families that disagree on the SIGN have measured two habits, not one rate, so the slope is
per-family bookkeeping and an unrun family inherits nothing from either.

**Nothing the ladder publishes moves.** Both families still fit to 7900, both still predict 12 of
12 two-fold cases and measure 12, and `fit_prediction_reading` is still
`every_fitted_guard_predicted_the_case_count_its_family_measured`. At 7900 the margin is the whole
200-character scan on both families, so an 11- or 22-character error is nowhere near enough to move
the count. The correction changes what can be read at guards the fit did NOT pick; it does not
change the guard it picks.

### Why the declared guard stays hard to count, and what a count is stated as instead

The residual is not a missing third calibration point. Two things the run's own rows show, and both
are re-read from the persisted per-case fields rather than measured again:

- **The flip window at 7000 is narrower than the family's case-to-case spread.** `qwen3:14b`'s
  control fold lengths run 263-304 -- a 41-character spread -- while the predicted count at 7000
  flips within 10 characters. `gemma4:e4b` is wider on both: a 237-369 spread against a
  30-character flip window. No per-family constant, level or slope or both, resolves a window that
  narrow out of that much case-to-case variation.
- **A case's verbosity does not transfer between cells.** Pair each case's control fold length
  against its own first fold at the fitted cell and the correlation is **-0.03 on both families**
  over 12 paired cases each. The level the fit carries across is not a property of the case.

The second reading is what decides the level rule, and it decides it against the per-case reading:
summing "how many of these twelve lengths land on the rung" is not twelve per-case predictions, it
is the family's own SPREAD sampled twelve times, read as a rate and multiplied back to a count. So
the level is a family constant plus an irreducible width, and the count is reported as an interval
-- the Wilson interval on that rate scaled back to cases and rounded outward. The point value does
not move, which is why the guard the fit picks does not move either.

| family | control spread | guard | flip window | point count | interval | shared-guard run measured |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `qwen3:14b` | 41 chars (263-304) | 7000 declared | 10 | 10 of 12 | **6-12** | 11 |
| `qwen3:14b` | 41 chars (263-304) | 7900 fitted | 200 | 12 of 12 | **9-12** | 12 |
| `gemma4:e4b` | 132 chars (237-369) | 7000 declared | 30 | 0 of 12 | **0-3** | 1 |
| `gemma4:e4b` | 132 chars (237-369) | 7900 fitted | 200 | 12 of 12 | **9-12** | 12 |

**The interval covers what the point estimate missed, on both families.** The earlier shared-guard
run measured 11 of `qwen3:14b`'s 12 cases and 1 of `gemma4:e4b`'s on the two-fold rung at the
declared 7000 guard. The point counts there are 10 and 0 -- wrong on both -- and 11 and 1 both sit
inside the intervals. At the fitted 7900 guard the interval is 9-12 on both families and both
measured 12, so the published ladder reads exactly as it did before.

**A guard is refused as a count when its flip window is inside the spread.** That comparison is a
named reading on every fit record rather than a caveat a reader has to make:
`the_flip_window_is_narrower_than_the_case_to_case_fold_length_spread` at the declared 7000 guard on
both families, and `the_flip_window_is_wider_than_the_case_to_case_fold_length_spread` at the fitted
7900. A refused guard is still RANKED -- that is what the fit exists to do and it is unaffected --
it just cannot be handed to an operator as a number. A family that writes one length on every case
has no measured spread to width a count with, and gets
`no_fold_length_spread_was_measured_to_widen_a_count_with` instead of a zero-width interval
pretending to be certainty.

The cross-family verdict is `level_transfer_reading`:
`no_qualified_family_carries_a_per_case_fold_length_level`, with both correlations named. One family
that DID carry a per-case level would make the level worth keeping per case and the interval
conservative; neither does.

Every number in this section is emitted by the shipped command rather than derived by hand. CUDA
host evidence (2026-08-29, RTX 4060 Ti 16 GB): `qwen3:14b` at 21.99 tok/s and `gemma4:e4b` at
53.26 tok/s, Ollama `num_ctx=8192`, seed 727, the same 144 episodes in the same control ->
three-fold -> fitted order. It reproduced the run above field for field -- both families qualified
12/12, both fitted to 7900, both predicted and measured 12, the same `+0.00814` and `-0.00384`
slopes, `ladder_fully_powered` true, `model_written_summary_sufficient` on `qwen3:14b` and
`typed_memory_marker_required` on `gemma4:e4b` -- so nothing the ladder publishes moves and the
interval, the countability refusal and the level correlation are additions to what a run reports,
not changes to what it measures. Lookup key: run
`agent-context-policy-repeated-fold-completion-replication`, run ids `b4f5f00462f6` (the fold
groups and the span slopes above) and `1502561eda1a` (the intervals, the refusals, and the level
correlations).

What would overturn this reading: a third qualified family whose slope agrees in sign with one of
these two, which would turn a disagreement into a majority; a task set whose fold lengths vary less
between cases than the flip window at the guard being read, which would make a point count readable
where it is not now; a family whose paired levels correlate, which would put the per-case level back
in; or a per-case span correction that recovers the transfer the -0.03 correlation says is absent.

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

**What bounds it: the two mechanism arms run as blocks, marker arm first.** Every cell walks all
of `typed_marker` and then all of `model_summary_only` against one stateful endpoint, so "won the
case" and "ran first" are the same column for the arm that wins here -- and all 3 discordant cases
across the two runs sit in that first arm. The summary-fold adoption study measured this seam
directly rather than assuming it away, and found that reaching the fold does NOT track position
while paired completion tracks the ARM
([summary-input bounds and elision](summary-input-elision.md#arm-order-is-balanced-not-fixed)), so
the effect here most likely is the marker. But this lane cannot yet say so from its own evidence,
and it reads in the conservative direction anyway: the caveat could only weaken a case for KEEPING
the shipped default, never manufacture one. Removing the confound here is not the same one-line
change the adoption study made -- the cell ladder is ordered by a one-fold control gate that stops
the run when the control fails, so a balanced schedule has to keep that gate.

The cell-POSITION half of that seam did move, in the useful direction, and for a reason that had
nothing to do with the marker. Fitting the middle rung's guard requires the fitted cell to run
after the never-fitted cell it calibrates against, so the ladder now runs one fold, then three,
then two. Run position no longer increases with fold count, which means an endpoint that drifted
over the run would no longer show up as a monotone fold-count effect. Both families' fold groups
and both marker readings came back identical to the run in the declared order, so nothing here
rests on the change -- but the confound between "later in the run" and "more folds" is gone, and
only the arm confound above remains.

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
| Repeated-fold completion design and compact-only runner | `src/llb/bench/memory/repeated_fold/design.py`, `src/llb/bench/memory/repeated_fold/completion.py` |
| Completion/mechanism readings, report, and command | `src/llb/bench/memory/repeated_fold/reading.py`, `src/llb/bench/memory/repeated_fold/report.py`, `src/llb/cli/bench/memory/repeated_fold.py` |
| Replication design contract and roster | `src/llb/bench/memory/repeated_fold/replication_design.py`, `samples/benchmarks/agentic_compact_repeated_fold_replication_design.json` |
| Per-fold paired uncertainty, evidence floor, and the cross-family reading | `src/llb/bench/memory/repeated_fold/replication_reading.py` |
| Two-family runner, aggregate, and roster driving | `src/llb/bench/memory/repeated_fold/replication.py`, `src/llb/bench/memory/repeated_fold/replication_report.py`, `src/llb/cli/bench/memory/repeated_fold_replication.py` |
| Measured fold length telemetry and the fold-length probe | `src/llb/bench/agentic/context.py`, `src/llb/bench/agentic/context_summary.py`, `src/llb/bench/memory/boundary/probe.py` |
| Measured per-step transcript growth (`step_entry_chars`) and the padded replay | `src/llb/bench/agentic/context.py`, `src/llb/bench/memory/boundary/probe.py` |
| Per-family two-fold guard fit and its predeclared band | `src/llb/bench/memory/repeated_fold/guard_fit.py` |
| The run order the fit needs, and the runner's guard seam | `src/llb/bench/memory/repeated_fold/fit_seam.py` (`fitted_cell_order`, `guard_resolver`), `src/llb/bench/memory/repeated_fold/completion.py` |
| Span-aware replayed fold length, its two calibration points, and its refusals | `src/llb/bench/memory/repeated_fold/fold_span.py`, `src/llb/bench/agentic/context_summary.py` (`summary_offered_chars`), `src/llb/bench/memory/boundary/probe.py` |
| The model-free replay a candidate guard is scored on, and its per-guard margin | `src/llb/bench/memory/repeated_fold/guard_replay.py` |
| The fit's prediction error and the cross-family calibration verdicts | `src/llb/bench/memory/repeated_fold/fit_calibration.py`, `src/llb/bench/memory/repeated_fold/replication_report.py` |
| Whether the per-case fold-length level transfers, the count interval, and the not-countable refusal | `src/llb/bench/memory/repeated_fold/level_transfer.py` |
| Whether every rung of every family's ladder carries evidence | `src/llb/bench/memory/repeated_fold/ladder_coverage.py` |
| Tests | `tests/llb/bench/memory/test_agentic_memory_repeated_fold_completion.py`, `tests/llb/bench/memory/test_agentic_memory_repeated_fold_replication.py`, `tests/llb/bench/memory/test_agentic_memory_repeated_fold_guard_fit.py`, `tests/llb/bench/memory/test_agentic_memory_repeated_fold_fold_span.py`, `tests/llb/bench/memory/test_agentic_memory_repeated_fold_level_transfer.py` |

The geometry, readings, gate, persistence, guard fit, and marker ablation use deterministic fakes in
`make ci`; every completion value above comes from the named CUDA run.
