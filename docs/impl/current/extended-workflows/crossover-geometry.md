# Cap-Fitting Boundary And Crossover Geometry

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

## Cap-fitting boundary surface

`make bench-agentic-context-compact-memory-boundary-surface` answers what one cap-fitting cell
cannot: WHERE compact stops repaying its summary call. It pins the family the replication qualified,
holds the typed memory tasks, observation cap, `compact_share`, activation floor, task count, and
summarizer-inclusive accounting fixed, and varies ONLY transcript depth and the prompt guard over a
predeclared grid of cap-fitting cells. The committed design is
`samples/benchmarks/agentic_compact_memory_boundary_surface_design.json`, and the pinned family is
re-qualified against the unchanged token-chain control before any cell runs.

A cap-fitting cell is usable only inside a narrow band, and that band needs NO model to compute. The
memory-dependent tool world is deterministic, so an oracle controller that always plays the next
workflow token reproduces the exact prompt sequence a perfect controller would send
(`src/llb/bench/agentic_memory_boundary_probe.py`): depth 6 peaks at 8,374 prompt chars and depth 10
at 11,926, and the probe's cap totals (13,258 and 27,343 model-input tokens per task) are the
numbers the host then measured. A guard BELOW the peak overflows cap; a guard at or above
`peak / compact_share` never lets compact fire. Design validation refuses any cell outside that open
band, a depth that does not predeclare cells on both sides of the crossover, a grid that drops the
replication's anchor geometry, and a declared window too narrow to carry the widest guard -- all in
CI, with no GPU.

The interpolation rule is predeclared with the grid: read the compact-minus-cap total model-input
delta on the guard axis, take the FIRST adjacent pair of cost-separated cells whose means have
opposite signs, and interpolate linearly to the zero crossing. A cell whose cost sign is not
readable is skipped rather than blocking a bracket around it, and a depth with no sign change
reports a BOUND (the crossing lies above or below the tested guards) instead of extrapolating. Every
cell must also keep its preconditions -- zero cap overflows, zero compact overflows, compaction
above the activation floor, paired completion, and both policies above the cell completion floor --
or it is reported invalid with the named reason instead of bending the crossover.

Core locations are `src/llb/bench/agentic_memory_boundary_probe.py` (the oracle episode walk that
measures the peak), `src/llb/bench/agentic_memory_fold_step_ladder.py` (`usable_guard_band` and
`guard_is_cap_fitting`, the band arithmetic over that peak),
`src/llb/bench/agentic_memory_boundary_gate.py` (the direction-aware lower-is-better cost gate,
shared with the replication in [compact versus cap](compact-versus-cap.md)),
`src/llb/bench/agentic_memory_boundary_surface_cells.py` (grid contract and per-cell validity),
`src/llb/bench/agentic_memory_boundary_crossover.py` (the interpolation and the routing lines),
`src/llb/bench/agentic_memory_boundary_surface.py` (design, run, and analysis),
`src/llb/bench/agentic_memory_boundary_surface_report.py`,
`src/llb/cli/bench/category_agentic_memory_boundary_surface.py`, and
`tests/llb/bench/test_agentic_memory_boundary_surface.py`, whose fake-model pass over the committed
grid proves every predeclared cell keeps cap fitting and compact firing.

```bash
make bench-agentic-context-compact-memory-boundary-surface
```

CUDA host evidence (2026-08-02, RTX 4060 Ti 16 GB): `mistral-small3.1:24b` on Ollama with
`num_ctx=8192`, seven depth-matched memory tasks per cell, `compact_share=0.5`, an 800-char
observation cap, 88 episodes at 11.21 tok/s over about 50 minutes. The pinned family re-passed the
unchanged depth-10 control at 4/4. Every cell completed 7/7 under both policies with zero overflows
and exactly one compaction per compact episode, so each cost delta is one summary call against
smaller later controller prompts. The aggregate is
`$DATA_DIR/agentic-compact-memory-boundary-surface/20260802T154634.305722Z-c668820b6c4d/manifest.json`;
its source cell bundles are under `.data/agentic-compact-vs-cap/20260802T1506*` through
`.data/agentic-compact-vs-cap/20260802T1546*`.

| cell | depth | guard | cap input tok | compact input tok | paired d(input tok) | cost pairs | side |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `surface-d6-g12000` | 6 | 12000 | 13258.0 | **12466.0** | -792.0 [-815.3, -762.0] | 0/7 | compact |
| `surface-d6-g14000` | 6 | 14000 | 13258.0 | **13132.1** | -125.9 [-138.4, -112.7] | 0/7 | compact |
| `surface-d6-g15500` | 6 | 15500 | **13258.0** | 14312.6 | +1054.6 [+1049.9, +1061.7] | 7/0 | cap |
| `surface-d10-g14000` | 10 | 14000 | 27343.0 | **22884.7** | -4458.3 [-4501.4, -4411.4] | 0/7 | compact |
| `surface-d10-g20000` | 10 | 20000 | 27343.0 | **24709.6** | -2633.4 [-2652.7, -2614.9] | 0/7 | compact |
| `surface-d10-g23000` | 10 | 23000 | **27343.0** | 28867.9 | +1524.9 [+1517.3, +1533.9] | 7/0 | cap |

Every cost row is unanimous across its seven pairs (two-sided exact sign-test p = 0.015625), every
completion delta is +0.000, and compact adds exactly +1.000 model call per task in all six cells.
All six cells landed on the side the design predeclared. The interpolated crossings are:

| depth | cap peak prompt | bracket | crossover guard | guard / peak |
| ---: | ---: | --- | ---: | ---: |
| 6 | 8374 | [14000, 15500] | 14160 | 1.69 |
| 10 | 11926 | [20000, 23000] | 21900 | 1.84 |

Routing rule: for memory-dependent transcripts, use `compact` while the prompt guard is below about
1.7-1.8x the transcript's cap peak prompt, and `observation_cap` above it. Verdict: **surface
mapped**. The replication's single cap-fitting cell was not a universal result and not a knife-edge
one either -- it sits inside a measured compact-cheaper region that ends at a crossover both tested
depths agree on in peak-relative terms (spread 0.15x). The mechanism is visible in the numbers: cap
costs exactly what the deterministic probe says regardless of guard, while compact's cost rises with
the guard because a later trigger folds a bigger transcript and leaves fewer steps to spend the
smaller prompt on. This does not change the shipped default; the medium-search shape still prefers
`observation_cap`.

## The routing rule lives on the trigger axis

`make bench-agentic-context-compact-trigger-collapse` closes the axis question the surface leaves
open. The surface swept the prompt guard at ONE `compact_share`, but the policy never reads the
guard directly: it folds when the prompt crosses `int(guard * compact_share)`, so the reported
guard is a stand-in for that trigger. The study measures the difference -- FAMILIES of cells that
hold the trigger fixed while moving share and guard inversely, plus one contrast family that holds
the guard at 12,000 chars and moves the trigger, which is the positive control that the measurement
can see a trigger change at all. The committed design is
`samples/benchmarks/agentic_compact_trigger_guard_collapse_design.json`.

Equivalence is predeclared on the scale the operator pays rather than on interval overlap: the
paired intervals here are tighter than any difference worth acting on, so overlap would reject a
practical equivalence over a few tokens. A family collapses when its SPREAD of compact-minus-cap
total model-input tokens stays within 2% of what the cap baseline costs at that depth AND every
member lands on the same cost side; the contrast family must EXCEED that same band or the study
reports `no_resolving_power` instead of a collapse. The probe also predicts, with no model, which
step each trigger folds at (`first_fold_step` over the deterministic cap prompt sequence), which is
the mechanism the claim rests on.

Core locations are `src/llb/bench/agentic_memory_trigger_collapse_design.py` (family/axis contract
and the cap-fitting band per pair), `src/llb/bench/agentic_memory_trigger_collapse_reading.py`
(vocabulary, fold-step annotation, family spread, and the reading),
`src/llb/bench/agentic_memory_trigger_collapse.py` (run and analysis),
`src/llb/bench/agentic_memory_trigger_collapse_report.py`,
`src/llb/cli/bench/category_agentic_memory_trigger_collapse.py`, and
`tests/llb/bench/test_agentic_memory_trigger_collapse.py`.

```bash
make bench-agentic-context-compact-trigger-collapse
```

CUDA host evidence (2026-08-02, RTX 4060 Ti 16 GB): `mistral-small3.1:24b` on Ollama with
`num_ctx=8192`, the same seven memory tasks per cell, eight cells at 10.64 tok/s over about an hour,
the pinned family re-passing the control at 4/4. Every cell completed 7/7 under both policies with
zero overflows, one compaction per compact episode, and +1.000 model calls per task. The
fold-annotated aggregate is
`$DATA_DIR/agentic-compact-trigger-guard-collapse/20260802T171326.479910Z-eed680be10aa/manifest.json`.

| family | kind | depth | share / guard | trigger | fold step | d(input tok) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `d6-trigger-7000` | equal trigger | 6 | 0.40 / 17500 | 7000 | 6 | -125.9 |
| `d6-trigger-7000` | equal trigger | 6 | 0.50 / 14000 | 7000 | 6 | -125.9 |
| `d6-trigger-7000` | equal trigger | 6 | 0.60 / 11667 | 7000 | 6 | -125.9 |
| `d6-guard-12000` | equal guard | 6 | 0.40 / 12000 | 4800 | 4 | -873.6 |
| `d6-guard-12000` | equal guard | 6 | 0.50 / 12000 | 6000 | 5 | -792.0 |
| `d6-guard-12000` | equal guard | 6 | 0.60 / 12000 | 7200 | 6 | -125.9 |
| `d10-trigger-10000` | equal trigger | 10 | 0.45 / 22224 | 10000 | 9 | -2633.4 |
| `d10-trigger-10000` | equal trigger | 10 | 0.60 / 16667 | 10000 | 9 | -2633.4 |

Both equal-trigger families have a spread of **0.0 tokens** -- bit-identical deltas across a 1.5x
and a 1.3x range of prompt guards -- against bands of 265.2 and 546.9 tokens, while the contrast
family moved 747.7 tokens across the same band. Verdict: **the trigger ratio collapses the
surface**. Share and guard act only through their product, so a crossover measured at one share
converts to any other: at depth 6 the 14,160-char crossover guard is a 7,080-char trigger (0.85x the
cap peak prompt) and at depth 10 the 21,900-char guard is a 10,950-char trigger (0.92x), giving the
portable form -- use `compact` while `compact_share * guard` stays below about 0.85-0.92x the
transcript's cap peak prompt, and `observation_cap` above it.

The fold step says why, and it is the more useful statement of the rule: the trigger reaches the
transcript ONLY by choosing which step compacts, so the cost delta is a STEP function of the
trigger, not a smooth one. The contrast family's 7,200-char trigger folds at step 6 exactly like the
7,000-char triggers and reproduces their -125.9 to the token; its 4,800- and 6,000-char triggers
fold at steps 4 and 5 and cost -873.6 and -792.0. Two independent runs agree to the token as well:
the surface's own (0.5, 14000) and (0.5, 20000) cells are the trigger-7000 and trigger-10000 values
above. An operator therefore does not need a grid for a new window -- only the trigger that lands on
the intended fold step. The shipped `compact_share` is unchanged.

## The crossover is a fold step, not a char guard

`make bench-agentic-context-compact-fold-step` restates the crossover on the axis the mechanism
actually has. The surface interpolated a char guard, and the trigger collapse then showed why that
number cannot be read literally: the trigger reaches the transcript ONLY by choosing which step
folds, so every trigger inside one step's interval produces the identical transcript. The crossover
is the boundary between two fold steps, and the interpolated char value is an artifact of fitting a
continuous rule to a discrete mechanism. The committed design is
`samples/benchmarks/agentic_compact_fold_step_crossover_design.json`.

The geometry is the inverse of the fold-step prediction and needs no model.
`fold_step_trigger_interval` returns the half-open `[low, high)` trigger interval whose every value
folds at one step -- `low` is the largest earlier step prompt, `high` is the step's own prompt --
and `fold_step_guard_interval` converts it to prompt guards through the runtime's own truncating
`int(guard * share)` rather than a float inverse. A step whose prompt does not exceed the running
maximum before it is UNREACHABLE (no trigger selects it), which `reachable_fold_steps` answers; the
ladder a design is placed against is `foldable_fold_steps`, the reachable steps an episode can
actually fold AT. The two differ by step 1: a guard under the first prompt selects it, and its
prompt is built from zero entries, so `compact_state` finds nothing older to summarize and returns
unchanged. A cell declared there would measure a `compact` arm that never compacts and publish it as
a fold-step cost. That used to be refused only indirectly -- a step-1 guard sits far below the cap
peak, so the per-CELL cap-fitting rule rejected it with a message about the usable guard band, and
only for a geometry where those two facts happen to contradict each other. The ladder rule now
refuses it directly, naming the ladder the design should have declared against.

Placement is what lets the grid tell a step change apart from a smooth slide, and all four rules are
checked in CI with no GPU: every declared cell must fold at the step it claims, the tested steps must
be ADJACENT on the foldable ladder, the guards inside one step must span at least half of that
step's guard interval (otherwise "same step, same cost" is measured over two nearly identical
guards), and the guards on either side of a step change must straddle it within 8 chars (otherwise
the flip is localized no better than the old bracket was). Cell preconditions -- cap fits, compact
fires above the activation floor, completion paired -- are the surface's unchanged gate, and a cell
whose measured fold step drifts from its declared one aborts the analysis rather than being re-read.

That arithmetic is separated from the episode walk that feeds it, because the two cost wildly
different things to call and a shared module name hid which one an import line was reaching for.
`src/llb/bench/agentic_memory_fold_step_ladder.py` holds the pure half -- the trigger/guard interval
inverse, `first_fold_step`, the reachable and foldable ladders, `live_entries_at_fold_step`,
`compaction_trigger_chars`, `smallest_guard_reaching`, `measured_cap_peak`, `usable_guard_band`, and
`guard_is_cap_fitting` -- every one a function of a prompt-size SEQUENCE plus a share or a guard,
cheap enough for the placement rules, the summarize-cap ladder, and the policy-change band solver to
sweep in a tight loop. `src/llb/bench/agentic_memory_boundary_probe.py` keeps the half that RUNS
episodes (`oracle_controller`, `oracle_compacting_controller`, `compact_fold_input_probe`,
`cap_prompt_sequence`, `cap_peak_prompt_chars`), each call a full workflow walk per task. No
re-export layer bridges them: every caller imports from the module that owns the name, so an import
line says which cost it is paying.

The tests mirror that split. `tests/llb/bench/test_agentic_memory_fold_step_ladder.py` holds the
interval algebra on synthetic prompt sequences -- the trigger interval and its inverse, the guard
interval through the runtime truncation, and the unreachable-versus-unfoldable step -- with no design
file, no episode, and no model in reach, so a failure there names the ladder rather than the study
that happened to exercise it. Its second half is the EDGES, which the callers reach only by accident
and which used to fail two layers up: a step outside the sequence (0 or one past the end, through
both `fold_step_trigger_interval` and `fold_step_guard_interval`) is refused rather than read as a
neighbour's interval; an unreachable step read directly is the empty interval itself (`low >= high`,
with the triggers on either side folding sooner and later); a trigger no prompt exceeds -- including
one equal to the largest prompt, since the crossing is STRICT -- gives `first_fold_step` of `None`;
`smallest_guard_reaching` and `usable_guard_band` refuse a share outside `(0, 1]` and answer at the
closed end (share 1 makes the guard the trigger, and makes the usable band empty rather than
refused); `usable_guard_band` refuses a non-positive peak, which is a probe that measured no prompt
at all; `measured_cap_peak` names the GEOMETRY behind such a walk, including the empty one a bare
`max` could only call an empty iterable; and an empty sequence folds at no step, has no reachable or
foldable step, and puts every step out of range.
`tests/llb/bench/test_agentic_memory_fold_step_crossover.py` keeps the study: the committed design's
placement, the validation refusals, the readings over fixture rows, and the end-to-end run under
perfect play. The usable band keeps its own home in
`tests/llb/bench/test_agentic_memory_boundary_surface.py`, where it is asserted against the
probe-measured cap peak it is only meaningful next to. The two policy-change interaction tests
(`test_agentic_policy_change_interaction_{band,cap}.py`) import ladder names to BUILD their geometry
rather than to test it, so they stay where they are.

Every one of those edges is reachable only through a CALLER, and every caller that passes a
probe-measured value into the arithmetic now states what it does when the ladder refuses, with the
choice asserted rather than implied. They do not all answer the same way, because what the refusal
MEANS differs per caller.

`_step_row` (`agentic_memory_fold_step_rows.py`) TRANSLATES. The step is a measured cell property
(`predicted_fold_step`) and the sequence is the depth's own oracle walk, so a step the sequence
cannot answer for means the two describe different geometries -- a probe that measured nothing, or
rows grouped against another depth's ladder. Both now read as `cells [...] cannot be read against
the measured N-step prompt sequence: they sit at fold step k, off its foldable ladder [...]`, and a
group whose trigger no prompt exceeds (`predicted_fold_step` of `None`, which used to reach the
ladder as a comparison against `None` and die as a `TypeError`) reads as `they fold at no step of
it`.

The cap PEAK is translated once, for everyone. `usable_guard_band` refuses a non-positive peak, but
the peak is the `max` of a probe walk, so a geometry whose oracle episodes end before their first
prompt failed inside that `max` instead -- one layer lower still, and as the bare builtin `max()
iterable argument is empty`, which names neither the geometry nor what the peak was wanted for.
`measured_cap_peak(sequence, geometry=...)` in `agentic_memory_fold_step_ladder.py` is the reduction
every cap-fitting caller now takes, and it states the fact above the band rather than inside it:
`depth 6 measured no prompt under perfect play (0 steps), so it has no cap peak and no usable guard
band to place cells in`. The `geometry` label is the caller's own vocabulary, so a surface depth and
a summarize-cap arm ladder read differently while the refusal stays one refusal. Seven readers share
it: `cap_peak_prompt_chars` (the probe's own reduction), `_validate_band`
(`agentic_memory_boundary_surface_cells.py`), `fold_step_cap_peaks` and `_validate_ladder`
(`agentic_memory_fold_step_design.py`), `_validate_ladder` (`agentic_memory_summary_cap_design.py`),
`_arm_row` (`agentic_memory_summary_cap_rows.py`), and `analyze_summary_cap`
(`agentic_memory_summary_cap.py`). In the two ladder validators the read moved AHEAD of the shape
rule as well: an unmeasured depth used to be reported as declared steps that are not adjacent on the
foldable ladder `[]`, which points the operator at the design instead of at the geometry.

The SHARE is deliberately left alone in all of them: it is a `held_fixed` value the design states
verbatim, so `compact share must be in (0, 1]` already names what the operator wrote as precisely as
a translation would.

`_interpolated_row` (`agentic_memory_crossover_restatement_rows.py`) TRANSLATES, and its refusal is
the sharpest of the set because the step comes from a COMMITTED artifact rather than from a probe of
the same geometry. Every other caller reads a step it measured itself; this one interpolates a fresh
guard against a freshly measured sequence and then asks where the PUBLISHED fold step's guard
interval was. The two can therefore disagree about the task world without either being malformed --
and that disagreement is exactly the drift the restatement exists to catch, not an argument error.
It read as the arithmetic's bare `fold step 10 is outside a 6-step sequence`, naming neither the
number nor the study that published it, and now reads as `the published
compact_memory_boundary_surface crossover at depth 10 is stated at fold step 10, which the
re-measured 6-step prompt sequence no longer has (its foldable ladder is [2, 3, 4, 5, 6]): the task
world moved under the published number, so there is no interval left to restate it inside`. The
check is the FOLDABLE ladder rather than the sequence bounds, because a published crossover is a
cost measured at a step an episode folded at: a step still in range but no longer reachable (an
earlier prompt grew past it) has been retired by the geometry just as completely.

`share_bound_conditions` (`agentic_policy_change_interaction_conditions.py`) PROPAGATES, and the
reachable fault was one level up. Every step the solver states conditions at comes from
`foldable_fold_steps` of the same sequence, so a step `fold_step_trigger_interval` cannot answer for
is a fabricated `StepGeometry` rather than a geometry anyone declared, and the ladder's own words are
the accurate ones. What WAS reachable is silent rather than loud: a geometry with no foldable step
yields no bands at all, and `format_band_report` then prints the same "no fold step separates the two
readings at this depth" that a genuinely contradictory coupling gets -- a derivation over an episode
nothing was measured over. `separating_guard_bands` refuses that before any condition is stated,
naming the reachable steps the geometry does offer, the way the placement rule refuses an empty
foldable ladder.

Each choice is tested with its own caller, so a failure names the study whose vocabulary moved: the
shared peak read itself in `test_agentic_memory_fold_step_ladder.py` (both refusing sequences, the
two `geometry` vocabularies, and the band built on what it returns); the step-row translation and
both fold-step peak readers in `test_agentic_memory_fold_step_crossover.py` (with the positive
control that the same cells still read as one step row on their own depth's sequence); the surface's
peak translation and its propagated share in `test_agentic_memory_boundary_surface.py`; the arm
ladder's design-side and analysis-side peaks in `test_agentic_memory_summary_cap.py` (the analysis
peak describes the geometry rather than the cells, so it is read even for an ineligible family); the
published-step translation in `test_agentic_memory_crossover_restatement.py`, against the control
that the committed number still restates inside its own interval; and both band-solver halves in
`test_agentic_policy_change_interaction_band.py`.

Core locations are `src/llb/bench/agentic_memory_fold_step_ladder.py` (the interval arithmetic
above, shared with the collapse study, the summarize-cap arms, and the policy-change band solver),
`src/llb/bench/agentic_memory_boundary_probe.py` (the oracle prompt sequence it is computed over),
`src/llb/bench/agentic_memory_fold_step_design.py` (the placement contract),
`src/llb/bench/agentic_memory_fold_step_rows.py` (step and depth rows, and the caller-side refusal
for a step the measured sequence cannot answer for),
`src/llb/bench/agentic_memory_boundary_surface_cells.py` (the depth-side walk behind the shared peak
read), `src/llb/bench/agentic_memory_crossover_restatement_rows.py` (the published-fold-step
refusal), `src/llb/bench/agentic_policy_change_interaction_band.py` (the empty-foldable-ladder
refusal),
`src/llb/bench/agentic_memory_fold_step_reading.py` (vocabulary, readings, routing lines),
`src/llb/bench/agentic_memory_fold_step.py` (run and analysis),
`src/llb/bench/agentic_memory_fold_step_report.py`,
`src/llb/cli/bench/category_agentic_memory_fold_step.py`,
`tests/llb/bench/test_agentic_memory_fold_step_ladder.py` (the interval arithmetic), and
`tests/llb/bench/test_agentic_memory_fold_step_crossover.py` (the study).

```bash
make bench-agentic-context-compact-fold-step
```

CUDA host evidence (2026-08-02, RTX 4060 Ti 16 GB): `mistral-small3.1:24b` on Ollama with
`num_ctx=8192`, seven depth-matched memory tasks per cell, `compact_share=0.5`, eight cells at
10.56 tok/s over about 68 minutes. The pinned family re-passed the unchanged depth-10 control at
4/4; every cell
completed 7/7 under both policies with zero overflows, one compaction per compact episode, and all
eight landed on the side the design predeclared. The aggregate is
`$DATA_DIR/agentic-compact-fold-step-crossover/20260802T185212.038607Z-24e73063cba6/manifest.json`.

| cell | depth | guard | trigger | fold step | cap tok | compact tok | d(input tok) | side |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fold-d6-step6-lo` | 6 | 13136 | 6568 | 6 | 13258.0 | **13132.1** | -125.9 | compact |
| `fold-d6-step6-hi` | 6 | 14910 | 7455 | 6 | 13258.0 | **13132.1** | -125.9 | compact |
| `fold-d6-step7-lo` | 6 | 14912 | 7456 | 7 | **13258.0** | 14312.6 | +1054.6 | cap |
| `fold-d6-step7-hi` | 6 | 16746 | 8373 | 7 | **13258.0** | 14312.6 | +1054.6 | cap |
| `fold-d10-step10-lo` | 10 | 20240 | 10120 | 10 | 27343.0 | **26434.4** | -908.6 | compact |
| `fold-d10-step10-hi` | 10 | 22014 | 11007 | 10 | 27343.0 | **26541.0** | -802.0 | compact |
| `fold-d10-step11-lo` | 10 | 22016 | 11008 | 11 | **27343.0** | 28698.4 | +1355.4 | cap |
| `fold-d10-step11-hi` | 10 | 23040 | 11520 | 11 | **27343.0** | 28878.6 | +1535.6 | cap |

Verdict: **fold-step boundary confirmed** at both depths. The cost side changes between two guards
**2 chars apart** -- 14910 and 14912 at depth 6, 22014 and 22016 at depth 10 -- while guards up to
1834 chars apart inside one step stay on the same side. The step change moves 1180.4 tokens
at depth 6 and 2300.8 at depth 10, against within-step bands of 265.2 and 546.9.

| depth | last compact-cheaper fold step | trigger interval | guard interval (share 0.5) | within-step residual |
| ---: | ---: | --- | --- | ---: |
| 6 | 6 | `[6568, 7456)` | `[13136, 14912)` | 0.0 tok |
| 10 | 10 | `[10120, 11008)` | `[20240, 22016)` | 180.1 tok |

The routing rule an operator applies exactly: **fold no later than step k** -- keep
`compact_share * guard` below step k's own cap prompt (7456 chars at depth 6, 11008 at depth 10),
which at `compact_share=0.5` is a guard below 14912 and 22016 chars. Both interpolated crossovers
land INSIDE the cheap step and name a point at which nothing changes: 14160 sits 752 chars below the
depth-6 step change (5.0% low) and 21900 sits 116 chars below the depth-10 one. Neither is wrong as
an approximation -- both fall in the correct step -- but only the step boundary is where the cost
actually moves, and only it converts to another `compact_share` without re-deriving anything.

One term survives inside a step, and the run isolates it. At depth 6 the whole cost is bit-identical
across the guard interval; at depth 10 it moves 180.1 tokens, of which 171.0 is the summarize call
and 9.1 is later controller prompts. The cause is that the summarize call's input cap was the trigger
at the time of this run, so a larger trigger inside one step fed the summarizer more of the folded
transcript -- and the summary it returned was then carried by every later prompt. Depth 6 folds a
transcript smaller than either cap, so nothing was trimmed and the residual is exactly zero. The
residual is 8% of the depth-10 step change and stays far inside the equivalence band, so it does not
move the boundary; each cell row records `compact_mean_controller_prompt_tokens` and
`compact_mean_compaction_prompt_tokens` so the split is readable rather than inferred. This does not
change the shipped `compact_share` or the guard-axis interpolation the surface publishes. The bound
that produced the residual is what
[the summarize-input cap](#the-summarize-input-cap-is-step-aligned) then replaced; this study's
design pins `summary_input_cap: "trigger"` so the numbers above reproduce unchanged.

## The summarize-input cap is step-aligned

`make bench-agentic-context-compact-summary-input-cap` closes the one term the fold-step study left
moving. The compact policy has to bound the summarize call's input -- that input is the transcript
that just blew the step prompt, so an uncapped summarizer is the one call in the loop guaranteed to
overflow -- but the bound it used, the compaction trigger, is the ONLY part of the compact cost that
is not a step function of the fold step. Two guards inside one step fold the identical transcript and
send bit-identical controller prompts, yet feed the summarizer different amounts of it, and the
summary that comes back is then carried by every later prompt. The bound also ELIDES the folded
transcript head-and-tail once it outgrows the trigger, so a transcript that would have fit the window
was summarized with its middle missing.

The shipped bound is now `summary_input_cap="window"`: the resolved prompt budget minus the summary
template's own overhead, which includes the elision marker `trim_observation` writes ON TOP of the
cap it is given (a bound that ignores the marker sends a summarize prompt a few chars over the
window -- exactly the silent truncation the cap exists to prevent). It is a property of the resolved
budget alone, so it does not move with `compact_share` and the folded transcript is summarized at its
own size whenever it fits. The legacy `trigger` bound stays selectable, and the boundary-surface,
trigger-collapse, replication, transfer, and fold-step designs all pin it explicitly so their
published numbers reproduce against the current runtime instead of silently re-measuring a different
summarizer. The committed design is
`samples/benchmarks/agentic_compact_summary_input_cap_design.json`.

The study is two ARMS over ONE fold-step ladder -- the same depth-10 ladder the fold-step crossover
published, with the two bounds as the only difference -- and it reads two independent things: whether
the step-aligned bound drives the within-step residual to zero WITHOUT moving the fold step the
routing rule is stated on, and whether the span the trigger bound elided was carrying completion.
The second question needs an elision to exist, and that is decided with no model at all:
`compact_fold_input_probe` walks the deterministic tool world with an oracle controller and a fixed
summary reply, and reports what each arm offers the summarizer and how much its bound elides. Design
validation refuses a ladder whose reference arm elides nothing (no trimmed span to price), a
step-aligned arm that elides anything (it is not step-aligned), and a step-aligned arm whose
summarize input is not identical across the guards inside one step. The fold-step placement rules --
declared step, adjacency on the foldable ladder, within-step guard span, straddle gap -- are shared
verbatim with the crossover study (`src/llb/bench/agentic_memory_fold_step_placement.py`), so a
residual measured here is on exactly the scale that study publishes.

Core locations are `src/llb/bench/agentic/context.py` (`SUMMARY_INPUT_CAPS`, the elision telemetry),
`src/llb/bench/agentic/context_budget.py` (`summary_input_cap_chars`),
`src/llb/bench/agentic/episode.py` (the bound resolver),
`src/llb/bench/agentic_memory_boundary_probe.py` (`compact_fold_input_probe`),
`src/llb/bench/agentic_memory_fold_step_placement.py` (shared placement rules),
`src/llb/bench/agentic_memory_summary_cap_design.py`,
`src/llb/bench/agentic_memory_summary_cap_reading.py`,
`src/llb/bench/agentic_memory_summary_cap_rows.py`,
`src/llb/bench/agentic_memory_summary_cap.py`,
`src/llb/bench/agentic_memory_summary_cap_report.py`,
`src/llb/cli/bench/category_agentic_memory_summary_cap.py`, and
`tests/llb/bench/test_agentic_memory_summary_cap.py`.

```bash
make bench-agentic-context-compact-summary-input-cap
```

CUDA host evidence (2026-08-05, RTX 4060 Ti 16 GB): `mistral-small3.1:24b` on Ollama with
`num_ctx=8192`, the fold-step study's seven depth-10 memory tasks per cell, `compact_share=0.5`,
eight cells (four guards under each bound) at 10.63 tok/s over about 79 minutes. The pinned family
re-passed the unchanged depth-10 control at 4/4. Every cell completed 7/7 under both policies with
zero overflows, exactly one compaction per compact episode, and all eight landed on the side the
design predeclared. The aggregate is
`$DATA_DIR/agentic-compact-summary-input-cap/20260805T185837.832318Z-0f86b57558a1/manifest.json`.

| arm | cell | guard | fold step | summarizer offered | elided | compact tok | d(input tok) | side |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `trigger` | `cap-d10-step10-lo` | 20240 | 10 | 10494 | **374** | 26434.4 | -908.6 | compact |
| `trigger` | `cap-d10-step10-hi` | 22014 | 10 | 10494 | 0 | 26541.0 | -802.0 | compact |
| `trigger` | `cap-d10-step11-lo` | 22016 | 11 | 11802 | **794** | 28698.4 | +1355.4 | cap |
| `trigger` | `cap-d10-step11-hi` | 23040 | 11 | 11802 | **282** | 28878.6 | +1535.6 | cap |
| `window` | `cap-d10-step10-lo` | 20240 | 10 | 10494 | 0 | **26541.0** | -802.0 | compact |
| `window` | `cap-d10-step10-hi` | 22014 | 10 | 10494 | 0 | **26541.0** | -802.0 | compact |
| `window` | `cap-d10-step11-lo` | 22016 | 11 | 11802 | 0 | **28953.3** | +1610.3 | cap |
| `window` | `cap-d10-step11-hi` | 23040 | 11 | 11802 | 0 | **28953.3** | +1610.3 | cap |

Verdict: **the step-aligned cap is an exact step function**. The within-step residual goes from
180.1 tokens to **exactly 0.0** -- both guards inside fold step 10 and both inside fold step 11 now
cost the same to the token, controller and summarizer alike -- while the boundary the routing rule is
stated on stays at fold step 10 and the step change grows slightly, from 2300.8 to 2412.3 tokens
against the same 546.9-token band. Every measured summarizer input and elided span reproduces the
model-free probe's prediction to the character, so the mechanism was settled before the GPU ran and
the run only confirmed it costs what the geometry says.

| arm | step-10 spread | step-11 spread | residual (summarizer / controller) | last compact-cheaper step |
| --- | ---: | ---: | --- | ---: |
| `trigger` | 106.6 | 180.1 | 180.1 (171.0 / 9.1) | 10 |
| `window` | **0.0** | **0.0** | **0.0 (0.0 / 0.0)** | 10 |

The elision was free: the reference arm cut up to 794 chars out of the summarizer's input and the
paired compact completion between the arms is +0.000 [+0.000, +0.000] over 28 pairs (0 wins, 0
losses, 28 ties, sign-test p = 1.0000) -- a `flat` reading, so the trimmed span carried nothing the
summary needed on this shape. Pin the cap for predictability, not for completion.

What the trigger cap WAS doing is visible in the compact column: a trimmed summarize input is a
smaller prompt, so the elision quietly discounted compact's own measured cost -- by 106.6 tokens at
fold step 10 and 180.1 at fold step 11, always in compact's favor, and always at the cells the
routing rule is read from. The `window` numbers are the undiscounted ones. Both arms still land on
the predeclared sides at every guard, so the depth-10 fold-step crossover is unchanged; what that
discount does to every OTHER published crossover is settled in
[the restatement](published-values.md#published-crossovers-under-the-shipped-cap) below.
