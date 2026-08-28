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

```bash
make bench-agentic-context-compact-repeated-fold
```

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
| Tests | `tests/llb/bench/memory/test_agentic_memory_worst_case_probe.py`, `tests/llb/bench/memory/test_agentic_memory_two_fold_geometry.py`, `tests/llb/bench/memory/test_agentic_memory_repeated_fold_completion.py` |

The geometry, readings, gate, persistence, and marker ablation use deterministic fakes in `make ci`;
the completion values above come from the named CUDA run.
