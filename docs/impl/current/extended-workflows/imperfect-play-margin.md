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
`src/llb/bench/agentic_memory_worst_case_probe.py` walks the same deterministic world with a
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

## What design validation does with it

`validate_surface_cells` certifies every predeclared guard against the margin-narrowed band
(`imperfect_play_guard_band` in `src/llb/bench/agentic_memory_fold_step_ladder.py`). The two bounds
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

The probe prices what the budget ALLOWS; `src/llb/bench/agentic_memory_extra_steps.py` reads what a
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
geometry rather than lucky: a cap-fitting guard puts the compact trigger inside the prefix the two
walks SHARE, so the first fold offers the summarizer the same bytes under both, and every measured
cap-fitting cell folds exactly once. The published invariance is therefore a statement about the
transcripts a real controller produces, and the check is what would catch a future cell -- a
repeatedly folding one, say -- where that stops holding.

Worst-case replay is defined only for the memory-chain task builder and refuses any other
(`worst_case_replay_controller`): the pipeline shapes end when their planted files run out, so
stalling one is a different task rather than a longer walk of the same one.

## Implementation map

| What | Where |
| --- | --- |
| Stalling controller, worst-case sequence / fold-input probe, `cap_peak_margin` | `src/llb/bench/agentic_memory_worst_case_probe.py` |
| Controller seam on the perfect-play probes | `src/llb/bench/agentic_memory_boundary_probe.py` |
| `imperfect_play_guard_band`, `guard_is_cap_fitting_under_imperfect_play` | `src/llb/bench/agentic_memory_fold_step_ladder.py` |
| The validation change and `depth_cap_peak_margin` | `src/llb/bench/agentic_memory_boundary_surface_cells.py` |
| Per-depth margin and per-cell observed steps in the analysis | `src/llb/bench/agentic_memory_boundary_surface.py` |
| Margin lines in the rendered surface | `src/llb/bench/agentic_memory_boundary_surface_report.py` |
| Observed extra steps read out of the run bundles | `src/llb/bench/agentic_memory_extra_steps.py` |
| Worst-case bound-invariance verdict and its roll-up | `src/llb/bench/agentic_memory_cap_audit.py` |
| Controller seam on the audit replay | `src/llb/bench/agentic_policy_change_replay.py`, `src/llb/bench/agentic_policy_change_audit.py`, `src/llb/bench/agentic_policy_change_tasks.py` |
| Tests | `tests/llb/bench/test_agentic_memory_worst_case_probe.py` |

Everything above is deterministic replay over the tool world, so it runs in `make ci` with no
backend and no GPU.
