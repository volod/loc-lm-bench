"""Where the separating guards are, solved from the interval arithmetic -- no GPU.

The interaction fixture only separates the two policy-change readings inside a narrow guard band,
and the band is a property of the task world: shift every prompt size by a few chars and it moves.
So the band is solved rather than searched for, and these are the tests that keep the solver honest
-- exact at both edges, empty where nothing can separate, and refused for a change whose direction
the arithmetic cannot answer.
"""

import pytest

from llb.bench.agentic.context_policy import SUMMARY_INPUT_CAP_TRIGGER, SUMMARY_INPUT_CAP_WINDOW
from llb.bench.agentic_memory_boundary_probe import cap_prompt_sequence
from llb.bench.agentic_memory_fold_step_ladder import (
    MIN_LIVE_ENTRIES_TO_FOLD,
    foldable_fold_steps,
    live_entries_at_fold_step,
    reachable_fold_steps,
)
from llb.bench.agentic_policy_change_audit import PolicyChange
from llb.bench.agentic_policy_change_interaction import audit_interaction_cell
from llb.bench.agentic_policy_change_interaction_band import (
    band_at_fold_step,
    format_band_report,
    separating_guard_bands,
)
from llb.bench.agentic_policy_change_interaction_conditions import share_bound_conditions
from llb.bench.agentic_policy_change_interaction_couplings import coupling_for, held_baseline
from llb.bench.agentic_policy_change_interaction_fixture import (
    declared_expectations,
    geometry_kwargs,
    interaction_change,
    load_interaction_design,
)
from llb.bench.agentic_policy_change_interaction_terms import (
    FIELD_KEEP_RECENT,
    FIELD_SHARE,
    StepGeometry,
)

# A depth whose fold never offers the summarizer enough to overtake a trigger, so it separates
# nothing at any guard -- the other side of the band solver's answer.
SHALLOW_DEPTH = 8
# The step whose prompt is built from zero entries: a trigger below the first prompt selects it, and
# `compact_state` folds nothing there.
UNFOLDABLE_FIRST_STEP = 1
# A pair no geometry separates, whose blocked steps are what the no-band report is read on.
NO_GEOMETRY_PAIR = (FIELD_SHARE, FIELD_KEEP_RECENT)


@pytest.fixture(scope="module")
def design() -> dict[str, object]:
    return load_interaction_design()


def test_every_committed_guard_lies_where_the_solved_band_says_it_should(
    design: dict[str, object],
):
    """The fixture's guards are SOLVED, so a shifted task world names its replacement guards."""
    change, held = interaction_change(design), design["held_fixed"]
    expected = declared_expectations(design)
    for declared in design["cells"]:
        depth = declared["depth"]
        bands = separating_guard_bands(change, depth=depth, held=held)
        report = format_band_report(change, depth, bands)
        band = band_at_fold_step(bands, declared["predeclared"]["fold_step"])
        assert band is not None, f"{declared['cell_id']}\n{report}"
        inside = band.contains(declared["max_prompt_chars"])
        assert inside is expected[declared["cell_id"]]["separates"], (
            f"{declared['cell_id']} guard {declared['max_prompt_chars']}\n{report}"
        )


def test_the_solved_band_is_exact_at_both_of_its_edges(design: dict[str, object]):
    """Solved, not approximated: the audit separates at `low` and stops separating at `high`."""
    change, held = interaction_change(design), design["held_fixed"]
    depth = design["cells"][0]["depth"]
    bands = separating_guard_bands(change, depth=depth, held=held)
    assert bands, format_band_report(change, depth, bands)
    for band in bands:
        for guard, wanted in (
            (band.low - 1, False),
            (band.low, True),
            (band.high - 1, True),
            (band.high, False),
        ):
            cell = {
                "cell_id": f"edge-d{depth}-g{guard}",
                "depth": depth,
                "compact_share": held["compact_share"],
                "max_prompt_chars": guard,
                "pinned_fields": [],
            }
            row = audit_interaction_cell(cell, held, change)
            assert row["separates"] is wanted, f"{cell['cell_id']}\n{band.describe()}"


def test_a_depth_too_shallow_to_interact_reports_no_band_rather_than_a_wrong_one(
    design: dict[str, object],
):
    """An empty answer is still an answer: a shallow fold cannot overtake a trigger at any guard.

    The transcript a fold hands the summarizer grows faster per step than the step prompt does, but
    it starts a whole task prompt and tool schema behind, so it only catches up with the trigger on
    a long enough episode. A shallow depth therefore has nothing to separate, and the solver says so
    instead of returning an interval nobody can use.
    """
    change, held = interaction_change(design), design["held_fixed"]
    bands = separating_guard_bands(change, depth=SHALLOW_DEPTH, held=held)
    assert bands == [], format_band_report(change, SHALLOW_DEPTH, bands)
    assert "no fold step separates" in format_band_report(change, SHALLOW_DEPTH, bands)


def test_the_ladder_skips_the_first_step_because_no_episode_folds_there(design: dict[str, object]):
    """A trigger can SELECT step 1; the loop can never fold at it, so the solver does not ask.

    The step-1 prompt is built from zero entries, and `compact_state` returns False on a state with
    nothing older to summarize -- measured in
    `test_agentic_policy_change_interaction_cap.py::test_folded_entries_is_what_compact_state_
    actually_folds`. The shipped geometry really does offer that step (a small enough guard trips on
    the first prompt), which is why the filter has work to do here rather than only in principle.
    """
    depth, held = design["cells"][0]["depth"], design["held_fixed"]
    sequence = cap_prompt_sequence(**geometry_kwargs(depth, held))
    assert UNFOLDABLE_FIRST_STEP in reachable_fold_steps(sequence)
    foldable = foldable_fold_steps(sequence)
    assert UNFOLDABLE_FIRST_STEP not in foldable
    assert all(live_entries_at_fold_step(step) >= MIN_LIVE_ENTRIES_TO_FOLD for step in foldable)


def test_no_solved_band_moves_when_the_unfoldable_step_is_dropped(design: dict[str, object]):
    """The skipped step separated nothing to begin with, so the committed bands are untouched.

    Two halves: the solver still solves exactly the fold steps the fixture predeclares, and the step
    the ladder now skips states a condition no guard satisfies -- so nothing solvable was filtered
    away rather than merely nothing that happened to be solved today.
    """
    change, held = interaction_change(design), design["held_fixed"]
    depth = design["cells"][0]["depth"]
    geometry = geometry_kwargs(depth, held)
    sequence = cap_prompt_sequence(**geometry)
    bands = separating_guard_bands(change, depth=depth, held=held)
    assert {band.fold_step for band in bands} == {
        cell["predeclared"]["fold_step"] for cell in design["cells"]
    }, format_band_report(change, depth, bands)
    skipped = set(reachable_fold_steps(sequence)) - set(foldable_fold_steps(sequence))
    assert skipped == {UNFOLDABLE_FIRST_STEP}
    for step in sorted(skipped):
        stated = share_bound_conditions(
            StepGeometry(
                change=change,
                step=step,
                sequence=sequence,
                geometry=geometry,
                share=float(held[FIELD_SHARE]),
            )
        )
        assert any(condition.is_empty for condition in stated.conditions), stated


def test_a_blocked_report_opens_with_a_fold_that_can_actually_happen(design: dict[str, object]):
    """The no-band answer's FIRST line is its most informative one, not a vacuous step.

    `_blocking_reasons` reports each condition once, at the first step that blocks it, so a step the
    loop never folds at would lead the report with a count of zero live entries -- the one reading
    that says nothing about the geometry a drifted commit has to fix.
    """
    coupling = coupling_for(NO_GEOMETRY_PAIR)
    change, depth = coupling.example_change(), design["cells"][0]["depth"]
    held = held_baseline(coupling, design["held_fixed"])
    first_foldable = foldable_fold_steps(cap_prompt_sequence(**geometry_kwargs(depth, held)))[0]
    bands = separating_guard_bands(change, depth=depth, held=held, include_empty=True)
    report = format_band_report(change, depth, bands)
    assert bands and all(band.is_empty for band in bands), report
    assert min(band.fold_step for band in bands) == first_foldable, report
    # Line 0 is the header, 1 the no-band statement, 2 the coupling; the reasons follow.
    reasons = report.splitlines()[3:]
    assert reasons, report
    assert reasons[0].startswith(f"  fold step {first_foldable}: "), report
    live = live_entries_at_fold_step(first_foldable)
    assert f"step {first_foldable} has {live}" in reasons[0], report


def test_a_geometry_with_no_foldable_step_is_refused_rather_than_read_as_no_band(
    design: dict[str, object],
):
    """ "No fold step separates" has to mean the steps were solved, not that there were none.

    The solver states its conditions only at `foldable_fold_steps` of the measured sequence, so an
    unmeasured geometry does not reach a ladder refusal at all -- the comprehension simply yields
    nothing and `format_band_report` prints the same no-band line a genuinely contradictory coupling
    gets. That is the one caller-side edge here that is silent rather than loud, so it is refused
    with what the geometry does offer, the way the placement rule refuses an empty ladder.
    """
    change, held = interaction_change(design), design["held_fixed"]
    depth = design["cells"][0]["depth"]
    # Two ways to measure nothing: episodes that end before their first prompt, and a single prompt
    # a trigger can select but no episode can fold at.
    for margin, steps in ((-depth, 0), (1 - depth, 1)):
        starved = {**held, "max_steps_margin": margin}
        assert len(cap_prompt_sequence(**geometry_kwargs(depth, starved))) == steps
        with pytest.raises(ValueError, match=f"no foldable fold step on its measured {steps}-step"):
            separating_guard_bands(change, depth=depth, held=starved)


def test_a_fabricated_fold_step_is_left_to_fail_in_the_ladders_own_words(design: dict[str, object]):
    """The conditions do NOT translate the ladder's step refusal, and that is the decision.

    Nothing the solver builds can name a step off the sequence, so a `StepGeometry` that does is a
    fabricated one rather than a geometry an operator declared -- and the accurate thing to say
    about it is exactly what `fold_step_trigger_interval` says. Translating it would give the
    coupling vocabulary for a state the coupling cannot be asked about.
    """
    change, held = interaction_change(design), design["held_fixed"]
    depth = design["cells"][0]["depth"]
    geometry = geometry_kwargs(depth, held)
    sequence = cap_prompt_sequence(**geometry)
    assert foldable_fold_steps(sequence), "the committed geometry must have a ladder to solve on"
    for step in (0, len(sequence) + 1):
        fabricated = StepGeometry(
            change=change,
            step=step,
            sequence=sequence,
            geometry=geometry,
            share=float(held[FIELD_SHARE]),
        )
        with pytest.raises(ValueError, match=f"outside a {len(sequence)}-step sequence"):
            share_bound_conditions(fabricated)


def test_a_change_the_band_arithmetic_cannot_answer_is_refused(design: dict[str, object]):
    """The reverse bound move cannot separate at all, so the solver refuses instead of answering."""
    held, depth = design["held_fixed"], design["cells"][0]["depth"]
    reversed_bound = PolicyChange(
        baseline={"compact_share": 0.5, "summary_input_cap": SUMMARY_INPUT_CAP_TRIGGER},
        candidate={"compact_share": 0.48, "summary_input_cap": SUMMARY_INPUT_CAP_WINDOW},
    )
    with pytest.raises(ValueError, match="bound move can separate"):
        separating_guard_bands(reversed_bound, depth=depth, held=held)
    with pytest.raises(ValueError, match="only defined for"):
        separating_guard_bands(PolicyChange.of("compact_share", 0.5, 0.48), depth=depth, held=held)
