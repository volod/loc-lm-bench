"""Where the separating guards are, solved from the interval arithmetic -- no GPU.

The interaction fixture only separates the two policy-change readings inside a narrow guard band,
and the band is a property of the task world: shift every prompt size by a few chars and it moves.
So the band is solved rather than searched for, and these are the tests that keep the solver honest
-- exact at both edges, empty where nothing can separate, and refused for a change whose direction
the arithmetic cannot answer.
"""

import pytest

from llb.bench.agentic.context import SUMMARY_INPUT_CAP_TRIGGER, SUMMARY_INPUT_CAP_WINDOW
from llb.bench.agentic_policy_change_audit import PolicyChange
from llb.bench.agentic_policy_change_interaction import audit_interaction_cell
from llb.bench.agentic_policy_change_interaction_band import (
    band_at_fold_step,
    format_band_report,
    separating_guard_bands,
)
from llb.bench.agentic_policy_change_interaction_fixture import (
    declared_expectations,
    interaction_change,
    load_interaction_design,
)

# A depth whose fold never offers the summarizer enough to overtake a trigger, so it separates
# nothing at any guard -- the other side of the band solver's answer.
SHALLOW_DEPTH = 8


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
