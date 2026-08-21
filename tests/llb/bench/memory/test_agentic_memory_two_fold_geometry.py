"""The repeatedly folding regime: why a cap-fitting cell cannot reach it, and what it says there.

Two claims live here and they are complements. A cap-fitting cell folds exactly ONCE, which is why
every published bound-invariance verdict was read in the one-fold regime and why the two walks agree
there. Below the cap peak a cell folds repeatedly, and there the stalling walk reaches a summarize
input the oracle transcript never produces -- so the same verdict, read on the same change, comes
out differently. That is the validity limit on the invariance statement.
"""

from copy import deepcopy

import pytest

from llb.bench.agentic.context_policy import (
    POLICY_COMPACT,
    POLICY_OBSERVATION_CAP,
    SUMMARY_INPUT_CAP_TRIGGER,
)
from llb.bench.memory.boundary.probe import compact_fold_input_probe
from llb.bench.memory.boundary.surface import load_surface_design, surface_cap_peaks
from llb.bench.memory.cap_audit import VERDICT_INVARIANT, VERDICT_SENSITIVE
from llb.bench.memory.two_fold.fixture import (
    MIN_DECLARED_FOLDS,
    probe_two_fold_cell,
    two_fold_cells,
    two_fold_change,
    load_two_fold_design,
    validate_two_fold_design,
)
from llb.bench.memory.two_fold.reading import (
    READING_DRIFTED,
    READING_ONE_FOLD_ONLY,
    analyze_two_fold,
    declaration_drift,
    margin_scaling,
)
from llb.bench.memory.worst_case_probe import worst_case_fold_input_probe
from llb.bench.agentic.design_fields import as_mapping, as_rows

SURFACE_DESIGN = "samples/benchmarks/agentic_compact_memory_boundary_surface_design.json"


def test_every_committed_cap_fitting_cell_folds_exactly_once_under_both_walks():
    """The one-fold regime is a property of the band, not of the cells that happened to be picked.

    Hysteresis raises the trigger to the FULL guard after the first summary, and a fold replaces
    entries with a strictly shorter summary line -- so the post-fold prompt is below the walk's peak
    while a cap-fitting guard is above it. Spending the whole step budget does not change that,
    which is why the published verdicts' two walks cannot disagree.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    design = load_surface_design(root / SURFACE_DESIGN)
    held = design["held_fixed"]
    peaks = surface_cap_peaks(design)
    for cell in design["surface"]["cells"]:
        geometry = {
            "depth": cell["depth"],
            "n_tasks": held["n_tasks"],
            "pad_chars": held["pad_chars"],
            "max_steps_margin": held["max_steps_margin"],
            "observation_cap_chars": held["observation_cap_chars"],
            "observation_head_share": held["observation_head_share"],
            "max_prompt_chars": cell["max_prompt_chars"],
            "compact_share": held["compact_share"],
        }
        assert cell["max_prompt_chars"] > peaks[cell["depth"]]
        assert compact_fold_input_probe(**geometry)["n_compactions"] == 1
        assert worst_case_fold_input_probe(**geometry)["n_compactions"] == 1


def test_the_committed_fixture_is_in_the_regime_it_declares():
    design = load_two_fold_design()
    validate_two_fold_design(design)
    change = two_fold_change(design)
    held = as_mapping(design, "held_fixed")
    for cell in two_fold_cells(design):
        measured = probe_two_fold_cell(cell, held, change)
        # Repeatedly folding, and BELOW its cap peak -- the two halves of the regime.
        assert measured["oracle_folds"] >= MIN_DECLARED_FOLDS
        assert cell["max_prompt_chars"] < measured["cap_peak_prompt_chars"]
        assert cell["policies"] == [POLICY_COMPACT]


def test_the_fixture_separates_the_two_walks_and_states_the_validity_limit():
    """The finding: one cell is invariant under perfect play and sensitive under imperfect play."""
    analysis = analyze_two_fold(load_two_fold_design())
    rows = {row["cell_id"]: row for row in analysis["cells"]}
    assert not any(row["declaration_drift"] for row in rows.values())

    separating = rows["twofold-d10-g7000"]
    assert separating["oracle_verdict"] == VERDICT_INVARIANT
    assert separating["worst_case_verdict"] == VERDICT_SENSITIVE
    assert separating["separates"] is True
    # The mechanism: the stalling walk does not grow the folds it SHARES with the oracle, it adds a
    # later one -- and that fold is the first to outgrow the trigger bound's cap.
    assert separating["worst_case_folds"] == separating["oracle_folds"] + 1
    assert separating["fold_input_margin_chars"] == [0] * separating["oracle_folds"]
    assert separating["worst_case_only_folds"] == 1
    assert separating["oracle_elided_chars_under_baseline"] == 0
    assert separating["worst_case_elided_chars_under_baseline"] > 0
    assert separating["worst_case_first_divergent_step"] is not None

    control = rows["twofold-d10-g6500"]
    assert control["oracle_verdict"] == control["worst_case_verdict"] == VERDICT_INVARIANT
    assert control["separates"] is False
    assert control["worst_case_first_divergent_step"] is None

    assert analysis["separating_cell_ids"] == ["twofold-d10-g7000"]
    assert analysis["reading"] == READING_ONE_FOLD_ONLY
    assert "one-fold transcript states nothing about a repeatedly folding one" in analysis["reason"]
    assert analysis["changes_shipped_default"] is False


def test_the_peak_margin_is_a_rate_per_wasted_step_not_a_constant():
    """The +453 chars the one-fold studies measured is `max_steps_margin=4`, not a universal number."""
    rows = margin_scaling(load_two_fold_design())
    assert len(rows) >= 2
    assert [row["max_steps_margin"] for row in rows] == sorted(
        row["max_steps_margin"] for row in rows
    )
    # A wider step budget buys the controller more wasted entries, so the margin grows with it...
    margins = [row["margin_chars"] for row in rows]
    assert margins == sorted(margins) and margins[0] < margins[-1]
    # ...at one price per extra step, which is what makes it a rate rather than a number.
    assert len({row["margin_chars_per_extra_step"] for row in rows}) == 1
    for row in rows:
        assert (
            row["margin_chars"] == row["margin_chars_per_extra_step"] * row["budgeted_extra_steps"]
        )


def test_a_cap_fitting_guard_is_refused_as_out_of_regime():
    """A cell above its cap peak claims a repeatedly folding cap-fitting cell, which cannot exist."""
    design = load_two_fold_design()
    out_of_band = deepcopy(design)
    as_rows(out_of_band, "cells")[0]["max_prompt_chars"] = 20000
    with pytest.raises(ValueError, match="cap-fitting band"):
        validate_two_fold_design(out_of_band)


def test_a_one_fold_cell_a_cap_arm_and_a_non_separating_grid_are_all_refused():
    design = load_two_fold_design()

    one_fold = deepcopy(design)
    # A guard this wide never folds a second time: perfect play crosses its trigger once.
    as_rows(one_fold, "cells")[0]["max_prompt_chars"] = 11000
    with pytest.raises(ValueError, match="below the 2 this fixture exists to reach"):
        validate_two_fold_design(one_fold)

    with_cap_arm = deepcopy(design)
    as_rows(with_cap_arm, "cells")[0]["policies"] = [POLICY_OBSERVATION_CAP, POLICY_COMPACT]
    with pytest.raises(ValueError, match="compact arm alone"):
        validate_two_fold_design(with_cap_arm)

    flat = deepcopy(design)
    for row in as_rows(flat, "cells"):
        as_mapping(row, "expected")["separates"] = False
    with pytest.raises(ValueError, match="SEPARATING cell"):
        validate_two_fold_design(flat)

    pinned = deepcopy(design)
    as_rows(pinned, "cells")[0]["summary_input_cap"] = SUMMARY_INPUT_CAP_TRIGGER
    with pytest.raises(ValueError, match="cannot audit it"):
        validate_two_fold_design(pinned)


def test_one_step_budget_cannot_say_whether_the_margin_is_a_constant():
    design = load_two_fold_design()
    single = deepcopy(design)
    as_mapping(single, "margin_scaling")["max_steps_margins"] = [4]
    with pytest.raises(ValueError, match="two distinct step budgets"):
        validate_two_fold_design(single)


def test_a_measured_geometry_that_leaves_its_declaration_is_a_finding_not_a_new_expectation():
    """Predeclared, so a runtime change that moves the regime reads as drift rather than as truth."""
    analysis = analyze_two_fold(load_two_fold_design())
    row = analysis["cells"][0]
    moved = {
        "predeclared": {"oracle_folds": row["oracle_folds"] + 1},
        "expected": {"separates": not row["separates"]},
    }
    drift = declaration_drift(row, moved)
    assert len(drift) == 2
    assert any(item.startswith("oracle_folds:") for item in drift)

    from llb.bench.memory.two_fold.reading import two_fold_reading

    reading, reason = two_fold_reading(
        [{**row, "declaration_drift": drift}], margin_scaling(load_two_fold_design())
    )
    assert reading == READING_DRIFTED
    assert "oracle_folds" in reason
