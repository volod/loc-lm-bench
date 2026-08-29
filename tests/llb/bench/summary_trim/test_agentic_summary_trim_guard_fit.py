"""Per-family middle-critical guard fitting: the band's filter, the choice, and its refusals.

The live run is a two-family GPU measurement; what CI owns is the model-free half. Which guards the
declared band can actually use is a property of the workload geometry alone, and so is the choice a
given set of measured walks makes among them -- so both are decided here, over injected walk
lengths, before any host is warmed.
"""

from typing import cast

import pytest

from llb.bench.context_policy.guard_band import guard_grid, median_int, select_guard
from llb.bench.summary_trim.design import (
    load_summary_trim_design,
    validate_summary_trim_design,
    workloads,
)
from llb.bench.summary_trim.guard_fit import (
    FIT_APPLIED,
    FIT_DECLARED,
    FIT_UNDERPOWERED,
    FIT_UNMEASURED,
    GUARD_FIT_FIELD,
    WALK_CONTROL_ARM,
    fit_middle_guard,
    fitted_workload_name,
    guard_band_reading,
    guard_fit_spec,
    validate_guard_fit,
)
from llb.bench.summary_trim.guard_regime import (
    REFUSED_FOLD_COUNT,
    REFUSED_NO_ELISION,
    REFUSED_PLACEMENT,
    REFUSED_UNFOLDED_FACT,
    scan_guard_band,
    usable_guards,
)
from llb.bench.summary_trim.workloads import build_workload_tasks

# The two cases the measured host has walked short before, named here so a fixture can inject the
# same shape CI has no GPU to reproduce. How short is derived from the band rather than pinned:
# the point is a walk that ends before the fitted fold, whatever step that fold lands on.
_SHORT_WALK_CASES = ("window-elision-m-001-d10", "window-elision-m-002-d10")
# A transcript that grows SLOWER than the committed one: the retired middle-critical shape, whose
# padding lets the band span two fold steps. It is what still exercises a fit that MOVES a guard,
# a trade the committed shape's one-step band cannot offer.
_SLOWER_GROWING_SHAPE = {
    "depth": 10,
    "pad_chars": 1600,
    "fact_stages": {"head": 4, "middle": 5, "tail": 7},
}
_SLOWER_GROWING_BAND = {"search_min_chars": 9000, "search_max_chars": 16000, "step_chars": 250}
_SLOWER_GROWING_GUARD = 14000
# The verdict gates on the policy-change audit, which is a whole separate model-free study;
# what is under test here is the fit, so the audit enters as a stub.
_NO_AUDIT: dict[str, object] = {
    "change": "stub",
    "n_cells": 0,
    "n_prompt_invariant": 0,
    "n_invalidated": 0,
    "invalidated_cells": [],
    "affected_published_values": [],
    "invariant": True,
}


@pytest.fixture(scope="module")
def design() -> dict[str, object]:
    return load_summary_trim_design()


@pytest.fixture(scope="module")
def fitted(design: dict[str, object]) -> dict[str, object]:
    name = fitted_workload_name(design)
    return next(row for row in workloads(design) if row["workload"] == name)


@pytest.fixture(scope="module")
def scan(design: dict[str, object], fitted: dict[str, object]) -> list[dict[str, object]]:
    return scan_guard_band(fitted, design["held_fixed"], guard_fit_spec(design))  # type: ignore[arg-type]


def _slower_growing(
    design: dict[str, object], fitted: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    """The same study over a slower-growing transcript, whose band spans more than one fold step."""
    workload = {
        **fitted,
        "max_prompt_chars": _SLOWER_GROWING_GUARD,
        "task_shape": _SLOWER_GROWING_SHAPE,
    }
    spec = {**guard_fit_spec(design), **_SLOWER_GROWING_BAND}
    return {**design, GUARD_FIT_FIELD: spec}, workload


def _walks(fitted: dict[str, object], short: dict[str, int] | None = None) -> dict[str, int]:
    """Every case walking the full workflow, except the ones a caller names as ending early."""
    short = short or {}
    ids = [str(record["id"]) for record in build_workload_tasks(fitted)]
    return {item: short.get(item, 11) for item in ids}


def test_every_usable_guard_still_produces_the_workload_regime(
    design: dict[str, object], scan: list[dict[str, object]]
):
    """The band is filtered BEFORE it is scored, so a fit cannot buy an early fold with the regime."""
    usable = [row for row in scan if row["refusal"] is None]
    assert usable, "the declared band must contain at least the declared guard"
    for row in usable:
        assert row["n_compactions"] == 1, row
        assert int(row["summary_input_elided_chars"]) > 0, row  # type: ignore[arg-type]
        assert int(row["fold_step"]) > 0, row  # type: ignore[arg-type]


def test_the_band_brackets_every_way_a_guard_stops_measuring_the_stratum(
    scan: list[dict[str, object]],
):
    """A band that only held usable guards would not show where the fit runs out of room."""
    refusals = {row["refusal"] for row in scan}
    assert {
        REFUSED_NO_ELISION,
        REFUSED_UNFOLDED_FACT,
        REFUSED_PLACEMENT,
        REFUSED_FOLD_COUNT,
    } <= refusals


def test_folding_earlier_costs_the_stratum_the_fact_it_was_built_around(
    scan: list[dict[str, object]],
):
    """Why the band bottoms out: below its floor the fold is early enough to move the fact out.

    The elided span is the middle of a transcript whose length the guard sets, so a guard low
    enough to fold before the walk's end folds a transcript that has not reached every fact its
    tasks plant, and stops being the experiment at all. Which of the two placement properties bites
    at the floor is a property of the SHAPE: a fast-growing transcript runs out of stages first, a
    slow-growing one moves the boundaries under a fact it does contain.
    """
    usable = usable_guards(scan)
    floor = min(usable, key=lambda guard: (usable[guard], guard))
    below = [row for row in scan if int(row["max_prompt_chars"]) < floor]  # type: ignore[arg-type]
    assert below and below[-1]["refusal"] == REFUSED_UNFOLDED_FACT
    # Above the usable run the fold does contain every fact, and the trim boundaries move instead.
    above = [row for row in scan if int(row["max_prompt_chars"]) > max(usable)]  # type: ignore[arg-type]
    assert above and above[0]["refusal"] == REFUSED_PLACEMENT
    # The usable guards are one contiguous run: the refusals bracket them rather than interleave.
    guards = sorted(usable)
    assert guards == [row["max_prompt_chars"] for row in scan if row["refusal"] is None]
    assert [row for row in scan if int(row["max_prompt_chars"]) > max(guards)]  # type: ignore[arg-type]


def test_the_band_is_readable_with_no_family_at_all(
    design: dict[str, object], fitted: dict[str, object]
):
    """What bounds every family's fit is model-free, so an audit-only run still reports it."""
    band = guard_band_reading(design, fitted)
    assert band["search_band"] == {"min_chars": 4000, "max_chars": 15000, "step_chars": 250}
    assert band["n_candidates"] == len(band["usable_guards"]) + len(band["refused_guards"])  # type: ignore[arg-type]
    assert band["declared_max_prompt_chars"] in band["usable_guards"]  # type: ignore[operator]
    assert band["band_floor_reason"] == REFUSED_UNFOLDED_FACT
    assert band["band_fold_steps"] == sorted(set(band["usable_guards"].values()))  # type: ignore[union-attr]


def test_a_family_that_walks_every_case_keeps_the_declared_guard(
    design: dict[str, object], fitted: dict[str, object]
):
    """The declared guard wins its own tie, so a family the constant suits keeps its geometry."""
    record = fit_middle_guard(design, fitted, design["held_fixed"], _walks(fitted))  # type: ignore[arg-type]
    assert record["fit_reading"] == FIT_DECLARED
    assert record["fitted_max_prompt_chars"] == fitted["max_prompt_chars"]
    assert record["meets_evidence_floor"] is True
    assert record["short_walk_cases"] == []


def test_a_walk_a_lower_guard_can_reach_moves_the_guard(
    design: dict[str, object], fitted: dict[str, object]
):
    """The fit exists for this case: a family that stops short of the declared fold gets an earlier one.

    Read on a slower-growing transcript, because moving a guard is only possible where the band
    spans two fold steps -- which is a property of the workload's padding, not of the fit.
    """
    slower_design, slower = _slower_growing(design, fitted)
    band = guard_band_reading(slower_design, slower)
    steps = cast(list[int], band["band_fold_steps"])
    assert len(steps) > 1, steps
    walks = {item: steps[0] for item in _walks(slower)}
    record = fit_middle_guard(slower_design, slower, design["held_fixed"], walks)  # type: ignore[arg-type]
    assert record["fit_reading"] == FIT_APPLIED
    assert int(record["fitted_fold_step"]) == steps[0]  # type: ignore[arg-type]
    assert record["fitted_max_prompt_chars"] != slower["max_prompt_chars"]
    assert record["meets_evidence_floor"] is True


def test_a_band_of_one_fold_step_can_only_confirm_the_declared_guard_or_refuse_the_walk(
    design: dict[str, object], fitted: dict[str, object], scan: list[dict[str, object]]
):
    """The committed shape's own limit, stated rather than left to be derived from the guard list.

    Its padding grows the transcript fast enough that the fold lands inside a short walk, and the
    same speed leaves the elided middle one entry wide -- so every guard that still holds the
    regime folds at the SAME step, and no fit can trade fold position against walk length.
    """
    steps = sorted(set(usable_guards(scan).values()))
    assert steps == [7], steps
    reached = fit_middle_guard(design, fitted, design["held_fixed"], _walks(fitted))  # type: ignore[arg-type]
    missed = fit_middle_guard(  # type: ignore[arg-type]
        design, fitted, design["held_fixed"], {item: steps[0] - 1 for item in _walks(fitted)}
    )
    assert reached["fit_reading"] == FIT_DECLARED
    assert missed["fit_reading"] == FIT_UNDERPOWERED
    assert {reached["fitted_max_prompt_chars"], missed["fitted_max_prompt_chars"]} == {
        fitted["max_prompt_chars"]
    }
    assert "no candidate folds earlier" in str(missed["fit_reason"])


def test_a_walk_no_guard_in_the_band_reaches_is_reported_rather_than_widened(
    design: dict[str, object], fitted: dict[str, object], scan: list[dict[str, object]]
):
    """The declared negative outcome: name the closest guard and what refused the one below it."""
    earliest = min(usable_guards(scan).values())
    walks = _walks(fitted, {case: earliest - 1 for case in _SHORT_WALK_CASES})
    record = fit_middle_guard(design, fitted, design["held_fixed"], walks)  # type: ignore[arg-type]
    assert record["fit_reading"] == FIT_UNDERPOWERED
    assert record["meets_evidence_floor"] is False
    assert record["short_walk_cases"] == sorted(_SHORT_WALK_CASES)
    assert record["predicted_folding_cases"] == len(walks) - len(_SHORT_WALK_CASES)
    # The band is exhausted, not unexplored: the reason names what the band had left and why the
    # candidate below it is not one.
    reason = str(record["fit_reason"])
    assert "The band is exhausted, not unexplored" in reason
    assert str(record["band_floor_reason"]) in reason
    assert record["band_floor_reason"] == REFUSED_UNFOLDED_FACT


def test_a_run_with_no_measured_walk_leaves_the_declared_guard_alone(
    design: dict[str, object], fitted: dict[str, object]
):
    """Nothing measured is not evidence for a move, so the published geometry stands."""
    record = fit_middle_guard(design, fitted, design["held_fixed"], {})  # type: ignore[arg-type]
    assert record["fit_reading"] == FIT_UNMEASURED
    assert record["fitted_max_prompt_chars"] == fitted["max_prompt_chars"]


def test_the_walk_control_must_not_fold(design: dict[str, object]):
    """A control that folds measures a folded walk, which is not the walk the fit is about."""
    spec = {**guard_fit_spec(design), "walk_control": {"max_prompt_chars": 14000}}
    with pytest.raises(ValueError, match="walk control folds"):
        validate_guard_fit({**design, GUARD_FIT_FIELD: spec}, workloads(design))


def test_the_declared_guard_must_be_a_candidate_in_its_own_band(design: dict[str, object]):
    """A band that excludes the declared guard cannot report 'the declared one already fits'."""
    spec = {**guard_fit_spec(design), "search_min_chars": 9750}
    with pytest.raises(ValueError, match="not a candidate in its own band"):
        validate_guard_fit({**design, GUARD_FIT_FIELD: spec}, workloads(design))


def test_the_fit_must_name_a_declared_workload(design: dict[str, object]):
    spec = {**guard_fit_spec(design), "workload": "nowhere"}
    with pytest.raises(ValueError, match="not a declared workload"):
        validate_guard_fit({**design, GUARD_FIT_FIELD: spec}, workloads(design))


def test_the_committed_design_carries_a_fit_the_gate_accepts(design: dict[str, object]):
    """The whole design gate, including the band the fitted workload may move inside."""
    validate_summary_trim_design(design)
    assert guard_fit_spec(design)["walk_control"]["arm"] == WALK_CONTROL_ARM  # type: ignore[index]


def test_the_shared_band_picks_the_declared_guard_on_a_tie_and_the_widest_run_otherwise():
    """The two selection rules, stated on their own: they decide every per-family guard fit."""
    assert guard_grid({"search_min_chars": 10, "search_max_chars": 40, "step_chars": 10}) == [
        10,
        20,
        30,
        40,
    ]
    assert select_guard({10: 2, 20: 2, 30: 1}, declared=20) == 20
    # No tie at the declared guard: the centre of the widest contiguous run of winners wins, which
    # is the choice furthest from the edge where the score changes.
    assert select_guard({10: 2, 20: 2, 30: 2, 40: 1, 50: 2}, declared=40) == 20
    assert median_int([]) == 0 and median_int([3, 1, 2]) == 2
    with pytest.raises(ValueError, match="at least one candidate"):
        select_guard({}, declared=1)


def test_the_walk_control_is_measured_but_never_paired_against_an_arm(
    design: dict[str, object], oracle_family
):
    """It is a measurement, not a third arm: it is persisted, and nothing pairs against it."""
    run = oracle_family("fixture")
    assert [row["arm"] for row in run.walk_control] == [WALK_CONTROL_ARM]
    control = run.walk_control[0]
    assert control["workload"] == fitted_workload_name(design)
    assert {case["measured_folds"] for case in control["cases"]} == {0}  # type: ignore[union-attr]
    assert all(row["arm"] != WALK_CONTROL_ARM for row in run.rows)
    assert run.guard_fit["walk_lengths"] == {
        str(case["item_id"]): case["n_steps"]
        for case in control["cases"]  # type: ignore[union-attr]
    }


def test_the_aggregate_carries_the_control_it_measured_the_walk_with(
    design: dict[str, object], oracle_family
):
    """The verdict has to be readable beside the measurement the fit consumed, not just the fit."""
    from llb.bench.summary_trim.analysis import analyze_summary_trim_runs
    from llb.bench.summary_trim.report import format_summary_trim_table

    analysis = analyze_summary_trim_runs(design, [oracle_family("fixture")], audit=_NO_AUDIT)
    control = cast(list[dict[str, object]], analysis["families"])[0]["walk_control"]  # type: ignore[index]
    assert control == [
        {
            "workload": fitted_workload_name(design),
            "arm": WALK_CONTROL_ARM,
            "max_prompt_chars": guard_fit_spec(design)["walk_control"]["max_prompt_chars"],  # type: ignore[index]
            "n_tasks": 12,
            "completion": 1.0,
            "n_folded_cases": 0,
        }
    ]
    table = format_summary_trim_table(analysis)
    assert "middle-critical guard fit" in table and "walk control: guard" in table


def test_the_fitted_workload_runs_at_the_guard_the_fit_chose(
    design: dict[str, object], oracle_family
):
    """The arms execute the fit's answer, not the declaration, and only for the fitted workload."""
    run = oracle_family("fixture")
    name = fitted_workload_name(design)
    fitted_rows = [row for row in run.rows if row["workload"] == name]
    assert {row["max_prompt_chars"] for row in fitted_rows} == {
        run.guard_fit["fitted_max_prompt_chars"]
    }
    others = {
        str(row["workload"]): row["max_prompt_chars"] for row in run.rows if row["workload"] != name
    }
    declared = {
        str(row["workload"]): row["max_prompt_chars"]
        for row in workloads(design)
        if row["workload"] != name
    }
    assert others == declared
