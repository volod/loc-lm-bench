"""Replaying a fold length at the span the fold it stands for actually offered.

The fit's other two measurements -- the fold length and the per-step entry -- are covered beside
the fit itself. This file covers the third: the run order that puts a second never-fitted cell on
disk before the fitted one, the (offered span, written length) slope those two cells give the
replay, and what that slope is worth against the flat replay it replaces.
"""

import json
import re

from llb.bench.agentic.context_summary import summary_offered_chars
from llb.bench.memory.repeated_fold.completion import run_repeated_fold_completion
from llb.bench.memory.repeated_fold.design import completion_cells
from llb.bench.memory.repeated_fold.fold_span import (
    SPAN_LENGTH_INTERPOLATED,
    SPAN_LENGTH_SINGLE,
    SPAN_LENGTH_UNMEASURED,
    fold_length_span_model,
    measured_fold_points,
)
from llb.bench.memory.repeated_fold.guard_fit import (
    fit_fold_guard,
    fitted_cell_order,
    guard_fit_spec,
    measured_fold_lengths,
    measured_step_entry_chars,
)
from llb.bench.memory.repeated_fold.replication import (
    SPAN_SLOPE_AGREES,
    SPAN_SLOPE_DISAGREES,
    SPAN_SLOPE_UNREAD,
    analyze_replication_runs,
    run_replication_family,
    span_slope_reading,
)
from llb.bench.memory.repeated_fold.replication_design import (
    load_repeated_fold_replication_design,
    replication_roster,
    validate_replication_design,
)
from llb.bench.memory.repeated_fold.replication_report import format_replication_table


class WritesSummariesOfLength:
    """Perfect play, with a summarizer whose output length is the family's whole personality."""

    def __init__(self, summary_chars: int):
        self.summary_chars = summary_chars

    def __call__(self, prompt: str) -> str:
        if "Стисло підсумуй" in prompt:
            return ("підсумок попередніх кроків; " * 400)[: self.summary_chars]
        if "[workflow complete]" not in prompt:
            tokens = re.findall(r'(?:токеном "|next token: )(wf-\d{3}-\d+)', prompt)
            assert tokens
            return json.dumps({"name": "advance", "arguments": {"token": tokens[-1]}})
        code = re.search(r"MEM-\d{3}-\d{3}", prompt)
        return json.dumps(
            {"name": "finish", "arguments": {"answer": code.group(0) if code else "LOST"}}
        )


def _fitted_cell(design: dict[str, object]) -> dict[str, object]:
    cell_id = guard_fit_spec(design)["cell_id"]
    return next(cell for cell in completion_cells(design) if cell["cell_id"] == cell_id)


def _fake_design(n_tasks: int = 6) -> dict[str, object]:
    design = load_repeated_fold_replication_design()
    held = {**design["held_fixed"], "n_tasks": n_tasks, "minimum_paired_cases_per_fold": 2}
    return {**design, "held_fixed": held}


class WritesInProportionToWhatItIsOffered(WritesSummariesOfLength):
    """A summarizer whose output tracks the span it was shown, which is what real ones do.

    The constant-length fake above cannot show the defect this calibration exists for: it writes
    the same summary whether it is folding a whole transcript or three entries, so replaying the
    control's length at the fitted cell's spans is exactly right for it. A family with a floor and
    a slope writes LESS at the shorter spans a tighter guard folds, which is where a length
    carried across from the control alone goes wrong.
    """

    def __init__(self, floor_chars: int, offered_divisor: int):
        super().__init__(floor_chars)
        self.floor_chars = floor_chars
        self.offered_divisor = offered_divisor

    def __call__(self, prompt: str) -> str:
        if "Стисло підсумуй" not in prompt:
            return super().__call__(prompt)
        chars = self.floor_chars + summary_offered_chars(prompt) // self.offered_divisor
        return ("підсумок попередніх кроків; " * 400)[:chars]


def test_two_cells_of_different_span_give_the_replay_a_slope():
    """The second point is what turns one measured length into a length per offered span."""
    model = fold_length_span_model([(8000, 300)] * 3, [(3000, 150)] * 3)
    assert model.reading == SPAN_LENGTH_INTERPOLATED
    assert model.chars_per_offered_char == (150 - 300) / (3000 - 8000)
    assert model.length_at(300, 8000) == 300
    assert model.length_at(300, 3000) == 150
    assert model.length_at(300, 5500) == 225


def test_one_measured_span_is_refused_a_slope_rather_than_given_a_zero_one():
    """A run that measured one span keeps the flat replay, named -- it does not invent a rate."""
    single = fold_length_span_model([(8000, 300)], [])
    assert single.reading == SPAN_LENGTH_SINGLE
    assert single.chars_per_offered_char == 0.0
    assert single.length_at(274, 2500) == 274
    same_span = fold_length_span_model([(8000, 300)], [(8000, 220)])
    assert same_span.reading == SPAN_LENGTH_SINGLE
    assert same_span.length_at(274, 2500) == 274
    assert fold_length_span_model([], [(3000, 150)]).reading == SPAN_LENGTH_UNMEASURED


def test_the_replayed_length_is_never_extrapolated_past_the_measured_spans():
    """Two points give a slope, not a curve, so the model stops where the measurements stop."""
    model = fold_length_span_model([(8000, 300)], [(3000, 150)])
    assert model.length_at(300, 500) == model.length_at(300, 3000)
    assert model.length_at(300, 20000) == model.length_at(300, 8000)


def test_a_case_keeps_its_own_level_while_the_family_shares_one_slope():
    """Level per case, slope per family: the fit still predicts a case COUNT, not one episode."""
    model = fold_length_span_model([(8000, 300)], [(3000, 150)])
    assert model.length_at(400, 3000) - model.length_at(300, 3000) == 100
    assert model.length_at(400, 8000) == 400


def test_the_run_carries_each_folds_offered_span_beside_the_length_written_against_it():
    rows = [
        {
            "cell_id": "twofold-d10-g6500",
            "arm": "typed_marker",
            "cases": [
                {"summary_fold_input_chars": [2400, 3100], "summary_output_chars": [140, 170]},
                {"summary_fold_input_chars": [2500], "summary_output_chars": [150]},
            ],
        },
        {
            "cell_id": "twofold-d10-g6500",
            "arm": "model_summary_only",
            "cases": [{"summary_fold_input_chars": [9999], "summary_output_chars": [9999]}],
        },
    ]
    assert measured_fold_points(rows, "twofold-d10-g6500") == [
        (2400, 140),
        (3100, 170),
        (2500, 150),
    ]


def test_the_fitted_cell_runs_last_so_every_cell_it_measures_is_already_on_disk():
    design = load_repeated_fold_replication_design()
    order = [cell["cell_id"] for cell in fitted_cell_order(design)]
    fitted = guard_fit_spec(design)["cell_id"]
    span_source = guard_fit_spec(design)["span_length_source"]
    control = next(cell for cell in completion_cells(design) if cell["cap_fitting_control"])
    assert order[0] == control["cell_id"]
    assert order[-1] == fitted
    assert order.index(span_source) < order.index(fitted)
    assert set(order) == {cell["cell_id"] for cell in completion_cells(design)}


def test_a_run_order_that_does_not_lead_with_the_control_is_refused():
    design = _fake_design(n_tasks=2)
    reversed_cells = list(reversed(completion_cells(design)))
    try:
        run_repeated_fold_completion(
            design,
            model="fake",
            backend="fake",
            complete=WritesSummariesOfLength(120),
            cell_order=reversed_cells,
        )
    except ValueError as exc:
        assert "must run first" in str(exc)
    else:
        raise AssertionError("an order that drops the control-first gate must be refused")


def test_a_replication_without_a_span_source_is_refused():
    design = load_repeated_fold_replication_design()
    spec = {
        key: value for key, value in guard_fit_spec(design).items() if key != "span_length_source"
    }
    try:
        validate_replication_design({**design, "two_fold_guard_fit": spec})
    except ValueError as exc:
        assert "span_length_source" in str(exc)
    else:
        raise AssertionError("a one-span replay must be refused by the committed design")


def test_a_span_source_that_folds_no_more_often_than_the_fitted_cell_is_refused():
    design = load_repeated_fold_replication_design()
    control = next(cell for cell in completion_cells(design) if cell["cap_fitting_control"])
    spec = {**guard_fit_spec(design), "span_length_source": control["cell_id"]}
    try:
        validate_replication_design({**design, "two_fold_guard_fit": spec})
    except ValueError as exc:
        assert "no slope" in str(exc) or "same spans" in str(exc)
    else:
        raise AssertionError("a second point at the control's own span must be refused")


def test_a_span_source_naming_the_fitted_cell_is_refused():
    design = load_repeated_fold_replication_design()
    spec = {**guard_fit_spec(design), "span_length_source": guard_fit_spec(design)["cell_id"]}
    try:
        validate_replication_design({**design, "two_fold_guard_fit": spec})
    except ValueError as exc:
        assert "the cell it is fitting" in str(exc)
    else:
        raise AssertionError("a fit reading its own cell must be refused")


def _shared_guard_two_fold_cases(analysis: dict[str, object], cell_id: str) -> int:
    """How many shipped-policy cases a cell actually landed on the two-fold rung."""
    return sum(
        1
        for row in analysis["cells"]
        if row["cell_id"] == cell_id and row["arm"] == "typed_marker"
        for case in row["cases"]
        if case["measured_folds"] == 2
    )


def test_the_span_aware_replay_counts_the_declared_guard_the_flat_one_could_not():
    """The published defect, reproduced and fixed in one fixture.

    A family whose summaries track the span it is shown folds the DECLARED 7000 guard twice on
    every case. Replaying the control's own fold length there predicts none of them, because the
    control folds one long span and writes long against it; replaying at the span the 7000 guard's
    folds actually offer predicts all of them, and keeps the declared guard instead of moving one
    that never needed moving.
    """
    design = _fake_design()
    roster = replication_roster(design)
    shared = {key: value for key, value in design.items() if key != "two_fold_guard_fit"}
    fitted_run = run_replication_family(
        design, roster[0], complete=WritesInProportionToWhatItIsOffered(40, 40)
    )
    measured_run = run_replication_family(
        shared, roster[0], complete=WritesInProportionToWhatItIsOffered(40, 40)
    )
    fit = fitted_run.analysis["guard_fits"][0]
    cell = _fitted_cell(design)
    flat = fit_fold_guard(
        design,
        cell,
        design["held_fixed"],
        measured_fold_lengths(fitted_run.analysis["cells"], fit["fold_length_source"]),
        evidence_floor=2,
        step_entry_chars=measured_step_entry_chars(
            fitted_run.analysis["cells"], fit["step_length_source"]
        ),
    )
    measured = _shared_guard_two_fold_cases(measured_run.analysis, cell["cell_id"])
    assert measured > 0
    assert fit["fold_span_reading"] == SPAN_LENGTH_INTERPOLATED
    assert fit["declared_target_cases"] == measured
    assert flat["declared_target_cases"] != measured


def test_the_span_aware_replay_lands_nearer_the_length_the_fitted_cell_wrote():
    """The calibration error the fit is left with, stated against the one it started from."""
    design = _fake_design()
    run = run_replication_family(
        design, replication_roster(design)[0], complete=WritesInProportionToWhatItIsOffered(80, 20)
    )
    fit = run.analysis["guard_fits"][0]
    low, high = fit["fitted_cell_fold_span_range"]
    assert low < high < fit["anchor_fold_span_chars"]
    assert abs(fit["span_replay_error_chars"]) < abs(fit["fold_length_replay_error_chars"])
    assert fit["prediction_within_fold_length_margin"] is True
    assert "span:" in format_replication_table(analyze_replication_runs(design, [run]))


def test_families_that_slope_opposite_ways_are_named_rather_than_averaged():
    """Two habits are not one rate: a sign disagreement is the finding, so the run states it."""

    def family(name: str, slope: float) -> dict[str, object]:
        return {
            "model_family": name,
            "control_eligible": True,
            "guard_fits": [
                {
                    "chars_written_per_offered_char": slope,
                    "fold_span_reading": SPAN_LENGTH_INTERPOLATED,
                }
            ],
        }

    disagree, reason = span_slope_reading([family("qwen", 0.008), family("gemma4", -0.004)])
    assert disagree == SPAN_SLOPE_DISAGREES
    assert "qwen +0.00800" in reason and "gemma4 -0.00400" in reason
    agree, _reason = span_slope_reading([family("qwen", 0.008), family("gemma4", 0.004)])
    assert agree == SPAN_SLOPE_AGREES


def test_a_family_with_only_one_measured_span_slopes_against_nothing():
    unread, reason = span_slope_reading(
        [
            {
                "model_family": "qwen",
                "control_eligible": True,
                "guard_fits": [
                    {
                        "chars_written_per_offered_char": 0.0,
                        "fold_span_reading": SPAN_LENGTH_SINGLE,
                    }
                ],
            }
        ]
    )
    assert unread == SPAN_SLOPE_UNREAD
    assert "second fold span" in reason


def test_the_run_states_whether_its_span_correction_generalizes():
    design = _fake_design()
    roster = replication_roster(design)[:2]
    runs = [
        run_replication_family(
            design, roster[0], complete=WritesInProportionToWhatItIsOffered(40, 40)
        ),
        run_replication_family(design, roster[1], complete=WritesSummariesOfLength(200)),
    ]
    analysis = analyze_replication_runs(design, runs)
    assert analysis["span_slope_reading"] in {SPAN_SLOPE_AGREES, SPAN_SLOPE_DISAGREES}
    assert "span correction:" in format_replication_table(analysis)
