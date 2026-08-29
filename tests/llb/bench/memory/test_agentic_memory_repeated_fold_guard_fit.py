"""Per-family fold-guard fitting: the contract, the model-free fit, and its refusals."""

import json
import re

from llb.bench.agentic.context import ContextState, format_entry
from llb.bench.agentic.context_policy import POLICY_COMPACT, ContextPolicy
from llb.bench.agentic.context_summary import compact_state
from llb.bench.memory.boundary.probe import compact_fold_input_probe, fold_length_controller
from llb.bench.memory.repeated_fold.design import completion_cells
from llb.bench.context_policy.guard_band import search_band
from llb.bench.memory.repeated_fold.fold_span import fold_length_span_model
from llb.bench.memory.repeated_fold.guard_fit import (
    FIT_APPLIED,
    FIT_DECLARED,
    FIT_UNDERPOWERED,
    FIT_UNMEASURED,
    STEP_LENGTH_ABOVE_ORACLE,
    STEP_LENGTH_AT_ORACLE,
    STEP_LENGTH_UNMEASURED,
    fit_fold_guard,
    guard_fit_spec,
    measured_fold_lengths,
    measured_step_entry_chars,
)
from llb.bench.memory.repeated_fold.guard_replay import (
    FOLD_COUNT_MARGIN_SCAN_CHARS,
    FOLD_COUNT_MARGIN_STEP_CHARS,
    fold_count_margin_chars,
    fold_count_predictor,
    oracle_step_entry_chars,
)
from llb.bench.memory.repeated_fold.ladder_coverage import ladder_coverage
from llb.bench.memory.repeated_fold.fit_calibration import (
    PREDICTION_CALIBRATED,
    PREDICTION_DIVERGED,
    fit_prediction_reading,
)
from llb.bench.memory.repeated_fold.replication import (
    analyze_replication_runs,
    run_replication_family,
)
from llb.bench.memory.repeated_fold.replication_design import (
    load_repeated_fold_replication_design,
    replication_roster,
    validate_replication_design,
)
from llb.bench.memory.repeated_fold.replication_report import format_replication_table

_FLAT = fold_length_span_model([], [])

GEOMETRY = {
    "depth": 10,
    "n_tasks": 2,
    "pad_chars": 1200,
    "max_steps_margin": 4,
    "observation_cap_chars": 800,
    "observation_head_share": 0.6,
    "compact_share": 0.8,
    "summary_input_cap": "window",
}


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


def test_a_longer_summary_folds_a_shared_guard_one_more_time():
    """The defect the fit exists for, priced with no model: same guard, two fold lengths."""
    short = compact_fold_input_probe(
        max_prompt_chars=7000, controller=fold_length_controller(120), **GEOMETRY
    )
    long = compact_fold_input_probe(
        max_prompt_chars=7000, controller=fold_length_controller(700), **GEOMETRY
    )
    assert short["n_compactions"] == 2
    assert long["n_compactions"] > short["n_compactions"]


def test_the_committed_design_predeclares_a_band_that_holds_the_declared_guard():
    design = load_repeated_fold_replication_design()
    validate_replication_design(design)
    spec = guard_fit_spec(design)
    low, high, step = search_band(spec)
    cell = _fitted_cell(design)
    assert low <= cell["max_prompt_chars"] <= high
    assert step > 0
    assert spec["target_folds"] == cell["expected_oracle_folds"]
    assert cell["cap_fitting_control"] is False


def test_a_band_reaching_the_cap_peak_is_refused():
    design = load_repeated_fold_replication_design()
    spec = {**guard_fit_spec(design), "search_max_chars": 20000}
    try:
        validate_replication_design({**design, "two_fold_guard_fit": spec})
    except ValueError as exc:
        assert "cap peak" in str(exc)
    else:
        raise AssertionError("a band that reaches the cap peak must be refused")


def test_a_band_reaching_the_deeper_cells_guard_is_refused():
    design = load_repeated_fold_replication_design()
    spec = {**guard_fit_spec(design), "search_min_chars": 6000}
    try:
        validate_replication_design({**design, "two_fold_guard_fit": spec})
    except ValueError as exc:
        assert "deeper cell" in str(exc)
    else:
        raise AssertionError("a band that overlaps the deeper cell must be refused")


def test_a_band_that_excludes_the_declared_guard_is_refused():
    design = load_repeated_fold_replication_design()
    spec = {**guard_fit_spec(design), "search_min_chars": 7100}
    try:
        validate_replication_design({**design, "two_fold_guard_fit": spec})
    except ValueError as exc:
        assert "declared guard" in str(exc)
    else:
        raise AssertionError("a band that cannot reproduce the declared guard must be refused")


def test_fitting_the_cap_fitting_control_is_refused():
    design = load_repeated_fold_replication_design()
    control = next(cell for cell in completion_cells(design) if cell["cap_fitting_control"])
    spec = {**guard_fit_spec(design), "cell_id": control["cell_id"], "target_folds": 1}
    try:
        validate_replication_design({**design, "two_fold_guard_fit": spec})
    except ValueError as exc:
        assert "control" in str(exc)
    else:
        raise AssertionError("fitting the eligibility control must be refused")


def test_a_replication_without_a_guard_fit_is_refused():
    design = load_repeated_fold_replication_design()
    try:
        validate_replication_design(
            {key: value for key, value in design.items() if key != "two_fold_guard_fit"}
        )
    except ValueError as exc:
        assert "two_fold_guard_fit" in str(exc)
    else:
        raise AssertionError("a shared-guard replication must be refused")


def test_a_short_summary_family_keeps_the_declared_guard():
    design = load_repeated_fold_replication_design()
    record = fit_fold_guard(
        design, _fitted_cell(design), design["held_fixed"], [120] * 12, evidence_floor=4
    )
    assert record["fitted_max_prompt_chars"] == record["declared_max_prompt_chars"]
    assert record["fit_reading"] == FIT_DECLARED
    assert record["meets_evidence_floor"] is True


def test_a_long_summary_family_moves_to_a_guard_that_reaches_the_rung():
    design = load_repeated_fold_replication_design()
    cell = _fitted_cell(design)
    record = fit_fold_guard(design, cell, design["held_fixed"], [700] * 12, evidence_floor=4)
    assert record["fitted_max_prompt_chars"] > record["declared_max_prompt_chars"]
    assert record["fit_reading"] == FIT_APPLIED
    assert record["declared_target_cases"] < record["predicted_target_cases"]
    assert record["meets_evidence_floor"] is True
    low, high, _step = search_band(guard_fit_spec(design))
    assert low <= record["fitted_max_prompt_chars"] <= high


def test_a_fit_that_cannot_reach_the_floor_is_named_rather_than_smoothed():
    design = load_repeated_fold_replication_design()
    record = fit_fold_guard(
        design, _fitted_cell(design), design["held_fixed"], [120], evidence_floor=4
    )
    assert record["fit_reading"] == FIT_UNDERPOWERED
    assert record["meets_evidence_floor"] is False
    assert "floor 4" in record["fit_reason"]


def test_a_control_that_measured_no_fold_leaves_the_declared_guard_standing():
    design = load_repeated_fold_replication_design()
    record = fit_fold_guard(
        design, _fitted_cell(design), design["held_fixed"], [], evidence_floor=4
    )
    assert record["fit_reading"] == FIT_UNMEASURED
    assert record["fitted_max_prompt_chars"] == record["declared_max_prompt_chars"]
    assert record["meets_evidence_floor"] is False


def test_the_fold_length_is_read_from_the_shipped_control_arm_alone():
    rows = [
        {
            "cell_id": "onefold-d10-g14000",
            "arm": "typed_marker",
            "cases": [{"summary_output_chars": [211]}, {"summary_output_chars": [199]}],
        },
        {
            "cell_id": "onefold-d10-g14000",
            "arm": "model_summary_only",
            "cases": [{"summary_output_chars": [9999]}],
        },
        {
            "cell_id": "twofold-d10-g7000",
            "arm": "typed_marker",
            "cases": [{"summary_output_chars": [8888, 8888]}],
        },
    ]
    assert measured_fold_lengths(rows, "onefold-d10-g14000") == [211, 199]


def test_the_summary_length_is_measured_before_typed_markers_are_prepended():
    state = ContextState()
    state.entries = [("read_file", {}, "hits [memory: final_code=MEM-001-002]")]
    folded = compact_state(ContextPolicy(name=POLICY_COMPACT), state, lambda _entries: "x" * 40)
    assert folded is True
    assert state.telemetry.summary_output_chars == [40]
    assert len(state.summary) > 40


def test_two_families_of_different_verbosity_both_reach_the_two_fold_rung():
    design = _fake_design()
    roster = replication_roster(design)[:2]
    runs = [
        run_replication_family(design, roster[0], complete=WritesSummariesOfLength(120)),
        run_replication_family(design, roster[1], complete=WritesSummariesOfLength(700)),
    ]
    analysis = analyze_replication_runs(design, runs)
    fits = {family["model_family"]: family["guard_fits"][0] for family in analysis["families"]}
    assert fits[roster[0]["model_family"]]["fit_reading"] == FIT_DECLARED
    assert fits[roster[1]["model_family"]]["fit_reading"] == FIT_APPLIED
    for family in analysis["families"]:
        measured = {row["measured_folds"] for row in family["fold_groups"]}
        assert {1, 2, 3} <= measured
    assert analysis["ladder_fully_powered"] is True
    assert "guard fit" in format_replication_table(analysis)


def test_the_shared_guard_leaves_the_verbose_family_off_the_two_fold_rung():
    """Without the fit, the defect reproduces: the long-summary family skips the middle rung."""
    design = _fake_design()
    shared = {key: value for key, value in design.items() if key != "two_fold_guard_fit"}
    roster = replication_roster(design)[:2]
    runs = [
        run_replication_family(shared, roster[0], complete=WritesSummariesOfLength(120)),
        run_replication_family(shared, roster[1], complete=WritesSummariesOfLength(700)),
    ]
    analysis = analyze_replication_runs(shared, runs)
    verbose = analysis["families"][1]
    assert verbose["guard_fits"] == []
    assert 2 not in {row["measured_folds"] for row in verbose["fold_groups"]}


def test_an_underfloor_rung_names_its_family_fold_count_and_guard():
    coverage = ladder_coverage(
        [
            {
                "model_family": "gemma4",
                "model": "gemma4:e4b",
                "evidence_floor": 4,
                "fold_groups": [
                    {"measured_folds": 1, "n_evidence": 12, "meets_evidence_floor": True},
                    {"measured_folds": 2, "n_evidence": 1, "meets_evidence_floor": False},
                ],
                "cells": [
                    {
                        "cell_id": "twofold-d10-g7000",
                        "arm": "typed_marker",
                        "max_prompt_chars": 7000,
                        "cases": [{"measured_folds": 2}, {"measured_folds": 3}],
                    }
                ],
            }
        ]
    )
    assert coverage["ladder_fully_powered"] is False
    hole = coverage["underpowered_ladder_rungs"][0]
    assert hole["model_family"] == "gemma4"
    assert hole["measured_folds"] == 2
    assert hole["max_prompt_chars"] == [7000]
    assert "below the floor of 4" in coverage["ladder_coverage_reason"]


class WritesLongSteps(WritesSummariesOfLength):
    """A family whose CALLS run long: same summarizer, a fatter transcript entry per step."""

    def __init__(self, summary_chars: int, note_chars: int):
        super().__init__(summary_chars)
        self.note_chars = note_chars

    def __call__(self, prompt: str) -> str:
        reply = super().__call__(prompt)
        if not reply.startswith("{"):
            return reply
        call = json.loads(reply)
        if call["name"] != "advance":
            return reply
        call["arguments"]["нотатка"] = "к" * self.note_chars
        return json.dumps(call, ensure_ascii=False)


def test_a_step_that_renders_longer_folds_a_shared_guard_one_more_time():
    """The half of the walk the replay used to assume, priced on its own."""
    lean = compact_fold_input_probe(
        max_prompt_chars=7000,
        controller=fold_length_controller(120, step_entry_chars=36),
        **GEOMETRY,
    )
    fat = compact_fold_input_probe(
        max_prompt_chars=7000,
        controller=fold_length_controller(120, step_entry_chars=600),
        **GEOMETRY,
    )
    assert fat["n_compactions"] > lean["n_compactions"]
    assert max(fat["step_entry_chars"]) == 600
    assert max(lean["step_entry_chars"]) < 600


def test_a_measured_step_at_the_oracle_length_leaves_the_replayed_walk_untouched():
    """Carrying the measurement must not move a family whose steps already match the oracle."""
    assumed = compact_fold_input_probe(
        max_prompt_chars=7000, controller=fold_length_controller(274), **GEOMETRY
    )
    walked = max(assumed["step_entry_chars"])
    measured = compact_fold_input_probe(
        max_prompt_chars=7000,
        controller=fold_length_controller(274, step_entry_chars=walked),
        **GEOMETRY,
    )
    assert measured == assumed


def test_a_step_length_below_the_oracles_own_call_is_never_shortened():
    """The oracle's call is the shortest one that plays this world; a fit cannot go under it."""
    floor = compact_fold_input_probe(
        max_prompt_chars=7000,
        controller=fold_length_controller(274, step_entry_chars=1),
        **GEOMETRY,
    )
    plain = compact_fold_input_probe(
        max_prompt_chars=7000, controller=fold_length_controller(274), **GEOMETRY
    )
    assert floor == plain


def test_the_episode_measures_what_each_step_appended_to_the_transcript():
    state = ContextState()
    state.record(ContextPolicy(name=POLICY_COMPACT), "advance", {"token": "wf-000-0"}, "x" * 500)
    assert state.telemetry.step_entry_chars == [
        len(format_entry(("advance", {"token": "wf-000-0"}, "")))
    ]
    assert state.telemetry.step_entry_chars[0] < 500


def test_the_step_length_is_read_from_the_shipped_control_arm_alone():
    rows = [
        {
            "cell_id": "onefold-d10-g14000",
            "arm": "typed_marker",
            "cases": [{"step_entry_chars": [36, 36]}, {"step_entry_chars": [37]}],
        },
        {
            "cell_id": "onefold-d10-g14000",
            "arm": "model_summary_only",
            "cases": [{"step_entry_chars": [9999]}],
        },
        {
            "cell_id": "twofold-d10-g7000",
            "arm": "typed_marker",
            "cases": [{"step_entry_chars": [8888]}],
        },
    ]
    assert measured_step_entry_chars(rows, "onefold-d10-g14000") == [36, 36, 37]


def test_a_family_whose_steps_run_long_is_fitted_to_a_wider_guard():
    """Two families with the SAME summarizer and different step lengths land on different guards."""
    design = load_repeated_fold_replication_design()
    cell = _fitted_cell(design)
    lean = fit_fold_guard(
        design,
        cell,
        design["held_fixed"],
        [200] * 12,
        evidence_floor=4,
        step_entry_chars=[oracle_step_entry_chars(cell, design["held_fixed"])] * 120,
    )
    fat = fit_fold_guard(
        design,
        cell,
        design["held_fixed"],
        [200] * 12,
        evidence_floor=4,
        step_entry_chars=[500] * 120,
    )
    assert lean["step_length_reading"] == STEP_LENGTH_AT_ORACLE
    assert fat["step_length_reading"] == STEP_LENGTH_ABOVE_ORACLE
    assert fat["fitted_max_prompt_chars"] > lean["fitted_max_prompt_chars"]
    assert fat["median_step_entry_chars"] == 500


def test_an_unmeasured_step_length_is_named_rather_than_assumed():
    design = load_repeated_fold_replication_design()
    record = fit_fold_guard(
        design, _fitted_cell(design), design["held_fixed"], [200] * 12, evidence_floor=4
    )
    assert record["step_length_reading"] == STEP_LENGTH_UNMEASURED
    assert record["median_step_entry_chars"] == 0
    assert record["step_entry_chars_range"] == []


def test_a_design_whose_step_length_comes_off_a_folding_cell_is_refused():
    design = load_repeated_fold_replication_design()
    spec = {**guard_fit_spec(design), "step_length_source": "twofold-d10-g6500"}
    try:
        validate_replication_design({**design, "two_fold_guard_fit": spec})
    except ValueError as exc:
        assert "step length" in str(exc)
    else:
        raise AssertionError("a step length measured off a fitted-regime cell must be refused")


def test_the_run_states_its_fits_prediction_error_and_names_a_divergence():
    design = _fake_design()
    roster = replication_roster(design)[:2]
    runs = [
        run_replication_family(design, roster[0], complete=WritesSummariesOfLength(120)),
        run_replication_family(design, roster[1], complete=WritesLongSteps(120, 300)),
    ]
    analysis = analyze_replication_runs(design, runs)
    table = format_replication_table(analysis)
    assert analysis["fit_prediction_reading"] in {PREDICTION_CALIBRATED, PREDICTION_DIVERGED}
    for family in analysis["families"]:
        fit = family["guard_fits"][0]
        assert fit["prediction_error_cases"] == (
            fit["measured_target_cases"] - fit["predicted_target_cases"]
        )
        assert fit["median_fitted_cell_fold_length_chars"] > 0
    assert "calibration: step-length" in table
    assert "fit prediction:" in table


def test_a_fit_whose_prediction_misses_is_named_with_its_family_and_guard():
    families = [
        {
            "model_family": "gemma4",
            "control_eligible": True,
            "guard_fits": [
                {
                    "cell_id": "twofold-d10-g7000",
                    "target_folds": 2,
                    "fitted_max_prompt_chars": 7900,
                    "predicted_target_cases": 12,
                    "measured_target_cases": 7,
                    "prediction_exact": False,
                    "median_fold_length_chars": 274,
                    "median_fitted_cell_fold_length_chars": 214,
                }
            ],
        }
    ]
    reading, reason = fit_prediction_reading(families)
    assert reading == PREDICTION_DIVERGED
    assert "gemma4/twofold-d10-g7000" in reason
    assert "guard 7900" in reason
    assert "274" in reason and "214" in reason


def test_the_fit_states_how_wrong_its_replayed_fold_length_may_be_per_guard():
    """The countability number: slack at the guard the fit picks against the one it was handed."""
    design = load_repeated_fold_replication_design()
    record = fit_fold_guard(
        design,
        _fitted_cell(design),
        design["held_fixed"],
        [274] * 12,
        evidence_floor=4,
        step_entry_chars=[36] * 120,
    )
    assert record["fold_count_margin_chars"] == FOLD_COUNT_MARGIN_SCAN_CHARS
    assert 0 <= record["declared_fold_count_margin_chars"] < record["fold_count_margin_chars"]


def test_a_guard_whose_fold_count_flips_nearby_reports_a_narrow_margin():
    """A margin is a scan, not a constant: the flip point decides it."""
    design = load_repeated_fold_replication_design()
    cell = _fitted_cell(design)
    predict = fold_count_predictor(cell, design["held_fixed"], 36, _FLAT)
    wide = fold_count_margin_chars(predict, 7900, 274)
    narrow = fold_count_margin_chars(predict, 7000, 274)
    assert wide > narrow
    assert predict(7000, 274 - narrow - FOLD_COUNT_MARGIN_STEP_CHARS) != predict(7000, 274)
    assert predict(7000, 274 - narrow) == predict(7000, 274)


def test_an_unmeasured_fold_length_has_no_margin_to_state():
    design = load_repeated_fold_replication_design()
    cell = _fitted_cell(design)
    predict = fold_count_predictor(cell, design["held_fixed"], 0, _FLAT)
    assert fold_count_margin_chars(predict, 7900, 0) == 0
