"""Summarize-input cap: the step-aligned bound, its arm contract, and the two readings it cuts."""

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from llb.bench.agentic.context_policy import (
    POLICY_COMPACT,
    SUMMARY_INPUT_CAP_TRIGGER,
    SUMMARY_INPUT_CAP_WINDOW,
    ContextPolicy,
)
from llb.bench.agentic.context_summary import summary_prompt_overhead_chars
from llb.bench.agentic.context_budget import fixed_budget
from llb.bench.agentic.episode_prompt import summary_input_cap_chars
from llb.bench.agentic_memory_boundary_probe import compact_fold_input_probe, oracle_controller
from llb.bench.agentic_memory_summary_cap import analyze_summary_cap, run_summary_cap_arms
from llb.bench.agentic_memory_summary_cap_design import (
    arm_fold_input_probes,
    declared_cells,
    load_summary_cap_design,
    summary_cap_prompt_sequence,
    validate_summary_cap_design,
)
from llb.bench.agentic_memory_summary_cap_reading import (
    ELISION_FREE,
    ELISION_NONE_TO_PRICE,
    READING_BOUNDARY_MOVED,
    READING_EXACT,
    READING_INELIGIBLE,
    READING_INVALID,
    READING_LADDER_UNREADABLE,
    READING_RESIDUAL_SURVIVES,
    ROLE_REFERENCE,
    ROLE_STEP_ALIGNED,
    STUDY_KIND,
)
from llb.bench.agentic_memory_summary_cap_report import (
    format_summary_cap_table,
    persist_summary_cap,
)

ROOT = Path(__file__).resolve().parents[3]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_summary_input_cap_design.json"

CONTROL_PASS = {
    "eligible": True,
    "model": "pinned",
    "completion": 1.0,
    "minimum_completion_rate": 0.75,
}
_MEMORY = re.compile(r"final_code=([A-Z0-9-]+)")


def _fake_model(prompt: str) -> str:
    """Perfect play, plus a fixed summary whenever the compact policy folds the transcript."""
    if "Стисло підсумуй" in prompt:
        return "[memory: final_code=" + (_MEMORY.findall(prompt) or [""])[0] + "] крок виконано"
    return oracle_controller(prompt)


# --- the runtime bound ----------------------------------------------------------------------


def test_the_window_cap_is_the_budget_minus_the_template_and_the_trigger_cap_is_the_trigger():
    budget = fixed_budget(20_000)
    window = ContextPolicy(name=POLICY_COMPACT, compact_share=0.5)
    trigger = ContextPolicy(
        name=POLICY_COMPACT, compact_share=0.5, summary_input_cap=SUMMARY_INPUT_CAP_TRIGGER
    )
    assert window.summary_input_cap == SUMMARY_INPUT_CAP_WINDOW  # the shipped bound
    assert summary_input_cap_chars(trigger, budget) == 10_000
    assert summary_input_cap_chars(window, budget) == 20_000 - summary_prompt_overhead_chars()
    # The window bound does not move with compact_share; the trigger bound is nothing but it.
    shifted = ContextPolicy(name=POLICY_COMPACT, compact_share=0.25)
    assert summary_input_cap_chars(shifted, budget) == summary_input_cap_chars(window, budget)
    # An unresolvable window refuses nothing under either bound.
    assert summary_input_cap_chars(window, fixed_budget(0)) == 0


def test_an_unknown_summary_input_cap_is_refused():
    with pytest.raises(ValueError, match="unknown summary input cap"):
        ContextPolicy(name=POLICY_COMPACT, summary_input_cap="half")


def test_the_step_aligned_bound_summarizes_the_folded_transcript_whole():
    """Two guards folding the SAME step offer the summarizer the same bytes -- and elide none."""
    geometry = dict(depth=10, n_tasks=3, compact_share=0.5)
    lo, hi = (
        compact_fold_input_probe(max_prompt_chars=guard, **geometry) for guard in (20240, 22014)
    )
    assert lo["summary_input_chars"] == hi["summary_input_chars"]
    assert lo["summary_input_elided_chars"] == hi["summary_input_elided_chars"] == 0
    # Under the trigger bound the same fold step feeds the summarizer different amounts.
    elided = [
        compact_fold_input_probe(
            max_prompt_chars=guard, summary_input_cap=SUMMARY_INPUT_CAP_TRIGGER, **geometry
        )["summary_input_elided_chars"]
        for guard in (20240, 22014)
    ]
    assert elided[0] > 0 and elided[1] == 0


# --- the design contract --------------------------------------------------------------------


def test_the_committed_design_places_two_arms_over_one_adjacent_fold_step_ladder():
    design = load_summary_cap_design(DESIGN_PATH)
    validate_summary_cap_design(design)
    cells = declared_cells(design)
    assert {cell["arm_id"] for cell in cells} == {"trigger-cap", "window-cap"}
    assert {cell["role"] for cell in cells} == {ROLE_REFERENCE, ROLE_STEP_ALIGNED}
    steps = sorted({int(cell["fold_step"]) for cell in cells})
    assert steps == list(range(steps[0], steps[0] + len(steps)))
    assert summary_cap_prompt_sequence(design)[-1] > 0
    probes = arm_fold_input_probes(design)
    reference = [probe for probe in probes if probe["summary_input_cap"] == "trigger"]
    aligned = [probe for probe in probes if probe["summary_input_cap"] == "window"]
    assert max(probe["summary_input_elided_chars"] for probe in reference) > 0
    assert all(probe["summary_input_elided_chars"] == 0 for probe in aligned)


def test_the_design_refuses_arms_that_do_not_isolate_the_summarize_input_bound():
    design = load_summary_cap_design(DESIGN_PATH)

    one_arm = deepcopy(design)
    one_arm["arms"] = one_arm["arms"][:1]
    with pytest.raises(ValueError, match="exactly two uniquely named arms"):
        validate_summary_cap_design(one_arm)

    same_cap = deepcopy(design)
    same_cap["arms"][0]["summary_input_cap"] = "window"
    with pytest.raises(ValueError, match="distinct summarize-input cap"):
        validate_summary_cap_design(same_cap)

    swapped = deepcopy(design)
    swapped["arms"][0]["summary_input_cap"] = "window"
    swapped["arms"][1]["summary_input_cap"] = "trigger"
    with pytest.raises(ValueError, match="reference arm must pin the trigger cap"):
        validate_summary_cap_design(swapped)

    # The cap is the ARM; holding it fixed would make the study measure nothing.
    held = deepcopy(design)
    held["held_fixed"]["summary_input_cap"] = "window"
    with pytest.raises(ValueError, match="this study's ARM"):
        validate_summary_cap_design(held)


def test_the_design_refuses_a_ladder_with_no_elision_to_price_or_a_loose_placement():
    design = load_summary_cap_design(DESIGN_PATH)

    mislabelled = deepcopy(design)
    mislabelled["ladder"]["steps"][0]["cells"][0]["max_prompt_chars"] = 22016
    with pytest.raises(ValueError, match="folds at step 11, not the declared 10"):
        validate_summary_cap_design(mislabelled)

    narrow = deepcopy(design)
    narrow["ladder"]["steps"][0]["cells"][1]["max_prompt_chars"] = 20244
    with pytest.raises(ValueError, match="guards span"):
        validate_summary_cap_design(narrow)

    loose = deepcopy(design)
    loose["ladder"]["steps"][1]["cells"][0]["max_prompt_chars"] = 22400
    loose["ladder"]["steps"][1]["cells"][1]["max_prompt_chars"] = 23851
    with pytest.raises(ValueError, match="straddles the step"):
        validate_summary_cap_design(loose)

    # Depth 6 folds a transcript neither trigger cap trims, so it has no elided span to price.
    shallow = deepcopy(design)
    shallow["ladder"] = {
        "depth": 6,
        "steps": [
            {
                "fold_step": 6,
                "expected_side": "compact_cheaper",
                "cells": [
                    {"cell_id": "a", "max_prompt_chars": 13136, "expected_side": "compact_cheaper"},
                    {"cell_id": "b", "max_prompt_chars": 14910, "expected_side": "compact_cheaper"},
                ],
            },
            {
                "fold_step": 7,
                "expected_side": "cap_cheaper",
                "cells": [
                    {"cell_id": "c", "max_prompt_chars": 14912, "expected_side": "cap_cheaper"},
                    {"cell_id": "d", "max_prompt_chars": 16746, "expected_side": "cap_cheaper"},
                ],
            },
        ],
    }
    with pytest.raises(ValueError, match="no trimmed span to price"):
        validate_summary_cap_design(shallow)


def test_a_ladder_the_probe_measured_nothing_over_names_the_ladder_not_an_empty_iterable():
    """The design and the analysis each reduce the same walk to a peak, and both say which ladder.

    The analysis peak is read even when the family is ineligible -- it describes the geometry, not
    the cells -- so an unmeasured ladder has to be named there too rather than dying as a builtin
    `max() iterable argument is empty` while reporting an ineligible run.
    """
    design = load_summary_cap_design(DESIGN_PATH)
    depth = design["ladder"]["depth"]

    unmeasured = deepcopy(design)
    unmeasured["held_fixed"]["max_steps_margin"] = -depth
    for read in (
        validate_summary_cap_design,
        lambda moved: analyze_summary_cap(moved, {**CONTROL_PASS, "eligible": False}, []),
    ):
        with pytest.raises(ValueError, match=f"the depth {depth} ladder measured no prompt"):
            read(unmeasured)


# --- the readings ---------------------------------------------------------------------------


def test_the_committed_arms_run_and_persist_under_perfect_play(tmp_path: Path):
    design = load_summary_cap_design(DESIGN_PATH)
    rows = run_summary_cap_arms(
        design, model="fake", backend="fake", complete=_fake_model, data_dir=tmp_path
    )
    analysis = analyze_summary_cap(design, CONTROL_PASS, rows)
    for cell in analysis["cells"]:
        assert cell["valid"] is True, cell["invalid_reason"]
        assert cell["compaction_activation_rate"] == 1.0, cell["cell_id"]
    arms = {row["role"]: row for row in analysis["arms"]}
    reference, aligned = arms[ROLE_REFERENCE], arms[ROLE_STEP_ALIGNED]
    # The whole point: under the trigger bound the cost slides inside a fold step, and pinning the
    # bound to a step-aligned quantity drives that residual to zero.
    assert reference["within_step_residual_tokens"] > 0.0
    assert aligned["within_step_residual_tokens"] == pytest.approx(0.0)
    assert reference["max_summary_input_elided_chars"] > 0.0
    assert aligned["max_summary_input_elided_chars"] == 0.0
    # ... without moving the fold step the routing rule is stated on.
    assert (
        aligned["last_compact_cheaper_fold_step"]
        == reference["last_compact_cheaper_fold_step"]
        == 10
    )
    assert analysis["summary_cap_reading"] == READING_EXACT
    assert analysis["elision_reading"] == ELISION_FREE
    assert analysis["shipped_summary_input_cap"] == SUMMARY_INPUT_CAP_WINDOW
    assert any("step function of the fold step" in line for line in analysis["operator_lines"])

    table = format_summary_cap_table(analysis)
    paths = persist_summary_cap(
        design, analysis, data_dir=tmp_path, table=table, tokens_per_s=0.0, mirror=lambda *_: None
    )
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["config"]["study_kind"] == STUDY_KIND
    assert "model-free summarizer-input probe" in table and "arm window-cap" in table


def test_an_ineligible_family_an_invalid_cell_and_a_surviving_residual_are_all_named(
    tmp_path: Path,
):
    design = load_summary_cap_design(DESIGN_PATH)
    rows = run_summary_cap_arms(
        design, model="fake", backend="fake", complete=_fake_model, data_dir=tmp_path
    )

    ineligible = analyze_summary_cap(design, {**CONTROL_PASS, "eligible": False}, rows)
    assert ineligible["summary_cap_reading"] == READING_INELIGIBLE
    assert ineligible["elision_reading"] == ELISION_NONE_TO_PRICE
    assert ineligible["changes_shipped_default"] is False

    overflowed = deepcopy(rows)
    overflowed[0]["cap_context_overflows"] = 1
    assert analyze_summary_cap(design, CONTROL_PASS, overflowed)["summary_cap_reading"] == (
        READING_INVALID
    )

    # A step-aligned arm whose cost still slides inside a step is a named failure, not a pass.
    tightened = deepcopy(design)
    tightened["cap_rule"]["residual_tolerance_tokens"] = 0.0
    slid = deepcopy(rows)
    aligned_id = next(arm["arm_id"] for arm in design["arms"] if arm["role"] == ROLE_STEP_ALIGNED)
    for row in slid:
        if row["arm_id"] == aligned_id and row["declared_cell_id"].endswith("step10-hi"):
            row["compact_mean_total_model_input_tokens"] += 50.0
            row["paired"]["total_model_input_tokens"]["delta"] = {
                "mean": row["paired"]["total_model_input_tokens"]["delta"]["mean"] + 50.0,
                "lo": row["paired"]["total_model_input_tokens"]["delta"]["lo"] + 50.0,
                "hi": row["paired"]["total_model_input_tokens"]["delta"]["hi"] + 50.0,
            }
    survived = analyze_summary_cap(tightened, CONTROL_PASS, slid)
    assert survived["summary_cap_reading"] == READING_RESIDUAL_SURVIVES
    # The residual is named in the operator lines rather than papered over by the headline claim.
    assert not any(
        "step function of the fold step alone" in line for line in survived["operator_lines"]
    )
    assert any("not yet a pure step function" in line for line in survived["operator_lines"])


def test_a_moved_fold_step_boundary_withdraws_the_operator_line(tmp_path: Path):
    """The routing rule is stated on the fold step; a cap that moves it invalidates the rule."""
    design = load_summary_cap_design(DESIGN_PATH)
    rows = run_summary_cap_arms(
        design, model="fake", backend="fake", complete=_fake_model, data_dir=tmp_path
    )
    aligned_id = next(arm["arm_id"] for arm in design["arms"] if arm["role"] == ROLE_STEP_ALIGNED)
    moved = deepcopy(rows)
    for row in moved:
        # Flip the step-10 cells of the step-aligned arm to the cap-cheaper side, which drags the
        # last compact-cheaper fold step below the one the reference arm reports.
        if row["arm_id"] == aligned_id and "step10" in row["declared_cell_id"]:
            row["verdict"] = "prefer_cap"
            delta = row["paired"]["total_model_input_tokens"]["delta"]
            shift = 2 * abs(delta["mean"]) + 100.0
            row["paired"]["total_model_input_tokens"].update(
                {
                    "delta": {key: value + shift for key, value in delta.items()},
                    "wins": row["paired"]["total_model_input_tokens"]["losses"],
                    "losses": row["paired"]["total_model_input_tokens"]["wins"],
                }
            )
    analysis = analyze_summary_cap(design, CONTROL_PASS, moved)
    assert analysis["summary_cap_reading"] in (READING_BOUNDARY_MOVED, READING_LADDER_UNREADABLE)
    assert analysis["changes_shipped_default"] is False
    assert not any("summarized at its own size" in line for line in analysis["operator_lines"])


def test_the_run_refuses_rows_that_do_not_match_the_declared_arm_and_cell_order(tmp_path: Path):
    design = load_summary_cap_design(DESIGN_PATH)
    rows = run_summary_cap_arms(
        design, model="fake", backend="fake", complete=_fake_model, data_dir=tmp_path
    )
    with pytest.raises(ValueError, match="declared arm/cell order"):
        analyze_summary_cap(design, CONTROL_PASS, list(reversed(rows)))
