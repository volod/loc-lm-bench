"""Fold-step crossover: ladder placement, within-step agreement, and the step-change boundary.

The study's subject is the committed design and what a ladder of runs reads off it. The interval
arithmetic it stands on -- `agentic_memory_fold_step_ladder` -- is tested on its own in
`test_agentic_memory_fold_step_ladder.py`.
"""

from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from llb.bench.agentic_memory_boundary_probe import oracle_controller
from llb.bench.agentic_memory_fold_step_ladder import (
    compaction_trigger_chars,
    first_fold_step,
    foldable_fold_steps,
    guard_is_cap_fitting,
    reachable_fold_steps,
)
from llb.bench.agentic_memory_fold_step import analyze_fold_steps, run_fold_step_ladders
from llb.bench.agentic_memory_fold_step_design import (
    declared_cells,
    fold_step_cap_peaks,
    fold_step_prompt_sequences,
    load_fold_step_design,
    validate_fold_step_design,
)
from llb.bench.agentic_memory_fold_step_reading import (
    CONTROLLER_TOKEN_QUANTIZATION,
    READING_CONFIRMED,
    READING_INELIGIBLE,
    READING_INVALID,
    READING_NO_FLIP,
    READING_NO_POWER,
    READING_PARTIAL,
    READING_WITHIN_STEP,
    STUDY_KIND,
)
from llb.bench.agentic_memory_fold_step_report import format_fold_step_table, persist_fold_steps

ROOT = Path(__file__).resolve().parents[3]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_fold_step_crossover_design.json"

CONTROL_PASS = {
    "eligible": True,
    "model": "pinned",
    "completion": 1.0,
    "minimum_completion_rate": 0.75,
}
_MEMORY = re.compile(r"final_code=([A-Z0-9-]+)")

CAP_BASELINE = {6: 13258.0, 10: 27343.0}

# One value per declared cell, in declared order: both step-6 guards agree, both step-7 guards
# agree, and the side changes only between the two steps.
CONFIRMING = [-125.9, -125.9, +1054.6, +1054.6, -554.0, -554.0, +1524.9, +1524.9]


def _cell_row(cell: dict[str, object], cost_delta: float, **overrides: object) -> dict[str, object]:
    depth = int(cell["depth"])
    row: dict[str, object] = {
        "cell_id": cell["cell_id"],
        "depth": depth,
        "compact_share": 0.5,
        "max_prompt_chars": cell["max_prompt_chars"],
        "declared_fold_step": cell["fold_step"],
        "require_cap_fits": True,
        "verdict": "prefer_compact" if cost_delta < 0 else "prefer_cap",
        "cap_completion": 1.0,
        "compact_completion": 1.0,
        "cap_mean_total_model_input_tokens": CAP_BASELINE[depth],
        "compact_mean_total_model_input_tokens": CAP_BASELINE[depth] + cost_delta,
        "cap_context_overflows": 0,
        "compact_context_overflows": 0,
        "compaction_activation_rate": 1.0,
        "paired": {
            "completion": {
                "delta": {"mean": 0.0, "lo": 0.0, "hi": 0.0},
                "wins": 0,
                "losses": 0,
                "ties": 7,
                "sign_test_p": 1.0,
                "stability": {"tighter_reading": "flat"},
            },
            "total_model_input_tokens": {
                "delta": {"mean": cost_delta, "lo": cost_delta - 20, "hi": cost_delta + 20},
                "wins": 7 if cost_delta > 0 else 0,
                "losses": 0 if cost_delta > 0 else 7,
                "ties": 0,
                "sign_test_p": 0.015625,
            },
        },
    }
    row.update(overrides)
    return row


def _rows(
    design: dict[str, object], deltas: list[float], **overrides: object
) -> list[dict[str, object]]:
    rows = [
        _cell_row(cell, delta) for cell, delta in zip(declared_cells(design), deltas, strict=True)
    ]
    if overrides:
        rows[0].update(overrides)
    return rows


def test_committed_ladders_sit_one_fold_step_apart_inside_the_usable_band():
    design = load_fold_step_design(DESIGN_PATH)
    validate_fold_step_design(design)
    sequences = fold_step_prompt_sequences(design)
    peaks = fold_step_cap_peaks(design)
    for cell in declared_cells(design):
        depth, guard = int(cell["depth"]), int(cell["max_prompt_chars"])
        assert (
            first_fold_step(sequences[depth], compaction_trigger_chars(guard, 0.5))
            == (cell["fold_step"])
        )
        assert guard_is_cap_fitting(guard, peaks[depth], 0.5)
    for ladder in design["ladders"]:
        steps = [step["fold_step"] for step in ladder["steps"]]
        assert steps == list(range(steps[0], steps[0] + len(steps)))


def test_design_refuses_a_mislabelled_step_a_gap_in_the_ladder_and_a_loose_placement():
    design = load_fold_step_design(DESIGN_PATH)

    mislabelled = deepcopy(design)
    mislabelled["ladders"][0]["steps"][0]["cells"][0]["max_prompt_chars"] = 14912
    with pytest.raises(ValueError, match="folds at step 7, not the declared 6"):
        validate_fold_step_design(mislabelled)

    # A ladder that skips a step measures a two-step jump, not a step change.
    skipped = deepcopy(design)
    skipped["ladders"][0]["steps"][0].update(
        {
            "fold_step": 4,
            "cells": [
                {"cell_id": "a", "max_prompt_chars": 9584, "expected_side": "compact_cheaper"},
                {"cell_id": "b", "max_prompt_chars": 11358, "expected_side": "compact_cheaper"},
            ],
        }
    )
    with pytest.raises(ValueError, match="ADJACENT"):
        validate_fold_step_design(skipped)

    # Two nearly identical guards inside one step cannot show that the whole interval agrees.
    narrow = deepcopy(design)
    narrow["ladders"][0]["steps"][0]["cells"][1]["max_prompt_chars"] = 13140
    with pytest.raises(ValueError, match="guards span"):
        validate_fold_step_design(narrow)

    # A wide gap across the step change puts the flip anywhere in between, which is the old reading.
    loose = deepcopy(design)
    loose["ladders"][0]["steps"][1]["cells"][0]["max_prompt_chars"] = 15500
    with pytest.raises(ValueError, match="straddles the step"):
        validate_fold_step_design(loose)

    sided = deepcopy(design)
    sided["ladders"][0]["steps"][0]["cells"][0]["expected_side"] = "cap_cheaper"
    with pytest.raises(ValueError, match="the step's own side"):
        validate_fold_step_design(sided)

    unfittable = deepcopy(design)
    unfittable["step_rule"]["within_step_cap_cost_fraction"] = 0.5
    with pytest.raises(ValueError, match="placement bounds"):
        validate_fold_step_design(unfittable)


def test_a_ladder_that_declares_a_step_no_episode_folds_at_is_refused_by_the_ladder_rule():
    """The placement ladder is the steps a fold HAPPENS at, not the steps a trigger can select.

    Step 1 is reachable on both committed geometries -- a guard under the first prompt trips on it --
    and folds nothing, so a cell there would measure a `compact` arm that never compacts. The refusal
    used to be indirect and geometry-dependent: a step-1 guard sits far under the cap peak, so the
    per-CELL cap-fitting rule rejected it with a message about the usable guard band, which reads as
    a guard-placement mistake rather than as a ladder naming an impossible fold.
    """
    design = load_fold_step_design(DESIGN_PATH)
    sequences = fold_step_prompt_sequences(design)
    for sequence in sequences.values():
        assert reachable_fold_steps(sequence)[0] == 1
        assert foldable_fold_steps(sequence)[0] == 2

    unfoldable = deepcopy(design)
    unfoldable["ladders"][0]["steps"][0]["fold_step"] = 1
    with pytest.raises(ValueError, match="ADJACENT on the foldable ladder"):
        validate_fold_step_design(unfoldable)


def test_the_boundary_is_a_fold_step_and_the_interpolated_guard_is_an_artifact():
    design = load_fold_step_design(DESIGN_PATH)
    analysis = analyze_fold_steps(design, CONTROL_PASS, _rows(design, CONFIRMING))
    assert analysis["fold_step_reading"] == READING_CONFIRMED
    ladders = {row["depth"]: row for row in analysis["depth_ladders"]}
    assert ladders[6]["last_compact_cheaper_fold_step"] == 6
    boundary = ladders[6]["boundary"]
    assert boundary["trigger_boundary_chars"] == 7456
    assert boundary["guard_boundary_chars"] == 14912
    assert boundary["straddling_guard_gap_chars"] == 2
    assert boundary["step_change_separates"] is True
    # The surface's 14160-char crossover sits INSIDE step 6, where nothing changes.
    artifact = ladders[6]["interpolated_guard_artifact"]
    assert artifact["fold_step"] == 6
    assert artifact["guard_interval"] == [13136, 14912]
    assert artifact["gap_to_boundary_chars"] == 752
    assert ladders[10]["last_compact_cheaper_fold_step"] == 10
    assert ladders[10]["boundary"]["guard_boundary_chars"] == 22016
    rule = analysis["fold_step_routing_rule"]
    assert any("fold no later than step 6" in line for line in rule)
    assert any("nothing changes" in line for line in rule)
    # Equal deltas inside every step: the whole cost, not just the controller half, is a step.
    assert any("bit-identical across the guard interval" in line for line in rule)
    assert analysis["expectation_matches"] == 8
    assert analysis["changes_shipped_default"] is False


def test_two_guards_inside_one_fold_step_that_disagree_refuse_the_step_reading():
    design = load_fold_step_design(DESIGN_PATH)
    moved = list(CONFIRMING)
    moved[1] = +900.0  # the second step-6 guard flips side without a step change
    analysis = analyze_fold_steps(design, CONTROL_PASS, _rows(design, moved))
    assert analysis["fold_step_reading"] == READING_PARTIAL
    ladders = {row["depth"]: row for row in analysis["depth_ladders"]}
    assert ladders[6]["reading"] == READING_WITHIN_STEP
    assert ladders[6]["boundary"] is None
    assert ladders[10]["reading"] == READING_CONFIRMED

    every = [+900.0 if index < 2 else delta for index, delta in enumerate(CONFIRMING)]
    every[4] = -554.0 + 4000.0
    both = analyze_fold_steps(design, CONTROL_PASS, _rows(design, every))
    assert both["fold_step_reading"] == READING_WITHIN_STEP
    assert both["fold_step_routing_rule"] == []


def test_a_ladder_that_never_changes_side_reports_a_bound_instead_of_a_boundary():
    design = load_fold_step_design(DESIGN_PATH)
    analysis = analyze_fold_steps(
        design, CONTROL_PASS, _rows(design, [-125.9, -125.9, -100.0, -100.0] + CONFIRMING[4:])
    )
    ladders = {row["depth"]: row for row in analysis["depth_ladders"]}
    assert ladders[6]["reading"] == READING_NO_FLIP
    assert ladders[6]["last_compact_cheaper_fold_step"] is None
    assert analysis["fold_step_reading"] == READING_PARTIAL


def test_a_step_change_smaller_than_the_within_step_band_has_no_resolving_power():
    design = load_fold_step_design(DESIGN_PATH)
    # Band at depth 6 is 2% of 13258 = 265.2 tokens; a -1 -> +1 flip is far inside it.
    analysis = analyze_fold_steps(
        design, CONTROL_PASS, _rows(design, [-1.0, -1.0, +1.0, +1.0] + CONFIRMING[4:])
    )
    ladders = {row["depth"]: row for row in analysis["depth_ladders"]}
    assert ladders[6]["reading"] == READING_NO_POWER
    assert ladders[6]["boundary"]["step_change_separates"] is False


def test_cell_preconditions_and_eligibility_gate_the_study():
    design = load_fold_step_design(DESIGN_PATH)
    overflowed = analyze_fold_steps(
        design, CONTROL_PASS, _rows(design, CONFIRMING, cap_context_overflows=2)
    )
    assert overflowed["fold_step_reading"] == READING_INVALID
    assert "not cap-fitting" in overflowed["reason"]

    ineligible = analyze_fold_steps(design, {"eligible": False, "model": "pinned"}, [])
    assert ineligible["fold_step_reading"] == READING_INELIGIBLE
    assert ineligible["cells"] == [] and ineligible["depth_ladders"] == []

    drifted = _rows(design, CONFIRMING)
    drifted[0]["declared_fold_step"] = 5
    with pytest.raises(ValueError, match="no longer fold where they were declared"):
        analyze_fold_steps(design, CONTROL_PASS, drifted)


def _fake_model(prompt: str) -> str:
    """Perfect play, plus a fixed summary whenever the compact policy folds the transcript."""
    if "Стисло підсумуй" in prompt or "підсумок" in prompt.split("\n")[0]:
        return "[memory: final_code=" + (_MEMORY.findall(prompt) or [""])[0] + "] крок виконано"
    return oracle_controller(prompt)


def test_the_committed_ladders_run_and_persist_under_perfect_play(tmp_path: Path):
    design = load_fold_step_design(DESIGN_PATH)
    rows = run_fold_step_ladders(
        design, model="fake", backend="fake", complete=_fake_model, data_dir=tmp_path
    )
    analysis = analyze_fold_steps(design, CONTROL_PASS, rows)
    for cell in analysis["cells"]:
        assert cell["valid"] is True, cell["invalid_reason"]
        assert cell["compaction_activation_rate"] == 1.0, cell["cell_id"]
    for ladder in analysis["depth_ladders"]:
        assert ladder["reading"] != READING_WITHIN_STEP, ladder["depth"]
        # The fold step fixes the transcript the controller sees, so two guards spanning the whole
        # step interval send bit-identical controller prompts.
        assert ladder["controller_cost_is_exact_step"] is True, ladder["depth"]
        means = []
        for step in ladder["steps"]:
            assert step["controller_prompt_spread"] <= CONTROLLER_TOKEN_QUANTIZATION
            assert step["guard_span_chars"] >= 0.5 * step["guard_interval_width_chars"]
            # Whatever within-step movement survives is the summarize call, whose input cap IS the
            # trigger -- and it must stay far inside the equivalence band.
            assert step["spread"] == pytest.approx(
                step["summarizer_prompt_spread"], abs=CONTROLLER_TOKEN_QUANTIZATION
            )
            assert step["within_band"] is True, (ladder["depth"], step["fold_step"])
            means.append(step["mean_cost_delta"])
        # Adjacent steps must differ, or the ladder could not see a step change at all.
        assert len(set(means)) == len(means), ladder["depth"]
    # Depth 6 folds a transcript smaller than either trigger cap, so its steps are bit-identical;
    # depth 10 folds a longer one, so the cap bites and leaves a small residual.
    ladders = {row["depth"]: row for row in analysis["depth_ladders"]}
    assert ladders[6]["within_step_residual_tokens"] == pytest.approx(0.0)
    assert ladders[10]["within_step_residual_tokens"] > 0.0
    assert ladders[10]["within_step_summarizer_residual_tokens"] == pytest.approx(
        ladders[10]["within_step_residual_tokens"], abs=CONTROLLER_TOKEN_QUANTIZATION
    )
    assert any(
        "still move INSIDE a step" in line and "summarize call" in line
        for line in analysis["fold_step_routing_rule"]
    )
    table = format_fold_step_table(analysis)
    paths = persist_fold_steps(
        design, analysis, data_dir=tmp_path, table=table, tokens_per_s=0.0, mirror=lambda *_: None
    )
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["config"]["study_kind"] == STUDY_KIND
    assert "ladder" in table and "fold-step routing rule" in table
