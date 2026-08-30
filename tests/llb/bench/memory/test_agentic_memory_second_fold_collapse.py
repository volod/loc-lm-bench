"""Second-fold trigger restatement: regime contract, anchor pairing, noise floor, and readings."""

from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from llb.bench.memory.boundary.probe import oracle_controller
from llb.bench.memory.second_fold.design import (
    validate_second_fold_design,
)
from llb.bench.memory.second_fold.geometry import (
    load_second_fold_design,
    probe_second_fold_cell,
    second_fold_cells,
)
from llb.bench.memory.second_fold.reading import (
    MIN_REPEATED_FOLDS,
    READING_COLLAPSES,
    READING_GUARD_REENTERS,
    READING_INELIGIBLE,
    READING_INVALID,
    READING_NO_POWER,
    READING_UNSTABLE,
    STUDY_KIND,
)
from llb.bench.memory.second_fold.report import format_second_fold_table, persist_second_fold
from llb.bench.memory.second_fold.run import analyze_second_fold, run_second_fold_cells

ROOT = Path(__file__).resolve().parents[4]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_second_fold_trigger_design.json"

CONTROL_PASS = {
    "eligible": True,
    "model": "pinned",
    "completion": 1.0,
    "minimum_completion_rate": 0.75,
}
_MEMORY = re.compile(r"final_code=([A-Z0-9-]+)")
# One anchor cost per family, in declared order, plus the per-cell costs a fixture row carries.
_ANCHOR_TOKENS = 16000.0


def _fake_model(prompt: str) -> str:
    """Perfect play, plus a fixed summary whenever the compact policy folds the transcript."""
    if "Стисло підсумуй" in prompt or "підсумок" in prompt.split("\n")[0]:
        return "[memory: final_code=" + (_MEMORY.findall(prompt) or [""])[0] + "] крок виконано"
    return oracle_controller(prompt)


def _cell_row(design: dict, cell: dict, cost: float, **overrides: object) -> dict:
    held = design["held_fixed"]
    n_tasks = held["n_tasks"]
    row = {
        "cell_id": cell["cell_id"],
        "family_id": cell["family_id"],
        "depth": cell["depth"],
        "compact_share": cell["compact_share"],
        "max_prompt_chars": cell["max_prompt_chars"],
        "repeats_anchor": cell.get("repeats_anchor"),
        "expected_separation": cell["expected"]["separates_from_anchor"],
        **probe_second_fold_cell(cell, held),
        "n_tasks": n_tasks,
        "completion": 1.0,
        "case_success": [1.0] * n_tasks,
        "case_total_model_input_tokens": [cost] * n_tasks,
        "mean_total_model_input_tokens": cost,
        "mean_compaction_prompt_tokens": 500.0,
        "mean_controller_prompt_tokens": cost - 500.0,
        "measured_fold_counts": [2] * n_tasks,
        "measured_fold_input_chars": [
            {
                "fold": 1,
                "n_episodes": n_tasks,
                "mean_offered_chars": 2646.0,
                "max_offered_chars": 2646.0,
            }
        ],
        "context_overflows": 0,
        "statuses": ["ok"] * n_tasks,
        "valid": True,
        "invalid_reason": None,
    }
    row.update(overrides)
    return row


def _rows(design: dict, costs: dict[str, float], **overrides: object) -> list[dict]:
    rows = [_cell_row(design, cell, costs[cell["cell_id"]]) for cell in second_fold_cells(design)]
    if overrides:
        rows[0].update(overrides)
    return rows


# Every member equals its anchor: the trigger-only rule would still hold through the second fold.
COLLAPSING = {
    "secondfold-d10-s0.8-g7000": _ANCHOR_TOKENS,
    "secondfold-d10-s0.7-g8000": _ANCHOR_TOKENS,
    "secondfold-d10-s0.64-g8750": _ANCHOR_TOKENS,
    "secondfold-d10-s0.56-g10000": _ANCHOR_TOKENS,
    "contrast-d10-s0.8-g7000": _ANCHOR_TOKENS,
    "contrast-d10-s0.6-g7000": _ANCHOR_TOKENS + 900.0,
    "contrast-d10-s0.9-g7000": _ANCHOR_TOKENS + 100.0,
    "contrast-d10-s0.95-g7000": _ANCHOR_TOKENS + 700.0,
}


def test_committed_design_is_below_the_cap_peak_and_folds_repeatedly():
    design = load_second_fold_design(DESIGN_PATH)
    validate_second_fold_design(design)
    for cell in second_fold_cells(design):
        probe = probe_second_fold_cell(cell, design["held_fixed"])
        assert probe["below_cap_peak"] is True, cell["cell_id"]
        assert probe["oracle_folds"] >= MIN_REPEATED_FOLDS, cell["cell_id"]
    families = {family["family_id"]: family for family in design["families"]}
    trigger_family = families["d10-trigger-5600"]
    triggers = {
        probe_second_fold_cell({**cell, "depth": trigger_family["depth"]}, design["held_fixed"])[
            "compaction_trigger_chars"
        ]
        for cell in trigger_family["cells"]
    }
    guards = {cell["max_prompt_chars"] for cell in trigger_family["cells"]}
    assert triggers == {5600} and len(guards) == len(trigger_family["cells"])


def test_design_refuses_a_cap_fitting_cell_a_one_fold_cell_and_a_blind_contrast():
    design = load_second_fold_design(DESIGN_PATH)

    cap_fitting = deepcopy(design)
    cap_fitting["families"][0]["cells"][0].update({"compact_share": 0.5, "max_prompt_chars": 14000})
    with pytest.raises(ValueError, match="cap-fitting"):
        validate_second_fold_design(cap_fitting)

    # Same 5600-char trigger, but a guard so wide the post-fold prompt never crosses it again.
    one_fold = deepcopy(design)
    one_fold["families"][0]["cells"][3].update({"compact_share": 0.5, "max_prompt_chars": 11200})
    with pytest.raises(ValueError, match="folds 1 time"):
        validate_second_fold_design(one_fold)

    # Two contrast triggers inside ONE fold-step interval produce the identical transcript.
    blind = deepcopy(design)
    blind["families"][1]["cells"][1].update(
        {
            "compact_share": 0.75,
            "predeclared": {
                "compaction_trigger_chars": 5250,
                "first_fold_step": 4,
                "oracle_folds": 2,
                "oracle_fold_input_chars": [2646, 5327],
                "oracle_model_input_chars": 63922,
            },
        }
    )
    with pytest.raises(ValueError, match="identical transcript"):
        validate_second_fold_design(blind)

    drifted = deepcopy(design)
    drifted["families"][0]["cells"][1]["predeclared"]["oracle_fold_input_chars"] = [2646, 5327]
    with pytest.raises(ValueError, match="predeclares"):
        validate_second_fold_design(drifted)

    unchecked = deepcopy(design)
    unchecked["families"][0]["cells"][0]["predeclared"]["second_fold_step"] = 9
    with pytest.raises(ValueError, match="which the probe does not measure"):
        validate_second_fold_design(unchecked)

    no_floor = deepcopy(design)
    no_floor["held_fixed"]["minimum_measured_folds"] = 1
    with pytest.raises(ValueError, match="one-fold regime"):
        validate_second_fold_design(no_floor)

    no_repeat = deepcopy(design)
    no_repeat["families"][1]["cells"][0].pop("repeats_anchor")
    with pytest.raises(ValueError, match="noise floor"):
        validate_second_fold_design(no_repeat)


def test_equal_trigger_agreement_reads_as_a_collapse_only_with_a_moving_contrast():
    design = load_second_fold_design(DESIGN_PATH)
    analysis = analyze_second_fold(design, CONTROL_PASS, _rows(design, COLLAPSING))
    assert analysis["second_fold_reading"] == READING_COLLAPSES
    families = {row["family_id"]: row for row in analysis["families"]}
    assert families["d10-trigger-5600"]["triggers"] == [5600]
    assert families["d10-trigger-5600"]["spread"] == pytest.approx(0.0)
    assert families["d10-trigger-5600"]["equivalence_band"] == pytest.approx(320.0)
    assert families["d10-guard-7000"]["separated_members"] == [
        "contrast-d10-s0.6-g7000",
        "contrast-d10-s0.95-g7000",
    ]
    assert analysis["repeat_geometry"]["reproduces"] is True
    assert analysis["changes_shipped_default"] is False

    flat = analyze_second_fold(
        design,
        CONTROL_PASS,
        _rows(
            design,
            dict(
                COLLAPSING,
                **{
                    "contrast-d10-s0.6-g7000": _ANCHOR_TOKENS,
                    "contrast-d10-s0.9-g7000": _ANCHOR_TOKENS,
                    "contrast-d10-s0.95-g7000": _ANCHOR_TOKENS,
                },
            ),
        ),
    )
    assert flat["second_fold_reading"] == READING_NO_POWER
    assert "no member cleared" in flat["reason"]


def test_a_moved_equal_trigger_family_bounds_the_rule_to_one_fold():
    design = load_second_fold_design(DESIGN_PATH)
    moved = dict(COLLAPSING, **{"secondfold-d10-s0.56-g10000": _ANCHOR_TOKENS + 3000.0})
    analysis = analyze_second_fold(design, CONTROL_PASS, _rows(design, moved))
    assert analysis["second_fold_reading"] == READING_GUARD_REENTERS
    assert "d10-trigger-5600" in analysis["reason"]
    member = analysis["families"][0]["member_deltas"][-1]
    assert member["separates"] is True and member["paired_delta"]["mean"] == pytest.approx(3000.0)


def test_the_repeat_pair_and_the_cell_preconditions_gate_the_claim():
    design = load_second_fold_design(DESIGN_PATH)
    unstable = analyze_second_fold(
        design,
        CONTROL_PASS,
        _rows(design, dict(COLLAPSING, **{"contrast-d10-s0.8-g7000": _ANCHOR_TOKENS + 800.0})),
    )
    assert unstable["second_fold_reading"] == READING_UNSTABLE
    assert "no spread here is readable" in unstable["reason"]

    invalid = analyze_second_fold(
        design,
        CONTROL_PASS,
        _rows(design, COLLAPSING, valid=False, invalid_reason="the compact arm overflowed"),
    )
    assert invalid["second_fold_reading"] == READING_INVALID

    ineligible = analyze_second_fold(design, {"eligible": False, "model": "pinned"}, [])
    assert ineligible["second_fold_reading"] == READING_INELIGIBLE
    assert ineligible["cells"] == [] and ineligible["families"] == []


def test_perfect_play_separates_the_equal_trigger_family_and_persists(tmp_path: Path):
    design = load_second_fold_design(DESIGN_PATH)
    rows, reports = run_second_fold_cells(
        design, model="fake", backend="fake", complete=_fake_model
    )
    analysis = analyze_second_fold(design, CONTROL_PASS, rows)
    for cell in analysis["cells"]:
        assert cell["valid"] is True, cell["invalid_reason"]
        assert min(cell["measured_fold_counts"]) >= MIN_REPEATED_FOLDS, cell["cell_id"]
    # The mechanism, with no model in reach: one trigger, one first fold step, and a SECOND fold
    # whose offered transcript grows with the guard hysteresis raised the trigger to.
    trigger_family = analysis["families"][0]
    assert trigger_family["first_fold_steps"] == [4]
    assert analysis["second_fold_reading"] == READING_GUARD_REENTERS
    second_fold_inputs = [
        [row["mean_offered_chars"] for row in cell["measured_fold_input_chars"]][1]
        for cell in analysis["cells"]
        if cell["family_id"] == trigger_family["family_id"]
    ]
    assert second_fold_inputs == sorted(second_fold_inputs)
    assert len(set(second_fold_inputs)) == len(second_fold_inputs)
    assert analysis["repeat_geometry"]["reproduces"] is True
    for family in analysis["families"]:
        for member in family["member_deltas"]:
            assert member["separates"] is member["expected_separation"], member["cell_id"]

    table = format_second_fold_table(analysis)
    paths = persist_second_fold(
        design,
        analysis,
        reports,
        data_dir=tmp_path,
        table=table,
        tokens_per_s=0.0,
        mirror=lambda *_: None,
    )
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["config"]["study_kind"] == STUDY_KIND
    assert "per-fold summarize input" in table and "repeat geometry" in table
    assert len(list((tmp_path / "agentic-compact-vs-cap").glob("*/manifest.json"))) == 9
