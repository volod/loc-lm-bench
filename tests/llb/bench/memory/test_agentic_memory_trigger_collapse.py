"""Trigger/guard collapse: grid contract, equivalence band, positive control, and portable rule."""

from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from llb.bench.memory.boundary.probe import oracle_controller
from llb.bench.memory.fold_step.ladder import compaction_trigger_chars, guard_is_cap_fitting
from llb.bench.memory.trigger_collapse.run import analyze_collapse, run_collapse_families
from llb.bench.memory.trigger_collapse.design import (
    collapse_cap_peaks,
    load_collapse_design,
    validate_collapse_design,
)
from llb.bench.memory.trigger_collapse.reading import (
    READING_COLLAPSES,
    STUDY_KIND,
    READING_INELIGIBLE,
    READING_INTERACTS,
    READING_INVALID,
    READING_NO_POWER,
)
from llb.bench.memory.trigger_collapse.report import (
    format_collapse_table,
    persist_collapse,
)

ROOT = Path(__file__).resolve().parents[4]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_trigger_guard_collapse_design.json"

CONTROL_PASS = {
    "eligible": True,
    "model": "pinned",
    "completion": 1.0,
    "minimum_completion_rate": 0.75,
}
_MEMORY = re.compile(r"final_code=([A-Z0-9-]+)")

CAP_BASELINE = {6: 13258.0, 10: 27343.0}


def _cell_row(
    family: dict[str, object], cell: dict[str, object], cost_delta: float, **overrides: object
) -> dict[str, object]:
    depth = int(family["depth"])
    row: dict[str, object] = {
        "cell_id": cell["cell_id"],
        "family_id": family["family_id"],
        "depth": depth,
        "compact_share": cell["compact_share"],
        "max_prompt_chars": cell["max_prompt_chars"],
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


def _rows(design: dict[str, object], deltas: dict[str, list[float]], **overrides: object) -> list:
    rows: list[dict[str, object]] = []
    for family in design["families"]:
        for cell, delta in zip(family["cells"], deltas[family["family_id"]], strict=True):
            rows.append(_cell_row(family, cell, delta))
    if overrides:
        rows[0].update(overrides)
    return rows


COLLAPSING = {
    "d6-trigger-7000": [-130.0, -125.9, -121.0],
    "d6-guard-12000": [-1500.0, -792.0, -126.0],
    "d10-trigger-10000": [-2640.0, -2633.4],
}


def test_committed_design_holds_one_trigger_per_family_inside_the_usable_band():
    design = load_collapse_design(DESIGN_PATH)
    validate_collapse_design(design)
    peaks = collapse_cap_peaks(design)
    for family in design["families"]:
        triggers = {
            compaction_trigger_chars(cell["max_prompt_chars"], cell["compact_share"])
            for cell in family["cells"]
        }
        guards = {cell["max_prompt_chars"] for cell in family["cells"]}
        if family["kind"] == "equal_trigger":
            assert len(triggers) == 1 and len(guards) == len(family["cells"])
        else:
            assert len(guards) == 1 and len(triggers) == len(family["cells"])
        for cell in family["cells"]:
            assert guard_is_cap_fitting(
                cell["max_prompt_chars"], peaks[family["depth"]], cell["compact_share"]
            )


def test_design_refuses_a_broken_axis_a_missing_contrast_and_an_unfittable_guard():
    design = load_collapse_design(DESIGN_PATH)
    drifted = deepcopy(design)
    drifted["families"][0]["cells"][0]["max_prompt_chars"] = 17600
    with pytest.raises(ValueError, match="ONE trigger"):
        validate_collapse_design(drifted)

    no_contrast = deepcopy(design)
    no_contrast["families"][1]["kind"] = "equal_trigger"
    with pytest.raises(ValueError, match="one equal-guard"):
        validate_collapse_design(no_contrast)

    unnamed = deepcopy(design)
    unnamed["equivalence"]["contrast_family"] = "not-a-family"
    with pytest.raises(ValueError, match="equivalence band"):
        validate_collapse_design(unnamed)

    # Same 7000-char trigger, but a guard under the cap peak: cap would overflow, so the pair is
    # not a cap-fitting realization of that trigger at all.
    overflowing = deepcopy(design)
    overflowing["families"][0]["cells"][0].update({"compact_share": 0.9, "max_prompt_chars": 7778})
    with pytest.raises(ValueError, match="usable band"):
        validate_collapse_design(overflowing)


def test_collapse_holds_when_equal_trigger_families_stay_inside_the_band():
    design = load_collapse_design(DESIGN_PATH)
    analysis = analyze_collapse(design, CONTROL_PASS, _rows(design, COLLAPSING))
    assert analysis["collapse_reading"] == READING_COLLAPSES
    families = {row["family_id"]: row for row in analysis["families"]}
    assert families["d6-trigger-7000"]["triggers"] == [7000]
    assert families["d6-trigger-7000"]["spread"] == pytest.approx(9.0)
    assert families["d6-trigger-7000"]["equivalence_band"] == pytest.approx(265.16)
    assert families["d6-trigger-7000"]["within_band"] is True
    assert families["d6-guard-12000"]["within_band"] is False
    assert families["d10-trigger-10000"]["triggers"] == [10000]
    rule = analysis["portable_trigger_rule"]
    assert any("7080-char trigger" in line for line in rule)
    assert any("portable form" in line for line in rule)
    assert analysis["changes_shipped_default"] is False


def test_a_moved_equal_trigger_family_reports_an_interaction():
    design = load_collapse_design(DESIGN_PATH)
    moved = dict(COLLAPSING, **{"d6-trigger-7000": [-900.0, -125.9, +400.0]})
    analysis = analyze_collapse(design, CONTROL_PASS, _rows(design, moved))
    assert analysis["collapse_reading"] == READING_INTERACTS
    assert "d6-trigger-7000" in analysis["reason"]
    assert analysis["portable_trigger_rule"] == []


def test_a_contrast_family_that_does_not_move_refuses_to_claim_a_collapse():
    design = load_collapse_design(DESIGN_PATH)
    flat = dict(COLLAPSING, **{"d6-guard-12000": [-130.0, -125.9, -121.0]})
    analysis = analyze_collapse(design, CONTROL_PASS, _rows(design, flat))
    assert analysis["collapse_reading"] == READING_NO_POWER
    assert "would say nothing" in analysis["reason"]
    assert analysis["portable_trigger_rule"] == []


def test_cell_preconditions_and_eligibility_gate_the_study():
    design = load_collapse_design(DESIGN_PATH)
    overflowed = analyze_collapse(
        design, CONTROL_PASS, _rows(design, COLLAPSING, cap_context_overflows=2)
    )
    assert overflowed["collapse_reading"] == READING_INVALID
    assert "not cap-fitting" in overflowed["reason"]

    ineligible = analyze_collapse(design, {"eligible": False, "model": "pinned"}, [])
    assert ineligible["collapse_reading"] == READING_INELIGIBLE
    assert ineligible["cells"] == [] and ineligible["families"] == []


def _fake_model(prompt: str) -> str:
    """Perfect play, plus a fixed summary whenever the compact policy folds the transcript."""
    if "Стисло підсумуй" in prompt or "підсумок" in prompt.split("\n")[0]:
        return "[memory: final_code=" + (_MEMORY.findall(prompt) or [""])[0] + "] крок виконано"
    return oracle_controller(prompt)


def test_the_committed_families_run_and_persist_under_perfect_play(tmp_path: Path):
    design = load_collapse_design(DESIGN_PATH)
    rows = run_collapse_families(
        design, model="fake", backend="fake", complete=_fake_model, data_dir=tmp_path
    )
    analysis = analyze_collapse(design, CONTROL_PASS, rows)
    for cell in analysis["cells"]:
        assert cell["valid"] is True, cell["invalid_reason"]
        assert cell["compaction_activation_rate"] == 1.0, cell["cell_id"]
    # The equal-trigger families must agree exactly under a deterministic controller: the guard
    # only re-enters through hysteresis, which a single fold per episode never reaches.
    for row in analysis["families"]:
        if row["kind"] == "equal_trigger":
            assert row["spread"] == pytest.approx(0.0), row["family_id"]
            assert len(row["fold_steps"]) == 1, row["family_id"]
        else:
            assert len(row["fold_steps"]) == len(row["triggers"]), row["family_id"]
    # The delta is a step function of the trigger: cells that fold at the same step agree, whether
    # or not their triggers are equal.
    by_fold: dict[tuple[int, int], set[float]] = {}
    for cell in analysis["cells"]:
        key = (cell["depth"], cell["predicted_fold_step"])
        delta = cell["cost_evidence"]["compact_minus_cap_total_model_input_tokens"]["mean"]
        by_fold.setdefault(key, set()).add(round(delta, 6))
    assert all(len(deltas) == 1 for deltas in by_fold.values()), by_fold
    table = format_collapse_table(analysis)
    paths = persist_collapse(
        design, analysis, data_dir=tmp_path, table=table, tokens_per_s=0.0, mirror=lambda *_: None
    )
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["config"]["study_kind"] == STUDY_KIND
    assert "families" in table and "portable trigger rule" in table
