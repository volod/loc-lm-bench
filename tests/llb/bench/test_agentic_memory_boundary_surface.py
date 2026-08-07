"""Cap-fitting boundary surface: probe, grid contract, cell gate, and crossover reading."""

from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from llb.bench.agentic_memory_boundary_crossover import (
    READING_BRACKETED,
    READING_CAP_ACROSS_GRID,
    READING_COMPACT_ACROSS_GRID,
    interpolate_crossover,
)
from llb.bench.agentic_memory_boundary_probe import cap_peak_prompt_chars, oracle_controller
from llb.bench.agentic_memory_fold_step_ladder import guard_is_cap_fitting, usable_guard_band
from llb.bench.agentic_memory_boundary_surface import (
    READING_INELIGIBLE,
    READING_INVALID,
    READING_MAPPED,
    READING_NOT_BRACKETED,
    STUDY_KIND,
    analyze_surface,
    load_surface_design,
    run_surface_grid,
    surface_cap_peaks,
    validate_surface_design,
)
from llb.bench.agentic_memory_boundary_surface_report import (
    format_surface_table,
    persist_surface,
)

ROOT = Path(__file__).resolve().parents[3]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_memory_boundary_surface_design.json"

CONTROL_PASS = {
    "eligible": True,
    "model": "pinned",
    "completion": 1.0,
    "minimum_completion_rate": 0.75,
}
_MEMORY = re.compile(r"final_code=([A-Z0-9-]+)")


def _cell_row(cell: dict[str, object], cost_delta: float, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "cell_id": cell["cell_id"],
        "depth": cell["depth"],
        "max_prompt_chars": cell["max_prompt_chars"],
        "require_cap_fits": True,
        "verdict": "prefer_compact" if cost_delta < 0 else "prefer_cap",
        "cap_completion": 1.0,
        "compact_completion": 1.0,
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


def _rows(design: dict[str, object], deltas: list[float], **overrides: object) -> list[dict]:
    cells = design["surface"]["cells"]
    rows = [_cell_row(cell, delta) for cell, delta in zip(cells, deltas, strict=True)]
    if overrides:
        rows[-1].update(overrides)
    return rows


def test_probe_reproduces_the_measured_cap_geometry_without_a_model():
    peak = cap_peak_prompt_chars(depth=6, n_tasks=7)
    # The replication measured exactly this cap cost at depth 6: 8000 chars overflowed cap and
    # 12000 did not, and cap spent 13258 total model-input tokens per task.
    assert 8000 < peak < 12000
    low, high = usable_guard_band(peak, 0.5)
    assert (low, high) == (peak, 2 * peak)
    assert guard_is_cap_fitting(12000, peak, 0.5) is True
    assert guard_is_cap_fitting(8000, peak, 0.5) is False
    assert guard_is_cap_fitting(2 * peak, peak, 0.5) is False


def test_committed_design_is_cap_fitting_brackets_both_sides_and_anchors_the_reference():
    design = load_surface_design(DESIGN_PATH)
    validate_surface_design(design)
    peaks = surface_cap_peaks(design)
    share = design["held_fixed"]["compact_share"]
    assert set(peaks) == {6, 10}
    for cell in design["surface"]["cells"]:
        assert guard_is_cap_fitting(cell["max_prompt_chars"], peaks[cell["depth"]], share)
    anchor = design["reference"]
    assert any(
        cell["depth"] == anchor["depth"] and cell["max_prompt_chars"] == anchor["max_prompt_chars"]
        for cell in design["surface"]["cells"]
    )


def test_design_refuses_an_overflowing_guard_a_one_sided_depth_and_a_narrow_window():
    design = load_surface_design(DESIGN_PATH)
    overflowing = deepcopy(design)
    overflowing["surface"]["cells"][1]["max_prompt_chars"] = 8000
    with pytest.raises(ValueError, match="usable band"):
        validate_surface_design(overflowing)

    one_sided = deepcopy(design)
    for cell in one_sided["surface"]["cells"]:
        cell["expected_side"] = "compact_cheaper"
    with pytest.raises(ValueError, match="both sides"):
        validate_surface_design(one_sided)

    narrow = deepcopy(design)
    narrow["held_fixed"]["max_model_len"] = 4096
    with pytest.raises(ValueError, match="max_model_len"):
        validate_surface_design(narrow)

    unpinned = deepcopy(design)
    unpinned["surface"]["interpolation"]["rule"] = "eyeball"
    with pytest.raises(ValueError, match="zero-crossing"):
        validate_surface_design(unpinned)


def test_linear_zero_crossing_lands_between_the_bracketing_guards():
    assert interpolate_crossover(14000, -1000.0, 15500, 500.0) == pytest.approx(15000.0)
    with pytest.raises(ValueError):
        interpolate_crossover(15500, -1.0, 14000, 1.0)


def test_surface_reading_maps_one_crossover_per_depth_and_states_a_routing_rule():
    design = load_surface_design(DESIGN_PATH)
    analysis = analyze_surface(
        design, CONTROL_PASS, _rows(design, [-800.0, -400.0, 200.0, -9000.0, -2000.0, 1000.0])
    )
    assert analysis["surface_reading"] == READING_MAPPED
    assert analysis["expectation_matches"] == len(design["surface"]["cells"])
    depths = {row["depth"]: row for row in analysis["depth_surface"]}
    assert depths[6]["reading"] == READING_BRACKETED
    assert depths[6]["bracket"] == [14000, 15500]
    assert 14000 < depths[6]["crossover_max_prompt_chars"] < 15500
    assert depths[10]["bracket"] == [20000, 23000]
    assert depths[6]["crossover_guard_ratio"] == pytest.approx(
        depths[6]["crossover_max_prompt_chars"] / depths[6]["cap_peak_prompt_chars"]
    )
    assert any("use compact below" in line for line in analysis["routing_rule"])
    assert any("portable form" in line for line in analysis["routing_rule"])
    assert analysis["changes_shipped_default"] is False


def test_one_sided_grids_report_a_bound_instead_of_extrapolating():
    design = load_surface_design(DESIGN_PATH)
    analysis = analyze_surface(
        design, CONTROL_PASS, _rows(design, [-800.0, -400.0, -200.0, -9000.0, -2000.0, -100.0])
    )
    assert analysis["surface_reading"] == READING_NOT_BRACKETED
    readings = {row["depth"]: row["reading"] for row in analysis["depth_surface"]}
    assert readings == {6: READING_COMPACT_ACROSS_GRID, 10: READING_COMPACT_ACROSS_GRID}
    assert all(row["crossover_max_prompt_chars"] is None for row in analysis["depth_surface"])
    assert any("crossover is above the grid" in line for line in analysis["routing_rule"])

    cap_side = analyze_surface(
        design, CONTROL_PASS, _rows(design, [800.0, 400.0, 200.0, 9000.0, 2000.0, 100.0])
    )
    assert {row["reading"] for row in cap_side["depth_surface"]} == {READING_CAP_ACROSS_GRID}


def test_a_cell_whose_cost_does_not_separate_is_never_an_interpolation_endpoint():
    design = load_surface_design(DESIGN_PATH)
    rows = _rows(design, [-800.0, -400.0, 200.0, -9000.0, -2000.0, 1000.0])
    # The middle depth-10 cell keeps its preconditions but its cost sign is not readable.
    rows[4]["paired"]["total_model_input_tokens"].update(
        {"wins": 3, "losses": 4, "sign_test_p": 1.0}
    )
    analysis = analyze_surface(design, CONTROL_PASS, rows)
    depth10 = next(row for row in analysis["depth_surface"] if row["depth"] == 10)
    assert analysis["cells"][4]["measured_side"] == "cost_tied"
    assert analysis["cells"][4]["valid"] is True
    assert depth10["reading"] == READING_BRACKETED
    assert depth10["bracket"] == [14000, 23000]
    assert analysis["surface_reading"] == READING_MAPPED


def test_a_cell_that_loses_its_preconditions_invalidates_the_surface():
    design = load_surface_design(DESIGN_PATH)
    deltas = [-800.0, -400.0, 200.0, -9000.0, -2000.0, 1000.0]
    overflowed = analyze_surface(
        design, CONTROL_PASS, _rows(design, deltas, cap_context_overflows=3)
    )
    assert overflowed["surface_reading"] == READING_INVALID
    assert "not cap-fitting" in overflowed["reason"]

    inactive = analyze_surface(
        design, CONTROL_PASS, _rows(design, deltas, compaction_activation_rate=0.5)
    )
    assert inactive["surface_reading"] == READING_INVALID
    assert "activation floor" in inactive["reason"]

    rescued = _rows(design, deltas)
    rescued[-1]["paired"]["completion"]["delta"] = {"mean": 1.0, "lo": 1.0, "hi": 1.0}
    rescue = analyze_surface(design, CONTROL_PASS, rescued)
    assert rescue["surface_reading"] == READING_INVALID
    assert "not paired" in rescue["reason"]


def _fake_model(prompt: str) -> str:
    """Perfect play, plus a fixed summary whenever the compact policy folds the transcript."""
    if "Стисло підсумуй" in prompt or "підсумок" in prompt.split("\n")[0]:
        return "[memory: final_code=" + (_MEMORY.findall(prompt) or [""])[0] + "] крок виконано"
    return oracle_controller(prompt)


def test_the_committed_grid_keeps_both_policies_usable_under_perfect_play(tmp_path: Path):
    design = load_surface_design(DESIGN_PATH)
    validate_surface_design(design)
    rows = run_surface_grid(
        design, model="fake", backend="fake", complete=_fake_model, data_dir=tmp_path
    )
    analysis = analyze_surface(design, CONTROL_PASS, rows)
    for cell in analysis["cells"]:
        assert cell["cap_context_overflows"] == 0, cell["cell_id"]
        assert cell["compact_context_overflows"] == 0, cell["cell_id"]
        assert cell["compaction_activation_rate"] == 1.0, cell["cell_id"]
        assert cell["valid"] is True, cell["invalid_reason"]
    table = format_surface_table(analysis)
    paths = persist_surface(
        design, analysis, data_dir=tmp_path, table=table, tokens_per_s=0.0, mirror=lambda *_: None
    )
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["config"]["study_kind"] == STUDY_KIND
    assert manifest["config"]["analysis"]["surface_reading"] == analysis["surface_reading"]
    assert "routing rule" in table


def test_an_ineligible_pinned_family_reports_nothing_measured():
    design = load_surface_design(DESIGN_PATH)
    analysis = analyze_surface(design, {"eligible": False, "model": "pinned"}, [])
    assert analysis["surface_reading"] == READING_INELIGIBLE
    assert analysis["cells"] == []
    assert analysis["depth_surface"] == []
