"""Window-elision geometry, exact paired reading, run, and persistence contracts."""

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from llb.bench.agentic_design_fields import as_mapping, as_rows
from llb.bench.agentic_memory_window_elision import run_window_elision
from llb.bench.agentic_memory_window_elision_design import (
    ROLE_ELIDED,
    ROLE_FIT,
    elision_cells,
    load_window_elision_design,
    probe_elision_cell,
    validate_window_elision_design,
)
from llb.bench.agentic_memory_window_elision_reading import (
    READING_COSTS,
    READING_FREE,
    completion_reading,
)
from llb.bench.agentic_memory_window_elision_report import (
    format_window_elision_table,
    persist_window_elision_run,
)


class SummaryDropsMiddle:
    """Follow the token chain while the model-written summary retains no final code."""

    def __call__(self, prompt: str) -> str:
        if "Стисло підсумуй" in prompt:
            return "checkpoint retained"
        if "[workflow complete]" not in prompt:
            tokens = re.findall(r'(?:токеном "|next token: )(wf-\d{3}-\d+)', prompt)
            assert tokens
            return json.dumps({"name": "advance", "arguments": {"token": tokens[-1]}})
        code = re.search(r"MEM-\d{3}-\d{3}", prompt)
        return json.dumps(
            {"name": "finish", "arguments": {"answer": code.group(0) if code else "LOST"}}
        )


def test_design_predeclares_trigger_matched_fit_and_elision_cells():
    design = load_window_elision_design()
    validate_window_elision_design(design)
    held = design["held_fixed"]
    probes = {cell["role"]: probe_elision_cell(cell, held) for cell in elision_cells(design)}
    assert probes[ROLE_FIT]["summary_input_elided_chars"] == 0
    assert probes[ROLE_ELIDED]["summary_input_elided_chars"] == 2134
    assert {probe["compaction_trigger_chars"] for probe in probes.values()} == {11200}
    assert {tuple(probe["summary_fold_input_chars"]) for probe in probes.values()} == {(15402,)}


def test_design_rejects_a_drifted_trigger_and_a_control_that_elides():
    design = load_window_elision_design()
    moved = deepcopy(design)
    as_rows(moved, "cells")[0]["compact_share"] = 0.7
    with pytest.raises(ValueError, match="compaction trigger fixed"):
        validate_window_elision_design(moved)

    mislabeled = deepcopy(design)
    control = next(cell for cell in as_rows(mislabeled, "cells") if cell["role"] == ROLE_FIT)
    as_mapping(control, "expected")["summary_input_elided_chars"] = 1
    with pytest.raises(ValueError, match="summary_input_elided_chars drifted"):
        validate_window_elision_design(mislabeled)

    untyped = deepcopy(design)
    as_mapping(untyped, "held_fixed")["preserve_memory_markers"] = False
    with pytest.raises(ValueError, match="shipped typed-memory behavior"):
        validate_window_elision_design(untyped)


def test_fake_run_is_paired_and_persists_the_probe_backed_reading(tmp_path: Path):
    design = load_window_elision_design()
    run = run_window_elision(design, model="fake", backend="fake", complete=SummaryDropsMiddle())
    assert run.analysis["comparison_eligible"] is True
    assert run.analysis["completion_reading"] == READING_FREE
    assert run.analysis["paired_completion"] == {
        "n_pairs": 4,
        "fit_wins": 0,
        "elided_wins": 0,
        "unchanged": 4,
        "completion_delta": 0.0,
    }
    table = format_window_elision_table(run.analysis)
    paths = persist_window_elision_run(
        design,
        run,
        data_dir=tmp_path,
        table=table,
        tokens_per_s=12.5,
        mirror=lambda *_: None,
    )
    assert "window_elision_costs_no_completion" in table
    assert Path(paths["manifest"]).exists()
    assert len(list((tmp_path / "agentic-compact-window-elision").glob("*/manifest.json"))) == 3


def test_exact_paired_reading_reports_a_one_directional_completion_cost():
    fit = [{"item_id": "a", "success": True}, {"item_id": "b", "success": True}]
    elided = [{"item_id": "a", "success": False}, {"item_id": "b", "success": True}]
    reading, reason, paired = completion_reading(
        fit, elided, eligible=True, eligibility_reason="eligible"
    )
    assert reading == READING_COSTS
    assert "wins 1" in reason
    assert paired["completion_delta"] == 0.5
