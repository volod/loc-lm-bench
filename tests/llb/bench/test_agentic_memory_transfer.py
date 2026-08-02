"""Control gate, prospective design, and transfer reading contracts."""

import json
from pathlib import Path
import re

from llb.bench.agentic_memory_transfer import (
    READING_CONDITIONAL,
    READING_INELIGIBLE,
    analyze_transfer,
    load_transfer_design,
    run_control_pilot,
    validate_transfer_design,
)


ROOT = Path(__file__).resolve().parents[3]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_memory_transfer_design.json"


def control_complete(prompt: str) -> str:
    if "[workflow complete]" not in prompt:
        tokens = re.findall(r'(?:токеном "|next token: )(ctrl-\d{3}-\d+)', prompt)
        assert tokens
        return json.dumps({"name": "advance", "arguments": {"token": tokens[-1]}})
    code = re.search(r"CTRL-\d{3}-\d{3}", prompt)
    assert code is not None
    return json.dumps({"name": "finish", "arguments": {"answer": code.group(0)}})


def test_committed_design_is_non_qwen_and_brackets_reference_geometry():
    design = load_transfer_design(DESIGN_PATH)
    validate_transfer_design(design)
    assert all("qwen" not in str(row["model_family"]).lower() for row in design["candidate_roster"])
    assert design["matrix"]["depths"] == [6, 10]
    assert design["matrix"]["compact_shares"] == [0.4, 0.6]


def test_design_refuses_qwen_and_underpowered_cells():
    design = load_transfer_design(DESIGN_PATH)
    design["candidate_roster"][0]["model_family"] = "qwen"
    try:
        validate_transfer_design(design)
    except ValueError as exc:
        assert "non-Qwen" in str(exc)
    else:
        raise AssertionError("Qwen transfer candidate was accepted")
    design = load_transfer_design(DESIGN_PATH)
    design["matrix"]["n_tasks"] = 5
    try:
        validate_transfer_design(design)
    except ValueError as exc:
        assert "six paired tasks" in str(exc)
    else:
        raise AssertionError("underpowered transfer cell was accepted")


def test_control_pilot_passes_only_after_walking_the_memory_free_chain():
    design = load_transfer_design(DESIGN_PATH)
    report, row = run_control_pilot(
        design["control_pilot"],
        model="fake",
        backend="fake",
        complete=control_complete,
    )
    assert report.result.objective_score == 1.0
    assert row["eligible"] is True
    assert row["mean_steps"] == 11.0


def test_transfer_reading_requires_eligibility_and_exposes_geometry_dependence():
    design = load_transfer_design(DESIGN_PATH)
    ineligible = analyze_transfer(
        design,
        [{"eligible": False, "model": "failed"}],
        [],
    )
    assert ineligible["transfer_reading"] == READING_INELIGIBLE

    cells = [
        {"verdict": verdict}
        for verdict in ("prefer_compact", "prefer_cap", "still_tied", "prefer_compact")
    ]
    conditional = analyze_transfer(
        design,
        [{"eligible": True, "model": "selected"}],
        cells,
    )
    assert conditional["transfer_reading"] == READING_CONDITIONAL
    assert conditional["changes_shipped_default"] is False
