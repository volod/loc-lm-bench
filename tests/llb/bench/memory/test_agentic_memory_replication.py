"""Second-family, tighter-evidence, and cap-fitting replication contracts."""

from copy import deepcopy
from pathlib import Path

import pytest

from llb.bench.memory.replication.run import (
    READING_BOUNDARY_COMPACT,
    READING_BOUNDARY_FAILED,
    analyze_replication,
    load_replication_design,
    validate_replication_design,
)

ROOT = Path(__file__).resolve().parents[4]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_memory_replication_design.json"


def _completion_pair() -> dict[str, object]:
    return {
        "delta": {"mean": 1.0, "lo": 1.0, "hi": 1.0},
        "wins": 7,
        "losses": 0,
        "ties": 0,
        "stability": {"tighter_reading": "separated"},
    }


def _cost_pair() -> dict[str, object]:
    return {
        "delta": {"mean": -800.0, "lo": -820.0, "hi": -780.0},
        "wins": 0,
        "losses": 7,
        "ties": 0,
        "sign_test_p": 0.015625,
    }


def _rows(design: dict[str, object], *, boundary_overflows: int = 0) -> list[dict[str, object]]:
    cells = design["matrix"]["cells"]
    rows: list[dict[str, object]] = []
    for cell in cells:
        boundary = cell["require_cap_fits"]
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "require_cap_fits": boundary,
                "verdict": "prefer_compact",
                "cap_context_overflows": boundary_overflows if boundary else 7,
                "compaction_activation_rate": 1.0,
                "paired": {
                    "completion": _completion_pair(),
                    "total_model_input_tokens": _cost_pair(),
                },
            }
        )
    return rows


def test_committed_replication_design_requires_a_new_family_and_tighter_floor():
    design = load_replication_design(DESIGN_PATH)
    validate_replication_design(design)
    assert design["matrix"]["n_tasks"] == 7
    assert sum(cell["require_cap_fits"] for cell in design["matrix"]["cells"]) == 1
    assert set(design["reference_model_families"]).isdisjoint(
        row["model_family"] for row in design["candidate_roster"]
    )


def test_design_refuses_a_reference_family_and_six_pair_cells():
    design = load_replication_design(DESIGN_PATH)
    bad_family = deepcopy(design)
    bad_family["candidate_roster"][0]["model_family"] = "gemma4"
    with pytest.raises(ValueError, match="new non-Qwen"):
        validate_replication_design(bad_family)
    underpowered = deepcopy(design)
    underpowered["matrix"]["n_tasks"] = 6
    with pytest.raises(ValueError, match="97.5%"):
        validate_replication_design(underpowered)


def test_analysis_requires_tighter_transfer_cells_and_a_real_cap_fitting_boundary():
    design = load_replication_design(DESIGN_PATH)
    pilots = [{"eligible": True, "model_family": "mistral", "model": "candidate"}]
    analysis = analyze_replication(design, pilots, _rows(design))
    assert analysis["tighter_transfer_cells_passed"] is True
    assert analysis["cap_fitting_boundary_passed"] is True
    assert analysis["replication_reading"] == READING_BOUNDARY_COMPACT
    assert analysis["boundary_directional_evidence"]["cost_clears"] is True
    assert analysis["changes_shipped_default"] is False

    invalid = analyze_replication(design, pilots, _rows(design, boundary_overflows=1))
    assert invalid["cap_fitting_boundary_passed"] is False
    assert invalid["replication_reading"] == READING_BOUNDARY_FAILED
