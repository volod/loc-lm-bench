"""Tests for recommend render."""

import json
import pytest
from llb.board.recommend.build import (
    build_recommendation,
    load_config_cells,
)
from llb.board.recommend.model import HostInfo
from llb.board.recommend.render import (
    format_config_detail_md,
)
from llb.board.recommend.sections import format_finetune_campaign_section_md
from test_recommend import COHORT, _cell


def test_format_config_detail_groups_by_model_and_marks_best():
    # A swept at three depths, B at one. The per-config table shows every (model, top_k) cell and
    # marks each model's best config; with a real grid present the single-config note is absent.
    cells = [_cell("A", 3, 0.40), _cell("A", 8, 0.55), _cell("A", 5, 0.50), _cell("B", 5, 0.45)]
    md = format_config_detail_md(cells)
    assert "## RAG configuration detail (model x config)" in md
    assert "| model | top_k | best | objective | tok/s | peak VRAM | recall |" in md
    assert "| A | 8 | * |" in md  # A's best config is top_k 8 (0.55)
    assert "| A | 3 |  |" in md and "| A | 5 |  |" in md  # non-best cells listed, unmarked
    assert "Single configuration per model" not in md


def test_format_config_detail_notes_when_no_grid_swept():
    md = format_config_detail_md([_cell("A", 5, 0.5), _cell("B", 5, 0.4)])
    assert "Single configuration per model" in md
    assert 'SWEEP_RAG_GRID="top_k=3,5,8"' in md  # guides the operator to run the grid


def test_format_config_detail_empty_when_no_cells():
    assert format_config_detail_md([]) == ""


def test_format_finetune_campaign_section_ranks_completed_entries():
    md = format_finetune_campaign_section_md(
        {
            "campaign_dir": "/runs/ft",
            "report_path": "/runs/ft/report.md",
            "entries": [
                {
                    "model": "slow",
                    "status": "completed",
                    "base_objective": 0.2,
                    "tuned_objective": 0.3,
                    "delta": 0.1,
                    "train_wall_clock_s": 20,
                    "peak_vram_mb": 9000,
                },
                {
                    "model": "winner",
                    "status": "completed",
                    "base_objective": 0.2,
                    "tuned_objective": 0.5,
                    "delta": 0.3,
                    "train_wall_clock_s": 30,
                    "peak_vram_mb": 10000,
                },
                {"model": "too-big", "status": "skipped", "reason": "does not fit"},
            ],
        }
    )

    assert "## Fine-tune campaign" in md
    assert "| 1 | winner |" in md
    assert "| 2 | slow |" in md
    assert "skipped: does not fit" in md


def test_load_config_cells_keeps_each_top_k_but_dedups_reruns(tmp_path):
    # A RAG grid: the same model at top_k 3 and 8 is TWO cells (not collapsed to best-per-model);
    # a re-run of the SAME cell keeps only its best objective.
    def write_bundle(name, top_k, obj):
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(
            json.dumps(
                {
                    "split": "final",
                    "n_cases": 82,
                    "config": {"model": "modelA", "backend": "ollama", "top_k": top_k},
                    "metrics": {"objective_score": obj, "reliability": 1.0, "tokens_per_s": 10.0},
                    "telemetry": {"peak_vram_mb": 9000},
                    "retrieval": {"recall": 0.9, "mrr": 0.8},
                }
            ),
            encoding="utf-8",
        )

    write_bundle("k3", 3, 0.40)
    write_bundle("k8-a", 8, 0.50)
    write_bundle("k8-b", 8, 0.55)  # re-run of the top_k=8 cell, higher objective

    cells = load_config_cells(tmp_path)
    by_top_k = {c.record.config["top_k"]: c for c in cells}
    assert set(by_top_k) == {3, 8}  # both depths kept as distinct cells
    assert by_top_k[8].result.objective_score == 0.55  # best re-run of the top_k=8 cell


@pytest.mark.slow
def test_render_comparison_chart_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from llb.board.charts import render_comparison_chart

    rec = build_recommendation(COHORT, HostInfo(16, 16380, "RTX 4060 Ti", True))
    out = render_comparison_chart(rec, tmp_path / "comparison.png")
    assert out is not None and out.exists() and out.stat().st_size > 0
