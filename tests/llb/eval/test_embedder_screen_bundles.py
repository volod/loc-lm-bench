"""Focused tests split from ``test_embedder_adoption_screen.py``."""

import json
from pathlib import Path

import pytest
from _embedder_adoption_screen_helpers import (
    CANDIDATE,
    FOCUS,
    _paired,
    _row,
    _sweep_report,
    _wide_sweep,
)

from llb.eval.embedder_adoption.screen_models import DECISION_SCREEN_SUPPORTED
from llb.eval.embedder_adoption.screen_data import cell_item_deltas
from llb.eval.embedder_adoption.screen_report import (
    format_screen,
    format_screen_summary,
)
from llb.eval.embedder_adoption.screen import run_screen_study
from llb.eval.embedder_adoption.comparison_run import run_screen_study_over_paths


def test_per_item_deltas_come_back_from_the_run_bundles(tmp_path: Path):
    base = [_row("q0", 0.0, 4), _row("q1", 0.2, 2)]
    cand = [_row("q0", 1.0, 1), _row("q1", 0.2, 1)]
    deltas = cell_item_deltas(_sweep_report(tmp_path, base=base, cand=cand), FOCUS)
    assert deltas.item_ids == ["q0", "q1"]
    assert deltas.objective == pytest.approx([1.0, 0.0])
    # reciprocal rank: 1/1 - 1/4 = 0.75 ; 1/1 - 1/2 = 0.5
    assert deltas.reciprocal_rank == pytest.approx([0.75, 0.5])
    assert len(deltas) == 2


def test_an_unknown_cell_is_refused(tmp_path: Path):
    report = _sweep_report(tmp_path, base=[_row("q0", 0.0, 1)], cand=[_row("q0", 1.0, 1)])
    with pytest.raises(ValueError, match="not in this sweep"):
        cell_item_deltas(report, "k99")


def test_a_lane_that_scored_a_different_item_set_fails_loudly(tmp_path: Path):
    base = [_row("q0", 0.0, 1), _row("q1", 0.0, 1)]
    cand = [_row("q0", 1.0, 1)]
    report = _sweep_report(tmp_path, base=base, cand=cand)
    with pytest.raises(ValueError, match="different item sets"):
        cell_item_deltas(report, FOCUS)


def test_a_missing_bundle_is_named(tmp_path: Path):
    report = _sweep_report(tmp_path, base=[_row("q0", 0.0, 1)], cand=[_row("q0", 1.0, 1)])
    report["cells"][1]["lanes"][CANDIDATE]["run_dirs"] = [str(tmp_path / "gone")]
    with pytest.raises(ValueError, match="missing run bundle scores"):
        cell_item_deltas(report, FOCUS)


def test_the_study_reports_a_curve_per_model_and_a_verdict(tmp_path: Path):
    study = run_screen_study(
        [_wide_sweep(tmp_path, "a"), _wide_sweep(tmp_path, "b")],
        focus_cell=FOCUS,
        sizes=(10, 15),
        draws=15,
        resamples=150,
    )
    assert [m["model"] for m in study["models"]] == ["a", "b"]
    assert all(m["reproduced"] for m in study["models"])
    assert study["verdict"]["decision"] == DECISION_SCREEN_SUPPORTED
    assert study["verdict"]["bundles_full_grid"] == 4  # 2 cells x 2 encoders x 1 split
    assert study["verdict"]["bundles_focus_cell"] == 2


def test_the_study_refuses_vectors_that_disagree_with_the_recorded_reading(tmp_path: Path):
    """The self-check: a pipeline that cannot reproduce the published answer is not trusted."""
    sweep = _wide_sweep(tmp_path, "a")
    # The bundles say a large answer gain; claim the sweep recorded no separation at all.
    sweep["cells"][1]["paired"] = {
        metric: _paired(0.0, -0.1, 0.1, wins=12, losses=12)
        for metric in ("objective_score", "reciprocal_rank")
    }
    with pytest.raises(ValueError, match="do not reproduce the recorded reading"):
        run_screen_study([sweep], focus_cell=FOCUS, sizes=(10,), draws=5, resamples=100)


def test_a_sweep_without_the_focus_cell_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="is not in the sweep"):
        run_screen_study([_wide_sweep(tmp_path, "a")], focus_cell="k99", sizes=(10,), draws=5)


def test_an_empty_study_is_refused():
    with pytest.raises(ValueError, match="at least one recorded sweep"):
        run_screen_study([])


def test_report_renders_ascii_with_the_curve_and_the_verdict(tmp_path: Path):
    study = run_screen_study(
        [_wide_sweep(tmp_path, "a")], focus_cell=FOCUS, sizes=(10, 15), draws=10, resamples=100
    )
    text = format_screen(study, metadata={"goldset": "gs.jsonl"})
    assert "# Embedder adoption bar" in text
    assert "### Screen agreement with the full-set reading" in text
    assert text.isascii()
    assert format_screen_summary(study).isascii()


def test_the_study_round_trips_through_disk(tmp_path: Path):
    sweep = _wide_sweep(tmp_path, "a")
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    (sweep_dir / "comparison.json").write_text(json.dumps(sweep), encoding="utf-8")
    run = run_screen_study_over_paths(
        [sweep_dir],
        out_dir=tmp_path / "screen",
        focus_cell=FOCUS,
        sizes=(10,),
        draws=10,
        resamples=100,
    )
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["verdict"]["decision"] == DECISION_SCREEN_SUPPORTED
    assert persisted["metadata"]["goldset"] == "gs.jsonl"
    assert Path(run.paths["report"]).read_text(encoding="utf-8").startswith("# Embedder adoption")
