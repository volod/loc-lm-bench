"""adoption-bar-per-model-screen -- what does deciding the reranker question for ONE model cost?

Pure and file-driven: per-item deltas come back from run bundles on disk and the resampling study
is pure Python, so the whole vertical is unit-tested with dict rows -- no backend, store, or GPU.
"""

import json
from pathlib import Path

import pytest

from llb.eval.embedder_adoption import (
    DECISION_FULL_SET_REQUIRED,
    DECISION_SCREEN_SUPPORTED,
    READING_ANSWER,
    READING_NEITHER,
    READING_RANK_ONLY,
    cell_item_deltas,
    decide_screen,
    format_screen,
    format_screen_summary,
    reading_from_deltas,
    run_screen_study,
    screen_model,
)
from llb.eval.embedder_adoption.run import run_screen_study_over_paths
from llb.eval.embedder_adoption.models import ItemDeltas
from llb.rag.fusion_evidence.stats import bootstrap_index_sets

FOCUS = "k10+rerank"
BASELINE = "intfloat/multilingual-e5-base"
CANDIDATE = "BAAI/bge-m3"


def _deltas(objective: list[float], rr: list[float] | None = None) -> ItemDeltas:
    return ItemDeltas(
        item_ids=[f"q{i}" for i in range(len(objective))],
        objective=objective,
        reciprocal_rank=rr if rr is not None else [0.0] * len(objective),
    )


def _index_sets(n: int, resamples: int = 300):
    return bootstrap_index_sets(n, resamples, 13)


# --- the reading rule over delta vectors --------------------------------------------------


def test_reading_from_deltas_matches_the_cell_reading_order():
    """Objective first, then first-hit rank -- the same order `cell_reading` uses."""
    n = 20
    both = _deltas([0.5] * n, [0.5] * n)
    assert reading_from_deltas(both, _index_sets(n)) == READING_ANSWER
    rank_only = _deltas([0.0] * n, [0.5] * n)
    assert reading_from_deltas(rank_only, _index_sets(n)) == READING_RANK_ONLY
    flat = _deltas([0.0] * n, [0.0] * n)
    assert reading_from_deltas(flat, _index_sets(n)) == READING_NEITHER


def test_misaligned_delta_vectors_are_refused():
    with pytest.raises(ValueError, match="must align"):
        ItemDeltas(item_ids=["q0", "q1"], objective=[0.1], reciprocal_rank=[0.1, 0.2])


def test_take_restricts_every_vector_together():
    deltas = _deltas([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    subset = deltas.take([2, 0])
    assert subset.item_ids == ["q2", "q0"]
    assert subset.objective == pytest.approx([0.3, 0.1])
    assert subset.reciprocal_rank == pytest.approx([0.6, 0.4])


def test_a_delta_whose_interval_spans_zero_is_not_an_answer():
    n = 20
    marginal = _deltas([1.0] + [0.0] * (n - 1))
    assert reading_from_deltas(marginal, _index_sets(n)) != READING_ANSWER


# --- the agreement curve ------------------------------------------------------------------


def test_a_strong_effect_is_reproduced_by_small_screens():
    """A wide margin survives subsampling, so a screen is cheap for an obvious effect."""
    n = 40
    screen = screen_model(
        "strong",
        _deltas([0.6] * n, [0.6] * n),
        READING_ANSWER,
        sizes=(10, 20),
        draws=20,
        resamples=200,
    )
    assert screen["full_reading"] == READING_ANSWER
    assert screen["reproduced"] is True
    assert all(entry["agreement"] == 1.0 for entry in screen["sizes"])
    assert screen["min_size"] == 10


def test_a_marginal_effect_is_lost_by_small_screens():
    """The case the study exists to catch: the full set detects it, a screen does not."""
    # 12 wins of +0.3 against 28 zeros: the full-set interval clears zero, subsets often do not.
    deltas = _deltas([0.3] * 12 + [0.0] * 28)
    screen = screen_model(
        "marginal", deltas, READING_ANSWER, sizes=(10, 15), draws=40, resamples=200, target=0.9
    )
    assert screen["full_reading"] == READING_ANSWER
    assert min(entry["agreement"] for entry in screen["sizes"]) < 0.9
    assert screen["min_size"] is None


def test_sizes_at_or_above_the_full_set_are_not_measured():
    n = 12
    screen = screen_model(
        "m", _deltas([0.5] * n), READING_ANSWER, sizes=(10, 12, 20), draws=10, resamples=100
    )
    assert [entry["size"] for entry in screen["sizes"]] == [10]


def test_the_agreement_curve_is_deterministic_at_a_fixed_seed():
    kwargs = dict(sizes=(10,), draws=25, resamples=150)
    deltas = _deltas([0.3] * 10 + [0.0] * 20)
    a = screen_model("m", deltas, READING_ANSWER, seed=7, **kwargs)
    b = screen_model("m", deltas, READING_ANSWER, seed=7, **kwargs)
    assert a["sizes"] == b["sizes"]


# --- the cost verdict ---------------------------------------------------------------------


def _screen(model: str, min_size: int | None, n: int = 40):
    return {
        "model": model,
        "n": n,
        "full_reading": READING_ANSWER,
        "recorded_reading": READING_ANSWER,
        "reproduced": True,
        "sizes": [],
        "min_size": min_size,
    }


def test_a_screen_is_claimed_only_when_every_model_survives_it():
    verdict = decide_screen(
        [_screen("a", 15), _screen("b", 25)],
        focus_cell=FOCUS,
        target=0.9,
        bundles_full_grid=24,
        bundles_focus_cell=6,
    )
    assert verdict["decision"] == DECISION_SCREEN_SUPPORTED
    assert verdict["min_size"] == 25  # the WORST model sets the screen size
    assert "25 of 40 items" in verdict["reason"]


def test_one_model_losing_its_reading_blocks_the_item_saving():
    """A screen that reproduces three models and loses the fourth is the dangerous one."""
    verdict = decide_screen(
        [_screen("a", 15), _screen("b", None)],
        focus_cell=FOCUS,
        target=0.9,
        bundles_full_grid=24,
        bundles_focus_cell=6,
    )
    assert verdict["decision"] == DECISION_FULL_SET_REQUIRED
    assert verdict["min_size"] is None
    assert "`b`" in verdict["reason"]


def test_the_cell_saving_is_reported_either_way():
    """Dropping the other cells is exact and unconditional, so both verdicts must state it."""
    for models in ([_screen("a", 15)], [_screen("a", None)]):
        verdict = decide_screen(
            models, focus_cell=FOCUS, target=0.9, bundles_full_grid=24, bundles_focus_cell=6
        )
        assert "24 to 6 run-eval bundles (4x)" in verdict["reason"]


# --- re-deriving per-item deltas from run bundles -----------------------------------------


def _row(item_id: str, objective: float, rank: int | None) -> dict:
    return {
        "item_id": item_id,
        "split": "final",
        "status": "ok",
        "objective_score": objective,
        "token_f1": objective,
        "retrieval_hit": 1.0 if rank else 0.0,
        "first_hit_rank": rank,
    }


def _bundle(path: Path, rows: list[dict]) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return str(path)


def _paired(mean: float, lo: float, hi: float, *, wins: int, losses: int = 0) -> dict:
    """A finished paired block, ledger included -- the gate reads the ledger, not only the bounds."""
    return {
        "delta": {"mean": mean, "lo": lo, "hi": hi},
        "wins": wins,
        "losses": losses,
        "ties": 0,
        "sign_test_p": 0.0,
    }


def _sweep_report(tmp_path: Path, *, base: list[dict], cand: list[dict], model="m") -> dict:
    """A finished-sweep shape whose focus cell names real bundles on disk."""
    return {
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "item_ids": [r["item_id"] for r in base],
        "metrics": [],
        "resamples": 200,
        "confidence": 0.95,
        "seed": 13,
        "cells": [
            {
                "label": "k10",
                "top_k": 10,
                "reranker": None,
                "n": len(base),
                "lanes": {
                    BASELINE: {"run_dirs": [_bundle(tmp_path / "k10-b", base)], "metrics": {}},
                    CANDIDATE: {"run_dirs": [_bundle(tmp_path / "k10-c", cand)], "metrics": {}},
                },
                "paired": {},
            },
            {
                "label": FOCUS,
                "top_k": 10,
                "reranker": "x",
                "n": len(base),
                "lanes": {
                    BASELINE: {"run_dirs": [_bundle(tmp_path / "f-b", base)], "metrics": {}},
                    CANDIDATE: {"run_dirs": [_bundle(tmp_path / "f-c", cand)], "metrics": {}},
                },
                "paired": {
                    metric: _paired(0.5, 0.4, 0.6, wins=len(base))
                    for metric in ("objective_score", "reciprocal_rank")
                },
            },
        ],
        "verdict": {"decision": "extend_bar"},
        "metadata": {"model": model, "goldset": "gs.jsonl", "corpus": "c", "split": "final"},
    }


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


# --- the whole study ----------------------------------------------------------------------


def _wide_sweep(tmp_path: Path, model: str, n: int = 24) -> dict:
    base = [_row(f"q{i}", 0.0, 4) for i in range(n)]
    cand = [_row(f"q{i}", 1.0, 1) for i in range(n)]
    return _sweep_report(tmp_path / model, base=base, cand=cand, model=model)


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
