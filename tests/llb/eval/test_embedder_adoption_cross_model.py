"""embedder-adoption-bar-second-model -- do two generation models agree on the reading?

Pure and file-driven: the input is two finished `AdoptionBarReport`s, so the cross-model reading
is unit-tested with dict reports -- no backend, no store, no GPU.
"""

import json
from pathlib import Path

import pytest

from llb.eval.embedder_adoption import (
    READING_ANSWER,
    READING_NEITHER,
    READING_RANK_ONLY,
    cell_reading,
    compare_cells,
    compare_models,
    format_cross_model,
    format_cross_summary,
    load_report,
    run_cross_model_comparison,
)
from llb.eval.embedder_adoption.models import CellSpec

BASELINE = "intfloat/multilingual-e5-base"
CANDIDATE = "BAAI/bge-m3"
MODEL_A = "mamaylm-12b"
MODEL_B = "gemma4-31b"


def _row(item_id: str, objective: float, rank: int | None = 1, hit: float = 1.0) -> dict:
    return {
        "item_id": item_id,
        "split": "final",
        "status": "ok",
        "objective_score": objective,
        "token_f1": objective,
        "exact": 0.0,
        "contains": 1.0 if objective > 0 else 0.0,
        "retrieval_hit": hit,
        "first_hit_rank": rank,
    }


def _ids(n: int) -> list[str]:
    return [f"q{i}" for i in range(n)]


def _report(cells: dict[str, tuple[list[dict], list[dict]]], *, model: str, resamples: int = 200):
    """A finished single-model sweep report with a generation-model metadata stamp."""
    specs = {"k10": CellSpec(10, None), "k3": CellSpec(3, None), "k3+rerank": CellSpec(3, "x")}
    report = compare_cells(
        [
            (specs[label], {BASELINE: base, CANDIDATE: cand})
            for label, (base, cand) in cells.items()
        ],
        {},
        baseline=BASELINE,
        candidate=CANDIDATE,
        resamples=resamples,
    )
    report["metadata"] = {
        "model": model,
        "goldset": "gs.jsonl",
        "corpus": "corpus",
        "split": "final",
    }
    return report


# The three per-cell reading building blocks, as scored-row pairs (baseline, candidate).
def _answer_cell(ids: list[str]):  # candidate answers better AND ranks earlier
    return ([_row(i, 0.0, rank=None, hit=0.0) for i in ids], [_row(i, 1.0, rank=1) for i in ids])


def _rank_only_cell(ids: list[str]):  # candidate ranks earlier, answers identically
    return ([_row(i, 1.0, rank=3) for i in ids], [_row(i, 1.0, rank=1) for i in ids])


def _neither_cell(ids: list[str]):  # identical on both
    return ([_row(i, 1.0, rank=1) for i in ids], [_row(i, 1.0, rank=1) for i in ids])


# --- the per-cell reading -----------------------------------------------------------------


def test_cell_reading_names_the_three_outcomes():
    ids = _ids(12)
    answer = compare_cells(
        [(CellSpec(3, None), {BASELINE: _answer_cell(ids)[0], CANDIDATE: _answer_cell(ids)[1]})],
        {},
        baseline=BASELINE,
        candidate=CANDIDATE,
        resamples=200,
    )
    assert cell_reading(answer["cells"][0]) == READING_ANSWER

    rank = compare_cells(
        [
            (
                CellSpec(10, None),
                {BASELINE: _rank_only_cell(ids)[0], CANDIDATE: _rank_only_cell(ids)[1]},
            )
        ],
        {},
        baseline=BASELINE,
        candidate=CANDIDATE,
        resamples=200,
    )
    assert cell_reading(rank["cells"][0]) == READING_RANK_ONLY

    neither = compare_cells(
        [(CellSpec(10, None), {BASELINE: _neither_cell(ids)[0], CANDIDATE: _neither_cell(ids)[1]})],
        {},
        baseline=BASELINE,
        candidate=CANDIDATE,
        resamples=200,
    )
    assert cell_reading(neither["cells"][0]) == READING_NEITHER


# --- agreement ----------------------------------------------------------------------------


def test_two_models_that_read_every_cell_the_same_agree():
    ids = _ids(12)
    cells = {"k10": _rank_only_cell(ids), "k3": _answer_cell(ids)}
    cross = compare_models([_report(cells, model=MODEL_A), _report(cells, model=MODEL_B)])
    assert cross["verdicts"] == {MODEL_A: "extend_bar", MODEL_B: "extend_bar"}
    assert cross["verdicts_agree"] is True
    assert cross["agree_count"] == 2
    assert [c["agree"] for c in cross["cells"]] == [True, True]
    assert cross["cells"][0]["readings"] == {MODEL_A: READING_RANK_ONLY, MODEL_B: READING_RANK_ONLY}
    assert cross["cells"][1]["readings"] == {MODEL_A: READING_ANSWER, MODEL_B: READING_ANSWER}


def test_a_model_specific_answer_gain_surfaces_as_a_disagreement():
    """The whole point: one model turns the rank gain into an answer, the other does not."""
    ids = _ids(12)
    a = {"k10": _rank_only_cell(ids), "k3": _answer_cell(ids)}  # extend_bar
    b = {"k10": _rank_only_cell(ids), "k3": _rank_only_cell(ids)}  # keep_bar
    cross = compare_models([_report(a, model=MODEL_A), _report(b, model=MODEL_B)])
    assert cross["verdicts"] == {MODEL_A: "extend_bar", MODEL_B: "keep_bar"}
    assert cross["verdicts_agree"] is False
    assert cross["agree_count"] == 1
    assert cross["cells"][0]["agree"] is True  # both rank_only on k10
    assert cross["cells"][1]["agree"] is False  # answer vs rank_only on k3


# --- comparability guards -----------------------------------------------------------------


def test_different_encoder_pairs_are_not_comparable():
    ids = _ids(8)
    a = _report({"k10": _neither_cell(ids)}, model=MODEL_A)
    b = _report({"k10": _neither_cell(ids)}, model=MODEL_B)
    b["candidate"] = "some/other-encoder"
    with pytest.raises(ValueError, match="different encoder pairs"):
        compare_models([a, b])


def test_different_cell_grids_are_not_comparable():
    ids = _ids(8)
    a = _report({"k10": _neither_cell(ids), "k3": _neither_cell(ids)}, model=MODEL_A)
    b = _report({"k10": _neither_cell(ids)}, model=MODEL_B)
    with pytest.raises(ValueError, match="different cell grids"):
        compare_models([a, b])


def test_different_item_sets_are_not_comparable():
    a = _report({"k10": _neither_cell(_ids(8))}, model=MODEL_A)
    b = _report({"k10": _neither_cell(_ids(7))}, model=MODEL_B)
    with pytest.raises(ValueError, match="different item sets"):
        compare_models([a, b])


def test_different_seeds_are_not_comparable():
    ids = _ids(8)
    a = _report({"k10": _neither_cell(ids)}, model=MODEL_A)
    b = _report({"k10": _neither_cell(ids)}, model=MODEL_B)
    b["seed"] = a["seed"] + 1
    with pytest.raises(ValueError, match="different bootstrap seeds"):
        compare_models([a, b])


def test_the_same_model_scored_twice_is_not_a_cross_model_reading():
    ids = _ids(8)
    a = _report({"k10": _neither_cell(ids)}, model=MODEL_A)
    b = _report({"k10": _neither_cell(ids)}, model=MODEL_A)
    with pytest.raises(ValueError, match="same model"):
        compare_models([a, b])


def test_exactly_two_reports_are_required():
    a = _report({"k10": _neither_cell(_ids(8))}, model=MODEL_A)
    with pytest.raises(ValueError, match="exactly two"):
        compare_models([a])


# --- reporting ----------------------------------------------------------------------------


def test_cross_model_report_renders_ascii_with_agreement_and_verdicts():
    ids = _ids(12)
    a = {"k10": _rank_only_cell(ids), "k3": _answer_cell(ids)}
    b = {"k10": _rank_only_cell(ids), "k3": _rank_only_cell(ids)}
    cross = compare_models([_report(a, model=MODEL_A), _report(b, model=MODEL_B)])
    text = format_cross_model(cross, metadata={"goldset": "gs.jsonl"})
    assert "# Embedder adoption bar" in text
    assert "### Per-cell reading agreement" in text
    assert "DISAGREE" in _headline_or_summary(text, cross)
    assert MODEL_A in text and MODEL_B in text
    assert text.isascii()
    assert format_cross_summary(cross).isascii()


def _headline_or_summary(text: str, cross) -> str:
    return text + format_cross_summary(cross)


# --- disk round-trip ----------------------------------------------------------------------


def test_load_and_persist_round_trip(tmp_path: Path):
    ids = _ids(12)
    a = _report({"k10": _rank_only_cell(ids), "k3": _answer_cell(ids)}, model=MODEL_A)
    b = _report({"k10": _rank_only_cell(ids), "k3": _rank_only_cell(ids)}, model=MODEL_B)
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    for d, rep in ((dir_a, a), (dir_b, b)):
        d.mkdir()
        (d / "comparison.json").write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")

    # load_report accepts either the dir or the file
    assert load_report(dir_a)["baseline"] == BASELINE
    assert load_report(dir_a / "comparison.json")["candidate"] == CANDIDATE

    run = run_cross_model_comparison([dir_a, dir_b], out_dir=tmp_path / "cross")
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["verdicts_agree"] is False
    assert persisted["metadata"]["goldset"] == "gs.jsonl"
    assert Path(run.paths["report"]).read_text(encoding="utf-8").startswith("# Embedder adoption")


def test_load_report_rejects_a_non_sweep_json(tmp_path: Path):
    bad = tmp_path / "comparison.json"
    bad.write_text(json.dumps({"not": "a sweep"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not an adoption-bar comparison"):
        load_report(bad)
