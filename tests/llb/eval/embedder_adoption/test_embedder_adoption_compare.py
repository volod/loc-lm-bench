"""Focused tests split from ``test_embedder_adoption.py``."""

from pathlib import Path

import pytest
from tests.llb.eval._embedder_adoption_helpers import (
    BASELINE,
    CANDIDATE,
    _ids,
    _row,
    _sweep,
)

from llb.core.config import RunConfig
from llb.eval.embedder_adoption.models import (
    DECISION_EXTEND_BAR,
    DECISION_KEEP_BAR,
    DECISION_NO_EVIDENCE,
    CellSpec,
    EmbedderLane,
)
from llb.eval.embedder_adoption.cells import (
    build_cells,
    cell_config,
    parse_rerankers,
)
from llb.eval.retrieval_budgets import parse_top_ks
from llb.eval.embedder_adoption.compare import (
    compare_cells,
    with_reciprocal_rank,
)
from llb.eval.embedder_adoption.report import (
    format_report,
    format_summary,
)
from llb.eval.embedder_adoption.models import (
    METRIC_OBJECTIVE,
    METRIC_RECIPROCAL_RANK,
)
from llb.rag.rerank import DEFAULT_RERANKER


def test_the_grid_is_the_product_of_the_two_knobs_that_make_rank_bind():
    cells = build_cells(parse_top_ks("10,3"), parse_rerankers("off,on"))
    assert [cell.label for cell in cells] == ["k10", "k10+rerank", "k3", "k3+rerank"]
    assert [cell.reranker for cell in cells] == [None, DEFAULT_RERANKER, None, DEFAULT_RERANKER]


def test_the_grid_selections_are_deduplicated_and_validated():
    assert parse_top_ks("10, 3 ,10") == [10, 3]
    assert parse_rerankers("off,off,cross/encoder-x") == [None, "cross/encoder-x"]
    with pytest.raises(ValueError, match="must be an integer"):
        parse_top_ks("ten")
    with pytest.raises(ValueError, match="at least 1"):
        parse_top_ks("0")
    with pytest.raises(ValueError, match="at least one top_k"):
        parse_top_ks(" , ")


def test_a_cell_config_carries_the_encoder_its_store_was_built_with(tmp_path: Path):
    """The store root moves WITH the encoder: `index_dir()` resolves from `data_dir`."""
    base = RunConfig(data_dir=tmp_path, top_k=5, reranker="stale/cross-encoder")
    cfg = cell_config(base, CellSpec(3, None), EmbedderLane(CANDIDATE, tmp_path / "bge"))
    assert cfg.embedding_model == CANDIDATE
    assert cfg.index_dir() == tmp_path / "bge" / "llb" / "rag"
    assert cfg.top_k == 3
    assert cfg.reranker is None  # the off half of the grid must be able to CLEAR the knob
    assert cfg.run_name == f"embedder-adoption-k3-{CANDIDATE}"


def test_reciprocal_rank_is_derived_from_the_bundles_first_hit_rank():
    rows = with_reciprocal_rank([_row("q0", 1.0, rank=1), _row("q1", 1.0, rank=4)])
    assert [row[METRIC_RECIPROCAL_RANK] for row in rows] == [1.0, 0.25]


def test_an_item_whose_context_carried_no_gold_span_has_reciprocal_rank_zero():
    """No hit is rank 0.0, the same convention `llb.rag.retrieval.reciprocal_rank` uses."""
    rows = with_reciprocal_rank([_row("q0", 0.0, rank=None, hit=0.0)])
    assert rows[0][METRIC_RECIPROCAL_RANK] == 0.0


def test_a_rank_gain_that_reaches_the_answer_in_one_cell_extends_the_bar():
    """The only outcome that justifies a second bar -- and it is SCOPED to the cell that showed it."""
    ids = _ids(12)
    report = _sweep(
        {
            # k=10: the candidate ranks earlier, and both encoders answer identically.
            "k10": (
                [_row(i, 1.0, rank=3) for i in ids],
                [_row(i, 1.0, rank=1) for i in ids],
            ),
            # k=3: the baseline's rank-3 evidence falls off the budget and its answers collapse.
            "k3": (
                [_row(i, 0.0, rank=None, hit=0.0) for i in ids],
                [_row(i, 1.0, rank=1) for i in ids],
            ),
        }
    )
    verdict = report["verdict"]
    assert verdict["decision"] == DECISION_EXTEND_BAR
    assert verdict["answer_cells"] == ["k3"]
    assert verdict["rank_cells"] == ["k10", "k3"]
    assert "k3" in verdict["reason"]


def test_per_cell_answer_gains_must_survive_the_four_cell_selection_family():
    ids = _ids(7)
    labels = ("k10", "k10+rerank", "k3", "k3+rerank")
    cells = {}
    for loss, label in enumerate(labels):
        baseline = [_row(item_id, 0.5, rank=3) for item_id in ids]
        candidate = [
            _row(item_id, 0.25 if index == loss else 1.0, rank=1)
            for index, item_id in enumerate(ids)
        ]
        cells[label] = (baseline, candidate)
    report = _sweep(cells)
    verdict = report["verdict"]

    assert verdict["per_row_answer_cells"] == list(labels)
    assert verdict["answer_cells"] == []
    assert verdict["decision"] == DECISION_KEEP_BAR
    assert verdict["selection_adjustment"]["family_size"] == 4
    assert all(
        entry["unadjusted_p"] <= 0.025 < entry["adjusted_p"]
        for entry in verdict["selection_adjustment"]["p_values"].values()
    )


def test_a_rank_gain_that_never_reaches_the_answer_keeps_recall_as_the_sole_bar():
    """The measured negative: the encoder does rank better and the answers do not move."""
    ids = _ids(12)
    report = _sweep(
        {
            "k10": (
                [_row(i, 1.0, rank=3) for i in ids],
                [_row(i, 1.0, rank=1) for i in ids],
            ),
            "k3": (
                [_row(i, 1.0, rank=2) for i in ids],
                [_row(i, 1.0, rank=1) for i in ids],
            ),
        }
    )
    verdict = report["verdict"]
    assert verdict["decision"] == DECISION_KEEP_BAR
    assert verdict["answer_cells"] == []
    assert verdict["rank_cells"] == ["k10", "k3"]
    assert "recall@k stays the sole adoption bar" in verdict["reason"]


def test_a_sweep_where_the_rank_gain_never_reproduces_decides_nothing():
    """An unmet premise is not a `keep` -- the sweep never tested the question."""
    ids = _ids(12)
    report = _sweep(
        {"k10": ([_row(i, 1.0, rank=1) for i in ids], [_row(i, 1.0, rank=1) for i in ids])}
    )
    verdict = report["verdict"]
    assert verdict["decision"] == DECISION_NO_EVIDENCE
    assert verdict["rank_cells"] == []
    assert "never tested" in verdict["reason"]


def test_a_one_item_answer_gain_does_not_clear_the_interval():
    """The bar reads the paired INTERVAL, never the point estimate."""
    ids = _ids(12)
    report = _sweep(
        {
            "k3": (
                [_row(i, 0.0, rank=2) for i in ids],
                [_row(i, 1.0 if i == "q0" else 0.0, rank=1) for i in ids],
            )
        }
    )
    assert report["cells"][0]["paired"][METRIC_OBJECTIVE]["delta"]["mean"] > 0.0
    assert report["verdict"]["decision"] == DECISION_KEEP_BAR


def test_every_cell_is_paired_on_the_same_items_and_the_same_resample_draw():
    """Common random numbers across cells: the cells are comparable to each other, not just within."""
    ids = _ids(10)
    rows = [_row(i, 1.0, rank=1) for i in ids]
    report = _sweep({"k10": (rows, rows), "k3": (rows, rows)})
    assert report["item_ids"] == sorted(ids)
    assert all(cell["n"] == 10 for cell in report["cells"])
    intervals = [cell["paired"][METRIC_OBJECTIVE]["delta"] for cell in report["cells"]]
    assert intervals[0] == intervals[1]


def test_a_cell_that_scored_a_different_item_set_is_not_a_comparison():
    ids = _ids(4)
    with pytest.raises(ValueError, match="different item sets"):
        _sweep(
            {
                "k10": ([_row(i, 1.0) for i in ids], [_row(i, 1.0) for i in ids]),
                "k3": ([_row(i, 1.0) for i in ids], [_row(i, 1.0) for i in ids[:3]]),
            }
        )


def test_a_cell_missing_an_encoder_fails_loudly():
    ids = _ids(4)
    with pytest.raises(ValueError, match="did not score"):
        compare_cells(
            [(CellSpec(10, None), {BASELINE: [_row(i, 1.0) for i in ids]})],
            {},
            baseline=BASELINE,
            candidate=CANDIDATE,
            resamples=0,
        )


def test_an_empty_sweep_is_refused():
    with pytest.raises(ValueError, match="at least one cell"):
        compare_cells([], {}, baseline=BASELINE, candidate=CANDIDATE)


def test_report_renders_ascii_tables_with_the_verdict_and_every_cell():
    ids = _ids(6)
    report = _sweep(
        {
            "k10": ([_row(i, 1.0, rank=3) for i in ids], [_row(i, 1.0, rank=1) for i in ids]),
            "k3": ([_row(i, 0.0, rank=None, hit=0.0) for i in ids], [_row(i, 1.0) for i in ids]),
        }
    )
    text = format_report(report, metadata={"model": "m", "backend": "ollama"})
    assert "# Embedder adoption bar" in text
    assert "### Answer-side delta per cell" in text
    assert "### Per-encoder means" in text
    assert all(label in text for label in ("k10", "k3"))
    assert DECISION_EXTEND_BAR in text
    assert text.isascii()
    summary = format_summary(report)
    assert "d objective" in summary and summary.isascii()
