"""Reranker bake-off lane: shared pools, rank quality, cost columns, and the keep-or-swap verdict.

Every case runs over fake cross-encoders (`_rerank_bakeoff_helpers`) -- no download, no GPU.
"""

import pytest

from tests.llb.rag._rerank_bakeoff_helpers import (
    BASELINE,
    CANDIDATE,
    REMOTE_CODE_CANDIDATE,
    fake_loader,
    items,
    pools,
)

from llb.rag.rerank_bakeoff.fit import fit_verdict
from llb.rag.rerank_bakeoff.lane import run_rerank_bakeoff
from llb.rag.rerank_bakeoff.models import (
    ROW_NO_RERANK,
    SKIP_LOAD_FAILED,
    SKIP_NO_HEADROOM,
    KIND_RETRIEVAL_ORDER,
    LoadedScorer,
    ScorerLoadError,
)

# Gold sits at rank 4 for two thirds of the items and outside the kept top-3 nowhere else, so a
# perfect reranker can move rank while a flat one cannot.
GOLD_POSITIONS = [4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1]
K = 3
POOL_DEPTH = 6


def _run(**overrides):
    scored_items = items(len(GOLD_POSITIONS))
    kwargs = {
        "corpus_root": "corpus",
        "embedding_model": "intfloat/multilingual-e5-base",
        "chunking": "recursive@800/120",
        "pool_depth": POOL_DEPTH,
        "batch_size": 8,
        "candidates": [BASELINE, CANDIDATE],
        "load_scorer": fake_loader(),
        "resamples": 200,
    }
    kwargs.update(overrides)
    return run_rerank_bakeoff(scored_items, pools(GOLD_POSITIONS, POOL_DEPTH), K, **kwargs)


def _row(report, model):
    return next(row for row in report["candidates"] if row["model"] == model)


def test_reranker_off_row_is_the_pool_in_retrieval_order():
    report = _run()
    off = _row(report, ROW_NO_RERANK)
    assert off["kind"] == KIND_RETRIEVAL_ORDER
    # Gold at rank 4 is outside k=3: only the four rank-1 items are found without a reranker.
    assert off["recall_at_k"] == pytest.approx(4 / 12)
    assert off["first_hit_rank_mean"] == pytest.approx(1.0)
    assert off["hit_items"] == 4
    assert off["rerank_ms_per_query"] == 0.0  # the row costs no model


def test_a_perfect_reranker_lifts_every_item_to_rank_one():
    report = _run()
    candidate = _row(report, CANDIDATE)
    assert candidate["recall_at_k"] == 1.0
    assert candidate["mrr"] == 1.0
    assert candidate["first_hit_rank_mean"] == pytest.approx(1.0)
    assert candidate["hit_items"] == 12
    assert candidate["pool_depth"] == POOL_DEPTH
    assert report["best_recall"] == CANDIDATE and report["best_first_hit"] == CANDIDATE


def test_a_flat_reranker_reproduces_retrieval_order_exactly():
    """The incumbent's scorer here scores every candidate equally: a stable sort is a no-op."""
    report = _run()
    flat, off = _row(report, BASELINE), _row(report, ROW_NO_RERANK)
    assert (flat["recall_at_k"], flat["mrr"]) == (off["recall_at_k"], off["mrr"])


def test_every_candidate_is_scored_on_the_identical_pool():
    """One pool per item is retrieved once and re-sorted, so the rows differ only by the model."""
    report = _run()
    assert {row["n"] for row in report["candidates"]} == {12}
    assert {row["k"] for row in report["candidates"]} == {K}
    assert {row["pool_depth"] for row in report["candidates"]} == {POOL_DEPTH}


def test_cost_columns_record_what_the_swap_is_paid_in():
    report = _run(load_scorer=fake_loader({BASELINE: 1200.0, CANDIDATE: 2400.0}))
    candidate = _row(report, CANDIDATE)
    assert candidate["vram_mb"] == 2400.0 and candidate["vram_peak_mb"] == 2400.0
    assert candidate["load_seconds"] == 0.5 and candidate["device"] == "cpu"
    assert candidate["rerank_ms_per_query"] >= 0.0 and candidate["pairs_per_second"] > 0.0


def test_a_candidate_over_the_declared_headroom_is_skipped_with_its_footprint():
    headroom = {
        "total_mb": 16000.0,
        "generator_mb": 14000.0,
        "reserve_mb": 512.0,
        "headroom_mb": 1488.0,
    }
    report = _run(load_scorer=fake_loader({BASELINE: 1200.0, CANDIDATE: 2400.0}), headroom=headroom)
    assert CANDIDATE not in {row["model"] for row in report["candidates"]}
    skipped = next(row for row in report["skipped"] if row["model"] == CANDIDATE)
    assert skipped["reason"] == SKIP_NO_HEADROOM
    assert "2400" in skipped["detail"] and "1488" in skipped["detail"]
    # The one that fits still ranks, and says so on its row.
    assert _row(report, BASELINE)["fits_headroom"] is True


def test_an_unloadable_candidate_is_recorded_and_the_roster_continues():
    def loader(model: str) -> LoadedScorer:
        if model == CANDIDATE:
            raise ScorerLoadError("CUDA out of memory")
        return fake_loader()(model)

    report = _run(load_scorer=loader)
    assert {row["model"] for row in report["candidates"]} == {ROW_NO_RERANK, BASELINE}
    skipped = next(row for row in report["skipped"] if row["model"] == CANDIDATE)
    assert skipped["reason"] == SKIP_LOAD_FAILED and "out of memory" in skipped["detail"]


def test_screened_entries_ride_into_the_report_beside_the_scored_rows():
    declined = [
        {
            "model": REMOTE_CODE_CANDIDATE,
            "family": "jina-reranker-v2",
            "reason": "trust_remote_code_not_opted_in",
            "detail": "needs trust_remote_code",
        }
    ]
    report = _run(skipped=declined)
    assert [row["model"] for row in report["skipped"]] == [REMOTE_CODE_CANDIDATE]


def test_the_verdict_swaps_to_a_candidate_that_clears_both_bars():
    report = _run()
    verdict = report["verdict"]
    assert verdict["decision"] == "adopt" and verdict["model"] == CANDIDATE
    assert set(verdict["cleared"][CANDIDATE]) == {"recall_at_k", "mrr"}
    # Both bars are on by default in this lane, and each row carries every paired metric --
    # the two bars plus the intactness pair, which is reported but never gates adoption.
    assert report["uncertainty"]["bars"] == ["recall_at_k", "mrr"]
    assert set(_row(report, CANDIDATE)["paired_vs_baseline"]["metrics"]) == {
        "recall_at_k",
        "mrr",
        "span_char_coverage_at_k",
        "span_intact_at_k",
    }


def test_a_baseline_the_run_did_not_score_leaves_the_verdict_undecided():
    report = _run(candidates=[CANDIDATE], baseline="acme/not-scored")
    assert report["verdict"]["decision"] == "undecided"
    assert "paired_vs_baseline" not in _row(report, CANDIDATE)


def test_the_paired_ledger_is_keyed_by_item_id():
    report = _run(item_ids=[f"gold-{i}" for i in range(len(GOLD_POSITIONS))])
    ledger = report["paired_items"]
    assert [entry["item_id"] for entry in ledger][:2] == ["gold-0", "gold-1"]
    assert set(ledger[0]["models"]) == {ROW_NO_RERANK, BASELINE, CANDIDATE}


def test_a_pool_per_item_is_required():
    with pytest.raises(ValueError, match="one candidate pool per scored item"):
        run_rerank_bakeoff(
            items(3),
            pools([1, 1], POOL_DEPTH),
            K,
            corpus_root="c",
            embedding_model="e",
            chunking="recursive@800/120",
            pool_depth=POOL_DEPTH,
            batch_size=8,
            candidates=[],
            load_scorer=fake_loader(),
        )


def test_the_floor_reads_each_lane_on_its_own_rerank_scores():
    report = _run(noise_floor=True, noise_floor_replicates=8)
    floor = report["noise_floor"]
    assert set(floor["lanes"]) == {ROW_NO_RERANK, BASELINE, CANDIDATE}
    # A perfect reranker separates gold from every other candidate by a full point, so no replicate
    # of that lane can flip an item: its band is exactly zero.
    assert floor["lanes"][CANDIDATE]["recall_at_k"]["half_width"] == 0.0
    assert floor["lanes"][CANDIDATE]["recall_at_k"]["base"] == 1.0
    assert set(floor["jitter_by_lane"]) == {ROW_NO_RERANK, BASELINE, CANDIDATE}


@pytest.mark.parametrize(
    "vram,headroom,expected",
    [
        (2400.0, None, None),  # no budget declared -> the gate does not run
        (2400.0, 1488.0, False),
        (1200.0, 1488.0, True),
        (None, 1488.0, None),  # no GPU reader -> nothing measured to gate on
    ],
)
def test_fit_verdict_needs_both_a_measurement_and_a_budget(vram, headroom, expected):
    budget = (
        None
        if headroom is None
        else {
            "total_mb": 16000.0,
            "generator_mb": 14000.0,
            "reserve_mb": 512.0,
            "headroom_mb": headroom,
        }
    )
    assert fit_verdict(vram, budget) is expected
