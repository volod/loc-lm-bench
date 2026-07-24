"""Retrieval-comparison measurement floor (`llb.rag.noise_floor`).

Pure: fake stores expose the `.retrieve` seam and carry explicit `retrieval_score`s, so the
spread statistic is exercised with no FAISS, no GPU, and no embedder.
"""

import pytest

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.compare import compare_retrieval, format_comparison
from llb.rag.noise_floor import DEFAULT_SCORE_JITTER, measure_noise_floor
from llb.rag.noise_floor_report import format_noise_floor

REPLICATES = 32


class _ScoredStore:
    """Returns the same scored candidates every time, truncated to the requested depth."""

    def __init__(self, hits: list[ChunkRecord]) -> None:
        self._hits = hits
        self.depths: list[int] = []

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        self.depths.append(k)
        return self._hits[:k]


def _chunk(doc: str, start: int, end: int, score: float | None = None) -> ChunkRecord:
    chunk: ChunkRecord = {"doc_id": doc, "char_start": start, "char_end": end, "text": "x"}
    if score is not None:
        chunk["retrieval_score"] = score
    return chunk


def _span(doc: str, start: int, end: int) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "g"}


def _items() -> list[tuple[str, list[SourceSpanRecord]]]:
    return [("питання", [_span("d1", 0, 10)])]


def _separated_hits() -> list[ChunkRecord]:
    # The gold-hitting chunk sits at rank 1 with a wide score margin: noise cannot move it.
    return [_chunk("d1", 0, 10, 0.9), _chunk("d2", 0, 10, 0.5), _chunk("d3", 0, 10, 0.1)]


def _tied_hits(k: int) -> list[ChunkRecord]:
    # k fillers and the gold chunk all share one score, so the rank-k cut is arbitrary and
    # whether the gold chunk lands inside the top-k is decided by tie order alone.
    fillers = [_chunk("d2", i * 10, i * 10 + 5, 0.5) for i in range(k)]
    return [*fillers, _chunk("d1", 0, 10, 0.5)]


def test_floor_is_zero_when_scores_are_well_separated():
    report = measure_noise_floor(
        {"faiss": _ScoredStore(_separated_hits())}, _items(), k=2, replicates=REPLICATES
    )
    lane = report["lanes"]["faiss"]
    assert lane["recall_at_k"] == {
        "base": 1.0,
        "min": 1.0,
        "max": 1.0,
        "mean": 1.0,
        "std": 0.0,
        "half_width": 0.0,
    }
    assert report["floor_recall_at_k"] == 0.0 and report["floor_mrr"] == 0.0


def test_floor_is_nonzero_when_the_cut_sits_on_a_tie():
    report = measure_noise_floor(
        {"faiss": _ScoredStore(_tied_hits(k=3))}, _items(), k=3, replicates=REPLICATES
    )
    recall = report["lanes"]["faiss"]["recall_at_k"]
    assert recall["min"] == 0.0 and recall["max"] == 1.0  # the tie flips the verdict both ways
    assert recall["half_width"] == 0.5
    assert report["floor_recall_at_k"] == 0.5


def test_floor_retrieves_the_pool_once_and_the_base_at_k():
    # One deep pass feeds every replicate; `base` re-uses the comparison's own depth-k call, so
    # the reported base cannot drift from the comparison row.
    store = _ScoredStore(_separated_hits())
    measure_noise_floor({"faiss": store}, _items(), k=2, replicates=REPLICATES)
    assert store.depths == [6, 2]  # CANDIDATE_DEPTH_FACTOR * k, then k


def test_fragile_items_count_the_ties_at_the_cut():
    tied = measure_noise_floor(
        {"faiss": _ScoredStore(_tied_hits(k=3))}, _items(), k=3, replicates=REPLICATES
    )
    assert tied["lanes"]["faiss"] == {**tied["lanes"]["faiss"], "n": 1, "fragile_items": 1}
    separated = measure_noise_floor(
        {"faiss": _ScoredStore(_separated_hits())}, _items(), k=2, replicates=REPLICATES
    )
    assert separated["lanes"]["faiss"]["fragile_items"] == 0


def test_floor_is_reproducible_for_the_same_seed_and_differs_per_lane():
    hits = _tied_hits(k=3)
    stores = {"a": _ScoredStore(hits), "b": _ScoredStore(list(hits))}
    first = measure_noise_floor(stores, _items(), k=3, replicates=REPLICATES)
    second = measure_noise_floor(stores, _items(), k=3, replicates=REPLICATES)
    assert first["lanes"] == second["lanes"]  # seeded: byte-identical across calls
    # Lanes are seeded independently, so identical stores do not draw the identical noise.
    assert first["lanes"]["a"]["recall_at_k"]["mean"] != first["lanes"]["b"]["recall_at_k"]["mean"]


def test_floor_skips_a_lane_whose_candidates_carry_no_score():
    stores = {
        "scored": _ScoredStore(_separated_hits()),
        "unscored": _ScoredStore([_chunk("d1", 0, 10)]),
    }
    report = measure_noise_floor(stores, _items(), k=2, replicates=REPLICATES)
    assert report["unscored"] == ["unscored"]
    assert set(report["lanes"]) == {"scored"}


@pytest.mark.parametrize(
    "kwargs,message",
    [({"replicates": 1}, "at least 2"), ({"jitter": 0.0}, "must be > 0")],
)
def test_floor_rejects_a_degenerate_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        measure_noise_floor({}, _items(), k=2, **kwargs)


def test_floor_of_no_lanes_is_zero():
    report = measure_noise_floor({}, _items(), k=2, replicates=REPLICATES)
    assert report["lanes"] == {} and report["floor_recall_at_k"] == 0.0


def test_default_jitter_covers_the_measured_between_process_score_drift():
    # Anchor for the constant: the measured maximum drift was 6.0e-7 per cosine score.
    assert DEFAULT_SCORE_JITTER >= 6.0e-7


def test_format_noise_floor_is_ascii_and_states_the_floor():
    report = measure_noise_floor(
        {"faiss": _ScoredStore(_tied_hits(k=3))}, _items(), k=3, replicates=REPLICATES
    )
    text = "\n".join(format_noise_floor(report))
    assert text.isascii()  # AGENTS.md: ASCII-only output
    assert "noise floor" in text and "faiss" in text
    assert "read any smaller delta as noise" in text


def test_margin_reads_the_top_two_lanes_against_the_floor():
    stores = {
        "hit": _ScoredStore(_separated_hits()),
        "miss": _ScoredStore([_chunk("d9", 0, 10, 0.9), _chunk("d8", 0, 10, 0.5)]),
    }
    margin = measure_noise_floor(stores, _items(), k=2, replicates=REPLICATES)["margin"]
    assert margin["leader"] == "hit" and margin["runner_up"] == "miss"
    assert margin["delta"] == 1.0 and margin["floor"] == 0.0
    assert margin["clears_floor"] is True


def test_margin_of_a_lead_inside_the_floor_does_not_clear_it():
    # Two lanes over the same tie: identical base recall, and a band the tie order can move.
    stores = {"a": _ScoredStore(_tied_hits(k=3)), "b": _ScoredStore(_tied_hits(k=3))}
    report = measure_noise_floor(stores, _items(), k=3, replicates=REPLICATES)
    margin = report["margin"]
    assert margin["delta"] == 0.0 and margin["floor"] > 0.0
    assert margin["clears_floor"] is False
    assert "does NOT clear the floor" in "\n".join(format_noise_floor(report))


def test_margin_of_a_single_lane_has_nothing_to_clear():
    report = measure_noise_floor(
        {"faiss": _ScoredStore(_separated_hits())}, _items(), k=2, replicates=REPLICATES
    )
    margin = report["margin"]
    assert margin["runner_up"] is None and margin["clears_floor"] is False
    assert "nothing to clear" in "\n".join(format_noise_floor(report))


class _PoolStore:
    """A lane whose deeper ranking is NOT its top-k extended -- the fused-row case.

    Asking such a lane for `3k` results re-ranks a deeper pool and answers for a DIFFERENT row, so
    it exposes the pool seam: the published top-k ranking with the cut moved instead.
    """

    def __init__(self) -> None:
        self.pool_calls: list[tuple[int, int]] = []
        self.depths: list[int] = []

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        self.depths.append(k)
        if k > 2:  # a deeper request fuses a deeper pool: another row, and it loses the gold
            return [_chunk("dX", i, i + 5, 0.9 - i / 100) for i in range(k)]
        return self._published(k)

    def retrieve_candidate_pool(self, question: str, k: int, candidates: int) -> list[ChunkRecord]:
        self.pool_calls.append((k, candidates))
        return self._published(candidates)

    def _published(self, cut: int) -> list[ChunkRecord]:
        ranking = [_chunk("d1", 0, 10, 0.9)]
        ranking += [_chunk("dY", i * 5, i * 5 + 5, 0.1) for i in range(cut)]
        return ranking[:cut]


def test_floor_perturbs_the_pool_seam_when_a_lane_exposes_one():
    store = _PoolStore()
    report = measure_noise_floor({"fused": store}, _items(), k=2, replicates=REPLICATES)
    assert store.pool_calls == [(2, 6)]  # asked for its own k, cut at CANDIDATE_DEPTH_FACTOR * k
    assert store.depths == [2]  # the deep `retrieve` that would answer for another row never runs
    # The pool's top-k IS the published row, so the band brackets the published recall.
    lane = report["lanes"]["fused"]
    assert lane["recall_at_k"]["base"] == 1.0 and lane["recall_at_k"]["min"] == 1.0


def test_comparison_renders_the_floor_when_it_is_attached():
    stores = {"faiss": _ScoredStore(_tied_hits(k=3))}
    report = compare_retrieval(stores, _items(), k=3)
    assert "noise_floor" not in report  # opt-in: the default comparison is unchanged
    report["noise_floor"] = measure_noise_floor(stores, _items(), k=3, replicates=REPLICATES)
    rendered = format_comparison(report)
    assert "noise floor" in rendered and "best (recall@k)" in rendered


def test_margin_breaks_a_recall_tie_by_mrr_like_the_tables_do():
    # Same recall, different first-hit rank: the runner-up must be the row the table prints second.
    stores = {
        "top": _ScoredStore(_separated_hits()),
        "mid-mrr": _ScoredStore([_chunk("d2", 0, 10, 0.9), _chunk("d1", 0, 10, 0.5)]),
        "low-mrr": _ScoredStore(
            [_chunk("d2", 0, 10, 0.9), _chunk("d3", 0, 10, 0.7), _chunk("d1", 0, 10, 0.5)]
        ),
    }
    margin = measure_noise_floor(stores, _items(), k=3, replicates=REPLICATES)["margin"]
    assert margin["leader"] == "top" and margin["runner_up"] == "mid-mrr"
