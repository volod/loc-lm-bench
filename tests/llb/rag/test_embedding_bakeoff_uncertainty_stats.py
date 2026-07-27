"""Paired sampling uncertainty of the embedder bake-off (`embedding_bakeoff_uncertainty`).

Pure: per-item metric vectors, the shared-index paired bootstrap, the adopt-or-retain verdict, and
the report columns all run over fake stores and plain vectors -- no FAISS, no GPU, no numpy.
"""

from llb.rag.embedding_bakeoff import score_candidate


from llb.rag.embedding_bakeoff_models import BuiltStore


from llb.rag.embedding_bakeoff_uncertainty import (
    METRIC_MRR,
    METRIC_RECALL,
    item_vectors,
    paired_rows,
    recall_delta,
)


from llb.rag.embedding_bakeoff_verdict import (
    separates_from_baseline,
)


from _embedding_bakeoff_uncertainty_helpers import (
    BASELINE,
    _HitSetStore,
    _items,
    _vectors,
)


def test_item_vectors_mean_matches_the_published_row():
    items = _items(4)
    store = _HitSetStore({items[0][0], items[2][0]})
    built = BuiltStore(store=store, embed_seconds=1.0, index_bytes=10)
    row = score_candidate("m", built, items, k=2)
    vectors = item_vectors([(store.retrieve(q, 2), spans) for q, spans in items], k=2)
    assert vectors[METRIC_RECALL] == [1.0, 1.0, 1.0, 1.0]  # k=2 retrieves both chunks
    assert vectors[METRIC_MRR] == [1.0, 0.5, 1.0, 0.5]  # gold first only where the store hits
    assert row["recall_at_k"] == sum(vectors[METRIC_RECALL]) / 4
    assert row["mrr"] == sum(vectors[METRIC_MRR]) / 4


def test_paired_delta_keeps_the_item_pairing():
    # Candidate wins 8 items outright and never loses one: a consistent, separated lead.
    baseline = [1.0] * 6 + [0.0] * 14
    candidate = [1.0] * 6 + [1.0] * 8 + [0.0] * 6
    paired = paired_rows(
        {BASELINE: _vectors(baseline), "cand": _vectors(candidate)}, BASELINE, resamples=500
    )
    delta = recall_delta(paired["cand"])
    assert (delta["wins"], delta["losses"], delta["ties"]) == (8, 0, 12)
    assert delta["delta"]["mean"] == 0.4
    assert delta["delta"]["lo"] > 0.0  # the interval clears zero -> a separated candidate
    assert separates_from_baseline(paired["cand"]) is True
    assert paired["cand"]["baseline"] == BASELINE


def test_baseline_row_is_paired_against_itself_at_exactly_zero():
    paired = paired_rows({BASELINE: _vectors([1.0, 0.0, 1.0])}, BASELINE, resamples=64)
    delta = recall_delta(paired[BASELINE])["delta"]
    assert (delta["mean"], delta["lo"], delta["hi"]) == (0.0, 0.0, 0.0)
    assert separates_from_baseline(paired[BASELINE]) is False


def test_a_one_item_lead_does_not_separate():
    # The exact shape the bake-off re-read produced: a two-question lead on a 40-item set.
    baseline = [1.0] * 37 + [0.0] * 3
    candidate = [1.0] * 39 + [0.0]
    paired = paired_rows(
        {BASELINE: _vectors(baseline), "cand": _vectors(candidate)}, BASELINE, resamples=500
    )
    assert recall_delta(paired["cand"])["delta"]["lo"] == 0.0  # touches zero -> not separated
    assert separates_from_baseline(paired["cand"]) is False


def test_paired_rows_share_one_index_set_and_are_seed_deterministic():
    vectors = {
        BASELINE: _vectors([1.0, 0.0] * 10),
        "cand": _vectors([1.0] * 14 + [0.0] * 6),
    }
    first = paired_rows(vectors, BASELINE, resamples=200, seed=7)
    again = paired_rows(vectors, BASELINE, resamples=200, seed=7)
    other_seed = paired_rows(vectors, BASELINE, resamples=200, seed=8)
    assert first == again  # same seed -> byte-identical intervals
    assert recall_delta(first["cand"])["delta"] != recall_delta(other_seed["cand"])["delta"]


def test_paired_rows_are_empty_when_the_baseline_was_not_scored():
    assert paired_rows({"cand": _vectors([1.0])}, BASELINE) == {}
