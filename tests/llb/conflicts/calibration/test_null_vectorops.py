"""Focused tests split from ``test_null_calibration.py``."""

import math
import random

import pytest
from tests.llb.conflicts._null_calibration_helpers import (
    _distribution,
    _synthetic_store,
)
from tests.llb.conflicts.conflict_helpers import (
    FIXTURE_CORPUS,
    fake_store_view,
)

from llb.conflicts.calibration.distribution import NullDistribution
from llb.conflicts.calibration.sampling import estimate_null_distribution
from llb.conflicts.tiers.semantic import content_ordinals
from llb.conflicts.semantic_tree.vectorops import VectorSet


def test_pair_similarities_matches_the_scalar_path():
    vectors, _ = _synthetic_store(n_docs=3, per_doc=4, dim=16)
    pairs = [(0, 1), (2, 5), (3, 11)]
    batched = vectors.pair_similarities(pairs)
    assert batched == [pytest.approx(vectors.similarity(left, right)) for left, right in pairs]
    assert vectors.pair_similarities([]) == []


def test_pair_similarities_agrees_without_numpy():
    vectors, chunks = _synthetic_store(n_docs=3, per_doc=4, dim=16)
    rows = [vectors.row(index) for index in range(len(chunks))]
    pure = VectorSet(rows, use_numpy=False)
    pairs = [(0, 1), (2, 5), (3, 11)]
    assert pure.pair_similarities(pairs) == [
        pytest.approx(value) for value in vectors.pair_similarities(pairs)
    ]


def test_calibration_samples_only_comparable_chunks():
    """Front matter and low-content chunks must not inflate the estimated tail."""
    store = fake_store_view()
    from llb.conflicts.corpus import load_corpus_docs

    docs = load_corpus_docs(FIXTURE_CORPUS)
    body_offsets = {doc.doc_id: doc.body_offset for doc in docs}
    allowed = content_ordinals(store.chunks, body_offsets)
    everything = set(range(len(store.chunks)))
    assert allowed < everything  # the fixture really does carry excluded chunks

    narrow = estimate_null_distribution(store.vectors, store.chunks, allowed, sample_pairs=10_000)
    wide = estimate_null_distribution(store.vectors, store.chunks, everything, sample_pairs=10_000)
    assert narrow is not None and wide is not None
    assert narrow.n_pairs < wide.n_pairs


def test_distribution_payload_is_json_safe():
    distribution = NullDistribution(
        similarities=[0.0, 0.25, 0.5, 0.75, 1.0],
        n_pairs=5,
        total_pairs=5,
        seed=2,
        exhaustive=False,
    )
    payload = distribution.payload(0.5)
    assert payload["resolved_cos_threshold"] == 0.5
    assert payload["mean"] == 0.5
    assert payload["min"] == 0.0 and payload["max"] == 1.0
    assert all(isinstance(value, float) for value in payload["quantiles"].values())
    assert all(math.isfinite(value) for value in payload["quantiles"].values())


def test_candidate_budget_selects_exactly_n_pairs_over_an_exhaustive_distribution():
    """The budget is a RANK cutoff, and this pins that contract exactly.

    Over an exhaustive distribution the null and the observed population are the same set, so a
    budget of N cuts at the Nth largest similarity and the scan returns precisely N pairs. That
    exactness is the feature; the absence of any false-positive claim is the documented
    limitation (see `no independent null` in
    docs/impl/current/data-prep/conflict-detection.md).
    """
    rng = random.Random(0)
    values = sorted(rng.gauss(0.0, 1.0) for _ in range(10_000))
    assert len(set(values)) == len(values)  # a re-seeded rng would make these all identical
    distribution = _distribution(similarities=values, n_pairs=10_000, total_pairs=10_000)
    for budget in (1, 5, 12, 50):
        cutoff = distribution.quantile(distribution.quantile_for_top_n(budget))
        assert sum(1 for value in values if value >= cutoff) == budget
