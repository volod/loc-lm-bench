"""Focused tests split from ``test_null_calibration.py``."""

import random

import pytest
from _null_calibration_helpers import (
    _distribution,
    _synthetic_store,
)

from llb.conflicts.null_distribution import (
    MIN_NULL_PAIRS,
    _quantile,
)
from llb.conflicts.null_sampling import (
    _cross_document_sample,
    estimate_null_distribution,
)


def test_quantile_matches_linear_interpolation():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _quantile(values, 0.0) == 0.0
    assert _quantile(values, 1.0) == 4.0
    assert _quantile(values, 0.5) == 2.0
    assert _quantile(values, 0.25) == 1.0
    # Between samples the result interpolates rather than snapping to a neighbour.
    assert _quantile(values, 0.3) == pytest.approx(1.2)


def test_quantile_agrees_with_numpy_when_available():
    numpy = pytest.importorskip("numpy")
    rng = random.Random(3)
    values = sorted(rng.gauss(0.0, 1.0) for _ in range(500))
    for q in (0.5, 0.9, 0.99, 0.999):
        assert _quantile(values, q) == pytest.approx(float(numpy.quantile(values, q)))


def test_estimate_is_deterministic_run_to_run():
    vectors, chunks = _synthetic_store()
    allowed = set(range(len(chunks)))
    first = estimate_null_distribution(vectors, chunks, allowed)
    second = estimate_null_distribution(vectors, chunks, allowed)
    assert first is not None and second is not None
    assert first.similarities == second.similarities


def test_sampling_is_deterministic_per_seed_and_varies_across_seeds():
    """The sampler itself; the estimator prefers exhaustive enumeration at this corpus size."""
    _, chunks = _synthetic_store()
    ordinals = list(range(len(chunks)))
    first, _ = _cross_document_sample(chunks, ordinals, sample_pairs=300, seed=11)
    again, _ = _cross_document_sample(chunks, ordinals, sample_pairs=300, seed=11)
    other, _ = _cross_document_sample(chunks, ordinals, sample_pairs=300, seed=12)
    assert first == again
    assert first != other


def test_enumeration_is_preferred_over_sampling_when_the_pair_space_is_small():
    """Sampling puts a 1/N floor under the estimable tail; enumeration removes it entirely."""
    vectors, chunks = _synthetic_store(n_docs=8, per_doc=8)
    allowed = set(range(len(chunks)))
    distribution = estimate_null_distribution(vectors, chunks, allowed, sample_pairs=10)
    assert distribution is not None
    # Even with a tiny sample budget, a small corpus is enumerated exactly.
    assert distribution.exhaustive is True
    assert distribution.n_pairs == distribution.total_pairs
    assert distribution.resolvable_quantile() == 1.0


def test_a_sampled_estimate_reports_the_tail_it_cannot_resolve():
    sampled = _distribution(n_pairs=1_000, total_pairs=1_000_000, exhaustive=False)
    assert sampled.resolvable_quantile() == pytest.approx(0.999)
    # A budget of 1 over a million pairs needs a rarer tail than 1000 samples can express.
    assert sampled.quantile_for_top_n(1) > sampled.resolvable_quantile()


def test_sample_never_pairs_a_document_with_itself():
    """A same-document pair is not a NULL pair -- it is the corpus restating itself."""
    _, chunks = _synthetic_store()
    ordinals = list(range(len(chunks)))
    pairs, exhaustive = _cross_document_sample(chunks, ordinals, sample_pairs=400, seed=5)
    assert exhaustive is False
    assert pairs
    assert all(chunks[left]["doc_id"] != chunks[right]["doc_id"] for left, right in pairs)
    assert all(left < right for left, right in pairs)
    assert len(set(pairs)) == len(pairs)  # sampled without replacement


def test_small_corpus_enumerates_exhaustively_and_ignores_the_seed():
    vectors, chunks = _synthetic_store(n_docs=6, per_doc=5)
    allowed = set(range(len(chunks)))
    first = estimate_null_distribution(vectors, chunks, allowed, sample_pairs=10_000, seed=1)
    second = estimate_null_distribution(vectors, chunks, allowed, sample_pairs=10_000, seed=99)
    assert first is not None and second is not None
    assert first.exhaustive is True
    # No sampling error to differ over: two seeds resolve the identical threshold.
    assert first.similarities == second.similarities
    # 30 chunks over 6 docs: every cross-document pair, and no same-document pair.
    assert first.n_pairs == (30 * 29 // 2) - 6 * (5 * 4 // 2)


def test_too_few_pairs_declines_to_calibrate():
    """Two tiny documents cannot estimate a tail; the run keeps its fixed threshold."""
    vectors, chunks = _synthetic_store(n_docs=2, per_doc=3)
    allowed = set(range(len(chunks)))
    # 2 docs x 3 chunks -> 9 cross-document pairs, far under the guard.
    pairs, _ = _cross_document_sample(chunks, sorted(allowed), sample_pairs=100, seed=0)
    assert len(pairs) == 9 < MIN_NULL_PAIRS
    assert estimate_null_distribution(vectors, chunks, allowed, sample_pairs=100) is None


def test_quantile_resolves_above_the_bulk_of_unrelated_pairs():
    vectors, chunks = _synthetic_store(n_docs=20, per_doc=20)
    allowed = set(range(len(chunks)))
    distribution = estimate_null_distribution(vectors, chunks, allowed, sample_pairs=20_000)
    assert distribution is not None
    tail = distribution.quantile(0.999)
    assert tail > distribution.quantile(0.5)
    assert tail > distribution.quantile(0.99)
    # Random 32-dim vectors: the 99.9th percentile of unrelated pairs sits well below 1.0.
    assert 0.0 < tail < 1.0
