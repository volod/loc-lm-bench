"""Corpus-calibrated cosine threshold: null-distribution sampling and knob precedence."""

import math
import random

import pytest

from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.constants import DEFAULT_COSINE_THRESHOLD, TIER_SEMANTIC
from llb.conflicts.null_calibration import resolve_cos_threshold
from llb.conflicts.null_distribution import MIN_NULL_PAIRS, NullDistribution, _quantile
from llb.conflicts.null_sampling import _cross_document_sample, estimate_null_distribution
from llb.conflicts.semantic_tier import content_ordinals
from llb.conflicts.vectorops import VectorSet
from tests.llb.conflicts.conflict_helpers import FIXTURE_CORPUS, fake_store_view


def _semantic_stats(result):
    return next(stat for stat in result.tiers if stat.tier == TIER_SEMANTIC)


def _synthetic_store(n_docs: int = 12, per_doc: int = 12, dim: int = 32, seed: int = 7):
    """Random unit vectors labelled with doc ids -- a corpus with no real duplication."""
    rng = random.Random(seed)
    chunks = []
    rows = []
    for doc in range(n_docs):
        for index in range(per_doc):
            chunks.append(
                {
                    "doc_id": f"doc-{doc}.md",
                    "chunk_id": f"doc-{doc}-{index}",
                    "char_start": 0,
                    "char_end": 100,
                    "text": "word " * 40,
                }
            )
            rows.append([rng.gauss(0.0, 1.0) for _ in range(dim)])
    return VectorSet(rows), chunks


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


def _distribution(**kwargs) -> NullDistribution:
    defaults = dict(
        similarities=[0.1, 0.2, 0.3, 0.4], n_pairs=4, total_pairs=4, seed=0, exhaustive=True
    )
    return NullDistribution(**{**defaults, **kwargs})


def test_precedence_explicit_beats_quantile_beats_budget_beats_default():
    distribution = _distribution()
    assert resolve_cos_threshold(
        explicit=0.75, quantile=0.999, default=0.9, distribution=distribution
    ) == (0.75, "explicit", None)

    threshold, source, used = resolve_cos_threshold(
        explicit=None, quantile=1.0, default=0.9, distribution=distribution
    )
    assert (threshold, source, used) == (0.4, "calibrated", 1.0)

    # A raw quantile wins over a budget when both are given.
    _, _, used = resolve_cos_threshold(
        explicit=None, quantile=1.0, default=0.9, distribution=distribution, max_candidate_pairs=2
    )
    assert used == 1.0

    assert resolve_cos_threshold(explicit=None, quantile=None, default=0.9, distribution=None) == (
        0.9,
        "default",
        None,
    )
    # A knob that could not be estimated falls back rather than inventing a number.
    assert resolve_cos_threshold(explicit=None, quantile=0.999, default=0.9, distribution=None) == (
        0.9,
        "default",
        None,
    )


def test_candidate_budget_tightens_the_quantile_as_the_corpus_grows():
    """The whole point of the budget knob: the same budget scales with the pair space."""
    small = _distribution(total_pairs=1_000)
    large = _distribution(total_pairs=1_000_000)
    assert small.quantile_for_top_n(1) == pytest.approx(0.999)
    assert large.quantile_for_top_n(1) == pytest.approx(0.999999)
    # A fixed quantile would admit 1000x more chance flags on the larger corpus; the budget does
    # not, which is what makes it portable across corpus sizes.
    assert large.quantile_for_top_n(1) > small.quantile_for_top_n(1)
    assert small.quantile_for_top_n(10) < small.quantile_for_top_n(1)


def test_budget_resolves_through_a_real_audit_and_records_the_rank_cutoff():
    result = run_audit(
        FIXTURE_CORPUS,
        AuditParams(effort=TIER_SEMANTIC, max_candidate_pairs=1),
        store=fake_store_view(),
    )
    extra = _semantic_stats(result).extra
    assert extra["cos_threshold_source"] == "calibrated"
    null = extra["null_distribution"]
    assert null["max_candidate_pairs"] == 1
    assert null["selected_rank"] == pytest.approx(1.0, abs=0.01)
    assert null["resolved_quantile"] == pytest.approx(1.0 - 1.0 / null["total_pairs"])


def test_audit_records_the_resolved_threshold_and_distribution():
    result = run_audit(
        FIXTURE_CORPUS,
        AuditParams(effort=TIER_SEMANTIC, cos_quantile=0.9, null_sample_pairs=10_000),
        store=fake_store_view(),
    )
    extra = _semantic_stats(result).extra
    assert extra["cos_threshold_source"] == "calibrated"
    null = extra["null_distribution"]
    assert null["resolved_quantile"] == 0.9
    assert null["resolved_cos_threshold"] == pytest.approx(extra["cos_threshold"], abs=1e-6)
    assert null["exhaustive"] is True
    assert set(null["quantiles"]) == {"0.5", "0.9", "0.99", "0.999", "0.9999"}
    # The resolved absolute cosine is what the tier actually scanned with.
    assert result.tree_meta["cos_threshold"] == pytest.approx(extra["cos_threshold"])


def test_explicit_threshold_overrides_the_calibrated_one_in_a_real_audit():
    params = AuditParams(effort=TIER_SEMANTIC, cos_threshold=0.85, cos_quantile=0.999)
    result = run_audit(FIXTURE_CORPUS, params, store=fake_store_view())
    extra = _semantic_stats(result).extra
    assert extra["cos_threshold"] == 0.85
    assert extra["cos_threshold_source"] == "explicit"


def test_audit_without_either_knob_keeps_the_fixed_default():
    result = run_audit(FIXTURE_CORPUS, AuditParams(effort=TIER_SEMANTIC), store=fake_store_view())
    extra = _semantic_stats(result).extra
    assert extra["cos_threshold"] == DEFAULT_COSINE_THRESHOLD
    assert extra["cos_threshold_source"] == "default"
    assert "null_distribution" not in extra


def test_calibrated_audit_is_reproducible_run_to_run():
    def once():
        return run_audit(
            FIXTURE_CORPUS,
            AuditParams(effort=TIER_SEMANTIC, cos_quantile=0.99, null_seed=3),
            store=fake_store_view(),
        )

    first, second = once(), once()
    assert _semantic_stats(first).extra["cos_threshold"] == pytest.approx(
        _semantic_stats(second).extra["cos_threshold"]
    )
    assert len(first.findings) == len(second.findings)


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
    limitation (see `no independent null` in docs/impl/current/data-prep.md).
    """
    rng = random.Random(0)
    values = sorted(rng.gauss(0.0, 1.0) for _ in range(10_000))
    assert len(set(values)) == len(values)  # a re-seeded rng would make these all identical
    distribution = _distribution(similarities=values, n_pairs=10_000, total_pairs=10_000)
    for budget in (1, 5, 12, 50):
        cutoff = distribution.quantile(distribution.quantile_for_top_n(budget))
        assert sum(1 for value in values if value >= cutoff) == budget
