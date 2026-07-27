"""Focused tests split from ``test_null_calibration.py``."""

import pytest
from _null_calibration_helpers import (
    _distribution,
    _semantic_stats,
)
from conflict_helpers import (
    FIXTURE_CORPUS,
    fake_store_view,
)

from llb.conflicts.audit import (
    AuditParams,
    run_audit,
)
from llb.conflicts.constants import (
    DEFAULT_COSINE_THRESHOLD,
    TIER_SEMANTIC,
)
from llb.conflicts.null_calibration import resolve_cos_threshold


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
