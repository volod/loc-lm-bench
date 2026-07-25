"""adoption-cell-reading-reports-a-binary-on-borderline-evidence -- is a reading settled?

Pure: the input is per-item delta vectors, so the whole borderline lane is unit-tested with lists
-- no backend, store, or GPU.
"""

import pytest

from llb.eval.embedder_adoption import READING_ANSWER, READING_NEITHER, READING_RANK_ONLY
from llb.eval.embedder_adoption.models import ItemDeltas
from llb.eval.embedder_adoption.stability import (
    LOOSER_CONFIDENCE,
    SIDE_ABOVE,
    SIDE_BELOW,
    decision_probability,
    exceedance,
    format_reading,
    row_stability,
)
from llb.rag.fusion_evidence.stats import bootstrap_index_sets


def _deltas(objective: list[float], rr: list[float] | None = None) -> ItemDeltas:
    return ItemDeltas(
        item_ids=[f"q{i}" for i in range(len(objective))],
        objective=objective,
        reciprocal_rank=rr if rr is not None else [0.0] * len(objective),
    )


# --- the exceedance probability -----------------------------------------------------------


def test_exceedance_is_the_share_of_resamples_above_zero():
    idx = bootstrap_index_sets(20, 400, 13)
    assert exceedance([1.0] * 20, idx) == 1.0
    assert exceedance([-1.0] * 20, idx) == 0.0
    # A symmetric spread straddles zero, so it lands near a half.
    mixed = exceedance([1.0] * 10 + [-1.0] * 10, idx)
    assert 0.3 < mixed < 0.7


def test_exceedance_of_nothing_is_zero_not_a_crash():
    assert exceedance([], bootstrap_index_sets(0, 10, 13)) == 0.0
    assert exceedance([1.0], []) == 0.0


def test_the_decision_probability_matches_the_interval_rule():
    """`lo > 0` at 95% is exactly `p_positive > 0.975` -- the two must not drift apart."""
    assert decision_probability(0.95) == pytest.approx(0.975)
    assert decision_probability(0.90) == pytest.approx(0.95)


# --- the borderline flag ------------------------------------------------------------------


def test_a_decisive_reading_is_settled_at_both_levels():
    stability = row_stability(_deltas([0.5] * 30), resamples=400)
    assert stability["reading"] == READING_ANSWER
    assert stability["looser_reading"] == READING_ANSWER
    assert stability["tighter_reading"] == READING_ANSWER
    assert stability["borderline"] is False
    assert stability["side"] is None
    assert stability["p_positive"] == 1.0


def test_a_flat_reading_is_settled_too():
    stability = row_stability(_deltas([0.0] * 30), resamples=400)
    assert stability["reading"] == READING_NEITHER
    assert stability["borderline"] is False


def test_a_reading_that_a_looser_interval_would_change_is_borderline():
    """The measured lapa case: just misses at 95%, clears at 90%."""
    # A small positive mean with enough spread to put p_positive between the two cut points
    # (measured: p_positive ~= 0.970, between the 90% cut at 0.950 and the 95% cut at 0.975).
    deltas = _deltas([1.0] * 8 + [-1.0] * 2 + [0.0] * 20)
    stability = row_stability(deltas, resamples=2000)
    cut_95, cut_90 = decision_probability(0.95), decision_probability(LOOSER_CONFIDENCE)
    assert cut_90 < stability["p_positive"] < cut_95
    assert stability["reading"] == READING_NEITHER
    assert stability["looser_reading"] == READING_ANSWER
    assert stability["borderline"] is True
    assert stability["side"] == SIDE_BELOW


def test_a_reading_that_only_just_clears_the_bar_is_borderline_above():
    """The two-sided half: it PASSES at 95% but a 97.5% interval would drop it."""
    # Measured on the recorded k3 rows: p_positive just over the 0.975 cut, under 0.9875.
    deltas = _deltas([1.0] * 9 + [-1.0] * 2 + [0.0] * 19)
    stability = row_stability(deltas, resamples=2000)
    assert decision_probability(0.95) < stability["p_positive"] < decision_probability(0.975)
    assert stability["reading"] == READING_ANSWER
    assert stability["looser_reading"] == READING_ANSWER  # a looser level agrees
    assert stability["tighter_reading"] != READING_ANSWER  # a tighter one drops it
    assert stability["borderline"] is True
    assert stability["side"] == SIDE_ABOVE


def test_the_borderline_check_reuses_one_resample_draw():
    """Both readings come from the SAME index sets, so only the percentile cut differs."""
    deltas = _deltas([1.0] * 8 + [-1.0] * 2 + [0.0] * 20)
    a = row_stability(deltas, resamples=800, seed=5)
    b = row_stability(deltas, resamples=800, seed=5)
    assert a == b


def test_a_rank_only_row_can_be_borderline_on_the_objective():
    """The reading is three-state, so the flag tracks the READING, not one metric."""
    objective = [1.0] * 8 + [-1.0] * 2 + [0.0] * 20
    stability = row_stability(_deltas(objective, [0.5] * 30), resamples=2000)
    assert stability["reading"] == READING_RANK_ONLY  # objective misses, rank clears
    assert stability["looser_reading"] == READING_ANSWER
    assert stability["borderline"] is True


def test_the_neighbouring_levels_must_straddle_the_reporting_confidence():
    with pytest.raises(ValueError, match="must straddle"):
        row_stability(_deltas([0.5] * 10), resamples=100, looser_confidence=0.99)
    with pytest.raises(ValueError, match="must straddle"):
        row_stability(_deltas([0.5] * 10), resamples=100, tighter_confidence=0.90)


# --- rendering ----------------------------------------------------------------------------


def test_format_reading_marks_only_borderline_rows():
    settled = row_stability(_deltas([0.5] * 30), resamples=400)
    assert format_reading(settled, READING_ANSWER) == "answer"
    marked = row_stability(_deltas([1.0] * 8 + [-1.0] * 2 + [0.0] * 20), resamples=2000)
    assert format_reading(marked, READING_NEITHER) == "neither (borderline)"


def test_format_reading_falls_back_when_no_stability_was_measured():
    """An archived roster whose run bundles are gone still renders its readings."""
    assert format_reading(None, READING_RANK_ONLY) == "rank only"
