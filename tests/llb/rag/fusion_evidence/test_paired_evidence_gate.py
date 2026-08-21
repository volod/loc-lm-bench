"""Focused tests split from ``test_paired_minimum_evidence.py``."""

import pytest
from tests.llb.rag._paired_minimum_evidence_helpers import (
    BOUND,
    RESAMPLES,
    SEED,
    _unanimous,
)

from llb.rag.fusion_evidence.evidence_gate import (
    READING_FLAT,
    READING_INSUFFICIENT_EVIDENCE,
    READING_SEPARATED,
    apply_evidence_gate,
    evidence_gate_note,
    evidence_gate_summary,
    minimum_discordant_pairs,
    reaches_reporting_level,
    smallest_reachable_sign_test_p,
)
from llb.rag.fusion_evidence.paired import (
    discordant_deltas,
    discordant_pairs,
    evidence_gate_clause,
    gated_readings,
    paired_comparison,
    reading_of,
    separates,
    sign_test_p,
)
from llb.rag.fusion_evidence.stability import (
    LOOSER_CONFIDENCE,
    TIGHTER_CONFIDENCE,
)
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    bootstrap_index_sets,
    bootstrap_interval,
)


def test_the_bound_is_the_sign_tests_own_resolution_limit():
    """`2 * 0.5^d <= 1 - confidence`, evaluated -- no constant is introduced anywhere."""
    assert BOUND == 6
    assert smallest_reachable_sign_test_p(BOUND) <= 1.0 - DEFAULT_CONFIDENCE
    assert smallest_reachable_sign_test_p(BOUND - 1) > 1.0 - DEFAULT_CONFIDENCE
    # The same arithmetic the block already publishes beside its interval.
    assert sign_test_p(BOUND, 0) == pytest.approx(smallest_reachable_sign_test_p(BOUND))
    assert sign_test_p(BOUND - 1, 0) == pytest.approx(smallest_reachable_sign_test_p(BOUND - 1))


def test_a_tighter_convention_needs_more_items_and_a_looser_one_fewer():
    """The bound moves WITH the level, which is why a reading can be gated one convention up."""
    assert minimum_discordant_pairs(LOOSER_CONFIDENCE) == 5
    assert minimum_discordant_pairs(TIGHTER_CONFIDENCE) == 7
    with pytest.raises(ValueError, match="no discordant count"):
        minimum_discordant_pairs(1.0)


@pytest.mark.parametrize(
    ("discordant", "expected"),
    [
        (BOUND - 1, READING_INSUFFICIENT_EVIDENCE),
        (BOUND, READING_SEPARATED),
        (BOUND + 1, READING_SEPARATED),
    ],
)
def test_the_reading_flips_at_the_reachable_minimum(discordant: int, expected: str):
    row = _unanimous(discordant)
    assert row["delta"]["lo"] > 0.0  # every one of them clears zero
    assert discordant_pairs(row) == discordant
    assert reading_of(row) == expected
    assert row["stability"]["reading"] == expected
    assert separates(row) is (expected == READING_SEPARATED)


def test_the_gate_changes_the_reading_and_nothing_else():
    """The interval, the ledger, the point estimate, and the sign-test p are all untouched."""
    n = 40
    candidate = [1.0 if i < BOUND - 1 else 0.0 for i in range(n)]
    baseline = [0.0] * n
    index_sets = bootstrap_index_sets(n, RESAMPLES, SEED)
    row = paired_comparison(candidate, baseline, index_sets, DEFAULT_CONFIDENCE)
    deltas = [c - b for c, b in zip(candidate, baseline)]
    assert row["delta"] == bootstrap_interval(deltas, index_sets, DEFAULT_CONFIDENCE)
    assert (row["wins"], row["losses"], row["ties"]) == (BOUND - 1, 0, n - BOUND + 1)
    assert row["sign_test_p"] == sign_test_p(BOUND - 1, 0)
    assert row["stability"]["reading"] == READING_INSUFFICIENT_EVIDENCE
    assert row["stability"]["discordant"] == BOUND - 1


def test_a_row_at_the_bound_is_borderline_because_a_tighter_level_cannot_read_it():
    """The gate and the borderline flag are one scale: 6 items reach 95% and not 97.5%."""
    stability = _unanimous(BOUND)["stability"]
    assert stability["reading"] == READING_SEPARATED
    assert stability["tighter_reading"] == READING_INSUFFICIENT_EVIDENCE
    assert stability["borderline"] is True and stability["side"] == "above"


def test_only_a_claim_is_gated_never_the_null_reading():
    """A thin item set may say it found nothing; what it may not do is say it found something."""
    assert apply_evidence_gate(READING_FLAT, discordant=0) == READING_FLAT
    assert apply_evidence_gate(READING_SEPARATED, discordant=0) == READING_INSUFFICIENT_EVIDENCE
    assert apply_evidence_gate("neither", discordant=1, null_reading="neither") == "neither"
    assert apply_evidence_gate("answer", discordant=1, null_reading="neither") == (
        READING_INSUFFICIENT_EVIDENCE
    )
    # A row that does not clear zero reads `flat` however few items differ.
    assert reading_of(_unanimous(0)) == READING_FLAT


def test_the_two_discordant_counts_agree_on_the_same_data():
    """A lane holding delta vectors and one holding a ledger must gate identically."""
    for wins in (0, 1, BOUND - 1, BOUND, 12):
        row = _unanimous(wins)
        deltas = [1.0] * wins + [0.0] * (40 - wins)
        assert discordant_deltas(deltas) == discordant_pairs(row) == wins
        assert reaches_reporting_level(wins) is separates(row)


def test_a_report_states_how_many_readings_the_gate_relabeled():
    rows = [_unanimous(BOUND - 1), _unanimous(BOUND), _unanimous(0)]
    assert gated_readings(rows) == (1, 3)
    text = "\n".join(evidence_gate_summary(*gated_readings(rows)))
    assert "1 of 3" in text and "insufficient evidence" in text and str(BOUND) in text
    assert text.isascii()
    # Always stated, so a clean table is a statement rather than an absence.
    assert evidence_gate_summary(0, 3)
    assert evidence_gate_summary(0, 0) == []


def test_the_shared_clause_names_the_thin_rows_and_stays_quiet_otherwise():
    assert evidence_gate_note([]) == ""
    assert evidence_gate_note([("recall", BOUND, 40)]) == ""
    clause = evidence_gate_note([("recall", BOUND - 1, 40), ("mrr", 1, 40)])
    assert "INSUFFICIENT EVIDENCE" in clause and "`recall` differs on 5 of 40" in clause
    assert "2 of the rows" in clause
    # A row that never cleared zero is not "insufficient evidence", it is a measured non-result.
    assert evidence_gate_clause([("recall", _unanimous(0))]) == ""
    assert "INSUFFICIENT EVIDENCE" in evidence_gate_clause([("recall", _unanimous(BOUND - 1))])
