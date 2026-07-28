"""adoption-borderline-annotation-on-the-other-paired-lanes -- one cut scale for every lane.

Every adopt-or-retain call in the repo is a `lo > 0` cut of a paired interval, so every one of them
can be produced by the convention rather than by the evidence. These tests cover the shared
annotation: that it falls out of the draw the interval already takes (so no lane pays for it), that
it rides on `paired_comparison` and therefore reaches all four lanes without per-lane wiring, and
that each lane's verdict sentence says when the row it names sits on the cut.

Pure: value vectors and dict reports, so the whole vertical runs in the lightweight CI install.
"""

from random import Random

import pytest

from llb.rag.fusion_evidence.evidence_gate import (
    READING_FLAT,
    READING_SEPARATED,
)
from llb.rag.fusion_evidence.stability import (
    LOOSER_CONFIDENCE,
    SIDE_ABOVE,
    SIDE_BELOW,
    TIGHTER_CONFIDENCE,
    borderline_note,
    boundary_table,
    decision_probability,
    exceedance,
    format_reading,
    stability_from_readings,
    unsettled,
)
from llb.rag.fusion_evidence.stats import (
    bootstrap_index_sets,
    bootstrap_interval,
    bootstrap_ratio,
    bootstrap_samples,
)
from llb.rag.fusion_evidence.paired import paired_comparison

RESAMPLES = 2000
SEED = 13


def _paired(wins: int, losses: int, n: int = 30, confidence: float = 0.95):
    """A paired row with `wins` items up, `losses` down, the rest tied -- a tunable near-miss."""
    candidate = [1.0 if i < wins else 0.0 for i in range(n)]
    baseline = [1.0 if wins <= i < wins + losses else 0.0 for i in range(n)]
    return paired_comparison(
        candidate, baseline, bootstrap_index_sets(n, RESAMPLES, SEED), confidence
    )


# --- the annotation is free, and it agrees with the interval it annotates ------------------


def test_exceedance_counts_the_same_draw_the_bounds_are_read_from():
    """The whole reason the annotation costs nothing: one draw answers both questions."""
    values = [1.0] * 8 + [-1.0] * 2 + [0.0] * 20
    index_sets = bootstrap_index_sets(len(values), RESAMPLES, SEED)
    samples = bootstrap_samples(values, index_sets)
    assert exceedance(samples) == sum(1 for s in samples if s > 0.0) / len(samples)


def test_the_decision_probability_is_exactly_the_interval_rule():
    """`lo > 0` at 95% is `p_positive > 0.975` -- the two readings must not drift apart."""
    assert decision_probability(0.95) == pytest.approx(0.975)
    assert decision_probability(0.90) == pytest.approx(0.95)


def test_the_persisted_reading_matches_the_interval_the_row_publishes():
    """A row whose delta clears zero must read `separated`, and one that does not must read `flat`."""
    for wins, losses in ((20, 0), (6, 1), (7, 1), (2, 8)):
        row = _paired(wins, losses)
        separated = row["delta"]["lo"] > 0.0
        expected = READING_SEPARATED if separated else READING_FLAT
        assert row["stability"]["reading"] == expected, (wins, losses)


@pytest.mark.parametrize("n", [1, 2, 5, 12, 35, 40, 82, 95])
@pytest.mark.parametrize("resamples", [0, 50, RESAMPLES])
@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.975, 0.99])
def test_the_interval_is_unchanged_by_carrying_the_annotation(
    n: int, resamples: int, confidence: float
):
    """Nothing recorded may move: the bounds must equal what the plain interval helper returns.

    `bootstrap_interval` is the pre-annotation code path, so this is a differential check against
    it -- swept across the shapes and settings the repo's recorded artifacts were produced at,
    because that is what says a recorded number could not have moved.
    """
    rng = Random(f"{SEED}:{n}:{resamples}:{confidence}")
    candidate = [rng.random() for _ in range(n)]
    baseline = [rng.random() for _ in range(n)]
    index_sets = bootstrap_index_sets(n, resamples, SEED)
    deltas = [c - b for c, b in zip(candidate, baseline)]
    row = paired_comparison(candidate, baseline, index_sets, confidence)
    assert row["delta"] == bootstrap_interval(deltas, index_sets, confidence)


def test_a_bootstrap_ratio_carries_the_same_cut_annotation_without_a_sign_test_gate():
    """Three positive events put a non-negative route ratio just below the default lower-bound cut."""
    n = 30
    estimate = bootstrap_ratio(
        [i < 3 for i in range(n)],
        [True] * n,
        bootstrap_index_sets(n, 5000, SEED),
    )
    stability = estimate["stability"]
    assert estimate == {**estimate, "mean": 0.1}
    assert stability["reading"] == READING_FLAT
    assert stability["looser_reading"] == READING_SEPARATED
    assert stability["borderline"] is True and stability["side"] == SIDE_BELOW
    assert "discordant" not in stability and "pairs" not in stability


# --- the flag is two-sided and discriminating ----------------------------------------------


def test_a_near_miss_negative_is_marked_below_the_cut():
    """The undecided negative: fails at 95%, would clear a 90% bar."""
    stability = _paired(6, 1)["stability"]
    assert decision_probability(LOOSER_CONFIDENCE) < stability["p_positive"]
    assert stability["p_positive"] < decision_probability(0.95)
    assert stability["reading"] == READING_FLAT
    assert stability["looser_reading"] == READING_SEPARATED
    assert stability["borderline"] is True and stability["side"] == SIDE_BELOW


def test_a_near_miss_positive_is_marked_above_the_cut():
    """The other half: it PASSES at 95% but a 97.5% interval would drop it."""
    stability = _paired(7, 1)["stability"]
    assert decision_probability(0.95) < stability["p_positive"]
    assert stability["p_positive"] < decision_probability(TIGHTER_CONFIDENCE)
    assert stability["reading"] == READING_SEPARATED
    assert stability["borderline"] is True and stability["side"] == SIDE_ABOVE


def test_a_decisive_row_and_a_flat_row_are_both_settled():
    assert _paired(20, 0)["stability"]["borderline"] is False
    assert _paired(0, 0)["stability"]["borderline"] is False


def test_no_draw_means_no_annotation_rather_than_a_confident_zero():
    """`p_positive` is a share OF resamples; a persisted 0.0 would read as a settled negative."""
    assert "stability" not in paired_comparison([1.0, 0.0], [0.0, 0.0], [])
    assert "stability" not in paired_comparison([], [], bootstrap_index_sets(0, 10, SEED))


def test_a_confidence_outside_the_two_conventions_carries_no_annotation():
    """The flag is defined against 90% / 97.5%; at 99% it would compare nothing meaningful."""
    row = paired_comparison(
        [1.0] * 10, [0.0] * 10, bootstrap_index_sets(10, 200, SEED), confidence=0.99
    )
    assert "stability" not in row
    assert row["delta"]["lo"] > 0.0  # the interval itself is still reported


# --- assembly, rendering, and the shared clause ---------------------------------------------


def test_stability_from_readings_accepts_a_richer_reading_than_separated_or_flat():
    """The adoption bar reads three states; the shared assembly must not assume two."""
    stability = stability_from_readings(
        reading="neither",
        looser_reading="answer",
        tighter_reading="neither",
        p_positive=0.969,
        discordant=12,
        pairs=40,
    )
    assert stability["borderline"] is True and stability["side"] == SIDE_BELOW
    assert stability["discordant"] == 12 and stability["pairs"] == 40


def test_unsettled_and_format_reading_only_mark_a_borderline_row():
    settled = _paired(20, 0)["stability"]
    marked = _paired(6, 1)["stability"]
    assert unsettled(settled) is None and unsettled(marked) is marked
    assert unsettled(None) is None
    assert format_reading(settled, READING_SEPARATED) == "separated"
    assert format_reading(marked, READING_FLAT) == "flat (borderline)"
    assert format_reading(None, "rank_only") == "rank only"


def test_the_shared_clause_is_empty_when_every_named_row_is_settled():
    assert borderline_note([("recall", _paired(20, 0)["stability"]), ("mrr", None)]) == ""


def test_the_shared_clause_names_the_row_the_side_and_the_level_that_would_flip_it():
    note = borderline_note([("recall_at_k", _paired(6, 1)["stability"])])
    assert "BORDERLINE" in note and "below the cut" not in note
    assert "`recall_at_k`" in note and "p_positive" in note
    assert "0.90 interval would read it `separated`" in note
    assert "too close to call" in note
    above = borderline_note([("mrr", _paired(7, 1)["stability"])])
    assert "0.975 interval would read it `flat`" in above


def test_the_clause_counts_the_rows_when_several_are_unsettled():
    note = borderline_note(
        [("recall_at_k", _paired(6, 1)["stability"]), ("mrr", _paired(7, 1)["stability"])]
    )
    assert "2 of the rows" in note


def test_the_boundary_table_is_ascii_and_empty_when_nothing_was_measured():
    rows = [("`bge-m3` recall_at_k", _paired(6, 1)["stability"])]
    lines = boundary_table(rows, title="T", key_header="row", subject="the candidate")
    text = "\n".join(lines)
    assert "### T" in text and "p_positive" in text and "NO (below)" in text
    assert "at 90%" in text and "at 97.5%" in text
    assert text.isascii()
    assert boundary_table([], title="T", key_header="row", subject="x") == []
