"""Focused tests split from ``test_embedder_adoption_screen.py``."""

import pytest
from _embedder_adoption_screen_helpers import (
    FOCUS,
    _deltas,
    _index_sets,
    _screen,
)

from llb.eval.embedder_adoption.screen_models import (
    DECISION_FULL_SET_REQUIRED,
    DECISION_SCREEN_SUPPORTED,
)
from llb.eval.embedder_adoption.cross_model import (
    READING_ANSWER,
    READING_NEITHER,
    READING_RANK_ONLY,
)
from llb.eval.embedder_adoption.screen import (
    decide_screen,
    screen_model,
)
from llb.eval.embedder_adoption.stability import reading_from_deltas
from llb.eval.embedder_adoption.models import ItemDeltas


def test_reading_from_deltas_matches_the_cell_reading_order():
    """Objective first, then first-hit rank -- the same order `cell_reading` uses."""
    n = 20
    both = _deltas([0.5] * n, [0.5] * n)
    assert reading_from_deltas(both, _index_sets(n)) == READING_ANSWER
    rank_only = _deltas([0.0] * n, [0.5] * n)
    assert reading_from_deltas(rank_only, _index_sets(n)) == READING_RANK_ONLY
    flat = _deltas([0.0] * n, [0.0] * n)
    assert reading_from_deltas(flat, _index_sets(n)) == READING_NEITHER


def test_misaligned_delta_vectors_are_refused():
    with pytest.raises(ValueError, match="must align"):
        ItemDeltas(item_ids=["q0", "q1"], objective=[0.1], reciprocal_rank=[0.1, 0.2])


def test_take_restricts_every_vector_together():
    deltas = _deltas([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    subset = deltas.take([2, 0])
    assert subset.item_ids == ["q2", "q0"]
    assert subset.objective == pytest.approx([0.3, 0.1])
    assert subset.reciprocal_rank == pytest.approx([0.6, 0.4])


def test_a_delta_whose_interval_spans_zero_is_not_an_answer():
    n = 20
    marginal = _deltas([1.0] + [0.0] * (n - 1))
    assert reading_from_deltas(marginal, _index_sets(n)) != READING_ANSWER


def test_a_strong_effect_is_reproduced_by_small_screens():
    """A wide margin survives subsampling, so a screen is cheap for an obvious effect."""
    n = 40
    screen = screen_model(
        "strong",
        _deltas([0.6] * n, [0.6] * n),
        READING_ANSWER,
        sizes=(10, 20),
        draws=20,
        resamples=200,
    )
    assert screen["full_reading"] == READING_ANSWER
    assert screen["reproduced"] is True
    assert all(entry["agreement"] == 1.0 for entry in screen["sizes"])
    assert screen["min_size"] == 10


def test_a_marginal_effect_is_lost_by_small_screens():
    """The case the study exists to catch: the full set detects it, a screen does not."""
    # 12 wins of +0.3 against 28 zeros: the full-set interval clears zero, subsets often do not.
    deltas = _deltas([0.3] * 12 + [0.0] * 28)
    screen = screen_model(
        "marginal", deltas, READING_ANSWER, sizes=(10, 15), draws=40, resamples=200, target=0.9
    )
    assert screen["full_reading"] == READING_ANSWER
    assert min(entry["agreement"] for entry in screen["sizes"]) < 0.9
    assert screen["min_size"] is None


def test_sizes_at_or_above_the_full_set_are_not_measured():
    n = 12
    screen = screen_model(
        "m", _deltas([0.5] * n), READING_ANSWER, sizes=(10, 12, 20), draws=10, resamples=100
    )
    assert [entry["size"] for entry in screen["sizes"]] == [10]


def test_the_agreement_curve_is_deterministic_at_a_fixed_seed():
    kwargs = dict(sizes=(10,), draws=25, resamples=150)
    deltas = _deltas([0.3] * 10 + [0.0] * 20)
    a = screen_model("m", deltas, READING_ANSWER, seed=7, **kwargs)
    b = screen_model("m", deltas, READING_ANSWER, seed=7, **kwargs)
    assert a["sizes"] == b["sizes"]


def test_a_screen_is_claimed_only_when_every_model_survives_it():
    verdict = decide_screen(
        [_screen("a", 15), _screen("b", 25)],
        focus_cell=FOCUS,
        target=0.9,
        bundles_full_grid=24,
        bundles_focus_cell=6,
    )
    assert verdict["decision"] == DECISION_SCREEN_SUPPORTED
    assert verdict["min_size"] == 25  # the WORST model sets the screen size
    assert "25 of 40 items" in verdict["reason"]


def test_one_model_losing_its_reading_blocks_the_item_saving():
    """A screen that reproduces three models and loses the fourth is the dangerous one."""
    verdict = decide_screen(
        [_screen("a", 15), _screen("b", None)],
        focus_cell=FOCUS,
        target=0.9,
        bundles_full_grid=24,
        bundles_focus_cell=6,
    )
    assert verdict["decision"] == DECISION_FULL_SET_REQUIRED
    assert verdict["min_size"] is None
    assert "`b`" in verdict["reason"]


def test_the_cell_saving_is_reported_either_way():
    """Dropping the other cells is exact and unconditional, so both verdicts must state it."""
    for models in ([_screen("a", 15)], [_screen("a", None)]):
        verdict = decide_screen(
            models, focus_cell=FOCUS, target=0.9, bundles_full_grid=24, bundles_focus_cell=6
        )
        assert "24 to 6 run-eval bundles (4x)" in verdict["reason"]
