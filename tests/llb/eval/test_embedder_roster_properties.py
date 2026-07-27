"""Focused tests split from ``test_embedder_adoption_roster.py``."""

import pytest
from _embedder_adoption_roster_helpers import (
    FOCUS,
    OTHER,
    _report,
    _roster,
)

from llb.eval.embedder_adoption.roster_models import (
    DECISION_INSUFFICIENT_VARIATION,
    DECISION_NO_PROPERTY_PREDICTS,
    DECISION_PROPERTY_PREDICTS,
)
from llb.eval.embedder_adoption.roster import compare_roster
from llb.eval.embedder_adoption.roster_models import (
    PROPERTY_FAMILY,
    PROPERTY_PARAMS,
)


def test_the_roster_records_every_models_reading_per_cell():
    roster = _roster([("a", True), ("b", False), ("c", False)])
    assert roster["models"] == ["a", "b", "c"]
    focus = next(c for c in roster["cells"] if c["label"] == FOCUS)
    assert focus["answer_models"] == ["a"]
    assert focus["unanimous"] is False
    other = next(c for c in roster["cells"] if c["label"] == OTHER)
    assert other["unanimous"] is True  # every model reads k10 as rank_only
    assert roster["verdicts"] == {m: "extend_bar" for m in ("a",)} | {
        m: "keep_bar" for m in ("b", "c")
    }


def test_a_parameter_count_threshold_that_separates_is_named_with_its_chance_probability():
    """The headline claim -- and it must carry how easily 4 models split by luck."""
    roster = _roster(
        [("big-a", True), ("big-b", True), ("small-a", False), ("small-b", False)],
        {
            "big-a": {"params_b": 27, "family": "gemma"},
            "big-b": {"params_b": 24, "family": "mistral"},
            "small-a": {"params_b": 12, "family": "gemma"},
            "small-b": {"params_b": 8, "family": "llama"},
        },
    )
    verdict = roster["verdict"]
    assert verdict["decision"] == DECISION_PROPERTY_PREDICTS
    params = next(s for s in verdict["separations"] if s["property"] == PROPERTY_PARAMS)
    assert params["separates"] is True
    assert "separates upward" in params["reason"]
    # 2 / C(4,2) = 2/6
    assert params["chance_probability"] == pytest.approx(2 / 6)
    assert "0.33" in params["reason"]
    # family is shared across the split (gemma on both sides), so it must NOT separate
    family = next(s for s in verdict["separations"] if s["property"] == PROPERTY_FAMILY)
    assert family["separates"] is False
    assert "shared across the split" in family["reason"]


def test_an_overlapping_parameter_count_does_not_predict():
    """A capturing model on either side of a non-capturing one: no threshold can split them."""
    roster = _roster(
        [("a", True), ("b", False), ("c", True)],
        {
            # `family` is deliberately shared across the split too, so this isolates `params_b`.
            "a": {"params_b": 27, "family": "gemma"},
            "b": {"params_b": 24, "family": "gemma"},
            "c": {"params_b": 12, "family": "qwen"},
        },
    )
    verdict = roster["verdict"]
    assert verdict["decision"] == DECISION_NO_PROPERTY_PREDICTS
    params = next(s for s in verdict["separations"] if s["property"] == PROPERTY_PARAMS)
    assert params["separates"] is False
    assert "overlaps across the split" in params["reason"]


def test_a_family_that_groups_models_separates_but_one_family_per_model_does_not():
    """A property has to GROUP models to predict anything about a new one."""
    grouped = _roster(
        [("a", True), ("b", True), ("c", False), ("d", False)],
        {
            "a": {"family": "gemma"},
            "b": {"family": "gemma"},
            "c": {"family": "qwen"},
            "d": {"family": "qwen"},
        },
    )
    family = next(s for s in grouped["verdict"]["separations"] if s["property"] == PROPERTY_FAMILY)
    assert family["separates"] is True
    assert grouped["verdict"]["decision"] == DECISION_PROPERTY_PREDICTS

    unique = _roster(
        [("a", True), ("b", False), ("c", False)],
        {"a": {"family": "gemma"}, "b": {"family": "qwen"}, "c": {"family": "mistral"}},
    )
    family = next(s for s in unique["verdict"]["separations"] if s["property"] == PROPERTY_FAMILY)
    assert family["separates"] is False
    assert "only restates the model list" in family["reason"]
    assert unique["verdict"]["decision"] == DECISION_NO_PROPERTY_PREDICTS


def test_an_undeclared_property_is_reported_not_silently_skipped():
    roster = _roster(
        [("a", True), ("b", False), ("c", False)],
        {"a": {"params_b": 27}, "b": {"params_b": 12}},  # c has no profile at all
    )
    params = next(s for s in roster["verdict"]["separations"] if s["property"] == PROPERTY_PARAMS)
    assert params["missing"] == ["c"]
    family = next(s for s in roster["verdict"]["separations"] if s["property"] == PROPERTY_FAMILY)
    assert family["separates"] is False
    assert "undeclared" in family["reason"]
    assert sorted(family["missing"]) == ["a", "b", "c"]


def test_a_roster_with_no_profiles_still_reads_the_cells_and_says_nothing_predicts():
    roster = _roster([("a", True), ("b", False), ("c", False)])
    assert roster["verdict"]["decision"] == DECISION_NO_PROPERTY_PREDICTS
    assert all(not s["separates"] for s in roster["verdict"]["separations"])


def test_a_unanimous_focus_cell_has_no_split_to_explain():
    """Vacuous separation is refused: with no split every property would 'predict' it."""
    for answers in (True, False):
        roster = _roster(
            [("a", answers), ("b", answers), ("c", answers)],
            {"a": {"params_b": 27}, "b": {"params_b": 12}, "c": {"params_b": 8}},
        )
        verdict = roster["verdict"]
        assert verdict["decision"] == DECISION_INSUFFICIENT_VARIATION
        assert verdict["separations"] == []
        assert "no split" in verdict["reason"]


def test_a_roster_needs_at_least_three_sweeps():
    with pytest.raises(ValueError, match="at least three"):
        _roster([("a", True), ("b", False)])


def test_the_same_model_twice_is_refused():
    with pytest.raises(ValueError, match="same model more than once"):
        _roster([("a", True), ("a", False), ("b", False)])


def test_incomparable_sweeps_are_refused():
    reports = [_report(model=m, focus_answers=False) for m in ("a", "b", "c")]
    reports[2]["seed"] = reports[0]["seed"] + 1
    with pytest.raises(ValueError, match="different bootstrap seeds"):
        compare_roster(reports, {}, focus_cell=FOCUS)


def test_an_unknown_focus_cell_is_refused():
    with pytest.raises(ValueError, match="not in the swept grid"):
        _roster([("a", True), ("b", False), ("c", False)], focus_cell="k99")
