import pytest

from llb.scoring import text_analysis as ta
from llb.scoring import text_analysis_labels as ta_labels


def make_similarity(table):
    """Injected cosine: looks up (a, b) (order-insensitive); 0.0 otherwise."""

    def similarity(a, b):
        return table.get((a, b), table.get((b, a), 0.0))

    return similarity


ZERO_SIM = make_similarity({})


def label(label_id, value, kind=ta_labels.TOPIC, aliases=(), scoring=""):
    return ta_labels.PlantedLabel(
        label_id=label_id, kind=kind, value=value, aliases=tuple(aliases), scoring=scoring
    )


def test_normalize_surface():
    assert ta_labels.normalize_surface("  Київ,  Столиця.  ") == "київ, столиця"
    assert ta_labels.normalize_surface("«Україна»") == "україна"


def test_exact_match_full_credit():
    labels = [label("L1", "Київ"), label("L2", "Львів")]
    score = ta.score_subtask(["київ", "ЛЬВІВ"], labels, ZERO_SIM)
    assert score["f1"] == 1.0
    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert sorted(m[0] for m in score["matched"]) == ["L1", "L2"]


def test_alias_match():
    labels = [label("L1", "Володимир Зеленський", aliases=["Зеленський"])]
    score = ta.score_subtask(["зеленський"], labels, ZERO_SIM)
    assert score["recall"] == 1.0


def test_cosine_full_credit_band():
    sim = make_similarity({("прибуток зріс", "Зростання доходу"): 0.9})
    labels = [label("L1", "Зростання доходу", kind=ta_labels.TREND)]
    score = ta.score_subtask(["прибуток зріс"], labels, sim)
    assert score["matched"] == [("L1", 1.0)]
    assert score["f1"] == 1.0


def test_cosine_partial_credit_band():
    sim = make_similarity({("прибуток", "Зростання доходу"): 0.75})
    labels = [label("L1", "Зростання доходу", kind=ta_labels.TREND)]
    score = ta.score_subtask(["прибуток"], labels, sim)
    assert score["matched"] == [("L1", ta_labels.PARTIAL_CREDIT)]
    assert score["recall"] == ta_labels.PARTIAL_CREDIT
    assert score["precision"] == ta_labels.PARTIAL_CREDIT
    assert score["f1"] == ta_labels.PARTIAL_CREDIT


def test_below_partial_threshold_no_match():
    sim = make_similarity({("щось", "Зростання доходу"): 0.5})
    labels = [label("L1", "Зростання доходу", kind=ta_labels.TREND)]
    score = ta.score_subtask(["щось"], labels, sim)
    assert score["matched"] == []
    assert score["f1"] == 0.0


def test_greedy_one_to_one_assignment():
    # Two predictions both match the single label exactly; only one is credited, the other is
    # a false positive that lowers precision.
    labels = [label("L1", "Київ")]
    score = ta.score_subtask(["Київ", "київ"], labels, ZERO_SIM)
    assert score["recall"] == 1.0
    assert score["precision"] == 0.5
    assert round(score["f1"], 4) == 0.6667


def test_hallucinated_prediction_penalizes_precision():
    labels = [label("L1", "Київ")]
    score = ta.score_subtask(["Київ", "Марс"], labels, ZERO_SIM)
    assert score["recall"] == 1.0
    assert score["precision"] == 0.5
    assert score["n_pred"] == 2


def test_score_document_objective_headline_excludes_judged():
    labels = [
        label("T1", "економіка", kind=ta_labels.TOPIC),
        label("I1", "ринок перегрітий", kind=ta_labels.INSIGHT),
    ]
    preds = {ta_labels.TOPIC: ["економіка"], ta_labels.INSIGHT: ["ринок перегрітий"]}
    result = ta.score_document(preds, labels, ZERO_SIM)
    assert result["objective_score"] == 1.0  # only the topic sub-task feeds the headline
    assert result["n_objective_subtasks"] == 1
    assert set(result["subtasks"]) == {ta_labels.TOPIC, ta_labels.INSIGHT}
    assert result["subtasks"][ta_labels.INSIGHT]["objective"] is False


def test_contradiction_paired_spans_require_both_sides():
    label = ta_labels.PlantedLabel(
        label_id="c0",
        kind=ta_labels.CONTRADICTION,
        value="суперечність про дохід",
        attrs={"spans": ["дохід зріс", "дохід впав"]},
    )
    sim = lambda _a, _b: 0.0  # noqa: E731 - exact-surface only
    # a prediction naming BOTH contradicting sides earns full credit
    assert ta._credit("дохід зріс, але потім дохід впав", label, sim) == 1.0
    # naming only ONE side earns zero (min of the two side credits)
    assert ta._credit("дохід зріс", label, sim) == 0.0


def test_contradiction_without_spans_falls_back_to_surface():
    label = ta_labels.PlantedLabel(
        label_id="c1", kind=ta_labels.CONTRADICTION, value="внутрішня суперечність"
    )
    sim = lambda _a, _b: 0.0  # noqa: E731
    assert ta._credit("внутрішня суперечність", label, sim) == 1.0


def test_from_record_round_trip():
    record = {
        "label_id": "L1",
        "kind": ta_labels.TREND,
        "value": "Зростання доходу",
        "aliases": ["дохід зріс"],
        "attrs": {"subject": "дохід", "direction": "up"},
        "scoring": "objective",
    }
    planted = ta_labels.PlantedLabel.from_record(record)
    assert planted.surfaces == ("Зростання доходу", "дохід зріс")
    assert planted.attrs["direction"] == "up"
    assert planted.is_objective is True


def test_load_planted_labels_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown text-analysis label kind"):
        ta_labels.load_planted_labels([{"label_id": "x", "kind": "bogus", "value": "v"}])


def test_judged_kind_objective_floor_still_scored():
    labels = [label("N1", "піднесення і занепад", kind=ta_labels.NARRATIVE)]
    score = ta.score_subtask(["піднесення і занепад"], labels, ZERO_SIM)
    assert score["objective"] is False
    assert score["recall"] == 1.0  # objective floor is computed even for judged kinds


def trend_label(label_id, value, direction, aliases=()):
    return ta_labels.PlantedLabel(
        label_id=label_id,
        kind=ta_labels.TREND,
        value=value,
        aliases=tuple(aliases),
        attrs={"direction": direction},
    )


def test_direction_of_lexicon():
    assert ta_labels.direction_of("частка ВДЕ зросла") == ta_labels.DIRECTION_UP
    assert ta_labels.direction_of("ціни впали") == ta_labels.DIRECTION_DOWN
    assert ta_labels.direction_of("показник стабільний") == ta_labels.DIRECTION_FLAT
    assert ta_labels.direction_of("просто текст") is None


def test_trend_direction_conflict_zeroes_surface_match():
    # Surface exact-matches the label value, but the prediction states the WRONG direction.
    labels = [trend_label("L1", "частка ВДЕ зросла", "up")]
    conflict = ta.score_subtask(["частка ВДЕ впала"], labels, ZERO_SIM)
    assert conflict["recall"] == 0.0  # wrong direction -> credit zeroed -> label unrecovered
    assert conflict["precision"] == 0.0  # the prediction is now an unmatched false positive


def test_trend_direction_agreement_keeps_exact_credit():
    labels = [trend_label("L1", "частка ВДЕ зросла", "up")]
    ok = ta.score_subtask(["частка ВДЕ зросла"], labels, ZERO_SIM)
    assert ok["recall"] == 1.0  # right subject + right direction -> full credit


def test_trend_no_detectable_direction_keeps_surface_credit():
    labels = [trend_label("L1", "інфляція", "up")]
    # prediction surface-matches but carries no direction word -> we cannot penalize it.
    keep = ta.score_subtask(["інфляція"], labels, ZERO_SIM)
    assert keep["recall"] == 1.0
