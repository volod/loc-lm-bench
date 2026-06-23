import pytest

from llb.scoring import text_analysis as ta


def make_similarity(table):
    """Injected cosine: looks up (a, b) (order-insensitive); 0.0 otherwise."""

    def similarity(a, b):
        return table.get((a, b), table.get((b, a), 0.0))

    return similarity


ZERO_SIM = make_similarity({})


def label(label_id, value, kind=ta.TOPIC, aliases=(), scoring=""):
    return ta.PlantedLabel(
        label_id=label_id, kind=kind, value=value, aliases=tuple(aliases), scoring=scoring
    )


def test_normalize_surface():
    assert ta.normalize_surface("  Київ,  Столиця.  ") == "київ, столиця"
    assert ta.normalize_surface("«Україна»") == "україна"


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
    labels = [label("L1", "Зростання доходу", kind=ta.TREND)]
    score = ta.score_subtask(["прибуток зріс"], labels, sim)
    assert score["matched"] == [("L1", 1.0)]
    assert score["f1"] == 1.0


def test_cosine_partial_credit_band():
    sim = make_similarity({("прибуток", "Зростання доходу"): 0.75})
    labels = [label("L1", "Зростання доходу", kind=ta.TREND)]
    score = ta.score_subtask(["прибуток"], labels, sim)
    assert score["matched"] == [("L1", ta.PARTIAL_CREDIT)]
    assert score["recall"] == ta.PARTIAL_CREDIT
    assert score["precision"] == ta.PARTIAL_CREDIT
    assert score["f1"] == ta.PARTIAL_CREDIT


def test_below_partial_threshold_no_match():
    sim = make_similarity({("щось", "Зростання доходу"): 0.5})
    labels = [label("L1", "Зростання доходу", kind=ta.TREND)]
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
        label("T1", "економіка", kind=ta.TOPIC),
        label("I1", "ринок перегрітий", kind=ta.INSIGHT),
    ]
    preds = {ta.TOPIC: ["економіка"], ta.INSIGHT: ["ринок перегрітий"]}
    result = ta.score_document(preds, labels, ZERO_SIM)
    assert result["objective_score"] == 1.0  # only the topic sub-task feeds the headline
    assert result["n_objective_subtasks"] == 1
    assert set(result["subtasks"]) == {ta.TOPIC, ta.INSIGHT}
    assert result["subtasks"][ta.INSIGHT]["objective"] is False


def test_from_record_round_trip():
    record = {
        "label_id": "L1",
        "kind": ta.TREND,
        "value": "Зростання доходу",
        "aliases": ["дохід зріс"],
        "attrs": {"subject": "дохід", "direction": "up"},
        "scoring": "objective",
    }
    planted = ta.PlantedLabel.from_record(record)
    assert planted.surfaces == ("Зростання доходу", "дохід зріс")
    assert planted.attrs["direction"] == "up"
    assert planted.is_objective is True


def test_load_planted_labels_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown text-analysis label kind"):
        ta.load_planted_labels([{"label_id": "x", "kind": "bogus", "value": "v"}])


def test_judged_kind_objective_floor_still_scored():
    labels = [label("N1", "піднесення і занепад", kind=ta.NARRATIVE)]
    score = ta.score_subtask(["піднесення і занепад"], labels, ZERO_SIM)
    assert score["objective"] is False
    assert score["recall"] == 1.0  # objective floor is computed even for judged kinds
