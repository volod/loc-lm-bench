"""embedder-adoption-bar-reranker-model-dependence -- is the reranker gain predictable in advance?

Pure and file-driven: the input is N finished `AdoptionBarReport`s plus a declared profile per
model, so the whole roster reading is unit-tested with dict reports -- no backend, store, or GPU.
"""

import json
from pathlib import Path

import pytest

from llb.eval.embedder_adoption import (
    DECISION_INSUFFICIENT_VARIATION,
    DECISION_NO_PROPERTY_PREDICTS,
    DECISION_PROPERTY_PREDICTS,
    compare_cells,
    compare_roster,
    format_roster,
    format_roster_summary,
    load_profiles,
    run_roster_comparison,
)
from llb.eval.embedder_adoption.models import CellSpec
from llb.eval.embedder_adoption.roster import PROPERTY_FAMILY, PROPERTY_PARAMS

BASELINE = "intfloat/multilingual-e5-base"
CANDIDATE = "BAAI/bge-m3"
FOCUS = "k10+rerank"
OTHER = "k10"


def _row(item_id: str, objective: float, rank: int | None = 1, hit: float = 1.0) -> dict:
    return {
        "item_id": item_id,
        "split": "final",
        "status": "ok",
        "objective_score": objective,
        "token_f1": objective,
        "exact": 0.0,
        "contains": 1.0 if objective > 0 else 0.0,
        "retrieval_hit": hit,
        "first_hit_rank": rank,
    }


def _ids(n: int = 12) -> list[str]:
    return [f"q{i}" for i in range(n)]


def _answer_pair(ids):
    """Candidate answers better AND ranks earlier -> READING_ANSWER."""
    return ([_row(i, 0.0, rank=None, hit=0.0) for i in ids], [_row(i, 1.0, rank=1) for i in ids])


def _rank_only_pair(ids):
    """Candidate ranks earlier, answers identically -> READING_RANK_ONLY."""
    return ([_row(i, 1.0, rank=3) for i in ids], [_row(i, 1.0, rank=1) for i in ids])


def _report(*, model: str, focus_answers: bool, ids=None):
    """A finished sweep whose FOCUS cell either reaches the answer or is rank-only."""
    ids = ids or _ids()
    focus = _answer_pair(ids) if focus_answers else _rank_only_pair(ids)
    other = _rank_only_pair(ids)
    report = compare_cells(
        [
            (CellSpec(10, None), {BASELINE: other[0], CANDIDATE: other[1]}),
            (CellSpec(10, "x"), {BASELINE: focus[0], CANDIDATE: focus[1]}),
        ],
        {},
        baseline=BASELINE,
        candidate=CANDIDATE,
        resamples=200,
    )
    report["metadata"] = {"model": model, "goldset": "gs.jsonl", "corpus": "c", "split": "final"}
    return report


def _roster(spec: list[tuple[str, bool]], profiles=None, **kwargs):
    return compare_roster(
        [_report(model=m, focus_answers=a) for m, a in spec],
        profiles,
        focus_cell=kwargs.pop("focus_cell", FOCUS),
        **kwargs,
    )


# --- per-cell roster readings -------------------------------------------------------------


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


# --- the property separation test ---------------------------------------------------------


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


# --- guards -------------------------------------------------------------------------------


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


# --- profiles + reporting -----------------------------------------------------------------


def test_profiles_are_declared_and_validated(tmp_path: Path):
    good = tmp_path / "profiles.json"
    good.write_text(json.dumps({"a": {"params_b": 27, "family": "gemma"}}), encoding="utf-8")
    assert load_profiles(good) == {"a": {"params_b": 27, "family": "gemma"}}
    assert load_profiles(None) == {}

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"a": {"parameters": 27}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown propert"):
        load_profiles(bad)

    not_object = tmp_path / "list.json"
    not_object.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="keyed by model id"):
        load_profiles(not_object)


def test_roster_report_renders_ascii_with_readings_properties_and_separations():
    roster = _roster(
        [("big-a", True), ("big-b", True), ("small-a", False), ("small-b", False)],
        {
            "big-a": {"params_b": 27, "family": "gemma"},
            "big-b": {"params_b": 24, "family": "gemma"},
            "small-a": {"params_b": 12, "family": "qwen"},
            "small-b": {"params_b": 8, "family": "qwen"},
        },
    )
    text = format_roster(roster, metadata={"goldset": "gs.jsonl"})
    assert "# Embedder adoption bar" in text
    assert "### Per-cell reading by model" in text
    assert "### Declared model properties" in text
    assert "### Property separation" in text
    assert DECISION_PROPERTY_PREDICTS in text
    assert text.isascii()
    assert format_roster_summary(roster).isascii()


def test_roster_round_trips_through_disk(tmp_path: Path):
    dirs = []
    for name, answers in (("a", True), ("b", False), ("c", False)):
        d = tmp_path / name
        d.mkdir()
        (d / "comparison.json").write_text(
            json.dumps(_report(model=name, focus_answers=answers), ensure_ascii=False),
            encoding="utf-8",
        )
        dirs.append(d)
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps({"a": {"params_b": 27}, "b": {"params_b": 12}, "c": {"params_b": 8}}),
        encoding="utf-8",
    )

    run = run_roster_comparison(
        dirs, out_dir=tmp_path / "roster", profiles_path=profiles, focus_cell=FOCUS
    )
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["verdict"]["decision"] == DECISION_PROPERTY_PREDICTS
    assert persisted["metadata"]["goldset"] == "gs.jsonl"
    assert Path(run.paths["report"]).read_text(encoding="utf-8").startswith("# Embedder adoption")
