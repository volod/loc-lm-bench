"""Decoding stability: what an identical re-run does to a lane's own numbers.

The comparison's intervals resample the ITEM SET; none of them can see a lane that answers
differently the second time it is asked. These tests pin the band statistic over committed fixture
rows, so the spread is covered without a backend or a GPU.
"""

import json
from pathlib import Path

import pytest
from tests.llb.eval._context_ablation_helpers import (
    _gold_item,
    _lanes,
    _repeating_lane,
    _row,
    _types,
    _write_bundle,
)

from llb.core.config import RunConfig
from llb.eval.context_ablation import run_context_ablation
from llb.eval.context_ablation.compare import compare_context_strategies
from llb.eval.context_ablation.decoding_stability import measure_decoding_stability
from llb.eval.context_ablation.models import (
    DERIVED_RETRIEVAL_UPLIFT,
    LANE_CLOSED_BOOK,
    LANE_RAG,
    STABILITY_DRIFTS,
    STABILITY_REPRODUCIBLE,
)
from llb.eval.context_ablation.report import format_report

ITEMS = ("q1", "q2", "q3", "q4")


def _stability(closed_repeats, rag_repeats, derived=()):
    return measure_decoding_stability(
        {LANE_CLOSED_BOOK: closed_repeats, LANE_RAG: rag_repeats}, ITEMS, derived
    )


def _lane(*objectives: float, contaminated: tuple[str, ...] = (), answer: str = "a"):
    """One repeat of a lane: one row per fixture item, at the objectives given."""
    return [
        _row(
            item_id,
            objective,
            contains=1.0 if item_id in contaminated else 0.0,
            answer_preview=f"{answer}-{objective}",
        )
        for item_id, objective in zip(ITEMS, objectives, strict=True)
    ]


def test_the_repeat_groups_say_whether_a_lane_settled_or_never_repeated_itself():
    settles = [_lane(0.0, 0.0, 0.0, 0.0), _lane(1.0, 0.0, 0.0, 0.0), _lane(1.0, 0.0, 0.0, 0.0)]
    never = [_lane(0.0, 0.0, 0.0, 0.0), _lane(1.0, 0.0, 0.0, 0.0), _lane(1.0, 1.0, 0.0, 0.0)]

    assert _stability(settles, [_lane(1.0, 1.0, 1.0, 1.0)] * 3)["lanes"][LANE_CLOSED_BOOK][
        "outcome_groups"
    ] == [1, 2]
    assert _stability(never, [_lane(1.0, 1.0, 1.0, 1.0)] * 3)["lanes"][LANE_CLOSED_BOOK][
        "outcome_groups"
    ] == [1, 1, 1]


def test_a_lane_that_reproduces_exactly_has_a_zero_band():
    repeat = _lane(0.0, 0.2, 0.4, 0.6)

    report = _stability([repeat, repeat], [_lane(1.0, 1.0, 1.0, 1.0)] * 2)

    assert report["reading"] == STABILITY_REPRODUCIBLE
    assert report["baseline_floor"] == 0.0
    assert report["lanes"][LANE_CLOSED_BOOK]["divergent_items"] == 0
    assert report["lanes"][LANE_CLOSED_BOOK]["outcome_groups"] == [2]
    assert report["repeats"] == 2


def test_the_band_is_the_spread_of_the_lane_mean_quoted_against_the_first_repeat():
    first, second = _lane(0.0, 0.0, 0.0, 0.0), _lane(1.0, 1.0, 0.0, 0.0)

    report = _stability([first, second], [_lane(1.0, 1.0, 1.0, 1.0)] * 2)

    closed = report["lanes"][LANE_CLOSED_BOOK]
    assert closed["objective"] == {
        "base": 0.0,
        "min": 0.0,
        "max": 0.5,
        "mean": 0.25,
        "std": 0.25,
        "half_width": 0.25,
    }
    assert closed["divergent_items"] == 2
    assert closed["answer_divergent_items"] == 2
    assert report["reading"] == STABILITY_DRIFTS


def test_the_contamination_rate_carries_its_own_band():
    first = _lane(1.0, 1.0, 0.0, 0.0, contaminated=("q1", "q2"))
    second = _lane(1.0, 0.0, 0.0, 0.0, contaminated=("q1",))

    report = _stability([first, second], [_lane(1.0, 1.0, 1.0, 1.0)] * 2)

    match_rate = report["lanes"][LANE_CLOSED_BOOK]["match_rate"]
    assert (match_rate["base"], match_rate["min"], match_rate["max"]) == (0.5, 0.25, 0.5)
    assert match_rate["half_width"] == pytest.approx(0.125)


def test_the_ungrounded_band_is_read_against_the_grounded_lanes_own_band():
    closed = [_lane(0.0, 0.0, 0.0, 0.0), _lane(1.0, 1.0, 0.0, 0.0)]
    rag = [_lane(1.0, 1.0, 1.0, 1.0), _lane(1.0, 1.0, 1.0, 0.5)]

    report = _stability(closed, rag)

    assert report["grounded_floor"] == pytest.approx(0.0625)
    assert report["noise_multiple"] == pytest.approx(4.0)
    assert "4.0x the widest grounded band" in report["reason"]
    assert "the noisier measurement" in report["reason"]


def test_a_quieter_ungrounded_lane_is_reported_as_such_rather_than_as_a_small_multiple():
    """The premise is a hypothesis: a run that refutes it must say so, not print `0.2x`."""
    closed = [_lane(0.0, 0.0, 0.0, 0.0), _lane(1.0, 0.0, 0.0, 0.0)]
    rag = [_lane(1.0, 1.0, 1.0, 1.0), _lane(0.0, 0.0, 1.0, 1.0)]

    report = _stability(closed, rag)

    assert report["noise_multiple"] == pytest.approx(0.5)
    assert "is NOT the noisier measurement here" in report["reason"]


def test_a_grounded_lane_that_never_moves_leaves_no_multiple_to_state():
    closed = [_lane(0.0, 0.0, 0.0, 0.0), _lane(1.0, 0.0, 0.0, 0.0)]
    rag = [_lane(1.0, 1.0, 1.0, 1.0)] * 2

    report = _stability(closed, rag)

    assert report["noise_multiple"] is None
    assert "every grounded lane reproduced exactly" in report["reason"]


def test_a_derived_delta_is_floored_by_the_two_lanes_it_is_taken_over():
    lanes = _lanes(_lane(0.0, 0.0, 0.0, 0.0), _lane(1.0, 1.0, 1.0, 1.0))
    comparison = compare_context_strategies(lanes, _types(*ITEMS), resamples=20)
    closed = [_lane(0.0, 0.0, 0.0, 0.0), _lane(1.0, 1.0, 0.0, 0.0)]

    report = _stability(closed, [_lane(1.0, 1.0, 1.0, 1.0)] * 2, comparison["derived"])

    uplift = next(m for m in report["deltas"] if m["label"] == DERIVED_RETRIEVAL_UPLIFT)
    assert uplift["floor"] == pytest.approx(0.25)
    assert uplift["delta"] == pytest.approx(1.0)
    assert uplift["clears_floor"] is True
    assert uplift["floor_multiple"] == pytest.approx(4.0)


def test_a_delta_inside_the_decoding_floor_does_not_clear_it():
    lanes = _lanes(_lane(0.0, 0.0, 0.0, 0.0), _lane(0.0, 0.0, 0.0, 0.4))
    comparison = compare_context_strategies(lanes, _types(*ITEMS), resamples=20)
    closed = [_lane(0.0, 0.0, 0.0, 0.0), _lane(1.0, 1.0, 0.0, 0.0)]

    report = _stability(closed, [_lane(0.0, 0.0, 0.0, 0.4)] * 2, comparison["derived"])

    uplift = next(m for m in report["deltas"] if m["label"] == DERIVED_RETRIEVAL_UPLIFT)
    assert uplift["delta"] == pytest.approx(0.1)
    assert uplift["clears_floor"] is False


def test_a_delta_whose_lanes_were_not_both_repeated_is_absent_rather_than_floored_at_zero():
    lanes = _lanes(
        _lane(0.0, 0.0, 0.0, 0.0), _lane(1.0, 1.0, 1.0, 1.0), long_context=_lane(1.0, 1.0, 1.0, 1.0)
    )
    comparison = compare_context_strategies(lanes, _types(*ITEMS), resamples=20)

    report = _stability(
        [_lane(0.0, 0.0, 0.0, 0.0)] * 2, [_lane(1.0, 1.0, 1.0, 1.0)] * 2, comparison["derived"]
    )

    labels = {margin["label"] for margin in report["deltas"]}
    assert DERIVED_RETRIEVAL_UPLIFT in labels
    assert not any(label.startswith("long_context") for label in labels)


def test_one_pass_is_not_a_spread():
    with pytest.raises(ValueError, match="at least 2 repeats"):
        _stability([_lane(0.0, 0.0, 0.0, 0.0)], [_lane(1.0, 1.0, 1.0, 1.0)])


def test_lanes_repeated_a_different_number_of_times_are_not_comparable():
    with pytest.raises(ValueError, match="same number of times"):
        _stability([_lane(0.0, 0.0, 0.0, 0.0)] * 2, [_lane(1.0, 1.0, 1.0, 1.0)] * 3)


def test_a_repeat_that_scored_a_different_item_set_fails_loudly():
    short = [_row(item_id, 0.0) for item_id in ITEMS[:2]]
    with pytest.raises(ValueError, match="different item set"):
        _stability([_lane(0.0, 0.0, 0.0, 0.0), short], [_lane(1.0, 1.0, 1.0, 1.0)] * 2)


def test_repeated_lanes_land_in_the_persisted_artifact_and_the_report(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)

    run = run_context_ablation(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        [LANE_CLOSED_BOOK, LANE_RAG],
        out_dir=tmp_path / "context-ablation",
        resamples=20,
        repeats=3,
        run_lane=_repeating_lane(tmp_path, {LANE_CLOSED_BOOK: [0.0, 0.5, 0.0], LANE_RAG: [1.0]}),
    )

    stability = run.report["decoding_stability"]
    assert stability["repeats"] == 3
    assert stability["lanes"][LANE_CLOSED_BOOK]["objective"]["half_width"] == pytest.approx(0.25)
    assert stability["lanes"][LANE_RAG]["objective"]["half_width"] == 0.0
    assert len(stability["lanes"][LANE_CLOSED_BOOK]["run_dirs"]) == 3
    # The comparison itself is still the FIRST repeat, not a pooled average of the three.
    assert run.report["lanes"][LANE_CLOSED_BOOK]["overall"]["metrics"]["objective_score"][
        "mean"
    ] == pytest.approx(0.0)
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["decoding_stability"]["reading"] == STABILITY_DRIFTS
    body = Path(run.paths["report"]).read_text(encoding="utf-8")
    assert "How far a re-run moves each number" in body
    assert "identical repeats" in body
    # The quoted repeat is the transient one here (0.0, then 0.5, then 0.0 again is 1+1+1), so
    # the report must not claim it settled.
    assert "The quoted repeat is the odd one out" not in body


def test_the_report_says_when_the_quoted_repeat_is_the_one_that_never_came_back(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)

    run = run_context_ablation(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        [LANE_CLOSED_BOOK, LANE_RAG],
        out_dir=tmp_path / "context-ablation",
        resamples=20,
        repeats=3,
        run_lane=_repeating_lane(tmp_path, {LANE_CLOSED_BOOK: [0.0, 0.5, 0.5], LANE_RAG: [1.0]}),
    )

    assert run.report["decoding_stability"]["lanes"][LANE_CLOSED_BOOK]["outcome_groups"] == [1, 2]
    body = Path(run.paths["report"]).read_text(encoding="utf-8")
    assert "The quoted repeat is the odd one out" in body
    assert "`closed_book`" in body


def test_a_single_pass_leaves_the_artifact_exactly_as_it_was(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)

    run = run_context_ablation(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        [LANE_CLOSED_BOOK, LANE_RAG],
        out_dir=tmp_path / "context-ablation",
        resamples=20,
        run_lane=_repeating_lane(tmp_path, {LANE_CLOSED_BOOK: [0.0], LANE_RAG: [1.0]}),
    )

    assert "decoding_stability" not in run.report
    assert "How far a re-run moves each number" not in format_report(run.report)


def test_the_lane_is_re_scored_with_the_identical_config_on_the_identical_items(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    run_context_ablation(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        [LANE_CLOSED_BOOK, LANE_RAG],
        out_dir=tmp_path / "context-ablation",
        resamples=0,
        repeats=2,
        run_lane=_repeating_lane(tmp_path, {LANE_CLOSED_BOOK: [0.0], LANE_RAG: [1.0]}, seen),
    )

    assert len(seen) == 4
    assert seen[0] == seen[2] and seen[1] == seen[3]


def test_a_gold_item_is_still_scored_once_per_lane_without_repeats(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(_gold_item("q1").model_dump_json(exclude_none=True) + "\n", encoding="utf-8")
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    run_context_ablation(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        [LANE_CLOSED_BOOK, LANE_RAG],
        out_dir=tmp_path / "context-ablation",
        resamples=0,
        run_lane=_repeating_lane(tmp_path, {LANE_CLOSED_BOOK: [0.0], LANE_RAG: [1.0]}, seen),
    )

    assert len(seen) == 2
