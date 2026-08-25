"""The ablation read one question type at a time (context-ablation-question-type-slices)."""

import pytest
from tests.llb.eval._context_ablation_helpers import (
    _derived,
    _lanes,
    _row,
    _types,
)

from llb.eval import common as eval_common
from llb.eval.context_ablation import (
    compare_context_strategies,
    format_report,
)
from llb.eval.context_ablation.models import (
    DERIVED_LONG_CONTEXT_DELTA_FITTING,
    DERIVED_RETRIEVAL_UPLIFT,
    LANE_LONG_CONTEXT,
    VERDICT_LONG_CONTEXT_WINS,
    VERDICT_NO_RETRIEVAL_GAIN,
    VERDICT_RAG_PAYS_OFF,
)

FACTOID = "factoid"
MULTI_HOP = "multi-hop"


def _split_types(ids: list[str], first: int) -> dict[str, str]:
    """The first `first` items are factoid, the rest multi-hop."""
    return {item_id: (FACTOID if i < first else MULTI_HOP) for i, item_id in enumerate(ids)}


def _reading(report, name):
    return next(entry for entry in report["slice_readings"] if entry["slice"] == name)


def _slice_derived(report, name, label):
    return next(entry for entry in _reading(report, name)["derived"] if entry["label"] == label)


def _uneven_report(**kwargs):
    """Retrieval pays for factoids on this set and does nothing at all for multi-hop items."""
    ids = [f"q{i}" for i in range(16)]
    types = _split_types(ids, 8)
    closed = [_row(item_id, 0.1, hit=0.0) for item_id in ids]
    rag = [_row(item_id, 0.9 if types[item_id] == FACTOID else 0.1) for item_id in ids]
    return compare_context_strategies(_lanes(closed, rag), types, resamples=200, **kwargs)


def test_a_pooled_uplift_hides_the_slice_it_was_not_paid_on():
    report = _uneven_report()
    assert _derived(report, DERIVED_RETRIEVAL_UPLIFT)["paired"]["delta"]["mean"] == pytest.approx(
        0.4
    )
    factoid = _slice_derived(report, FACTOID, DERIVED_RETRIEVAL_UPLIFT)
    multi_hop = _slice_derived(report, MULTI_HOP, DERIVED_RETRIEVAL_UPLIFT)
    assert (factoid["n"], multi_hop["n"]) == (8, 8)
    assert factoid["paired"]["delta"]["mean"] == pytest.approx(0.8)
    assert multi_hop["paired"]["delta"]["mean"] == pytest.approx(0.0)
    assert _reading(report, FACTOID)["verdict"]["decision"] == VERDICT_RAG_PAYS_OFF
    assert _reading(report, MULTI_HOP)["verdict"]["decision"] == VERDICT_NO_RETRIEVAL_GAIN


def test_every_slice_is_judged_by_the_same_cut_as_the_pooled_verdict():
    """A slice where retrieval pays reaches the pooled decision on its own items alone."""
    ids = [f"q{i}" for i in range(12)]
    types = _split_types(ids, 6)
    report = compare_context_strategies(
        _lanes([_row(i, 0.0, hit=0.0) for i in ids], [_row(i, 1.0) for i in ids]),
        types,
        resamples=200,
    )
    assert report["verdict"]["decision"] == VERDICT_RAG_PAYS_OFF
    for name in (FACTOID, MULTI_HOP):
        assert _reading(report, name)["verdict"]["decision"] == VERDICT_RAG_PAYS_OFF


def test_a_slice_carries_its_own_contamination_rate_not_the_run_wide_one():
    ids = [f"q{i}" for i in range(8)]
    types = _split_types(ids, 4)
    closed = [
        _row(item_id, 0.1, contains=1.0 if types[item_id] == FACTOID else 0.0) for item_id in ids
    ]
    report = compare_context_strategies(
        _lanes(closed, [_row(i, 0.6) for i in ids]), types, resamples=200
    )
    assert report["contamination"]["n_contaminated"] == 4
    assert _reading(report, FACTOID)["contamination"]["n_contaminated"] == 4
    assert _reading(report, MULTI_HOP)["contamination"]["n_contaminated"] == 0
    assert _reading(report, MULTI_HOP)["verdict"]["contamination_rate"] == pytest.approx(0.0)


def test_a_slice_the_document_lane_skipped_entirely_is_not_measurable_rather_than_zero():
    """A lane that skipped every item of a question type compared nothing on that slice."""
    ids = [f"q{i}" for i in range(12)]
    types = _split_types(ids, 6)
    long_rows = [
        _row(item_id, 1.0)
        if types[item_id] == FACTOID
        else _row(item_id, 0.0, hit=0.0, status=eval_common.CONTEXT_OVERFLOW)
        for item_id in ids
    ]
    report = compare_context_strategies(
        _lanes([_row(i, 0.0, hit=0.0) for i in ids], [_row(i, 0.5) for i in ids], long_rows),
        types,
        resamples=200,
    )
    multi_hop = _reading(report, MULTI_HOP)
    assert multi_hop["verdict"]["skipped"][LANE_LONG_CONTEXT] == 6
    assert _slice_derived(report, MULTI_HOP, DERIVED_LONG_CONTEXT_DELTA_FITTING)["n"] == 0
    assert multi_hop["verdict"]["decision"] != VERDICT_LONG_CONTEXT_WINS
    assert "not measurable" in format_report(report)


def test_a_slice_no_lane_skipped_carries_no_fitting_cut_of_its_own():
    """The fitting cut is scoped to the slice: an untouched slice has nothing to cut."""
    ids = [f"q{i}" for i in range(12)]
    types = _split_types(ids, 6)
    long_rows = [_row(item_id, 1.0) for item_id in ids]
    long_rows[-1] = _row(ids[-1], 0.0, hit=0.0, status=eval_common.CONTEXT_OVERFLOW)
    report = compare_context_strategies(
        _lanes([_row(i, 0.0, hit=0.0) for i in ids], [_row(i, 0.5) for i in ids], long_rows),
        types,
        resamples=200,
    )
    labels = [entry["label"] for entry in _reading(report, FACTOID)["derived"]]
    assert DERIVED_LONG_CONTEXT_DELTA_FITTING not in labels
    multi_hop_labels = [entry["label"] for entry in _reading(report, MULTI_HOP)["derived"]]
    assert DERIVED_LONG_CONTEXT_DELTA_FITTING in multi_hop_labels
    assert _slice_derived(report, MULTI_HOP, DERIVED_LONG_CONTEXT_DELTA_FITTING)["n"] == 5


def test_a_slice_reading_never_carries_the_shippable_adoption_call():
    """Adoption is a per-corpus decision; a dozen items of one question type cannot make it."""
    report = _uneven_report()
    assert "retrieved_document" in report["verdict"]
    for reading in report["slice_readings"]:
        assert "retrieved_document" not in reading["verdict"]


def test_the_report_names_every_slice_its_n_and_the_reading_it_reached():
    text = format_report(_uneven_report())
    assert "### Per-slice reading" in text
    assert f"| {FACTOID} | 8 |" in text
    assert f"| {MULTI_HOP} | 8 |" in text
    assert VERDICT_NO_RETRIEVAL_GAIN in text
    assert f"#### Derived numbers -- {MULTI_HOP}" in text


def test_a_gold_set_with_no_sidecar_says_so_instead_of_reporting_an_empty_slice():
    ids = [f"q{i}" for i in range(6)]
    report = compare_context_strategies(
        _lanes([_row(i, 0.0, hit=0.0) for i in ids], [_row(i, 0.6) for i in ids]),
        {},
        resamples=200,
    )
    assert report["slice_readings"] == []
    assert "ships no `needle_items.jsonl`" in format_report(report)


def test_slicing_does_not_move_the_pooled_numbers_it_was_added_beside():
    """The pooled table is the same table it was before the slices existed."""
    ids = [f"q{i}" for i in range(10)]
    lanes = _lanes([_row(i, 0.0, hit=0.0) for i in ids], [_row(i, 0.7) for i in ids])
    pooled = compare_context_strategies(lanes, _types(*ids), resamples=200)
    sliced = compare_context_strategies(lanes, _split_types(ids, 5), resamples=200)
    assert pooled["derived"] == sliced["derived"]
    assert pooled["verdict"]["decision"] == sliced["verdict"]["decision"]
