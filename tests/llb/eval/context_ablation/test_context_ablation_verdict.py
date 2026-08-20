"""Focused tests split from ``test_context_ablation.py``."""

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
from llb.eval.context_ablation.derived import (
    is_contaminated,
    skipped_item_ids,
)
from llb.eval.context_ablation.models import (
    DERIVED_LONG_CONTEXT_DELTA,
    DERIVED_LONG_CONTEXT_DELTA_FITTING,
    DERIVED_RETRIEVAL_UPLIFT,
    LANE_CLOSED_BOOK,
    LANE_LONG_CONTEXT,
    LANE_RAG,
    VERDICT_LONG_CONTEXT_WINS,
    VERDICT_NO_RETRIEVAL_GAIN,
    VERDICT_RAG_PAYS_OFF,
    VERDICT_RETRIEVAL_INCONCLUSIVE,
)


def test_retrieval_uplift_is_rag_minus_closed_book_paired_per_item():
    ids = [f"q{i}" for i in range(8)]
    report = compare_context_strategies(
        _lanes([_row(i, 0.0, hit=0.0) for i in ids], [_row(i, 1.0) for i in ids]),
        _types(*ids),
        resamples=200,
    )
    uplift = _derived(report, DERIVED_RETRIEVAL_UPLIFT)
    assert (uplift["candidate"], uplift["reference"]) == (LANE_RAG, LANE_CLOSED_BOOK)
    assert uplift["paired"]["delta"]["mean"] == pytest.approx(1.0)
    assert report["verdict"]["decision"] == VERDICT_RAG_PAYS_OFF


def test_the_long_context_delta_is_stated_against_rag_not_against_the_baseline():
    ids = [f"q{i}" for i in range(8)]
    report = compare_context_strategies(
        _lanes(
            [_row(i, 0.0, hit=0.0) for i in ids],
            [_row(i, 0.5) for i in ids],
            [_row(i, 1.0) for i in ids],
        ),
        _types(*ids),
        resamples=200,
    )
    delta = _derived(report, DERIVED_LONG_CONTEXT_DELTA)
    assert (delta["candidate"], delta["reference"]) == (LANE_LONG_CONTEXT, LANE_RAG)
    assert delta["paired"]["delta"]["mean"] == pytest.approx(0.5)
    assert report["verdict"]["decision"] == VERDICT_LONG_CONTEXT_WINS


def test_a_skipped_item_gets_a_second_delta_over_the_items_the_lane_could_answer():
    """A skipped item scores zero; counting it as a long-context loss would be a lie."""
    ids = [f"q{i}" for i in range(8)]
    long_rows = [_row(i, 1.0) for i in ids[:-1]]
    long_rows.append(_row(ids[-1], 0.0, hit=0.0, status=eval_common.CONTEXT_OVERFLOW))
    report = compare_context_strategies(
        _lanes([_row(i, 0.0, hit=0.0) for i in ids], [_row(i, 0.5) for i in ids], long_rows),
        _types(*ids),
        resamples=200,
    )
    assert report["lanes"][LANE_LONG_CONTEXT]["skipped_item_ids"] == [ids[-1]]
    assert _derived(report, DERIVED_LONG_CONTEXT_DELTA)["n"] == 8
    fitting = _derived(report, DERIVED_LONG_CONTEXT_DELTA_FITTING)
    assert fitting["n"] == 7
    assert fitting["paired"]["delta"]["mean"] == pytest.approx(0.5)
    assert report["verdict"]["skipped"][LANE_LONG_CONTEXT] == 1


def test_without_a_skip_there_is_no_second_population_to_report():
    ids = ["q0", "q1"]
    report = compare_context_strategies(
        _lanes(
            [_row(i, 0.0) for i in ids], [_row(i, 0.0) for i in ids], [_row(i, 0.0) for i in ids]
        ),
        _types(*ids),
        resamples=0,
    )
    assert [entry["label"] for entry in report["derived"]] == [
        DERIVED_RETRIEVAL_UPLIFT,
        DERIVED_LONG_CONTEXT_DELTA,
    ]


def test_a_two_lane_comparison_reports_only_the_uplift():
    ids = ["q0", "q1"]
    report = compare_context_strategies(
        _lanes([_row(i, 0.0) for i in ids], [_row(i, 1.0) for i in ids]), _types(*ids), resamples=0
    )
    assert [entry["label"] for entry in report["derived"]] == [DERIVED_RETRIEVAL_UPLIFT]


def test_skips_are_read_from_the_terminal_status_not_from_a_missing_row():
    rows = [_row("a", 0.0), _row("b", 0.0, status=eval_common.CONTEXT_OVERFLOW)]
    assert skipped_item_ids(rows) == ["b"]


def test_a_closed_book_answer_that_matches_the_reference_is_flagged():
    assert is_contaminated({"exact": 1.0, "contains": 0.0})
    assert is_contaminated({"exact": 0.0, "contains": 1.0})
    assert not is_contaminated({"exact": 0.0, "contains": 0.0, "objective_score": 0.9})


def test_the_contamination_rate_is_measured_on_the_closed_book_lane_only():
    ids = ["q0", "q1", "q2", "q3"]
    closed = [_row(i, 0.0) for i in ids]
    closed[0] = _row("q0", 1.0, exact=1.0, contains=1.0)
    report = compare_context_strategies(
        _lanes(closed, [_row(i, 1.0, exact=1.0, contains=1.0) for i in ids]),
        _types(*ids),
        resamples=0,
    )
    contamination = report["contamination"]
    assert contamination["lane"] == LANE_CLOSED_BOOK
    assert contamination["item_ids"] == ["q0"]
    assert contamination["rate"] == pytest.approx(0.25)
    assert report["verdict"]["contamination_rate"] == pytest.approx(0.25)
    assert [item["contaminated"] for item in report["items"]] == [True, False, False, False]


def test_a_noisy_uplift_stays_inconclusive_instead_of_claiming_rag_pays_off():
    ids = [f"q{i}" for i in range(12)]
    rag = [_row(i, 0.0) for i in ids]
    rag[0] = _row("q0", 1.0)
    report = compare_context_strategies(
        _lanes([_row(i, 0.0) for i in ids], rag), _types(*ids), resamples=200
    )
    assert report["verdict"]["decision"] == VERDICT_RETRIEVAL_INCONCLUSIVE
    assert "calibrated test does not separate" in report["verdict"]["reason"]


def test_a_model_that_answers_as_well_from_its_weights_records_no_retrieval_gain():
    ids = [f"q{i}" for i in range(6)]
    report = compare_context_strategies(
        _lanes([_row(i, 1.0) for i in ids], [_row(i, 1.0) for i in ids]), _types(*ids), resamples=50
    )
    assert report["verdict"]["decision"] == VERDICT_NO_RETRIEVAL_GAIN


def test_the_verdict_reads_the_fitting_delta_so_skips_cannot_sink_the_long_context_lane():
    # Six items fit -- the fewest whose exact sign test can reach 95%, so the fitting subset is
    # readable on its own and the verdict turns on WHICH population it reads, not on the count.
    ids = [f"q{i}" for i in range(16)]
    long_rows = [_row(i, 1.0) for i in ids[:6]]
    long_rows += [_row(i, 0.0, hit=0.0, status=eval_common.CONTEXT_OVERFLOW) for i in ids[6:]]
    report = compare_context_strategies(
        _lanes([_row(i, 0.0) for i in ids], [_row(i, 0.5) for i in ids], long_rows),
        _types(*ids),
        resamples=200,
    )
    assert _derived(report, DERIVED_LONG_CONTEXT_DELTA)["paired"]["delta"]["mean"] < 0
    assert report["verdict"]["decision"] == VERDICT_LONG_CONTEXT_WINS


def test_an_unknown_baseline_lane_is_rejected():
    with pytest.raises(ValueError, match="baseline lane"):
        compare_context_strategies({LANE_RAG: [_row("a", 1.0)]}, {}, baseline=LANE_CLOSED_BOOK)


def test_the_report_leads_with_the_derived_numbers_and_stays_ascii():
    ids = [f"q{i}" for i in range(6)]
    closed = [_row(i, 0.0) for i in ids]
    closed[0] = _row("q0", 1.0, exact=1.0, contains=1.0)
    report = compare_context_strategies(
        _lanes(closed, [_row(i, 1.0) for i in ids], [_row(i, 1.0) for i in ids]),
        _types(*ids),
        resamples=50,
    )
    text = format_report(report, metadata={"model": "m", "backend": "b"})
    assert "# RAG versus long context" in text
    assert text.index("### Derived numbers") < text.index("### Per lane")
    assert "retrieval_uplift" in text
    assert "### Flagged items" in text
    assert "contaminated" in text
    assert text.isascii()
