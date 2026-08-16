"""Focused tests split from ``test_answer_quality.py``."""

import json
from pathlib import Path

import pytest
from _answer_quality_helpers import (
    FUSED,
    VECTOR,
    _lanes,
    _report,
    _retrieval_record,
    _row,
    _types,
)

from llb.eval.answer_quality import (
    compare_answer_quality,
    format_report,
)
from llb.eval.answer_quality.coverage import (
    read_case_coverage,
    with_coverage,
)
from llb.eval.answer_quality.models import (
    METRIC_OBJECTIVE,
    METRIC_RETRIEVAL_HIT,
    VERDICT_ANSWER_GAIN,
    VERDICT_INCONCLUSIVE,
    VERDICT_NO_EVIDENCE,
    VERDICT_NO_GAIN,
    VERDICT_RETRIEVAL_ONLY,
)


def test_a_consistent_objective_gain_is_an_answer_quality_gain():
    verdict = _report([0.0] * 12, [1.0] * 12)["verdict"]
    assert verdict["decision"] == VERDICT_ANSWER_GAIN
    assert verdict["best_lane"] == FUSED
    assert verdict["focus_n"] == 12


def test_a_gain_whose_interval_includes_zero_stays_inconclusive():
    verdict = _report([0.0] * 12, [1.0] + [0.0] * 11)["verdict"]
    assert verdict["decision"] == VERDICT_INCONCLUSIVE
    assert "calibrated test does not separate" in verdict["reason"]


def test_better_retrieval_that_does_not_reach_the_answer_is_a_retrieval_only_effect():
    """The finding this lane exists to produce: more evidence retrieved, no better answers."""
    verdict = _report(
        [0.0] * 12,
        [0.0] * 12,
        hits=([0.0] * 12, [1.0] * 6 + [0.0] * 6),
    )["verdict"]
    assert verdict["decision"] == VERDICT_RETRIEVAL_ONLY
    assert "retrieval-only effect" in verdict["reason"]
    assert verdict["coverage_metric"] == METRIC_RETRIEVAL_HIT


def test_every_candidate_lane_keeps_its_own_decision_not_just_the_winner():
    """A three-lane comparison has a result per lane; the headline verdict names only one."""
    overlap = "fused/global_community@0.30/d50/ioverlap"
    ids = [f"q{i}" for i in range(12)]
    report = compare_answer_quality(
        {
            VECTOR: [_row(i, 0.0, 0.0) for i in ids],
            FUSED: [_row(i, 0.0, 1.0) for i in ids],  # more evidence, identical answers
            overlap: [_row(i, 1.0, 1.0) for i in ids],  # more evidence AND better answers
        },
        _types(*ids),
        baseline=VECTOR,
        resamples=200,
    )
    verdict = report["verdict"]
    assert verdict["best_lane"] == overlap
    assert verdict["lane_decisions"][overlap]["decision"] == VERDICT_ANSWER_GAIN
    assert verdict["lane_decisions"][FUSED]["decision"] == VERDICT_RETRIEVAL_ONLY
    assert VECTOR not in verdict["lane_decisions"]  # the baseline is not judged against itself
    text = format_report(report)
    assert "Per-lane decisions" in text
    assert all(label in text for label in (FUSED, overlap))


def test_a_two_lane_comparison_does_not_repeat_its_verdict_as_a_lane_list():
    report = _report([0.0] * 12, [1.0] * 12)
    assert list(report["verdict"]["lane_decisions"]) == [FUSED]
    assert "Per-lane decisions" not in format_report(report)


def test_a_measured_coverage_gain_outranks_a_noisy_objective_gain():
    """A +0.01 objective whose interval spans zero must not hide a coverage gain that does not."""
    verdict = _report(
        [0.0] * 11 + [0.0],
        [0.0] * 11 + [0.2],
        hits=([0.0] * 12, [1.0] * 12),
    )["verdict"]
    assert verdict["decision"] == VERDICT_RETRIEVAL_ONLY
    assert "not separable" in verdict["reason"]


def test_coverage_columns_are_recomputed_from_the_bundles_retrieval_sidecar(tmp_path: Path):
    """`retrieval_hit` credits a one-hop context; `all_spans_at_k` is what a two-hop item needs."""
    (tmp_path / "retrieval.jsonl").write_text(
        _retrieval_record("q0", 1) + _retrieval_record("q1", 2), encoding="utf-8"
    )
    coverage = read_case_coverage(tmp_path, 10)
    assert coverage["q0"] == {
        "all_spans_at_k": 0.0,
        "span_coverage": 0.5,
        "context_chars": 10.0,
    }
    assert coverage["q1"] == {
        "all_spans_at_k": 1.0,
        "span_coverage": 1.0,
        "context_chars": 20.0,
    }
    enriched = with_coverage([_row("q0", 1.0)], coverage)
    assert enriched[0]["all_spans_at_k"] == 0.0
    assert enriched[0]["objective_score"] == 1.0


def test_a_second_hop_carried_by_a_collapsed_duplicate_counts_as_covered(tmp_path: Path):
    """One retrieved chunk stands for both hops when its text is indexed once for two documents."""
    record = {
        "item_id": "q0",
        "retrieved": [
            {
                "doc_id": "d1",
                "char_start": 0,
                "char_end": 10,
                "rank": 1,
                "duplicate_count": 2,
                "duplicate_occurrences": [{"doc_id": "d2", "char_start": 0, "char_end": 10}],
            }
        ],
        "gold_spans": [
            {"doc_id": "d1", "char_start": 0, "char_end": 10, "text": "a"},
            {"doc_id": "d2", "char_start": 0, "char_end": 10, "text": "b"},
        ],
    }
    (tmp_path / "retrieval.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert read_case_coverage(tmp_path, 10)["q0"] == {
        "all_spans_at_k": 1.0,
        "span_coverage": 1.0,
        # One SERVED record, so the context bill counts it once however many documents it stands
        # for -- the collapsed occurrences restore coverage, never context.
        "context_chars": 10.0,
    }


def test_a_bundle_without_the_sidecar_keeps_its_rows_unchanged(tmp_path: Path):
    rows = [_row("q0", 1.0)]
    assert with_coverage(rows, read_case_coverage(tmp_path, 10)) == rows


def test_the_verdict_prefers_the_graded_coverage_metric_the_sidecar_supplied():
    """`all_spans_at_k` is uniformly 0.0 on a hard multi-hop slice; graded coverage still moves."""
    # Six items: the fewest a coverage separation can be read on at 95%, so the assertion is
    # about which METRIC the verdict reads rather than about the minimum-evidence gate.
    ids = [f"q{i}" for i in range(6)]
    vector = [dict(_row(i, 0.0), all_spans_at_k=0.0, span_coverage=0.0) for i in ids]
    fused = [dict(_row(i, 0.0), all_spans_at_k=0.0, span_coverage=0.5) for i in ids]
    report = compare_answer_quality(_lanes(vector, fused), _types(*ids), baseline=VECTOR)
    assert report["metrics"] == [
        "objective_score",
        "token_f1",
        "retrieval_hit",
        "all_spans_at_k",
        "span_coverage",
    ]
    assert report["verdict"]["coverage_metric"] == "span_coverage"
    assert report["verdict"]["decision"] == VERDICT_RETRIEVAL_ONLY
    assert "span_coverage +0.500" in report["verdict"]["reason"]


def test_a_coverage_column_only_one_lane_measured_is_dropped_rather_than_zero_filled():
    ids = ["q0", "q1"]
    vector = [_row(i, 0.0) for i in ids]
    fused = [dict(_row(i, 0.0), all_spans_at_k=1.0, span_coverage=1.0) for i in ids]
    report = compare_answer_quality(_lanes(vector, fused), _types(*ids), baseline=VECTOR)
    assert "all_spans_at_k" not in report["metrics"]
    assert report["verdict"]["coverage_metric"] == METRIC_RETRIEVAL_HIT


def test_no_gain_on_either_axis_is_recorded_as_such():
    assert _report([1.0] * 6, [1.0] * 6)["verdict"]["decision"] == VERDICT_NO_GAIN


def test_a_set_without_a_focus_slice_item_claims_no_evidence():
    report = compare_answer_quality(
        _lanes([_row("a", 1.0)], [_row("a", 0.0)]), {"a": "factoid"}, baseline=VECTOR, resamples=0
    )
    assert report["verdict"]["decision"] == VERDICT_NO_EVIDENCE
    assert report["lanes"][VECTOR]["slices"]["multi-hop"]["n"] == 0
    assert report["lanes"][VECTOR]["slices"]["factoid"]["n"] == 1


def test_untyped_items_score_overall_but_join_no_slice():
    report = compare_answer_quality(
        _lanes([_row("a", 1.0), _row("b", 0.0)], [_row("a", 1.0), _row("b", 0.0)]),
        {"a": "multi-hop"},
        baseline=VECTOR,
        resamples=0,
    )
    assert report["lanes"][VECTOR]["overall"]["n"] == 2
    assert report["lanes"][VECTOR]["slices"]["multi-hop"]["n"] == 1


def test_focus_item_ledger_carries_every_lane_per_item():
    report = _report([0.0, 1.0], [1.0, 1.0], resamples=0)
    ledger = report["focus_items"]
    assert [item["item_id"] for item in ledger] == ["q0", "q1"]
    assert ledger[0]["lanes"][FUSED][METRIC_OBJECTIVE] == pytest.approx(1.0)
    assert ledger[0]["lanes"][VECTOR][METRIC_RETRIEVAL_HIT] == pytest.approx(1.0)


def test_report_renders_ascii_tables_with_the_verdict_and_the_item_ledger():
    text = format_report(_report([0.0] * 6, [1.0] * 6), metadata={"model": "m", "backend": "b"})
    assert "# Multi-hop answer quality" in text
    assert VERDICT_ANSWER_GAIN in text
    assert "### Focus slice: multi-hop" in text
    assert "### Item-level outcomes (multi-hop)" in text
    assert text.isascii()
