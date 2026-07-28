"""Focused tests split from ``test_fusion_evidence.py``."""

import pytest
from _fusion_evidence_helpers import (
    _MULTI_HOP_N,
    _ByQuestion,
    _chunk,
    _fusion_report,
    _multi_hop_item,
    _span,
)

from llb.rag.fusion_evidence import (
    EvidenceItem,
    build_sweep_rows,
    evaluate_fusion_evidence,
    format_report,
)
from llb.rag.fusion_evidence.evidence_gate import READING_INSUFFICIENT_EVIDENCE
from llb.rag.fusion_evidence.models import (
    METRIC_ALL_SPANS,
    METRIC_RECALL,
    VERDICT_ADOPT,
    VERDICT_INCONCLUSIVE,
    VERDICT_NO_EVIDENCE,
    VERDICT_REJECT,
)
from llb.rag.fusion_evidence.rows import VECTOR_ROW


def test_fusion_that_completes_the_multi_hop_evidence_is_adopted():
    report = _fusion_report()
    focus = report["rows"]["fused/local_khop@0.30/d10"]["slices"]["multi-hop"]
    assert focus["n"] == _MULTI_HOP_N
    assert focus["metrics"][METRIC_ALL_SPANS]["mean"] == 1.0
    assert (
        report["rows"][VECTOR_ROW]["slices"]["multi-hop"]["metrics"][METRIC_ALL_SPANS]["mean"]
        == 0.0
    )
    assert focus["paired_vs_baseline"][METRIC_ALL_SPANS]["wins"] == _MULTI_HOP_N
    verdict = report["verdict"]
    assert verdict["decision"] == VERDICT_ADOPT
    assert verdict["best_row"] == "fused/local_khop@0.30/d10"
    assert verdict["focus_n"] == _MULTI_HOP_N


def test_the_same_gain_on_too_few_items_is_not_adopted():
    """The identical fixture one item short of the reachable minimum states no separation."""
    report = _fusion_report(multi_hop=_MULTI_HOP_N - 1)
    focus = report["rows"]["fused/local_khop@0.30/d10"]["slices"]["multi-hop"]
    # Same point estimate, same interval, same unanimous ledger -- only the reading changes.
    assert focus["paired_vs_baseline"][METRIC_ALL_SPANS]["delta"]["lo"] > 0.0
    assert focus["paired_vs_baseline"][METRIC_ALL_SPANS]["stability"]["reading"] == (
        READING_INSUFFICIENT_EVIDENCE
    )
    verdict = report["verdict"]
    assert verdict["decision"] == VERDICT_INCONCLUSIVE
    assert "INSUFFICIENT EVIDENCE" in verdict["reason"]
    assert format_report(report).count("insufficient evidence") > 0


def test_zero_weight_fused_row_ties_the_vector_baseline_exactly():
    report = _fusion_report()
    passthrough = report["rows"]["fused/local_khop@0.00/d10"]["overall"]
    for metric, comparison in passthrough["paired_vs_baseline"].items():
        assert comparison["wins"] == comparison["losses"] == 0, metric
        assert comparison["delta"]["mean"] == 0.0, metric


def test_no_multi_hop_item_yields_no_evidence_not_a_recommendation():
    items = [EvidenceItem("f-1", "q", [_span("d1", 0, 10)], "factoid")]
    vector = _ByQuestion({"q": [_chunk("d1", 0, 10)]})
    graph = _ByQuestion({"q": []})
    rows = build_sweep_rows(vector, {"local_khop": graph}, ["q"], k=10, weights=(0.3,))
    report = evaluate_fusion_evidence(rows, items, 10, baseline=VECTOR_ROW, resamples=50)
    assert report["verdict"]["decision"] == VERDICT_NO_EVIDENCE
    assert report["verdict"]["focus_n"] == 0
    assert report["focus_items"] == []


def test_a_multi_hop_gain_paid_for_in_overall_recall_is_rejected():
    # A heavy graph share completes each multi-hop item's second span but crowds the factoids'
    # only gold chunk out of the top-k: a real gain that the overall lane pays for. Both slices
    # carry enough differing items for their readings to be reachable, so what the verdict turns
    # on is the overall cost, not the item count.
    multi = [f"q{i}" for i in range(_MULTI_HOP_N)]
    factoid = [f"f{i}" for i in range(_MULTI_HOP_N)]
    items = [
        *(_multi_hop_item(f"mh-{i}", question) for i, question in enumerate(multi)),
        *(
            EvidenceItem(f"f-{i}", q, [_span("d1", 0, 10)], "factoid")
            for i, q in enumerate(factoid)
        ),
    ]
    vector = _ByQuestion(
        {
            **{q: [_chunk("d1", 0, 10), _chunk("d3", 0, 10)] for q in multi},
            **{q: [_chunk("d8", 0, 10), _chunk("d1", 0, 10)] for q in factoid},
        }
    )
    graph = _ByQuestion(
        {
            **{q: [_chunk("d2", 0, 10)] for q in multi},
            **{q: [_chunk("d7", 0, 10), _chunk("d6", 0, 10)] for q in factoid},
        }
    )
    rows = build_sweep_rows(vector, {"local_khop": graph}, multi + factoid, k=2, weights=(0.9,))
    report = evaluate_fusion_evidence(rows, items, 2, baseline=VECTOR_ROW, resamples=50)
    fused = report["rows"]["fused/local_khop@0.90/d2"]
    assert (
        fused["slices"]["multi-hop"]["paired_vs_baseline"][METRIC_ALL_SPANS]["wins"] == _MULTI_HOP_N
    )
    assert fused["overall"]["paired_vs_baseline"][METRIC_RECALL]["delta"]["mean"] < 0
    assert report["verdict"]["decision"] == VERDICT_REJECT
    assert "overall recall@k" in report["verdict"]["reason"]


def test_evaluate_rejects_an_unknown_baseline_row():
    with pytest.raises(ValueError, match="baseline row"):
        evaluate_fusion_evidence({}, [], 10, baseline=VECTOR_ROW)


def test_report_is_ascii_and_carries_the_slice_uncertainty_and_item_ledger():
    text = format_report(_fusion_report())
    assert text.isascii()  # AGENTS.md: ASCII-only output
    assert "Focus slice: multi-hop" in text
    assert "all-spans@k" in text and "bootstrap CI" in text
    assert "Item-level outcomes (multi-hop)" in text and "mh-1" in text
    assert "Slice: factoid" in text


def test_report_says_so_when_the_multi_hop_slice_is_empty():
    items = [EvidenceItem("f-1", "q", [_span("d1", 0, 10)], "factoid")]
    rows = build_sweep_rows(
        _ByQuestion({"q": [_chunk("d1", 0, 10)]}),
        {"local_khop": _ByQuestion({})},
        ["q"],
        10,
        (0.3,),
    )
    text = format_report(
        evaluate_fusion_evidence(rows, items, 10, baseline=VECTOR_ROW, resamples=0)
    )
    assert "No multi-hop item was scored." in text
    # an all-zero metric table for an empty slice would read like a measured result
    assert "No item falls in this slice" in text
    assert text.count("| vector |") == 2  # overall + factoid only, not the empty focus slice


def test_a_gain_whose_interval_includes_zero_is_inconclusive_not_adopted():
    # Four multi-hop items, one of which fusion recovers: the mean delta is +0.25 but resamples
    # that omit the single winner make the interval touch zero, so the lane must NOT recommend.
    items = [_multi_hop_item(f"mh-{i}", f"q{i}") for i in range(4)]
    covered = [_chunk("d1", 0, 10), _chunk("d2", 0, 10)]
    vector = _ByQuestion({"q0": [], "q1": covered, "q2": covered, "q3": covered})
    graph = _ByQuestion({"q0": [_chunk("d1", 0, 10)], "q1": [], "q2": [], "q3": []})
    rows = build_sweep_rows(
        vector, {"local_khop": graph}, [i.question for i in items], k=10, weights=(0.3,)
    )
    report = evaluate_fusion_evidence(rows, items, 10, baseline=VECTOR_ROW, resamples=500)
    focus = report["rows"]["fused/local_khop@0.30/d10"]["slices"]["multi-hop"]
    delta = focus["paired_vs_baseline"][METRIC_RECALL]["delta"]
    assert delta["mean"] == pytest.approx(0.25) and delta["lo"] == 0.0
    assert report["verdict"]["decision"] == VERDICT_INCONCLUSIVE
    assert "calibrated randomization test does not separate" in report["verdict"]["reason"]
