"""Held-out calibration of the sidecar-free graph-fusion router."""

import pytest

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.fusion_calibration import (
    calibrate_routing,
    format_report,
    parse_thresholds,
    policy_grid,
)
from llb.rag.fusion_calibration.evaluate import (
    DECISION_NO_RECOMMENDATION,
    DECISION_RECOMMEND,
)
from llb.rag.fusion_evidence.models import EvidenceItem
from llb.rag.fusion_routing import HeuristicPolicy, QuestionTypeRouter


def _chunk(doc: str) -> ChunkRecord:
    return {"doc_id": doc, "char_start": 0, "char_end": 1, "text": "x"}


def _span(doc: str) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": 0, "char_end": 1, "text": "x"}


class _Store:
    def __init__(self, hits: dict[str, list[ChunkRecord]]) -> None:
        self.hits = hits
        self.calls = 0

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        self.calls += 1
        return self.hits.get(question, [])[:k]


LONG_MULTI = "one two three four five six seven"
LONG_SINGLE = "alpha beta gamma delta epsilon zeta eta"
SHORT_MULTI = "one two"


def _multi(item_id: str, question: str) -> EvidenceItem:
    return EvidenceItem(item_id, question, [_span("d1"), _span("d2")], "multi-hop")


def _single(item_id: str, question: str) -> EvidenceItem:
    return EvidenceItem(item_id, question, [_span("d1")], "factoid")


def test_threshold_grid_parsing_and_policy_validation():
    assert parse_thresholds("5, 10,5") == (5, 10)
    assert parse_thresholds("0,2", allow_zero=True) == (0, 2)
    assert [policy.label for policy in policy_grid((5,), (0, 2))] == ["w5/e0", "w5/e2"]
    with pytest.raises(ValueError, match="at least 1"):
        HeuristicPolicy(0, 2)
    with pytest.raises(ValueError, match="non-negative"):
        HeuristicPolicy(5, -1)


def test_zero_entity_threshold_makes_length_sufficient_without_a_sidecar():
    decision = QuestionTypeRouter(0.3, {}, HeuristicPolicy(5, 0)).decide(LONG_MULTI)
    assert decision.route == "graph"
    assert decision.source == "heuristic"
    assert decision.signals == ("entity_requirement_disabled", "long_question")


def test_calibration_selects_on_tuning_and_scores_only_the_frozen_policy_on_final():
    tuning = [_multi("tm", LONG_MULTI), _single("ts", LONG_SINGLE)]
    final = [_multi("fm", LONG_MULTI), _single("fs", "short fact")]
    questions = [item.question for item in [*tuning, *final]]
    vector = _Store({question: [_chunk("d1"), _chunk("noise")] for question in questions})
    graph = _Store({question: [_chunk("d2")] for question in questions})
    report = calibrate_routing(
        vector,
        graph,
        tuning,
        final,
        policy_grid((5,), (0, 2)),
        k=2,
        graph_strategy="global_community",
        graph_weight=0.5,
        candidates=2,
        span_identity="exact",
        tuning_split="tuning",
        final_split="final",
        resamples=20,
    )
    assert report["decision"] == DECISION_RECOMMEND
    assert report["frozen_policy"] == report["recommended_policy"] == "w5/e0"
    assert set(report["tuning"]) == {"w5/e0", "w5/e2"}
    assert report["final"]["policy"]["label"] == "w5/e0"
    assert vector.calls == len(questions) and graph.calls == len(questions)
    assert "Final is evaluated" in format_report(report)
    assert report["final"]["route_errors"] == []


def test_final_gate_can_veto_a_policy_selected_on_tuning():
    tuning = [_multi("tm", LONG_MULTI), _single("ts", "short fact")]
    final = [_multi("fm", SHORT_MULTI), _single("fs", "brief fact")]
    questions = [item.question for item in [*tuning, *final]]
    vector = _Store({question: [_chunk("d1"), _chunk("noise")] for question in questions})
    graph = _Store({question: [_chunk("d2")] for question in questions})
    report = calibrate_routing(
        vector,
        graph,
        tuning,
        final,
        (HeuristicPolicy(5, 0),),
        k=2,
        graph_strategy="global_community",
        graph_weight=0.5,
        candidates=2,
        span_identity="exact",
        tuning_split="tuning",
        final_split="final",
        resamples=20,
    )
    assert report["tuning"]["w5/e0"]["recommendation_gate"] is True
    assert report["final"]["recommendation_gate"] is False
    assert report["decision"] == DECISION_NO_RECOMMENDATION
    assert report["recommended_policy"] is None
    assert report["final"]["route_errors"] == [
        {
            "item_id": "fm",
            "predicted": "vector",
            "expected": "graph",
            "signals": ("entity_requirement_disabled",),
        }
    ]
    assert "Frozen final routing errors" in format_report(report)


def test_calibration_never_recommends_a_policy_without_a_retrieval_gain():
    tuning = [_multi("tm", LONG_MULTI), _single("ts", LONG_SINGLE)]
    final = [_multi("fm", LONG_MULTI), _single("fs", LONG_SINGLE)]
    questions = [item.question for item in [*tuning, *final]]
    vector = _Store({question: [_chunk("d1")] for question in questions})
    graph = _Store({})
    report = calibrate_routing(
        vector,
        graph,
        tuning,
        final,
        (HeuristicPolicy(5, 0),),
        k=2,
        graph_strategy="global_community",
        graph_weight=0.5,
        candidates=2,
        span_identity="exact",
        tuning_split="tuning",
        final_split="final",
        resamples=0,
    )
    assert report["decision"] == DECISION_NO_RECOMMENDATION
    assert report["recommended_policy"] is None
    assert report["frozen_policy"] == "w5/e0"
