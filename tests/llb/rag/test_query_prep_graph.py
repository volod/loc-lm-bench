"""Focused tests split from ``test_query_prep_pipeline.py``."""

import pytest
from _query_prep_pipeline_helpers import (
    _UK_PLAUSIBLE,
)
from _query_prep_helpers import RecordingStore

from llb.eval import graph
from llb.rag.query_prep.base import STEP_NORMALIZE
from llb.rag.query_prep.pipeline import QueryPrep
from llb.rag.query_prep.report import (
    cumulative_pipelines,
    format_query_prep_ab,
    query_prep_ab_report,
)


def test_ab_report_attributes_per_step_delta():
    # the fake store only "finds" the gold span when the query is transliterated to Cyrillic
    def retrieve(result, k):
        return (
            [{"doc_id": "d", "char_start": 0, "char_end": 5}] if "закон" in result.processed else []
        )

    items = [("zakon", [{"doc_id": "d", "char_start": 0, "char_end": 5}])]
    stages = cumulative_pipelines([STEP_NORMALIZE])
    report = query_prep_ab_report(items, retrieve, 5, stages)
    assert [row["stage"] for row in report["stages"]] == ["baseline", "+normalize"]
    assert report["stages"][0]["recall_at_k"] == 0.0
    assert report["stages"][1]["recall_at_k"] == 1.0
    assert report["stages"][1]["delta_recall"] == pytest.approx(1.0)
    assert report["stages"][1]["cases"][0]["query_processed"] == "закон"
    assert "query-prep A/B" in format_query_prep_ab(report)


def test_retrieve_node_uses_processed_query_and_preserves_raw():
    chunks = [{"doc_id": "a", "text": "закон україни", "char_start": 0, "char_end": 13}]
    store = RecordingStore(chunks)
    pipeline = QueryPrep.build([STEP_NORMALIZE])
    node = graph.make_retrieve_node(store, k=5, query_prep=pipeline)
    update = node({"question": "Zakon Ukrainy"})
    assert store.seen == ["закон украіни"]  # retrieval used the transliterated query
    assert update["query_processed"] == "закон украіни"
    assert update["query_corrections"] == 2  # two transliterations


def test_pipeline_language_gate_passes_romanized_ukrainian():
    pipeline = QueryPrep.build([STEP_NORMALIZE], plausible=_UK_PLAUSIBLE)
    result = pipeline.process("Zakon Ukrainy")
    assert result.processed == "закон украіни"  # still transliterated
    assert result.normalize_gate is not None and result.normalize_gate.transliterate
    assert "query_normalize_gate" not in result.provenance()  # only surfaced on refusal


def test_pipeline_language_gate_leaves_foreign_query_unchanged():
    pipeline = QueryPrep.build([STEP_NORMALIZE], plausible=_UK_PLAUSIBLE)
    result = pipeline.process("What does the law say")
    assert result.processed == "What does the law say"  # verbatim, not Cyrillic nonsense
    assert result.provenance()["query_processed"] == "What does the law say"
    gate = result.provenance()["query_normalize_gate"]
    assert gate["transliterated"] is False and gate["plausible_tokens"] == 0


def test_pipeline_language_gate_off_without_probe_still_transliterates():
    # No probe wired: the gate is inert and per-token transliteration runs as before.
    result = QueryPrep.build([STEP_NORMALIZE]).process("what does the")
    assert result.processed != "what does the"  # mangled, exactly the pre-gate behavior
    assert result.normalize_gate is None
