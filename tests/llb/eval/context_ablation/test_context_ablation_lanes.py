"""Focused tests split from ``test_context_ablation.py``."""

from pathlib import Path

import pytest
from tests.llb.eval._context_ablation_helpers import (
    ALWAYS_FITS,
    NEVER_FITS,
)

from llb.core.config import RunConfig
from llb.eval import common as eval_common
from llb.eval.context_ablation import (
    lane_config,
    parse_lanes,
)
from llb.eval.context_ablation.models import (
    LANE_CLOSED_BOOK,
    LANE_LONG_CONTEXT,
    LANE_RAG,
)
from llb.eval.context_ablation.sources import (
    build_context_lane,
    closed_book_source,
    long_context_source,
    whole_document_chunk,
)
from llb.eval.graph import (
    CLOSED_BOOK_TEMPLATE,
    build_messages,
)


def test_the_default_selection_is_all_three_lanes_with_closed_book_first():
    assert parse_lanes("closed_book,rag,long_context") == [
        LANE_CLOSED_BOOK,
        LANE_RAG,
        LANE_LONG_CONTEXT,
    ]


def test_closed_book_is_pulled_to_the_front_so_the_baseline_never_moves():
    assert parse_lanes("long_context,rag,closed_book")[0] == LANE_CLOSED_BOOK


def test_a_lane_selection_deduplicates_in_the_order_given():
    assert parse_lanes("rag, closed_book ,rag") == [LANE_CLOSED_BOOK, LANE_RAG]


@pytest.mark.parametrize("spec", ["", "faiss", "rag,oracle"])
def test_an_unknown_context_lane_is_rejected(spec: str):
    with pytest.raises(ValueError):
        parse_lanes(spec)


def test_lane_config_selects_the_strategy_run_eval_reproduces_the_bundle_from():
    base = RunConfig(model="m")
    lane = lane_config(base, LANE_LONG_CONTEXT, run_name_prefix="context-ablation")
    assert lane.context_strategy == LANE_LONG_CONTEXT
    assert lane.run_name == "context-ablation-long_context"
    assert lane_config(base, LANE_RAG, run_name_prefix="x").context_strategy == LANE_RAG


def test_closed_book_sends_no_context_but_must_not_raise_a_retrieval_miss():
    """`retrieval_miss` short-circuits generation, and a lane that never calls the model
    measures nothing -- an empty context is the POINT of this lane, not a failure."""
    update = closed_book_source()({"question": "q", "gold_spans": []})
    assert update["retrieved"] == []
    assert update["context"] == ""
    assert "status" not in update


def test_the_closed_book_prompt_carries_the_question_and_no_context_block():
    messages = build_messages("Столиця України?", "IGNORED", template_id=CLOSED_BOOK_TEMPLATE)
    assert "IGNORED" not in "".join(message["content"] for message in messages)
    assert any("Столиця України?" in message["content"] for message in messages)


def test_long_context_lays_the_whole_gold_document_in_offset_exact():
    documents = {"d1.txt": "abcdef"}
    state = {"gold_spans": [{"doc_id": "d1.txt", "char_start": 2, "char_end": 4, "text": "cd"}]}
    update = long_context_source(documents, ALWAYS_FITS)(state)
    chunk = update["retrieved"][0]
    assert (chunk["char_start"], chunk["char_end"], chunk["text"]) == (0, 6, "abcdef")
    assert "abcdef" in update["context"]
    assert "status" not in update


def test_a_multi_document_item_carries_every_gold_document_once():
    documents = {"a.txt": "AAA", "b.txt": "BBB"}
    spans = [
        {"doc_id": "b.txt", "char_start": 0, "char_end": 1, "text": "B"},
        {"doc_id": "a.txt", "char_start": 0, "char_end": 1, "text": "A"},
        {"doc_id": "b.txt", "char_start": 1, "char_end": 2, "text": "B"},
    ]
    update = long_context_source(documents, ALWAYS_FITS)({"gold_spans": spans})
    assert [chunk["doc_id"] for chunk in update["retrieved"]] == ["b.txt", "a.txt"]


def test_a_document_that_does_not_fit_is_skipped_never_truncated():
    documents = {"d1.txt": "x" * 100_000}
    state = {"gold_spans": [{"doc_id": "d1.txt", "char_start": 0, "char_end": 1, "text": "x"}]}
    update = long_context_source(documents, NEVER_FITS)(state)
    assert update["status"] == eval_common.CONTEXT_OVERFLOW
    assert update["retrieved"] == []
    assert update["context"] == ""


def test_a_gold_document_missing_from_the_corpus_fails_loudly():
    state = {"gold_spans": [{"doc_id": "gone.txt", "char_start": 0, "char_end": 1, "text": "x"}]}
    with pytest.raises(SystemExit, match="gone.txt"):
        long_context_source({}, ALWAYS_FITS)(state)


def test_a_skipped_case_never_reaches_the_model():
    from llb.eval import graph

    calls: list[object] = []

    class Launcher:
        def chat(self, messages, **kwargs):  # pragma: no cover - must never run
            calls.append(messages)
            raise AssertionError("a skipped case must not call the backend")

    node = graph.make_generate_node(Launcher(), max_tokens=8, temperature=0.0, timeout=1)
    update = node({"question": "q", "status": eval_common.CONTEXT_OVERFLOW})
    assert update == {"answer": "", "usage": {}}
    assert calls == []


def test_the_rag_strategy_installs_no_context_source_at_all():
    assert build_context_lane(RunConfig(context_strategy=LANE_RAG), ALWAYS_FITS) is None


def test_the_closed_book_strategy_selects_the_closed_book_prompt():
    lane = build_context_lane(RunConfig(context_strategy=LANE_CLOSED_BOOK), ALWAYS_FITS)
    assert lane is not None and lane.template_id == CLOSED_BOOK_TEMPLATE


def test_the_long_context_strategy_reads_the_corpus_and_keeps_the_rag_prompt(tmp_path: Path):
    (tmp_path / "d1.txt").write_text("документ", encoding="utf-8")
    lane = build_context_lane(
        RunConfig(context_strategy=LANE_LONG_CONTEXT, corpus_root=tmp_path), ALWAYS_FITS
    )
    assert lane is not None and lane.template_id is None
    state = {
        "gold_spans": [{"doc_id": "d1.txt", "char_start": 0, "char_end": 8, "text": "документ"}]
    }
    assert lane.source(state)["retrieved"][0]["text"] == "документ"


def test_the_context_budget_is_what_decides_a_skip():
    """`fits_context_chars` is the one budget rule; a small explicit budget must skip a big doc."""
    config = RunConfig(context_budget=1024, max_tokens=256)
    lane = build_context_lane(RunConfig(context_strategy=LANE_CLOSED_BOOK), ALWAYS_FITS)
    assert lane is not None  # sanity: the strategy switch itself is wired
    from llb.backends.context_fit import fits_context_chars

    assert fits_context_chars(config, None, 0, 0, 500)
    assert not fits_context_chars(config, None, 0, 0, 100_000)


def test_a_whole_document_chunk_is_a_verbatim_corpus_slice():
    chunk = whole_document_chunk("d.md", "text")
    assert chunk["text"] == "text"
    assert (chunk["char_start"], chunk["char_end"]) == (0, 4)
