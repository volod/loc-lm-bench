"""The shippable sibling of the oracle lane: retrieve the chunk, send its whole document.

Two things have to hold for this lane to mean anything. It must select its documents from the
RANKING and never from the gold label -- otherwise it is the oracle lane with extra steps -- and
adding it must not move a single number the three existing lanes already report, or every earlier
context-ablation reading would need re-taking.
"""

from pathlib import Path

import pytest
from tests.llb.eval._context_ablation_helpers import (
    ALWAYS_FITS,
    NEVER_FITS,
    _derived,
    _lanes,
    _row,
    _types,
)

from llb.core.config import RunConfig
from llb.eval import common as eval_common
from llb.eval.context_ablation import compare_context_strategies, parse_lanes
from llb.eval.context_ablation.models import (
    ADOPT_RETRIEVED_DOCUMENT,
    DERIVED_LONG_CONTEXT_DELTA,
    DERIVED_LONG_CONTEXT_DELTA_FITTING,
    DERIVED_ORACLE_DOCUMENT_GAP,
    DERIVED_RETRIEVAL_UPLIFT,
    DERIVED_RETRIEVED_DOCUMENT_DELTA,
    DERIVED_RETRIEVED_DOCUMENT_DELTA_FITTING,
    LANE_CLOSED_BOOK,
    LANE_LONG_CONTEXT,
    LANE_RAG,
    LANE_RETRIEVED_DOCUMENT,
    REJECT_RETRIEVED_DOCUMENT,
    RETRIEVED_DOCUMENT_INCONCLUSIVE,
    RETRIEVED_DOCUMENT_NOT_MEASURED,
)
from llb.eval.context_ablation.sources import (
    build_context_lane,
    ranked_doc_ids,
    retrieved_document_refiner,
)

DOCUMENTS = {"a.txt": "AAAA AAAA AAAA", "b.txt": "BBBB BBBB", "c.txt": "CCCC"}


def _chunk(doc_id: str, text: str = "chunk") -> dict:
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}#0",
        "char_start": 0,
        "char_end": len(text),
        "text": text,
        "strategy": "fixed",
        "metadata": {},
    }


def _retrieved(*doc_ids: str) -> dict:
    return {"retrieved": [_chunk(doc_id) for doc_id in doc_ids], "context": "CHUNKS"}


# --- document selection -------------------------------------------------------------------


def test_the_selection_rule_walks_the_ranking_and_counts_documents_not_chunks():
    chunks = [_chunk("a.txt"), _chunk("a.txt"), _chunk("b.txt"), _chunk("c.txt")]
    assert ranked_doc_ids(chunks, 1) == ["a.txt"]
    assert ranked_doc_ids(chunks, 2) == ["a.txt", "b.txt"]
    assert ranked_doc_ids(chunks, 9) == ["a.txt", "b.txt", "c.txt"]


def test_the_top_ranked_chunks_document_replaces_the_chunk_block_in_the_prompt():
    refine = retrieved_document_refiner(DOCUMENTS, ALWAYS_FITS)
    update = refine({"question": "q"}, _retrieved("b.txt", "a.txt"))
    assert [chunk["doc_id"] for chunk in update["retrieved"]] == ["b.txt"]
    assert update["retrieved"][0]["text"] == DOCUMENTS["b.txt"]
    assert "BBBB BBBB" in update["context"] and "CHUNKS" not in update["context"]


def test_the_gold_label_never_enters_the_selection():
    """The whole point of the lane: a gold span pointing elsewhere must change nothing."""
    refine = retrieved_document_refiner(DOCUMENTS, ALWAYS_FITS)
    state = {"gold_spans": [{"doc_id": "c.txt", "char_start": 0, "char_end": 4, "text": "CCCC"}]}
    update = refine(state, _retrieved("a.txt"))
    assert [chunk["doc_id"] for chunk in update["retrieved"]] == ["a.txt"]


def test_a_wider_document_budget_lays_in_more_distinct_documents():
    refine = retrieved_document_refiner(DOCUMENTS, ALWAYS_FITS, top_n=2)
    update = refine({}, _retrieved("a.txt", "a.txt", "c.txt"))
    assert [chunk["doc_id"] for chunk in update["retrieved"]] == ["a.txt", "c.txt"]


def test_documents_that_do_not_fit_are_skipped_never_truncated():
    refine = retrieved_document_refiner(DOCUMENTS, NEVER_FITS)
    update = refine({}, _retrieved("a.txt"))
    assert update["status"] == eval_common.CONTEXT_OVERFLOW
    assert update["retrieved"] == [] and update["context"] == ""


def test_a_retrieved_document_missing_from_the_corpus_fails_loudly():
    """A store built from another corpus would otherwise silently degrade the lane back to rag."""
    refine = retrieved_document_refiner(DOCUMENTS, ALWAYS_FITS)
    with pytest.raises(SystemExit, match="gone.txt"):
        refine({}, _retrieved("gone.txt"))


def test_the_measured_cost_of_the_retrieval_that_chose_the_document_is_kept():
    refine = retrieved_document_refiner(DOCUMENTS, ALWAYS_FITS)
    base = {**_retrieved("a.txt"), "retrieve_latency_s": 0.25, "query_corrections": 2}
    update = refine({}, base)
    assert update["retrieve_latency_s"] == 0.25
    assert update["query_corrections"] == 2


def test_no_restored_header_is_claimed_for_a_prompt_that_carries_whole_documents():
    refine = retrieved_document_refiner(DOCUMENTS, ALWAYS_FITS)
    base = {
        **_retrieved("a.txt"),
        "prompt_chunks": [_chunk("a.txt", "header + chunk")],
        "table_headers_restored": 1,
        "table_header_chars": 9,
    }
    update = refine({}, base)
    assert "prompt_chunks" not in update
    assert (update["table_headers_restored"], update["table_header_chars"]) == (0, 0)


# --- wiring -------------------------------------------------------------------------------


def test_the_lane_installs_a_refiner_and_keeps_the_rag_prompt(tmp_path: Path):
    (tmp_path / "a.txt").write_text("документ", encoding="utf-8")
    lane = build_context_lane(
        RunConfig(context_strategy=LANE_RETRIEVED_DOCUMENT, corpus_root=tmp_path), ALWAYS_FITS
    )
    assert lane is not None
    assert lane.source is None, "the lane retrieves; it must not replace the retrieve node"
    assert lane.template_id is None, "same generation prompt as rag, so the delta is the context"
    update = lane.refiner({}, _retrieved("a.txt"))
    assert update["retrieved"][0]["text"] == "документ"


def test_the_refiner_runs_after_real_retrieval_inside_the_retrieve_node(tmp_path: Path):
    from llb.eval.graph import make_retrieve_node

    class Store:
        def retrieve(self, question: str, k: int) -> list[dict]:
            return [_chunk("a.txt"), _chunk("b.txt")][:k]

    node = make_retrieve_node(
        Store(), 2, context_refiner=retrieved_document_refiner(DOCUMENTS, ALWAYS_FITS)
    )
    update = node({"question": "q"})
    assert [chunk["doc_id"] for chunk in update["retrieved"]] == ["a.txt"]
    assert update["retrieve_latency_s"] >= 0.0


def test_a_retrieval_miss_short_circuits_before_the_lane_widens_anything():
    from llb.eval.graph import make_retrieve_node

    class EmptyStore:
        def retrieve(self, question: str, k: int) -> list[dict]:
            return []

    def explode(state, update):  # pragma: no cover - must never run
        raise AssertionError("nothing was retrieved, so there is no document to widen to")

    node = make_retrieve_node(EmptyStore(), 3, context_refiner=explode)
    assert node({"question": "q"})["status"] == eval_common.RETRIEVAL_MISS


def test_the_lane_is_selectable_by_name_in_any_order():
    assert parse_lanes("long_context,retrieved_document,rag,closed_book") == [
        LANE_CLOSED_BOOK,
        LANE_LONG_CONTEXT,
        LANE_RETRIEVED_DOCUMENT,
        LANE_RAG,
    ]
    assert parse_lanes("closed_book,retrieved_document") == [
        LANE_CLOSED_BOOK,
        LANE_RETRIEVED_DOCUMENT,
    ]


# --- the four-lane comparison -------------------------------------------------------------


def _four_lane_report(**kwargs):
    ids = [f"q{i}" for i in range(8)]
    return compare_context_strategies(
        _lanes(
            [_row(item_id, 0.0) for item_id in ids],
            [_row(item_id, 0.4) for item_id in ids],
            long_context=[_row(item_id, 1.0) for item_id in ids],
            retrieved_document=[_row(item_id, 0.7) for item_id in ids],
        ),
        _types(*ids),
        resamples=50,
        **kwargs,
    )


def test_the_oracle_gap_splits_into_the_capturable_part_and_the_gold_label_part():
    report = _four_lane_report()
    captured = _derived(report, DERIVED_RETRIEVED_DOCUMENT_DELTA)["paired"]["delta"]["mean"]
    residual = _derived(report, DERIVED_ORACLE_DOCUMENT_GAP)["paired"]["delta"]["mean"]
    oracle = _derived(report, DERIVED_LONG_CONTEXT_DELTA)["paired"]["delta"]["mean"]
    assert captured == pytest.approx(0.3)
    assert residual == pytest.approx(0.3)
    assert captured + residual == pytest.approx(oracle)


def test_adding_the_lane_moves_no_number_the_three_existing_lanes_already_reported():
    ids = [f"q{i}" for i in range(8)]
    closed = [_row(item_id, 0.0) for item_id in ids]
    rag = [_row(item_id, 0.4) for item_id in ids]
    long_context = [_row(item_id, 1.0) for item_id in ids]
    three = compare_context_strategies(
        _lanes(closed, rag, long_context=long_context), _types(*ids), resamples=50
    )
    four = compare_context_strategies(
        _lanes(
            closed,
            rag,
            long_context=long_context,
            retrieved_document=[_row(item_id, 0.7) for item_id in ids],
        ),
        _types(*ids),
        resamples=50,
    )
    for label in (DERIVED_RETRIEVAL_UPLIFT, DERIVED_LONG_CONTEXT_DELTA):
        assert _derived(four, label) == _derived(three, label)
    for lane in (LANE_CLOSED_BOOK, LANE_RAG, LANE_LONG_CONTEXT):
        assert four["lanes"][lane] == three["lanes"][lane]
    assert four["verdict"]["decision"] == three["verdict"]["decision"]


def test_a_measured_gain_with_no_gold_label_is_an_explicit_adopt():
    adoption = _four_lane_report()["verdict"]["retrieved_document"]
    assert adoption["decision"] == ADOPT_RETRIEVED_DOCUMENT
    assert adoption["delta"] == pytest.approx(0.3)
    assert adoption["captured_share"] == pytest.approx(0.5)
    assert "no gold label" in adoption["reason"]


def test_a_measured_loss_is_an_explicit_reject():
    ids = [f"q{i}" for i in range(8)]
    report = compare_context_strategies(
        _lanes(
            [_row(item_id, 0.0) for item_id in ids],
            [_row(item_id, 0.6) for item_id in ids],
            long_context=[_row(item_id, 0.9) for item_id in ids],
            retrieved_document=[_row(item_id, 0.2) for item_id in ids],
        ),
        _types(*ids),
        resamples=50,
    )
    adoption = report["verdict"]["retrieved_document"]
    assert adoption["decision"] == REJECT_RETRIEVED_DOCUMENT
    assert "keep the chunked configuration" in adoption["reason"]


def test_a_delta_that_straddles_zero_is_not_adopted():
    ids = [f"q{i}" for i in range(8)]
    swing = [0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.0, 0.0]
    report = compare_context_strategies(
        _lanes(
            [_row(item_id, 0.0) for item_id in ids],
            [_row(item_id, 0.5) for item_id in ids],
            long_context=[_row(item_id, 0.8) for item_id in ids],
            retrieved_document=[_row(item_id, 0.5 + delta) for item_id, delta in zip(ids, swing)],
        ),
        _types(*ids),
        resamples=50,
    )
    adoption = report["verdict"]["retrieved_document"]
    assert adoption["decision"] == RETRIEVED_DOCUMENT_INCONCLUSIVE
    assert "keep the chunked configuration" in adoption["reason"]


def test_a_three_lane_run_says_there_is_nothing_to_adopt_rather_than_guessing():
    ids = [f"q{i}" for i in range(4)]
    report = compare_context_strategies(
        _lanes(
            [_row(item_id, 0.0) for item_id in ids],
            [_row(item_id, 0.5) for item_id in ids],
            long_context=[_row(item_id, 0.9) for item_id in ids],
        ),
        _types(*ids),
        resamples=50,
    )
    adoption = report["verdict"]["retrieved_document"]
    assert adoption["decision"] == RETRIEVED_DOCUMENT_NOT_MEASURED
    assert adoption["captured_share"] is None


def test_a_skip_in_one_document_lane_does_not_shrink_the_other_lanes_population():
    """Each fitting cut is scoped to its OWN pair; pooling every lane's skips would silently
    re-take a delta that was fully measured."""
    ids = [f"q{i}" for i in range(6)]
    overflowed = ids[0]
    report = compare_context_strategies(
        _lanes(
            [_row(item_id, 0.0) for item_id in ids],
            [_row(item_id, 0.4) for item_id in ids],
            long_context=[_row(item_id, 0.9) for item_id in ids],
            retrieved_document=[
                _row(item_id, 0.0 if item_id == overflowed else 0.7)
                | ({"status": eval_common.CONTEXT_OVERFLOW} if item_id == overflowed else {})
                for item_id in ids
            ],
        ),
        _types(*ids),
        resamples=50,
    )
    labels = [entry["label"] for entry in report["derived"]]
    assert DERIVED_RETRIEVED_DOCUMENT_DELTA_FITTING in labels
    assert DERIVED_LONG_CONTEXT_DELTA_FITTING not in labels
    assert _derived(report, DERIVED_LONG_CONTEXT_DELTA)["n"] == len(ids)
    assert _derived(report, DERIVED_RETRIEVED_DOCUMENT_DELTA_FITTING)["n"] == len(ids) - 1
    assert report["verdict"]["retrieved_document"]["skipped"] == 1


def test_the_report_states_the_adoption_call_beside_the_ablation_verdict():
    from llb.eval.context_ablation import format_report

    text = format_report(_four_lane_report())
    assert f"- retrieved-document lane: **{ADOPT_RETRIEVED_DOCUMENT}**" in text
    assert f"`{DERIVED_ORACLE_DOCUMENT_GAP}`" in text
    assert "capturing 50% of the oracle gap" in text
