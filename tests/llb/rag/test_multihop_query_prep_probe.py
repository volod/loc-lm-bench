"""Paired query-prep conversion over fake decomposition and retrieval seams."""

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.multihop_probe import (
    DIAGNOSIS_BUDGET,
    DIAGNOSIS_COVERED,
    DIAGNOSIS_QUERY,
    EvidenceItem,
    compare_multihop_query_prep,
    format_query_prep_probe_report,
)
from llb.rag.query_prep.pipeline import QueryPrep

_BUDGETS = (10, 25, 50)


def _chunk(doc: str, start: int = 0) -> ChunkRecord:
    return {"doc_id": doc, "char_start": start, "char_end": start + 10, "text": "x"}


def _span(doc: str, text: str) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": 0, "char_end": 10, "text": text}


def _filler(n: int) -> list[ChunkRecord]:
    return [_chunk("filler", 100 * index) for index in range(n)]


class _ByQuery:
    def __init__(self, hits: dict[str, list[ChunkRecord]]) -> None:
        self.hits = hits

    def retrieve(self, query: str, k: int) -> list[ChunkRecord]:
        return self.hits.get(query, [])[:k]


def _two_hop(item_id: str, question: str) -> EvidenceItem:
    return EvidenceItem(
        item_id,
        question,
        [_span("d1", f"{item_id}-a"), _span("d2", f"{item_id}-b")],
        "multi-hop",
    )


def test_query_prep_conversion_is_paired_by_raw_diagnosis_and_reports_budget_cost():
    query_item = _two_hop("mh-query", "query-q")
    budget_item = _two_hop("mh-budget", "budget-q")
    flat_item = EvidenceItem("flat", "flat-q", [_span("d1", "flat-a")], "factoid")
    shared_noise = _filler(10)
    store = _ByQuery(
        {
            "query-q": [_chunk("d1")],
            "query-hop-a": [_chunk("d1")],
            "query-hop-b": [_chunk("d2")],
            "budget-q": [_chunk("d1"), *_filler(28), _chunk("d2")],
            **{f"noise-{index}": shared_noise for index in range(5)},
            "mh-query-a": [_chunk("d1")],
            "mh-query-b": [_chunk("d2")],
            "mh-budget-a": [_chunk("d1")],
            "mh-budget-b": [_chunk("d2")],
        }
    )
    generated: list[str] = []

    def decompose(question: str) -> str:
        generated.append(question)
        if question == "query-q":
            return '["query-hop-a", "query-hop-b"]'
        return "\n".join(f"noise-{index}" for index in range(5))

    pipeline = QueryPrep.build(["decompose"], decomposer=decompose)
    report = compare_multihop_query_prep(
        store,
        [query_item, flat_item, budget_item],
        pipeline,
        budgets=_BUDGETS,
        resamples=32,
    )

    assert generated == ["query-q", "budget-q"]  # one model response defines every k and depth
    assert report["baseline"]["n_items"] == 2
    query = report["conversion"]["cohorts"][DIAGNOSIS_QUERY]
    assert (query["n"], query["all_spans_gained"], query["newly_reachable_at_depth"]) == (1, 1, 1)
    budget = report["conversion"]["cohorts"][DIAGNOSIS_BUDGET]
    assert (budget["n"], budget["span_coverage_regressed"]) == (1, 1)
    assert report["conversion"]["transitions"][DIAGNOSIS_QUERY][DIAGNOSIS_COVERED] == 1
    prepared = {item["item_id"]: item for item in report["prepared"]["items"]}
    assert prepared["mh-query"]["query_prep"]["query_subqueries"] == [
        "query-hop-a",
        "query-hop-b",
    ]

    text = format_query_prep_probe_report(report)
    assert text.isascii()
    assert "Query-diagnosed conversion: 1/1" in text
    assert "Budget-diagnosed cost: 1/1" in text
    assert "| query | 1 | 0 -> 1 | +1 / -0 |" in text
    assert "| mh-query | query -> covered | 0/1 | 0.500/1.000 |" in text
