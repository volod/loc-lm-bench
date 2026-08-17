"""multihop-both-hops-ceiling -- the per-hop retrievability probe.

Pure: driven by fake stores exposing the `.retrieve` seam, so it runs in the lightweight CI
install (no FAISS, no DuckDB, no GPU). The CLI wiring layers real lane stores on top.
"""

import pytest

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.multihop_probe import (
    DIAGNOSIS_BUDGET,
    DIAGNOSIS_COVERED,
    DIAGNOSIS_QUERY,
    DIAGNOSIS_UNREACHABLE,
    EXPLANATION_BUDGET,
    EXPLANATION_MIXED,
    EXPLANATION_NONE,
    EXPLANATION_QUERY,
    EvidenceItem,
    format_probe_report,
    parse_budgets,
    probe_multihop_hops,
)

_BUDGETS = (10, 25, 50)


def _chunk(doc: str, start: int = 0) -> ChunkRecord:
    return {"doc_id": doc, "char_start": start, "char_end": start + 10, "text": "x"}


def _span(doc: str, text: str) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": 0, "char_end": 10, "text": text}


def _filler(n: int) -> list[ChunkRecord]:
    """Chunks that hit no labeled span, so a hop's rank is exactly where it is placed."""
    return [_chunk("filler", 100 * i) for i in range(n)]


class _ByQuery:
    """A store returning a fixed ranking per query string (truncated to k)."""

    def __init__(self, hits: dict[str, list[ChunkRecord]]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[ChunkRecord]:
        self.calls.append((query, k))
        return self.hits.get(query, [])[:k]


def _two_hop(item_id: str, question: str) -> EvidenceItem:
    return EvidenceItem(
        item_id, question, [_span("d1", f"{item_id}-a"), _span("d2", f"{item_id}-b")], "multi-hop"
    )


def _at(rank: int, chunk: ChunkRecord) -> list[ChunkRecord]:
    return [*_filler(rank - 1), chunk]


def _probe(store, items, **kwargs):
    return probe_multihop_hops(store, items, budgets=_BUDGETS, resamples=32, **kwargs)


def test_a_hop_below_the_cut_is_a_budget_diagnosis_and_the_curve_shows_where_it_arrives():
    item = _two_hop("mh-1", "q")
    store = _ByQuery(
        {
            "q": [_chunk("d1"), *_filler(28), _chunk("d2")],  # second hop at rank 30
            "mh-1-a": [_chunk("d1")],
            "mh-1-b": [_chunk("d2")],
        }
    )
    report = _probe(store, [item])
    probe = report["items"][0]
    assert [hop["question_rank"] for hop in probe["hops"]] == [1, 30]
    assert probe["diagnosis"] == DIAGNOSIS_BUDGET
    assert probe["limiting_rank"] == 30
    assert probe["min_budget"] == 50
    curve = {point["k"]: point for point in report["slices"]["multi-hop"]["curve"]}
    assert curve[10]["all_spans_at_k"]["mean"] == 0.0
    assert curve[25]["all_spans_at_k"]["mean"] == 0.0
    assert curve[50]["all_spans_at_k"]["mean"] == 1.0
    assert curve[10]["recall_at_k"] == 1.0  # the flat metric never saw the miss
    assert (curve[10]["hop_hit_rate"], curve[50]["hop_hit_rate"]) == (0.5, 1.0)


def test_a_hop_only_its_own_text_reaches_is_a_query_diagnosis():
    item = _two_hop("mh-1", "q")
    store = _ByQuery(
        {
            "q": [_chunk("d1")],  # the question never returns the second hop at any depth
            "mh-1-a": [_chunk("d1")],
            "mh-1-b": [_chunk("d2")],  # its own text reaches it at rank 1
        }
    )
    report = _probe(store, [item])
    probe = report["items"][0]
    assert [hop["question_rank"] for hop in probe["hops"]] == [1, None]
    assert [hop["span_query_rank"] for hop in probe["hops"]] == [1, 1]
    assert probe["diagnosis"] == DIAGNOSIS_QUERY
    assert probe["min_budget"] == "beyond"
    diagnosis = report["slices"]["multi-hop"]["diagnosis"]
    assert diagnosis["explanation"] == EXPLANATION_QUERY
    assert diagnosis["budget_histogram"]["beyond"] == 1


def test_a_hop_no_query_form_reaches_is_neither_budget_nor_query():
    item = _two_hop("mh-1", "q")
    store = _ByQuery({"q": [_chunk("d1")], "mh-1-a": [_chunk("d1")], "mh-1-b": []})
    report = _probe(store, [item])
    assert report["items"][0]["diagnosis"] == DIAGNOSIS_UNREACHABLE


def test_a_hop_its_own_text_reaches_only_below_the_operating_budget_is_not_a_query_diagnosis():
    """The control is read AT the operating budget: an ideal query that also needs depth 30 does
    not show that a decomposed question would have delivered the hop."""
    item = _two_hop("mh-1", "q")
    store = _ByQuery(
        {
            "q": [_chunk("d1")],
            "mh-1-a": [_chunk("d1")],
            "mh-1-b": _at(30, _chunk("d2")),
        }
    )
    report = _probe(store, [item])
    assert report["items"][0]["diagnosis"] == DIAGNOSIS_UNREACHABLE


def test_an_item_carrying_every_hop_at_the_operating_budget_explains_nothing():
    item = _two_hop("mh-1", "q")
    store = _ByQuery(
        {
            "q": [_chunk("d1"), _chunk("d2")],
            "mh-1-a": [_chunk("d1")],
            "mh-1-b": [_chunk("d2")],
        }
    )
    report = _probe(store, [item])
    diagnosis = report["slices"]["multi-hop"]["diagnosis"]
    assert report["items"][0]["diagnosis"] == DIAGNOSIS_COVERED
    assert diagnosis["failing_items"] == 0
    assert diagnosis["explanation"] == EXPLANATION_NONE
    assert diagnosis["budget_histogram"]["10"] == 1


def _mixed_store(items: list[EvidenceItem], deep: dict[str, list[ChunkRecord]]) -> _ByQuery:
    hits = dict(deep)
    for item in items:
        for span in item.spans:
            hits.setdefault(span["text"], [_chunk(span["doc_id"])])
    return _ByQuery(hits)


def test_the_majority_failing_diagnosis_names_the_explanation():
    items = [_two_hop(f"mh-{i}", f"q{i}") for i in range(3)]
    deep = {
        "q0": [_chunk("d1"), *_filler(28), _chunk("d2")],  # budget
        "q1": [_chunk("d1"), *_filler(13), _chunk("d2")],  # budget
        "q2": [_chunk("d1")],  # query
    }
    report = _probe(_mixed_store(items, deep), items)
    diagnosis = report["slices"]["multi-hop"]["diagnosis"]
    assert diagnosis["counts"] == {"covered": 0, "budget": 2, "query": 1, "unreachable": 0}
    assert diagnosis["explanation"] == EXPLANATION_BUDGET
    assert diagnosis["budget_histogram"] == {"10": 0, "25": 1, "50": 1, "200": 0, "beyond": 1}


def test_a_tie_between_two_causes_is_reported_as_mixed_not_as_a_winner():
    items = [_two_hop(f"mh-{i}", f"q{i}") for i in range(2)]
    deep = {"q0": [_chunk("d1"), *_filler(28), _chunk("d2")], "q1": [_chunk("d1")]}
    report = _probe(_mixed_store(items, deep), items)
    assert report["slices"]["multi-hop"]["diagnosis"]["explanation"] == EXPLANATION_MIXED


def test_only_the_focus_slice_is_laddered_into_the_item_ledger_but_every_slice_is_measured():
    multi = _two_hop("mh-1", "q0")
    factoid = EvidenceItem("f-1", "q1", [_span("d1", "f-1-a")], "factoid")
    store = _mixed_store([multi, factoid], {"q0": [_chunk("d1")], "q1": [_chunk("d1")]})
    report = _probe(store, [multi, factoid])
    assert [item["item_id"] for item in report["items"]] == ["mh-1"]
    assert report["slices"]["factoid"]["n"] == 1
    assert report["slices"]["factoid"]["curve"][0]["all_spans_at_k"]["mean"] == 1.0
    assert report["n_items"] == 2
    assert report["overall"]["n"] == 2  # the focus slice is read against every scored item
    assert report["overall"]["n_hops"] == 3
    assert report["overall"]["curve"][0]["all_spans_at_k"]["mean"] == 0.5


def test_an_empty_focus_slice_is_rejected_instead_of_reporting_every_item_covered():
    factoid = EvidenceItem("f-1", "q1", [_span("d1", "f-1-a")], "factoid")
    with pytest.raises(ValueError, match="probe focus slice is empty: multi-hop"):
        _probe(_ByQuery({}), [factoid])


def test_each_budget_is_retrieved_at_that_budget_and_the_depth_covers_the_widest_one():
    item = _two_hop("mh-1", "q")
    store = _ByQuery({"q": [_chunk("d1")], "mh-1-a": [_chunk("d1")], "mh-1-b": [_chunk("d2")]})
    report = probe_multihop_hops(store, [item], budgets=(50, 10), probe_depth=1, resamples=0)
    assert report["budgets"] == [10, 50]
    assert report["probe_depth"] == 50  # never shallower than the widest compared budget
    question_calls = [k for query, k in store.calls if query == "q"]
    assert sorted(question_calls) == [10, 50, 50]  # one per budget, plus the deep pass


def test_the_report_states_the_curve_the_ledger_and_the_named_explanation_in_ascii():
    items = [_two_hop(f"mh-{i}", f"q{i}") for i in range(2)]
    deep = {"q0": [_chunk("d1"), *_filler(28), _chunk("d2")], "q1": [_chunk("d1")]}
    text = format_probe_report(_probe(_mixed_store(items, deep), items))
    assert text.isascii()
    assert "| k | all-spans@k |" in text
    assert "Explanation supported: mixed" in text
    assert "| every item | 2 | 0.000 | 0.000 | 0.500 | 0/1/1/0 | mixed |" in text
    assert "| mh-0 | 2 | 1 / 30 | 1 / 1 | 50 | budget |" in text
    assert "| mh-1 | 2 | 1 / - | 1 / 1 | never | query |" in text
    assert "missing a hop at k=10" in text


def test_budget_grid_parsing_sorts_deduplicates_and_rejects_a_non_budget():
    assert parse_budgets("50, 10,25,10") == (10, 25, 50)
    for bad in ("", "0", "k", "10,-5"):
        with pytest.raises(ValueError):
            parse_budgets(bad)
