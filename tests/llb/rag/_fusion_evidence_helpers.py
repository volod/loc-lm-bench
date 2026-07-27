"""graph-vector-fusion-multihop-evidence -- the multi-hop fusion evidence lane.

Pure: driven by fake stores exposing the `.retrieve` seam, so it runs in the lightweight CI
install (no FAISS, no DuckDB, no GPU). The CLI wiring layers real stores on top.
"""

from llb.core.contracts.rag import (
    ChunkRecord,
    SourceSpanRecord,
)
from llb.rag.fusion_evidence import (
    EvidenceItem,
    build_sweep_rows,
    evaluate_fusion_evidence,
)
from llb.rag.fusion_evidence.rows import VECTOR_ROW


def _chunk(doc: str, start: int, end: int) -> ChunkRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "x"}


def _span(doc: str, start: int, end: int) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "g"}


class _ByQuestion:
    """A store returning a fixed ranking per question (truncated to k)."""

    def __init__(self, hits: dict[str, list[ChunkRecord]]) -> None:
        self.hits = hits
        self.calls = 0

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        self.calls += 1
        return self.hits.get(question, [])[:k]


def _multi_hop_item(item_id: str, question: str) -> EvidenceItem:
    return EvidenceItem(item_id, question, [_span("d1", 0, 10), _span("d2", 0, 10)], "multi-hop")


_MULTI_HOP_N = 6


def _fusion_report(multi_hop: int = _MULTI_HOP_N, **kwargs):
    """Multi-hop items the vector lane half-covers and the graph lane completes, plus a factoid."""
    questions = [f"q{i}" for i in range(multi_hop)]
    items = [
        *(_multi_hop_item(f"mh-{i}", question) for i, question in enumerate(questions)),
        EvidenceItem("f-1", "qf", [_span("d1", 0, 10)], "factoid"),
    ]
    vector = _ByQuestion({q: [_chunk("d1", 0, 10)] for q in [*questions, "qf"]})
    graph = _ByQuestion({q: [_chunk("d2", 0, 10)] for q in questions} | {"qf": []})
    rows = build_sweep_rows(
        vector, {"local_khop": graph}, [i.question for i in items], k=10, weights=(0.0, 0.3)
    )
    return evaluate_fusion_evidence(rows, items, 10, baseline=VECTOR_ROW, resamples=100, **kwargs)


def _scored(doc: str, start: int, score: float) -> ChunkRecord:
    chunk = _chunk(doc, start, start + 10)
    chunk["retrieval_score"] = score
    return chunk


def _floor_rows(pool_depth: int = 0):
    """A one-strategy sweep whose lanes carry scores, so every row has a floor to measure."""
    vector = _ByQuestion({"q": [_scored("d1", 0, 0.9), *(_scored("dv", i, 0.5) for i in range(9))]})
    graph = _ByQuestion({"q": [_scored("d2", 0, 0.8), *(_scored("dg", i, 0.4) for i in range(9))]})
    return build_sweep_rows(
        vector,
        {"local_khop": graph},
        ["q"],
        k=3,
        weights=(0.0, 0.3, 1.0),
        pool_depth=pool_depth,
    )


def _identities(hits: list[ChunkRecord]) -> list[tuple[str, int]]:
    return [(hit["doc_id"], hit["char_start"]) for hit in hits]
