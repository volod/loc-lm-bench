"""GraphRAG backend residual 3 -- graph-vs-FAISS retrieval comparison core (`llb.rag.compare`).

Pure: driven by fake stores exposing the `.retrieve` seam, so it runs in the lightweight CI install
(no FAISS, no DuckDB, no GPU). The CLI wiring (`compare-retrieval`) layers real stores on top.
"""

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord


class _FakeStore:
    """A store that always returns the same fixed hits (truncated to k)."""

    def __init__(self, hits: list[ChunkRecord]) -> None:
        self._hits = hits

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        return self._hits[:k]


def _chunk(doc: str, start: int, end: int) -> ChunkRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "x"}


def _span(doc: str, start: int, end: int) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "g"}


def _items() -> list[tuple[str, list[SourceSpanRecord]]]:
    return [("питання", [_span("d1", 0, 10)])]


class _MetaStore(_FakeStore):
    """A store that also carries build meta, like a real `RagStore`."""

    def __init__(
        self,
        hits: list[ChunkRecord],
        duplicates: dict,
        collapse_duplicates: bool = True,
        strategy: str = "recursive",
    ) -> None:
        super().__init__(hits)
        self.meta = {
            "duplicates": duplicates,
            "collapse_duplicates": collapse_duplicates,
            "strategy": strategy,
        }
