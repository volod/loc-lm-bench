"""Embedding bake-off core (`llb.rag.embedding_bakeoff`).

Pure: fake stores expose the `.retrieve` + `.meta` seam and a fake store-builder stands in for the
heavy FAISS build, so scoring, ranking, the consent/open-data gate, and report shaping run in the
lightweight CI install (no GPU, no FAISS, no numpy, no network).
"""

from llb.core.contracts.rag import (
    ChunkRecord,
    SourceSpanRecord,
)
from llb.rag.embedding_bakeoff import run_bakeoff
from llb.rag.embedding_bakeoff_models import BuiltStore


class _FakeStore:
    """Returns fixed hits (truncated to k) and carries store meta (dim / n_indexed / model)."""

    def __init__(
        self, hits: list[ChunkRecord], *, dim: int = 8, n_indexed: int = 3, model: str = "m"
    ):
        self._hits = hits
        self.meta = {"dim": dim, "n_indexed": n_indexed, "embedding_model": model}

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        return self._hits[:k]


def _chunk(doc: str, start: int, end: int) -> ChunkRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "x"}


def _span(doc: str, start: int, end: int) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "g"}


def _items() -> list[tuple[str, list[SourceSpanRecord]]]:
    return [("питання", [_span("d1", 0, 10)])]


def _fixed_builder(store: _FakeStore):
    return lambda model: BuiltStore(store=store, embed_seconds=1.0, index_bytes=100)


def _scored(doc: str, start: int, score: float) -> ChunkRecord:
    chunk = _chunk(doc, start, start + 10)
    chunk["retrieval_score"] = score
    return chunk


class _PerQuestionStore:
    """Fake store whose ranking depends on the question, so items can differ within one lane."""

    def __init__(self, hits: dict[str, list[ChunkRecord]]):
        self._hits = hits
        self.meta = {"dim": 8, "n_indexed": 3, "embedding_model": "m"}

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        return self._hits[question][:k]


def _floor_items() -> list[tuple[str, list[SourceSpanRecord]]]:
    return [("резолюція", [_span("d1", 0, 10)]), ("протокол", [_span("d9", 0, 10)])]


_RESOLVED_ITEM = [_scored("d1", 0, 0.9), _scored("d2", 20, 0.2)]

_TIED_ITEM = [_scored("d7", 70, 0.5), _scored("d8", 80, 0.5), _scored("d9", 0, 0.5)]

_TIED_ITEM_REORDERED = [_TIED_ITEM[1], _TIED_ITEM[0], _TIED_ITEM[2]]

_RESOLVED_SECOND_ITEM = [_scored("d9", 0, 0.9), _scored("d7", 70, 0.2)]


def _floor_bakeoff(runner_up_second_item: list[ChunkRecord], k: int = 2):
    """A two-candidate bake-off with the floor measured over the same two items."""
    stores = {
        "cand-a": _PerQuestionStore({"резолюція": _RESOLVED_ITEM, "протокол": _TIED_ITEM}),
        "cand-b": _PerQuestionStore(
            {"резолюція": _RESOLVED_ITEM, "протокол": runner_up_second_item}
        ),
    }
    return run_bakeoff(
        _floor_items(),
        k=k,
        corpus_root="corpus",
        local_models=["cand-a", "cand-b"],
        build_local=lambda model: BuiltStore(
            store=stores[model], embed_seconds=1.0, index_bytes=100
        ),
        noise_floor=True,
        noise_floor_replicates=16,
    )
