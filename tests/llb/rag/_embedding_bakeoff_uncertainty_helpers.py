"""Paired sampling uncertainty of the embedder bake-off (`embedding_bakeoff_uncertainty`).

Pure: per-item metric vectors, the shared-index paired bootstrap, the adopt-or-retain verdict, and
the report columns all run over fake stores and plain vectors -- no FAISS, no GPU, no numpy.
"""

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord


from llb.rag.embedding_bakeoff.run import run_bakeoff


from llb.rag.embedding_bakeoff.models import BuiltStore


from llb.rag.embedding_bakeoff.uncertainty import (
    METRIC_MRR,
    METRIC_RECALL,
)


# Real roster ids, not placeholders: `compare-embeddings` screens its roster against the
# convention registry (`llb.rag.embedding_bakeoff.roster`), so a CLI-level test has to name
# candidates whose query/passage format is actually declared.
BASELINE = "intfloat/multilingual-e5-base"
CLI_CANDIDATE = "BAAI/bge-m3"


def _chunk(doc: str, start: int) -> ChunkRecord:
    return {"doc_id": doc, "char_start": start, "char_end": start + 10, "text": "x"}


def _span(doc: str) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": 0, "char_end": 10, "text": "g"}


class _HitSetStore:
    """Retrieves the gold chunk for the questions in `hits`, and a miss for every other."""

    def __init__(self, hits: set[str]):
        self._hits = hits
        self.meta = {"dim": 8, "n_indexed": 3, "embedding_model": "m"}

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        gold = _chunk("d1", 0)
        miss = _chunk("d9", 500)
        return ([gold, miss] if question in self._hits else [miss, gold])[:k]


def _questions(n: int) -> list[str]:
    return [f"питання-{i:02d}" for i in range(n)]


def _items(n: int) -> list[tuple[str, list[SourceSpanRecord]]]:
    return [(question, [_span("d1")]) for question in _questions(n)]


def _vectors(recall: list[float], mrr: list[float] | None = None) -> dict[str, list[float]]:
    return {METRIC_RECALL: recall, METRIC_MRR: mrr if mrr is not None else list(recall)}


def _bakeoff(baseline: str | None = BASELINE, n: int = 20):
    items = _items(n)
    questions = _questions(n)
    stores = {
        BASELINE: _HitSetStore(set(questions[:4])),
        "cand": _HitSetStore(set(questions[:14])),
    }
    return run_bakeoff(
        items,
        k=1,  # k=1 so a miss is a miss: the store puts the gold chunk second
        corpus_root="corpus",
        local_models=[BASELINE, "cand"],
        build_local=lambda model: BuiltStore(
            store=stores[model], embed_seconds=1.0, index_bytes=100
        ),
        baseline=baseline,
        resamples=500,
    )
