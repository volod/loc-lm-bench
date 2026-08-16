"""Shared fixtures for the reranker bake-off tests: fake pools and fake cross-encoders.

The whole lane is exercised with a scorer that is a plain function of the chunk text, so ranking,
the rows, the fit gate, the paired verdict, and the report run with no download and no GPU.
"""

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.rerank_bakeoff.models import LoadedScorer

GOLD_TEXT = "GOLD"
MISS_TEXT = "MISS"

# Registered ids: the lane's roster screen resolves conventions, so tests name real candidates.
BASELINE = "BAAI/bge-reranker-v2-m3"
CANDIDATE = "mixedbread-ai/mxbai-rerank-base-v2"
REMOTE_CODE_CANDIDATE = "jinaai/jina-reranker-v2-base-multilingual"


def span(doc: str = "d1") -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": 0, "char_end": 10, "text": "g"}


def chunk(doc: str, start: int, text: str, score: float) -> ChunkRecord:
    return {
        "doc_id": doc,
        "char_start": start,
        "char_end": start + 10,
        "text": text,
        "retrieval_score": score,
        "rank": 0,
    }


def pool(gold_position: int, depth: int = 6) -> list[ChunkRecord]:
    """A candidate pool in retrieval order with the gold chunk at `gold_position` (1-based).

    A `gold_position` past `depth` is a pool that simply does not contain the evidence, which is the
    case no reranker can fix -- exactly what the reranker-off row exists to make visible.
    """
    records = []
    for rank in range(1, depth + 1):
        is_gold = rank == gold_position
        record = chunk(
            "d1" if is_gold else f"d{rank + 10}",
            0 if is_gold else 500 + rank,
            GOLD_TEXT if is_gold else MISS_TEXT,
            1.0 - rank / 100.0,
        )
        record["rank"] = rank
        records.append(record)
    return records


def items(n: int) -> list[tuple[str, list[SourceSpanRecord]]]:
    return [(f"питання-{i:02d}", [span()]) for i in range(n)]


def pools(gold_positions: list[int], depth: int = 6) -> list[list[ChunkRecord]]:
    return [pool(position, depth) for position in gold_positions]


def perfect_scorer(_question: str, texts: list[str]) -> list[float]:
    """Puts the gold chunk first for every item."""
    return [1.0 if text == GOLD_TEXT else 0.0 for text in texts]


def flat_scorer(_question: str, texts: list[str]) -> list[float]:
    """Scores everything identically: a stable sort, so retrieval order survives untouched."""
    return [0.0 for _ in texts]


def harmful_scorer(_question: str, texts: list[str]) -> list[float]:
    """Pushes the gold chunk to the bottom of the pool."""
    return [-1.0 if text == GOLD_TEXT else 0.0 for text in texts]


SCORERS = {
    BASELINE: flat_scorer,
    CANDIDATE: perfect_scorer,
    REMOTE_CODE_CANDIDATE: harmful_scorer,
}


def fake_loader(vram_mb: dict[str, float] | None = None, loaded: list[str] | None = None):
    """A `ScorerLoader` over the fake scorers, optionally reporting a per-model footprint."""
    footprints = vram_mb or {}

    def load(model: str) -> LoadedScorer:
        if loaded is not None:
            loaded.append(model)
        return LoadedScorer(
            scorer=SCORERS[model],
            device="cpu",
            load_seconds=0.5,
            vram_mb=footprints.get(model),
            read_vram=lambda: footprints.get(model),
        )

    return load
