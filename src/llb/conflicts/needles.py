"""Needle lane: which gold answers are reachable from more than one place in the corpus.

The needle-in-a-haystack test already asks whether a gold item's evidence can be found at all
(`annotate_needle_retrieval` scores exactly that). This asks the complementary question: can it be
found in more than one document? A needle whose answer span has a near-duplicate in another
document is ambiguous -- retrieval has two defensible answers to return, and whichever it ranks
first is luck rather than relevance.

That makes the non-unique needle set an independent corroboration of the tree's findings: it is
derived from the gold set rather than from the corpus geometry, so agreement between the two is
evidence, not circularity.
"""

import time
from dataclasses import dataclass

from llb.conflicts.constants import DEFAULT_COSINE_THRESHOLD
from llb.conflicts.vectorops import VectorSet
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.goldset.schema import GoldItem
from llb.rag.retrieval import chunk_hits_any


def _spans(item: GoldItem) -> list[SourceSpanRecord]:
    return [
        SourceSpanRecord(
            doc_id=span.doc_id,
            char_start=span.char_start,
            char_end=span.char_end,
            text=span.text,
        )
        for span in item.source_spans
    ]


@dataclass(frozen=True)
class NeedleAmbiguity:
    """One gold item and the foreign documents that also carry its answer."""

    item_id: str
    gold_chunks: int
    foreign_docs: list[str]
    max_similarity: float

    @property
    def is_ambiguous(self) -> bool:
        return bool(self.foreign_docs)

    def payload(self) -> JsonObject:
        return {
            "item_id": self.item_id,
            "gold_chunks": self.gold_chunks,
            "foreign_docs": self.foreign_docs,
            "max_similarity": round(self.max_similarity, 4),
            "ambiguous": self.is_ambiguous,
        }


def _gold_chunk_ordinals(item: GoldItem, chunks: list[ChunkRecord]) -> list[int]:
    """Ordinals of the chunks overlapping this item's gold spans."""
    spans = _spans(item)
    return [ordinal for ordinal, chunk in enumerate(chunks) if chunk_hits_any(chunk, spans)]


def _neighbours_by_ordinal(
    vectors: VectorSet, cos_threshold: float
) -> dict[int, list[tuple[int, float]]]:
    """Near-duplicate neighbours of every chunk, keyed both ways so lookup is one step."""
    neighbours: dict[int, list[tuple[int, float]]] = {}
    for left, right, similarity in vectors.pairs_above(cos_threshold):
        neighbours.setdefault(left, []).append((right, similarity))
        neighbours.setdefault(right, []).append((left, similarity))
    return neighbours


def _foreign_documents(
    gold: list[int],
    chunks: list[ChunkRecord],
    neighbours: dict[int, list[tuple[int, float]]],
) -> dict[str, float]:
    """Other documents carrying a near-duplicate of these gold chunks, with the best similarity."""
    foreign: dict[str, float] = {}
    for ordinal in gold:
        own_doc = chunks[ordinal]["doc_id"]
        for other, similarity in neighbours.get(ordinal, []):
            other_doc = chunks[other]["doc_id"]
            if other_doc != own_doc:
                foreign[other_doc] = max(foreign.get(other_doc, 0.0), similarity)
    return foreign


def analyze_needles(
    items: list[GoldItem],
    chunks: list[ChunkRecord],
    vectors: VectorSet,
    *,
    cos_threshold: float = DEFAULT_COSINE_THRESHOLD,
) -> tuple[list[NeedleAmbiguity], JsonObject]:
    """Flag each gold item whose answer span is near-duplicated in another document."""
    started = time.monotonic()
    neighbours = _neighbours_by_ordinal(vectors, cos_threshold)

    rows: list[NeedleAmbiguity] = []
    for item in items:
        gold = _gold_chunk_ordinals(item, chunks)
        foreign = _foreign_documents(gold, chunks, neighbours)
        rows.append(
            NeedleAmbiguity(
                item_id=item.id,
                gold_chunks=len(gold),
                foreign_docs=sorted(foreign),
                max_similarity=max(foreign.values(), default=0.0),
            )
        )

    ambiguous = [row for row in rows if row.is_ambiguous]
    unlocated = [row.item_id for row in rows if row.gold_chunks == 0]
    report: JsonObject = {
        "enabled": True,
        "cos_threshold": cos_threshold,
        "items": len(rows),
        "ambiguous_items": len(ambiguous),
        "non_unique_needle_fraction": (round(len(ambiguous) / len(rows), 4) if rows else 0.0),
        "unlocated_items": len(unlocated),
        "unlocated_ids": unlocated,
        "ambiguous_ids": [row.item_id for row in ambiguous],
        "seconds": round(time.monotonic() - started, 3),
    }
    return rows, report
