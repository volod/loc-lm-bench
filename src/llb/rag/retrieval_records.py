"""The persisted retrieved-span record: build it from a live chunk, read it back as a chunk.

`run-eval` writes one `retrieval.jsonl` line per scored case, and several lanes recompute
retrieval metrics from that sidecar instead of from the live store -- miss classification
(`llb.board.miss_analysis`) and multi-span answer coverage (`llb.eval.answer_quality.coverage`).
Those recomputations must agree with the metric the run itself reported, which is why the record
carries the DUPLICATE OCCURRENCES of a collapsed chunk (`llb.rag.duplicates`): a chunk that stands
for the same text in several documents hits a gold span labeled at any of them, and a record that
kept only the surviving copy's offsets would report a retrieval miss the run did not have.

The occurrence list is bounded, because a converted-PDF corpus can repeat one passage dozens of
times and the sidecar is written per case per hit. The bound is content-aware rather than blind:
every occurrence that overlaps one of the item's OWN gold spans is kept (so the recomputation
stays exact), the remaining slots go to the first other occurrences in build order, and
`duplicate_count` always states the true total -- so a reader can say "also in 3 shown of 58
places" without the record growing with the corpus.
"""

from llb.core.contracts.rag import (
    ChunkRecord,
    RetrievedOccurrence,
    RetrievedSpanRecord,
    SourceSpanRecord,
)
from llb.rag.duplicates import (
    COUNT_KEY,
    OCCURRENCES_KEY,
    DuplicateOccurrence,
    duplicate_occurrences,
)
from llb.rag.retrieval import chunk_hits_any

# Bounded per-chunk text carried into `retrieval.jsonl` for observability; the span coordinates
# (not the text) drive the miss classifier, so the preview stays small like `answer_preview`.
RETRIEVED_TEXT_PREVIEW_CHARS = 160

# How many other places of a collapsed chunk the record shows. Gold-overlapping occurrences are
# never dropped, so this is a floor on the list length, not a hard cap.
RETRIEVED_OCCURRENCE_LIMIT = 8


def retrieved_span(
    chunk: ChunkRecord, rank: int, gold_spans: list[SourceSpanRecord] | None = None
) -> RetrievedSpanRecord:
    """One persisted retrieved-span record. A chunk with no collapsed copies is unchanged."""
    record: RetrievedSpanRecord = {
        "doc_id": str(chunk.get("doc_id", "")),
        "char_start": int(chunk.get("char_start", 0)),
        "char_end": int(chunk.get("char_end", 0)),
        "rank": rank,
        "text_preview": str(chunk.get("text", ""))[:RETRIEVED_TEXT_PREVIEW_CHARS],
    }
    score = chunk.get("retrieval_score")
    if score is not None:
        record["retrieval_score"] = float(score)
    copies = duplicate_occurrences(chunk)
    if copies:
        record["duplicate_count"] = len(copies) + 1
        record["duplicate_occurrences"] = _bounded_occurrences(copies, gold_spans or [])
    return record


def record_as_chunk(record: RetrievedSpanRecord) -> ChunkRecord:
    """A persisted record read back as a `ChunkRecord`, occurrences restored.

    Span matching (`llb.rag.retrieval.chunk_hits_span`) reads occurrences off `metadata`, so a
    lane that recomputes a metric from the sidecar sees the same places the live chunk did.
    """
    chunk: ChunkRecord = {
        "doc_id": str(record.get("doc_id", "")),
        "char_start": int(record.get("char_start", 0)),
        "char_end": int(record.get("char_end", 0)),
        "text": str(record.get("text_preview", "")),
    }
    copies = record.get("duplicate_occurrences")
    if copies:
        chunk["metadata"] = {
            OCCURRENCES_KEY: list(copies),
            COUNT_KEY: int(record.get("duplicate_count", len(copies) + 1)),
        }
    return chunk


def record_documents(record: RetrievedSpanRecord) -> list[str]:
    """The documents one retrieved row stands for: its own, plus those of its shown occurrences.

    An operator asking "did my context come from the document I expected?" needs every place, not
    the copy the build happened to keep -- and for a collapsed chunk those are different documents.
    Order is the record's own; each document appears once.
    """
    documents: list[str] = []
    copies = record.get("duplicate_occurrences") or []
    for doc_id in [str(record.get("doc_id", "")), *(str(c.get("doc_id", "")) for c in copies)]:
        if doc_id and doc_id not in documents:
            documents.append(doc_id)
    return documents


def _bounded_occurrences(
    copies: list[DuplicateOccurrence], gold_spans: list[SourceSpanRecord]
) -> list[RetrievedOccurrence]:
    """Gold-overlapping copies plus the first others, back in build order, projected to a span."""
    required = {
        index
        for index, copy in enumerate(copies)
        if chunk_hits_any(_as_chunk(copy), gold_spans)  # keeps the sidecar metric exact
    }
    kept = set(required)
    for index in range(len(copies)):
        if len(kept) >= max(RETRIEVED_OCCURRENCE_LIMIT, len(required)):
            break
        kept.add(index)
    return [_projected(copies[index]) for index in sorted(kept)]


def _as_chunk(occurrence: DuplicateOccurrence) -> ChunkRecord:
    return {
        "doc_id": str(occurrence.get("doc_id", "")),
        "char_start": int(occurrence.get("char_start", 0)),
        "char_end": int(occurrence.get("char_end", 0)),
        "text": "",
    }


def _projected(occurrence: DuplicateOccurrence) -> RetrievedOccurrence:
    """Only the place, never the collapsed copy's own metadata -- the store keeps the rest."""
    projected: RetrievedOccurrence = {
        "doc_id": str(occurrence.get("doc_id", "")),
        "char_start": int(occurrence.get("char_start", 0)),
        "char_end": int(occurrence.get("char_end", 0)),
    }
    chunk_id = occurrence.get("chunk_id")
    if chunk_id is not None:
        projected["chunk_id"] = str(chunk_id)
    return projected
