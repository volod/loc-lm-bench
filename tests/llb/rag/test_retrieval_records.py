"""The persisted retrieved-span record (`llb.rag.retrieval_records`).

Pure: hand-built chunks and records, no store, no embedder. The properties under test are the
ones the sidecar readers depend on -- an uncollapsed chunk records exactly what it always did, a
collapsed one records enough that recomputing its metric from the sidecar agrees with the run,
and the list stays bounded however often the corpus repeats a passage.
"""

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.duplicates import collapse_duplicate_chunks
from llb.rag.retrieval import chunk_hits_any, recall_at_k
from llb.rag.retrieval_records import (
    RETRIEVED_OCCURRENCE_LIMIT,
    RETRIEVED_TEXT_PREVIEW_CHARS,
    record_as_chunk,
    record_documents,
    retrieved_span,
)

FURNITURE = "повторюваний колонтитул"


def _chunk(doc: str, start: int, text: str = FURNITURE, chunk_id: str | None = None) -> ChunkRecord:
    return {
        "doc_id": doc,
        "chunk_id": chunk_id or f"{doc}#0000",
        "char_start": start,
        "char_end": start + len(text),
        "text": text,
        "metadata": {"pages": [1, 1]},
    }


def _span(doc: str, start: int, end: int) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "еталон"}


def _collapsed(n_copies: int) -> ChunkRecord:
    """One survivor standing for `n_copies` documents (a.md, b.md, ...)."""
    copies = [_chunk(f"doc{i:02d}.md", i * 10) for i in range(n_copies)]
    return collapse_duplicate_chunks(copies).chunks[0]


def test_an_uncollapsed_chunk_records_exactly_what_it_always_did():
    chunk = _chunk("a.md", 0, text="т" * 500)
    chunk["retrieval_score"] = 0.9
    assert retrieved_span(chunk, 1, [_span("a.md", 0, 5)]) == {
        "doc_id": "a.md",
        "char_start": 0,
        "char_end": 500,
        "rank": 1,
        "text_preview": "т" * RETRIEVED_TEXT_PREVIEW_CHARS,
        "retrieval_score": 0.9,
    }


def test_a_collapsed_chunk_records_its_other_places():
    record = retrieved_span(_collapsed(3), 2, [])
    assert record["duplicate_count"] == 3  # the survivor plus its two copies
    assert [copy["doc_id"] for copy in record["duplicate_occurrences"]] == ["doc01.md", "doc02.md"]
    copy = record["duplicate_occurrences"][0]
    assert copy == {
        "doc_id": "doc01.md",
        "char_start": 10,
        "char_end": 10 + len(FURNITURE),
        "chunk_id": "doc01.md#0000",
    }
    assert "metadata" not in copy and "text" not in copy  # the place only; the store keeps the rest


def test_the_occurrence_list_stays_bounded_on_a_heavily_repeated_passage():
    record = retrieved_span(_collapsed(58), 1, [])
    assert record["duplicate_count"] == 58  # the true total is always stated
    assert len(record["duplicate_occurrences"]) == RETRIEVED_OCCURRENCE_LIMIT


def test_the_bound_never_drops_an_occurrence_the_items_metric_depends_on():
    """A gold span on the 40th copy must survive the bound, or the sidecar would report a miss."""
    gold = [_span("doc40.md", 400, 406)]
    record = retrieved_span(_collapsed(58), 1, gold)
    documents = [copy["doc_id"] for copy in record["duplicate_occurrences"]]
    assert "doc40.md" in documents
    assert len(record["duplicate_occurrences"]) == RETRIEVED_OCCURRENCE_LIMIT
    assert documents == sorted(documents)  # kept in build order, not gold-first
    assert chunk_hits_any(record_as_chunk(record), gold)


def test_a_record_read_back_hits_every_place_it_recorded():
    record = retrieved_span(_collapsed(3), 1, [])
    chunk = record_as_chunk(record)
    assert chunk_hits_any(chunk, [_span("doc00.md", 0, 5)])  # the survivor's own place
    assert chunk_hits_any(chunk, [_span("doc02.md", 20, 25)])  # a collapsed copy's place
    assert not chunk_hits_any(chunk, [_span("doc02.md", 0, 5)])  # not everywhere in that doc
    assert recall_at_k([chunk], [_span("doc01.md", 12, 14)], 10) == 1.0


def test_an_uncollapsed_record_reads_back_without_occurrence_metadata():
    chunk = record_as_chunk(retrieved_span(_chunk("a.md", 0), 1, []))
    assert "metadata" not in chunk
    assert chunk_hits_any(chunk, [_span("a.md", 1, 3)])
    assert not chunk_hits_any(chunk, [_span("b.md", 1, 3)])


def test_record_documents_lists_every_place_once_in_record_order():
    assert record_documents(retrieved_span(_collapsed(3), 1, [])) == [
        "doc00.md",
        "doc01.md",
        "doc02.md",
    ]
    assert record_documents(retrieved_span(_chunk("a.md", 0), 1, [])) == ["a.md"]


def test_two_copies_inside_one_document_collapse_to_one_document_entry():
    copies = [_chunk("a.md", 0), _chunk("a.md", 90, chunk_id="a.md#0009"), _chunk("b.md", 0)]
    record = retrieved_span(collapse_duplicate_chunks(copies).chunks[0], 1, [])
    assert record["duplicate_count"] == 3
    assert record_documents(record) == ["a.md", "b.md"]
