"""Exact-duplicate chunk collapse (`llb.rag.duplicates`) and what it fixes downstream.

Pure unit tests over the committed `samples/corpora/duplicate_chunks_uk_v1/` fixture (three
Ukrainian manuals repeating the same page furniture) plus hand-built records: no FAISS, no GPU,
no embedder. The store-level tests that need a vector index live in `test_duplicates_store.py`.
"""

from pathlib import Path

import pytest

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.duplicates import (
    COUNT_KEY,
    OCCURRENCES_KEY,
    collapse_duplicate_chunks,
    duplicate_occurrences,
    duplicate_stats,
    expand_duplicate_chunks,
    format_duplicate_stats,
    occurrence_spans,
)
from llb.rag.retrieval import chunk_hits_span, recall_at_k
from llb.rag.store_build import _children_to_parents, order_by_score

FIXTURE = Path("samples/corpora/duplicate_chunks_uk_v1/corpus")

# The fixture's planted rate under `heading@400`; see its README -- these numbers ARE the fixture.
FIXTURE_STRATEGY, FIXTURE_SIZE, FIXTURE_OVERLAP = "heading", 400, 30
FIXTURE_CHUNKS, FIXTURE_UNIQUE, FIXTURE_GROUPS, FIXTURE_LARGEST = 12, 6, 3, 3


def fixture_chunks() -> list[ChunkRecord]:
    return chunk_corpus(FIXTURE, FIXTURE_STRATEGY, FIXTURE_SIZE, FIXTURE_OVERLAP)


def _chunk(doc: str, start: int, end: int, text: str, chunk_id: str) -> ChunkRecord:
    return {
        "doc_id": doc,
        "chunk_id": chunk_id,
        "char_start": start,
        "char_end": end,
        "text": text,
        "metadata": {},
    }


def _span(doc: str, start: int, end: int) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "g"}


def test_fixture_plants_the_documented_duplicate_rate():
    stats = duplicate_stats(fixture_chunks())
    assert (stats["n"], stats["unique"]) == (FIXTURE_CHUNKS, FIXTURE_UNIQUE)
    assert (stats["groups"], stats["largest_group"]) == (FIXTURE_GROUPS, FIXTURE_LARGEST)
    assert stats["duplicate_chunks"] == FIXTURE_GROUPS * FIXTURE_LARGEST
    assert stats["duplicate_share"] == pytest.approx(0.75)
    # every repeated group in this fixture is the same furniture shared ACROSS the three manuals
    assert stats["cross_document_groups"] == FIXTURE_GROUPS
    assert stats["intra_document_groups"] == 0


def test_collapse_indexes_each_distinct_text_once_and_stays_offset_exact():
    chunks = fixture_chunks()
    collapse = collapse_duplicate_chunks(chunks)
    assert len(collapse.chunks) == FIXTURE_UNIQUE
    assert len({c["text"] for c in collapse.chunks}) == FIXTURE_UNIQUE
    for survivor in collapse.chunks:
        source = (FIXTURE / str(survivor["doc_id"])).read_text(encoding="utf-8")
        assert source[survivor["char_start"] : survivor["char_end"]] == survivor["text"]
        for copy in duplicate_occurrences(survivor):
            other = (FIXTURE / str(copy["doc_id"])).read_text(encoding="utf-8")
            assert other[copy["char_start"] : copy["char_end"]] == survivor["text"]


def test_collapse_keeps_the_first_copy_and_records_the_rest():
    chunks = [
        _chunk("a.md", 0, 5, "furniture", "a#0"),
        _chunk("a.md", 5, 9, "unique", "a#1"),
        _chunk("b.md", 3, 8, "furniture", "b#0"),
        _chunk("c.md", 7, 12, "furniture", "c#0"),
    ]
    survivors = collapse_duplicate_chunks(chunks).chunks
    assert [c["chunk_id"] for c in survivors] == ["a#0", "a#1"]
    metadata = survivors[0]["metadata"]
    assert metadata[COUNT_KEY] == 3  # the survivor plus its two copies
    assert [copy["chunk_id"] for copy in metadata[OCCURRENCES_KEY]] == ["b#0", "c#0"]
    assert "text" not in metadata[OCCURRENCES_KEY][0]  # identical to the survivor's by design
    assert survivors[1]["metadata"] == {}  # a chunk with no copies is left alone


def test_collapse_leaves_a_duplicate_free_set_byte_identical():
    chunks = [_chunk("a.md", 0, 5, "one", "a#0"), _chunk("b.md", 0, 5, "two", "b#0")]
    collapse = collapse_duplicate_chunks(chunks)
    assert collapse.chunks == chunks
    assert collapse.kept == [0, 1]
    assert collapse.stats["collapsed"] == 0
    assert collapse.stats["largest_group"] == 1


def test_kept_positions_track_the_input_rows():
    chunks = [
        _chunk("a.md", 0, 5, "dup", "a#0"),
        _chunk("b.md", 0, 5, "dup", "b#0"),
        _chunk("c.md", 0, 3, "own", "c#0"),
    ]
    assert collapse_duplicate_chunks(chunks).kept == [0, 2]


def test_a_survivor_still_hits_a_span_labeled_on_a_collapsed_copy():
    chunks = [
        _chunk("a.md", 0, 5, "furniture", "a#0"),
        _chunk("b.md", 30, 35, "furniture", "b#0"),
    ]
    survivor = collapse_duplicate_chunks(chunks).chunks[0]
    assert occurrence_spans(survivor) == [survivor, *duplicate_occurrences(survivor)]
    assert chunk_hits_span(survivor, _span("a.md", 1, 3))  # its own place
    assert chunk_hits_span(survivor, _span("b.md", 31, 33))  # the collapsed copy's place
    assert not chunk_hits_span(survivor, _span("b.md", 0, 5))  # not everywhere in that doc
    assert not chunk_hits_span(survivor, _span("c.md", 0, 5))  # not in a doc it never appears in
    assert recall_at_k([survivor], [_span("b.md", 31, 33)], 10) == 1.0


def test_uncollapsed_chunk_matching_is_unchanged():
    chunk = _chunk("a.md", 0, 5, "text", "a#0")
    assert occurrence_spans(chunk) == [chunk]
    assert chunk_hits_span(chunk, _span("a.md", 4, 9))
    assert not chunk_hits_span(chunk, _span("a.md", 5, 9))  # half-open, no touching-edge hit


def test_expand_reverses_collapse_exactly():
    chunks = fixture_chunks()
    survivors = collapse_duplicate_chunks(chunks).chunks
    expanded, rows = expand_duplicate_chunks(survivors)
    by_doc = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc_id"], []).append(chunk)
    restored = {}
    for chunk in expanded:
        restored.setdefault(chunk["doc_id"], []).append(chunk)
    assert restored == by_doc  # every copy back in its document, in build order
    assert len(rows) == len(expanded)
    # every expanded record points at the row its text was indexed under
    for chunk, row in zip(expanded, rows):
        assert chunk["text"] == survivors[row]["text"]


def test_expand_passes_through_an_uncollapsed_store():
    chunks = [_chunk("a.md", 0, 5, "one", "a#0"), _chunk("b.md", 0, 5, "two", "b#0")]
    expanded, rows = expand_duplicate_chunks(chunks)
    assert expanded == chunks
    assert rows == [0, 1]


def test_order_by_score_breaks_exact_ties_on_chunk_id_not_backend_order():
    chunks = [_chunk("a.md", 0, 5, "t", cid) for cid in ("c#0", "a#0", "b#0")]
    backend_order = [(0, 0.5), (1, 0.5), (2, 0.5)]
    shuffled = [(2, 0.5), (0, 0.5), (1, 0.5)]
    assert order_by_score(backend_order, chunks) == order_by_score(shuffled, chunks)
    assert [i for i, _ in order_by_score(backend_order, chunks)] == [1, 2, 0]  # a#0, b#0, c#0


def test_order_by_score_keeps_the_score_ranking():
    chunks = [_chunk("a.md", 0, 5, "t", cid) for cid in ("z#0", "a#0")]
    assert order_by_score([(1, 0.1), (0, 0.9)], chunks) == [(0, 0.9), (1, 0.1)]


def test_a_collapsed_child_surfaces_every_occurrence_parent():
    parents = {pid: _chunk(f"{pid}.md", 0, 100, "parent", pid) for pid in ("p1", "p2", "p3")}
    children = [
        {**_chunk("p1.md", 0, 9, "furniture", "p1::c0"), "parent_id": "p1"},
        {**_chunk("p2.md", 0, 9, "furniture", "p2::c0"), "parent_id": "p2"},
        {**_chunk("p3.md", 0, 6, "unique", "p3::c0"), "parent_id": "p3"},
    ]
    survivors = collapse_duplicate_chunks(children).chunks
    out = _children_to_parents(survivors, parents)
    assert [p["chunk_id"] for p in out] == ["p1", "p2", "p3"]
    assert [p["rank"] for p in out] == [1, 2, 3]


def test_format_duplicate_stats_reports_both_dispositions():
    stats = duplicate_stats(fixture_chunks())
    collapsed = format_duplicate_stats(stats)
    assert "9/12 chunks (75.0%)" in collapsed
    assert "largest 3 copies" in collapsed
    assert "6 indexed (6 collapsed)" in collapsed
    assert "0 intra-document, 3 cross-document" in collapsed
    kept = format_duplicate_stats(stats, collapsed=False)
    assert "all 12 indexed" in kept
    assert "chunk_id" in kept  # the tie-break the operator gets instead


def test_duplicate_stats_split_intra_from_cross_document():
    chunks = [
        _chunk("a.md", 0, 3, "hdr", "a#0"),  # same text in two documents -> cross
        _chunk("b.md", 0, 3, "hdr", "b#0"),
        _chunk("a.md", 3, 6, "boi", "a#1"),  # same text twice in ONE document -> intra
        _chunk("a.md", 6, 9, "boi", "a#2"),
        _chunk("a.md", 9, 12, "boi", "a#3"),
    ]
    stats = duplicate_stats(chunks)
    assert (stats["groups"], stats["intra_document_groups"], stats["cross_document_groups"]) == (
        2,
        1,
        1,
    )
    # the collapse path measures the same split from the survivors' folded occurrences
    assert collapse_duplicate_chunks(chunks).stats["intra_document_groups"] == 1
