"""Contiguous-chunk stitching (`llb.rag.stitching`): what merges, what refuses, and what it costs.

Pure: plain chunk dicts through the `.retrieve` seam, so it runs in the lightweight CI install
(no FAISS, no GPU). The lever's whole claim is that it reflows evidence without changing it, so
these tests pin BOTH sides of that: the merges that convert fragments into whole spans, and the
retrieval metrics that must not move when they happen.
"""

from llb.rag import retrieval
from llb.rag.stitching import (
    STITCHED_FROM_KEY,
    StitchingRetriever,
    served_chars,
    stitch_contiguous,
)

DOC = "".join(str(index % 10) for index in range(200))


def chunk(start: int, end: int, doc: str = "a.txt", **extra: object) -> dict:
    """A chunk whose text is exactly `DOC[start:end]`, like every chunk the builder emits."""
    return {
        "doc_id": doc,
        "chunk_id": f"{doc}#{start:04d}",
        "char_start": start,
        "char_end": end,
        "text": DOC[start:end],
        **extra,
    }


def span(start: int, end: int, doc: str = "a.txt") -> dict:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": DOC[start:end]}


def test_adjacent_chunks_merge_into_one_block_with_exact_offsets():
    blocks = stitch_contiguous([chunk(0, 50), chunk(50, 100)])
    assert len(blocks) == 1
    block = blocks[0]
    assert (block["char_start"], block["char_end"]) == (0, 100)
    assert block["text"] == DOC[0:100]
    assert block["rank"] == 1
    assert [part["char_start"] for part in block["metadata"][STITCHED_FROM_KEY]] == [0, 50]


def test_overlapping_chunks_merge_with_the_overlap_served_once():
    # The shipped chunker overlaps by design; a merge must not serve the shared text twice.
    blocks = stitch_contiguous([chunk(0, 50), chunk(30, 80)])
    assert len(blocks) == 1
    assert blocks[0]["text"] == DOC[0:80]
    assert served_chars(blocks) == 80 < served_chars([chunk(0, 50), chunk(30, 80)])


def test_a_contained_chunk_adds_no_characters_to_its_block():
    blocks = stitch_contiguous([chunk(0, 100), chunk(20, 40)])
    assert len(blocks) == 1
    assert blocks[0]["text"] == DOC[0:100]


def test_a_gap_is_never_bridged():
    # Bridging would serve text nobody retrieved, which is the one thing the lever must not do.
    chunks = [chunk(0, 50), chunk(60, 100)]
    assert stitch_contiguous(chunks) is chunks


def test_chunks_from_different_documents_are_left_alone():
    chunks = [chunk(0, 50, doc="a.txt"), chunk(50, 100, doc="b.txt")]
    assert stitch_contiguous(chunks) is chunks


def test_a_chunk_whose_text_disagrees_with_its_offsets_is_refused():
    # A governance overlay may rewrite chunk text; a merged block could then no longer be laid
    # back onto the source offsets, so the chunk is served as its own block instead.
    rewritten = {**chunk(50, 100), "text": "rewritten"}
    chunks = [chunk(0, 50), rewritten]
    assert stitch_contiguous(chunks) is chunks


def test_a_collapsed_duplicate_chunk_is_refused():
    # Its recorded occurrences describe THAT text at other places; a merged block is at none.
    collapsed = chunk(50, 100)
    collapsed["metadata"] = {
        "duplicate_count": 2,
        "duplicate_occurrences": [{"doc_id": "b.txt", "char_start": 0, "char_end": 50}],
    }
    chunks = [chunk(0, 50), collapsed]
    assert stitch_contiguous(chunks) is chunks


def test_a_block_takes_its_best_ranked_part_position_and_nothing_reorders():
    # Retrieval order: a far chunk, then two contiguous ones. The merged block sits where its
    # best-ranked part sat -- second -- and the far chunk keeps first place.
    chunks = [chunk(150, 180), chunk(50, 100), chunk(0, 50)]
    blocks = stitch_contiguous(chunks)
    assert [(b["char_start"], b["char_end"]) for b in blocks] == [(150, 180), (0, 100)]
    assert [b["rank"] for b in blocks] == [1, 2]


def test_a_merged_block_keeps_the_best_ranked_parts_identity_and_score():
    chunks = [chunk(50, 100, retrieval_score=0.9), chunk(0, 50, retrieval_score=0.4)]
    block = stitch_contiguous(chunks)[0]
    assert block["chunk_id"] == "a.txt#0050" and block["retrieval_score"] == 0.9


def test_stitching_converts_a_cut_span_to_intact_without_moving_recall_or_coverage():
    spans = [span(40, 60)]
    fragmented = [chunk(0, 50), chunk(50, 100)]
    stitched = stitch_contiguous(fragmented)
    for metric in (retrieval.recall_at_k, retrieval.span_char_coverage_at_k):
        assert metric(stitched, spans, 10) == metric(fragmented, spans, 10)
    assert retrieval.span_intact_at_k(fragmented, spans, 10) == 0.0
    assert retrieval.span_intact_at_k(stitched, spans, 10) == 1.0
    assert retrieval.served_chars_at_k(stitched, 10) == retrieval.served_chars_at_k(fragmented, 10)


def test_retriever_stitches_the_stores_own_top_k_and_censuses_the_reflow():
    class _Store:
        meta = {"strategy": "recursive"}

        def retrieve(self, question: str, k: int) -> list[dict]:
            return [chunk(0, 50), chunk(50, 100), chunk(150, 180)][:k]

    retriever = StitchingRetriever(_Store())
    blocks = retriever.retrieve("питання", 10)
    assert [(b["char_start"], b["char_end"]) for b in blocks] == [(0, 100), (150, 180)]
    census = retriever.census()
    assert census["parts_per_query"] == 3.0
    assert census["blocks_per_query"] == 2.0
    assert census["merged_per_query"] == 1.0
    assert census["chars_delta_per_query"] == 0.0
    assert retriever.meta["strategy"] == "recursive"  # unknown attributes delegate


def test_retriever_census_is_zero_before_any_query():
    assert StitchingRetriever(object()).census()["queries"] == 0.0


def test_retriever_preserves_a_hybrid_stores_split_query_path():
    seen: list[tuple[str, str]] = []

    class _Hybrid:
        def retrieve(self, question: str, k: int) -> list[dict]:
            raise AssertionError("the split path must not fall back to retrieve()")

        def retrieve_queries(self, dense: str, lexical: str, k: int) -> list[dict]:
            seen.append((dense, lexical))
            return [chunk(0, 50), chunk(50, 100)]

    blocks = StitchingRetriever(_Hybrid()).retrieve_queries("dense", "lexical", 10)
    assert seen == [("dense", "lexical")] and len(blocks) == 1


def test_retriever_falls_back_to_retrieve_when_the_store_has_no_split_path():
    class _Flat:
        def retrieve(self, question: str, k: int) -> list[dict]:
            return [chunk(0, 50)]

    assert (
        StitchingRetriever(_Flat()).retrieve_queries("dense", "lexical", 10)[0]["char_start"] == 0
    )
