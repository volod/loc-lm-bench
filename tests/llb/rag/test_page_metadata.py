"""Chunk page/section provenance join (chunk-page-metadata, task 18)."""

import pytest

from llb.rag.chunking.corpus import chunk_corpus, chunk_text
from llb.rag.chunking.dispatch import STRATEGIES
from llb.rag.page_metadata import (
    annotate_page_metadata,
    heading_breadcrumb,
    intersect_pages,
    load_page_citations,
)
from llb.rag.store_build import _build_children
from llb.core.paths import PROJECT_ROOT

FIXTURE = PROJECT_ROOT / "samples" / "pdf_pages"
PDF_DOC = "pdf-37e9918f8c51.md"
PLAIN_DOC = "plain_note.md"

# Hand-read from the committed sidecar: page 1 spans chars [30, 288), page 2 [288, 531).
PAGE_SPANS = [
    {"page": 1, "char_start": 30, "char_end": 288},
    {"page": 2, "char_start": 288, "char_end": 531},
]
PAGE_SPAN_TUPLES = [(span["char_start"], span["char_end"]) for span in PAGE_SPANS]
ANNOTATION_STRATEGIES = [
    pytest.param(strategy, marks=pytest.mark.slow) if strategy == "recursive" else strategy
    for strategy in STRATEGIES
    if strategy != "semantic"
]


def _rec(start: int, end: int, doc_id: str = PDF_DOC, metadata=None):
    text = (FIXTURE / doc_id).read_text(encoding="utf-8")
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}#{start}",
        "char_start": start,
        "char_end": end,
        "text": text[start:end],
        "metadata": {} if metadata is None else metadata,
    }


def test_intersect_pages_inside_single_page() -> None:
    assert intersect_pages(100, 200, PAGE_SPANS) == [1]


def test_intersect_pages_straddling_boundary() -> None:
    assert intersect_pages(250, 400, PAGE_SPANS) == [1, 2]


def test_intersect_pages_header_region_has_no_pages() -> None:
    assert intersect_pages(5, 20, PAGE_SPANS) == []


def test_load_page_citations_reads_sidecar() -> None:
    result = load_page_citations(FIXTURE, PDF_DOC)
    assert result is not None
    source, spans = result
    assert source == "regulation.pdf"
    assert [s["page"] for s in spans] == [1, 2]


def test_load_page_citations_absent_for_plain_doc() -> None:
    assert load_page_citations(FIXTURE, PLAIN_DOC) is None


def test_heading_breadcrumb_nested() -> None:
    text = (FIXTURE / PDF_DOC).read_text(encoding="utf-8")
    # A char inside the page-2 body is under both the section h1 and the subsection h2.
    assert heading_breadcrumb(text, 400) == {
        "h1": "Розділ 1. Загальні положення",
        "h2": "Підрозділ 1.1. Визначення термінів",
    }


def test_heading_breadcrumb_page_one_body() -> None:
    text = (FIXTURE / PDF_DOC).read_text(encoding="utf-8")
    assert heading_breadcrumb(text, 150) == {"h1": "Розділ 1. Загальні положення"}


def test_annotate_attaches_exact_pages_and_source() -> None:
    inside = _rec(100, 200)
    straddle = _rec(250, 400)
    coverage = annotate_page_metadata([inside, straddle], FIXTURE)
    assert coverage == 1.0
    assert inside["metadata"]["pages"] == [1, 1]
    assert inside["metadata"]["source_pdf"] == "regulation.pdf"
    assert straddle["metadata"]["pages"] == [1, 2]
    assert straddle["metadata"]["headers"]["h1"] == "Розділ 1. Загальні положення"


def test_annotate_leaves_text_and_offsets_byte_identical() -> None:
    rec = _rec(100, 200)
    before_text, before_start, before_end = rec["text"], rec["char_start"], rec["char_end"]
    annotate_page_metadata([rec], FIXTURE)
    assert rec["text"] == before_text
    assert (rec["char_start"], rec["char_end"]) == (before_start, before_end)


def test_annotate_no_page_fields_for_plain_doc() -> None:
    rec = _rec(0, 40, doc_id=PLAIN_DOC)
    coverage = annotate_page_metadata([rec], FIXTURE)
    assert coverage == 0.0
    assert "pages" not in rec["metadata"]
    assert "source_pdf" not in rec["metadata"]
    # A plain markdown doc still gets its heading breadcrumb.
    assert rec["metadata"]["headers"] == {"h1": "Нотатка"}


def test_annotate_preserves_existing_markdown_headers() -> None:
    rec = _rec(300, 350, metadata={"headers": {"h9": "hand-set"}})
    annotate_page_metadata([rec], FIXTURE)
    assert rec["metadata"]["headers"] == {"h9": "hand-set"}  # not overwritten


def test_annotate_breaks_shared_metadata_aliasing() -> None:
    shared = {}
    a = _rec(100, 150, metadata=shared)
    b = _rec(300, 350, metadata=shared)
    annotate_page_metadata([a, b], FIXTURE)
    assert a["metadata"] is not b["metadata"]
    assert a["metadata"]["pages"] == [1, 1]
    assert b["metadata"]["pages"] == [2, 2]


@pytest.mark.parametrize("strategy", ANNOTATION_STRATEGIES)
def test_offsets_round_trip_after_annotation(strategy: str) -> None:
    doc_text = (FIXTURE / PDF_DOC).read_text(encoding="utf-8")
    if strategy == "page":
        chunks = chunk_text(
            doc_text, PDF_DOC, strategy, size=1000, overlap=40, page_spans=PAGE_SPAN_TUPLES
        )
    else:
        chunks = chunk_corpus(FIXTURE, strategy, size=1000, overlap=40)
    annotate_page_metadata(chunks, FIXTURE)
    for chunk in chunks:
        if chunk["doc_id"] != PDF_DOC:
            continue
        # Offsets still resolve to the exact source slice.
        assert doc_text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]
        pages = chunk["metadata"].get("pages")
        hit = intersect_pages(chunk["char_start"], chunk["char_end"], PAGE_SPANS)
        if pages is not None:
            assert pages == [hit[0], hit[-1]]


@pytest.mark.slow
def test_parent_child_propagation_annotates_children() -> None:
    parents = chunk_corpus(FIXTURE, "recursive", size=400, overlap=0)
    parents = [p for p in parents if p["doc_id"] == PDF_DOC]
    annotate_page_metadata(parents, FIXTURE)
    children = _build_children(parents, "recursive", child_size=120, overlap=0, embedder=None)
    annotate_page_metadata(children, FIXTURE)
    assert children
    # Propagation actually happened: page fields reached the children.
    assert any("pages" in c["metadata"] for c in children)
    # Child page ranges are their own precise intersection, not blindly the parent's. A child
    # whose span predates page 1 (e.g. a chunk that is only the "# Source PDF" preamble) correctly
    # carries no `pages`, so assert page fields per-child rather than assuming every child is paged.
    for child in children:
        hit = intersect_pages(child["char_start"], child["char_end"], PAGE_SPANS)
        if hit:
            assert child["metadata"]["pages"] == [hit[0], hit[-1]]
        else:
            assert "pages" not in child["metadata"]


def test_retrieval_hits_expose_page_fields_flat() -> None:
    class FakeEmbedder:
        def encode_queries(self, texts):
            return [[1.0]]

    class FakeIndex:
        def search(self, query, k):
            return [[0.9, 0.8]], [[0, 1]]

    from llb.rag.store import RagStore

    chunks = [_rec(100, 200), _rec(300, 400)]
    annotate_page_metadata(chunks, FIXTURE)
    store = RagStore(chunks, FakeIndex(), FakeEmbedder(), {"mode": "flat"}, None)
    hits = store.retrieve("q", 2)
    assert hits[0]["metadata"]["pages"] == [1, 1]
    assert hits[1]["metadata"]["pages"] == [2, 2]


def test_retrieval_hits_expose_page_fields_parent_child() -> None:
    class FakeEmbedder:
        def encode_queries(self, texts):
            return [[1.0]]

    class FakeIndex:
        def search(self, query, k):
            return [[0.9]], [[0]]

    from llb.rag.store import RagStore

    parents = [_rec(100, 200)]
    annotate_page_metadata(parents, FIXTURE)
    children = [{"chunk_id": "c0", "parent_id": parents[0]["chunk_id"]}]
    store = RagStore(children, FakeIndex(), FakeEmbedder(), {"mode": "parent_child"}, parents)
    hits = store.retrieve("q", 1)
    assert hits[0]["metadata"]["pages"] == [1, 1]
    assert hits[0]["metadata"]["source_pdf"] == "regulation.pdf"
