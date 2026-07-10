"""Corpus-chunking strategies (page / heading / late): offset-exact, CI-safe with fakes."""

from pathlib import Path

import pytest

from llb.rag.chunking import (
    STRATEGIES,
    chunk_corpus,
    chunk_text,
    doc_page_spans,
    heading_spans,
    recursive_spans,
)
from llb.rag.late_encoding import (
    encode_records_late,
    pool_span_vectors,
    window_char_spans,
)
from llb.core.paths import PROJECT_ROOT

FIXTURE = PROJECT_ROOT / "samples" / "pdf_pages"
PDF_DOC = "pdf-37e9918f8c51.md"
PLAIN_DOC = "plain_note.md"
# Hand-read from the committed sidecar: page 1 spans chars [30, 288), page 2 [288, 531).
PAGE_SPANS = [(30, 288), (288, 531)]

NEW_STRATEGIES = ("page", "heading", "late")


def test_new_strategies_registered():
    assert set(NEW_STRATEGIES) <= set(STRATEGIES)


@pytest.mark.parametrize("strategy", NEW_STRATEGIES)
def test_offsets_resolve_for_new_strategies(strategy):
    if strategy == "page":
        text = (FIXTURE / PDF_DOC).read_text(encoding="utf-8")
        chunks = chunk_text(text, PDF_DOC, strategy, size=1000, overlap=20, page_spans=PAGE_SPANS)
        texts = {PDF_DOC: text}
    else:
        chunks = chunk_corpus(FIXTURE, strategy, size=1000, overlap=20)
        texts = {
            doc: (FIXTURE / doc).read_text(encoding="utf-8")
            for doc in {c["doc_id"] for c in chunks}
        }
    assert chunks
    for c in chunks:
        assert texts[c["doc_id"]][c["char_start"] : c["char_end"]] == c["text"]


# --- page: boundaries never cross a page-sidecar span ---


def test_doc_page_spans_reads_sidecar():
    assert doc_page_spans(FIXTURE, PDF_DOC) == PAGE_SPANS
    assert doc_page_spans(FIXTURE, PLAIN_DOC) is None


def test_page_never_crosses_a_page_boundary():
    text = (FIXTURE / PDF_DOC).read_text(encoding="utf-8")
    regions = [(0, 30), *PAGE_SPANS, (531, len(text))]
    chunks = chunk_text(text, PDF_DOC, "page", size=1000, overlap=20, page_spans=PAGE_SPANS)
    assert chunks
    for c in chunks:
        assert any(rs <= c["char_start"] and c["char_end"] <= re_ for rs, re_ in regions), (
            f"chunk [{c['char_start']}, {c['char_end']}) straddles a page boundary"
        )


@pytest.mark.slow
def test_page_subsplits_within_a_long_page():
    # size 100 < page-1 span length, so page 1 must yield several chunks, all inside it.
    chunks = [
        c for c in chunk_corpus(FIXTURE, "page", size=100, overlap=20) if c["doc_id"] == PDF_DOC
    ]
    page_one = [c for c in chunks if 30 <= c["char_start"] and c["char_end"] <= 288]
    assert len(page_one) >= 2


@pytest.mark.slow
def test_page_without_sidecar_falls_back_to_recursive():
    text = (FIXTURE / PLAIN_DOC).read_text(encoding="utf-8")
    page = chunk_text(text, PLAIN_DOC, "page", size=60, overlap=10)
    expected = recursive_spans(text, 60, 10)
    assert [(c["char_start"], c["char_end"]) for c in page] == expected


# --- heading: layout-aware hierarchy packing, breadcrumbs, heading lines included ---

HEADING_TEXT = (
    "# Розділ\n\nВступний текст розділу.\n\n"
    "## Перший підрозділ\n\nТекст першого підрозділу.\n\n"
    "## Другий підрозділ\n\nТекст другого підрозділу."
)


def test_heading_packs_whole_subtree_into_one_chunk():
    spans = heading_spans(HEADING_TEXT, size=1000, overlap=0)
    assert len(spans) == 1
    start, end, meta = spans[0]
    assert HEADING_TEXT[start:end].startswith("# Розділ")  # heading line INSIDE the chunk
    assert "## Другий підрозділ" in HEADING_TEXT[start:end]
    assert meta["headers"] == {"h1": "Розділ"}


@pytest.mark.slow
def test_heading_oversized_subtree_recurses_with_full_breadcrumb():
    spans = heading_spans(HEADING_TEXT, size=60, overlap=0)
    texts = [HEADING_TEXT[s:e] for s, e, _ in spans]
    assert any(t.startswith("# Розділ") for t in texts)  # own section keeps its heading line
    sub = [m for _, _, m in spans if m["headers"].get("h2")]
    assert sub and all(m["headers"]["h1"] == "Розділ" for m in sub)  # full breadcrumb
    assert {m["headers"]["h2"] for m in sub} == {"Перший підрозділ", "Другий підрозділ"}


@pytest.mark.slow
def test_heading_skips_heading_only_sections():
    text = "# Порожній\n\n## Дитина\n\nТекст дитини тут."
    spans = heading_spans(text, size=20, overlap=0)  # too small to pack the subtree
    assert all(text[s:e].strip() != "# Порожній" for s, e, _ in spans)
    assert any(m["headers"] == {"h1": "Порожній", "h2": "Дитина"} for _, _, m in spans)


def test_heading_without_headers_covers_text():
    text = "Просто текст без заголовків. Ще одне речення."
    spans = heading_spans(text, size=1000, overlap=0)
    assert spans == [(0, len(text), {"headers": {}})]


# --- late: sentence-identical spans, late-pooled vectors ---


def test_late_spans_match_sentence_spans():
    text = (FIXTURE / PDF_DOC).read_text(encoding="utf-8")
    late = chunk_text(text, PDF_DOC, "late", size=120, overlap=20)
    sentence = chunk_text(text, PDF_DOC, "sentence", size=120, overlap=20)
    assert [(c["char_start"], c["char_end"]) for c in late] == [
        (c["char_start"], c["char_end"]) for c in sentence
    ]


def test_pool_span_vectors_means_and_normalizes():
    token_spans = [(0, 5), (5, 10), (10, 15)]
    token_vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    pooled = pool_span_vectors([(0, 10), (20, 30)], token_spans, token_vectors)
    assert pooled[1] is None  # no token overlaps the second chunk
    assert pooled[0] == pytest.approx([0.7071, 0.7071], abs=1e-4)  # mean [0.5, 0.5], normalized


def test_pool_span_vectors_partial_token_overlap_counts():
    # A token straddling the chunk edge still contributes (char-overlap rule).
    pooled = pool_span_vectors([(3, 8)], [(0, 5)], [[2.0, 0.0]])
    assert pooled[0] == [1.0, 0.0]


def test_window_char_spans_groups_consecutive_tokens():
    offsets = [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert window_char_spans(offsets, 2) == [(0, 6), (6, 12)]
    assert window_char_spans(offsets, 10) == [(0, 12)]


def _record(doc_id, start, end, text):
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}#{start}",
        "char_start": start,
        "char_end": end,
        "text": text,
    }


def test_encode_records_late_groups_per_doc_and_falls_back():
    docs = {"a.md": "текст першого документа", "b.md": "текст другого"}
    records = [
        _record("a.md", 0, 5, "текст"),
        _record("b.md", 0, 5, "текст"),
        _record("a.md", 6, 14, "першого"),
    ]
    calls: list[str] = []

    def encode_doc(text, spans):
        calls.append(text)
        # First span of every doc gets a vector; a.md's second span gets none (fallback).
        return [[1.0, 0.0]] + [None] * (len(spans) - 1)

    def fallback(texts):
        assert texts == ["першого"]
        return [[0.0, 2.0]]

    vectors = encode_records_late(records, docs.__getitem__, encode_doc, fallback)
    assert [c for c in calls] == [docs["a.md"], docs["b.md"]]  # one whole-doc pass per doc
    assert vectors == [[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]]  # record order preserved


def test_encode_records_late_incomplete_fallback_fails_loudly():
    records = [_record("a.md", 0, 5, "текст")]
    with pytest.raises(ValueError, match="fewer vectors"):
        encode_records_late(
            records, lambda _d: "текст", lambda _t, s: [None] * len(s), lambda _texts: []
        )


def test_store_refuses_late_with_parent_child():
    from llb.rag.store import RagStore

    with pytest.raises(ValueError, match="flat mode only"):
        RagStore.build(FIXTURE, "late", 200, 20, mode="parent_child")


def test_build_chunking_comparison_rejects_unknown_strategy():
    from llb.rag.compare import build_chunking_comparison

    with pytest.raises(ValueError, match="unknown chunking strategy"):
        build_chunking_comparison(object(), ["markdown", "nope"])


def test_build_chunking_comparison_builds_flat_store_per_strategy(monkeypatch, tmp_path):
    import llb.rag.store as store_mod
    from llb.rag.compare import build_chunking_comparison

    class FakeConfig:
        corpus_root = tmp_path
        chunk_size = 300
        chunk_overlap = 30
        embedding_model = "fake-embedder"

    class FakeStore:
        def __init__(self, strategy):
            self.strategy = strategy
            self.saved_to = None

        def save(self, path):
            self.saved_to = Path(path)

        def retrieve(self, question, k):
            return []

    def fake_build(corpus_root, strategy, size, overlap, model, mode):
        assert (corpus_root, size, overlap, model, mode) == (
            tmp_path,
            300,
            30,
            "fake-embedder",
            "flat",
        )
        return FakeStore(strategy)

    monkeypatch.setattr(store_mod.RagStore, "build", staticmethod(fake_build))
    stores = build_chunking_comparison(FakeConfig(), ["page", "heading"], stores_root=tmp_path)
    assert list(stores) == ["page", "heading"]
    assert stores["page"].saved_to == tmp_path / "page"
    assert stores["heading"].saved_to == tmp_path / "heading"
