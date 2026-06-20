import pytest

from llb.rag.chunking import (
    chunk_text,
    fixed_spans,
    markdown_spans,
    recursive_spans,
    semantic_spans,
    sentence_chunk_spans,
    sentence_spans,
)

TEXT = "Перше речення. Друге речення! Третє речення?\n\nНовий абзац тут. І ще одне."


def test_offsets_resolve_for_all_strategies():
    for strategy in ("fixed", "sentence", "recursive"):
        chunks = chunk_text(TEXT, "d.txt", strategy, size=30, overlap=5)
        assert chunks, strategy
        for c in chunks:
            assert TEXT[c["char_start"] : c["char_end"]] == c["text"]


def test_fixed_covers_and_respects_size():
    spans = fixed_spans("x" * 100, size=30, overlap=5)
    assert all(end - start <= 30 for start, end in spans)
    assert spans[0][0] == 0 and spans[-1][1] == 100


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, -1), (10, 10), (10, 11)])
def test_chunking_rejects_invalid_window(size, overlap):
    with pytest.raises(ValueError):
        chunk_text(TEXT, "d.txt", "fixed", size=size, overlap=overlap)


def test_sentence_never_cuts_midsentence():
    spans = sentence_chunk_spans(TEXT, size=20)
    sentence_ends = {end for _, end in sentence_spans(TEXT)}
    for _, end in spans:
        assert end in sentence_ends or end == len(TEXT)


def test_recursive_starts_at_zero():
    spans = recursive_spans(TEXT, size=25, overlap=5)
    assert spans and spans[0][0] == 0


def test_chunk_text_carries_metadata_field():
    for strategy in ("fixed", "sentence", "recursive"):
        for c in chunk_text(TEXT, "d.txt", strategy, size=30, overlap=5):
            assert "metadata" in c  # uniform shape; empty for non-structured strategies


# --- native semantic chunking (offset-exact, fake embedder -> CI-safe) ---

class FakeEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode_passages(self, texts):
        return [self.mapping[t] for t in texts]


def test_semantic_spans_breaks_on_distance_spike():
    text = "Кіт сидить вдома. Кіт спить тут. Банк відкрито сьогодні."
    sents = sentence_spans(text)
    assert len(sents) == 3
    # first two sentences identical embeddings, third orthogonal -> one breakpoint
    vecs = dict(zip((text[s:e] for s, e in sents), ([1.0, 0.0], [1.0, 0.0], [0.0, 1.0])))
    spans = semantic_spans(text, 1000, FakeEmbedder(vecs), threshold_pct=50)
    assert len(spans) == 2
    assert text[spans[0][0]:spans[0][1]] == text[sents[0][0]:sents[1][1]]  # exact source offsets
    assert text[spans[1][0]:spans[1][1]] == text[sents[2][0]:sents[2][1]]


def test_semantic_spans_single_sentence():
    text = "Лише одне речення."
    assert semantic_spans(text, 1000, FakeEmbedder({}), threshold_pct=50) == sentence_spans(text)


# --- langchain-backed strategies (skip when [rag] is absent, e.g. in CI) ---

def test_recursive_langchain_offsets_in_range():
    pytest.importorskip("langchain_text_splitters")
    from llb.rag.chunking import _recursive_langchain

    text = "Абзац один тут.\n\nАбзац два значно довший і має більше слів для поділу на частини зараз."
    spans = _recursive_langchain(text, 40, 8)
    assert spans and spans[0][0] == 0
    assert all(0 <= s < e <= len(text) and text[s:e] for s, e in spans)


def test_markdown_spans_carry_headers_and_exact_offsets():
    # markdown parses headers from the source, so it is offset-exact without langchain.
    text = ("# Заголовок\n\nТекст розділу тут. Ще одне речення.\n\n"
            "## Підрозділ\n\nІнший текст підрозділу зараз.")
    spans = markdown_spans(text, size=1000, overlap=0)
    assert spans
    for s, e, meta in spans:
        assert 0 <= s < e <= len(text)
        assert text[s:e].strip() and "#" not in text[s:e]  # body only, header line excluded
        assert "headers" in meta
    # the breadcrumb stack nests: the subsection chunk carries both h1 and h2
    deepest = [m for _, _, m in spans if m["headers"].get("h2")]
    assert deepest and deepest[0]["headers"]["h1"] == "Заголовок"
