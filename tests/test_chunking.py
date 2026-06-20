from llb.rag.chunking import (
    chunk_text,
    fixed_spans,
    recursive_spans,
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


def test_sentence_never_cuts_midsentence():
    spans = sentence_chunk_spans(TEXT, size=20)
    sentence_ends = {end for _, end in sentence_spans(TEXT)}
    for _, end in spans:
        assert end in sentence_ends or end == len(TEXT)


def test_recursive_starts_at_zero():
    spans = recursive_spans(TEXT, size=25, overlap=5)
    assert spans and spans[0][0] == 0
