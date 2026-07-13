"""Primitive character-span helpers shared by every chunking strategy.

Every span is (start, end) into the SOURCE text so retrieval can be scored against source-span
gold labels by overlap. These helpers have zero heavy dependencies.
"""

import re

_TERM = re.compile(r"[.!?…]+")
_CLOSERS = '”»")]’'


def validate_chunking(size: int, overlap: int) -> None:
    """Validate invariants shared by every chunking implementation."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) char spans for sentences, covering the whole text."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _TERM.finditer(text):
        end = m.end()
        while end < len(text) and text[end] in _CLOSERS:
            end += 1
        if end >= len(text) or text[end].isspace():
            if text[start:end].strip():
                spans.append((start, end))
            start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans


def fixed_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    validate_chunking(size, overlap)
    step = size - overlap
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        spans.append((i, min(n, i + size)))
        if i + size >= n:
            break
        i += step
    return spans


def _pack(spans: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    """Greedily merge adjacent spans while the merged span fits within `size`."""
    out: list[tuple[int, int]] = []
    cur_start = cur_end = None
    for start, end in spans:
        if cur_start is None:
            cur_start, cur_end = start, end
        elif end - cur_start <= size:
            cur_end = end
        else:
            assert cur_end is not None
            out.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    if cur_start is not None:
        assert cur_end is not None
        out.append((cur_start, cur_end))
    return out


def sentence_chunk_spans(text: str, size: int) -> list[tuple[int, int]]:
    return _pack(sentence_spans(text), size)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(pct / 100.0 * (len(ordered) - 1)))
    return ordered[idx]
