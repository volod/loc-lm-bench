"""The chunk-size cap every strategy shares: `size` is a CAP, not a packing target.

A unit-packing strategy (`sentence`, `late`, `semantic`) emits a single unit whole however long
it is, and a structure-aware strategy (`markdown`, `heading`, `page`) does the same for a whole
section. On converted Ukrainian PDFs that leaks badly: a markdown table, page furniture, or a
heading block carries no sentence terminator, so it packs into one multi-hundred-character span
and an operator who asked for small chunks silently does not get them.

Every oversized span is split here on the recursive splitter's separators (paragraph -> line ->
word -> character), so the fallback keeps the largest natural boundary that still fits. Offsets
stay exact: sub-spans are resolved inside the oversized slice and shifted back to SOURCE
coordinates, and metadata is inherited unchanged by every sub-span.
"""

from llb.core.contracts.common import JsonObject
from llb.rag.chunking.recursive import recursive_spans


def cap_span(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """`text[start:end]` as one or more spans that all fit `size`, in SOURCE coordinates."""
    if end - start <= size:
        return [(start, end)]
    spans = [(start + s, start + e) for s, e in recursive_spans(text[start:end], size, overlap)]
    # The pinned splitter's last-resort separator is per-character, so it cannot leave an
    # oversized split; fail loudly rather than let an over-budget chunk reach the index.
    over = [(s, e) for s, e in spans if e - s > size]
    if over:
        raise ValueError(
            f"the recursive fallback left {len(over)} span(s) longer than size={size} "
            f"(longest {max(e - s for s, e in over)} chars); refusing to index them."
        )
    return spans


def cap_spans(
    text: str, spans: list[tuple[int, int, JsonObject]], size: int, overlap: int
) -> list[tuple[int, int, JsonObject]]:
    """`cap_span` over (start, end, metadata) triples; sub-spans inherit their span's metadata."""
    out: list[tuple[int, int, JsonObject]] = []
    for start, end, meta in spans:
        out.extend((s, e, meta) for s, e in cap_span(text, start, end, size, overlap))
    return out
