"""Strip PDF page furniture (running headers/footers, page numbers, image/comment noise) so a
passage that crosses a page break still grounds contiguously.

Repetition is measured across the WHOLE document, so `strip_page_furniture` must see every page at
once; only short, frequently repeating lines qualify, so body content is preserved.
"""

import re
from collections import Counter

from llb.prep.pdf.model import (
    _FURNITURE_DECORATION,
    _FURNITURE_MAX_LEN,
    _FURNITURE_REPEAT_FRACTION,
    _HTML_COMMENT_BLOCK,
    _MANY_BLANK_LINES,
    _PAGE_NUMBER_LINE,
    _PICTURE_PLACEHOLDER,
)


def _furniture_key(line: str) -> str:
    """Decoration-insensitive key for detecting a line that recurs as a running header/footer."""
    return re.sub(r"\s+", " ", _FURNITURE_DECORATION.sub(" ", line)).strip().casefold()


def strip_page_furniture(page_texts: list[str]) -> list[str]:
    """Drop PDF page furniture so a passage that crosses a page break still grounds contiguously.

    Removes lines that recur across a large fraction of pages (running headers/footers), standalone
    page-number lines, image placeholders, and HTML comments. Repetition is measured across the
    WHOLE document, so this must see every page at once. Body content is preserved: only short,
    frequently repeating lines qualify. Returns one cleaned string per input page (possibly empty).
    """
    counts = _furniture_line_counts(page_texts)
    threshold = max(8, int(_FURNITURE_REPEAT_FRACTION * max(len(page_texts), 1)))
    return [_strip_page(text, counts, threshold) for text in page_texts]


def _furniture_line_counts(page_texts: list[str]) -> Counter[str]:
    """How often each normalized line recurs across the whole document."""
    counts: Counter[str] = Counter()
    for text in page_texts:
        for line in text.split("\n"):
            key = _furniture_key(line)
            if key:
                counts[key] += 1
    return counts


def _is_furniture_line(line: str, counts: Counter[str], threshold: int) -> bool:
    """A short line repeating on many pages (running header/footer) or a bare page number."""
    stripped = line.strip()
    key = _furniture_key(line)
    if key and len(stripped) <= _FURNITURE_MAX_LEN and counts[key] >= threshold:
        return True
    return bool(_PAGE_NUMBER_LINE.match(line))


def _strip_page(text: str, counts: Counter[str], threshold: int) -> str:
    """One page with furniture lines, placeholders, and comment blocks removed."""
    kept = [
        line
        for line in text.split("\n")
        if not line.strip() or not _is_furniture_line(line, counts, threshold)
    ]
    body = _HTML_COMMENT_BLOCK.sub(" ", _PICTURE_PLACEHOLDER.sub(" ", "\n".join(kept)))
    return _MANY_BLANK_LINES.sub("\n\n", body).strip()
