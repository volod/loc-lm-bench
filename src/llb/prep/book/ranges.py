"""1-based inclusive PDF page ranges and the chapter-sized batches a run walks.

`parse_page_ranges("1-5,8,10-", page_count)` returns the sorted page numbers. Every refusal names
the offending part: a page before 1, a page past the document's last page, a reversed range, an
unparseable part, and two parts that select the same page.
"""

import re
from collections.abc import Sequence

from llb.prep.book.constants import PAGE_ID_MIN_DIGITS, PAGE_ID_PREFIX

_PART = re.compile(r"^(?P<start>\d*)\s*(?P<dash>-)?\s*(?P<end>\d*)$")


class PageRangeError(ValueError):
    """A `--pages` specification this document cannot satisfy."""


def _parse_part(part: str, page_count: int) -> tuple[int, int]:
    match = _PART.match(part)
    if match is None or (not match["start"] and not match["end"]):
        raise PageRangeError(f"page range part {part!r} is not N, N-M, N-, or -M")
    if not match["dash"]:
        start = end = int(match["start"])
    else:
        start = int(match["start"]) if match["start"] else 1
        end = int(match["end"]) if match["end"] else page_count
    if start < 1:
        raise PageRangeError(f"page range part {part!r} starts at {start}; pages are 1-based")
    if end < start:
        raise PageRangeError(f"page range part {part!r} is reversed ({start} > {end})")
    if end > page_count:
        raise PageRangeError(
            f"page range part {part!r} reaches page {end}, past the last page {page_count}"
        )
    return start, end


def parse_page_ranges(spec: str, page_count: int) -> list[int]:
    """Sorted 1-based pages selected by `spec`, refusing overlap and out-of-range parts by name."""
    if page_count < 1:
        raise PageRangeError("the document has no pages")
    parts = [part.strip() for part in spec.split(",")]
    if not spec.strip() or any(not part for part in parts):
        raise PageRangeError(f"page range {spec!r} has an empty part")
    owner: dict[int, str] = {}
    for part in parts:
        start, end = _parse_part(part, page_count)
        for page in range(start, end + 1):
            if page in owner:
                overlap = [p for p in range(start, end + 1) if p in owner]
                raise PageRangeError(
                    f"page range parts {owner[page]!r} and {part!r} overlap on "
                    f"{_format_span(overlap)}"
                )
            owner[page] = part
    return sorted(owner)


def _format_span(pages: Sequence[int]) -> str:
    first, last = pages[0], pages[-1]
    return f"page {first}" if first == last else f"pages {first}-{last}"


def batches(pages: Sequence[int], batch_pages: int) -> list[list[int]]:
    """Consecutive batches of at most `batch_pages` pages, in page order."""
    if batch_pages < 1:
        raise PageRangeError(f"batch size must be at least 1, got {batch_pages}")
    ordered = list(pages)
    return [ordered[i : i + batch_pages] for i in range(0, len(ordered), batch_pages)]


def page_id(pdf_page: int, page_count: int) -> str:
    """Deterministic directory name for a 1-based PDF page: `p0007` in a 1,067-page book."""
    width = max(PAGE_ID_MIN_DIGITS, len(str(page_count)))
    return f"{PAGE_ID_PREFIX}{pdf_page:0{width}d}"
