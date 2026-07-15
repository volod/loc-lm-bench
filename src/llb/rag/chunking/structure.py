"""Structure-aware strategies: markdown-header, heading-hierarchy, and PDF page-aligned splits.

All three parse boundaries from the SOURCE so every span stays an exact source substring, and
sub-split oversized sections with the pinned `recursive_spans`.
"""

import re
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.rag.chunking.recursive import recursive_spans
from llb.rag.chunking.spans import _trim, validate_chunking

_MD_HEADER = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.M)

# One parsed markdown header: (line_start, line_end, level, title).
_Header = tuple[int, int, int, str]


def _parse_headers(text: str) -> list[_Header]:
    """Markdown header lines parsed from the SOURCE, so spans stay exact source substrings."""
    return [
        (m.start(), m.end(), len(m.group(1)), m.group(2).strip()) for m in _MD_HEADER.finditer(text)
    ]


def _header_breadcrumbs(headers: list[_Header]) -> list[dict[str, str]]:
    """Full enclosing-heading breadcrumb per header (h1..h6 stack rule)."""
    crumbs: list[dict[str, str]] = []
    stack: dict[int, str] = {}
    for _, _, level, title in headers:
        stack = {lvl: t for lvl, t in stack.items() if lvl < level}
        stack[level] = title
        crumbs.append({f"h{lvl}": stack[lvl] for lvl in sorted(stack)})
    return crumbs


def _emit_section(
    out: list[tuple[int, int, JsonObject]],
    text: str,
    body_start: int,
    body_end: int,
    meta: JsonObject,
    size: int,
    overlap: int,
) -> None:
    """Append the trimmed section as one span, or recursive sub-spans when it exceeds `size`."""
    bs, be = _trim(text, body_start, body_end)
    if be <= bs:
        return
    if be - bs <= size:
        out.append((bs, be, meta))
    else:
        for rs, re_end in recursive_spans(text[bs:be], size, overlap):
            out.append((bs + rs, bs + re_end, meta))


def markdown_spans(text: str, size: int, overlap: int) -> list[tuple[int, int, JsonObject]]:
    """Structure-aware split on markdown headers; header breadcrumbs land in metadata.

    Headers are parsed from the SOURCE so every span is an exact source substring. (langchain's
    MarkdownHeaderTextSplitter rejoins section content and loses offsets, which would break the
    source-span metric.) Sections longer than `size` are sub-split with `recursive_spans`
    (the pinned langchain RecursiveCharacterTextSplitter).
    """
    headers = _parse_headers(text)
    out: list[tuple[int, int, JsonObject]] = []
    if not headers:
        _emit_section(out, text, 0, len(text), {"headers": {}}, size, overlap)
        return out

    if headers[0][0] > 0:  # preamble before the first header
        _emit_section(out, text, 0, headers[0][0], {"headers": {}}, size, overlap)

    crumbs = _header_breadcrumbs(headers)
    for i, (_h_start, h_end, _level, _title) in enumerate(headers):
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        _emit_section(out, text, h_end, body_end, {"headers": crumbs[i]}, size, overlap)
    return out


def page_aligned_spans(
    text: str, size: int, overlap: int, page_spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Spans whose boundaries never cross a page-sidecar span (PDF page/citation-aware).

    The document is partitioned into regions: each sidecar page span plus the gaps around
    them (front matter, inter-page markers). A region that fits `size` becomes one span; a
    longer region is sub-split WITHIN itself via `recursive_spans`, so no chunk ever
    straddles a page boundary and every citation resolves to exactly one page range.
    """
    validate_chunking(size, overlap)
    n = len(text)
    regions: list[tuple[int, int]] = []
    cursor = 0
    for raw_start, raw_end in sorted(page_spans):
        start, end = max(cursor, raw_start, 0), min(raw_end, n)
        if end <= start:
            continue
        if start > cursor:
            regions.append((cursor, start))  # gap before this page (never merged into it)
        regions.append((start, end))
        cursor = end
    if cursor < n:
        regions.append((cursor, n))

    spans: list[tuple[int, int]] = []
    for region_start, region_end in regions:
        rs, re_ = _trim(text, region_start, region_end)
        if re_ <= rs:
            continue
        if re_ - rs <= size:
            spans.append((rs, re_))
        else:
            spans.extend((rs + s, rs + e) for s, e in recursive_spans(text[rs:re_], size, overlap))
    return spans


class _HeadingWalker:
    """Recursive heading-subtree chunker behind `heading_spans`."""

    def __init__(self, text: str, headers: list[_Header], size: int, overlap: int) -> None:
        self.text = text
        self.headers = headers
        self.crumbs = _header_breadcrumbs(headers)
        self.size = size
        self.overlap = overlap
        self.out: list[tuple[int, int, JsonObject]] = []

    def _emit(self, body_start: int, body_end: int, meta: JsonObject) -> None:
        _emit_section(self.out, self.text, body_start, body_end, meta, self.size, self.overlap)

    def _subtree_end(self, i: int) -> int:
        """Char offset where header i's subtree ends (the next heading at <= its level)."""
        level = self.headers[i][2]
        for j in range(i + 1, len(self.headers)):
            if self.headers[j][2] <= level:
                return self.headers[j][0]
        return len(self.text)

    def _next_after_subtree(self, i: int) -> int:
        """Index of the first header NOT inside header i's subtree."""
        end = self._subtree_end(i)
        j = i + 1
        while j < len(self.headers) and self.headers[j][0] < end:
            j += 1
        return j

    def _emit_subtree(self, i: int) -> None:
        h_start, h_end, _, _ = self.headers[i]
        end = self._subtree_end(i)
        meta: JsonObject = {"headers": self.crumbs[i]}
        if end - h_start <= self.size:  # the whole subtree, heading line included, is one chunk
            self._emit(h_start, end, meta)
            return
        j = i + 1
        in_subtree = j < len(self.headers) and self.headers[j][0] < end
        first_child = self.headers[j][0] if in_subtree else end
        body_s, body_e = _trim(self.text, h_end, first_child)
        if body_e > body_s:  # own section text (skip heading-only chunks; the breadcrumb
            self._emit(h_start, first_child, meta)  # already carries the title to child chunks)
        while j < len(self.headers) and self.headers[j][0] < end:
            self._emit_subtree(j)
            j = self._next_after_subtree(j)

    def walk(self) -> list[tuple[int, int, JsonObject]]:
        if self.headers[0][0] > 0:  # preamble before the first heading
            self._emit(0, self.headers[0][0], {"headers": {}})
        i = 0
        while i < len(self.headers):
            self._emit_subtree(i)
            i = self._next_after_subtree(i)
        return self.out


def heading_spans(text: str, size: int, overlap: int) -> list[tuple[int, int, JsonObject]]:
    """Heading-hierarchy (layout-aware) split: whole subtrees pack into one chunk when they fit.

    Unlike `markdown_spans` (one chunk per leaf section BODY, header lines stripped), this
    strategy keeps heading lines INSIDE the chunk text -- the layout the embedder sees matches
    the layout a reader sees -- and a heading whose entire subtree (itself + all nested
    subsections) fits within `size` becomes a single chunk. Oversized subtrees emit their own
    section (heading line + immediate body, recursively sub-split) and then recurse into each
    child heading. Every chunk carries the full breadcrumb of enclosing headings in
    `metadata.headers`.
    """
    headers = _parse_headers(text)
    if not headers:
        out: list[tuple[int, int, JsonObject]] = []
        _emit_section(out, text, 0, len(text), {"headers": {}}, size, overlap)
        return out
    return _HeadingWalker(text, headers, size, overlap).walk()


def doc_page_spans(corpus_root: Path, doc_id: str) -> list[tuple[int, int]] | None:
    """Page char spans for `doc_id` from its citation sidecar, or None when it has none."""
    # Function-level import: page_metadata imports this package for `_MD_HEADER`.
    from llb.rag.page_metadata import load_page_citations

    cite = load_page_citations(Path(corpus_root), doc_id)
    if cite is None:
        return None
    _, spans = cite
    out = [
        (span["char_start"], span["char_end"])
        for span in spans
        if isinstance(span.get("char_start"), int) and isinstance(span.get("char_end"), int)
    ]
    return out or None
