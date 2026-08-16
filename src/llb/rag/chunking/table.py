"""Table-aware chunking: a markdown table row is never cut, and a split table keeps its header.

Converted Ukrainian PDFs carry their numeric and comparative answers in markdown tables, and no
other strategy knows what a table row is. A unit-packing strategy (`sentence`, `late`, `semantic`)
sees no sentence terminator in a table at all, so the shared `size` cap cuts it at an arbitrary
character and the code, the unit, and the price of one line land in different chunks. `recursive`
is better by accident -- one of the pinned splitter's separators is the newline, so it usually
lands on row boundaries -- but nothing in it is AWARE of a row, and nothing measures that per
build. Row integrity here is a guarantee instead: see
`docs/impl/current/rag-core/chunking.md#table-aware-chunking` for the measured row-cut census and
what the guarantee turned out to be worth against each strategy.

This strategy partitions a document into TABLE regions and everything else. Non-table text routes
through the `recursive` splitter, so a `table`-versus-`recursive` delta isolates the table
handling. A table that fits `size` stays ONE chunk carrying the breadcrumb of its nearest enclosing
heading. A longer table splits BETWEEN rows -- whole rows packed up to `size` -- and every chunk of
that table records the header row's SOURCE offsets in `metadata.table_header_span`, so a consumer
can restore the column names a middle block would otherwise have lost.

Chunk text stays a verbatim corpus slice with exact offsets: no header row is ever copied into the
text of a later block, because that would break the source-span metric that scores retrieval.

A single row longer than `size` is the one case where a row IS cut: it falls through to the shared
cap split, exactly as an over-long sentence does under `sentence`.
"""

import re
from typing import NamedTuple

from llb.core.contracts.common import JsonObject
from llb.rag.chunking.cap import cap_span
from llb.rag.chunking.spans import _pack, _trim, validate_chunking
from llb.rag.chunking.structure import _Header, _header_breadcrumbs, _parse_headers

# A GFM table row: any non-blank line carrying a cell separator.
_PIPE = "|"
# The delimiter line under the header (`| --- | :---: |`): separators, dashes, colons, spaces
# only, with at least one dash and at least one separator. Requiring the separator keeps a plain
# `---` horizontal rule from turning the paragraph above it into a table.
_DELIMITER_ROW = re.compile(r"^(?=[^\n]*\|)[ \t|:-]*-[ \t|:-]*$")

TABLE_HEADER_SPAN_KEY = "table_header_span"


class _Table(NamedTuple):
    """One markdown table: its full span, its header row, and every row line in source order."""

    start: int
    end: int
    header: tuple[int, int]
    rows: list[tuple[int, int]]


def _line_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) per line, excluding the line break, covering the whole text."""
    spans: list[tuple[int, int]] = []
    start = 0
    for line in text.split("\n"):
        spans.append((start, start + len(line)))
        start += len(line) + 1
    return spans


def _is_row(text: str, span: tuple[int, int]) -> bool:
    line = text[span[0] : span[1]]
    return _PIPE in line and bool(line.strip())


def _is_delimiter(text: str, span: tuple[int, int]) -> bool:
    return bool(_DELIMITER_ROW.match(text[span[0] : span[1]]))


def find_tables(text: str) -> list[_Table]:
    """Every markdown table in `text`, in source order (GFM rule: header row + delimiter row)."""
    lines = _line_spans(text)
    tables: list[_Table] = []
    index = 0
    while index + 1 < len(lines):
        header = lines[index]
        if not (_is_row(text, header) and _is_delimiter(text, lines[index + 1])):
            index += 1
            continue
        end = index + 2
        while end < len(lines) and _is_row(text, lines[end]):
            end += 1
        rows = lines[index:end]
        tables.append(_Table(rows[0][0], rows[-1][1], header, rows))
        index = end
    return tables


class _Breadcrumbs:
    """Nearest-heading breadcrumb lookup that walks headings and tables in one pass."""

    def __init__(self, text: str) -> None:
        self.headers: list[_Header] = _parse_headers(text)
        self.crumbs = _header_breadcrumbs(self.headers)
        self._index = 0

    def at(self, offset: int) -> JsonObject:
        """Breadcrumb of the nearest heading ABOVE `offset` ({} before the first heading).

        Offsets must arrive in increasing order -- tables are visited in source order.
        """
        while self._index < len(self.headers) and self.headers[self._index][0] < offset:
            self._index += 1
        return dict(self.crumbs[self._index - 1]) if self._index else {}


def shifted_metadata(meta: JsonObject, offset: int) -> JsonObject:
    """Metadata whose recorded SOURCE spans are moved by `offset`.

    A `parent_child` child re-chunks its PARENT'S text, so a header row found there is in
    parent-local coordinates. The child's own offsets are shifted back to source coordinates and
    the recorded span has to move with them, or it points at unrelated text.
    """
    span = meta.get(TABLE_HEADER_SPAN_KEY)
    if not isinstance(span, list) or len(span) != 2:
        return meta
    return {**meta, TABLE_HEADER_SPAN_KEY: [span[0] + offset, span[1] + offset]}


def _table_meta(table: _Table, breadcrumb: JsonObject) -> JsonObject:
    """Heading breadcrumb plus the additive header-row offsets every table chunk carries."""
    return {"headers": breadcrumb, TABLE_HEADER_SPAN_KEY: [table.header[0], table.header[1]]}


def _table_chunk_spans(
    text: str, table: _Table, size: int, overlap: int, meta: JsonObject
) -> list[tuple[int, int, JsonObject]]:
    """One span for a table that fits `size`, otherwise row-aligned blocks that each fit it."""
    start, end = _trim(text, table.start, table.end)
    if end <= start:
        return []
    if end - start <= size:
        return [(start, end, meta)]
    return [
        (block_start, block_end, meta)
        for row_start, row_end in _pack(table.rows, size)
        # A single row longer than `size` is the only row the cap may cut.
        for block_start, block_end in cap_span(text, row_start, row_end, size, overlap)
    ]


def _emit_prose(
    out: list[tuple[int, int, JsonObject]],
    text: str,
    start: int,
    end: int,
    size: int,
    overlap: int,
) -> None:
    """Append the non-table region `text[start:end]` as recursive-split spans (no metadata)."""
    region_start, region_end = _trim(text, start, end)
    if region_end <= region_start:
        return
    out.extend((s, e, {}) for s, e in cap_span(text, region_start, region_end, size, overlap))


def table_spans(text: str, size: int, overlap: int) -> list[tuple[int, int, JsonObject]]:
    """Table-aware (start, end, metadata) spans: rows stay whole, prose routes through recursive."""
    validate_chunking(size, overlap)
    breadcrumbs = _Breadcrumbs(text)
    out: list[tuple[int, int, JsonObject]] = []
    cursor = 0
    for table in find_tables(text):
        _emit_prose(out, text, cursor, table.start, size, overlap)
        meta = _table_meta(table, breadcrumbs.at(table.start))
        out.extend(_table_chunk_spans(text, table, size, overlap, meta))
        cursor = table.end
    _emit_prose(out, text, cursor, len(text), size, overlap)
    return out
