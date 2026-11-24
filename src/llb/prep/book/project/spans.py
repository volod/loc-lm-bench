"""Turn projected fragments into document text and exact page citations.

Page spans partition the text, including the gap before a block and the
trailing newline. A marker's character range is the marker token itself.
"""

from dataclasses import dataclass

from llb.core.contracts.common import JsonObject
from llb.goldset.markers import MARKER


@dataclass(frozen=True, slots=True)
class Fragment:
    """One block's projected text, and the gap that precedes it."""

    page: int
    text: str
    block_id: str
    region: list[float] | None
    anchor: str
    vault_file: str
    gap_before: str


@dataclass(frozen=True, slots=True)
class PageSpan:
    """One PDF page's character range in a projected document."""

    page: int
    char_start: int
    char_end: int
    blocks: tuple[JsonObject, ...]


@dataclass(frozen=True, slots=True)
class RenderedDocument:
    """The projected text of one content file and the pages it cites."""

    text: str
    pages: tuple[PageSpan, ...]


def assemble(fragments: list[Fragment]) -> RenderedDocument:
    """Join fragments and cite every character of the resulting text."""
    parts: list[str] = []
    ranges: list[tuple[int, int, int]] = []
    blocks: list[JsonObject] = []
    cursor = 0
    for fragment in fragments:
        start = cursor + len(fragment.gap_before)
        parts.append(fragment.gap_before)
        parts.append(fragment.text)
        end = start + len(fragment.text)
        ranges.append((fragment.page, cursor, end))
        blocks.append(_block(fragment, start, end))
        cursor = end
    text = "".join(parts)
    if text and not text.endswith("\n"):
        text += "\n"
        page, start, _end = ranges[-1]
        ranges[-1] = (page, start, len(text))
    merged = _merge(ranges)
    _cover(text, merged, blocks)
    return RenderedDocument(text, tuple(_pages(merged, blocks)))


def _block(fragment: Fragment, start: int, end: int) -> JsonObject:
    return {
        "block_id": fragment.block_id,
        "region": fragment.region,
        "char_start": start,
        "char_end": end,
        "vault_file": fragment.vault_file,
        "vault_anchor": fragment.anchor,
        "markers": _markers(fragment.text, start),
    }


def _markers(text: str, base: int) -> list[JsonObject]:
    found: list[JsonObject] = []
    for match in MARKER.finditer(text):
        found.append(
            {
                "reason": match.group(1),
                "block_id": match.group(2),
                "start": int(match.group(3)),
                "end": int(match.group(4)),
                "char_start": base + match.start(),
                "char_end": base + match.end(),
            }
        )
    return found


def _merge(ranges: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    merged: list[tuple[int, int, int]] = []
    for page, start, end in ranges:
        if merged and merged[-1][0] == page and merged[-1][2] == start:
            merged[-1] = (page, merged[-1][1], end)
            continue
        merged.append((page, start, end))
    return merged


def _pages(ranges: list[tuple[int, int, int]], blocks: list[JsonObject]) -> list[PageSpan]:
    pages: list[PageSpan] = []
    for page, start, end in ranges:
        own = tuple(block for block in blocks if _inside(block, start, end))
        pages.append(PageSpan(page=page, char_start=start, char_end=end, blocks=own))
    return pages


def _inside(block: JsonObject, start: int, end: int) -> bool:
    char_start = block["char_start"]
    char_end = block["char_end"]
    return (
        isinstance(char_start, int)
        and isinstance(char_end, int)
        and start <= char_start
        and char_end <= end
    )


def _cover(text: str, ranges: list[tuple[int, int, int]], blocks: list[JsonObject]) -> None:
    if not ranges or ranges[0][1] != 0 or ranges[-1][2] != len(text):
        raise ValueError("citation page spans do not cover the document")
    cursor = 0
    for _page, start, end in ranges:
        if start != cursor or end < start:
            raise ValueError("citation page spans do not cover the document")
        cursor = end
    for block in blocks:
        for marker in _marker_rows(block.get("markers")):
            slice_ = text[int(marker["char_start"]) : int(marker["char_end"])]
            if not slice_.startswith("{{llb-marker:"):
                raise ValueError("marker offset does not match the document text")


def _marker_rows(markers: object) -> list[JsonObject]:
    if not isinstance(markers, list):
        return []
    return [marker for marker in markers if isinstance(marker, dict)]
