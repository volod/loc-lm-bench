"""The visible uncertainty marker, and the passage a reader sees it in.

The marker is a Markdown link Obsidian shows as text, plus an optional embed of the crop. A
machine reads the same line back. The three kinds are `unreadable`, `low-confidence`, and
`formula-table`.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llb.core.contracts.data_prep.book_resolve import ResolvedBlock
from llb.core.contracts.data_prep.book_review import MarkerKind

_MARKER = re.compile(
    r"\[!(unreadable|low-confidence|formula-table)\]"
    r"\(([^)\s]+) \"([^\"]+)\"\)"
    r"(?: !\[\[([^]]+)\]\])?"
)


@dataclass(frozen=True, slots=True)
class ParsedMarker:
    """One marker found in a passage."""

    kind: MarkerKind
    pdf_link: str
    block_id: str
    start: int
    end: int
    crop: str | None


def render_marker(
    kind: MarkerKind,
    pdf_link: str,
    block_id: str,
    start: int,
    end: int,
    crop: str | None,
) -> str:
    """One visible marker. The title carries the block and the offsets, which the link does not."""
    title = f"{block_id} {start}-{end}"
    text = f'[!{kind}]({pdf_link} "{title}")'
    if crop:
        text += f" ![[{crop}]]"
    return text


def parse_marker(text: str) -> ParsedMarker | None:
    """The first marker in `text`, or none when the line is ordinary prose."""
    found = parse_markers(text)
    return found[0] if found else None


def marker_layout_size(text: str) -> int:
    """Count markup consistently for an in-vault PDF and an external PDF."""
    return len(text) - sum(len(match[2].partition("#")[0]) for match in _MARKER.finditer(text))


def parse_markers(text: str) -> list[ParsedMarker]:
    """Every marker in `text`, in order."""
    found: list[ParsedMarker] = []
    for match in _MARKER.finditer(text):
        kind, link, title, crop = match.groups()
        parsed = _title(title)
        marker_kind = _kind(kind)
        if parsed is None or marker_kind is None:
            continue
        block_id, start, end = parsed
        found.append(
            ParsedMarker(
                kind=marker_kind,
                pdf_link=link,
                block_id=block_id,
                start=start,
                end=end,
                crop=crop,
            )
        )
    return found


def render_passage(
    block: ResolvedBlock,
    replacements: Mapping[tuple[int, int], str],
    prefixes: Sequence[str],
) -> str:
    """The block a reader sees: adopted text, with a marker where a span was left out.

    An empty passage keeps the marker and drops the blank. A low-confidence or orphaned block
    keeps its text and leads with the marker.
    """
    parts: list[str] = []
    for span in block.spans:
        replacement = replacements.get((span.start, span.end))
        if span.provenance == "unresolved" and replacement is not None:
            parts.append(replacement)
        elif span.adopted:
            parts.append(span.text)
    body = "".join(parts)
    if prefixes and not body.strip():
        return " ".join(prefixes)
    if prefixes:
        return " ".join(prefixes) + " " + body
    return body


def _kind(value: str) -> MarkerKind | None:
    if value == "unreadable":
        return "unreadable"
    if value == "low-confidence":
        return "low-confidence"
    if value == "formula-table":
        return "formula-table"
    return None


def _title(title: str) -> tuple[str, int, int] | None:
    if " " not in title:
        return None
    block_id, offsets = title.rsplit(" ", 1)
    if "-" not in offsets:
        return None
    start_text, end_text = offsets.split("-", 1)
    if not start_text.isdigit() or not end_text.isdigit():
        return None
    return block_id, int(start_text), int(end_text)
