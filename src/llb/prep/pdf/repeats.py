"""Intra-document repeated blocks: census them, and handle them at CONVERSION time.

A converted manual repeats whole blocks INSIDE one document -- a boilerplate procedure step
restated in section after section, a note repeated under every table, a running footer the
line-level `strip_page_furniture` pass did not reach. Index-time collapse
(`llb.rag.duplicates`) already removes their index and tie cost, but it cannot restore what the
repetition destroys in the SOURCE: the document's own reading order stops tracking its chunk
ordinals, and one surviving copy answers a question asked about any of the sections that carry it.

Both handling options rewrite the rendered document, so every edit is recorded as a `TextEdit`
and every offset that survives is remappable (`remap_span`) -- gold spans, page-citation
sidecars, and anything else anchored to the pre-strip text follow the rewrite instead of
silently pointing at moved characters. Repeat detection is EXACT (byte-identical block text):
near-duplicate matching is a corpus-hygiene question owned by the conflict lane, not this one.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

from typing_extensions import TypedDict

REPEAT_KEEP = "keep"
REPEAT_DROP = "drop"
REPEAT_ANCHOR = "anchor"
REPEAT_MODES = (REPEAT_KEEP, REPEAT_DROP, REPEAT_ANCHOR)

# A block must repeat at least this many times inside ONE document before either mode touches it.
# Two occurrences is ordinary prose reuse; a third is a pattern, and it is the threshold the
# census reports against so the number an operator reads is the number that was acted on.
DEFAULT_MIN_REPEATS = 3

_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_MD_HEADING_LINE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.M)
_TABLE_LINE = re.compile(r"^[ \t]*\|", re.M)
_BREADCRUMB_SEPARATOR = " > "
# The anchor is a markdown blockquote glued to the block with no blank line between, so every
# splitter that breaks on blank lines keeps the anchor and its block in one chunk.
_ANCHOR_PREFIX = "> "


class TextEdit(NamedTuple):
    """One rewrite of `[start, end)` into `replacement`, in ascending non-overlapping order.

    `moved_to` is set only for a DROPPED repeat: it is the start offset (in the pre-strip text) of
    the surviving copy of the same text, which is where an offset inside the dropped block still
    resolves after the rewrite.
    """

    start: int
    end: int
    replacement: str
    moved_to: int | None = None


class RepeatCensus(TypedDict):
    """Block-level repetition of ONE document, measured before any rewrite."""

    blocks: int  # blank-line separated blocks in the document
    repeated_blocks: int  # blocks whose text occurs at least `min_repeats` times
    groups: int  # distinct texts repeating at least `min_repeats` times
    largest_group: int  # occurrences of the most repeated text (0 when nothing repeats)
    handled_groups: int  # of those groups, the ones a mode is allowed to rewrite
    handled_blocks: int  # blocks the mode actually rewrote


@dataclass(frozen=True)
class RepeatRewrite:
    """A rewritten document plus the edits that produced it and the census that motivated them."""

    text: str
    census: RepeatCensus
    edits: list[TextEdit] = field(default_factory=list)


class _Block(NamedTuple):
    start: int
    end: int
    text: str


def rewrite_repeated_blocks(
    text: str, mode: str = REPEAT_KEEP, min_repeats: int = DEFAULT_MIN_REPEATS
) -> RepeatRewrite:
    """Census the document's repeated blocks and apply `mode` to the eligible ones.

    `keep` measures and changes nothing (byte-identical text, no edits), so a census is never
    paid for with a rewrite. `drop` keeps the FIRST occurrence of a repeated block and removes
    the rest, which loses no text -- every dropped copy is byte-identical to the survivor.
    `anchor` keeps every occurrence and prefixes each with its enclosing-heading breadcrumb, so
    copies under different sections stop being identical and each is retrievable in its own
    section instead of collapsing onto one.
    """
    if mode not in REPEAT_MODES:
        raise ValueError(f"unknown repeat mode: {mode!r}; choose one of {REPEAT_MODES}")
    if min_repeats < 2:
        raise ValueError(f"min_repeats must be >= 2 (got {min_repeats})")
    blocks = _blocks(text)
    counts = Counter(block.text for block in blocks)
    repeated = {body for body, count in counts.items() if count >= min_repeats}
    handled = {body for body in repeated if _is_handleable(body)}
    edits = [] if mode == REPEAT_KEEP else _edits(text, blocks, handled, mode)
    census: RepeatCensus = {
        "blocks": len(blocks),
        "repeated_blocks": sum(counts[body] for body in repeated),
        "groups": len(repeated),
        "largest_group": max((counts[body] for body in repeated), default=0),
        "handled_groups": len(handled),
        "handled_blocks": len(edits),
    }
    return RepeatRewrite(text=apply_edits(text, edits), census=census, edits=edits)


def apply_edits(text: str, edits: list[TextEdit]) -> str:
    """Apply ascending non-overlapping edits to `text`."""
    parts: list[str] = []
    cursor = 0
    for edit in edits:
        parts.append(text[cursor : edit.start])
        parts.append(edit.replacement)
        cursor = edit.end
    parts.append(text[cursor:])
    return "".join(parts)


def remap_span(edits: list[TextEdit], start: int, end: int) -> tuple[int, int] | None:
    """Where `[start, end)` of the pre-strip text lands after the rewrite, or None if it cannot.

    A span inside a DROPPED block resolves onto the surviving copy of the same text, so a gold
    label on any copy keeps pointing at its own words. A span that straddles an edit boundary has
    no single image -- its two halves move apart -- so it is refused rather than silently moved,
    which the length invariant below detects; the caller decides what to do with it (the goldset
    lane drops the item and reports it).
    """
    if end <= start:
        return None
    new_start = _remap_offset(edits, start)
    new_last = _remap_offset(edits, end - 1)
    if new_start is None or new_last is None or new_last + 1 - new_start != end - start:
        return None
    return new_start, new_last + 1


def span_rehomed(edits: list[TextEdit], start: int, end: int) -> bool:
    """True when `[start, end)` sat inside a DROPPED copy, so its remap lands on another section.

    `drop` keeps the first copy of a repeated block and removes the rest; a span labeled on one of
    those removed copies is not lost -- `remap_span` follows it onto the byte-identical survivor --
    but the survivor sits in a DIFFERENT section, so the question's evidence has been re-homed.
    This flags exactly those spans for the yield audit, which then asks whether retrieval still
    reaches the survivor.
    """
    return any(
        edit.moved_to is not None and edit.start <= start < edit.end and start < end
        for edit in edits
    )


def heading_breadcrumb(text: str, offset: int) -> str:
    """The enclosing markdown heading chain at `offset` (`h1 > h2 > ...`), '' when there is none.

    A heading at offset 0 is skipped: the conversion lane opens every rendered document with its
    own `# Source PDF: <source>` title, which encloses every block equally and so distinguishes
    no copy from another -- it would only make each anchor longer.
    """
    stack: dict[int, str] = {}
    for match in _MD_HEADING_LINE.finditer(text, 0, offset):
        if match.start() == 0:
            continue
        level = len(match.group(1))
        stack = {lvl: title for lvl, title in stack.items() if lvl < level}
        stack[level] = match.group(2).strip()
    return _BREADCRUMB_SEPARATOR.join(stack[level] for level in sorted(stack))


def _blocks(text: str) -> list[_Block]:
    """Blank-line separated blocks with their offsets; leading/trailing whitespace excluded."""
    blocks: list[_Block] = []
    cursor = 0
    bounds = [match.start() for match in _BLANK_LINE.finditer(text)] + [len(text)]
    for bound in bounds:
        body = text[cursor:bound]
        stripped = body.strip()
        if stripped:
            start = cursor + (len(body) - len(body.lstrip()))
            blocks.append(_Block(start=start, end=start + len(stripped), text=stripped))
        cursor = bound
    return blocks


def _is_handleable(body: str) -> bool:
    """Only repeated PROSE is rewritten: table rows and headings carry structure, not content.

    A repeated table header or `|`-row is what makes the tables under it readable, and a repeated
    heading is a section anchor -- removing either from every copy but the first would corrupt the
    document even though the removed characters exist elsewhere.
    """
    return not _MD_HEADING_LINE.match(body) and not _TABLE_LINE.search(body)


def _edits(text: str, blocks: list[_Block], handled: set[str], mode: str) -> list[TextEdit]:
    """One edit per rewritten block: a deletion (`drop`) or a breadcrumb insertion (`anchor`)."""
    edits: list[TextEdit] = []
    first_seen: dict[str, int] = {}
    starts = [block.start for block in blocks[1:]] + [len(text)]
    for block, next_start in zip(blocks, starts):
        survivor = first_seen.setdefault(block.text, block.start)
        if block.text not in handled:
            continue
        if mode == REPEAT_ANCHOR:
            edits.extend(_anchor_edit(text, block))
        elif survivor != block.start:
            # swallow the blank line after the block too, so removal leaves no widening gap
            edits.append(
                TextEdit(start=block.start, end=next_start, replacement="", moved_to=survivor)
            )
    return edits


def _anchor_edit(text: str, block: _Block) -> list[TextEdit]:
    """Prefix one occurrence with its enclosing-heading breadcrumb (nothing when there is none)."""
    breadcrumb = heading_breadcrumb(text, block.start)
    if not breadcrumb:
        return []
    anchor = f"{_ANCHOR_PREFIX}{breadcrumb}\n"
    return [TextEdit(start=block.start, end=block.start, replacement=anchor)]


def _remap_offset(edits: list[TextEdit], offset: int) -> int | None:
    """The post-rewrite image of one pre-strip offset (None when it fell into removed text)."""
    shift = 0
    for edit in edits:
        if edit.start == edit.end:  # insertion: it shifts everything from its own position on
            if edit.start > offset:
                break
            shift += len(edit.replacement)
        elif edit.end <= offset:
            shift += len(edit.replacement) - (edit.end - edit.start)
        elif edit.start <= offset:  # inside a dropped block: follow the surviving copy
            if edit.moved_to is None:
                return None
            return _remap_offset(edits, edit.moved_to + (offset - edit.start))
        else:
            break
    return offset + shift
