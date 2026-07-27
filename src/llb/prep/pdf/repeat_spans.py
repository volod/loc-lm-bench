"""Offset and span remapping across repeated-block rewrites."""

from llb.prep.pdf.repeat_models import TextEdit


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


def remap_span_split(edits: list[TextEdit], start: int, end: int) -> list[tuple[int, int]] | None:
    """Remap `[start, end)` as one or MORE images, recovering a span that straddles a rewrite.

    `remap_span` refuses a span crossing an edit boundary because its two halves move apart -- but
    with `drop` the halves are not lost: the part inside a removed block still exists on the
    byte-identical survivor, and the part after it stays in place. Splitting the span at every edit
    boundary it crosses maps each piece cleanly (each piece lies in one region, so `remap_span`
    succeeds) and returns the pieces as separate images. Because `recall_at_k` credits an item when
    ANY of its spans is covered, a span split this way keeps exactly the original retrieval
    semantics. Returns None only when a piece is genuinely unanchorable (removed with no survivor);
    contiguous images are merged so a non-straddling span still returns a single pair.
    """
    if end <= start:
        return None
    cuts = sorted({c for edit in edits for c in (edit.start, edit.end) if start < c < end})
    bounds = [start, *cuts, end]
    images: list[tuple[int, int]] = []
    for lo, hi in zip(bounds, bounds[1:]):
        piece = remap_span(edits, lo, hi)
        if piece is None:
            return None
        if images and images[-1][1] == piece[0]:
            images[-1] = (images[-1][0], piece[1])  # merge a piece that stayed contiguous
        else:
            images.append(piece)
    return images


def span_rehomed(edits: list[TextEdit], start: int, end: int) -> bool:
    """True when `[start, end)` OVERLAPS a DROPPED copy, so part of it re-homes onto the survivor.

    `drop` keeps the first copy of a repeated block and removes the rest; a span labeled on one of
    those removed copies is not lost -- `remap_span` (or `remap_span_split` for a straddler)
    follows it onto the byte-identical survivor -- but the survivor sits in a DIFFERENT section, so
    the question's evidence has been re-homed. This flags any span that touches a removed block for
    the yield audit, which then asks whether retrieval still reaches the survivor.
    """
    return any(
        edit.moved_to is not None and edit.start < end and start < edit.end for edit in edits
    )


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
