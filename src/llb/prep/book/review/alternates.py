"""The other reading a reviewer can accept, and which marker kind a span takes."""

from collections.abc import Sequence

from llb.core.contracts.data_prep.book_blocks import (
    Block,
    RawReading,
    UncertaintySlice,
    UncertaintySpan,
)
from llb.core.contracts.data_prep.book_review import MarkerKind
from llb.prep.book.review.constants import FORMULA_TABLE_KINDS, LOW_CONFIDENCE_BELOW


def marker_kind(block_kind: str, text: str) -> MarkerKind:
    """Formula and table content is its own marker. Blank text is unreadable."""
    if block_kind in FORMULA_TABLE_KINDS:
        return "formula-table"
    if not text.strip():
        return "unreadable"
    return "low-confidence"


def marker_kind_of_replacement(replacement: str, raw: str) -> MarkerKind:
    """The kind a previous marker used, or a guess from the raw slice when it did not."""
    if replacement.startswith("[!unreadable]"):
        return "unreadable"
    if replacement.startswith("[!low-confidence]"):
        return "low-confidence"
    if replacement.startswith("[!formula-table]"):
        return "formula-table"
    return "unreadable" if not raw.strip() else "low-confidence"


def is_low_confidence(reading: RawReading) -> bool:
    """True when the engine reported a score below the routing cut."""
    return reading.confidence is not None and reading.confidence < LOW_CONFIDENCE_BELOW


def candidate_text(block: Block, primary: RawReading, start: int, end: int) -> str | None:
    """The other reading of this span, or none when every reading says the same thing.

    The span's primary characters are kept wherever no other reading differs, so a span widened
    around a character another reading inserted reads as that reading: `.5` becomes `.05`.
    """
    pieces = _span_pieces(block, primary, start, end)
    if pieces:
        return _composed(pieces, primary.text, start, end)
    if start == 0 and end == len(primary.text):
        return other_reading(block, primary)
    return None


_Piece = tuple[int, int, tuple[float, str, str], str]


def _span_pieces(block: Block, primary: RawReading, start: int, end: int) -> list[_Piece]:
    pieces: list[_Piece] = []
    for span in block.uncertainty:
        piece = _piece(span, primary, start, end, block.readings)
        if piece is not None:
            pieces.append(piece)
    return pieces


def _piece(
    span: UncertaintySpan,
    primary: RawReading,
    start: int,
    end: int,
    readings: Sequence[RawReading],
) -> _Piece | None:
    own = [item for item in span.slices if _is_primary(primary, item)]
    others = [item for item in span.slices if not _is_primary(primary, item)]
    if not own or not others:
        return None
    sliver = own[0]
    if not _touches(sliver.start, sliver.end, start, end):
        return None
    best = _best_slice(others, readings)
    return sliver.start, sliver.end, _rank(readings, best.engine, best.variant), best.text


def _touches(own_start: int, own_end: int, start: int, end: int) -> bool:
    if own_start == own_end:
        return start <= own_start <= end
    return own_end > start and own_start < end


def _composed(pieces: list[_Piece], primary: str, start: int, end: int) -> str | None:
    """The primary slice with each other reading's text in place of the slice it disagreed on."""
    parts: list[str] = []
    cursor = start
    seen: set[tuple[int, int]] = set()
    for own_start, own_end, _rank_key, text in sorted(pieces):
        if (own_start, own_end) in seen or own_start < cursor:
            continue
        seen.add((own_start, own_end))
        parts.append(primary[cursor:own_start])
        parts.append(text)
        cursor = max(cursor, min(own_end, end))
    parts.append(primary[cursor:end])
    composed = "".join(parts)
    if composed == primary[start:end]:
        return None
    return composed


def other_reading(block: Block, primary: RawReading) -> str | None:
    """The strongest other reading of the whole block, when its text differs."""
    others = [
        reading
        for reading in block.readings
        if reading is not primary and reading.text != primary.text
    ]
    if not others:
        return None
    return _best_reading(others).text


def _best_slice(
    slices: Sequence[UncertaintySlice], readings: Sequence[RawReading]
) -> UncertaintySlice:
    return min(slices, key=lambda item: _rank(readings, item.engine, item.variant))


def _best_reading(readings: Sequence[RawReading]) -> RawReading:
    return min(readings, key=lambda item: _rank(readings, item.engine, item.variant))


def _rank(readings: Sequence[RawReading], engine: str, variant: str) -> tuple[float, str, str]:
    return (-_score(readings, engine, variant), engine, variant)


def _score(readings: Sequence[RawReading], engine: str, variant: str) -> float:
    for reading in readings:
        if (
            reading.engine == engine
            and reading.variant == variant
            and reading.confidence is not None
        ):
            return reading.confidence
    return -1.0


def _is_primary(primary: RawReading, sliver: UncertaintySlice) -> bool:
    if sliver.engine != primary.engine or sliver.variant != primary.variant:
        return False
    if sliver.end > len(primary.text) or sliver.end < sliver.start:
        return False
    return primary.text[sliver.start : sliver.end] == sliver.text
