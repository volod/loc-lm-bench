"""The other reading a reviewer can accept, and which marker kind a span takes."""

from collections.abc import Sequence

from llb.core.contracts.data_prep.book_blocks import (
    Block,
    RawReading,
    UncertaintySlice,
    UncertaintySpan,
)
from llb.core.contracts.data_prep.book_resolve import ResolvedBlock
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


def primary_reading(block: Block, resolved: ResolvedBlock) -> RawReading | None:
    """The ledger reading the resolved block was taken from."""
    for reading in block.readings:
        if reading.engine != resolved.engine or reading.variant != resolved.variant:
            continue
        if resolved.model is None or reading.model == resolved.model:
            return reading
    return block.readings[0] if block.readings else None


def candidate_text(block: Block, primary: RawReading, start: int, end: int) -> str | None:
    """The other reading of this span, or none when every reading says the same thing."""
    pieces = _span_pieces(block, primary, start, end)
    if pieces:
        return _joined(pieces, primary.text[start:end])
    if start == 0 and end == len(primary.text):
        return other_reading(block, primary)
    return None


def _span_pieces(block: Block, primary: RawReading, start: int, end: int) -> list[tuple[int, str]]:
    pieces: list[tuple[int, str]] = []
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
) -> tuple[int, str] | None:
    own = [item for item in span.slices if _is_primary(primary, item)]
    others = [item for item in span.slices if not _is_primary(primary, item)]
    if not own or not others:
        return None
    sliver = own[0]
    if sliver.end <= start or sliver.start >= end:
        return None
    return sliver.start, _best_slice(others, readings).text


def _joined(pieces: list[tuple[int, str]], primary_slice: str) -> str | None:
    text = "".join(item[1] for item in sorted(pieces))
    if text == primary_slice:
        return None
    return text


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
