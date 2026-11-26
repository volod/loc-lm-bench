"""A span that still needs a person, before its marker string is built."""

from dataclasses import dataclass

from llb.core.contracts.data_prep.book_blocks import Region, UncertaintyClass
from llb.core.contracts.data_prep.book_review import (
    MarkerKind,
    PriorDecision,
    QueueReason,
    ReviewVerdict,
)


@dataclass(frozen=True, slots=True)
class SpanMark:
    """One open span on one block."""

    page_id: str
    pdf_page: int
    block_id: str
    ordinal: int
    anchor: str
    kind: MarkerKind
    reason: QueueReason
    start: int
    end: int
    raw_text: str
    candidate: str | None
    classes: tuple[UncertaintyClass, ...]
    priors: tuple[PriorDecision, ...] = ()
    verdict: ReviewVerdict = ""
    block_present: bool = True
    region: Region | None = None
