"""Which spans of a resolved page become review entries.

An unresolved character span is a marker. An empty content block is unreadable, because a blank
where a passage should be is an omission. A low confidence score routes the whole block. Formula
and table blocks use their own marker.
"""

from collections.abc import Sequence
from dataclasses import replace

from llb.core.contracts.data_prep.book_blocks import Block, UncertaintyClass
from llb.core.contracts.data_prep.book_resolve import (
    CorrectionEdit,
    PageCorrectionOverlay,
    PageResolvedText,
    ResolvedBlock,
    ResolvedSpan,
)
from llb.core.contracts.data_prep.book_review import QueueReason, ReviewVerdict
from llb.prep.book.review.alternates import (
    candidate_text,
    is_low_confidence,
    marker_kind,
    other_reading,
)
from llb.prep.book.resolve.primary import resolved_reading
from llb.prep.book.review.constants import REVIEW_TEXT_KINDS
from llb.prep.book.review.marks import SpanMark
from llb.prep.book.review.orphans import attach_orphans


def page_marks(
    ledger_page_id: str,
    blocks: dict[str, Block],
    resolved: PageResolvedText,
    overlay: PageCorrectionOverlay,
    edits: Sequence[CorrectionEdit],
) -> list[SpanMark]:
    """The open spans on one page, in reading order."""
    marks: list[SpanMark] = []
    for resolved_block in resolved.blocks:
        source = blocks.get(resolved_block.block_id)
        if source is not None:
            marks.extend(_block_marks(ledger_page_id, source, resolved_block))
    marks = _annotate(marks, overlay, edits, ledger_page_id)
    marks = attach_orphans(marks, resolved, overlay, blocks)
    marks.sort(key=lambda item: (item.pdf_page, not item.block_present, item.ordinal, item.start))
    return marks


def _block_marks(page_id: str, block: Block, resolved: ResolvedBlock) -> list[SpanMark]:
    primary = resolved_reading(block, resolved)
    unresolved = [span for span in resolved.spans if span.provenance == "unresolved"]
    if unresolved:
        return [_from_span(page_id, block, resolved, span) for span in unresolved]
    if block.kind not in REVIEW_TEXT_KINDS:
        return []
    if not resolved.primary_text.strip():
        return [_whole(page_id, block, resolved, "empty")]
    if primary is not None and is_low_confidence(primary):
        return [_whole(page_id, block, resolved, "low-confidence")]
    return []


def _from_span(page_id: str, block: Block, resolved: ResolvedBlock, span: ResolvedSpan) -> SpanMark:
    primary = resolved_reading(block, resolved)
    candidate = None if primary is None else candidate_text(block, primary, span.start, span.end)
    return _make(
        page_id,
        block,
        resolved,
        "unresolved",
        span.start,
        span.end,
        span.text,
        candidate,
        tuple(span.classes),
    )


def _whole(page_id: str, block: Block, resolved: ResolvedBlock, reason: QueueReason) -> SpanMark:
    primary = resolved_reading(block, resolved)
    text = resolved.primary_text
    candidate = None if primary is None else other_reading(block, primary)
    return _make(page_id, block, resolved, reason, 0, len(text), text, candidate, ())


def _make(
    page_id: str,
    block: Block,
    resolved: ResolvedBlock,
    reason: QueueReason,
    start: int,
    end: int,
    text: str,
    candidate: str | None,
    classes: tuple[UncertaintyClass, ...],
) -> SpanMark:
    return SpanMark(
        page_id=page_id,
        pdf_page=block.pdf_page,
        block_id=block.block_id,
        ordinal=block.ordinal,
        anchor=resolved.anchor,
        kind=marker_kind(block.kind, text),
        reason=reason,
        start=start,
        end=end,
        raw_text=text,
        candidate=candidate,
        classes=classes,
    )


def _annotate(
    marks: list[SpanMark],
    overlay: PageCorrectionOverlay,
    edits: Sequence[CorrectionEdit],
    page_id: str,
) -> list[SpanMark]:
    """Drop a span a reviewer edit already adopted. Keep one that is only recorded so far."""
    applied = {edit.edit_id for edit in overlay.applied if edit.author == "reviewer"}
    orphaned = {item.edit.edit_id for item in overlay.orphaned}
    kept: list[SpanMark] = []
    for mark in marks:
        updated = _with_verdict(mark, edits, page_id, applied, orphaned)
        if updated is not None:
            kept.append(updated)
    return kept


def _with_verdict(
    mark: SpanMark,
    edits: Sequence[CorrectionEdit],
    page_id: str,
    applied: set[str],
    orphaned: set[str],
) -> SpanMark | None:
    covering = _covering(edits, page_id, mark)
    if any(edit.edit_id in applied for edit in covering):
        return None
    pending = [edit for edit in covering if edit.edit_id not in orphaned]
    verdict = _token(pending[0].evidence) if pending else ""
    if verdict == mark.verdict:
        return mark
    return replace(mark, verdict=verdict)


def _covering(
    edits: Sequence[CorrectionEdit], page_id: str, mark: SpanMark
) -> list[CorrectionEdit]:
    return [
        edit
        for edit in edits
        if edit.page_id == page_id and edit.block_id == mark.block_id and _covers(edit, mark)
    ]


def _covers(edit: CorrectionEdit, mark: SpanMark) -> bool:
    return edit.start <= mark.start and edit.end >= mark.end


def _token(evidence: str) -> ReviewVerdict:
    if evidence == "accept-raw":
        return "accept-raw"
    if evidence == "accept-candidate":
        return "accept-candidate"
    if evidence == "enter-text":
        return "enter-text"
    if evidence == "mark-illegible":
        return "mark-illegible"
    return ""
