"""Orphaned overlay entries return to the queue with the decision they held."""

from llb.core.contracts.data_prep.book_blocks import Block
from llb.core.contracts.data_prep.book_resolve import (
    CorrectionEdit,
    OrphanedEdit,
    PageCorrectionOverlay,
    PageResolvedText,
    ResolvedBlock,
)
from llb.core.contracts.data_prep.book_review import PriorDecision
from llb.prep.book.review.alternates import (
    marker_kind,
    marker_kind_of_replacement,
    other_reading,
)
from llb.prep.book.resolve.primary import resolved_reading
from llb.prep.book.review.constants import DECISION_TOKENS
from llb.prep.book.review.marks import SpanMark


def attach_orphans(
    marks: list[SpanMark],
    resolved: PageResolvedText,
    overlay: PageCorrectionOverlay,
    blocks: dict[str, Block],
) -> list[SpanMark]:
    """Keep every refused edit. Attach it to the span it touched, or open a new entry."""
    resolved_by_id = {block.block_id: block for block in resolved.blocks}
    current = list(marks)
    for orphan in overlay.orphaned:
        prior = prior_decision(orphan)
        hosts = [
            mark for mark in current if mark.block_id == orphan.edit.block_id and mark.block_present
        ]
        if hosts:
            current = _attach(current, hosts, orphan.edit, prior)
            continue
        current.append(_requeue(resolved.page_id, orphan, prior, blocks, resolved_by_id))
    return current


def prior_decision(orphan: OrphanedEdit) -> PriorDecision:
    """The decision text the workbench shows for an edit the ledger refused."""
    edit = orphan.edit
    return PriorDecision(
        edit_id=edit.edit_id,
        decision=edit.evidence if edit.evidence in DECISION_TOKENS else "",
        replacement=edit.replacement,
        evidence=edit.evidence,
        reason=orphan.reason,
    )


def _attach(
    marks: list[SpanMark], hosts: list[SpanMark], edit: CorrectionEdit, prior: PriorDecision
) -> list[SpanMark]:
    target = next((mark for mark in hosts if _overlaps(edit, mark)), hosts[0])
    updated = _with_prior(target, prior)
    return [updated if mark is target else mark for mark in marks]


def _with_prior(mark: SpanMark, prior: PriorDecision) -> SpanMark:
    from dataclasses import replace

    return replace(mark, priors=(*mark.priors, prior))


def _requeue(
    page_id: str,
    orphan: OrphanedEdit,
    prior: PriorDecision,
    blocks: dict[str, Block],
    resolved: dict[str, ResolvedBlock],
) -> SpanMark:
    edit = orphan.edit
    block = blocks.get(edit.block_id)
    resolved_block = resolved.get(edit.block_id)
    if block is None or resolved_block is None:
        return _absent(page_id, edit, prior)
    text = resolved_block.primary_text
    primary = resolved_reading(block, resolved_block)
    return SpanMark(
        page_id=page_id,
        pdf_page=block.pdf_page,
        block_id=block.block_id,
        ordinal=block.ordinal,
        anchor=resolved_block.anchor,
        kind=marker_kind(block.kind, text),
        reason="orphaned",
        start=0,
        end=len(text),
        raw_text=text,
        candidate=None if primary is None else other_reading(block, primary),
        classes=(),
        priors=(prior,),
    )


def _absent(page_id: str, edit: CorrectionEdit, prior: PriorDecision) -> SpanMark:
    return SpanMark(
        page_id=page_id,
        pdf_page=_page_number(edit.page_id),
        block_id=edit.block_id,
        ordinal=0,
        anchor=edit.anchor,
        kind=marker_kind_of_replacement(edit.replacement, edit.raw_text),
        reason="orphaned",
        start=edit.start,
        end=edit.end,
        raw_text=edit.raw_text,
        candidate=None,
        classes=(),
        priors=(prior,),
        block_present=False,
    )


def _overlaps(edit: CorrectionEdit, mark: SpanMark) -> bool:
    if edit.start == edit.end and mark.start == mark.end:
        return edit.start == mark.start
    return edit.start < mark.end and edit.end > mark.start


def _page_number(page_id: str) -> int:
    digits = page_id[1:] if page_id.startswith("p") else page_id
    return int(digits) if digits.isdigit() else 1
