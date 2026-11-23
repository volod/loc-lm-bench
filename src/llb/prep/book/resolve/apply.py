"""Apply the resolution rule to one page.

The adopted text is the primary reading, with three exceptions. Characters every reading shares
are an agreement edit. An uncertainty span stays unresolved unless an overlay edit covers it.
Reviewer edits win over model edits, and both win over agreement. An edit whose anchor does not
match the block is orphaned and re-queued; it is not tried against any other block.
"""

from llb.core.contracts.data_prep.book_blocks import Block, PageBlockLedger, RawReading
from llb.core.contracts.data_prep.book_resolve import (
    CorrectionEdit,
    OrphanedEdit,
    PageCorrectionOverlay,
    PageResolvedText,
    ResolvedBlock,
)
from llb.prep.book.resolve.anchor import block_anchor
from llb.prep.book.resolve.constants import (
    AUTHOR_RANK,
    OVERLAY_KIND,
    RESOLVED_KIND,
    SCHEMA_VERSION,
)
from llb.prep.book.resolve.primary import PrimaryChoice, select_reading
from llb.prep.book.resolve.spans import cover_spans


def resolve_page(
    ledger: PageBlockLedger, choice: PrimaryChoice, edits: list[CorrectionEdit]
) -> tuple[PageCorrectionOverlay, PageResolvedText]:
    """Resolve every block. `edits` are model and reviewer corrections, never agreement."""
    by_id = {block.block_id: block for block in ledger.blocks}
    grouped: dict[str, list[CorrectionEdit]] = {block_id: [] for block_id in by_id}
    orphaned: list[OrphanedEdit] = []
    for edit in edits:
        if edit.page_id != ledger.page_id:
            continue
        if edit.block_id not in by_id:
            orphaned.append(OrphanedEdit(edit=edit, reason="block_missing", requeued=True))
            continue
        grouped[edit.block_id].append(edit)
    applied: list[CorrectionEdit] = []
    resolved: list[ResolvedBlock] = []
    for block in ledger.blocks:
        block_resolved, block_applied, block_orphans = resolve_block(
            ledger.page_id, block, choice, grouped[block.block_id]
        )
        resolved.append(block_resolved)
        applied.extend(block_applied)
        orphaned.extend(block_orphans)
    orphaned.sort(key=lambda item: (item.edit.block_id, item.edit.edit_id))
    overlay = PageCorrectionOverlay(
        schema_id="llb.book-ocr-correction-overlay",
        schema_version=SCHEMA_VERSION,
        kind=OVERLAY_KIND,
        page_id=ledger.page_id,
        pdf_page=ledger.pdf_page,
        applied=applied,
        orphaned=orphaned,
    )
    page = PageResolvedText(
        schema_id="llb.book-ocr-resolved-page",
        schema_version=SCHEMA_VERSION,
        kind=RESOLVED_KIND,
        page_id=ledger.page_id,
        pdf_page=ledger.pdf_page,
        page_class=choice.page_class,
        primary_stage=choice.stage,
        primary_engine=choice.engine,
        primary_variant=choice.variant,
        primary_model=choice.model,
        blocks=resolved,
    )
    return overlay, page


def resolve_block(
    page_id: str, block: Block, choice: PrimaryChoice, edits: list[CorrectionEdit]
) -> tuple[ResolvedBlock, list[CorrectionEdit], list[OrphanedEdit]]:
    """One block: the adopted text, the edits that produced it, and the edits it refused."""
    anchor = block_anchor(block)
    primary, fallback = select_reading(block, choice)
    kept, orphaned = _attach(primary, anchor, edits)
    winners = _winners(kept)
    spans, agreement = cover_spans(page_id, block, primary, anchor, winners)
    applied = sorted([*winners, *agreement], key=lambda edit: (edit.start, edit.edit_id))
    text = "".join(span.text for span in spans if span.adopted)
    resolved = ResolvedBlock(
        block_id=block.block_id,
        anchor=anchor,
        engine=primary.engine,
        variant=primary.variant,
        model=primary.model,
        fallback=fallback,
        primary_text=primary.text,
        text=text,
        spans=spans,
    )
    return resolved, applied, orphaned


def _attach(
    primary: RawReading, anchor: str, edits: list[CorrectionEdit]
) -> tuple[list[CorrectionEdit], list[OrphanedEdit]]:
    kept: list[CorrectionEdit] = []
    orphaned: list[OrphanedEdit] = []
    for edit in edits:
        if edit.anchor != anchor:
            orphaned.append(OrphanedEdit(edit=edit, reason="block_changed", requeued=True))
            continue
        if edit.end > len(primary.text) or primary.text[edit.start : edit.end] != edit.raw_text:
            orphaned.append(OrphanedEdit(edit=edit, reason="raw_mismatch", requeued=True))
            continue
        kept.append(edit)
    return kept, orphaned


def _winners(edits: list[CorrectionEdit]) -> list[CorrectionEdit]:
    """Higher-precedence edits consume their span. An overlapped edit is not applied."""
    ordered = sorted(edits, key=lambda edit: (AUTHOR_RANK[edit.author], edit.start, edit.edit_id))
    chosen: list[CorrectionEdit] = []
    covered: list[tuple[int, int]] = []
    for edit in ordered:
        if any(edit.start < end and start < edit.end for start, end in covered):
            continue
        chosen.append(edit)
        covered.append((edit.start, edit.end))
    return chosen
