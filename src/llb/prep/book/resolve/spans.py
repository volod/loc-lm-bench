"""Tile one block's primary reading into resolved spans.

Overlay edits win their whole span. The gaps are agreement where every reading matches the
primary, unresolved where an uncertainty span is still open, and the primary reading elsewhere.
"""

from typing import Literal

from llb.core.contracts.data_prep.book_blocks import Block, RawReading
from llb.core.contracts.data_prep.book_resolve import CorrectionEdit, ResolvedSpan
from llb.prep.book.resolve.constants import AGREEMENT_EVIDENCE, AGREEMENT_SOURCE
from llb.prep.book.resolve.masks import agreed_mask, classes_covering, uncertain_mask

BaseKind = Literal["primary", "agreement", "unresolved"]


def cover_spans(
    page_id: str,
    block: Block,
    primary: RawReading,
    anchor: str,
    winners: list[CorrectionEdit],
) -> tuple[list[ResolvedSpan], list[CorrectionEdit]]:
    """The spans of one block, and the agreement edits those spans record."""
    agreed = agreed_mask(primary, block.readings)
    uncertain = uncertain_mask(primary, block.uncertainty)
    edits = sorted(winners, key=lambda edit: (edit.start, edit.edit_id))
    spans: list[ResolvedSpan] = []
    agreement: list[CorrectionEdit] = []
    cursor, edit_index, length = 0, 0, len(primary.text)
    while cursor < length or edit_index < len(edits):
        edit_index, taken = _take_edit(edits, edit_index, cursor, spans)
        if taken is not None:
            cursor = taken
            continue
        if cursor >= length:
            break
        cursor = _append_base(
            page_id,
            block,
            primary,
            anchor,
            agreed,
            uncertain,
            edits,
            edit_index,
            cursor,
            length,
            spans,
            agreement,
        )
    return spans, agreement


def _take_edit(
    edits: list[CorrectionEdit], edit_index: int, cursor: int, spans: list[ResolvedSpan]
) -> tuple[int, int | None]:
    """Apply the edit that starts at `cursor`. Edits that start earlier were already passed."""
    while edit_index < len(edits) and edits[edit_index].start < cursor:
        edit_index += 1
    if edit_index >= len(edits) or edits[edit_index].start != cursor:
        return edit_index, None
    edit = edits[edit_index]
    spans.append(_edit_span(edit))
    return edit_index + 1, edit.end


def _append_base(
    page_id: str,
    block: Block,
    primary: RawReading,
    anchor: str,
    agreed: list[bool],
    uncertain: list[bool],
    edits: list[CorrectionEdit],
    edit_index: int,
    cursor: int,
    length: int,
    spans: list[ResolvedSpan],
    agreement: list[CorrectionEdit],
) -> int:
    """Append one base run and return the cursor just after it."""
    edit = edits[edit_index] if edit_index < len(edits) else None
    stop = length if edit is None else min(length, edit.start)
    kind = _kind(agreed, uncertain, cursor)
    index = cursor + 1
    while index < stop and _kind(agreed, uncertain, index) == kind:
        index += 1
    spans.append(_base_span(page_id, block, primary, anchor, kind, cursor, index, agreement))
    return index


def _edit_span(edit: CorrectionEdit) -> ResolvedSpan:
    return ResolvedSpan(
        provenance=edit.author,
        text=edit.replacement,
        start=edit.start,
        end=edit.end,
        adopted=True,
        edit_id=edit.edit_id,
    )


def _base_span(
    page_id: str,
    block: Block,
    primary: RawReading,
    anchor: str,
    kind: BaseKind,
    start: int,
    end: int,
    agreement: list[CorrectionEdit],
) -> ResolvedSpan:
    text = primary.text[start:end]
    edit_id = None
    classes = []
    if kind == "agreement":
        edit_id = f"agreement:{block.block_id}:{start}:{end}"
        agreement.append(
            CorrectionEdit(
                edit_id=edit_id,
                page_id=page_id,
                block_id=block.block_id,
                anchor=anchor,
                author="agreement",
                start=start,
                end=end,
                raw_text=text,
                replacement=text,
                evidence=AGREEMENT_EVIDENCE,
                source=AGREEMENT_SOURCE,
            )
        )
    elif kind == "unresolved":
        classes = classes_covering(primary, block.uncertainty, start, end)
    return ResolvedSpan(
        provenance=kind,
        text=text,
        start=start,
        end=end,
        adopted=kind != "unresolved",
        edit_id=edit_id,
        classes=classes,
    )


def _kind(agreed: list[bool], uncertain: list[bool], index: int) -> BaseKind:
    if uncertain[index]:
        return "unresolved"
    if agreed[index]:
        return "agreement"
    return "primary"
