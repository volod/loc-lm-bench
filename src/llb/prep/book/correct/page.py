"""Correct one page: propose per block, guard each proposal, write corrections.json.

The block ledger is read and checked unchanged. Refusals stay in the corrections file so
resolve can mark those spans instead of applying them.
"""

import logging
from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book_blocks import Block
from llb.core.contracts.data_prep.book_resolve import (
    CorrectionEdit,
    ModelCorrections,
    RefusedCorrection,
)
from llb.prep.book.cache import page_dir
from llb.prep.book.constants import BLOCKS_FILENAME
from llb.prep.book.correct.cache import cache_key, read_cache, write_cache
from llb.prep.book.correct.constants import CORRECT_VERSION, CROP_PAD_PX, SKIP_KINDS
from llb.prep.book.correct.context import Neighbours, neighbour_window, reading_order
from llb.prep.book.correct.edges import edge_text
from llb.prep.book.correct.guard import Refusal, judge, marker_classes
from llb.prep.book.correct.params import scripted_proposals
from llb.prep.book.correct.proposal import BlockPrompt, Proposal, ProposalResult, Proposer
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.preprocess.images import encode_png, read_grey
from llb.prep.book.resolve.anchor import block_anchor
from llb.prep.book.resolve.constants import (
    CORRECT_STAGE,
    CORRECTIONS_FILENAME,
    MODEL_KIND,
    MODEL_SCHEMA_ID,
    SCHEMA_VERSION,
)
from llb.prep.book.resolve.io import write_corrections
from llb.prep.book.resolve.primary import PrimaryChoice, choice_from_params, select_reading
from llb.prep.book.stages import StageContext, StageResult
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RENDER_STAGE
from llb.prep.book.stages.recognize import BLOCKS_STAGE

_LOG = logging.getLogger(__name__)


def correct_page(ctx: StageContext, proposer: Proposer | None) -> StageResult:
    """Write this page's admitted and refused edits. Raise if the ledger bytes move."""
    ledger_path = ctx.inputs[BLOCKS_STAGE] / BLOCKS_FILENAME
    before = ledger_path.read_bytes()
    ledger = read_ledger(ledger_path)
    image = read_grey(ctx.inputs[RENDER_STAGE] / PAGE_IMAGE_FILENAME)
    choice = _choice(ctx)
    ordered = reading_order(ledger.blocks)
    texts = [select_reading(block, choice)[0].text for block in ordered]
    page_count = int(getattr(ctx.document, "page_count", ctx.pdf_page) or ctx.pdf_page)
    previous = edge_text(ctx.run_dir, ctx.pdf_page - 1, page_count, choice, last=True)
    nxt = edge_text(ctx.run_dir, ctx.pdf_page + 1, page_count, choice, last=False)
    prior = page_dir(ctx.run_dir, ctx.page_id) / CORRECT_STAGE
    edits: list[CorrectionEdit] = []
    refused: list[RefusedCorrection] = []
    cached = 0
    unparsed = 0
    considered = 0
    for index, block in enumerate(ordered):
        if block.kind in SKIP_KINDS or not texts[index]:
            continue
        considered += 1
        window = neighbour_window(ordered, index, texts, previous, nxt)
        crop = _crop(image, block)
        saw_crop = crop.size > 0
        result, hit = _reply(ctx, proposer, block, texts[index], window, crop, prior)
        cached += int(hit)
        unparsed += int(result.unparsed)
        judgement = judge(texts[index], list(result.proposals), saw_crop=saw_crop)
        edits.extend(_edits(ctx, block, texts[index], judgement.admitted))
        refused.extend(_refusals(ctx, block, judgement.refused))
    write_corrections(
        ctx.work_dir / CORRECTIONS_FILENAME,
        ModelCorrections(
            schema_id=MODEL_SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            kind=MODEL_KIND,
            page_id=ctx.page_id,
            pdf_page=ctx.pdf_page,
            model=ctx.model or "",
            edits=edits,
            refused=refused,
        ),
    )
    if ledger_path.read_bytes() != before:
        raise RuntimeError("correct changed the block ledger")
    _LOG.info(
        "correct page %s: admitted %s, refused %s, cached %s",
        ctx.page_id,
        len(edits),
        len(refused),
        cached,
    )
    return StageResult(
        meta={
            "version": CORRECT_VERSION,
            "blocks": considered,
            "admitted": len(edits),
            "refused": len(refused),
            "cached": cached,
            "unparsed": unparsed,
        }
    )


def _choice(ctx: StageContext) -> PrimaryChoice:
    return choice_from_params(ctx.params)


def _reply(
    ctx: StageContext,
    proposer: Proposer | None,
    block: Block,
    text: str,
    window: Neighbours,
    crop: np.ndarray,
    prior: Path,
) -> tuple[ProposalResult, bool]:
    if proposer is None:
        proposals = scripted_proposals(ctx.params, ctx.page_id, text)
        return ProposalResult(tuple(proposals)), False
    prompt = BlockPrompt(
        page_id=ctx.page_id,
        block_id=block.block_id,
        text=text,
        before=window.before,
        after=window.after,
        previous_page=window.previous_page,
        next_page=window.next_page,
        crop_png=encode_png(crop),
    )
    key = cache_key(
        ctx.model or "",
        block_anchor(block),
        prompt.crop_png,
        text,
        window.before,
        window.after,
        window.previous_page,
        window.next_page,
    )
    stored = read_cache(prior, block.block_id, key)
    if stored is not None:
        write_cache(ctx.work_dir, block.block_id, key, stored)
        return stored, True
    result = proposer.propose(prompt)
    write_cache(ctx.work_dir, block.block_id, key, result)
    return result, False


def _crop(image: np.ndarray, block: Block) -> np.ndarray:
    height, width = image.shape[:2]
    region = block.region
    x0 = max(0, int(region.x0) - CROP_PAD_PX)
    y0 = max(0, int(region.y0) - CROP_PAD_PX)
    x1 = min(width, int(np.ceil(region.x1)) + CROP_PAD_PX)
    y1 = min(height, int(np.ceil(region.y1)) + CROP_PAD_PX)
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return image[y0:y1, x0:x1]


def _edits(
    ctx: StageContext, block: Block, text: str, proposals: tuple[Proposal, ...]
) -> list[CorrectionEdit]:
    anchor = block_anchor(block)
    edits: list[CorrectionEdit] = []
    for proposal in proposals:
        raw = text[proposal.start : proposal.end]
        edits.append(
            CorrectionEdit(
                edit_id=_edit_id(block.block_id, proposal),
                page_id=ctx.page_id,
                block_id=block.block_id,
                anchor=anchor,
                author="model",
                start=proposal.start,
                end=proposal.end,
                raw_text=raw,
                replacement=proposal.replacement,
                evidence=proposal.evidence,
                source=ctx.model or "model",
                confusion_class=proposal.confusion_class,
            )
        )
    return edits


def _refusals(
    ctx: StageContext, block: Block, refusals: tuple[Refusal, ...]
) -> list[RefusedCorrection]:
    anchor = block_anchor(block)
    rows: list[RefusedCorrection] = []
    for refusal in refusals:
        proposal = refusal.proposal
        rows.append(
            RefusedCorrection(
                page_id=ctx.page_id,
                block_id=block.block_id,
                anchor=anchor,
                start=proposal.start,
                end=proposal.end,
                raw_text=refusal.raw_text,
                replacement=proposal.replacement,
                confusion_class=proposal.confusion_class,
                reason=refusal.reason,
                evidence=proposal.evidence,
                classes=list(marker_classes(proposal.confusion_class)),
            )
        )
    return rows


def _edit_id(block_id: str, proposal: Proposal) -> str:
    return f"model:{block_id}:{proposal.start}:{proposal.end}:{proposal.confusion_class}"
