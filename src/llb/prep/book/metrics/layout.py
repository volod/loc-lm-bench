"""Where a page's blocks sit: figure overlap, captions, reading order, and the printed page.

A text-only lane cannot express a figure box, a caption link, or a table grid, so those metrics
are absent there. A resolved page that did not emit one of those fields is absent too. Reading
order on a text-only lane is the order its plain text happens to contain the reference readings.
"""

import re
from collections.abc import Sequence

from llb.prep.book.metrics.constants import FIELD_CAPTION, FIELD_FIGURE, FIELD_PRINTED
from llb.prep.book.metrics.geometry import (
    Box,
    intersection_over_union,
    match_indices,
    mean_overlap,
    rank_agreement,
)
from llb.prep.book.metrics.reading import reference_reading
from llb.prep.book.metrics.reference import RefBlock, RefPage
from llb.prep.book.metrics.report import MetricValue, absent, maybe, not_applicable, scored
from llb.prep.book.metrics.submission import HypBlock, PageSubmission


def figures(page: RefPage, submission: PageSubmission) -> MetricValue:
    """Mean overlap of reference figures. An unmatched figure contributes zero."""
    found = [block for block in page.blocks if block.kind == "figure"]
    if not found:
        return not_applicable()
    if submission.text_only or FIELD_FIGURE in submission.unsupported:
        return absent()
    hypothesis = [
        block for block in submission.blocks if block.kind == "figure" and block.region is not None
    ]
    boxes = [block.region for block in hypothesis if block.region is not None]
    overlaps = [
        (left, intersection_over_union(box_of(found[left]), boxes[right]))
        for left, right in match_indices([box_of(block) for block in found], boxes)
    ]
    return scored(mean_overlap([box_of(block) for block in found], overlaps))


def captions(page: RefPage, submission: PageSubmission) -> MetricValue:
    """Share of figures whose linked caption is the reference caption, after matching both."""
    found = _reference_kind(page, "figure")
    if not found:
        return not_applicable()
    if submission.text_only or FIELD_CAPTION in submission.unsupported:
        return absent()
    figure_match = _match_blocks(found, _hypothesis_kind(submission, "figure"))
    reference_captions = _reference_kind(page, "caption")
    caption_match = _match_blocks(reference_captions, _hypothesis_kind(submission, "caption"))
    by_id = {block.block_id: index for index, block in enumerate(reference_captions)}
    return scored(_caption_hits(found, figure_match, caption_match, by_id) / len(found))


def _reference_kind(page: RefPage, kind: str) -> list[RefBlock]:
    return [block for block in page.blocks if block.kind == kind]


def _hypothesis_kind(submission: PageSubmission, kind: str) -> list[HypBlock]:
    return [block for block in submission.blocks if block.kind == kind]


def _match_blocks(reference: list[RefBlock], hypothesis: list[HypBlock]) -> dict[int, HypBlock]:
    boxes = [box_of(block) for block in reference]
    regions = [block.region for block in hypothesis]
    return {left: hypothesis[right] for left, right in match_indices(boxes, regions)}


def _caption_hits(
    found: list[RefBlock],
    figure_match: dict[int, HypBlock],
    caption_match: dict[int, HypBlock],
    by_id: dict[str, int],
) -> int:
    hits = 0
    for index, figure in enumerate(found):
        if _caption_matches(figure_match, caption_match, by_id, index, figure):
            hits += 1
    return hits


def _caption_matches(
    figure_match: dict[int, HypBlock],
    caption_match: dict[int, HypBlock],
    by_id: dict[str, int],
    index: int,
    figure: RefBlock,
) -> bool:
    linked = figure_match.get(index)
    if figure.caption_id is None:
        return linked is not None and linked.caption_id is None
    target = by_id.get(figure.caption_id)
    if target is None:
        return False
    matched = caption_match.get(target)
    if linked is None or matched is None:
        return False
    return linked.caption_id == matched.block_id


def order(page: RefPage, submission: PageSubmission, hypothesis_text: str) -> MetricValue:
    """Rank agreement of reading order. A text-only lane uses character offsets in its plain text."""
    if submission.text_only:
        return maybe(rank_agreement(_text_ranks(page, hypothesis_text)))
    pairs_found: list[tuple[int, int]] = []
    for kind in {block.kind for block in page.blocks}:
        for reference, hypothesis in pairs(page.blocks, submission.blocks, kind):
            if hypothesis is not None:
                pairs_found.append((reference.reading_order, hypothesis.reading_order))
    return maybe(rank_agreement(pairs_found))


def printed(page: RefPage, submission: PageSubmission, hypothesis_text: str) -> MetricValue:
    """Score a structured folio, or look for a known folio in text-only output.

    A resolved page scores the printed-page mapping the renderers publish. A matching
    token in the body, or on a page-number block, does not stand in for that mapping.
    """
    if FIELD_PRINTED in submission.unsupported:
        return absent()
    if submission.uses_emitted_folio:
        return _emitted_folio(page, submission.emitted_folio)
    numbers = [block for block in submission.blocks if block.kind == "page-number"]
    if page.printed_page is None:
        if submission.text_only:
            return absent()
        return scored(float(submission.present and not numbers))
    want = _page_token(page.printed_page)
    if not submission.text_only:
        return scored(float(len(numbers) == 1 and _page_token(numbers[0].text) == want))
    return scored(
        1.0 if re.search(rf"(?<![\w]){re.escape(want)}(?![\w])", hypothesis_text) else 0.0
    )


def _emitted_folio(page: RefPage, folio: str | None) -> MetricValue:
    if page.printed_page is None:
        return scored(float(folio is None))
    if folio is None:
        return scored(0.0)
    return scored(float(_page_token(folio) == _page_token(page.printed_page)))


def pairs(
    reference: Sequence[RefBlock], hypothesis: Sequence[HypBlock], kind: str
) -> list[tuple[RefBlock, HypBlock | None]]:
    """Reference blocks of ``kind`` paired with a hypothesis block, or None when unmatched."""
    refs = [block for block in reference if block.kind == kind]
    hyps = [block for block in hypothesis if block.kind == kind]
    if hyps and _all_boxed(hyps):
        return _paired_by_overlap(refs, hyps)
    return _paired_by_order(refs, hyps)


def _all_boxed(hyps: list[HypBlock]) -> bool:
    return all(block.region is not None for block in hyps)


def _paired_by_overlap(
    refs: list[RefBlock], hyps: list[HypBlock]
) -> list[tuple[RefBlock, HypBlock | None]]:
    boxes = [block.region for block in hyps if block.region is not None]
    matched = {
        left: hyps[right] for left, right in match_indices([box_of(block) for block in refs], boxes)
    }
    return [(block, matched.get(index)) for index, block in enumerate(refs)]


def _paired_by_order(
    refs: list[RefBlock], hyps: list[HypBlock]
) -> list[tuple[RefBlock, HypBlock | None]]:
    return [(block, hyps[index] if index < len(hyps) else None) for index, block in enumerate(refs)]


def box_of(block: RefBlock) -> Box:
    """The reference region as a box. A reference block always has one."""
    region = block.region
    return Box(region.x0, region.y0, region.x1, region.y1)


def _text_ranks(page: RefPage, hypothesis_text: str) -> list[tuple[int, int]]:
    """Character offsets of each block's reading. A hit is blanked so a repeat can match again."""
    mutable = list(hypothesis_text)
    found_pairs: list[tuple[int, int]] = []
    for block in sorted(page.blocks, key=lambda item: item.reading_order):
        needle = reference_reading(block)
        if not needle:
            continue
        found = "".join(mutable).find(needle)
        if found < 0:
            continue
        found_pairs.append((block.reading_order, found))
        for index in range(found, found + len(needle)):
            mutable[index] = "\0"
    return found_pairs


def _page_token(value: str) -> str:
    return re.sub(r"\s+", "", value).strip(".,;:")
