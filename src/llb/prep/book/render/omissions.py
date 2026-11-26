"""Blank pages and unread regions, attached after the resolved blocks.

A blank scan becomes one blank block. Each unread region becomes one block
in original pixel coordinates. A page that already has blocks keeps them.
"""

from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book_blocks import PageBlockLedger
from llb.prep.book.review.coverage import UnreadRegion, account_page
from llb.prep.book.review.crops import crop_filename
from llb.prep.book.render.types import RenderBlock


def with_coverage(
    blocks: list[RenderBlock],
    image: np.ndarray | None,
    ledger: PageBlockLedger,
    page_dir: Path,
) -> list[RenderBlock]:
    """The page's blocks plus a blank marker or one block per unread region."""
    account = account_page(image, ledger, page_dir)
    if account.coverage == "blank":
        return [_blank(ledger.pdf_page, ledger.page_id)]
    if not account.regions:
        return blocks
    extra = [
        _omission(ledger.page_id, ledger.pdf_page, index, region)
        for index, region in enumerate(account.regions)
    ]
    return [*blocks, *extra]


def _blank(pdf_page: int, page_id: str) -> RenderBlock:
    return RenderBlock(
        pdf_page=pdf_page,
        page_id=page_id,
        ordinal=0,
        reading_order=0,
        kind="unknown",
        role="blank",
        block_id=page_id,
        digest="",
        text="",
        region=None,
        resolved=None,
    )


def _omission(page_id: str, pdf_page: int, index: int, region: UnreadRegion) -> RenderBlock:
    crop = f"review/{crop_filename(region.block_id, 0, 0)}"
    return RenderBlock(
        pdf_page=pdf_page,
        page_id=page_id,
        ordinal=10_000 + index,
        reading_order=10**6 + index,
        kind="unknown",
        role="omission",
        block_id=region.block_id,
        digest=region.anchor,
        text="",
        region=region.region,
        resolved=None,
        crops=((0, 0, crop),),
    )
