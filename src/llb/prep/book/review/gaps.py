"""Queue entries for an unread region and for a page whose recognition failed."""

import hashlib
from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book_blocks import Region
from llb.core.contracts.data_prep.book_review import ReviewQueueEntry
from llb.prep.book.cache import page_dir
from llb.prep.book.preprocess.images import read_pixels
from llb.prep.book.resolve.constants import REVIEW_DIRNAME
from llb.prep.book.review.coverage import PageAccount
from llb.prep.book.review.crops import crop_filename, write_block_crop
from llb.prep.book.review.marks import SpanMark
from llb.prep.book.stages.render import PAGE_IMAGE_FILENAME, RENDER_STAGE


def omission_marks(account: PageAccount, page_id: str, pdf_page: int) -> list[SpanMark]:
    """One unreadable mark per unread region, in the order the regions were found."""
    marks: list[SpanMark] = []
    for index, region in enumerate(account.regions):
        marks.append(
            SpanMark(
                page_id=page_id,
                pdf_page=pdf_page,
                block_id=region.block_id,
                ordinal=10_000 + index,
                anchor=region.anchor,
                kind="unreadable",
                reason="omission",
                start=0,
                end=0,
                raw_text="",
                candidate=None,
                classes=(),
                block_present=False,
                region=region.region,
            )
        )
    return marks


def failure_entry(
    page_id: str, pdf_page: int, pdf_link: str, reason: str, run_dir: Path
) -> ReviewQueueEntry:
    """The queue row for a page that will not gain a resolved text."""
    block_id = f"{page_id}-r0000"
    marker = f'[!page-failed]({pdf_link} "{page_id} {reason}")'
    image, box = _page_box(run_dir, page_id)
    crop = _crop(image, box, block_id, run_dir) if image is not None and box is not None else None
    return ReviewQueueEntry(
        entry_id=crop_filename(block_id, 0, 0).removesuffix(".png"),
        page_id=page_id,
        pdf_page=pdf_page,
        block_id=block_id,
        ordinal=0,
        anchor=_anchor(page_id),
        kind="page-failed",
        reason="page-failed",
        start=0,
        end=0,
        raw_text="",
        marker=marker,
        crop=crop,
        pdf_link=pdf_link,
        rendered=marker,
        block_present=False,
        box=_numbers(box),
    )


def _page_box(run_dir: Path, page_id: str) -> tuple[np.ndarray | None, Region | None]:
    path = page_dir(run_dir, page_id) / RENDER_STAGE / PAGE_IMAGE_FILENAME
    if not path.is_file():
        return None, None
    image = read_pixels(path)
    height, width = int(image.shape[0]), int(image.shape[1])
    region = Region(x0=0, y0=0, x1=float(width), y1=float(height), width=width, height=height)
    return image, region


def _crop(image: np.ndarray, region: Region, block_id: str, run_dir: Path) -> str | None:
    name = crop_filename(block_id, 0, 0)
    if not write_block_crop(image, region, run_dir / REVIEW_DIRNAME / name):
        return None
    return f"{REVIEW_DIRNAME}/{name}"


def _numbers(region: Region | None) -> list[float]:
    if region is None:
        return []
    return [region.x0, region.y0, region.x1, region.y1]


def _anchor(page_id: str) -> str:
    return hashlib.sha256(f"{page_id}:page-failed".encode("ascii")).hexdigest()
