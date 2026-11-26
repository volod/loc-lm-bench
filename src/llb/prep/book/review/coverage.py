"""Compare a page image with the regions recognition covered.

A blank page has no ink and no blocks. An unread region is ink that a text
column or a layout box shows and that no OCR, formula, or table box covers.
The region is the unread band, so a stretch that was read stays out of the crop.
A confident reading of a neighbouring column does not cover the gap. The
coordinates are pixels of the rendered page.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from llb.core.contracts.data_prep.book_blocks import PageBlockLedger, Region
from llb.core.contracts.data_prep.book_review import CoverageStatus
from llb.prep.book.classes.content import bands, ink_mask
from llb.prep.book.classes.layout import column_spans
from llb.prep.book.constants import READING_FILENAME
from llb.prep.book.engines.protocol import read_reading

# Ink below this share of the page, and no blocks, is a blank scan.
BLANK_INK = 0.004
# Uncovered ink smaller than this is a speck, not a missed column.
MIN_INK_PIXELS = 150
# Share of a column's ink that recognition must cover.
COVERED_INK = 0.5
# The body band column detection uses. Margins are not a second column.
BODY_TOP = 0.08
BODY_BOTTOM = 0.92
BOX_PAD = 4
# Empty rows shorter than this share of the page stay in one unread region.
# A line gap merges. A run of covered lines splits the region.
BAND_MERGE = 0.06
# A thinner strip is a rule or a fringe, not a missed line.
MIN_BAND_PX = 24
_RECOGNITION = ("ocr", "formula", "table")
_LAYOUT = ("layout",)
_Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class UnreadRegion:
    """One region of original ink that recognition did not cover."""

    block_id: str
    anchor: str
    region: Region


@dataclass(frozen=True, slots=True)
class PageAccount:
    """How this page's ink relates to the blocks that were returned.

    `read` means the ink is covered, not that every character is sure.
    """

    coverage: CoverageStatus
    regions: tuple[UnreadRegion, ...]


def account_page(
    image: np.ndarray | None, ledger: PageBlockLedger, page_dir: Path | None = None
) -> PageAccount:
    """Blank, read, or one unread region per missed column or layout box."""
    if image is None:
        return PageAccount("read", ())
    grey = _grey(image)
    ink = ink_mask(grey)
    if float(ink.mean()) < BLANK_INK and not ledger.blocks:
        return PageAccount("blank", ())
    recognition = _boxes(page_dir, _RECOGNITION)
    if recognition is None:
        recognition = [_corners(block.region) for block in ledger.blocks]
    covered = _covered(ink.shape, recognition)
    gaps = _layout_gaps(ink, covered, _boxes(page_dir, _LAYOUT) or [])
    gaps.extend(_column_gaps(ink, covered))
    regions = _regions(ledger.page_id, _dedupe(gaps), grey.shape)
    coverage: CoverageStatus = "omission" if regions else "read"
    return PageAccount(coverage, tuple(regions))


def _grey(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.asarray(image, dtype=np.uint8)
    pixels = image[:, :, :3] if image.ndim == 3 and image.shape[2] == 4 else image
    return np.asarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY), dtype=np.uint8)


def _boxes(page_dir: Path | None, stages: tuple[str, ...]) -> list[_Box] | None:
    """Stage boxes, or None when none of those stages wrote a reading."""
    if page_dir is None:
        return None
    found = False
    boxes: list[_Box] = []
    for stage in stages:
        path = page_dir / stage / READING_FILENAME
        if not path.is_file():
            continue
        found = True
        for block in read_reading(path).blocks:
            boxes.append((block.x0, block.y0, block.x1, block.y1))
    return boxes if found else None


def _corners(region: Region) -> _Box:
    return (region.x0, region.y0, region.x1, region.y1)


def _covered(shape: tuple[int, ...], boxes: list[_Box]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        left = max(0, int(np.floor(x0)) - BOX_PAD)
        top = max(0, int(np.floor(y0)) - BOX_PAD)
        right = min(width, int(np.ceil(x1)) + BOX_PAD)
        bottom = min(height, int(np.ceil(y1)) + BOX_PAD)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
    return mask


def _layout_gaps(ink: np.ndarray, covered: np.ndarray, boxes: list[_Box]) -> list[_Box]:
    height, width = ink.shape
    gaps: list[_Box] = []
    for box in boxes:
        gaps.extend(_gap(ink, covered, _clamp(box, width, height)))
    return gaps


def _column_gaps(ink: np.ndarray, covered: np.ndarray) -> list[_Box]:
    height, width = ink.shape
    _count, spans = column_spans(ink)
    y0, y1 = int(height * BODY_TOP), int(height * BODY_BOTTOM)
    if y1 <= y0:
        return []
    gaps: list[_Box] = []
    for left, right in spans:
        gaps.extend(_gap(ink, covered, (float(left), float(y0), float(right), float(y1))))
    return gaps


def _gap(ink: np.ndarray, covered: np.ndarray, window: _Box) -> list[_Box]:
    """Unread bands inside one column or layout box.

    The window is unread when recognition covers less than half its ink.
    The returned boxes are the unread bands, so a read stretch in the middle
    is not part of the crop.
    """
    height, width = ink.shape
    x0, y0, x1, y1 = (int(window[0]), int(window[1]), int(window[2]), int(window[3]))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return []
    patch = ink[y0:y1, x0:x1]
    total = int(patch.sum())
    if total < MIN_INK_PIXELS:
        return []
    seen = covered[y0:y1, x0:x1]
    if int((patch & seen).sum()) / total >= COVERED_INK:
        return []
    gap = patch & ~seen
    return _bands(gap, x0, y0, height)


def _bands(gap: np.ndarray, origin_x: int, origin_y: int, page_height: int) -> list[_Box]:
    min_row = max(8, int(gap.shape[1] * 0.02))
    merge_px = max(MIN_BAND_PX, int(page_height * BAND_MERGE))
    merged: list[tuple[int, int]] = []
    for start, end in bands(gap.sum(axis=1) >= min_row):
        if merged and start - merged[-1][1] <= merge_px:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    found: list[_Box] = []
    for top, bottom in merged:
        ys, xs = np.nonzero(gap[top:bottom])
        if ys.size == 0 or int(ys.max()) - int(ys.min()) + 1 < MIN_BAND_PX:
            continue
        if int(ys.size) < MIN_INK_PIXELS:
            continue
        found.append(
            (
                float(origin_x + int(xs.min())),
                float(origin_y + top + int(ys.min())),
                float(origin_x + int(xs.max()) + 1),
                float(origin_y + top + int(ys.max()) + 1),
            )
        )
    return found


def _clamp(box: _Box, width: int, height: int) -> _Box:
    return (
        max(0.0, box[0]),
        max(0.0, box[1]),
        min(float(width), box[2]),
        min(float(height), box[3]),
    )


def _dedupe(boxes: list[_Box]) -> list[_Box]:
    kept: list[_Box] = []
    for box in sorted(boxes, key=lambda item: (item[1], item[0], item[3], item[2])):
        if any(_overlap(box, other) >= COVERED_INK for other in kept):
            continue
        kept.append(box)
    return kept


def _overlap(left: _Box, right: _Box) -> float:
    width = min(left[2], right[2]) - max(left[0], right[0])
    height = min(left[3], right[3]) - max(left[1], right[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    smaller = min(_area(left), _area(right))
    if smaller <= 0:
        return 0.0
    return shared / smaller


def _area(box: _Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _regions(page_id: str, boxes: list[_Box], shape: tuple[int, ...]) -> list[UnreadRegion]:
    height, width = int(shape[0]), int(shape[1])
    found: list[UnreadRegion] = []
    for box in boxes:
        region = _fit(box, width, height)
        if region is None:
            continue
        block_id = f"{page_id}-r{len(found):04d}"
        found.append(UnreadRegion(block_id, _anchor(block_id, region), region))
    return found


def _fit(box: _Box, width: int, height: int) -> Region | None:
    x0, y0, x1, y1 = _clamp(box, width, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return Region(x0=x0, y0=y0, x1=x1, y1=y1, width=width, height=height)


def _anchor(block_id: str, region: Region) -> str:
    payload = f"{block_id}|{region.x0:.1f}|{region.y0:.1f}|{region.x1:.1f}|{region.y1:.1f}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
