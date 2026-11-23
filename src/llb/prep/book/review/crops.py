"""Padded crops of a block region, written as deterministic PNGs."""

import math
import re
from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book_blocks import Region
from llb.prep.book.preprocess.images import write_png
from llb.prep.book.review.constants import CROP_PADDING_PX

CROP_NAME = re.compile(r"^p\d+-b\d+-s\d+-\d+\.png$")


def crop_filename(block_id: str, start: int, end: int) -> str:
    """`p0001-b0000-s0000-0004.png`. The same span always gets the same name."""
    return f"{block_id}-s{start:04d}-{end:04d}.png"


def write_block_crop(
    image: np.ndarray, region: Region, dest: Path, padding: int = CROP_PADDING_PX
) -> bool:
    """Write the region plus `padding` pixels, clipped to the page. False when the box is empty."""
    box = padded_box(region, image.shape, padding)
    if box is None:
        return False
    x0, y0, x1, y1 = box
    write_png(dest, image[y0:y1, x0:x1])
    return True


def padded_box(
    region: Region, shape: tuple[int, ...], padding: int
) -> tuple[int, int, int, int] | None:
    """Pixel box of `region` expanded by `padding` and clipped to `shape`."""
    height, width = int(shape[0]), int(shape[1])
    x0 = max(0, math.floor(region.x0) - padding)
    y0 = max(0, math.floor(region.y0) - padding)
    x1 = min(width, math.ceil(region.x1) + padding)
    y1 = min(height, math.ceil(region.y1) + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1
