"""Background, contrast, blur, noise, gutter shadow, and neighbour-page bleed.

These are read on the original render, so a bleed box and a gutter side are already in the frame
the block ledger uses. None of them crops the page.
"""

import cv2
import numpy as np

from llb.core.contracts.data_prep.book_preprocess import BleedBand
from llb.prep.book.preprocess.constants import (
    BLEED_DENSITY,
    BLEED_EDGE_FRACTION,
    GUTTER_DARK_FRACTION,
    TILES,
)
from llb.prep.book.preprocess.images import background_kernel, ink_mask


def unevenness(grey: np.ndarray) -> float:
    """Spread of the background's tile medians, as a fraction of the grey range."""
    kernel = background_kernel(grey.shape)
    background = cv2.medianBlur(grey, kernel)
    tiles = _tile_medians(background, TILES)
    return round(float(np.std(tiles) / 255.0), 4)


def contrast(grey: np.ndarray) -> float:
    """The gap between the 5th and 95th percentile, as a fraction of the grey range."""
    low, high = np.percentile(grey, (5, 95))
    return round(float((high - low) / 255.0), 4)


def blur_score(grey: np.ndarray) -> float:
    """Variance of the Laplacian. A sharp page scores higher than a blurred one."""
    return round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 2)


def noise_score(grey: np.ndarray) -> float:
    """Grain of the background: the std of a 3-pixel median residual, ink excluded."""
    residual = grey.astype(np.float32) - cv2.medianBlur(grey, 3).astype(np.float32)
    background = residual[~ink_mask(grey)]
    if background.size == 0:
        return 0.0
    return round(float(background.std()), 3)


def gutter(grey: np.ndarray) -> tuple[str, int]:
    """The darkest edge band, or `none` when no edge is darker than the page centre."""
    columns = np.median(grey, axis=0)
    rows = np.median(grey, axis=1)
    centre = float(np.median(columns[len(columns) // 4 : 3 * len(columns) // 4]))
    candidates = (
        ("left", _dark_width(columns, centre)),
        ("right", _dark_width(columns[::-1], centre)),
        ("top", _dark_width(rows, centre)),
        ("bottom", _dark_width(rows[::-1], centre)),
    )
    side, width = max(candidates, key=lambda item: item[1])
    if width == 0:
        return "none", 0
    return side, int(width)


def bleed_band(grey: np.ndarray) -> BleedBand | None:
    """A text-like strip in the outer margin, separated from the body by a gap."""
    ink = ink_mask(grey)
    height, width = ink.shape
    density = ink.mean(axis=0)
    body = _widest(density >= BLEED_DENSITY)
    if body is None:
        return None
    edge = max(8, int(width * BLEED_EDGE_FRACTION))
    left = _side_band(density, body, 0, edge, "left", height)
    right = _side_band(density, body, width - edge, width, "right", height)
    return right or left


def _tile_medians(image: np.ndarray, tiles: int) -> np.ndarray:
    height, width = image.shape[:2]
    values = []
    for y0, y1 in _edges(height, tiles):
        for x0, x1 in _edges(width, tiles):
            values.append(float(np.median(image[y0:y1, x0:x1])))
    return np.asarray(values, dtype=np.float64)


def _edges(length: int, tiles: int) -> list[tuple[int, int]]:
    cuts = [round(index * length / tiles) for index in range(tiles + 1)]
    return [
        (cuts[index], cuts[index + 1]) for index in range(tiles) if cuts[index + 1] > cuts[index]
    ]


def _dark_width(profile: np.ndarray, centre: float) -> int:
    limit = centre * (1.0 - GUTTER_DARK_FRACTION)
    width = 0
    for value in profile.tolist():
        if float(value) >= limit:
            break
        width += 1
    return width


def _widest(mask: np.ndarray) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            best = _longer(best, start, index)
            start = None
    if start is not None:
        best = _longer(best, start, len(mask))
    return best


def _longer(best: tuple[int, int] | None, start: int, end: int) -> tuple[int, int]:
    if best is None or end - start > best[1] - best[0]:
        return start, end
    return best


def _side_band(
    density: np.ndarray,
    body: tuple[int, int],
    x0: int,
    x1: int,
    side: str,
    height: int,
) -> BleedBand | None:
    region = density[x0:x1] >= BLEED_DENSITY
    if not region.any():
        return None
    run = _widest(region)
    if run is None:
        return None
    left, right = x0 + run[0], x0 + run[1]
    if _overlaps(left, right, body) or not _separated(left, right, body, density):
        return None
    return BleedBand(side=side, x0=left, y0=0, x1=right, y1=height)  # type: ignore[arg-type]


def _overlaps(left: int, right: int, body: tuple[int, int]) -> bool:
    return left < body[1] and right > body[0]


def _separated(left: int, right: int, body: tuple[int, int], density: np.ndarray) -> bool:
    gap = density[right : body[0]] if right <= body[0] else density[body[1] : left]
    return gap.size > 2 and float(gap.max()) < BLEED_DENSITY
