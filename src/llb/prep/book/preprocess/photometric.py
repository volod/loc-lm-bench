"""Corrections that change tone and leave every pixel where it was.

None of these binarises the page. The background division is the survey's illumination: a wide
median estimates the paper, and dividing by it lifts a shadow without turning the text into a
one-bit mask.
"""

import cv2
import numpy as np

from llb.prep.book.preprocess.constants import (
    BILATERAL_D,
    BILATERAL_SIGMA,
    CLAHE_CLIP,
    CLAHE_TILE,
    GUTTER_MAX_GAIN,
)
from llb.prep.book.preprocess.images import background_kernel


def flatten_background(grey: np.ndarray) -> np.ndarray:
    """Divide the page by a wide median background and keep it in 0..255."""
    kernel = background_kernel(grey.shape)
    background = cv2.medianBlur(grey, kernel).astype(np.float32) + 1.0
    flat = grey.astype(np.float32) / background * 255.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def local_contrast(grey: np.ndarray) -> np.ndarray:
    """CLAHE over small tiles. The clip limit keeps a thin stroke from blowing out."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    return np.asarray(clahe.apply(grey), dtype=np.uint8)


def denoise(grey: np.ndarray) -> np.ndarray:
    """A small bilateral filter: grain goes, an edge stays."""
    filtered = cv2.bilateralFilter(grey, BILATERAL_D, BILATERAL_SIGMA, BILATERAL_SIGMA)
    return np.asarray(filtered, dtype=np.uint8)


def lift_gutter(grey: np.ndarray, side: str, width: int) -> np.ndarray:
    """Brighten one edge band toward the page centre, and leave the rest of the page alone."""
    if side in ("left", "right"):
        return _lift_columns(grey, side, width)
    if side in ("top", "bottom"):
        return _lift_rows(grey, side, width)
    return grey


def _lift_columns(grey: np.ndarray, side: str, width: int) -> np.ndarray:
    profile = np.median(grey, axis=0).astype(np.float32)
    centre = float(np.median(profile[len(profile) // 4 : 3 * len(profile) // 4]))
    out = grey.astype(np.float32)
    span = max(width, 1)
    columns = _edge_indexes(grey.shape[1], span, side == "left")
    for index, x in enumerate(columns):
        feather = 1.0 - index / span
        out[:, x] *= _gain(float(profile[x]), centre, feather)
    return np.clip(out, 0, 255).astype(np.uint8)


def _lift_rows(grey: np.ndarray, side: str, width: int) -> np.ndarray:
    profile = np.median(grey, axis=1).astype(np.float32)
    centre = float(np.median(profile[len(profile) // 4 : 3 * len(profile) // 4]))
    out = grey.astype(np.float32)
    span = max(width, 1)
    rows = _edge_indexes(grey.shape[0], span, side == "top")
    for index, y in enumerate(rows):
        feather = 1.0 - index / span
        out[y, :] *= _gain(float(profile[y]), centre, feather)
    return np.clip(out, 0, 255).astype(np.uint8)


def _edge_indexes(length: int, span: int, from_start: bool) -> range:
    """Indexes from the outer edge inward, so the first one is the page border."""
    if from_start:
        return range(span)
    return range(length - 1, length - 1 - span, -1)


def _gain(value: float, centre: float, feather: float) -> float:
    raw = centre / max(float(value), 1.0)
    lifted = 1.0 + (min(raw, GUTTER_MAX_GAIN) - 1.0) * feather
    return max(lifted, 1.0)
