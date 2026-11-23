"""Rotations, deskews, and the point maps that send a variant pixel back to the render.

Quarter turns use OpenCV's exact rotations, so an upright correction of a sideways page is the
original pixels, not a resample. Deskew expands the canvas until the whole page still fits: a map
that drops a corner is a crop, and the caller refuses it.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from llb.prep.book.preprocess.constants import COVER_CELL

_TURN = {
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}


@dataclass(frozen=True)
class Affine:
    """Variant pixel to original pixel, as a 2x3 matrix, plus both image sizes."""

    matrix: tuple[tuple[float, float, float], tuple[float, float, float]]
    width: int
    height: int
    src_width: int
    src_height: int


def identity(width: int, height: int) -> Affine:
    """The map of an image that was not moved."""
    return Affine(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), width, height, width, height)


def rotate_upright(grey: np.ndarray, angle: int) -> tuple[np.ndarray, Affine]:
    """Turn `grey` so content rotated `angle` degrees clockwise reads upright."""
    height, width = grey.shape[:2]
    if angle == 0:
        return grey, identity(width, height)
    turned = np.asarray(cv2.rotate(grey, _TURN[angle]), dtype=np.uint8)
    new_h, new_w = turned.shape[:2]
    return turned, Affine(_rotation_matrix(angle, width, height), new_w, new_h, width, height)


def deskew(grey: np.ndarray, angle: float) -> tuple[np.ndarray, Affine]:
    """Rotate by `angle` degrees (OpenCV's counter-clockwise) and keep the whole page."""
    height, width = grey.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    forward = np.asarray(cv2.getRotationMatrix2D(center, angle, 1.0), dtype=np.float64)
    cosine, sine = abs(float(forward[0, 0])), abs(float(forward[0, 1]))
    new_w = int(round(height * sine + width * cosine))
    new_h = int(round(height * cosine + width * sine))
    forward[0, 2] += (new_w - width) / 2.0
    forward[1, 2] += (new_h - height) / 2.0
    image = cv2.warpAffine(grey, forward, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=255)
    inverse = _invert(forward)
    return np.asarray(image, dtype=np.uint8), Affine(_rows(inverse), new_w, new_h, width, height)


def compose(inner: Affine, outer: Affine) -> Affine:
    """`outer` maps the variant to an intermediate image; `inner` maps that image to the original."""
    full = _homogeneous(inner.matrix) @ _homogeneous(outer.matrix)
    return Affine(_rows(full[:2]), outer.width, outer.height, inner.src_width, inner.src_height)


def apply_affine(
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]], x: float, y: float
) -> tuple[float, float]:
    """One point through a 2x3 matrix."""
    row0, row1 = matrix
    return row0[0] * x + row0[1] * y + row0[2], row1[0] * x + row1[1] * y + row1[2]


def affine_covers(transform: Affine) -> bool:
    """Every inset corner and edge midpoint of the original lands inside the variant."""
    forward = _invert(np.asarray(transform.matrix, dtype=np.float64))
    mapped = _rows(forward)
    inset = 2.0
    width, height = transform.src_width, transform.src_height
    points = (
        (inset, inset),
        (width - 1 - inset, inset),
        (inset, height - 1 - inset),
        (width - 1 - inset, height - 1 - inset),
        (width / 2, inset),
        (width / 2, height - 1 - inset),
    )
    return all(_inside(apply_affine(mapped, x, y), transform) for x, y in points)


def grid_covers(map_x: np.ndarray, map_y: np.ndarray, src_width: int, src_height: int) -> bool:
    """The sampled map hits every cell of the original page, inset by one cell."""
    cells_y = max(src_height // COVER_CELL, 1)
    cells_x = max(src_width // COVER_CELL, 1)
    occupied = np.zeros((cells_y, cells_x), dtype=bool)
    ox = np.clip((map_x / COVER_CELL).astype(np.int32), 0, cells_x - 1)
    oy = np.clip((map_y / COVER_CELL).astype(np.int32), 0, cells_y - 1)
    occupied[oy.ravel(), ox.ravel()] = True
    if cells_y < 5 or cells_x < 5:
        return bool(occupied.mean() > 0.9)
    return bool(occupied[1:-1, 1:-1].all())


def push_through(
    map_x: np.ndarray,
    map_y: np.ndarray,
    transform: Affine,
) -> tuple[np.ndarray, np.ndarray]:
    """Send a variant-to-intermediate grid the rest of the way to the original."""
    row0, row1 = transform.matrix
    original_x = row0[0] * map_x + row0[1] * map_y + row0[2]
    original_y = row1[0] * map_x + row1[1] * map_y + row1[2]
    return original_x.astype(np.float32), original_y.astype(np.float32)


def _rotation_matrix(
    angle: int, width: int, height: int
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if angle == 90:
        return ((0.0, -1.0, float(width - 1)), (1.0, 0.0, 0.0))
    if angle == 180:
        return ((-1.0, 0.0, float(width - 1)), (0.0, -1.0, float(height - 1)))
    if angle == 270:
        return ((0.0, 1.0, 0.0), (-1.0, 0.0, float(height - 1)))
    raise ValueError(f"rotation {angle} is not a quarter turn")


def _invert(matrix: np.ndarray) -> np.ndarray:
    linear = matrix[:, :2]
    translation = matrix[:, 2]
    inverse = np.linalg.inv(linear)
    return np.hstack((inverse, (-inverse @ translation).reshape(2, 1)))


def _homogeneous(
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> np.ndarray:
    return np.vstack((np.asarray(matrix, dtype=np.float64), (0.0, 0.0, 1.0)))


def _rows(matrix: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
    )


def _inside(point: tuple[float, float], transform: Affine) -> bool:
    x, y = point
    return -0.5 <= x < transform.width + 0.5 and -0.5 <= y < transform.height + 0.5
