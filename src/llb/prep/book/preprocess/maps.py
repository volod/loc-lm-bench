"""Sample a stored point map, and send a variant box back to the original render."""

from collections.abc import Callable

import numpy as np

from llb.core.contracts.data_prep.book_preprocess import AffineTransform, GridTransform, Transform
from llb.prep.book.engines.protocol import Proposed
from llb.prep.book.preprocess.geometry import Affine, apply_affine

Locate = Callable[[float, float], tuple[float, float] | None]


def locate_affine(transform: Affine) -> Locate:
    """Original pixel to variant pixel, the inverse of the stored matrix."""
    forward = _inverse_rows(transform.matrix)

    def locate(x: float, y: float) -> tuple[float, float] | None:
        vx, vy = apply_affine(forward, x, y)
        if vx < -0.5 or vy < -0.5 or vx >= transform.width or vy >= transform.height:
            return None
        return vx, vy

    return locate


def locate_dense(map_x: np.ndarray, map_y: np.ndarray, src_width: int, src_height: int) -> Locate:
    """Original pixel to variant pixel, from a dense variant-to-original remap."""
    inverse_x, inverse_y = _scatter(map_x, map_y, src_width, src_height)

    def locate(x: float, y: float) -> tuple[float, float] | None:
        ix, iy = int(round(x)), int(round(y))
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                px, py = ix + dx, iy + dy
                if px < 0 or py < 0 or px >= src_width or py >= src_height:
                    continue
                vx, vy = int(inverse_x[py, px]), int(inverse_y[py, px])
                if vx >= 0:
                    return float(vx), float(vy)
        return None

    return locate


def map_box(
    transform: Transform, box: Proposed, src_width: int, src_height: int
) -> Proposed | None:
    """The axis-aligned box of `box`'s corners sent back to the original render."""
    corners = ((box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1))
    mapped = [_point(transform, x, y) for x, y in corners]
    xs = [point[0] for point in mapped]
    ys = [point[1] for point in mapped]
    x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
    x1, y1 = min(float(src_width), max(xs)), min(float(src_height), max(ys))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return Proposed(box.kind, box.order, x0, y0, x1, y1, box.text, box.confidence)


def _point(transform: Transform, x: float, y: float) -> tuple[float, float]:
    if isinstance(transform, AffineTransform):
        return apply_affine(_matrix(transform), x, y)
    return _bilinear(transform, x, y)


def _bilinear(transform: GridTransform, x: float, y: float) -> tuple[float, float]:
    row0, row1, ty = _bracket(transform.rows, y)
    col0, col1, tx = _bracket(transform.cols, x)
    return (
        _blend(transform.map_x, row0, row1, col0, col1, ty, tx),
        _blend(transform.map_y, row0, row1, col0, col1, ty, tx),
    )


def _blend(
    grid: list[list[float]], row0: int, row1: int, col0: int, col1: int, ty: float, tx: float
) -> float:
    top = grid[row0][col0] * (1.0 - tx) + grid[row0][col1] * tx
    bottom = grid[row1][col0] * (1.0 - tx) + grid[row1][col1] * tx
    return top * (1.0 - ty) + bottom * ty


def _bracket(coords: list[int], value: float) -> tuple[int, int, float]:
    if value <= coords[0]:
        return 0, 0, 0.0
    if value >= coords[-1]:
        last = len(coords) - 1
        return last, last, 0.0
    hi = 1
    while hi < len(coords) and coords[hi] < value:
        hi += 1
    lo = hi - 1
    span = coords[hi] - coords[lo]
    if span <= 0:
        return lo, hi, 0.0
    return lo, hi, (value - coords[lo]) / span


def _matrix(
    transform: AffineTransform,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    row0, row1 = transform.matrix
    return (tuple(row0), tuple(row1))  # type: ignore[return-value]


def _inverse_rows(
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    linear = np.asarray(matrix, dtype=np.float64)[:, :2]
    translation = np.asarray(matrix, dtype=np.float64)[:, 2]
    inverse = np.linalg.inv(linear)
    shifted = -inverse @ translation
    return (
        (float(inverse[0, 0]), float(inverse[0, 1]), float(shifted[0])),
        (float(inverse[1, 0]), float(inverse[1, 1]), float(shifted[1])),
    )


def _scatter(
    map_x: np.ndarray, map_y: np.ndarray, src_width: int, src_height: int
) -> tuple[np.ndarray, np.ndarray]:
    inverse_x = np.full((src_height, src_width), -1, dtype=np.int32)
    inverse_y = np.full((src_height, src_width), -1, dtype=np.int32)
    height, width = map_x.shape
    yy, xx = np.mgrid[0:height, 0:width]
    ox = np.rint(map_x).astype(np.int32)
    oy = np.rint(map_y).astype(np.int32)
    valid = (ox >= 0) & (ox < src_width) & (oy >= 0) & (oy < src_height)
    inverse_x[oy[valid], ox[valid]] = xx[valid]
    inverse_y[oy[valid], ox[valid]] = yy[valid]
    return inverse_x, inverse_y
