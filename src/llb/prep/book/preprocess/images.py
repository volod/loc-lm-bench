"""Grey page images: load, save, and the ink mask the signals share.

PNG bytes are written with a fixed compression level and no ancillary chunks, so two runs of the
same correction produce the same file.
"""

from pathlib import Path

import cv2
import numpy as np

from llb.prep.book.preprocess.constants import BACKGROUND_KERNEL

_PNG_PARAMS = (cv2.IMWRITE_PNG_COMPRESSION, 3)


def read_grey(path: Path) -> np.ndarray:
    """The page as one uint8 channel. A colour render is converted; the file is not modified."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"unreadable image: {path}")
    return np.asarray(image, dtype=np.uint8)


def encode_png(grey: np.ndarray) -> bytes:
    """PNG bytes of `grey`. The same pixels always encode to the same bytes."""
    ok, encoded = cv2.imencode(".png", np.asarray(grey, dtype=np.uint8), list(_PNG_PARAMS))
    if not ok:
        raise RuntimeError("could not encode a page image")
    return encoded.tobytes()


def write_png(path: Path, grey: np.ndarray) -> None:
    """Write `grey` as a deterministic PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_png(grey))


def background_kernel(shape: tuple[int, ...]) -> int:
    """An odd kernel wider than a glyph and narrower than the page."""
    limit = int(min(shape[:2]))
    if limit % 2 == 0:
        limit -= 1
    size = min(BACKGROUND_KERNEL, max(3, limit))
    if size % 2 == 0:
        size -= 1
    return max(size, 3)


def ink_mask(grey: np.ndarray) -> np.ndarray:
    """Pixels darker than three quarters of a wide median background."""
    kernel = background_kernel(grey.shape)
    background = cv2.medianBlur(grey, kernel)
    return grey.astype(np.int16) * 4 < background.astype(np.int16) * 3


def text_bands(ink: np.ndarray, fraction: float) -> list[tuple[int, int]]:
    """`(y0, y1)` of each horizontal run of ink, top to bottom."""
    rows = ink.sum(axis=1)
    peak = float(rows.max()) if rows.size else 0.0
    if peak <= 0:
        return []
    hot = rows >= peak * fraction
    return _runs(hot)


def _runs(hot: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(hot.tolist()):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            _keep(bands, start, index)
            start = None
    if start is not None:
        _keep(bands, start, len(hot))
    return bands


def _keep(bands: list[tuple[int, int]], start: int, end: int) -> None:
    if end - start >= 2:
        bands.append((start, end))
