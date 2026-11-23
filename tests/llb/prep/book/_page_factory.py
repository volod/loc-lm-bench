"""Synthetic grey pages with one planted scan defect each.

The glyphs are darker on the bottom half, so orientation can tell upright from upside down.
Solid bars cannot. Nothing here is a page of a real book.
"""

import cv2
import numpy as np

BACKGROUND = 230
INK = 32
UPPER = 90
WIDTH = 420
HEIGHT = 540


def clean_page(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Straight, even, sharp text on a flat background."""
    page = np.full((height, width), BACKGROUND, dtype=np.uint8)
    paint_glyphs(page, 40, width - 40)
    return page


def paint_glyphs(page: np.ndarray, x0: int, x1: int) -> None:
    """Bottom-heavy text lines from `x0` to `x1`.

    A line, not a row of blocks: the horizontal ink projection then dominates the vertical one,
    which is what makes upright distinguishable from sideways.
    """
    height, _width = page.shape
    # Twelve pixels of ink would be most of a 61-wide median window, so the background
    # estimate would follow the text. A line of 8 in a gap of 28 leaves the paper in the majority.
    for top in range(40, height - 40, 36):
        page[top : top + 4, x0:x1] = UPPER
        page[top + 4 : top + 8, x0:x1] = INK


def tilt(page: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the page counter-clockwise by `angle` degrees, on the same canvas."""
    height, width = page.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, 1.0)
    warped = cv2.warpAffine(
        page, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=BACKGROUND
    )
    return np.asarray(warped, dtype=np.uint8)


def bow(page: np.ndarray, amplitude: float) -> np.ndarray:
    """Bend text lines down in the middle by `amplitude` pixels. The page edges stay put."""
    height, width = page.shape[:2]
    xs = np.arange(width, dtype=np.float32)
    shift = amplitude * 4.0 * xs * (width - 1 - xs) / float(width - 1) ** 2
    map_x = np.broadcast_to(xs, (height, width)).copy()
    map_y = (np.arange(height, dtype=np.float32)[:, None] - shift[None, :]).astype(np.float32)
    warped = cv2.remap(
        page, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=BACKGROUND
    )
    return np.asarray(warped, dtype=np.uint8)


def darken_gutter(page: np.ndarray, columns: int = 40) -> np.ndarray:
    """Shade the left edge the way a binding shadow does."""
    out = page.copy()
    out[:, :columns] = (out[:, :columns].astype(np.float32) * 0.55).astype(np.uint8)
    return out


def blur(page: np.ndarray) -> np.ndarray:
    return np.asarray(cv2.GaussianBlur(page, (0, 0), 2.5), dtype=np.uint8)


def add_noise(page: np.ndarray, std: float = 12.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    noisy = page.astype(np.float32) + rng.normal(0.0, std, page.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def fade(page: np.ndarray) -> np.ndarray:
    """Compress the grey range so ink and paper nearly meet."""
    scaled = 190.0 + (page.astype(np.float32) - 128.0) * 0.12
    return np.clip(scaled, 0, 255).astype(np.uint8)


def uneven(page: np.ndarray) -> np.ndarray:
    """A smooth left-to-right falloff. It is not a sharp edge band."""
    _height, width = page.shape
    ramp = np.linspace(0.62, 1.0, width, dtype=np.float32)
    shaded = page.astype(np.float32) * ramp[None, :]
    return np.clip(shaded, 0, 255).astype(np.uint8)


def bleed_page() -> np.ndarray:
    """Body text, a gap, and a neighbour-page strip in the outer margin."""
    page = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)
    paint_glyphs(page, 150, WIDTH - 40)
    paint_glyphs(page, 8, 40)
    return page


def with_marks(page: np.ndarray) -> np.ndarray:
    """A decimal point, a minus sign, and rules on both page edges."""
    out = page.copy()
    out[90:92, 200:202] = 0
    out[130, 180:200] = 0
    out[2:4, :] = 0
    out[-4:-2, :] = 0
    return out


def with_blob(page: np.ndarray, x: int, y: int, width: int = 40, height: int = 24) -> np.ndarray:
    """A black rectangle darker than any glyph, so a finder can name it twice."""
    out = page.copy()
    out[y : y + height, x : x + width] = 0
    return out


def upside_down(page: np.ndarray) -> np.ndarray:
    return np.asarray(cv2.rotate(page, cv2.ROTATE_180), dtype=np.uint8)


def blob_box(page: np.ndarray) -> tuple[int, int, int, int]:
    """The box of the planted black rectangle.

    Cubic resampling rings text edges down to black, so the darkest pixels are not all the blob.
    The blob is the one dark component that is thick in both directions.
    """
    ink = (page < 8).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    best: tuple[int, int, int, int, int] | None = None
    for index in range(1, count):
        x, y, width, height, area = (int(stats[index, column]) for column in range(5))
        if width < 8 or height < 8:
            continue
        if best is None or area > best[0]:
            best = (area, x, y, x + width, y + height)
    if best is None:
        raise ValueError("no planted blob")
    return best[1], best[2], best[3], best[4]
