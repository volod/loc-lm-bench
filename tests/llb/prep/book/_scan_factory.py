"""Deterministic synthetic scanned PDFs for the book-ocr tests -- no page of a real book in git.

Every page is one raster image placed over the page, the way a scanner's PDF is built: a grey
"text block" pattern seeded by the page number, placed so its native resolution is
`SCAN_DPI`. One page is landscape, one carries an RGB image, one places a landscape scan rotated a
quarter turn onto a portrait page (as the target book does for 45 pages), and one also carries an
embedded text line, so the inventory has something to tell apart.
"""

from dataclasses import dataclass
from pathlib import Path

SCAN_DPI = 150
PAGE_W_PT = 288.0  # 4 in x 5 in, small enough to keep the fixture fast
PAGE_H_PT = 360.0
POINTS_PER_INCH = 72.0
BACKGROUND = 0xE0
INK = 0x20


@dataclass(frozen=True)
class ScanSpec:
    pages: int = 6
    landscape_page: int | None = 3
    rgb_page: int | None = 2
    text_page: int | None = 5
    rotated_page: int | None = 4


def _grey_samples(width: int, height: int, seed: int) -> bytes:
    """Horizontal "text lines" whose lengths depend on the page, on a uniform background."""
    rows = bytearray([BACKGROUND]) * (width * height)
    line_h, gap = 6, 10
    for index, top in enumerate(range(20, height - 20, line_h + gap)):
        length = (width - 40) * (50 + (seed * 37 + index * 11) % 50) // 100
        for y in range(top, min(top + line_h, height)):
            start = y * width + 20
            rows[start : start + length] = bytes([INK]) * length
    return bytes(rows)


def make_duplicate_scan(path: Path) -> Path:
    """Two pages with the same raster, so a cache keyed only on the image would alias them."""
    import fitz

    doc = fitz.open()
    w_px = round(PAGE_W_PT / POINTS_PER_INCH * SCAN_DPI)
    h_px = round(PAGE_H_PT / POINTS_PER_INCH * SCAN_DPI)
    samples = _grey_samples(w_px, h_px, 1)
    for _page in range(2):
        page = doc.new_page(width=PAGE_W_PT, height=PAGE_H_PT)
        pixmap = fitz.Pixmap(fitz.csGRAY, w_px, h_px, samples, False)
        page.insert_image(page.rect, pixmap=pixmap)
    doc.save(path, garbage=3, deflate=True, no_new_id=True)
    doc.close()
    return path


def make_scan(path: Path, spec: ScanSpec = ScanSpec()) -> Path:
    """Write the synthetic scan to `path` and return it."""
    import fitz

    doc = fitz.open()
    for number in range(1, spec.pages + 1):
        landscape = number == spec.landscape_page
        w_pt, h_pt = (PAGE_H_PT, PAGE_W_PT) if landscape else (PAGE_W_PT, PAGE_H_PT)
        page = doc.new_page(width=w_pt, height=h_pt)
        rotated = number == spec.rotated_page
        w_px = round((h_pt if rotated else w_pt) / POINTS_PER_INCH * SCAN_DPI)
        h_px = round((w_pt if rotated else h_pt) / POINTS_PER_INCH * SCAN_DPI)
        grey = fitz.Pixmap(fitz.csGRAY, w_px, h_px, _grey_samples(w_px, h_px, number), False)
        pix = fitz.Pixmap(fitz.csRGB, grey) if number == spec.rgb_page else grey
        page.insert_image(page.rect, pixmap=pix, rotate=90 if rotated else 0)
        if number == spec.text_page:
            page.insert_text((20, 20), "embedded caption", fontsize=8)
    doc.save(path, garbage=3, deflate=True, no_new_id=True)
    doc.close()
    return path
