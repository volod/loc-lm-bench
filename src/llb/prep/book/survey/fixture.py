"""The smoke page a survey runs every candidate on: synthetic with a transcription, or real.

The synthetic page (`page_content` draws it) is degraded the way the target scan is -- skewed,
curved toward the gutter, shadowed, on grey paper, blurred, noisy -- and saved as a 150 dpi
greyscale JPEG inside an image-only PDF. The degradation is numpy with a fixed seed, so the page is
byte-identical on every host and no page of a real book is needed. A page of a real scan is
rendered at its native resolution and has no transcription; it stays under the run directory.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llb.prep.book.survey.page_content import (
    PAGE_H_PT,
    PAGE_W_PT,
    FixtureTruth,
    TruthBlock,
    fixture_truth,
    render_clean,
)

SEED = 20261119
JPEG_QUALITY = 80

# Degradation strengths, chosen to resemble the target scan: lines bend toward the gutter by about
# two text heights, the gutter side is shadowed, the page sits on grey paper, and the whole page
# is slightly rotated and soft.
SKEW_DEG = 1.2
CURL_PX = 14.0
CURL_ZONE = 0.35  # fraction of the width, from the gutter edge, over which lines bend
PAPER_LEVEL = 0.90
SHADOW_DEPTH = 0.45
SHADOW_WIDTH = 0.07  # fraction of the width
BLUR_SIGMA_PX = 0.7
NOISE_SIGMA = 6.0

PAGE_IMAGE_NAME = "page.jpg"
PAGE_PDF_NAME = "page.pdf"
TRUTH_NAME = "truth.json"


@dataclass(frozen=True)
class SmokePage:
    """Where a materialized smoke page lives; `truth` is None for a page of a real scan."""

    image: Path
    pdf: Path
    truth: FixtureTruth | None
    label: str


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample `img` at fractional coordinates, white outside the page."""
    h, w = img.shape
    padded = np.pad(img, 1, constant_values=1.0)
    xs, ys = np.clip(xs + 1, 0, w), np.clip(ys + 1, 0, h)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    top = padded[y0, x0] * (1 - fx) + padded[y0, x0 + 1] * fx
    bottom = padded[y0 + 1, x0] * (1 - fx) + padded[y0 + 1, x0 + 1] * fx
    out: np.ndarray = top * (1 - fy) + bottom * fy
    return out


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(3 * sigma))
    taps = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    taps /= taps.sum()
    rows = np.apply_along_axis(lambda r: np.convolve(r, taps, mode="same"), 1, img)
    out: np.ndarray = np.apply_along_axis(lambda c: np.convolve(c, taps, mode="same"), 0, rows)
    return out


def degrade(clean: np.ndarray) -> np.ndarray:
    """Skew, gutter curl, gutter shadow, grey paper, blur, and noise; ink stays dark."""
    h, w = clean.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    angle = np.deg2rad(SKEW_DEG)
    cx, cy = w / 2, h / 2
    sx = cx + (xs - cx) * np.cos(angle) - (ys - cy) * np.sin(angle)
    sy = cy + (xs - cx) * np.sin(angle) + (ys - cy) * np.cos(angle)
    zone = CURL_ZONE * w
    sy = sy - CURL_PX * np.clip((zone - sx) / zone, 0, 1) ** 2
    warped = _bilinear(clean / 255.0, sx, sy)
    shadow = 1 - SHADOW_DEPTH * np.exp(-xs / (SHADOW_WIDTH * w))
    light = PAPER_LEVEL * shadow * (0.94 + 0.06 * ys / h)
    page = _gaussian_blur(warped * light, BLUR_SIGMA_PX) * 255.0
    page += np.random.default_rng(SEED).normal(0, NOISE_SIGMA, page.shape)
    out: np.ndarray = np.clip(page, 0, 255).astype(np.uint8)
    return out


def _grey_jpeg(pixels: np.ndarray) -> bytes:
    import fitz

    h, w = pixels.shape
    pix = fitz.Pixmap(fitz.csGRAY, w, h, pixels.tobytes(), False)
    data: bytes = pix.tobytes("jpg", jpg_quality=JPEG_QUALITY)
    return data


def _image_pdf(jpeg: bytes, path: Path) -> None:
    """One page placing the scan at its native resolution, the way the target book is built."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=PAGE_W_PT, height=PAGE_H_PT)
    page.insert_image(page.rect, stream=jpeg)
    doc.save(path, garbage=3, deflate=True, no_new_id=True)
    doc.close()


def write_fixture(out_dir: Path) -> SmokePage:
    """Write the degraded page image, its image-only PDF, and its transcription to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jpeg = _grey_jpeg(degrade(render_clean().astype(float)))
    image, pdf = out_dir / PAGE_IMAGE_NAME, out_dir / PAGE_PDF_NAME
    image.write_bytes(jpeg)
    _image_pdf(jpeg, pdf)
    truth = fixture_truth()
    payload: dict[str, Any] = asdict(truth)
    (out_dir / TRUTH_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return SmokePage(image=image, pdf=pdf, truth=truth, label="synthetic")


def read_truth(path: Path) -> FixtureTruth:
    """A transcription `write_fixture` wrote, read back."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FixtureTruth(
        blocks=[TruthBlock(**block) for block in payload["blocks"]],
        dangerous_tokens=list(payload["dangerous_tokens"]),
    )


def write_real_page(source_pdf: Path, pdf_page: int, out_dir: Path) -> SmokePage:
    """One page of a real scan, rendered at its native resolution; it has no transcription.

    The output stays under the run directory and never enters git: a real book is copyrighted.
    """
    import fitz

    from llb.prep.book.inventory import (
        open_pdf,
        inventory_page,
        render_dpi,
        render_page_png,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = open_pdf(source_pdf)
    try:
        if not 1 <= pdf_page <= doc.page_count:
            raise ValueError(f"{source_pdf} has {doc.page_count} pages; page {pdf_page} is out")
        dpi = render_dpi(inventory_page(doc, pdf_page), "native")
        png = render_page_png(doc, pdf_page, dpi, True)
        single = fitz.open()
        single.insert_pdf(doc, from_page=pdf_page - 1, to_page=pdf_page - 1)
        single.save(out_dir / PAGE_PDF_NAME, garbage=3, deflate=True, no_new_id=True)
        single.close()
    finally:
        doc.close()
    image = out_dir / "page.png"
    image.write_bytes(png)
    return SmokePage(image=image, pdf=out_dir / PAGE_PDF_NAME, truth=None, label=f"pdf-p{pdf_page}")
