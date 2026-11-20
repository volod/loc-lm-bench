"""Page inventory and native-resolution rendering, both straight from PyMuPDF.

The inventory is what the PDF itself says about a page before anything renders it: its size and
rotation, its orientation, how many images it places, the pixel size of the largest one, the
resolution that image was scanned at, and whether the page carries any embedded text. Rendering
then uses that native resolution, so a 150 dpi scan is rendered at 150 dpi -- neither blurred by
downsampling nor inflated by an upsample that adds pixels and no information.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from llb.core.contracts.data_prep.book import ImageRotation, PageInventoryRecord
from llb.prep.pdf.model import PdfDiagnostics, inspect_pdf

if TYPE_CHECKING:
    import fitz

POINTS_PER_INCH = 72.0
# A page that places no raster image (born-digital, vector-only) has no native resolution; render
# it at a resolution OCR engines are tuned for. A scan claiming more than the cap is clamped, so a
# mis-stated image transform cannot ask for a gigapixel render.
FALLBACK_RENDER_DPI = 300
MAX_RENDER_DPI = 600
GREY_COMPONENTS = 1
QUARTER_TURN = 90
FULL_TURN = 360


class PdfOpenError(ValueError):
    """The source cannot be opened, or is encrypted, so no page of it can be inventoried."""


@dataclass(frozen=True)
class DocumentInventory:
    """Document-level facts the PDF lane's diagnostics already collect, plus the page count."""

    page_count: int
    diagnostics: PdfDiagnostics


def open_pdf(path: Path) -> "fitz.Document":
    """Open `path` read-only through PyMuPDF, refusing an encrypted or unreadable document."""
    import fitz

    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise PdfOpenError(f"cannot open {path}: {type(exc).__name__}: {exc}") from exc
    if doc.needs_pass:
        doc.close()
        raise PdfOpenError(f"{path} is password-protected; decrypt it before recovery")
    if doc.page_count < 1:
        doc.close()
        raise PdfOpenError(f"{path} has no pages")
    return doc


def inventory_document(path: Path) -> DocumentInventory:
    """The PDF lane's cheap diagnostics, refusing what no page-level stage could read."""
    diagnostics = inspect_pdf(path)
    if diagnostics.error:
        raise PdfOpenError(diagnostics.error)
    if diagnostics.needs_password:
        raise PdfOpenError(f"{path} is password-protected; decrypt it before recovery")
    if not diagnostics.page_count:
        raise PdfOpenError(f"{path} has no pages")
    return DocumentInventory(page_count=diagnostics.page_count, diagnostics=diagnostics)


def _image_axis(info: dict[str, Any]) -> tuple[float, float]:
    """The page-space vector the image's pixel rows run along (`a`, `b` of its placement matrix)."""
    a, b = info["transform"][:2]
    return float(a), float(b)


def _image_dpi(info: dict[str, Any]) -> float | None:
    """Pixels per inch along the image's own rows.

    Measured along the placement matrix rather than the bounding box, because a scanner PDF often
    places a landscape scan rotated onto a portrait page: the target book places 45 pages that way,
    and dividing their 1,654-pixel rows by the 595-point box width reads 200 dpi for a 150 dpi scan.
    """
    length_in = math.hypot(*_image_axis(info)) / POINTS_PER_INCH
    if length_in <= 0 or not info.get("width"):
        return None
    return float(info["width"]) / length_in


def _image_rotation(info: dict[str, Any]) -> ImageRotation:
    """The angle the image is placed at, snapped to a quarter turn (0 = upright placement)."""
    a, b = _image_axis(info)
    degrees = math.degrees(math.atan2(b, a))
    return cast(ImageRotation, int(round(degrees / QUARTER_TURN) * QUARTER_TURN) % FULL_TURN)


def inventory_page(doc: "fitz.Document", pdf_page: int) -> PageInventoryRecord:
    """Inventory one 1-based PDF page."""
    page = doc[pdf_page - 1]
    images = [info for info in page.get_image_info() if info.get("width")]
    largest = max(images, key=lambda info: info["width"] * info["height"], default=None)
    dpi = _image_dpi(largest) if largest is not None else None
    rect = page.rect  # already accounts for /Rotate
    return PageInventoryRecord(
        width_pt=round(float(rect.width), 3),
        height_pt=round(float(rect.height), 3),
        rotation=int(page.rotation),
        orientation="landscape" if rect.width > rect.height else "portrait",
        image_count=len(images),
        image_width_px=int(largest["width"]) if largest is not None else None,
        image_height_px=int(largest["height"]) if largest is not None else None,
        native_dpi=round(dpi) if dpi is not None else None,
        image_rotation=_image_rotation(largest) if largest is not None else None,
        embedded_text_chars=len((page.get_text("text") or "").strip()),
    )


def render_dpi(inventory: PageInventoryRecord, requested: int | str) -> int:
    """The resolution a page renders at: its native one for `"native"`, else the requested one."""
    if requested != "native":
        return min(int(requested), MAX_RENDER_DPI)
    native = inventory.native_dpi or FALLBACK_RENDER_DPI
    return max(1, min(native, MAX_RENDER_DPI))


def page_is_grey(doc: "fitz.Document", pdf_page: int) -> bool:
    """True when every image the page places is single-channel, so a grey render loses nothing.

    The colorspace NAME cannot say this (an `ICCBased` profile may hold one channel or three), so
    the component count is read from the image stream itself.
    """
    xrefs = [entry[0] for entry in doc[pdf_page - 1].get_images(full=True)]
    return bool(xrefs) and all(
        (doc.extract_image(xref) or {}).get("colorspace") == GREY_COMPONENTS for xref in xrefs
    )


def render_page_png(doc: "fitz.Document", pdf_page: int, dpi: int, grey: bool) -> bytes:
    """PNG bytes of one page rendered at `dpi`, single-channel when `grey`."""
    import fitz

    page = doc[pdf_page - 1]
    colorspace = fitz.csGRAY if grey else fitz.csRGB
    pix = page.get_pixmap(dpi=dpi, colorspace=colorspace, alpha=False)
    return bytes(pix.tobytes("png"))
