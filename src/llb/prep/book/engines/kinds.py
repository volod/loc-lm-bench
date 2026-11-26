"""Block-kind names and the geometry an engine's boxes are brought into.

Figure text is dropped here, whatever the engine wrote for the picture: the spec keeps the image
region and forbids a generated description. A label the ledger does not know becomes `unknown`
rather than body, so an unrecognised class stays visible.
"""

import re

from llb.core.contracts.data_prep.book_blocks import BLOCK_KINDS
from llb.prep.book.engines.protocol import Proposed

_PAGE_NUMBER = re.compile(r"^\d{1,4}$")
_INVENTED = re.compile(
    r"^(technical illustration|the figure shows|this figure shows)\b", re.IGNORECASE
)
_ALIASES = {
    "text": "body",
    "paragraph": "body",
    "list": "body",
    "list-item": "body",
    "code": "body",
    "document-index": "body",
    "title": "heading",
    "section-header": "heading",
    "heading": "heading",
    "formula": "formula",
    "equation": "formula",
    "display-formula": "formula",
    "table": "table",
    "picture": "figure",
    "figure": "figure",
    "image": "figure",
    "chart": "figure",
    "caption": "caption",
    "footnote": "footnote",
    "page-header": "running-header",
    "running-header": "running-header",
    "header": "running-header",
    "page-number": "page-number",
    "page-footer": "page-number",
    "footer": "page-number",
    "ornament": "ornament",
    "neighbour-page-bleed": "neighbour-page-bleed",
    "unknown": "unknown",
    "aside": "unknown",
    "sidebar": "unknown",
    "text-box": "unknown",
    "boxed": "unknown",
    "boxed-text": "unknown",
}
# A box whose largest coordinate is at most this is treated as 0..1 normalized, not pixels.
_NORMALIZED_MAX = 1.5
_MILLI_MAX = 1000.0


def clean_kind_text(raw: str, text: str) -> tuple[str, str]:
    """Map an engine label onto a ledger kind, and drop text that belongs to a figure."""
    kind = _kind(raw, text)
    if kind == "figure" or _INVENTED.match(text.strip()):
        return "figure", ""
    return kind, text


def _kind(raw: str, text: str) -> str:
    key = raw.strip().lower().replace("_", "-").replace(" ", "-")
    if key in {"page-footer", "footer"} and not _PAGE_NUMBER.match(text.strip()):
        return "running-header"
    if key in BLOCK_KINDS:
        return key
    return _ALIASES.get(key, "unknown")


def to_pixels(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Pixel box for one proposal. Prefer `fit_proposals`, which decides the space for the page."""
    return _scale((x0, y0, x1, y1), _space_of_one(x0, y0, x1, y1, width, height), width, height)


def fit_proposals(
    proposals: list[Proposed], sent_width: int, sent_height: int, width: int, height: int
) -> list[Proposed]:
    """Kind cleanup, one coordinate space for the whole page, then back onto the rendered page.

    The space is decided from every box together. A line whose corners all sit inside the top
    thousand pixels of a tall page is still pixels when any other box on the page extends past
    that, which is what a detector returns. A page whose every box is in 0..1000 on a larger image
    is the thousandth-of-a-page space some document models use.
    """
    space = _space(proposals, sent_width, sent_height)
    converted = [_to_sent_pixels(item, space, sent_width, sent_height) for item in proposals]
    fitted = []
    for item in rescale(converted, sent_width, sent_height, width, height):
        clamped = _clamp(item, width, height)
        if clamped is not None:
            fitted.append(clamped)
    return fitted


def _space(proposals: list[Proposed], width: int, height: int) -> str:
    if not proposals:
        return "pixel"
    peak = max(max(item.x0, item.y0, item.x1, item.y1) for item in proposals)
    return _space_of_peak(peak, width, height)


def _space_of_one(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> str:
    return _space_of_peak(max(x0, y0, x1, y1), width, height)


def _space_of_peak(peak: float, width: int, height: int) -> str:
    if peak <= _NORMALIZED_MAX:
        return "unit"
    if peak <= _MILLI_MAX and max(width, height) > _MILLI_MAX:
        return "milli"
    return "pixel"


def _to_sent_pixels(proposed: Proposed, space: str, width: int, height: int) -> Proposed:
    kind, text = clean_kind_text(proposed.kind, proposed.text)
    x0, y0, x1, y1 = _scale(
        (proposed.x0, proposed.y0, proposed.x1, proposed.y1), space, width, height
    )
    return Proposed(kind, proposed.order, x0, y0, x1, y1, text, proposed.confidence)


def _scale(
    box: tuple[float, float, float, float], space: str, width: int, height: int
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    if space == "unit":
        return (x0 * width, y0 * height, x1 * width, y1 * height)
    if space == "milli":
        return (
            x0 / _MILLI_MAX * width,
            y0 / _MILLI_MAX * height,
            x1 / _MILLI_MAX * width,
            y1 / _MILLI_MAX * height,
        )
    return box


def _clamp(proposed: Proposed, width: int, height: int) -> Proposed | None:
    x0, y0 = max(0.0, proposed.x0), max(0.0, proposed.y0)
    x1, y1 = min(float(width), proposed.x1), min(float(height), proposed.y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return Proposed(
        proposed.kind, proposed.order, x0, y0, x1, y1, proposed.text, proposed.confidence
    )


def rescale(
    proposals: list[Proposed], sent_width: int, sent_height: int, width: int, height: int
) -> list[Proposed]:
    """Map boxes from the image the model saw back onto the rendered page."""
    if (sent_width, sent_height) == (width, height) or sent_width < 1 or sent_height < 1:
        return proposals
    scale_x, scale_y = width / sent_width, height / sent_height
    return [
        Proposed(
            item.kind,
            item.order,
            item.x0 * scale_x,
            item.y0 * scale_y,
            item.x1 * scale_x,
            item.y1 * scale_y,
            item.text,
            item.confidence,
        )
        for item in proposals
    ]
