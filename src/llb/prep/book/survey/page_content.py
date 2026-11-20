"""The synthetic smoke page's CONTENT: what is printed on it, where, and its transcription.

A running header, a numbered heading, two columns of body text with decimals, minus signs, Greek
letters, and cross-references, a display formula with subscripts and an equation number, a ruled
table, a figure with a caption, and a page number -- the block kinds the target handbook carries.
It is drawn with PyMuPDF's built-in Base-14 fonts (Times for Latin, Symbol for Greek and the true
minus sign), so the clean rendering is identical on every host; `fixture` degrades it into a scan.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import fitz

PAGE_W_PT = 432.0  # 6 x 8 in at 72 pt/in
PAGE_H_PT = 576.0
SCAN_DPI = 150
POINTS_PER_INCH = 72.0

TIMES = "tiro"
SYMBOL = "symb"
# Symbol encodes Greek and the mathematical minus on ASCII code points.
SYMBOL_CODES = {"α": "a", "β": "b", "Δ": "D", "μ": "m", "−": "-"}


@dataclass(frozen=True)
class TruthBlock:
    """One block of the reference transcription, in reading order."""

    kind: str
    text: str
    latex: str | None = None
    cells: list[list[str]] | None = None


@dataclass(frozen=True)
class FixtureTruth:
    blocks: list[TruthBlock]
    # Tokens whose misreading changes a value: a decimal point, a sign, a Greek letter, an
    # equation number. A smoke reading reports each as found, found with a lookalike, or missing.
    dangerous_tokens: list[str] = field(default_factory=list)

    def reading_text(self, kinds: tuple[str, ...] | None = None) -> str:
        """The transcription as one string in reading order, optionally restricted by kind."""
        parts: list[str] = []
        for block in self.blocks:
            if kinds is not None and block.kind not in kinds:
                continue
            if block.cells:
                parts.extend(" ".join(row) for row in block.cells)
            else:
                parts.append(block.text)
        return "\n".join(parts)


HEADER = "SPACECRAFT DESIGN NOTES"
HEADER_RIGHT = "3.2 Orbit Transfer"
HEADING = "3.2 Orbit Transfer Budget"
LEFT_COLUMN = (
    "The velocity increment for a single",
    "burn is 0.05 km/s when the thrust is",
    "−3.2 N and the mass ratio is 1.07.",
    "Table 3-1 lists the budget; see",
    "Eq. (3-4) and Fig. 3-2 for the trend.",
)
RIGHT_COLUMN = (
    "The angle α is 12.5 deg and β is",
    "0.8 deg at an altitude of 400 km.",
    "A margin of 5% covers the losses",
    "from gravity and steering, and the",
    "residual error is below 2 μm/s.",
)
FORMULA_TEXT = "Δv = Isp g0 ln(m0/mf)"
FORMULA_LATEX = r"\Delta v = I_{sp} g_0 \ln(m_0/m_f)"
EQUATION_NUMBER = "(3-4)"
TABLE_CAPTION = "Table 3-1. Transfer budget."
TABLE = (
    ("Maneuver", "Δv (m/s)", "Margin"),
    ("Insertion", "1,240.5", "5%"),
    ("Plane change", "0.05", "10%"),
    ("Deorbit", "−98.2", "2%"),
)
FIGURE_CAPTION = "Fig. 3-2. Altitude versus time."
PAGE_NUMBER = "57"
DANGEROUS = ("0.05", "−3.2", "1.07", "12.5", "α", "β", "Δv", "1,240.5", "−98.2", "(3-4)", "μm/s")


def fixture_truth() -> FixtureTruth:
    """The reference transcription of the synthetic page, in reading order."""
    return FixtureTruth(
        blocks=[
            TruthBlock("running_header", f"{HEADER} {HEADER_RIGHT}"),
            TruthBlock("heading", HEADING),
            TruthBlock("body", " ".join(LEFT_COLUMN)),
            TruthBlock("body", " ".join(RIGHT_COLUMN)),
            TruthBlock("formula", f"{FORMULA_TEXT} {EQUATION_NUMBER}", latex=FORMULA_LATEX),
            TruthBlock("caption", TABLE_CAPTION),
            TruthBlock("table", "", cells=[list(row) for row in TABLE]),
            TruthBlock("figure", ""),
            TruthBlock("caption", FIGURE_CAPTION),
            TruthBlock("page_number", PAGE_NUMBER),
        ],
        dangerous_tokens=list(DANGEROUS),
    )


def _runs(text: str) -> list[tuple[str, str]]:
    """Split `text` into (drawn text, font) runs, Greek and minus going to the Symbol font."""
    runs: list[tuple[str, str]] = []
    for char in text:
        font, drawn = (SYMBOL, SYMBOL_CODES[char]) if char in SYMBOL_CODES else (TIMES, char)
        if runs and runs[-1][1] == font:
            runs[-1] = (runs[-1][0] + drawn, font)
        else:
            runs.append((drawn, font))
    return runs


def _width(text: str, size: float) -> float:
    import fitz

    return float(sum(fitz.get_text_length(t, fontname=f, fontsize=size) for t, f in _runs(text)))


def _draw(page: "fitz.Page", x: float, y: float, text: str, size: float = 10.0) -> float:
    """Draw `text` at baseline (x, y); returns the x after it."""
    import fitz

    for drawn, font in _runs(text):
        page.insert_text((x, y), drawn, fontname=font, fontsize=size)
        x += fitz.get_text_length(drawn, fontname=font, fontsize=size)
    return x


def _draw_formula(page: "fitz.Page", y: float) -> None:
    """The display formula with real subscripts, centred, and its equation number at the right."""
    pieces: tuple[tuple[str, float], ...] = (
        ("Δv = I", 0.0),
        ("sp", 3.0),
        (" g", 0.0),
        ("0", 3.0),
        (" ln(m", 0.0),
        ("0", 3.0),
        ("/m", 0.0),
        ("f", 3.0),
        (")", 0.0),
    )
    x = PAGE_W_PT / 2 - 55
    for text, drop in pieces:
        x = _draw(page, x, y + drop, text, 8.0 if drop else 11.0)
    _draw(page, PAGE_W_PT - 60 - _width(EQUATION_NUMBER, 10.0), y, EQUATION_NUMBER)


def _draw_table(page: "fitz.Page", top: float) -> float:
    import fitz

    left, widths, row_h = 90.0, (110.0, 80.0, 60.0), 16.0
    right = left + sum(widths)
    for index, row in enumerate(TABLE):
        y = top + index * row_h
        x = left
        for cell, width in zip(row, widths):
            _draw(page, x + 4, y + 12, cell, 9.5)
            x += width
    bottom = top + len(TABLE) * row_h
    for y in (top, top + row_h, bottom):
        page.draw_line(fitz.Point(left, y), fitz.Point(right, y), width=0.8)
    return bottom


def _draw_figure(page: "fitz.Page", top: float) -> float:
    import fitz

    box = fitz.Rect(130, top, 300, top + 70)
    page.draw_line(box.bl, box.tl, width=0.8)
    page.draw_line(box.bl, box.br, width=0.8)
    points = [fitz.Point(box.x0 + i * 17, box.y1 - 6 - (i * 7) % 50) for i in range(11)]
    page.draw_polyline(points, width=0.8)
    return float(box.y1)


def _compose(doc: "fitz.Document") -> None:
    page = doc.new_page(width=PAGE_W_PT, height=PAGE_H_PT)
    _draw(page, 48, 40, HEADER, 8.0)
    _draw(page, PAGE_W_PT - 48 - _width(HEADER_RIGHT, 8.0), 40, HEADER_RIGHT, 8.0)
    _draw(page, 48, 78, HEADING, 13.0)
    for index, (left, right) in enumerate(zip(LEFT_COLUMN, RIGHT_COLUMN)):
        y = 104 + index * 14
        _draw(page, 48, y, left)
        _draw(page, 228, y, right)
    _draw_formula(page, 196)
    _draw(page, 90, 236, TABLE_CAPTION, 9.5)
    bottom = _draw_table(page, 244)
    figure_bottom = _draw_figure(page, bottom + 24)
    _draw(page, 130, figure_bottom + 18, FIGURE_CAPTION, 9.5)
    _draw(page, PAGE_W_PT / 2 - _width(PAGE_NUMBER, 9.0) / 2, PAGE_H_PT - 30, PAGE_NUMBER, 9.0)


def render_clean() -> np.ndarray:
    """The undegraded page as a greyscale array at the scan resolution."""
    import fitz

    doc = fitz.open()
    _compose(doc)
    pix = doc[0].get_pixmap(dpi=SCAN_DPI, colorspace=fitz.csGRAY, alpha=False)
    doc.close()
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
