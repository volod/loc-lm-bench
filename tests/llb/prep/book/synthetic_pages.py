"""The eleven pages of the generated scan: pixels and the text the engines return.

Boxes are fractions of the page. A folio sits below the contents-density band, in
the footer, so a page number does not make the next page look like a contents
continuation. No page of a real book is stored here.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from llb.core.contracts.data_prep.book_blocks import FormulaReading, TableReading
from llb.prep.book.grid import formula_from_text, table_from_grid
from tests.llb.prep.book._page_factory import BACKGROUND, HEIGHT, WIDTH, upside_down

PAGE_COUNT = 11
FAILED_PAGE = 1
CONTENTS = 2
CHAPTER = 3
DISPUTE = 4
GAP = 5
COLUMNS = 6
FORMULA_PAGE = 7
TABLE_PAGE = 8
FIGURE_PAGE = 9
INDEX_PAGE = 10
UPSIDE_PAGE = 11
KILL_PAGE = FORMULA_PAGE
HEADER = "Mission Handbook"
DISPUTE_TEXT = "The delta is 3.2 kilometres."
DISPUTE_ALT = "The delta is 3.3 kilometres."
FORMULA_TEXT = "E = mc^2 (2-1)"
LEFT_COLUMN = "Left column states the constraint."
RIGHT_COLUMN = "Right column states the margin."
FIGURE_CAPTION = "Figure 2-1. Transfer geometry."
TABLE_CAPTION = "Table 2-1. Stage loads."
Box = tuple[float, float, float, float]
_INK = 30
_LIGHT = 90


@dataclass(frozen=True)
class Plant:
    """One engine proposal on one PDF page."""

    page: int
    kind: str
    box: Box
    text: str
    engine: str
    formula: FormulaReading | None = None
    table: TableReading | None = None


def _plants() -> tuple[Plant, ...]:
    rows = [["Item", "Mass"], ["Tank", "1200"], ["Valve", "35"]]
    table = table_from_grid(rows, header_rows=1)
    formula = formula_from_text(FORMULA_TEXT, display=True)
    if table is None or formula is None:
        raise RuntimeError("the synthetic formula or table did not parse")
    dispute = (0.08, 0.28, 0.78, 0.34)
    left = (0.06, 0.28, 0.42, 0.36)
    right = (0.56, 0.28, 0.92, 0.36)
    planted = [
        *_contents(),
        _line(CHAPTER, 0.10, "1 Mission Design", kind="heading"),
        _line(CHAPTER, 0.22, "The window opens on day one."),
        _line(DISPUTE, 0.10, "1.1 Orbit Window", kind="heading"),
        Plant(DISPUTE, "body", dispute, DISPUTE_TEXT, "reader"),
        Plant(DISPUTE, "body", dispute, DISPUTE_ALT, "layout"),
        _line(DISPUTE, 0.42, "See Equation 2-1 and Figure 2-1."),
        _line(DISPUTE, 0.50, "Figure 9-9 is absent."),
        _line(DISPUTE, 0.58, "The ignition load is documented in Table 2-1."),
        _line(GAP, 0.30, "A plate fills this opening."),
        Plant(COLUMNS, "body", left, LEFT_COLUMN, "reader"),
        Plant(COLUMNS, "body", left, LEFT_COLUMN, "layout"),
        Plant(COLUMNS, "body", right, RIGHT_COLUMN, "reader"),
        Plant(COLUMNS, "body", right, RIGHT_COLUMN, "layout"),
        _line(FORMULA_PAGE, 0.10, "2 Propulsion", kind="heading"),
        Plant(FORMULA_PAGE, "formula", (0.08, 0.24, 0.62, 0.34), FORMULA_TEXT, "formula", formula),
        _line(FORMULA_PAGE, 0.24, "(2-1)", x0=0.72, x1=0.90),
        _line(FORMULA_PAGE, 0.42, "Equation 2-1 sets the burn."),
        _line(TABLE_PAGE, 0.16, TABLE_CAPTION, kind="caption"),
        Plant(
            TABLE_PAGE, "table", (0.08, 0.28, 0.80, 0.58), _grid_text(rows), "table", table=table
        ),
        Plant(FIGURE_PAGE, "figure", (0.12, 0.20, 0.70, 0.50), "", "layout"),
        _line(FIGURE_PAGE, 0.56, FIGURE_CAPTION, kind="caption"),
        _line(FIGURE_PAGE, 0.68, "Figure 2-1 shows the path."),
        _line(INDEX_PAGE, 0.12, "Index", kind="heading"),
        _line(INDEX_PAGE, 0.26, "transfer, 11"),
        _line(INDEX_PAGE, 0.34, "propulsion, 14"),
        _line(UPSIDE_PAGE, 0.30, "The rotated page still carries its folio."),
        _folio_text(CONTENTS, "i"),
    ]
    for page, printed in _folios():
        planted.append(_folio(page, printed))
        planted.append(_line(page, 0.02, HEADER, x1=0.62, height=0.045))
    return tuple(planted)


def _contents() -> list[Plant]:
    lines = (
        (0.12, "Contents"),
        (0.22, "1 Mission Design .... 10"),
        (0.30, "1.1 Orbit Window .... 11"),
        (0.38, "2 Propulsion .... 14"),
        (0.46, "Index .... 17"),
    )
    return [_line(CONTENTS, top, text) for top, text in lines]


def _folios() -> tuple[tuple[int, int], ...]:
    """Printed folios. PDF page 5 has none: 11 is followed by 13, so 12 is the gap."""
    return (
        (CHAPTER, 10),
        (DISPUTE, 11),
        (COLUMNS, 13),
        (FORMULA_PAGE, 14),
        (TABLE_PAGE, 15),
        (FIGURE_PAGE, 16),
        (INDEX_PAGE, 17),
        (UPSIDE_PAGE, 18),
    )


def _line(
    page: int,
    top: float,
    text: str,
    *,
    kind: str = "body",
    x0: float = 0.08,
    x1: float = 0.78,
    height: float = 0.05,
) -> Plant:
    return Plant(page, kind, (x0, top, x1, top + height), text, "reader")


def _folio(page: int, printed: int) -> Plant:
    return _folio_text(page, str(printed))


def _folio_text(page: int, text: str) -> Plant:
    return Plant(page, "body", (0.84, 0.965, 0.96, 0.995), text, "reader")


def _grid_text(rows: Sequence[Sequence[str]]) -> str:
    return "\n".join(" | ".join(row) for row in rows)


def _pixels(number: int) -> np.ndarray:
    if number == UPSIDE_PAGE:
        return _upside()
    if number == COLUMNS:
        return _columns()
    page = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)
    if number == TABLE_PAGE:
        _rules(page)
    if number == FIGURE_PAGE:
        _blob(page)
    return page


def _upside() -> np.ndarray:
    """A bottom-heavy sentence band, rotated 180 so it lands inside the reader box."""
    page = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)
    # Inverse of the rendered sentence box under a half turn: (x, y) -> (1-x, 1-y).
    _band(page, 0.22, 0.65, 0.92, 0.70, top_dark=False)
    return upside_down(page)


def _columns() -> np.ndarray:
    """Ink inside the two reader boxes, so the columns are read and still two columns."""
    page = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)
    _band(page, 0.08, 0.30, 0.40, 0.34, top_dark=False)
    _band(page, 0.58, 0.30, 0.90, 0.34, top_dark=False)
    return page


def _band(page: np.ndarray, x0: float, y0: float, x1: float, y1: float, *, top_dark: bool) -> None:
    top, bottom = int(y0 * HEIGHT) + 2, int(y1 * HEIGHT) - 2
    left, right = int(x0 * WIDTH) + 2, int(x1 * WIDTH) - 2
    mid = (top + bottom) // 2
    dark, light = _INK, _LIGHT
    if not top_dark:
        dark, light = light, dark
    page[top:mid, left:right] = dark
    page[mid:bottom, left:right] = light


def _rules(page: np.ndarray) -> None:
    """A ruled grid inset in the table's box. The engine, not these lines, supplies the cells."""
    top, bottom = int(0.28 * HEIGHT) + 6, int(0.58 * HEIGHT) - 6
    left, right = int(0.08 * WIDTH) + 6, int(0.80 * WIDTH) - 6
    for y_pos in range(top, bottom, max(1, (bottom - top) // 4)):
        page[y_pos : y_pos + 2, left:right] = _INK
    for x_pos in (left, (left + right) // 2, right - 2):
        page[top:bottom, x_pos : x_pos + 2] = _INK


def _blob(page: np.ndarray) -> None:
    """Thin lines in the figure box. A solid block disappears into the local background.

    No text engine covers these lines, so the drawing stays an unread region. The gaps
    keep the paper in the median window, which is what makes the ink visible to coverage.
    """
    left, right = int(0.20 * WIDTH), int(0.60 * WIDTH)
    top, bottom = int(0.24 * HEIGHT), int(0.48 * HEIGHT)
    for y_pos in range(top, bottom, 36):
        page[y_pos : y_pos + 4, left:right] = _LIGHT
        page[y_pos + 4 : y_pos + 8, left:right] = _INK
