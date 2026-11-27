"""A four-page planted scan whose technical blocks two engines read differently.

`alpha` answers the formula stage and is the primary reading. `beta` answers the table stage;
`gamma` replaces it when a test rebuilds the ledger. Both stages are structural, so the two
engines' proposals cluster into one block per region and every block below carries the readings
the case needs. Boxes are fractions of the rendered page.

- `p0001` an index and a sign: a body line whose primary drops a minus and shortens `0.05`, a
  formula whose index digit, sign, and index letter the engines disagree on, and an agreed formula.
- `p0002` tables: a disputed digit cell, a merged header against a blank, a primary that left a
  cell blank and omitted a column, and two captions that disagree on a digit.
- `p0003` figures: a diagram with labels and no caption beside a sentence that cites it, a
  drawing between two competing unnumbered captions, and a caption whose number is unresolved.
- `p0004` engine fallback: a formula the primary engine did not read.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book import BookStageSpec
from llb.core.contracts.data_prep.book_blocks import FormulaReading, TableReading
from llb.prep.book.config import RunRequest
from llb.prep.book.engines.protocol import EngineRead, EngineRequest, Proposed
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.grid import formula_from_text, table_from_grid
from llb.prep.book.runner import start_run
from llb.prep.book.stages import StageRegistry
from llb.prep.book.stages.recognize import recognition_stages
from llb.prep.book.stages.render import RenderStage
from llb.prep.book.stages.resolve import ResolveStage

PRIMARY = {"primary": {"stage": "formula", "engine": "alpha", "variant": "original"}}
OPEN_FORMULA = r"v_1 - v_i = \Delta v (2-4)"
AGREED_FORMULA = r"E = mc^2 (2-5)"
FALLBACK_FORMULA = r"F = m a (4-1)"
BODY = "The burn lasts 0.5 s at x = -3."
CITING = "Figure 3-1 shows the transfer."
DISPUTED_CAPTION = "Figure 12-3. Orbit geometry."
_PAPER = 235
_WIDTH = 600
_HEIGHT = 800
Box = tuple[float, float, float, float]
Cell = str | dict[str, object]


@dataclass(frozen=True)
class Planted:
    """One proposal: page, kind, box, text, and the structure the engine attaches."""

    page: str
    kind: str
    box: Box
    text: str
    formula: FormulaReading | None = None
    table: TableReading | None = None


def _formula(page: str, box: Box, text: str) -> Planted:
    return Planted(page, "formula", box, text, formula=formula_from_text(text, display=True))


def _table(page: str, box: Box, rows: Sequence[Sequence[Cell]], caption: str | None = None):
    text = "\n".join(" | ".join(_cell_text(cell) for cell in row) for row in rows)
    return Planted(page, "table", box, text, table=table_from_grid(rows, caption))


def _cell_text(cell: Cell) -> str:
    return str(cell.get("text", "")) if isinstance(cell, dict) else cell


_HEADING = ("p0001", "heading", (0.1, 0.04, 0.9, 0.08), "2.1 Transfer burn")
_T2_TEXT = "Stage 1\nThrust | Isp\n450 | 310"
_SHARED = (
    Planted(*_HEADING),
    _formula("p0001", (0.1, 0.32, 0.9, 0.40), AGREED_FORMULA),
    Planted("p0001", "body", (0.1, 0.44, 0.9, 0.50), "The transfer follows Equation 2-5."),
    Planted("p0004", "body", (0.1, 0.40, 0.9, 0.45), "Fallback page."),
)
_T4_ROWS = [["a", "b"], ["1", "2"]]

ALPHA = (
    *_SHARED,
    Planted("p0001", "body", (0.1, 0.10, 0.9, 0.16), BODY),
    _formula("p0001", (0.1, 0.20, 0.9, 0.28), OPEN_FORMULA),
    _table(
        "p0002", (0.1, 0.05, 0.9, 0.22), [["Item", "Mass (kg)"], ["Tank", "1200"], ["Valve", "35"]]
    ),
    Planted(
        "p0002",
        "table",
        (0.1, 0.27, 0.9, 0.44),
        _T2_TEXT,
        table=table_from_grid(
            [[{"text": "Stage 1", "col_span": 2}], ["Thrust", "Isp"], ["450", "310"]]
        ),
    ),
    _table("p0002", (0.1, 0.49, 0.9, 0.66), [["Valve", "Mass (kg)"], ["V1", ""]]),
    _table("p0002", (0.1, 0.71, 0.9, 0.88), _T4_ROWS, "Table 2-1. Loads"),
    Planted("p0003", "figure", (0.05, 0.05, 0.45, 0.40), ""),
    Planted("p0003", "body", (0.08, 0.36, 0.12, 0.39), "0"),
    Planted("p0003", "body", (0.20, 0.36, 0.30, 0.39), "t (s)"),
    Planted("p0003", "body", (0.38, 0.36, 0.44, 0.39), "10"),
    Planted("p0003", "body", (0.05, 0.45, 0.45, 0.50), CITING),
    Planted("p0003", "figure", (0.55, 0.10, 0.95, 0.30), ""),
    Planted("p0003", "caption", (0.55, 0.06, 0.95, 0.09), "Burn sequence."),
    Planted("p0003", "caption", (0.55, 0.31, 0.95, 0.34), "Thrust profile."),
    Planted("p0003", "figure", (0.05, 0.60, 0.95, 0.85), ""),
    Planted("p0003", "caption", (0.05, 0.86, 0.95, 0.90), DISPUTED_CAPTION),
)

BETA = (
    *_SHARED,
    Planted("p0001", "body", (0.1, 0.10, 0.9, 0.16), "The burn lasts 0.05 s at x = 3."),
    _formula("p0001", (0.1, 0.20, 0.9, 0.28), r"v_l + v_j = \Delta v (2-4)"),
    _table(
        "p0002", (0.1, 0.05, 0.9, 0.22), [["Item", "Mass (kg)"], ["Tank", "1200"], ["Valve", "36"]]
    ),
    Planted(
        "p0002",
        "table",
        (0.1, 0.27, 0.9, 0.44),
        _T2_TEXT,
        table=table_from_grid([["Stage 1", ""], ["Thrust", "Isp"], ["450", "310"]]),
    ),
    _table("p0002", (0.1, 0.49, 0.9, 0.66), [["Valve", "Mass (kg)", "Count"], ["V1", "12", "4"]]),
    _table("p0002", (0.1, 0.71, 0.9, 0.88), _T4_ROWS, "Table 2-7. Loads"),
    Planted("p0003", "caption", (0.05, 0.86, 0.95, 0.90), "Figure 12-8. Orbit geometry."),
    _formula("p0004", (0.1, 0.20, 0.9, 0.30), FALLBACK_FORMULA),
)

# The rebuilt ledger: the formula's other reading now agrees on the digit and the sign.
GAMMA = tuple(
    _formula("p0001", item.box, r"v_1 - v_j = \Delta v (2-4)")
    if item.page == "p0001" and item.kind == "formula" and item.text.startswith("v_l")
    else item
    for item in BETA
)


class PlantedEngine:
    """Answers one stage with its planted proposals for the page being read."""

    def __init__(self, engine_id: str, stage: str, planted: Sequence[Planted]) -> None:
        self.engine_id = engine_id
        self.stages = ("formula", "table")
        self._stage = stage
        self._planted = tuple(planted)

    def available(self) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, request: EngineRequest, stage: str) -> EngineRead:
        page = request.image.parent.parent.name
        found: list[Proposed] = []
        if stage == self._stage:
            for item in self._planted:
                if item.page == page:
                    found.append(_scaled(request, _order(item.box), item))
        return EngineRead(self.engine_id, "planted-1", request.variant, stage, tuple(found))


def _order(box: Box) -> int:
    """Top to bottom, then left to right, so two engines number one region alike."""
    return round(box[1] * 1000) * 1000 + round(box[0] * 1000)


def _scaled(request: EngineRequest, order: int, item: Planted) -> Proposed:
    x0, y0, x1, y1 = item.box
    width, height = request.width, request.height
    return Proposed(
        item.kind,
        order,
        x0 * width,
        y0 * height,
        x1 * width,
        y1 * height,
        item.text,
        None,
        item.formula,
        item.table,
    )


def registry(other: str = "beta") -> StageRegistry:
    """Render, the two planted engines, and resolve. `other` names the table-stage engine."""
    second = GAMMA if other == "gamma" else BETA
    engines = (PlantedEngine("alpha", "formula", ALPHA), PlantedEngine(other, "table", second))
    return StageRegistry(
        (RenderStage(), *recognition_stages(AdapterRegistry(engines)), ResolveStage())
    )


def stages(other: str = "beta") -> tuple[BookStageSpec, ...]:
    """The configured run: both structural stages, the ledger, and resolve."""
    return (
        BookStageSpec(name="render"),
        BookStageSpec(name="formula", params={"engine": "alpha"}),
        BookStageSpec(name="table", params={"engine": other}),
        BookStageSpec(name="blocks"),
        BookStageSpec(name="resolve", params=PRIMARY),
    )


def recover(pdf: Path, out: Path) -> Path:
    """Run every page through resolve."""
    request = RunRequest(source=pdf, pages="1-4", batch_pages=4, stages=stages())
    return start_run(request, out, registry()).run_dir


def planted_pdf(path: Path) -> Path:
    """Four plain pages. The engines are told what is on them."""
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((_HEIGHT, _WIDTH), _PAPER, dtype=np.uint8)
    with fitz.open() as document:
        for _page in range(4):
            page = document.new_page(width=_WIDTH, height=_HEIGHT)
            pixmap = fitz.Pixmap(fitz.csGRAY, _WIDTH, _HEIGHT, image.tobytes(), False)
            page.insert_image(page.rect, pixmap=pixmap)
        document.save(path, garbage=3, deflate=True, no_new_id=True)
    return path
