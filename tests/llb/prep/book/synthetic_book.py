"""Fake engines over the generated scan, and the run that reads it.

Two engines disagree on one digit. The reader can refuse one page, and it can
die once mid-run. Correction stays off: the open value is the disagreement.
"""

from collections.abc import Sequence
from pathlib import Path

from llb.core.contracts.data_prep.book import BookStageSpec
from llb.prep.book.config import RunRequest
from llb.prep.book.engines.protocol import EngineRead, EngineRequest, Proposed
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.runner import start_run
from llb.prep.book.stages import StageRegistry
from llb.prep.book.stages.classes import ClassesStage
from llb.prep.book.stages.preprocess import SignalsStage, VariantsStage
from llb.prep.book.stages.recognize import recognition_stages
from llb.prep.book.stages.render import RenderStage
from llb.prep.book.stages.resolve import ResolveStage
from tests.llb.prep.book._page_factory import HEIGHT, WIDTH
from tests.llb.prep.book._stages import SimulatedKill
from tests.llb.prep.book.synthetic_pages import (
    CHAPTER,
    COLUMNS,
    CONTENTS,
    DISPUTE,
    DISPUTE_ALT,
    DISPUTE_TEXT,
    FAILED_PAGE,
    FIGURE_CAPTION,
    FIGURE_PAGE,
    FORMULA_PAGE,
    FORMULA_TEXT,
    GAP,
    HEADER,
    INDEX_PAGE,
    KILL_PAGE,
    LEFT_COLUMN,
    PAGE_COUNT,
    Plant,
    RIGHT_COLUMN,
    TABLE_CAPTION,
    TABLE_PAGE,
    UPSIDE_PAGE,
    _pixels,
    _plants,
)

PRIMARY = {"primary": {"stage": "ocr", "engine": "reader", "variant": "original"}}

__all__ = [
    "CHAPTER",
    "COLUMNS",
    "CONTENTS",
    "DISPUTE",
    "DISPUTE_ALT",
    "DISPUTE_TEXT",
    "FAILED_PAGE",
    "FIGURE_CAPTION",
    "FIGURE_PAGE",
    "FORMULA_PAGE",
    "FORMULA_TEXT",
    "GAP",
    "HEADER",
    "INDEX_PAGE",
    "KILL_PAGE",
    "LEFT_COLUMN",
    "PAGE_COUNT",
    "PRIMARY",
    "RIGHT_COLUMN",
    "TABLE_CAPTION",
    "TABLE_PAGE",
    "UPSIDE_PAGE",
    "begin",
    "make_registry",
    "stage_specs",
    "write_scan",
]


def write_scan(path: Path) -> Path:
    """Write the eleven-page scan. One page's pixels are rotated 180 degrees."""
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as document:
        for number in range(1, PAGE_COUNT + 1):
            image = _pixels(number)
            page = document.new_page(width=WIDTH, height=HEIGHT)
            pixmap = fitz.Pixmap(fitz.csGRAY, WIDTH, HEIGHT, image.tobytes(), False)
            page.insert_image(page.rect, pixmap=pixmap)
        document.save(path, garbage=3, deflate=True, no_new_id=True)
    return path


def stage_specs() -> tuple[BookStageSpec, ...]:
    """Render through resolve. Correction stays off: the disagreement is the engines."""
    return (
        BookStageSpec(name="render"),
        BookStageSpec(name="signals"),
        BookStageSpec(name="variants"),
        BookStageSpec(name="classes"),
        BookStageSpec(name="layout", params={"engine": "layout"}),
        BookStageSpec(name="ocr", params={"engine": "reader"}),
        BookStageSpec(name="formula", params={"engine": "formula"}),
        BookStageSpec(name="table", params={"engine": "table"}),
        BookStageSpec(name="blocks"),
        BookStageSpec(name="resolve", params=PRIMARY),
    )


def make_registry(*, kill: bool) -> StageRegistry:
    """Four fake engines. `kill` dies once, mid-run, on the formula page's OCR."""
    reader = _Engine(
        "reader",
        "ocr",
        _plants(),
        fail_page=FAILED_PAGE,
        kill_page=KILL_PAGE if kill else 0,
    )
    engines = (
        _Engine("layout", "layout", _plants()),
        reader,
        _Engine("formula", "formula", _plants()),
        _Engine("table", "table", _plants()),
    )
    return StageRegistry(
        (
            RenderStage(),
            SignalsStage(),
            VariantsStage(),
            ClassesStage(),
            *recognition_stages(AdapterRegistry(engines)),
            ResolveStage(),
        )
    )


def begin(scan: Path, out: Path, registry: StageRegistry) -> Path:
    """Start the run. A planted kill returns the partial directory instead of finishing."""
    request = RunRequest(source=scan, pages=f"1-{PAGE_COUNT}", max_attempts=1, stages=stage_specs())
    try:
        return start_run(request, out, registry).run_dir
    except SimulatedKill:
        runs = [path for path in out.iterdir() if path.is_dir()]
        if len(runs) != 1:
            raise
        return runs[0]


class _Engine:
    """One stage of the planted book. OCR can fail one page or die once."""

    def __init__(
        self,
        engine_id: str,
        stage: str,
        planted: Sequence[Plant],
        *,
        fail_page: int = 0,
        kill_page: int = 0,
    ) -> None:
        self.engine_id = engine_id
        self.stages = (stage,)
        self._stage = stage
        self._planted = tuple(planted)
        self._fail_page = fail_page
        self._kill_page = kill_page

    def available(self) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, request: EngineRequest, stage: str) -> EngineRead:
        page = int(request.image.parent.parent.name[1:])
        if stage == self._stage and page == self._kill_page:
            self._kill_page = 0
            raise SimulatedKill()
        if stage == self._stage and page == self._fail_page:
            raise RuntimeError(f"recognition failed on PDF page {page}")
        found = [
            _scaled(request, item)
            for item in self._planted
            if item.page == page and item.engine == self.engine_id and stage == self._stage
        ]
        return EngineRead(self.engine_id, "synthetic-1", request.variant, stage, tuple(found))


def _scaled(request: EngineRequest, item: Plant) -> Proposed:
    x0, y0, x1, y1 = item.box
    order = round(y0 * 1000) * 1000 + round(x0 * 1000)
    return Proposed(
        item.kind,
        order,
        x0 * request.width,
        y0 * request.height,
        x1 * request.width,
        y1 * request.height,
        item.text,
        None,
        item.formula,
        item.table,
    )
