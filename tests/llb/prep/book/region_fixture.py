"""Six planted pages: two neighbours, an empty response, a missed column, a blank, a failure."""

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from llb.cli import app
from llb.core.contracts.data_prep.book import BookStageSpec
from llb.prep.book.classes.content import ink_mask
from llb.prep.book.classes.layout import column_spans
from llb.prep.book.config import RunRequest
from llb.prep.book.engines.protocol import EngineRead, EngineRequest, Proposed
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.preprocess.images import read_grey
from llb.prep.book.runner import start_run
from llb.prep.book.stages import StageRegistry
from llb.prep.book.stages.recognize import recognition_stages
from llb.prep.book.stages.render import RenderStage
from llb.prep.book.stages.resolve import ResolveStage

_WIDTH = 640
_HEIGHT = 800
_PAPER = 230
_INK = 30
BEFORE = "p0001"
EMPTY = "p0002"
COLUMNS = "p0003"
BLANK = "p0004"
FAILED = "p0005"
AFTER = "p0006"
TEXT = {
    BEFORE: "Neighbour before stays.",
    COLUMNS: "Left column stays.",
    AFTER: "Neighbour after stays.",
}
RIGHT = "Right column was read."
_PRIMARY = {"primary": {"stage": "ocr", "engine": "scripted", "variant": "original"}}
_RUNNER = CliRunner()


class Scripted:
    """Returns the columns it was told to read, or fails one page outright."""

    engine_id = "scripted"
    stages = ("ocr",)

    def available(self) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, request: EngineRequest, stage: str) -> EngineRead:
        page = request.image.parent.parent.name
        if page == FAILED:
            raise RuntimeError("terminal engine failure")
        if stage != "ocr" or page in {EMPTY, BLANK}:
            return EngineRead(self.engine_id, None, request.variant, stage, ())
        return EngineRead(self.engine_id, None, request.variant, stage, _blocks(request, page))


def recover(pdf: Path, out: Path) -> Path:
    """Run the six pages, then build structure and the review queue."""
    stages = (
        BookStageSpec(name="render"),
        BookStageSpec(name="ocr", params={"engine": "scripted"}),
        BookStageSpec(name="blocks"),
        BookStageSpec(name="resolve", params=_PRIMARY),
    )
    registry = StageRegistry(
        (RenderStage(), *recognition_stages(AdapterRegistry((Scripted(),))), ResolveStage())
    )
    outcome = start_run(
        RunRequest(source=pdf, pages="1-6", max_attempts=1, stages=stages), out, registry
    )
    built = command("book-ocr", "structure", str(outcome.run_dir))
    assert built.exit_code == 0, built.output
    queued = command("book-ocr", "review-queue", str(outcome.run_dir))
    assert queued.exit_code == 0, queued.output
    return outcome.run_dir


def command(*args: str):
    return _RUNNER.invoke(app, list(args))


def planted_pdf(path: Path) -> None:
    write_pdf(path, [single(), single(), columns(), blank(), single(), single()])


def write_pdf(path: Path, pages: list[np.ndarray]) -> None:
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as document:
        for image in pages:
            height, width = image.shape
            page = document.new_page(width=width, height=height)
            pixmap = fitz.Pixmap(fitz.csGRAY, width, height, image.tobytes(), False)
            page.insert_image(page.rect, pixmap=pixmap)
        document.save(path, garbage=3, deflate=True, no_new_id=True)


def single() -> np.ndarray:
    page = np.full((_HEIGHT, _WIDTH), _PAPER, dtype=np.uint8)
    _lines(page, 80, 560)
    return page


def columns() -> np.ndarray:
    page = np.full((_HEIGHT, _WIDTH), _PAPER, dtype=np.uint8)
    _lines(page, 40, 160)
    _lines(page, 400, 560)
    return page


def blank() -> np.ndarray:
    return np.full((_HEIGHT, _WIDTH), _PAPER, dtype=np.uint8)


def _blocks(request: EngineRequest, page: str) -> tuple[Proposed, ...]:
    ink = ink_mask(read_grey(request.image))
    _count, spans = column_spans(ink)
    if page == COLUMNS and len(spans) > 1:
        spans = spans[:1]
    blocks = []
    for index, (left, right) in enumerate(spans):
        box = _ink_box(ink, left, right)
        text = TEXT.get(page, "")
        if box is None or not text:
            continue
        blocks.append(Proposed("body", index, box[0], box[1], box[2], box[3], text, 0.99))
    return tuple(blocks)


def _lines(page: np.ndarray, x0: int, x1: int) -> None:
    for top in range(80, page.shape[0] - 40, 36):
        page[top : top + 8, x0:x1] = _INK


def _ink_box(ink: np.ndarray, left: int, right: int) -> tuple[float, float, float, float] | None:
    patch = ink[:, left:right]
    ys, xs = np.nonzero(patch)
    if xs.size == 0:
        return None
    return (
        float(left + int(xs.min())),
        float(int(ys.min())),
        float(left + int(xs.max()) + 1),
        float(int(ys.max()) + 1),
    )
