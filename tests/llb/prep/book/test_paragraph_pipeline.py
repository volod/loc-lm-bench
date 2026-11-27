"""Render, project, and review a planted scan with two competing readings."""

import json
from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book import BookStageSpec
from llb.core.contracts.data_prep.book_classes import (
    PAGE_CLASS_IDS,
    PilotPageEntry,
    PilotSelection,
    PlacedSignals,
)
from llb.prep.book.config import RunRequest
from llb.prep.book.engines.protocol import EngineRead, EngineRequest, Proposed
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.ledger.io import ledger_path, read_ledger
from llb.prep.book.pilot_review.prepare import DraftInput, prepare
from llb.prep.book.preprocess.constants import BOX_TOLERANCE_PX
from llb.prep.book.preprocess.images import read_grey
from llb.prep.book.project.build import project_corpus
from llb.prep.book.ranges import page_id
from llb.prep.book.render.build import RenderOptions, build_vault
from llb.prep.book.runner import start_run
from llb.prep.book.source import sha256_file
from llb.prep.book.stages import StageRegistry
from llb.prep.book.stages.preprocess import SignalsStage, VariantsStage
from llb.prep.book.stages.recognize import recognition_stages
from llb.prep.book.stages.render import RenderStage
from llb.prep.book.stages.resolve import ResolveStage
from llb.prep.book.structure.build import build_structure
from tests.llb.prep.book._page_factory import (
    bleed_page,
    blob_box,
    image_pdf,
    upside_down,
    with_blob,
)
from tests.llb.prep.book._planted_page import HEIGHT, LAYOUT, LINES, WIDTH, page_key, scaled
from tests.llb.prep.book.test_paragraph_reading import _ORDER

_PRIMARY = {"primary": {"stage": "layout", "engine": "layout-plant", "variant": "original"}}


class _Planted:
    """One engine, one stage. The other stage returns nothing."""

    stages = ("layout", "ocr", "formula", "table")

    def __init__(self, engine_id: str, stage: str) -> None:
        self.engine_id = engine_id
        self._stage = stage

    def available(self) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, request: EngineRequest, stage: str) -> EngineRead:
        blocks: tuple[Proposed, ...] = ()
        if stage == self._stage:
            table = LAYOUT if stage == "layout" else LINES
            blocks = scaled(request, table[page_key(request)])
        return EngineRead(self.engine_id, "planted-1", request.variant, stage, blocks)


class _Mark:
    """A blob and a bleed strip, read on the rotated variant."""

    engine_id = "mark"
    stages = ("layout", "ocr", "formula", "table")

    def available(self) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, request: EngineRequest, stage: str) -> EngineRead:
        assert request.variant == "rotated"
        if stage not in {"layout", "ocr"}:
            return EngineRead(self.engine_id, "planted-1", request.variant, stage, ())
        x0, y0, x1, y1 = blob_box(read_grey(request.image))
        if stage == "layout":
            blocks = (
                Proposed("body", 0, x0, y0, x1, y1, "MARK", None),
                Proposed("neighbour-page-bleed", 1, 0, 120, 16, 220, "NEIGHBOUR", None),
            )
        else:
            blocks = (
                Proposed(
                    "body",
                    0,
                    x0 + 1,
                    y0 + 1,
                    max(x0 + 2, x1 - 1),
                    max(y0 + 2, y1 - 1),
                    "MARK",
                    None,
                ),
                Proposed("neighbour-page-bleed", 1, 1, 140, 12, 200, "NEIGHBOUR", None),
            )
        return EngineRead(self.engine_id, "planted-1", request.variant, stage, blocks)


def _pdf(path: Path) -> Path:
    import fitz

    image = np.full((HEIGHT, WIDTH), 230, dtype=np.uint8)
    with fitz.open() as document:
        for _page in range(2):
            page = document.new_page(width=WIDTH, height=HEIGHT)
            pixels = fitz.Pixmap(fitz.csGRAY, WIDTH, HEIGHT, image.tobytes(), False)
            page.insert_image(page.rect, pixmap=pixels)
        document.save(path, garbage=3, deflate=True, no_new_id=True)
    return path


def _registry() -> StageRegistry:
    engines = (_Planted("layout-plant", "layout"), _Planted("line-plant", "ocr"))
    return StageRegistry(
        (RenderStage(), *recognition_stages(AdapterRegistry(engines)), ResolveStage())
    )


def _run(pdf: Path, out: Path, stages: tuple[BookStageSpec, ...]) -> Path:
    request = RunRequest(source=pdf, pages="1-2", stages=stages)
    return start_run(request, out, _registry()).run_dir


def _markdown(path: Path) -> str:
    files = sorted(path.rglob("*.md"))
    return "\n".join(item.read_text(encoding="utf-8") for item in files)


def _corpus_text(path: Path) -> str:
    files = [
        item
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.suffix not in {".json", ".png", ".pdf"}
    ]
    return "\n".join(item.read_text(encoding="utf-8") for item in files)


def test_vault_corpus_and_worksheet_keep_paragraphs(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    pdf = _pdf(library / "planted.pdf")
    stages = (
        BookStageSpec(name="render"),
        BookStageSpec(name="layout", params={"engine": "layout-plant"}),
        BookStageSpec(name="ocr", params={"engine": "line-plant"}),
        BookStageSpec(name="blocks"),
        BookStageSpec(name="resolve", params=_PRIMARY),
    )
    combined = _run(pdf, tmp_path / "runs", stages)
    ledger = read_ledger(ledger_path(combined, "p0001"))
    assert [_visible(block) for block in ledger.blocks] == _ORDER
    build_structure(combined)
    vault = build_vault(combined, RenderOptions(output=tmp_path / "vault"))
    prose = _markdown(vault)
    corpus_dir = tmp_path / "corpus"
    project_corpus(combined, corpus_dir, "redistributable", language="en")
    corpus = _corpus_text(corpus_dir)
    for sentence in (
        "Left alpha begins and holds.",
        "Left beta ends with necessary",
        "Right alpha stays apart.",
        "word follows.",
        "The discussion continuation",
        "from the next page.",
        "A fresh paragraph starts.",
        "BOXED EXAMPLE stays apart.",
        "STRAY LINE",
        "WHICH WAY",
    ):
        assert sentence in prose
    for sentence in (
        "Left alpha begins and holds.",
        "Left beta ends with neces-",
        "sary word follows.",
        "Right alpha stays apart.",
        "The discussion continua-",
        "tion from the next page.",
        "A fresh paragraph starts.",
        "BOXED EXAMPLE stays apart.",
        "STRAY LINE",
        "WHICH WAY",
    ):
        assert sentence in corpus
    assert "INTERLEAVED" not in prose and "INTERLEAVED" not in corpus
    assert "NEIGHBOUR" not in corpus and "> [page-number]" not in corpus
    assert prose.index("Left alpha") < prose.index("Right alpha")
    assert corpus.index("Left alpha") < corpus.index("Right alpha")
    assert prose.index("STRAY LINE") < prose.index("BOXED EXAMPLE")
    assert "STRAY LINE" not in prose.split("Right alpha stays apart.")[1].split("\n\n")[0]
    assert "Right alpha" not in prose.split("necessary")[0].split("Left beta")[-1]
    _worksheet(tmp_path, pdf, combined)


def _visible(block) -> str:
    for reading in block.readings:
        if reading.engine == "layout-plant":
            return reading.text
    return block.readings[0].text


def _worksheet(tmp_path: Path, pdf: Path, combined: Path) -> None:
    del combined
    layout = _run(
        pdf,
        tmp_path / "layout-runs",
        (
            BookStageSpec(name="render"),
            BookStageSpec(name="layout", params={"engine": "layout-plant"}),
            BookStageSpec(name="blocks"),
        ),
    )
    lines = _run(
        pdf,
        tmp_path / "line-runs",
        (
            BookStageSpec(name="render"),
            BookStageSpec(name="ocr", params={"engine": "line-plant"}),
            BookStageSpec(name="blocks"),
        ),
    )
    pilot = tmp_path / "pilot.json"
    _pilot(pdf, pilot)
    review = tmp_path / "review"
    prepare(DraftInput(source=pdf, pilot=pilot, primary=layout, alternate=lines), review)
    draft = json.loads((review / "predraft" / "pages" / "p0001.json").read_text(encoding="utf-8"))
    assert [block["text"] for block in draft["blocks"]] == _ORDER
    manifest = json.loads((review / "review.json").read_text(encoding="utf-8"))
    notes = {note["block_id"]: note for note in manifest["pages"][0]["notes"]}
    stray = next(block for block in draft["blocks"] if block["text"] == "STRAY LINE")
    unclear = next(block for block in draft["blocks"] if block["text"] == "WHICH WAY")
    assert notes[stray["block_id"]]["flags"] == ["No region owns this line."]
    assert "Placement is unclear." in notes[unclear["block_id"]]["flags"]
    rival = {item["text"] for item in notes[unclear["block_id"]]["alternate"]}
    assert rival == {"Left gamma holds.", "Right gamma holds."}
    assert "layout-plant" not in json.dumps(manifest)


def _pilot(pdf: Path, path: Path) -> None:
    signals = PlacedSignals(
        line_curvature_px=0,
        gutter_width_px=0,
        gutter_side="none",
        background_unevenness=0,
        contrast=0,
        orientation_deg=0,
        image_rotation=None,
        page_orientation="portrait",
        has_bleed=True,
        formula_density=0,
        equation_lines=0,
        ruling_lines=0,
        vertical_rules=0,
        ruled_fraction=0,
        figure_area=0,
        diagram_labels=0,
        column_count=2,
        footnote_band=0,
        corner_components=0,
        gutter_ink=0,
    )
    pages = [
        PilotPageEntry(
            pdf_page=number,
            page_id=page_id(number, 2),
            represents="multi_column",
            classes=["multi_column"],
            primary="multi_column",
            signals=signals,
        )
        for number in (1, 2)
    ]
    coverage = {name: 0 for name in PAGE_CLASS_IDS}
    coverage["multi_column"] = 2
    selection = PilotSelection(
        schema_id="llb.book-ocr-pilot-selection",
        schema_version="1.0.0",
        kind="book-ocr-pilot-selection",
        source_sha256=sha256_file(pdf),
        seed=1,
        size=2,
        pages=pages,
        empty_classes=[name for name in PAGE_CLASS_IDS if name != "multi_column"],
        unfilled_classes=[],
        coverage=coverage,
    )
    path.write_text(selection.model_dump_json(), encoding="utf-8")


def test_a_rotated_page_returns_the_mark_to_its_source_pixels(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    scan = upside_down(with_blob(bleed_page(), 180, 200, width=36, height=20))
    source = image_pdf(library / "rotated.pdf", scan)
    registry = StageRegistry(
        (
            RenderStage(),
            SignalsStage(),
            VariantsStage(),
            *recognition_stages(AdapterRegistry((_Mark(),))),
        )
    )
    stages = (
        BookStageSpec(name="render"),
        BookStageSpec(name="signals"),
        BookStageSpec(name="variants"),
        BookStageSpec(name="layout", params={"engine": "mark", "variant": "rotated"}),
        BookStageSpec(name="ocr", params={"engine": "mark", "variant": "rotated"}),
        BookStageSpec(name="blocks"),
    )
    result = start_run(
        RunRequest(source=source, pages="1", stages=stages), tmp_path / "runs", registry
    )
    page = read_ledger(ledger_path(result.run_dir, "p0001"))
    body = next(block for block in page.blocks if block.kind == "body")
    expected = blob_box(scan)
    for got, wanted in zip(
        (body.region.x0, body.region.y0, body.region.x1, body.region.y1), expected, strict=True
    ):
        assert abs(got - wanted) <= BOX_TOLERANCE_PX
    assert all("NEIGHBOUR" not in reading.text for reading in body.readings)
    assert any(block.kind == "neighbour-page-bleed" for block in page.blocks)
