"""The classes stage, engine routing from it, and the pilot command."""

import json
from pathlib import Path

from typer.testing import CliRunner

from llb.cli import app
from llb.core.contracts.data_prep.book import BookStageSpec
from llb.core.contracts.data_prep.book_classes import PAGE_CLASS_IDS
from llb.core.fsutil import atomic_write_text
from llb.prep.book.cache import list_outputs, read_record, stage_dir
from llb.prep.book.classes.assign import matching_classes
from llb.prep.book.classes.records import read_classes, read_selection, write_classes
from llb.prep.book.config import RunRequest
from llb.prep.book.constants import STAGE_RECORD_FILENAME
from llb.prep.book.engines.protocol import EngineRead, EngineRequest, Proposed
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.ranges import page_id
from llb.prep.book.runner import resume_run, start_run
from llb.prep.book.stages import StageContext, StageRegistry, StageResult
from llb.prep.book.stages.classes import ClassesStage
from llb.prep.book.stages.preprocess import SignalsStage
from llb.prep.book.stages.recognize import recognition_stages
from llb.prep.book.stages.render import RenderStage

RUNNER = CliRunner()
_SOURCE_SHA = "89101813d4c695e9f992d540e751c4e2f4b80c6550f24d5d23e1a795383cb7c8"


class _Counting:
    """A test engine that records every time a batch opens it."""

    def __init__(self, engine_id: str, text: str) -> None:
        self.engine_id = engine_id
        self.text = text
        self.stages = ("layout", "ocr")
        self.opens: list[str] = []
        self.reads = 0

    def available(self) -> None:
        return None

    def open(self) -> None:
        self.opens.append(self.engine_id)

    def close(self) -> None:
        return None

    def read(self, request: EngineRequest, stage: str) -> EngineRead:
        self.reads += 1
        block = Proposed(
            "body", 0, 0.0, 0.0, float(request.width), float(request.height), self.text, None
        )
        return EngineRead(self.engine_id, "script", request.variant, stage, (block,))


class _Alternating:
    """Writes `dense_formulas` on odd pages and `normal` on even ones. No measurement."""

    name = "classes"
    version = "1"
    inputs = ("render",)
    reads_source = False
    model_required = False

    def tool_identity(self) -> str:
        return "alternating-classes"

    def validate_params(self, params: object) -> list[str]:
        del params
        return []

    def effective_params(
        self, params: object, inventory: object, document: object, pdf_page: int
    ) -> dict[str, int]:
        del params, inventory, document
        return {"pdf_page": pdf_page}

    def run(self, ctx: StageContext) -> StageResult:
        from llb.prep.book.classes.records import classes_path
        from tests.llb.prep.book.test_page_classes import _page

        primary = "dense_formulas" if ctx.pdf_page % 2 else "normal"
        write_classes(classes_path(ctx.work_dir), _page(ctx.pdf_page, [primary]))
        return StageResult(meta={"primary": primary})


def test_classes_mark_the_landscape_pages_and_repeat_byte_for_byte(scan_pdf: Path, out_root: Path):
    registry = StageRegistry((RenderStage(), SignalsStage(), ClassesStage()))
    request = RunRequest(
        source=scan_pdf,
        pages="1-6",
        batch_pages=6,
        stages=(
            BookStageSpec(name="render"),
            BookStageSpec(name="signals"),
            BookStageSpec(name="classes"),
        ),
    )
    first = start_run(request, out_root, registry)
    second = start_run(request, out_root, registry)
    for number in range(1, 7):
        pid = page_id(number, 6)
        left = (first.run_dir / "pages" / pid / "classes.json").read_bytes()
        right = (second.run_dir / "pages" / pid / "classes.json").read_bytes()
        assert left == right
        record = read_classes(first.run_dir / "pages" / pid / "classes.json")
        if number in (3, 4):
            assert record.primary == "landscape" or "landscape" in record.classes
        else:
            assert "landscape" not in record.classes


def test_alternating_classes_open_each_routed_engine_once(scan_pdf: Path, out_root: Path):
    alpha, beta = _Counting("alpha", "ALPHA"), _Counting("beta", "BETA")
    registry = StageRegistry(
        (RenderStage(), _Alternating(), *recognition_stages(AdapterRegistry((alpha, beta))))
    )
    request = RunRequest(
        source=scan_pdf,
        pages="1-4",
        batch_pages=4,
        stages=(
            BookStageSpec(name="render"),
            BookStageSpec(name="classes"),
            BookStageSpec(
                name="ocr",
                params={
                    "engine": "alpha",
                    "by_page_class": {"dense_formulas": "beta", "normal": "alpha"},
                },
            ),
        ),
    )
    outcome = start_run(request, out_root, registry)
    assert alpha.opens == ["alpha"]
    assert beta.opens == ["beta"]
    odd = read_classes(outcome.run_dir / "pages" / "p0001" / "classes.json")
    even = read_classes(outcome.run_dir / "pages" / "p0002" / "classes.json")
    assert odd.primary == "dense_formulas" and even.primary == "normal"


def test_a_changed_class_reruns_only_the_stage_that_routes_on_it(scan_pdf: Path, out_root: Path):
    alpha, beta = _Counting("alpha", "ALPHA"), _Counting("beta", "BETA")
    routed = {
        name: "alpha"
        for name in (
            "normal",
            "landscape",
            "multi_column",
            "grey_background",
            "gutter_corner",
            "small_footnotes",
            "labelled_diagrams",
            "large_tables",
            "severe_curvature",
        )
    }
    routed["dense_formulas"] = "beta"
    registry = StageRegistry(
        (
            RenderStage(),
            SignalsStage(),
            ClassesStage(),
            *recognition_stages(AdapterRegistry((alpha, beta))),
        )
    )
    request = RunRequest(
        source=scan_pdf,
        pages="1-2",
        batch_pages=2,
        stages=(
            BookStageSpec(name="render"),
            BookStageSpec(name="signals"),
            BookStageSpec(name="classes"),
            BookStageSpec(name="layout", params={"engine": "alpha"}),
            BookStageSpec(name="ocr", params={"engine": "alpha", "by_page_class": routed}),
        ),
    )
    outcome = start_run(request, out_root, registry)
    run_dir = outcome.run_dir
    current = read_classes(stage_dir(run_dir, page_id(1, 6), "classes") / "classes.json")
    target = "normal" if current.primary == "dense_formulas" else "dense_formulas"
    before = _keys(run_dir)
    _retarget(run_dir, 1, target)
    alpha.reads = 0
    beta.reads = 0
    resume_run(run_dir, registry=registry)
    after = _keys(run_dir)
    assert after["layout"] == before["layout"]
    assert after["ocr"][2] == before["ocr"][2]
    assert after["ocr"][1] != before["ocr"][1]
    assert (beta.reads, alpha.reads) == ((1, 0) if target == "dense_formulas" else (0, 1))


def test_pilot_select_writes_the_checksum_and_names_empty_classes(scan_pdf: Path, out_root: Path):
    registry = StageRegistry((RenderStage(), SignalsStage(), ClassesStage()))
    outcome = start_run(
        RunRequest(
            source=scan_pdf,
            pages="1-4",
            batch_pages=4,
            stages=(
                BookStageSpec(name="render"),
                BookStageSpec(name="signals"),
                BookStageSpec(name="classes"),
            ),
        ),
        out_root,
        registry,
    )
    result = RUNNER.invoke(
        app, ["book-ocr", "pilot-select", str(outcome.run_dir), "--size", "2", "--seed", "0"]
    )
    assert result.exit_code == 0, result.output
    selection = read_selection(outcome.run_dir / "pilot" / "selection.json")
    assert selection.seed == 0 and len(selection.pages) == 2
    assert len(selection.source_sha256) == 64
    assert "severe curvature" in result.output or "landscape" in result.output
    missing = RUNNER.invoke(app, ["book-ocr", "pilot-select", str(out_root)])
    assert missing.exit_code == 2


def test_the_committed_handbook_pilot_names_every_class():
    path = Path("samples/book_ocr/space-mission-engineering-the-new-smad/pilot_pages.json")
    selection = read_selection(path)
    assert selection.source_sha256 == _SOURCE_SHA
    assert (selection.seed, selection.size, len(selection.pages)) == (0, 25, 25)
    assert (selection.empty_classes, selection.unfilled_classes) == ([], [])
    assert selection.coverage == dict(zip(PAGE_CLASS_IDS, [3] * 5 + [2] * 5, strict=True))
    for page in selection.pages:
        assert 1 <= page.pdf_page <= 1067
        assert page.page_id == page_id(page.pdf_page, 1067)
        assert page.classes == matching_classes(page.signals)
        assert page.primary == page.classes[0]
    text = path.read_text(encoding="utf-8")
    assert "page.png" not in text and "transcription" not in text


def _keys(run_dir: Path) -> dict[str, dict[int, str]]:
    found: dict[str, dict[int, str]] = {"layout": {}, "ocr": {}}
    for number in (1, 2):
        pid = page_id(number, 6)
        for stage in ("layout", "ocr"):
            record = read_record(stage_dir(run_dir, pid, stage))
            assert record is not None and record.cache_key
            found[stage][number] = record.cache_key
    return found


def _retarget(run_dir: Path, pdf_page: int, primary: str) -> None:
    pid = page_id(pdf_page, 6)
    directory = stage_dir(run_dir, pid, "classes")
    current = read_classes(directory / "classes.json")
    write_classes(
        directory / "classes.json",
        current.model_copy(update={"classes": [primary], "primary": primary}),
    )
    stage = read_record(directory)
    assert stage is not None
    rewritten = stage.model_copy(update={"outputs": list_outputs(directory)})
    atomic_write_text(
        directory / STAGE_RECORD_FILENAME,
        json.dumps(rewritten.model_dump(mode="json"), ensure_ascii=True) + "\n",
    )
