"""Engine adapters: the fake engine, page-class selection, and a named refusal."""

import sys

import pytest
from typer.testing import CliRunner

from llb.cli import app
from llb.core.contracts.data_prep.book import BookStageSpec
from llb.prep.book.config import RunRequest
from llb.prep.book.engines.fake import FAKE_ENGINE
from llb.prep.book.engines.kinds import fit_proposals
from llb.prep.book.engines.parse_docling import parse_docling
from llb.prep.book.engines.parse_text import parse_dots, parse_grounding, parse_markdown
from llb.prep.book.engines.protocol import EngineRead, EngineRequest, Proposed
from llb.prep.book.engines.registry import (
    AdapterRegistry,
    default_adapter_registry,
    describe_engines,
)
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.runner import BookRunError, start_run
from llb.prep.book.stages import StageRegistry
from llb.prep.book.stages.recognize import recognition_stages
from llb.prep.book.stages.render import RenderStage

RUNNER = CliRunner()
_SHORTLIST = {
    "rapidocr",
    "mineru-basic",
    "mineru-standard",
    "docling",
    "docling-vlm-mineru2-pro",
    "docling-vlm-surya-ocr2-5",
    "paddleocr-vl",
    "marker-balanced",
    "dots-ocr",
    "deepseek-ocr-2",
    "olmocr-2",
    "ollama-glm-ocr-latest",
    "ollama-deepseek-ocr-3b",
}
_DROPPED = {
    "marker-fast",
    "pp-structurev3",
    "granite-docling",
    "docling-vlm-granite-docling",
    "docling-vlm-chandra-ocr2",
    "docling-vlm-lightonocr3-4b",
    "ollama-qwen3-vl-8b",
    "ollama-qwen2.5vl-7b",
}


class _Scripted:
    """A test engine whose whole page is one block of fixed text."""

    def __init__(self, engine_id: str, text: str, stages: tuple[str, ...]) -> None:
        self.engine_id = engine_id
        self.text = text
        self.stages = stages

    def available(self) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, request: EngineRequest, stage: str) -> EngineRead:
        block = Proposed(
            "body", 0, 0.0, 0.0, float(request.width), float(request.height), self.text, None
        )
        return EngineRead(self.engine_id, "script", request.variant, stage, (block,))


def _request(scan, stages) -> RunRequest:
    return RunRequest(source=scan, pages="1", batch_pages=1, stages=tuple(stages))


def test_importing_the_package_does_not_load_an_ocr_library():
    import llb.prep.book.engines  # noqa: F401
    from llb.prep.book.stages import default_registry

    default_registry()
    assert "rapidocr" not in sys.modules
    assert "paddleocr" not in sys.modules


def test_shortlist_is_registered_and_the_dropped_engines_are_not():
    ids = {adapter.engine_id for adapter in default_adapter_registry().all()}
    assert ids == _SHORTLIST | {FAKE_ENGINE}
    assert ids.isdisjoint(_DROPPED)
    by_id = {adapter.engine_id: adapter for adapter in default_adapter_registry().all()}
    assert by_id["rapidocr"].stages == ("ocr",)
    assert "layout" not in by_id["olmocr-2"].stages
    assert "layout" in by_id["dots-ocr"].stages


def test_fake_engine_ledger_is_stable_across_runs(scan_pdf, out_root):
    stages = (
        BookStageSpec(name="render"),
        BookStageSpec(name="ocr", params={"engine": "fake"}),
        BookStageSpec(name="blocks"),
    )
    registry = StageRegistry((RenderStage(), *recognition_stages(default_adapter_registry())))
    first = start_run(_request(scan_pdf, stages), out_root, registry)
    second = start_run(_request(scan_pdf, stages), out_root, registry)
    published = [
        (run.run_dir / "pages" / "p0001" / "blocks.json").read_bytes() for run in (first, second)
    ]
    assert published[0] == published[1]
    ledger = read_ledger(first.run_dir / "pages" / "p0001" / "blocks.json")
    assert [block.block_id for block in ledger.blocks] == [
        "p0001-b0000",
        "p0001-b0001",
        "p0001-b0002",
    ]
    assert ledger.blocks[1].readings[0].text == "The value is 0.05."
    stage_dir = first.run_dir / "pages" / "p0001" / "ocr"
    assert (stage_dir / "reading.json").is_file()
    assert not (stage_dir / "page.pdf").exists()
    assert not list(stage_dir.glob("*.log"))


def test_page_class_selects_the_configured_engine(scan_pdf, out_root):
    adapters = AdapterRegistry(
        (
            _Scripted("alpha", "ALPHA", ("ocr",)),
            _Scripted("beta", "BETA", ("ocr",)),
        )
    )
    stages = (
        BookStageSpec(name="render"),
        BookStageSpec(
            name="ocr",
            params={
                "engine": "alpha",
                "page_class": "formula",
                "by_page_class": {"formula": "beta"},
            },
        ),
        BookStageSpec(name="blocks"),
    )
    outcome = start_run(
        _request(scan_pdf, stages),
        out_root,
        StageRegistry((RenderStage(), *recognition_stages(adapters))),
    )
    reading = read_ledger(outcome.run_dir / "pages" / "p0001" / "blocks.json").blocks[0].readings[0]
    assert reading.engine == "beta" and reading.text == "BETA"


def test_absent_engine_is_a_named_refusal(scan_pdf, out_root, monkeypatch, tmp_path):
    monkeypatch.setattr("llb.prep.book.engines.candidate.resolve_data_dir", lambda: tmp_path)
    stages = (
        BookStageSpec(name="render"),
        BookStageSpec(name="ocr", params={"engine": "rapidocr"}),
        BookStageSpec(name="blocks"),
    )
    registry = StageRegistry((RenderStage(), *recognition_stages(default_adapter_registry())))
    with pytest.raises(BookRunError) as caught:
        start_run(_request(scan_pdf, stages), out_root, registry)
    message = str(caught.value)
    assert message.startswith("EngineUnavailable:") or "EngineUnavailable:" in message
    assert "Traceback" not in message
    assert "rapidocr" in message


def test_blocks_without_a_recognition_stage_is_refused(scan_pdf):
    from llb.prep.book.runner import check_request

    plan = check_request(
        RunRequest(
            source=scan_pdf,
            pages="1",
            stages=(BookStageSpec(name="render"), BookStageSpec(name="blocks")),
        )
    )
    assert any("needs a layout, ocr, formula, or table" in error for error in plan.errors)


def test_engines_command_lists_ready_and_refused(monkeypatch):
    class _Down:
        engine_id = "missing"
        stages = ("ocr",)

        def available(self) -> None:
            from llb.prep.book.engines.protocol import EngineUnavailable

            raise EngineUnavailable("missing", "not installed")

    monkeypatch.setattr(
        "llb.prep.book.engines.registry.default_adapter_registry",
        lambda: AdapterRegistry((_Down(),)),
    )
    lines = describe_engines()
    assert lines == ["missing\tocr\trefused: not installed"]
    result = RUNNER.invoke(app, ["book-ocr", "engines"])
    assert result.exit_code == 0, result.output
    assert "missing" in result.output and "refused: not installed" in result.output
    help_text = RUNNER.invoke(app, ["book-ocr", "--help"])
    assert help_text.exit_code == 0 and "engines" in help_text.output


def test_pixel_boxes_are_not_rescaled_one_line_at_a_time():
    """A line in the top of a tall page stays in pixels when another line extends past 1000."""
    page = (1241, 1755)
    proposals = [
        Proposed("body", 0, 493.0, 116.0, 715.0, 156.0, "Title", None),
        Proposed("body", 1, 642.0, 207.0, 1119.0, 264.0, "715.2 days", None),
    ]
    fitted = fit_proposals(proposals, *page, *page)
    assert (fitted[0].y0, fitted[0].y1) == (116.0, 156.0)
    assert fitted[1].x1 == 1119.0
    thousandths = fit_proposals(
        [
            Proposed("body", 0, 0.0, 0.0, 500.0, 100.0, "A", None),
            Proposed("body", 1, 0.0, 800.0, 500.0, 900.0, "B", None),
        ],
        *page,
        *page,
    )
    assert thousandths[1].y0 == pytest.approx(800.0 / 1000.0 * 1755.0)


def test_parsers_keep_boxes_and_drop_figure_prose():
    width, height = 100, 200
    docling = parse_docling(
        {
            "pages": {"1": {"size": {"width": 100, "height": 200}}},
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "label": "text",
                    "text": "Hello",
                    "prov": [
                        {
                            "bbox": {
                                "l": 0,
                                "t": 200,
                                "r": 50,
                                "b": 180,
                                "coord_origin": "BOTTOMLEFT",
                            }
                        }
                    ],
                }
            ],
            "groups": [
                {"self_ref": "#/groups/0", "children": [{"$ref": "#/pictures/0"}]},
            ],
            "pictures": [
                {
                    "self_ref": "#/pictures/0",
                    "prov": [
                        {
                            "bbox": {
                                "l": 0,
                                "t": 20,
                                "r": 10,
                                "b": 0,
                                "coord_origin": "BOTTOMLEFT",
                            }
                        }
                    ],
                }
            ],
            "body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/groups/0"}]},
        },
        width,
        height,
    )
    fitted = fit_proposals(docling, width, height, width, height)
    assert [(item.kind, item.text, item.y0) for item in fitted] == [
        ("body", "Hello", 0.0),
        ("figure", "", 180.0),
    ]
    grounded = fit_proposals(
        parse_grounding("text[[0, 0, 500, 100]]\nHello", 1000, 1000), 1000, 1000, 1000, 1000
    )
    assert grounded[0].text == "Hello"
    assert (grounded[0].x1, grounded[0].y1) == (500.0, 100.0)
    dots = fit_proposals(
        parse_dots('[{"bbox": [0, 0, 0.5, 0.5], "category": "Text", "text": "Hi"}]', 100, 100),
        100,
        100,
        100,
        100,
    )
    assert dots[0].kind == "body" and dots[0].x1 == 50.0
    figure = fit_proposals(
        parse_markdown("![a rocket](page_10_20_30_40.png)\n\nThe figure shows a rocket.", 100, 100),
        100,
        100,
        100,
        100,
    )
    assert all(item.text == "" and item.kind == "figure" for item in figure)
