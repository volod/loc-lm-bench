"""Repeated whole-page readings collapse, and native formula and table fields come back."""

from llb.core.contracts.data_prep.book import BookStageSpec
from llb.prep.book.config import RunRequest
from llb.prep.book.engines.parse_middle import parse_middle
from llb.prep.book.engines.parse_text import parse_markdown
from llb.prep.book.engines.protocol import EngineRead, EngineRequest, Proposed
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.engines.shortlist import shortlist_adapters
from llb.prep.book.grid import formula_from_text, table_from_text
from llb.prep.book.ledger.assemble import _Tagged, assemble
from llb.prep.book.ledger.repeat import collapse_repeated
from llb.prep.book.runner import start_run
from llb.prep.book.stages import StageRegistry
from llb.prep.book.stages.recognize import recognition_stages
from llb.prep.book.stages.render import RenderStage

_STAGES = ("layout", "ocr", "formula", "table")


def test_same_box_copies_collapse_and_a_line_inside_a_region_does_not() -> None:
    formula = Proposed("formula", 0, 0.0, 0.0, 80.0, 40.0, r"E = mc^2 \tag {10-28}", None)
    copies = [_Tagged(stage, "docling", "m", "original", formula) for stage in _STAGES]
    merged = collapse_repeated(copies)
    assert len(merged) == 1
    assert merged[0].proposed.formula is not None
    assert merged[0].proposed.formula.equation_number == "10-28"
    assert "tag" not in merged[0].proposed.formula.latex
    region = Proposed("body", 0, 0.0, 0.0, 100.0, 100.0, "Paragraph", None)
    line = Proposed("body", 1, 0.0, 0.0, 100.0, 20.0, "Paragraph", None)
    kept = collapse_repeated(
        [
            _Tagged("layout", "docling", "m", "original", region),
            _Tagged("ocr", "docling", "m", "original", line),
        ]
    )
    assert len(kept) == 2
    other = collapse_repeated(
        [
            _Tagged("layout", "docling", "m", "original", formula),
            _Tagged("layout", "paddle", "m", "original", formula),
        ]
    )
    assert len(other) == 2


def test_four_stage_copies_become_one_ledger_block() -> None:
    proposed = Proposed("body", 0, 10.0, 10.0, 90.0, 40.0, "Once", None)
    reads = tuple(EngineRead("docling", "m", "original", stage, (proposed,)) for stage in _STAGES)
    ledger = assemble("p0001", 1, 200, 400, reads)
    assert len(ledger.blocks) == 1
    assert ledger.blocks[0].readings[0].text == "Once"


def test_spaced_tag_and_display_wrapper_become_an_equation_number() -> None:
    reading = formula_from_text(r"\[R = e \tag {24-9}\]", display=True)
    assert reading is not None
    assert reading.equation_number == "24-9"
    assert reading.number_side == "right"
    assert "tag" not in reading.latex
    assert reading.latex.startswith("R")


def test_repeated_markdown_header_keeps_spans_and_an_empty_cell() -> None:
    text = "\n".join(
        [
            "| Name | Rate | Rate | Notes |",
            "| Name | Low | High | Notes |",
            "| A | 1 |  |  |",
        ]
    )
    table = table_from_text(text)
    assert table is not None
    assert table.header_rows == 2
    assert table.faithful
    assert [(cell.text, cell.colspan, cell.rowspan) for cell in table.rows[0]] == [
        ("Name", 1, 2),
        ("Rate", 2, 1),
        ("Notes", 1, 2),
    ]
    assert [cell.text for cell in table.rows[1]] == ["Low", "High"]
    assert table.rows[2][2].text == ""
    plain = table_from_text("| Name | Rate |\n| --- | --- |\n| A | 1 |")
    assert plain is not None
    assert plain.header_rows == 1
    assert plain.rows[0][0].colspan == 1
    trailing = table_from_text("Name | Low | High |\nA | 1 | 2 |")
    assert trailing is not None
    assert [cell.text for cell in trailing.rows[0]] == ["Name", "Low", "High", ""]
    assert [cell.text for cell in trailing.rows[1]] == ["A", "1", "2", ""]


def test_multiline_display_math_is_one_formula() -> None:
    blocks = parse_markdown("$$\nE = mc^2 \\tag{10-28}\n$$\n\nAfter.", 100, 200)
    formulas = [block for block in blocks if block.kind == "formula"]
    assert len(formulas) == 1
    assert formulas[0].formula is not None
    assert formulas[0].formula.equation_number == "10-28"
    assert formulas[0].text.startswith("E")
    assert blocks[-1].kind == "body"
    inline = parse_markdown("$$E = mc^2$$", 100, 100)
    assert len(inline) == 1 and inline[0].kind == "formula"


def test_middle_json_scales_boxes_and_keeps_a_caption_off_the_grid() -> None:
    payload = {
        "schema": "docvortex.middle",
        "pages": [
            {
                "blocks": [
                    {
                        "type": "text",
                        "bbox": [0.1, 0.2, 0.4, 0.3],
                        "content": [
                            {"type": "text", "content": "Hello "},
                            {"type": "equation_inline", "content": "x"},
                        ],
                    },
                    {
                        "type": "equation",
                        "bbox": [10, 20, 80, 40],
                        "content": r"E = mc^2 \tag{1-2}",
                    },
                    {
                        "type": "image",
                        "bbox": [0.5, 0.1, 0.9, 0.3],
                        "content": "a drawing of a rocket",
                    },
                    {
                        "type": "table",
                        "bbox": [0.1, 0.6, 0.9, 0.9],
                        "content": [
                            {
                                "type": "table_caption",
                                "content": [{"type": "text", "content": "Table 1. Parts"}],
                            },
                            {"type": "table_body", "content": "| Name | Rate |\n| A | 1 |"},
                        ],
                    },
                ]
            }
        ],
    }
    proposed = parse_middle(payload, 1000, 2000)
    assert proposed[0].text == "Hello $x$"
    assert proposed[0].x0 == 100.0 and proposed[0].y0 == 400.0
    assert proposed[1].x0 == 10.0
    assert proposed[1].formula is not None
    assert proposed[1].formula.equation_number == "1-2"
    assert proposed[2].kind == "figure" and proposed[2].text == ""
    assert proposed[3].table is not None
    assert proposed[3].table.caption == "Table 1. Parts"
    assert proposed[3].text.startswith("| Name")


def test_a_stored_one_row_header_is_rebuilt_from_html() -> None:
    from llb.core.contracts.data_prep.book_blocks import TableCell, TableReading

    text = (
        "<table><tr><td rowspan='2'>Name</td><td colspan='2'>Rate</td>"
        "<td rowspan='2'>Notes</td></tr>"
        "<tr><td>Low</td><td>High</td></tr>"
        "<tr><td>A</td><td>1</td><td></td><td></td></tr></table>"
    )
    stored = TableReading(
        rows=[[TableCell(text="Name"), TableCell(text="Rate")]],
        header_rows=1,
        caption="Table 1. Parts",
    )
    proposed = Proposed("table", 0, 0.0, 0.0, 80.0, 40.0, text, None, None, stored)
    ledger = assemble(
        "p0001",
        1,
        200,
        400,
        (EngineRead("mineru-standard", "m", "original", "layout", (proposed,)),),
    )
    table = ledger.blocks[0].readings[0].table
    assert table is not None
    assert table.header_rows == 2
    assert table.caption == "Table 1. Parts"
    assert table.rows[2][2].text == ""


def test_html_merged_header_is_two_rows_when_the_cells_are_not_th() -> None:
    text = (
        "<table><tr><td rowspan='2'>Name</td><td colspan='2'>Rate</td>"
        "<td rowspan='2'>Notes</td></tr>"
        "<tr><td>Low</td><td>High</td></tr>"
        "<tr><td>A</td><td>1</td><td></td><td></td></tr></table>"
    )
    table = table_from_text(text)
    assert table is not None
    assert table.header_rows == 2
    assert table.rows[0][1].colspan == 2
    assert table.rows[0][0].rowspan == 2
    assert table.rows[2][2].text == ""


def test_mineru_recovery_asks_for_middle_json() -> None:
    mineru = next(item for item in shortlist_adapters() if item.engine_id == "mineru-standard")
    assert mineru.spec.argv is not None
    assert "middle_json" in mineru.spec.argv
    assert mineru.shares_reading


class _Counting:
    """A whole-page engine. `read` ignores the stage, as the real adapters do."""

    def __init__(self) -> None:
        self.calls = 0
        self.engine_id = "whole"
        self.stages = _STAGES
        self.shares_reading = True

    def available(self) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, request: EngineRequest, stage: str) -> EngineRead:
        self.calls += 1
        block = Proposed("body", 0, 10.0, 10.0, 80.0, 40.0, "Once", None)
        return EngineRead(self.engine_id, "script", request.variant, stage, (block,))


def test_a_later_recognition_stage_reuses_the_saved_reading(scan_pdf, out_root) -> None:
    engine = _Counting()
    stages = (
        BookStageSpec(name="render"),
        *[BookStageSpec(name=name, params={"engine": "whole"}) for name in _STAGES],
        BookStageSpec(name="blocks"),
    )
    outcome = start_run(
        RunRequest(source=scan_pdf, pages="1", batch_pages=1, stages=stages),
        out_root,
        StageRegistry((RenderStage(), *recognition_stages(AdapterRegistry((engine,))))),
    )
    assert engine.calls == 1
    page = outcome.run_dir / "pages" / "p0001"
    ocr = (page / "ocr" / "stage.json").read_text(encoding="utf-8")
    assert '"shared_from": "layout"' in ocr
    reading = (page / "ocr" / "reading.json").read_text(encoding="utf-8")
    assert '"stage": "ocr"' in reading
    assert '"text": "Once"' in reading
