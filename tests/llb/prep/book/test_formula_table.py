"""Formula and table structure: native grids, crops, and separate table faults."""

from pathlib import Path

from llb.core.contracts.data_prep.book_blocks import (
    Block,
    FormulaReading,
    RawReading,
    Region,
    TableCell,
    TableReading,
)
from llb.prep.book.engines.parse_common import _mineru, _raw
from llb.prep.book.engines.parse_docling import parse_docling
from llb.prep.book.engines.parse_text import parse_markdown
from llb.prep.book.engines.protocol import EngineRead, Proposed
from llb.prep.book.grid import formula_from_text, table_from_grid
from llb.prep.book.ledger.assemble import assemble
from llb.prep.book.ledger.cells import cell_disputes
from llb.prep.book.metrics.constants import CLASS_FIVE_S
from llb.prep.book.metrics.latex import latex_parses
from llb.prep.book.metrics.resolved import _table as resolved_table
from llb.prep.book.metrics.submission import _formula as scored_formula
from llb.prep.book.metrics.submission import _table as scored_table
from llb.prep.book.metrics.table_faults import (
    ALTERED_CELL,
    COLLAPSED_HEADER,
    LOST_EMPTY_CELL,
    WRONG_CAPTION,
    table_faults,
)
from llb.prep.book.render.bars import formula_needs_crop
from llb.prep.book.render.mathform import (
    IMAGE,
    render_body,
    render_formula,
    render_table,
    table_mode,
)

_WIDTH = 200
_HEIGHT = 400
_BOX = {
    "bbox": {"l": 0, "t": 40, "r": 80, "b": 0, "coord_origin": "BOTTOMLEFT"},
}


def test_spaced_equation_numbers_become_tags_and_exponents_stay() -> None:
    numbered = formula_from_text(r"F _ { s } = \lambda t \quad ( 2 4 - 1 1 )", display=True)
    assert numbered is not None
    assert numbered.equation_number == "24-11"
    assert numbered.number_side == "right"
    assert "( 2 4" not in numbered.latex
    suffixed = formula_from_text(
        r"W _ { F } = R _ { E } \quad ( 1 0 \, { - } 2 a ) \\", display=True
    )
    assert suffixed is not None and suffixed.equation_number == "10-2a"
    assert not suffixed.latex.endswith("\\\\")
    texted = formula_from_text(r"A \quad \text {(10-5)}", display=True)
    assert texted is not None and texted.equation_number == "10-5"
    exponent = formula_from_text(r"e ^ { ( 2 4 - 9 ) }", display=True)
    assert exponent is not None and exponent.equation_number is None
    assert "24-9" not in (exponent.equation_number or "")
    cdot = formula_from_text(r"\approx D \sin \theta & & ( 1 0 \cdot 2 b )", display=True)
    assert cdot is not None and cdot.equation_number == "10-2b"
    divided = formula_from_text(r"F \quad ( 1 0 \div 3 )", display=True)
    assert divided is not None and divided.equation_number == "10-3"
    product = formula_from_text(r"a = ( 2 \cdot 3 )", display=True)
    assert product is not None and product.equation_number is None


def test_identical_stage_texts_are_not_joined_twice() -> None:
    formula = formula_from_text(r"E = mc^2 \tag{10-28}", display=True)
    assert formula is not None
    proposed = Proposed("formula", 0, 0.0, 0.0, 80.0, 40.0, formula.latex, None, formula, None)
    reads = (
        EngineRead("docling", "m", "original", "formula", (proposed,)),
        EngineRead("docling", "m", "original", "table", (proposed,)),
    )
    ledger = assemble("p0001", 1, _WIDTH, _HEIGHT, reads)
    assert ledger.blocks[0].readings[0].text == formula.latex


def test_emitted_formula_parses_and_keeps_its_equation_number() -> None:
    reading = formula_from_text(r"(10-28) E = mc^2", display=True)
    assert reading is not None
    assert reading.equation_number == "10-28"
    assert reading.number_side == "left"
    rendered = render_formula(reading.latex, reading.equation_number or "", True, None)
    assert r"\tag{10-28}" in rendered
    inner = rendered.strip().removeprefix("$$").removesuffix("$$").strip()
    assert latex_parses(inner)


def test_formula_below_either_bar_keeps_its_crop() -> None:
    broken = _formula_block(r"\frac{a}{", 0.99)
    low = _formula_block(r"E = mc^2", 0.5)
    split = _formula_block(r"E = mc^2", 0.99)
    other = split.readings[0].model_copy(
        update={"formula": FormulaReading(latex=r"E = mc", equation_number="10-28")}
    )
    disagreed = split.model_copy(update={"readings": [split.readings[0], other]})
    sure = _formula_block("TOK-FORMULA", 0.99)
    assert formula_needs_crop(broken, broken.readings[0])
    assert formula_needs_crop(low, low.readings[0])
    assert formula_needs_crop(disagreed, disagreed.readings[0])
    assert not formula_needs_crop(sure, sure.readings[0])
    crop = render_formula(r"\frac{a}{", "", True, "review/p0001-b0000-formula.png")
    assert "review/p0001-b0000-formula.png" in crop
    assert render_body("a & b") == r"\begin{aligned}a & b\end{aligned}"


def test_table_modes_keep_empty_cells_spans_and_drop_an_unfaithful_grid() -> None:
    plain = _grid([["a", "b"], ["1", ""]], caption=None)
    assert table_mode(plain) != IMAGE
    markdown = render_table(plain, None, None, emit_caption=False)
    assert "| 1 |  |" in markdown
    spanned = table_from_grid([[{"text": "h", "col_span": 2, "column_header": True}], ["a", "b"]])
    assert spanned is not None
    html = render_table(spanned, None, None, emit_caption=False)
    assert 'colspan="2"' in html
    overlap = TableReading(
        rows=[
            [TableCell(text="a", colspan=2, rowspan=2), TableCell(text="b")],
            [TableCell(text="c", colspan=2)],
        ],
        faithful=False,
    )
    image = render_table(overlap, "assets/ch/p0001_0_table.png", None, emit_caption=False)
    assert image == "![[assets/ch/p0001_0_table.png]]"
    assert "|" not in image and "<table" not in image
    reading = RawReading(engine="a", variant="original", text="grid", table=overlap)
    assert scored_table("table", reading, "grid") is not None
    assert resolved_table("table", "grid", reading.table, reading) is None


def test_cell_dispute_keeps_a_dangerous_substitution_and_both_sources() -> None:
    left = _table_reading("docling", "5")
    right = _table_reading("mineru", "S")
    disputes = cell_disputes([left, right])
    assert len(disputes) == 1
    dispute = disputes[0]
    assert dispute.classes == ["digit"]
    assert CLASS_FIVE_S in dispute.substitutions
    assert [source.text for source in dispute.sources] == ["5", "S"]
    assert cell_disputes([_table_reading("docling", "cat"), _table_reading("mineru", "cot")]) == []


def test_each_table_fault_fires_alone() -> None:
    base = _grid([["h1", "h2"], ["a", "b"]], caption="Table 1")
    altered = _grid([["h1", "h2"], ["a", "c"]], caption="Table 1")
    collapsed = _grid([["h1", "h2"], ["sub", "head"], ["a", "b"]], caption="Table 1", headers=2)
    dropped = _grid([["h1"], ["a", "b"]], caption="Table 1")
    emptied = _grid([["h1", "h2"], ["a", ""]], caption="Table 1")
    missing = _grid([["h1", "h2"], ["a"]], caption="Table 1")
    retitled = _grid([["h1", "h2"], ["a", "b"]], caption="Table 2")
    assert table_faults(base, altered) == (ALTERED_CELL,)
    assert table_faults(collapsed, dropped) == (COLLAPSED_HEADER,)
    assert table_faults(emptied, missing) == (LOST_EMPTY_CELL,)
    assert table_faults(base, retitled) == (WRONG_CAPTION,)


def test_markdown_keeps_a_caption_an_empty_cell_and_inline_math() -> None:
    proposed = parse_markdown(
        "Table 1-1. Costs\n| a | b |\n| --- | --- |\n| 1 |  |\n\n"
        r"$$E = mc^2 \tag{10-28}$$"
        "\n$a+b$\n",
        _WIDTH,
        _HEIGHT,
    )
    table = next(item for item in proposed if item.kind == "table")
    formula = next(item for item in proposed if item.kind == "formula")
    inline = next(item for item in proposed if item.kind == "inline-formula")
    assert table.table is not None
    assert table.table.caption == "Table 1-1. Costs"
    assert table.table.rows[1][1].text == ""
    assert "Table 1-1" not in table.text
    assert formula.formula is not None
    assert formula.formula.equation_number == "10-28"
    assert formula.formula.number_side == "right"
    assert inline.formula is not None
    assert inline.formula.display is False


def test_docling_keeps_a_span_a_caption_and_an_equation_number() -> None:
    native = _docling()
    kept = next(item for item in native if item.kind == "table")
    numbered = next(item for item in native if item.kind == "formula")
    assert kept.table is not None
    assert kept.table.caption == "Table 1-1. Costs"
    assert kept.table.rows[0][0].colspan == 2
    assert kept.table.rows[1][0].text == ""
    assert numbered.formula is not None
    assert numbered.formula.equation_number == "10-28"
    assert numbered.formula.number_side == "left"


def test_assemble_and_scoring_use_the_native_grid() -> None:
    formula = formula_from_text(r"E = mc^2 \tag{10-28}", display=True)
    assert formula is not None
    proposed = Proposed("formula", 0, 0.0, 0.0, 80.0, 40.0, formula.latex, 0.99, formula, None)
    ledger = assemble(
        "p0001", 1, _WIDTH, _HEIGHT, (EngineRead("a", "m", "original", "formula", (proposed,)),)
    )
    stored = ledger.blocks[0].readings[0]
    assert stored.formula is not None and stored.formula.equation_number == "10-28"
    latex, number = scored_formula("formula", stored, stored.text)
    assert number == "10-28"
    assert latex is not None and "mc^2" in latex
    grid = _grid([["h"], ["5"]])
    reading = RawReading(engine="a", variant="original", text="h | 5", table=grid)
    scored = scored_table("table", reading, reading.text)
    assert scored is not None and scored[1][0].text == "5"


def test_raw_blocks_round_trip_formula_and_table(tmp_path: Path) -> None:
    path = tmp_path / "raw_blocks.json"
    path.write_text(
        '{"blocks": [{"kind": "formula", "x0": 0, "y0": 0, "x1": 1, "y1": 1, "text": "E",'
        ' "formula": {"latex": "E = mc^2", "equation_number": "10-28", "number_side": "right"}}]}',
        encoding="utf-8",
    )
    loaded = _raw(path)
    assert loaded[0].formula is not None and loaded[0].formula.equation_number == "10-28"
    kind, _text, formula, table = _mineru(
        {
            "type": "table",
            "text": "",
            "table_body": "<table><tr><td>a</td><td></td></tr></table>",
            "table_caption": ["Table 9"],
            "bbox": [0, 0, 1, 1],
        }
    )
    assert kind == "table" and table is not None and table.caption == "Table 9"
    assert table.rows[0][1].text == ""
    assert formula is None


def _formula_block(latex: str, confidence: float) -> Block:
    region = Region(x0=0, y0=0, x1=80, y1=40, width=_WIDTH, height=_HEIGHT)
    reading = formula_from_text(latex, display=True)
    raw = RawReading(
        engine="a", variant="original", text=latex, confidence=confidence, formula=reading
    )
    return Block(
        block_id="p0001-b0000",
        pdf_page=1,
        ordinal=0,
        kind="formula",
        region=region,
        reading_order=0,
        readings=[raw],
    )


def _grid(rows: list[list[str]], *, caption: str | None = None, headers: int = 1) -> TableReading:
    return TableReading(
        rows=[[TableCell(text=text) for text in row] for row in rows],
        header_rows=headers,
        caption=caption,
    )


def _table_reading(engine: str, text: str) -> RawReading:
    return RawReading(
        engine=engine,
        variant="original",
        text=text,
        table=_grid([[text]]),
    )


def _docling() -> list[Proposed]:
    payload: dict[str, object] = {
        "pages": {"1": {"size": {"width": _WIDTH, "height": _HEIGHT}}},
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "caption",
                "text": "Table 1-1. Costs",
                "prov": [_BOX],
            },
            {
                "self_ref": "#/texts/1",
                "label": "formula",
                "text": "(10-28) E = mc^2",
                "prov": [_BOX],
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "captions": [{"$ref": "#/texts/0"}],
                "prov": [_BOX],
                "data": {
                    "table_cells": [
                        {
                            "text": "h",
                            "start_row_offset_idx": 0,
                            "start_col_offset_idx": 0,
                            "col_span": 2,
                            "column_header": True,
                        },
                        {
                            "text": "",
                            "start_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "column_header": False,
                        },
                        {
                            "text": "b",
                            "start_row_offset_idx": 1,
                            "start_col_offset_idx": 1,
                            "column_header": False,
                        },
                    ]
                },
            }
        ],
        "body": {
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/tables/0"},
                {"$ref": "#/texts/1"},
            ]
        },
    }
    return parse_docling(payload, _WIDTH, _HEIGHT)
