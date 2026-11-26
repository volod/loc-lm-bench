"""Paragraph boundaries recovered from regions, one failure at a time."""

from dataclasses import replace

from llb.core.contracts.data_prep.book_blocks import Region
from llb.prep.book.engines.kinds import clean_kind_text
from llb.prep.book.engines.protocol import EngineRead
from llb.prep.book.ledger.assemble import assemble
from llb.prep.book.render.join import paragraph_runs, stitch_columns
from llb.prep.book.render.types import RenderBlock, Unit
from tests.llb.prep.book._planted_page import HEIGHT, LAYOUT, LINES, WIDTH, proposed

_ORDER = [
    "Span Title",
    "Left alpha begins and holds.",
    "NEIGHBOUR BLEED",
    "Left beta ends with neces-",
    "sary word follows.",
    "Right alpha stays apart.",
    "STRAY LINE",
    "BOXED EXAMPLE stays apart.",
    "Left gamma holds.",
    "Right gamma holds.",
    "WHICH WAY",
    "speed | 9.81",
    "Table 1-1 Planted caption.",
    "The discussion continua-",
]


def _ledger():
    layout = EngineRead("layout", "m", "original", "layout", proposed(LAYOUT["p0001"]))
    lines = EngineRead("ocr", "m", "original", "ocr", proposed(LINES["p0001"]))
    return assemble("p0001", 1, WIDTH, HEIGHT, [layout, lines])


def _shown(block) -> str:
    for reading in block.readings:
        if reading.engine == "layout":
            return reading.text
    return block.readings[0].text


def test_reading_order_is_columns_and_drops_the_interleaved_page_box():
    blocks = _ledger().blocks
    assert [_shown(block) for block in blocks] == _ORDER
    assert [block.reading_order for block in blocks] == list(range(len(blocks)))
    assert all("INTERLEAVED" not in reading.text for block in blocks for reading in block.readings)
    left = _shown(blocks[1])
    right = next(_shown(block) for block in blocks if _shown(block).startswith("Right alpha"))
    assert blocks[1].reading_order < next(
        block.reading_order for block in blocks if _shown(block) == right
    )
    assert left.startswith("Left")


def test_paragraph_text_keeps_its_region_and_leaves_strays_and_asides_out():
    blocks = _ledger().blocks
    by_text = {_shown(block): block for block in blocks}
    assert by_text["Left alpha begins and holds."].paragraph
    assert not by_text["STRAY LINE"].paragraph
    assert not by_text["WHICH WAY"].paragraph
    assert by_text["BOXED EXAMPLE stays apart."].kind == "unknown"
    bodies = [block for block in blocks if block.kind == "body"]
    assert [reading.text for reading in by_text["STRAY LINE"].readings] == ["STRAY LINE"]
    assert [reading.text for reading in by_text["WHICH WAY"].readings] == ["WHICH WAY"]
    assert all("BOXED" not in reading.text for block in bodies for reading in block.readings)
    assert all("Span Title" not in reading.text for block in bodies for reading in block.readings)
    assert all("NEIGHBOUR" not in reading.text for block in bodies for reading in block.readings)
    assert all(reading.text.strip() != "9.81" for block in bodies for reading in block.readings)
    stray = by_text["STRAY LINE"]
    assert stray.region.y0 == 390
    assert sum(1 for block in blocks if _shown(block) == "Left alpha begins and holds.") == 1


def test_a_caption_stays_beside_its_table_and_a_cell_stays_in_the_table():
    blocks = _ledger().blocks
    table = next(block for block in blocks if block.kind == "table")
    caption = next(block for block in blocks if block.kind == "caption")
    assert caption.reading_order == table.reading_order + 1
    assert all("Table 1-1" not in reading.text for reading in table.readings)
    assert all("speed" not in reading.text for reading in caption.readings)
    cell = [
        block for block in blocks for reading in block.readings if reading.text.strip() == "9.81"
    ]
    assert cell == [table]


def test_a_wide_body_splits_into_columns_instead_of_keeping_interleaved_text():
    wide = EngineRead(
        "layout",
        "m",
        "original",
        "layout",
        proposed((("body", 0, 40, 80, 760, 400, "LEFT right mixed"),)),
    )
    lines = EngineRead(
        "ocr",
        "m",
        "original",
        "ocr",
        proposed(
            (
                ("body", 0, 500, 100, 700, 140, "right one"),
                ("body", 1, 50, 100, 300, 140, "left one"),
                ("body", 2, 500, 200, 700, 240, "right two"),
                ("body", 3, 50, 200, 300, 240, "left two"),
            )
        ),
    )
    texts = [_shown(block) for block in assemble("p0001", 1, WIDTH, HEIGHT, [wide, lines]).blocks]
    assert texts == ["left one", "left two", "right one", "right two"]


def test_a_line_inside_a_table_stays_there_when_its_center_crosses_the_gutter():
    layout = EngineRead(
        "layout",
        "m",
        "original",
        "layout",
        proposed(
            (
                ("body", 0, 40, 20, 200, 80, "Left edge."),
                ("body", 1, 300, 100, 436, 130, "Left prose."),
                ("table", 2, 300, 180, 700, 340, "cell"),
                ("body", 3, 520, 380, 760, 480, "Right prose."),
            )
        ),
    )
    lines = EngineRead(
        "ocr",
        "m",
        "original",
        "ocr",
        proposed(
            (
                ("body", 0, 560, 220, 680, 260, "9.81"),
                ("body", 1, 368, 95, 700, 135, "clips the gutter"),
            )
        ),
    )
    blocks = assemble("p0001", 1, WIDTH, HEIGHT, [layout, lines]).blocks
    table = next(block for block in blocks if block.kind == "table")
    assert any(reading.text == "9.81" for reading in table.readings)
    assert all(
        reading.text != "9.81"
        for block in blocks
        if block.kind == "body"
        for reading in block.readings
    )
    left = next(
        block
        for block in blocks
        if any(reading.text == "Left prose." for reading in block.readings)
    )
    assert all(reading.text != "clips the gutter" for reading in left.readings)
    assert any(
        reading.text == "clips the gutter"
        for block in blocks
        if block is not left
        for reading in block.readings
    )


def test_a_bottom_folio_is_labelled_and_a_boxed_aside_is_not_body():
    layout = EngineRead(
        "layout",
        "m",
        "original",
        "layout",
        proposed((("body", 0, 40, 100, 300, 200, "Paragraph."),)),
    )
    lines = EngineRead(
        "ocr",
        "m",
        "original",
        "ocr",
        proposed(
            (
                ("body", 0, 50, 110, 280, 150, "Paragraph."),
                ("body", 1, 360, 900, 400, 940, "12"),
            )
        ),
    )
    blocks = assemble("p0001", 1, WIDTH, HEIGHT, [layout, lines]).blocks
    folio = [block for block in blocks if block.kind == "page-number"]
    assert [(_shown(block), block.paragraph) for block in folio] == [("12", False)]
    assert all(
        "12" not in reading.text
        for block in blocks
        if block.kind == "body"
        for reading in block.readings
    )
    assert clean_kind_text("text-box", "BOXED") == ("unknown", "BOXED")


def _line(text: str, x: float, y: float, page: int, paragraph: bool) -> RenderBlock:
    return RenderBlock(
        pdf_page=page,
        page_id=f"p{page:04d}",
        ordinal=0,
        reading_order=0,
        kind="body",
        role="body",
        block_id="b",
        digest="",
        text=text,
        region=Region(x0=x, y0=y, x1=x + 200, y1=y + 80, width=WIDTH, height=HEIGHT),
        resolved=None,
        paragraph=paragraph,
    )


def test_a_finished_paragraph_does_not_absorb_the_next_body_block():
    blocks = [
        _line("Left alpha begins and holds.", 40, 90, 1, True),
        _line("Left beta ends with neces-", 40, 120, 1, True),
    ]
    assert [run[0].text for run in paragraph_runs(blocks)] == [
        "Left alpha begins and holds.",
        "Left beta ends with neces-",
    ]


def test_a_hyphen_joins_the_next_page_and_stops_before_a_new_paragraph():
    units = [
        Unit(
            "paragraph",
            (_line("The discussion continua-", 40, 860, 1, True),),
            "The discussion continua-",
            False,
        ),
        Unit(
            "paragraph",
            (_line("tion from the next page.", 40, 80, 2, True),),
            "tion from the next page.",
            False,
        ),
        Unit(
            "paragraph",
            (_line("A fresh paragraph starts.", 40, 200, 2, True),),
            "A fresh paragraph starts.",
            False,
        ),
    ]
    stitched = stitch_columns(units)
    assert [unit.text for unit in stitched] == [
        "The discussion continuation",
        "from the next page.",
        "A fresh paragraph starts.",
    ]
    other_column = replace(units[1], blocks=(_line("tion from the next page.", 500, 80, 2, True),))
    assert [unit.text for unit in stitch_columns([units[0], other_column])] == [
        "The discussion continua-",
        "tion from the next page.",
    ]
