"""Printed pages, the hierarchy, and the numbering inventory."""

import json
from pathlib import Path

from typer.testing import CliRunner

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.cli import app
from llb.core.contracts.data_prep.book import BookOcrRunConfig, BookSourceRecord, BookStageSpec
from llb.core.contracts.data_prep.book_blocks import Block, PageBlockLedger, RawReading, Region
from llb.core.paths import PROJECT_ROOT
from llb.prep.book.config import write_config
from llb.prep.book.ledger.io import ledger_path, write_ledger
from llb.prep.book.ranges import page_id
from llb.prep.book.structure.build import build_structure, infer_structure
from llb.prep.book.structure.io import read_structure, structure_path
from llb.prep.book.structure.lines import Line

_W = 1000
_H = 1400
_SHA = "a" * 64
_SCHEMA = (
    PROJECT_ROOT / "schemas" / "artifacts" / "llb.book-ocr-document-structure" / "1.0.0.schema.json"
)
RUNNER = CliRunner()


class _Pages:
    """Synthetic lines. `kind` is stored so a test can show it is not consulted."""

    def __init__(self) -> None:
        self.lines: list[Line] = []
        self._count: dict[int, int] = {}

    def add(
        self,
        pdf: int,
        text: str,
        x: float,
        y: float,
        kind: str = "body",
        w: float = 240,
        h: float = 24,
    ) -> None:
        index = self._count.get(pdf, 0)
        self._count[pdf] = index + 1
        self.lines.append(
            Line(
                block_id=f"p{pdf:04d}-b{index:04d}",
                pdf_page=pdf,
                text=text,
                kind=kind,
                x0=x,
                y0=y,
                x1=x + w,
                y1=y + h,
                page_width=_W,
                page_height=_H,
            )
        )

    def infer(self, pages: list[int] | None = None):
        chosen = pages or sorted({line.pdf_page for line in self.lines})
        return infer_structure(list(self.lines), chosen, max(chosen), _SHA)


def _handbook() -> _Pages:
    book = _Pages()
    book.add(1, "Table of Contents", 80, 70, w=300)
    book.add(1, "Part I Mission Design ...... 1", 80, 160, w=420)
    book.add(1, "1 Introduction ...... 1", 80, 220, w=420)
    book.add(1, "1.1 Purpose ...... 2", 100, 280, w=400)
    book.add(1, "1.2 Scope ...... 3", 100, 340, w=400)
    book.add(1, "2 Orbits ...... 4", 80, 400, w=420)
    book.add(1, "2.1 Kepler ...... 4", 100, 460, w=400)
    for pdf, printed, header in (
        (2, "1", "Mission Design"),
        (3, "2", "Mission Design"),
        (4, "3", "Mission Design"),
    ):
        book.add(pdf, header, 80, 40, kind="running-header", w=280)
        book.add(pdf, printed, 900, 40, kind="page-number", w=40, h=20)
    book.add(2, "Part I Mission Design", 80, 180, kind="heading", w=360)
    book.add(2, "1 Introduction", 80, 250, kind="heading", w=280)
    book.add(2, "The introduction explains the scope of the work.", 80, 340, w=700)
    book.add(3, "1.1 Purpose", 80, 180, kind="heading")
    book.add(3, "Purpose is stated here in ordinary prose.", 80, 260, w=700)
    book.add(4, "1.2 Scope", 80, 180, kind="heading")
    book.add(4, "Scope is stated here in ordinary prose.", 80, 260, w=700)
    book.add(5, "Orbits", 80, 40, kind="running-header", w=200)
    book.add(5, "4", 900, 40, kind="page-number", w=40, h=20)
    book.add(5, "2 Orbits", 80, 180, kind="heading")
    book.add(5, "2.1 Kepler", 80, 250, kind="heading")
    book.add(5, "Figure 2-1 A sample orbit.", 80, 360, w=400)
    book.add(5, "Table 2-1 Elements.", 80, 430, w=400)
    book.add(5, "(2-1)", 860, 500, w=80)
    book.add(5, "Table 2-1, Fig. 2-2, Eq. 2-3", 80, 1320, w=420)
    return book


def _walk(node) -> list[tuple[str, str, str, int, int]]:
    rows = [(node.level, node.number, node.title, node.pdf_start, node.pdf_end)]
    for child in node.children:
        rows.extend(_walk(child))
    return rows


def _mapping(structure) -> list[tuple[int, str | None, str]]:
    return [(page.pdf_page, page.printed, page.status) for page in structure.pages]


def test_a_contents_book_reconstructs_exactly():
    structure = _handbook().infer()
    assert _walk(structure.hierarchy) == [
        ("book", "", "Book", 1, 5),
        ("part", "I", "Mission Design", 2, 5),
        ("chapter", "1", "Introduction", 2, 4),
        ("section", "1.1", "Purpose", 3, 3),
        ("section", "1.2", "Scope", 4, 4),
        ("chapter", "2", "Orbits", 5, 5),
        ("section", "2.1", "Kepler", 5, 5),
    ]
    assert _mapping(structure) == [
        (1, None, "front-matter"),
        (2, "1", "observed"),
        (3, "2", "observed"),
        (4, "3", "observed"),
        (5, "4", "observed"),
    ]
    assert structure.conflicts == []
    assert structure.contents_pages == [1]
    assert [(item.text, item.pdf_start, item.pdf_end) for item in structure.headers] == [
        ("Mission Design", 2, 4),
        ("Orbits", 5, 5),
    ]
    part = structure.hierarchy.children[0]
    assert "contents" in part.evidence and "running-header" in part.evidence
    numbers = {(item.kind, item.number, item.evidence) for item in structure.inventory}
    assert numbers == {
        ("figure", "2-1", "caption"),
        ("table", "2-1", "caption"),
        ("equation", "2-1", "display-number"),
        ("figure", "2-2", "page-footer"),
        ("equation", "2-3", "page-footer"),
    }


def test_relabeling_every_block_body_keeps_pages_and_headers():
    labeled = _handbook()
    labeled_structure = labeled.infer()
    body = _Pages()
    body.lines = [
        Line(
            line.block_id,
            line.pdf_page,
            line.text,
            "body",
            line.x0,
            line.y0,
            line.x1,
            line.y1,
            line.page_width,
            line.page_height,
        )
        for line in labeled.lines
    ]
    again = body.infer()
    assert _mapping(again) == _mapping(labeled_structure)
    assert [(item.text, item.pdf_start, item.pdf_end) for item in again.headers] == [
        (item.text, item.pdf_start, item.pdf_end) for item in labeled_structure.headers
    ]


def test_a_missing_page_number_is_a_gap_in_the_offset():
    book = _Pages()
    for pdf, printed in ((1, "10"), (2, "11"), (4, "13"), (5, "14")):
        book.add(pdf, printed, 900, 40, w=40, h=20)
        book.add(pdf, "Ordinary prose on this page.", 80, 200, w=600)
    book.add(3, "Ordinary prose on the page whose number was not read.", 80, 200, w=600)
    structure = book.infer()
    assert _mapping(structure) == [
        (1, "10", "observed"),
        (2, "11", "observed"),
        (3, "12", "gap"),
        (4, "13", "observed"),
        (5, "14", "observed"),
    ]
    assert structure.pages[2].fallback is True
    assert structure.pages[2].block_id is None


def test_an_unnumbered_plate_does_not_advance_the_printed_page():
    book = _Pages()
    for pdf, printed in ((1, "10"), (2, "11"), (4, "12")):
        book.add(pdf, printed, 900, 40, w=40, h=20)
    book.add(3, "A full-page plate.", 80, 400, w=600)
    structure = book.infer()
    assert _mapping(structure) == [
        (1, "10", "observed"),
        (2, "11", "observed"),
        (3, None, "unnumbered"),
        (4, "12", "observed"),
    ]
    assert structure.pages[2].fallback is False


def test_a_contents_entry_with_no_heading_is_reported_and_not_a_node():
    book = _handbook()
    book.add(1, "3 Missing ...... 9", 80, 520, w=420)
    structure = book.infer()
    assert [row[1] for row in _walk(structure.hierarchy) if row[0] == "chapter"] == ["1", "2"]
    missing = [item for item in structure.conflicts if item.kind == "contents-without-heading"]
    assert len(missing) == 1
    assert missing[0].number == "3"
    assert missing[0].contents_title == "Missing"
    assert all(node.number != "3" for node in _flat(structure.hierarchy))


def test_no_contents_and_no_numbered_headings_fall_back_to_page_ranges():
    book = _Pages()
    for pdf in (1, 2, 3, 4):
        book.add(pdf, "Notes", 80, 40, w=160)
        book.add(pdf, "Prose with no number and no contents.", 80, 200, w=640)
    structure = book.infer()
    assert structure.hierarchy.fallback is True
    assert [child.level for child in structure.hierarchy.children] == ["page-range"]
    child = structure.hierarchy.children[0]
    assert child.fallback is True
    assert child.title == "Pages 1-4: Notes"
    assert [page.status for page in structure.pages] == ["unnumbered"] * 4
    assert all(page.fallback for page in structure.pages)


def test_a_running_header_section_number_that_disagrees_is_a_conflict():
    book = _Pages()
    for pdf in (1, 2):
        book.add(pdf, "9.9", 80, 40, w=60)
        book.add(pdf, str(pdf), 900, 40, w=40, h=20)
        book.add(pdf, "Body prose.", 80, 240, w=400)
    book.add(1, "1.1 Purpose", 80, 160)
    structure = book.infer()
    assert any(
        item.kind == "header-mismatch" and item.number == "9.9" for item in structure.conflicts
    )


def test_a_split_contents_row_keeps_its_number_and_its_page():
    book = _Pages()
    book.add(1, "Table of Contents", 80, 70, w=300)
    book.add(1, "1.1", 100, 200, w=40)
    book.add(1, "Purpose", 180, 240, w=200)
    book.add(1, "...... 2", 700, 240, w=120)
    structure = book.infer()
    assert structure.contents_pages == [1]
    # The entry is evidence only. With no heading, it is a conflict, not a node.
    conflict = structure.conflicts[0]
    assert conflict.kind == "contents-without-heading"
    assert conflict.number == "1.1"
    assert conflict.contents_title == "Purpose"
    assert conflict.printed == "2"
    assert structure.hierarchy.children[0].level == "page-range"


def test_a_scanned_page_does_not_turn_numbers_and_tall_lines_into_chapters():
    book = _Pages()
    book.add(1, "12", 900, 133, w=40, h=24)
    book.add(1, "Ordinary prose on this page.", 80, 300, w=700)
    book.add(1, "106", 620, 500, w=40, h=24)
    book.add(1, "Units", 760, 500, w=80, h=24)
    book.add(1, "29.785 km", 820, 700, w=100, h=24)
    book.add(2, "2", 80, 200, w=40, h=28)
    book.add(2, "Orbits", 80, 240, w=200, h=28)
    book.add(2, "The orbit is described here.", 80, 320, w=700)
    for index, title in enumerate(("Alpha", "Beta", "Gamma")):
        book.add(3, title, 80, 160 + index * 40, w=300, h=70)
        book.add(3, "A line of ordinary prose.", 80, 500 + index * 40, w=700, h=24)
    structure = book.infer()
    assert (structure.pages[0].printed, structure.pages[0].status) == ("12", "observed")
    chapters = [node for node in _flat(structure.hierarchy) if node.level == "chapter"]
    assert [(node.number, node.title) for node in chapters] == [("2", "Orbits")]


def test_a_table_cell_and_a_page_hole_do_not_become_structure():
    book = _Pages()
    book.add(1, "1 Introduction", 80, 180, w=280)
    book.add(1, "6.6 m", 400, 400, w=40)
    book.add(1, "1.5", 80, 500, w=40)
    book.add(1, "Sources of more information", 80, 540, w=360)
    book.add(10, "10.7 Design of orbits", 80, 180, w=360)
    structure = book.infer([1, 10])
    nodes = [node for node in _flat(structure.hierarchy) if node.level != "book"]
    assert [(node.number, node.pdf_start, node.pdf_end) for node in nodes] == [
        ("1", 1, 1),
        ("10.7", 10, 10),
    ]
    assert structure.hierarchy.children[1].level == "section"


def test_a_list_of_tables_is_not_inventoried_as_captions():
    book = _Pages()
    for index in range(8):
        book.add(1, f"Table {index + 1}-1 Row.", 80, 200 + index * 30, w=400)
    book.add(2, "Table 9-1 A real caption.", 80, 300, w=400)
    structure = book.infer()
    tables = [item for item in structure.inventory if item.kind == "table"]
    assert [(item.number, item.pdf_page) for item in tables] == [("9-1", 2)]


def test_structure_round_trips_and_matches_its_schema():
    structure = _handbook().infer()
    again = type(structure).model_validate_json(structure.model_dump_json())
    assert again.model_dump(mode="json") == structure.model_dump(mode="json")
    payload = structure.model_dump(mode="json")
    assert DEFAULT_REGISTRY.read_current(payload).model_dump(mode="json") == payload
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    from jsonschema.validators import validator_for

    validator_for(schema)(schema).validate(payload)


def test_build_writes_the_artifact_and_a_second_build_matches(tmp_path: Path):
    book = _handbook()
    run = _write_run(tmp_path, book.lines)
    first = build_structure(run)
    raw = structure_path(run).read_bytes()
    second = build_structure(run)
    assert structure_path(run).read_bytes() == raw
    assert read_structure(structure_path(run)).model_dump() == first.model_dump()
    assert second.model_dump() == first.model_dump()
    result = RUNNER.invoke(app, ["book-ocr", "structure", str(run)])
    assert result.exit_code == 0, result.output
    assert "5 pages, 4 numbered, 0 gaps" in result.output
    assert "conflicts 0" in result.output


def test_a_missing_ledger_is_refused(tmp_path: Path):
    book = _handbook()
    kept = [line for line in book.lines if line.pdf_page != 2]
    run = _write_run(tmp_path, kept, pages=[1, 2, 3, 4, 5])
    result = RUNNER.invoke(app, ["book-ocr", "structure", str(run)])
    assert result.exit_code == 2
    assert "p0002" in result.output


def _flat(node):
    yield node
    for child in node.children:
        yield from _flat(child)


def _write_run(tmp_path: Path, lines: list[Line], pages: list[int] | None = None) -> Path:
    configured = pages or sorted({line.pdf_page for line in lines})
    run = tmp_path / "run"
    write_config(run, _config(configured))
    by_page: dict[int, list[Line]] = {}
    for line in lines:
        by_page.setdefault(line.pdf_page, []).append(line)
    for pdf, group in by_page.items():
        write_ledger(ledger_path(run, page_id(pdf, 5)), _ledger(pdf, group))
    return run


def _config(pages: list[int]) -> BookOcrRunConfig:
    return BookOcrRunConfig(
        schema_id="llb.book-ocr-run-config",
        schema_version="1.0.0",
        kind="book-ocr-run-config",
        run_id="structure",
        created_at="2026-11-23T00:00:00+00:00",
        source=BookSourceRecord(
            original_path="scan.pdf",
            sha256=_SHA,
            size_bytes=1,
            page_count=5,
            policy="reference",
            stored_path="scan.pdf",
        ),
        pages_spec="1-5",
        pages=pages,
        batch_pages=25,
        max_attempts=2,
        stages=[BookStageSpec(name="blocks")],
    )


def _ledger(pdf: int, lines: list[Line]) -> PageBlockLedger:
    blocks = []
    for ordinal, line in enumerate(lines):
        blocks.append(
            Block(
                block_id=line.block_id,
                pdf_page=pdf,
                ordinal=ordinal,
                kind="body",
                region=Region(
                    x0=line.x0,
                    y0=line.y0,
                    x1=line.x1,
                    y1=line.y1,
                    width=_W,
                    height=_H,
                ),
                reading_order=ordinal,
                readings=[RawReading(engine="fake", variant="original", text=line.text)],
            )
        )
    return PageBlockLedger(
        schema_id="llb.book-ocr-block-ledger",
        schema_version="1.0.0",
        kind="book-ocr-block-ledger",
        page_id=page_id(pdf, 5),
        pdf_page=pdf,
        blocks=blocks,
    )
