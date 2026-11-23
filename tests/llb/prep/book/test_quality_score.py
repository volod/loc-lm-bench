"""The fixture page, the existing-lane report, and the score command."""

import json
from pathlib import Path

from typer.testing import CliRunner

from llb.cli import app
from llb.core.contracts.data_prep.book_blocks import Block, PageBlockLedger, RawReading, Region
from llb.core.contracts.data_prep.corpus import PdfCitations, PdfPageCitationRecord
from llb.core.paths import PROJECT_ROOT
from llb.prep.book.constants import BLOCK_LEDGER_SCHEMA_ID, CONTRACT_VERSION
from llb.prep.book.ledger.io import write_ledger
from llb.prep.book.metrics.constants import STATUS_ABSENT, STATUS_SCORED
from llb.prep.book.metrics.geometry import Box
from llb.prep.book.metrics.page_score import score_page
from llb.prep.book.metrics.reading import page_reading, reference_reading
from llb.prep.book.metrics.reference import RefPage, load_reference
from llb.prep.book.metrics.report import PageScore, ScoreReport
from llb.prep.book.metrics.score import score_reference
from llb.prep.book.metrics.submission import HypBlock, PageSubmission
from llb.prep.book.metrics.tables import Cell

_FIXTURE = PROJECT_ROOT / "tests/llb/prep/book/fixtures/metrics/reference"
_RUNNER = CliRunner()


def test_the_fixture_page_scores_perfect() -> None:
    page = load_reference(_FIXTURE)[0]
    perfect = score_page(page, _mirror(page))
    assert perfect.character_error_rate.value == 0.0
    assert perfect.word_error_rate.value == 0.0
    assert perfect.missing_words == 0
    assert perfect.duplicated_words == 0
    assert perfect.reading_order_agreement.value == 1.0
    assert sum(perfect.dangerous_substitutions.values()) == 0
    assert perfect.formula_fidelity.value == 1.0
    assert perfect.formula_parse_valid is True
    assert perfect.table_similarity.value == 1.0
    assert perfect.figure_overlap.value == 1.0
    assert perfect.caption_association.value == 1.0
    assert perfect.printed_page_accuracy.value == 1.0


def test_the_existing_lane_reports_structure_absent(tmp_path: Path) -> None:
    page = load_reference(_FIXTURE)[0]
    citations = _write_citations(tmp_path, page)
    report = score_reference(_FIXTURE, citations=citations)
    lane = report.lanes[0]
    assert lane.lane == "existing-pdf"
    page = lane.pages[0]
    assert page.character_error_rate.value == 0.0
    assert page.reading_order_agreement.value == 1.0
    assert page.printed_page_accuracy.value == 1.0
    assert page.formula_fidelity.status == STATUS_SCORED
    assert page.formula_fidelity.value == 1.0
    assert page.table_similarity.status == STATUS_ABSENT
    assert page.figure_overlap.status == STATUS_ABSENT
    assert page.caption_association.status == STATUS_ABSENT
    assert page.table_similarity.value is None
    assert "score" not in ScoreReport.model_fields
    assert "headline" not in PageScore.model_fields
    rolled = lane.by_page_class["normal"]
    assert rolled.dangerous_substitution_rates["5/S"] == 0.0
    assert rolled.character_error_rate.value == 0.0


def test_score_command_reads_a_ledger_and_refuses_a_broken_reference(tmp_path: Path) -> None:
    pages = load_reference(_FIXTURE)
    run = tmp_path / "run"
    _write_ledger(run, pages[0])
    result = _RUNNER.invoke(
        app, ["book-ocr", "score", str(run), "--reference", str(_FIXTURE), "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["lanes"][0]["lane"] == "book-ocr/fake"
    assert "score" not in payload
    broken = tmp_path / "broken" / "pages"
    broken.mkdir(parents=True)
    bad = json.loads((_FIXTURE / "pages" / "p0001.json").read_text(encoding="utf-8"))
    bad["blocks"][3]["caption_id"] = "missing"
    (broken / "p0001.json").write_text(json.dumps(bad), encoding="utf-8")
    refused = _RUNNER.invoke(
        app, ["book-ocr", "score", str(run), "--reference", str(broken.parent), "--json"]
    )
    assert refused.exit_code == 2
    assert "caption_id" in refused.output


def _mirror(page: RefPage) -> PageSubmission:
    blocks = []
    for block in page.blocks:
        table = None
        if block.table is not None:
            table = tuple(
                tuple(Cell(cell.text, cell.colspan, cell.rowspan) for cell in row)
                for row in block.table.rows
            )
        region = block.region
        blocks.append(
            HypBlock(
                block.block_id,
                block.kind,
                Box(region.x0, region.y0, region.x1, region.y1),
                block.reading_order,
                block.text,
                block.latex,
                table,
                block.caption_id,
                block.equation_number,
            )
        )
    return PageSubmission("book-ocr/fake", page.pdf_page, tuple(blocks), False)


def _write_citations(tmp_path: Path, page: RefPage) -> Path:
    body = page_reading((block.reading_order, reference_reading(block)) for block in page.blocks)
    markdown = tmp_path / "page.md"
    markdown.write_text(body, encoding="utf-8")
    record = PdfCitations(
        schema_id="llb.pdf-citations",
        schema_version="1.0.0",
        kind="pdf-citations",
        source="scan.pdf",
        doc_id="page.md",
        parser="pymupdf4llm",
        diagnostics={},
        pages=[
            PdfPageCitationRecord(
                page=page.pdf_page,
                char_start=0,
                char_end=len(body),
                text_start=0,
                text_end=len(body),
                n_chars=len(body),
                parser="pymupdf4llm",
            )
        ],
    )
    path = tmp_path / "page.citations.json"
    path.write_text(record.model_dump_json(), encoding="utf-8")
    return path


def _write_ledger(run: Path, page: RefPage) -> None:
    blocks = []
    for block in page.blocks:
        text = block.latex or block.text
        if block.table is not None:
            lines = [
                "| " + " | ".join(cell.text for cell in row) + " |" for row in block.table.rows
            ]
            text = "\n".join(lines)
        region = block.region
        blocks.append(
            Block(
                block_id=block.block_id,
                pdf_page=page.pdf_page,
                ordinal=block.reading_order,
                kind=block.kind,
                region=Region(
                    x0=region.x0,
                    y0=region.y0,
                    x1=region.x1,
                    y1=region.y1,
                    width=page.width,
                    height=page.height,
                ),
                reading_order=block.reading_order,
                readings=[RawReading(engine="fake", variant="original", text=text, model="m")],
            )
        )
    ledger = PageBlockLedger(
        schema_id=BLOCK_LEDGER_SCHEMA_ID,
        schema_version=CONTRACT_VERSION,
        kind="book-ocr-block-ledger",
        page_id="p0001",
        pdf_page=page.pdf_page,
        blocks=blocks,
    )
    write_ledger(run / "pages" / "p0001" / "blocks.json", ledger)
