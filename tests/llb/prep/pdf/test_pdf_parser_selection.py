"""Tests for pdf parser selection."""

from pathlib import Path
import pytest
from llb.prep.pdf.ingest import ingest_pdf_corpus
from llb.prep.pdf.model import (
    DOCLING_TOOL,
    PARSER_AUTO,
    PYMUPDF4LLM_TOOL,
    PdfDiagnostics,
    PdfExtraction,
    PdfPageChunk,
)


def test_auto_parser_selects_highest_quality_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    source = pdf_root / "quality.pdf"
    source.write_bytes(b"%PDF")

    monkeypatch.setattr(
        "llb.prep.pdf.ingest.inspect_pdf",
        lambda _path: PdfDiagnostics(
            page_count=2,
            encrypted=False,
            needs_password=False,
            embedded_text_chars=1000,
            image_pages=0,
            image_only_pages=0,
        ),
    )

    def fake_extract(_path: Path, tool: str) -> PdfExtraction:
        if tool == PYMUPDF4LLM_TOOL:
            return PdfExtraction(
                text="Сторінкова цитата. " * 80,
                parser=PYMUPDF4LLM_TOOL,
                pages=[
                    PdfPageChunk(page=1, text="Сторінкова цитата. " * 40),
                    PdfPageChunk(page=2, text="Сторінкова цитата. " * 40),
                ],
            )
        raise RuntimeError(f"missing {tool}")

    monkeypatch.setattr("llb.prep.pdf.quality.extract_pdf_markdown", fake_extract)

    result = ingest_pdf_corpus(pdf_root, tmp_path / "corpus", parser=PARSER_AUTO, min_chars=10)

    item = result.items[0]
    assert item.status == "ok"
    assert item.parser == PYMUPDF4LLM_TOOL
    assert item.quality is not None
    assert item.quality.citation_coverage == 1.0
    selected = [attempt.parser for attempt in item.attempts if attempt.selected]
    assert selected == [PYMUPDF4LLM_TOOL]
    assert {attempt.parser for attempt in item.attempts} == {PYMUPDF4LLM_TOOL}


def test_short_image_only_pdf_reports_clean_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    source = pdf_root / "scan.pdf"
    source.write_bytes(b"%PDF")

    monkeypatch.setattr(
        "llb.prep.pdf.ingest.inspect_pdf",
        lambda _path: PdfDiagnostics(
            page_count=2,
            encrypted=False,
            needs_password=False,
            embedded_text_chars=0,
            image_pages=2,
            image_only_pages=2,
        ),
    )

    result = ingest_pdf_corpus(
        pdf_root,
        tmp_path / "corpus",
        parser=PARSER_AUTO,
        min_chars=10,
        extractor=lambda _path, _tool: "",
    )

    item = result.items[0]
    assert item.status == "too_short"
    assert "image-only PDF" in (item.error or "")
    assert item.page_count == 2
    assert item.image_only_pages == 2


def test_image_only_auto_attempts_ocr_layout_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    source = pdf_root / "scan.pdf"
    source.write_bytes(b"%PDF")

    monkeypatch.setattr(
        "llb.prep.pdf.ingest.inspect_pdf",
        lambda _path: PdfDiagnostics(
            page_count=2,
            encrypted=False,
            needs_password=False,
            embedded_text_chars=0,
            image_pages=2,
            image_only_pages=2,
        ),
    )
    monkeypatch.setattr(
        "llb.prep.pdf.quality.extract_pdf_markdown",
        lambda _path, tool: PdfExtraction(text="", parser=tool),
    )

    result = ingest_pdf_corpus(pdf_root, tmp_path / "corpus", parser=PARSER_AUTO, min_chars=10)

    assert result.items[0].status == "too_short"
    assert [attempt.parser for attempt in result.items[0].attempts] == [DOCLING_TOOL]
