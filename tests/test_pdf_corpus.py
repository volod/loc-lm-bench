"""PDF corpus ingestion: local PDFs -> canonical .md corpus files."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.cli import app
from llb.prep import pdf_corpus as pc

from llb.prep.pdf_corpus import (
    DEFAULT_MARKDOWN_DIRNAME,
    DOCLING_TOOL,
    PARSER_AUTO,
    PDF_CORPUS_MANIFEST,
    PDF_CORPUS_QUALITY,
    PYMUPDF4LLM_TOOL,
    PdfDiagnostics,
    PdfExtraction,
    PdfPageChunk,
    clean_pdf_text,
    default_markdown_out_dir,
    doc_id_for_pdf,
    ingest_pdf_corpus,
    iter_pdf_files,
)

RUNNER = CliRunner()


def test_iter_pdf_files_is_recursive_and_stable(tmp_path: Path) -> None:
    nested = tmp_path / "b"
    nested.mkdir()
    (nested / "two.PDF").write_bytes(b"%PDF")
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "note.txt").write_text("ignore", encoding="utf-8")

    assert [path.relative_to(tmp_path).as_posix() for path in iter_pdf_files(tmp_path)] == [
        "a.pdf",
        "b/two.PDF",
    ]


def test_clean_pdf_text_removes_form_feeds_and_extra_blank_lines() -> None:
    assert clean_pdf_text(" A  \n\n\n\x0c\n B \n") == "A\n\n B"


def test_default_markdown_out_dir_is_md_subdirectory(tmp_path: Path) -> None:
    assert (
        default_markdown_out_dir(tmp_path / "_doc") == tmp_path / "_doc" / DEFAULT_MARKDOWN_DIRNAME
    )


def test_ingest_pdf_corpus_writes_docs_and_manifest(tmp_path: Path) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    source = pdf_root / "Документ.pdf"
    source.write_bytes(b"%PDF")
    out_dir = tmp_path / "corpus"

    def extractor(path: Path, tool: str) -> str:
        assert path == source
        assert tool == PYMUPDF4LLM_TOOL
        return "Український текст документа. " * 30

    result = ingest_pdf_corpus(pdf_root, out_dir, extractor=extractor, min_chars=100)

    doc_id = doc_id_for_pdf(pdf_root, source)
    doc = out_dir / doc_id
    assert result.n_docs == 1 and result.n_skipped == 0
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert text.startswith("# Source PDF: Документ.pdf")
    assert "Український текст документа" in text

    manifest = json.loads((out_dir / PDF_CORPUS_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["n_docs"] == 1
    assert manifest["items"][0]["source"] == "Документ.pdf"
    assert manifest["items"][0]["doc_id"] == doc_id
    assert manifest["items"][0]["parser"] == PYMUPDF4LLM_TOOL

    citation_path = out_dir / manifest["items"][0]["citation_path"]
    assert citation_path.is_file()
    citations = json.loads(citation_path.read_text(encoding="utf-8"))
    assert citations["source"] == "Документ.pdf"
    assert citations["doc_id"] == doc_id

    quality = json.loads((out_dir / PDF_CORPUS_QUALITY).read_text(encoding="utf-8"))
    assert quality["n_docs"] == 1
    assert quality["items"][0]["source"] == "Документ.pdf"


def test_ingest_pdf_corpus_defaults_to_md_subdirectory(tmp_path: Path) -> None:
    pdf_root = tmp_path / "_doc"
    pdf_root.mkdir()
    (pdf_root / "source.pdf").write_bytes(b"%PDF")

    result = ingest_pdf_corpus(
        pdf_root,
        extractor=lambda _path, _tool: "Український текст документа. " * 30,
        min_chars=100,
    )

    assert result.out_dir == pdf_root / "_md"
    assert result.n_docs == 1
    assert (pdf_root / "_md" / PDF_CORPUS_MANIFEST).is_file()


def test_ingest_pdf_corpus_skips_short_extractions(tmp_path: Path) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "short.pdf").write_bytes(b"%PDF")

    result = ingest_pdf_corpus(
        pdf_root,
        tmp_path / "corpus",
        extractor=lambda _path, _tool: "мало",
        min_chars=10,
    )

    assert result.n_docs == 0 and result.n_skipped == 1
    assert result.items[0].status == "too_short"
    assert (tmp_path / "corpus" / PDF_CORPUS_QUALITY).is_file()


def test_ingest_pdf_corpus_writes_page_citation_spans(tmp_path: Path) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    source = pdf_root / "pages.pdf"
    source.write_bytes(b"%PDF")
    out_dir = tmp_path / "corpus"

    def extractor(_path: Path, _tool: str) -> PdfExtraction:
        return PdfExtraction(
            text="Перша сторінка.\n\nДруга сторінка.",
            parser=PYMUPDF4LLM_TOOL,
            pages=[
                PdfPageChunk(page=1, text="Перша сторінка."),
                PdfPageChunk(page=2, text="Друга сторінка."),
            ],
        )

    result = ingest_pdf_corpus(pdf_root, out_dir, extractor=extractor, min_chars=10)

    doc_id = result.items[0].doc_id
    assert doc_id is not None
    doc_text = (out_dir / doc_id).read_text(encoding="utf-8")
    citations = json.loads((out_dir / result.items[0].citation_path).read_text(encoding="utf-8"))
    assert [page["page"] for page in citations["pages"]] == [1, 2]
    first = citations["pages"][0]
    assert doc_text[first["text_start"] : first["text_end"]] == "Перша сторінка."


def test_auto_parser_selects_highest_quality_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    source = pdf_root / "quality.pdf"
    source.write_bytes(b"%PDF")

    monkeypatch.setattr(
        pc,
        "inspect_pdf",
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

    monkeypatch.setattr(pc, "extract_pdf_markdown", fake_extract)

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
        pc,
        "inspect_pdf",
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
        pc,
        "inspect_pdf",
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
        pc,
        "extract_pdf_markdown",
        lambda _path, tool: PdfExtraction(text="", parser=tool),
    )

    result = ingest_pdf_corpus(pdf_root, tmp_path / "corpus", parser=PARSER_AUTO, min_chars=10)

    assert result.items[0].status == "too_short"
    assert [attempt.parser for attempt in result.items[0].attempts] == [DOCLING_TOOL]


def test_ingest_pdf_corpus_rejects_empty_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no PDF files"):
        ingest_pdf_corpus(tmp_path, tmp_path / "corpus")


def test_pdf_to_markdown_cli_defaults_out_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_root = tmp_path / "_doc"
    pdf_root.mkdir()
    seen: dict[str, object] = {}

    def fake_ingest_pdf_corpus(
        pdf_root_arg: Path,
        out_dir_arg: Path | None,
        *,
        min_chars: int,
        parser: str,
        limit: int | None,
    ) -> pc.PdfCorpusResult:
        seen["pdf_root"] = pdf_root_arg
        seen["out_dir"] = out_dir_arg
        seen["min_chars"] = min_chars
        seen["parser"] = parser
        seen["limit"] = limit
        return pc.PdfCorpusResult(
            pdf_root=pdf_root_arg,
            out_dir=pc.default_markdown_out_dir(pdf_root_arg),
            items=[pc.PdfCorpusItem(source="a.pdf", doc_id="pdf-a.md", n_chars=500, status="ok")],
        )

    monkeypatch.setattr(pc, "ingest_pdf_corpus", fake_ingest_pdf_corpus)

    result = RUNNER.invoke(app, ["pdf-to-markdown", str(pdf_root), "--limit", "1"])

    assert result.exit_code == 0
    assert seen == {
        "pdf_root": pdf_root,
        "out_dir": None,
        "min_chars": 500,
        "parser": "auto",
        "limit": 1,
    }
    assert f"-> {pdf_root / '_md'}" in result.output
