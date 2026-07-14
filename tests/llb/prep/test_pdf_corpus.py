"""PDF corpus ingestion: local PDFs -> canonical .md corpus files."""

import json
from pathlib import Path

from typer.testing import CliRunner

from llb.prep.pdf.furniture import strip_page_furniture
from llb.prep.pdf.ingest import ingest_pdf_corpus
from llb.prep.pdf.model import (
    PDF_CORPUS_MANIFEST,
    PDF_CORPUS_QUALITY,
    PYMUPDF4LLM_TOOL,
    PdfExtraction,
    PdfPageChunk,
    clean_pdf_text,
)
from llb.prep.pdf.render import doc_id_for_pdf, iter_pdf_files

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


def test_strip_page_furniture_drops_running_headers_footers_and_noise() -> None:
    # a running header + footer + page number wrap each page's unique body
    pages = [
        f"MANUAL v1\n\n<!-- c -->\nBody paragraph {i} unique text.\n**==> picture <==**\nConf.\n**{i}**"
        for i in range(1, 13)
    ]
    cleaned = strip_page_furniture(pages)
    assert len(cleaned) == len(pages)
    joined = "\n".join(cleaned)
    # repeated furniture and per-page noise are gone
    assert "MANUAL v1" not in joined
    assert "Conf." not in joined
    assert "picture" not in joined and "<!--" not in joined
    assert not any(line.strip() in {str(i) for i in range(1, 13)} for line in joined.split("\n"))
    # unique body content survives on every page
    assert all(f"Body paragraph {i} unique text." in cleaned[i - 1] for i in range(1, 13))


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


def _counting_extractor(calls: list[Path]):
    def extractor(path: Path, _tool: str) -> str:
        calls.append(path)
        return "Український текст документа. " * 30

    return extractor
