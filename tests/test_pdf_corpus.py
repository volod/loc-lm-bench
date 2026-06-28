"""PDF corpus ingestion: local PDFs -> canonical .md corpus files."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.cli import app
from llb.prep import pdf_corpus as pc

from llb.prep.pdf_corpus import (
    DEFAULT_MARKDOWN_DIRNAME,
    PDF_CORPUS_MANIFEST,
    PYMUPDF4LLM_TOOL,
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
        limit: int | None,
    ) -> pc.PdfCorpusResult:
        seen["pdf_root"] = pdf_root_arg
        seen["out_dir"] = out_dir_arg
        seen["min_chars"] = min_chars
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
        "limit": 1,
    }
    assert f"-> {pdf_root / '_md'}" in result.output
