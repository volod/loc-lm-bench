"""PDF corpus ingestion: local PDFs -> canonical .md corpus files."""

import json
from pathlib import Path

import pytest

from llb.prep.pdf_corpus import (
    PDF_CORPUS_MANIFEST,
    clean_pdf_text,
    doc_id_for_pdf,
    ingest_pdf_corpus,
    iter_pdf_files,
)


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


def test_ingest_pdf_corpus_writes_docs_and_manifest(tmp_path: Path) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    source = pdf_root / "Документ.pdf"
    source.write_bytes(b"%PDF")
    out_dir = tmp_path / "corpus"

    def extractor(path: Path, _tool: str) -> str:
        assert path == source
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
