"""Tests for pdf corpus cache."""

import json
from pathlib import Path
import pytest
from llb.cli import app
from llb.prep.pdf import ingest as pdf_ingest
from llb.prep.pdf.ingest import ingest_pdf_corpus
from llb.prep.pdf.model import (
    PDF_CORPUS_MANIFEST,
    PdfCorpusItem,
    PdfCorpusResult,
)
from llb.prep.pdf.render import default_markdown_out_dir
from test_pdf_corpus import RUNNER, _counting_extractor


def test_ingest_pdf_corpus_rejects_empty_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no PDF files"):
        ingest_pdf_corpus(tmp_path, tmp_path / "corpus")


def test_ingest_pdf_corpus_reuses_unchanged_sources(tmp_path: Path) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    source = pdf_root / "doc.pdf"
    source.write_bytes(b"%PDF one")
    out_dir = tmp_path / "corpus"
    calls: list[Path] = []

    first = ingest_pdf_corpus(
        pdf_root, out_dir, extractor=_counting_extractor(calls), min_chars=100
    )
    assert len(calls) == 1
    assert first.items[0].source_sha256
    assert not first.items[0].reused

    second = ingest_pdf_corpus(
        pdf_root, out_dir, extractor=_counting_extractor(calls), min_chars=100
    )
    assert len(calls) == 1  # unchanged source: no reconversion
    assert second.n_docs == 1
    assert second.items[0].reused
    assert second.items[0].source_sha256 == first.items[0].source_sha256
    manifest = json.loads((out_dir / PDF_CORPUS_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["items"][0]["source_sha256"] == first.items[0].source_sha256

    source.write_bytes(b"%PDF two")  # changed source: reconverted
    third = ingest_pdf_corpus(
        pdf_root, out_dir, extractor=_counting_extractor(calls), min_chars=100
    )
    assert len(calls) == 2
    assert not third.items[0].reused


def test_ingest_pdf_corpus_refresh_forces_reconversion(tmp_path: Path) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "doc.pdf").write_bytes(b"%PDF")
    out_dir = tmp_path / "corpus"
    calls: list[Path] = []

    ingest_pdf_corpus(pdf_root, out_dir, extractor=_counting_extractor(calls), min_chars=100)
    result = ingest_pdf_corpus(
        pdf_root, out_dir, extractor=_counting_extractor(calls), min_chars=100, refresh=True
    )

    assert len(calls) == 2
    assert not result.items[0].reused


def test_ingest_pdf_corpus_does_not_reuse_when_output_is_missing(tmp_path: Path) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    source = pdf_root / "doc.pdf"
    source.write_bytes(b"%PDF")
    out_dir = tmp_path / "corpus"
    calls: list[Path] = []

    first = ingest_pdf_corpus(
        pdf_root, out_dir, extractor=_counting_extractor(calls), min_chars=100
    )
    doc_id = first.items[0].doc_id
    assert doc_id is not None
    (out_dir / doc_id).unlink()

    second = ingest_pdf_corpus(
        pdf_root, out_dir, extractor=_counting_extractor(calls), min_chars=100
    )

    assert len(calls) == 2
    assert not second.items[0].reused
    assert (out_dir / doc_id).is_file()


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
        refresh: bool,
    ) -> PdfCorpusResult:
        seen["pdf_root"] = pdf_root_arg
        seen["out_dir"] = out_dir_arg
        seen["min_chars"] = min_chars
        seen["parser"] = parser
        seen["limit"] = limit
        seen["refresh"] = refresh
        return PdfCorpusResult(
            pdf_root=pdf_root_arg,
            out_dir=default_markdown_out_dir(pdf_root_arg),
            items=[PdfCorpusItem(source="a.pdf", doc_id="pdf-a.md", n_chars=500, status="ok")],
        )

    monkeypatch.setattr(pdf_ingest, "ingest_pdf_corpus", fake_ingest_pdf_corpus)

    result = RUNNER.invoke(app, ["pdf-to-markdown", str(pdf_root), "--limit", "1"])

    assert result.exit_code == 0
    assert seen == {
        "pdf_root": pdf_root,
        "out_dir": None,
        "min_chars": 500,
        "parser": "auto",
        "limit": 1,
        "refresh": False,
    }
    assert f"-> {pdf_root / '_md'}" in result.output
