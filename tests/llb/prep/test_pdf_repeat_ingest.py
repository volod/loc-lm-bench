"""Focused tests split from ``test_pdf_repeats.py``."""

import json
from pathlib import Path

from _pdf_repeats_helpers import (
    FIXTURE,
    RUNNER,
)

from llb.main import app
from llb.prep.pdf.ingest import ingest_pdf_corpus
from llb.prep.pdf.model import (
    PDF_CITATION_SUFFIX,
    PdfExtraction,
    PdfPageChunk,
)
from llb.prep.pdf.repeats import (
    REPEAT_DROP,
    REPEAT_KEEP,
)


def test_conversion_applies_the_selected_repeat_mode_and_keeps_citations_exact(
    tmp_path: Path,
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "manual.pdf").write_bytes(b"%PDF")
    body = "Порядок збереження документа: натисніть кнопку Зберегти.\n\nУнікальний текст сторінки"

    def extractor(_path: Path, _tool: str) -> PdfExtraction:
        return PdfExtraction(
            text=body,
            parser="pymupdf4llm",
            pages=[PdfPageChunk(page=page, text=f"{body} {page}.") for page in (1, 2, 3)],
        )

    result = ingest_pdf_corpus(
        pdf_root, tmp_path / "md", extractor=extractor, min_chars=10, repeat_blocks=REPEAT_DROP
    )

    item = result.items[0]
    assert item.repeat_blocks == REPEAT_DROP
    assert item.doc_id is not None
    doc_text = (tmp_path / "md" / item.doc_id).read_text(encoding="utf-8")
    assert doc_text.count("Порядок збереження документа") == 1
    citations = json.loads(
        (tmp_path / "md" / f"{Path(item.doc_id).stem}{PDF_CITATION_SUFFIX}").read_text(
            encoding="utf-8"
        )
    )
    for page in citations["pages"]:
        assert doc_text[page["text_start"] : page["text_end"]].strip()


def test_conversion_reuse_is_keyed_on_the_repeat_mode(tmp_path: Path) -> None:
    """Switching the mode must reconvert: the previous output is a different rendering."""
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "manual.pdf").write_bytes(b"%PDF")
    calls: list[Path] = []

    def extractor(path: Path, _tool: str) -> str:
        calls.append(path)
        return "Український текст документа. " * 30

    out = tmp_path / "md"
    ingest_pdf_corpus(pdf_root, out, extractor=extractor, min_chars=100)
    ingest_pdf_corpus(pdf_root, out, extractor=extractor, min_chars=100)
    assert len(calls) == 1  # unchanged source and mode: reused

    ingest_pdf_corpus(pdf_root, out, extractor=extractor, min_chars=100, repeat_blocks=REPEAT_DROP)
    assert len(calls) == 2


def test_cli_censuses_without_touching_the_corpus(tmp_path: Path) -> None:
    report = tmp_path / "census.json"

    result = RUNNER.invoke(
        app,
        ["strip-corpus-repeats", "--corpus", str(FIXTURE), "--report", str(report)],
    )

    assert result.exit_code == 0, result.output
    assert "repeated block groups: 2" in result.output
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == REPEAT_KEEP and payload["out_root"] is None


def test_cli_refuses_a_rewriting_mode_without_an_output_root() -> None:
    result = RUNNER.invoke(
        app, ["strip-corpus-repeats", "--corpus", str(FIXTURE), "--mode", REPEAT_DROP]
    )

    assert result.exit_code == 2
    assert "--out" in result.output
