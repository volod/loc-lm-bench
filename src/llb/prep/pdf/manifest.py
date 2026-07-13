"""Write the PDF-corpus artifacts: the conversion manifest, the quality report, and the per-doc
page-citation sidecars.

The manifest doubles as the reuse fingerprint source (see `reuse.py`); the citation sidecar records
each rendered page's char offsets so span validation can cite the originating PDF page.
"""

import json
from pathlib import Path

from llb.prep.pdf.model import (
    PDF_CITATION_SUFFIX,
    PDF_CORPUS_MANIFEST,
    PDF_CORPUS_QUALITY,
    PdfCorpusResult,
    PdfDiagnostics,
    PdfExtraction,
    RenderedPdfDoc,
)
from dataclasses import asdict


def _manifest(result: PdfCorpusResult) -> dict[str, object]:
    return {
        "kind": "pdf-corpus",
        "pdf_root": str(result.pdf_root),
        "corpus_root": str(result.out_dir),
        "n_pdfs": len(result.items),
        "n_docs": result.n_docs,
        "n_skipped": result.n_skipped,
        "items": [asdict(item) for item in result.items],
    }


def _quality_report(result: PdfCorpusResult) -> dict[str, object]:
    return {
        "kind": "pdf-corpus-quality",
        "pdf_root": str(result.pdf_root),
        "corpus_root": str(result.out_dir),
        "n_pdfs": len(result.items),
        "n_docs": result.n_docs,
        "n_skipped": result.n_skipped,
        "items": [asdict(item) for item in result.items],
    }


def _write_manifest(result: PdfCorpusResult) -> None:
    (result.out_dir / PDF_CORPUS_MANIFEST).write_text(
        json.dumps(_manifest(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_quality_report(result: PdfCorpusResult) -> None:
    (result.out_dir / PDF_CORPUS_QUALITY).write_text(
        json.dumps(_quality_report(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _citation_path(doc_id: str) -> str:
    return Path(doc_id).with_suffix(PDF_CITATION_SUFFIX).name


def _write_citations(
    out_dir: Path,
    source: str,
    doc_id: str,
    rendered: RenderedPdfDoc,
    extraction: PdfExtraction,
    diagnostics: PdfDiagnostics,
) -> str:
    rel = _citation_path(doc_id)
    payload = {
        "kind": "pdf-citations",
        "source": source,
        "doc_id": doc_id,
        "parser": extraction.parser,
        "diagnostics": asdict(diagnostics),
        "pages": [asdict(page) for page in rendered.citations],
    }
    (out_dir / rel).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return rel
