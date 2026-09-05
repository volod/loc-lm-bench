"""Write the PDF-corpus artifacts: the conversion manifest, the quality report, and the per-doc
page-citation sidecars.

The manifest doubles as the reuse fingerprint source (see `reuse.py`); the citation sidecar records
each rendered page's char offsets so span validation can cite the originating PDF page.
"""

import json
from pathlib import Path
from typing import Literal

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.data_prep.corpus import (
    PdfCitations,
    PdfCorpusManifest,
    PdfItemRecord,
    PdfPageCitationRecord,
)

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

PdfManifestKind = Literal["pdf-corpus", "pdf-corpus-quality"]


def _pdf_items(result: PdfCorpusResult) -> list[PdfItemRecord]:
    return [PdfItemRecord.model_validate(asdict(item)) for item in result.items]


def _manifest(result: PdfCorpusResult, kind: PdfManifestKind) -> PdfCorpusManifest:
    """The conversion manifest and its quality report, which differ only in `kind`."""
    return PdfCorpusManifest(
        schema_id="llb.pdf-corpus-manifest",
        schema_version="1.0.0",
        kind=kind,
        pdf_root=str(result.pdf_root),
        corpus_root=str(result.out_dir),
        n_pdfs=len(result.items),
        n_docs=result.n_docs,
        n_skipped=result.n_skipped,
        items=_pdf_items(result),
    )


def _write_contract(path: Path, record: ArtifactContract) -> None:
    path.write_text(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_manifest(result: PdfCorpusResult) -> None:
    _write_contract(result.out_dir / PDF_CORPUS_MANIFEST, _manifest(result, "pdf-corpus"))


def _write_quality_report(result: PdfCorpusResult) -> None:
    _write_contract(result.out_dir / PDF_CORPUS_QUALITY, _manifest(result, "pdf-corpus-quality"))


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
    record = PdfCitations(
        schema_id="llb.pdf-citations",
        schema_version="1.0.0",
        kind="pdf-citations",
        source=source,
        doc_id=doc_id,
        parser=extraction.parser,
        diagnostics=asdict(diagnostics),
        pages=[PdfPageCitationRecord.model_validate(asdict(page)) for page in rendered.citations],
    )
    _write_contract(out_dir / rel, record)
    return rel
