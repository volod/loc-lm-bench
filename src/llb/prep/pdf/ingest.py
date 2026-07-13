"""The PDF-corpus ingestion orchestrator: convert one PDF into a rendered `.md` doc + citation
sidecar (`_ingest_one_pdf`), and drive the whole directory into a corpus with a manifest
(`ingest_pdf_corpus`), reusing unchanged sources.
"""

import logging
from pathlib import Path

from llb.prep.pdf.dispatch import _normalize_extraction
from llb.prep.pdf.manifest import _write_citations, _write_manifest, _write_quality_report
from llb.prep.pdf.model import (
    PARSER_AUTO,
    PDF_PARSERS,
    PYMUPDF4LLM_TOOL,
    PdfCorpusItem,
    PdfCorpusResult,
    PdfTextExtractor,
    inspect_pdf,
)
from llb.prep.pdf.quality import (
    _extract_with_fallbacks,
    _extraction_quality,
    _short_extraction_reason,
)
from llb.prep.pdf.render import (
    _attempt,
    _render_doc,
    _source_rel,
    default_markdown_out_dir,
    doc_id_for_pdf,
    iter_pdf_files,
)
from llb.prep.pdf.reuse import _previous_manifest_items, _reusable_item, _sha256_file

_LOG = logging.getLogger(__name__)


def _ingest_one_pdf(
    pdf_root: Path,
    pdf_path: Path,
    out_dir: Path,
    extractor: PdfTextExtractor | None,
    min_chars: int,
    parser: str,
    source_sha256: str | None = None,
) -> PdfCorpusItem:
    source = _source_rel(pdf_root, pdf_path)
    diagnostics = inspect_pdf(pdf_path)
    try:
        if extractor is not None:
            extraction = _normalize_extraction(
                extractor(pdf_path, PYMUPDF4LLM_TOOL), PYMUPDF4LLM_TOOL
            )
            quality = _extraction_quality(extraction, diagnostics, min_chars)
            status = "ok" if len(extraction.text) >= min_chars else "too_short"
            attempts = [
                _attempt(
                    extraction.parser,
                    status,
                    len(extraction.text),
                    quality=quality,
                    selected=True,
                )
            ]
        else:
            extraction = _extract_with_fallbacks(pdf_path, parser, min_chars, diagnostics)
            attempts = extraction.attempts
            quality = next(
                (attempt.quality for attempt in attempts if attempt.selected and attempt.quality),
                _extraction_quality(extraction, diagnostics, min_chars),
            )
    except RuntimeError as exc:
        return PdfCorpusItem(
            source=source,
            doc_id=None,
            n_chars=0,
            status="error",
            error=str(exc),
            parser=parser,
            page_count=diagnostics.page_count,
            embedded_text_chars=diagnostics.embedded_text_chars,
            image_only_pages=diagnostics.image_only_pages,
            diagnostics=diagnostics,
            source_sha256=source_sha256,
        )
    if len(extraction.text) < min_chars:
        return PdfCorpusItem(
            source=source,
            doc_id=None,
            n_chars=len(extraction.text),
            status="too_short",
            error=_short_extraction_reason(diagnostics, min_chars, attempts),
            parser=extraction.parser,
            page_count=diagnostics.page_count,
            embedded_text_chars=diagnostics.embedded_text_chars,
            image_only_pages=diagnostics.image_only_pages,
            attempts=attempts,
            diagnostics=diagnostics,
            quality=quality,
            source_sha256=source_sha256,
        )
    doc_id = doc_id_for_pdf(pdf_root, pdf_path)
    rendered = _render_doc(source, extraction)
    (out_dir / doc_id).write_text(rendered.text, encoding="utf-8")
    citation_path = _write_citations(out_dir, source, doc_id, rendered, extraction, diagnostics)
    return PdfCorpusItem(
        source=source,
        doc_id=doc_id,
        n_chars=len(extraction.text),
        status="ok",
        parser=extraction.parser,
        citation_path=citation_path,
        page_count=diagnostics.page_count,
        embedded_text_chars=diagnostics.embedded_text_chars,
        image_only_pages=diagnostics.image_only_pages,
        attempts=attempts,
        diagnostics=diagnostics,
        quality=quality,
        source_sha256=source_sha256,
    )


def ingest_pdf_corpus(
    pdf_root: Path | str,
    out_dir: Path | str | None = None,
    *,
    min_chars: int = 500,
    limit: int | None = None,
    parser: str = PARSER_AUTO,
    extractor: PdfTextExtractor | None = None,
    refresh: bool = False,
) -> PdfCorpusResult:
    """Extract a local PDF directory into a `.md` corpus and write a manifest.

    Unchanged sources are reused from the previous manifest (fingerprinted by sha256) instead of
    reconverted; `refresh=True` forces a full reconversion.
    """
    if parser not in PDF_PARSERS:
        raise ValueError(f"unknown PDF parser: {parser!r}; choose one of {PDF_PARSERS}")
    root = Path(pdf_root)
    if not root.exists():
        raise ValueError(f"PDF root does not exist: {root}")
    pdfs = iter_pdf_files(root)
    if limit is not None:
        pdfs = pdfs[:limit]
    if not pdfs:
        raise ValueError(f"no PDF files under {root}")
    target = Path(out_dir) if out_dir is not None else default_markdown_out_dir(root)
    target.mkdir(parents=True, exist_ok=True)
    previous = {} if refresh else _previous_manifest_items(target)
    items: list[PdfCorpusItem] = []
    n_reused = 0
    for pdf in pdfs:
        source = _source_rel(root, pdf)
        source_sha256 = _sha256_file(pdf)
        reused = _reusable_item(previous.get(source), source_sha256, target, min_chars, parser)
        if reused is not None:
            n_reused += 1
            _LOG.info("[pdf-corpus] reuse %s (unchanged source %s)", reused.doc_id, source)
            items.append(reused)
            continue
        items.append(
            _ingest_one_pdf(
                root, pdf, target, extractor, min_chars, parser, source_sha256=source_sha256
            )
        )
    result = PdfCorpusResult(pdf_root=root, out_dir=target, items=items)
    _write_manifest(result)
    _write_quality_report(result)
    if n_reused:
        _LOG.info(
            "[pdf-corpus] reused %d/%d unchanged conversions (refresh=True forces reconversion)",
            n_reused,
            len(items),
        )
    _LOG.info("[pdf-corpus] extracted %d/%d PDFs into %s", result.n_docs, len(result.items), target)
    return result
