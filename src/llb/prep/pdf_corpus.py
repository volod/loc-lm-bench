"""PDF corpus ingestion for local Ukrainian document collections.

The rest of the RAG/goldset pipeline consumes `.md` and `.txt` files with stable character
offsets. This module turns a local PDF directory into that canonical text corpus using
per-parser extractors. The extracted `.md` files become the source of truth for later span
validation; original PDFs are recorded only as provenance in the manifest.

The extraction backends live in the `llb.prep.pdf` package -- the shared data model in
`pdf.model`, one module per parser (`pdf.pymupdf`, `pdf.docling`, `pdf.marker`,
`pdf.unstructured`, `pdf.markitdown`), and the dispatcher in `pdf.dispatch`. This module keeps the
orchestration (parser selection, page-furniture stripping, corpus rendering, quality scoring,
manifest/citation I/O, and unchanged-source reuse) and re-exports the package's public names so
`llb.prep.pdf_corpus.<name>` keeps working.
"""

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from llb.prep.pdf.dispatch import _normalize_extraction, extract_pdf_markdown
from llb.prep.pdf.model import (
    _FURNITURE_DECORATION,
    _FURNITURE_MAX_LEN,
    _FURNITURE_REPEAT_FRACTION,
    _HTML_COMMENT_BLOCK,
    _MANY_BLANK_LINES,
    _MARKDOWN_HEADING,
    _PAGE_NUMBER_LINE,
    _PICTURE_PLACEHOLDER,
    DEFAULT_MARKDOWN_DIRNAME,
    DOCLING_TOOL,
    MARKER_TOOL,
    MARKITDOWN_TOOL,
    PARSER_AUTO,
    PARSER_QUALITY_PRIORITY,
    PDF_AUTO_IMAGE_CANDIDATES,
    PDF_AUTO_TEXT_CANDIDATES,
    PDF_CITATION_SUFFIX,
    PDF_CORPUS_MANIFEST,
    PDF_CORPUS_QUALITY,
    PDF_PARSERS,
    PDF_SUFFIX,
    PYMUPDF4LLM_TOOL,
    QUALITY_CHAR_DIVISOR,
    QUALITY_CITATION_COVERAGE_WEIGHT,
    QUALITY_HEADING_WEIGHT,
    QUALITY_MAX_CHAR_SCORE,
    QUALITY_MAX_HEADING_SCORE,
    QUALITY_MAX_TABLE_SCORE,
    QUALITY_PAGE_COVERAGE_WEIGHT,
    QUALITY_SHORT_PENALTY,
    QUALITY_TABLE_WEIGHT,
    SHA256_READ_CHUNK_BYTES,
    UNSTRUCTURED_TOOL,
    PdfCorpusItem,
    PdfCorpusResult,
    PdfDiagnostics,
    PdfExtraction,
    PdfExtractionQuality,
    PdfPageChunk,
    PdfPageCitation,
    PdfParserAttempt,
    PdfTextExtractor,
    RenderedPdfDoc,
    _is_image_only_pdf,
    clean_pdf_text,
    inspect_pdf,
)

_LOG = logging.getLogger(__name__)

__all__ = [
    # data model + constants (re-exported from pdf.model)
    "DEFAULT_MARKDOWN_DIRNAME",
    "DOCLING_TOOL",
    "MARKER_TOOL",
    "MARKITDOWN_TOOL",
    "PARSER_AUTO",
    "PDF_CITATION_SUFFIX",
    "PDF_CORPUS_MANIFEST",
    "PDF_CORPUS_QUALITY",
    "PDF_PARSERS",
    "PDF_SUFFIX",
    "PYMUPDF4LLM_TOOL",
    "UNSTRUCTURED_TOOL",
    "PdfCorpusItem",
    "PdfCorpusResult",
    "PdfDiagnostics",
    "PdfExtraction",
    "PdfExtractionQuality",
    "PdfPageChunk",
    "PdfPageCitation",
    "PdfParserAttempt",
    "PdfTextExtractor",
    "RenderedPdfDoc",
    "clean_pdf_text",
    "inspect_pdf",
    # extraction dispatch (re-exported from pdf.dispatch)
    "extract_pdf_markdown",
    # public orchestration
    "default_markdown_out_dir",
    "doc_id_for_pdf",
    "ingest_pdf_corpus",
    "iter_pdf_files",
    "strip_page_furniture",
]


# --- page-furniture stripping (running headers/footers, page numbers, image/comment noise) --------


def _furniture_key(line: str) -> str:
    """Decoration-insensitive key for detecting a line that recurs as a running header/footer."""
    return re.sub(r"\s+", " ", _FURNITURE_DECORATION.sub(" ", line)).strip().casefold()


def strip_page_furniture(page_texts: list[str]) -> list[str]:
    """Drop PDF page furniture so a passage that crosses a page break still grounds contiguously.

    Removes lines that recur across a large fraction of pages (running headers/footers), standalone
    page-number lines, image placeholders, and HTML comments. Repetition is measured across the
    WHOLE document, so this must see every page at once. Body content is preserved: only short,
    frequently repeating lines qualify. Returns one cleaned string per input page (possibly empty).
    """
    counts = _furniture_line_counts(page_texts)
    threshold = max(8, int(_FURNITURE_REPEAT_FRACTION * max(len(page_texts), 1)))
    return [_strip_page(text, counts, threshold) for text in page_texts]


def _furniture_line_counts(page_texts: list[str]) -> Counter[str]:
    """How often each normalized line recurs across the whole document."""
    counts: Counter[str] = Counter()
    for text in page_texts:
        for line in text.split("\n"):
            key = _furniture_key(line)
            if key:
                counts[key] += 1
    return counts


def _is_furniture_line(line: str, counts: Counter[str], threshold: int) -> bool:
    """A short line repeating on many pages (running header/footer) or a bare page number."""
    stripped = line.strip()
    key = _furniture_key(line)
    if key and len(stripped) <= _FURNITURE_MAX_LEN and counts[key] >= threshold:
        return True
    return bool(_PAGE_NUMBER_LINE.match(line))


def _strip_page(text: str, counts: Counter[str], threshold: int) -> str:
    """One page with furniture lines, placeholders, and comment blocks removed."""
    kept = [
        line
        for line in text.split("\n")
        if not line.strip() or not _is_furniture_line(line, counts, threshold)
    ]
    body = _HTML_COMMENT_BLOCK.sub(" ", _PICTURE_PLACEHOLDER.sub(" ", "\n".join(kept)))
    return _MANY_BLANK_LINES.sub("\n\n", body).strip()


# --- PDF discovery ------------------------------------------------------------------------


def iter_pdf_files(pdf_root: Path | str) -> list[Path]:
    """Return PDFs under `pdf_root` in stable corpus-relative order."""
    root = Path(pdf_root)
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == PDF_SUFFIX),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def doc_id_for_pdf(pdf_root: Path | str, pdf_path: Path | str) -> str:
    """Stable ASCII corpus id for a PDF path."""
    root = Path(pdf_root)
    rel = Path(pdf_path).relative_to(root).as_posix()
    digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:12]
    return f"pdf-{digest}.md"


def default_markdown_out_dir(pdf_root: Path | str) -> Path:
    """Return the default markdown output directory for a PDF corpus root."""
    return Path(pdf_root) / DEFAULT_MARKDOWN_DIRNAME


def _attempt(
    parser: str,
    status: str,
    n_chars: int = 0,
    error: str | None = None,
    quality: PdfExtractionQuality | None = None,
    selected: bool = False,
) -> PdfParserAttempt:
    return PdfParserAttempt(
        parser=parser,
        status=status,
        n_chars=n_chars,
        error=error,
        quality=quality,
        selected=selected,
    )


# --- corpus rendering ---------------------------------------------------------------------


def _source_rel(pdf_root: Path, pdf_path: Path) -> str:
    return pdf_path.relative_to(pdf_root).as_posix()


def _render_doc(source: str, extraction: PdfExtraction) -> RenderedPdfDoc:
    header = f"# Source PDF: {source}\n\n"
    parts = [header]
    citations: list[PdfPageCitation] = []

    pages = [page for page in extraction.pages if page.text.strip()]
    if not pages:
        text = f"{header}{extraction.text}\n"
        return RenderedPdfDoc(text=text, citations=citations)

    cleaned_texts = strip_page_furniture([page.text for page in pages])
    for page, cleaned in zip(pages, cleaned_texts):
        body = cleaned.strip()
        if not body:
            continue  # page was entirely furniture (e.g. a cover/running-header-only page)
        # block offsets are relative to the original page text; furniture removal shifts them.
        blocks = page.blocks if cleaned == page.text else []
        page_marker = (
            f"<!-- source_pdf: {source} page: {page.page} parser: {extraction.parser} -->\n\n"
        )
        page_start = sum(len(part) for part in parts)
        text_start = page_start + len(page_marker)
        page_text = f"{body}\n\n"
        text_end = text_start + len(body)
        page_end = page_start + len(page_marker) + len(page_text)
        parts.extend([page_marker, page_text])
        citations.append(
            PdfPageCitation(
                page=page.page,
                char_start=page_start,
                char_end=page_end,
                text_start=text_start,
                text_end=text_end,
                n_chars=len(body),
                parser=extraction.parser,
                blocks=_offset_blocks(blocks, text_start),
            )
        )
    return RenderedPdfDoc(text="".join(parts).rstrip() + "\n", citations=citations)


def _offset_blocks(blocks: list[dict[str, Any]], text_start: int) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for block in blocks:
        item = dict(block)
        page_start = item.get("page_char_start")
        page_end = item.get("page_char_end")
        if isinstance(page_start, int):
            item["char_start"] = text_start + page_start
        if isinstance(page_end, int):
            item["char_end"] = text_start + page_end
        shifted.append(item)
    return shifted


# --- manifest, quality report, citation sidecars ------------------------------------------


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


# --- quality scoring + parser selection ---------------------------------------------------


def _extraction_quality(
    extraction: PdfExtraction,
    diagnostics: PdfDiagnostics,
    min_chars: int,
) -> PdfExtractionQuality:
    n_chars = len(extraction.text)
    page_count = diagnostics.page_count
    page_text_pages = sum(1 for page in extraction.pages if page.text.strip())
    citation_pages = page_text_pages
    page_coverage = _coverage(page_text_pages, page_count)
    citation_coverage = _coverage(citation_pages, page_count)
    heading_count = len(_MARKDOWN_HEADING.findall(extraction.text))
    table_marker_count = extraction.text.casefold().count("<table") + extraction.text.count("|")
    score = _quality_score(
        extraction.parser,
        n_chars,
        page_coverage,
        citation_coverage,
        heading_count,
        table_marker_count,
        min_chars,
    )
    return PdfExtractionQuality(
        n_chars=n_chars,
        page_count=page_count,
        page_text_pages=page_text_pages,
        page_coverage=page_coverage,
        citation_pages=citation_pages,
        citation_coverage=citation_coverage,
        heading_count=heading_count,
        table_marker_count=table_marker_count,
        score=score,
    )


def _coverage(numerator: int, denominator: int | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return min(1.0, numerator / denominator)


def _quality_score(
    parser: str,
    n_chars: int,
    page_coverage: float | None,
    citation_coverage: float | None,
    heading_count: int,
    table_marker_count: int,
    min_chars: int,
) -> float:
    char_score = min(QUALITY_MAX_CHAR_SCORE, n_chars / QUALITY_CHAR_DIVISOR)
    page_score = (page_coverage or 0.0) * QUALITY_PAGE_COVERAGE_WEIGHT
    citation_score = (citation_coverage or 0.0) * QUALITY_CITATION_COVERAGE_WEIGHT
    heading_score = min(QUALITY_MAX_HEADING_SCORE, heading_count * QUALITY_HEADING_WEIGHT)
    table_score = min(QUALITY_MAX_TABLE_SCORE, table_marker_count * QUALITY_TABLE_WEIGHT)
    short_penalty = QUALITY_SHORT_PENALTY if n_chars < min_chars else 0.0
    return (
        char_score
        + page_score
        + citation_score
        + heading_score
        + table_score
        + PARSER_QUALITY_PRIORITY.get(parser, 0.0)
        - short_penalty
    )


def _mark_selected(
    attempts: list[PdfParserAttempt],
    selected_parser: str | None,
) -> list[PdfParserAttempt]:
    return [
        _attempt(
            attempt.parser,
            attempt.status,
            attempt.n_chars,
            attempt.error,
            attempt.quality,
            selected=attempt.parser == selected_parser,
        )
        for attempt in attempts
    ]


def _auto_candidates_for_pdf(diagnostics: PdfDiagnostics) -> tuple[str, ...]:
    if _is_image_only_pdf(diagnostics):
        return PDF_AUTO_IMAGE_CANDIDATES
    return PDF_AUTO_TEXT_CANDIDATES


def _extract_with_fallbacks(pdf_path: Path, parser: str, min_chars: int) -> PdfExtraction:
    diagnostics = inspect_pdf(pdf_path)
    if parser != PARSER_AUTO:
        extraction = extract_pdf_markdown(pdf_path, parser)
        quality = _extraction_quality(extraction, diagnostics, min_chars)
        status = "ok" if len(extraction.text) >= min_chars else "too_short"
        return PdfExtraction(
            text=extraction.text,
            parser=extraction.parser,
            pages=extraction.pages,
            attempts=[
                _attempt(
                    extraction.parser,
                    status,
                    len(extraction.text),
                    quality=quality,
                    selected=True,
                )
            ],
        )

    attempts: list[PdfParserAttempt] = []
    candidates: list[tuple[PdfExtraction, PdfExtractionQuality]] = []
    short_candidates: list[tuple[PdfExtraction, PdfExtractionQuality]] = []
    for candidate in _auto_candidates_for_pdf(diagnostics):
        try:
            extraction = extract_pdf_markdown(pdf_path, candidate)
        except RuntimeError as exc:
            attempts.append(_attempt(candidate, "error", error=str(exc)))
            continue
        n_chars = len(extraction.text)
        quality = _extraction_quality(extraction, diagnostics, min_chars)
        if n_chars >= min_chars:
            attempts.append(_attempt(candidate, "ok", n_chars, quality=quality))
            candidates.append((extraction, quality))
        else:
            attempts.append(_attempt(candidate, "too_short", n_chars, quality=quality))
            short_candidates.append((extraction, quality))

    if candidates:
        selected, _quality = max(candidates, key=lambda item: item[1].score)
        return PdfExtraction(
            text=selected.text,
            parser=selected.parser,
            pages=selected.pages,
            attempts=_mark_selected(attempts, selected.parser),
        )
    if short_candidates:
        selected, _quality = max(short_candidates, key=lambda item: item[1].score)
        return PdfExtraction(
            text=selected.text,
            parser=selected.parser,
            pages=selected.pages,
            attempts=_mark_selected(attempts, selected.parser),
        )
    raise RuntimeError(
        "all PDF parsers failed: " + "; ".join(a.error or a.status for a in attempts)
    )


def _short_extraction_reason(
    diagnostics: PdfDiagnostics,
    min_chars: int,
    attempts: list[PdfParserAttempt],
) -> str:
    if diagnostics.needs_password:
        return "PDF is encrypted or password-protected"
    if _is_image_only_pdf(diagnostics):
        errors = [
            f"{attempt.parser}={attempt.error}"
            for attempt in attempts
            if attempt.status == "error" and attempt.error
        ]
        suffix = f"; fallback errors: {'; '.join(errors)}" if errors else ""
        return (
            "image-only PDF with zero embedded text; OCR/layout fallback did not recover "
            f"{min_chars} chars{suffix}"
        )
    failed = [f"{a.parser}:{a.status}" for a in attempts]
    return f"extracted text shorter than {min_chars} chars; attempts={', '.join(failed)}"


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
            extraction = _extract_with_fallbacks(pdf_path, parser, min_chars)
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


# --- unchanged-source reuse ---------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(SHA256_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _previous_manifest_items(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the previous manifest of `out_dir` as `source -> item payload` (empty when absent)."""
    path = out_dir / PDF_CORPUS_MANIFEST
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items = payload.get("items") if isinstance(payload, dict) else None
    previous: dict[str, dict[str, Any]] = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get("source"), str):
            previous[item["source"]] = item
    return previous


def _dataclass_kwargs(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {key: value for key, value in payload.items() if key in names}


def _nested_from_payload(cls: type, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    return cls(**_dataclass_kwargs(cls, payload))


def _attempt_from_payload(payload: dict[str, Any]) -> PdfParserAttempt:
    data = dict(payload)
    data["quality"] = _nested_from_payload(PdfExtractionQuality, payload.get("quality"))
    return PdfParserAttempt(**_dataclass_kwargs(PdfParserAttempt, data))


def _item_from_payload(payload: dict[str, Any]) -> PdfCorpusItem:
    data = dict(payload)
    data["attempts"] = [
        _attempt_from_payload(attempt)
        for attempt in (payload.get("attempts") or [])
        if isinstance(attempt, dict)
    ]
    data["diagnostics"] = _nested_from_payload(PdfDiagnostics, payload.get("diagnostics"))
    data["quality"] = _nested_from_payload(PdfExtractionQuality, payload.get("quality"))
    data["reused"] = True
    return PdfCorpusItem(**_dataclass_kwargs(PdfCorpusItem, data))


def _reusable_item(
    payload: dict[str, Any] | None,
    source_sha256: str,
    out_dir: Path,
    min_chars: int,
    parser: str,
) -> PdfCorpusItem | None:
    """Rehydrate a previous ok item when the source and conversion request still match it."""
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return None
    if payload.get("source_sha256") != source_sha256:
        return None
    doc_id = payload.get("doc_id")
    citation_path = payload.get("citation_path")
    selected_parser = payload.get("parser")
    n_chars = payload.get("n_chars")
    if not doc_id or not citation_path or not selected_parser:
        return None
    if parser != PARSER_AUTO and selected_parser != parser:
        return None
    if not isinstance(n_chars, int) or n_chars < min_chars:
        return None
    if not (out_dir / doc_id).is_file() or not (out_dir / citation_path).is_file():
        return None
    try:
        return _item_from_payload(payload)
    except (TypeError, ValueError):
        return None


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
