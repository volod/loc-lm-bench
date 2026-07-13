"""Score a PDF extraction's quality and, in `auto` mode, select the best parser.

`_extraction_quality` blends character count, page/citation coverage, heading and table density, and
a per-parser priority into one comparable score; `_extract_with_fallbacks` runs the auto-candidate
parsers for the PDF's kind (text vs image-only) and keeps the highest-scoring result.
"""

from pathlib import Path

from llb.prep.pdf.dispatch import extract_pdf_markdown
from llb.prep.pdf.model import (
    PARSER_AUTO,
    PARSER_QUALITY_PRIORITY,
    PDF_AUTO_IMAGE_CANDIDATES,
    PDF_AUTO_TEXT_CANDIDATES,
    QUALITY_CHAR_DIVISOR,
    QUALITY_CITATION_COVERAGE_WEIGHT,
    QUALITY_HEADING_WEIGHT,
    QUALITY_MAX_CHAR_SCORE,
    QUALITY_MAX_HEADING_SCORE,
    QUALITY_MAX_TABLE_SCORE,
    QUALITY_PAGE_COVERAGE_WEIGHT,
    QUALITY_SHORT_PENALTY,
    QUALITY_TABLE_WEIGHT,
    _MARKDOWN_HEADING,
    _is_image_only_pdf,
    PdfDiagnostics,
    PdfExtraction,
    PdfExtractionQuality,
    PdfParserAttempt,
)
from llb.prep.pdf.render import _attempt


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


def _extract_with_fallbacks(
    pdf_path: Path, parser: str, min_chars: int, diagnostics: PdfDiagnostics
) -> PdfExtraction:
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
