"""Parser dispatch for the PDF corpus extractors."""

from pathlib import Path

from llb.prep.pdf.docling import _extract_with_docling
from llb.prep.pdf.marker import _extract_with_marker
from llb.prep.pdf.markitdown import _extract_with_markitdown
from llb.prep.pdf.model import (
    DOCLING_TOOL,
    MARKER_TOOL,
    MARKITDOWN_TOOL,
    PDF_PARSERS,
    PYMUPDF4LLM_TOOL,
    UNSTRUCTURED_TOOL,
    PdfExtraction,
    clean_pdf_text,
)
from llb.prep.pdf.pymupdf import _extract_with_pymupdf4llm
from llb.prep.pdf.unstructured import _extract_with_unstructured


def extract_pdf_markdown(pdf_path: Path, parser: str = PYMUPDF4LLM_TOOL) -> PdfExtraction:
    """Extract markdown from one PDF with a concrete parser."""
    if parser == PYMUPDF4LLM_TOOL:
        return _extract_with_pymupdf4llm(pdf_path)
    if parser == DOCLING_TOOL:
        return _extract_with_docling(pdf_path)
    if parser == MARKER_TOOL:
        return _extract_with_marker(pdf_path)
    if parser == UNSTRUCTURED_TOOL:
        return _extract_with_unstructured(pdf_path)
    if parser == MARKITDOWN_TOOL:
        return _extract_with_markitdown(pdf_path)
    raise RuntimeError(f"unknown PDF parser: {parser!r}; choose one of {PDF_PARSERS}")


def _normalize_extraction(value: str | PdfExtraction, parser: str) -> PdfExtraction:
    if isinstance(value, PdfExtraction):
        return value
    return PdfExtraction(text=clean_pdf_text(value), parser=parser)
