"""Unstructured `partition_pdf` markdown extraction (optional hi-res parser)."""

from pathlib import Path
from typing import Any

from llb.prep.pdf.model import (
    UNSTRUCTURED_TOOL,
    PdfExtraction,
    PdfPageChunk,
    clean_pdf_text,
)


def _extract_with_unstructured(pdf_path: Path) -> PdfExtraction:
    """Extract with Unstructured partition_pdf when the optional dependency is installed."""
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        raise RuntimeError("missing unstructured dependency") from exc
    try:
        kwargs: dict[str, Any] = {
            "filename": str(pdf_path),
            "strategy": "hi_res",
            "infer_table_structure": True,
            "include_page_breaks": True,
            "languages": ["ukr", "eng"],
        }
        try:
            elements = partition_pdf(**kwargs)
        except TypeError:
            kwargs.pop("languages", None)
            elements = partition_pdf(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"unstructured failed for {pdf_path.name}: {exc}") from exc

    by_page: dict[int, list[str]] = {}
    for element in elements:
        metadata = getattr(element, "metadata", None)
        page_number = getattr(metadata, "page_number", None) if metadata is not None else None
        page = int(page_number or 1)
        text_as_html = getattr(metadata, "text_as_html", None) if metadata is not None else None
        text = text_as_html or str(element)
        if text.strip():
            by_page.setdefault(page, []).append(text)
    pages = [
        PdfPageChunk(page=page, text=clean_pdf_text("\n\n".join(parts)))
        for page, parts in sorted(by_page.items())
    ]
    return PdfExtraction(
        text=clean_pdf_text("\n\n".join(page.text for page in pages)),
        parser=UNSTRUCTURED_TOOL,
        pages=pages,
    )
