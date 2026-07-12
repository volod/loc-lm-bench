"""MarkItDown markdown extraction (broad-format sanity baseline parser)."""

from pathlib import Path

from llb.prep.pdf.model import MARKITDOWN_TOOL, PdfExtraction, clean_pdf_text


def _extract_with_markitdown(pdf_path: Path) -> PdfExtraction:
    """Extract with MarkItDown as a broad-format sanity baseline."""
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError("missing markitdown dependency") from exc
    try:
        result = MarkItDown().convert(str(pdf_path))
        text = (
            getattr(result, "text_content", None)
            or getattr(result, "markdown", None)
            or str(result)
        )
    except Exception as exc:
        raise RuntimeError(f"markitdown failed for {pdf_path.name}: {exc}") from exc
    return PdfExtraction(text=clean_pdf_text(str(text)), parser=MARKITDOWN_TOOL)
