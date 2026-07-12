"""PyMuPDF4LLM markdown extraction (the default text-PDF parser)."""

from pathlib import Path
from typing import Any

from llb.prep.pdf.model import (
    PYMUPDF4LLM_TOOL,
    PdfExtraction,
    PdfPageChunk,
    _is_image_only_pdf,
    clean_pdf_text,
    inspect_pdf,
)


def _extract_with_pymupdf4llm(pdf_path: Path) -> PdfExtraction:
    """Extract markdown from one PDF with PyMuPDF4LLM, preserving page chunks when available."""
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise RuntimeError("missing pymupdf4llm dependency") from exc
    try:
        use_ocr = _is_image_only_pdf(inspect_pdf(pdf_path))
        chunks = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            use_ocr=use_ocr,
            force_ocr=use_ocr,
            ocr_language="ukr+eng",
        )
    except Exception as exc:
        raise RuntimeError(f"pymupdf4llm failed for {pdf_path.name}: {exc}") from exc
    if isinstance(chunks, list):
        pages = [
            _pymupdf_page_chunk(idx, chunk)
            for idx, chunk in enumerate(chunks)
            if isinstance(chunk, dict)
        ]
        text = clean_pdf_text("\n\n".join(page.text for page in pages if page.text))
        return PdfExtraction(text=text, parser=PYMUPDF4LLM_TOOL, pages=pages)

    if isinstance(chunks, str):
        return PdfExtraction(text=clean_pdf_text(chunks), parser=PYMUPDF4LLM_TOOL)
    raise RuntimeError(f"pymupdf4llm returned unsupported markdown for {pdf_path.name}")


def _pymupdf_page_chunk(idx: int, chunk: dict[str, Any]) -> PdfPageChunk:
    """One `PdfPageChunk` from a pymupdf4llm page dict (falls back to positional numbering)."""
    metadata_obj = chunk.get("metadata")
    metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
    page = int(metadata.get("page_number") or idx + 1)
    page_text = clean_pdf_text(str(chunk.get("text") or ""))
    return PdfPageChunk(page=page, text=page_text, blocks=_page_blocks(chunk.get("page_boxes")))


def _page_blocks(page_boxes: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if not isinstance(page_boxes, list):
        return blocks
    for box in page_boxes:
        if not isinstance(box, dict):
            continue
        pos = box.get("pos")
        start: int | None = None
        end: int | None = None
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            start = int(pos[0])
            end = int(pos[1])
        bbox_obj = box.get("bbox")
        bbox = list(bbox_obj) if isinstance(bbox_obj, (tuple, list)) else None
        blocks.append(
            {
                "class": str(box.get("class") or "unknown"),
                "bbox": bbox,
                "page_char_start": start,
                "page_char_end": end,
            }
        )
    return blocks
