"""PDF discovery and corpus rendering: enumerate PDFs in stable order, assign stable corpus ids,
and render one extraction into a `.md` document with per-page citation offsets.

`_render_doc` strips page furniture before laying out the page markers, so the citation offsets it
records point at the cleaned, contiguous body text the rest of the pipeline grounds against.
"""

import hashlib
from pathlib import Path
from typing import Any

from llb.prep.pdf.furniture import strip_page_furniture
from llb.prep.pdf.model import (
    DEFAULT_MARKDOWN_DIRNAME,
    PDF_SUFFIX,
    PdfExtraction,
    PdfExtractionQuality,
    PdfPageCitation,
    PdfParserAttempt,
    RenderedPdfDoc,
)


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
