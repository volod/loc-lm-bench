"""Shared data model + cheap helpers for the PDF corpus extractors (leaf of the pdf package).

Constants, the extraction/citation dataclasses, markdown normalization (`clean_pdf_text`), and the
cheap PyMuPDF diagnostics (`inspect_pdf`) that the per-parser extractor modules and the parent
the `llb.prep.pdf` orchestration modules both build on. Depends on nothing else in the package, so it
carries no import cycle.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PDF_CORPUS_MANIFEST = "pdf_corpus_manifest.json"
PDF_CORPUS_QUALITY = "pdf_corpus_quality.json"
PDF_CITATION_SUFFIX = ".citations.json"
PDF_SUFFIX = ".pdf"
DEFAULT_MARKDOWN_DIRNAME = "_md"
PARSER_AUTO = "auto"
PYMUPDF4LLM_TOOL = "pymupdf4llm"
DOCLING_TOOL = "docling"
MARKER_TOOL = "marker"
UNSTRUCTURED_TOOL = "unstructured"
MARKITDOWN_TOOL = "markitdown"
PDF_PARSERS = (
    PARSER_AUTO,
    PYMUPDF4LLM_TOOL,
    DOCLING_TOOL,
    MARKER_TOOL,
    UNSTRUCTURED_TOOL,
    MARKITDOWN_TOOL,
)
PDF_AUTO_IMAGE_CANDIDATES = (DOCLING_TOOL,)
PDF_AUTO_TEXT_CANDIDATES = (PYMUPDF4LLM_TOOL,)
QUALITY_CHAR_DIVISOR = 1000.0
QUALITY_MAX_CHAR_SCORE = 1200.0
QUALITY_PAGE_COVERAGE_WEIGHT = 500.0
QUALITY_CITATION_COVERAGE_WEIGHT = 800.0
QUALITY_HEADING_WEIGHT = 2.0
QUALITY_MAX_HEADING_SCORE = 200.0
QUALITY_TABLE_WEIGHT = 1.0
QUALITY_MAX_TABLE_SCORE = 200.0
QUALITY_SHORT_PENALTY = 10000.0
SHA256_READ_CHUNK_BYTES = 1 << 20  # stream source PDFs in 1 MiB chunks when fingerprinting
PARSER_QUALITY_PRIORITY = {
    MARKER_TOOL: 50.0,
    DOCLING_TOOL: 45.0,
    UNSTRUCTURED_TOOL: 30.0,
    PYMUPDF4LLM_TOOL: 25.0,
    MARKITDOWN_TOOL: 10.0,
}
_MANY_BLANK_LINES = re.compile(r"\n{3,}")
_MARKER_PAGE_ID = re.compile(r"/page/(\d+)(?:/|$)")
_MARKDOWN_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_HTML_TAG = re.compile(r"<[^>]+>")

# --- page-furniture stripping (running headers/footers, page numbers, image/comment noise) --------
_PICTURE_PLACEHOLDER = re.compile(r"\*\*==>.*?<==\*\*", re.DOTALL)  # extractor image stubs
_HTML_COMMENT_BLOCK = re.compile(r"<!--.*?-->", re.DOTALL)
_PAGE_NUMBER_LINE = re.compile(r"^\s*\**\s*\d{1,4}\s*\**\s*$")  # a bold-or-bare page number, alone
_FURNITURE_DECORATION = re.compile(r"[*#>_|♆•]")
_FURNITURE_MAX_LEN = 90  # running headers/footers are short; body paragraphs are not
_FURNITURE_REPEAT_FRACTION = 0.30  # a short line on >=30% of pages is page furniture, not content


@dataclass(frozen=True)
class PdfParserAttempt:
    """One parser attempt over a source PDF."""

    parser: str
    status: str
    n_chars: int = 0
    error: str | None = None
    quality: "PdfExtractionQuality | None" = None
    selected: bool = False


@dataclass(frozen=True)
class PdfDiagnostics:
    """PDF-level diagnostics used to explain skipped files."""

    page_count: int | None = None
    encrypted: bool | None = None
    needs_password: bool | None = None
    embedded_text_chars: int | None = None
    image_pages: int | None = None
    image_only_pages: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class PdfExtractionQuality:
    """Comparable extraction-quality features for parser selection and audit."""

    n_chars: int
    page_count: int | None
    page_text_pages: int
    page_coverage: float | None
    citation_pages: int
    citation_coverage: float | None
    heading_count: int
    table_marker_count: int
    score: float


@dataclass(frozen=True)
class PdfPageChunk:
    """Markdown extracted for one PDF page before final corpus rendering."""

    page: int
    text: str
    blocks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PdfExtraction:
    """A parser's markdown output plus optional page-local citation hints."""

    text: str
    parser: str
    pages: list[PdfPageChunk] = field(default_factory=list)
    attempts: list[PdfParserAttempt] = field(default_factory=list)


@dataclass(frozen=True)
class PdfPageCitation:
    """Mapping from a source PDF page to generated corpus character offsets."""

    page: int
    char_start: int
    char_end: int
    text_start: int
    text_end: int
    n_chars: int
    parser: str
    blocks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RenderedPdfDoc:
    """Rendered corpus text and citation spans."""

    text: str
    citations: list[PdfPageCitation]


PdfTextExtractor = Callable[[Path, str], str | PdfExtraction]


@dataclass(frozen=True)
class PdfCorpusItem:
    """One PDF ingestion outcome recorded in the manifest."""

    source: str
    doc_id: str | None
    n_chars: int
    status: str
    error: str | None = None
    parser: str | None = None
    citation_path: str | None = None
    page_count: int | None = None
    embedded_text_chars: int | None = None
    image_only_pages: int | None = None
    attempts: list[PdfParserAttempt] = field(default_factory=list)
    diagnostics: PdfDiagnostics | None = None
    quality: PdfExtractionQuality | None = None
    source_sha256: str | None = None  # reuse key: skip reconversion when the source is unchanged
    reused: bool = False


@dataclass(frozen=True)
class PdfCorpusResult:
    """Summary of one PDF corpus ingestion run."""

    pdf_root: Path
    out_dir: Path
    items: list[PdfCorpusItem]

    @property
    def n_docs(self) -> int:
        return sum(1 for item in self.items if item.status == "ok")

    @property
    def n_skipped(self) -> int:
        return sum(1 for item in self.items if item.status != "ok")


def clean_pdf_text(text: str) -> str:
    """Normalize extracted markdown enough for chunking while preserving readable content."""
    text = text.replace("\x0c", "\n\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return _MANY_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _is_image_only_pdf(diagnostics: PdfDiagnostics) -> bool:
    return bool(
        diagnostics.page_count
        and diagnostics.image_only_pages == diagnostics.page_count
        and diagnostics.embedded_text_chars == 0
    )


def inspect_pdf(pdf_path: Path) -> PdfDiagnostics:
    """Collect cheap diagnostics that explain parser failures and short extractions."""
    try:
        import fitz
    except ImportError as exc:
        return PdfDiagnostics(error=f"missing PyMuPDF dependency: {exc}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        return PdfDiagnostics(error=f"PyMuPDF could not open PDF: {type(exc).__name__}: {exc}")

    embedded_text_chars = 0
    image_pages = 0
    image_only_pages = 0
    for page in doc:
        text_chars = len((page.get_text("text") or "").strip())
        images = len(page.get_images(full=True))
        embedded_text_chars += text_chars
        if images:
            image_pages += 1
        if images and text_chars == 0:
            image_only_pages += 1
    return PdfDiagnostics(
        page_count=doc.page_count,
        encrypted=bool(doc.is_encrypted),
        needs_password=bool(doc.needs_pass),
        embedded_text_chars=embedded_text_chars,
        image_pages=image_pages,
        image_only_pages=image_only_pages,
    )
