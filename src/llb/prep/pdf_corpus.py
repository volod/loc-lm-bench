"""PDF corpus ingestion for local Ukrainian document collections.

The rest of the RAG/goldset pipeline consumes `.md` and `.txt` files with stable character
offsets. This module turns a local PDF directory into that canonical text corpus using
PyMuPDF4LLM markdown extraction. The extracted `.md` files become the source of truth for later
span validation; original PDFs are recorded only as provenance in the manifest.
"""

import hashlib
import html
import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

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


def clean_pdf_text(text: str) -> str:
    """Normalize extracted markdown enough for chunking while preserving readable content."""
    text = text.replace("\x0c", "\n\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return _MANY_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


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


def _normalize_extraction(value: str | PdfExtraction, parser: str) -> PdfExtraction:
    if isinstance(value, PdfExtraction):
        return value
    return PdfExtraction(text=clean_pdf_text(value), parser=parser)


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
        pages: list[PdfPageChunk] = []
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            metadata_obj = chunk.get("metadata")
            metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
            page = int(metadata.get("page_number") or idx + 1)
            page_text = clean_pdf_text(str(chunk.get("text") or ""))
            blocks = _page_blocks(chunk.get("page_boxes"))
            pages.append(PdfPageChunk(page=page, text=page_text, blocks=blocks))
        text = clean_pdf_text("\n\n".join(page.text for page in pages if page.text))
        return PdfExtraction(text=text, parser=PYMUPDF4LLM_TOOL, pages=pages)

    if isinstance(chunks, str):
        return PdfExtraction(text=clean_pdf_text(chunks), parser=PYMUPDF4LLM_TOOL)
    raise RuntimeError(f"pymupdf4llm returned unsupported markdown for {pdf_path.name}")


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


def _extract_with_docling(pdf_path: Path) -> PdfExtraction:
    """Extract with Docling when the optional CUDA-capable layout/OCR dependency is installed."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
        from docling.document_converter import DocumentConverter
        from docling.document_converter import PdfFormatOption
    except ImportError as exc:
        raise RuntimeError("missing docling dependency") from exc
    diagnostics = inspect_pdf(pdf_path)
    force_ocr = _is_image_only_pdf(diagnostics)
    last_error: Exception | None = None
    for ocr_options in _docling_ocr_option_candidates(force_ocr):
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = force_ocr
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
            _configure_docling_accelerator(pipeline_options)
            if force_ocr and ocr_options is not None:
                pipeline_options.ocr_options = ocr_options
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
            result = converter.convert(str(pdf_path))
            document = result.document
            pages = _docling_pages(document)
            markdown = clean_pdf_text(
                "\n\n".join(page.text for page in pages if page.text.strip())
                or _docling_export_markdown(document)
            )
            return PdfExtraction(text=markdown, parser=DOCLING_TOOL, pages=pages)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError(f"docling failed for {pdf_path.name}: {last_error}") from last_error
    raise RuntimeError(f"docling failed for {pdf_path.name}: no OCR options available")


def _configure_docling_accelerator(pipeline_options: Any) -> None:
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    except ImportError:
        return
    try:
        device = AcceleratorDevice.CUDA if _torch_cuda_available() else AcceleratorDevice.AUTO
        pipeline_options.accelerator_options = AcceleratorOptions(device=device)
    except Exception:
        return


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _docling_ocr_option_candidates(force_full_page_ocr: bool) -> list[Any | None]:
    if not force_full_page_ocr:
        return [None]
    try:
        from docling.datamodel import pipeline_options as options
    except ImportError:
        return [None]

    candidates = (
        (
            "TesseractCliOcrOptions",
            {"force_full_page_ocr": force_full_page_ocr, "lang": ["ukr", "eng"]},
        ),
        ("RapidOcrOptions", {"force_full_page_ocr": force_full_page_ocr, "backend": "torch"}),
        (
            "EasyOcrOptions",
            {"force_full_page_ocr": force_full_page_ocr, "lang": ["uk", "en"], "use_gpu": True},
        ),
        ("OcrAutoOptions", {"force_full_page_ocr": force_full_page_ocr, "lang": ["uk", "en"]}),
    )
    options_list: list[Any | None] = []
    for name, kwargs in candidates:
        cls = getattr(options, name, None)
        if cls is None:
            continue
        for variant in (
            kwargs,
            {key: value for key, value in kwargs.items() if key != "backend"},
            {key: value for key, value in kwargs.items() if key not in {"backend", "lang"}},
            {"force_full_page_ocr": force_full_page_ocr},
            {},
        ):
            try:
                options_list.append(cls(**variant))
                break
            except Exception:
                continue
    options_list.append(None)
    return options_list


def _docling_pages(document: Any) -> list[PdfPageChunk]:
    try:
        page_count = int(document.num_pages())
    except Exception:
        page_count = 0

    pages: list[PdfPageChunk] = []
    for page_number in range(1, page_count + 1):
        page_text = clean_pdf_text(_docling_export_markdown(document, page_number))
        if page_text:
            pages.append(PdfPageChunk(page=page_number, text=page_text))
    return pages


def _docling_export_markdown(document: Any, page_number: int | None = None) -> str:
    kwargs: dict[str, Any] = {
        "compact_tables": True,
        "page_break_placeholder": None,
    }
    if page_number is not None:
        kwargs["page_no"] = page_number
    try:
        return str(document.export_to_markdown(**kwargs))
    except TypeError:
        if page_number is not None:
            return str(document.export_to_markdown(page_no=page_number))
        return str(document.export_to_markdown())


def _extract_with_marker(pdf_path: Path) -> PdfExtraction:
    """Extract with Marker when its optional package and API are installed."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as exc:
        raise RuntimeError("missing marker dependency") from exc
    try:
        force_ocr = _is_image_only_pdf(inspect_pdf(pdf_path))
        try:
            converter = PdfConverter(
                artifact_dict=create_model_dict(),
                config={"force_ocr": force_ocr, "paginate_output": True},
            )
        except TypeError:
            converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(pdf_path))
        pages = _marker_pages(rendered)
        markdown = getattr(rendered, "markdown", None)
        if markdown is None:
            markdown = "\n\n".join(page.text for page in pages if page.text) or str(rendered)
    except Exception as exc:
        raise RuntimeError(f"marker failed for {pdf_path.name}: {exc}") from exc
    return PdfExtraction(text=clean_pdf_text(str(markdown)), parser=MARKER_TOOL, pages=pages)


def _marker_pages(rendered: Any) -> list[PdfPageChunk]:
    root = _object_to_mapping(rendered)
    children = root.get("children") if root else getattr(rendered, "children", None)
    if not isinstance(children, list):
        return []

    pages: list[PdfPageChunk] = []
    for idx, child in enumerate(children):
        item = _object_to_mapping(child)
        block_type = str(item.get("block_type") or "").casefold()
        if block_type != "page":
            continue
        page_number = _marker_page_number(item, idx + 1)
        text = clean_pdf_text(_marker_block_text(item))
        if text:
            pages.append(
                PdfPageChunk(page=page_number, text=text, blocks=_marker_block_boxes(item))
            )
    return pages


def _object_to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _marker_page_number(item: dict[str, Any], fallback: int) -> int:
    item_id = str(item.get("id") or "")
    match = _MARKER_PAGE_ID.search(item_id)
    if match is None:
        return fallback
    return int(match.group(1)) + 1


def _marker_block_text(item: dict[str, Any]) -> str:
    children = item.get("children")
    if isinstance(children, list) and children:
        parts = [_marker_block_text(_object_to_mapping(child)) for child in children]
        return "\n\n".join(part for part in parts if part.strip())
    raw_html = str(item.get("html") or item.get("text") or "")
    return _html_to_text(raw_html)


def _marker_block_boxes(item: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    children = item.get("children")
    if not isinstance(children, list):
        return blocks
    for child in children:
        child_map = _object_to_mapping(child)
        polygon = child_map.get("polygon")
        blocks.append(
            {
                "class": str(child_map.get("block_type") or "unknown"),
                "bbox": polygon if isinstance(polygon, list) else None,
                "page_char_start": None,
                "page_char_end": None,
            }
        )
    return blocks


def _html_to_text(value: str) -> str:
    return html.unescape(_HTML_TAG.sub(" ", value)).strip()


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

    for page in pages:
        page_marker = (
            f"<!-- source_pdf: {source} page: {page.page} parser: {extraction.parser} -->\n\n"
        )
        page_start = sum(len(part) for part in parts)
        text_start = page_start + len(page_marker)
        page_text = f"{page.text.strip()}\n\n"
        text_end = text_start + len(page.text.strip())
        page_end = page_start + len(page_marker) + len(page_text)
        parts.extend([page_marker, page_text])
        citations.append(
            PdfPageCitation(
                page=page.page,
                char_start=page_start,
                char_end=page_end,
                text_start=text_start,
                text_end=text_end,
                n_chars=len(page.text.strip()),
                parser=extraction.parser,
                blocks=_offset_blocks(page.blocks, text_start),
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
    )


def ingest_pdf_corpus(
    pdf_root: Path | str,
    out_dir: Path | str | None = None,
    *,
    min_chars: int = 500,
    limit: int | None = None,
    parser: str = PARSER_AUTO,
    extractor: PdfTextExtractor | None = None,
) -> PdfCorpusResult:
    """Extract a local PDF directory into a `.md` corpus and write a manifest."""
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
    items = [_ingest_one_pdf(root, pdf, target, extractor, min_chars, parser) for pdf in pdfs]
    result = PdfCorpusResult(pdf_root=root, out_dir=target, items=items)
    _write_manifest(result)
    _write_quality_report(result)
    _LOG.info("[pdf-corpus] extracted %d/%d PDFs into %s", result.n_docs, len(result.items), target)
    return result
