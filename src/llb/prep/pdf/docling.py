"""Docling markdown extraction (optional CUDA-capable layout/OCR parser)."""

from pathlib import Path
from typing import Any

from llb.prep.pdf.model import (
    DOCLING_TOOL,
    PdfExtraction,
    PdfPageChunk,
    _is_image_only_pdf,
    clean_pdf_text,
    inspect_pdf,
)


def _extract_with_docling(pdf_path: Path) -> PdfExtraction:
    """Extract with Docling when the optional CUDA-capable layout/OCR dependency is installed."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
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
        instance = _first_constructible(cls, kwargs, force_full_page_ocr)
        if instance is not None:
            options_list.append(instance)
    options_list.append(None)
    return options_list


def _first_constructible(cls: Any, kwargs: dict[str, Any], force_full_page_ocr: bool) -> Any:
    """Instantiate `cls` with progressively fewer kwargs (docling versions differ); None if all fail."""
    variants: tuple[dict[str, Any], ...] = (
        kwargs,
        {key: value for key, value in kwargs.items() if key != "backend"},
        {key: value for key, value in kwargs.items() if key not in {"backend", "lang"}},
        {"force_full_page_ocr": force_full_page_ocr},
        {},
    )
    for variant in variants:
        try:
            return cls(**variant)
        except Exception:
            continue
    return None


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
