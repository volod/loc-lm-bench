"""Existing-lane Docling extraction: ``python -I baseline_docling.py <pdf> <lang> <force> <out>``.

Runs inside the survey's docling environment, which has Docling and not this repo.
Stdlib only, apart from that import. ``force`` is ``1`` when the PDF has no text
layer. The Tesseract language is the tag the book recorded (``eng`` for the
handbook). This is the same order as ``llb.prep.pdf.docling._extract_with_docling``:
Tesseract, then RapidOCR, then EasyOCR, then auto.
"""

import json
import sys
from pathlib import Path
from typing import Any

_EASY = {"eng": "en", "ukr": "uk", "en": "en", "uk": "uk"}


def main(pdf: str, language: str, force: str, dest: str) -> None:
    pages = extract(Path(pdf), language, force == "1")
    Path(dest).write_text(json.dumps(pages), encoding="utf-8")


def extract(pdf: Path, language: str, force_ocr: bool) -> list[dict[str, Any]]:
    """One ``{page, text}`` row per PDF page. Text may be empty."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    last_error: Exception | None = None
    for ocr_options in _ocr_options(force_ocr, language):
        try:
            options = PdfPipelineOptions()
            options.do_ocr = force_ocr
            options.do_table_structure = True
            options.table_structure_options = TableStructureOptions(do_cell_matching=True)
            _accelerator(options)
            if force_ocr and ocr_options is not None:
                options.ocr_options = ocr_options
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
            document = converter.convert(str(pdf)).document
            return _pages(document)
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"docling failed for {pdf.name}: {last_error}")


def _pages(document: Any) -> list[dict[str, Any]]:
    try:
        count = int(document.num_pages())
    except Exception:
        count = 0
    rows = []
    for number in range(1, count + 1):
        rows.append({"page": number, "text": _markdown(document, number)})
    return rows


def _markdown(document: Any, page_number: int) -> str:
    kwargs: dict[str, Any] = {"compact_tables": True, "page_break_placeholder": None}
    kwargs["page_no"] = page_number
    try:
        return str(document.export_to_markdown(**kwargs))
    except TypeError:
        return str(document.export_to_markdown(page_no=page_number))


def _accelerator(options: Any) -> None:
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    except ImportError:
        return
    try:
        import torch

        device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.AUTO
        options.accelerator_options = AcceleratorOptions(device=device)
    except Exception:
        return


def _ocr_options(force_ocr: bool, language: str) -> list[Any | None]:
    if not force_ocr:
        return [None]
    from docling.datamodel import pipeline_options as options

    tesseract = [language]
    easy = [_EASY.get(language, language)]
    specs = (
        ("TesseractCliOcrOptions", {"force_full_page_ocr": True, "lang": tesseract}),
        ("RapidOcrOptions", {"force_full_page_ocr": True, "backend": "torch"}),
        ("EasyOcrOptions", {"force_full_page_ocr": True, "lang": easy, "use_gpu": True}),
        ("OcrAutoOptions", {"force_full_page_ocr": True, "lang": easy}),
    )
    built: list[Any | None] = []
    for name, kwargs in specs:
        cls = getattr(options, name, None)
        if cls is None:
            continue
        instance = _construct(cls, kwargs)
        if instance is not None:
            built.append(instance)
    built.append(None)
    return built


def _construct(cls: Any, kwargs: dict[str, Any]) -> Any:
    variants: tuple[dict[str, Any], ...] = (
        kwargs,
        {key: value for key, value in kwargs.items() if key != "backend"},
        {key: value for key, value in kwargs.items() if key not in {"backend", "lang"}},
        {"force_full_page_ocr": True},
        {},
    )
    for variant in variants:
        try:
            return cls(**variant)
        except Exception:
            continue
    return None


if __name__ == "__main__":
    main(*sys.argv[1:])
