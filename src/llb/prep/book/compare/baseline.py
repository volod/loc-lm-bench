"""The existing PDF lane, with the book's OCR language recorded on the sidecar.

The corpus default reads Ukrainian and English together. A book comparison passes
the book's own language and writes that tag into the citation diagnostics, so a
later score can see the setting beside the numbers. When Docling is not installed
in this process, the same extraction runs in the isolated ``book-docling``
environment.
"""

import json
import subprocess
from pathlib import Path

from llb.core.contracts.data_prep.corpus import PdfCitations, PdfPageCitationRecord
from llb.core.fsutil import atomic_write_text
from llb.core.paths import resolve_data_dir
from llb.prep.book.engines.protocol import EngineUnavailable
from llb.prep.book.survey.envs import installed_env, tool_environment

_BASELINE_TIMEOUT_S = 7200
_RUNNER = Path(__file__).resolve().parents[1] / "engines" / "runners" / "baseline_docling.py"
_LOG_TAIL = 2000

_DOC_NAME = "baseline.md"
_SIDECAR = "baseline.citations.json"


def citations_from_pages(
    pages: list[tuple[int, str]], *, language: str, source: str
) -> tuple[str, PdfCitations]:
    """Markdown and a citation sidecar. ``pages`` are ``(pdf_page, text)``."""
    if not language:
        raise ValueError("baseline OCR language is empty")
    chunks: list[str] = []
    records: list[PdfPageCitationRecord] = []
    cursor = 0
    for index, (pdf_page, text) in enumerate(pages):
        body = text.strip()
        if index and (chunks and chunks[-1]) and body:
            chunks.append("\n\n")
            cursor += 2
        start = cursor
        chunks.append(body)
        cursor += len(body)
        records.append(
            PdfPageCitationRecord(
                page=pdf_page,
                char_start=start,
                char_end=cursor,
                text_start=start,
                text_end=cursor,
                n_chars=len(body),
                parser="docling",
            )
        )
    markdown = "".join(chunks)
    citations = PdfCitations(
        schema_id="llb.pdf-citations",
        schema_version="1.0.0",
        kind="pdf-citations",
        source=source,
        doc_id=_DOC_NAME,
        parser="docling",
        diagnostics={"ocr_language": language},
        pages=records,
    )
    return markdown, citations


def write_baseline(dest: Path, pages: list[tuple[int, str]], *, language: str, source: str) -> Path:
    """Write the markdown and the sidecar. Returns the sidecar path."""
    markdown, citations = citations_from_pages(pages, language=language, source=source)
    dest.mkdir(parents=True, exist_ok=True)
    atomic_write_text(dest / _DOC_NAME, markdown)
    path = dest / _SIDECAR
    atomic_write_text(path, citations.model_dump_json(indent=2) + "\n")
    return path


def recorded_language(path: Path) -> str:
    """The OCR language stored on a baseline sidecar."""
    try:
        record = PdfCitations.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not a pdf citations sidecar: {exc}") from exc
    language = record.diagnostics.get("ocr_language")
    if not isinstance(language, str) or not language:
        raise ValueError(f"{path} does not record ocr_language")
    return language


def docling_available() -> bool:
    """True when the project environment can import Docling."""
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


def extract_baseline(pdf: Path, pages: list[int], language: str, dest: Path) -> Path:
    """Run the existing lane on ``pages`` and write its sidecar.

    Docling is optional. A host without it in this process or in ``book-docling``
    is refused by name before a page is read.
    """
    if not language:
        raise ValueError("baseline OCR language is empty")
    if not docling_available() and installed_env(resolve_data_dir(), "docling") is None:
        raise EngineUnavailable("docling", "missing docling dependency")
    subset = dest / "subset.pdf"
    dest.mkdir(parents=True, exist_ok=True)
    _write_subset(pdf, pages, subset)
    by_number = _read_subset(subset, language, dest)
    texts = [(original, by_number.get(index, "")) for index, original in enumerate(pages, start=1)]
    return write_baseline(dest, texts, language=language, source=pdf.name)


def _read_subset(subset: Path, language: str, dest: Path) -> dict[int, str]:
    if docling_available():
        from llb.prep.pdf.docling import _extract_with_docling

        extraction = _extract_with_docling(subset, ocr_languages=(language,))
        return {chunk.page: chunk.text for chunk in extraction.pages}
    return _read_subset_in_env(subset, language, dest)


def _read_subset_in_env(subset: Path, language: str, dest: Path) -> dict[int, str]:
    env_info = installed_env(resolve_data_dir(), "docling")
    if env_info is None:
        raise EngineUnavailable("docling", "missing docling dependency")
    from llb.prep.pdf.model import _is_image_only_pdf, clean_pdf_text, inspect_pdf

    force = "1" if _is_image_only_pdf(inspect_pdf(subset)) else "0"
    out = dest / "baseline-pages.json"
    log_path = dest / "baseline-docling.log"
    env = tool_environment(resolve_data_dir(), env_info.python.parent)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [str(env_info.python), "-I", str(_RUNNER), str(subset), language, force, str(out)],
                cwd=dest,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=_BASELINE_TIMEOUT_S,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise EngineUnavailable("docling", f"timed out after {_BASELINE_TIMEOUT_S:.0f}s") from exc
    if completed.returncode != 0 or not out.is_file():
        raise EngineUnavailable("docling", _tail(log_path) or f"exit {completed.returncode}")
    try:
        payload = json.loads(out.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EngineUnavailable("docling", f"page json is not readable: {exc}") from exc
    if not isinstance(payload, list):
        raise EngineUnavailable("docling", "page json is not a list")
    pages: dict[int, str] = {}
    for row in payload:
        if not isinstance(row, dict) or not isinstance(row.get("page"), int):
            continue
        text = row.get("text")
        pages[row["page"]] = clean_pdf_text(text if isinstance(text, str) else "")
    return pages


def _tail(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-_LOG_TAIL:].strip()


def _write_subset(pdf: Path, pages: list[int], dest: Path) -> None:
    import fitz

    source = fitz.open(pdf)
    try:
        output = fitz.open()
        try:
            for page in pages:
                output.insert_pdf(source, from_page=page - 1, to_page=page - 1)
            output.save(dest)
        finally:
            output.close()
    finally:
        source.close()


__all__ = [
    "citations_from_pages",
    "docling_available",
    "extract_baseline",
    "recorded_language",
    "write_baseline",
]
