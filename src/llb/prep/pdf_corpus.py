"""PDF corpus ingestion for local Ukrainian document collections.

The rest of the RAG/goldset pipeline consumes `.md` and `.txt` files with stable character
offsets. This module turns a local PDF directory into that canonical text corpus using
PyMuPDF4LLM markdown extraction. The extracted `.md` files become the source of truth for later
span validation; original PDFs are recorded only as provenance in the manifest.
"""

import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

_LOG = logging.getLogger(__name__)

PDF_CORPUS_MANIFEST = "pdf_corpus_manifest.json"
PDF_SUFFIX = ".pdf"
DEFAULT_MARKDOWN_DIRNAME = "_md"
PYMUPDF4LLM_TOOL = "pymupdf4llm"
_MANY_BLANK_LINES = re.compile(r"\n{3,}")

PdfTextExtractor = Callable[[Path, str], str]


@dataclass(frozen=True)
class PdfCorpusItem:
    """One PDF ingestion outcome recorded in the manifest."""

    source: str
    doc_id: str | None
    n_chars: int
    status: str
    error: str | None = None


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


def extract_pdf_markdown(pdf_path: Path) -> str:
    """Extract markdown from one PDF with PyMuPDF4LLM."""
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise RuntimeError("missing pymupdf4llm dependency") from exc
    try:
        markdown = pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"pymupdf4llm failed for {pdf_path.name}: {exc}") from exc
    if not isinstance(markdown, str):
        raise RuntimeError(f"pymupdf4llm returned non-text markdown for {pdf_path.name}")
    return markdown


def _source_rel(pdf_root: Path, pdf_path: Path) -> str:
    return pdf_path.relative_to(pdf_root).as_posix()


def _doc_text(source: str, text: str) -> str:
    return f"# Source PDF: {source}\n\n{text}\n"


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


def _write_manifest(result: PdfCorpusResult) -> None:
    (result.out_dir / PDF_CORPUS_MANIFEST).write_text(
        json.dumps(_manifest(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _ingest_one_pdf(
    pdf_root: Path,
    pdf_path: Path,
    out_dir: Path,
    extractor: PdfTextExtractor,
    min_chars: int,
) -> PdfCorpusItem:
    source = _source_rel(pdf_root, pdf_path)
    try:
        text = clean_pdf_text(extractor(pdf_path, PYMUPDF4LLM_TOOL))
    except RuntimeError as exc:
        return PdfCorpusItem(source=source, doc_id=None, n_chars=0, status="error", error=str(exc))
    if len(text) < min_chars:
        return PdfCorpusItem(
            source=source,
            doc_id=None,
            n_chars=len(text),
            status="too_short",
            error=f"extracted text shorter than {min_chars} chars",
        )
    doc_id = doc_id_for_pdf(pdf_root, pdf_path)
    (out_dir / doc_id).write_text(_doc_text(source, text), encoding="utf-8")
    return PdfCorpusItem(source=source, doc_id=doc_id, n_chars=len(text), status="ok")


def ingest_pdf_corpus(
    pdf_root: Path | str,
    out_dir: Path | str | None = None,
    *,
    min_chars: int = 500,
    limit: int | None = None,
    extractor: PdfTextExtractor | None = None,
) -> PdfCorpusResult:
    """Extract a local PDF directory into a `.md` corpus and write a manifest."""
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
    selected_extractor = extractor or (lambda path, _tool: extract_pdf_markdown(path))
    items = [_ingest_one_pdf(root, pdf, target, selected_extractor, min_chars) for pdf in pdfs]
    result = PdfCorpusResult(pdf_root=root, out_dir=target, items=items)
    _write_manifest(result)
    _LOG.info("[pdf-corpus] extracted %d/%d PDFs into %s", result.n_docs, len(result.items), target)
    return result
