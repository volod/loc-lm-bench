"""Unified mixed-corpus ingestion (`txt` / `md` / `pdf`) into the canonical `.md`/`.txt` corpus.

The rest of the pipeline (RAG index, ontology drafting, GraphRAG) consumes a directory of `.md`
and `.txt` files with stable character offsets. This module turns ONE mixed source directory into
that shape:

- PDFs are routed through the existing `ingest_pdf_corpus` converter (PyMuPDF4LLM / Docling OCR,
  `pdf-<digest>.md` ids, citation sidecars, its own incremental reuse).
- `.md` / `.txt` files are passed through verbatim (offsets preserved) under their relative path,
  with the SAME manifest shape as the PDF lane: a `source_sha256` fingerprint, incremental reuse
  when the source is unchanged, and skip diagnostics for short/failed documents.

A unified `corpus_manifest.json` records every source with `kind` (`pdf`|`text`), status, and
reuse flag, so a rerun over an unchanged mixed corpus reports `reused: true` for every document.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llb.prep.pdf_corpus import (
    PDF_SUFFIX,
    PdfTextExtractor,
    _sha256_file,
    default_markdown_out_dir,
    ingest_pdf_corpus,
)

_LOG = logging.getLogger(__name__)

CORPUS_MANIFEST = "corpus_manifest.json"
TEXT_SUFFIXES = (".md", ".txt")
KIND_PDF = "pdf"
KIND_TEXT = "text"
DEFAULT_MIN_CHARS = 500


@dataclass(frozen=True)
class CorpusItem:
    """One source-document ingestion outcome in the unified manifest.

    Shares the PDF lane's reuse contract: `source_sha256` fingerprints the source, `reused` marks a
    skipped reconversion, and a non-`ok` `status` carries an `error` explaining the skip.
    """

    source: str
    doc_id: str | None
    kind: str  # KIND_PDF | KIND_TEXT
    status: str  # "ok" | "too_short" | "error"
    n_chars: int
    source_sha256: str | None = None
    reused: bool = False
    error: str | None = None
    parser: str | None = None  # PDF lane only


@dataclass(frozen=True)
class CorpusIngestResult:
    """Summary of one mixed-corpus ingestion run."""

    source_root: Path
    out_dir: Path
    items: list[CorpusItem]

    @property
    def n_docs(self) -> int:
        return sum(1 for item in self.items if item.status == "ok")

    @property
    def n_skipped(self) -> int:
        return sum(1 for item in self.items if item.status != "ok")

    @property
    def n_reused(self) -> int:
        return sum(1 for item in self.items if item.reused)


def _iter_by_suffix(root: Path, out_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (pdfs, text files) under `root` in stable order, excluding the `out_dir` subtree.

    The staged corpus commonly lives at `<root>/_md`, so a rerun must not re-ingest its own output.
    """
    out_resolved = out_dir.resolve()
    pdfs: list[Path] = []
    texts: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix().casefold()):
        if not path.is_file():
            continue
        try:
            if out_resolved == path.resolve().parent or out_resolved in path.resolve().parents:
                continue
        except OSError:
            pass
        suffix = path.suffix.lower()
        if suffix == PDF_SUFFIX:
            pdfs.append(path)
        elif suffix in TEXT_SUFFIXES:
            texts.append(path)
    return pdfs, texts


def _previous_text_items(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Load prior `text` manifest items as `source -> payload` for incremental reuse."""
    path = out_dir / CORPUS_MANIFEST
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items = payload.get("items") if isinstance(payload, dict) else None
    previous: dict[str, dict[str, Any]] = {}
    for item in items if isinstance(items, list) else []:
        if (
            isinstance(item, dict)
            and item.get("kind") == KIND_TEXT
            and isinstance(item.get("source"), str)
        ):
            previous[item["source"]] = item
    return previous


def _ingest_text_file(
    root: Path,
    path: Path,
    out_dir: Path,
    min_chars: int,
    previous: dict[str, dict[str, Any]],
    refresh: bool,
) -> CorpusItem:
    source = path.relative_to(root).as_posix()
    doc_id = source  # preserve the relative path so RAG/ontology keep the same doc id
    source_sha256 = _sha256_file(path)
    target = out_dir / doc_id
    prev = previous.get(source)
    if (
        not refresh
        and prev is not None
        and prev.get("status") == "ok"
        and prev.get("source_sha256") == source_sha256
        and isinstance(prev.get("n_chars"), int)
        and target.is_file()
    ):
        _LOG.info("[corpus] reuse %s (unchanged source %s)", doc_id, source)
        return CorpusItem(
            source=source,
            doc_id=doc_id,
            kind=KIND_TEXT,
            status="ok",
            n_chars=int(prev["n_chars"]),
            source_sha256=source_sha256,
            reused=True,
        )
    text = path.read_text(encoding="utf-8")
    if len(text) < min_chars:
        return CorpusItem(
            source=source,
            doc_id=None,
            kind=KIND_TEXT,
            status="too_short",
            n_chars=len(text),
            source_sha256=source_sha256,
            error=f"text shorter than {min_chars} chars",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return CorpusItem(
        source=source,
        doc_id=doc_id,
        kind=KIND_TEXT,
        status="ok",
        n_chars=len(text),
        source_sha256=source_sha256,
    )


def _pdf_item_to_corpus_item(payload: dict[str, Any]) -> CorpusItem:
    return CorpusItem(
        source=str(payload.get("source", "")),
        doc_id=payload.get("doc_id"),
        kind=KIND_PDF,
        status=str(payload.get("status", "error")),
        n_chars=int(payload.get("n_chars") or 0),
        source_sha256=payload.get("source_sha256"),
        reused=bool(payload.get("reused", False)),
        error=payload.get("error"),
        parser=payload.get("parser"),
    )


def _manifest(result: CorpusIngestResult) -> dict[str, object]:
    return {
        "kind": "corpus",
        "source_root": str(result.source_root),
        "corpus_root": str(result.out_dir),
        "n_sources": len(result.items),
        "n_docs": result.n_docs,
        "n_skipped": result.n_skipped,
        "n_reused": result.n_reused,
        "items": [asdict(item) for item in result.items],
    }


def ingest_corpus(
    root: Path | str,
    out_dir: Path | str | None = None,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    parser: str = "auto",
    refresh: bool = False,
    extractor: PdfTextExtractor | None = None,
) -> CorpusIngestResult:
    """Ingest a mixed `txt`/`md`/`pdf` directory into one canonical `.md`/`.txt` corpus.

    PDFs route through `ingest_pdf_corpus`; `.md`/`.txt` pass through verbatim. Unchanged sources
    are reused (fingerprinted by sha256); `refresh=True` forces a full reconversion of both lanes.
    """
    source_root = Path(root)
    if not source_root.exists():
        raise ValueError(f"corpus root does not exist: {source_root}")
    target = Path(out_dir) if out_dir is not None else default_markdown_out_dir(source_root)
    pdfs, texts = _iter_by_suffix(source_root, target)
    if not pdfs and not texts:
        raise ValueError(f"no .txt/.md/.pdf documents under {source_root}")
    target.mkdir(parents=True, exist_ok=True)

    items: list[CorpusItem] = []
    if pdfs:
        pdf_result = ingest_pdf_corpus(
            source_root,
            target,
            min_chars=min_chars,
            parser=parser,
            extractor=extractor,
            refresh=refresh,
        )
        items.extend(_pdf_item_to_corpus_item(asdict(item)) for item in pdf_result.items)

    previous = {} if refresh else _previous_text_items(target)
    for path in texts:
        items.append(_ingest_text_file(source_root, path, target, min_chars, previous, refresh))

    result = CorpusIngestResult(source_root=source_root, out_dir=target, items=items)
    (target / CORPUS_MANIFEST).write_text(
        json.dumps(_manifest(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _LOG.info(
        "[corpus] ingested %d/%d documents (%d reused, %d skipped) -> %s",
        result.n_docs,
        len(result.items),
        result.n_reused,
        result.n_skipped,
        target,
    )
    return result
