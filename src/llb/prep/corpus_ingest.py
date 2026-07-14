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
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llb.prep.pdf.ingest import ingest_pdf_corpus
from llb.prep.pdf.model import PdfTextExtractor
from llb.prep.pdf.render import default_markdown_out_dir
from llb.prep.corpus_governance import (
    DEFAULT_SOURCE_SYSTEM,
    manifest_items_fingerprint,
    preserve_ingestion_time,
    utc_ingestion_time,
)
from llb.prep.corpus_ingest_text import (
    CORPUS_MANIFEST,
    CorpusIngestResult,
    CorpusItem,
    DEFAULT_MIN_CHARS,
    KIND_PDF,
    _LOG,
    _ingest_text_file,
    _iter_by_suffix,
    _previous_manifest_items,
)


def _pdf_item_to_corpus_item(
    payload: dict[str, Any],
    previous: dict[str, dict[str, Any]],
    *,
    default_language: str | None,
    default_source_system: str,
    default_acl_label: str | None,
    ingestion_time: str,
) -> CorpusItem:
    source = str(payload.get("source", ""))
    prev = previous.get(source)
    governance = preserve_ingestion_time(
        prev,
        {
            "language": payload.get("language") or default_language or "und",
            "version": payload.get("version"),
            "effective_date": payload.get("effective_date"),
            "ingestion_time": ingestion_time,
            "source_system": payload.get("source_system") or default_source_system,
            "acl_label": payload.get("acl_label") or default_acl_label,
        },
    )
    return CorpusItem(
        source=source,
        doc_id=payload.get("doc_id"),
        kind=KIND_PDF,
        status=str(payload.get("status", "error")),
        n_chars=int(payload.get("n_chars") or 0),
        source_sha256=payload.get("source_sha256"),
        reused=bool(payload.get("reused", False)),
        error=payload.get("error"),
        parser=payload.get("parser"),
        language=governance["language"],
        version=governance["version"],
        effective_date=governance["effective_date"],
        ingestion_time=governance["ingestion_time"],
        source_system=governance["source_system"],
        acl_label=governance["acl_label"],
    )


def _manifest(result: CorpusIngestResult) -> dict[str, object]:
    item_rows = [asdict(item) for item in result.items]
    return {
        "kind": "corpus",
        "source_root": str(result.source_root),
        "corpus_root": str(result.out_dir),
        "n_sources": len(result.items),
        "n_docs": result.n_docs,
        "n_skipped": result.n_skipped,
        "n_reused": result.n_reused,
        "n_removed_sources": result.n_removed_sources,
        "removed_sources": result.removed_sources,
        "corpus_fingerprint": manifest_items_fingerprint(item_rows),
        "items": item_rows,
    }


def _unlink_if_file(path: Path, description: str) -> None:
    """Best-effort delete of a staged output file; log (do not raise) on OS errors."""
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        _LOG.warning("[corpus] could not remove stale %s %s", description, path)


def _remove_stale_doc_files(out_dir: Path, doc_id: str, payload: dict[str, Any]) -> None:
    """Delete the staged document and its citation sidecar for one superseded manifest entry."""
    _unlink_if_file(out_dir / doc_id, "staged document")
    citation_path = payload.get("citation_path")
    if isinstance(citation_path, str):
        _unlink_if_file(out_dir / citation_path, "citation sidecar")


def _cleanup_stale_outputs(
    out_dir: Path, previous: dict[str, dict[str, Any]], current: list[CorpusItem]
) -> list[str]:
    """Remove old staged docs whose source disappeared or whose doc id changed."""
    current_sources = {item.source for item in current}
    current_doc_ids = {item.doc_id for item in current if item.status == "ok" and item.doc_id}
    removed_sources: list[str] = []
    for source, payload in sorted(previous.items()):
        if source not in current_sources:
            removed_sources.append(source)
        doc_id = payload.get("doc_id")
        if (
            payload.get("status") != "ok"
            or not isinstance(doc_id, str)
            or doc_id in current_doc_ids
        ):
            continue
        _remove_stale_doc_files(out_dir, doc_id, payload)
    return removed_sources


def ingest_corpus(
    root: Path | str,
    out_dir: Path | str | None = None,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    parser: str = "auto",
    refresh: bool = False,
    extractor: PdfTextExtractor | None = None,
    default_language: str | None = None,
    source_system: str = DEFAULT_SOURCE_SYSTEM,
    acl_label: str | None = None,
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
    ingestion_time = utc_ingestion_time()
    previous = {} if refresh else _previous_manifest_items(target)
    if pdfs:
        pdf_result = ingest_pdf_corpus(
            source_root,
            target,
            min_chars=min_chars,
            parser=parser,
            extractor=extractor,
            refresh=refresh,
        )
        items.extend(
            _pdf_item_to_corpus_item(
                asdict(item),
                previous,
                default_language=default_language,
                default_source_system=source_system,
                default_acl_label=acl_label,
                ingestion_time=ingestion_time,
            )
            for item in pdf_result.items
        )

    for path in texts:
        items.append(
            _ingest_text_file(
                source_root,
                path,
                target,
                min_chars,
                previous,
                refresh,
                default_language,
                source_system,
                acl_label,
                ingestion_time,
            )
        )

    removed_sources = _cleanup_stale_outputs(target, previous, items)
    result = CorpusIngestResult(
        source_root=source_root, out_dir=target, items=items, removed_sources=removed_sources
    )
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
