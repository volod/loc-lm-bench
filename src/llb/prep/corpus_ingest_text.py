"""Focused corpus ingest text implementation."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from llb.prep.pdf.model import PDF_SUFFIX
from llb.prep.pdf.reuse import _sha256_file
from llb.prep.corpus_governance import (
    preserve_ingestion_time,
    source_governance,
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
    language: str | None = None
    version: str | None = None
    effective_date: str | None = None
    ingestion_time: str | None = None
    source_system: str | None = None
    acl_label: str | None = None


@dataclass(frozen=True)
class CorpusIngestResult:
    """Summary of one mixed-corpus ingestion run."""

    source_root: Path
    out_dir: Path
    items: list[CorpusItem]
    removed_sources: list[str]

    @property
    def n_docs(self) -> int:
        return sum(1 for item in self.items if item.status == "ok")

    @property
    def n_skipped(self) -> int:
        return sum(1 for item in self.items if item.status != "ok")

    @property
    def n_reused(self) -> int:
        return sum(1 for item in self.items if item.reused)

    @property
    def n_removed_sources(self) -> int:
        return len(self.removed_sources)


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


def _previous_manifest_items(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Load prior unified manifest items as `source -> payload` for reuse and diff reporting."""
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
        if isinstance(item, dict) and isinstance(item.get("source"), str):
            previous[item["source"]] = item
    return previous


def _ingest_text_file(
    root: Path,
    path: Path,
    out_dir: Path,
    min_chars: int,
    previous: dict[str, dict[str, Any]],
    refresh: bool,
    default_language: str | None,
    default_source_system: str,
    default_acl_label: str | None,
    ingestion_time: str,
) -> CorpusItem:
    source = path.relative_to(root).as_posix()
    doc_id = source  # preserve the relative path so RAG/ontology keep the same doc id
    source_sha256 = _sha256_file(path)
    target = out_dir / doc_id
    prev = previous.get(source)
    text = path.read_text(encoding="utf-8")
    governance = source_governance(
        root,
        path,
        text=text,
        default_language=default_language,
        default_source_system=default_source_system,
        default_acl_label=default_acl_label,
        ingestion_time=ingestion_time,
    )
    governance = preserve_ingestion_time(prev, governance)
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
            language=governance["language"],
            version=governance["version"],
            effective_date=governance["effective_date"],
            ingestion_time=governance["ingestion_time"],
            source_system=governance["source_system"],
            acl_label=governance["acl_label"],
        )
    if len(text) < min_chars:
        return CorpusItem(
            source=source,
            doc_id=None,
            kind=KIND_TEXT,
            status="too_short",
            n_chars=len(text),
            source_sha256=source_sha256,
            error=f"text shorter than {min_chars} chars",
            language=governance["language"],
            version=governance["version"],
            effective_date=governance["effective_date"],
            ingestion_time=governance["ingestion_time"],
            source_system=governance["source_system"],
            acl_label=governance["acl_label"],
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
        language=governance["language"],
        version=governance["version"],
        effective_date=governance["effective_date"],
        ingestion_time=governance["ingestion_time"],
        source_system=governance["source_system"],
        acl_label=governance["acl_label"],
    )
