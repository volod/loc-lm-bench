"""Attach page/section provenance to chunks, strategy-independent and additive.

`build-index` chunks a corpus into offset-exact `(doc_id, char_start, char_end)` spans. When a
document was produced by the PDF lane (`llb.prep.pdf_corpus`), a `*.citations.json` sidecar sits
beside it mapping each source-PDF page to a character span in the rendered `.md`. This module
joins the two: every chunk whose char span intersects a page span gains

  - `metadata.pages`      = [first_page, last_page]  (source-PDF page numbers)
  - `metadata.source_pdf` = the original PDF path recorded in the sidecar

and every chunk (regardless of strategy) gains `metadata.headers`, the breadcrumb of enclosing
markdown headings located in the SOURCE text. Chunk ids, text, and offsets are never touched --
only `metadata` grows -- so `validate-goldset` and source-span scoring keep working.

Plain `.md`/`.txt` documents (no sidecar) get header breadcrumbs but no page fields.
"""

import json
import logging
from pathlib import Path

from llb.contracts import ChunkRecord, JsonObject
from llb.prep.pdf_corpus import PDF_CITATION_SUFFIX
from llb.rag.chunking import _MD_HEADER

_LOG = logging.getLogger(__name__)


def citation_sidecar_path(corpus_root: Path, doc_id: str) -> Path:
    """Path of the `*.citations.json` sidecar that would sit beside `doc_id`."""
    return corpus_root / Path(doc_id).with_suffix(PDF_CITATION_SUFFIX)


def load_page_citations(
    corpus_root: Path, doc_id: str
) -> tuple[str | None, list[JsonObject]] | None:
    """Return `(source_pdf, page_spans)` for `doc_id`, or None when no sidecar exists.

    `page_spans` are the sidecar's `pages` entries carrying `page`/`char_start`/`char_end`.
    """
    path = citation_sidecar_path(corpus_root, doc_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _LOG.warning("[page-metadata] unreadable citation sidecar %s; skipping", path)
        return None
    if not isinstance(payload, dict):
        return None
    pages = payload.get("pages")
    spans = [p for p in pages if isinstance(p, dict)] if isinstance(pages, list) else []
    source = payload.get("source")
    return (source if isinstance(source, str) else None), spans


def intersect_pages(start: int, end: int, page_spans: list[JsonObject]) -> list[int]:
    """Source-PDF page numbers whose char span overlaps `[start, end)`, sorted and unique."""
    hits: set[int] = set()
    for span in page_spans:
        page = span.get("page")
        cs = span.get("char_start")
        ce = span.get("char_end")
        if not (isinstance(page, int) and isinstance(cs, int) and isinstance(ce, int)):
            continue
        if cs < end and ce > start:  # half-open overlap
            hits.add(page)
    return sorted(hits)


def heading_breadcrumb(text: str, pos: int) -> dict[str, str]:
    """Breadcrumb of markdown headings enclosing character `pos` (e.g. {"h1": ..., "h2": ...}).

    Mirrors the stack logic in `chunking.markdown_spans`: a deeper heading replaces same/lower
    levels, so the breadcrumb is the last heading seen at each level up to `pos`.
    """
    stack: dict[int, str] = {}
    for m in _MD_HEADER.finditer(text):
        if m.start() > pos:
            break
        level = len(m.group(1))
        stack = {lvl: title for lvl, title in stack.items() if lvl < level}
        stack[level] = m.group(2).strip()
    return {f"h{lvl}": stack[lvl] for lvl in sorted(stack)}


def annotate_page_metadata(records: list[ChunkRecord], corpus_root: Path | str) -> float:
    """Attach page/section provenance to `records` in place; return page-annotation coverage.

    Coverage is the fraction of `records` that received a `pages` field (0.0 for a corpus with no
    PDF sidecars). Each record's `metadata` is replaced with a fresh dict, so records that share a
    metadata object (parent_child children inheriting a parent's dict) never alias after this.
    """
    corpus_root = Path(corpus_root)
    doc_texts: dict[str, str] = {}
    citations: dict[str, tuple[str | None, list[JsonObject]] | None] = {}
    n_paged = 0
    for record in records:
        doc_id = record["doc_id"]
        if doc_id not in citations:
            citations[doc_id] = load_page_citations(corpus_root, doc_id)
        meta: JsonObject = dict(record.get("metadata") or {})
        record["metadata"] = meta

        cite = citations[doc_id]
        if cite is not None:
            source, spans = cite
            pages = intersect_pages(record["char_start"], record["char_end"], spans)
            if pages:
                meta["pages"] = [pages[0], pages[-1]]
                if source is not None:
                    meta["source_pdf"] = source
                n_paged += 1
            else:
                # This record's own span intersects no page. Drop any page fields carried in
                # from an inherited metadata dict (parent_child children start from the parent's
                # metadata) so `pages` always reflects THIS record's span, never the parent's.
                meta.pop("pages", None)
                meta.pop("source_pdf", None)

        if not meta.get("headers"):
            if doc_id not in doc_texts:
                doc_texts[doc_id] = _read_doc_text(corpus_root, doc_id)
            breadcrumb = heading_breadcrumb(doc_texts[doc_id], record["char_start"])
            if breadcrumb:
                meta["headers"] = breadcrumb

    return n_paged / len(records) if records else 0.0


def _read_doc_text(corpus_root: Path, doc_id: str) -> str:
    path = corpus_root / doc_id
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""
