"""Chunk-metadata filter seam: restrict retrieval candidates BEFORE fusion/ranking.

A filter is any `ChunkRecord -> bool` predicate; `RagStore.retrieve` applies it to the indexed
units before RRF fusion (hybrid) or before the top-k cut (dense-only), so a scoped query never
surfaces an out-of-scope chunk. `metadata_filter` builds the standard predicate over the fields
every chunk already carries: `doc_id` plus the section breadcrumb (`metadata.headers`) and PDF
page range (`metadata.pages`) that the page-metadata join attaches. Task 17's ACL label applies
through this same seam.
"""

from collections.abc import Callable

from llb.core.contracts import ChunkRecord, JsonObject

ChunkFilter = Callable[[ChunkRecord], bool]


def metadata_filter(
    doc_ids: set[str] | None = None,
    heading_contains: str | None = None,
    page_range: tuple[int, int] | None = None,
) -> ChunkFilter:
    """Predicate matching chunks by document, enclosing-heading substring, and/or page overlap.

    All given conditions must hold (AND). `heading_contains` casefold-matches any level of the
    `metadata.headers` breadcrumb; `page_range` is an inclusive source-PDF page interval that
    must overlap the chunk's `metadata.pages` span. A chunk without the needed metadata field
    fails that condition (a page-scoped query never returns an un-paged chunk).
    """
    needle = heading_contains.casefold() if heading_contains is not None else None

    def accept(chunk: ChunkRecord) -> bool:
        if doc_ids is not None and chunk.get("doc_id") not in doc_ids:
            return False
        meta: JsonObject = chunk.get("metadata") or {}
        if needle is not None:
            headers = meta.get("headers")
            titles = headers.values() if isinstance(headers, dict) else []
            if not any(needle in str(title).casefold() for title in titles):
                return False
        if page_range is not None:
            pages = meta.get("pages")
            if not (isinstance(pages, list) and len(pages) == 2):
                return False
            first, last = int(pages[0]), int(pages[1])
            if last < page_range[0] or first > page_range[1]:
                return False
        return True

    return accept
