"""Chunk-metadata filter seam: restrict retrieval candidates BEFORE fusion/ranking.

A filter is any `ChunkRecord -> bool` predicate; `RagStore.retrieve` applies it to the indexed
units before RRF fusion (hybrid) or before the top-k cut (dense-only), so a scoped query never
surfaces an out-of-scope chunk. `metadata_filter` builds the standard predicate over the fields
every chunk already carries: `doc_id` plus the section breadcrumb (`metadata.headers`), PDF
page range (`metadata.pages`), and governance ACL label (`metadata.acl_label`).
"""

from collections.abc import Callable

from llb.core.contracts import ChunkRecord, JsonObject

ChunkFilter = Callable[[ChunkRecord], bool]


def metadata_filter(
    doc_ids: set[str] | None = None,
    heading_contains: str | None = None,
    page_range: tuple[int, int] | None = None,
    acl_label: str | None = None,
) -> ChunkFilter:
    """Predicate matching chunks by document, heading substring, page overlap, and/or ACL label.

    All given conditions must hold (AND). `heading_contains` casefold-matches any level of the
    `metadata.headers` breadcrumb; `page_range` is an inclusive source-PDF page interval that
    must overlap the chunk's `metadata.pages` span. `acl_label` matches the governance tag copied
    into `metadata.acl_label`. A chunk without the needed metadata field fails that condition.
    """
    needle = heading_contains.casefold() if heading_contains is not None else None

    def accept(chunk: ChunkRecord) -> bool:
        if doc_ids is not None and chunk.get("doc_id") not in doc_ids:
            return False
        meta: JsonObject = chunk.get("metadata") or {}
        if acl_label is not None and meta.get("acl_label") != acl_label:
            return False
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
