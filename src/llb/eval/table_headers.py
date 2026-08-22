"""Opt-in, prompt-side restoration of a table chunk's header row (table-header-context-restoration).

The `table` chunker records every table chunk's header-row SOURCE offsets in
`metadata.table_header_span` and never copies the header text into a later block, because chunk
text must stay a verbatim corpus slice for the source-span metrics
(`llb.rag.chunking.table`). The consequence at generation time is that a middle row block reaches
the model as rows of bare values whose column names sit in a different chunk -- exactly the shape a
numeric or comparative question cannot be answered from.

This module restores those column names IN THE PROMPT ONLY. It returns COPIES of the retrieved
chunks whose `text` carries the header row on top; the stored chunk, its `char_start` /
`char_end`, and every retrieval metric read from them stay untouched, so retrieval recall@k / MRR
are unchanged by construction and only answer quality can move.

The header text is read from the corpus the offsets refer to, and every read is guarded: a chunk is
restored only when the resolved document reproduces that chunk's own text at its own offsets. A
corpus that has drifted from the store (or is simply absent) therefore restores NOTHING rather than
prepending unrelated text.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from llb.core.contracts.rag import ChunkRecord
from llb.eval import common as eval_common
from llb.eval.graph_contracts import RagState
from llb.rag.chunking.table import TABLE_HEADER_SPAN_KEY

logger = logging.getLogger(__name__)

# The header row is laid on its own line above the rows it labels. A row block is not a well-formed
# GFM table either way (the delimiter line is not part of the recorded span), so the restoration
# adds the recorded header row and nothing invented around it.
HEADER_SEPARATOR = "\n"

# One resolver call per doc_id; `None` means "this document is not readable here", which disables
# restoration for its chunks instead of guessing.
DocTextResolver = Callable[[str], str | None]


class HeaderRestoration(NamedTuple):
    """Prompt chunks plus what restoring their headers added.

    `chunks` is the list handed to `format_context`; it is the INPUT list itself when nothing was
    restored, so the caller can tell a no-op apart with `is`.
    """

    chunks: list[ChunkRecord]
    restored: int
    added_chars: int


HeaderRestorer = Callable[[list[ChunkRecord]], HeaderRestoration]


def header_span(chunk: ChunkRecord) -> tuple[int, int] | None:
    """The recorded `[start, end]` header-row offsets of a table chunk, or None."""
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        return None
    span = metadata.get(TABLE_HEADER_SPAN_KEY)
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return None
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError):
        return None
    return (start, end) if 0 <= start < end else None


def _chunk_reproduces(doc: str, chunk: ChunkRecord) -> bool:
    """True when `doc` carries this chunk's own text at this chunk's own offsets.

    The one guard that makes reading a header by offset safe: it fails whenever the resolved
    document is not the one the chunk was cut from (a drifted corpus, a governance overlay that
    rewrote the text, a chunk whose text was assembled rather than sliced).
    """
    start, end = int(chunk.get("char_start", 0)), int(chunk.get("char_end", 0))
    return doc[start:end] == chunk.get("text", "")


def _already_carries(chunk: ChunkRecord, span: tuple[int, int], header: str) -> bool:
    """True when this chunk already shows the header -- by span, or by text for a repeated one."""
    start, end = int(chunk.get("char_start", 0)), int(chunk.get("char_end", 0))
    if start <= span[0] and span[1] <= end:
        return True
    return header in str(chunk.get("text", ""))


def restored_chunk(chunk: ChunkRecord, doc: str) -> ChunkRecord | None:
    """A copy of `chunk` with its header row prepended, or None when the rule does not fire.

    The rule does not fire when the chunk records no header span, when the document does not
    reproduce the chunk, when the recorded span resolves to blank text, or when the chunk already
    carries the header.
    """
    span = header_span(chunk)
    if span is None or not _chunk_reproduces(doc, chunk):
        return None
    header = doc[span[0] : span[1]]
    if not header.strip() or _already_carries(chunk, span, header):
        return None
    return {**chunk, "text": f"{header}{HEADER_SEPARATOR}{chunk.get('text', '')}"}


def restore_headers(chunks: list[ChunkRecord], doc_text: DocTextResolver) -> HeaderRestoration:
    """Prompt copies of `chunks` carrying their table headers, plus the character cost.

    Pure with respect to the input: no chunk is mutated, and the input list is returned unchanged
    when no chunk was restored.
    """
    out: list[ChunkRecord] = []
    restored = 0
    added = 0
    docs: dict[str, str | None] = {}
    for chunk in chunks:
        candidate = None
        if header_span(chunk) is not None:
            doc_id = str(chunk.get("doc_id", ""))
            if doc_id not in docs:
                docs[doc_id] = doc_text(doc_id)
            doc = docs[doc_id]
            candidate = restored_chunk(chunk, doc) if doc is not None else None
        if candidate is None:
            out.append(chunk)
            continue
        restored += 1
        added += len(candidate["text"]) - len(str(chunk.get("text", "")))
        out.append(candidate)
    if not restored:
        return HeaderRestoration(chunks, 0, 0)
    return HeaderRestoration(out, restored, added)


def corpus_doc_text(corpus_root: Path) -> DocTextResolver:
    """Resolve `doc_id` -> document text under `corpus_root`, reading each document at most once.

    A missing or unreadable document resolves to None, which disables restoration for its chunks.
    """
    root = Path(corpus_root)
    cache: dict[str, str | None] = {}

    def read(doc_id: str) -> str | None:
        if doc_id not in cache:
            path = root / doc_id
            try:
                cache[doc_id] = path.read_text(encoding="utf-8")
            except OSError:
                logger.warning(
                    "[table-headers] no readable corpus document for %s under %s; "
                    "header restoration is off for its chunks",
                    doc_id,
                    root,
                )
                cache[doc_id] = None
        return cache[doc_id]

    return read


def corpus_header_restorer(corpus_root: Path) -> HeaderRestorer:
    """The restorer used by a run: header text read from the corpus the store was built from."""
    doc_text = corpus_doc_text(corpus_root)

    def restore(chunks: list[ChunkRecord]) -> HeaderRestoration:
        return restore_headers(chunks, doc_text)

    return restore


def prompt_context(
    chunks: list[ChunkRecord],
    order: str = eval_common.ORDER_RANK,
    restorer: HeaderRestorer | None = None,
) -> RagState:
    """The prompt-side half of one retrieval: the rendered context plus its assembly accounting.

    `retrieved` is deliberately NOT in the returned update -- this is what the model reads, and the
    caller keeps the stored records separately. The accounting columns are always present (0 / 0
    with no restorer), so a lane with the step off and a lane with it on carry the same measured
    column; `prompt_chunks` appears only when assembly actually changed something.
    """
    restoration = restorer(chunks) if restorer is not None else HeaderRestoration(chunks, 0, 0)
    update: RagState = {
        "context": eval_common.format_context(restoration.chunks, order=order),
        "table_headers_restored": restoration.restored,
        "table_header_chars": restoration.added_chars,
    }
    if restoration.chunks is not chunks:
        update["prompt_chunks"] = restoration.chunks
    return update


__all__ = [
    "DocTextResolver",
    "HEADER_SEPARATOR",
    "HeaderRestoration",
    "HeaderRestorer",
    "corpus_doc_text",
    "corpus_header_restorer",
    "header_span",
    "prompt_context",
    "restore_headers",
    "restored_chunk",
]
