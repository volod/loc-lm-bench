"""Stitch CONTIGUOUS retrieved chunks back into one block (assembly-time evidence reflow).

The intactness pair (`llb.rag.retrieval`) can report that a gold span ARRIVED -- every character of
it is somewhere in the top-k -- while no single chunk carries it whole, which is what a model
reassembling a procedure from fragments actually reads. Two levers can convert fragments into whole
spans: raise `size` (a different index, a different retrieval), or reflow what was already
retrieved. This module is the second one.

The rule is deliberately narrow, because the whole value of the lever is that it CANNOT change what
was retrieved: two retrieved chunks merge only when they belong to the same document and their
character ranges touch or overlap. A gap is never bridged (that would serve text nobody retrieved),
nothing is reordered (a block sits at its best-ranked part's position), and no text is invented (the
merged text is the parts' own text with the overlap counted once). So on any lane, `recall@k` and
`span_char_coverage_at_k` are invariant by construction -- the retrieved character UNION is
identical -- and only `span_intact_at_k` can move, upward, when a merge closes a cut span.

Two things a merge is refused for, both about staying exactly reversible:

- a chunk whose `text` length disagrees with its own offsets (a governance overlay may rewrite chunk
  text), because the merged text could then no longer be laid back onto the source offsets;
- a chunk that collapsed byte-identical copies (`llb.rag.duplicates.collapse`), because its recorded
  occurrences describe THAT text at other places and a merged block appears at none of them.

A refused chunk is still served -- it stays its own block -- so refusing costs intactness, never
evidence.

`mrr` is NOT readable on a stitched lane: merging shortens the list, so the first hit can only move
to an earlier position. Read `span_intact_at_k` for the gain and `served_chars_at_k` for the price.
"""

import time
from typing import Any, Callable, cast

from typing_extensions import TypedDict

from llb.core.contracts.rag import ChunkRecord
from llb.rag.duplicates.collapse import duplicate_occurrences

# Additive metadata key on a merged block: its parts, in source-offset order.
STITCHED_FROM_KEY = "stitched_from"


class StitchCensus(TypedDict):
    """Per-query stitching accounting: how much merged, and what it cost in served characters."""

    queries: float
    parts_per_query: float
    blocks_per_query: float
    merged_per_query: float
    chars_per_query: float
    chars_delta_per_query: float
    stitch_ms_per_query: float


def offsets_exact(chunk: ChunkRecord) -> bool:
    """True when the chunk's text is exactly the source slice its offsets name."""
    return len(str(chunk.get("text", ""))) == int(chunk["char_end"]) - int(chunk["char_start"])


def stitchable(chunk: ChunkRecord) -> bool:
    """True when merging this chunk stays exactly reversible onto the source offsets."""
    return offsets_exact(chunk) and not duplicate_occurrences(chunk)


def served_chars(chunks: list[ChunkRecord]) -> int:
    """Characters this context serves the model (overlapping copies counted as served twice)."""
    return sum(len(str(chunk.get("text", ""))) for chunk in chunks)


def _runs(group: list[tuple[int, ChunkRecord]]) -> list[list[tuple[int, ChunkRecord]]]:
    """Split one document's chunks (offset order) into maximal touching-or-overlapping runs."""
    runs: list[list[tuple[int, ChunkRecord]]] = []
    reach: int | None = None
    for entry in group:
        start, end = int(entry[1]["char_start"]), int(entry[1]["char_end"])
        if reach is None or start > reach:
            runs.append([entry])
        else:
            runs[-1].append(entry)
        reach = end if reach is None else max(reach, end)
    return runs


def _merged_text(run: list[tuple[int, ChunkRecord]]) -> str:
    """The run's own text with overlaps counted once -- never a character it did not retrieve."""
    text = str(run[0][1]["text"])
    reach = int(run[0][1]["char_end"])
    for _, chunk in run[1:]:
        end = int(chunk["char_end"])
        if end <= reach:
            continue
        text += str(chunk["text"])[reach - int(chunk["char_start"]) :]
        reach = end
    return text


def _merge(run: list[tuple[int, ChunkRecord]]) -> ChunkRecord:
    """One block from a run: the best-ranked part's identity over the run's whole character range."""
    base = min(run, key=lambda entry: entry[0])[1]
    block = cast(ChunkRecord, dict(base))
    block["char_start"] = int(run[0][1]["char_start"])
    block["char_end"] = max(int(chunk["char_end"]) for _, chunk in run)
    block["text"] = _merged_text(run)
    metadata = dict(base.get("metadata") or {})
    metadata[STITCHED_FROM_KEY] = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "char_start": int(chunk["char_start"]),
            "char_end": int(chunk["char_end"]),
        }
        for _, chunk in run
    ]
    block["metadata"] = metadata
    return block


def stitch_contiguous(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    """Merge contiguous same-document chunks into blocks, preserving retrieval order.

    Returns the input list ITSELF when nothing merged, so a no-op costs nothing. Otherwise every
    returned record is a copy: merged blocks carry `metadata.stitched_from` and every block is
    renumbered `rank` 1..n, because the pre-stitch rank names a position this list no longer has.
    """
    by_doc: dict[str, list[tuple[int, ChunkRecord]]] = {}
    blocks: list[tuple[int, ChunkRecord]] = []
    for position, chunk in enumerate(chunks):
        if stitchable(chunk):
            by_doc.setdefault(str(chunk["doc_id"]), []).append((position, chunk))
        else:
            blocks.append((position, chunk))
    for group in by_doc.values():
        group.sort(key=lambda entry: (int(entry[1]["char_start"]), int(entry[1]["char_end"])))
        for run in _runs(group):
            best = min(position for position, _ in run)
            blocks.append((best, run[0][1] if len(run) == 1 else _merge(run)))
    if len(blocks) == len(chunks):
        return chunks
    blocks.sort(key=lambda entry: entry[0])
    ranked: list[ChunkRecord] = []
    for rank, (_, block) in enumerate(blocks, 1):
        record = cast(ChunkRecord, dict(block))
        record["rank"] = rank
        ranked.append(record)
    return ranked


class StitchingRetriever:
    """Wrap any RAG-store-contract retriever with a contiguous-stitching step after the top-k cut.

    `retrieve(question, k)` asks the wrapped store for the SAME k it always would and stitches what
    comes back, so the lane retrieves exactly what its base lane retrieves and differs only in how
    many blocks that evidence arrives in. Unknown attributes delegate to the wrapped store, so the
    wrapper drops into every seam a bare store fits, and `census()` reports the per-query blocks and
    served characters the reflow actually produced.
    """

    def __init__(self, store: Any, clock: Callable[[], float] = time.perf_counter):
        self.store = store
        self.n_queries = 0
        self.parts = 0
        self.blocks = 0
        self.chars_in = 0
        self.chars_out = 0
        self.stitch_s = 0.0
        self._clock = clock

    def retrieve(self, question: str, k: int, **kwargs: Any) -> list[ChunkRecord]:
        return self._stitched(self.store.retrieve(question, k, **kwargs))

    def retrieve_queries(
        self, dense_query: str, lexical_query: str, k: int, **kwargs: Any
    ) -> list[ChunkRecord]:
        """Preserve the split dense/lexical path of a hybrid store."""
        if not callable(getattr(self.store, "retrieve_queries", None)):
            return self.retrieve(lexical_query, k, **kwargs)
        return self._stitched(self.store.retrieve_queries(dense_query, lexical_query, k, **kwargs))

    def _stitched(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        started = self._clock()
        blocks = stitch_contiguous(chunks)
        self.stitch_s += self._clock() - started
        self.n_queries += 1
        self.parts += len(chunks)
        self.blocks += len(blocks)
        self.chars_in += served_chars(chunks)
        self.chars_out += served_chars(blocks)
        return blocks

    def census(self) -> StitchCensus:
        """Per-query means (zeros when unused): what merged, and the served-character delta."""
        queries = self.n_queries or 1
        return StitchCensus(
            queries=float(self.n_queries),
            parts_per_query=self.parts / queries,
            blocks_per_query=self.blocks / queries,
            merged_per_query=(self.parts - self.blocks) / queries,
            chars_per_query=self.chars_out / queries,
            chars_delta_per_query=(self.chars_out - self.chars_in) / queries,
            stitch_ms_per_query=self.stitch_s * 1000.0 / queries,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)
