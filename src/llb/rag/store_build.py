"""Focused store build implementation."""

from pathlib import Path
from typing import Any, cast
from llb.core.contracts.rag import ChunkRecord
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.chunking.dispatch import chunk_spans
from llb.rag.duplicates import (
    DuplicateStats,
    collapse_duplicate_chunks,
    duplicate_occurrences,
    duplicate_stats,
)

CHUNKS_FILE = "chunks.jsonl"  # the INDEXED units (children in parent_child mode)

PARENTS_FILE = "parents.jsonl"  # the parent docstore (parent_child mode only)

META_FILE = "store_meta.json"

LEXICAL_FILE = "lexical_index.json"  # BM25 postings beside the vector index (hybrid mode)

MODE_HYBRID = "hybrid"


def _child_parent_ids(child: ChunkRecord) -> list[str]:
    """Every parent a child hit stands for: its own, plus those of its collapsed copies.

    A child that collapsed byte-identical copies is indexed once but belongs to each document
    that carries the text, so a hit must still surface all of their parents -- the same set the
    tied duplicate children returned before collapse, in build order.
    """
    own = child.get("parent_id")
    ids = [own] if own is not None else []
    for occurrence in duplicate_occurrences(child):
        parent_id = occurrence.get("parent_id")
        if parent_id is not None:
            ids.append(parent_id)
    return ids


def _children_to_parents(
    child_hits: list[ChunkRecord], parent_by_id: dict[str, ChunkRecord]
) -> list[ChunkRecord]:
    """Map ranked child hits to their unique parents (preserving rank). Pure + testable."""
    out: list[ChunkRecord] = []
    seen: set[str] = set()
    for child in child_hits:
        for pid in _child_parent_ids(child):
            if pid in seen or pid not in parent_by_id:
                continue
            seen.add(pid)
            parent = cast(ChunkRecord, dict(parent_by_id[pid]))
            parent["retrieval_score"] = child.get("retrieval_score")
            parent["rank"] = len(out) + 1
            child_id = child.get("chunk_id")
            if child_id is not None:
                parent["matched_child_id"] = child_id
            out.append(parent)
    return out


def _build_children(
    parents: list[ChunkRecord],
    strategy: str,
    child_size: int,
    overlap: int,
    embedder: Any,
) -> list[ChunkRecord]:
    sem = embedder if strategy == "semantic" else None
    children: list[ChunkRecord] = []
    for parent in parents:
        text = parent["text"]
        for j, (start, end, meta) in enumerate(
            chunk_spans(text, strategy, child_size, overlap, sem)
        ):
            metadata = {**(parent.get("metadata") or {}), **(meta or {})}
            children.append(
                {
                    "doc_id": parent["doc_id"],
                    "chunk_id": f"{parent['chunk_id']}::c{j:03d}",
                    "char_start": parent["char_start"] + start,
                    "char_end": parent["char_start"] + end,
                    "text": text[start:end],
                    "parent_id": parent["chunk_id"],
                    "strategy": strategy,
                    "size": child_size,
                    "metadata": metadata,
                }
            )
    return children


def order_by_score(
    candidates: list[tuple[int, float]], chunks: list[ChunkRecord]
) -> list[tuple[int, float]]:
    """Sort `(chunk index, score)` candidates best-first, breaking EXACT ties on `chunk_id`.

    A vector backend returns tied candidates in whatever order it stored them, so any exact tie
    -- byte-identical chunks that survive collapse, or a backend that rounds its scores -- makes
    the reported ranking depend on build order rather than on the data. Breaking the tie on the
    stable `chunk_id` (then on the backend's own position) makes it documented and reproducible
    across rebuilds and across backends.
    """
    return sorted(candidates, key=lambda pair: (-pair[1], _tie_break_key(chunks[pair[0]], pair[0])))


def _tie_break_key(chunk: ChunkRecord, position: int) -> tuple[str, int]:
    return str(chunk.get("chunk_id", "")), position


def _validate_build_params(mode: str, strategy: str, child_size: int) -> None:
    """Reject retrieval mode / strategy / child-size combinations a store cannot support."""
    if mode not in ("flat", "parent_child", MODE_HYBRID):
        raise ValueError(f"unknown retrieval mode: {mode}")
    if strategy == "late" and mode == "parent_child":
        raise ValueError(
            "the 'late' strategy supports flat mode only (children re-chunk parent "
            "slices, so their vectors could not pool over whole-document tokens)"
        )
    if child_size <= 0:
        raise ValueError("child_size must be > 0")


def _indexed_units(
    corpus_root: Path,
    strategy: str,
    size: int,
    overlap: int,
    mode: str,
    child_size: int,
    embedder: Any,
    collapse_duplicates: bool = True,
) -> tuple[list[ChunkRecord], list[ChunkRecord] | None, DuplicateStats]:
    """(indexed, parents, duplicates): units to embed, the parent docstore, and the duplicate rate.

    With `collapse_duplicates` the INDEXED units carry one record per distinct text (see
    `llb.rag.duplicates`); the parent docstore is never collapsed, because a parent is returned as
    generation context for its own document. The stats are measured either way, so a store that
    keeps its duplicates still reports what it is spending on them.
    """
    sem = embedder if strategy == "semantic" else None
    units = chunk_corpus(corpus_root, strategy, size, overlap, sem)
    if not units:
        raise ValueError(f"no chunks produced from corpus at {corpus_root}")
    if mode == "parent_child":
        units_or_children = _build_children(units, strategy, child_size, overlap, embedder)
        if not units_or_children:
            raise ValueError("parent_child mode produced no child chunks")
        parents: list[ChunkRecord] | None = units
    else:
        units_or_children, parents = units, None
    if not collapse_duplicates:
        return units_or_children, parents, duplicate_stats(units_or_children)
    collapse = collapse_duplicate_chunks(units_or_children)
    return collapse.chunks, parents, collapse.stats
