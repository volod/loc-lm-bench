"""Focused store build implementation."""

from pathlib import Path
from typing import Any, cast
from llb.core.contracts.rag import ChunkRecord
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.chunking.dispatch import chunk_spans

CHUNKS_FILE = "chunks.jsonl"  # the INDEXED units (children in parent_child mode)

PARENTS_FILE = "parents.jsonl"  # the parent docstore (parent_child mode only)

META_FILE = "store_meta.json"

LEXICAL_FILE = "lexical_index.json"  # BM25 postings beside the vector index (hybrid mode)

MODE_HYBRID = "hybrid"


def _children_to_parents(
    child_hits: list[ChunkRecord], parent_by_id: dict[str, ChunkRecord]
) -> list[ChunkRecord]:
    """Map ranked child hits to their unique parents (preserving rank). Pure + testable."""
    out: list[ChunkRecord] = []
    seen: set[str] = set()
    for child in child_hits:
        pid = child.get("parent_id")
        if pid is None or pid in seen or pid not in parent_by_id:
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
) -> tuple[list[ChunkRecord], list[ChunkRecord] | None]:
    """(indexed, parents): the units to embed, plus the parent docstore in parent_child mode."""
    sem = embedder if strategy == "semantic" else None
    units = chunk_corpus(corpus_root, strategy, size, overlap, sem)
    if not units:
        raise ValueError(f"no chunks produced from corpus at {corpus_root}")
    if mode != "parent_child":
        return units, None
    children = _build_children(units, strategy, child_size, overlap, embedder)
    if not children:
        raise ValueError("parent_child mode produced no child chunks")
    return children, units
