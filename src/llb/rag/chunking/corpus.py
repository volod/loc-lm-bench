"""Build `ChunkRecord`s from a corpus: per-document chunking, governance merge, and summaries."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llb.conflicts.overlay import apply_to_chunks, directives_by_doc, load_applied_overlay
from llb.core.contracts.rag import ChunkRecord, ChunkSummary
from llb.prep.corpus_governance import manifest_governance_by_doc
from llb.rag.chunking.dispatch import chunk_spans
from llb.rag.chunking.structure import doc_page_spans


def chunk_text(
    text: str,
    doc_id: str,
    strategy: str,
    size: int,
    overlap: int,
    embedder: Any = None,
    page_spans: list[tuple[int, int]] | None = None,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    spans = chunk_spans(text, strategy, size, overlap, embedder, page_spans=page_spans)
    for k, (start, end, meta) in enumerate(spans):
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}#{strategy}#{k:04d}",
                "char_start": start,
                "char_end": end,
                "text": text[start:end],
                "strategy": strategy,
                "size": size,
                "overlap": overlap,
                "metadata": meta,
            }
        )
    return chunks


def iter_doc_paths(corpus_root: Path) -> Iterator[str]:
    """Corpus doc ids (relative paths) in the canonical sorted build order, without reading."""
    root = Path(corpus_root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".txt", ".md"):
            yield str(path.relative_to(root))


def iter_docs(corpus_root: Path) -> Iterator[tuple[str, str]]:
    root = Path(corpus_root)
    for doc_id in iter_doc_paths(root):
        yield doc_id, (root / doc_id).read_text(encoding="utf-8")


def chunk_corpus(
    corpus_root: Path,
    strategy: str,
    size: int,
    overlap: int,
    embedder: Any = None,
    only_docs: set[str] | None = None,
) -> list[ChunkRecord]:
    """Chunk the corpus (or, with `only_docs`, just those doc_ids -- the refresh subset path)."""
    chunks: list[ChunkRecord] = []
    governance_by_doc = manifest_governance_by_doc(corpus_root)
    overlay_by_doc = directives_by_doc(load_applied_overlay(corpus_root))
    for doc_id, text in iter_docs(corpus_root):
        if only_docs is not None and doc_id not in only_docs:
            continue
        page_spans = doc_page_spans(corpus_root, doc_id) if strategy == "page" else None
        doc_chunks = chunk_text(
            text, doc_id, strategy, size, overlap, embedder, page_spans=page_spans
        )
        governance = governance_by_doc.get(doc_id)
        if governance:
            for chunk in doc_chunks:
                chunk["metadata"] = {**(chunk.get("metadata") or {}), **governance}
        chunks.extend(apply_to_chunks(doc_chunks, overlay_by_doc.get(doc_id)))
    return chunks


def summarize(chunks: list[ChunkRecord]) -> ChunkSummary:
    sizes = [c["char_end"] - c["char_start"] for c in chunks]
    n = len(sizes)
    return {
        "n": n,
        "avg": sum(sizes) // n if n else 0,
        "min": min(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0,
    }
