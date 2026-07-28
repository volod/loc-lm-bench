"""The merge plan behind an incremental refresh: what to re-chunk, what to reuse, in which order.

`refresh_vector_store` (`llb.rag.refresh.store_refresh`) owns the diff, the embedder, and the
published generation; this module owns the content. It chunks the changed documents, interleaves
their fresh units with the kept ones in the exact from-scratch build order, decides per row
whether an embedding can be reused, re-applies duplicate collapse over the merged corpus state,
and assembles the merged embedding matrix. Keeping it separate is what makes "a refreshed store
equals a rebuild" a property of one readable unit.
"""

from pathlib import Path
from typing import Any

from llb.core.contracts.rag import ChunkRecord, RagStoreMeta
from llb.rag.chunking.corpus import chunk_corpus, iter_doc_paths
from llb.rag.duplicate_tiers import TIER_EXACT
from llb.rag.duplicates import (
    collapse_duplicate_chunks,
    duplicate_stats,
)
from llb.rag.page_metadata import annotate_page_metadata
from llb.rag.refresh.merge_assembly import MergeAssemblyBuilder
from llb.rag.refresh.merge_models import MergedUnits
from llb.rag.store_build import _build_children

MODE_PARENT_CHILD = "parent_child"


def chunk_changed_docs(
    corpus_root: Path,
    changed: set[str],
    meta: RagStoreMeta,
    embedder: Any,
) -> tuple[dict[str, list[ChunkRecord]], dict[str, list[ChunkRecord]] | None]:
    """(indexed units, parents) per changed doc, mirroring the from-scratch build path."""
    strategy = str(meta.get("strategy", "recursive"))
    sem = embedder if strategy == "semantic" else None
    units = chunk_corpus(
        corpus_root,
        strategy,
        int(meta.get("size", 800)),
        int(meta.get("overlap", 120)),
        sem,
        only_docs=changed,
    )
    if str(meta.get("mode", "flat")) != MODE_PARENT_CHILD:
        annotate_page_metadata(units, corpus_root)
        return _records_by_doc(units), None
    children = _build_children(
        units, strategy, int(meta.get("child_size", 400)), int(meta.get("overlap", 120)), embedder
    )
    annotate_page_metadata(children, corpus_root)
    annotate_page_metadata(units, corpus_root)
    return _records_by_doc(children), _records_by_doc(units)


def _records_by_doc(records: list[ChunkRecord]) -> dict[str, list[ChunkRecord]]:
    out: dict[str, list[ChunkRecord]] = {}
    for record in records:
        out.setdefault(str(record["doc_id"]), []).append(record)
    return out


def assemble(
    corpus_root: Path,
    changed: set[str],
    old_chunks: list[ChunkRecord],
    old_parents: list[ChunkRecord] | None,
    new_by_doc: dict[str, list[ChunkRecord]],
    new_parents_by_doc: dict[str, list[ChunkRecord]] | None,
    modified: set[str] | None = None,
) -> MergedUnits:
    """Interleave kept and fresh units in the exact from-scratch build order.

    `modified` names the changed docs eligible for the annotation-only fast path (the diff's
    modified class); added docs and legacy full refreshes always embed fresh rows.
    """
    builder = MergeAssemblyBuilder(
        old_chunks,
        old_parents,
        include_parents=new_parents_by_doc is not None,
    )
    for doc_id in iter_doc_paths(corpus_root):
        if doc_id in changed:
            fresh = new_by_doc.get(doc_id, [])
            builder.add_fresh(
                doc_id,
                fresh,
                (new_parents_by_doc.get(doc_id, []) if new_parents_by_doc is not None else []),
                annotation_only=bool(modified and doc_id in modified),
            )
            continue
        builder.add_reused(doc_id)
    return builder.build()


def text_row_map(chunks: list[ChunkRecord]) -> dict[str, int]:
    """`{chunk text -> stored embedding row}` for the live store, first row per text wins.

    Built once from the stored survivors (each `chunks[row]` is the row of `stored_vectors`),
    it holds REFERENCES to the texts already in `chunks`, so its cost is one dict entry per stored
    row -- no copy of the corpus text -- which bounds it on a large store. Collapsed stores have
    distinct survivor texts so no key ever collides; `--keep-duplicate-chunks` stores can repeat a
    text, and first-wins then points at the lowest (survivor-equivalent) row, whose vector is
    identical to every copy's anyway.
    """
    rows: dict[str, int] = {}
    for row, chunk in enumerate(chunks):
        rows.setdefault(str(chunk["text"]), row)
    return rows


def resolve_duplicates(
    merged: MergedUnits,
    vector_rows: list[int | None],
    collapse: bool,
    text_rows: dict[str, int] | None = None,
    tier: str = TIER_EXACT,
) -> MergedUnits:
    """Point the reuse plan back at stored embedding rows, then re-collapse the merged units.

    Expansion (see `refresh_vector_store`) gave every duplicate copy its own record so the
    per-document merge stays exact, at the cost of reuse ordinals that address expanded records
    instead of stored rows; `vector_rows` maps them back. Re-collapsing AFTER the merge is what
    keeps a refreshed store identical to a rebuild even when the document that happened to carry
    a survivor was the one edited or deleted.

    `text_rows` (when supplied) recovers a fresh unit's embedding by its TEXT: the position map
    can only reuse a row a fresh unit inherits from its own stored chunk, so a repeated passage
    whose stored survivor lived in the EDITED document -- or an unchanged chunk of a modified
    document -- re-embeds text the store already holds. Keying the leftover fresh rows on stored
    text reuses that row regardless of which document now carries it. Only valid where a chunk
    vector is a pure function of its text (every strategy but `late`, whose vectors are
    document-contextual), so the caller passes `None` for `late`.

    `tier` re-collapses at the tier the store was built with. Under a coarser tier than `exact`,
    expansion hands back no row for a copy whose text differs from its survivor's, so that copy is
    re-embedded instead of inheriting a vector encoded from another wording.
    """
    rows: list[int | None] = [
        None if source is None else vector_rows[source] for source in merged.row_sources
    ]
    reused: set[int] = set()
    if text_rows:
        for position, (unit, row) in enumerate(zip(merged.indexed, rows)):
            if row is not None:
                continue
            hit = text_rows.get(str(unit["text"]))
            if hit is not None:
                rows[position] = hit
                reused.add(position)
    if not collapse:
        return MergedUnits(
            indexed=merged.indexed,
            parents=merged.parents,
            row_sources=rows,
            duplicates=duplicate_stats(merged.indexed, tier),
            text_reused=len(reused),
        )
    collapsed = collapse_duplicate_chunks(merged.indexed, tier)
    return MergedUnits(
        indexed=collapsed.chunks,
        parents=merged.parents,
        row_sources=[rows[position] for position in collapsed.kept],
        duplicates=collapsed.stats,
        text_reused=sum(1 for position in collapsed.kept if position in reused),
    )


def merged_vectors(
    old_vectors: Any,
    merged: MergedUnits,
    meta: RagStoreMeta,
    corpus_root: Path,
    embedder: Any,
) -> Any:
    """The merged float32 matrix: kept rows from the old index, fresh rows from the embedder."""
    import numpy as np

    new_units = merged.new_units
    new_vectors: Any = None
    if new_units:
        if str(meta.get("strategy")) == "late":
            from llb.rag.late_encoding import encode_store_vectors

            new_vectors = encode_store_vectors(new_units, corpus_root, embedder)
        else:
            new_vectors = embedder.encode_passages([str(u["text"]) for u in new_units])
        new_vectors = np.asarray(new_vectors, dtype="float32")
    old = np.asarray(old_vectors, dtype="float32")
    dim = int(old.shape[1]) if old.size else int(new_vectors.shape[1])
    out = np.empty((len(merged.row_sources), dim), dtype="float32")
    fresh_row = 0
    for row, src in enumerate(merged.row_sources):
        if src is None:
            out[row] = new_vectors[fresh_row]
            fresh_row += 1
        else:
            out[row] = old[src]
    return out
