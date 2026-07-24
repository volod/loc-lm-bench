"""The merge plan behind an incremental refresh: what to re-chunk, what to reuse, in which order.

`refresh_vector_store` (`llb.rag.refresh.store_refresh`) owns the diff, the embedder, and the
published generation; this module owns the content. It chunks the changed documents, interleaves
their fresh units with the kept ones in the exact from-scratch build order, decides per row
whether an embedding can be reused, re-applies duplicate collapse over the merged corpus state,
and assembles the merged embedding matrix. Keeping it separate is what makes "a refreshed store
equals a rebuild" a property of one readable unit.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.core.contracts.rag import ChunkRecord, RagStoreMeta
from llb.rag.chunking.corpus import chunk_corpus, iter_doc_paths
from llb.rag.duplicate_tiers import TIER_EXACT
from llb.rag.duplicates import DuplicateStats, collapse_duplicate_chunks, duplicate_stats
from llb.rag.page_metadata import annotate_page_metadata
from llb.rag.refresh.lexical_merge import MergeEntry
from llb.rag.store_build import _build_children

MODE_PARENT_CHILD = "parent_child"


@dataclass
class MergedUnits:
    """The merged build-order store content plus the per-row reuse plan."""

    indexed: list[ChunkRecord]
    parents: list[ChunkRecord] | None
    # per indexed row: the stored embedding row to reuse, or None for a fresh unit to embed
    row_sources: list[int | None]
    duplicates: DuplicateStats | None = None  # measured after the merge, None when not collapsed
    # kept rows whose embedding was recovered by TEXT (a changed doc re-introduced text the store
    # already held), i.e. the encoder calls the text-keyed reuse saved beyond the position map.
    text_reused: int = 0

    @property
    def new_units(self) -> list[ChunkRecord]:
        return [u for u, src in zip(self.indexed, self.row_sources) if src is None]

    def lexical_entries(self) -> list[MergeEntry]:
        return [
            src if src is not None else str(unit["text"])
            for unit, src in zip(self.indexed, self.row_sources)
        ]


def group_by_doc(records: list[ChunkRecord]) -> dict[str, list[int]]:
    """Build-order ordinals per doc_id, order preserved."""
    out: dict[str, list[int]] = {}
    for ordinal, record in enumerate(records):
        out.setdefault(str(record["doc_id"]), []).append(ordinal)
    return out


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


def _annotation_only_sources(
    fresh: list[ChunkRecord], old_chunks: list[ChunkRecord], old_ordinals: list[int]
) -> list[int | None]:
    """Row sources for one changed doc's fresh units: old ordinals when the diff is
    annotation-only, fresh embeds otherwise.

    A modified document whose re-chunked `(char_start, char_end, text)` grid reproduces the
    stored one exactly (sidecar-driven page-span regeneration, governance-only manifest
    changes) has embedding rows unchanged by construction: the fresh records replace the
    stored ones (carrying the re-annotated metadata) while every embedding row is reused.
    """
    if len(fresh) != len(old_ordinals):
        return [None] * len(fresh)
    spans_unchanged = all(
        unit["char_start"] == old_chunks[ordinal]["char_start"]
        and unit["char_end"] == old_chunks[ordinal]["char_end"]
        and unit["text"] == old_chunks[ordinal]["text"]
        for unit, ordinal in zip(fresh, old_ordinals)
    )
    return list(old_ordinals) if spans_unchanged else [None] * len(fresh)


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
    old_ordinals = group_by_doc(old_chunks)
    old_parent_ordinals = group_by_doc(old_parents) if old_parents is not None else {}
    indexed: list[ChunkRecord] = []
    parents: list[ChunkRecord] | None = [] if new_parents_by_doc is not None else None
    row_sources: list[int | None] = []
    for doc_id in iter_doc_paths(corpus_root):
        if doc_id in changed:
            fresh = new_by_doc.get(doc_id, [])
            indexed.extend(fresh)
            if modified and doc_id in modified:
                sources = _annotation_only_sources(fresh, old_chunks, old_ordinals.get(doc_id, []))
            else:
                sources = [None] * len(fresh)
            row_sources.extend(sources)
            if parents is not None and new_parents_by_doc is not None:
                parents.extend(new_parents_by_doc.get(doc_id, []))
            continue
        for ordinal in old_ordinals.get(doc_id, []):
            indexed.append(old_chunks[ordinal])
            row_sources.append(ordinal)
        if parents is not None and old_parents is not None:
            parents.extend(old_parents[ordinal] for ordinal in old_parent_ordinals.get(doc_id, []))
    return MergedUnits(indexed=indexed, parents=parents, row_sources=row_sources)


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
