"""Incremental vector-store refresh: re-chunk/re-embed changed documents only.

`refresh_vector_store` diffs the store's recorded per-doc fingerprints against the current
corpus, keeps every unchanged document's chunk records and embedding rows verbatim, chunks and
embeds only added/modified documents, and drops deleted ones. A modified document whose
re-chunked span grid matches the stored one exactly (an annotation-only diff: sidecar page-span
regeneration, governance-only manifest changes) rewrites its chunk records but reuses every
embedding row instead of re-embedding. The merged store preserves the
exact from-scratch build order (documents in sorted corpus order, chunks in per-doc order), so
the refreshed store is identical to a rebuild on the same corpus state -- for the dense index
(every `VectorIndex` backend rebuilds exactly from the merged matrix), the lexical BM25 side
(token counts of kept chunks are recovered from the old postings), and the persisted records.

The refreshed store is published as a new immutable generation under
``<index-dir>/generations/<utc-timestamp>/`` (see `llb.core.store_generations`); the source
store is never touched, so deleting the new generation is the rollback.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from llb.core.config_validation import DEFAULT_EMBEDDING_MODEL
from llb.core.contracts.rag import ChunkRecord, RagStoreMeta
from llb.core.store_generations import (
    generation_timestamp,
    new_generation_paths,
    publish_generation,
    resolve_store_dir,
)
from llb.prep.corpus_governance import corpus_doc_fingerprints, corpus_fingerprint
from llb.rag.chunking.corpus import chunk_corpus, iter_doc_paths
from llb.rag.lexical import Lemmatizer, LexicalIndex
from llb.rag.page_metadata import annotate_page_metadata
from llb.rag.refresh.diff import ManifestDiff, diff_fingerprints
from llb.rag.refresh.lexical_merge import MergeEntry, merge_lexical_index
from llb.rag.store import RagStore
from llb.rag.store_build import (
    CHUNKS_FILE,
    LEXICAL_FILE,
    META_FILE,
    MODE_HYBRID,
    PARENTS_FILE,
    _build_children,
)
from llb.rag.store_io import _read_jsonl
from llb.rag.vector_index import RAG_BACKEND_FAISS, build_vector_index, load_vector_index

_LOG = logging.getLogger(__name__)

MODE_PARENT_CHILD = "parent_child"


@dataclass
class VectorRefreshResult:
    """Outcome of one vector-store refresh (`refreshed=False` == corpus unchanged, no-op)."""

    diff: ManifestDiff
    refreshed: bool
    source_dir: Path
    generation_dir: Path | None = None
    n_reused: int = 0
    n_embedded: int = 0
    old_store: RagStore | None = None
    new_store: RagStore | None = None


@dataclass
class _MergedUnits:
    """The merged build-order store content plus the per-row reuse plan."""

    indexed: list[ChunkRecord]
    parents: list[ChunkRecord] | None
    # per indexed row: the old build-order ordinal to reuse, or None for a fresh unit to embed
    row_sources: list[int | None]

    @property
    def new_units(self) -> list[ChunkRecord]:
        return [u for u, src in zip(self.indexed, self.row_sources) if src is None]

    def lexical_entries(self) -> list[MergeEntry]:
        return [
            src if src is not None else str(unit["text"])
            for unit, src in zip(self.indexed, self.row_sources)
        ]


def stored_vectors(index: Any) -> Any:
    """The build-order embedding matrix persisted by the loaded vector index."""
    vectors = getattr(index, "vectors", None)
    if vectors is None:
        raise SystemExit(
            "[refresh] the loaded vector index does not expose its stored vectors; "
            "rebuild the store once with `llb build-index` to enable refresh"
        )
    return vectors()


def _group_by_doc(records: list[ChunkRecord]) -> dict[str, list[int]]:
    """Build-order ordinals per doc_id, order preserved."""
    out: dict[str, list[int]] = {}
    for ordinal, record in enumerate(records):
        out.setdefault(str(record["doc_id"]), []).append(ordinal)
    return out


def _chunk_changed_docs(
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


def _assemble(
    corpus_root: Path,
    changed: set[str],
    old_chunks: list[ChunkRecord],
    old_parents: list[ChunkRecord] | None,
    new_by_doc: dict[str, list[ChunkRecord]],
    new_parents_by_doc: dict[str, list[ChunkRecord]] | None,
    modified: set[str] | None = None,
) -> _MergedUnits:
    """Interleave kept and fresh units in the exact from-scratch build order.

    `modified` names the changed docs eligible for the annotation-only fast path (the diff's
    modified class); added docs and legacy full refreshes always embed fresh rows.
    """
    old_ordinals = _group_by_doc(old_chunks)
    old_parent_ordinals = _group_by_doc(old_parents) if old_parents is not None else {}
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
    return _MergedUnits(indexed=indexed, parents=parents, row_sources=row_sources)


def _merged_vectors(
    old_vectors: Any,
    merged: _MergedUnits,
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


def _load_old_lexical(live_dir: Path, meta: RagStoreMeta) -> LexicalIndex | None:
    if str(meta.get("mode")) != MODE_HYBRID:
        return None
    lexical_path = live_dir / LEXICAL_FILE
    if not lexical_path.is_file():
        raise SystemExit(
            f"[refresh] the hybrid store at {live_dir} is missing its lexical index "
            f"({LEXICAL_FILE}); rebuild it with `build-index --retrieval-mode hybrid`."
        )
    return LexicalIndex.load(lexical_path)


def _refreshed_meta(
    meta: RagStoreMeta,
    merged: _MergedUnits,
    vectors: Any,
    lexical: LexicalIndex | None,
    current_fingerprints: dict[str, str],
    corpus_root: Path,
    live_dir: Path,
) -> RagStoreMeta:
    n = len(merged.indexed)
    coverage = sum(1 for u in merged.indexed if (u.get("metadata") or {}).get("pages")) / n
    new_meta = cast(
        RagStoreMeta,
        {
            **meta,
            "n_indexed": n,
            "n_parents": len(merged.parents) if merged.parents else 0,
            "dim": int(vectors.shape[1]),
            "page_annotation_coverage": round(coverage, 4),
            "corpus_fingerprint": corpus_fingerprint(corpus_root),
            "doc_fingerprints": current_fingerprints,
            "refreshed_from": str(live_dir),
        },
    )
    if lexical is not None:
        new_meta["lexical"] = {"lemmatize": lexical.lemmatize, "n_terms": len(lexical.postings)}
    return new_meta


def refresh_vector_store(
    index_dir: Path | str,
    corpus_root: Path | str,
    *,
    embedder: Any = None,
    lemmatizer: Lemmatizer | None = None,
    timestamp: str | None = None,
) -> VectorRefreshResult:
    """Diff the corpus against the live store and publish an incrementally refreshed generation.

    `embedder` and `lemmatizer` inject fakes for tests; by default the store's recorded
    embedding model is loaded (only once changes exist). Returns a no-op result when the corpus
    fingerprints match the store.
    """
    corpus_root = Path(corpus_root)
    base_dir = Path(index_dir)
    live_dir = resolve_store_dir(base_dir, META_FILE)
    meta_path = live_dir / META_FILE
    if not meta_path.is_file():
        raise SystemExit(
            f"[refresh] no RAG store at {base_dir}; build one first with `llb build-index`"
        )
    meta = cast(RagStoreMeta, json.loads(meta_path.read_text(encoding="utf-8")))
    current = corpus_doc_fingerprints(corpus_root)
    if not current:
        raise SystemExit(f"[refresh] no documents found in corpus at {corpus_root}")
    recorded_raw = meta.get("doc_fingerprints")
    if not isinstance(recorded_raw, dict):
        _LOG.warning(
            "[refresh] store at %s records no doc_fingerprints (built before refresh support); "
            "treating every document as changed",
            live_dir,
        )
        recorded_raw = {}
    recorded = {str(doc_id): str(fp) for doc_id, fp in recorded_raw.items()}
    diff = diff_fingerprints(recorded, current)
    if recorded and not diff.has_changes:
        return VectorRefreshResult(diff=diff, refreshed=False, source_dir=live_dir)

    old_chunks = _read_jsonl(live_dir / CHUNKS_FILE)
    mode = str(meta.get("mode", "flat"))
    old_parents = _read_jsonl(live_dir / PARENTS_FILE) if mode == MODE_PARENT_CHILD else None
    backend = str(meta.get("backend", RAG_BACKEND_FAISS))
    old_index = load_vector_index(backend, live_dir)
    old_lexical = _load_old_lexical(live_dir, meta)
    if embedder is None:
        from llb.rag.embedding import Embedder

        embedder = Embedder(str(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL)))

    new_by_doc, new_parents_by_doc = _chunk_changed_docs(corpus_root, diff.changed, meta, embedder)
    merged = _assemble(
        corpus_root,
        diff.changed,
        old_chunks,
        old_parents,
        new_by_doc,
        new_parents_by_doc,
        modified=set(diff.modified),
    )
    if not merged.indexed:
        raise SystemExit(f"[refresh] no chunks produced from corpus at {corpus_root}")
    vectors = _merged_vectors(stored_vectors(old_index), merged, meta, corpus_root, embedder)
    index = build_vector_index(backend, vectors)
    lexical = (
        merge_lexical_index(old_lexical, merged.lexical_entries(), lemmatizer)
        if old_lexical is not None
        else None
    )
    new_meta = _refreshed_meta(meta, merged, vectors, lexical, current, corpus_root, live_dir)

    new_store = RagStore(
        merged.indexed, index, embedder, new_meta, parents=merged.parents, lexical=lexical
    )
    staging_dir, final_dir = new_generation_paths(base_dir, timestamp or generation_timestamp())
    new_store.save(staging_dir)
    generation_dir = publish_generation(staging_dir, final_dir)
    old_store = RagStore(
        old_chunks, old_index, embedder, meta, parents=old_parents, lexical=old_lexical
    )
    n_embedded = sum(1 for src in merged.row_sources if src is None)
    _LOG.info(
        "[refresh] %s -> %s: %s; %d rows reused, %d embedded",
        live_dir,
        generation_dir,
        diff.summary(),
        len(merged.row_sources) - n_embedded,
        n_embedded,
    )
    return VectorRefreshResult(
        diff=diff,
        refreshed=True,
        source_dir=live_dir,
        generation_dir=generation_dir,
        n_reused=len(merged.row_sources) - n_embedded,
        n_embedded=n_embedded,
        old_store=old_store,
        new_store=new_store,
    )
