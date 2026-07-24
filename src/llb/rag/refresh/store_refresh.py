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
from llb.core.contracts.rag import RagStoreMeta
from llb.core.store_generations import (
    generation_timestamp,
    new_generation_paths,
    publish_generation,
    resolve_store_dir,
)
from llb.prep.corpus_governance import corpus_doc_fingerprints, corpus_fingerprint
from llb.rag.duplicate_tiers import TIER_EXACT
from llb.rag.duplicates import expand_duplicate_chunks
from llb.rag.lexical import Lemmatizer, LexicalIndex
from llb.rag.refresh.diff import ManifestDiff, diff_fingerprints
from llb.rag.refresh.lexical_merge import merge_lexical_index
from llb.rag.refresh.merge import (
    MODE_PARENT_CHILD,
    MergedUnits,
    assemble,
    chunk_changed_docs,
    merged_vectors,
    resolve_duplicates,
    text_row_map,
)
from llb.rag.store import RagStore
from llb.rag.store_build import (
    CHUNKS_FILE,
    LEXICAL_FILE,
    META_FILE,
    MODE_HYBRID,
    PARENTS_FILE,
)
from llb.rag.store_io import _read_jsonl
from llb.rag.vector_index import RAG_BACKEND_FAISS, build_vector_index, load_vector_index

_LOG = logging.getLogger(__name__)


@dataclass
class VectorRefreshResult:
    """Outcome of one vector-store refresh (`refreshed=False` == corpus unchanged, no-op)."""

    diff: ManifestDiff
    refreshed: bool
    source_dir: Path
    generation_dir: Path | None = None
    n_reused: int = 0
    n_embedded: int = 0
    n_reused_by_text: int = (
        0  # of n_reused, rows a changed doc recovered by text (else re-embedded)
    )
    old_store: RagStore | None = None
    new_store: RagStore | None = None


def stored_vectors(index: Any) -> Any:
    """The build-order embedding matrix persisted by the loaded vector index."""
    vectors = getattr(index, "vectors", None)
    if vectors is None:
        raise SystemExit(
            "[refresh] the loaded vector index does not expose its stored vectors; "
            "rebuild the store once with `llb build-index` to enable refresh"
        )
    return vectors()


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
    merged: MergedUnits,
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
    if merged.duplicates is not None:
        new_meta["duplicates"] = cast(Any, merged.duplicates)
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

    # Duplicate collapse is undone before the per-document merge and re-applied after it, so the
    # merge still sees every document's complete chunk list (`llb.rag.duplicates`).
    merge_chunks, vector_rows = expand_duplicate_chunks(old_chunks)
    new_by_doc, new_parents_by_doc = chunk_changed_docs(corpus_root, diff.changed, meta, embedder)
    # Recover a changed doc's fresh row from any stored chunk with the same text -- valid only
    # where the vector is a pure function of the text (`late` pools document context, so its rows
    # cannot be reused across documents and it re-encodes each changed doc instead).
    text_rows = None if str(meta.get("strategy")) == "late" else text_row_map(old_chunks)
    merged = resolve_duplicates(
        assemble(
            corpus_root,
            diff.changed,
            merge_chunks,
            old_parents,
            new_by_doc,
            new_parents_by_doc,
            modified=set(diff.modified),
        ),
        vector_rows,
        collapse=bool(meta.get("collapse_duplicates", True)),
        text_rows=text_rows,
        tier=str(meta.get("duplicate_tier", TIER_EXACT)),
    )
    if not merged.indexed:
        raise SystemExit(f"[refresh] no chunks produced from corpus at {corpus_root}")
    vectors = merged_vectors(stored_vectors(old_index), merged, meta, corpus_root, embedder)
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
        "[refresh] %s -> %s: %s; %d rows reused (%d recovered by text), %d embedded",
        live_dir,
        generation_dir,
        diff.summary(),
        len(merged.row_sources) - n_embedded,
        merged.text_reused,
        n_embedded,
    )
    return VectorRefreshResult(
        diff=diff,
        refreshed=True,
        source_dir=live_dir,
        generation_dir=generation_dir,
        n_reused=len(merged.row_sources) - n_embedded,
        n_embedded=n_embedded,
        n_reused_by_text=merged.text_reused,
        old_store=old_store,
        new_store=new_store,
    )
