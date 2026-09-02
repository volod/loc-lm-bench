"""Build the immutable components consumed by :class:`llb.rag.vector_store.store.RagStore`."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.core.config_validation import DEFAULT_EMBEDDING_MODEL
from llb.core.contracts.rag import ChunkRecord, RagStoreMeta
from llb.prep.corpus.fingerprints import corpus_doc_fingerprints, corpus_fingerprint
from llb.prep.corpus.governance_fields import GOVERNANCE_FIELDS
from llb.rag.duplicates.tiers import TIER_EXACT
from llb.rag.duplicates.collapse import collapse_is_lossless
from llb.rag.encoders.embedder import Embedder
from llb.rag.encoders.tuned import embedder_fingerprint
from llb.rag.late_encoding import encode_store_vectors
from llb.rag.vector_store.lexical import Lemmatizer
from llb.rag.vector_store.lexical_index import LexicalIndex
from llb.rag.page_metadata import annotate_page_metadata
from llb.rag.vector_store.build import MODE_HYBRID, _indexed_units, _validate_build_params
from llb.rag.vector_store.vector_index import RAG_BACKEND_FAISS, VectorIndex, build_vector_index

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoreParts:
    chunks: list[ChunkRecord]
    index: VectorIndex
    embedder: Embedder
    meta: RagStoreMeta
    parents: list[ChunkRecord] | None
    lexical: LexicalIndex | None


def build_store_parts(
    corpus_root: Path | str,
    strategy: str = "recursive",
    size: int = 800,
    overlap: int = 120,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    mode: str = "flat",
    child_size: int = 400,
    vector_store: str = RAG_BACKEND_FAISS,
    embedder: Any = None,
    lexical_lemmas: bool = False,
    lemmatizer: Lemmatizer | None = None,
    collapse_duplicates: bool = True,
    duplicate_tier: str = TIER_EXACT,
) -> StoreParts:
    """Chunk, annotate, encode, and index a corpus.

    `collapse_duplicates` is DOWNGRADED to False for a strategy whose vector is not a pure
    function of its text (`collapse_is_lossless`), and the meta records the effective value, so
    an incremental refresh -- which reads `collapse_duplicates` back from the meta -- keeps
    matching a from-scratch rebuild.
    """
    _validate_build_params(mode, strategy, child_size)
    if collapse_duplicates and not collapse_is_lossless(strategy):
        log.info(
            "[rag] strategy %s pools document context -- keeping duplicate chunks "
            "(collapse would discard position-dependent vectors)",
            strategy,
        )
        collapse_duplicates = False
    resolved_embedder = embedder if embedder is not None else Embedder(embedding_model)
    embedding_model = getattr(resolved_embedder, "model_name", embedding_model)
    indexed, parents, duplicates = _indexed_units(
        Path(corpus_root),
        strategy,
        size,
        overlap,
        mode,
        child_size,
        resolved_embedder,
        collapse_duplicates=collapse_duplicates,
        duplicate_tier=duplicate_tier,
    )

    page_coverage = annotate_page_metadata(indexed, corpus_root)
    if parents is not None:
        annotate_page_metadata(parents, corpus_root)

    vectors = (
        encode_store_vectors(indexed, corpus_root, resolved_embedder)
        if strategy == "late"
        else resolved_embedder.encode_passages([chunk["text"] for chunk in indexed])
    )
    index = build_vector_index(vector_store, vectors)
    lexical = (
        LexicalIndex.build(
            [chunk["text"] for chunk in indexed],
            lemmatize=lexical_lemmas,
            lemmatizer=lemmatizer,
        )
        if mode == MODE_HYBRID
        else None
    )
    meta: RagStoreMeta = {
        "mode": mode,
        "strategy": strategy,
        "size": size,
        "overlap": overlap,
        "child_size": child_size,
        "embedding_model": embedding_model,
        "embedder_fingerprint": embedder_fingerprint(embedding_model),
        "n_indexed": len(indexed),
        "n_parents": len(parents) if parents else 0,
        "dim": int(vectors.shape[1]),
        "backend": vector_store,
        "page_annotation_coverage": round(page_coverage, 4),
        "corpus_fingerprint": corpus_fingerprint(corpus_root),
        "doc_fingerprints": corpus_doc_fingerprints(corpus_root),
        "corpus_manifest": "corpus_manifest.json",
        "governance_fields": list(GOVERNANCE_FIELDS),
        "collapse_duplicates": collapse_duplicates,
        "duplicate_tier": duplicate_tier,
        "duplicates": dict(duplicates),
    }
    if lexical is not None:
        meta["lexical"] = {"lemmatize": lexical.lemmatize, "n_terms": len(lexical.postings)}
    return StoreParts(indexed, index, resolved_embedder, meta, parents, lexical)
