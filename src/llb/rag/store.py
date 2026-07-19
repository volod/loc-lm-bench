"""Chunked dense, parent-child, and hybrid retrieval over source-span-preserving records.

Every mode returns offset-exact chunks. Parent-child retrieval indexes precise children and
surfaces their generation-sized parents; hybrid retrieval fuses dense and lexical rankings before
an optional candidate filter.
"""

import json
from pathlib import Path
from typing import Any, cast

from llb.core.config_validation import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FUSION_CANDIDATES,
    DEFAULT_FUSION_WEIGHT,
)
from llb.core.contracts.rag import ChunkRecord, RagStoreMeta
from llb.rag.embedding import Embedder
from llb.rag.filters import ChunkFilter
from llb.rag.late_encoding import encode_store_vectors
from llb.rag.lexical import Lemmatizer, LexicalIndex, rrf_fuse
from llb.rag.page_metadata import annotate_page_metadata
from llb.core.store_generations import resolve_store_dir
from llb.prep.corpus_governance import (
    GOVERNANCE_FIELDS,
    corpus_doc_fingerprints,
    corpus_fingerprint,
)
from llb.rag.vector_index import (
    RAG_BACKEND_FAISS,
    VectorIndex,
    build_vector_index,
    load_vector_index,
    save_vector_index,
)
from llb.rag.store_build import (
    CHUNKS_FILE,
    LEXICAL_FILE,
    META_FILE,
    MODE_HYBRID,
    PARENTS_FILE,
    _children_to_parents,
    _indexed_units,
    _validate_build_params,
)
from llb.rag.store_io import _read_jsonl, _renumber, _write_jsonl


class RagStore:
    """In-process retrieval over one chunked + embedded corpus (flat or parent_child)."""

    def __init__(
        self,
        chunks: list[ChunkRecord],
        index: VectorIndex,
        embedder: Embedder,
        meta: RagStoreMeta,
        parents: list[ChunkRecord] | None = None,
        lexical: LexicalIndex | None = None,
    ):
        self.chunks = chunks  # indexed units (children when parent_child)
        self.index = index
        self.embedder = embedder
        self.meta = meta
        self.parents = parents
        self.lexical = lexical  # BM25 side of hybrid mode (None otherwise)
        # Query-time fusion knobs; `_load_store` overwrites them from the RunConfig so the
        # manifest-recorded values are the ones actually used.
        self.fusion_weight = DEFAULT_FUSION_WEIGHT
        self.fusion_candidates = DEFAULT_FUSION_CANDIDATES
        self.backend = str(
            meta.get("backend", RAG_BACKEND_FAISS)
        )  # platform matrix vector-store backend
        self._parent_by_id = {p["chunk_id"]: p for p in parents} if parents else {}

    @classmethod
    def build(
        cls,
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
    ) -> "RagStore":
        """Chunk + embed a corpus into a retrievable store.

        `embedder` injects an alternative encoder exposing `encode_passages`/`encode_queries`
        (e.g. the `compare-embeddings` API lane's `ApiEmbedder`); its `model_name` overrides
        `embedding_model` in the persisted meta so a store always records the encoder it was
        built with. Defaults to the pinned local `Embedder(embedding_model)`.

        `mode="hybrid"` additionally builds the lexical BM25 index over the same chunks;
        `lexical_lemmas` opts its tokenization into Ukrainian lemmatization (`lemmatizer`
        injects a fake for tests). The stored chunk text is byte-identical either way.
        """
        _validate_build_params(mode, strategy, child_size)
        embedder = embedder if embedder is not None else Embedder(embedding_model)
        embedding_model = getattr(embedder, "model_name", embedding_model)
        indexed, parents = _indexed_units(
            Path(corpus_root), strategy, size, overlap, mode, child_size, embedder
        )

        # Attach page/section provenance from PDF citation sidecars (strategy-independent,
        # additive metadata only). Coverage is measured over the INDEXED units; parents are
        # annotated too so their metadata surfaces on parent_child retrieval hits.
        page_coverage = annotate_page_metadata(indexed, corpus_root)
        if parents is not None:
            annotate_page_metadata(parents, corpus_root)

        if strategy == "late":
            # Late chunking: pool whole-document token embeddings per chunk span instead of
            # encoding each chunk text in isolation (see `llb.rag.late_encoding`).
            vectors = encode_store_vectors(indexed, corpus_root, embedder)
        else:
            vectors = embedder.encode_passages([c["text"] for c in indexed])
        index = build_vector_index(vector_store, vectors)
        lexical = None
        if mode == MODE_HYBRID:
            lexical = LexicalIndex.build(
                [c["text"] for c in indexed], lemmatize=lexical_lemmas, lemmatizer=lemmatizer
            )
        meta: RagStoreMeta = {
            "mode": mode,
            "strategy": strategy,
            "size": size,
            "overlap": overlap,
            "child_size": child_size,
            "embedding_model": embedding_model,
            "n_indexed": len(indexed),
            "n_parents": len(parents) if parents else 0,
            "dim": int(vectors.shape[1]),
            "backend": vector_store,
            "page_annotation_coverage": round(page_coverage, 4),
            "corpus_fingerprint": corpus_fingerprint(corpus_root),
            "doc_fingerprints": corpus_doc_fingerprints(corpus_root),
            "corpus_manifest": "corpus_manifest.json",
            "governance_fields": list(GOVERNANCE_FIELDS),
        }
        if lexical is not None:
            meta["lexical"] = {"lemmatize": lexical.lemmatize, "n_terms": len(lexical.postings)}
        return cls(indexed, index, embedder, meta, parents=parents, lexical=lexical)

    def retrieve(
        self, question: str, k: int, chunk_filter: ChunkFilter | None = None
    ) -> list[ChunkRecord]:
        """Top-k results. Flat: the matched chunks. parent_child: their unique parents.
        Hybrid: the weighted-RRF fusion of the dense and lexical rankings.

        `chunk_filter` (see `llb.rag.filters.metadata_filter`) restricts candidates BEFORE
        fusion/ranking; with a filter the whole index is scanned, so the cut is exact.
        """
        query_vec = self.embedder.encode_queries([question])
        if self.lexical is not None and self.meta.get("mode") == MODE_HYBRID:
            return self._retrieve_hybrid(question, query_vec, k, chunk_filter)
        base_k = k * 4 if self.parents else k
        search_k = len(self.chunks) if chunk_filter else min(len(self.chunks), base_k)
        while True:
            hits = self._filtered_search(query_vec, search_k, chunk_filter)
            if self.parents is None:
                return hits[:k]
            parent_hits = _children_to_parents(hits, self._parent_by_id)
            if len(parent_hits) >= k or search_k >= len(self.chunks):
                return parent_hits[:k]
            # Child hits can cluster under one parent. Expand until k unique parents are
            # found or the complete child index has been searched.
            search_k = min(len(self.chunks), max(search_k + 1, search_k * 2))

    def _filtered_search(
        self, query_vec: Any, search_k: int, chunk_filter: ChunkFilter | None
    ) -> list[ChunkRecord]:
        """Dense search, with candidates re-ranked/renumbered after the metadata cut."""
        hits = self._search(query_vec, max(1, search_k))
        if chunk_filter is not None:
            hits = _renumber([hit for hit in hits if chunk_filter(hit)])
        return hits

    def _retrieve_hybrid(
        self, question: str, query_vec: Any, k: int, chunk_filter: ChunkFilter | None
    ) -> list[ChunkRecord]:
        """Fuse the dense and lexical top candidates with weighted RRF; return the top k."""
        assert self.lexical is not None
        depth = max(self.fusion_candidates, k)
        search_k = len(self.chunks) if chunk_filter else min(len(self.chunks), depth)
        _scores, ids = self.index.search(query_vec, max(1, search_k))
        dense_ids = [int(cid) for cid in ids[0] if cid >= 0]
        allowed: set[int] | None = None
        if chunk_filter is not None:
            allowed = {i for i, c in enumerate(self.chunks) if chunk_filter(c)}
            dense_ids = [cid for cid in dense_ids if cid in allowed]
        dense_ids = dense_ids[:depth]
        lexical_ids = [cid for cid, _ in self.lexical.search(question, depth, allowed)]
        fused = rrf_fuse(dense_ids, lexical_ids, self.fusion_weight)
        hits: list[ChunkRecord] = []
        for rank, (cid, score) in enumerate(fused[:k], 1):
            chunk = cast(ChunkRecord, dict(self.chunks[cid]))
            chunk["retrieval_score"] = float(score)
            chunk["rank"] = rank
            hits.append(chunk)
        return hits

    def _search(self, query_vec: Any, search_k: int) -> list[ChunkRecord]:
        """Return ranked indexed units for an already encoded query."""
        scores, ids = self.index.search(query_vec, search_k)
        hits: list[ChunkRecord] = []
        for rank, (cid, score) in enumerate(zip(ids[0], scores[0]), 1):
            if cid < 0:  # faiss pads with -1 when fewer than k results exist
                continue
            chunk = cast(ChunkRecord, dict(self.chunks[cid]))
            chunk["retrieval_score"] = float(score)
            chunk["rank"] = rank
            hits.append(chunk)
        return hits

    def save(self, index_dir: Path | str) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(self.chunks, index_dir / CHUNKS_FILE)
        if self.parents is not None:
            _write_jsonl(self.parents, index_dir / PARENTS_FILE)
        if self.lexical is not None:
            self.lexical.save(index_dir / LEXICAL_FILE)
        save_vector_index(self.index, self.backend, index_dir)
        (index_dir / META_FILE).write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path | str) -> "RagStore":
        # A refresh publishes immutable `generations/<ts>/` children; resolve the live one.
        index_dir = resolve_store_dir(index_dir, META_FILE)
        chunks = _read_jsonl(index_dir / CHUNKS_FILE)
        meta = json.loads((index_dir / META_FILE).read_text(encoding="utf-8"))
        lexical = None
        if meta.get("mode") == MODE_HYBRID:
            lexical_path = index_dir / LEXICAL_FILE
            if not lexical_path.is_file():
                raise SystemExit(
                    f"[rag] the hybrid store at {index_dir} is missing its lexical index "
                    f"({LEXICAL_FILE}); rebuild it with `build-index --retrieval-mode hybrid`."
                )
            lexical = LexicalIndex.load(lexical_path)
        index = load_vector_index(meta.get("backend", RAG_BACKEND_FAISS), index_dir)
        embedder = Embedder(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
        parents = None
        if meta.get("mode") == "parent_child":
            parents = _read_jsonl(index_dir / PARENTS_FILE)
        return cls(chunks, index, embedder, meta, parents=parents, lexical=lexical)
