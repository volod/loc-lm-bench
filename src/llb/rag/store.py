"""Chunked dense, parent-child, and hybrid retrieval over source-span-preserving records.

Every mode returns offset-exact chunks. Parent-child retrieval indexes precise children and
surfaces their generation-sized parents; hybrid retrieval fuses dense and lexical rankings before
an optional candidate filter.
"""

from pathlib import Path
from typing import Any, cast

from llb.core.config_validation import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FUSION_CANDIDATES,
    DEFAULT_FUSION_WEIGHT,
)
from llb.core.contracts.rag import ChunkRecord, RagStoreMeta
from llb.rag.duplicate_tiers import TIER_EXACT
from llb.rag.embedding import Embedder
from llb.rag.filters import ChunkFilter
from llb.rag.lexical import (
    Lemmatizer,
    rrf_fuse,
)
from llb.rag.lexical_index import LexicalIndex
from llb.rag.vector_index import RAG_BACKEND_FAISS, VectorIndex
from llb.rag.store_build import (
    MODE_HYBRID,
    _children_to_parents,
    order_by_score,
)
from llb.rag.store_factory import build_store_parts
from llb.rag.store_hybrid import allowed_chunk_ids, dense_hybrid_ids, hybrid_chunks
from llb.rag.store_persistence import load_store, save_store
from llb.rag.store_io import _renumber


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
        collapse_duplicates: bool = True,
        duplicate_tier: str = TIER_EXACT,
    ) -> "RagStore":
        """Chunk + embed a corpus into a retrievable store.

        `embedder` injects an alternative encoder exposing `encode_passages`/`encode_queries`
        (e.g. the `compare-embeddings` API lane's `ApiEmbedder`); its `model_name` overrides
        `embedding_model` in the persisted meta so a store always records the encoder it was
        built with. Defaults to the pinned local `Embedder(embedding_model)`.

        `mode="hybrid"` additionally builds the lexical BM25 index over the same chunks;
        `lexical_lemmas` opts its tokenization into Ukrainian lemmatization (`lemmatizer`
        injects a fake for tests). The stored chunk text is byte-identical either way.

        `collapse_duplicates` (default) indexes each distinct chunk text once and records the
        dropped copies as additive occurrence metadata (`llb.rag.duplicates`); pass False to keep
        every copy indexed, which only changes the index budget and the tie rate -- the measured
        duplicate stats land in the store meta either way. `duplicate_tier` selects WHEN two texts
        count as the same passage (`llb.rag.duplicate_tiers`); only the default `exact` tier is
        loss-free. A strategy whose vector is NOT a pure function of its text (`late`) keeps its
        duplicates whatever is requested, and the meta records what the build actually did
        (`collapse_is_lossless` in `llb.rag.store_build`).
        """
        parts = build_store_parts(
            corpus_root,
            strategy,
            size,
            overlap,
            embedding_model,
            mode,
            child_size,
            vector_store,
            embedder,
            lexical_lemmas,
            lemmatizer,
            collapse_duplicates,
            duplicate_tier,
        )
        return cls(
            parts.chunks,
            parts.index,
            parts.embedder,
            parts.meta,
            parents=parts.parents,
            lexical=parts.lexical,
        )

    def retrieve(
        self, question: str, k: int, chunk_filter: ChunkFilter | None = None
    ) -> list[ChunkRecord]:
        """Top-k results. Flat: the matched chunks. parent_child: their unique parents.
        Hybrid: the weighted-RRF fusion of the dense and lexical rankings.

        `chunk_filter` (see `llb.rag.filters.metadata_filter`) restricts candidates BEFORE
        fusion/ranking; with a filter the whole index is scanned, so the cut is exact.
        """
        return self.retrieve_queries(question, question, k, chunk_filter=chunk_filter)

    def retrieve_queries(
        self,
        dense_query: str,
        lexical_query: str,
        k: int,
        chunk_filter: ChunkFilter | None = None,
    ) -> list[ChunkRecord]:
        """Retrieve with independently selected dense and lexical query text.

        The split is useful for HyDE: its hypothetical passage drives the embedding while the
        user's processed question remains the BM25 query. Dense-only stores ignore
        `lexical_query`.
        """
        query_vec = self.embedder.encode_queries([dense_query])
        if self.lexical is not None and self.meta.get("mode") == MODE_HYBRID:
            return self._retrieve_hybrid(lexical_query, query_vec, k, chunk_filter)
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
        allowed = allowed_chunk_ids(self.chunks, chunk_filter)
        dense_ids = dense_hybrid_ids(
            self.chunks,
            self.index,
            self._ranked_candidates,
            query_vec,
            depth,
            chunk_filter,
            allowed,
            self.fusion_weight,
        )
        lexical_ids = [cid for cid, _ in self.lexical.search(question, depth, allowed)]
        fused = rrf_fuse(dense_ids, lexical_ids, self.fusion_weight)
        return hybrid_chunks(self.chunks, fused, k)

    def _search(self, query_vec: Any, search_k: int) -> list[ChunkRecord]:
        """Return ranked indexed units for an already encoded query.

        Candidates are re-sorted through `order_by_score`, so an exact score tie is broken by
        `chunk_id` rather than by the backend's candidate order.
        """
        scores, ids = self.index.search(query_vec, search_k)
        hits: list[ChunkRecord] = []
        for rank, (cid, score) in enumerate(self._ranked_candidates(ids, scores), 1):
            chunk = cast(ChunkRecord, dict(self.chunks[cid]))
            chunk["retrieval_score"] = float(score)
            chunk["rank"] = rank
            hits.append(chunk)
        return hits

    def _ranked_candidates(self, ids: Any, scores: Any) -> list[tuple[int, float]]:
        """Backend `(id, score)` rows, padding dropped and exact ties broken deterministically."""
        candidates = [
            (int(cid), float(score))
            for cid, score in zip(ids[0], scores[0])
            if cid >= 0  # faiss pads with -1 when fewer than k results exist
        ]
        return order_by_score(candidates, self.chunks)

    def save(self, index_dir: Path | str) -> None:
        save_store(
            index_dir,
            self.chunks,
            self.index,
            self.backend,
            self.meta,
            self.parents,
            self.lexical,
        )

    @classmethod
    def load(cls, index_dir: Path | str) -> "RagStore":
        loaded = load_store(index_dir)
        return cls(
            loaded.chunks,
            loaded.index,
            loaded.embedder,
            loaded.meta,
            parents=loaded.parents,
            lexical=loaded.lexical,
        )
