"""RAG store: chunked corpus + pinned embedding + FAISS index, retrievable by question.

Three retrieval modes:
  - flat:          index `chunk_size` chunks; retrieve returns those chunks.
  - parent_child:  index small `child_chunk_size` children for precise matching, but return
                   their larger PARENT chunk for generation context (retrieve a child ->
                   surface its parent). Precision from the child, context from the parent.
  - hybrid:        index like `flat`, but ALSO build a lexical BM25 index over the same
                   offset-exact chunks and fuse the dense + lexical rankings with weighted
                   reciprocal-rank fusion at query time (hybrid-retrieval-uk). Fusion happens
                   inside `retrieve`, so every dense `VectorIndex` backend gains hybrid
                   identically.

`retrieve` returns chunk dicts (doc_id + char offsets) in every mode, so recall@k / MRR by
SOURCE-SPAN overlap score directly against the gold labels. The optional `chunk_filter`
predicate (see `llb.rag.filters`) restricts candidates BEFORE fusion/ranking.
"""

import json
from pathlib import Path
from typing import Any, cast

from llb.core.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FUSION_CANDIDATES,
    DEFAULT_FUSION_WEIGHT,
)
from llb.core.contracts import ChunkRecord, RagStoreMeta
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.chunking.dispatch import chunk_spans
from llb.rag.embedding import Embedder
from llb.rag.filters import ChunkFilter
from llb.rag.late_encoding import encode_store_vectors
from llb.rag.lexical import Lemmatizer, LexicalIndex, rrf_fuse
from llb.rag.page_metadata import annotate_page_metadata
from llb.prep.corpus_governance import GOVERNANCE_FIELDS, corpus_fingerprint
from llb.rag.vector_index import (
    RAG_BACKEND_FAISS,
    VectorIndex,
    build_vector_index,
    load_vector_index,
    save_vector_index,
)

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
        index_dir = Path(index_dir)
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


def store_embedder_mismatch(meta: RagStoreMeta, expected_model: str) -> str | None:
    """Return the store's built embedder id when it differs from `expected_model`, else None.

    A store is embedded and queried by the SAME encoder (recorded in `store_meta.json`), so a
    config that names a different `embedding_model` than the store on disk would silently score
    the wrong encoder. Callers refuse the run with this signal (embedding bake-off fingerprint).
    """
    built = str(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    return built if built != expected_model else None


def stale_store_message(
    meta: RagStoreMeta, corpus_root: Path | str, index_dir: Path | str
) -> str | None:
    """Return a rebuild message when the store fingerprint differs from the current corpus."""
    built = meta.get("corpus_fingerprint")
    if not isinstance(built, str):
        return None
    current = corpus_fingerprint(corpus_root)
    if built == current:
        return None
    return (
        f"[rag] stale store at {index_dir}: corpus manifest fingerprint changed. "
        "Rebuild with `llb build-index --corpus-root <corpus-dir>` so removed sources and "
        "governance metadata propagate into chunks."
    )


def _renumber(hits: list[ChunkRecord]) -> list[ChunkRecord]:
    """Reassign contiguous 1-based ranks after a filter removed candidates."""
    for rank, hit in enumerate(hits, 1):
        hit["rank"] = rank
    return hits


def _write_jsonl(rows: list[ChunkRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[ChunkRecord]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return cast(list[ChunkRecord], rows)
