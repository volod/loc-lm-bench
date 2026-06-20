"""RAG store: chunked corpus + pinned embedding + FAISS index, retrievable by question.

Two retrieval modes:
  - flat:          index `chunk_size` chunks; retrieve returns those chunks.
  - parent_child:  index small `child_chunk_size` children for precise matching, but return
                   their larger PARENT chunk for generation context (retrieve a child ->
                   surface its parent). Precision from the child, context from the parent.

`retrieve` returns chunk dicts (doc_id + char offsets) in both modes, so recall@k / MRR by
SOURCE-SPAN overlap score directly against the gold labels.
"""

import json
from pathlib import Path

from llb.config import DEFAULT_EMBEDDING_MODEL
from llb.rag.chunking import chunk_corpus, chunk_spans
from llb.rag.embedding import Embedder
from llb.rag.index import FaissIndex

CHUNKS_FILE = "chunks.jsonl"      # the INDEXED units (children in parent_child mode)
PARENTS_FILE = "parents.jsonl"    # the parent docstore (parent_child mode only)
INDEX_FILE = "index.faiss"
META_FILE = "store_meta.json"


def _children_to_parents(child_hits: list[dict], parent_by_id: dict) -> list[dict]:
    """Map ranked child hits to their unique parents (preserving rank). Pure + testable."""
    out: list[dict] = []
    seen: set[str] = set()
    for child in child_hits:
        pid = child.get("parent_id")
        if pid is None or pid in seen or pid not in parent_by_id:
            continue
        seen.add(pid)
        parent = dict(parent_by_id[pid])
        parent["retrieval_score"] = child.get("retrieval_score")
        parent["rank"] = len(out) + 1
        parent["matched_child_id"] = child.get("chunk_id")
        out.append(parent)
    return out


def _build_children(parents: list[dict], strategy: str, child_size: int, overlap: int,
                    embedder) -> list[dict]:
    sem = embedder if strategy == "semantic" else None
    children: list[dict] = []
    for parent in parents:
        text = parent["text"]
        for j, (start, end, meta) in enumerate(chunk_spans(text, strategy, child_size, overlap, sem)):
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
                    "metadata": meta or parent.get("metadata", {}),
                }
            )
    return children


class RagStore:
    """In-process retrieval over one chunked + embedded corpus (flat or parent_child)."""

    def __init__(self, chunks: list[dict], index: FaissIndex, embedder: Embedder, meta: dict,
                 parents: list[dict] | None = None):
        self.chunks = chunks            # indexed units (children when parent_child)
        self.index = index
        self.embedder = embedder
        self.meta = meta
        self.parents = parents
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
    ) -> "RagStore":
        if mode not in ("flat", "parent_child"):
            raise ValueError(f"unknown retrieval mode: {mode}")
        if child_size <= 0:
            raise ValueError("child_size must be > 0")
        embedder = Embedder(embedding_model)
        sem = embedder if strategy == "semantic" else None
        units = chunk_corpus(Path(corpus_root), strategy, size, overlap, sem)
        if not units:
            raise ValueError(f"no chunks produced from corpus at {corpus_root}")

        parents = None
        if mode == "parent_child":
            parents = units
            indexed = _build_children(parents, strategy, child_size, overlap, embedder)
            if not indexed:
                raise ValueError("parent_child mode produced no child chunks")
        else:
            indexed = units

        vectors = embedder.encode_passages([c["text"] for c in indexed])
        index = FaissIndex.build(vectors)
        meta = {
            "mode": mode,
            "strategy": strategy,
            "size": size,
            "overlap": overlap,
            "child_size": child_size,
            "embedding_model": embedding_model,
            "n_indexed": len(indexed),
            "n_parents": len(parents) if parents else 0,
            "dim": int(vectors.shape[1]),
        }
        return cls(indexed, index, embedder, meta, parents=parents)

    def retrieve(self, question: str, k: int) -> list[dict]:
        """Top-k results. Flat: the matched chunks. parent_child: their unique parents."""
        query_vec = self.embedder.encode_queries([question])
        search_k = min(len(self.chunks), k * 4 if self.parents else k)
        while True:
            hits = self._search(query_vec, max(1, search_k))
            if self.parents is None:
                return hits[:k]
            parent_hits = _children_to_parents(hits, self._parent_by_id)
            if len(parent_hits) >= k or search_k >= len(self.chunks):
                return parent_hits[:k]
            # Child hits can cluster under one parent. Expand until k unique parents are
            # found or the complete child index has been searched.
            search_k = min(len(self.chunks), max(search_k + 1, search_k * 2))

    def _search(self, query_vec, search_k: int) -> list[dict]:
        """Return ranked indexed units for an already encoded query."""
        scores, ids = self.index.search(query_vec, search_k)
        hits: list[dict] = []
        for rank, (cid, score) in enumerate(zip(ids[0], scores[0]), 1):
            if cid < 0:  # faiss pads with -1 when fewer than k results exist
                continue
            chunk = dict(self.chunks[cid])
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
        self.index.save(index_dir / INDEX_FILE)
        (index_dir / META_FILE).write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path | str) -> "RagStore":
        index_dir = Path(index_dir)
        chunks = _read_jsonl(index_dir / CHUNKS_FILE)
        meta = json.loads((index_dir / META_FILE).read_text(encoding="utf-8"))
        index = FaissIndex.load(index_dir / INDEX_FILE)
        embedder = Embedder(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
        parents = None
        if meta.get("mode") == "parent_child":
            parents = _read_jsonl(index_dir / PARENTS_FILE)
        return cls(chunks, index, embedder, meta, parents=parents)


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
