"""RAG store: chunked corpus + pinned embedding + FAISS index, retrievable by question.

Build once with `RagStore.build(...)` (chunks the corpus via `llb.rag.chunking`, embeds
with the pinned `Embedder`, indexes with `FaissIndex`), persist with `.save(...)`, reload
with `.load(...)`, and query with `.retrieve(question, k)` -> ranked chunk dicts.

`retrieve` returns the same chunk-dict shape the retrieval metrics consume (doc_id +
char offsets), so recall@k / MRR score directly against gold source spans.
"""

import json
from pathlib import Path

from llb.config import DEFAULT_EMBEDDING_MODEL
from llb.rag.chunking import chunk_corpus
from llb.rag.embedding import Embedder
from llb.rag.index import FaissIndex

CHUNKS_FILE = "chunks.jsonl"
INDEX_FILE = "index.faiss"
META_FILE = "store_meta.json"


class RagStore:
    """In-process retrieval over one chunked + embedded corpus."""

    def __init__(self, chunks: list[dict], index: FaissIndex, embedder: Embedder, meta: dict):
        self.chunks = chunks
        self.index = index
        self.embedder = embedder
        self.meta = meta

    @classmethod
    def build(
        cls,
        corpus_root: Path | str,
        strategy: str = "recursive",
        size: int = 800,
        overlap: int = 120,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> "RagStore":
        chunks = chunk_corpus(Path(corpus_root), strategy, size, overlap)
        if not chunks:
            raise ValueError(f"no chunks produced from corpus at {corpus_root}")
        embedder = Embedder(embedding_model)
        vectors = embedder.encode_passages([c["text"] for c in chunks])
        index = FaissIndex.build(vectors)
        meta = {
            "strategy": strategy,
            "size": size,
            "overlap": overlap,
            "embedding_model": embedding_model,
            "n_chunks": len(chunks),
            "dim": int(vectors.shape[1]),
        }
        return cls(chunks, index, embedder, meta)

    def retrieve(self, question: str, k: int) -> list[dict]:
        """Top-k chunks for a question, each annotated with its retrieval score + rank."""
        query_vec = self.embedder.encode_queries([question])
        scores, ids = self.index.search(query_vec, min(k, len(self.chunks)))
        out: list[dict] = []
        for rank, (cid, score) in enumerate(zip(ids[0], scores[0]), 1):
            if cid < 0:  # faiss pads with -1 when fewer than k results exist
                continue
            chunk = dict(self.chunks[cid])
            chunk["retrieval_score"] = float(score)
            chunk["rank"] = rank
            out.append(chunk)
        return out

    def save(self, index_dir: Path | str) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        self.index.save(index_dir / INDEX_FILE)
        (index_dir / META_FILE).write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path | str) -> "RagStore":
        index_dir = Path(index_dir)
        chunks = [
            json.loads(line)
            for line in (index_dir / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        meta = json.loads((index_dir / META_FILE).read_text(encoding="utf-8"))
        index = FaissIndex.load(index_dir / INDEX_FILE)
        embedder = Embedder(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
        return cls(chunks, index, embedder, meta)
