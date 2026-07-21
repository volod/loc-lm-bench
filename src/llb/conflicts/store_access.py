"""Read chunks and vectors from a built store without constructing a query embedder.

`RagStore.load` eagerly builds an `Embedder`, which downloads and loads the encoder. Conflict
detection compares STORED vectors to each other and never encodes a query, so it reads the
persisted artifacts directly instead -- keeping the audit runnable (and testable) without the
sentence-transformers stack.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.conflicts.vectorops import VectorSet
from llb.core.contracts.rag import ChunkRecord
from llb.core.store_generations import resolve_store_dir
from llb.rag.store_build import CHUNKS_FILE, META_FILE
from llb.rag.store_io import _read_jsonl
from llb.rag.vector_index import RAG_BACKEND_FAISS, load_vector_index


@dataclass
class StoreView:
    """The parts of a built store the conflict tiers need."""

    index_dir: Path
    chunks: list[ChunkRecord]
    vectors: VectorSet
    meta: dict[str, Any]

    @property
    def embedding_model(self) -> str:
        return str(self.meta.get("embedding_model", ""))

    @property
    def dim(self) -> int:
        return self.vectors.dim

    @property
    def doc_fingerprints(self) -> dict[str, str]:
        recorded = self.meta.get("doc_fingerprints")
        return dict(recorded) if isinstance(recorded, dict) else {}


def load_store_view(index_dir: Path | str) -> StoreView:
    """Load chunk records plus their stored vectors from the live store generation."""
    resolved = resolve_store_dir(Path(index_dir), META_FILE)
    meta_path = resolved / META_FILE
    if not meta_path.is_file():
        raise SystemExit(
            f"[conflicts] no store at {resolved}: run `make build-index` first "
            "(the semantic and claim tiers need chunk vectors)."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks = _read_jsonl(resolved / CHUNKS_FILE)
    index = load_vector_index(str(meta.get("backend", RAG_BACKEND_FAISS)), resolved)
    vectors = VectorSet.from_any(index.vectors())  # type: ignore[attr-defined]
    if len(vectors) != len(chunks):
        raise SystemExit(
            f"[conflicts] store at {resolved} is inconsistent: {len(chunks)} chunks but "
            f"{len(vectors)} vectors."
        )
    return StoreView(index_dir=resolved, chunks=chunks, vectors=vectors, meta=meta)
