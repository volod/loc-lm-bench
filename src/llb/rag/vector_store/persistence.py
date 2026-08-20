"""Persistence for RAG store chunks, indexes, metadata, and lexical sidecars."""

import json
from dataclasses import dataclass
from pathlib import Path

from llb.core.config_validation import DEFAULT_EMBEDDING_MODEL
from llb.core.contracts.rag import ChunkRecord, RagStoreMeta
from llb.core.store_generations import resolve_store_dir
from llb.rag.encoders.embedder import Embedder
from llb.rag.vector_store.lexical_index import LexicalIndex
from llb.rag.vector_store.build import (
    CHUNKS_FILE,
    LEXICAL_FILE,
    META_FILE,
    PARENTS_FILE,
    MODE_HYBRID,
)
from llb.rag.vector_store.io import _read_jsonl, _write_jsonl
from llb.rag.vector_store.vector_index import (
    RAG_BACKEND_FAISS,
    VectorIndex,
    load_vector_index,
    save_vector_index,
)


@dataclass(frozen=True)
class LoadedStore:
    chunks: list[ChunkRecord]
    index: VectorIndex
    embedder: Embedder
    meta: RagStoreMeta
    parents: list[ChunkRecord] | None
    lexical: LexicalIndex | None


def save_store(
    index_dir: Path | str,
    chunks: list[ChunkRecord],
    index: VectorIndex,
    backend: str,
    meta: RagStoreMeta,
    parents: list[ChunkRecord] | None,
    lexical: LexicalIndex | None,
) -> None:
    target = Path(index_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_jsonl(chunks, target / CHUNKS_FILE)
    if parents is not None:
        _write_jsonl(parents, target / PARENTS_FILE)
    if lexical is not None:
        lexical.save(target / LEXICAL_FILE)
    save_vector_index(index, backend, target)
    (target / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_store(index_dir: Path | str) -> LoadedStore:
    target = resolve_store_dir(index_dir, META_FILE)
    chunks = _read_jsonl(target / CHUNKS_FILE)
    meta = json.loads((target / META_FILE).read_text(encoding="utf-8"))
    lexical = None
    if meta.get("mode") == MODE_HYBRID:
        lexical_path = target / LEXICAL_FILE
        if not lexical_path.is_file():
            raise SystemExit(
                f"[rag] the hybrid store at {target} is missing its lexical index "
                f"({LEXICAL_FILE}); rebuild it with `build-index --retrieval-mode hybrid`."
            )
        lexical = LexicalIndex.load(lexical_path)
    index = load_vector_index(meta.get("backend", RAG_BACKEND_FAISS), target)
    embedder = Embedder(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    parents = _read_jsonl(target / PARENTS_FILE) if meta.get("mode") == "parent_child" else None
    return LoadedStore(chunks, index, embedder, meta, parents, lexical)
