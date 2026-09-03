"""Persistence for RAG store chunks, indexes, metadata, and lexical sidecars.

Every project-owned member is written through its registered artifact contract, so a store carries
its own identity and a store written by a newer build refuses here instead of retrieving with
fields this reader cannot see. The vector index and the lexical postings are NOT modelled: they
belong to their own formats and are bound opaquely by the store's dataset manifest
(`llb.artifacts.retrieval.datasets`).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llb.artifacts.datasets import publish_dataset_manifest
from llb.artifacts.gates import refuse_tampered_dataset
from llb.artifacts.retrieval.datasets import store_dataset_manifest
from llb.artifacts.records import decode, encode
from llb.core.config_validation import DEFAULT_EMBEDDING_MODEL
from llb.core.contracts.rag import ChunkRecord, RagStoreMeta
from llb.core.contracts.retrieval.store import RAG_CHUNK_SCHEMA_ID, RAG_STORE_META_SCHEMA_ID
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

CHUNK_CONTRACT_VERSION = "1.0.0"
STORE_META_CONTRACT_VERSION = "1.0.0"


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
    _write_jsonl(_encoded_chunks(chunks), target / CHUNKS_FILE)
    if parents is not None:
        _write_jsonl(_encoded_chunks(parents), target / PARENTS_FILE)
    if lexical is not None:
        lexical.save(target / LEXICAL_FILE)
    save_vector_index(index, backend, target)
    (target / META_FILE).write_text(
        json.dumps(_encoded_meta(meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Published last: the manifest binds every member that now exists, including the index and
    # postings files whose bytes no contract describes.
    publish_dataset_manifest(target, store_dataset_manifest(target, resolve_live=False))


def read_store_chunks(path: Path | str) -> list[ChunkRecord]:
    """The chunk rows of one store file at the current contract, identity removed.

    Every consumer that opens `chunks.jsonl` or `parents.jsonl` outside `RagStore.load` -- the
    incremental refresh, the conflict tiers, the duplicate-residue report -- reads it here, so
    exactly one place knows that the rows on disk carry identity and the records in memory do not.
    """
    return _decoded_chunks(Path(path))


def load_store(index_dir: Path | str) -> LoadedStore:
    target = resolve_store_dir(index_dir, META_FILE)
    refuse_tampered_dataset(target)
    chunks = read_store_chunks(target / CHUNKS_FILE)
    meta = read_store_meta(target / META_FILE)
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
    parents = (
        read_store_chunks(target / PARENTS_FILE) if meta.get("mode") == "parent_child" else None
    )
    return LoadedStore(chunks, index, embedder, meta, parents, lexical)


def read_store_meta(path: Path) -> RagStoreMeta:
    """The store metadata at the current contract, migrating a pre-contract file forward."""
    record = json.loads(path.read_text(encoding="utf-8"))
    return cast(RagStoreMeta, decode(RAG_STORE_META_SCHEMA_ID, record, source=str(path)))


def _encoded_meta(meta: RagStoreMeta) -> dict[str, object]:
    return encode(RAG_STORE_META_SCHEMA_ID, STORE_META_CONTRACT_VERSION, meta)


def _encoded_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    return [
        cast(ChunkRecord, encode(RAG_CHUNK_SCHEMA_ID, CHUNK_CONTRACT_VERSION, chunk))
        for chunk in chunks
    ]


def _decoded_chunks(path: Path) -> list[ChunkRecord]:
    rows = _read_jsonl(path)
    return [
        cast(
            ChunkRecord,
            decode(RAG_CHUNK_SCHEMA_ID, row, source=f"{path}#record-{index}"),
        )
        for index, row in enumerate(rows, start=1)
    ]
