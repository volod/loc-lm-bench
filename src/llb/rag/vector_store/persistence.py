"""Persistence for RAG store chunks, indexes, metadata, and lexical sidecars.

Publication is contract-checked in both directions: every chunk and parent row is validated
against `llb.rag-chunk` before it is written, the metadata is published with its own identity and
with a digest of each opaque index member beside it, and a load resolves that metadata -- and
re-checks those digests -- before a vector backend is imported.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llb.artifacts.retrieval_graph.opaque import describe_member
from llb.artifacts.retrieval_graph.stores import (
    readable_store_meta,
    refuse_unreadable_store,
    validated_chunk_rows,
    write_store_meta,
)
from llb.core.config_validation import DEFAULT_EMBEDDING_MODEL
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord, RagStoreMeta
from llb.core.contracts.retrieval_graph.common import OpaqueIndexMember
from llb.core.store_generations import resolve_store_dir
from llb.rag.encoders.embedder import Embedder
from llb.rag.vector_store.lexical import LEXICAL_INDEX_VERSION
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
    VECTOR_INDEX_OWNERS,
    VectorIndex,
    load_vector_index,
    save_vector_index,
    vector_index_format_version,
    vector_index_relative_path,
)

VECTOR_INDEX_MEMBER_ID = "vector-index"
LEXICAL_INDEX_MEMBER_ID = "lexical-index"


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
    _write_jsonl(_contract_rows(chunks), target / CHUNKS_FILE)
    if parents is not None:
        _write_jsonl(_contract_rows(parents), target / PARENTS_FILE)
    if lexical is not None:
        lexical.save(target / LEXICAL_FILE)
    save_vector_index(index, backend, target)
    write_store_meta(target / META_FILE, dict(meta), _index_members(target, backend, lexical))


def load_store(index_dir: Path | str) -> LoadedStore:
    target = resolve_store_dir(index_dir, META_FILE)
    refuse_unreadable_store(target)
    meta = cast(RagStoreMeta, readable_store_meta(target / META_FILE))
    chunks = _read_jsonl(target / CHUNKS_FILE)
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


def _contract_rows(rows: list[ChunkRecord]) -> list[ChunkRecord]:
    """Every row, validated against `llb.rag-chunk` and written in the compact store form."""
    return cast(list[ChunkRecord], list(validated_chunk_rows(cast(list[JsonObject], rows))))


def _index_members(
    target: Path, backend: str, lexical: LexicalIndex | None
) -> list[OpaqueIndexMember]:
    """The opaque files this generation was just published with, digested as written."""
    owner, artifact_format, description = VECTOR_INDEX_OWNERS[backend]
    members = [
        describe_member(
            target,
            VECTOR_INDEX_MEMBER_ID,
            vector_index_relative_path(backend),
            owner=owner,
            artifact_format=artifact_format,
            format_version=vector_index_format_version(backend),
            description=description,
        )
    ]
    if lexical is not None:
        members.append(
            describe_member(
                target,
                LEXICAL_INDEX_MEMBER_ID,
                LEXICAL_FILE,
                owner="llb.rag.vector_store.lexical",
                artifact_format="bm25-postings",
                format_version=LEXICAL_INDEX_VERSION,
                description=(
                    "BM25 postings over this build's tokenizer output; a different tokenizer "
                    "version would not match the queries."
                ),
            )
        )
    return members
