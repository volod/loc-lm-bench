"""platform matrix vector-index seam -- pick the ANN backend behind `RagStore` (FAISS is the default).

`RagStore` keeps the chunk records (ids + source-span offsets) and only asks its index to map a
query to build-order ids + similarity scores, so swapping the vector backend never changes the
`.retrieve(question, k)` contract or the gold-span metrics. This module is the single dispatch
point: every backend exposes the same `build` / `search` / `save` / `load`, and the chosen backend
is recorded in the store meta so `load` re-selects it without a config.

Heavy clients are imported lazily inside each adapter, so this module imports in the base install;
each non-FAISS backend needs its matching optional extra (`[rag-chroma]` / `[rag-qdrant]` /
`[rag-lancedb]`).
"""

from pathlib import Path
from typing import Any, Protocol

from llb.rag.vector_store.faiss_index import FaissIndex

RAG_BACKEND_FAISS = "faiss"
RAG_BACKEND_CHROMA = "chroma"
RAG_BACKEND_QDRANT = "qdrant"
RAG_BACKEND_LANCEDB = "lancedb"
RAG_BACKENDS = (
    RAG_BACKEND_FAISS,
    RAG_BACKEND_CHROMA,
    RAG_BACKEND_QDRANT,
    RAG_BACKEND_LANCEDB,
)

# FAISS persists a single file; the adapter backends persist a subdirectory (vectors.npy).
FAISS_INDEX_FILE = "index.faiss"

# Who owns each persisted vector index, and what it is. The bytes belong to the named library, so
# a store BINDS the member rather than modelling it: enough for a reader to say "this is a FAISS
# index written by faiss 1.x, and I do not have faiss" instead of guessing from a file extension.
VECTOR_INDEX_OWNERS: dict[str, tuple[str, str, str]] = {
    RAG_BACKEND_FAISS: (
        "faiss",
        "faiss-index",
        "FAISS inner-product index over the store's normalized embeddings.",
    ),
    RAG_BACKEND_CHROMA: (
        "numpy",
        "npy",
        "Float32 vector matrix a Chroma collection is rebuilt from on load.",
    ),
    RAG_BACKEND_QDRANT: (
        "numpy",
        "npy",
        "Float32 vector matrix a Qdrant collection is rebuilt from on load.",
    ),
    RAG_BACKEND_LANCEDB: (
        "numpy",
        "npy",
        "Float32 vector matrix a LanceDB table is rebuilt from on load.",
    ),
}


def vector_index_relative_path(backend: str) -> str:
    """Where `save_vector_index` puts this backend's persisted index, relative to the store."""
    if backend == RAG_BACKEND_FAISS:
        return FAISS_INDEX_FILE
    from llb.rag.stores.base import VECTORS_FILE

    return f"{backend}/{VECTORS_FILE}"


def vector_index_format_version(backend: str) -> str:
    """The owning library's version, read from the library that just wrote the file."""
    if backend == RAG_BACKEND_FAISS:
        import faiss

        return str(getattr(faiss, "__version__", "unversioned"))
    import numpy

    return str(numpy.__version__)


class VectorIndex(Protocol):
    """The minimal ANN-index seam `RagStore` depends on (FAISS + every platform matrix adapter satisfy it)."""

    def search(self, query_vectors: Any, k: int) -> tuple[list[list[float]], list[list[int]]]: ...

    def save(self, path: Path | str) -> None: ...


def _adapter_class(backend: str) -> Any:
    if backend == RAG_BACKEND_CHROMA:
        from llb.rag.stores.chroma import ChromaIndex

        return ChromaIndex
    if backend == RAG_BACKEND_QDRANT:
        from llb.rag.stores.qdrant import QdrantIndex

        return QdrantIndex
    if backend == RAG_BACKEND_LANCEDB:
        from llb.rag.stores.lancedb import LanceDBIndex

        return LanceDBIndex
    raise ValueError(f"unknown vector store backend: {backend!r}; choose one of {RAG_BACKENDS}")


def build_vector_index(backend: str, vectors: Any) -> VectorIndex:
    """Build the chosen vector index over `vectors` (build-order ids 0..n-1)."""
    if backend == RAG_BACKEND_FAISS:
        return FaissIndex.build(vectors)
    return _adapter_class(backend).build(vectors)  # type: ignore[no-any-return]


def save_vector_index(index: VectorIndex, backend: str, index_dir: Path | str) -> None:
    """Persist `index` under `index_dir` using the backend's on-disk layout."""
    index_dir = Path(index_dir)
    if backend == RAG_BACKEND_FAISS:
        index.save(index_dir / FAISS_INDEX_FILE)
        return
    index.save(index_dir / backend)


def load_vector_index(backend: str, index_dir: Path | str) -> VectorIndex:
    """Reload the persisted vector index for `backend` from `index_dir`."""
    index_dir = Path(index_dir)
    if backend == RAG_BACKEND_FAISS:
        return FaissIndex.load(index_dir / FAISS_INDEX_FILE)
    return _adapter_class(backend).load(index_dir / backend)  # type: ignore[no-any-return]
