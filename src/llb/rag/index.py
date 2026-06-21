"""FAISS inner-product index over normalized embeddings (lazy-loaded).

Vectors are L2-normalized by the embedder, so inner product == cosine similarity. The
index stores only vectors; chunk metadata is kept alongside by `rag.store`. `faiss` and
`numpy` are imported lazily so this module loads in the base install; the real path needs
the `[rag]` extra.
"""

from pathlib import Path
from typing import Any


class FaissIndex:
    """Thin wrapper over a faiss.IndexFlatIP."""

    def __init__(self, index: Any = None):
        self._index = index

    @classmethod
    def build(cls, vectors: Any) -> "FaissIndex":
        faiss = _import_faiss()
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2-D (n, dim) array")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index)

    def search(self, query_vectors: Any, k: int) -> tuple[list[list[float]], list[list[int]]]:
        """Return (scores, ids): top-k per query row. Ids index into the build order."""
        if self._index is None:
            raise RuntimeError("index is empty; build or load it first")
        scores, ids = self._index.search(query_vectors, k)
        return scores.tolist(), ids.tolist()

    def save(self, path: Path | str) -> None:
        faiss = _import_faiss()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path | str) -> "FaissIndex":
        faiss = _import_faiss()
        return cls(faiss.read_index(str(path)))


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise SystemExit(
            'ERROR: FAISS index needs the [rag] extra. Run: uv pip install -e ".[rag]"'
        ) from exc
    return faiss
