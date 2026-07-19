"""Shared base for the platform matrix vector-store adapters (Chroma / Qdrant / LanceDB).

Each adapter is a thin ANN index over the SAME normalized embeddings FAISS indexes, behind the
SAME `VectorIndex` contract (`build` / `search` / `save` / `load`), so `RagStore` is unchanged and
chunk ids + source-span offsets + the `.retrieve(question, k)` contract are preserved by the store
(the index only maps build-order ids -> similarity, exactly like FAISS).

The canonical persisted artifact is the float32 vector matrix (`vectors.npy`): every adapter
rebuilds its native collection from it on `load`, so persistence is uniform and never depends on a
store's on-disk format. Heavy imports (numpy + the backend client) are deferred to first use so the
module imports in the base install; the real path needs the matching optional extra.
"""

from pathlib import Path
from typing import Any

VECTORS_FILE = "vectors.npy"


def cosine_distance_to_similarity(distance: float) -> float:
    """Map a cosine DISTANCE (lower is better) to a cosine SIMILARITY (higher is better).

    Chroma and LanceDB report cosine distance `1 - cos`; FAISS inner product over normalized
    vectors reports cosine similarity directly. Converting keeps every adapter's `retrieval_score`
    on the same higher-is-better cosine scale the gold-span metrics expect."""
    return 1.0 - float(distance)


class VectorStoreAdapter:
    """Base adapter: vector persistence + (scores, ids) shaping shared by every backend.

    Subclasses implement two small hooks -- `_index(vectors)` (load the matrix into the native
    collection) and `_search_row(query, k)` (one query -> ranked `(build_order_id, similarity)`
    pairs). The base handles the build-order id mapping, the per-query shaping that matches the
    FAISS `search` contract, and the uniform `vectors.npy` persistence."""

    name = "base"

    def __init__(self, vectors: Any) -> None:
        self._vectors = vectors  # float32 (n, dim), L2-normalized by the embedder
        self._index(vectors)

    @classmethod
    def build(cls, vectors: Any) -> "VectorStoreAdapter":
        import numpy as np

        arr = np.asarray(vectors, dtype="float32")
        if arr.ndim != 2:
            raise ValueError("vectors must be a 2-D (n, dim) array")
        return cls(arr)

    def _index(self, vectors: Any) -> None:
        raise NotImplementedError

    def _search_row(self, query: list[float], k: int) -> list[tuple[int, float]]:
        raise NotImplementedError

    def vectors(self) -> Any:
        """The stored float32 (n, dim) matrix in build order (refresh reuses these rows)."""
        return self._vectors

    def search(self, query_vectors: Any, k: int) -> tuple[list[list[float]], list[list[int]]]:
        """Top-k per query row as (scores, ids); ids index into the build order (FAISS contract)."""
        rows = (
            query_vectors.tolist()
            if hasattr(query_vectors, "tolist")
            else [list(row) for row in query_vectors]
        )
        scores: list[list[float]] = []
        ids: list[list[int]] = []
        for row in rows:
            pairs = self._search_row(list(row), k)
            ids.append([int(i) for i, _s in pairs])
            scores.append([float(s) for _i, s in pairs])
        return scores, ids

    def save(self, index_dir: Path | str) -> None:
        import numpy as np

        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / VECTORS_FILE, self._vectors)

    @classmethod
    def load(cls, index_dir: Path | str) -> "VectorStoreAdapter":
        import numpy as np

        vectors = np.load(Path(index_dir) / VECTORS_FILE)
        return cls(vectors)
