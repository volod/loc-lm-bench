"""M7.4 Qdrant vector-store adapter (opt-in `[rag-qdrant]` extra).

An in-memory cosine Qdrant collection over the same normalized embeddings FAISS indexes, behind the
shared `VectorStoreAdapter` contract. Qdrant's cosine SCORE is already a similarity (higher is
better), so it maps straight onto the FAISS-comparable cosine scale.
"""

from typing import Any

from llb.rag.stores.base import VectorStoreAdapter

_COLLECTION = "llb"


class QdrantIndex(VectorStoreAdapter):
    name = "qdrant"

    def _index(self, vectors: Any) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise SystemExit(
                "ERROR: the qdrant store needs the [rag-qdrant] extra. "
                'Run: uv pip install -e ".[rag-qdrant]"'
            ) from exc
        self._models = models
        self._client = QdrantClient(location=":memory:")
        dim = int(vectors.shape[1])
        if self._client.collection_exists(collection_name=_COLLECTION):
            self._client.delete_collection(collection_name=_COLLECTION)
        self._client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        self._client.upsert(
            collection_name=_COLLECTION,
            points=[models.PointStruct(id=i, vector=row.tolist()) for i, row in enumerate(vectors)],
        )

    def _search_row(self, query: list[float], k: int) -> list[tuple[int, float]]:
        result = self._client.query_points(collection_name=_COLLECTION, query=query, limit=k)
        hits = result.points
        return [(int(hit.id), float(hit.score)) for hit in hits]
