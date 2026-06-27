"""M7.4 Chroma vector-store adapter (opt-in `[rag-chroma]` extra).

A cosine-space Chroma collection over the same normalized embeddings FAISS indexes, behind the
shared `VectorStoreAdapter` contract. Chroma reports cosine DISTANCE, converted back to similarity
so `retrieval_score` stays on the FAISS-comparable cosine scale.
"""

from typing import Any
from uuid import uuid4

from llb.rag.stores.base import VectorStoreAdapter, cosine_distance_to_similarity

_COLLECTION_PREFIX = "llb"


class ChromaIndex(VectorStoreAdapter):
    name = "chroma"

    def _index(self, vectors: Any) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise SystemExit(
                "ERROR: the chroma store needs the [rag-chroma] extra. "
                'Run: uv pip install -e ".[rag-chroma]"'
            ) from exc
        client = chromadb.EphemeralClient()
        # cosine space so the distance is 1 - cosine_similarity over the normalized vectors.
        collection_name = f"{_COLLECTION_PREFIX}-{uuid4().hex}"
        self._collection: Any = client.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._collection.add(
            ids=[str(i) for i in range(len(vectors))],
            embeddings=[row.tolist() for row in vectors],
        )

    def _search_row(self, query: list[float], k: int) -> list[tuple[int, float]]:
        result: dict[str, Any] = self._collection.query(query_embeddings=[query], n_results=k)
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [
            (int(cid), cosine_distance_to_similarity(dist)) for cid, dist in zip(ids, distances)
        ]
