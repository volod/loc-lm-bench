"""M7.4 LanceDB vector-store adapter (opt-in `[rag-lancedb]` extra).

A cosine LanceDB table over the same normalized embeddings FAISS indexes, behind the shared
`VectorStoreAdapter` contract. LanceDB needs an on-disk connection even for a transient table, so
the adapter keeps a `TemporaryDirectory` alive for the table's lifetime; it reports cosine DISTANCE,
converted back to similarity so `retrieval_score` stays on the FAISS-comparable cosine scale.
"""

from tempfile import TemporaryDirectory
from typing import Any

from llb.rag.stores.base import VectorStoreAdapter, cosine_distance_to_similarity

_TABLE = "llb"


class LanceDBIndex(VectorStoreAdapter):
    name = "lancedb"

    def _index(self, vectors: Any) -> None:
        try:
            import lancedb
        except ImportError as exc:
            raise SystemExit(
                "ERROR: the lancedb store needs the [rag-lancedb] extra. "
                'Run: uv pip install -e ".[rag-lancedb]"'
            ) from exc
        # The temp dir backs the transient table; held on the instance so it outlives _index.
        self._tmp = TemporaryDirectory(prefix="llb-lancedb-")
        db = lancedb.connect(self._tmp.name)
        rows = [{"id": i, "vector": row.tolist()} for i, row in enumerate(vectors)]
        self._table = db.create_table(_TABLE, data=rows, mode="overwrite")

    def _search_row(self, query: list[float], k: int) -> list[tuple[int, float]]:
        hits = self._table.search(query).metric("cosine").limit(k).to_list()
        return [(int(hit["id"]), cosine_distance_to_similarity(hit["_distance"])) for hit in hits]
