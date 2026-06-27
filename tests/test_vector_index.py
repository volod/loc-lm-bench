"""M7.4 vector-store seam -- dispatch, adapter contracts, FAISS round-trip, store backend tag."""

import pytest

from llb.rag import vector_index as vi
from llb.rag.stores.base import VectorStoreAdapter, cosine_distance_to_similarity


# --- pure helpers + dispatch (no heavy deps -- runs in the lightweight CI) -----------------


def test_cosine_distance_to_similarity():
    assert cosine_distance_to_similarity(0.0) == 1.0
    assert cosine_distance_to_similarity(0.25) == 0.75


def test_adapter_class_dispatch_and_unknown():
    from llb.rag.stores.chroma import ChromaIndex
    from llb.rag.stores.lancedb import LanceDBIndex
    from llb.rag.stores.qdrant import QdrantIndex

    assert vi._adapter_class(vi.RAG_BACKEND_CHROMA) is ChromaIndex
    assert vi._adapter_class(vi.RAG_BACKEND_QDRANT) is QdrantIndex
    assert vi._adapter_class(vi.RAG_BACKEND_LANCEDB) is LanceDBIndex
    with pytest.raises(ValueError, match="unknown vector store backend"):
        vi._adapter_class("bogus")


def test_build_vector_index_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown vector store backend"):
        vi.build_vector_index("bogus", None)


@pytest.mark.parametrize(
    "backend,module",
    [
        (vi.RAG_BACKEND_CHROMA, "chromadb"),
        (vi.RAG_BACKEND_QDRANT, "qdrant_client"),
        (vi.RAG_BACKEND_LANCEDB, "lancedb"),
    ],
)
def test_adapter_raises_systemexit_when_extra_missing(backend, module):
    """With the optional extra absent, instantiating the adapter names the install path."""
    try:
        __import__(module)
        pytest.skip(f"{module} installed; the missing-extra path is exercised only when absent")
    except ImportError:
        pass
    cls = vi._adapter_class(backend)
    with pytest.raises(SystemExit, match=backend):
        cls([[0.1, 0.2]])  # __init__ -> _index -> lazy import fails -> SystemExit


# --- base adapter shaping (fake subclass, no numpy needed) ---------------------------------


class _StubAdapter(VectorStoreAdapter):
    """Brute-force, dependency-free subclass to exercise the base seam (build-order ids/shaping)."""

    name = "stub"

    def _index(self, vectors):
        self._rows = [list(row) for row in vectors]

    def _search_row(self, query, k):
        # dot product over the stored rows; return top-k (id, score) by score desc.
        scored = [(i, sum(a * b for a, b in zip(query, row))) for i, row in enumerate(self._rows)]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]


def test_base_search_shapes_scores_and_ids():
    index = _StubAdapter([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    scores, ids = index.search([[1.0, 0.0]], 2)
    assert ids == [[0, 2]]  # row 0 is the closest, then row 2
    assert scores[0][0] == 1.0 and scores[0][1] == pytest.approx(0.9)


# --- FAISS round-trip through the seam (needs the [rag] extra) -----------------------------


@pytest.mark.slow
def test_faiss_seam_build_search_save_load(tmp_path):
    pytest.importorskip("faiss")
    import numpy as np

    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype="float32")
    index = vi.build_vector_index(vi.RAG_BACKEND_FAISS, vectors)
    scores, ids = index.search(np.array([[1.0, 0.0]], dtype="float32"), 2)
    assert ids[0][0] == 0  # exact match ranks first
    vi.save_vector_index(index, vi.RAG_BACKEND_FAISS, tmp_path)
    assert (tmp_path / vi.FAISS_INDEX_FILE).exists()
    reloaded = vi.load_vector_index(vi.RAG_BACKEND_FAISS, tmp_path)
    scores2, ids2 = reloaded.search(np.array([[1.0, 0.0]], dtype="float32"), 2)
    assert ids2 == ids and scores2 == scores


@pytest.mark.slow
def test_base_adapter_vector_roundtrip(tmp_path):
    pytest.importorskip("numpy")
    import numpy as np

    index = _StubAdapter.build(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    index.save(tmp_path)
    assert (tmp_path / "vectors.npy").exists()
    reloaded = _StubAdapter.load(tmp_path)
    _scores, ids = reloaded.search([[0.0, 1.0]], 1)
    assert ids == [[1]]
