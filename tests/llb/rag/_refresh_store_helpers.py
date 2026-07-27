"""dynamic-corpus-refresh: incremental refresh == from-scratch rebuild, per store kind.

Every test builds a v1 store with the fake hashed-BoW embedder, edits the corpus to v2
(modify b.md, delete c.md, add d.md), refreshes incrementally, and compares against a
from-scratch rebuild on the same corpus state: chunk records, embedding matrices, lexical
postings, and ranked retrieval must be identical, and only the changed documents' chunks may
reach the embedder.

The helper module and both importing test modules are marked `heavy_env`: module marks do not
propagate through imports. These tests are quick and run in the default local environment, but are
deselected by `make ci-github`, whose base `[dev]` environment lacks the store extras. The default
local environment includes FAISS, Chroma, and Qdrant. The LanceDB parameter is marked `opt_in_env`
because that adapter remains an explicitly installed lane.
"""

import numpy as np
import pytest
from refresh_helpers import (
    QUESTIONS,
    V1_DOCS,
    V2_DOCS,
    CountingEmbedder,
    build_store,
    retrieval_ids,
    write_corpus,
)

from llb.rag.refresh.store_refresh import stored_vectors
from llb.rag.store import RagStore

pytestmark = pytest.mark.heavy_env

TS = "20990101T000000Z"

META_EQUIVALENCE_KEYS = (
    "mode",
    "strategy",
    "size",
    "overlap",
    "n_indexed",
    "n_parents",
    "dim",
    "backend",
    "page_annotation_coverage",
    "corpus_fingerprint",
    "doc_fingerprints",
    "lexical",
)


def _setup(tmp_path, *, mode="flat", backend="faiss", lemmatizer=None):
    """v1 store on disk + edited corpus; returns (corpus, index_dir)."""
    corpus = write_corpus(tmp_path / "corpus", V1_DOCS)
    store = build_store(
        corpus, CountingEmbedder(), mode=mode, backend=backend, lemmatizer=lemmatizer
    )
    index_dir = tmp_path / "rag"
    store.save(index_dir)
    write_corpus(corpus, V2_DOCS)
    return corpus, index_dir


def _assert_equivalent(
    refreshed: RagStore, rebuilt: RagStore, *, check_retrieval: bool = True
) -> None:
    assert refreshed.chunks == rebuilt.chunks
    assert refreshed.parents == rebuilt.parents
    np.testing.assert_array_equal(
        np.asarray(stored_vectors(refreshed.index)), np.asarray(stored_vectors(rebuilt.index))
    )
    for key in META_EQUIVALENCE_KEYS:
        assert refreshed.meta.get(key) == rebuilt.meta.get(key), key
    if rebuilt.lexical is not None:
        assert refreshed.lexical is not None
        assert refreshed.lexical.postings == rebuilt.lexical.postings
        assert refreshed.lexical.doc_lengths == rebuilt.lexical.doc_lengths
    # ANN adapters (HNSW) may reorder exact score TIES between two collection instances, so
    # ranked-list equality is only asserted for the deterministic FAISS flat index; for the
    # adapters the persisted artifact IS the vector matrix (collections rebuild from
    # vectors.npy on load), so matrix + chunk equality above is the complete equivalence.
    if check_retrieval:
        assert retrieval_ids(refreshed, QUESTIONS) == retrieval_ids(rebuilt, QUESTIONS)
