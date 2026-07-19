"""dynamic-corpus-refresh: incremental refresh == from-scratch rebuild, per store kind.

Every test builds a v1 store with the fake hashed-BoW embedder, edits the corpus to v2
(modify b.md, delete c.md, add d.md), refreshes incrementally, and compares against a
from-scratch rebuild on the same corpus state: chunk records, embedding matrices, lexical
postings, and ranked retrieval must be identical, and only the changed documents' chunks may
reach the embedder.

Every test builds real FAISS-backed stores, so the module is marked `heavy_env`: quick, run by
local `make ci` / `make test` (full install), deselected by `make ci-github` in the base
`[dev]`-only GitHub env where faiss (the `[rag]` extra) is absent.
"""

import json

import numpy as np
import pytest

from llb.core.store_generations import resolve_store_dir
from llb.rag.refresh.siblings import refresh_sibling_stores, sibling_store_dirs
from llb.rag.refresh.store_refresh import refresh_vector_store, stored_vectors
from llb.rag.store import RagStore
from llb.rag.store_build import CHUNKS_FILE, META_FILE

from refresh_helpers import (
    QUESTIONS,
    V1_DOCS,
    V2_DOCS,
    CountingEmbedder,
    TokenLevelEmbedder,
    build_store,
    retrieval_ids,
    write_citation_sidecar,
    write_corpus,
)

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


def test_noop_when_corpus_unchanged(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", V1_DOCS)
    build_store(corpus, CountingEmbedder()).save(tmp_path / "rag")
    result = refresh_vector_store(tmp_path / "rag", corpus, embedder=CountingEmbedder())
    assert result.refreshed is False
    assert result.generation_dir is None
    assert not (tmp_path / "rag" / "generations").exists()


def test_add_modify_delete_matches_rebuild_faiss_flat(tmp_path):
    corpus, index_dir = _setup(tmp_path)
    embedder = CountingEmbedder()
    result = refresh_vector_store(index_dir, corpus, embedder=embedder, timestamp=TS)
    assert result.refreshed
    assert result.diff.added == ["d.md"]
    assert result.diff.modified == ["b.md"]
    assert result.diff.deleted == ["c.md"]
    rebuilt = build_store(corpus, CountingEmbedder())
    _assert_equivalent(result.new_store, rebuilt)
    # only changed documents' chunks reached the embedder
    assert embedder.embedded_texts
    assert all(
        text in V2_DOCS["b.md"] or text in V2_DOCS["d.md"] for text in embedder.embedded_texts
    )
    assert result.n_embedded == len(embedder.embedded_texts)
    assert result.n_reused == len(rebuilt.chunks) - result.n_embedded
    assert result.n_reused > 0


def test_refresh_publishes_immutable_generation(tmp_path):
    corpus, index_dir = _setup(tmp_path)
    before = (index_dir / CHUNKS_FILE).read_bytes()
    result = refresh_vector_store(index_dir, corpus, embedder=CountingEmbedder(), timestamp=TS)
    # the source store is untouched (rollback unit); the generation holds the refreshed store
    assert (index_dir / CHUNKS_FILE).read_bytes() == before
    generation = result.generation_dir
    assert generation == index_dir / "generations" / TS
    assert (generation / CHUNKS_FILE).is_file() and (generation / META_FILE).is_file()
    meta = json.loads((generation / META_FILE).read_text(encoding="utf-8"))
    assert meta["refreshed_from"] == str(index_dir)
    # resolution serves the new generation; deleting it rolls back to the source store
    assert resolve_store_dir(index_dir, META_FILE) == generation
    for file in sorted(generation.rglob("*")):
        if file.is_file():
            file.unlink()
    generation.rmdir()
    assert resolve_store_dir(index_dir, META_FILE) == index_dir


def test_deletion_only_refresh_embeds_nothing_and_retires_chunks(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", V1_DOCS)
    build_store(corpus, CountingEmbedder(), mode="hybrid").save(tmp_path / "rag")
    v1_minus_c = {name: text for name, text in V1_DOCS.items() if name != "c.md"}
    write_corpus(corpus, v1_minus_c)
    embedder = CountingEmbedder()
    result = refresh_vector_store(tmp_path / "rag", corpus, embedder=embedder, timestamp=TS)
    assert result.refreshed and result.n_embedded == 0 and embedder.passage_calls == []
    new_store = result.new_store
    # deletion propagates to the persisted chunks, the dense matrix, and the lexical postings
    assert all(chunk["doc_id"] != "c.md" for chunk in new_store.chunks)
    assert len(np.asarray(stored_vectors(new_store.index))) == len(new_store.chunks)
    assert "унікальний-термін-лесі" not in new_store.lexical.postings
    _assert_equivalent(new_store, build_store(corpus, CountingEmbedder(), mode="hybrid"))


def test_hybrid_lexical_matches_rebuild(tmp_path):
    corpus, index_dir = _setup(tmp_path, mode="hybrid")
    result = refresh_vector_store(index_dir, corpus, embedder=CountingEmbedder(), timestamp=TS)
    rebuilt = build_store(corpus, CountingEmbedder(), mode="hybrid")
    _assert_equivalent(result.new_store, rebuilt)


def test_hybrid_lemmatized_merge_tokenizes_only_new_texts(tmp_path):
    lemma_calls: list[str] = []

    def fake_lemmatizer(token: str) -> str:
        lemma_calls.append(token)
        return token.rstrip("иіау")  # crude deterministic stemmer, same on both paths

    corpus, index_dir = _setup(tmp_path, mode="hybrid", lemmatizer=fake_lemmatizer)
    lemma_calls.clear()
    result = refresh_vector_store(
        index_dir,
        corpus,
        embedder=CountingEmbedder(),
        lemmatizer=fake_lemmatizer,
        timestamp=TS,
    )
    changed_tokens = set(lemma_calls)
    rebuilt = build_store(corpus, CountingEmbedder(), mode="hybrid", lemmatizer=fake_lemmatizer)
    _assert_equivalent(result.new_store, rebuilt)
    # unchanged chunks were recovered from the old postings, never re-lemmatized: no token
    # unique to the untouched a.md reaches the lemmatizer during the refresh
    assert changed_tokens  # the changed docs' texts were tokenized
    assert "шевченко" not in changed_tokens
    assert "кобзар" not in changed_tokens


def test_parent_child_matches_rebuild(tmp_path):
    corpus, index_dir = _setup(tmp_path, mode="parent_child")
    result = refresh_vector_store(index_dir, corpus, embedder=CountingEmbedder(), timestamp=TS)
    rebuilt = build_store(corpus, CountingEmbedder(), mode="parent_child")
    _assert_equivalent(result.new_store, rebuilt)
    assert result.new_store.parents is not None


@pytest.mark.parametrize(
    "backend,module",
    [("chroma", "chromadb"), ("qdrant", "qdrant_client"), ("lancedb", "lancedb")],
)
def test_alternative_vector_backends_match_rebuild(tmp_path, backend, module):
    pytest.importorskip(module)
    corpus, index_dir = _setup(tmp_path, backend=backend)
    result = refresh_vector_store(index_dir, corpus, embedder=CountingEmbedder(), timestamp=TS)
    rebuilt = build_store(corpus, CountingEmbedder(), backend=backend)
    _assert_equivalent(result.new_store, rebuilt, check_retrieval=False)


def test_legacy_store_without_doc_fingerprints_refreshes_fully(tmp_path):
    corpus, index_dir = _setup(tmp_path)
    meta_path = index_dir / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["doc_fingerprints"]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    embedder = CountingEmbedder()
    result = refresh_vector_store(index_dir, corpus, embedder=embedder, timestamp=TS)
    assert result.refreshed and result.n_reused == 0
    _assert_equivalent(result.new_store, build_store(corpus, CountingEmbedder()))


def test_refresh_refuses_missing_store_and_empty_corpus(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", V1_DOCS)
    with pytest.raises(SystemExit, match="no RAG store"):
        refresh_vector_store(tmp_path / "rag", corpus)
    build_store(corpus, CountingEmbedder()).save(tmp_path / "rag")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no documents"):
        refresh_vector_store(tmp_path / "rag", empty)


def test_sidecar_only_change_reannotates_the_documents_chunks(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", V1_DOCS)
    write_citation_sidecar(corpus, "a.md", page=1)
    build_store(corpus, CountingEmbedder()).save(tmp_path / "rag")
    # regenerate the page spans only: the document text is untouched
    write_citation_sidecar(corpus, "a.md", page=7)
    embedder = CountingEmbedder()
    result = refresh_vector_store(tmp_path / "rag", corpus, embedder=embedder, timestamp=TS)
    assert result.refreshed
    assert result.diff.modified == ["a.md"] and not result.diff.added and not result.diff.deleted
    # annotation-only fast path: records are rewritten, no chunk reaches the embedder
    assert result.n_embedded == 0 and embedder.passage_calls == []
    assert result.n_reused == len(result.new_store.chunks)
    a_chunks = [c for c in result.new_store.chunks if c["doc_id"] == "a.md"]
    assert a_chunks and all(c["metadata"]["pages"] == [7, 7] for c in a_chunks)
    _assert_equivalent(result.new_store, build_store(corpus, CountingEmbedder()))
    # the refreshed generation records the sidecar-aware fingerprints: a second pass is a no-op
    again = refresh_vector_store(tmp_path / "rag", corpus, embedder=CountingEmbedder())
    assert again.refreshed is False


@pytest.mark.parametrize("mode", ["hybrid", "parent_child"])
def test_annotation_only_fast_path_per_store_mode(tmp_path, mode):
    corpus = write_corpus(tmp_path / "corpus", V1_DOCS)
    write_citation_sidecar(corpus, "b.md", page=2)
    build_store(corpus, CountingEmbedder(), mode=mode).save(tmp_path / "rag")
    write_citation_sidecar(corpus, "b.md", page=9)
    embedder = CountingEmbedder()
    result = refresh_vector_store(tmp_path / "rag", corpus, embedder=embedder, timestamp=TS)
    assert result.refreshed and result.n_embedded == 0 and embedder.passage_calls == []
    _assert_equivalent(result.new_store, build_store(corpus, CountingEmbedder(), mode=mode))


def test_same_span_text_edit_still_reembeds(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", V1_DOCS)
    build_store(corpus, CountingEmbedder()).save(tmp_path / "rag")
    # equal-length replacement keeps every chunk span identical; only the text differs
    edited = {**V1_DOCS, "a.md": V1_DOCS["a.md"].replace("Кобзар", "Гайдам")}
    write_corpus(corpus, edited)
    embedder = CountingEmbedder()
    result = refresh_vector_store(tmp_path / "rag", corpus, embedder=embedder, timestamp=TS)
    assert result.diff.modified == ["a.md"]
    assert result.n_embedded > 0 and embedder.embedded_texts
    _assert_equivalent(result.new_store, build_store(corpus, CountingEmbedder()))


def test_late_strategy_matches_rebuild(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", V1_DOCS)
    build_store(corpus, TokenLevelEmbedder(), strategy="late").save(tmp_path / "rag")
    write_corpus(corpus, V2_DOCS)
    embedder = TokenLevelEmbedder()
    result = refresh_vector_store(tmp_path / "rag", corpus, embedder=embedder, timestamp=TS)
    assert result.refreshed and result.n_reused > 0
    _assert_equivalent(result.new_store, build_store(corpus, TokenLevelEmbedder(), strategy="late"))
    # late re-encoding (encode_store_vectors per changed doc) touched only the changed documents
    assert embedder.token_windows
    assert all(
        window in V2_DOCS["b.md"] or window in V2_DOCS["d.md"] for window in embedder.token_windows
    )


def test_sibling_comparison_stores_refresh_to_rebuild_equivalence(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", V1_DOCS)
    index_dir = tmp_path / "rag"
    build_store(corpus, TokenLevelEmbedder()).save(index_dir)
    build_store(corpus, TokenLevelEmbedder(), strategy="sentence").save(index_dir / "sentence")
    build_store(corpus, TokenLevelEmbedder(), strategy="late").save(index_dir / "late")
    write_corpus(corpus, V2_DOCS)
    # the main store's refresh leaves a generations/ child behind; it is never a sibling
    refresh_vector_store(index_dir, corpus, embedder=TokenLevelEmbedder(), timestamp=TS)
    assert [d.name for d in sibling_store_dirs(index_dir)] == ["late", "sentence"]
    results = refresh_sibling_stores(index_dir, corpus, embedder=TokenLevelEmbedder(), timestamp=TS)
    assert [name for name, _ in results] == ["late", "sentence"]
    for name, result in results:
        assert result.refreshed
        assert result.generation_dir == index_dir / name / "generations" / TS
        _assert_equivalent(
            result.new_store, build_store(corpus, TokenLevelEmbedder(), strategy=name)
        )
    # every sibling now serves the refreshed generation, and a second pass is a per-store no-op
    again = refresh_sibling_stores(index_dir, corpus, embedder=TokenLevelEmbedder())
    assert [name for name, _ in again] == ["late", "sentence"]
    assert all(not result.refreshed for _, result in again)


def test_second_refresh_chains_from_the_previous_generation(tmp_path):
    corpus, index_dir = _setup(tmp_path)
    refresh_vector_store(index_dir, corpus, embedder=CountingEmbedder(), timestamp=TS)
    v3 = {**V2_DOCS, "e.md": "Григорій Сковорода був мандрівним філософом і поетом."}
    write_corpus(corpus, v3)
    embedder = CountingEmbedder()
    result = refresh_vector_store(
        index_dir, corpus, embedder=embedder, timestamp="20990102T000000Z"
    )
    assert result.source_dir == index_dir / "generations" / TS
    assert result.diff.added == ["e.md"] and not result.diff.modified
    _assert_equivalent(result.new_store, build_store(corpus, CountingEmbedder()))
