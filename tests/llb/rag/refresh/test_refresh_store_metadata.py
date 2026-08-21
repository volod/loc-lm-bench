"""Focused tests split from ``test_refresh_store.py``."""

import pytest
from tests.llb.rag._refresh_store_helpers import (
    TS,
    _assert_equivalent,
    _setup,
)
from tests.llb.rag.refresh_helpers import (
    V1_DOCS,
    V2_DOCS,
    CountingEmbedder,
    TokenLevelEmbedder,
    build_store,
    write_citation_sidecar,
    write_corpus,
)

from llb.rag.refresh.siblings import (
    refresh_sibling_stores,
    sibling_store_dirs,
)
from llb.rag.refresh.store_refresh import refresh_vector_store

pytestmark = pytest.mark.heavy_env


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
