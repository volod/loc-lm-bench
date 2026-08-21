"""Focused tests split from ``test_refresh_store.py``."""

import json

import numpy as np
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
    build_store,
    write_corpus,
)

from llb.core.store_generations import resolve_store_dir
from llb.rag.refresh.store_refresh import (
    refresh_vector_store,
    stored_vectors,
)
from llb.rag.vector_store.build import (
    CHUNKS_FILE,
    META_FILE,
)

pytestmark = pytest.mark.heavy_env


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
    [
        ("chroma", "chromadb"),
        ("qdrant", "qdrant_client"),
        pytest.param("lancedb", "lancedb", marks=pytest.mark.opt_in_env),
    ],
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
    # no fingerprints means every document is treated as added, so the position map reuses nothing;
    # the text-keyed reuse still recovers unchanged a.md's rows from the store's own vectors.
    assert result.refreshed
    assert result.n_reused == result.n_reused_by_text > 0
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
