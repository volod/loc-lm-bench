"""Duplicate collapse through a real store: index budget, retrieval, tie floor, and refresh.

Every test builds a store over the committed `duplicate_chunks_uk_v1` fixture (or a tiny inline
corpus) with the fake hashed-BoW embedder, so identical text embeds to an identical vector -- the
exact-tie mechanism the collapse exists to remove -- with no GPU and no sentence-transformers.
The dense index still goes through the real vector-index seam, so the module needs `[rag]`.
"""

import numpy as np
import pytest

from llb.rag.duplicate_tiers import TIER_EXACT, TIER_MASKED
from llb.rag.duplicates import duplicate_occurrences
from llb.rag.noise_floor import measure_noise_floor
from llb.rag.refresh.store_refresh import refresh_vector_store, stored_vectors
from llb.rag.retrieval import recall_at_k
from llb.rag.store import RagStore

from refresh_helpers import CountingEmbedder, retrieval_ids, write_corpus
from test_duplicates import (
    FIXTURE,
    FIXTURE_CHUNKS,
    FIXTURE_OVERLAP,
    FIXTURE_SIZE,
    FIXTURE_STRATEGY,
    FIXTURE_UNIQUE,
)

pytestmark = pytest.mark.heavy_env

SERVICE_QUESTION = "Який телефон гарячої лінії сервісної служби?"
FURNITURE = "Телефон гарячої лінії"

SHARED_BLOCK = (
    "## Загальні положення\n\n"
    "Цей документ підготовлено відповідно до вимог чинного законодавства.\n"
)
V1_DUP_DOCS = {
    "a.md": f"# А\n\n{SHARED_BLOCK}\n## Розділ А\n\nНасос подає двісті кубічних метрів.\n",
    "b.md": f"# Б\n\n{SHARED_BLOCK}\n## Розділ Б\n\nКомпресор дає сім кубічних метрів.\n",
    "c.md": f"# В\n\n{SHARED_BLOCK}\n## Розділ В\n\nВентилятор дає дванадцять тисяч.\n",
}
# v2 modifies b.md and adds d.md, both repeating the shared block a.md already carries.
V2_DUP_DOCS = {
    "a.md": V1_DUP_DOCS["a.md"],
    "b.md": f"# Б\n\n{SHARED_BLOCK}\n## Розділ Б\n\nКомпресор дає вісім кубічних метрів.\n",
    "c.md": V1_DUP_DOCS["c.md"],
    "d.md": f"# Г\n\n{SHARED_BLOCK}\n## Розділ Г\n\nТурбіна дає сорок мегават.\n",
}
# a.md carries the SURVIVING copy of the shared block, so deleting it is the case an incremental
# refresh gets wrong unless collapse is undone before the merge and re-applied after it.
V2_DELETED_SURVIVOR_DOCS = {k: v for k, v in V2_DUP_DOCS.items() if k != "a.md"}
# a.md is EDITED but keeps the shared block: the survivor's own document is the one that changed,
# so its fresh copy re-embeds the shared block (which unchanged b.md/c.md still hold) unless the
# reuse is keyed on stored text rather than on chunk position.
V2_EDITED_SURVIVOR_DOCS = {
    "a.md": f"# А\n\n{SHARED_BLOCK}\n## Розділ А\n\nНасос подає триста кубічних метрів.\n",
    "b.md": V1_DUP_DOCS["b.md"],
    "c.md": V1_DUP_DOCS["c.md"],
}
DUP_QUESTIONS = ["Що подає насос?", "Скільки дає компресор?", "Про що загальні положення?"]

# The same shared block carrying a PAGE NUMBER, so the copies are equivalent only at the `masked`
# tier -- the survivor's stored vector encodes a.md's wording, not b.md's.
NUMBERED_BLOCK = (
    "## Колонтитул\n\nСторінка {page} з 12. Редакція документа. Внутрішній службовий обіг.\n"
)
V1_NEAR_DUP_DOCS = {
    f"{name}.md": f"# {name}\n\n{NUMBERED_BLOCK.format(page=page)}\n## Розділ\n\n{body}\n"
    for name, page, body in (
        ("a", 1, "Насос подає двісті кубічних метрів на годину."),
        ("b", 4, "Компресор дає сім кубічних метрів на хвилину."),
        ("c", 9, "Вентилятор дає дванадцять тисяч метрів."),
    )
}
V2_NEAR_DUP_DOCS = {k: v for k, v in V1_NEAR_DUP_DOCS.items() if k != "a.md"}


def _fixture_store(collapse: bool = True) -> RagStore:
    return RagStore.build(
        FIXTURE,
        FIXTURE_STRATEGY,
        FIXTURE_SIZE,
        FIXTURE_OVERLAP,
        embedder=CountingEmbedder(),
        collapse_duplicates=collapse,
    )


def _dup_store(corpus, embedder=None, tier=TIER_EXACT) -> RagStore:
    """One chunk per `##` section (size 120 keeps sections apart), so the shared block repeats."""
    return RagStore.build(
        corpus, "heading", 120, 0, embedder=embedder or CountingEmbedder(), duplicate_tier=tier
    )


def test_store_indexes_each_distinct_chunk_once():
    store = _fixture_store()
    assert store.meta["n_indexed"] == FIXTURE_UNIQUE
    assert len(store.chunks) == FIXTURE_UNIQUE
    assert store.meta["collapse_duplicates"] is True
    assert store.meta["duplicates"]["n"] == FIXTURE_CHUNKS
    assert store.meta["duplicates"]["collapsed"] == FIXTURE_CHUNKS - FIXTURE_UNIQUE
    assert np.asarray(stored_vectors(store.index)).shape[0] == FIXTURE_UNIQUE


def test_keeping_duplicates_still_reports_what_they_cost():
    store = _fixture_store(collapse=False)
    assert store.meta["n_indexed"] == FIXTURE_CHUNKS
    assert store.meta["collapse_duplicates"] is False
    assert store.meta["duplicates"]["duplicate_share"] == pytest.approx(0.75)
    assert all(not duplicate_occurrences(chunk) for chunk in store.chunks)


def test_a_collapsed_chunk_is_still_retrievable_for_every_document_it_appears_in():
    store = _fixture_store()
    hits = store.retrieve(SERVICE_QUESTION, 3)
    survivor = next(hit for hit in hits if FURNITURE in hit["text"])
    places = [survivor["doc_id"], *(copy["doc_id"] for copy in duplicate_occurrences(survivor))]
    assert len(places) == len(set(places)) == 3  # all three manuals, once each
    for place in places:
        span = {
            "doc_id": place,
            "char_start": survivor["char_start"]
            if place == survivor["doc_id"]
            else next(
                c["char_start"] for c in duplicate_occurrences(survivor) if c["doc_id"] == place
            ),
            "char_end": survivor["char_end"]
            if place == survivor["doc_id"]
            else next(
                c["char_end"] for c in duplicate_occurrences(survivor) if c["doc_id"] == place
            ),
            "text": survivor["text"],
        }
        assert recall_at_k(hits, [span], 3) == 1.0


def test_collapse_removes_the_tie_driven_measurement_floor():
    """The plan's premise, measured: identical chunks tie, and the tie IS the floor."""
    items = [(SERVICE_QUESTION, [{"doc_id": "x", "char_start": 0, "char_end": 1, "text": "g"}])]
    report = measure_noise_floor(
        {"keep": _fixture_store(collapse=False), "collapse": _fixture_store()},
        items,
        k=2,
        replicates=8,
    )
    assert report["lanes"]["keep"]["fragile_items"] == 1
    assert report["lanes"]["collapse"]["fragile_items"] == 0


def test_refresh_matches_a_rebuild_when_the_surviving_copys_document_is_deleted(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", V1_DUP_DOCS)
    index_dir = tmp_path / "rag"
    _dup_store(corpus).save(index_dir)
    write_corpus(corpus, V2_DELETED_SURVIVOR_DOCS)

    result = refresh_vector_store(index_dir, corpus, embedder=CountingEmbedder(), timestamp="T")
    rebuilt = _dup_store(corpus)
    assert result.new_store is not None
    assert result.new_store.chunks == rebuilt.chunks
    assert result.new_store.meta["duplicates"] == rebuilt.meta["duplicates"]
    np.testing.assert_array_equal(
        np.asarray(stored_vectors(result.new_store.index)),
        np.asarray(stored_vectors(rebuilt.index)),
    )
    assert retrieval_ids(result.new_store, DUP_QUESTIONS) == retrieval_ids(rebuilt, DUP_QUESTIONS)
    # the shared block survives the deletion of the document that used to carry it
    survivor = next(c for c in rebuilt.chunks if "Загальні положення" in str(c["text"]))
    assert survivor["doc_id"] == "b.md"
    assert [copy["doc_id"] for copy in duplicate_occurrences(survivor)] == ["c.md", "d.md"]


def test_refresh_only_embeds_the_changed_documents_distinct_text(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", V1_DUP_DOCS)
    index_dir = tmp_path / "rag"
    _dup_store(corpus).save(index_dir)
    write_corpus(corpus, V2_DUP_DOCS)
    embedder = CountingEmbedder()

    refresh_vector_store(index_dir, corpus, embedder=embedder, timestamp="T")
    # the modified and added documents repeat the shared block that unchanged a.md still carries,
    # so only their own sections are new text -- the repeat costs no embedding call
    assert embedder.embedded_texts
    assert all("Загальні положення" not in text for text in embedder.embedded_texts)
    assert any("Турбіна" in text for text in embedder.embedded_texts)


def test_refresh_recovers_by_text_when_the_edited_document_carried_the_survivor(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", V1_DUP_DOCS)
    index_dir = tmp_path / "rag"
    _dup_store(corpus).save(index_dir)
    # a.md holds the surviving copy of the shared block AND is the document being edited
    write_corpus(corpus, V2_EDITED_SURVIVOR_DOCS)
    embedder = CountingEmbedder()

    result = refresh_vector_store(index_dir, corpus, embedder=embedder, timestamp="T")
    # only a.md's new section is genuinely new text; the shared block a.md re-emits is recovered
    # from the store's own vectors even though the survivor's own document is the one that changed
    assert embedder.embedded_texts
    assert all("Загальні положення" not in text for text in embedder.embedded_texts)
    assert any("триста" in text for text in embedder.embedded_texts)
    assert result.n_reused_by_text == 1  # the one shared-block row a position map would re-embed
    # and the refreshed store is still byte-for-byte a rebuild
    rebuilt = _dup_store(corpus)
    assert result.new_store.chunks == rebuilt.chunks
    np.testing.assert_array_equal(
        np.asarray(stored_vectors(result.new_store.index)),
        np.asarray(stored_vectors(rebuilt.index)),
    )
    assert retrieval_ids(result.new_store, DUP_QUESTIONS) == retrieval_ids(rebuilt, DUP_QUESTIONS)


def test_a_coarse_tier_store_indexes_the_page_numbered_block_once(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", V1_NEAR_DUP_DOCS)
    exact = _dup_store(corpus)
    masked = _dup_store(corpus, tier=TIER_MASKED)
    assert exact.meta["n_indexed"] == masked.meta["n_indexed"] + 2  # three footers -> one
    assert masked.meta["duplicate_tier"] == TIER_MASKED
    survivor = next(c for c in masked.chunks if "Колонтитул" in str(c["text"]))
    # the merged copies differ from the survivor, so each carries its OWN text and offsets
    copies = duplicate_occurrences(survivor)
    assert [copy["doc_id"] for copy in copies] == ["b.md", "c.md"]
    for copy in copies:
        source = (corpus / str(copy["doc_id"])).read_text(encoding="utf-8")
        assert source[copy["char_start"] : copy["char_end"]] == copy["text"] != survivor["text"]


def test_coarse_tier_refresh_reembeds_the_copy_that_becomes_the_survivor(tmp_path):
    """Deleting a.md promotes b.md's DIFFERENT wording; reusing a.md's row would drift."""
    corpus = write_corpus(tmp_path / "corpus", V1_NEAR_DUP_DOCS)
    index_dir = tmp_path / "rag"
    _dup_store(corpus, tier=TIER_MASKED).save(index_dir)
    write_corpus(corpus, V2_NEAR_DUP_DOCS)

    result = refresh_vector_store(index_dir, corpus, embedder=CountingEmbedder(), timestamp="T")
    rebuilt = _dup_store(corpus, tier=TIER_MASKED)
    assert result.new_store is not None
    assert result.new_store.chunks == rebuilt.chunks
    np.testing.assert_array_equal(
        np.asarray(stored_vectors(result.new_store.index)),
        np.asarray(stored_vectors(rebuilt.index)),
    )
    assert retrieval_ids(result.new_store, DUP_QUESTIONS) == retrieval_ids(rebuilt, DUP_QUESTIONS)
