"""Lexical BM25 + Ukrainian normalization + RRF fusion (hybrid-retrieval-uk core, pure Python)."""

import pytest

from llb.rag.lexical import (
    LexicalIndex,
    normalize_token,
    rrf_fuse,
    tokenize,
)


def test_normalize_token_unifies_apostrophe_variants_and_casefolds():
    # U+2019, U+02BC, and ASCII ' all normalize to the same token
    assert normalize_token("м’яч") == "м'яч"
    assert normalize_token("мʼяч") == "м'яч"
    assert normalize_token("М'ЯЧ") == "м'яч"


def test_tokenize_strips_punctuation_and_keeps_numbers():
    tokens = tokenize("Наказ № 4821, від 12.03.2024 (ДСТУ 8134:2020)!")
    assert tokens == ["наказ", "4821", "від", "12", "03", "2024", "дсту", "8134", "2020"]


def test_tokenize_applies_injected_lemmatizer():
    fake = {"начальника": "начальник", "служби": "служба"}.get
    tokens = tokenize("начальника служби", lambda t: fake(t) or t)
    assert tokens == ["начальник", "служба"]


def _index(texts, **kwargs):
    return LexicalIndex.build(texts, **kwargs)


def test_bm25_ranks_the_exact_term_chunk_first():
    idx = _index(
        [
            "Наказ № 4801 видано начальником служби.",
            "Наказ № 4802 видано начальником служби.",
            "Наказ № 4821 видано начальником служби.",
        ]
    )
    ranked = idx.search("Хто видав наказ № 4821?", k=3)
    assert ranked[0][0] == 2  # the chunk containing the exact number wins


def test_bm25_is_deterministic_and_breaks_ties_by_ordinal():
    texts = ["однакові слова тут", "однакові слова тут", "інший текст зовсім"]
    a = _index(texts).search("однакові слова", k=3)
    b = _index(texts).search("однакові слова", k=3)
    assert a == b
    assert [ordinal for ordinal, _ in a] == [0, 1]  # equal scores -> build order


def test_bm25_allowed_set_restricts_candidates():
    idx = _index(["наказ один", "наказ два", "наказ три"])
    ranked = idx.search("наказ", k=3, allowed={1})
    assert [ordinal for ordinal, _ in ranked] == [1]


def test_lemmatized_index_matches_inflected_query():
    lemmas = {"начальника": "начальник", "начальник": "начальник"}
    idx = _index(
        ["начальник служби затвердив", "зовсім інша тема"],
        lemmatize=True,
        lemmatizer=lambda t: lemmas.get(t, t),
    )
    # genitive query form collapses to the nominative corpus lemma
    ranked = idx.search("начальника", k=2)
    assert ranked and ranked[0][0] == 0


def test_lexical_index_save_load_round_trip(tmp_path):
    idx = _index(["наказ № 4821 тут", "інший запис"])
    idx.save(tmp_path / "lexical_index.json")
    loaded = LexicalIndex.load(tmp_path / "lexical_index.json")
    assert loaded.search("4821", k=2) == idx.search("4821", k=2)
    assert loaded.lemmatize is False


def test_build_with_lemmas_never_mutates_the_texts():
    texts = ["Начальника служби призначено.", "М’яч на полі."]
    snapshot = list(texts)
    _index(texts, lemmatize=True, lemmatizer=lambda t: t[:4])
    assert texts == snapshot  # normalization/lemmas live in the index only


def test_rrf_weight_extremes_reproduce_each_side():
    dense = [10, 11, 12]
    lexical = [12, 11, 10]
    assert [cid for cid, _ in rrf_fuse(dense, lexical, 1.0)][:3] == dense
    assert [cid for cid, _ in rrf_fuse(dense, lexical, 0.0)][:3] == lexical


def test_rrf_balanced_fusion_prefers_agreement():
    # id 5 is mid-rank on BOTH sides; id 1 is top-dense only, id 9 top-lexical only
    fused = rrf_fuse([1, 5, 2], [9, 5, 3], 0.5)
    assert fused[0][0] == 5


def test_rrf_rejects_out_of_range_weight():
    with pytest.raises(ValueError, match="fusion weight"):
        rrf_fuse([1], [2], 1.5)


def test_rrf_zero_weight_side_does_not_append_disabled_candidates():
    assert [cid for cid, _ in rrf_fuse([1], [2, 3], 1.0)] == [1]
    assert [cid for cid, _ in rrf_fuse([1, 2], [3], 0.0)] == [3]
