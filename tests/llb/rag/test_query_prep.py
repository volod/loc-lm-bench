"""Query-side processing lane (uk-query-processing): pure pipeline, glossary, A/B, and wiring."""

import pytest

from llb.rag.query_prep.glossary import (
    Glossary,
    GlossaryEntry,
)
from llb.rag.query_prep.normalize import (
    apply_normalize,
    cyrillic_to_latin,
    transliterate_latin_to_cyrillic,
)
from llb.rag.query_prep.pipeline import QueryPrep
from llb.rag.query_prep.typos import (
    apply_typos,
    build_vocabulary,
    damerau_levenshtein,
    nearest_vocab_token,
)


# --------------------------------------------------------------------------------------------
# normalize: casefold + apostrophe unification + transliteration
# --------------------------------------------------------------------------------------------


def test_normalize_unifies_apostrophes_and_casefolds():
    processed, edits = apply_normalize("М'ЯЧ")
    assert processed == "м'яч"  # U+2019 apostrophe unified to ASCII, casefolded
    assert edits == []  # silent normalization, no transliteration edit


def test_normalize_transliterates_latin_typed_tokens():
    processed, edits = apply_normalize("zakon про pravo")
    assert processed == "закон про право"
    kinds = {(e.original, e.replacement) for e in edits}
    assert ("zakon", "закон") in kinds and ("pravo", "право") in kinds


def test_normalize_leaves_cyrillic_tokens_untouched():
    processed, edits = apply_normalize("рішення суду")
    assert processed == "рішення суду"
    assert edits == []


@pytest.mark.parametrize("word", ["закон", "право", "щит", "якіст", "рішення", "суд"])
def test_transliteration_table_round_trips(word):
    romanized = cyrillic_to_latin(word)
    assert transliterate_latin_to_cyrillic(romanized) == word


def test_romanization_drops_soft_sign():
    assert "ь" not in cyrillic_to_latin("власність")


# --------------------------------------------------------------------------------------------
# typos: Damerau-Levenshtein correction that never touches in-vocabulary tokens
# --------------------------------------------------------------------------------------------


def test_damerau_levenshtein_counts_transposition_as_one():
    assert damerau_levenshtein("наказ", "накза", 2) == 1  # adjacent transposition
    assert damerau_levenshtein("abcd", "abdc", 2) == 1
    assert damerau_levenshtein("наказ", "приказ", 2) == 3  # bounded -> max+1


def test_typos_correct_out_of_vocabulary_token():
    vocab = build_vocabulary(["наказ видано начальником служби"])
    processed, edits = apply_typos("виданоо начальнком", vocab)
    assert processed == "видано начальником"
    assert {(e.original, e.replacement) for e in edits} == {
        ("виданоо", "видано"),
        ("начальнком", "начальником"),
    }


def test_typos_never_alter_in_vocabulary_token():
    vocab = build_vocabulary(["наказ видано начальником", "накат хвилі"])
    # "наказ" IS in the corpus; even though "накат" is one edit away, it must stay unchanged.
    processed, edits = apply_typos("наказ", vocab)
    assert processed == "наказ"
    assert edits == []


def test_typos_leave_numeric_codes_untouched():
    vocab = build_vocabulary(["наказ 4821 від 2024"])
    processed, edits = apply_typos("4822", vocab)  # a code one edit from 4821
    assert processed == "4822"
    assert edits == []


def test_typos_long_token_allows_distance_two():
    vocab = build_vocabulary(["інтелектуальної власності"])
    processed, _ = apply_typos("інтелектуальнох", vocab)  # 12 chars, 2 edits away
    assert processed == "інтелектуальної"


def test_nearest_vocab_token_is_deterministic_under_ties():
    # "хіт" is one edit from BOTH; the lexicographically smaller candidate wins deterministically
    vocab = frozenset({"кіт", "літ"})
    assert nearest_vocab_token("хіт", vocab, 1) == "кіт"


# --------------------------------------------------------------------------------------------
# typos morphology guard (morphology-aware-typo-guard): a valid inflection is not a misspelling
# --------------------------------------------------------------------------------------------


def test_typo_guard_skips_known_word_form_but_still_corrects_misspelling():
    vocab = build_vocabulary(["документа поділяти наказ"])
    known = {"документами"}.__contains__  # fake probe: the inflection is a known word form
    # unguarded: the valid inflection is "corrected" to the corpus surface form
    unguarded, _ = apply_typos("документами", vocab)
    assert unguarded == "документа"
    # guarded: the known inflection stays; lemmatization is the lane that matches it
    guarded, edits = apply_typos("документами", vocab, known_word=known)
    assert guarded == "документами"
    assert edits == []
    # a genuine misspelling stays unknown to the probe and is still corrected
    corrected, edits = apply_typos("накза", vocab, known_word=known)
    assert corrected == "наказ"
    assert [(e.original, e.replacement) for e in edits] == [("накза", "наказ")]


def test_typo_guard_requires_typos_step():
    with pytest.raises(ValueError, match="typo morphology guard"):
        QueryPrep.build(("normalize",), known_word=lambda token: True)


def test_pipeline_threads_typo_guard_probe():
    vocab = build_vocabulary(["документа наказ"])
    pipeline = QueryPrep.build(
        ("typos",), vocabulary=vocab, known_word={"документами"}.__contains__
    )
    assert pipeline.process("документами").processed == "документами"


# --------------------------------------------------------------------------------------------
# glossary: deterministic alias expansion + builder
# --------------------------------------------------------------------------------------------


def _glossary():
    return Glossary(
        (
            GlossaryEntry("інтелектуальна власність", ("ІВ", "intelektualna vlasnist")),
            GlossaryEntry("авторське право", ()),
        )
    )


# --------------------------------------------------------------------------------------------
# rewrite: off by default, injected callable
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# pipeline: ordering, exact no-op, dependency validation
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# A/B report over a fake retriever
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# graph wiring: raw question preserved, processed query retrieved with, both recorded
# --------------------------------------------------------------------------------------------


class _RecordingStore:
    def __init__(self, chunks):
        self.chunks = chunks  # mirrors RagStore.chunks (query-prep reads it for the vocabulary)
        self.seen: list[str] = []

    def retrieve(self, question, k):
        self.seen.append(question)
        return self.chunks[:k]


# --------------------------------------------------------------------------------------------
# runner resolver: dependency wiring from RunConfig + store + launcher
# --------------------------------------------------------------------------------------------
