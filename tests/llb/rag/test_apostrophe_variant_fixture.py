"""The planted mixed-apostrophe-variant fixture: its variant mix and what it buys the lexical lane.

Pure: the real `LexicalIndex` over the committed corpus, no dense side, no GPU, no `[rag]` extra.
The "pre-fix" arm restores the v1 tokenizer (apostrophe variants were NOT in-word characters and
unification ran after tokenizing), so the delta the shipped tokenizer is worth on a corpus that
MIXES variants is measured here rather than argued -- per variant, since the pre-fix tokenizer
broke on only two of the four.
"""

import re
from pathlib import Path

import pytest

from llb.goldset.schema import load_goldset
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.lexical import LexicalIndex
from llb.rag.retrieval import evaluate_retrieval

FIXTURE = Path("samples/goldsets/apostrophe_variants_uk")
KEYBOARD = "'"
VARIANTS = {"keyboard": KEYBOARD, "typographic": "’", "modifier": "ʼ", "grave": "`"}
# What each document was "converted" with (README): the corpus mixes variants, each document is
# internally consistent except the copy-pasted appendix, which alternates between two sources.
DOC_VARIANTS = {
    "reyestr-osnovnyy.md": {"keyboard"},
    "reyestr-perevydannya.md": {"typographic"},
    "reyestr-arkhiv.md": {"modifier"},
    "dodatok-zmishanyy.md": {"typographic", "grave"},
}
ENTRIES_PER_VARIANT = {"keyboard": 15, "typographic": 23, "modifier": 15, "grave": 7}
# Only the punctuation-class variants were unreachable before the fix: U+02BC is a Unicode
# modifier LETTER, so `\w` kept `памʼятка` whole even under v1, while U+2019 and the grave
# accent split it into two half-words.
RESCUED_VARIANTS = ("typographic", "grave")
# v1 token regex: only the ASCII apostrophe was an explicit in-word character.
V1_TOKEN_RE = re.compile(r"[\w']+")
K = 10


def _corpus_text(name: str) -> str:
    return (FIXTURE / "corpus" / name).read_text(encoding="utf-8")


def _items():
    return load_goldset(FIXTURE / "goldset.jsonl")


def _entry_text(item) -> str:
    span = item.source_spans[0]
    return _corpus_text(span.doc_id)[span.char_start : span.char_end]


def _entry_variant(item) -> str:
    """Which apostrophe the gold entry was written with (the question always types U+0027)."""
    return next(label for label, char in VARIANTS.items() if char in _entry_text(item))


def _subject_term(item) -> str:
    """The one apostrophe-bearing noun the question asks about (the only discriminating token)."""
    return item.question.rsplit("— ", 1)[1].rstrip("?")


def _by_variant(items) -> dict[str, list]:
    grouped: dict[str, list] = {label: [] for label in VARIANTS}
    for item in items:
        grouped[_entry_variant(item)].append(item)
    return grouped


def test_corpus_mixes_apostrophe_variants_across_documents():
    for name, expected in DOC_VARIANTS.items():
        text = _corpus_text(name)
        assert {label for label, char in VARIANTS.items() if char in text} == expected, name
    assert {variant for expected in DOC_VARIANTS.values() for variant in expected} == set(VARIANTS)


def test_every_question_types_the_keyboard_apostrophe_over_a_unique_subject_term():
    items = _items()
    assert len(items) == 60
    subjects = [_subject_term(item) for item in items]
    assert len(set(subjects)) == len(subjects)  # one discriminating token per entry
    for item, subject in zip(items, subjects):
        assert KEYBOARD in subject
        assert not any(char in item.question for char in "’ʼ`")
        # the entry states the same term, written in ITS document's variant
        variant = _entry_variant(item)
        assert subject.replace(KEYBOARD, VARIANTS[variant]) in _entry_text(item)
    assert {label: len(group) for label, group in _by_variant(items).items()} == ENTRIES_PER_VARIANT


@pytest.mark.slow
def test_subject_term_reaches_a_mismatched_entry_only_with_the_shipped_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
):
    """The tokenizer's whole purpose, measured: a term typed with one variant must find the entry
    that wrote it with another. Under v1 a term facing a punctuation-class variant returned ZERO
    BM25 candidates -- the query kept it whole while the index held two half-words."""
    chunks = chunk_corpus(FIXTURE / "corpus", "recursive", 800, 120, None)
    grouped = _by_variant(_items())

    def candidates() -> dict[str, int]:
        index = LexicalIndex.build([c["text"] for c in chunks])
        return {
            label: sum(bool(index.search(_subject_term(item), K)) for item in group)
            for label, group in grouped.items()
        }

    assert candidates() == ENTRIES_PER_VARIANT
    monkeypatch.setattr("llb.rag.lexical._TOKEN_RE", V1_TOKEN_RE)
    assert candidates() == {
        label: 0 if label in RESCUED_VARIANTS else count
        for label, count in ENTRIES_PER_VARIANT.items()
    }


@pytest.mark.slow
def test_lexical_recall_on_the_mixed_corpus_halves_under_the_variant_blind_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
):
    chunks = chunk_corpus(FIXTURE / "corpus", "recursive", 800, 120, None)
    items = _items()
    grouped = _by_variant(items)
    assert len(chunks) > K  # recall@k must be non-trivial

    def lexical_recall(subset) -> float:
        index = LexicalIndex.build([c["text"] for c in chunks])
        pairs = [
            (
                [chunks[cid] for cid, _ in index.search(item.question, K)],
                [span.model_dump() for span in item.source_spans],
            )
            for item in subset
        ]
        return evaluate_retrieval(pairs, K)["recall_at_k"]

    assert lexical_recall(items) == 1.0
    monkeypatch.setattr("llb.rag.lexical._TOKEN_RE", V1_TOKEN_RE)
    for label, group in grouped.items():
        # no residue: a rescued item is not retrieved by boilerplate ties either
        assert lexical_recall(group) == (0.0 if label in RESCUED_VARIANTS else 1.0), label
    assert lexical_recall(items) == 0.5  # 30 of 60 entries use a punctuation-class variant
