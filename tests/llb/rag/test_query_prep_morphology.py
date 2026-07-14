"""Tests for query prep morphology."""

from llb.rag.query_prep.typos import (
    apply_typos,
    build_vocabulary,
)


def test_typo_guard_with_real_pymorphy_probe():
    from llb.rag.lexical import load_uk_word_probe

    known = load_uk_word_probe()
    vocab = build_vocabulary(["поділяти документа"])
    # both plan examples: grammatically valid inflections survive the guard
    assert apply_typos("поділяють документами", vocab, known_word=known)[0] == (
        "поділяють документами"
    )
    # the misspelling "поділяяти" is unknown to pymorphy3 and is still corrected
    assert apply_typos("поділяяти", vocab, known_word=known)[0] == "поділяти"
