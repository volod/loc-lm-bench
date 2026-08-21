"""Ambiguity-aware restoration: surface compatibility, morphology, context, and refusal.

The constraints that decide WHICH corpus surface the `typos` step may restore a noisy token to,
and when restoring at all is a guess rather than a repair.
"""

import pytest

from llb.rag.query_prep.base import (
    KIND_HOMOGLYPH,
    KIND_TRANSLITERATE,
    KIND_TYPO,
    STEP_NORMALIZE,
    STEP_TYPOS,
    QueryEdit,
)
from llb.rag.query_prep.pipeline import QueryPrep
from llb.rag.query_prep.restore import (
    TokenProvenance,
    VocabularyContext,
    normalization_provenance,
    select_restoration,
    surface_distance,
)
from llb.rag.query_prep.typos import apply_typos, build_vocabulary


def test_normalization_provenance_keeps_single_origin_and_drops_ambiguous_ones():
    edits = [
        QueryEdit(STEP_NORMALIZE, KIND_TRANSLITERATE, original="sut", replacement="сут"),
        QueryEdit(STEP_NORMALIZE, KIND_HOMOGLYPH, original="нaказ", replacement="наказ"),
        # two different noisy forms collapsing onto one token leave no usable constraint
        QueryEdit(STEP_NORMALIZE, KIND_TRANSLITERATE, original="myach", replacement="мяч"),
        QueryEdit(STEP_NORMALIZE, KIND_TRANSLITERATE, original="mjach", replacement="мяч"),
        QueryEdit(STEP_TYPOS, KIND_TYPO, original="накза", replacement="наказ"),
    ]
    provenance = normalization_provenance(edits)
    assert provenance["сут"] == TokenProvenance("sut", KIND_TRANSLITERATE)
    assert provenance["наказ"] == TokenProvenance("нaказ", KIND_HOMOGLYPH)
    assert "мяч" not in provenance


def test_surface_distance_reverses_the_lossy_transform_per_kind():
    typed = TokenProvenance("sut", KIND_TRANSLITERATE)
    assert surface_distance("суть", typed) == 0  # only the soft sign was lost
    assert surface_distance("суд", typed) > 0  # a different word, not a restoration
    homoglyph = TokenProvenance("нaказ", KIND_HOMOGLYPH)  # Latin "a" inside a Cyrillic token
    assert surface_distance("наказ", homoglyph) == 0
    assert surface_distance("накат", homoglyph) > 0


def test_transliteration_provenance_refuses_an_incompatible_nearest_neighbor():
    vocab = build_vocabulary(["суд ухвалив рішення"])
    # unconstrained, the OOV "сут" is one edit from "суд" and would silently become it
    assert apply_typos("сут", vocab)[0] == "суд"
    typed = {"сут": TokenProvenance("sut", KIND_TRANSLITERATE)}
    processed, edits = apply_typos("сут", vocab, provenance=typed)
    assert (processed, edits) == ("сут", [])


def test_transliteration_provenance_restores_the_form_the_user_could_have_typed():
    # both are one edit away; only "суть" romanizes back to the "sut" that was actually typed
    vocab = build_vocabulary(["суд розглянув суть справи"])
    typed = {"сут": TokenProvenance("sut", KIND_TRANSLITERATE)}
    processed, edits = apply_typos("сут", vocab, provenance=typed)
    assert processed == "суть"
    assert [(edit.original, edit.replacement) for edit in edits] == [("сут", "суть")]


def test_pipeline_threads_normalization_provenance_into_the_typos_step():
    vocab = build_vocabulary(["суд ухвалив рішення"])
    pipeline = QueryPrep.build(("normalize", "typos"), vocabulary=vocab)
    # "sut" normalizes to the OOV "сут"; the typo step may not turn that into the corpus's "суд"
    assert pipeline.process("sut").processed == "сут"
    # without the normalize step there is no provenance, so the nearest neighbor still applies
    assert QueryPrep.build(("typos",), vocabulary=vocab).process("сут").processed == "суд"


def test_short_token_refuses_an_unresolved_tie_instead_of_guessing():
    vocab = build_vocabulary(["кіт спить", "літ минуло"])
    # "хіт" is one edit from both and nothing separates them: refuse rather than pick alphabetically
    assert apply_typos("хіт", vocab) == ("хіт", [])


def test_short_token_refuses_a_length_changing_candidate():
    # "кв" is one deletion from the noisy "якв", but at three characters that is a different short
    # word, not a repair; the same deletion on a long token stays a plausible dropped letter.
    vocab = build_vocabulary(["кв м", "розташовані будівлі"])
    assert apply_typos("якв", vocab) == ("якв", [])
    assert apply_typos("розьташовані", vocab)[0] == "розташовані"


def test_transliteration_provenance_licenses_a_short_length_change():
    # romanization is what dropped the soft sign, so restoring it is not a resize of a short token
    vocab = build_vocabulary(["князь підписав"])
    typed = {"княз": TokenProvenance("knyaz", KIND_TRANSLITERATE)}
    assert apply_typos("княз", vocab, provenance=typed)[0] == "князь"


def test_long_token_tie_still_corrects_deterministically():
    vocab = build_vocabulary(["постанову ухвалено", "постанови ухвалено"])
    processed, edits = apply_typos("постановю", vocab)  # 9 chars: above the ambiguity cutoff
    assert processed == "постанови"
    assert len(edits) == 1


def test_local_context_prefers_the_candidate_the_query_co_occurs_with():
    chunks = ["наказ про відпустку працівника", "накат хвилі на морському березі"]
    context = VocabularyContext.build(chunks)
    vocab = context.tokens
    query = "накас хвилі на березі"
    # alphabetically "наказ" wins, and without the context index that is what the step picks
    assert apply_typos(query, vocab)[0].startswith("наказ")
    assert apply_typos(query, vocab, context=context)[0].startswith("накат")


def test_morphology_prefers_the_candidate_that_keeps_the_typed_ending():
    # Same distance and same known-word status: the candidate that rewrites the typed inflection
    # loses to the one that only repairs the stem, even though it sorts first.
    assert select_restoration("заквнами", [(1, "заквнамі"), (1, "законами")]) == "законами"


def test_morphology_prefers_a_known_word_form_over_a_corpus_surface():
    known = {"наказу"}.__contains__
    assert select_restoration("наказі", [(1, "наказа"), (1, "наказу")]) == "наказа"
    assert (
        select_restoration("наказі", [(1, "наказа"), (1, "наказу")], known_word=known) == "наказу"
    )


def test_query_context_index_requires_the_typos_step():
    with pytest.raises(ValueError, match="query-context index"):
        QueryPrep.build(("normalize",), context=VocabularyContext.build(["наказ"]))
