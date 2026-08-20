"""Dense-lane casing: the lexical lane keeps the fold, the dense lane gets the user's case."""

import pytest

from llb.rag.query_prep.base import STEP_DECOMPOSE, STEP_GLOSSARY, STEP_NORMALIZE, STEP_TYPOS
from llb.rag.query_prep.casing import apply_case_pattern, restore_query_case
from llb.rag.query_prep.pipeline import QueryPrep
from llb.rag.query_prep.retrieval import retrieve_prepared


class RecordingStore:
    """Fake store that records the (dense, lexical) text pair of every retrieval call."""

    def __init__(self):
        self.calls = []

    def retrieve_queries(self, dense, lexical, k, chunk_filter=None):
        self.calls.append((dense, lexical))
        return [{"doc_id": dense, "char_start": 0, "char_end": 1, "text": dense}]


def test_case_pattern_transfers_capitalization_without_replacing_characters():
    assert apply_case_pattern("Кобзар", "кобзар") == "Кобзар"
    assert apply_case_pattern("NP", "np") == "NP"
    assert apply_case_pattern("кобзар", "кобзар") == "кобзар"
    # The transliterated replacement keeps its own characters; only the case comes from the source.
    assert apply_case_pattern("Kyiv", "київ") == "Київ"
    # A mixed-case source of a different length falls back to the leading capital.
    assert apply_case_pattern("McDonald", "макдоналдс") == "Макдоналдс"


def test_case_pattern_never_overwrites_case_a_step_produced():
    # Only the fold's own lowercase output is restorable; a model rewrite's casing is its own.
    assert apply_case_pattern("КОБЗАР", "Kobzar") == "Kobzar"


def test_restore_case_leaves_unaligned_and_already_lower_tokens_alone():
    # An appended glossary expansion has no raw counterpart and keeps the case the step produced.
    assert restore_query_case("Що таке ІВ", "що таке ів інтелектуальна власність") == (
        "Що таке ІВ інтелектуальна власність"
    )
    assert restore_query_case("що таке ів", "що таке ів") == "що таке ів"
    assert restore_query_case("", "що таке ів") == "що таке ів"


def test_restore_case_preserves_processed_characters_and_punctuation():
    # Apostrophe unification and transliteration survive; only capitalization is transferred.
    restored = restore_query_case("Хто написав «Кобзар»?", "хто написав «кобзар»?")
    assert restored == "Хто написав «Кобзар»?"
    assert restore_query_case("Пам'ять", "пам'ять") == "Пам'ять"


def test_dense_case_lane_splits_dense_and_lexical_text():
    prep = QueryPrep.build([STEP_NORMALIZE], dense_case=True)
    result = prep.process("Хто написав Кобзар?")

    assert result.processed == "хто написав кобзар?"
    assert result.dense_query == "Хто написав Кобзар?"
    assert result.provenance()["query_dense"] == "Хто написав Кобзар?"

    store = RecordingStore()
    retrieve_prepared(store, result, 3)
    assert store.calls == [("Хто написав Кобзар?", "хто написав кобзар?")]


def test_dense_case_off_keeps_one_query_on_both_lanes():
    result = QueryPrep.build([STEP_NORMALIZE]).process("Хто написав Кобзар?")

    assert result.dense_processed is None
    assert result.dense_query == result.processed
    assert "query_dense" not in result.provenance()

    store = RecordingStore()
    retrieve_prepared(store, result, 3)
    assert store.calls == [("хто написав кобзар?", "хто написав кобзар?")]


def test_dense_case_recovers_the_acronym_the_normalize_step_folds():
    result = QueryPrep.build([STEP_NORMALIZE], dense_case=True).process("What does the NP mean?")

    assert "np" in result.processed
    assert "NP" in result.dense_query


def test_dense_case_carries_capitalization_onto_a_corrected_token():
    vocabulary = frozenset({"кобзар", "написав", "хто"})
    result = QueryPrep.build(
        [STEP_NORMALIZE, STEP_TYPOS], vocabulary=vocabulary, dense_case=True
    ).process("Хто написав Кобзаp?")

    assert result.processed == "хто написав кобзар?"
    assert result.dense_query == "Хто написав Кобзар?"


def test_dense_case_is_an_exact_noop_when_the_query_is_already_lowercase():
    result = QueryPrep.build([STEP_NORMALIZE], dense_case=True).process("хто написав кобзар?")

    assert result.dense_processed is None
    assert result.dense_query == result.processed


def test_dense_case_needs_the_normalize_step():
    with pytest.raises(ValueError, match="dense-case lane needs the 'normalize' step"):
        QueryPrep.build([STEP_TYPOS], vocabulary=frozenset({"кобзар"}), dense_case=True)


def test_dense_case_routes_only_the_original_query_lane_of_a_decomposition():
    prep = QueryPrep.build(
        [STEP_NORMALIZE, STEP_DECOMPOSE],
        decomposer=lambda _q: '["перша частина", "друга частина"]',
        dense_case=True,
    )
    result = prep.process("Хто написав Кобзар і коли?")

    store = RecordingStore()
    retrieve_prepared(store, result, 4)
    dense, lexical = store.calls[0]
    assert dense == "Хто написав Кобзар і коли?"
    assert lexical == result.processed
    # Subqueries are model output that never went through the fold, so both lanes get them as-is.
    assert store.calls[1:] == [("перша частина",) * 2, ("друга частина",) * 2]


def test_glossary_expansion_stays_folded_on_both_lanes():
    from tests.llb.rag._query_prep_helpers import glossary as build_glossary

    result = QueryPrep.build(
        [STEP_NORMALIZE, STEP_GLOSSARY], glossary=build_glossary(), dense_case=True
    ).process("Що таке ІВ")

    assert result.dense_query.startswith("Що таке ІВ")
    assert "інтелектуальна власність" in result.dense_query
