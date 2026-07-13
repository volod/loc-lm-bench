"""Query-side processing lane (uk-query-processing): pure pipeline, glossary, A/B, and wiring."""

import json

import pytest

from llb.eval import graph
from llb.rag.query_prep.base import STEP_GLOSSARY, STEP_NORMALIZE, STEP_REWRITE, STEP_TYPOS
from llb.rag.query_prep.glossary import (
    Glossary,
    GlossaryEntry,
    apply_glossary,
    build_glossary_from_candidates,
)
from llb.rag.query_prep.normalize import (
    apply_normalize,
    cyrillic_to_latin,
    transliterate_latin_to_cyrillic,
)
from llb.rag.query_prep.pipeline import QueryPrep
from llb.rag.query_prep.rewrite import apply_rewrite
from llb.rag.query_prep.report import (
    cumulative_pipelines,
    format_query_prep_ab,
    query_prep_ab_report,
)
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


def test_typo_guard_with_real_pymorphy_probe():
    pytest.importorskip("pymorphy3")
    from llb.rag.lexical import load_uk_word_probe

    known = load_uk_word_probe()
    vocab = build_vocabulary(["поділяти документа"])
    # both plan examples: grammatically valid inflections survive the guard
    assert apply_typos("поділяють документами", vocab, known_word=known)[0] == (
        "поділяють документами"
    )
    # the misspelling "поділяяти" is unknown to pymorphy3 and is still corrected
    assert apply_typos("поділяяти", vocab, known_word=known)[0] == "поділяти"


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


def test_glossary_expands_matched_alias_deterministically():
    processed, edits = apply_glossary("що таке ІВ", _glossary())
    # the raw query is preserved; canonical + other aliases are appended
    assert processed.startswith("що таке ІВ")
    assert "інтелектуальна власність" in processed
    assert "intelektualna vlasnist" in processed
    # deterministic: same input -> same output
    assert apply_glossary("що таке ІВ", _glossary())[0] == processed
    assert [e.replacement for e in edits]


def test_glossary_no_match_is_noop():
    processed, edits = apply_glossary("погода у Києві", _glossary())
    assert processed == "погода у Києві"
    assert edits == []


def test_glossary_matches_multiword_canonical_as_phrase():
    processed, _ = apply_glossary("що охороняє авторське право у творах", _glossary())
    # already present canonical is matched but there are no other forms to add -> unchanged
    assert processed == "що охороняє авторське право у творах"


def test_build_glossary_from_candidates_seeds_transliteration_and_sorts():
    rows = [
        {"term": "патент", "aliases": ["patent"]},
        {"term": "авторське право", "aliases": []},
    ]
    glossary = build_glossary_from_candidates(rows)
    canonicals = [e.canonical for e in glossary.entries]
    assert canonicals == ["авторське право", "патент"]  # sorted by canonical
    patent = next(e for e in glossary.entries if e.canonical == "патент")
    assert "patent" in patent.aliases  # recorded alias kept, not duplicated by romanization


def test_build_glossary_without_transliterations():
    glossary = build_glossary_from_candidates(
        [{"term": "патент", "aliases": []}], add_transliterations=False
    )
    assert glossary.entries[0].aliases == ()


def test_glossary_json_round_trip(tmp_path):
    glossary = _glossary()
    path = tmp_path / "query_glossary.json"
    path.write_text(json.dumps(glossary.to_dict(source_bundle="b")), encoding="utf-8")
    loaded = Glossary.load(path)
    assert [e.canonical for e in loaded.entries] == [e.canonical for e in glossary.entries]


# --------------------------------------------------------------------------------------------
# rewrite: off by default, injected callable
# --------------------------------------------------------------------------------------------


def test_rewrite_records_both_forms():
    processed, edits, rewrite = apply_rewrite("q", lambda q: "розширений запит")
    assert processed == "розширений запит"
    assert rewrite == "розширений запит"
    assert edits and edits[0].kind == "rewrite"


def test_rewrite_blank_is_noop():
    processed, edits, rewrite = apply_rewrite("q", lambda q: "  ")
    assert processed == "q"
    assert edits == []


# --------------------------------------------------------------------------------------------
# pipeline: ordering, exact no-op, dependency validation
# --------------------------------------------------------------------------------------------


def test_empty_pipeline_is_exact_noop():
    result = QueryPrep.build([]).process("Незмінне Питання?")
    assert result.processed == "Незмінне Питання?"
    assert result.changed is False
    assert result.edits == ()


def test_pipeline_applies_steps_in_order():
    vocab = build_vocabulary(["закон україни"])
    pipeline = QueryPrep.build([STEP_NORMALIZE, STEP_TYPOS], vocabulary=vocab)
    result = QueryPrep.process(pipeline, "Zakon")  # normalize -> закон, already in vocab
    assert result.processed == "закон"
    assert result.steps == (STEP_NORMALIZE, STEP_TYPOS)


def test_pipeline_rejects_unknown_step():
    with pytest.raises(ValueError, match="unknown query-prep step"):
        QueryPrep.build(["nope"])


def test_pipeline_rejects_duplicate_step():
    with pytest.raises(ValueError, match="duplicate"):
        QueryPrep.build([STEP_NORMALIZE, STEP_NORMALIZE])


def test_pipeline_requires_dependencies():
    with pytest.raises(ValueError, match="vocabulary"):
        QueryPrep.build([STEP_TYPOS])
    with pytest.raises(ValueError, match="glossary"):
        QueryPrep.build([STEP_GLOSSARY])
    with pytest.raises(ValueError, match="rewrite endpoint"):
        QueryPrep.build([STEP_REWRITE])


# --------------------------------------------------------------------------------------------
# A/B report over a fake retriever
# --------------------------------------------------------------------------------------------


def test_ab_report_attributes_per_step_delta():
    # the fake store only "finds" the gold span when the query is transliterated to Cyrillic
    def retrieve(query, k):
        return [{"doc_id": "d", "char_start": 0, "char_end": 5}] if "закон" in query else []

    items = [("zakon", [{"doc_id": "d", "char_start": 0, "char_end": 5}])]
    stages = cumulative_pipelines([STEP_NORMALIZE])
    report = query_prep_ab_report(items, retrieve, 5, stages)
    assert [row["stage"] for row in report["stages"]] == ["baseline", "+normalize"]
    assert report["stages"][0]["recall_at_k"] == 0.0
    assert report["stages"][1]["recall_at_k"] == 1.0
    assert report["stages"][1]["delta_recall"] == pytest.approx(1.0)
    assert "query-prep A/B" in format_query_prep_ab(report)


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


def test_retrieve_node_uses_processed_query_and_preserves_raw():
    chunks = [{"doc_id": "a", "text": "закон україни", "char_start": 0, "char_end": 13}]
    store = _RecordingStore(chunks)
    pipeline = QueryPrep.build([STEP_NORMALIZE])
    node = graph.make_retrieve_node(store, k=5, query_prep=pipeline)
    update = node({"question": "Zakon Ukrainy"})
    assert store.seen == ["закон украіни"]  # retrieval used the transliterated query
    assert update["query_processed"] == "закон украіни"
    assert update["query_corrections"] == 2  # two transliterations


def test_retrieve_node_without_query_prep_records_nothing():
    store = _RecordingStore([{"doc_id": "a", "text": "x", "char_start": 0, "char_end": 1}])
    node = graph.make_retrieve_node(store, k=5)
    update = node({"question": "Zakon"})
    assert store.seen == ["Zakon"]  # untouched
    assert "query_processed" not in update


# --------------------------------------------------------------------------------------------
# runner resolver: dependency wiring from RunConfig + store + launcher
# --------------------------------------------------------------------------------------------


def test_build_query_prep_returns_none_when_off():
    from llb.core.config import RunConfig
    from llb.executor.runner_setup import build_query_prep

    assert build_query_prep(RunConfig(), _RecordingStore([]), None) is None


def test_build_query_prep_reads_vocabulary_from_store_chunks():
    from llb.core.config import RunConfig
    from llb.executor.runner_setup import build_query_prep

    store = _RecordingStore(
        [{"doc_id": "a", "text": "видано наказ", "char_start": 0, "char_end": 1}]
    )
    cfg = RunConfig().with_overrides(query_prep=["typos"])
    pipeline = build_query_prep(cfg, store, None)
    assert pipeline.process("виданоо").processed == "видано"  # corrected against store vocab


def test_build_query_prep_glossary_needs_path():
    from llb.core.config import RunConfig
    from llb.executor.runner_setup import build_query_prep

    cfg = RunConfig().with_overrides(query_prep=["glossary"])
    with pytest.raises(SystemExit, match="query_glossary_path"):
        build_query_prep(cfg, _RecordingStore([]), None)


def test_build_query_prep_rewrite_needs_launcher():
    from llb.core.config import RunConfig
    from llb.executor.runner_setup import build_query_prep

    cfg = RunConfig().with_overrides(query_prep=["rewrite"])
    with pytest.raises(SystemExit, match="backend launcher"):
        build_query_prep(cfg, _RecordingStore([]), None)
