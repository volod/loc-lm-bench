"""Query-side processing lane (uk-query-processing): pure pipeline, glossary, A/B, and wiring."""

import json

import pytest

from llb.eval import graph
from llb.rag import query_prep as qp


# --------------------------------------------------------------------------------------------
# normalize: casefold + apostrophe unification + transliteration
# --------------------------------------------------------------------------------------------


def test_normalize_unifies_apostrophes_and_casefolds():
    processed, edits = qp.apply_normalize("М'ЯЧ")
    assert processed == "м'яч"  # U+2019 apostrophe unified to ASCII, casefolded
    assert edits == []  # silent normalization, no transliteration edit


def test_normalize_transliterates_latin_typed_tokens():
    processed, edits = qp.apply_normalize("zakon про pravo")
    assert processed == "закон про право"
    kinds = {(e.original, e.replacement) for e in edits}
    assert ("zakon", "закон") in kinds and ("pravo", "право") in kinds


def test_normalize_leaves_cyrillic_tokens_untouched():
    processed, edits = qp.apply_normalize("рішення суду")
    assert processed == "рішення суду"
    assert edits == []


@pytest.mark.parametrize("word", ["закон", "право", "щит", "якіст", "рішення", "суд"])
def test_transliteration_table_round_trips(word):
    romanized = qp.cyrillic_to_latin(word)
    assert qp.transliterate_latin_to_cyrillic(romanized) == word


def test_romanization_drops_soft_sign():
    assert "ь" not in qp.cyrillic_to_latin("власність")


# --------------------------------------------------------------------------------------------
# typos: Damerau-Levenshtein correction that never touches in-vocabulary tokens
# --------------------------------------------------------------------------------------------


def test_damerau_levenshtein_counts_transposition_as_one():
    assert qp.damerau_levenshtein("наказ", "накза", 2) == 1  # adjacent transposition
    assert qp.damerau_levenshtein("abcd", "abdc", 2) == 1
    assert qp.damerau_levenshtein("наказ", "приказ", 2) == 3  # bounded -> max+1


def test_typos_correct_out_of_vocabulary_token():
    vocab = qp.build_vocabulary(["наказ видано начальником служби"])
    processed, edits = qp.apply_typos("виданоо начальнком", vocab)
    assert processed == "видано начальником"
    assert {(e.original, e.replacement) for e in edits} == {
        ("виданоо", "видано"),
        ("начальнком", "начальником"),
    }


def test_typos_never_alter_in_vocabulary_token():
    vocab = qp.build_vocabulary(["наказ видано начальником", "накат хвилі"])
    # "наказ" IS in the corpus; even though "накат" is one edit away, it must stay unchanged.
    processed, edits = qp.apply_typos("наказ", vocab)
    assert processed == "наказ"
    assert edits == []


def test_typos_leave_numeric_codes_untouched():
    vocab = qp.build_vocabulary(["наказ 4821 від 2024"])
    processed, edits = qp.apply_typos("4822", vocab)  # a code one edit from 4821
    assert processed == "4822"
    assert edits == []


def test_typos_long_token_allows_distance_two():
    vocab = qp.build_vocabulary(["інтелектуальної власності"])
    processed, _ = qp.apply_typos("інтелектуальнох", vocab)  # 12 chars, 2 edits away
    assert processed == "інтелектуальної"


def test_nearest_vocab_token_is_deterministic_under_ties():
    # "хіт" is one edit from BOTH; the lexicographically smaller candidate wins deterministically
    vocab = frozenset({"кіт", "літ"})
    assert qp.nearest_vocab_token("хіт", vocab, 1) == "кіт"


# --------------------------------------------------------------------------------------------
# glossary: deterministic alias expansion + builder
# --------------------------------------------------------------------------------------------


def _glossary():
    return qp.Glossary(
        (
            qp.GlossaryEntry("інтелектуальна власність", ("ІВ", "intelektualna vlasnist")),
            qp.GlossaryEntry("авторське право", ()),
        )
    )


def test_glossary_expands_matched_alias_deterministically():
    processed, edits = qp.apply_glossary("що таке ІВ", _glossary())
    # the raw query is preserved; canonical + other aliases are appended
    assert processed.startswith("що таке ІВ")
    assert "інтелектуальна власність" in processed
    assert "intelektualna vlasnist" in processed
    # deterministic: same input -> same output
    assert qp.apply_glossary("що таке ІВ", _glossary())[0] == processed
    assert [e.replacement for e in edits]


def test_glossary_no_match_is_noop():
    processed, edits = qp.apply_glossary("погода у Києві", _glossary())
    assert processed == "погода у Києві"
    assert edits == []


def test_glossary_matches_multiword_canonical_as_phrase():
    processed, _ = qp.apply_glossary("що охороняє авторське право у творах", _glossary())
    # already present canonical is matched but there are no other forms to add -> unchanged
    assert processed == "що охороняє авторське право у творах"


def test_build_glossary_from_candidates_seeds_transliteration_and_sorts():
    rows = [
        {"term": "патент", "aliases": ["patent"]},
        {"term": "авторське право", "aliases": []},
    ]
    glossary = qp.build_glossary_from_candidates(rows)
    canonicals = [e.canonical for e in glossary.entries]
    assert canonicals == ["авторське право", "патент"]  # sorted by canonical
    patent = next(e for e in glossary.entries if e.canonical == "патент")
    assert "patent" in patent.aliases  # recorded alias kept, not duplicated by romanization


def test_build_glossary_without_transliterations():
    glossary = qp.build_glossary_from_candidates(
        [{"term": "патент", "aliases": []}], add_transliterations=False
    )
    assert glossary.entries[0].aliases == ()


def test_glossary_json_round_trip(tmp_path):
    glossary = _glossary()
    path = tmp_path / "query_glossary.json"
    path.write_text(json.dumps(glossary.to_dict(source_bundle="b")), encoding="utf-8")
    loaded = qp.Glossary.load(path)
    assert [e.canonical for e in loaded.entries] == [e.canonical for e in glossary.entries]


# --------------------------------------------------------------------------------------------
# rewrite: off by default, injected callable
# --------------------------------------------------------------------------------------------


def test_rewrite_records_both_forms():
    processed, edits, rewrite = qp.apply_rewrite("q", lambda q: "розширений запит")
    assert processed == "розширений запит"
    assert rewrite == "розширений запит"
    assert edits and edits[0].kind == "rewrite"


def test_rewrite_blank_is_noop():
    processed, edits, rewrite = qp.apply_rewrite("q", lambda q: "  ")
    assert processed == "q"
    assert edits == []


# --------------------------------------------------------------------------------------------
# pipeline: ordering, exact no-op, dependency validation
# --------------------------------------------------------------------------------------------


def test_empty_pipeline_is_exact_noop():
    result = qp.QueryPrep.build([]).process("Незмінне Питання?")
    assert result.processed == "Незмінне Питання?"
    assert result.changed is False
    assert result.edits == ()


def test_pipeline_applies_steps_in_order():
    vocab = qp.build_vocabulary(["закон україни"])
    pipeline = qp.QueryPrep.build([qp.STEP_NORMALIZE, qp.STEP_TYPOS], vocabulary=vocab)
    result = qp.QueryPrep.process(pipeline, "Zakon")  # normalize -> закон, already in vocab
    assert result.processed == "закон"
    assert result.steps == (qp.STEP_NORMALIZE, qp.STEP_TYPOS)


def test_pipeline_rejects_unknown_step():
    with pytest.raises(ValueError, match="unknown query-prep step"):
        qp.QueryPrep.build(["nope"])


def test_pipeline_rejects_duplicate_step():
    with pytest.raises(ValueError, match="duplicate"):
        qp.QueryPrep.build([qp.STEP_NORMALIZE, qp.STEP_NORMALIZE])


def test_pipeline_requires_dependencies():
    with pytest.raises(ValueError, match="vocabulary"):
        qp.QueryPrep.build([qp.STEP_TYPOS])
    with pytest.raises(ValueError, match="glossary"):
        qp.QueryPrep.build([qp.STEP_GLOSSARY])
    with pytest.raises(ValueError, match="rewrite endpoint"):
        qp.QueryPrep.build([qp.STEP_REWRITE])


# --------------------------------------------------------------------------------------------
# A/B report over a fake retriever
# --------------------------------------------------------------------------------------------


def test_ab_report_attributes_per_step_delta():
    # the fake store only "finds" the gold span when the query is transliterated to Cyrillic
    def retrieve(query, k):
        return [{"doc_id": "d", "char_start": 0, "char_end": 5}] if "закон" in query else []

    items = [("zakon", [{"doc_id": "d", "char_start": 0, "char_end": 5}])]
    stages = qp.cumulative_pipelines([qp.STEP_NORMALIZE])
    report = qp.query_prep_ab_report(items, retrieve, 5, stages)
    assert [row["stage"] for row in report["stages"]] == ["baseline", "+normalize"]
    assert report["stages"][0]["recall_at_k"] == 0.0
    assert report["stages"][1]["recall_at_k"] == 1.0
    assert report["stages"][1]["delta_recall"] == pytest.approx(1.0)
    assert "query-prep A/B" in qp.format_query_prep_ab(report)


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
    pipeline = qp.QueryPrep.build([qp.STEP_NORMALIZE])
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
    from llb.executor.runner import build_query_prep

    assert build_query_prep(RunConfig(), _RecordingStore([]), None) is None


def test_build_query_prep_reads_vocabulary_from_store_chunks():
    from llb.core.config import RunConfig
    from llb.executor.runner import build_query_prep

    store = _RecordingStore(
        [{"doc_id": "a", "text": "видано наказ", "char_start": 0, "char_end": 1}]
    )
    cfg = RunConfig().with_overrides(query_prep=["typos"])
    pipeline = build_query_prep(cfg, store, None)
    assert pipeline.process("виданоо").processed == "видано"  # corrected against store vocab


def test_build_query_prep_glossary_needs_path():
    from llb.core.config import RunConfig
    from llb.executor.runner import build_query_prep

    cfg = RunConfig().with_overrides(query_prep=["glossary"])
    with pytest.raises(SystemExit, match="query_glossary_path"):
        build_query_prep(cfg, _RecordingStore([]), None)


def test_build_query_prep_rewrite_needs_launcher():
    from llb.core.config import RunConfig
    from llb.executor.runner import build_query_prep

    cfg = RunConfig().with_overrides(query_prep=["rewrite"])
    with pytest.raises(SystemExit, match="backend launcher"):
        build_query_prep(cfg, _RecordingStore([]), None)
