"""Focused tests split from ``test_query_prep_pipeline.py``."""

import json

import pytest
from tests.llb.rag._query_prep_helpers import glossary as build_glossary

from llb.rag.query_prep.base import (
    STEP_DECOMPOSE,
    STEP_GLOSSARY,
    STEP_HYDE,
    STEP_NORMALIZE,
    STEP_REWRITE,
    STEP_TYPOS,
)
from llb.rag.query_prep.decompose import parse_subqueries
from llb.rag.query_prep.glossary import (
    Glossary,
    apply_glossary,
    build_glossary_from_candidates,
)
from llb.rag.query_prep.pipeline import QueryPrep
from llb.rag.query_prep.retrieval import retrieve_prepared
from llb.rag.query_prep.rewrite import apply_rewrite
from llb.rag.query_prep.typos import build_vocabulary


def test_glossary_expands_matched_alias_deterministically():
    processed, edits = apply_glossary("що таке ІВ", build_glossary())
    # the raw query is preserved; canonical + other aliases are appended
    assert processed.startswith("що таке ІВ")
    assert "інтелектуальна власність" in processed
    assert "intelektualna vlasnist" in processed
    # deterministic: same input -> same output
    assert apply_glossary("що таке ІВ", build_glossary())[0] == processed
    assert [e.replacement for e in edits]


def test_glossary_no_match_is_noop():
    processed, edits = apply_glossary("погода у Києві", build_glossary())
    assert processed == "погода у Києві"
    assert edits == []


def test_glossary_matches_multiword_canonical_as_phrase():
    processed, _ = apply_glossary("що охороняє авторське право у творах", build_glossary())
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
    glossary = build_glossary()
    path = tmp_path / "query_glossary.json"
    path.write_text(json.dumps(glossary.to_dict(source_bundle="b")), encoding="utf-8")
    loaded = Glossary.load(path)
    assert [e.canonical for e in loaded.entries] == [e.canonical for e in glossary.entries]


def test_rewrite_records_both_forms():
    processed, edits, rewrite = apply_rewrite("q", lambda q: "розширений запит")
    assert processed == "розширений запит"
    assert rewrite == "розширений запит"
    assert edits and edits[0].kind == "rewrite"


def test_rewrite_blank_is_noop():
    processed, edits, rewrite = apply_rewrite("q", lambda q: "  ")
    assert processed == "q"
    assert edits == []


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
    with pytest.raises(ValueError, match="hypothetical-answer"):
        QueryPrep.build([STEP_HYDE])
    with pytest.raises(ValueError, match="decomposition endpoint"):
        QueryPrep.build([STEP_DECOMPOSE])


def test_model_steps_record_generated_text_and_subqueries():
    pipeline = QueryPrep.build(
        [STEP_HYDE, STEP_DECOMPOSE],
        hypothesizer=lambda _q: "hypothetical passage",
        decomposer=lambda _q: '{"subqueries":["first", "second"]}',
    )
    result = pipeline.process("compound question")
    assert result.processed == "compound question"
    assert result.hypothetical_answer == "hypothetical passage"
    assert result.subqueries == ("first", "second")
    assert result.provenance()["query_corrections"] == 0
    assert str(result.provenance()["query_decomposition"]).startswith("{")


def test_decomposition_parser_accepts_fences_lines_and_bounds_output():
    assert parse_subqueries('```json\n{"subqueries":["a","b","a"]}\n```') == ("a", "b")
    assert parse_subqueries("1. a\n- b\n* c", limit=2) == ("a", "b")


def test_prepared_retrieval_splits_hyde_dense_and_raw_lexical_queries():
    class Store:
        def __init__(self):
            self.calls = []

        def retrieve_queries(self, dense, lexical, k, chunk_filter=None):
            self.calls.append((dense, lexical, k, chunk_filter))
            return [{"doc_id": "d", "char_start": 0, "char_end": 1, "text": "x"}]

    store = Store()
    result = QueryPrep.build([STEP_HYDE], hypothesizer=lambda _q: "hypothetical passage").process(
        "raw question"
    )
    retrieve_prepared(store, result, 3)
    assert store.calls == [("hypothetical passage", "raw question", 3, None)]


def test_decomposition_retrieves_each_subquery_and_rrf_deduplicates_spans():
    shared = {"doc_id": "d", "char_start": 0, "char_end": 1, "text": "shared"}

    class Store:
        def retrieve_queries(self, dense, lexical, k, chunk_filter=None):
            unique = {"doc_id": dense, "char_start": 1, "char_end": 2, "text": dense}
            return [shared, unique]

    result = QueryPrep.build(
        [STEP_DECOMPOSE], decomposer=lambda _q: '["part-a", "part-b"]'
    ).process("compound")
    hits = retrieve_prepared(Store(), result, 4)
    assert [(hit["doc_id"], hit["char_start"]) for hit in hits] == [
        ("d", 0),
        ("compound", 1),
        ("part-a", 1),
        ("part-b", 1),
    ]
