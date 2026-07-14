"""Tests for query prep pipeline."""

import json
import pytest
from llb.eval import graph
from llb.rag.query_prep.base import STEP_GLOSSARY, STEP_NORMALIZE, STEP_REWRITE, STEP_TYPOS
from llb.rag.query_prep.glossary import (
    Glossary,
    apply_glossary,
    build_glossary_from_candidates,
)
from llb.rag.query_prep.pipeline import QueryPrep
from llb.rag.query_prep.rewrite import apply_rewrite
from llb.rag.query_prep.report import (
    cumulative_pipelines,
    format_query_prep_ab,
    query_prep_ab_report,
)
from llb.rag.query_prep.typos import (
    build_vocabulary,
)
from test_query_prep import _RecordingStore, _glossary


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


def test_build_query_prep_returns_none_when_off():
    from llb.core.config import RunConfig
    from llb.executor.runner_retrieval import build_query_prep

    assert build_query_prep(RunConfig(), _RecordingStore([]), None) is None


def test_build_query_prep_reads_vocabulary_from_store_chunks():
    from llb.core.config import RunConfig
    from llb.executor.runner_retrieval import build_query_prep

    store = _RecordingStore(
        [{"doc_id": "a", "text": "видано наказ", "char_start": 0, "char_end": 1}]
    )
    cfg = RunConfig().with_overrides(query_prep=["typos"])
    pipeline = build_query_prep(cfg, store, None)
    assert pipeline.process("виданоо").processed == "видано"  # corrected against store vocab


def test_build_query_prep_glossary_needs_path():
    from llb.core.config import RunConfig
    from llb.executor.runner_retrieval import build_query_prep

    cfg = RunConfig().with_overrides(query_prep=["glossary"])
    with pytest.raises(SystemExit, match="query_glossary_path"):
        build_query_prep(cfg, _RecordingStore([]), None)


def test_build_query_prep_rewrite_needs_launcher():
    from llb.core.config import RunConfig
    from llb.executor.runner_retrieval import build_query_prep

    cfg = RunConfig().with_overrides(query_prep=["rewrite"])
    with pytest.raises(SystemExit, match="backend launcher"):
        build_query_prep(cfg, _RecordingStore([]), None)
