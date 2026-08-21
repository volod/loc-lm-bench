"""Focused tests split from ``test_query_prep_pipeline.py``."""

import pytest
from tests.llb.rag._query_prep_pipeline_helpers import (
    _UK_PLAUSIBLE,
)
from tests.llb.rag._query_prep_helpers import RecordingStore

from llb.eval import graph
from llb.rag.query_prep.base import STEP_TYPOS
from llb.rag.query_prep.pipeline import QueryPrep


def test_language_gate_requires_normalize_step():
    with pytest.raises(ValueError, match="normalize language gate"):
        QueryPrep.build([STEP_TYPOS], vocabulary=frozenset({"x"}), plausible=_UK_PLAUSIBLE)


def test_retrieve_node_without_query_prep_records_nothing():
    store = RecordingStore([{"doc_id": "a", "text": "x", "char_start": 0, "char_end": 1}])
    node = graph.make_retrieve_node(store, k=5)
    update = node({"question": "Zakon"})
    assert store.seen == ["Zakon"]  # untouched
    assert "query_processed" not in update


def test_build_query_prep_returns_none_when_off():
    from llb.core.config import RunConfig
    from llb.executor.runner_retrieval import build_query_prep

    assert build_query_prep(RunConfig(), RecordingStore([]), None) is None


def test_build_query_prep_reads_vocabulary_from_store_chunks():
    from llb.core.config import RunConfig
    from llb.executor.runner_retrieval import build_query_prep

    store = RecordingStore(
        [{"doc_id": "a", "text": "видано наказ", "char_start": 0, "char_end": 1}]
    )
    cfg = RunConfig().with_overrides(query_prep=["typos"])
    pipeline = build_query_prep(cfg, store, None)
    assert pipeline.process("виданоо").processed == "видано"  # corrected against store vocab


def test_build_query_prep_wires_language_gate_when_flag_on(monkeypatch):
    from llb.core.config import RunConfig
    from llb.executor.runner_retrieval import build_query_prep

    # Fake morphology probe: only genuine Ukrainian forms are "known".
    monkeypatch.setattr(
        "llb.rag.vector_store.lexical.load_uk_word_probe", lambda: {"закон", "право"}.__contains__
    )
    store = RecordingStore([{"doc_id": "a", "text": "закон право", "char_start": 0, "char_end": 1}])
    cfg = RunConfig().with_overrides(query_prep=["normalize"], query_prep_language_gate=True)
    pipeline = build_query_prep(cfg, store, None)
    assert pipeline.plausible is not None
    assert pipeline.process("zakon pravo").processed == "закон право"  # romanized UA transliterated
    assert pipeline.process("what does the").processed == "what does the"  # foreign left untouched


def test_build_query_prep_language_gate_off_by_default(monkeypatch):
    from llb.core.config import RunConfig
    from llb.executor.runner_retrieval import build_query_prep

    loaded = []
    monkeypatch.setattr(
        "llb.rag.vector_store.lexical.load_uk_word_probe",
        lambda: loaded.append(True) or (lambda _t: False),
    )
    cfg = RunConfig().with_overrides(query_prep=["normalize"])
    pipeline = build_query_prep(cfg, RecordingStore([]), None)
    assert pipeline.plausible is None and loaded == []  # gate inert, no probe loaded
    assert pipeline.process("what does the").processed != "what does the"  # pre-gate behavior


def test_config_language_gate_needs_normalize_step():
    from llb.core.config import RunConfig

    with pytest.raises(ValueError, match="query_prep_language_gate"):
        RunConfig().with_overrides(query_prep=["typos"], query_prep_language_gate=True)


def test_build_query_prep_dense_case_reaches_the_pipeline():
    from llb.core.config import RunConfig
    from llb.executor.runner_retrieval import build_query_prep

    store = RecordingStore([{"doc_id": "a", "text": "кобзар", "char_start": 0, "char_end": 1}])
    cfg = RunConfig().with_overrides(query_prep=["normalize"], query_prep_dense_case=True)
    pipeline = build_query_prep(cfg, store, None)
    assert pipeline.dense_case is True
    assert pipeline.process("Хто написав Кобзар?").dense_query == "Хто написав Кобзар?"


def test_config_dense_case_needs_normalize_step():
    from llb.core.config import RunConfig

    with pytest.raises(ValueError, match="query_prep_dense_case"):
        RunConfig().with_overrides(query_prep=["typos"], query_prep_dense_case=True)


def test_build_query_prep_glossary_needs_path():
    from llb.core.config import RunConfig
    from llb.executor.runner_retrieval import build_query_prep

    cfg = RunConfig().with_overrides(query_prep=["glossary"])
    with pytest.raises(SystemExit, match="query_glossary_path"):
        build_query_prep(cfg, RecordingStore([]), None)


def test_build_query_prep_rewrite_needs_launcher():
    from llb.core.config import RunConfig
    from llb.executor.runner_retrieval import build_query_prep

    cfg = RunConfig().with_overrides(query_prep=["rewrite"])
    with pytest.raises(SystemExit, match="backend launcher"):
        build_query_prep(cfg, RecordingStore([]), None)
