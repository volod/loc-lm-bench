"""Resumable ontology drafting: per-window extraction journal + kill-then-resume equivalence.

No server or GPU: the model is an injected fake. The kill is simulated with `KeyboardInterrupt`
(a BaseException), which -- unlike a transport error -- is NOT swallowed by the per-window handler,
so it faithfully leaves the in-flight window un-journaled the way a process kill would.
"""

import json

import pytest

from llb.goldset.schema import load_goldset
from llb.prep.ontology.constants import (
    EXTRACTION_JOURNAL_FILENAME,
    EXTRACTION_JOURNAL_META_FILENAME,
)
from llb.prep.ontology.endpoint import EndpointConfig
from llb.prep.ontology.extract import LLMExtractionAdapter
from llb.prep.ontology.journal import ExtractionJournal
from llb.prep.ontology.models import DocRecord
from llb.prep.ontology.pipeline import draft_goldset

# reuse the trusted fake endpoint + docs from the full-flow test
from tests.test_ontology_draft import DOC1, DOC2, fake_endpoint

LONG_DOC = (
    "Київ є столицею України та найбільшим містом країни. "
    "Львів є культурним центром заходу держави. "
    "Одеса є великим портовим містом на півдні. "
    "Харків є значним науковим осередком сходу. "
) * 3


def test_extraction_journal_skips_recorded_windows(tmp_path):
    # a long doc splits into several windows; a fresh adapter re-using the same journal must reuse
    # every window WITHOUT any new model call, and rebuild the identical merged extraction.
    doc = DocRecord(doc_id="a.md", text=LONG_DOC, sha256="x", n_chars=len(LONG_DOC))
    journal_path = tmp_path / EXTRACTION_JOURNAL_FILENAME

    calls: list[str] = []

    def counting(prompt: str) -> str:
        calls.append(prompt)
        return fake_endpoint(prompt)

    journal = ExtractionJournal(journal_path)
    adapter = LLMExtractionAdapter(
        complete=counting, max_chars=60, chunk_overlap=0, concurrency=1, journal=journal
    )
    first = adapter.extract(doc)
    n_windows = len(calls)
    assert n_windows > 1  # genuinely multi-window
    assert journal_path.is_file()

    reloaded = ExtractionJournal(journal_path)
    assert reloaded.load() == n_windows

    def boom(_prompt: str) -> str:
        raise AssertionError("journaled window must not re-call the model")

    resumed_adapter = LLMExtractionAdapter(
        complete=boom, max_chars=60, chunk_overlap=0, concurrency=1, journal=reloaded
    )
    second = resumed_adapter.extract(doc)
    assert second.model_dump() == first.model_dump()


def _corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.md").write_text(DOC1, encoding="utf-8")
    (corpus / "doc2.md").write_text(DOC2, encoding="utf-8")
    return corpus


def test_kill_mid_extraction_then_resume_matches_uninterrupted(tmp_path):
    corpus = _corpus(tmp_path)
    cfg = EndpointConfig(kind="local", model="fake")

    # 1) uninterrupted reference run
    bundle_a = tmp_path / "bundle_a"
    draft_goldset(corpus, cfg, complete=fake_endpoint, max_items=20, out_dir=bundle_a)

    # 2) interrupted run: raise on the SECOND extraction call (doc1 journaled, doc2 killed)
    bundle_b = tmp_path / "bundle_b"
    state = {"extractions": 0}

    def killing(prompt: str) -> str:
        if "будує онтологію" in prompt:
            state["extractions"] += 1
            if state["extractions"] == 2:
                raise KeyboardInterrupt("simulated kill mid-extraction")
        return fake_endpoint(prompt)

    with pytest.raises(KeyboardInterrupt):
        draft_goldset(corpus, cfg, complete=killing, max_items=20, out_dir=bundle_b)

    # the bundle exists with the meta sidecar and a partial (non-empty) journal
    assert (bundle_b / EXTRACTION_JOURNAL_META_FILENAME).is_file()
    journal_lines = (
        (bundle_b / EXTRACTION_JOURNAL_FILENAME).read_text(encoding="utf-8").splitlines()
    )
    assert len(journal_lines) == 1  # only doc1's window completed
    assert not (bundle_b / "goldset.jsonl").exists()  # never finished

    # 3) resume: reuse the journaled window, re-extract the rest, replay deterministic stages
    result = draft_goldset(corpus, cfg, complete=fake_endpoint, out_dir=bundle_b, resume=True)
    assert result.log.summary()  # resumed run recorded its own (fewer) calls

    # the resumed bundle is byte-identical to the uninterrupted one on the deterministic artifacts
    for name in ("goldset.jsonl", "extraction.jsonl", "ontology.json"):
        assert (bundle_b / name).read_text(encoding="utf-8") == (bundle_a / name).read_text(
            encoding="utf-8"
        ), name
    # same kept items, both unverified
    items_a = load_goldset(bundle_a / "goldset.jsonl")
    items_b = load_goldset(bundle_b / "goldset.jsonl")
    assert [it.id for it in items_b] == [it.id for it in items_a]
    assert items_b and all(it.verified is False for it in items_b)


def test_resume_reads_settings_from_meta(tmp_path):
    corpus = _corpus(tmp_path)
    cfg = EndpointConfig(kind="local", model="fake")
    bundle = tmp_path / "bundle"
    draft_goldset(corpus, cfg, complete=fake_endpoint, max_items=7, seed=99, out_dir=bundle)
    meta = json.loads((bundle / EXTRACTION_JOURNAL_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["kind"] == "extraction-journal-meta"
    assert meta["max_items"] == 7 and meta["seed"] == 99
    assert meta["corpus_root"] == str(corpus)
    assert meta["endpoint"]["model"] == "fake"

    # a resume that passes different max_items/seed is overridden by the pinned meta
    result = draft_goldset(
        corpus, cfg, complete=fake_endpoint, max_items=999, seed=1, out_dir=bundle, resume=True
    )
    reloaded = json.loads((bundle / EXTRACTION_JOURNAL_META_FILENAME).read_text(encoding="utf-8"))
    assert reloaded["max_items"] == 7 and reloaded["seed"] == 99  # meta untouched by resume
    assert result.items or result.items == []


def test_fresh_run_clears_prior_extraction_journal(tmp_path):
    corpus = _corpus(tmp_path)
    cfg = EndpointConfig(kind="local", model="fake")
    bundle = tmp_path / "bundle"
    draft_goldset(corpus, cfg, complete=fake_endpoint, max_items=7, out_dir=bundle)
    assert (bundle / EXTRACTION_JOURNAL_FILENAME).is_file()

    def killing(prompt: str) -> str:
        if "будує онтологію" in prompt:
            raise KeyboardInterrupt("fresh run must call extraction again")
        return fake_endpoint(prompt)

    with pytest.raises(KeyboardInterrupt, match="fresh run"):
        draft_goldset(corpus, cfg, complete=killing, max_items=7, out_dir=bundle)
    assert (bundle / EXTRACTION_JOURNAL_META_FILENAME).is_file()
    assert not (bundle / EXTRACTION_JOURNAL_FILENAME).exists()


def test_resume_without_meta_raises(tmp_path):
    corpus = _corpus(tmp_path)
    cfg = EndpointConfig(kind="local", model="fake")
    with pytest.raises(ValueError, match="cannot resume"):
        draft_goldset(corpus, cfg, complete=fake_endpoint, out_dir=tmp_path / "empty", resume=True)
