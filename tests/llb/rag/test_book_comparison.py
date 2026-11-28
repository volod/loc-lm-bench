"""Synthetic acceptance for the projection/PDF retrieval comparison."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.cli import app
from llb.goldset.schema import dump_goldset, load_goldset
from llb.prep.pdf.ingest import ingest_pdf_corpus
from llb.prep.pdf.model import PYMUPDF4LLM_TOOL
from llb.rag.book_comparison import run
from llb.rag.book_comparison.inputs import load_book_pair, load_book_questions
from llb.rag.book_comparison.metrics import Ranks, retrieve_ranks
from llb.rag.vector_store import factory
from llb.rag.vector_store.layout import META_FILE
from tests.llb.rag.book_fixture import (
    MARKER,
    MDE,
    N_ITEMS,
    PlantedEmbedder,
    evidence,
    source_span,
    write_inputs,
)

RUNNER = CliRunner()


@pytest.fixture
def book(tmp_path):
    pdf, projection, gold, config = write_inputs(tmp_path)
    baseline = tmp_path / "baseline"
    result = ingest_pdf_corpus(pdf, baseline, parser=PYMUPDF4LLM_TOOL, min_chars=1)
    assert result.n_docs == 1
    return projection, baseline, gold, config


def _invoke(book):
    projection, baseline, gold, config = book
    return RUNNER.invoke(
        app,
        [
            "compare-book-retrieval",
            "--projection",
            str(projection),
            "--baseline",
            str(baseline),
            "--goldset",
            str(gold),
            "--config",
            str(config),
            "--mde",
            str(MDE),
            "--k",
            "1",
            "--resamples",
            "256",
        ],
    )


@pytest.mark.heavy_env
def test_compare_book_replays_both_indexes_and_pins_verdicts(book, monkeypatch):
    monkeypatch.setattr(factory, "Embedder", PlantedEmbedder)
    result = _invoke(book)
    assert result.exit_code == 0, result.output
    report_path = Path(result.output.strip().rsplit(" -> ", 1)[-1])
    report = json.loads(report_path.read_text())
    settings = report["settings"]
    assert (settings["k"], settings["minimum_detectable_delta"]) == (1, 0.25)
    assert settings["minimum_discordant_pairs"] == 6
    assert report["baseline_parser"] == PYMUPDF4LLM_TOOL
    _assert_indexes(report_path.parent, report["indexes"])

    all_items, present, missing = (
        report["slices"][slice_name]
        for slice_name in ("all", "baseline_text_present", "baseline_text_missing")
    )
    assert [(row["n"], row["verdict"]) for row in (all_items, present, missing)] == [
        (12, "inconclusive"),
        (6, "regressed"),
        (6, "improved"),
    ]
    for metric in ("span_recall_at_k", "span_mrr"):
        _assert_delta(all_items, metric, 0.5, 0.5, 6, 6, 0)
        _assert_delta(present, metric, 0, 1, 0, 6, 0)
        _assert_delta(missing, metric, 1, 0, 6, 0, 0)
    for metric in ("page_recall_at_k", "page_mrr"):
        _assert_delta(all_items, metric, 0.5, 1, 0, 6, 6)
        _assert_delta(present, metric, 0, 1, 0, 6, 0)
        _assert_delta(missing, metric, 1, 1, 0, 0, 6)
    assert present["metrics"]["span_recall_at_k"]["sensitivity"]["target_reached"]
    assert not all_items["metrics"]["span_recall_at_k"]["sensitivity"]["target_reached"]
    assert report["items"][0]["baseline"] == {"span": None, "page": 1}
    assert report["items"][0]["baseline_span_matches"] == 0
    assert report["items"][0]["pdf_pages"] == [1]
    replay = _invoke(book)
    assert replay.exit_code == 0, replay.output
    replay_path = Path(replay.output.strip().rsplit(" -> ", 1)[-1])
    assert replay_path != report_path
    assert replay_path.read_bytes() == report_path.read_bytes()


def _assert_indexes(run_dir, indexes):
    left, right = indexes["projection"], indexes["baseline"]
    settings = (
        "embedding_model",
        "embedder_fingerprint",
        "strategy",
        "size",
        "overlap",
        "mode",
        "backend",
        "child_size",
        "collapse_duplicates",
        "duplicate_tier",
    )
    assert {key: left[key] for key in settings} == {key: right[key] for key in settings}
    assert left["strategy"] == "page"
    assert left["corpus_fingerprint"] != right["corpus_fingerprint"]
    for lane in ("projection", "baseline"):
        assert (run_dir / lane / META_FILE).is_file()


def _assert_delta(row, key, projection, baseline, wins, losses, ties):
    metric = row["metrics"][key]
    assert (metric["projection"], metric["baseline"]) == (projection, baseline)
    assert metric["paired"]["delta"]["mean"] == projection - baseline
    assert [metric["paired"][key] for key in ("wins", "losses", "ties")] == [wins, losses, ties]


@pytest.mark.parametrize("failure", ["unverified", "span mismatch", "unresolved", "duplicate id"])
def test_compare_book_refuses_invalid_gold_before_indexing(book, monkeypatch, failure):
    projection, _, gold, _ = book
    items = load_goldset(gold)
    item = items[-1]
    span = item.source_spans[0]
    if failure == "unverified":
        item.verified = False
    elif failure == "span mismatch":
        span.text = span.text.replace("42", "43")
    elif failure == "unresolved":
        text = (projection / span.doc_id).read_text()
        item.source_spans = [source_span(text, span.doc_id, MARKER)]
    else:
        item.id = items[0].id
    dump_goldset(items, gold)

    def unexpected_build(*args, **kwargs):
        pytest.fail("invalid gold reached the index builder")

    monkeypatch.setattr(run, "build_configured_index", unexpected_build)
    result = _invoke(book)
    assert result.exit_code == 2, result.output
    assert failure in result.output
    assert not (gold.parent / run.METHOD).exists()


def test_compare_book_refuses_another_scan(book):
    projection, _, _, _ = book
    citations = next(projection.glob("*.citations.json"))
    payload = json.loads(citations.read_text())
    payload["diagnostics"]["source_sha256"] = "0" * 64
    citations.write_text(json.dumps(payload))
    result = _invoke(book)
    assert result.exit_code == 2
    assert "different source PDFs" in result.output


def test_cross_page_span_maps_fragments_and_page_hits_use_exact_offsets(book):
    projection, baseline, gold, _ = book
    items = load_goldset(gold)
    item = items[0]
    doc_id = item.source_doc_id
    text = (projection / doc_id).read_text()
    label = text[text.index(evidence(6)) : text.index(evidence(7)) + len(evidence(7))]
    item.source_spans = [source_span(text, doc_id, label)]
    dump_goldset([item], gold)
    pair = load_book_pair(projection, baseline)
    [question] = load_book_questions(gold, projection, pair)
    assert question.pages == {6, 7}
    assert (question.present_parts, question.evidence_parts) == (1, 2)
    assert not question.baseline_text_present
    assert [span["text"] for span in question.baseline_spans] == [evidence(7)]

    pages = pair.baseline.citations.pages
    document = pair.baseline.doc_id
    # Meta advertises a page range that includes the gold pages, but only the actual
    # occurrence offsets count. The correct page without exact evidence ranks earlier.
    chunks = [
        {
            "doc_id": document,
            "char_start": page.text_start,
            "char_end": page.text_end,
            "text": pair.baseline.text[page.text_start : page.text_end],
            "metadata": {"pages": [1, N_ITEMS]},
        }
        for page in (pages[0], pages[5], pages[6])
    ]

    class FixedStore:
        def retrieve(self, question, k):
            return chunks

    docs = {document: pair.baseline}
    assert retrieve_ranks(FixedStore(), [question], docs, k=2, baseline=True) == [Ranks(None, 2)]
    assert retrieve_ranks(FixedStore(), [question], docs, k=3, baseline=True) == [Ranks(3, 2)]


def test_whitespace_only_normalization_and_insufficient_pairs(book):
    from llb.rag.book_comparison.metrics import PairedRank, ReadingSettings, read_slices

    projection, baseline, gold, _ = book
    baseline_md = next(baseline.glob("*.md"))
    baseline_md.write_text(baseline_md.read_text().replace(evidence(7), "evidence07\t42"))
    pair = load_book_pair(projection, baseline)
    questions = load_book_questions(gold, projection, pair)
    present = questions[6]
    assert present.baseline_text_present
    assert [span["text"] for span in present.baseline_spans] == ["evidence07\t42"]
    rows = [PairedRank(present, Ranks(None, None), Ranks(1, 1))]
    slices = read_slices(rows, ReadingSettings(1, MDE, resamples=32))
    assert slices["all"]["verdict"] == "inconclusive"
    assert slices["baseline_text_missing"] == {"n": 0, "verdict": "inconclusive", "metrics": {}}


def test_baseline_text_distinguishes_numeric_substitutions(book):
    from llb.rag.book_comparison.inputs import CitedDocument, _baseline_matches

    projection, baseline, _, _ = book
    original = load_book_pair(projection, baseline).baseline
    text = "42.5, -42, +42, 1.42, 142 and 42. Another 42!"
    page = original.citations.pages[0].model_copy(update={"text_start": 0, "text_end": len(text)})
    doc = CitedDocument(original.doc_id, text, original.citations)
    matches = _baseline_matches(doc, page, "42")
    assert [span["char_start"] for span in matches] == [text.index("42.", 4), text.rindex("42")]
