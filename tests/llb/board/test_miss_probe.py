"""Tests for miss probe."""

import json
from pathlib import Path
import pytest
from llb.board.miss_analysis.classify import (
    analyze_run,
    retrieval_hit_from_record,
    retrieved_docs_from_record,
)
from llb.board.miss_analysis.recommendations import refresh_recommendations
from llb.board import miss_probe as mp
from llb.executor import durability_journal
from llb.executor.cases import CaseBatch, batch_retrieval_records
from llb.tracking.manifest import RunManifest, persist_run
from miss_analysis_helpers import DOC_A, RUN_ID, _analyze, _goldset, _score_row
from miss_probe_helpers import _probe_manifest, _probe_subset, _write_probe_bundle


def test_persist_run_writes_additive_retrieval_records(tmp_path):
    items = _goldset()[:2]
    batch = CaseBatch(
        rows=[_score_row(items[0].id, "ok", 1.0, 1.0), _score_row(items[1].id, "ok", 0.0, 0.0)],
        retrieval_pairs=[
            (
                [
                    {
                        "doc_id": DOC_A,
                        "char_start": 0,
                        "char_end": 40,
                        "text": "т" * 500,
                        "retrieval_score": 0.9,
                    }
                ],
                [span.model_dump() for span in items[0].source_spans],
            ),
            ([], [span.model_dump() for span in items[1].source_spans]),
        ],
        answers=[(items[0], "відповідь"), (items[1], "")],
    )
    records = batch_retrieval_records(batch)
    manifest = RunManifest(run_id="r1", run_name="t", split="final", config={}, n_cases=2)
    out_dir = tmp_path / "run-eval" / "bundle"
    paths = persist_run(
        manifest, batch.rows, out_dir, mirror=lambda *a: None, retrieval_rows=records
    )
    lines = [json.loads(line) for line in Path(paths["retrieval"]).read_text().splitlines()]
    assert [rec["item_id"] for rec in lines] == [items[0].id, items[1].id]
    first = lines[0]["retrieved"][0]
    assert first["rank"] == 1 and first["retrieval_score"] == 0.9
    assert len(first["text_preview"]) == 160  # bounded preview, never the full chunk
    assert lines[0]["gold_spans"][0]["doc_id"] == DOC_A
    assert lines[1]["retrieved"] == []


def test_persisted_records_keep_the_places_a_collapsed_chunk_stands_for(tmp_path):
    """End to end: a hit the run scored through a duplicate copy is a hit for miss analysis too."""
    item = _goldset()[0]
    gold = [span.model_dump() for span in item.source_spans]
    survivor = {
        "doc_id": "other.md",  # the indexed copy lives elsewhere; the gold span is on the copy
        "chunk_id": "other.md#0000",
        "char_start": 0,
        "char_end": 40,
        "text": "повторюваний блок",
        "metadata": {
            "duplicate_count": 2,
            "duplicate_occurrences": [
                {"doc_id": DOC_A, "chunk_id": f"{DOC_A}#0000", "char_start": 0, "char_end": 40}
            ],
        },
    }
    batch = CaseBatch(
        rows=[_score_row(item.id, "ok", 0.1, 1.0)],
        retrieval_pairs=[([survivor], gold)],
        answers=[(item, "відповідь")],
    )
    record = batch_retrieval_records(batch)[0]
    persisted = record["retrieved"][0]
    assert persisted["duplicate_count"] == 2
    assert [c["doc_id"] for c in persisted["duplicate_occurrences"]] == [DOC_A]
    assert retrieval_hit_from_record(dict(record)) is True
    assert retrieved_docs_from_record(dict(record)) == ["other.md", DOC_A]


def test_probe_runs_miss_subset_and_confirms_retrieval_hypothesis(tmp_path):
    manifest = _probe_manifest(tmp_path)
    analysis = analyze_run(
        tmp_path / "run-eval" / ("20260101T000000.000000Z-" + RUN_ID), _goldset()
    )
    calls: list[dict] = []

    def fake_run_eval(cfg, *, items, split, resume=None, emit=True):
        calls.append({"cfg": cfg, "items": items, "split": split, "resume": resume})
        name = f"fake-{cfg.top_k}"
        _write_probe_bundle(tmp_path, name, cfg.run_name, items)
        return {"run_timestamp": name}

    outcomes = mp.run_probes(manifest, analysis.misses, _goldset(), [8], run_eval_fn=fake_run_eval)
    assert len(calls) == 1
    assert [item.id for item in calls[0]["items"]] == sorted(m.item_id for m in analysis.misses)
    assert calls[0]["cfg"].top_k == 8
    assert calls[0]["cfg"].run_name == mp.probe_run_name(RUN_ID, 8)
    outcome = outcomes[0]
    assert outcome["n_items"] == 5 and outcome["recovered_retrieval"] == 1
    assert outcome["n_retrieval_misses"] == 1 and not outcome["reused"]

    analysis.probes = outcomes
    refresh_recommendations(analysis)
    raise_line = next(
        rec["line"] for rec in analysis.recommendations if rec["action"] == "raise_top_k"
    )
    assert "CONFIRMED" in raise_line and "top_k=8" in raise_line


def test_probe_reuses_finalized_bundle_without_rerunning(tmp_path):
    manifest = _probe_manifest(tmp_path)
    analysis = analyze_run(
        tmp_path / "run-eval" / ("20260101T000000.000000Z-" + RUN_ID), _goldset()
    )
    subset = _probe_subset(analysis.misses)
    _write_probe_bundle(tmp_path, "existing-probe", mp.probe_run_name(RUN_ID, 8), subset)

    def must_not_run(*args, **kwargs):
        raise AssertionError("a finalized probe bundle must be reused, never re-run")

    outcomes = mp.run_probes(manifest, analysis.misses, _goldset(), [8], run_eval_fn=must_not_run)
    assert outcomes[0]["reused"] and outcomes[0]["recovered_retrieval"] == 1


def test_probe_resumes_interrupted_staging_via_journal_meta(tmp_path):
    manifest = _probe_manifest(tmp_path)
    analysis = analyze_run(
        tmp_path / "run-eval" / ("20260101T000000.000000Z-" + RUN_ID), _goldset()
    )
    subset = _probe_subset(analysis.misses)
    cfg = mp.probe_config(manifest, 8)
    staging = tmp_path / "run-eval" / ".interrupted-probe.tmp"
    durability_journal.write_journal_meta(
        staging, config_fingerprint=cfg.fingerprint(), items=subset, run_id="x", split="final"
    )
    seen: dict = {}

    def fake_run_eval(cfg, *, items, split, resume=None, emit=True):
        seen["resume"] = resume
        _write_probe_bundle(tmp_path, "resumed-probe", cfg.run_name, items)
        return {"run_timestamp": "resumed-probe"}

    outcomes = mp.run_probes(manifest, analysis.misses, _goldset(), [8], run_eval_fn=fake_run_eval)
    assert seen["resume"] == tmp_path / "run-eval" / "interrupted-probe"
    assert outcomes[0]["resumed"] and not outcomes[0]["reused"]


def test_probe_skips_depth_equal_to_run_top_k(tmp_path):
    manifest = _probe_manifest(tmp_path)
    analysis = analyze_run(
        tmp_path / "run-eval" / ("20260101T000000.000000Z-" + RUN_ID), _goldset()
    )

    def must_not_run(*args, **kwargs):
        raise AssertionError("probing the run's own top_k re-measures the baseline")

    assert mp.run_probes(manifest, analysis.misses, _goldset(), [5], run_eval_fn=must_not_run) == []


def test_lower_top_k_recommendation_requires_measured_gain(tmp_path):
    analysis = _analyze(tmp_path)
    analysis.probes = [
        {
            "top_k": 3,
            "n_items": 5,
            "mean_objective": 0.4,
            "base_mean_objective": 0.08,
            "recovered_retrieval": 0,
            "n_retrieval_misses": 1,
            "run_dir": "x",
            "reused": False,
            "resumed": False,
        }
    ]
    refresh_recommendations(analysis)
    line = next(rec["line"] for rec in analysis.recommendations if rec["action"] == "lower_top_k")
    assert "0.400" in line and "0.080" in line


def test_parse_probe_depths_validates_input():
    assert mp.parse_probe_depths("8,3,8") == [3, 8]
    with pytest.raises(SystemExit):
        mp.parse_probe_depths("three")
    with pytest.raises(SystemExit):
        mp.parse_probe_depths("0")
    with pytest.raises(SystemExit):
        mp.parse_probe_depths(",")
