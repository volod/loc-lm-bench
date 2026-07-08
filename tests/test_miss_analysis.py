"""Miss analysis (miss-analysis-recommendations): classifier, clusters, recommendations, probes.

Drives `llb.board.miss_analysis` + `llb.board.miss_probe` over a synthetic scored bundle that
contains exactly one case of every miss class plus a clean hit, so class separation (zero
cross-class leakage), numeric evidence in every recommendation line, and probe reuse/resume are
all provable without a backend, store, or GPU.
"""

import json
import re
from pathlib import Path

import pytest

from llb.board import miss_analysis as ma
from llb.board import miss_probe as mp
from llb.board.recommend import format_miss_section_md
from llb.core.config import RunConfig
from llb.executor import durability
from llb.executor.cases import CaseBatch, batch_retrieval_records
from llb.goldset.schema import GoldItem
from llb.tracking.manifest import RunManifest, persist_run

DOC_A = "doc_a.txt"
DOC_B = "doc_b.txt"
RUN_ID = "cafe01234567"


def _item(item_id: str, question: str, doc_id: str = DOC_A) -> GoldItem:
    return GoldItem(
        id=item_id,
        lang="uk",
        question=question,
        reference_answer="еталон",
        source_doc_id=doc_id,
        source_spans=[{"doc_id": doc_id, "char_start": 10, "char_end": 16, "text": "еталон"}],
        provenance="sample-generated",
        verified=True,
        split="final",
    )


def _goldset() -> list[GoldItem]:
    return [
        _item("m-retr", "Коли ухвалили закон про авторське право?"),
        _item("m-gen", "Хто підписав закон про авторське право?"),
        _item("m-refuse", "Що каже закон про авторське право?"),
        _item("m-empty", "Де зареєстрrelated закон?"),
        _item("m-judge", "Скільки статей у законі про авторське право?"),
        _item("hit", "Яка столиця України?", doc_id=DOC_B),
    ]


def _hit_retrieval(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "retrieved": [
            {"doc_id": DOC_A, "char_start": 0, "char_end": 40, "rank": 1, "text_preview": "x"}
        ],
        "gold_spans": [{"doc_id": DOC_A, "char_start": 10, "char_end": 16, "text": "еталон"}],
    }


def _miss_retrieval(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "retrieved": [
            {"doc_id": DOC_B, "char_start": 0, "char_end": 40, "rank": 1, "text_preview": "y"}
        ],
        "gold_spans": [{"doc_id": DOC_A, "char_start": 10, "char_end": 16, "text": "еталон"}],
    }


def _score_row(item_id: str, status: str, objective: float, hit: float, **extra) -> dict:
    row = {
        "item_id": item_id,
        "split": "final",
        "status": status,
        "objective_score": objective,
        "token_f1": objective,
        "exact": 0.0,
        "contains": 0.0,
        "retrieval_hit": hit,
        "first_hit_rank": 1 if hit else None,
        "tokens_per_s": 10.0,
        "latency_s": 0.5,
        "completion_tokens": 12,
        "answer_preview": "відповідь",
    }
    row.update(extra)
    return row


def _bundle_config(tmp_path: Path, **overrides) -> RunConfig:
    goldset_path = tmp_path / "goldset.jsonl"
    return RunConfig(
        data_dir=tmp_path,
        model="fake-uk",
        backend="ollama",
        top_k=5,
        run_name="rag-eval",
        goldset_path=goldset_path,
        **overrides,
    )


def _write_bundle(
    tmp_path: Path,
    rows: list[dict],
    retrieval_rows: list[dict] | None,
    *,
    name: str = "20260101T000000.000000Z-" + RUN_ID,
    objective: float = 0.3,
) -> Path:
    config = _bundle_config(tmp_path)
    run_dir = tmp_path / "run-eval" / name
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": RUN_ID,
        "run_name": config.run_name,
        "split": "final",
        "config": config.fingerprint(),
        "metrics": {"objective_score": objective, "reliability": 1.0, "tokens_per_s": 10.0},
        "n_cases": len(rows),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "scores.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    if retrieval_rows is not None:
        (run_dir / "retrieval.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in retrieval_rows),
            encoding="utf-8",
        )
    return run_dir


def _all_class_rows() -> tuple[list[dict], list[dict]]:
    """One case per miss class + one clean hit; retrieval records agree with the story."""
    rows = [
        _score_row("m-retr", "ok", 0.1, 0.0),
        _score_row("m-gen", "ok", 0.2, 1.0),
        _score_row("m-refuse", "refusal", 0.0, 1.0),
        _score_row("m-empty", "empty", 0.0, 1.0),
        _score_row("m-judge", "ok", 0.1, 1.0, judge_score=0.9),
        _score_row("hit", "ok", 1.0, 1.0),
    ]
    retrieval = [
        _miss_retrieval("m-retr"),
        _hit_retrieval("m-gen"),
        _hit_retrieval("m-refuse"),
        _hit_retrieval("m-empty"),
        _hit_retrieval("m-judge"),
        _hit_retrieval("hit"),
    ]
    return rows, retrieval


def _analyze(tmp_path: Path, **kwargs) -> ma.MissAnalysis:
    rows, retrieval = _all_class_rows()
    run_dir = _write_bundle(tmp_path, rows, retrieval)
    return ma.analyze_run(run_dir, _goldset(), **kwargs)


# --------------------------------------------------------------------------- classification


def test_classifier_separates_every_class_with_zero_leakage(tmp_path):
    analysis = _analyze(tmp_path)
    by_id = {m.item_id: m.miss_class for m in analysis.misses}
    assert by_id == {
        "m-retr": ma.MISS_RETRIEVAL,
        "m-gen": ma.MISS_GENERATION,
        "m-refuse": ma.MISS_REFUSAL,
        "m-empty": ma.MISS_ARTIFACT,
        "m-judge": ma.MISS_JUDGE,
    }
    assert "hit" not in by_id
    assert analysis.class_counts == {cls: 1 for cls in ma.MISS_CLASSES}


def test_span_overlap_beats_scored_hit_flag(tmp_path):
    """The persisted retrieved spans, not the scored retrieval_hit float, decide the class."""
    rows = [_score_row("m-retr", "ok", 0.1, 1.0)]  # scored flag LIES (says hit)
    run_dir = _write_bundle(tmp_path, rows, [_miss_retrieval("m-retr")])
    analysis = ma.analyze_run(run_dir, _goldset())
    assert analysis.misses[0].miss_class == ma.MISS_RETRIEVAL


def test_legacy_bundle_without_retrieval_jsonl_falls_back_to_hit_flag(tmp_path):
    rows = [_score_row("m-retr", "ok", 0.1, 0.0), _score_row("m-gen", "ok", 0.2, 1.0)]
    run_dir = _write_bundle(tmp_path, rows, None)
    analysis = ma.analyze_run(run_dir, _goldset())
    by_id = {m.item_id: m.miss_class for m in analysis.misses}
    assert by_id == {"m-retr": ma.MISS_RETRIEVAL, "m-gen": ma.MISS_GENERATION}


def test_retrieval_miss_status_maps_directly(tmp_path):
    rows = [_score_row("m-retr", "retrieval_miss", 0.0, 0.0)]
    run_dir = _write_bundle(tmp_path, rows, None)
    analysis = ma.analyze_run(run_dir, _goldset())
    assert analysis.misses[0].miss_class == ma.MISS_RETRIEVAL


def test_transport_statuses_classify_as_artifacts(tmp_path):
    rows = [
        _score_row("m-retr", "timeout", 0.0, 0.0),
        _score_row("m-gen", "backend_error", 0.0, 0.0),
    ]
    run_dir = _write_bundle(tmp_path, rows, None)
    analysis = ma.analyze_run(run_dir, _goldset())
    assert {m.miss_class for m in analysis.misses} == {ma.MISS_ARTIFACT}


# --------------------------------------------------------------------------- clusters + labels


def test_clusters_by_document_topic_and_question_type(tmp_path):
    analysis = _analyze(tmp_path)
    doc_rows = {row.key: row for row in analysis.clusters["document"]}
    assert doc_rows[DOC_A].n_misses == 5
    assert doc_rows[DOC_A].n_cases == 5
    assert DOC_B not in doc_rows  # the clean hit never forms a cluster row
    qtype_rows = {row.key: row for row in analysis.clusters["question_type"]}
    assert qtype_rows["when"].n_misses == 1  # "Коли ..." -> when
    assert qtype_rows["who"].n_misses == 1  # "Хто ..." -> who
    assert analysis.clusters["topic"]  # topic dimension always materializes


def test_provenance_sidecar_labels_win_over_heuristics(tmp_path):
    rows, retrieval = _all_class_rows()
    run_dir = _write_bundle(tmp_path, rows, retrieval)
    provenance = {"m-retr": {"id": "m-retr", "question_type": "multi_hop", "topic": "ip-law"}}
    analysis = ma.analyze_run(run_dir, _goldset(), provenance=provenance)
    miss = next(m for m in analysis.misses if m.item_id == "m-retr")
    assert miss.question_type == "multi_hop"
    assert miss.topic == "ip-law"


# --------------------------------------------------------------------------- recommendations


def test_every_recommendation_line_names_numeric_evidence(tmp_path):
    analysis = _analyze(tmp_path, alternatives=[("better-uk", 0.9)])
    assert analysis.recommendations
    for rec in analysis.recommendations:
        assert re.search(r"\d", rec["line"]), rec["line"]


def test_alternative_model_recommendation_cites_measured_scores(tmp_path):
    analysis = _analyze(tmp_path, alternatives=[("better-uk", 0.9), ("fake-uk", 0.3)])
    line = next(
        rec["line"] for rec in analysis.recommendations if rec["action"] == "alternative_model"
    )
    assert "better-uk" in line and "0.900" in line and "0.300" in line


def test_no_alternative_recommendation_when_nothing_scores_higher(tmp_path):
    analysis = _analyze(tmp_path, alternatives=[("worse-uk", 0.1)])
    assert all(rec["action"] != "alternative_model" for rec in analysis.recommendations)


def test_recommendations_are_ranked_by_miss_weight(tmp_path):
    analysis = _analyze(tmp_path)
    weights = [int(rec["weight"]) for rec in analysis.recommendations]
    assert weights == sorted(weights, reverse=True)


# --------------------------------------------------------------------------- artifacts + report


def test_write_analysis_emits_report_misses_and_payload(tmp_path):
    analysis = _analyze(tmp_path)
    out_dir = tmp_path / "miss-analysis" / "20260101T000000Z"
    paths = ma.write_analysis(analysis, out_dir)
    report = Path(paths["report"]).read_text(encoding="utf-8")
    assert "## Miss classes" in report and "## Recommendations" in report
    misses = [json.loads(line) for line in Path(paths["misses"]).read_text().splitlines()]
    assert {m["miss_class"] for m in misses} == set(ma.MISS_CLASSES)
    payload = json.loads(Path(paths["analysis"]).read_text(encoding="utf-8"))
    assert payload["n_misses"] == 5 and payload["class_counts"][ma.MISS_RETRIEVAL] == 1

    latest = ma.latest_analysis(tmp_path)
    assert latest is not None and latest["n_misses"] == 5
    assert latest["report_path"] == paths["report"]


def test_recommend_summary_gains_miss_section_only_when_analysis_exists(tmp_path):
    assert format_miss_section_md(None) == ""
    analysis = _analyze(tmp_path)
    ma.write_analysis(analysis, tmp_path / "miss-analysis" / "20260101T000000Z")
    section = format_miss_section_md(ma.latest_analysis(tmp_path))
    assert section.startswith("## Miss analysis")
    assert "5 of 6" in section
    assert re.search(r"\d\. ", section)  # ranked recommendation lines are quoted


# --------------------------------------------------------------------------- retrieval.jsonl


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


# --------------------------------------------------------------------------- probe mode


def _probe_manifest(tmp_path: Path) -> dict:
    run_dir = _write_bundle(tmp_path, *_all_class_rows())
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _probe_subset(analysis_misses: list[ma.MissRecord]) -> list[GoldItem]:
    items_by_id = {item.id: item for item in _goldset()}
    return sorted((items_by_id[m.item_id] for m in analysis_misses), key=lambda item: item.id)


def _write_probe_bundle(tmp_path: Path, name: str, run_name: str, subset: list[GoldItem]) -> Path:
    """A finalized probe bundle where every case hit retrieval and scored well."""
    run_dir = tmp_path / "run-eval" / name
    run_dir.mkdir(parents=True)
    rows = [_score_row(item.id, "ok", 0.8, 1.0) for item in subset]
    manifest = {
        "run_id": "probe" + name[-4:],
        "run_name": run_name,
        "split": "final",
        "n_cases": len(rows),
        "config": {},
        "metrics": {"objective_score": 0.8},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return run_dir


def test_probe_runs_miss_subset_and_confirms_retrieval_hypothesis(tmp_path):
    manifest = _probe_manifest(tmp_path)
    analysis = ma.analyze_run(
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
    ma.refresh_recommendations(analysis)
    raise_line = next(
        rec["line"] for rec in analysis.recommendations if rec["action"] == "raise_top_k"
    )
    assert "CONFIRMED" in raise_line and "top_k=8" in raise_line


def test_probe_reuses_finalized_bundle_without_rerunning(tmp_path):
    manifest = _probe_manifest(tmp_path)
    analysis = ma.analyze_run(
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
    analysis = ma.analyze_run(
        tmp_path / "run-eval" / ("20260101T000000.000000Z-" + RUN_ID), _goldset()
    )
    subset = _probe_subset(analysis.misses)
    cfg = mp.probe_config(manifest, 8)
    staging = tmp_path / "run-eval" / ".interrupted-probe.tmp"
    durability.write_journal_meta(
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
    analysis = ma.analyze_run(
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
    ma.refresh_recommendations(analysis)
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
