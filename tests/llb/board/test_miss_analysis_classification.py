"""Tests for miss analysis classification."""

import json
import re
from pathlib import Path
from llb.board.miss_analysis.classify import analyze_run, topic_of
from llb.board.miss_analysis.model import (
    MISS_ARTIFACT,
    MISS_CLASSES,
    MISS_GENERATION,
    MISS_RETRIEVAL,
)
from llb.board.miss_analysis.report import latest_analysis, write_analysis
from llb.board.recommend.sections import format_miss_section_md
from miss_analysis_helpers import (
    DOC_A,
    DOC_B,
    _all_class_rows,
    _analyze,
    _goldset,
    _miss_retrieval,
    _score_row,
    _write_bundle,
)


def test_span_overlap_beats_scored_hit_flag(tmp_path):
    """The persisted retrieved spans, not the scored retrieval_hit float, decide the class."""
    rows = [_score_row("m-retr", "ok", 0.1, 1.0)]  # scored flag LIES (says hit)
    run_dir = _write_bundle(tmp_path, rows, [_miss_retrieval("m-retr")])
    analysis = analyze_run(run_dir, _goldset())
    assert analysis.misses[0].miss_class == MISS_RETRIEVAL


def test_legacy_bundle_without_retrieval_jsonl_falls_back_to_hit_flag(tmp_path):
    rows = [_score_row("m-retr", "ok", 0.1, 0.0), _score_row("m-gen", "ok", 0.2, 1.0)]
    run_dir = _write_bundle(tmp_path, rows, None)
    analysis = analyze_run(run_dir, _goldset())
    by_id = {m.item_id: m.miss_class for m in analysis.misses}
    assert by_id == {"m-retr": MISS_RETRIEVAL, "m-gen": MISS_GENERATION}


def test_retrieval_miss_status_maps_directly(tmp_path):
    rows = [_score_row("m-retr", "retrieval_miss", 0.0, 0.0)]
    run_dir = _write_bundle(tmp_path, rows, None)
    analysis = analyze_run(run_dir, _goldset())
    assert analysis.misses[0].miss_class == MISS_RETRIEVAL


def test_transport_statuses_classify_as_artifacts(tmp_path):
    rows = [
        _score_row("m-retr", "timeout", 0.0, 0.0),
        _score_row("m-gen", "backend_error", 0.0, 0.0),
    ]
    run_dir = _write_bundle(tmp_path, rows, None)
    analysis = analyze_run(run_dir, _goldset())
    assert {m.miss_class for m in analysis.misses} == {MISS_ARTIFACT}


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


def test_topic_of_collapses_ukrainian_case_forms():
    genitive = topic_of("Хто виконує обов'язки начальника?", None)
    nominative = topic_of("Що робить начальник установи щодня?", None)
    assert genitive == nominative == "начальник"


def test_provenance_sidecar_labels_win_over_heuristics(tmp_path):
    rows, retrieval = _all_class_rows()
    run_dir = _write_bundle(tmp_path, rows, retrieval)
    provenance = {"m-retr": {"id": "m-retr", "question_type": "multi_hop", "topic": "ip-law"}}
    analysis = analyze_run(run_dir, _goldset(), provenance=provenance)
    miss = next(m for m in analysis.misses if m.item_id == "m-retr")
    assert miss.question_type == "multi_hop"
    assert miss.topic == "ip-law"


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


def test_write_analysis_emits_report_misses_and_payload(tmp_path):
    analysis = _analyze(tmp_path)
    out_dir = tmp_path / "miss-analysis" / "20260101T000000Z"
    paths = write_analysis(analysis, out_dir)
    report = Path(paths["report"]).read_text(encoding="utf-8")
    assert "## Miss classes" in report and "## Recommendations" in report
    misses = [json.loads(line) for line in Path(paths["misses"]).read_text().splitlines()]
    assert {m["miss_class"] for m in misses} == set(MISS_CLASSES)
    payload = json.loads(Path(paths["analysis"]).read_text(encoding="utf-8"))
    assert payload["n_misses"] == 5 and payload["class_counts"][MISS_RETRIEVAL] == 1

    latest = latest_analysis(tmp_path)
    assert latest is not None and latest["n_misses"] == 5
    assert latest["report_path"] == paths["report"]


def test_recommend_summary_gains_miss_section_only_when_analysis_exists(tmp_path):
    assert format_miss_section_md(None) == ""
    analysis = _analyze(tmp_path)
    write_analysis(analysis, tmp_path / "miss-analysis" / "20260101T000000Z")
    section = format_miss_section_md(latest_analysis(tmp_path))
    assert section.startswith("## Miss analysis")
    assert "5 of 6" in section
    assert re.search(r"\d\. ", section)  # ranked recommendation lines are quoted
