"""multi-hop-answer-quality -- the end-to-end answer comparison of two retrieval lanes.

Pure and file-driven: the comparison consumes canonical per-case rows, and the orchestration takes
an injected lane runner, so the whole vertical runs in the lightweight CI install (no FAISS, no
backend, no GPU). The CLI wiring layers real stores and `run-eval` on top.
"""

import json
from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.eval.answer_quality import (
    compare_answer_quality,
    format_report,
    lane_config,
    lane_labels_from_comparison,
    parse_lane_label,
    parse_lanes,
    run_answer_quality,
    shared_item_ids,
)
from llb.eval.answer_quality.coverage import read_case_coverage, with_coverage
from llb.eval.answer_quality.models import (
    METRIC_OBJECTIVE,
    METRIC_RETRIEVAL_HIT,
    VERDICT_ANSWER_GAIN,
    VERDICT_INCONCLUSIVE,
    VERDICT_NO_EVIDENCE,
    VERDICT_NO_GAIN,
    VERDICT_RETRIEVAL_ONLY,
)
from llb.goldset.schema import GoldItem
from llb.rag.fusion_evidence.models import FUSED_ROW_TEMPLATE

VECTOR = "vector"
FUSED = "fused/global_community@0.10/d10"


def _row(item_id: str, objective: float, hit: float = 1.0) -> dict:
    return {
        "item_id": item_id,
        "split": "final",
        "status": "ok",
        "objective_score": objective,
        "token_f1": objective,
        "exact": 0.0,
        "contains": 0.0,
        "retrieval_hit": hit,
    }


def _lanes(vector: list[dict], fused: list[dict]) -> dict[str, list[dict]]:
    return {VECTOR: vector, FUSED: fused}


def _types(*item_ids: str) -> dict[str, str]:
    return {item_id: "multi-hop" for item_id in item_ids}


# --- lane labels --------------------------------------------------------------------------


def test_lane_label_parses_every_sweep_row_shape():
    assert parse_lane_label("vector").retrieval_backend == "faiss"
    graph = parse_lane_label("graph/local_khop")
    assert (graph.retrieval_backend, graph.retrieval_strategy) == ("graph", "local_khop")
    fused = parse_lane_label("fused/global_community@0.10/d50")
    assert fused.retrieval_backend == "fused"
    assert fused.retrieval_strategy == "global_community"
    assert fused.graph_weight == pytest.approx(0.1)
    assert fused.graph_fusion_candidates == 50


def test_fused_label_round_trips_the_sweeps_own_template():
    """The parser must never drift from the one place the sweep FORMATS a fused row label."""
    label = FUSED_ROW_TEMPLATE.format(strategy="global_community", weight=0.1, depth=10)
    spec = parse_lane_label(label)
    assert spec.label == label
    assert (spec.retrieval_strategy, spec.graph_fusion_candidates) == ("global_community", 10)
    assert spec.graph_weight == pytest.approx(0.1)


def test_fused_label_without_depth_leaves_the_lane_pool_at_top_k():
    assert parse_lane_label("fused/local_khop@0.30").graph_fusion_candidates is None


@pytest.mark.parametrize(
    "label", ["", "faiss", "fused/global_community", "fused/@0.3", "fused/x@1.5", "fused/x@0.3/d0"]
)
def test_unparseable_lane_labels_are_rejected(label: str):
    with pytest.raises(ValueError):
        parse_lane_label(label)


def test_lane_selection_deduplicates_in_the_order_given():
    assert [spec.label for spec in parse_lanes("vector, fused/x@0.3 ,vector")] == [
        "vector",
        "fused/x@0.3",
    ]


def test_lanes_are_read_from_the_sweep_verdict_that_named_the_best_row(tmp_path: Path):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"verdict": {"baseline": VECTOR, "best_row": FUSED}}))
    assert lane_labels_from_comparison(comparison) == [VECTOR, FUSED]


def test_a_sweep_verdict_without_a_best_row_is_not_scorable(tmp_path: Path):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"verdict": {"baseline": VECTOR, "best_row": None}}))
    with pytest.raises(ValueError, match="no fused row"):
        lane_labels_from_comparison(comparison)


def test_lane_config_can_reset_the_candidate_depth_that_with_overrides_would_drop():
    base = RunConfig(retrieval_backend="fused", graph_fusion_candidates=50, graph_weight=0.3)
    vector = lane_config(base, parse_lane_label(VECTOR), run_name_prefix="answer-quality")
    assert vector.retrieval_backend == "faiss"
    assert vector.graph_fusion_candidates is None
    assert vector.run_name == "answer-quality-vector"
    fused = lane_config(base, parse_lane_label("fused/local_khop@0.00"), run_name_prefix="aq")
    assert fused.graph_weight == pytest.approx(0.0)
    assert fused.retrieval_strategy == "local_khop"


# --- identical item sets ------------------------------------------------------------------


def test_lanes_scoring_different_item_sets_is_not_a_comparison():
    with pytest.raises(ValueError, match="different item sets"):
        shared_item_ids(_lanes([_row("a", 1.0)], [_row("b", 1.0)]))


def test_a_lane_that_scored_an_item_twice_is_rejected():
    with pytest.raises(ValueError, match="more than once"):
        shared_item_ids(_lanes([_row("a", 1.0), _row("a", 0.0)], [_row("a", 1.0)]))


def test_shared_item_ids_are_sorted_so_every_lane_aligns_item_by_item():
    lanes = _lanes([_row("b", 1.0), _row("a", 0.0)], [_row("a", 0.0), _row("b", 1.0)])
    assert shared_item_ids(lanes) == ["a", "b"]


def test_comparison_aligns_rows_by_item_id_not_by_file_order():
    report = compare_answer_quality(
        _lanes([_row("b", 1.0), _row("a", 0.0)], [_row("a", 0.0), _row("b", 1.0)]),
        _types("a", "b"),
        baseline=VECTOR,
        resamples=0,
    )
    focus = report["lanes"][FUSED]["slices"]["multi-hop"]
    assert focus["paired_vs_baseline"][METRIC_OBJECTIVE]["delta"]["mean"] == pytest.approx(0.0)
    assert focus["paired_vs_baseline"][METRIC_OBJECTIVE]["ties"] == 2


def test_an_unknown_baseline_lane_is_rejected():
    with pytest.raises(ValueError, match="baseline lane"):
        compare_answer_quality(_lanes([_row("a", 1.0)], [_row("a", 1.0)]), {}, baseline="missing")


# --- slices and verdicts ------------------------------------------------------------------


def _report(vector: list[float], fused: list[float], hits=None, resamples: int = 200):
    ids = [f"q{i}" for i in range(len(vector))]
    vector_hits, fused_hits = hits or ([1.0] * len(vector), [1.0] * len(fused))
    return compare_answer_quality(
        _lanes(
            [_row(i, s, h) for i, s, h in zip(ids, vector, vector_hits)],
            [_row(i, s, h) for i, s, h in zip(ids, fused, fused_hits)],
        ),
        _types(*ids),
        baseline=VECTOR,
        resamples=resamples,
    )


def test_a_consistent_objective_gain_is_an_answer_quality_gain():
    verdict = _report([0.0] * 12, [1.0] * 12)["verdict"]
    assert verdict["decision"] == VERDICT_ANSWER_GAIN
    assert verdict["best_lane"] == FUSED
    assert verdict["focus_n"] == 12


def test_a_gain_whose_interval_includes_zero_stays_inconclusive():
    verdict = _report([0.0] * 12, [1.0] + [0.0] * 11)["verdict"]
    assert verdict["decision"] == VERDICT_INCONCLUSIVE
    assert "includes no difference" in verdict["reason"]


def test_better_retrieval_that_does_not_reach_the_answer_is_a_retrieval_only_effect():
    """The finding this lane exists to produce: more evidence retrieved, no better answers."""
    verdict = _report(
        [0.0] * 12,
        [0.0] * 12,
        hits=([0.0] * 12, [1.0] * 6 + [0.0] * 6),
    )["verdict"]
    assert verdict["decision"] == VERDICT_RETRIEVAL_ONLY
    assert "retrieval-only effect" in verdict["reason"]
    assert verdict["coverage_metric"] == METRIC_RETRIEVAL_HIT


def test_a_measured_coverage_gain_outranks_a_noisy_objective_gain():
    """A +0.01 objective whose interval spans zero must not hide a coverage gain that does not."""
    verdict = _report(
        [0.0] * 11 + [0.0],
        [0.0] * 11 + [0.2],
        hits=([0.0] * 12, [1.0] * 12),
    )["verdict"]
    assert verdict["decision"] == VERDICT_RETRIEVAL_ONLY
    assert "not separable" in verdict["reason"]


# --- multi-span coverage columns ----------------------------------------------------------


def _retrieval_record(item_id: str, covered_hops: int) -> str:
    """A retrieval sidecar row for a two-hop item whose context carries `covered_hops` of them."""
    gold = [
        {"doc_id": "d1", "char_start": 0, "char_end": 10, "text": "a"},
        {"doc_id": "d2", "char_start": 0, "char_end": 10, "text": "b"},
    ]
    return (
        json.dumps(
            {
                "item_id": item_id,
                "retrieved": [dict(span, rank=i) for i, span in enumerate(gold[:covered_hops], 1)],
                "gold_spans": gold,
            }
        )
        + "\n"
    )


def test_coverage_columns_are_recomputed_from_the_bundles_retrieval_sidecar(tmp_path: Path):
    """`retrieval_hit` credits a one-hop context; `all_spans_at_k` is what a two-hop item needs."""
    (tmp_path / "retrieval.jsonl").write_text(
        _retrieval_record("q0", 1) + _retrieval_record("q1", 2), encoding="utf-8"
    )
    coverage = read_case_coverage(tmp_path, 10)
    assert coverage["q0"] == {"all_spans_at_k": 0.0, "span_coverage": 0.5}
    assert coverage["q1"] == {"all_spans_at_k": 1.0, "span_coverage": 1.0}
    enriched = with_coverage([_row("q0", 1.0)], coverage)
    assert enriched[0]["all_spans_at_k"] == 0.0
    assert enriched[0]["objective_score"] == 1.0


def test_a_bundle_without_the_sidecar_keeps_its_rows_unchanged(tmp_path: Path):
    rows = [_row("q0", 1.0)]
    assert with_coverage(rows, read_case_coverage(tmp_path, 10)) == rows


def test_the_verdict_prefers_the_graded_coverage_metric_the_sidecar_supplied():
    """`all_spans_at_k` is uniformly 0.0 on a hard multi-hop slice; graded coverage still moves."""
    ids = ["q0", "q1", "q2", "q3"]
    vector = [dict(_row(i, 0.0), all_spans_at_k=0.0, span_coverage=0.0) for i in ids]
    fused = [dict(_row(i, 0.0), all_spans_at_k=0.0, span_coverage=0.5) for i in ids]
    report = compare_answer_quality(_lanes(vector, fused), _types(*ids), baseline=VECTOR)
    assert report["metrics"] == [
        "objective_score",
        "token_f1",
        "retrieval_hit",
        "all_spans_at_k",
        "span_coverage",
    ]
    assert report["verdict"]["coverage_metric"] == "span_coverage"
    assert report["verdict"]["decision"] == VERDICT_RETRIEVAL_ONLY
    assert "span_coverage +0.500" in report["verdict"]["reason"]


def test_a_coverage_column_only_one_lane_measured_is_dropped_rather_than_zero_filled():
    ids = ["q0", "q1"]
    vector = [_row(i, 0.0) for i in ids]
    fused = [dict(_row(i, 0.0), all_spans_at_k=1.0, span_coverage=1.0) for i in ids]
    report = compare_answer_quality(_lanes(vector, fused), _types(*ids), baseline=VECTOR)
    assert "all_spans_at_k" not in report["metrics"]
    assert report["verdict"]["coverage_metric"] == METRIC_RETRIEVAL_HIT


def test_no_gain_on_either_axis_is_recorded_as_such():
    assert _report([1.0] * 6, [1.0] * 6)["verdict"]["decision"] == VERDICT_NO_GAIN


def test_a_set_without_a_focus_slice_item_claims_no_evidence():
    report = compare_answer_quality(
        _lanes([_row("a", 1.0)], [_row("a", 0.0)]), {"a": "factoid"}, baseline=VECTOR, resamples=0
    )
    assert report["verdict"]["decision"] == VERDICT_NO_EVIDENCE
    assert report["lanes"][VECTOR]["slices"]["multi-hop"]["n"] == 0
    assert report["lanes"][VECTOR]["slices"]["factoid"]["n"] == 1


def test_untyped_items_score_overall_but_join_no_slice():
    report = compare_answer_quality(
        _lanes([_row("a", 1.0), _row("b", 0.0)], [_row("a", 1.0), _row("b", 0.0)]),
        {"a": "multi-hop"},
        baseline=VECTOR,
        resamples=0,
    )
    assert report["lanes"][VECTOR]["overall"]["n"] == 2
    assert report["lanes"][VECTOR]["slices"]["multi-hop"]["n"] == 1


def test_focus_item_ledger_carries_every_lane_per_item():
    report = _report([0.0, 1.0], [1.0, 1.0], resamples=0)
    ledger = report["focus_items"]
    assert [item["item_id"] for item in ledger] == ["q0", "q1"]
    assert ledger[0]["lanes"][FUSED][METRIC_OBJECTIVE] == pytest.approx(1.0)
    assert ledger[0]["lanes"][VECTOR][METRIC_RETRIEVAL_HIT] == pytest.approx(1.0)


def test_report_renders_ascii_tables_with_the_verdict_and_the_item_ledger():
    text = format_report(_report([0.0] * 6, [1.0] * 6), metadata={"model": "m", "backend": "b"})
    assert "# Multi-hop answer quality" in text
    assert VERDICT_ANSWER_GAIN in text
    assert "### Focus slice: multi-hop" in text
    assert "### Item-level outcomes (multi-hop)" in text
    assert text.isascii()


# --- orchestration ------------------------------------------------------------------------


def _gold_item(item_id: str, verified: bool = True) -> GoldItem:
    return GoldItem(
        id=item_id,
        lang="uk",
        question=f"питання {item_id}",
        reference_answer="відповідь",
        source_doc_id="doc",
        source_spans=[{"doc_id": "doc", "char_start": 0, "char_end": 9, "text": "відповідь"}],
        provenance="human-authored",
        verified=verified,
        split="final",
    )


def _write_bundle(goldset: Path, verified: bool = True) -> None:
    """A two-item gold set whose question types live in the needle sidecar beside it."""
    items = [_gold_item("q1", verified), _gold_item("q2", verified)]
    goldset.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items),
        encoding="utf-8",
    )
    (goldset.parent / "needle_items.jsonl").write_text(
        '{"id": "q1", "question_type": "multi-hop"}\n{"id": "q2", "question_type": "factoid"}\n',
        encoding="utf-8",
    )


def _recording_lane(tmp_path: Path, seen: list[tuple[str, str, tuple[str, ...]]]):
    """A fake lane runner that persists a `scores.jsonl` the fused lane always answers better."""

    def fake_lane(config: RunConfig, items: list[GoldItem], split: str) -> Path:
        seen.append((config.run_name, config.retrieval_backend, tuple(i.id for i in items)))
        run_dir = tmp_path / "run-eval" / f"{config.run_name}-{split}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scores = run_dir / "scores.jsonl"
        objective = 1.0 if config.retrieval_backend == "fused" else 0.0
        scores.write_text(
            "".join(json.dumps(_row(item.id, objective)) + "\n" for item in items),
            encoding="utf-8",
        )
        return scores

    return fake_lane


def test_every_lane_scores_the_same_selected_items_and_the_comparison_persists(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    run = run_answer_quality(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        parse_lanes(f"{VECTOR},{FUSED}"),
        out_dir=tmp_path / "answer-quality",
        resamples=50,
        run_lane=_recording_lane(tmp_path, seen),
    )

    assert [entry[1] for entry in seen] == ["faiss", "fused"]
    assert {entry[2] for entry in seen} == {("q1", "q2")}
    assert run.report["item_ids"] == ["q1", "q2"]
    assert run.report["lanes"][FUSED]["run_dirs"] == [
        str(tmp_path / "run-eval" / f"answer-quality-{FUSED}-final")
    ]
    assert run.report["verdict"]["focus_n"] == 1
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["metadata"]["split"] == "final"
    assert persisted["metadata"]["grounding"] == "verified"
    assert Path(run.paths["report"]).read_text(encoding="utf-8").startswith("# Multi-hop")


def test_a_drafted_ledger_is_scorable_only_on_request_and_says_so_in_every_artifact(
    tmp_path: Path,
):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset, verified=False)
    lanes = parse_lanes(f"{VECTOR},{FUSED}")
    cfg = RunConfig(data_dir=tmp_path, goldset_path=goldset)
    with pytest.raises(SystemExit, match="no verified"):
        run_answer_quality(cfg, lanes, run_lane=_recording_lane(tmp_path, []))

    run = run_answer_quality(
        cfg,
        lanes,
        out_dir=tmp_path / "answer-quality",
        resamples=0,
        verified_only=False,
        run_lane=_recording_lane(tmp_path, []),
    )
    assert run.report["item_ids"] == ["q1", "q2"]
    assert "grounding: `drafted`" in Path(run.paths["report"]).read_text(encoding="utf-8")


def test_a_single_lane_is_not_a_comparison(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    with pytest.raises(ValueError, match="at least one candidate lane"):
        run_answer_quality(RunConfig(data_dir=tmp_path, goldset_path=goldset), parse_lanes(VECTOR))


def test_several_splits_pool_into_one_compared_item_set(tmp_path: Path):
    """One ordinary run bundle per (lane, split); the comparison covers the pooled ledger."""
    goldset = tmp_path / "goldset.jsonl"
    items = [_gold_item("q1"), _gold_item("q2")]
    items[1].split = "tuning"
    goldset.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items), encoding="utf-8"
    )
    (goldset.parent / "needle_items.jsonl").write_text(
        '{"id": "q1", "question_type": "multi-hop"}\n{"id": "q2", "question_type": "multi-hop"}\n',
        encoding="utf-8",
    )
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    run = run_answer_quality(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        parse_lanes(f"{VECTOR},{FUSED}"),
        splits=["final", "tuning"],
        out_dir=tmp_path / "answer-quality",
        resamples=0,
        run_lane=_recording_lane(tmp_path, seen),
    )

    assert [entry[2] for entry in seen] == [("q1",), ("q2",), ("q1",), ("q2",)]
    assert run.report["item_ids"] == ["q1", "q2"]
    assert run.report["verdict"]["focus_n"] == 2
    assert len(run.report["lanes"][VECTOR]["run_dirs"]) == 2


def test_a_split_that_selects_nothing_fails_instead_of_shrinking_the_item_set(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    with pytest.raises(SystemExit, match="tuning"):
        run_answer_quality(
            RunConfig(data_dir=tmp_path, goldset_path=goldset),
            parse_lanes(f"{VECTOR},{FUSED}"),
            splits=["final", "tuning"],
            run_lane=_recording_lane(tmp_path, []),
        )
