"""Focused tests split from ``test_paired_minimum_evidence.py``."""

import pytest
from _paired_minimum_evidence_helpers import (
    BOUND,
    RESAMPLES,
    SEED,
    _lane_rows,
)

from llb.rag.fusion_evidence.evidence_gate import READING_INSUFFICIENT_EVIDENCE
from llb.rag.fusion_evidence.paired import paired_comparison
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    bootstrap_index_sets,
)


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_embedder_bake_off_adopts_only_on_a_reachable_reading(wins: int):
    from llb.rag.embedding_bakeoff_uncertainty import METRIC_MRR, METRIC_RECALL, paired_rows
    from llb.rag.embedding_bakeoff_verdict import DECISION_ADOPT, DECISION_RETAIN, decide_verdict

    candidate, baseline = _lane_rows(wins)
    vectors = {
        "e5": {METRIC_RECALL: baseline, METRIC_MRR: baseline},
        "bge": {METRIC_RECALL: candidate, METRIC_MRR: candidate},
    }
    verdict = decide_verdict(paired_rows(vectors, "e5", resamples=RESAMPLES, seed=SEED), "e5")
    if wins >= BOUND:
        assert verdict["decision"] == DECISION_ADOPT and verdict["model"] == "bge"
        assert "INSUFFICIENT EVIDENCE" not in verdict["reason"]
    else:
        assert verdict["decision"] == DECISION_RETAIN and verdict["model"] == "e5"
        assert "INSUFFICIENT EVIDENCE" in verdict["reason"]


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_context_ablation_reads_a_thin_uplift_as_inconclusive(wins: int):
    from llb.eval.context_ablation.compare import compare_context_strategies
    from llb.eval.context_ablation.models import (
        LANE_CLOSED_BOOK,
        LANE_RAG,
        VERDICT_RAG_PAYS_OFF,
        VERDICT_RETRIEVAL_INCONCLUSIVE,
    )
    from llb.eval.context_ablation.report import format_report

    rag, closed = _lane_rows(wins)
    lanes = {
        LANE_CLOSED_BOOK: [
            {"item_id": f"q{i:02d}", "status": "ok", "objective_score": v, "retrieval_hit": 0.0}
            for i, v in enumerate(closed)
        ],
        LANE_RAG: [
            {"item_id": f"q{i:02d}", "status": "ok", "objective_score": v, "retrieval_hit": 1.0}
            for i, v in enumerate(rag)
        ],
    }
    report = compare_context_strategies(lanes, {}, resamples=RESAMPLES, seed=SEED)
    if wins >= BOUND:
        assert report["verdict"]["decision"] == VERDICT_RAG_PAYS_OFF
    else:
        assert report["verdict"]["decision"] == VERDICT_RETRIEVAL_INCONCLUSIVE
        assert "INSUFFICIENT EVIDENCE" in report["verdict"]["reason"]
        text = format_report(report)
        assert "insufficient evidence" in text and text.isascii()


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_answer_quality_lane_needs_a_reachable_objective_to_call_a_gain(wins: int):
    from llb.eval.answer_quality.compare import compare_answer_quality
    from llb.eval.answer_quality.models import VERDICT_ANSWER_GAIN, VERDICT_INCONCLUSIVE

    fused, vector = _lane_rows(wins)
    ids = [f"q{i:02d}" for i in range(len(fused))]

    def rows(values):
        return [
            {
                "item_id": item_id,
                "status": "ok",
                "objective_score": value,
                "token_f1": value,
                "exact": 0.0,
                "contains": 0.0,
                "retrieval_hit": 1.0,
            }
            for item_id, value in zip(ids, values)
        ]

    report = compare_answer_quality(
        {"vector": rows(vector), "fused/x@0.10/d10": rows(fused)},
        dict.fromkeys(ids, "multi-hop"),
        baseline="vector",
        focus_slice="multi-hop",
        resamples=RESAMPLES,
        seed=SEED,
    )
    if wins >= BOUND:
        assert report["verdict"]["decision"] == VERDICT_ANSWER_GAIN
    else:
        # The objective still LEADS, so the honest fall-through is `inconclusive`, not `no_gain`.
        assert report["verdict"]["decision"] == VERDICT_INCONCLUSIVE
        assert "INSUFFICIENT EVIDENCE" in report["verdict"]["reason"]


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_adoption_bar_extends_only_on_a_reachable_cell(wins: int):
    from llb.eval.embedder_adoption.cross_model import (
        READING_ANSWER,
        cell_reading,
    )
    from llb.eval.embedder_adoption.models import (
        DECISION_EXTEND_BAR,
        DECISION_NO_EVIDENCE,
        METRIC_OBJECTIVE,
        METRIC_RECIPROCAL_RANK,
    )
    from llb.eval.embedder_adoption.verdict import decide_bar

    candidate, baseline = _lane_rows(wins)
    index_sets = bootstrap_index_sets(len(candidate), RESAMPLES, SEED)
    cell = {
        "label": "k10+rerank",
        "top_k": 10,
        "reranker": "x",
        "n": len(candidate),
        "lanes": {},
        "paired": {
            metric: paired_comparison(candidate, baseline, index_sets, DEFAULT_CONFIDENCE)
            for metric in (METRIC_OBJECTIVE, METRIC_RECIPROCAL_RANK)
        },
    }
    verdict = decide_bar([cell], baseline="e5", candidate="bge")
    if wins >= BOUND:
        assert verdict["decision"] == DECISION_EXTEND_BAR
        assert cell_reading(cell) == READING_ANSWER
    else:
        # Not `keep_bar`: the rank half is just as unreadable, so the sweep decided nothing.
        assert verdict["decision"] == DECISION_NO_EVIDENCE
        assert cell_reading(cell) == READING_INSUFFICIENT_EVIDENCE
        assert "INSUFFICIENT EVIDENCE" in verdict["reason"]


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_fusion_sweep_adopts_only_on_a_reachable_focus_gain(wins: int):
    from llb.rag.fusion_evidence.models import METRICS, VERDICT_ADOPT, VERDICT_INCONCLUSIVE
    from llb.rag.fusion_evidence.slices import slice_report
    from llb.rag.fusion_evidence.verdict import decide

    fused, vector = _lane_rows(wins)
    index_sets = bootstrap_index_sets(len(fused), RESAMPLES, SEED)
    positions = list(range(len(fused)))

    def row(values):
        report = slice_report(
            dict.fromkeys(METRICS, values),
            dict.fromkeys(METRICS, vector),
            positions,
            index_sets,
            DEFAULT_CONFIDENCE,
            METRICS,
        )
        return {"overall": report, "slices": {"multi-hop": report}}

    verdict = decide(
        {"vector": row(vector), "fused/x@0.10/d10": row(fused)},
        baseline="vector",
        focus_slice="multi-hop",
    )
    if wins >= BOUND:
        assert verdict["decision"] == VERDICT_ADOPT
    else:
        assert verdict["decision"] == VERDICT_INCONCLUSIVE
        assert "INSUFFICIENT EVIDENCE" in verdict["reason"]
        assert "rests on enough differing items" in verdict["reason"]


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_long_context_power_resolution_refuses_a_thin_direction(wins: int):
    from llb.eval.context_ablation.models import (
        DERIVED_LONG_CONTEXT_DELTA,
        LANE_LONG_CONTEXT,
        POWER_RESOLUTION_SEPARATED,
        POWER_RESOLUTION_UNDECIDABLE,
    )
    from llb.eval.context_ablation.power import resolve_power_analysis

    candidate, baseline = _lane_rows(wins)
    entry = {
        "label": DERIVED_LONG_CONTEXT_DELTA,
        "candidate": LANE_LONG_CONTEXT,
        "reference": "rag",
        "n": len(candidate),
        "population": "all",
        "paired": paired_comparison(
            candidate,
            baseline,
            bootstrap_index_sets(len(candidate), RESAMPLES, SEED),
            DEFAULT_CONFIDENCE,
        ),
    }
    plan = {
        "alpha": round(1.0 - DEFAULT_CONFIDENCE, 12),
        "minimum_detectable_delta": 0.01,
        "target_reached": True,
    }
    resolved = resolve_power_analysis({"derived": [entry]}, plan)  # type: ignore[arg-type]
    if wins >= BOUND:
        assert resolved["resolution"] == POWER_RESOLUTION_SEPARATED
        assert resolved["direction"] == LANE_LONG_CONTEXT
    else:
        assert resolved["resolution"] == POWER_RESOLUTION_UNDECIDABLE
        assert "differ on only" in resolved["reason"]


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_the_routing_calibration_gate_reads_the_same_rule(wins: int):
    """The coverage half of the gate is a separation; the single-span half is not, and stays."""
    from llb.rag.fusion_evidence.paired import separates as lane_separates

    candidate, baseline = _lane_rows(wins)
    multi = paired_comparison(
        candidate,
        baseline,
        bootstrap_index_sets(len(candidate), RESAMPLES, SEED),
        DEFAULT_CONFIDENCE,
    )
    single = paired_comparison(
        baseline, baseline, bootstrap_index_sets(len(baseline), RESAMPLES, SEED), DEFAULT_CONFIDENCE
    )
    gate = lane_separates(multi, DEFAULT_CONFIDENCE) and single["delta"]["lo"] >= 0.0
    assert gate is (wins >= BOUND)
