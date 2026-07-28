"""Shared paired-power arithmetic."""

from pathlib import Path

import pytest
from _paired_minimum_evidence_helpers import BOUND, RESAMPLES, SEED, _lane_rows

from llb.eval.context_ablation.models import (
    LANE_LONG_CONTEXT,
    LANE_RAG,
    POWER_RESOLUTION_SEPARATED,
    POWER_RESOLUTION_UNDECIDABLE,
)
from llb.rag.fusion_evidence.paired import paired_comparison
from llb.rag.fusion_evidence.power import (
    plan_from_deltas,
    required_sample_size,
    resolvable_mde,
    resolve_power_analysis,
)
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, bootstrap_index_sets


def test_inverted_mde_and_realized_sd_recheck_use_the_same_arithmetic(tmp_path: Path):
    plan = plan_from_deltas(
        tmp_path / "reference.json",
        [-0.01, 0.0, 0.01, 0.0],
        minimum_detectable_delta=0.1,
        target_power=0.8,
        confidence=0.95,
        planned_n=20,
        selector={
            "lane": "test",
            "candidate": "candidate",
            "baseline": "base",
            "metric": "recall_at_k",
            "population": "all",
        },
    )
    realized = [1.0, -1.0] * 10
    paired = paired_comparison(
        realized,
        [0.0] * len(realized),
        bootstrap_index_sets(len(realized), 200, 17),
    )
    analysis = resolve_power_analysis(
        plan, realized, paired, candidate="candidate", baseline="base"
    )
    assert plan["target_reached"]
    assert analysis["planned_target_reached"]
    assert not analysis["target_reached"]
    assert analysis["realized_sd_exceeds_plan"]
    assert analysis["resolvable_mde"] == pytest.approx(
        resolvable_mde(
            analysis["realized_sample_sd"],
            len(realized),
            alpha=0.05,
            target_power=0.8,
        )
    )
    assert analysis["realized_required_n"] == required_sample_size(
        analysis["realized_sample_sd"], 0.1, alpha=0.05, target_power=0.8
    )


@pytest.mark.parametrize("wins", [BOUND - 1, BOUND])
def test_long_context_resolution_refuses_a_thin_direction(wins: int):
    candidate, baseline = _lane_rows(wins)
    deltas = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(candidate, baseline)
    ]
    paired = paired_comparison(
        candidate,
        baseline,
        bootstrap_index_sets(len(candidate), RESAMPLES, SEED),
        DEFAULT_CONFIDENCE,
    )
    plan = plan_from_deltas(
        Path("reference.json"),
        deltas,
        minimum_detectable_delta=0.01,
        target_power=0.8,
        confidence=DEFAULT_CONFIDENCE,
        planned_n=len(candidate),
        selector={
            "lane": "compare-context-strategies",
            "candidate": LANE_LONG_CONTEXT,
            "baseline": LANE_RAG,
            "metric": "objective_score",
            "population": "all",
        },
    )
    resolved = resolve_power_analysis(
        plan,
        deltas,
        paired,
        candidate=LANE_LONG_CONTEXT,
        baseline=LANE_RAG,
    )
    if wins >= BOUND:
        assert resolved["resolution"] == POWER_RESOLUTION_SEPARATED
        assert resolved["direction"] == LANE_LONG_CONTEXT
    else:
        assert resolved["resolution"] == POWER_RESOLUTION_UNDECIDABLE
        assert "paired items differ" in resolved["reason"]
