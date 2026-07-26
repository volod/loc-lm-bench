"""Shared paired-power arithmetic and the recorded compatibility contract."""

import json
from pathlib import Path

import pytest

from llb.rag.fusion_evidence.power import (
    plan_from_deltas,
    required_sample_size,
    resolvable_mde,
    resolve_power_analysis,
)
from llb.rag.fusion_evidence.stats import bootstrap_index_sets, paired_comparison


def test_the_recorded_context_plan_reproduces_exactly_through_the_shared_seam():
    root = Path(__file__).resolve().parents[3]
    run = root / ".data/context-ablation-host/context-ablation/20260725T-power-resolution"
    if not (run / "power-plan.json").is_file():
        pytest.skip("the host's recorded context-ablation artifact is not present")
    recorded = json.loads((run / "power-plan.json").read_text(encoding="utf-8"))
    reference = root / recorded["reference_artifact"]
    if not reference.is_file():
        pytest.skip("the host's recorded context-ablation reference is not present")
    payload = json.loads(reference.read_text(encoding="utf-8"))
    deltas = [
        item["lanes"]["long_context"]["objective_score"] - item["lanes"]["rag"]["objective_score"]
        for item in payload["items"]
    ]
    regenerated = plan_from_deltas(
        Path(recorded["reference_artifact"]),
        deltas,
        minimum_detectable_delta=recorded["minimum_detectable_delta"],
        target_power=recorded["target_power"],
        confidence=1.0 - recorded["alpha"],
        planned_n=recorded["planned_n"],
    )
    assert json.dumps(regenerated, indent=2) + "\n" == (run / "power-plan.json").read_text(
        encoding="utf-8"
    )


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
