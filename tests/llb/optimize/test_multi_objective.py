"""Unit tests for multi-objective vocabulary, knobs, Pareto picks, and NSGA-II studies."""

import json

import pytest

from llb.core.config import RunConfig
from llb.optimize.objectives import (
    ParetoPoint,
    TrialMetrics,
    normalize_outcome,
    parse_objectives,
    select_goal_picks,
    study_directions,
)
from llb.optimize.pareto_report import write_pareto_report
from llb.optimize.tuning_space import (
    CONTEXT_BUDGET_CHOICES,
    fits_context,
    suggest_overrides,
)
from test_tuner import FakeTrial, SMALL_CTX_SPEC


def test_parse_objectives_requires_quality_and_two_goals():
    assert parse_objectives("quality,latency") == ("quality", "latency")
    assert parse_objectives("quality,latency,cost") == ("quality", "latency", "cost")
    with pytest.raises(ValueError, match="at least two"):
        parse_objectives("quality")
    with pytest.raises(ValueError, match="unknown"):
        parse_objectives("quality,vram")
    with pytest.raises(ValueError, match="must include 'quality'"):
        parse_objectives("latency,cost")


def test_study_directions_maximize_quality_minimize_rest():
    assert study_directions(("quality", "latency", "cost")) == [
        "maximize",
        "minimize",
        "minimize",
    ]


def test_normalize_outcome_shapes():
    assert normalize_outcome(0.5).quality == 0.5
    assert normalize_outcome((0.7, 40.0)).throughput == 40.0
    triple = normalize_outcome((0.6, 1.2, 0.05))
    assert triple.latency_s == 1.2 and triple.cost_usd == 0.05
    assert normalize_outcome(TrialMetrics(0.9, 2.0)).latency_s == 2.0


def test_suggest_overrides_samples_embedder_and_context_budget():
    vals = {
        "strategy": "recursive",
        "chunk_size": 512,
        "overlap_frac": 0.1,
        "retrieval_mode": "flat",
        "top_k": 5,
        "embedding_model": "BAAI/bge-m3",
        "context_budget": 4096,
        "gpu_memory_utilization": 0.8,
    }
    over = suggest_overrides(
        FakeTrial(vals),
        backend="vllm",
        embedders=["intfloat/multilingual-e5-base", "BAAI/bge-m3"],
        tune_context_budget=True,
    )
    assert over["embedding_model"] == "BAAI/bge-m3"
    assert over["context_budget"] == 4096
    assert over["max_model_len"] == 4096  # coupled to budget
    assert 4096 in CONTEXT_BUDGET_CHOICES


def test_fits_context_respects_explicit_budget():
    big = RunConfig(top_k=12, chunk_size=1200, max_tokens=128, context_budget=2048)
    small = RunConfig(top_k=3, chunk_size=256, max_tokens=128, context_budget=2048)
    assert fits_context(big, None, 0, 0) is False
    assert fits_context(small, None, 0, 0) is True
    assert fits_context(big, SMALL_CTX_SPEC, 0, 0) is False


def test_select_goal_picks_names_quality_and_efficiency():
    front = [
        ParetoPoint(0, quality=0.9, latency_s=10.0, cost_usd=1.0, throughput=5.0, overrides={}),
        ParetoPoint(1, quality=0.8, latency_s=1.0, cost_usd=0.2, throughput=50.0, overrides={}),
        ParetoPoint(2, quality=0.85, latency_s=2.0, cost_usd=0.1, throughput=40.0, overrides={}),
    ]
    picks = {p.goal: p.point for p in select_goal_picks(front, include_cost=True)}
    assert picks["best_quality"].number == 0
    assert picks["best_quality_per_second"].number == 1  # 0.8 / 1.0 beats 0.9 / 10
    assert picks["cheapest_within_floor"].number == 2  # within 0.9*0.9 floor, lowest cost


def test_write_pareto_report_json_and_markdown(tmp_path):
    front = [
        ParetoPoint(0, 0.5, 1.0, 0.0, 10.0, overrides={"top_k": 5, "strategy": "recursive"}),
        ParetoPoint(1, 0.4, 0.5, 0.0, 20.0, overrides={"top_k": 3}),
    ]
    picks = select_goal_picks(front)
    paths = write_pareto_report(
        tmp_path,
        study_name="s1",
        objectives=("quality", "latency"),
        front=front,
        picks=picks,
        n_trials=10,
        n_complete=8,
        n_pruned=2,
    )
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["study_name"] == "s1" and len(payload["pareto_front"]) == 2
    md = paths["markdown"].read_text(encoding="utf-8")
    assert "Pareto front" in md and "best_quality" in md


def test_below_step_median_needs_warmup():
    from llb.optimize.multi_objective_trial import PRUNE_WARMUP_TRIALS, _below_step_median

    class _T:
        def __init__(self, number, attrs, study_trials):
            self.number = number
            self.user_attrs = attrs
            self.study = type("S", (), {"trials": study_trials})()

    priors = [_T(i, {"prune_quality_step_0": 0.5}, []) for i in range(PRUNE_WARMUP_TRIALS)]
    for t in priors:
        t.study.trials = priors
    current = _T(99, {}, priors)
    assert _below_step_median(current, 0, 0.1) is True
    assert _below_step_median(current, 0, 0.9) is False
    assert _below_step_median(_T(0, {}, priors[:2]), 0, 0.0) is False
