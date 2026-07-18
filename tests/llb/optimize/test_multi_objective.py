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


@pytest.mark.slow
def test_tune_multi_builds_pareto_front_with_fake_eval(tmp_path):
    pytest.importorskip("optuna")
    from llb.optimize.multi_objective import tune_multi

    base = RunConfig(data_dir=tmp_path)

    def evaluate(config: RunConfig, limit: int | None = None) -> TrialMetrics:
        del limit
        # Higher top_k -> higher quality AND higher latency (classic trade-off).
        quality = config.top_k / 12.0
        latency = float(config.top_k)
        return TrialMetrics(quality=quality, latency_s=latency, throughput=100.0 / latency)

    result = tune_multi(
        base,
        n_trials=40,
        study_name="mo1",
        objectives="quality,latency",
        evaluate=evaluate,
        storage=None,
        seed=1,
        embedders=None,
        tune_context_budget=False,
        write_report=True,
        report_dir=tmp_path / "report",
    )
    assert result.n_complete >= 2
    assert len(result.front) >= 2
    goals = {p.goal for p in result.picks}
    assert "best_quality" in goals and "best_quality_per_second" in goals
    assert (tmp_path / "report" / "pareto.json").is_file()


@pytest.mark.slow
def test_tune_multi_rebuilds_store_when_embedder_changes(tmp_path):
    pytest.importorskip("optuna")
    from llb.optimize.multi_objective import tune_multi
    from llb.optimize.tuner_runtime import StoreRegistry

    base = RunConfig(data_dir=tmp_path)
    registry = StoreRegistry()
    seen_embedders: list[str] = []

    def evaluate(config: RunConfig, limit: int | None = None) -> TrialMetrics:
        del limit
        # Simulate store-registry rebuild tracking without building a real FAISS store.
        key = (config.embedding_model, config.strategy, config.chunk_size)
        if key not in registry._cache:
            registry.builds.append(key)
            registry._cache[key] = object()
            seen_embedders.append(config.embedding_model)
        quality = 0.5 + (0.1 if "bge" in config.embedding_model else 0.0)
        return TrialMetrics(quality=quality, latency_s=1.0 + config.top_k * 0.1)

    embedders = ["intfloat/multilingual-e5-base", "BAAI/bge-m3"]
    result = tune_multi(
        base,
        n_trials=20,
        study_name="embed",
        objectives="quality,latency",
        evaluate=evaluate,
        storage=None,
        seed=3,
        embedders=embedders,
        tune_context_budget=False,
        write_report=False,
    )
    assert set(seen_embedders) == set(embedders)
    assert len(registry.builds) >= 2
    assert result.n_complete >= 1


@pytest.mark.slow
def test_two_stage_multi_scores_each_pick(tmp_path):
    pytest.importorskip("optuna")
    from llb.optimize.multi_objective import two_stage_multi

    base = RunConfig(data_dir=tmp_path)
    seen: list[str] = []

    def evaluate(config: RunConfig) -> TrialMetrics:
        return TrialMetrics(quality=config.top_k / 12.0, latency_s=float(config.top_k))

    def final_runner(config: RunConfig):
        seen.append(f"top_k={config.top_k}")
        return {"rows": [{"model": "m", "quality": 0.5}], "table": "ok"}

    out = two_stage_multi(
        base,
        n_trials=20,
        study_name="mo2",
        objectives="quality,latency",
        evaluate=evaluate,
        final_runner=final_runner,
        storage=None,
        seed=2,
        embedders=None,
        tune_context_budget=False,
        write_report=False,
    )
    assert len(out.tune.picks) >= 1
    assert len(seen) == len(out.tune.picks)
    assert set(out.finals) == {p.goal for p in out.tune.picks}


@pytest.mark.slow
def test_tune_multi_with_cost_objective(tmp_path):
    pytest.importorskip("optuna")
    from llb.optimize.multi_objective import tune_multi

    base = RunConfig(data_dir=tmp_path)

    def evaluate(config: RunConfig) -> TrialMetrics:
        return TrialMetrics(
            quality=config.top_k / 12.0,
            latency_s=float(config.top_k),
            cost_usd=0.01 * config.top_k,
        )

    result = tune_multi(
        base,
        n_trials=30,
        study_name="cost",
        objectives="quality,latency,cost",
        evaluate=evaluate,
        storage=None,
        seed=4,
        embedders=None,
        tune_context_budget=False,
        write_report=False,
    )
    assert "cheapest_within_floor" in {p.goal for p in result.picks}


def test_below_step_median_needs_warmup():
    from llb.optimize.multi_objective import PRUNE_WARMUP_TRIALS, _below_step_median

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
