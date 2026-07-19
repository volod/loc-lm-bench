"""Integration tests for multi-objective Optuna studies."""

import pytest

from llb.core.config import RunConfig
from llb.optimize.objectives import TrialMetrics


@pytest.mark.slow
def test_tune_multi_builds_pareto_front_with_fake_eval(tmp_path):
    pytest.importorskip("optuna")
    from llb.optimize.multi_objective_study import tune_multi

    base = RunConfig(data_dir=tmp_path)

    def evaluate(config: RunConfig, limit: int | None = None) -> TrialMetrics:
        del limit
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
    goals = {pick.goal for pick in result.picks}
    assert "best_quality" in goals and "best_quality_per_second" in goals
    assert (tmp_path / "report" / "pareto.json").is_file()


@pytest.mark.slow
def test_tune_multi_rebuilds_store_when_embedder_changes(tmp_path):
    pytest.importorskip("optuna")
    from llb.optimize.multi_objective_study import tune_multi
    from llb.optimize.tuner_runtime import StoreRegistry

    base = RunConfig(data_dir=tmp_path)
    registry = StoreRegistry()
    seen_embedders: list[str] = []

    def evaluate(config: RunConfig, limit: int | None = None) -> TrialMetrics:
        del limit
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
    from llb.optimize.multi_objective_study import two_stage_multi

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
    assert set(out.finals) == {pick.goal for pick in out.tune.picks}


@pytest.mark.slow
def test_tune_multi_with_cost_objective(tmp_path):
    pytest.importorskip("optuna")
    from llb.optimize.multi_objective_study import tune_multi

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
    assert "cheapest_within_floor" in {pick.goal for pick in result.picks}
