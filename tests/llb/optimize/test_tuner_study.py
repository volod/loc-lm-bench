"""Tests for tuner study."""

import pytest
from llb.core.config import RunConfig
from llb.optimize.tuner import (
    TwoStageResult,
    make_objective,
    suggest_overrides,
    tune,
    two_stage,
)
from test_tuner import BASE_OVERRIDES, FakeTrial, _GPU


@pytest.mark.slow
def test_tune_optimizes_toward_higher_quality(tmp_path):
    pytest.importorskip("optuna")
    base = RunConfig(data_dir=tmp_path)

    def evaluate(config: RunConfig) -> float:
        return config.top_k / 12.0 + (0.1 if config.strategy == "markdown" else 0.0)

    result = tune(base, n_trials=40, study_name="t1", evaluate=evaluate, storage=None, seed=1)
    assert result.n_complete >= 1
    assert result.best_config.top_k >= 9  # search pushed top_k high
    assert result.best_value == pytest.approx(evaluate(result.best_config))


@pytest.mark.slow
def test_two_stage_scores_winner_on_final(tmp_path):
    pytest.importorskip("optuna")
    base = RunConfig(data_dir=tmp_path)
    seen = {}

    def evaluate(config: RunConfig) -> float:
        return config.top_k / 12.0

    def final_runner(config: RunConfig):
        seen["config"] = config
        return {"rows": [{"model": "m", "quality": 0.9}]}

    out = two_stage(
        base,
        n_trials=15,
        study_name="t2",
        evaluate=evaluate,
        final_runner=final_runner,
        storage=None,
        seed=2,
    )
    assert isinstance(out, TwoStageResult)
    assert seen["config"].top_k == out.tune.best_config.top_k  # stage 2 ran the stage-1 winner
    assert out.final["rows"][0]["quality"] == 0.9


@pytest.mark.slow
def test_tune_persists_sqlite_study_for_resume(tmp_path):
    pytest.importorskip("optuna")
    base = RunConfig(data_dir=tmp_path)
    tune(base, n_trials=3, study_name="resume_me", evaluate=lambda c: c.top_k / 12.0, seed=1)
    assert (tmp_path / "optuna" / "resume_me.db").exists()  # persistent -> resumable


def test_suggest_overrides_samples_serving_params_only_for_vllm():
    vllm = suggest_overrides(FakeTrial(BASE_OVERRIDES), backend="vllm")
    assert vllm["gpu_memory_utilization"] == 0.8 and vllm["max_model_len"] == 8192
    ollama = suggest_overrides(FakeTrial(BASE_OVERRIDES), backend="ollama")
    assert "gpu_memory_utilization" not in ollama and "max_model_len" not in ollama


def test_objective_prunes_measured_oom(tmp_path):
    optuna = pytest.importorskip("optuna")
    base = RunConfig(data_dir=tmp_path)

    def evaluate(_config):
        raise RuntimeError("CUDA error: out of memory")

    objective = make_objective(base, evaluate)
    trial = optuna.trial.FixedTrial(
        {
            "strategy": "recursive",
            "chunk_size": 512,
            "overlap_frac": 0.0,
            "retrieval_mode": "flat",
            "top_k": 4,
        }
    )
    with pytest.raises(optuna.TrialPruned, match="measured OOM"):
        objective(trial)


@pytest.mark.slow
def test_tune_breaks_quality_ties_by_throughput(tmp_path):
    pytest.importorskip("optuna")
    base = RunConfig(data_dir=tmp_path)
    # constant quality -> every trial ties; throughput rises with top_k, so the fastest wins.
    result = tune(
        base,
        n_trials=40,
        study_name="tput",
        evaluate=lambda c: (1.0, float(c.top_k)),
        storage=None,
        seed=1,
    )
    assert result.best_config.top_k >= 10  # tie broken toward higher throughput


@pytest.mark.slow
def test_tune_invokes_on_trial_callback(tmp_path):
    pytest.importorskip("optuna")
    base = RunConfig(data_dir=tmp_path)
    seen = []
    tune(
        base,
        n_trials=3,
        study_name="cb",
        evaluate=lambda c: (c.top_k / 12.0, 50.0),
        storage=None,
        on_trial=seen.append,
    )
    assert len(seen) == 3 and "quality" in seen[0] and "throughput" in seen[0]


def test_with_isolation_runs_trial_under_gate(tmp_path):
    from llb.optimize.tuner import with_isolation

    base = RunConfig(data_dir=tmp_path, backend="vllm")
    seen = []

    def evaluate(config):
        seen.append(config.backend)
        return (1.0, 50.0)

    wrapped = with_isolation(
        evaluate, vram_reader=lambda: 1000, gpu_sampler=lambda: _GPU, sleep=lambda _s: None
    )
    assert wrapped(base) == (1.0, 50.0) and seen == ["vllm"]  # trial ran, gate reclaimed


def test_with_isolation_aborts_trial_on_leak(tmp_path):
    from llb.executor.vram import VramNotReclaimed
    from llb.optimize.tuner import with_isolation

    base = RunConfig(data_dir=tmp_path, backend="vllm")
    reads = iter([1000] + [9000] * 100)  # residual 8000 stuck
    usage = iter([{100: 100}, {100: 100, 200: 3000}])  # a leaked launched pid
    wrapped = with_isolation(
        lambda c: (1.0, 0.0),
        vram_reader=lambda: next(reads),
        pid_usage_reader=lambda: next(usage),
        gpu_sampler=lambda: _GPU,
        sleep=lambda _s: None,
    )
    with pytest.raises(VramNotReclaimed):
        wrapped(base)


@pytest.mark.slow
def test_tune_isolate_runs_each_trial_isolated(tmp_path):
    pytest.importorskip("optuna")
    base = RunConfig(data_dir=tmp_path, backend="vllm")
    result = tune(
        base,
        n_trials=3,
        study_name="iso",
        evaluate=lambda c: (c.top_k / 12.0, 50.0),
        storage=None,
        isolate=True,
        vram_reader=lambda: 1000,  # constant -> each trial's gate reclaims
        gpu_sampler=lambda: _GPU,
    )
    assert result.n_complete >= 1
