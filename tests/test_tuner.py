"""Two-stage Optuna tuner (M3.4): pure search-space/fit logic + Optuna-driven studies."""

import pytest

from llb.config import RunConfig
from llb.contracts import ModelSpec
from llb.optimize.tuner import (
    TwoStageResult,
    estimate_prompt_tokens,
    fits_context,
    make_objective,
    suggest_overrides,
    tune,
    two_stage,
)

SMALL_CTX_SPEC: ModelSpec = {
    "name": "m",
    "backend": "vllm",
    "source": "org/m",
    "params_b": 3.0,
    "quant": "q4_k_m",
    "n_layers": 28,
    "kv_dim": 1024,
    "max_context": 2048,
}


class FakeTrial:
    def __init__(self, vals):
        self.vals = vals
        self.attrs = {}

    def suggest_categorical(self, name, choices):
        return self.vals[name]

    def suggest_int(self, name, lo, hi, step=1):
        return self.vals[name]

    def suggest_float(self, name, lo, hi):
        return self.vals[name]

    def set_user_attr(self, key, value):
        self.attrs[key] = value


def test_suggest_overrides_flat_keeps_overlap_below_size():
    over = suggest_overrides(
        FakeTrial(
            {
                "strategy": "markdown",
                "chunk_size": 800,
                "overlap_frac": 0.25,
                "retrieval_mode": "flat",
                "top_k": 6,
            }
        )
    )
    assert over["strategy"] == "markdown" and over["chunk_size"] == 800
    assert over["chunk_overlap"] == 200 and over["chunk_overlap"] < over["chunk_size"]
    assert "child_chunk_size" not in over  # flat mode


def test_suggest_overrides_parent_child_clamps_child_below_size():
    over = suggest_overrides(
        FakeTrial(
            {
                "strategy": "recursive",
                "chunk_size": 300,
                "overlap_frac": 0.0,
                "retrieval_mode": "parent_child",
                "top_k": 4,
                "child_chunk_size": 600,  # bigger than chunk_size -> must be clamped
            }
        )
    )
    assert over["child_chunk_size"] < over["chunk_size"]


def test_estimate_prompt_tokens_grows_with_topk_and_size():
    base = RunConfig(max_tokens=128)
    big = base.with_overrides(top_k=12, chunk_size=1200)
    small = base.with_overrides(top_k=3, chunk_size=256)
    assert estimate_prompt_tokens(big) > estimate_prompt_tokens(small)


def test_fits_context_prunes_when_retrieved_context_too_big():
    base = RunConfig(max_tokens=128)
    big = base.with_overrides(top_k=12, chunk_size=1200)  # ~4800+ tok > 2048
    small = base.with_overrides(top_k=3, chunk_size=256)  # well under 2048
    assert fits_context(big, SMALL_CTX_SPEC, 0, 0) is False
    assert fits_context(small, SMALL_CTX_SPEC, 0, 0) is True
    assert fits_context(big, None, 0, 0) is True  # no spec -> cannot judge -> not pruned


def test_objective_prunes_over_context_trial(tmp_path):
    optuna = pytest.importorskip("optuna")
    base = RunConfig(max_tokens=128, data_dir=tmp_path)
    objective = make_objective(base, lambda _c: 1.0, model_spec=SMALL_CTX_SPEC)
    trial = optuna.trial.FixedTrial(
        {
            "strategy": "recursive",
            "chunk_size": 1280,
            "overlap_frac": 0.1,
            "retrieval_mode": "flat",
            "top_k": 12,
        }
    )
    with pytest.raises(optuna.TrialPruned):
        objective(trial)


def test_tune_optimizes_toward_higher_quality(tmp_path):
    pytest.importorskip("optuna")
    base = RunConfig(data_dir=tmp_path)

    def evaluate(config: RunConfig) -> float:
        return config.top_k / 12.0 + (0.1 if config.strategy == "markdown" else 0.0)

    result = tune(base, n_trials=40, study_name="t1", evaluate=evaluate, storage=None, seed=1)
    assert result.n_complete >= 1
    assert result.best_config.top_k >= 9  # search pushed top_k high
    assert result.best_value == pytest.approx(evaluate(result.best_config))


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


def test_tune_persists_sqlite_study_for_resume(tmp_path):
    pytest.importorskip("optuna")
    base = RunConfig(data_dir=tmp_path)
    tune(base, n_trials=3, study_name="resume_me", evaluate=lambda c: c.top_k / 12.0, seed=1)
    assert (tmp_path / "optuna" / "resume_me.db").exists()  # persistent -> resumable
