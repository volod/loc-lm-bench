"""Optuna study resume tests for joint-search finalists."""

from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.core.contracts.models import ResolvedModel
from llb.optimize.objectives import TrialMetrics
from llb.optimize.tuning_space import FINAL_SPLIT


def test_remaining_optuna_trials_zero_when_study_complete(tmp_path: Path):
    optuna = pytest.importorskip("optuna")
    from llb.optimize.joint_search.resume import remaining_optuna_trials, study_name_for

    study_name = study_name_for("ci-resume-optuna", "bravo")
    db_dir = tmp_path / "optuna"
    db_dir.mkdir(parents=True)
    storage = f"sqlite:///{db_dir / f'{study_name}.db'}"
    study = optuna.create_study(
        directions=["maximize", "minimize"], study_name=study_name, storage=storage
    )

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        value = trial.suggest_float("x", 0.0, 1.0)
        return value, 1.0 - value

    study.optimize(objective, n_trials=5)
    assert remaining_optuna_trials(tmp_path, study_name, 5) == 0
    assert remaining_optuna_trials(tmp_path, study_name, 8) == 3
    assert remaining_optuna_trials(tmp_path, "missing-study", 5) == 5


def test_default_tune_finalist_adds_zero_trials_when_study_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    optuna = pytest.importorskip("optuna")
    from llb.optimize.joint_search.hooks import default_tune_finalist
    from llb.optimize.joint_search.resume import study_name_for

    run_id = "optuna-complete"
    name = "bravo"
    study_name = study_name_for(run_id, name)
    db_dir = tmp_path / "optuna"
    db_dir.mkdir(parents=True)
    storage = f"sqlite:///{db_dir / f'{study_name}.db'}"
    study = optuna.create_study(
        directions=["maximize", "minimize"], study_name=study_name, storage=storage
    )
    study.optimize(lambda trial: (trial.suggest_float("x", 0.0, 1.0), 0.1), n_trials=3)
    before = len(study.trials)

    def fake_metrics(config: RunConfig, limit: int | None = None, stores=None):
        del config, limit, stores
        return TrialMetrics(quality=0.5, latency_s=0.2)

    def fake_final(config: RunConfig):
        del config
        return {
            "rows": [{"model": name, "quality": 0.55}],
            "metrics": {"objective_score": 0.55},
            "manifest": {"split": FINAL_SPLIT},
            "table": "ok",
            "retrieval": {},
            "paths": {},
            "telemetry": None,
            "run_timestamp": "t",
        }

    monkeypatch.setattr("llb.optimize.tuner_runtime._run_eval_metrics", fake_metrics)
    monkeypatch.setattr("llb.optimize.tuner_runtime._run_eval_final", fake_final)

    cell_dir = tmp_path / "joint-search" / run_id / "finalists" / name
    cell_dir.mkdir(parents=True)
    resolution = ResolvedModel(
        name=name,
        chosen_backend="ollama",
        chosen_source="bravo:tag",
        verdict="gpu",
        candidates=[],
        note="ok",
    )
    result = default_tune_finalist(
        RunConfig(data_dir=tmp_path, model="bravo:tag", backend="ollama"),
        resolution,
        cell_dir,
        n_trials=3,
        objectives=["quality", "latency"],
        seed=1,
        isolate=False,
        vram_reader=None,
        pid_usage_reader=None,
        vram_mib=0,
        ram_mib=0,
        max_model_len=8192,
    )
    reloaded = optuna.load_study(study_name=study_name, storage=storage)
    assert len(reloaded.trials) == before
    assert result.study_name == study_name
    assert "best_quality" in result.finals or result.finals
