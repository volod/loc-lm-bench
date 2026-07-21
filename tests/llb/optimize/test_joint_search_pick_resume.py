"""Finalist pick-scoring resume acceptance test."""

from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.optimize.joint_search.pick_scoring import score_finalist_picks
from llb.optimize.objectives import GoalPick, ParetoPoint
from llb.optimize.tuner_models import MultiObjectiveResult
from llb.optimize.tuning_space import FINAL_SPLIT


def test_pick_scoring_resume_skips_completed_pick_evals(tmp_path: Path):
    cell_dir = tmp_path / "finalists" / "bravo"
    cell_dir.mkdir(parents=True)
    front = [
        ParetoPoint(
            number=0,
            quality=0.8,
            latency_s=0.5,
            cost_usd=0.0,
            throughput=2.0,
            overrides={"top_k": 5},
        ),
        ParetoPoint(
            number=1,
            quality=0.7,
            latency_s=0.2,
            cost_usd=0.0,
            throughput=5.0,
            overrides={"top_k": 3},
        ),
    ]
    tune = MultiObjectiveResult(
        study_name="joint-ci-pick-scoring-bravo",
        storage=None,
        objectives=("quality", "latency"),
        n_trials=2,
        n_complete=2,
        n_pruned=0,
        front=front,
        picks=[
            GoalPick("best_quality", front[0]),
            GoalPick("best_quality_per_second", front[1]),
        ],
    )
    eval_calls: list[str] = []

    def final_runner(config: RunConfig):
        goal = "best_quality" if config.top_k == 5 else "best_quality_per_second"
        eval_calls.append(goal)
        if goal == "best_quality_per_second" and eval_calls.count(goal) == 1:
            raise RuntimeError("simulated kill mid-pick-scoring")
        quality = 0.6 if goal == "best_quality" else 0.55
        return {
            "rows": [{"model": "bravo", "quality": quality}],
            "metrics": {"objective_score": quality},
            "manifest": {"split": FINAL_SPLIT},
            "table": "ok",
            "retrieval": {},
            "paths": {},
            "telemetry": None,
            "run_timestamp": "t",
        }

    base = RunConfig(data_dir=tmp_path, model="bravo:tag", backend="ollama")
    with pytest.raises(RuntimeError, match="simulated kill mid-pick-scoring"):
        score_finalist_picks(tune, base, cell_dir, final_runner=final_runner)

    assert eval_calls == ["best_quality", "best_quality_per_second"]
    assert (cell_dir / "picks" / "best_quality.json").is_file()
    assert not (cell_dir / "picks" / "best_quality_per_second.json").is_file()

    before = list(eval_calls)
    finals = score_finalist_picks(tune, base, cell_dir, final_runner=final_runner)
    assert eval_calls == before + ["best_quality_per_second"]
    assert set(finals) == {"best_quality", "best_quality_per_second"}
    assert finals["best_quality"]["rows"][0]["quality"] == 0.6


def test_pick_scoring_reuses_identical_goal_configs(tmp_path: Path):
    cell_dir = tmp_path / "finalists" / "same"
    cell_dir.mkdir(parents=True)
    point = ParetoPoint(
        number=0,
        quality=0.8,
        latency_s=0.5,
        cost_usd=0.0,
        throughput=2.0,
        overrides={"top_k": 5},
    )
    tune = MultiObjectiveResult(
        study_name="joint-ci-identical-picks",
        storage=None,
        objectives=("quality", "latency"),
        n_trials=1,
        n_complete=1,
        n_pruned=0,
        front=[point],
        picks=[
            GoalPick("best_quality", point),
            GoalPick("best_quality_per_second", point),
        ],
    )
    calls: list[int] = []

    def final_runner(config: RunConfig):
        calls.append(config.top_k)
        return {
            "rows": [{"model": "same", "quality": 0.6}],
            "metrics": {"objective_score": 0.6},
            "manifest": {"split": FINAL_SPLIT},
            "table": "ok",
            "retrieval": {},
            "paths": {},
            "telemetry": None,
            "run_timestamp": "t",
        }

    finals = score_finalist_picks(
        tune,
        RunConfig(data_dir=tmp_path, model="same:tag", backend="ollama"),
        cell_dir,
        final_runner=final_runner,
    )

    assert calls == [5]
    assert set(finals) == {"best_quality", "best_quality_per_second"}
    assert (cell_dir / "picks" / "best_quality_per_second.json").is_file()


def test_pick_scoring_forwards_case_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cell_dir = tmp_path / "finalists" / "limited"
    cell_dir.mkdir(parents=True)
    point = ParetoPoint(
        number=0,
        quality=0.8,
        latency_s=0.5,
        cost_usd=0.0,
        throughput=2.0,
        overrides={"top_k": 5},
    )
    tune = MultiObjectiveResult(
        study_name="joint-ci-limited-picks",
        storage=None,
        objectives=("quality", "latency"),
        n_trials=1,
        n_complete=1,
        n_pruned=0,
        front=[point],
        picks=[GoalPick("best_quality", point)],
    )
    limits: list[int | None] = []

    def fake_final(config: RunConfig, *, limit: int | None = None):
        limits.append(limit)
        return {
            "rows": [{"model": config.model, "quality": 0.6}],
            "metrics": {"objective_score": 0.6},
            "manifest": {"split": FINAL_SPLIT},
            "table": "ok",
            "retrieval": {},
            "paths": {},
            "telemetry": None,
            "run_timestamp": "t",
        }

    monkeypatch.setattr("llb.optimize.tuner_runtime._run_eval_final", fake_final)
    score_finalist_picks(
        tune,
        RunConfig(data_dir=tmp_path, model="limited:tag", backend="ollama"),
        cell_dir,
        case_limit=4,
    )

    assert limits == [4]
