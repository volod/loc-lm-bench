"""Tests for finetune hparam study."""

import json
from pathlib import Path
import pytest
from llb.core.contracts.common import JsonObject
from llb.finetune.hparam_search.model import (
    HPARAMS_MANIFEST,
    STATE_COMPLETE,
    STATE_PRUNED,
)
from llb.finetune.hparam_search.search import search_hyperparameters
from finetune_hparam_helpers import (
    MODEL,
    _config,
    _dataset,
    _fake_clock,
    _rank_objective,
    _trainer_fn,
)


@pytest.mark.slow
def test_study_is_deterministic_for_a_seed(tmp_path: Path):
    dataset = _dataset(tmp_path)
    config = _config(tmp_path)

    def run(name: str):
        return search_hyperparameters(
            config,
            model=MODEL,
            dataset_dir=dataset,
            max_trials=4,
            seed=7,
            trainer="fake",
            out_dir=tmp_path / name,
            trainer_fn=_trainer_fn,
            objective_fn=_rank_objective,
        )

    first, second = run("study-a"), run("study-b")

    assert [trial.hyperparameters for trial in first.trials] == [
        trial.hyperparameters for trial in second.trials
    ]
    assert first.best_hyperparameters == second.best_hyperparameters
    assert first.best_objective == second.best_objective


@pytest.mark.slow
def test_sampled_effective_batch_is_the_product_of_its_geometry(tmp_path: Path):
    """finetune-hparams-effective-batch-axis: the two batch knobs are never independently drawn."""
    from llb.finetune.hparam_search.model import BATCH_GEOMETRY_CHOICES, MAX_LENGTH_CHOICES

    dataset = _dataset(tmp_path)
    config = _config(tmp_path)
    result = search_hyperparameters(
        config,
        model=MODEL,
        dataset_dir=dataset,
        max_trials=6,
        seed=3,
        trainer="fake",
        out_dir=tmp_path / "study",
        trainer_fn=_trainer_fn,
        objective_fn=_rank_objective,
    )
    assert result.trials
    for trial in result.trials:
        params = trial.hyperparameters
        per_device, grad_accum = BATCH_GEOMETRY_CHOICES[str(params["batch_geometry"])]
        assert params["per_device_train_batch_size"] == per_device
        assert params["gradient_accumulation_steps"] == grad_accum
        assert params["effective_batch_size"] == per_device * grad_accum
        assert params["max_length"] in MAX_LENGTH_CHOICES


@pytest.mark.slow
def test_budget_abort_stops_between_trials_and_the_study_resumes(tmp_path: Path):
    dataset = _dataset(tmp_path)
    config = _config(tmp_path)
    study_dir = tmp_path / "study"
    trained: list[int] = []

    def counting_trainer(dataset_dir: Path, model: str, adapter_dir: Path, seed: int, params):
        trained.append(int(params["lora_r"]))
        return _trainer_fn(dataset_dir, model, adapter_dir, seed, params)

    # Every clock read advances 10 minutes, so a 0.25h budget is spent after the first trial.
    aborted = search_hyperparameters(
        config,
        model=MODEL,
        dataset_dir=dataset,
        max_trials=6,
        max_hours=0.25,
        trainer="fake",
        out_dir=study_dir,
        trainer_fn=counting_trainer,
        objective_fn=_rank_objective,
        clock=_fake_clock(600.0),
    )

    assert aborted.budget_exhausted
    assert len(aborted.trials) == 1, "the in-flight trial completes; the next never starts"
    assert json.loads(aborted.manifest_path.read_text(encoding="utf-8"))["budget_exhausted"] is True
    before = len(trained)

    resumed = search_hyperparameters(
        config,
        model=MODEL,
        dataset_dir=dataset,
        max_trials=6,
        trainer="fake",
        resume=study_dir,
        trainer_fn=counting_trainer,
        objective_fn=_rank_objective,
    )

    assert not resumed.budget_exhausted
    assert len(resumed.trials) == 6
    assert [trial.number for trial in resumed.trials] == [0, 1, 2, 3, 4, 5]
    assert len(trained) - before == 5, "a resume retrains only the unfinished trials"


@pytest.mark.slow
def test_resume_at_the_same_budget_runs_no_further_trial(tmp_path: Path):
    dataset = _dataset(tmp_path)
    config = _config(tmp_path)
    study_dir = tmp_path / "study"
    scored: list[JsonObject] = []

    def counting_objective(adapter_dir: Path, params: JsonObject) -> float:
        scored.append(params)
        return _rank_objective(adapter_dir, params)

    search_hyperparameters(
        config,
        model=MODEL,
        dataset_dir=dataset,
        max_trials=3,
        trainer="fake",
        out_dir=study_dir,
        trainer_fn=_trainer_fn,
        objective_fn=counting_objective,
    )
    assert len(scored) == 3

    result = search_hyperparameters(
        config,
        model=MODEL,
        dataset_dir=dataset,
        max_trials=3,
        trainer="fake",
        resume=study_dir,
        trainer_fn=_trainer_fn,
        objective_fn=counting_objective,
    )

    assert len(scored) == 3, "an exhausted trial budget starts no further trial on resume"
    assert len(result.trials) == 3


@pytest.mark.slow
def test_a_measured_oom_prunes_the_trial_instead_of_killing_the_study(tmp_path: Path):
    dataset = _dataset(tmp_path)

    def flaky_objective(adapter_dir: Path, params: JsonObject) -> float:
        if int(params["lora_r"]) >= 64:
            raise RuntimeError("CUDA error: out of memory")
        return _rank_objective(adapter_dir, params)

    result = search_hyperparameters(
        _config(tmp_path),
        model=MODEL,
        dataset_dir=dataset,
        max_trials=4,
        trainer="fake",
        out_dir=tmp_path / "study",
        trainer_fn=_trainer_fn,
        objective_fn=flaky_objective,
    )

    states = [trial.state for trial in result.trials]
    assert STATE_PRUNED in states, "the rank-64 trial must prune on the measured OOM"
    assert states.count(STATE_COMPLETE) >= 1
    assert result.best_hyperparameters is not None
    assert int(result.best_hyperparameters["lora_r"]) < 64


def test_a_failed_trial_still_leaves_an_inspectable_manifest(tmp_path: Path):
    """A trial that fails for an unprunable reason has still cost a fine-tune; do not lose the study."""
    dataset = _dataset(tmp_path)
    study_dir = tmp_path / "study"

    def exploding_objective(_adapter_dir: Path, _params: JsonObject) -> float:
        raise RuntimeError("the backend refused this adapter")

    with pytest.raises(RuntimeError, match="refused this adapter"):
        search_hyperparameters(
            _config(tmp_path),
            model=MODEL,
            dataset_dir=dataset,
            max_trials=3,
            trainer="fake",
            out_dir=study_dir,
            trainer_fn=_trainer_fn,
            objective_fn=exploding_objective,
        )

    manifest = json.loads((study_dir / HPARAMS_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["n_trials"] == 1
    assert manifest["trials"][0]["state"] == "failed"
    assert manifest["best_hyperparameters"] is None
    assert (study_dir / "study.db").is_file(), "the study stays resumable"
