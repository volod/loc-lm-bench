"""Drive one model's budgeted Optuna study: carve the frozen dev slice, run trials against the
persistent SQLite study (resumable, wall-clock bounded), and write `hparams_manifest.json`.

`search_hyperparameters` is the entry point; `_finish` builds the manifest + result from the study.
"""

import logging
import time
from pathlib import Path
from typing import Any

from llb.bench.common import new_run_timestamp
from llb.core.config import RunConfig
from llb.core.contracts import JsonObject
from llb.finetune.dataset import load_dataset_manifest
from llb.finetune.hparam_search.dev_slice import _carve_dev_slice
from llb.finetune.hparam_search.manifest_io import _finish
from llb.finetune.hparam_search.model import (
    DEFAULT_DEV_FRACTION,
    DEFAULT_MAX_TRIALS,
    DEFAULT_SEED,
    DIGEST_TAG_CHARS,
    HPARAMS_METHOD,
    SECONDS_PER_HOUR,
    STATE_COMPLETE,
    STATE_FAILED,
    STATE_PRUNED,
    STUDY_DB,
    Clock,
    HparamSearchResult,
    ObjectiveFn,
    TrialRecord,
    TrialTrainerFn,
)
from llb.finetune.hparam_search.objective import (
    _default_objective_fn,
    _default_trainer_fn,
    _make_objective,
)
from llb.finetune.hparam_search.space import _default_estimate_fn, assert_tuning_only
from llb.finetune.naming import model_slug

_LOG = logging.getLogger(__name__)


class _WallClockBudget:
    """Between-trial wall-clock budget: an Optuna callback that stops the study past a deadline.

    A trial is atomic (a whole fine-tune), so the budget is checked BETWEEN trials. One in-flight
    trial may therefore overrun the deadline; it is never killed mid-training.
    """

    def __init__(self, now: Clock, max_hours: float | None) -> None:
        self._now = now
        self._deadline = now() + max_hours * SECONDS_PER_HOUR if max_hours else None
        self.exhausted = self._deadline is not None and now() >= self._deadline

    def __call__(self, running: Any, _trial: Any) -> None:
        if self._deadline is not None and self._now() >= self._deadline:
            self.exhausted = True
            running.stop()


def search_hyperparameters(
    config: RunConfig,
    *,
    model: str,
    dataset_dir: Path | str,
    max_trials: int = DEFAULT_MAX_TRIALS,
    max_hours: float | None = None,
    seed: int = DEFAULT_SEED,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    trainer: str = "auto",
    out_dir: Path | str | None = None,
    resume: Path | str | None = None,
    goldset_path: Path | str | None = None,
    stratify_by_base_score: Path | str | None = None,
    vram_headroom_mib: float | None = None,
    model_arch: JsonObject | None = None,
    trainer_fn: TrialTrainerFn | None = None,
    objective_fn: ObjectiveFn | None = None,
    clock: Clock | None = None,
) -> HparamSearchResult:
    """Search the LoRA space for one model on a tuning-split dataset, within a bounded budget.

    `vram_headroom_mib` (finetune-hparams-infeasible-point-prune) enables the pre-run prune: a
    trial whose estimated adapter TRAINING footprint (rank x target modules x layers, through
    `estimated_adapter_train_mib`) exceeds the headroom left beside the base model is pruned
    BEFORE the trainer runs, with the estimate in the prune reason. `model_arch` overrides the
    hidden-size/layer-count arch read from the model's cached HF config. Without a headroom the
    pre-run prune is off; the measured-OOM prune always stays in place.
    """
    import optuna

    if max_trials < 1:
        raise ValueError("max_trials must be >= 1")
    dataset_dir = Path(dataset_dir)
    dataset_manifest = load_dataset_manifest(dataset_dir)
    goldset = _resolve_goldset(goldset_path, config)
    assert_tuning_only(dataset_manifest, goldset_path=goldset)
    dev_slice = _carve_dev_slice(
        dataset_manifest, stratify_by_base_score, seed=seed, dev_fraction=dev_fraction
    )

    root = Path(resume) if resume is not None else Path(out_dir or _default_out_dir(config, model))
    root.mkdir(parents=True, exist_ok=True)
    now = clock or time.monotonic
    trainer_fn = trainer_fn or _default_trainer_fn(trainer)
    objective_fn = objective_fn or _default_objective_fn(config, dev_slice, goldset)

    study = optuna.create_study(
        direction="maximize",
        study_name=_study_name(model, str(dataset_manifest["dataset_digest"]), seed),
        storage=f"sqlite:///{root / STUDY_DB}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    remaining = max(0, max_trials - len(study.trials))
    budget = _WallClockBudget(now, max_hours)

    # A trial that fails for a reason the study cannot prune has still cost a fine-tune. The manifest
    # is written before the error propagates, so the study stays inspectable and resumable rather
    # than leaving only a `study.db` behind.
    failure: BaseException | None = None
    if remaining and not budget.exhausted:
        try:
            study.optimize(
                _make_objective(
                    model=model,
                    dataset_dir=dataset_dir,
                    dev_slice=dev_slice,
                    root=root,
                    seed=seed,
                    trainer_fn=trainer_fn,
                    objective_fn=objective_fn,
                    clock=now,
                    estimate_fn=_default_estimate_fn(model, model_arch),
                    vram_headroom_mib=vram_headroom_mib,
                ),
                n_trials=remaining,
                callbacks=[budget],
            )
        except BaseException as exc:
            failure = exc
    elif budget.exhausted:
        _LOG.warning("[finetune-hparams] time budget already exhausted; no trial started")

    trials = [_record_from_trial(trial) for trial in study.trials]
    result = _finish(
        root=root,
        model=model,
        dataset_dir=dataset_dir,
        dataset_manifest=dataset_manifest,
        dev_slice=dev_slice,
        study=study,
        trials=trials,
        seed=seed,
        max_trials=max_trials,
        max_hours=max_hours,
        budget_exhausted=budget.exhausted,
    )
    _LOG.info(
        "[finetune-hparams] %s best=%s over %d trials (%d complete, budget_exhausted=%s)",
        model,
        result.best_objective,
        len(trials),
        result.n_complete,
        result.budget_exhausted,
    )
    if failure is not None:
        _LOG.error("[finetune-hparams] study aborted; manifest -> %s", result.manifest_path)
        raise failure
    return result


def _resolve_goldset(goldset_path: Path | str | None, config: RunConfig) -> Path | None:
    if goldset_path is not None:
        return Path(goldset_path)
    return config.goldset_path if config.goldset_path.is_file() else None


def _study_name(model: str, dataset_digest: str, seed: int) -> str:
    """Pin the study to (model, dataset, seed) so a resume can never fold in a different dataset."""
    return f"{HPARAMS_METHOD}-{model_slug(model)}-{dataset_digest[:DIGEST_TAG_CHARS]}-s{seed}"


def _default_out_dir(config: RunConfig, model: str) -> Path:
    _run_id, stamp = new_run_timestamp()
    return config.data_dir / HPARAMS_METHOD / model_slug(model) / stamp


def _record_from_trial(trial: Any) -> TrialRecord:
    import optuna

    states = {
        optuna.trial.TrialState.COMPLETE: STATE_COMPLETE,
        optuna.trial.TrialState.PRUNED: STATE_PRUNED,
    }
    estimate = trial.user_attrs.get("estimated_adapter_mib")
    return TrialRecord(
        number=int(trial.number),
        state=states.get(trial.state, STATE_FAILED),
        objective=float(trial.value) if trial.value is not None else None,
        hyperparameters=dict(trial.user_attrs.get("hyperparameters") or {}),
        duration_s=trial.duration.total_seconds() if trial.duration else 0.0,
        estimated_adapter_mib=float(estimate) if estimate is not None else None,
    )
