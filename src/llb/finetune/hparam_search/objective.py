"""The per-trial objective: subset the train sub-slice, fine-tune an adapter, and score it on the
held-out dev slice -- with the measured-OOM prune and the live trial journal.

`_make_objective` builds the Optuna objective closure; `_default_trainer_fn` and
`_default_objective_fn` supply the real (injectable) trainer and dev-slice scorer.
"""

import json
from pathlib import Path
from typing import Any, Callable

from llb.core.contracts import JsonObject
from llb.finetune.dataset import TUNING_SPLIT, subset_dataset
from llb.finetune.hparam_search.dev_slice import _dev_items
from llb.finetune.hparam_search.model import (
    STATE_COMPLETE,
    STATE_FAILED,
    STATE_PRUNED,
    TRAIN_ROLE,
    TRIAL_JOURNAL,
    TRIALS_DIRNAME,
    Clock,
    DevSlice,
    EstimateFn,
    ObjectiveFn,
    TrialRecord,
    TrialTrainerFn,
)
from llb.finetune.hparam_search.space import suggest_lora_hyperparameters
from llb.finetune.serving import BACKEND_VLLM
from llb.finetune.trainer import train_adapter
from llb.optimize.tuner import is_oom
from llb.core.config import RunConfig


def _make_objective(
    *,
    model: str,
    dataset_dir: Path,
    dev_slice: DevSlice,
    root: Path,
    seed: int,
    trainer_fn: TrialTrainerFn,
    objective_fn: ObjectiveFn,
    clock: Clock,
    estimate_fn: EstimateFn | None = None,
    vram_headroom_mib: float | None = None,
) -> Callable[[Any], float]:
    import optuna

    journal = root / TRIAL_JOURNAL

    def objective(trial: Any) -> float:
        params = suggest_lora_hyperparameters(trial)
        trial.set_user_attr("hyperparameters", params)
        estimate = estimate_fn(params) if estimate_fn is not None else None
        if estimate is not None:
            estimate = round(estimate, 1)
            trial.set_user_attr("estimated_adapter_mib", estimate)
        if estimate is not None and vram_headroom_mib is not None and estimate > vram_headroom_mib:
            # Known-infeasible before any training is paid for; the measured-OOM prune below
            # still catches what this coarse estimate cannot.
            _append_trial(
                journal,
                TrialRecord(
                    trial.number, STATE_PRUNED, None, params, estimated_adapter_mib=estimate
                ),
            )
            raise optuna.TrialPruned(
                f"estimated adapter training footprint {estimate:.0f} MiB exceeds "
                f"VRAM headroom {vram_headroom_mib:.0f} MiB"
            )
        trial_dir = root / TRIALS_DIRNAME / f"trial-{trial.number}"
        started = clock()
        try:
            subset_dataset(
                dataset_dir=dataset_dir,
                out_dir=trial_dir / "dataset",
                item_ids=dev_slice.train_ids,
                role=TRAIN_ROLE,
            )
            adapter_dir = trial_dir / "adapter"
            trainer_fn(trial_dir / "dataset", model, adapter_dir, seed, params)
            value = float(objective_fn(adapter_dir, params))
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            if is_oom(exc):
                _append_trial(
                    journal,
                    TrialRecord(
                        trial.number, STATE_PRUNED, None, params, estimated_adapter_mib=estimate
                    ),
                )
                raise optuna.TrialPruned(f"measured OOM: {exc}") from None
            _append_trial(
                journal,
                TrialRecord(
                    trial.number, STATE_FAILED, None, params, estimated_adapter_mib=estimate
                ),
            )
            raise
        _append_trial(
            journal,
            TrialRecord(
                trial.number,
                STATE_COMPLETE,
                value,
                params,
                clock() - started,
                estimated_adapter_mib=estimate,
            ),
        )
        return value

    return objective


def _default_trainer_fn(trainer: str) -> TrialTrainerFn:
    def train(
        dataset_dir: Path, model: str, adapter_dir: Path, seed: int, params: JsonObject
    ) -> JsonObject:
        return train_adapter(
            dataset_dir=dataset_dir,
            model=model,
            out_dir=adapter_dir,
            seed=seed,
            trainer=trainer,
            hyperparameters=params,
        )

    return train


def _default_objective_fn(
    config: RunConfig, dev_slice: DevSlice, goldset_path: Path | str | None
) -> ObjectiveFn:
    """Score the trial adapter on the held-out dev sub-slice, never on calibration or final.

    Both preconditions are checked HERE, before the study is created, because the first trial has
    to fine-tune a model before it ever reaches the objective: a backend that cannot serve a LoRA
    would otherwise surface only after that training is already paid for.
    """
    if config.backend != BACKEND_VLLM:
        raise SystemExit(
            f"[finetune-hparams] scoring a trial adapter needs the {BACKEND_VLLM} backend "
            f"(direct LoRA serving), but the config names {config.backend!r}; pass "
            "--backend vllm, or inject an objective"
        )
    if goldset_path is None:
        raise SystemExit(
            "[finetune-hparams] the default objective scores the dev slice through run-eval and "
            "needs a goldset; pass --goldset or inject an objective"
        )
    dev_items = _dev_items(goldset_path, dev_slice)

    def score(adapter_dir: Path, _params: JsonObject) -> float:
        from llb.executor.runner import run_eval

        result = run_eval(
            config.with_overrides(adapter_path=adapter_dir),
            items=dev_items,
            split=TUNING_SPLIT,
            emit=False,
        )
        rows = result["rows"]
        return float(rows[0]["quality"]) if rows else 0.0

    return score


def _append_trial(journal: Path, record: TrialRecord) -> None:
    """Live progress log; the manifest's trial table is rebuilt from the study, not from this."""
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
