"""Budgeted per-model LoRA hyperparameter search that never leaves the tuning split.

The split discipline of `optimize/tuner.py` is extended one level down: that tuner searches RAG and
serving knobs on the tuning split while `final` stays held out. Here the search space is the LoRA
configuration itself, and the held-out set is carved from *inside* the tuning split -- a seeded dev
sub-slice that no trial ever trains on. Calibration and final never enter a trial at all, and a
guard refuses a dataset that so much as names one of their item ids.

The Optuna conventions are the tuner's: a seeded `TPESampler`, a persistent SQLite study so a killed
search resumes, and pruned rather than crashed trials on a measured OOM. The trainer and the
objective are injectable, so CI runs a complete study over the fake trainer with a synthetic
objective and no CUDA.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Callable

from llb.bench.common import new_run_timestamp
from llb.core.config import RunConfig
from llb.core.contracts import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.finetune.dataset import TUNING_SPLIT, load_dataset_manifest, subset_dataset
from llb.finetune.guard import PROTECTED_SPLITS
from llb.finetune.naming import model_slug
from llb.finetune.serving import BACKEND_VLLM
from llb.finetune.trainer import DEFAULT_TARGET_MODULES, train_adapter
from llb.goldset.schema import GoldItem, load_goldset
from llb.optimize.tuner import is_oom

_LOG = logging.getLogger(__name__)

HPARAMS_METHOD = "finetune-hparams"
HPARAMS_MANIFEST = "hparams_manifest.json"
TRIAL_JOURNAL = "trials.jsonl"
STUDY_DB = "study.db"
TRIALS_DIRNAME = "trials"

DEFAULT_SEED = 13
DEFAULT_MAX_TRIALS = 8
DEFAULT_DEV_FRACTION = 0.25
# Below two items a "held-out dev slice" is not held out from anything: one of train/dev is empty.
MIN_SLICE_ITEMS = 2
SECONDS_PER_HOUR = 3600.0
DIGEST_TAG_CHARS = 12

# The dev sub-slice is never materialized as a dataset: only the train sub-slice is trained on, and
# the objective scores dev items straight from the goldset.
TRAIN_ROLE = "train"

STATE_COMPLETE = "complete"
STATE_PRUNED = "pruned"
STATE_FAILED = "failed"

# The LoRA search space. `lora_alpha` is sampled as a MULTIPLE of the rank rather than
# independently: the effective update scale is alpha/r, so an independent alpha would spend most of
# the budget on rank/alpha pairs that differ only in a scale the optimizer can already reach.
LORA_R_CHOICES = [4, 8, 16, 32, 64]
LORA_ALPHA_MULTIPLIERS = [1, 2, 4]
LORA_DROPOUT_RANGE = (0.0, 0.2)
LORA_DROPOUT_STEP = 0.05
LEARNING_RATE_RANGE = (1e-5, 5e-4)
EPOCHS_RANGE = (1, 4)
# Named module sets rather than a free subset: PEFT target modules are architecture-specific, and a
# sampled arbitrary subset would mostly produce configurations that fail to attach.
TARGET_MODULE_PRESETS: dict[str, list[str]] = {
    "qv": ["q_proj", "v_proj"],
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn_mlp": list(DEFAULT_TARGET_MODULES),
}

# Effective-batch axis (finetune-hparams-effective-batch-axis). Effective batch size interacts
# strongly with the learning rate, so a best learning rate is only best AT its batch geometry.
# The two knobs are sampled as ONE named geometry (per_device x grad_accum) rather than
# independently -- independent draws would mostly differ only in a VRAM/wall-clock trade at the
# same effective batch, wasting budget on gradient-equivalent points. Geometries stay
# single-per-device-heavy because the constrained 16 GB host is the design target.
BATCH_GEOMETRY_CHOICES: dict[str, tuple[int, int]] = {
    "1x4": (1, 4),  # the trainer's conservative default
    "1x8": (1, 8),
    "2x4": (2, 4),
    "2x8": (2, 8),
}
MAX_LENGTH_CHOICES = [512, 1024, 2048]

# (adapter_dir, hyperparameters) -> dev-slice objective. Injectable so CI needs no backend.
ObjectiveFn = Callable[[Path, JsonObject], float]
# (dataset_dir, model, adapter_dir, seed, hyperparameters) -> adapter manifest.
TrialTrainerFn = Callable[[Path, str, Path, int, JsonObject], JsonObject]
Clock = Callable[[], float]


@dataclass(frozen=True)
class DevSlice:
    """A seeded, disjoint split of the tuning-split item ids into train and held-out dev."""

    train_ids: tuple[str, ...]
    dev_ids: tuple[str, ...]
    seed: int
    dev_fraction: float

    def as_dict(self) -> JsonObject:
        return {
            "seed": self.seed,
            "dev_fraction": self.dev_fraction,
            "n_train": len(self.train_ids),
            "n_dev": len(self.dev_ids),
            "train_ids": list(self.train_ids),
            "dev_ids": list(self.dev_ids),
        }


@dataclass(frozen=True)
class TrialRecord:
    number: int
    state: str
    objective: float | None
    hyperparameters: JsonObject
    duration_s: float = 0.0

    def as_dict(self) -> JsonObject:
        return {
            "number": self.number,
            "state": self.state,
            "objective": self.objective,
            "hyperparameters": self.hyperparameters,
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class HparamSearchResult:
    out_dir: Path
    manifest_path: Path
    dev_slice: DevSlice
    trials: list[TrialRecord] = field(default_factory=list)
    best_trial: int | None = None
    best_objective: float | None = None
    best_hyperparameters: JsonObject | None = None
    budget_exhausted: bool = False

    @property
    def n_complete(self) -> int:
        return sum(1 for trial in self.trials if trial.state == STATE_COMPLETE)


def carve_dev_slice(
    item_ids: list[str] | tuple[str, ...],
    *,
    seed: int = DEFAULT_SEED,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
) -> DevSlice:
    """Split tuning item ids into disjoint train/dev sub-slices, deterministically for a seed."""
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(f"dev_fraction must lie strictly between 0 and 1, got {dev_fraction}")
    unique = sorted({str(item_id) for item_id in item_ids})
    if len(unique) < MIN_SLICE_ITEMS:
        raise ValueError(
            f"a held-out dev slice needs at least {MIN_SLICE_ITEMS} tuning items, got {len(unique)}"
        )
    shuffled = list(unique)
    Random(seed).shuffle(shuffled)
    # Always leave at least one item on each side, whatever the fraction rounds to.
    n_dev = min(len(unique) - 1, max(1, round(len(unique) * dev_fraction)))
    return DevSlice(
        train_ids=tuple(sorted(shuffled[n_dev:])),
        dev_ids=tuple(sorted(shuffled[:n_dev])),
        seed=seed,
        dev_fraction=dev_fraction,
    )


def assert_tuning_only(
    dataset_manifest: JsonObject, *, goldset_path: Path | str | None = None
) -> None:
    """Refuse a search dataset that carries anything but tuning-split items.

    The manifest's own `split_counts` is checked first, then -- when a goldset is available -- the
    item ids are cross-checked against the real calibration/final ids. A dataset manifest is
    operator-writable, so its split counts alone are not proof.
    """
    counts = dataset_manifest.get("split_counts") or {}
    leaked = sorted(split for split in counts if split != TUNING_SPLIT)
    if leaked:
        raise SystemExit(
            f"[finetune-hparams] search dataset carries non-tuning splits: {', '.join(leaked)}"
        )
    if goldset_path is None:
        return
    protected = {item.id for item in load_goldset(goldset_path) if item.split in PROTECTED_SPLITS}
    dataset_ids = {str(item_id) for item_id in dataset_manifest.get("item_ids") or []}
    overlap = sorted(dataset_ids & protected)
    if overlap:
        raise SystemExit(
            "[finetune-hparams] search dataset holds protected-split item ids: "
            + ", ".join(overlap)
        )


def suggest_lora_hyperparameters(trial: Any) -> JsonObject:
    """Sample one LoRA configuration. The keys are exactly what `train_adapter` consumes.

    The batch geometry rides one categorical (`batch_geometry`), so the recorded best config is
    self-consistent: the learning rate was chosen AT the recorded effective batch size, and both
    land in `hparams_manifest.json` together.
    """
    rank = trial.suggest_categorical("lora_r", LORA_R_CHOICES)
    multiplier = trial.suggest_categorical("lora_alpha_multiplier", LORA_ALPHA_MULTIPLIERS)
    preset = trial.suggest_categorical("target_modules_preset", sorted(TARGET_MODULE_PRESETS))
    geometry = trial.suggest_categorical("batch_geometry", sorted(BATCH_GEOMETRY_CHOICES))
    per_device, grad_accum = BATCH_GEOMETRY_CHOICES[geometry]
    return {
        "method": "lora",
        "lora_r": int(rank),
        "lora_alpha": int(rank) * int(multiplier),
        "lora_dropout": trial.suggest_float(
            "lora_dropout", *LORA_DROPOUT_RANGE, step=LORA_DROPOUT_STEP
        ),
        "learning_rate": trial.suggest_float("learning_rate", *LEARNING_RATE_RANGE, log=True),
        "num_train_epochs": float(trial.suggest_int("num_train_epochs", *EPOCHS_RANGE)),
        "target_modules": list(TARGET_MODULE_PRESETS[preset]),
        "target_modules_preset": preset,
        "batch_geometry": geometry,
        "per_device_train_batch_size": int(per_device),
        "gradient_accumulation_steps": int(grad_accum),
        "effective_batch_size": int(per_device) * int(grad_accum),
        "max_length": int(trial.suggest_categorical("max_length", MAX_LENGTH_CHOICES)),
    }


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
    trainer_fn: TrialTrainerFn | None = None,
    objective_fn: ObjectiveFn | None = None,
    clock: Clock | None = None,
) -> HparamSearchResult:
    """Search the LoRA space for one model on a tuning-split dataset, within a bounded budget."""
    import optuna

    if max_trials < 1:
        raise ValueError("max_trials must be >= 1")
    dataset_dir = Path(dataset_dir)
    dataset_manifest = load_dataset_manifest(dataset_dir)
    goldset = _resolve_goldset(goldset_path, config)
    assert_tuning_only(dataset_manifest, goldset_path=goldset)
    dev_slice = carve_dev_slice(
        [str(item_id) for item_id in dataset_manifest.get("item_ids") or []],
        seed=seed,
        dev_fraction=dev_fraction,
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
    finished = len(study.trials)
    remaining = max(0, max_trials - finished)
    deadline = now() + max_hours * SECONDS_PER_HOUR if max_hours else None
    budget_exhausted = deadline is not None and now() >= deadline

    def stop_when_over_budget(running: Any, _trial: Any) -> None:
        # A trial is atomic (a whole fine-tune), so the wall-clock budget is checked BETWEEN trials.
        # One in-flight trial may therefore overrun the deadline; it is never killed mid-training.
        nonlocal budget_exhausted
        if deadline is not None and now() >= deadline:
            budget_exhausted = True
            running.stop()

    # A trial that fails for a reason the study cannot prune has still cost a fine-tune. The manifest
    # is written before the error propagates, so the study stays inspectable and resumable rather
    # than leaving only a `study.db` behind.
    failure: BaseException | None = None
    if remaining and not budget_exhausted:
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
                ),
                n_trials=remaining,
                callbacks=[stop_when_over_budget],
            )
        except BaseException as exc:
            failure = exc
    elif budget_exhausted:
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
        budget_exhausted=budget_exhausted,
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


def latest_hparams_manifest(data_dir: Path | str, model: str) -> Path | None:
    """Newest `$DATA_DIR/finetune-hparams/<model>/*/hparams_manifest.json`, or None."""
    root = Path(data_dir) / HPARAMS_METHOD / model_slug(model)
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir(), reverse=True):
        manifest = candidate / HPARAMS_MANIFEST
        if manifest.is_file():
            return manifest
    return None


def load_hparams_manifest(path: Path | str) -> JsonObject:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"hparams manifest must be a JSON object: {path}")
    return data


def trainer_defaults(data_dir: Path | str, model: str) -> JsonObject:
    """`train_adapter` kwargs from this model's recorded search, or `{}` when none exists.

    A recorded best config becomes the model's default for self-improvement and campaign rounds;
    the manifest path travels into `adapter_manifest.json` so a tuned row names the search that
    chose its hyperparameters.
    """
    path = latest_hparams_manifest(data_dir, model)
    if path is None:
        return {}
    try:
        manifest = load_hparams_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _LOG.warning("[finetune-hparams] ignoring unreadable manifest %s: %s", path, exc)
        return {}
    best = manifest.get("best_hyperparameters")
    if not isinstance(best, dict) or not best:
        return {}
    return {"hyperparameters": dict(best), "hparams_manifest": str(path)}


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
) -> Callable[[Any], float]:
    import optuna

    journal = root / TRIAL_JOURNAL

    def objective(trial: Any) -> float:
        params = suggest_lora_hyperparameters(trial)
        trial.set_user_attr("hyperparameters", params)
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
                _append_trial(journal, TrialRecord(trial.number, STATE_PRUNED, None, params))
                raise optuna.TrialPruned(f"measured OOM: {exc}") from None
            _append_trial(journal, TrialRecord(trial.number, STATE_FAILED, None, params))
            raise
        _append_trial(
            journal, TrialRecord(trial.number, STATE_COMPLETE, value, params, clock() - started)
        )
        return value

    return objective


def _finish(
    *,
    root: Path,
    model: str,
    dataset_dir: Path,
    dataset_manifest: JsonObject,
    dev_slice: DevSlice,
    study: Any,
    trials: list[TrialRecord],
    seed: int,
    max_trials: int,
    max_hours: float | None,
    budget_exhausted: bool,
) -> HparamSearchResult:
    complete = [trial for trial in trials if trial.state == STATE_COMPLETE]
    best = max(complete, key=lambda trial: trial.objective or 0.0) if complete else None
    manifest: JsonObject = {
        "kind": "llb.finetune.hparams",
        "model": model,
        "dataset_dir": str(dataset_dir),
        "dataset_digest": dataset_manifest["dataset_digest"],
        "study_name": study.study_name,
        "study_seed": seed,
        "storage": f"sqlite:///{root / STUDY_DB}",
        "dev_slice": dev_slice.as_dict(),
        "max_trials": max_trials,
        "max_hours": max_hours,
        "budget_exhausted": budget_exhausted,
        "n_trials": len(trials),
        "n_complete": len(complete),
        "n_pruned": sum(1 for trial in trials if trial.state == STATE_PRUNED),
        "best_trial": best.number if best else None,
        "best_objective": best.objective if best else None,
        "best_hyperparameters": best.hyperparameters if best else None,
        "trials": [trial.as_dict() for trial in trials],
        "created_at": new_run_timestamp()[1],
    }
    manifest_path = root / HPARAMS_MANIFEST
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return HparamSearchResult(
        out_dir=root,
        manifest_path=manifest_path,
        dev_slice=dev_slice,
        trials=trials,
        best_trial=best.number if best else None,
        best_objective=best.objective if best else None,
        best_hyperparameters=best.hyperparameters if best else None,
        budget_exhausted=budget_exhausted,
    )


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


def _dev_items(goldset_path: Path | str, dev_slice: DevSlice) -> list[GoldItem]:
    wanted = set(dev_slice.dev_ids)
    items = [item for item in load_goldset(goldset_path) if item.id in wanted]
    if not items:
        raise SystemExit(
            f"[finetune-hparams] none of the {len(wanted)} dev-slice ids exist in {goldset_path}"
        )
    return items


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
    return TrialRecord(
        number=int(trial.number),
        state=states.get(trial.state, STATE_FAILED),
        objective=float(trial.value) if trial.value is not None else None,
        hyperparameters=dict(trial.user_attrs.get("hyperparameters") or {}),
        duration_s=trial.duration.total_seconds() if trial.duration else 0.0,
    )


def _append_trial(journal: Path, record: TrialRecord) -> None:
    """Live progress log; the manifest's trial table is rebuilt from the study, not from this."""
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
