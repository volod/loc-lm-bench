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

# Base-score strata for `--stratify-by-base-score`. An item is ANSWERABLE when the base model
# already scores above zero on it; a dev slice with no answerable item makes the trial objective
# a near-constant that ranks every configuration the same (the first CUDA search on this repo hit
# exactly that with a uniform 3-item slice). Buckets are drawn from in ANSWERABLE-FIRST priority
# order so the floor-of-one lands on discriminating items before zeros.
BUCKET_HIGH = "high"  # objective_score >= HIGH_SCORE_BOUNDARY
BUCKET_LOW = "low"  # 0 < objective_score < HIGH_SCORE_BOUNDARY
BUCKET_ZERO = "zero"  # objective_score == 0.0
BUCKET_UNSCORED = "unscored"  # item absent from the base run's scores.jsonl
HIGH_SCORE_BOUNDARY = 0.5
BUCKET_PRIORITY = (BUCKET_HIGH, BUCKET_LOW, BUCKET_ZERO, BUCKET_UNSCORED)
SCORES_FILENAME = "scores.jsonl"
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

# Pre-run VRAM feasibility (finetune-hparams-infeasible-point-prune). A LoRA pair holds two
# matrices of `hidden x rank` per targeted module per layer, and training multiplies each
# parameter by weight + gradient (bf16) plus fp32 Adam moments and master copy. The estimate is
# deliberately coarse -- it exists to prune KNOWN-infeasible points before a fine-tune is paid
# for, not to replace the measured-OOM prune that stays in place after it.
LORA_MATRICES_PER_MODULE = 2
ADAPTER_TRAIN_BYTES_PER_PARAM = 16.0  # 2 weight + 2 grad + 8 Adam m/v + 4 fp32 master

# (adapter_dir, hyperparameters) -> dev-slice objective. Injectable so CI needs no backend.
ObjectiveFn = Callable[[Path, JsonObject], float]
# (dataset_dir, model, adapter_dir, seed, hyperparameters) -> adapter manifest.
TrialTrainerFn = Callable[[Path, str, Path, int, JsonObject], JsonObject]
Clock = Callable[[], float]
# hyperparameters -> estimated training footprint in MiB, or None when the arch is unknown.
EstimateFn = Callable[[JsonObject], float | None]


@dataclass(frozen=True)
class DevSlice:
    """A seeded, disjoint split of the tuning-split item ids into train and held-out dev.

    `strata` is set only by the stratified carve: per-bucket population/dev counts and the
    base-score distribution the slice was drawn against, recorded into `hparams_manifest.json`.
    """

    train_ids: tuple[str, ...]
    dev_ids: tuple[str, ...]
    seed: int
    dev_fraction: float
    strata: JsonObject | None = None

    def as_dict(self) -> JsonObject:
        payload: JsonObject = {
            "seed": self.seed,
            "dev_fraction": self.dev_fraction,
            "n_train": len(self.train_ids),
            "n_dev": len(self.dev_ids),
            "train_ids": list(self.train_ids),
            "dev_ids": list(self.dev_ids),
        }
        if self.strata is not None:
            payload["strata"] = self.strata
        return payload


@dataclass(frozen=True)
class TrialRecord:
    number: int
    state: str
    objective: float | None
    hyperparameters: JsonObject
    duration_s: float = 0.0
    estimated_adapter_mib: float | None = None

    def as_dict(self) -> JsonObject:
        return {
            "number": self.number,
            "state": self.state,
            "objective": self.objective,
            "hyperparameters": self.hyperparameters,
            "duration_s": round(self.duration_s, 3),
            "estimated_adapter_mib": self.estimated_adapter_mib,
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


def load_base_scores(run_dir: Path | str) -> dict[str, float]:
    """Per-item base-model `objective_score` from a scored run bundle's `scores.jsonl`."""
    path = Path(run_dir) / SCORES_FILENAME
    if not path.is_file():
        raise ValueError(f"--stratify-by-base-score run has no {SCORES_FILENAME}: {run_dir}")
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        item_id = row.get("item_id")
        value = row.get("objective_score")
        if item_id is not None and value is not None:
            scores[str(item_id)] = float(value)
    return scores


def base_score_bucket(score: float | None) -> str:
    """The base-score stratum an item falls into (see `BUCKET_PRIORITY`)."""
    if score is None:
        return BUCKET_UNSCORED
    if score <= 0.0:
        return BUCKET_ZERO
    return BUCKET_HIGH if score >= HIGH_SCORE_BOUNDARY else BUCKET_LOW


def _dev_quota_per_bucket(buckets: dict[str, list[str]], n_dev: int, total: int) -> dict[str, int]:
    """Proportional dev quota per bucket: floor of one (answerable buckets first), then
    largest-remainder top-up, each bucket capped at its own size. Deterministic."""
    quotas = {bucket: 0 for bucket in buckets}
    remaining = n_dev
    for bucket in BUCKET_PRIORITY:
        if remaining > 0 and buckets.get(bucket):
            quotas[bucket] = 1
            remaining -= 1
    while remaining > 0:
        ideal = {b: len(ids) * n_dev / total for b, ids in buckets.items()}
        open_buckets = [b for b in BUCKET_PRIORITY if b in buckets and quotas[b] < len(buckets[b])]
        if not open_buckets:
            break
        winner = max(open_buckets, key=lambda b: ideal[b] - quotas[b])
        quotas[winner] += 1
        remaining -= 1
    return quotas


def carve_stratified_dev_slice(
    item_ids: list[str] | tuple[str, ...],
    base_scores: dict[str, float],
    *,
    seed: int = DEFAULT_SEED,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    base_score_run: str | None = None,
) -> DevSlice:
    """Carve the dev slice proportionally per base-score bucket (answerable items guaranteed).

    Same guarantees as `carve_dev_slice` -- train/dev disjoint, deterministic for a seed, at
    least one item on each side -- plus: every non-empty bucket is represented (floor of one,
    answerable buckets first), so a small dev slice still carries items the base model can
    answer and the trial objective can discriminate. Refuses a population with NO answerable
    item: a study cannot rank trials against a constant objective.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(f"dev_fraction must lie strictly between 0 and 1, got {dev_fraction}")
    unique = sorted({str(item_id) for item_id in item_ids})
    if len(unique) < MIN_SLICE_ITEMS:
        raise ValueError(
            f"a held-out dev slice needs at least {MIN_SLICE_ITEMS} tuning items, got {len(unique)}"
        )
    if not any(base_scores.get(item_id, 0.0) > 0.0 for item_id in unique):
        raise ValueError(
            "stratified dev slice refused: the base model scores 0.0 on every tuning item, so "
            "no dev slice can discriminate between trials; grow or diversify the dataset "
            "(or drop --stratify-by-base-score to accept a constant objective knowingly)"
        )
    buckets: dict[str, list[str]] = {}
    for item_id in unique:
        buckets.setdefault(base_score_bucket(base_scores.get(item_id)), []).append(item_id)

    n_dev = min(len(unique) - 1, max(1, round(len(unique) * dev_fraction)))
    quotas = _dev_quota_per_bucket(buckets, n_dev, len(unique))
    rng = Random(seed)
    dev: set[str] = set()
    for bucket in BUCKET_PRIORITY:
        ids = list(buckets.get(bucket, []))
        rng.shuffle(ids)
        dev.update(ids[: quotas.get(bucket, 0)])

    strata: JsonObject = {
        bucket: {
            "population": len(ids),
            "dev": sum(1 for item_id in ids if item_id in dev),
            "mean_base_score": round(sum(base_scores.get(i, 0.0) for i in ids) / len(ids), 6),
        }
        for bucket, ids in sorted(buckets.items())
    }
    if base_score_run is not None:
        strata = {"base_score_run": base_score_run, "buckets": strata}
    return DevSlice(
        train_ids=tuple(sorted(set(unique) - dev)),
        dev_ids=tuple(sorted(dev)),
        seed=seed,
        dev_fraction=dev_fraction,
        strata=strata,
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


def adapter_param_estimate(params: JsonObject, *, hidden_size: int, n_layers: int) -> int:
    """Estimated LoRA parameter count: rank x targeted modules x layers x two `hidden x r` mats."""
    rank = int(params.get("lora_r") or 0)
    n_modules = len(params.get("target_modules") or [])
    return n_layers * n_modules * LORA_MATRICES_PER_MODULE * hidden_size * rank


def estimated_adapter_train_mib(params: JsonObject, *, hidden_size: int, n_layers: int) -> float:
    """The adapter's TRAINING footprint in MiB (weights + grads + optimizer states)."""
    from llb.backends.planner import MIB

    count = adapter_param_estimate(params, hidden_size=hidden_size, n_layers=n_layers)
    return count * ADAPTER_TRAIN_BYTES_PER_PARAM / MIB


def _cached_model_arch(model: str) -> JsonObject | None:
    """`hidden_size` / `n_layers` from the model's locally cached HF config, or None."""
    from llb.backends.planner import arch_from_config, cached_config_path

    path = cached_config_path(model)
    if path is None:
        return None
    try:
        return arch_from_config(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _default_estimate_fn(model: str, model_arch: JsonObject | None) -> EstimateFn:
    """Footprint estimator over the model's arch; returns None (never prunes) when unknown."""
    arch = model_arch if model_arch is not None else _cached_model_arch(model)

    def estimate(params: JsonObject) -> float | None:
        if not arch:
            return None
        hidden_size = arch.get("hidden_size")
        n_layers = arch.get("n_layers")
        if not isinstance(hidden_size, int) or not isinstance(n_layers, int):
            return None
        return estimated_adapter_train_mib(params, hidden_size=hidden_size, n_layers=n_layers)

    return estimate


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
    item_ids = [str(item_id) for item_id in dataset_manifest.get("item_ids") or []]
    if stratify_by_base_score is not None:
        dev_slice = carve_stratified_dev_slice(
            item_ids,
            load_base_scores(stratify_by_base_score),
            seed=seed,
            dev_fraction=dev_fraction,
            base_score_run=str(stratify_by_base_score),
        )
    else:
        dev_slice = carve_dev_slice(item_ids, seed=seed, dev_fraction=dev_fraction)

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
                    estimate_fn=_default_estimate_fn(model, model_arch),
                    vram_headroom_mib=vram_headroom_mib,
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
    estimate = trial.user_attrs.get("estimated_adapter_mib")
    return TrialRecord(
        number=int(trial.number),
        state=states.get(trial.state, STATE_FAILED),
        objective=float(trial.value) if trial.value is not None else None,
        hyperparameters=dict(trial.user_attrs.get("hyperparameters") or {}),
        duration_s=trial.duration.total_seconds() if trial.duration else 0.0,
        estimated_adapter_mib=float(estimate) if estimate is not None else None,
    )


def _append_trial(journal: Path, record: TrialRecord) -> None:
    """Live progress log; the manifest's trial table is rebuilt from the study, not from this."""
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
