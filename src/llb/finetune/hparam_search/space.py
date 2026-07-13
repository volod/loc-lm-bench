"""The LoRA search space and its guards: sample one configuration (`suggest_lora_hyperparameters`),
refuse a dataset that leaks a protected split (`assert_tuning_only`), and estimate a trial's
training footprint for the pre-run infeasibility prune.

The sampled keys are exactly what `train_adapter` consumes; the estimator returns None (never
prunes) when the model's architecture is unknown.
"""

import json
from pathlib import Path
from typing import Any

from llb.core.contracts import JsonObject
from llb.finetune.dataset import TUNING_SPLIT
from llb.finetune.guard import PROTECTED_SPLITS
from llb.finetune.hparam_search.model import (
    ADAPTER_TRAIN_BYTES_PER_PARAM,
    BATCH_GEOMETRY_CHOICES,
    EPOCHS_RANGE,
    LEARNING_RATE_RANGE,
    LORA_ALPHA_MULTIPLIERS,
    LORA_DROPOUT_RANGE,
    LORA_DROPOUT_STEP,
    LORA_MATRICES_PER_MODULE,
    LORA_R_CHOICES,
    MAX_LENGTH_CHOICES,
    TARGET_MODULE_PRESETS,
    EstimateFn,
)
from llb.goldset.schema import load_goldset


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
