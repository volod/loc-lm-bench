"""Focused adapter manifest implementation."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from llb.core.contracts import JsonObject
from llb.core.fsutil import atomic_write_text

ADAPTER_MANIFEST = "adapter_manifest.json"

PEFT_ADAPTER_CONFIG = "adapter_config.json"

ADAPTER_DIGEST_SHORT_CHARS = 12

DEFAULT_LORA_R = 16

DEFAULT_LORA_ALPHA = 32

DEFAULT_LORA_DROPOUT = 0.05

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def adapter_label(model: str, adapter_digest: str) -> str:
    return f"{model}+adapter-{adapter_digest[:ADAPTER_DIGEST_SHORT_CHARS]}"


def adapter_digest(
    model: str, dataset_digest_value: str, seed: int, hyperparameters: JsonObject
) -> str:
    payload = {
        "model": model,
        "dataset_digest": dataset_digest_value,
        "seed": seed,
        "hyperparameters": hyperparameters,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_adapter_manifest(adapter_dir: Path | str) -> JsonObject:
    path = Path(adapter_dir) / ADAPTER_MANIFEST
    if not path.is_file():
        raise ValueError(f"adapter manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"adapter manifest must be a JSON object: {path}")
    return data


def adapter_lora_rank(adapter_dir: Path | str | None) -> int | None:
    """The LoRA rank a serving backend must be sized for, or None when it cannot be determined.

    PEFT's own `adapter_config.json` wins: it describes the weights on disk and exists for adapters
    this project did not train. The `adapter_manifest.json` hyperparameters are the fallback (the
    fake trainer writes no PEFT config).
    """
    if adapter_dir is None:
        return None
    peft_config = Path(adapter_dir) / PEFT_ADAPTER_CONFIG
    if peft_config.is_file():
        rank = _read_json_key(peft_config, "r")
        if rank is not None:
            return rank
    try:
        hyperparameters = load_adapter_manifest(adapter_dir).get("hyperparameters") or {}
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return _as_positive_int(hyperparameters.get("lora_r"))


def _read_json_key(path: Path, key: str) -> int | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _as_positive_int(data.get(key)) if isinstance(data, dict) else None


def _as_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def _has_native_quantization(config: Any) -> bool:
    """True when the checkpoint config already declares its own quantization scheme."""
    quantization = getattr(config, "quantization_config", None)
    return bool(quantization)


def _adapter_manifest(
    *,
    model: str,
    dataset: JsonObject,
    dataset_manifest_path: Path,
    seed: int,
    hyperparameters: JsonObject,
    adapter_digest: str,
    trainer: str,
    loss_curve: list[float],
    hparams_manifest: Path | str | None = None,
) -> JsonObject:
    return {
        "kind": "llb.finetune.adapter",
        "base_model": model,
        "adapter_digest": adapter_digest,
        "adapter_label": adapter_label(model, adapter_digest),
        "dataset_digest": dataset["dataset_digest"],
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_item_ids": list(dataset.get("item_ids") or []),
        "dataset_split_counts": dict(dataset.get("split_counts") or {}),
        "seed": seed,
        "hyperparameters": hyperparameters,
        "hparams_manifest": str(hparams_manifest) if hparams_manifest else None,
        "trainer": trainer,
        "loss_curve": loss_curve,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _default_hyperparameters(overrides: JsonObject | None) -> JsonObject:
    params: JsonObject = {
        "method": "lora",
        "lora_r": DEFAULT_LORA_R,
        "lora_alpha": DEFAULT_LORA_ALPHA,
        "lora_dropout": DEFAULT_LORA_DROPOUT,
        "target_modules": DEFAULT_TARGET_MODULES,
    }
    if overrides:
        params.update(overrides)
    return params


def _write_manifest(out_dir: Path, manifest: JsonObject) -> None:
    atomic_write_text(
        out_dir / ADAPTER_MANIFEST,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
