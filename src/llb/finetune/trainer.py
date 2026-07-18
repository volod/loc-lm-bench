"""Adapter training seam with lazy optional CUDA backends."""

from pathlib import Path
from typing import Any, Callable, cast

from llb.core.contracts.common import JsonObject
from llb.finetune.dataset import DATASET_MANIFEST, load_dataset_manifest
from llb.finetune.adapter_manifest import (
    _adapter_manifest,
    _default_hyperparameters,
    _write_manifest,
    adapter_digest,
)
from llb.finetune.training_runtime import (
    DEFAULT_MAX_LENGTH,
    ensure_pad_token,
    load_quantized_base,
    load_sft_dataset,
    load_tokenizer,
    lora_config,
    require_finetune_stack,
    run_sft_training,
)

TrainerFn = Callable[..., JsonObject]

# `--trainer` values accepted by the seam. "auto" and "peft-trl" both select the PEFT/TRL path;
# the manifest always records the concrete trainer that ran, never "auto".
TRAINER_AUTO = "auto"
TRAINER_PEFT_TRL = "peft-trl"
TRAINER_UNSLOTH = "unsloth"
TRAINER_FAKE = "fake"
KNOWN_TRAINERS = (TRAINER_AUTO, TRAINER_PEFT_TRL, TRAINER_UNSLOTH, TRAINER_FAKE)


def train_adapter(
    *,
    dataset_dir: Path | str,
    model: str,
    out_dir: Path | str,
    seed: int = 13,
    trainer: str = TRAINER_AUTO,
    hyperparameters: JsonObject | None = None,
    hparams_manifest: Path | str | None = None,
) -> JsonObject:
    """Train or fake-train a LoRA adapter and write `adapter_manifest.json`.

    `hparams_manifest` is pure provenance: the path of the `finetune-hparams` study whose best
    config was passed in as `hyperparameters`. It is recorded, never re-read, and never enters
    `adapter_digest` -- two adapters with identical hyperparameters are the same adapter whether or
    not a search chose them.
    """
    if trainer not in KNOWN_TRAINERS:
        raise SystemExit(
            f"[finetune-adapter] unknown --trainer {trainer!r}; expected one of "
            + " | ".join(KNOWN_TRAINERS)
        )
    kwargs: dict[str, Any] = {
        "dataset_dir": dataset_dir,
        "model": model,
        "out_dir": out_dir,
        "seed": seed,
        "hyperparameters": hyperparameters,
        "hparams_manifest": hparams_manifest,
    }
    if trainer == TRAINER_FAKE:
        return fake_train_adapter(**kwargs)
    if trainer == TRAINER_UNSLOTH:
        return unsloth_train_adapter(**kwargs)
    return real_train_adapter(**kwargs)


def fake_train_adapter(
    *,
    dataset_dir: Path | str,
    model: str,
    out_dir: Path | str,
    seed: int = 13,
    hyperparameters: JsonObject | None = None,
    hparams_manifest: Path | str | None = None,
) -> JsonObject:
    """CI trainer: deterministic manifest + tiny marker file, no CUDA dependency."""
    dataset = load_dataset_manifest(dataset_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = _default_hyperparameters(hyperparameters)
    manifest = _finalize_adapter(
        model=model,
        dataset=dataset,
        dataset_dir=dataset_dir,
        seed=seed,
        params=params,
        trainer=TRAINER_FAKE,
        loss_curve=[1.0, 0.5],
        hparams_manifest=hparams_manifest,
        out=out,
    )
    (out / "adapter.fake").write_text(
        f"adapter_digest={manifest['adapter_digest']}\n", encoding="utf-8"
    )
    return manifest


def real_train_adapter(
    *,
    dataset_dir: Path | str,
    model: str,
    out_dir: Path | str,
    seed: int = 13,
    hyperparameters: JsonObject | None = None,
    hparams_manifest: Path | str | None = None,
) -> JsonObject:
    """Real LoRA/QLoRA training entrypoint (PEFT/TRL).

    The dependency check is explicit so operators get a clear install action instead of a late
    import traceback. Hyperparameters are intentionally conservative defaults; operators can inject
    a richer site trainer through `run_self_improve` without changing manifests or guards.
    """
    require_finetune_stack()
    from peft import get_peft_model

    dataset, sft_rows = load_sft_dataset(dataset_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = _default_hyperparameters(hyperparameters)
    tokenizer = load_tokenizer(model)
    base = load_quantized_base(model, params)
    lora = lora_config(params)
    # `get_peft_model` is typed `PeftModel | PeftMixedModel`, and only `mixed=True` yields the
    # latter, which SFTTrainer does not accept. The peft/trl stubs cannot express that.
    peft_model = cast(Any, get_peft_model(base, lora))
    loss_curve = run_sft_training(
        peft_model=peft_model,
        tokenizer=tokenizer,
        sft_rows=sft_rows,
        out=out,
        seed=seed,
        params=params,
    )
    return _finalize_adapter(
        model=model,
        dataset=dataset,
        dataset_dir=dataset_dir,
        seed=seed,
        params=params,
        trainer=TRAINER_PEFT_TRL,
        loss_curve=loss_curve,
        hparams_manifest=hparams_manifest,
        out=out,
    )


def unsloth_train_adapter(
    *,
    dataset_dir: Path | str,
    model: str,
    out_dir: Path | str,
    seed: int = 13,
    hyperparameters: JsonObject | None = None,
    hparams_manifest: Path | str | None = None,
) -> JsonObject:
    """Unsloth-accelerated LoRA/QLoRA training path (single CUDA GPU).

    Unsloth patches transformers/TRL at import time, so it MUST be imported before the rest of the
    training stack -- keep its import first in this function. The package is intentionally NOT a
    project extra (like marker, it pins a hardware-matched torch/triton stack); install it in the
    CUDA training environment when selecting `--trainer unsloth`.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "[finetune-adapter] trainer=unsloth needs the unsloth package on the CUDA host: "
            "uv pip install unsloth"
        ) from exc
    require_finetune_stack()
    dataset, sft_rows = load_sft_dataset(dataset_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = _default_hyperparameters(hyperparameters)
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=int(params.get("max_length", DEFAULT_MAX_LENGTH)),
        load_in_4bit=bool(params.get("load_in_4bit", True)),
    )
    ensure_pad_token(tokenizer)
    peft_model = FastLanguageModel.get_peft_model(
        base,
        r=int(params["lora_r"]),
        lora_alpha=int(params["lora_alpha"]),
        lora_dropout=float(params["lora_dropout"]),
        bias="none",
        target_modules=params.get("target_modules"),
        random_state=seed,
        use_gradient_checkpointing="unsloth",
    )
    loss_curve = run_sft_training(
        peft_model=peft_model,
        tokenizer=tokenizer,
        sft_rows=sft_rows,
        out=out,
        seed=seed,
        params=params,
    )
    return _finalize_adapter(
        model=model,
        dataset=dataset,
        dataset_dir=dataset_dir,
        seed=seed,
        params=params,
        trainer=TRAINER_UNSLOTH,
        loss_curve=loss_curve,
        hparams_manifest=hparams_manifest,
        out=out,
    )


def _finalize_adapter(
    *,
    model: str,
    dataset: JsonObject,
    dataset_dir: Path | str,
    seed: int,
    params: JsonObject,
    trainer: str,
    loss_curve: list[float],
    hparams_manifest: Path | str | None,
    out: Path,
) -> JsonObject:
    """Compute the adapter digest, write `adapter_manifest.json`, and return the manifest."""
    digest = adapter_digest(model, str(dataset["dataset_digest"]), seed, params)
    manifest = _adapter_manifest(
        model=model,
        dataset=dataset,
        dataset_manifest_path=Path(dataset_dir) / DATASET_MANIFEST,
        seed=seed,
        hyperparameters=params,
        adapter_digest=digest,
        trainer=trainer,
        loss_curve=loss_curve,
        hparams_manifest=hparams_manifest,
    )
    _write_manifest(out, manifest)
    return manifest
