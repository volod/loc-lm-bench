"""Adapter training seam for local self-improvement.

The fake trainer writes a complete adapter manifest for CI. The real trainer is deliberately
lazy-imported behind `.[finetune]` so base installs do not pull CUDA training stacks.
"""

import json
from pathlib import Path
from typing import Any, Callable, cast

from llb.core.contracts.common import JsonObject
from llb.finetune.dataset import DATASET_MANIFEST, load_dataset_manifest
from llb.finetune.adapter_manifest import (
    _adapter_manifest,
    _default_hyperparameters,
    _has_native_quantization,
    _write_manifest,
    adapter_digest,
)

# PEFT writes this beside the adapter weights; it is the authoritative record of the trained rank.
# Digest prefix length used everywhere an adapter is named short: labels, registry rows, merged
# artifact directories, and Ollama tags.

TrainerFn = Callable[..., JsonObject]


def train_adapter(
    *,
    dataset_dir: Path | str,
    model: str,
    out_dir: Path | str,
    seed: int = 13,
    trainer: str = "auto",
    hyperparameters: JsonObject | None = None,
    hparams_manifest: Path | str | None = None,
) -> JsonObject:
    """Train or fake-train a LoRA adapter and write `adapter_manifest.json`.

    `hparams_manifest` is pure provenance: the path of the `finetune-hparams` study whose best
    config was passed in as `hyperparameters`. It is recorded, never re-read, and never enters
    `adapter_digest` -- two adapters with identical hyperparameters are the same adapter whether or
    not a search chose them.
    """
    if trainer == "fake":
        return fake_train_adapter(
            dataset_dir=dataset_dir,
            model=model,
            out_dir=out_dir,
            seed=seed,
            hyperparameters=hyperparameters,
            hparams_manifest=hparams_manifest,
        )
    return real_train_adapter(
        dataset_dir=dataset_dir,
        model=model,
        out_dir=out_dir,
        seed=seed,
        hyperparameters=hyperparameters,
        hparams_manifest=hparams_manifest,
    )


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
    digest = adapter_digest(model, str(dataset["dataset_digest"]), seed, params)
    (out / "adapter.fake").write_text(f"adapter_digest={digest}\n", encoding="utf-8")
    manifest = _adapter_manifest(
        model=model,
        dataset=dataset,
        dataset_manifest_path=Path(dataset_dir) / DATASET_MANIFEST,
        seed=seed,
        hyperparameters=params,
        adapter_digest=digest,
        trainer="fake",
        loss_curve=[1.0, 0.5],
        hparams_manifest=hparams_manifest,
    )
    _write_manifest(out, manifest)
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
    """Real LoRA/QLoRA training entrypoint.

    The dependency check is explicit so operators get a clear install action instead of a late
    import traceback. Hyperparameters are intentionally conservative defaults; operators can inject
    a richer site trainer through `run_self_improve` without changing manifests or guards.
    """
    try:
        import bitsandbytes  # noqa: F401  (the default 4-bit load path needs it at model load)
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "[finetune-adapter] install the finetune extra on the CUDA host: "
            'uv pip install -e ".[finetune]"'
        ) from exc
    dataset = load_dataset_manifest(dataset_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = _default_hyperparameters(hyperparameters)
    sft_rows = _read_sft_rows(Path(dataset_dir) / "sft.jsonl")
    if not sft_rows:
        raise SystemExit("[finetune-adapter] no SFT records found in dataset")

    pretrained_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = [
        {"text": _format_chat(tokenizer, row["messages"], str(row["response"]))} for row in sft_rows
    ]
    hf_dataset = Dataset.from_list(train_rows)
    quantization_config = None
    if bool(params.get("load_in_4bit", True)) and not _has_native_quantization(pretrained_config):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(params.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=getattr(torch, str(params.get("compute_dtype", "bfloat16"))),
        )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": params.get("device_map", "auto"),
        "config": pretrained_config,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    base = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    if quantization_config is not None:
        base = prepare_model_for_kbit_training(base)
    lora = LoraConfig(
        r=int(params["lora_r"]),
        lora_alpha=int(params["lora_alpha"]),
        lora_dropout=float(params["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=params.get("target_modules"),
    )
    # `get_peft_model` is typed `PeftModel | PeftMixedModel`, and only `mixed=True` yields the
    # latter, which SFTTrainer does not accept. The peft/trl stubs cannot express that.
    peft_model = cast(Any, get_peft_model(base, lora))
    args = SFTConfig(
        output_dir=str(out / "trainer"),
        seed=seed,
        per_device_train_batch_size=int(params.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(params.get("gradient_accumulation_steps", 4)),
        learning_rate=float(params.get("learning_rate", 2e-4)),
        num_train_epochs=float(params.get("num_train_epochs", 1.0)),
        max_steps=int(params.get("max_steps", -1)),
        logging_steps=int(params.get("logging_steps", 1)),
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_length=int(params.get("max_length", 1024)),
    )
    trainer = SFTTrainer(
        model=peft_model,
        args=args,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    # Save what was actually trained: SFTTrainer may re-wrap the model during `accelerate` prepare,
    # so `trainer.model` is the adapter to persist, not `peft_model`. Its stub type is `Module | None`.
    cast(Any, trainer.model).save_pretrained(out)
    tokenizer.save_pretrained(out)
    loss_curve = [
        float(row["loss"])
        for row in trainer.state.log_history
        if isinstance(row, dict) and row.get("loss") is not None
    ]
    digest = adapter_digest(model, str(dataset["dataset_digest"]), seed, params)
    manifest = _adapter_manifest(
        model=model,
        dataset=dataset,
        dataset_manifest_path=Path(dataset_dir) / DATASET_MANIFEST,
        seed=seed,
        hyperparameters=params,
        adapter_digest=digest,
        trainer="peft-trl",
        loss_curve=loss_curve,
        hparams_manifest=hparams_manifest,
    )
    _write_manifest(out, manifest)
    return manifest


def _read_sft_rows(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _format_chat(tokenizer: Any, messages: object, response: str) -> str:
    chat = list(messages) if isinstance(messages, list) else []
    if hasattr(tokenizer, "apply_chat_template"):
        train_chat = [*chat, {"role": "assistant", "content": response}]
        rendered = tokenizer.apply_chat_template(
            train_chat,
            tokenize=False,
            add_generation_prompt=False,
        )
        return str(rendered)
    parts = []
    for message in chat:
        if isinstance(message, dict):
            parts.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
    parts.append(f"assistant: {response}")
    return "\n".join(parts)
