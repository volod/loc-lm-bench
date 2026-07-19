"""Shared PEFT/TRL runtime helpers for adapter trainers."""

import json
from pathlib import Path
from typing import Any, cast

from llb.core.contracts.common import JsonObject
from llb.finetune.adapter_manifest import _has_native_quantization
from llb.finetune.dataset import load_dataset_manifest

DEFAULT_MAX_LENGTH = 1024


def require_finetune_stack() -> None:
    """Fail fast with an install action instead of a late import traceback."""
    try:
        import bitsandbytes  # noqa: F401
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "[finetune-adapter] install the finetune extra on the CUDA host: "
            'uv pip install -e ".[finetune]"'
        ) from exc


def load_sft_dataset(dataset_dir: Path | str) -> tuple[JsonObject, list[JsonObject]]:
    """Load the dataset manifest and require at least one SFT row."""
    dataset = load_dataset_manifest(dataset_dir)
    sft_rows = _read_sft_rows(Path(dataset_dir) / "sft.jsonl")
    if not sft_rows:
        raise SystemExit("[finetune-adapter] no SFT records found in dataset")
    return dataset, sft_rows


def load_tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    return tokenizer


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_quantized_base(model: str, params: JsonObject) -> Any:
    """Load a base model, applying QLoRA unless disabled or already quantized."""
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

    pretrained_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
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
    return base


def lora_config(params: JsonObject) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=int(params["lora_r"]),
        lora_alpha=int(params["lora_alpha"]),
        lora_dropout=float(params["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=params.get("target_modules"),
    )


def run_sft_training(
    *,
    peft_model: Any,
    tokenizer: Any,
    sft_rows: list[JsonObject],
    out: Path,
    seed: int,
    params: JsonObject,
) -> list[float]:
    """Run the shared TRL loop, persist the adapter, and return its loss curve."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    train_rows = [
        {"text": _format_chat(tokenizer, row["messages"], str(row["response"]))} for row in sft_rows
    ]
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
        max_length=int(params.get("max_length", DEFAULT_MAX_LENGTH)),
    )
    trainer = SFTTrainer(
        model=peft_model,
        args=args,
        train_dataset=Dataset.from_list(train_rows),
        processing_class=tokenizer,
    )
    trainer.train()
    cast(Any, trainer.model).save_pretrained(out)
    tokenizer.save_pretrained(out)
    return [
        float(row["loss"])
        for row in trainer.state.log_history
        if isinstance(row, dict) and row.get("loss") is not None
    ]


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
        rendered = tokenizer.apply_chat_template(
            [*chat, {"role": "assistant", "content": response}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return str(rendered)
    parts = [
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in chat
        if isinstance(message, dict)
    ]
    return "\n".join([*parts, f"assistant: {response}"])
