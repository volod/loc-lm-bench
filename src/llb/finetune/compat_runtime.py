"""Focused compat runtime implementation."""

import json
import logging
from pathlib import Path
from typing import Any
from llb.bench.common import new_run_timestamp
from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.finetune.naming import model_slug

_LOG = logging.getLogger(__name__)

COMPAT_METHOD = "finetune-compat"

COMPAT_REPORT = "compat_report.json"


def _attach_and_step(model_obj: Any, target_modules: list[str]) -> None:
    """Attach a minimal rank-4 LoRA and run one forward/backward micro-step."""
    import torch
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    peft_model = get_peft_model(model_obj, lora)
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA attached but produced no trainable parameters")
    device = trainable[0].device
    token_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    out = peft_model(input_ids=token_ids, labels=token_ids)
    out.loss.backward()


def _default_config_loader(model: str, *, local_only: bool = False) -> Any:
    try:
        from transformers import AutoConfig
    except ImportError:
        raise SystemExit(
            "[finetune-compat] install the finetune extra on the CUDA host: "
            'uv pip install -e ".[finetune]"'
        ) from None
    return AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=local_only)


def _default_model_loader(model: str) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, device_map="auto")


def _write_report(report: JsonObject, out_root: Path | str) -> JsonObject:
    out_dir = Path(out_root) / COMPAT_METHOD / model_slug(str(report["model"]))
    out_dir = out_dir / new_run_timestamp()[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / COMPAT_REPORT
    atomic_write_text(path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["report_path"] = str(path)
    _LOG.info(
        "[finetune-compat] %s -> %s%s",
        report["model"],
        report["verdict"],
        f" ({report['blocker']})" if report.get("blocker") else "",
    )
    return report
