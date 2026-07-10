"""Compressed-QAT checkpoint trainability probes (compressed-qat-adapter-support).

Compressed-tensors QAT checkpoints (e.g. `*-qat-w4a16-ct`) serve well on vLLM but their linear
layers are NOT ordinary `nn.Linear`: PEFT can only inject LoRA into layer types it has a dispatch
for (full-precision, bitsandbytes 4/8-bit, GPTQ, AWQ, EETQ, HQQ). A campaign that only discovers
this after loading a 10 GB checkpoint has already paid for the load and then crashes mid-campaign.

This module answers "can this checkpoint take an adapter on this host?" in two stages:

- `inspect_quantization` + `assess_quantization`: config-only introspection (no weights, no CUDA)
  that classifies the checkpoint's native quantization scheme against PEFT's dispatch table and
  names the exact blocker for an unsupported scheme. The campaign runner uses this stage to skip
  a doomed model BEFORE its base eval or training run.
- `probe_trainability`: the heavy CUDA-host probe behind `llb finetune-compat`: load the model,
  scan its actual linear module classes, select per-architecture target modules from the modules
  that EXIST (never assume llama naming), attach a rank-4 LoRA, and run one forward/backward
  micro-step. The verdict plus every stage's evidence lands in
  `$DATA_DIR/finetune-compat/<model>/<timestamp>/compat_report.json`.

Everything except the real loader is pure and unit-tested with fake modules/configs; the loader
and trainer stacks are injectable seams.
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from llb.bench.common import new_run_timestamp
from llb.core.contracts import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.finetune.naming import model_slug

_LOG = logging.getLogger(__name__)

COMPAT_METHOD = "finetune-compat"
COMPAT_REPORT = "compat_report.json"

VERDICT_TRAINABLE = "trainable"
VERDICT_NOT_TRAINABLE = "not-trainable"
VERDICT_UNKNOWN = "unknown"

# Native quantization schemes PEFT has a LoRA layer dispatch for, mapped to the injection
# strategy the trainer should use. A scheme absent here (notably `compressed-tensors`) has no
# dispatch: `get_peft_model` either raises or silently wraps nothing trainable.
PEFT_SUPPORTED_QUANT_METHODS: dict[str, str] = {
    "bitsandbytes": "peft-bnb",
    "gptq": "peft-gptq",
    "awq": "peft-awq",
    "eetq": "peft-eetq",
    "hqq": "peft-hqq",
}

# Linear-layer CLASS names PEFT can wrap (matched by name so introspection never imports the
# quantization backends themselves).
PEFT_SUPPORTED_LINEAR_CLASSES = frozenset(
    {
        "Linear",
        "Linear4bit",
        "Linear8bitLt",
        "QuantLinear",
        "WQLinear_GEMM",
        "EetqLinear",
        "HQQLinear",
    }
)

# The documented fallback recorded beside every unsupported-scheme blocker.
FALLBACK_NOTE = (
    "train the adapter on the uncompressed base checkpoint and serve it merged/quantized, or "
    "pick the bitsandbytes-quantized path (load_in_4bit over the base weights)"
)

# Module-name suffixes that are never adapter targets (output head / embeddings).
_NON_TARGET_SUFFIXES = frozenset({"lm_head", "embed_tokens", "embed_out", "wte", "wpe"})

# Known attention/MLP projection suffix vocabulary, tried in preference order. Selection is
# grounded in the modules that actually exist in the loaded model, so a non-llama naming scheme
# (gpt2 `c_attn`, falcon `query_key_value`) still yields an attachable set.
_KNOWN_TARGET_SUFFIXES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
    "c_attn",
    "c_proj",
    "c_fc",
    "qkv_proj",
    "out_proj",
    "fc1",
    "fc2",
)

# (model id, RunConfig-ish) -> compat payload; the campaign's injectable probe seam.
CompatFn = Callable[[str], JsonObject]


@dataclass(frozen=True)
class QuantizationInfo:
    """The checkpoint's native quantization scheme, normalized from its pretrained config."""

    quant_method: str | None
    details: JsonObject

    @property
    def is_native(self) -> bool:
        return self.quant_method is not None


def inspect_quantization(config_like: Any) -> QuantizationInfo:
    """Normalize `quantization_config` from a pretrained config object OR a plain config dict."""
    if isinstance(config_like, dict):
        quant = config_like.get("quantization_config")
    else:
        quant = getattr(config_like, "quantization_config", None)
    if not quant:
        return QuantizationInfo(None, {})
    if isinstance(quant, dict):
        details: JsonObject = dict(quant)
    else:  # transformers QuantizationConfigMixin
        to_dict = getattr(quant, "to_dict", None)
        details = dict(to_dict()) if callable(to_dict) else {"repr": repr(quant)}
    method = details.get("quant_method")
    return QuantizationInfo(str(method) if method else "unknown", details)


def assess_quantization(quant: QuantizationInfo) -> tuple[str, str | None, str | None]:
    """(verdict, injection strategy, blocker) from config-only evidence.

    A full-precision checkpoint and every PEFT-dispatched scheme are `trainable`; an unsupported
    native scheme (compressed-tensors and friends) is `not-trainable` with the exact blocker and
    the documented fallback. An unrecognizable scheme stays `unknown` so the heavy probe decides.
    """
    if not quant.is_native:
        return VERDICT_TRAINABLE, "peft-lora", None
    method = (quant.quant_method or "").lower()
    strategy = PEFT_SUPPORTED_QUANT_METHODS.get(method)
    if strategy is not None:
        return VERDICT_TRAINABLE, strategy, None
    if method in {"compressed-tensors", "compressed_tensors"}:
        return (
            VERDICT_NOT_TRAINABLE,
            None,
            "native quant_method 'compressed-tensors' has no PEFT LoRA dispatch "
            f"(CompressedLinear layers cannot take adapters); fallback: {FALLBACK_NOTE}",
        )
    return (
        VERDICT_UNKNOWN,
        None,
        f"native quant_method {quant.quant_method!r} is not in PEFT's dispatch table; "
        "run the heavy probe (llb finetune-compat) to decide",
    )


def linear_class_summary(named_modules: Iterable[tuple[str, Any]]) -> dict[str, int]:
    """Count leaf linear-like modules by class name (any class whose name contains 'Linear')."""
    counts: Counter[str] = Counter()
    for _name, module in named_modules:
        cls = type(module).__name__
        if "Linear" in cls:
            counts[cls] += 1
    return dict(counts)


def unsupported_linear_classes(summary: dict[str, int]) -> list[str]:
    return sorted(cls for cls in summary if cls not in PEFT_SUPPORTED_LINEAR_CLASSES)


def select_target_modules(named_modules: Iterable[tuple[str, Any]]) -> list[str]:
    """Per-architecture target modules: known projection suffixes that EXIST in this model.

    Falls back to the most frequent non-head linear suffixes when the known vocabulary matches
    nothing, so an exotic architecture still yields an attachable (if unnamed) set.
    """
    suffix_counts: Counter[str] = Counter()
    for name, module in named_modules:
        if "Linear" not in type(module).__name__:
            continue
        suffix = name.rsplit(".", 1)[-1]
        if suffix and suffix not in _NON_TARGET_SUFFIXES and not re.fullmatch(r"\d+", suffix):
            suffix_counts[suffix] += 1
    known = [suffix for suffix in _KNOWN_TARGET_SUFFIXES if suffix in suffix_counts]
    if known:
        return known
    return [suffix for suffix, _count in suffix_counts.most_common(4)]


def probe_trainability(
    model: str,
    *,
    out_root: Path | str,
    loader: Callable[[str], Any] | None = None,
    config_loader: Callable[[str], Any] | None = None,
    attach_fn: Callable[[Any, list[str]], None] | None = None,
) -> JsonObject:
    """The staged trainability probe; writes and returns the compat report.

    Stage 1 (config) always runs. Stage 2 (load + module scan + LoRA attach + one
    forward/backward micro-step) runs unless stage 1 already proves the checkpoint
    not-trainable. Any stage-2 exception becomes the blocker, never a crash.
    """
    report: JsonObject = {
        "kind": "llb.finetune.compat",
        "model": model,
        "verdict": VERDICT_UNKNOWN,
        "injection_strategy": None,
        "blocker": None,
        "quantization": {},
        "linear_classes": {},
        "target_modules": [],
        "created_at": new_run_timestamp()[1],
    }
    config_loader = config_loader or _default_config_loader
    try:
        quant = inspect_quantization(config_loader(model))
    except Exception as exc:  # unreadable config is itself the blocker
        report["verdict"] = VERDICT_NOT_TRAINABLE
        report["blocker"] = f"cannot read pretrained config: {exc}"
        return _write_report(report, out_root)
    report["quantization"] = {"quant_method": quant.quant_method, **quant.details}
    verdict, strategy, blocker = assess_quantization(quant)
    report["injection_strategy"] = strategy
    report["blocker"] = blocker
    if verdict == VERDICT_NOT_TRAINABLE:
        report["verdict"] = verdict
        return _write_report(report, out_root)

    loader = loader or _default_model_loader
    try:
        loaded = loader(model)
        modules = list(loaded.named_modules())
        summary = linear_class_summary(modules)
        report["linear_classes"] = summary
        unsupported = unsupported_linear_classes(summary)
        targets = select_target_modules(modules)
        report["target_modules"] = targets
        if unsupported:
            report["verdict"] = VERDICT_NOT_TRAINABLE
            report["blocker"] = (
                "linear classes without a PEFT LoRA dispatch: "
                + ", ".join(unsupported)
                + f"; fallback: {FALLBACK_NOTE}"
            )
            return _write_report(report, out_root)
        if not targets:
            report["verdict"] = VERDICT_NOT_TRAINABLE
            report["blocker"] = "no adapter-targetable linear modules found"
            return _write_report(report, out_root)
        (attach_fn or _attach_and_step)(loaded, targets)
    except SystemExit:
        raise
    except Exception as exc:
        report["verdict"] = VERDICT_NOT_TRAINABLE
        report["blocker"] = f"{type(exc).__name__}: {exc}"
        return _write_report(report, out_root)
    report["verdict"] = VERDICT_TRAINABLE
    report["injection_strategy"] = strategy or "peft-lora"
    report["blocker"] = None
    return _write_report(report, out_root)


def config_compat_probe(model: str, *, local_only: bool = True) -> JsonObject:
    """The campaign's cheap pre-training probe: config-only, no weights, no CUDA.

    Returns an UNKNOWN verdict (never a false skip) when transformers or the config itself is
    unavailable -- the campaign only skips on a POSITIVE not-trainable verdict. `local_only`
    (the default) reads only an already-cached config, so the probe never touches the network
    for Ollama tags or not-yet-downloaded models.
    """
    try:
        quant = inspect_quantization(_default_config_loader(model, local_only=local_only))
    except (Exception, SystemExit) as exc:
        return {"verdict": VERDICT_UNKNOWN, "blocker": f"config unavailable: {exc}"}
    verdict, strategy, blocker = assess_quantization(quant)
    return {
        "verdict": verdict,
        "injection_strategy": strategy,
        "blocker": blocker,
        "quant_method": quant.quant_method,
    }


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
