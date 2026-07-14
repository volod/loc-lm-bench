"""Focused compat quantization implementation."""

from dataclasses import dataclass
from typing import Any
from llb.core.contracts import JsonObject

VERDICT_TRAINABLE = "trainable"

VERDICT_NOT_TRAINABLE = "not-trainable"

VERDICT_UNKNOWN = "unknown"

PEFT_SUPPORTED_QUANT_METHODS: dict[str, str] = {
    "bitsandbytes": "peft-bnb",
    "gptq": "peft-gptq",
    "awq": "peft-awq",
    "eetq": "peft-eetq",
    "hqq": "peft-hqq",
}

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

FALLBACK_NOTE = (
    "train the adapter on the uncompressed base checkpoint and serve it merged/quantized, or "
    "pick the bitsandbytes-quantized path (load_in_4bit over the base weights)"
)


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
