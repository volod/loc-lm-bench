"""Quantization-aware model-weight estimates."""

from llb.backends.planner.constants import EMBED_BPW, MIB, PARTIAL_QUANT_FORMATS, QUANT_BPW
from llb.core.contracts import ModelSpec


def resolve_bpw(spec: ModelSpec) -> float | None:
    if spec.get("bpw") is not None:
        return float(spec["bpw"])
    return QUANT_BPW.get(str(spec.get("quant", "")).lower())


def weights_mib(params_b: float, bpw: float) -> float:
    """Estimate MiB when every weight uses one bits-per-weight value."""
    return params_b * 1e9 * bpw / 8 / MIB


def embedding_params(vocab_size: int, hidden_size: int, tied: bool = True) -> float:
    """Parameters in the token embedding and, when untied, output head."""
    return vocab_size * hidden_size * (1 if tied else 2)


def hi_precision_params(spec: ModelSpec) -> float:
    """Count weights that remain high precision under a partial quantization format."""
    override = spec.get("hi_precision_params_b")
    if override is not None:
        return max(0.0, float(override) * 1e9)
    if str(spec.get("quant", "")).lower() not in PARTIAL_QUANT_FORMATS:
        return 0.0
    vocab = spec.get("vocab_size")
    hidden = spec.get("hidden_size")
    if vocab and hidden:
        return embedding_params(
            int(vocab), int(hidden), bool(spec.get("tie_word_embeddings", True))
        )
    return 0.0


def weights_mib_detailed(
    params_b: float, quant_bpw: float, hi_params: float, embed_bpw: float = EMBED_BPW
) -> float:
    """Price high-precision mass separately from the quantized model body."""
    total = params_b * 1e9
    hi = min(max(0.0, hi_params), total)
    hi_bpw = max(embed_bpw, quant_bpw)
    body_bytes = (total - hi) * quant_bpw / 8
    hi_bytes = hi * hi_bpw / 8
    return (body_bytes + hi_bytes) / MIB
