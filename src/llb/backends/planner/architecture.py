"""Offline architecture discovery from cached Hugging Face configuration."""

import json
from pathlib import Path
from typing import Any, cast

from llb.core.contracts.models import ModelSpec

_INTEGER_ARCH_FIELDS = (
    ("vocab_size", "vocab_size"),
    ("hidden_size", "hidden_size"),
    ("num_hidden_layers", "n_layers"),
    ("max_position_embeddings", "max_context"),
    ("sliding_window", "sliding_window"),
    ("sliding_window_pattern", "sliding_window_pattern"),
)
_ATTENTION_LAYER_TYPES = {"full_attention", "sliding_attention"}


def _nested_text_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("text_config")
    return value if isinstance(value, dict) else {}


def _config_value(config: dict[str, Any], text: dict[str, Any], key: str) -> Any:
    return text.get(key, config.get(key))


def _copy_integer_arch_fields(
    out: dict[str, Any], config: dict[str, Any], text: dict[str, Any]
) -> None:
    for source, destination in _INTEGER_ARCH_FIELDS:
        value = _config_value(config, text, source)
        if isinstance(value, int) and not isinstance(value, bool):
            out[destination] = value


def _copy_tied_embeddings(
    out: dict[str, Any], config: dict[str, Any], text: dict[str, Any]
) -> None:
    value = config.get("tie_word_embeddings", text.get("tie_word_embeddings"))
    if isinstance(value, bool):
        out["tie_word_embeddings"] = value


def _copy_layer_architecture(
    out: dict[str, Any], config: dict[str, Any], text: dict[str, Any]
) -> None:
    layer_types = _config_value(config, text, "layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        return
    full_attention_layers = layer_types.count("full_attention")
    has_non_kv_attention = any(
        layer_type not in _ATTENTION_LAYER_TYPES for layer_type in layer_types
    )
    if full_attention_layers and has_non_kv_attention:
        out["kv_layers"] = full_attention_layers
    if "sliding_window_pattern" not in out and 0 < full_attention_layers < len(layer_types):
        out["sliding_window_pattern"] = max(2, len(layer_types) // full_attention_layers)


def _copy_kv_dimension(out: dict[str, Any], config: dict[str, Any], text: dict[str, Any]) -> None:
    n_kv_heads = _config_value(config, text, "num_key_value_heads")
    head_dim = _config_value(config, text, "head_dim")
    if isinstance(n_kv_heads, int) and isinstance(head_dim, int):
        out["kv_dim"] = n_kv_heads * head_dim


def arch_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract planning fields from a Hugging Face config, including nested text config."""
    text = _nested_text_config(config)
    out: dict[str, Any] = {}
    _copy_integer_arch_fields(out, config, text)
    _copy_tied_embeddings(out, config, text)
    _copy_layer_architecture(out, config, text)
    _copy_kv_dimension(out, config, text)
    return out


def cached_config_path(repo_id: str) -> Path | None:
    """Return a cached config path without downloading the repository."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return None
    try:
        hit = try_to_load_from_cache(repo_id, "config.json")
    except Exception:
        return None
    return Path(hit) if isinstance(hit, str) and Path(hit).is_file() else None


def enrich_arch(spec: ModelSpec, *, override: bool = False) -> ModelSpec:
    """Fill or replace planning fields from a locally cached model configuration."""
    source = spec.get("source", "")
    if not source or source.count("/") != 1 or source.startswith("hf.co/"):
        return spec
    path = cached_config_path(source)
    if path is None:
        return spec
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return spec
    if not isinstance(config, dict):
        return spec
    merged: dict[str, Any] = dict(spec)
    for key, value in arch_from_config(config).items():
        if override or merged.get(key) is None:
            merged[key] = value
    return cast(ModelSpec, merged)
