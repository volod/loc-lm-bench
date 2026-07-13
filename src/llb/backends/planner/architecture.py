"""Offline architecture discovery from cached Hugging Face configuration."""

import json
from pathlib import Path
from typing import Any, cast

from llb.core.contracts import ModelSpec


def arch_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract planning fields from a Hugging Face config, including nested text config."""
    text = config["text_config"] if isinstance(config.get("text_config"), dict) else {}
    out: dict[str, Any] = {}
    for key, dest in (
        ("vocab_size", "vocab_size"),
        ("hidden_size", "hidden_size"),
        ("num_hidden_layers", "n_layers"),
        ("sliding_window", "sliding_window"),
        ("sliding_window_pattern", "sliding_window_pattern"),
    ):
        value = text.get(key, config.get(key))
        if isinstance(value, int) and not isinstance(value, bool):
            out[dest] = value
    tie = config.get("tie_word_embeddings", text.get("tie_word_embeddings"))
    if isinstance(tie, bool):
        out["tie_word_embeddings"] = tie
    layer_types = text.get("layer_types", config.get("layer_types"))
    if "sliding_window_pattern" not in out and isinstance(layer_types, list) and layer_types:
        full = sum(1 for layer_type in layer_types if layer_type == "full_attention")
        if 0 < full < len(layer_types):
            out["sliding_window_pattern"] = max(2, len(layer_types) // full)
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
    wanted = ("vocab_size", "hidden_size", "n_layers", "tie_word_embeddings")
    if not override and all(spec.get(key) is not None for key in wanted):
        return spec
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
