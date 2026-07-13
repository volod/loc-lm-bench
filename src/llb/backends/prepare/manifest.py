"""Load candidate-model preparation targets: validate an external candidate manifest (YAML) or read
a generated serving `tier.json` into concrete `ModelSpec` prep targets.
"""

import json
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from llb.core.contracts import ModelSpec


class _ModelSpecSchema(BaseModel):
    """Validation model for one external candidate-manifest entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    backend: str
    source: str
    min_vram_gb: int | float = 0
    notes: str | None = None
    license_url: str | None = None
    gated: bool = False
    params_b: float | None = None
    quant: str | None = None
    bpw: float | None = None
    n_layers: int | None = None
    kv_dim: int | None = None
    max_context: int | None = None
    vocab_size: int | None = None
    hidden_size: int | None = None
    tie_word_embeddings: bool | None = None
    embed_bpw: float | None = None
    hi_precision_params_b: float | None = None
    sources: dict[str, "str | dict[str, object] | list[str | dict[str, object]]"] | None = None


def load_manifest(path: Path | str) -> list[ModelSpec]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML -- {exc}") from None
    models = data.get("models") if isinstance(data, dict) else None
    if not models:
        raise ValueError(f"{path}: expected a top-level 'models:' list")
    for model in models:
        if not isinstance(model, dict):
            raise ValueError(f"{path}: each model entry must be a mapping, got: {model!r}")
    try:
        validated = [_ModelSpecSchema.model_validate(model) for model in models]
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid model entry -- {exc}") from None
    return [cast(ModelSpec, model.model_dump(exclude_none=True)) for model in validated]


def load_serving_targets(path: Path | str) -> list[ModelSpec]:
    """Read a generated serving tier.json as concrete preparation targets."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON -- {exc}") from None
    targets = data.get("targets") if isinstance(data, dict) else None
    if not isinstance(targets, list):
        raise ValueError(f"{path}: expected a top-level 'targets' list")

    models: list[ModelSpec] = []
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError(f"{path}: each target entry must be a mapping, got: {target!r}")
        target_id = target.get("target")
        backend = target.get("backend")
        source = target.get("model")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError(f"{path}: serving target is missing a non-empty target id")
        if not isinstance(backend, str) or not isinstance(source, str) or not source:
            raise ValueError(f"{path}: target {target_id!r} must include backend and model")
        models.append(
            {
                "name": f"serving-{target_id}",
                "backend": backend,
                "source": source,
                "min_vram_gb": 0,
                "notes": "generated serving-tier target",
            }
        )
    return models
