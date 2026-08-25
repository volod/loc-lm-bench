"""Load candidate-model preparation targets: validate an external candidate manifest (YAML) or read
a generated serving `tier.json` into concrete `ModelSpec` prep targets, plus the manifest's family
register (`families:`) that says which generation of a family the roster currently carries.
"""

import json
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from llb.core.contracts.models import FamilySpec, ModelSpec


class _ModelSpecSchema(BaseModel):
    """Validation model for one external candidate-manifest entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    backend: str
    source: str
    family: str | None = None
    generation: str | None = None
    min_vram_gb: int | float = 0
    notes: str | None = None
    license: str | None = None
    license_url: str | None = None
    gated: bool = False
    params_b: float | None = None
    quant: str | None = None
    bpw: float | None = None
    n_layers: int | None = None
    kv_layers: int | None = None
    kv_dim: int | None = None
    max_context: int | None = None
    sliding_window: int | None = None
    sliding_window_pattern: int | None = None
    vocab_size: int | None = None
    hidden_size: int | None = None
    tie_word_embeddings: bool | None = None
    embed_bpw: float | None = None
    hi_precision_params_b: float | None = None
    sources: dict[str, "str | dict[str, object] | list[str | dict[str, object]]"] | None = None


class _UpstreamSchema(BaseModel):
    """Where a family's artifacts come from, for a later currency check."""

    model_config = ConfigDict(extra="forbid")

    hf_author: str | None = None
    hf_prefix: str | None = None
    ollama_namespace: str | None = None


class _GenerationSchema(BaseModel):
    """One generation of a family and its status in the roster."""

    model_config = ConfigDict(extra="forbid")

    id: str
    status: str
    label: str | None = None
    license: str | None = None
    license_url: str | None = None
    weights_url: str | None = None


class _FamilySchema(BaseModel):
    """One `families:` entry: the family and the generations the roster carries for it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    role: str
    focus: str | None = None
    upstream: _UpstreamSchema | None = None
    generations: list[_GenerationSchema]


def _read_yaml(path: Path | str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML -- {exc}") from None
    return data if isinstance(data, dict) else {}


def load_families(path: Path | str) -> list[FamilySpec]:
    """The manifest's family register, or an empty list when the manifest declares none."""
    families = _read_yaml(path).get("families")
    if families is None:
        return []
    if not isinstance(families, list):
        raise ValueError(f"{path}: 'families:' must be a list of family entries")
    try:
        validated = [_FamilySchema.model_validate(family) for family in families]
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid family entry -- {exc}") from None
    return [cast(FamilySpec, family.model_dump(exclude_none=True)) for family in validated]


def load_manifest(path: Path | str) -> list[ModelSpec]:
    models = _read_yaml(path).get("models")
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
