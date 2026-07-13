"""Pure planning: decide a per-model action from the detected hardware, expand a logical model into
concrete per-backend / per-quant prep artifacts, and derive the license-acceptance URL.

No side effects -- this is the unit-testable core the `prepare_models` orchestrator drives.
"""

from typing import cast

from llb.backends.prepare.base import (
    ACTION_CACHE,
    ACTION_PULL,
    ACTION_SKIP,
    SUPPORTED_BACKENDS,
)
from llb.core.contracts import ModelSpec, PreparedModel, SourceRecord


def decide(backend: str, need_mb: int, max_mb: int, has_gpu: bool, force: bool) -> tuple[str, str]:
    """Per-model action + reason given the detected hardware."""
    if backend == "ollama":
        if need_mb > max_mb and not force:
            return ACTION_PULL, f"larger than {max_mb} MB VRAM; Ollama offloads to CPU (slow)"
        return ACTION_PULL, "ok"
    if backend == "vllm":
        if not has_gpu and not force:
            return ACTION_SKIP, "vLLM needs a CUDA GPU; none detected (use --force to cache anyway)"
        if need_mb > max_mb and not force:
            return ACTION_SKIP, f"needs ~{need_mb} MB VRAM, host has {max_mb} MB (use --force)"
        return ACTION_CACHE, "ok"
    return ACTION_SKIP, f"unknown backend '{backend}'"


def _normalize_source_record(value: "str | SourceRecord | dict[str, object]") -> dict[str, object]:
    if isinstance(value, str):
        return {"source": value}
    return {k: v for k, v in value.items() if v is not None}


def _normalize_source_records(value: object) -> list[dict[str, object]]:
    """A backend's `sources` value is one source or a LIST of them (multiple quants of one model)."""
    if isinstance(value, list):
        return [_normalize_source_record(v) for v in value]
    return [_normalize_source_record(value)]  # type: ignore[arg-type]


def _spec_row(
    model: ModelSpec, backend: str, record: dict[str, object], *, multi: bool
) -> ModelSpec | None:
    """One concrete prep artifact for a source record, or None when the record has no source."""
    source = record.get("source")
    if not isinstance(source, str) or not source:
        return None
    row = {**model, **record, "backend": backend, "source": source}
    if backend != model["backend"] or source != model["source"]:
        # Several quants of one backend (e.g. vLLM fp8 + w4a16) need distinct prep names.
        quant = record.get("quant")
        suffix = f"-{quant}" if multi and quant else ""
        row["name"] = f"{model['name']}-{backend}{suffix}"
    return cast(ModelSpec, row)


def _expand_model_spec(
    records: dict[str, list[dict[str, object]]], model: ModelSpec, expanded: list[ModelSpec]
) -> None:
    for backend, recs in records.items():
        if backend not in SUPPORTED_BACKENDS:
            continue
        for record in recs:
            row = _spec_row(model, backend, record, multi=len(recs) > 1)
            if row is not None:
                expanded.append(row)


def _expand_prepare_sources(models: list[ModelSpec]) -> list[ModelSpec]:
    """Expand a logical model into concrete backend artifacts that can be prepared.

    The resolver already understands per-backend `sources:` records. Model preparation needs the
    same expansion so a 16 GB host pulls Ollama GGUF fallbacks such as MamayLM/Lapa while also
    caching vLLM Hugging Face weights that fit the GPU. A backend that maps to a LIST of records
    (several vLLM quants of one model) expands to one prep artifact per quant.
    """
    expanded: list[ModelSpec] = []
    for model in models:
        records: dict[str, list[dict[str, object]]] = {
            backend: _normalize_source_records(source)
            for backend, source in (model.get("sources") or {}).items()
        }
        records.setdefault(model["backend"], [{"source": model["source"]}])

        _expand_model_spec(records, model, expanded)

    return expanded


def plan(
    models: list[ModelSpec],
    max_mb: int,
    has_gpu: bool,
    backend_filter: str,
    force: bool,
) -> list[PreparedModel]:
    """Annotate each in-scope model with an action + reason (no side effects)."""
    rows: list[PreparedModel] = []
    for m in _expand_prepare_sources(models):
        backend = m["backend"]
        if backend_filter != "all" and backend != backend_filter:
            continue
        if backend not in SUPPORTED_BACKENDS:
            continue
        need_mb = int(m.get("min_vram_gb", 0)) * 1024
        action, reason = decide(backend, need_mb, max_mb, has_gpu, force)
        rows.append({**m, "action": action, "reason": reason})
    return rows


def acceptance_url(spec: ModelSpec | PreparedModel) -> str | None:
    """Where to accept a gated model's license. Explicit `license_url`, else derived from the
    HF repo id when `gated: true`. None for ungated / non-HF entries."""
    if spec.get("license_url"):
        return str(spec["license_url"])
    if spec.get("gated") and spec.get("backend") == "vllm":
        return f"https://huggingface.co/{spec['source']}"
    return None
