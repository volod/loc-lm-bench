"""Shared CLI resolution for opt-in model-backed query preparation."""

from typing import Any

MODEL_QUERY_PREP_STEPS = frozenset({"rewrite", "hyde", "decompose"})


def parse_query_prep_steps(value: str | None) -> list[str]:
    """Parse the comma-separated CLI spelling while preserving configured step order."""
    return [step.strip() for step in value.split(",") if step.strip()] if value else []


def resolve_query_prep_endpoint(
    cfg: Any,
    steps: list[str],
    *,
    model: str | None,
    backend: str | None,
) -> tuple[Any, Any | None, dict[str, str] | None]:
    """Resolve a launcher only when a configured query-prep step calls a model."""
    from llb.executor.runner_backend import _make_launcher

    model_steps = MODEL_QUERY_PREP_STEPS.intersection(steps)
    if model_steps and model is None:
        raise ValueError("model-backed query prep needs --query-prep-model")
    endpoint_cfg = cfg.with_overrides(
        model=model,
        backend=backend or ("ollama" if model else None),
    )
    launcher = _make_launcher(endpoint_cfg) if model_steps else None
    endpoint = (
        {"model": endpoint_cfg.model, "backend": endpoint_cfg.backend} if model_steps else None
    )
    return endpoint_cfg, launcher, endpoint
