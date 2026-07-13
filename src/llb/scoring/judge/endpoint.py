"""Local OpenAI-compatible judge endpoint resolution and metadata."""

import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from llb.core import env
from llb.core.paths import load_project_env


def _normalize_openai_base_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("judge base URL must be an http(s) URL with a host")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(
            "judge base URL must not contain credentials, query parameters, or a fragment; "
            f"use {env.DEEPEVAL_JUDGE_API_KEY} for authentication"
        )
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def resolve_judge_endpoint(
    judge_model: str, explicit_base_url: str | None = None
) -> tuple[str, str | None]:
    """Resolve the served model id and local OpenAI-compatible endpoint."""
    load_project_env()
    base_url = explicit_base_url or os.environ.get(env.DEEPEVAL_JUDGE_BASE_URL)
    if base_url is not None:
        base_url = _normalize_openai_base_url(base_url)
    return judge_model, base_url


def judge_experiment_metadata(judge_model: str, base_url: str | None = None) -> dict[str, Any]:
    """Return the non-secret judge configuration recorded with experiments."""
    served_model, resolved_base_url = resolve_judge_endpoint(judge_model, base_url)
    return {
        "provider": "deepeval-geval",
        "model": served_model,
        "base_url": resolved_base_url,
        "prompt_language": "uk",
        "metrics": ["faithfulness", "answer_relevancy"],
    }
