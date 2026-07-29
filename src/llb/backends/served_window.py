"""Probe the context window a local backend is actually serving.

The agent-loop prompt guard used to resolve its budget from the DECLARED window alone
(host planner cap, model card, `max_model_len`, `context_budget`). That is wrong for Ollama:
its served `num_ctx` defaults to 4096 even when a GGUF advertises 131072, so a declared 32k
guard is 8x looser than the window that will silently truncate the prompt. This module asks
each backend what it is serving right now so the guard can take the MINIMUM of declared and
probed.
"""

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from llb.core.config_validation import (
    DEFAULT_LLAMACPP_HOST,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_VLLM_HOST,
)

_LOG = logging.getLogger(__name__)

# Provenance labels persisted beside the resolved prompt-char budget.
BUDGET_SOURCE_DECLARED = "declared"
BUDGET_SOURCE_SERVED = "served"
BUDGET_SOURCE_FIXED = "fixed"
BUDGET_SOURCE_UNBOUNDED = "unbounded"

HttpGet = Callable[[str], tuple[int, str] | None]


def _http_get(url: str, timeout: float = 3.0) -> tuple[int, str] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(resp.status), resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def native_root(url: str) -> str:
    """Scheme+host root for a backend base URL (drops an OpenAI-compatible `/v1` suffix)."""
    parsed = urlsplit(url)
    root = f"{parsed.scheme or 'http'}://{parsed.netloc or parsed.path}".rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def backend_host(backend: str, *, ollama_host: str, vllm_host: str, llamacpp_host: str) -> str:
    """The HTTP root to probe for `backend`."""
    if backend == "ollama":
        return native_root(ollama_host or DEFAULT_OLLAMA_HOST)
    if backend == "vllm":
        return native_root(vllm_host or DEFAULT_VLLM_HOST)
    if backend == "llamacpp":
        return native_root(llamacpp_host or DEFAULT_LLAMACPP_HOST)
    return native_root(ollama_host or DEFAULT_OLLAMA_HOST)


def _model_aliases(model: str) -> set[str]:
    """Name forms Ollama may report for the same loaded model (`tag` vs `tag:latest`)."""
    aliases = {model}
    if ":" in model:
        aliases.add(model.rsplit(":", 1)[0])
    else:
        aliases.add(f"{model}:latest")
    return aliases


def parse_ollama_served_context(ps_body: str, model: str) -> int | None:
    """Pull the loaded model's context from an Ollama `/api/ps` body (best-effort).

    Recent Ollama builds report `context_length`; older ones used `context`. Both are accepted,
    including a nested `details.context_length`.
    """
    try:
        data = json.loads(ps_body)
    except (ValueError, TypeError):
        return None
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    aliases = _model_aliases(model)
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("model") or "")
        if name not in aliases and name.rsplit(":", 1)[0] not in aliases:
            continue
        for key in ("context_length", "context"):
            value = entry.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return int(value)
        details = entry.get("details")
        if isinstance(details, dict):
            for key in ("context_length", "context"):
                nested = details.get(key)
                if isinstance(nested, int) and not isinstance(nested, bool) and nested > 0:
                    return int(nested)
    return None


def probe_served_max_model_len(
    backend: str,
    *,
    model: str,
    host: str,
    http_get: HttpGet | None = None,
) -> int | None:
    """Ask `backend` for the context length it is serving for `model`. None when unreachable.

    Ollama: `/api/ps` -> loaded entry `context`. vLLM: `/v1/models` -> `max_model_len`.
    llama.cpp: `/props` -> `n_ctx`. A miss is not a failure -- the caller falls back to the
    declared window and records that the probe was unavailable.
    """
    get = http_get or _http_get
    root = native_root(host)
    if backend == "ollama":
        got = get(f"{root}/api/ps")
        if not got or got[0] != 200:
            _LOG.info("[served-window] ollama /api/ps unreachable at %s", root)
            return None
        served = parse_ollama_served_context(got[1], model)
        if served is None:
            _LOG.info("[served-window] ollama model %s not resident at %s", model, root)
        return served
    if backend == "vllm":
        from llb.backends import vllm_command

        got = get(f"{root}/v1/models")
        if not got or got[0] != 200:
            _LOG.info("[served-window] vllm /v1/models unreachable at %s", root)
            return None
        return vllm_command.parse_served_context(got[1])
    if backend == "llamacpp":
        from llb.backends import llamacpp_command

        got = get(f"{root}/props")
        if not got or got[0] != 200:
            _LOG.info("[served-window] llamacpp /props unreachable at %s", root)
            return None
        return llamacpp_command.parse_served_context(got[1])
    _LOG.info("[served-window] no probe for backend %s", backend)
    return None


def bind_window(declared: int, served: int | None) -> tuple[int, str]:
    """Take the minimum of declared and probed windows; name which one bound the result.

    `declared` of 0 means "cannot bound from config/model"; a positive `served` then becomes the
    window and the source is `served`. When both are positive the smaller wins and `budget_source`
    names that smaller side.
    """
    if served is None or served <= 0:
        return declared, BUDGET_SOURCE_DECLARED if declared > 0 else BUDGET_SOURCE_UNBOUNDED
    if declared <= 0:
        return served, BUDGET_SOURCE_SERVED
    if served < declared:
        return served, BUDGET_SOURCE_SERVED
    return declared, BUDGET_SOURCE_DECLARED


def probe_config_served_max_model_len(
    config: Any, *, http_get: HttpGet | None = None
) -> int | None:
    """Probe using the host fields already on a `RunConfig`."""
    host = backend_host(
        config.backend,
        ollama_host=getattr(config, "ollama_host", DEFAULT_OLLAMA_HOST),
        vllm_host=getattr(config, "vllm_host", DEFAULT_VLLM_HOST),
        llamacpp_host=getattr(config, "llamacpp_host", DEFAULT_LLAMACPP_HOST),
    )
    return probe_served_max_model_len(
        config.backend, model=config.model, host=host, http_get=http_get
    )
