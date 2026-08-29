"""Shared CLI setup for agent context-policy benchmark lanes."""

from llb.backends.context_budget import (
    ContextBudget,
    fixed_budget,
    resolve_context_budget,
)
from llb.backends.served_window import (
    is_ollama_base_url,
    native_root,
    probe_served_window,
)
from llb.core.config import RunConfig


def agent_probe_host(cfg: RunConfig, base_url: str | None) -> str | None:
    """The native host to probe when `--base-url` names an Ollama daemon the config does not.

    Ollama exposes its OpenAI-compatible `/v1` layer on the same port as its native API, so a lane
    driven with `--base-url http://host:11434/v1` still runs against Ollama -- and against a
    different daemon than `cfg.ollama_host` may name. None means "no override": probe the host the
    config already names, which is also what a non-Ollama `--base-url` gets, since the run's own
    launcher is what such a URL configures.
    """
    if cfg.backend != "ollama" or not base_url:
        return None
    return native_root(base_url) if is_ollama_base_url(base_url, cfg.ollama_host) else None


def resolve_agent_context_budget(
    cfg: RunConfig,
    *,
    base_url: str | None,
    max_prompt_chars: int | None,
) -> ContextBudget:
    """Resolve one prompt budget, warm-loading Ollama first so the served window is observable.

    The guard is the MINIMUM of the declared and the served window, so it is only worth what its
    probe is worth. Ollama reports nothing on `/api/ps` until a request has loaded the model, so
    the probe has to warm it -- unconditionally, not only when the run pins `num_ctx`. An UNPINNED
    run is exactly the case that needs it: Ollama then serves its 4096 default however large a
    window the model card declares, and a probe that reads "unknown" there leaves a guard 32x
    looser than the window that will silently truncate the prompt.

    An unreachable backend resolves to the declared window rather than raising. The probe is
    telemetry about a window; the launcher the run itself starts is what refuses a dead backend,
    and it says so by name instead of failing inside budget resolution.
    """
    if max_prompt_chars is not None:
        return fixed_budget(max_prompt_chars)
    served = probe_served_window(
        cfg,
        host=agent_probe_host(cfg, base_url),
        timeout=cfg.request_timeout_s,
    )
    return resolve_context_budget(cfg, served_max_model_len=served)
