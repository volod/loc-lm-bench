"""Shared CLI setup for agent context-policy benchmark lanes."""

from llb.backends.context_budget import (
    ContextBudget,
    fixed_budget,
    resolve_context_budget,
)
from llb.core.config import RunConfig


def resolve_agent_context_budget(
    cfg: RunConfig,
    *,
    base_url: str | None,
    max_prompt_chars: int | None,
) -> ContextBudget:
    """Resolve one prompt budget, warming Ollama first when an explicit num_ctx is requested."""
    if max_prompt_chars is not None:
        return fixed_budget(max_prompt_chars)
    ollama_num_ctx = cfg.max_model_len or cfg.context_budget
    if cfg.backend == "ollama" and ollama_num_ctx:
        from llb.backends.ollama import OllamaLauncher
        from llb.backends.served_window import is_ollama_base_url, native_root

        native_host = (
            native_root(base_url)
            if base_url and is_ollama_base_url(base_url, cfg.ollama_host)
            else cfg.ollama_host
        )
        warm = OllamaLauncher(cfg.model, host=native_host, num_ctx=ollama_num_ctx)
        warm.start()
        try:
            warm.ensure_num_ctx(timeout=cfg.request_timeout_s)
        finally:
            warm.stop()
    return resolve_context_budget(cfg, probe=True)
