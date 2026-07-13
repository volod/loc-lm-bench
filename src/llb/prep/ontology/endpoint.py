"""Inference-endpoint adapter for the ontology-assisted drafting pipeline (local default, frontier opt-in).

Every pipeline stage drives ONE injectable `LLMComplete` (prompt -> text). This module builds
that callable from an `EndpointConfig`:

  - local (default, NO corpus egress): an OpenAI-compatible endpoint -- the same `chat_once`
    seam the eval backends use -- pointed at a local server (Ollama/vLLM/llama.cpp). Token
    counts are recorded; cost is 0 (local compute).
  - frontier (opt-in -- the human decision egress decision): reuses `frontier.litellm_complete`,
    so provider/model/token/cost provenance accumulates exactly as the frontier drafting utilities do.

The completion is the only thing the stages see, so the whole pipeline is unit-tested with a
fake `complete` and never needs a server or a provider key.
"""

import logging
from time import monotonic

from llb.backends.openai_client import chat_once, make_client
from llb.core.contracts import ChatMessage
from llb.prep.frontier import litellm_complete
from llb.prep.frontier_telemetry import (
    DraftBudget,
    LLMComplete,
    ProvenanceLog,
    budgeted_complete,
)
from llb.prep.ontology.endpoint_config import (
    ENDPOINT_FRONTIER,
    LOCAL_BACKEND_OLLAMA,
    LOCAL_BACKEND_VLLM,
    EndpointCompleters,
    EndpointConfig,
    EndpointLogs,
    EndpointPlan,
)

_LOG = logging.getLogger(__name__)


def _openai_compatible_complete(
    cfg: EndpointConfig, log: ProvenanceLog, *, extra_body: dict[str, object] | None = None
) -> LLMComplete:
    client = make_client(cfg.base_url, api_key=cfg.api_key)

    def complete(prompt: str) -> str:
        started = monotonic()
        messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
        result = chat_once(
            client,
            cfg.model,
            messages,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            timeout=cfg.timeout,
            extra_body=extra_body,
        )
        log.record(
            cfg.model,
            result.prompt_tokens,
            result.completion_tokens,
            0.0,
            latency_s=getattr(result, "latency_s", 0.0) or monotonic() - started,
            error=result.error,
        )
        if result.error:
            raise RuntimeError(f"local endpoint error ({cfg.base_url}): {result.error}")
        return result.text

    return complete


def _vllm_extra_body(cfg: EndpointConfig) -> dict[str, object] | None:
    """vLLM request extras for reasoning-output control.

    vLLM exposes `chat_template_kwargs` through the OpenAI-compatible endpoint. Qwen-style
    reasoning templates use `enable_thinking`; vLLM's own request schema also accepts
    `include_reasoning` and `reasoning_effort`, so the response budget is spent on JSON instead
    of hidden reasoning when `--no-think` maps to `think=False`.
    """
    if cfg.think is None:
        return None
    body: dict[str, object] = {
        "chat_template_kwargs": {"enable_thinking": cfg.think},
    }
    if cfg.think is False:
        body["include_reasoning"] = False
        body["reasoning_effort"] = "none"
    return body


def _local_complete(cfg: EndpointConfig, log: ProvenanceLog) -> LLMComplete:
    if cfg.backend == LOCAL_BACKEND_VLLM:
        return _openai_compatible_complete(cfg, log, extra_body=_vllm_extra_body(cfg))

    # Ontology completions are JSON at every model-driven stage. Ollama's native endpoint supports
    # JSON mode as well as `think` and `num_ctx`, while its OpenAI-compatible layer cannot reliably
    # enforce all three. Keep every Ollama ontology call on the native structured-output path.
    if cfg.backend == LOCAL_BACKEND_OLLAMA:
        return _ollama_native_complete(cfg, log)
    if cfg.think is not None or cfg.num_ctx is not None:
        raise ValueError(f"backend {cfg.backend!r} does not support think/num_ctx controls")
    return _openai_compatible_complete(cfg, log)


def _native_chat_url(base_url: str) -> str:
    """Map an OpenAI-compatible base URL to Ollama's native chat endpoint."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/api/chat"


def _ollama_native_complete(cfg: EndpointConfig, log: ProvenanceLog) -> LLMComplete:
    """Completion via Ollama native chat with JSON output and optional reasoning controls."""
    import httpx

    url = _native_chat_url(cfg.base_url)

    def complete(prompt: str) -> str:
        started = monotonic()
        options: dict[str, object] = {"temperature": cfg.temperature, "num_predict": cfg.max_tokens}
        if cfg.num_ctx is not None:
            options["num_ctx"] = cfg.num_ctx
        payload = {
            "model": cfg.model,
            "format": "json",
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": options,
        }
        if cfg.think is not None:
            payload["think"] = cfg.think
        try:
            resp = httpx.post(url, json=payload, timeout=cfg.timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.record(cfg.model, 0, 0, 0.0, latency_s=monotonic() - started, error=str(exc))
            raise RuntimeError(f"local endpoint error ({url}): {exc}") from exc
        data = resp.json()
        log.record(
            cfg.model,
            int(data.get("prompt_eval_count", 0) or 0),
            int(data.get("eval_count", 0) or 0),
            0.0,
            latency_s=monotonic() - started,
        )
        message = data.get("message") or {}
        return str(message.get("content") or "")

    return complete


def build_complete(
    cfg: EndpointConfig, log: ProvenanceLog, *, budget: DraftBudget | None = None
) -> LLMComplete:
    """Return the injectable completion callable for `cfg`, recording cost into `log`."""
    if cfg.kind == ENDPOINT_FRONTIER:
        _LOG.info("[ontology] endpoint=frontier model=%s (CORPUS EGRESS)", cfg.model)
        raw = litellm_complete(cfg.model, temperature=cfg.temperature, log=log)
        active_budget = budget or DraftBudget(max_calls=cfg.max_calls, max_usd=cfg.max_usd)
        return budgeted_complete(raw, log, active_budget)
    _LOG.info(
        "[ontology] endpoint=local backend=%s model=%s base_url=%s",
        cfg.backend,
        cfg.model,
        cfg.base_url,
    )
    return _local_complete(cfg, log)


def build_completers(plan: EndpointPlan, logs: EndpointLogs) -> EndpointCompleters:
    """Build the two phase callables, sharing one frontier budget across the run."""
    frontier = next(
        (cfg for cfg in (plan.extraction, plan.drafting) if cfg.kind == ENDPOINT_FRONTIER), None
    )
    budget = (
        DraftBudget(max_calls=frontier.max_calls, max_usd=frontier.max_usd)
        if frontier is not None
        else None
    )
    return EndpointCompleters(
        extraction=build_complete(plan.extraction, logs.extraction, budget=budget),
        drafting=build_complete(plan.drafting, logs.drafting, budget=budget),
    )
