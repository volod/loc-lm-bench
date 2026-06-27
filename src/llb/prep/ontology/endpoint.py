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
from dataclasses import dataclass

from llb.backends.openai_client import chat_once, make_client
from llb.config import DEFAULT_OLLAMA_HOST
from llb.contracts import ChatMessage
from llb.prep.frontier import LLMComplete, ProvenanceLog, litellm_complete

_LOG = logging.getLogger(__name__)

ENDPOINT_LOCAL = "local"
ENDPOINT_FRONTIER = "frontier"
ENDPOINT_KINDS = (ENDPOINT_LOCAL, ENDPOINT_FRONTIER)

DEFAULT_LOCAL_BASE_URL = f"{DEFAULT_OLLAMA_HOST}/v1"


@dataclass(frozen=True)
class EndpointConfig:
    """Where (and how) the pipeline runs its LLM calls."""

    kind: str = ENDPOINT_LOCAL
    model: str = ""
    base_url: str = DEFAULT_LOCAL_BASE_URL  # local only
    api_key: str = "not-needed"  # local only
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: float = 120.0
    # Local reasoning models (gemma4, deepseek-r1, qwen3) spend the token budget on hidden thinking
    # before any JSON, so structured extraction comes back empty. `think=False` disables it (Ollama
    # `think`); None leaves the endpoint's own default. Pair with a larger `max_tokens`.
    think: bool | None = None

    def __post_init__(self) -> None:
        if self.kind not in ENDPOINT_KINDS:
            raise ValueError(f"endpoint kind must be one of {ENDPOINT_KINDS}, got {self.kind!r}")
        if not self.model:
            raise ValueError("endpoint model must be set")

    @property
    def egress(self) -> bool:
        """True when this endpoint sends the corpus off-box (frontier)."""
        return self.kind == ENDPOINT_FRONTIER

    def provenance(self) -> dict[str, object]:
        rec: dict[str, object] = {"kind": self.kind, "model": self.model, "egress": self.egress}
        if self.kind == ENDPOINT_LOCAL:
            rec["base_url"] = self.base_url
        if self.think is not None:
            rec["think"] = self.think
        return rec


def _local_complete(cfg: EndpointConfig, log: ProvenanceLog) -> LLMComplete:
    # Disabling a reasoning model's thinking is honored only by Ollama's NATIVE /api/chat `think`
    # field -- the OpenAI-compatible /v1 layer ignores it and the model burns the whole token budget
    # on hidden reasoning, returning empty structured output. So route the think-set case there.
    if cfg.think is not None:
        return _ollama_native_complete(cfg, log)

    client = make_client(cfg.base_url, api_key=cfg.api_key)

    def complete(prompt: str) -> str:
        messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
        result = chat_once(
            client,
            cfg.model,
            messages,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            timeout=cfg.timeout,
        )
        log.record(cfg.model, result.prompt_tokens, result.completion_tokens, 0.0)
        if result.error:
            raise RuntimeError(f"local endpoint error ({cfg.base_url}): {result.error}")
        return result.text

    return complete


def _native_chat_url(base_url: str) -> str:
    """Map an OpenAI-compatible base URL to Ollama's native chat endpoint."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/api/chat"


def _ollama_native_complete(cfg: EndpointConfig, log: ProvenanceLog) -> LLMComplete:
    """Completion via Ollama's native /api/chat, which honors `think` (reasoning on/off)."""
    import httpx

    url = _native_chat_url(cfg.base_url)

    def complete(prompt: str) -> str:
        payload = {
            "model": cfg.model,
            "think": cfg.think,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens},
        }
        try:
            resp = httpx.post(url, json=payload, timeout=cfg.timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"local endpoint error ({url}): {exc}") from exc
        data = resp.json()
        log.record(
            cfg.model,
            int(data.get("prompt_eval_count", 0) or 0),
            int(data.get("eval_count", 0) or 0),
            0.0,
        )
        message = data.get("message") or {}
        return str(message.get("content") or "")

    return complete


def build_complete(cfg: EndpointConfig, log: ProvenanceLog) -> LLMComplete:
    """Return the injectable completion callable for `cfg`, recording cost into `log`."""
    if cfg.kind == ENDPOINT_FRONTIER:
        _LOG.info("[ontology] endpoint=frontier model=%s (CORPUS EGRESS)", cfg.model)
        return litellm_complete(cfg.model, temperature=cfg.temperature, log=log)
    _LOG.info("[ontology] endpoint=local model=%s base_url=%s", cfg.model, cfg.base_url)
    return _local_complete(cfg, log)
