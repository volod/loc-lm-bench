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
from llb.core.config import DEFAULT_OLLAMA_HOST
from llb.core.contracts import ChatMessage
from llb.prep.frontier import LLMComplete, ProvenanceLog, litellm_complete

_LOG = logging.getLogger(__name__)

ENDPOINT_LOCAL = "local"
ENDPOINT_FRONTIER = "frontier"
ENDPOINT_KINDS = (ENDPOINT_LOCAL, ENDPOINT_FRONTIER)
LOCAL_BACKEND_OLLAMA = "ollama"
LOCAL_BACKEND_VLLM = "vllm"
LOCAL_BACKEND_OPENAI = "openai"
LOCAL_BACKENDS = (LOCAL_BACKEND_OLLAMA, LOCAL_BACKEND_VLLM, LOCAL_BACKEND_OPENAI)

DEFAULT_LOCAL_BASE_URL = f"{DEFAULT_OLLAMA_HOST}/v1"


@dataclass(frozen=True)
class EndpointConfig:
    """Where (and how) the pipeline runs its LLM calls."""

    kind: str = ENDPOINT_LOCAL
    model: str = ""
    backend: str = LOCAL_BACKEND_OLLAMA
    base_url: str = DEFAULT_LOCAL_BASE_URL  # local only
    api_key: str = "not-needed"  # local only
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: float = 120.0
    # Local reasoning models (gemma4, deepseek-r1, qwen3) spend the token budget on hidden thinking
    # before any JSON, so structured extraction comes back empty. `think=False` disables it (Ollama
    # `think`); None leaves the endpoint's own default. Pair with a larger `max_tokens`.
    think: bool | None = None
    # Ollama loads a model with its modelfile context length (often 128k+), which can force CPU
    # offload on VRAM-bound hosts even though drafting prompts are bounded and small. `num_ctx`
    # right-sizes the context (native /api/chat only); None keeps the endpoint default. Prompts
    # longer than `num_ctx` would be silently truncated by Ollama, so keep headroom over
    # `extract_max_chars` + completion budget.
    num_ctx: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ENDPOINT_KINDS:
            raise ValueError(f"endpoint kind must be one of {ENDPOINT_KINDS}, got {self.kind!r}")
        if not self.model:
            raise ValueError("endpoint model must be set")
        if self.backend not in LOCAL_BACKENDS:
            raise ValueError(f"local backend must be one of {LOCAL_BACKENDS}, got {self.backend!r}")
        if self.kind != ENDPOINT_LOCAL and self.backend != LOCAL_BACKEND_OLLAMA:
            raise ValueError("local backend can only be set when endpoint kind is local")

    @property
    def egress(self) -> bool:
        """True when this endpoint sends the corpus off-box (frontier)."""
        return self.kind == ENDPOINT_FRONTIER

    def provenance(self) -> dict[str, object]:
        rec: dict[str, object] = {"kind": self.kind, "model": self.model, "egress": self.egress}
        if self.kind == ENDPOINT_LOCAL:
            rec["backend"] = self.backend
            rec["base_url"] = self.base_url
        if self.think is not None:
            rec["think"] = self.think
        if self.num_ctx is not None:
            rec["num_ctx"] = self.num_ctx
        return rec


def _openai_compatible_complete(
    cfg: EndpointConfig, log: ProvenanceLog, *, extra_body: dict[str, object] | None = None
) -> LLMComplete:
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
            extra_body=extra_body,
        )
        log.record(cfg.model, result.prompt_tokens, result.completion_tokens, 0.0)
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
    _LOG.info(
        "[ontology] endpoint=local backend=%s model=%s base_url=%s",
        cfg.backend,
        cfg.model,
        cfg.base_url,
    )
    return _local_complete(cfg, log)
