"""Inference-endpoint adapter for the M4.4 pipeline (local default, frontier opt-in).

Every pipeline stage drives ONE injectable `LLMComplete` (prompt -> text). This module builds
that callable from an `EndpointConfig`:

  - local (default, NO corpus egress): an OpenAI-compatible endpoint -- the same `chat_once`
    seam the eval backends use -- pointed at a local server (Ollama/vLLM/llama.cpp). Token
    counts are recorded; cost is 0 (local compute).
  - frontier (opt-in -- the Milestone H egress decision): reuses `frontier.litellm_complete`,
    so provider/model/token/cost provenance accumulates exactly as the M3.5 utilities do.

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
        return rec


def _local_complete(cfg: EndpointConfig, log: ProvenanceLog) -> LLMComplete:
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


def build_complete(cfg: EndpointConfig, log: ProvenanceLog) -> LLMComplete:
    """Return the injectable completion callable for `cfg`, recording cost into `log`."""
    if cfg.kind == ENDPOINT_FRONTIER:
        _LOG.info("[ontology] endpoint=frontier model=%s (CORPUS EGRESS)", cfg.model)
        return litellm_complete(cfg.model, temperature=cfg.temperature, log=log)
    _LOG.info("[ontology] endpoint=local model=%s base_url=%s", cfg.model, cfg.base_url)
    return _local_complete(cfg, log)
