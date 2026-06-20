"""Backend launcher interface + the uniform chat result.

All v1 backends (Ollama now; vLLM / llama.cpp later) expose an OpenAI-compatible HTTP
API, so the eval/RAG/judge code is backend-agnostic -- only the launcher and the
telemetry hook are backend-specific (Premise 1). A launcher is a context manager: it
starts/verifies the serving process, hands out a chat callable, and reports telemetry.

`ChatResult.error` carries a normalized failure token ("timeout" / "backend_error" /
None) so the eval graph can classify reliability failures without catching SDK-specific
exceptions.
"""

from dataclasses import dataclass, field

# Normalized transport-level failure tokens (None == success).
ERR_TIMEOUT = "timeout"
ERR_BACKEND = "backend_error"


@dataclass
class ChatResult:
    """Outcome of one chat call, with token accounting + latency for telemetry."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None

    def tokens_per_s(self) -> float:
        if self.error or self.latency_s <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / self.latency_s


@dataclass
class BackendLauncher:
    """Base launcher. Subclasses provide `chat` and may override start/stop/telemetry."""

    model: str
    meta: dict = field(default_factory=dict)

    def start(self) -> None:
        """Ensure the backend is serving `self.model`. Default: assume it is up."""

    def stop(self) -> None:
        """Release the backend. Default: nothing to release."""

    def chat(self, messages: list[dict], max_tokens: int, temperature: float,
             timeout: float) -> ChatResult:
        raise NotImplementedError

    def telemetry(self) -> dict:
        """Backend-specific telemetry (tokens/sec, served context, VRAM). Default: meta."""
        return dict(self.meta)

    def __enter__(self) -> "BackendLauncher":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
