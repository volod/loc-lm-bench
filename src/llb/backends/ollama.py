"""Ollama launcher: the prebuilt backend that proves the full loop (Premise 1).

Ollama ships as a prebuilt binary -- it uses the host GPU (CUDA) itself but needs no
from-source build -- so it proves the whole eval loop before backend telemetry takes on the
heavy vLLM/flash-attn source build. It runs as a host daemon; this launcher verifies the daemon
is reachable, optionally pulls the model, and uses native `/api/chat` so `think=false` is honored
for bounded benchmark generations. Telemetry for RAG core is the steady-state tokens/sec observed
on the last call; richer per-backend telemetry lands in backend telemetry.

`urllib`/`subprocess` are stdlib; no backend-specific client is required. Nothing to compile.
"""

import json
import subprocess
import time
import urllib.error
import urllib.request
from typing import cast

from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT, BackendLauncher, ChatResult
from llb.core.contracts.hardware import BackendMetadata
from llb.core.contracts.common import ChatMessage


class OllamaLauncher(BackendLauncher):
    """Serve one Ollama model over its native chat endpoint."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        pull: bool = False,
        num_ctx: int | None = None,
    ):
        super().__init__(model=model, meta={"backend": "ollama", "host": host})
        self.host = host.rstrip("/")
        self.pull = pull
        # When set, every chat request asks Ollama to serve this context length. Without it
        # Ollama defaults to 4096 even for GGUFs that advertise a much larger window, which is
        # exactly the silent truncation the agent-loop prompt guard exists to prevent.
        self.num_ctx = num_ctx
        self._last: ChatResult | None = None
        self._served_context: int | None = None

    def _reachable(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as resp:
                return int(resp.status) == 200
        except (urllib.error.URLError, OSError):
            return False

    def start(self) -> None:
        if not self._reachable():
            raise RuntimeError(
                f"Ollama not reachable at {self.host}. Start it with `ollama serve`."
            )
        if self.pull:
            subprocess.run(["ollama", "pull", self.model], check=True)
        self._served_context = self._read_served_context()
        self.meta["served_context"] = self._served_context

    def _read_served_context(self) -> int | None:
        from llb.backends.served_window import parse_ollama_served_context

        try:
            with urllib.request.urlopen(f"{self.host}/api/ps", timeout=5) as resp:
                body = resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError):
            return None
        return parse_ollama_served_context(body, self.model)

    def served_context(self) -> int | None:
        return self._served_context

    def ensure_num_ctx(self, timeout: float = 120.0) -> int | None:
        """Warm-load so `/api/ps` reports the `num_ctx` this launcher will serve.

        Ollama keeps a previously loaded context until a request asks for a different one. Calling
        this before a budget probe makes the probe observe the window the run will actually use.
        """
        if self.num_ctx is None or self.num_ctx <= 0:
            return self._read_served_context()
        self.chat(
            [{"role": "user", "content": " "}],
            max_tokens=1,
            temperature=0.0,
            timeout=timeout,
        )
        self._served_context = self._read_served_context()
        self.meta["served_context"] = self._served_context
        return self._served_context

    def chat(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float, timeout: float
    ) -> ChatResult:
        options: dict[str, object] = {"num_predict": max_tokens, "temperature": temperature}
        if self.num_ctx is not None and self.num_ctx > 0:
            options["num_ctx"] = self.num_ctx
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": messages,
            "options": options,
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8", "replace"))
        except TimeoutError:
            self._last = ChatResult(
                text="", latency_s=time.monotonic() - started, error=ERR_TIMEOUT
            )
            return self._last
        except (urllib.error.URLError, OSError, ValueError):
            self._last = ChatResult(
                text="", latency_s=time.monotonic() - started, error=ERR_BACKEND
            )
            return self._last
        message = data.get("message") or {}
        self._last = ChatResult(
            text=str(message.get("content") or ""),
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            latency_s=time.monotonic() - started,
        )
        return self._last

    def telemetry(self) -> BackendMetadata:
        out = dict(self.meta)
        if self._last is not None and not self._last.error:
            out["tokens_per_s"] = round(self._last.tokens_per_s(), 2)
            out["last_completion_tokens"] = self._last.completion_tokens
        return cast(BackendMetadata, out)


def list_models(host: str = "http://localhost:11434") -> list[str]:
    """Names of models currently available to the local Ollama daemon (best-effort)."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return []
    return [m.get("name", "") for m in data.get("models", [])]
