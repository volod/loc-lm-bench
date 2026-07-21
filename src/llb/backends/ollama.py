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

    def __init__(self, model: str, host: str = "http://localhost:11434", pull: bool = False):
        super().__init__(model=model, meta={"backend": "ollama", "host": host})
        self.host = host.rstrip("/")
        self.pull = pull
        self._last: ChatResult | None = None

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

    def chat(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float, timeout: float
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": messages,
            "options": {"num_predict": max_tokens, "temperature": temperature},
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
