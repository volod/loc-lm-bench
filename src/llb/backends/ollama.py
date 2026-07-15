"""Ollama launcher: the prebuilt backend that proves the full loop (Premise 1).

Ollama ships as a prebuilt binary -- it uses the host GPU (CUDA) itself but needs no
from-source build -- so it proves the whole eval loop before backend telemetry takes on the
heavy vLLM/flash-attn source build. It runs as a host daemon exposing an OpenAI-compatible
endpoint at `<host>/v1`; this launcher verifies the daemon is reachable, optionally pulls
the model, and serves chat calls through `openai_client.chat_once`. Telemetry for RAG core is the
steady-state tokens/sec observed on the last call; richer per-backend telemetry (served
context, peak VRAM) lands in backend telemetry.

`urllib`/`subprocess` are stdlib; the `openai` client is a base dep. Nothing to compile.
"""

import json
import subprocess
import urllib.error
import urllib.request
from typing import cast

import openai

from llb.backends.base import BackendLauncher, ChatResult
from llb.backends.openai_client import chat_once, make_client
from llb.core.contracts.hardware import BackendMetadata
from llb.core.contracts.common import ChatMessage


class OllamaLauncher(BackendLauncher):
    """Serve one Ollama model over its OpenAI-compatible endpoint."""

    def __init__(self, model: str, host: str = "http://localhost:11434", pull: bool = False):
        super().__init__(model=model, meta={"backend": "ollama", "host": host})
        self.host = host.rstrip("/")
        self.pull = pull
        self._client: openai.OpenAI | None = None
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
        self._client = make_client(f"{self.host}/v1", api_key="ollama")

    def chat(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float, timeout: float
    ) -> ChatResult:
        if self._client is None:
            self._client = make_client(f"{self.host}/v1", api_key="ollama")
        self._last = chat_once(
            self._client,
            self.model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
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
