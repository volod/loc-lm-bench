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
from dataclasses import dataclass
from typing import cast

from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT, BackendLauncher, ChatResult
from llb.core.contracts.hardware import BackendMetadata
from llb.core.contracts.common import ChatMessage


# Ollama reports a resident model's total weight bytes and the share of them held in VRAM;
# the CPU/GPU percentages its own `ps` prints are derived from that pair, rounded on the CPU side.
PLACEMENT_GPU = "GPU-resident"
PLACEMENT_CPU = "CPU-only"


@dataclass(frozen=True)
class OllamaPlacement:
    """Where a resident model's weights actually sit: total bytes vs the bytes held in VRAM.

    This is MEASURED placement, not the planner's estimate: a q4 GGUF that the memory planner
    calls `offload` may still land wholly on the GPU at a small context, and the throughput a
    row records is only readable beside the split that produced it.
    """

    total_bytes: int
    vram_bytes: int

    @property
    def cpu_percent(self) -> int:
        """Percent of the weights held in system RAM, rounded the way Ollama's own `ps` rounds."""
        if self.total_bytes <= 0:
            return 0
        return int(round((self.total_bytes - self.vram_bytes) / self.total_bytes * 100))

    @property
    def gpu_percent(self) -> int:
        return 100 - self.cpu_percent

    @property
    def label(self) -> str:
        """The placement as a table cell: fully resident, fully on CPU, or the offload split."""
        if self.vram_bytes <= 0:
            return PLACEMENT_CPU
        if self.vram_bytes >= self.total_bytes:
            return PLACEMENT_GPU
        return f"offload {self.cpu_percent}%/{self.gpu_percent}% CPU/GPU"


def parse_ollama_placement(ps_body: str, model: str) -> OllamaPlacement | None:
    """Pull one loaded model's GPU/CPU weight split from an `/api/ps` body (best-effort)."""
    from llb.backends.served_window import model_aliases, names_this_model

    try:
        data = json.loads(ps_body)
    except (ValueError, TypeError):
        return None
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    aliases = model_aliases(model)
    for entry in models:
        if not isinstance(entry, dict) or not names_this_model(entry, aliases):
            continue
        total, vram = entry.get("size"), entry.get("size_vram")
        if isinstance(total, int) and isinstance(vram, int) and total > 0:
            return OllamaPlacement(total_bytes=total, vram_bytes=max(0, min(vram, total)))
    return None


class OllamaLauncher(BackendLauncher):
    """Serve one Ollama model over its native chat endpoint."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        pull: bool = False,
        num_ctx: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(model=model, meta={"backend": "ollama", "host": host})
        self.host = host.rstrip("/")
        self.pull = pull
        # When set, every chat request asks Ollama to serve this context length. Without it
        # Ollama defaults to 4096 even for GGUFs that advertise a much larger window, which is
        # exactly the silent truncation the agent-loop prompt guard exists to prevent.
        self.num_ctx = num_ctx
        self.seed = seed
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

    def _read_ps(self) -> str | None:
        """The raw `/api/ps` body, or None when the daemon cannot be reached."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/ps", timeout=5) as resp:
                return str(resp.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError):
            return None

    def _read_served_context(self) -> int | None:
        from llb.backends.served_window import parse_ollama_served_context

        body = self._read_ps()
        return parse_ollama_served_context(body, self.model) if body is not None else None

    def served_context(self) -> int | None:
        return self._served_context

    def placement(self) -> OllamaPlacement | None:
        """The GPU/CPU weight split Ollama reports for this model right now (None until loaded)."""
        body = self._read_ps()
        return parse_ollama_placement(body, self.model) if body is not None else None

    def ensure_num_ctx(self, timeout: float = 120.0) -> int | None:
        """Warm-load so `/api/ps` reports the window this launcher will serve.

        Ollama keeps a previously loaded context until a request asks for a different one, and
        reports nothing at all until some request has loaded the model. Calling this before a
        budget probe makes the probe observe the window the run will actually use. It warms even
        with no `num_ctx` pinned, because THAT is the case the probe exists for: unpinned, Ollama
        serves its 4096 default however large a window the GGUF advertises, and a probe that reads
        "unknown" there leaves the declared window to bound a run it cannot bound.
        """
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
        if self.seed is not None:
            options["seed"] = self.seed
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
