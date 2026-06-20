"""Per-backend telemetry: steady-state throughput, peak VRAM, served context (M2.2).

Tokens/sec is measured at STEADY STATE -- a fixed prompt set + fixed max_new_tokens + N
warmup iterations -- so it is comparable across models (the design's throughput protocol).
Cold-start load time is recorded separately by the launcher (`load_time_s`), never conflated
with steady-state throughput. Peak VRAM is sampled via NVML during the run.

The throughput measurement is backend-agnostic (it drives `launcher.chat`), so it works for
any launcher and is unit-testable with a fake. NVML is injected (`vram_reader`), so this
module imports in the base install; the GPU sampler needs the `[telemetry]` extra.
"""

import threading
from dataclasses import dataclass
from typing import Any, Callable

from llb.backends.base import ChatResult
from llb.contracts import ChatMessage, GpuSummary, TelemetryReport

# Fixed Ukrainian prompts for the throughput protocol (comparable across models).
DEFAULT_THROUGHPUT_PROMPTS = [
    "Поясни своїми словами, що таке авторське право і навіщо воно потрібне.",
    "Стисло опиши основні кроки реєстрації торговельної марки в Україні.",
    "Назви типові обов'язки сторін у договорі про надання послуг.",
]


def tokens_per_char(token_count: int, text: str) -> float:
    """Tokenizer efficiency: generated tokens per character (UA tokenizes differently)."""
    n = len(text)
    return token_count / n if n else 0.0


@dataclass
class ThroughputResult:
    steady_tokens_per_s: float
    mean_completion_tokens: float
    tokens_per_char: float
    n_measured: int
    n_failed: int
    n_warmup: int
    max_new_tokens: int


def measure_throughput(
    chat: Callable[[list[ChatMessage], int, float, float], ChatResult],
    prompts: list[str] | None = None,
    *,
    max_new_tokens: int = 128,
    warmup: int = 1,
    passes: int = 1,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> ThroughputResult:
    """Run warmup then timed generations over a fixed prompt set; report steady-state rate.

    `chat(messages, max_tokens, temperature, timeout) -> ChatResult` (a launcher's `chat`).
    """
    prompts = prompts or DEFAULT_THROUGHPUT_PROMPTS

    def call(prompt: str) -> ChatResult:
        messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
        return chat(messages, max_new_tokens, temperature, timeout)

    for _ in range(warmup):  # warm caches / JIT; results discarded
        for prompt in prompts:
            call(prompt)

    total_tokens = total_latency = total_chars = 0.0
    measured = failed = 0
    for _ in range(passes):
        for prompt in prompts:
            result = call(prompt)
            if result.error or result.completion_tokens <= 0:
                failed += 1
                continue
            total_tokens += result.completion_tokens
            total_latency += result.latency_s
            total_chars += len(result.text)
            measured += 1

    rate = total_tokens / total_latency if total_latency > 0 else 0.0
    mean_tokens = total_tokens / measured if measured else 0.0
    tpc = total_tokens / total_chars if total_chars else 0.0
    return ThroughputResult(
        steady_tokens_per_s=rate,
        mean_completion_tokens=mean_tokens,
        tokens_per_char=tpc,
        n_measured=measured,
        n_failed=failed,
        n_warmup=warmup,
        max_new_tokens=max_new_tokens,
    )


class VramSampler:
    """Track peak used VRAM (MB) by polling an injected reader in a background thread."""

    def __init__(self, reader: Callable[[], int] | None, interval: float = 0.5):
        self.reader = reader
        self.interval = interval
        self.peak_mb = 0
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def sample(self) -> int:
        if self.reader is None:
            return 0
        value = self.reader()
        self.peak_mb = max(self.peak_mb, value)
        return value

    def _loop(self) -> None:
        while self._stop is not None and not self._stop.is_set():
            try:
                self.sample()
            except Exception:  # a transient NVML error must not crash the run
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "VramSampler":
        if self.reader is not None:
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def collect_telemetry(
    launcher: Any,
    *,
    prompts: list[str] | None = None,
    max_new_tokens: int = 128,
    warmup: int = 1,
    temperature: float = 0.0,
    timeout: float = 120.0,
    requested_context: int | None = None,
    vram_reader: Callable[[], int] | None = None,
) -> TelemetryReport:
    """Measure throughput (peak-VRAM-sampled) and assemble the manifest telemetry record."""
    with VramSampler(vram_reader) as sampler:
        tput = measure_throughput(
            launcher.chat,
            prompts,
            max_new_tokens=max_new_tokens,
            warmup=warmup,
            temperature=temperature,
            timeout=timeout,
        )
        if vram_reader is not None:  # guarantee >= 1 reading even for very short runs
            sampler.sample()
    report: TelemetryReport = {
        "steady_tokens_per_s": round(tput.steady_tokens_per_s, 2),
        "mean_completion_tokens": round(tput.mean_completion_tokens, 1),
        "tokens_per_char": round(tput.tokens_per_char, 4),
        "max_new_tokens": tput.max_new_tokens,
        "n_warmup": tput.n_warmup,
        "n_measured": tput.n_measured,
        "n_failed": tput.n_failed,
        "load_time_s": round(getattr(launcher, "load_time_s", 0.0), 2),
        "peak_vram_mb": sampler.peak_mb or None,
        "requested_context": requested_context,
        "served_context": launcher.served_context()
        if hasattr(launcher, "served_context")
        else None,
        "backend": launcher.meta.get("backend") if hasattr(launcher, "meta") else None,
        "gpu_memory_utilization": (
            launcher.meta.get("gpu_memory_utilization") if hasattr(launcher, "meta") else None
        ),
        "gpus": _gpu_summary(),
    }
    return report


def _gpu_summary() -> list[GpuSummary]:
    """Detected GPU name/VRAM/driver for reproducibility (best-effort, stdlib nvidia-smi)."""
    try:
        from llb.backends.hardware import detect_gpus

        return [{"name": g.name, "total_mb": g.total_mb, "driver": g.driver} for g in detect_gpus()]
    except Exception:
        return []
