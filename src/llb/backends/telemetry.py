"""Per-backend telemetry: steady-state throughput, peak VRAM, served context (telemetry hook).

Tokens/sec is measured at STEADY STATE -- a fixed prompt set + fixed max_new_tokens + N
warmup iterations -- so it is comparable across models (the design's throughput protocol).
Cold-start load time is recorded separately by the launcher (`load_time_s`), never conflated
with steady-state throughput. Peak VRAM is sampled via NVML during the run.

The throughput measurement is backend-agnostic (it drives `launcher.chat`), so it works for
any launcher and is unit-testable with a fake. NVML is injected (`vram_reader`), so this
module imports in the base install; the GPU sampler needs the `[telemetry]` extra.
"""

import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Callable

from llb.backends.base import ChatResult
from llb.contracts import ChatMessage, GpuSummary, TelemetryReport
from llb.prompts import render_text_list

# Fixed Ukrainian prompts for the throughput protocol (comparable across models).
DEFAULT_THROUGHPUT_PROMPTS = render_text_list("telemetry.throughput")


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


@dataclass(frozen=True)
class _PowerTelemetry:
    mean_power_w: float
    peak_power_w: float
    power_samples: int
    tokens_per_watt: float


@dataclass(frozen=True)
class _SamplerTelemetry:
    sampler: str
    flashinfer_version: str | None


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


def nvidia_smi_power_reader() -> Callable[[], float | None] | None:
    """Return a reader for total GPU power draw in watts, or None when unavailable."""

    def read() -> float | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        values: list[float] = []
        for line in out.stdout.strip().splitlines():
            try:
                values.append(float(line.strip()))
            except ValueError:
                continue
        return sum(values) if values else None

    return read if read() is not None else None


class PowerSampler:
    """Track total GPU power draw while telemetry prompts are running."""

    def __init__(self, reader: Callable[[], float | None] | None, interval: float = 0.5):
        self.reader = reader
        self.interval = interval
        self.samples: list[float] = []
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def sample(self) -> float | None:
        if self.reader is None:
            return None
        value = self.reader()
        if value is not None:
            self.samples.append(value)
        return value

    def _loop(self) -> None:
        while self._stop is not None and not self._stop.is_set():
            try:
                self.sample()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "PowerSampler":
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

    @property
    def mean_w(self) -> float | None:
        return sum(self.samples) / len(self.samples) if self.samples else None

    @property
    def peak_w(self) -> float | None:
        return max(self.samples) if self.samples else None


def _required_report(
    tput: ThroughputResult,
    launcher: Any,
    sampler: VramSampler,
    requested_context: int | None,
) -> TelemetryReport:
    meta = _launcher_meta(launcher)
    return {
        "steady_tokens_per_s": round(tput.steady_tokens_per_s, 2),
        "mean_completion_tokens": round(tput.mean_completion_tokens, 1),
        "tokens_per_char": round(tput.tokens_per_char, 4),
        "max_new_tokens": tput.max_new_tokens,
        "n_warmup": tput.n_warmup,
        "n_measured": tput.n_measured,
        "n_failed": tput.n_failed,
        "load_time_s": _rounded_load_time(launcher),
        "peak_vram_mb": sampler.peak_mb or None,
        "requested_context": requested_context,
        "served_context": _served_context(launcher),
        "backend": meta.get("backend"),
        "gpu_memory_utilization": meta.get("gpu_memory_utilization"),
        "n_gpu_layers": meta.get("n_gpu_layers"),
        "gpus": _gpu_summary(),
    }


def _launcher_meta(launcher: Any) -> Any:
    meta = getattr(launcher, "meta", None)
    return meta if hasattr(meta, "get") else {}


def _rounded_load_time(launcher: Any) -> float | None:
    load_time = getattr(launcher, "load_time_s", None)
    return round(load_time, 2) if isinstance(load_time, int | float) else None


def _served_context(launcher: Any) -> int | None:
    served = getattr(launcher, "served_context", None)
    return served() if callable(served) else None


def _power_report(power: PowerSampler, tput: ThroughputResult) -> _PowerTelemetry | None:
    mean_power = power.mean_w
    peak_power = power.peak_w
    if mean_power is None:
        return None
    return _PowerTelemetry(
        mean_power_w=round(mean_power, 2),
        peak_power_w=round(peak_power, 2) if peak_power is not None else 0.0,
        power_samples=len(power.samples),
        tokens_per_watt=(
            round(tput.steady_tokens_per_s / mean_power, 4) if mean_power > 0 else 0.0
        ),
    )


def _sampler_report(launcher: Any) -> _SamplerTelemetry | None:
    meta = _launcher_meta(launcher)
    if meta.get("sampler") is None:
        return None
    return _SamplerTelemetry(
        sampler=str(meta["sampler"]),
        flashinfer_version=meta.get("flashinfer_version"),
    )


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
    power_reader: Callable[[], float | None] | None = None,
) -> TelemetryReport:
    """Measure throughput (peak-VRAM-sampled) and assemble the manifest telemetry record."""
    with VramSampler(vram_reader) as sampler, PowerSampler(power_reader) as power:
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
        if power_reader is not None:
            power.sample()
    report = _required_report(tput, launcher, sampler, requested_context)
    power_report = _power_report(power, tput)
    if power_report is not None:
        report["mean_power_w"] = power_report.mean_power_w
        report["peak_power_w"] = power_report.peak_power_w
        report["power_samples"] = power_report.power_samples
        report["tokens_per_watt"] = power_report.tokens_per_watt
    sampler_report = _sampler_report(launcher)
    if sampler_report is not None:
        report["sampler"] = sampler_report.sampler
        report["flashinfer_version"] = sampler_report.flashinfer_version
    return report


def _gpu_summary() -> list[GpuSummary]:
    """Detected GPU name/VRAM/driver for reproducibility (best-effort, stdlib nvidia-smi)."""
    try:
        from llb.backends.hardware import detect_gpus

        return [{"name": g.name, "total_mb": g.total_mb, "driver": g.driver} for g in detect_gpus()]
    except Exception:
        return []
