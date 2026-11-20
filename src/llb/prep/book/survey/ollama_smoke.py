"""Read one page with a vision model the local Ollama daemon serves, and measure it.

The model runs in the daemon's own runner process, which the survey does not own, so its VRAM is
read the two ways that do not need the PID: Ollama's own placement (`/api/ps` `size_vram`, the
resident weights and cache) and the device-wide rise over the pre-call baseline. The request asks
for temperature 0, thinking off, and `keep_alive: 0`, so the model is unloaded the moment the page
is read and the next candidate starts from the same baseline.
"""

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.backends.telemetry_samplers import BackgroundSampler
from llb.core import env
from llb.core.config_validation import DEFAULT_OLLAMA_HOST
from llb.prep.book.survey.measure import BYTES_PER_MB, device_used_reader

NUM_CTX = 16384
NUM_PREDICT = 4096
PLACEMENT_INTERVAL_S = 0.5


@dataclass(frozen=True)
class OllamaReading:
    text: str
    seconds: float
    vram_resident_mb: float | None
    ram_resident_mb: float | None
    peak_vram_rise_mb: float | None
    error: str | None


def ollama_host() -> str:
    return os.environ.get(env.OLLAMA_HOST) or DEFAULT_OLLAMA_HOST


class _Placement(BackgroundSampler[None]):
    """The largest placement Ollama reports for `model` while it reads, plus the device rise."""

    def __init__(self, host: str, model: str, baseline_mb: int | None) -> None:
        super().__init__(self._read, PLACEMENT_INTERVAL_S)
        self.host, self.model, self.baseline_mb = host, model, baseline_mb
        self.device = device_used_reader() if baseline_mb is not None else None
        self.vram_mb: float | None = None
        self.ram_mb: float | None = None
        self.rise_mb: float | None = None

    def _read(self) -> None:
        import httpx

        for row in httpx.get(f"{self.host}/api/ps", timeout=5).json().get("models", []):
            if row.get("name") != self.model and row.get("model") != self.model:
                continue
            vram = float(row.get("size_vram", 0)) / BYTES_PER_MB
            total = float(row.get("size", 0)) / BYTES_PER_MB
            self.vram_mb = max(self.vram_mb or 0.0, round(vram, 1))
            self.ram_mb = max(self.ram_mb or 0.0, round(total - vram, 1))
        if self.device is not None and self.baseline_mb is not None:
            rise = float(max(self.device() - self.baseline_mb, 0))
            self.rise_mb = max(self.rise_mb or 0.0, rise)

    def _record(self, value: None) -> None:
        return None


def read_page(
    model: str, prompt: str, image: Path, timeout_s: float, baseline_mb: int | None
) -> OllamaReading:
    """Ask `model` to read `image`; never raises for a model or transport failure."""
    import httpx

    host = ollama_host()
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(image.read_bytes()).decode("ascii")],
        "stream": False,
        "keep_alive": 0,
        # A transcription needs no reasoning trace; with thinking on, a thinking model can spend
        # the whole token budget before writing any of the page (Qwen3-VL did, on a dense page).
        "think": False,
        "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
    }
    placement = _Placement(host, model, baseline_mb)
    start = time.monotonic()
    text, error = "", None
    with placement:
        try:
            response = httpx.post(f"{host}/api/generate", json=payload, timeout=timeout_s)
            response.raise_for_status()
            text = str(response.json().get("response", ""))
        except Exception as exc:  # the smoke records the failure instead of stopping the survey
            error = f"{type(exc).__name__}: {exc}"
    return OllamaReading(
        text=text,
        seconds=round(time.monotonic() - start, 2),
        vram_resident_mb=placement.vram_mb,
        ram_resident_mb=placement.ram_mb,
        peak_vram_rise_mb=placement.rise_mb,
        error=error,
    )
