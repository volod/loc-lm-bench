"""What one candidate's smoke on one page produced: its status, measurements, and reading."""

import re
from dataclasses import dataclass, field
from typing import Any

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
STATUS_OVER_BUDGET = "over-vram-budget"
STATUS_NOT_INSTALLED = "not-installed"
STATUS_NOT_RUN = "not-run"
# A CUDA allocation failure, or vLLM refusing to start because its reservation leaves too little
# KV cache for the configured context -- both mean the model does not fit as configured.
OOM = re.compile(
    r"out of memory|OutOfMemoryError|larger than the available KV cache memory", re.IGNORECASE
)


@dataclass
class SmokeResult:
    candidate: str
    page: str
    status: str
    seconds: float | None = None
    peak_rss_mb: float | None = None
    peak_vram_tree_mb: float | None = None
    peak_vram_rise_mb: float | None = None
    vram_resident_mb: float | None = None
    vram_residual_mb: int | None = None
    score: dict[str, Any] | None = None
    reading: str | None = None  # the text while running; its file name once written
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
