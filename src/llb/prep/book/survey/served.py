"""A smoke against the survey's own vLLM container: start it, run the smoke, always stop it.

The container's VRAM belongs to no process the smoke owns, so it is read as the device-wide rise
over the baseline taken before the container started, sampled across load and reading alike.
"""

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from llb.prep.book.survey import vllm_server
from llb.prep.book.survey.candidates import Candidate, Smoke
from llb.prep.book.survey.fixture import SmokePage
from llb.prep.book.survey.measure import DeviceRiseSampler
from llb.prep.book.survey.results import (
    OOM,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_OVER_BUDGET,
    SmokeResult,
)

SERVER_LOG_NAME = "server.log"

Execute = Callable[
    [Candidate, Smoke, SmokePage, Path, Path, dict[str, str], int | None], SmokeResult
]


def run_served_read(candidate: Candidate, smoke: Smoke, page: SmokePage, base: str) -> SmokeResult:
    """One chat request to the smoke's own vLLM server."""
    import httpx

    start, text, error = time.monotonic(), "", None
    try:
        text = vllm_server.read_page(
            base,
            smoke.model or "",
            smoke.prompt or "",
            page.image,
            smoke.timeout_s,
            smoke.max_image_side,
        )
    except (httpx.HTTPError, vllm_server.ServerError, KeyError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    ok = error is None and bool(text.strip())
    return SmokeResult(
        candidate=candidate.id,
        page=page.label,
        status=STATUS_OK if ok else STATUS_FAILED,
        seconds=round(time.monotonic() - start, 2),
        reading=text,
        error=error or (None if ok else "empty response"),
    )


def run_with_server(
    candidate: Candidate,
    smoke: Smoke,
    page: SmokePage,
    out_dir: Path,
    data_dir: Path,
    values: dict[str, str],
    baseline: int | None,
    execute: Execute,
) -> SmokeResult:
    """Start the smoke's vLLM container, run the smoke against it, and always stop it."""
    assert smoke.server is not None
    server_log = out_dir / SERVER_LOG_NAME
    start, sampler = time.monotonic(), DeviceRiseSampler(baseline)
    try:
        with sampler:
            with vllm_server.running_server(
                smoke.server, candidate.id, data_dir, server_log
            ) as base:
                load_s = round(time.monotonic() - start, 1)
                result = execute(
                    candidate, smoke, page, out_dir, data_dir, {**values, "server": base}, baseline
                )
    except (vllm_server.ServerError, subprocess.TimeoutExpired) as exc:
        log = (
            server_log.read_text(encoding="utf-8", errors="replace") if server_log.exists() else ""
        )
        status = STATUS_OVER_BUDGET if OOM.search(log) else STATUS_FAILED
        return SmokeResult(
            candidate.id,
            page.label,
            status,
            seconds=round(time.monotonic() - start, 2),
            peak_vram_rise_mb=sampler.peak_mb,
            error=f"{exc}; server log tail: {log[-1200:]}",
        )
    result.extra["server_load_s"] = load_s
    log = server_log.read_text(encoding="utf-8", errors="replace") if server_log.exists() else ""
    if result.status == STATUS_FAILED and OOM.search(log):
        result.status = STATUS_OVER_BUDGET  # the server died of memory mid-read (HTTP 500)
    result.peak_vram_rise_mb = max(result.peak_vram_rise_mb or 0.0, sampler.peak_mb or 0.0) or None
    return result
