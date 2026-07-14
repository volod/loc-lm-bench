"""Focused isolation thermal implementation."""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Callable
from llb.core.contracts import CoolDownReport, GpuSample

_LOG = logging.getLogger(__name__)

DEFAULT_COOLDOWN_TEMP_C = 55

DEFAULT_COOLDOWN_MAX_S = 120.0

_SMI_QUERY = "index,temperature.gpu,power.draw,clocks.sm,clocks.mem"

GpuSampler = Callable[[], list[GpuSample]]


def parse_smi_samples(stdout: str) -> list[GpuSample]:
    """Parse `nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks.sm,clocks.mem`."""
    samples: list[GpuSample] = []
    for line in stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        def _int(value: str) -> int | None:
            try:
                return int(float(value))
            except ValueError:
                return None

        def _float(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None

        idx = _int(parts[0])
        if idx is None:
            continue
        samples.append(
            {
                "index": idx,
                "temp_c": _int(parts[1]),
                "power_w": _float(parts[2]),
                "sm_clock_mhz": _int(parts[3]),
                "mem_clock_mhz": _int(parts[4]),
            }
        )
    return samples


def sample_gpu() -> list[GpuSample]:
    """Sample temp/power/clocks via nvidia-smi. Returns [] when no GPU / no driver."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return parse_smi_samples(out.stdout) if out.returncode == 0 else []


def _hottest(samples: list[GpuSample]) -> int | None:
    temps = [s["temp_c"] for s in samples if s["temp_c"] is not None]
    return max(temps) if temps else None


def cool_down(
    threshold_c: int = DEFAULT_COOLDOWN_TEMP_C,
    max_wait_s: float = DEFAULT_COOLDOWN_MAX_S,
    poll_s: float = 2.0,
    sampler: GpuSampler | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> CoolDownReport:
    """Wait until the hottest GPU is <= `threshold_c`, capped at `max_wait_s`."""
    sampler = sampler or sample_gpu
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    start = clock()
    temp = _hottest(sampler())
    while temp is not None and temp > threshold_c:
        now = clock()
        if now - start >= max_wait_s:
            return {"waited_s": round(now - start, 1), "final_temp_c": temp, "capped": True}
        sleep(poll_s)
        temp = _hottest(sampler())
    return {"waited_s": round(clock() - start, 1), "final_temp_c": temp, "capped": False}


def _persist_thermal(
    run_dir: str | None, cd: CoolDownReport, gpu: list[GpuSample], threshold_c: int
) -> None:
    if not run_dir:
        return
    path = Path(run_dir)
    if not path.exists():
        return
    record = {
        "cooldown_s": cd["waited_s"],
        "cooldown_capped": cd["capped"],  # True -> the GPU stayed hot past the cap (elevated)
        "final_temp_c": cd["final_temp_c"],
        "threshold_c": threshold_c,
        "gpu": gpu,
    }
    (path / "thermal.json").write_text(json.dumps(record), encoding="utf-8")
