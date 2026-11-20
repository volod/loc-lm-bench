"""Run one smoke command and measure what it held: wall time, peak RAM, and peak VRAM.

RAM is the peak resident set of the command's whole process tree (a converter forks workers).
VRAM is read two ways, because the two failure modes differ: the peak held by the tree's own PIDs
(exact for an in-process engine), and the peak rise of device-wide used memory over the baseline
taken just before the command (the only reading for an engine served by another process -- an
Ollama runner or a model server -- which the tree never owns). A reading is None when the host
cannot provide it (no `[telemetry]` extra, no NVIDIA device), never a guessed zero.
"""

import logging
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.backends.telemetry_samplers import BackgroundSampler

_LOG = logging.getLogger(__name__)

SAMPLE_INTERVAL_S = 0.25
BYTES_PER_MB = 1024 * 1024
# How long VRAM may take to come back after a command (or an Ollama unload) before the residual is
# recorded as held; the next candidate's baseline is taken after this.
RECLAIM_WAIT_S = 20.0
RECLAIM_TOLERANCE_MB = 256
OUTPUT_TAIL_CHARS = 4000


@dataclass(frozen=True)
class Usage:
    """What one smoke command held, plus how it ended."""

    seconds: float
    peak_rss_mb: float | None
    peak_vram_tree_mb: float | None
    peak_vram_rise_mb: float | None
    returncode: int | None
    timed_out: bool
    output_tail: str


def device_used_reader() -> Callable[[], int] | None:
    """Total used VRAM (MB) across devices, or None without NVML."""
    try:
        from llb.executor.vram import nvml_reader

        reader = nvml_reader()
        reader()
        return reader
    except (Exception, SystemExit):
        return None


def _pid_reader() -> Callable[[], dict[int, int]] | None:
    try:
        from llb.executor.vram import nvml_process_reader

        return nvml_process_reader()
    except (Exception, SystemExit):
        return None


def _tree(pid: int) -> list[object]:
    """The psutil processes of `pid` and its descendants (empty once it has exited)."""
    try:
        import psutil

        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except Exception:
        return []


class _TreeSampler(BackgroundSampler[None]):
    """Peak RSS of a process tree, its PIDs' VRAM, and device-wide used VRAM."""

    def __init__(self, pid: int, baseline_mb: int | None) -> None:
        super().__init__(self._read, SAMPLE_INTERVAL_S)
        self.pid = pid
        self.baseline_mb = baseline_mb
        self.device = device_used_reader() if baseline_mb is not None else None
        self.per_pid = _pid_reader()
        self.peak_rss_mb: float | None = None
        self.peak_tree_mb: float | None = None
        self.peak_rise_mb: float | None = None

    def _read(self) -> None:
        procs = _tree(self.pid)
        rss = 0
        for proc in procs:
            try:
                rss += proc.memory_info().rss  # type: ignore[attr-defined]
            except Exception:
                continue
        if procs:
            self.peak_rss_mb = max(self.peak_rss_mb or 0.0, rss / BYTES_PER_MB)
        if self.per_pid is not None:
            pids = {getattr(proc, "pid", -1) for proc in procs}
            held = sum(mb for pid, mb in self.per_pid().items() if pid in pids)
            self.peak_tree_mb = max(self.peak_tree_mb or 0.0, float(held))
        if self.device is not None and self.baseline_mb is not None:
            rise = max(self.device() - self.baseline_mb, 0)
            self.peak_rise_mb = max(self.peak_rise_mb or 0.0, float(rise))

    def _record(self, value: None) -> None:
        return None


class DeviceRiseSampler(BackgroundSampler[None]):
    """Peak device-wide VRAM rise over `baseline_mb`, for a model served by another process."""

    def __init__(self, baseline_mb: int | None) -> None:
        self.device = device_used_reader() if baseline_mb is not None else None
        super().__init__(self._read if self.device is not None else None, SAMPLE_INTERVAL_S)
        self.baseline_mb = baseline_mb
        self.peak_mb: float | None = None

    def _read(self) -> None:
        if self.device is not None and self.baseline_mb is not None:
            rise = float(max(self.device() - self.baseline_mb, 0))
            self.peak_mb = max(self.peak_mb or 0.0, rise)

    def _record(self, value: None) -> None:
        return None


def run_measured(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
    log_path: Path,
    baseline_mb: int | None,
) -> Usage:
    """Run `argv` to completion (or `timeout_s`), logging its output, and measure it."""
    start = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            list(argv), cwd=cwd, env=dict(env), stdout=log, stderr=subprocess.STDOUT
        )
        sampler = _TreeSampler(proc.pid, baseline_mb)
        with sampler:
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                for child in reversed(_tree(proc.pid)):
                    try:
                        child.kill()  # type: ignore[attr-defined]
                    except Exception:
                        continue
                proc.kill()
                proc.wait()
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-OUTPUT_TAIL_CHARS:]
    return Usage(
        seconds=round(time.monotonic() - start, 2),
        peak_rss_mb=_round(sampler.peak_rss_mb),
        peak_vram_tree_mb=_round(sampler.peak_tree_mb),
        peak_vram_rise_mb=_round(sampler.peak_rise_mb),
        returncode=None if timed_out else proc.returncode,
        timed_out=timed_out,
        output_tail=tail,
    )


def wait_for_reclaim(baseline_mb: int | None) -> int | None:
    """Wait until device VRAM is back near `baseline_mb`; returns the residual (MB) left."""
    reader = device_used_reader()
    if reader is None or baseline_mb is None:
        return None
    deadline = time.monotonic() + RECLAIM_WAIT_S
    residual = max(reader() - baseline_mb, 0)
    while residual > RECLAIM_TOLERANCE_MB and time.monotonic() < deadline:
        time.sleep(1.0)
        residual = max(reader() - baseline_mb, 0)
    if residual > RECLAIM_TOLERANCE_MB:
        _LOG.warning("[survey] %d MB of VRAM still held after the smoke", residual)
    return residual


def device_total_mb() -> int | None:
    """The largest device's total memory (MB): the VRAM budget a candidate must fit."""
    try:
        from llb.backends.hardware import detect_gpus, max_vram_mb

        gpus = detect_gpus()
        return max_vram_mb(gpus) if gpus else None
    except Exception:
        return None


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)
