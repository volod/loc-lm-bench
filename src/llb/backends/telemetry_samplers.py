"""Focused telemetry samplers implementation."""

import subprocess
import threading
from typing import Callable


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
