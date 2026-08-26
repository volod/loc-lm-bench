"""Focused telemetry samplers implementation.

Every sampler here answers the same question -- what was true WHILE the generations ran, not
after them -- so they share one polling contract: an injected reader, a daemon thread that runs
for the length of a `with` block, and a swallowed read error (a transient NVML/HTTP failure must
never crash a run). Only what each one keeps from the readings differs.
"""

import subprocess
import threading
from typing import Callable, Generic, TypeVar

_T = TypeVar("_T")


class BackgroundSampler(Generic[_T]):
    """Poll an injected reader on a daemon thread for the length of a `with` block.

    A `None` reader makes the whole sampler a no-op, so a caller on a host without the signal
    (no NVML, no power readout, a backend that cannot report placement) needs no branch.
    """

    def __init__(self, reader: Callable[[], _T] | None, interval: float = 0.5):
        self.reader = reader
        self.interval = interval
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def _record(self, value: _T) -> None:
        """Keep what this sampler keeps from one reading."""
        raise NotImplementedError

    def sample(self) -> _T | None:
        if self.reader is None:
            return None
        value = self.reader()
        self._record(value)
        return value

    def _loop(self) -> None:
        while self._stop is not None and not self._stop.is_set():
            try:
                self.sample()
            except Exception:  # a transient reader error must not crash the run
                pass
            self._stop.wait(self.interval)

    def __enter__(self):  # type: ignore[no-untyped-def]
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


class VramSampler(BackgroundSampler[int]):
    """Track peak used VRAM (MB) by polling an injected reader in a background thread."""

    def __init__(self, reader: Callable[[], int] | None, interval: float = 0.5):
        super().__init__(reader, interval)
        self.peak_mb = 0

    def _record(self, value: int) -> None:
        self.peak_mb = max(self.peak_mb, value)

    def sample(self) -> int:
        return super().sample() or 0


class LastValueSampler(BackgroundSampler[_T]):
    """Keep the LAST non-None reading -- for a signal that can vanish before the run ends.

    Ollama's scheduler may evict and reload a runner mid-pass on a host where the model barely
    fits, so a placement read taken only after the last generation can find nothing resident. What
    the run was served by is what was observed WHILE it generated.
    """

    def __init__(self, reader: Callable[[], _T] | None, interval: float = 1.0):
        super().__init__(reader, interval)
        self.value: _T | None = None

    def _record(self, value: _T) -> None:
        if value is not None:
            self.value = value


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


class PowerSampler(BackgroundSampler[float | None]):
    """Track total GPU power draw while telemetry prompts are running."""

    def __init__(self, reader: Callable[[], float | None] | None, interval: float = 0.5):
        super().__init__(reader, interval)
        self.samples: list[float] = []

    def _record(self, value: float | None) -> None:
        if value is not None:
            self.samples.append(value)

    @property
    def mean_w(self) -> float | None:
        return sum(self.samples) / len(self.samples) if self.samples else None

    @property
    def peak_w(self) -> float | None:
        return max(self.samples) if self.samples else None
