"""VRAM reclaim gate (basic, Milestone 1).

The sequential-execution contract: after a backend is killed, VRAM must return to the
pre-run baseline within a TOLERANCE band (drivers/display hold a little, so the exact
byte is never hit). Residual above tolerance means the killed backend leaked, which would
bias the next run -- so we raise `VramNotReclaimed` and abort loudly. A baseline shift
from unrelated desktop processes is tolerated, not aborted.

`pynvml` is imported lazily (the `[telemetry]` extra). The reader and sleep are injectable
so the gate logic is unit-testable without a GPU.
"""

from typing import Callable

from llb.contracts import VramReclaimReport

DEFAULT_TOLERANCE_MB = 512
DEFAULT_MAX_POLLS = 30


class VramNotReclaimed(RuntimeError):
    """Raised when freed VRAM stays above the baseline tolerance after a run."""


def nvml_reader() -> Callable[[], int]:
    """A reader returning total used VRAM (MB) across GPUs. Needs the `[telemetry]` extra."""
    try:
        import pynvml
    except ImportError as exc:
        raise SystemExit(
            "ERROR: VRAM telemetry needs the [telemetry] extra. "
            'Run: uv pip install -e ".[telemetry]"'
        ) from exc

    pynvml.nvmlInit()

    def read() -> int:
        total = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            total += pynvml.nvmlDeviceGetMemoryInfo(handle).used
        return total // (1024 * 1024)

    return read


def read_baseline(reader: Callable[[], int] | None = None) -> int:
    """Current used VRAM (MB) -- snapshot before a run starts."""
    reader = reader or nvml_reader()
    return reader()


def wait_for_reclaim(
    baseline_mb: int,
    reader: Callable[[], int] | None = None,
    tolerance_mb: int = DEFAULT_TOLERANCE_MB,
    max_polls: int = DEFAULT_MAX_POLLS,
    poll_s: float = 1.0,
    sleep: Callable[[float], None] | None = None,
) -> VramReclaimReport:
    """Poll until used VRAM <= baseline + tolerance, or polls run out.

    Returns {reclaimed, residual_mb, polls}. Does not raise; the caller decides whether a
    non-reclaim aborts the suite.
    """
    import time

    reader = reader or nvml_reader()
    sleep = sleep or time.sleep
    residual = reader() - baseline_mb
    polls = 1
    while residual > tolerance_mb and polls < max_polls:
        sleep(poll_s)
        residual = reader() - baseline_mb
        polls += 1
    return {"reclaimed": residual <= tolerance_mb, "residual_mb": residual, "polls": polls}


def assert_reclaimed(
    baseline_mb: int,
    reader: Callable[[], int] | None = None,
    tolerance_mb: int = DEFAULT_TOLERANCE_MB,
    max_polls: int = DEFAULT_MAX_POLLS,
    poll_s: float = 1.0,
    sleep: Callable[[float], None] | None = None,
) -> VramReclaimReport:
    """Run the gate and raise `VramNotReclaimed` if VRAM did not return to baseline."""
    result = wait_for_reclaim(
        baseline_mb,
        reader=reader,
        tolerance_mb=tolerance_mb,
        max_polls=max_polls,
        poll_s=poll_s,
        sleep=sleep,
    )
    if not result["reclaimed"]:
        raise VramNotReclaimed(
            f"VRAM residual {result['residual_mb']} MB exceeds tolerance after "
            f"{result['polls']} polls; aborting to avoid biasing the next run."
        )
    return result
