"""VRAM reclaim gate (basic, RAG core).

The sequential-execution contract: after a backend is killed, VRAM must return to the
pre-run baseline within a TOLERANCE band (drivers/display hold a little, so the exact
byte is never hit). Residual above tolerance means the killed backend leaked, which would
bias the next run -- so we raise `VramNotReclaimed` and abort loudly. A baseline shift
from unrelated desktop processes is tolerated, not aborted.

`pynvml` is imported lazily (the `[telemetry]` extra). The reader and sleep are injectable
so the gate logic is unit-testable without a GPU.
"""

from typing import Callable

from llb.core.contracts import VramReclaimReport

DEFAULT_TOLERANCE_MB = 512
DEFAULT_MAX_POLLS = 30

# Residual classification (isolation reclaim): distinguish a real leak from an unrelated baseline shift.
VERDICT_RECLAIMED = "reclaimed"
VERDICT_LEAKED = "leaked"  # residual is held by the launched process tree -> abort
VERDICT_BASELINE_SHIFT = "baseline_shift"  # an unrelated process grew -> tolerate


class VramNotReclaimed(RuntimeError):
    """Raised when freed VRAM stays above the baseline tolerance after a run."""


def classify_residual(
    residual_mb: int, pid_held_mb: int, tolerance_mb: int = DEFAULT_TOLERANCE_MB
) -> str:
    """Attribute a post-run VRAM residual (isolation reclaim).

    A residual within tolerance is `reclaimed`. Above tolerance, it is a `leaked` cell only when
    the LAUNCHED process tree still holds VRAM (`pid_held_mb` above tolerance) -- the killed
    backend did not free its memory; otherwise it is a `baseline_shift` (an unrelated desktop
    process grew) and must be tolerated, not aborted.
    """
    if residual_mb <= tolerance_mb:
        return VERDICT_RECLAIMED
    return VERDICT_LEAKED if pid_held_mb > tolerance_mb else VERDICT_BASELINE_SHIFT


def pids_held_mb(usage: dict[int, int], pids: set[int]) -> int:
    """Total VRAM (MB) held by `pids` (and their listed children) in a {pid: used_mb} map."""
    return sum(mb for pid, mb in usage.items() if pid in pids)


def nvml_process_reader() -> Callable[[], dict[int, int]]:
    """A reader returning {pid: used VRAM MB} across GPUs, for PID attribution. Needs `[telemetry]`."""
    try:
        import pynvml
    except ImportError as exc:
        raise SystemExit(
            'ERROR: VRAM attribution needs the [telemetry] extra. uv pip install -e ".[telemetry]"'
        ) from exc

    pynvml.nvmlInit()

    def read() -> dict[int, int]:
        usage: dict[int, int] = {}
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                mb = (proc.usedGpuMemory or 0) // (1024 * 1024)
                usage[proc.pid] = usage.get(proc.pid, 0) + mb
        return usage

    return read


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
