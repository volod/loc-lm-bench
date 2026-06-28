"""Host GPU detection (nvidia-smi, stdlib only).

Used by model preparation and VRAM-fit checks. Parsing is split from the subprocess call
(`parse_smi`) so it is unit-testable without a GPU. No NVML / extras required -- this runs
on a bare host before any heavy dependency is installed.
"""

import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path

_SMI_QUERY = "name,memory.total,memory.free,driver_version"
_NVIDIA_SMI_CANDIDATES = ("nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi")


@dataclass
class Gpu:
    index: int
    name: str
    total_mb: int
    free_mb: int
    driver: str


def parse_smi(stdout: str) -> list[Gpu]:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` output."""
    gpus: list[Gpu] = []
    for i, line in enumerate(stdout.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, total, free, driver = parts[:4]
        try:
            gpus.append(Gpu(i, name, int(float(total)), int(float(free)), driver))
        except ValueError:
            continue
    return gpus


def _nvidia_smi_candidates() -> list[str]:
    """Executable candidates for nvidia-smi, de-duplicated in preference order."""
    candidates: list[str] = []
    resolved = shutil.which("nvidia-smi")
    if resolved is not None:
        candidates.append(resolved)
    candidates.extend(_NVIDIA_SMI_CANDIDATES)
    out: list[str] = []
    for candidate in candidates:
        if candidate not in out:
            out.append(candidate)
    return out


def detect_gpus() -> list[Gpu]:
    """Detect host GPUs via nvidia-smi. Returns [] when none / no driver."""
    for executable in _nvidia_smi_candidates():
        try:
            out = subprocess.run(
                [executable, f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
        if out.returncode != 0:
            continue
        gpus = parse_smi(out.stdout)
        if gpus:
            return gpus
    return []


def max_vram_mb(gpus: list[Gpu]) -> int:
    """Largest single-GPU total VRAM in MB (v1 targets single-GPU fit)."""
    return max((g.total_mb for g in gpus), default=0)


def select_target_gpu(gpus: list[Gpu], visible_devices: str | None = None) -> Gpu | None:
    """The GPU a single-GPU run targets (VRAM contention guard): the first `CUDA_VISIBLE_DEVICES` entry when set
    (matched by numeric index; a UUID falls through), else the GPU with the most FREE VRAM. None
    when no GPU is present. This is what the contention guard reads instead of always GPU 0."""
    if not gpus:
        return None
    if visible_devices:
        first = visible_devices.split(",")[0].strip()
        if first.isdigit():
            for gpu in gpus:
                if gpu.index == int(first):
                    return gpu
    return max(gpus, key=lambda g: g.free_mb)


def parse_meminfo(text: str) -> int:
    """Total system RAM in MB from /proc/meminfo contents (MemTotal is in kB)."""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            try:
                return int(line.split()[1]) // 1024
            except (IndexError, ValueError):
                return 0
    return 0


def detect_ram_mb() -> int:
    """Total host RAM in MB (Linux /proc/meminfo). Returns 0 if unavailable."""
    try:
        return parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
    except OSError:
        return 0
