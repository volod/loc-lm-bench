"""Host GPU detection (nvidia-smi, stdlib only).

Used by model preparation and VRAM-fit checks. Parsing is split from the subprocess call
(`parse_smi`) so it is unit-testable without a GPU. No NVML / extras required -- this runs
on a bare host before any heavy dependency is installed.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

_SMI_QUERY = "name,memory.total,memory.free,driver_version"


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


def detect_gpus() -> list[Gpu]:
    """Detect host GPUs via nvidia-smi. Returns [] when none / no driver."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return parse_smi(out.stdout)


def max_vram_mb(gpus: list[Gpu]) -> int:
    """Largest single-GPU total VRAM in MB (v1 targets single-GPU fit)."""
    return max((g.total_mb for g in gpus), default=0)


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
