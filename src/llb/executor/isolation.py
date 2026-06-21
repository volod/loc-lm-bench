"""Hard-isolation sweep executor (M3.3).

The M1 runner evaluates one (model, config) in-process. A multi-model sweep needs more: a
leak or crash in one cell must not bias the next, and a hot GPU must cool before the next
throughput measurement. So each cell runs in its OWN PROCESS (`vllm serve` already dies with
its launcher; the whole `run-eval` running as a subprocess guarantees the Python/CUDA context
is gone too), and between cells we:

  1. VRAM gate -- wait for used VRAM to return to the pre-cell baseline within tolerance;
     abort the whole sweep loudly on `VramNotReclaimed` (a leak would bias every later cell).
  2. Thermal cooldown -- wait for the GPU to drop below a temperature threshold, capped at a
     max wait so a warm room cannot stall the sweep forever.
  3. Record temp / clocks / power for the cell (throughput is only comparable at like clocks).

The sweep is RESUMABLE: each cell has a stable key (a hash of its reproducibility-relevant
config) and writes a marker on completion, so a re-run skips finished cells. Every
side-effect (the per-cell subprocess, the NVML reader, the GPU sampler, sleep) is injectable,
so the loop is unit-tested without a GPU or a real subprocess.
"""

import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import yaml

from llb.config import RunConfig
from llb.contracts import CellResult, CoolDownReport, GpuSample, SweepReport
from llb.executor.vram import DEFAULT_TOLERANCE_MB, assert_reclaimed

_LOG = logging.getLogger(__name__)

SWEEP_METHOD = "sweep"
DEFAULT_COOLDOWN_TEMP_C = 55
DEFAULT_COOLDOWN_MAX_S = 120.0
_SMI_QUERY = "index,temperature.gpu,power.draw,clocks.sm,clocks.mem"

# Backends whose serving process owns its VRAM and frees it on exit -- so the reclaim gate is
# meaningful. Ollama is an external daemon that keeps models warm (keep-alive) by design, so a
# non-zero residual there is expected, not a leak; gating it would falsely abort the sweep.
GATE_BACKENDS = ("vllm", "llamacpp")

# Injectable seams.
CellRunner = Callable[[RunConfig, str], str]  # (config, split) -> published run dir
GpuSampler = Callable[[], list[GpuSample]]


def cell_key(config: RunConfig) -> str:
    """Stable id for a (model, config) cell: a hash of its reproducibility-relevant fields.

    `run_name` is excluded so a relabeled but otherwise identical cell still resumes.
    """
    fp = dict(config.fingerprint())
    fp.pop("run_name", None)
    blob = json.dumps(fp, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


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


def _subprocess_cell_runner(data_dir: Path, sweep_id: str, telemetry: bool) -> CellRunner:
    """Default cell runner: run `run-eval` as its own process and return its published run dir."""
    run_eval_root = data_dir / "run-eval"
    cfg_dir = data_dir / SWEEP_METHOD / sweep_id / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    def run(config: RunConfig, split: str) -> str:
        cfg_path = cfg_dir / f"{cell_key(config)}.yaml"
        cfg_path.write_text(yaml.safe_dump(config.fingerprint(), sort_keys=True), encoding="utf-8")
        before = {p.name for p in run_eval_root.glob("*")} if run_eval_root.exists() else set()
        cmd = [
            sys.executable,
            "-m",
            "llb.main",
            "run-eval",
            "--config",
            str(cfg_path),
            "--split",
            split,
        ]
        if telemetry:
            cmd.append("--telemetry")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            raise RuntimeError(f"cell run-eval exited {proc.returncode}: {' | '.join(tail)}")
        after = {p.name for p in run_eval_root.glob("*")} if run_eval_root.exists() else set()
        new = sorted(after - before)
        return str(run_eval_root / new[-1]) if new else ""

    return run


def run_sweep(
    configs: list[RunConfig],
    *,
    sweep_id: str,
    split: str = "final",
    data_dir: Path | None = None,
    telemetry: bool = False,
    resume: bool = True,
    vram_tolerance_mb: int = DEFAULT_TOLERANCE_MB,
    cooldown_temp_c: int = DEFAULT_COOLDOWN_TEMP_C,
    cooldown_max_s: float = DEFAULT_COOLDOWN_MAX_S,
    cell_runner: CellRunner | None = None,
    vram_reader: Callable[[], int] | None = None,
    gpu_sampler: GpuSampler | None = None,
    sleep: Callable[[float], None] | None = None,
) -> SweepReport:
    """Run each (model, config) cell in isolation, gating VRAM + thermals between cells.

    Aborts loudly on `VramNotReclaimed` (a leaked cell would bias every later one). Completed
    cells write a marker under ``$DATA_DIR/sweep/<sweep_id>/cells/`` so `resume=True` skips them.
    """
    base_dir = (data_dir or configs[0].data_dir) if configs else (data_dir or Path(".data"))
    cells_dir = base_dir / SWEEP_METHOD / sweep_id / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    runner = cell_runner or _subprocess_cell_runner(base_dir, sweep_id, telemetry)
    sampler = gpu_sampler or sample_gpu

    results: list[CellResult] = []
    for config in configs:
        key = cell_key(config)
        marker = cells_dir / f"{key}.json"
        if resume and marker.exists():
            prior: CellResult = json.loads(marker.read_text(encoding="utf-8"))
            prior["status"] = "skipped"
            _LOG.info("[sweep] skip (done) %s %s/%s", key, config.backend, config.model)
            results.append(prior)
            continue

        _LOG.info("[sweep] cell %s %s/%s", key, config.backend, config.model)
        gate = vram_reader is not None and config.backend in GATE_BACKENDS
        baseline = vram_reader() if (gate and vram_reader is not None) else None
        try:
            run_dir = runner(config, split)
        except Exception as exc:
            results.append(_cell(key, config, "failed", None, None, None, sampler(), str(exc)))
            continue

        if gate and baseline is not None:
            report = assert_reclaimed(  # raises VramNotReclaimed -> abort the whole sweep
                baseline, reader=vram_reader, tolerance_mb=vram_tolerance_mb, sleep=sleep
            )
            residual = report["residual_mb"]
        else:
            residual = None
        cd = cool_down(cooldown_temp_c, cooldown_max_s, sampler=sampler, sleep=sleep)
        result = _cell(key, config, "done", run_dir, residual, cd, sampler(), "ok")
        marker.write_text(json.dumps(result), encoding="utf-8")
        results.append(result)

    return _report(sweep_id, results)


def _cell(
    key: str,
    config: RunConfig,
    status: str,
    run_dir: str | None,
    residual_mb: int | None,
    cooldown: CoolDownReport | None,
    gpu: list[GpuSample],
    detail: str,
) -> CellResult:
    return {
        "cell_key": key,
        "model": config.model,
        "backend": config.backend,
        "status": status,
        "run_dir": run_dir,
        "vram_residual_mb": residual_mb,
        "cooldown_s": cooldown["waited_s"] if cooldown else 0.0,
        "cooldown_capped": cooldown["capped"] if cooldown else False,
        "gpu": gpu,
        "detail": detail,
    }


def _report(sweep_id: str, results: list[CellResult]) -> SweepReport:
    return {
        "sweep_id": sweep_id,
        "n_cells": len(results),
        "completed": sum(1 for r in results if r["status"] == "done"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }
