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

import functools
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, TypeVar, cast

import yaml

from llb.config import RunConfig
from llb.contracts import CellResult, CoolDownReport, GpuSample, IsolationOutcome, SweepReport
from llb.executor.vram import (
    DEFAULT_TOLERANCE_MB,
    VERDICT_LEAKED,
    VERDICT_RECLAIMED,
    VramNotReclaimed,
    classify_residual,
    pids_held_mb,
    wait_for_reclaim,
)

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
PidUsageReader = Callable[[], dict[int, int]]  # {pid: used VRAM MB} for leak attribution
_T = TypeVar("_T")


def isolate_cell(
    work: Callable[[], _T],
    *,
    backend: str,
    vram_reader: Callable[[], int] | None = None,
    pid_usage_reader: PidUsageReader | None = None,
    gpu_sampler: GpuSampler | None = None,
    sleep: Callable[[float], None] | None = None,
    vram_tolerance_mb: int = DEFAULT_TOLERANCE_MB,
    cooldown_temp_c: int = DEFAULT_COOLDOWN_TEMP_C,
    cooldown_max_s: float = DEFAULT_COOLDOWN_MAX_S,
) -> tuple[_T, IsolationOutcome]:
    """Run `work()` under the per-cell isolation contract, the single reusable primitive shared
    by the sweep, the public screen, and every Optuna trial.

    For VRAM-owning backends (`GATE_BACKENDS`): snapshot the VRAM baseline + the set of PIDs
    already holding VRAM, run `work`, then wait for reclaim. If VRAM does NOT return to baseline,
    ATTRIBUTE the residual via `classify_residual`: a PID that appeared during the cell and still
    holds VRAM is a `leaked` cell (abort with `VramNotReclaimed`); a pre-existing process that
    grew is a `baseline_shift` (tolerated). Without a `pid_usage_reader` the gate is conservative
    (any over-tolerance residual aborts). Finally apply the capped thermal cooldown.
    """
    sampler = gpu_sampler or sample_gpu
    gate = vram_reader is not None and backend in GATE_BACKENDS
    pids_before: set[int] = set()
    baseline: int | None = None
    if gate and vram_reader is not None:
        if pid_usage_reader is not None:
            pids_before = set(pid_usage_reader())
        baseline = vram_reader()

    out = work()  # the cell itself (a subprocess run-eval, a screen, an in-process trial...)

    residual: int | None = None
    verdict: str | None = None
    if gate and vram_reader is not None and baseline is not None:
        report = wait_for_reclaim(
            baseline, reader=vram_reader, tolerance_mb=vram_tolerance_mb, sleep=sleep
        )
        residual = report["residual_mb"]
        if report["reclaimed"]:
            verdict = VERDICT_RECLAIMED
        elif pid_usage_reader is not None:
            usage = pid_usage_reader()
            leaked = sorted(set(usage) - pids_before)  # PIDs that appeared during the cell
            held = pids_held_mb(usage, set(leaked))
            verdict = classify_residual(residual, held, vram_tolerance_mb)
            if verdict == VERDICT_LEAKED:
                raise VramNotReclaimed(
                    f"cell leaked ~{held} MB still held by launched PID(s) {leaked}; "
                    "aborting to avoid biasing later cells."
                )
            _LOG.warning(
                "[isolate] VRAM residual %d MB attributed to a baseline shift (unrelated "
                "process), not the launched cell -- tolerated.",
                residual,
            )
        else:  # no PID attribution available -> conservative: an over-tolerance residual aborts
            verdict = VERDICT_LEAKED
            raise VramNotReclaimed(
                f"VRAM residual {residual} MB exceeds tolerance (no PID attribution); aborting."
            )

    cooldown = cool_down(cooldown_temp_c, cooldown_max_s, sampler=sampler, sleep=sleep)
    outcome: IsolationOutcome = {
        "vram_residual_mb": residual,
        "vram_verdict": verdict,
        "cooldown": cooldown,
        "gpu": sampler(),
    }
    return out, outcome


_MARKER_KEYS = frozenset(CellResult.__required_keys__)


def _read_marker(path: Path) -> CellResult | None:
    """Read a completed-cell marker; a truncated marker is treated as unfinished work."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("[sweep] ignore unreadable marker %s: %s", path, exc)
        return None
    if (
        not isinstance(value, dict)
        or value.get("status") != "done"
        or not _MARKER_KEYS.issubset(value)
    ):
        _LOG.warning("[sweep] ignore invalid marker %s", path)
        return None
    return cast(CellResult, value)


def _write_marker(path: Path, result: CellResult) -> None:
    """Publish a completion marker atomically so interruption cannot create a false resume."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp:
            json.dump(result, temp)
            temp_path = Path(temp.name)
        temp_path.replace(path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


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
    pid_usage_reader: PidUsageReader | None = None,
    gpu_sampler: GpuSampler | None = None,
    sleep: Callable[[float], None] | None = None,
) -> SweepReport:
    """Run each (model, config) cell in isolation, gating VRAM + thermals between cells.

    Each cell runs through `isolate_cell`, so an unreclaimed-VRAM leak (attributed to the
    launched PID tree when `pid_usage_reader` is given) aborts the whole sweep loudly, while an
    unrelated baseline shift is tolerated. Completed cells write a marker under
    ``$DATA_DIR/sweep/<sweep_id>/cells/`` so `resume=True` skips them.
    """
    if not configs:
        return _report(sweep_id, [])
    base_dir = data_dir or configs[0].data_dir
    cells_dir = base_dir / SWEEP_METHOD / sweep_id / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    runner = cell_runner or _subprocess_cell_runner(base_dir, sweep_id, telemetry)
    sampler = gpu_sampler or sample_gpu

    results: list[CellResult] = []
    for config in configs:
        key = cell_key(config)
        marker = cells_dir / f"{key}.json"
        if resume and marker.exists():
            prior = _read_marker(marker)
            if prior is not None:
                prior["status"] = "skipped"
                _LOG.info("[sweep] skip (done) %s %s/%s", key, config.backend, config.model)
                results.append(prior)
                continue

        _LOG.info("[sweep] cell %s %s/%s", key, config.backend, config.model)
        try:
            run_dir, iso = isolate_cell(
                functools.partial(runner, config, split),
                backend=config.backend,
                vram_reader=vram_reader,
                pid_usage_reader=pid_usage_reader,
                gpu_sampler=sampler,
                sleep=sleep,
                vram_tolerance_mb=vram_tolerance_mb,
                cooldown_temp_c=cooldown_temp_c,
                cooldown_max_s=cooldown_max_s,
            )
        except VramNotReclaimed:
            raise  # a leaked cell would bias every later one -> abort the whole sweep
        except Exception as exc:  # the cell itself failed (e.g. run-eval subprocess) -> record it
            results.append(_cell(key, config, "failed", None, None, None, sampler(), str(exc)))
            continue

        # Persist the thermal flag into the canonical run BUNDLE (not only the sweep marker), so a
        # capped/elevated cooldown is visible to anyone reading the run -- e.g. the board.
        _persist_thermal(run_dir, iso["cooldown"], iso["gpu"], cooldown_temp_c)
        result = _cell(
            key, config, "done", run_dir, iso["vram_residual_mb"], iso["cooldown"], iso["gpu"], "ok"
        )
        _write_marker(marker, result)
        results.append(result)

    return _report(sweep_id, results)


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
