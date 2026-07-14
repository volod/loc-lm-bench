"""Hard-isolation sweep executor (isolation reclaim).

The RAG core runner evaluates one (model, config) in-process. A multi-model sweep needs more: a
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
from pathlib import Path
from typing import Callable, TypeVar


from llb.core.config import RunConfig
from llb.core.contracts import CellResult, IsolationOutcome, SweepReport
from llb.executor.vram import (
    DEFAULT_TOLERANCE_MB,
    VERDICT_LEAKED,
    VERDICT_RECLAIMED,
    VramNotReclaimed,
    classify_residual,
    pids_held_mb,
    wait_for_reclaim,
)
from llb.executor.isolation_thermal import (
    DEFAULT_COOLDOWN_MAX_S,
    DEFAULT_COOLDOWN_TEMP_C,
    GpuSampler,
    _LOG,
    _persist_thermal,
    cool_down,
    sample_gpu,
)
from llb.executor.sweep_cells import (
    CellRunner,
    SWEEP_METHOD,
    _cell,
    _read_marker,
    _report,
    _subprocess_cell_runner,
    _write_marker,
    cell_key,
)


# Backends whose serving process owns its VRAM and frees it on exit -- so the reclaim gate is
# meaningful. Ollama is an external daemon that keeps models warm (keep-alive) by design, so a
# non-zero residual there is expected, not a leak; gating it would falsely abort the sweep.
GATE_BACKENDS = ("vllm", "llamacpp")

# Injectable seams.
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


def run_sweep(
    configs: list[RunConfig],
    *,
    sweep_id: str,
    split: str = "final",
    data_dir: Path | None = None,
    telemetry: bool = False,
    limit: int | None = None,
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
    runner = cell_runner or _subprocess_cell_runner(base_dir, sweep_id, telemetry, limit)
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
