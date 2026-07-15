"""Focused sweep cells implementation."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, cast
import yaml
from llb.core.config import RunConfig
from llb.core.contracts.hardware import CellResult, CoolDownReport, GpuSample, SweepReport
from llb.executor.isolation_thermal import (
    _LOG,
)

SWEEP_METHOD = "sweep"

CellRunner = Callable[[RunConfig, str], str]  # (config, split) -> published run dir

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


def _run_eval_command(cfg_path: Path, split: str, limit: int | None, telemetry: bool) -> list[str]:
    """The `run-eval` subprocess argv for one sweep cell."""
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
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if telemetry:
        cmd.append("--telemetry")
    return cmd


def _run_eval_dir_names(run_eval_root: Path) -> set[str]:
    """Names of the published run dirs currently under the run-eval root."""
    return {p.name for p in run_eval_root.glob("*")} if run_eval_root.exists() else set()


class _SubprocessCellRunner:
    """Default cell runner: run `run-eval` as its own process and return its published run dir."""

    def __init__(
        self, data_dir: Path, sweep_id: str, telemetry: bool, limit: int | None = None
    ) -> None:
        self._run_eval_root = data_dir / "run-eval"
        self._cfg_dir = data_dir / SWEEP_METHOD / sweep_id / "configs"
        self._cfg_dir.mkdir(parents=True, exist_ok=True)
        self._telemetry = telemetry
        self._limit = limit

    def __call__(self, config: RunConfig, split: str) -> str:
        cfg_path = self._cfg_dir / f"{cell_key(config)}.yaml"
        cfg_path.write_text(yaml.safe_dump(config.fingerprint(), sort_keys=True), encoding="utf-8")
        before = _run_eval_dir_names(self._run_eval_root)
        cmd = _run_eval_command(cfg_path, split, self._limit, self._telemetry)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            raise RuntimeError(f"cell run-eval exited {proc.returncode}: {' | '.join(tail)}")
        new = sorted(_run_eval_dir_names(self._run_eval_root) - before)
        return str(self._run_eval_root / new[-1]) if new else ""


def _subprocess_cell_runner(
    data_dir: Path, sweep_id: str, telemetry: bool, limit: int | None = None
) -> CellRunner:
    """Default cell runner factory (see `_SubprocessCellRunner`)."""
    return _SubprocessCellRunner(data_dir, sweep_id, telemetry, limit)


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
