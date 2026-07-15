"""Focused loop state implementation."""

import json
from dataclasses import dataclass
from pathlib import Path
from llb.bench.common import new_run_timestamp
from llb.core.config import RunConfig
from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.scoring.leaderboard import bootstrap_mean_ci

STATE_FILENAME = "state.json"

REPORT_FILENAME = "report.md"


@dataclass
class RoundReport:
    round_index: int
    dataset_dir: Path
    adapter_dir: Path
    final_run_dir: Path | None
    base_objective: float
    tuned_objective: float
    delta: float
    verdict: str
    base_ci: tuple[float, float] | None = None
    tuned_ci: tuple[float, float] | None = None

    def as_dict(self) -> JsonObject:
        return {
            "round": self.round_index,
            "dataset_dir": str(self.dataset_dir),
            "adapter_dir": str(self.adapter_dir),
            "final_run_dir": str(self.final_run_dir) if self.final_run_dir else None,
            "base_objective": self.base_objective,
            "tuned_objective": self.tuned_objective,
            "delta": self.delta,
            "verdict": self.verdict,
            "base_ci": self.base_ci,
            "tuned_ci": self.tuned_ci,
        }


def _default_out_dir(config: RunConfig) -> Path:
    _run_id, stamp = new_run_timestamp()
    return config.data_dir / "self-improve" / stamp


def _load_state(root: Path) -> JsonObject:
    path = root / STATE_FILENAME
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_state(root: Path, state: JsonObject) -> None:
    atomic_write_text(root / STATE_FILENAME, json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def _objective_from_run(run_dir: Path) -> float:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return float((manifest.get("metrics") or {}).get("objective_score", 0.0))


def _ci_from_run(run_dir: Path, *, seed: int) -> tuple[float, float] | None:
    values = []
    scores = run_dir / "scores.jsonl"
    if not scores.is_file():
        return None
    for line in scores.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            values.append(float(row.get("objective_score", 0.0)))
    return bootstrap_mean_ci(values, seed=seed)


def _ci_overlaps(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool:
    if a is None or b is None:
        return False
    return max(a[0], b[0]) <= min(a[1], b[1])


def _report_from_state(row: JsonObject) -> RoundReport:
    return RoundReport(
        round_index=int(row["round"]),
        dataset_dir=Path(str(row["dataset_dir"])),
        adapter_dir=Path(str(row["adapter_dir"])),
        final_run_dir=Path(str(row["final_run_dir"])) if row.get("final_run_dir") else None,
        base_objective=float(row["base_objective"]),
        tuned_objective=float(row["tuned_objective"]),
        delta=float(row["delta"]),
        verdict=str(row["verdict"]),
        base_ci=_ci_from_state(row.get("base_ci")),
        tuned_ci=_ci_from_state(row.get("tuned_ci")),
    )


def _ci_from_state(value: object) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    return (float(value[0]), float(value[1]))


def _write_report(root: Path, base_final_dir: Path, rounds: list[RoundReport]) -> None:
    lines = [
        "# Self-improvement report",
        "",
        f"Base final run: `{base_final_dir}`",
        "",
        "| round | base objective | tuned objective | delta | verdict | adapter |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rounds:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.round_index),
                    f"{row.base_objective:.4f}",
                    f"{row.tuned_objective:.4f}",
                    f"{row.delta:.4f}",
                    row.verdict,
                    f"`{row.adapter_dir}`",
                ]
            )
            + " |"
        )
    atomic_write_text(root / REPORT_FILENAME, "\n".join(lines) + "\n")


def _publish_run_pointer(dest: Path, target: Path) -> None:
    if dest.exists() or dest.is_symlink():
        return
    try:
        dest.symlink_to(target, target_is_directory=True)
    except OSError:
        dest.mkdir(parents=True, exist_ok=True)
        atomic_write_text(dest / "run_dir.txt", str(target) + "\n")
