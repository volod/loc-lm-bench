"""Self-improvement campaign orchestration.

The orchestration is resumable through `state.json`: each round advances through tuning eval,
miss analysis, export, train, and final eval. Heavy collaborators are injectable, which lets CI
exercise the loop with fake trainers/evaluators while production uses `run-eval`.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from llb.bench.common import new_run_timestamp
from llb.board.miss_analysis import analyze_run, load_item_provenance, write_analysis
from llb.core.config import RunConfig
from llb.core.contracts import EvalResult, JsonObject
from llb.core.fsutil import atomic_write_text
from llb.finetune.dataset import export_finetune_set
from llb.finetune.registry import registry_path, try_register_adapter
from llb.finetune.trainer import train_adapter
from llb.goldset.schema import load_goldset
from llb.scoring.aggregate import bootstrap_mean_ci

STATE_FILENAME = "state.json"
REPORT_FILENAME = "report.md"
MIN_GAIN = 0.0

EvalFn = Callable[[RunConfig, str, Path], EvalResult]
TrainerFn = Callable[[Path, str, Path, int], JsonObject]


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


@dataclass
class SelfImproveResult:
    out_dir: Path
    base_final_run_dir: Path
    rounds: list[RoundReport] = field(default_factory=list)
    verdict: str = "reject"


def run_self_improve(
    config: RunConfig,
    *,
    rounds: int,
    out_dir: Path | str | None = None,
    trainer: str = "auto",
    min_gain: float = MIN_GAIN,
    limit: int | None = None,
    eval_fn: EvalFn | None = None,
    trainer_fn: TrainerFn | None = None,
    resume: Path | str | None = None,
) -> SelfImproveResult:
    """Run a local model self-improvement campaign."""
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    eval_fn = eval_fn or _default_eval_fn(limit=limit)
    trainer_fn = trainer_fn or (
        lambda dataset, model, adapter, seed: train_adapter(
            dataset_dir=dataset, model=model, out_dir=adapter, seed=seed, trainer=trainer
        )
    )
    root = Path(resume) if resume is not None else Path(out_dir or _default_out_dir(config))
    root.mkdir(parents=True, exist_ok=True)
    state = _load_state(root)

    base_final_raw = state.get("base_final_run_dir")
    base_final_dir = Path(str(base_final_raw)) if base_final_raw else None
    if base_final_dir is None:
        base_result = eval_fn(config, "final", root / "base-final")
        base_final_dir = Path(base_result["paths"]["manifest"]).parent
        state["base_final_run_dir"] = str(base_final_dir)
        _write_state(root, state)
    base_objective = _objective_from_run(base_final_dir)
    base_ci = _ci_from_run(base_final_dir, seed=config.seed)
    reports = [_report_from_state(row) for row in state.get("rounds", [])]

    current_cfg = config
    if reports:
        latest_adapter = reports[-1].adapter_dir
        current_cfg = current_cfg.with_overrides(adapter_path=latest_adapter)
    for round_index in range(len(reports) + 1, rounds + 1):
        round_dir = root / f"round-{round_index}"
        round_dir.mkdir(parents=True, exist_ok=True)
        tuning = eval_fn(current_cfg, "tuning", round_dir / "run")
        tuning_dir = Path(tuning["paths"]["manifest"]).parent
        _publish_run_pointer(round_dir / "run", tuning_dir)
        miss_dir = round_dir / "miss-analysis"
        analysis = analyze_run(
            tuning_dir,
            load_goldset(config.goldset_path),
            provenance=load_item_provenance(config.goldset_path),
        )
        miss_paths = write_analysis(analysis, miss_dir)
        dataset_dir = round_dir / "dataset"
        export_finetune_set(
            run_dir=tuning_dir,
            goldset_path=config.goldset_path,
            out_dir=dataset_dir,
            misses_path=miss_paths["misses"],
        )
        adapter_dir = round_dir / "adapter"
        adapter_manifest = trainer_fn(dataset_dir, config.model, adapter_dir, config.seed)
        tuned_cfg = config.with_overrides(adapter_path=adapter_dir)
        final = eval_fn(tuned_cfg, "final", round_dir / "run-final")
        final_dir = Path(final["paths"]["manifest"]).parent
        _publish_run_pointer(round_dir / "run-final", final_dir)
        tuned_objective = _objective_from_run(final_dir)
        tuned_ci = _ci_from_run(final_dir, seed=config.seed + round_index)
        delta = tuned_objective - base_objective
        verdict = "accept" if delta > min_gain and not _ci_overlaps(base_ci, tuned_ci) else "reject"
        register_round_adapter(
            config,
            adapter_dir=adapter_dir,
            source_run=tuning_dir,
            eval_summary={
                "final_run_dir": str(final_dir),
                "objective_score": tuned_objective,
                "base_objective": base_objective,
                "delta": delta,
                "verdict": verdict,
            },
        )
        report = RoundReport(
            round_index=round_index,
            dataset_dir=dataset_dir,
            adapter_dir=adapter_dir,
            final_run_dir=final_dir,
            base_objective=base_objective,
            tuned_objective=tuned_objective,
            delta=delta,
            verdict=verdict,
            base_ci=base_ci,
            tuned_ci=tuned_ci,
        )
        reports.append(report)
        state["rounds"] = [row.as_dict() for row in reports]
        state["last_adapter_digest"] = adapter_manifest.get("adapter_digest")
        _write_state(root, state)
        _write_report(root, base_final_dir, reports)
        _write_report(round_dir, base_final_dir, [report])
        if verdict == "reject":
            break
        current_cfg = tuned_cfg

    verdict = "accept" if reports and reports[-1].verdict == "accept" else "reject"
    _write_report(root, base_final_dir, reports)
    return SelfImproveResult(root, base_final_dir, reports, verdict)


def register_round_adapter(
    config: RunConfig,
    *,
    adapter_dir: Path,
    source_run: Path,
    eval_summary: JsonObject,
) -> None:
    """Make a freshly trained adapter a first-class, traceable artifact.

    Registration happens after the adapter's own final eval, so the entry carries the evidence the
    board later cites, plus the goldset/corpus digests that decide staleness.
    """
    try_register_adapter(
        registry=registry_path(config.data_dir),
        adapter_dir=adapter_dir,
        goldset_path=config.goldset_path,
        corpus_root=config.corpus_root,
        source_run=source_run,
        eval_summary=eval_summary,
    )


def _default_eval_fn(*, limit: int | None) -> EvalFn:
    from llb.executor.runner import run_eval

    def run(config: RunConfig, split: str, _round_run_dir: Path) -> EvalResult:
        return run_eval(config, split=split, limit=limit)

    return run


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
