"""Self-improvement campaign orchestration.

The orchestration is resumable through `state.json`: each round advances through tuning eval,
miss analysis, export, train, and final eval. Heavy collaborators are injectable, which lets CI
exercise the loop with fake trainers/evaluators while production uses `run-eval`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from llb.board.miss_analysis.classify import analyze_run
from llb.board.miss_analysis.load import load_item_provenance
from llb.board.miss_analysis.report import write_analysis
from llb.core.config import RunConfig
from llb.core.contracts.runs import EvalResult
from llb.core.contracts.common import JsonObject
from llb.finetune.dataset import export_finetune_set
from llb.finetune.registry.io import registry_path
from llb.finetune.registry.register import try_register_adapter
from llb.finetune.trainer import train_adapter
from llb.goldset.schema import load_goldset
from llb.finetune.loop_state import (
    RoundReport,
    _ci_from_run,
    _ci_overlaps,
    _default_out_dir,
    _load_state,
    _objective_from_run,
    _publish_run_pointer,
    _report_from_state,
    _write_report,
    _write_state,
)

MIN_GAIN = 0.0

EvalFn = Callable[[RunConfig, str, Path], EvalResult]
TrainerFn = Callable[[Path, str, Path, int], JsonObject]


@dataclass
class SelfImproveResult:
    out_dir: Path
    base_final_run_dir: Path
    rounds: list[RoundReport] = field(default_factory=list)
    verdict: str = "reject"


@dataclass(frozen=True)
class _LoopContext:
    """Fixed collaborators + baseline shared by every self-improvement round."""

    config: RunConfig
    eval_fn: EvalFn
    trainer_fn: TrainerFn
    base_objective: float
    base_ci: tuple[float, float] | None
    min_gain: float


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
    trainer_fn = trainer_fn or _default_trainer_fn(config, trainer)
    root = Path(resume) if resume is not None else Path(out_dir or _default_out_dir(config))
    root.mkdir(parents=True, exist_ok=True)
    state = _load_state(root)

    base_final_dir = _ensure_base_final(state, root, config, eval_fn)
    ctx = _LoopContext(
        config=config,
        eval_fn=eval_fn,
        trainer_fn=trainer_fn,
        base_objective=_objective_from_run(base_final_dir),
        base_ci=_ci_from_run(base_final_dir, seed=config.seed),
        min_gain=min_gain,
    )
    reports = [_report_from_state(row) for row in state.get("rounds", [])]

    current_cfg = config
    if reports:
        current_cfg = current_cfg.with_overrides(adapter_path=reports[-1].adapter_dir)
    for round_index in range(len(reports) + 1, rounds + 1):
        round_dir = root / f"round-{round_index}"
        round_dir.mkdir(parents=True, exist_ok=True)
        report, adapter_digest = _run_round(ctx, current_cfg, round_index, round_dir)
        reports.append(report)
        state["rounds"] = [row.as_dict() for row in reports]
        state["last_adapter_digest"] = adapter_digest
        _write_state(root, state)
        _write_report(root, base_final_dir, reports)
        _write_report(round_dir, base_final_dir, [report])
        if report.verdict == "reject":
            break
        current_cfg = config.with_overrides(adapter_path=report.adapter_dir)

    verdict = "accept" if reports and reports[-1].verdict == "accept" else "reject"
    _write_report(root, base_final_dir, reports)
    return SelfImproveResult(root, base_final_dir, reports, verdict)


def _ensure_base_final(state: JsonObject, root: Path, config: RunConfig, eval_fn: EvalFn) -> Path:
    """The base model's final-split run dir, evaluating (and recording) it on first call."""
    base_final_raw = state.get("base_final_run_dir")
    if base_final_raw:
        return Path(str(base_final_raw))
    base_result = eval_fn(config, "final", root / "base-final")
    base_final_dir = Path(base_result["paths"]["manifest"]).parent
    state["base_final_run_dir"] = str(base_final_dir)
    _write_state(root, state)
    return base_final_dir


def _run_round(
    ctx: _LoopContext, current_cfg: RunConfig, round_index: int, round_dir: Path
) -> tuple[RoundReport, object]:
    """One round: tuning eval -> miss analysis -> export -> train -> final eval -> verdict."""
    config = ctx.config
    tuning = ctx.eval_fn(current_cfg, "tuning", round_dir / "run")
    tuning_dir = Path(tuning["paths"]["manifest"]).parent
    _publish_run_pointer(round_dir / "run", tuning_dir)
    analysis = analyze_run(
        tuning_dir,
        load_goldset(config.goldset_path),
        provenance=load_item_provenance(config.goldset_path),
    )
    miss_paths = write_analysis(analysis, round_dir / "miss-analysis")
    dataset_dir = round_dir / "dataset"
    export_finetune_set(
        run_dir=tuning_dir,
        goldset_path=config.goldset_path,
        out_dir=dataset_dir,
        misses_path=miss_paths["misses"],
    )
    adapter_dir = round_dir / "adapter"
    adapter_manifest = ctx.trainer_fn(dataset_dir, config.model, adapter_dir, config.seed)
    tuned_cfg = config.with_overrides(adapter_path=adapter_dir)
    final = ctx.eval_fn(tuned_cfg, "final", round_dir / "run-final")
    final_dir = Path(final["paths"]["manifest"]).parent
    _publish_run_pointer(round_dir / "run-final", final_dir)
    tuned_objective = _objective_from_run(final_dir)
    tuned_ci = _ci_from_run(final_dir, seed=config.seed + round_index)
    delta = tuned_objective - ctx.base_objective
    accepted = delta > ctx.min_gain and not _ci_overlaps(ctx.base_ci, tuned_ci)
    verdict = "accept" if accepted else "reject"
    register_round_adapter(
        config,
        adapter_dir=adapter_dir,
        source_run=tuning_dir,
        eval_summary={
            "final_run_dir": str(final_dir),
            "objective_score": tuned_objective,
            "base_objective": ctx.base_objective,
            "delta": delta,
            "verdict": verdict,
        },
    )
    report = RoundReport(
        round_index=round_index,
        dataset_dir=dataset_dir,
        adapter_dir=adapter_dir,
        final_run_dir=final_dir,
        base_objective=ctx.base_objective,
        tuned_objective=tuned_objective,
        delta=delta,
        verdict=verdict,
        base_ci=ctx.base_ci,
        tuned_ci=tuned_ci,
    )
    return report, adapter_manifest.get("adapter_digest")


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
        index_dir=config.index_dir(),
        source_run=source_run,
        eval_summary=eval_summary,
    )


def _default_trainer_fn(config: RunConfig, trainer: str) -> TrainerFn:
    """Train through the seam, defaulting hyperparameters to this model's recorded search."""
    from llb.finetune.hparam_search.manifest_io import trainer_defaults

    def train(dataset: Path, model: str, adapter: Path, seed: int) -> JsonObject:
        return train_adapter(
            dataset_dir=dataset,
            model=model,
            out_dir=adapter,
            seed=seed,
            trainer=trainer,
            **trainer_defaults(config.data_dir, model),
        )

    return train


def _default_eval_fn(*, limit: int | None) -> EvalFn:
    from llb.executor.runner import run_eval

    def run(config: RunConfig, split: str, _round_run_dir: Path) -> EvalResult:
        return run_eval(config, split=split, limit=limit)

    return run
