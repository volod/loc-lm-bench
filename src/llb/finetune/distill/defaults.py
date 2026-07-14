"""Default (real-backend) implementations of the injectable teacher/trainer/comparison seams."""

from pathlib import Path

from llb.bench.common import new_run_timestamp
from llb.core.config import RunConfig
from llb.core.contracts import EvalResult, JsonObject
from llb.core.fsutil import atomic_write_text
from llb.eval import common as eval_common
from llb.finetune.distill.model import (
    DISTILL_METHOD,
    DistillComparison,
    TeacherResponse,
    TrainerFn,
)
from llb.finetune.trainer import train_adapter
from llb.goldset.schema import GoldItem
from llb.scoring.leaderboard import bootstrap_mean_ci


def _default_trainer_fn(config: RunConfig, trainer: str) -> TrainerFn:
    from llb.finetune.hparam_search.manifest_io import trainer_defaults

    def train(dataset_dir: Path, model: str, adapter_dir: Path, seed: int) -> JsonObject:
        return train_adapter(
            dataset_dir=dataset_dir,
            model=model,
            out_dir=adapter_dir,
            seed=seed,
            trainer=trainer,
            **trainer_defaults(config.data_dir, model),
        )

    return train


def _default_teacher_fn(
    config: RunConfig, items: list[GoldItem], root: Path
) -> list[TeacherResponse]:
    from llb.executor.runner_backend import _resolve_eval_runner

    staging = root / "teacher-backend"
    staging.mkdir(parents=True, exist_ok=True)
    launcher, runner_fn, _store, _contention = _resolve_eval_runner(
        config,
        store=None,
        launcher=None,
        runner_fn=None,
        prompt_package=None,
        staging_dir=staging,
        evict=False,
        wait=False,
    )
    responses: list[TeacherResponse] = []
    with launcher:
        for item in items:
            state = runner_fn(item)
            responses.append(
                TeacherResponse(
                    item_id=item.id,
                    answer=str(state.get("answer") or ""),
                    status=str(state.get("status") or eval_common.OK),
                    context=str(state.get("context") or ""),
                    retrieved=tuple(state.get("retrieved") or ()),
                )
            )
    return responses


def _default_comparison_fn(
    config: RunConfig,
    adapter_dir: Path,
    reference_adapter_dir: Path,
    items: list[GoldItem],
    out_dir: Path,
) -> DistillComparison:
    from llb.executor.runner import run_eval

    out_dir.mkdir(parents=True, exist_ok=True)
    split = items[0].split
    distilled = run_eval(
        config.with_overrides(adapter_path=adapter_dir),
        items=items,
        split=split,
        emit=False,
    )
    reference = run_eval(
        config.with_overrides(adapter_path=reference_adapter_dir),
        items=items,
        split=split,
        emit=False,
    )
    distilled_run_dir = _run_dir_from_eval(distilled)
    reference_run_dir = _run_dir_from_eval(reference)
    atomic_write_text(out_dir / "distilled_run.txt", str(distilled_run_dir) + "\n")
    atomic_write_text(out_dir / "reference_run.txt", str(reference_run_dir) + "\n")
    distilled_scores = _scores_from_eval(distilled)
    reference_scores = _scores_from_eval(reference)
    distilled_objective = _objective_from_eval(distilled)
    reference_objective = _objective_from_eval(reference)
    return DistillComparison(
        split=split,
        n_items=len(items),
        distilled_objective=distilled_objective,
        reference_objective=reference_objective,
        delta=distilled_objective - reference_objective,
        distilled_ci=bootstrap_mean_ci(distilled_scores, seed=config.seed),
        reference_ci=bootstrap_mean_ci(reference_scores, seed=config.seed + 1),
        distilled_run_dir=distilled_run_dir,
        reference_run_dir=reference_run_dir,
    )


def _run_dir_from_eval(result: EvalResult) -> Path:
    return Path(result["paths"]["manifest"]).parent


def _scores_from_eval(result: EvalResult) -> list[float]:
    scores: list[float] = []
    for row in result["rows"]:
        value = row.get("objective_score", 0.0)
        scores.append(float(value) if isinstance(value, int | float | str) else 0.0)
    return scores


def _objective_from_eval(result: EvalResult) -> float:
    return float(result["metrics"].get("objective_score", 0.0))


def _default_out_dir(config: RunConfig) -> Path:
    _run_id, stamp = new_run_timestamp()
    return config.data_dir / DISTILL_METHOD / stamp
