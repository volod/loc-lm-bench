"""Orchestrate a sequential, resumable multi-model fine-tuning campaign."""

from pathlib import Path

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec
from llb.finetune.campaign.coerce import _parse_models
from llb.finetune.campaign.defaults import (
    _default_compat_fn,
    _default_eval_fn,
    _default_out_dir,
    _default_planner_fn,
    _default_reclaim_fn,
)
from llb.finetune.campaign.entry import _run_campaign_entry
from llb.finetune.campaign.model import (
    CampaignResult,
    CompatFn,
    EvalFn,
    PlannerFn,
    ReclaimFn,
    TrainerFn,
    _CampaignHooks,
)
from llb.finetune.campaign.report import _write_report
from llb.finetune.campaign.state import (
    _append_entry,
    _existing_shared_dataset,
    _read_completed_entries,
)
from llb.finetune.loop import _default_trainer_fn


def run_finetune_campaign(
    config: RunConfig,
    *,
    models: list[str],
    rounds: int,
    out_dir: Path | str | None = None,
    resume: Path | str | None = None,
    trainer: str = "auto",
    limit: int | None = None,
    model_specs: list[ModelSpec] | None = None,
    eval_fn: EvalFn | None = None,
    trainer_fn: TrainerFn | None = None,
    planner_fn: PlannerFn | None = None,
    reclaim_fn: ReclaimFn | None = None,
    compat_fn: CompatFn | None = None,
) -> CampaignResult:
    """Run a sequential, resumable adapter campaign for a roster of local models."""
    roster = _parse_models(models)
    if not roster:
        raise ValueError("finetune campaign requires at least one model")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    hooks = _CampaignHooks(
        eval_fn=eval_fn or _default_eval_fn(limit=limit),
        trainer_fn=trainer_fn or _default_trainer_fn(config, trainer),
        planner_fn=planner_fn or _default_planner_fn(model_specs or []),
        reclaim_fn=reclaim_fn or _default_reclaim_fn(),
        compat_fn=compat_fn or _default_compat_fn(),
    )

    root = Path(resume) if resume is not None else Path(out_dir or _default_out_dir(config))
    root.mkdir(parents=True, exist_ok=True)
    done = _read_completed_entries(root)
    entries = list(done.values())
    shared_dataset_dir = _existing_shared_dataset(root)

    for model_index, model in enumerate(roster):
        if model in done:
            continue
        entry, shared_dataset_dir = _run_campaign_entry(
            model,
            config,
            root,
            rounds,
            hooks,
            shared_dataset_dir,
            reclaim_must_raise=model_index < len(roster) - 1,
        )
        _append_entry(root, entry)
        entries.append(entry)

    _write_report(root, entries, shared_dataset_dir)
    return CampaignResult(root, entries, shared_dataset_dir)
