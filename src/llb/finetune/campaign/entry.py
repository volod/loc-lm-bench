"""Per-model campaign execution: skip pre-probes, the round loop, and entry finalization."""

import json
import time
from pathlib import Path

from llb.backends.planner import VERDICT_NO
from llb.board.miss_analysis.classify import analyze_run
from llb.board.miss_analysis.load import load_item_provenance
from llb.board.miss_analysis.report import write_analysis
from llb.core.config import RunConfig
from llb.core.contracts import JsonObject, ModelPlanRow
from llb.finetune.campaign.model import (
    COMPLETE_VERDICT,
    SHARED_DATASET_DIRNAME,
    SKIP_VERDICT,
    CampaignEntry,
    EvalFn,
    _CampaignHooks,
    _RoundsOutcome,
)
from llb.finetune.compat import VERDICT_NOT_TRAINABLE as COMPAT_NOT_TRAINABLE
from llb.finetune.dataset import DATASET_MANIFEST, export_finetune_set
from llb.finetune.loop import (
    _ci_from_run,
    _objective_from_run,
    _publish_run_pointer,
    register_round_adapter,
)
from llb.finetune.naming import model_slug
from llb.goldset.schema import load_goldset


def _skip_entry(model: str, hooks: _CampaignHooks, plan: ModelPlanRow) -> CampaignEntry | None:
    """Planner / trainability pre-probes: return a skip entry, or None to proceed.

    Compressed-QAT trainability pre-probe (compressed-qat-adapter-support): a checkpoint
    whose native quantization scheme has no PEFT dispatch is skipped WITH the exact blocker
    before its base eval or any training run is paid for. Only a positive not-trainable
    verdict skips; an unknown/unreachable config lets the entry proceed.
    """
    planner_payload = dict(plan)
    if str(plan.get("verdict")) == VERDICT_NO:
        return CampaignEntry(
            model=model,
            status=SKIP_VERDICT,
            reason=str(plan.get("note") or "planner rejected model"),
            planner=planner_payload,
        )
    compat_payload = dict(hooks.compat_fn(model))
    if str(compat_payload.get("verdict")) == COMPAT_NOT_TRAINABLE:
        return CampaignEntry(
            model=model,
            status=SKIP_VERDICT,
            reason=f"not trainable: {compat_payload.get('blocker') or 'compat probe'}",
            planner=planner_payload,
            compat=compat_payload,
        )
    return None


def _run_entry_rounds(
    model: str,
    model_cfg: RunConfig,
    entry_dir: Path,
    rounds: int,
    hooks: _CampaignHooks,
    root: Path,
    shared_dataset_dir: Path | None,
    base_objective: float,
) -> _RoundsOutcome:
    """The tuning-eval -> miss-analysis -> export -> train -> final-eval loop for one model."""
    current_cfg = model_cfg
    tuning_dir: Path | None = None
    preference_dir: Path | None = None
    adapter_dir: Path | None = None
    final_dir: Path | None = None
    train_wall_clock_s = 0.0
    for round_index in range(1, rounds + 1):
        round_dir = entry_dir / f"round-{round_index}"
        tuning_dir = _eval_to_dir(hooks.eval_fn, current_cfg, "tuning", round_dir / "tuning")
        _publish_run_pointer(round_dir / "run-tuning", tuning_dir)

        analysis = analyze_run(
            tuning_dir,
            load_goldset(model_cfg.goldset_path),
            provenance=load_item_provenance(model_cfg.goldset_path),
        )
        miss_paths = write_analysis(analysis, round_dir / "miss-analysis")
        if shared_dataset_dir is None:
            shared_dataset_dir = root / SHARED_DATASET_DIRNAME
            export_finetune_set(
                run_dir=tuning_dir,
                goldset_path=model_cfg.goldset_path,
                out_dir=shared_dataset_dir,
            )
        preference_dir = round_dir / "preference-dataset"
        export_finetune_set(
            run_dir=tuning_dir,
            goldset_path=model_cfg.goldset_path,
            out_dir=preference_dir,
            misses_path=miss_paths["misses"],
        )

        adapter_dir = round_dir / "adapter"
        start = time.monotonic()
        hooks.trainer_fn(shared_dataset_dir, model, adapter_dir, model_cfg.seed + round_index - 1)
        train_wall_clock_s += time.monotonic() - start
        current_cfg = model_cfg.with_overrides(adapter_path=adapter_dir)
        final_dir = _eval_to_dir(hooks.eval_fn, current_cfg, "final", round_dir / "final")
        _publish_run_pointer(round_dir / "run-final", final_dir)
        round_objective = _objective_from_run(final_dir)
        register_round_adapter(
            model_cfg,
            adapter_dir=adapter_dir,
            source_run=tuning_dir,
            eval_summary={
                "final_run_dir": str(final_dir),
                "objective_score": round_objective,
                "base_objective": base_objective,
                "delta": round_objective - base_objective,
                "round": round_index,
            },
        )
    if (
        tuning_dir is None
        or preference_dir is None
        or adapter_dir is None
        or final_dir is None
        or shared_dataset_dir is None
    ):
        raise RuntimeError("campaign entry did not run any rounds")
    return _RoundsOutcome(
        tuning_dir=tuning_dir,
        preference_dir=preference_dir,
        adapter_dir=adapter_dir,
        final_dir=final_dir,
        train_wall_clock_s=train_wall_clock_s,
        shared_dataset_dir=shared_dataset_dir,
    )


def _completed_entry(
    model: str,
    model_cfg: RunConfig,
    base_final_dir: Path,
    base_objective: float,
    outcome: _RoundsOutcome,
    planner_payload: JsonObject,
    reclaim: JsonObject,
) -> CampaignEntry:
    tuned_objective = _objective_from_run(outcome.final_dir)
    return CampaignEntry(
        model=model,
        status=COMPLETE_VERDICT,
        base_final_run_dir=base_final_dir,
        tuning_run_dir=outcome.tuning_dir,
        final_run_dir=outcome.final_dir,
        adapter_dir=outcome.adapter_dir,
        preference_dataset_dir=outcome.preference_dir,
        shared_dataset_digest=_dataset_digest(outcome.shared_dataset_dir),
        base_objective=base_objective,
        tuned_objective=tuned_objective,
        delta=tuned_objective - base_objective,
        base_ci=_ci_from_run(base_final_dir, seed=model_cfg.seed),
        tuned_ci=_ci_from_run(outcome.final_dir, seed=model_cfg.seed + 1),
        train_wall_clock_s=outcome.train_wall_clock_s,
        peak_vram_mb=_peak_vram(outcome.final_dir),
        planner=planner_payload,
        reclaim=reclaim,
    )


def _run_campaign_entry(
    model: str,
    config: RunConfig,
    root: Path,
    rounds: int,
    hooks: _CampaignHooks,
    shared_dataset_dir: Path | None,
    *,
    reclaim_must_raise: bool,
) -> tuple[CampaignEntry, Path | None]:
    """Run one roster model end to end; returns (entry, possibly-created shared dataset dir)."""
    entry_dir = root / model_slug(model)
    entry_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = config.with_overrides(model=model)
    plan = hooks.planner_fn(model, model_cfg)
    skipped = _skip_entry(model, hooks, plan)
    if skipped is not None:
        return skipped, shared_dataset_dir

    base_final_dir = _eval_to_dir(hooks.eval_fn, model_cfg, "final", entry_dir / "base-final")
    _publish_run_pointer(entry_dir / "run-base-final", base_final_dir)
    base_objective = _objective_from_run(base_final_dir)
    outcome = _run_entry_rounds(
        model, model_cfg, entry_dir, rounds, hooks, root, shared_dataset_dir, base_objective
    )
    try:
        reclaim = hooks.reclaim_fn()
    except Exception as exc:
        if reclaim_must_raise:
            raise
        reclaim = {"reclaimed": False, "reason": str(exc)}
    entry = _completed_entry(
        model, model_cfg, base_final_dir, base_objective, outcome, dict(plan), reclaim
    )
    return entry, outcome.shared_dataset_dir


def _eval_to_dir(eval_fn: EvalFn, config: RunConfig, split: str, run_dir: Path) -> Path:
    result = eval_fn(config, split, run_dir)
    return Path(result["paths"]["manifest"]).parent


def _dataset_digest(dataset_dir: Path) -> str:
    manifest = json.loads((dataset_dir / DATASET_MANIFEST).read_text(encoding="utf-8"))
    return str(manifest["dataset_digest"])


def _peak_vram(run_dir: Path) -> float | None:
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    telemetry = manifest.get("telemetry") or {}
    value = telemetry.get("peak_vram_mb")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
