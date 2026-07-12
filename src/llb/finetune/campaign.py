"""Multi-model fine-tuning campaign orchestration.

The campaign runner schedules the existing adapter self-improvement ingredients across a roster:
base/tuning/final evals, shared SFT export, per-model preference export, trainer seam, VRAM reclaim,
and a resumable JSONL journal. Heavy collaborators are injectable so CI can exercise the control
plane without launching models or training stacks.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from llb.backends.planner import VERDICT_NO, enrich_arch, plan_model
from llb.bench.common import new_run_timestamp
from llb.board.miss_analysis import analyze_run, load_item_provenance, write_analysis
from llb.core.config import RunConfig
from llb.core.contracts import EvalResult, JsonObject, ModelPlanRow, ModelSpec
from llb.core.fsutil import atomic_write_text
from llb.finetune.compat import VERDICT_NOT_TRAINABLE as COMPAT_NOT_TRAINABLE
from llb.finetune.dataset import DATASET_MANIFEST, export_finetune_set
from llb.finetune.loop import (
    _ci_from_run,
    _default_trainer_fn,
    _objective_from_run,
    _publish_run_pointer,
    register_round_adapter,
)
from llb.finetune.naming import model_slug
from llb.goldset.schema import load_goldset

PROGRESS_FILENAME = "campaign.progress.jsonl"
REPORT_FILENAME = "report.md"
SHARED_DATASET_DIRNAME = "shared-dataset"
SKIP_VERDICT = "skipped"
COMPLETE_VERDICT = "completed"

EvalFn = Callable[[RunConfig, str, Path], EvalResult]
TrainerFn = Callable[[Path, str, Path, int], JsonObject]
PlannerFn = Callable[[str, RunConfig], ModelPlanRow]
ReclaimFn = Callable[[], JsonObject]
# model id -> compat verdict payload (compressed-qat-adapter-support). Only a POSITIVE
# not-trainable verdict skips; an unknown verdict lets the entry proceed.
CompatFn = Callable[[str], JsonObject]


@dataclass
class CampaignEntry:
    model: str
    status: str
    reason: str | None = None
    base_final_run_dir: Path | None = None
    tuning_run_dir: Path | None = None
    final_run_dir: Path | None = None
    adapter_dir: Path | None = None
    preference_dataset_dir: Path | None = None
    shared_dataset_digest: str | None = None
    base_objective: float | None = None
    tuned_objective: float | None = None
    delta: float | None = None
    base_ci: tuple[float, float] | None = None
    tuned_ci: tuple[float, float] | None = None
    train_wall_clock_s: float | None = None
    peak_vram_mb: float | None = None
    planner: JsonObject = field(default_factory=dict)
    reclaim: JsonObject = field(default_factory=dict)
    compat: JsonObject = field(default_factory=dict)

    def as_dict(self) -> JsonObject:
        return {
            "model": self.model,
            "status": self.status,
            "reason": self.reason,
            "base_final_run_dir": _path_or_none(self.base_final_run_dir),
            "tuning_run_dir": _path_or_none(self.tuning_run_dir),
            "final_run_dir": _path_or_none(self.final_run_dir),
            "adapter_dir": _path_or_none(self.adapter_dir),
            "preference_dataset_dir": _path_or_none(self.preference_dataset_dir),
            "shared_dataset_digest": self.shared_dataset_digest,
            "base_objective": self.base_objective,
            "tuned_objective": self.tuned_objective,
            "delta": self.delta,
            "base_ci": self.base_ci,
            "tuned_ci": self.tuned_ci,
            "train_wall_clock_s": self.train_wall_clock_s,
            "peak_vram_mb": self.peak_vram_mb,
            "planner": self.planner,
            "reclaim": self.reclaim,
            "compat": self.compat,
        }


@dataclass
class CampaignResult:
    out_dir: Path
    entries: list[CampaignEntry]
    shared_dataset_dir: Path | None


@dataclass
class _CampaignHooks:
    """The injectable collaborators of one campaign run (real or CI fakes)."""

    eval_fn: EvalFn
    trainer_fn: TrainerFn
    planner_fn: PlannerFn
    reclaim_fn: ReclaimFn
    compat_fn: CompatFn


@dataclass
class _RoundsOutcome:
    """Artifacts of the last completed round for one campaign entry."""

    tuning_dir: Path
    preference_dir: Path
    adapter_dir: Path
    final_dir: Path
    train_wall_clock_s: float
    shared_dataset_dir: Path


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


def _default_eval_fn(*, limit: int | None) -> EvalFn:
    from llb.executor.runner import run_eval

    def run(config: RunConfig, split: str, _round_run_dir: Path) -> EvalResult:
        return run_eval(config, split=split, limit=limit)

    return run


def _default_planner_fn(model_specs: list[ModelSpec]) -> PlannerFn:
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb

    specs = {_model_key(spec): spec for spec in model_specs}
    vram_mib = max_vram_mb(detect_gpus())
    ram_mib = detect_ram_mb()

    def plan(model: str, config: RunConfig) -> ModelPlanRow:
        spec = specs.get(model) or {
            "name": model,
            "source": model,
            "backend": config.backend,
        }
        return plan_model(enrich_arch(spec), vram_mib=vram_mib, ram_mib=ram_mib)

    return plan


def _default_compat_fn() -> CompatFn:
    """Config-only trainability probe: cheap, and UNKNOWN (never a skip) when unreachable."""
    from llb.finetune.compat import config_compat_probe

    return config_compat_probe


def _default_reclaim_fn() -> ReclaimFn:
    baseline: int | None = None

    def reclaim() -> JsonObject:
        nonlocal baseline
        from llb.backends.hardware import detect_gpus
        from llb.executor.vram import assert_reclaimed, read_baseline

        if not detect_gpus():
            return {"skipped": True, "reason": "no GPU detected"}
        try:
            if baseline is None:
                baseline = read_baseline()
            return dict(assert_reclaimed(baseline))
        except SystemExit as exc:
            return {"skipped": True, "reason": str(exc)}

    return reclaim


def _default_out_dir(config: RunConfig) -> Path:
    _run_id, stamp = new_run_timestamp()
    return config.data_dir / "finetune-campaign" / stamp


def _eval_to_dir(eval_fn: EvalFn, config: RunConfig, split: str, run_dir: Path) -> Path:
    result = eval_fn(config, split, run_dir)
    return Path(result["paths"]["manifest"]).parent


def _append_entry(root: Path, entry: CampaignEntry) -> None:
    existing = list(_read_completed_entries(root).values())
    path = root / PROGRESS_FILENAME
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"event": "entry", "entry": entry.as_dict()}, ensure_ascii=False) + "\n"
        )
    _write_state(root, [*existing, entry])


def _write_state(root: Path, entries: list[CampaignEntry]) -> None:
    atomic_write_text(
        root / "campaign_state.json",
        json.dumps(
            {"entries": [entry.as_dict() for entry in entries]}, ensure_ascii=False, indent=2
        )
        + "\n",
    )


def _read_completed_entries(root: Path) -> dict[str, CampaignEntry]:
    path = root / PROGRESS_FILENAME
    if not path.is_file():
        return {}
    entries: dict[str, CampaignEntry] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = row.get("entry") if isinstance(row, dict) else None
        if isinstance(payload, dict) and payload.get("model"):
            entries[str(payload["model"])] = _entry_from_dict(payload)
    return entries


def _entry_from_dict(row: JsonObject) -> CampaignEntry:
    return CampaignEntry(
        model=str(row["model"]),
        status=str(row["status"]),
        reason=str(row["reason"]) if row.get("reason") is not None else None,
        base_final_run_dir=_path_from(row.get("base_final_run_dir")),
        tuning_run_dir=_path_from(row.get("tuning_run_dir")),
        final_run_dir=_path_from(row.get("final_run_dir")),
        adapter_dir=_path_from(row.get("adapter_dir")),
        preference_dataset_dir=_path_from(row.get("preference_dataset_dir")),
        shared_dataset_digest=_str_or_none(row.get("shared_dataset_digest")),
        base_objective=_float_or_none(row.get("base_objective")),
        tuned_objective=_float_or_none(row.get("tuned_objective")),
        delta=_float_or_none(row.get("delta")),
        base_ci=_ci_from_value(row.get("base_ci")),
        tuned_ci=_ci_from_value(row.get("tuned_ci")),
        train_wall_clock_s=_float_or_none(row.get("train_wall_clock_s")),
        peak_vram_mb=_float_or_none(row.get("peak_vram_mb")),
        planner=dict(row.get("planner") or {}),
        reclaim=dict(row.get("reclaim") or {}),
    )


def _existing_shared_dataset(root: Path) -> Path | None:
    path = root / SHARED_DATASET_DIRNAME
    return path if (path / DATASET_MANIFEST).is_file() else None


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


def _write_report(
    root: Path, entries: list[CampaignEntry], shared_dataset_dir: Path | None
) -> None:
    ranked = sorted(
        [entry for entry in entries if entry.status == COMPLETE_VERDICT],
        key=lambda entry: (
            entry.delta if entry.delta is not None else float("-inf"),
            -(entry.train_wall_clock_s or 0.0),
            -(entry.peak_vram_mb or 0.0),
        ),
        reverse=True,
    )
    lines = [
        "# Fine-tune campaign report",
        "",
        f"Shared dataset: `{shared_dataset_dir}`" if shared_dataset_dir else "Shared dataset: n/a",
        "",
        "| rank | model | base objective | tuned objective | delta | train s | peak VRAM | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rank_by_model = {entry.model: idx for idx, entry in enumerate(ranked, 1)}
    ordered = sorted(entries, key=lambda entry: rank_by_model.get(entry.model, 10_000))
    for entry in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank_by_model.get(entry.model, "")),
                    entry.model,
                    _fmt(entry.base_objective),
                    _fmt(entry.tuned_objective),
                    _fmt(entry.delta),
                    _fmt(entry.train_wall_clock_s),
                    _fmt(entry.peak_vram_mb),
                    entry.status if entry.reason is None else f"{entry.status}: {entry.reason}",
                ]
            )
            + " |"
        )
    atomic_write_text(root / REPORT_FILENAME, "\n".join(lines) + "\n")


def latest_campaign(data_dir: Path | str) -> JsonObject | None:
    """Newest `$DATA_DIR/finetune-campaign/*/campaign.progress.jsonl` with report path attached."""
    root = Path(data_dir) / "finetune-campaign"
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir(), reverse=True):
        progress = candidate / PROGRESS_FILENAME
        if not progress.is_file():
            continue
        entries = [entry.as_dict() for entry in _read_completed_entries(candidate).values()]
        if not entries:
            continue
        return {
            "campaign_dir": str(candidate),
            "report_path": str(candidate / REPORT_FILENAME),
            "entries": entries,
        }
    return None


def _parse_models(models: list[str]) -> list[str]:
    out: list[str] = []
    for value in models:
        for item in value.split(","):
            model = item.strip()
            if model and model not in out:
                out.append(model)
    return out


def _model_key(spec: ModelSpec) -> str:
    return str(spec.get("name") or spec.get("source"))


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _path_from(value: object) -> Path | None:
    return Path(str(value)) if value else None


def _str_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def _float_or_none(value: object) -> float | None:
    try:
        return None if value is None else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _ci_from_value(value: object) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    return (float(value[0]), float(value[1]))


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.4f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
