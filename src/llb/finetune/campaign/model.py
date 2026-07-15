"""Campaign vocabulary: filenames, verdicts, the fn seams, and the record dataclasses."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from llb.core.config import RunConfig
from llb.core.contracts.runs import EvalResult
from llb.core.contracts.common import JsonObject
from llb.core.contracts.models import ModelPlanRow
from llb.finetune.campaign.coerce import _path_or_none

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
