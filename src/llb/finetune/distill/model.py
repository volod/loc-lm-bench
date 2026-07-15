"""Distillation vocabulary: filenames, gate defaults, the record dataclasses, and the fn seams."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llb.core.config import RunConfig
from llb.core.contracts.common import ChatMessage, JsonObject
from llb.core.contracts.rag import ChunkRecord
from llb.eval import common as eval_common
from llb.goldset.schema import GoldItem

DISTILL_METHOD = "distill"
TEACHER_OUTPUTS = "teacher_outputs.jsonl"
DISTILL_MANIFEST = "distill_manifest.json"
REPORT_FILENAME = "report.md"
DATASET_DIRNAME = "dataset"
REFERENCE_DATASET_DIRNAME = "reference_dataset"
ADAPTER_DIRNAME = "adapter"
REFERENCE_ADAPTER_DIRNAME = "reference_adapter"
COMPARISON_DIRNAME = "comparison"
DEFAULT_GATE_THRESHOLD = 0.8
DEFAULT_COMPARE_SPLIT = "final"
REFERENCE_TARGET = "reference"
TEACHER_TARGET = "teacher"


@dataclass(frozen=True)
class TeacherResponse:
    """One raw teacher answer before the deterministic quality gate is applied."""

    item_id: str
    answer: str
    status: str = eval_common.OK
    context: str = ""
    retrieved: tuple[ChunkRecord, ...] = ()
    messages: tuple[ChatMessage, ...] = ()


@dataclass(frozen=True)
class GatedTeacherRecord:
    """Teacher answer plus objective gate signals."""

    item: GoldItem
    answer: str
    status: str
    gate_score: float
    token_f1: float
    exact: float
    contains: float
    accepted: bool
    context: str
    retrieved: tuple[ChunkRecord, ...]
    messages: tuple[ChatMessage, ...]

    def as_dict(self) -> JsonObject:
        return {
            "item_id": self.item.id,
            "split": self.item.split,
            "question": self.item.question,
            "reference_answer": self.item.reference_answer,
            "teacher_answer": self.answer,
            "status": self.status,
            "gate_score": round(self.gate_score, 6),
            "token_f1": round(self.token_f1, 6),
            "exact": round(self.exact, 6),
            "contains": round(self.contains, 6),
            "accepted": self.accepted,
            "retrieved": [dict(chunk) for chunk in self.retrieved],
        }


@dataclass(frozen=True)
class DistillComparison:
    """Paired comparison of distilled-vs-reference adapters over the same eval items."""

    split: str
    n_items: int
    distilled_objective: float
    reference_objective: float
    delta: float
    distilled_ci: tuple[float, float] | None = None
    reference_ci: tuple[float, float] | None = None
    distilled_run_dir: Path | None = None
    reference_run_dir: Path | None = None

    def as_dict(self) -> JsonObject:
        return {
            "split": self.split,
            "n_items": self.n_items,
            "distilled_objective": self.distilled_objective,
            "reference_objective": self.reference_objective,
            "delta": self.delta,
            "distilled_ci": self.distilled_ci,
            "reference_ci": self.reference_ci,
            "distilled_run_dir": str(self.distilled_run_dir) if self.distilled_run_dir else None,
            "reference_run_dir": str(self.reference_run_dir) if self.reference_run_dir else None,
        }


@dataclass(frozen=True)
class DistillResult:
    out_dir: Path
    teacher_outputs_path: Path
    dataset_dir: Path
    reference_dataset_dir: Path
    adapter_dir: Path
    reference_adapter_dir: Path
    report_path: Path
    manifest_path: Path
    accepted: int
    rejected: int
    comparison: DistillComparison
    registered_adapter_id: str | None = None


TeacherFn = Callable[[RunConfig, list[GoldItem], Path], list[TeacherResponse]]
TrainerFn = Callable[[Path, str, Path, int], JsonObject]
ComparisonFn = Callable[[RunConfig, Path, Path, list[GoldItem], Path], DistillComparison]
