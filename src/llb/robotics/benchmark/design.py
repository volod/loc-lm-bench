"""Load and validate the frozen final-split robotics task ledger."""

from pathlib import Path

from llb.robotics.benchmark.constants import MANDATORY_FAULT_CLASSES
from llb.robotics.benchmark.models import BenchmarkDesign, BenchmarkTask
from llb.robotics.digests import file_digest


def load_design(path: Path) -> tuple[BenchmarkDesign, tuple[BenchmarkTask, ...]]:
    try:
        design = BenchmarkDesign.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: invalid robotics benchmark design -- {exc}") from None
    ledger_path = (path.parent / design.task_ledger).resolve()
    if file_digest(ledger_path) != design.task_ledger_sha256:
        raise ValueError("robotics task ledger digest does not match the frozen design")
    tasks: list[BenchmarkTask] = []
    try:
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                tasks.append(BenchmarkTask.model_validate_json(line))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{ledger_path}: invalid robotics task ledger -- {exc}") from None
    if len(tasks) < design.minimum_evidence_count:
        raise ValueError("robotics task ledger is smaller than its minimum evidence count")
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("robotics task ledger contains duplicate task ids")
    observed = {task.safety_class for task in tasks if task.safety_class}
    declared = set(design.mandatory_fault_classes)
    if declared != MANDATORY_FAULT_CLASSES or not declared.issubset(observed):
        raise ValueError("robotics task ledger does not cover every mandatory fault class")
    if not any(task.expected_behavior == "complete" for task in tasks):
        raise ValueError("robotics task ledger has no normal completion workflow")
    return design, tuple(tasks)
