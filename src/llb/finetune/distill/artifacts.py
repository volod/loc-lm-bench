"""Write the distillation run manifest and the human-readable paired-comparison report."""

import json
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.finetune.distill.model import (
    ADAPTER_DIRNAME,
    DISTILL_MANIFEST,
    REFERENCE_ADAPTER_DIRNAME,
    REPORT_FILENAME,
    TEACHER_OUTPUTS,
    DistillComparison,
    GatedTeacherRecord,
)


def _write_manifest(
    root: Path,
    *,
    teacher: str,
    student: str,
    gate: float,
    dataset_manifest: JsonObject,
    reference_manifest: JsonObject,
    adapter_manifest: JsonObject,
    reference_adapter_manifest: JsonObject,
    records: list[GatedTeacherRecord],
    comparison: DistillComparison,
    registered_adapter_id: str | None,
) -> Path:
    accepted = [record for record in records if record.accepted]
    payload: JsonObject = {
        "kind": "llb.finetune.distill",
        "teacher_model": teacher,
        "student_model": student,
        "gate_threshold": gate,
        "n_teacher_outputs": len(records),
        "n_accepted": len(accepted),
        "n_rejected": len(records) - len(accepted),
        "teacher_outputs": str(root / TEACHER_OUTPUTS),
        "dataset": dataset_manifest,
        "reference_dataset": reference_manifest,
        "adapter": {
            "adapter_dir": str(root / ADAPTER_DIRNAME),
            "adapter_digest": adapter_manifest.get("adapter_digest"),
        },
        "reference_adapter": {
            "adapter_dir": str(root / REFERENCE_ADAPTER_DIRNAME),
            "adapter_digest": reference_adapter_manifest.get("adapter_digest"),
        },
        "comparison": comparison.as_dict(),
        "registered_adapter_id": registered_adapter_id,
    }
    path = root / DISTILL_MANIFEST
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def _write_report(
    root: Path,
    *,
    teacher: str,
    student: str,
    gate: float,
    records: list[GatedTeacherRecord],
    dataset_dir: Path,
    adapter_dir: Path,
    reference_adapter_dir: Path,
    comparison: DistillComparison,
    registered_adapter_id: str | None,
) -> Path:
    accepted = sum(1 for record in records if record.accepted)
    rejected = len(records) - accepted
    verdict = "distilled-better" if comparison.delta > 0 else "no-gain"
    lines = [
        "# Local distillation report",
        "",
        f"Teacher: `{teacher}`",
        f"Student: `{student}`",
        f"Gate: `{gate:.3f}`",
        f"Teacher outputs: `{root / TEACHER_OUTPUTS}`",
        f"Accepted: `{accepted}`",
        f"Rejected: `{rejected}`",
        f"Distilled dataset: `{dataset_dir}`",
        f"Distilled adapter: `{adapter_dir}`",
        f"Reference adapter: `{reference_adapter_dir}`",
        f"Registered adapter: `{registered_adapter_id or 'not registered'}`",
        "",
        "## Paired comparison",
        "",
        f"Split: `{comparison.split}`",
        f"Items: `{comparison.n_items}`",
        "",
        "| lane | objective | ci | run |",
        "| --- | --- | --- | --- |",
        _comparison_row(
            "distilled",
            comparison.distilled_objective,
            comparison.distilled_ci,
            comparison.distilled_run_dir,
        ),
        _comparison_row(
            "reference-sft",
            comparison.reference_objective,
            comparison.reference_ci,
            comparison.reference_run_dir,
        ),
        "",
        f"Delta distilled-minus-reference: `{comparison.delta:.4f}`",
        f"Verdict: `{verdict}`",
    ]
    path = root / REPORT_FILENAME
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def _comparison_row(
    label: str, objective: float, ci: tuple[float, float] | None, run_dir: Path | None
) -> str:
    ci_text = "-" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    run_text = "-" if run_dir is None else f"`{run_dir}`"
    return f"| {label} | {objective:.4f} | {ci_text} | {run_text} |"
