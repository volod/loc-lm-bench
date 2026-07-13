"""Orchestrate the local distillation lane: gate -> dataset -> train -> compare -> register."""

from pathlib import Path

from llb.core.config import RunConfig
from llb.finetune.distill.artifacts import _write_manifest, _write_report
from llb.finetune.distill.dataset_io import _write_jsonl, _write_training_dataset
from llb.finetune.distill.defaults import (
    _default_comparison_fn,
    _default_out_dir,
    _default_teacher_fn,
    _default_trainer_fn,
)
from llb.finetune.distill.gate import _gate_responses, _load_items
from llb.finetune.distill.model import (
    ADAPTER_DIRNAME,
    COMPARISON_DIRNAME,
    DATASET_DIRNAME,
    DEFAULT_COMPARE_SPLIT,
    DEFAULT_GATE_THRESHOLD,
    REFERENCE_ADAPTER_DIRNAME,
    REFERENCE_DATASET_DIRNAME,
    REFERENCE_TARGET,
    TEACHER_OUTPUTS,
    TEACHER_TARGET,
    ComparisonFn,
    DistillResult,
    TeacherFn,
    TrainerFn,
)
from llb.finetune.dataset import TUNING_SPLIT
from llb.finetune.registry import registry_path, try_register_adapter


def run_distillation(
    config: RunConfig,
    *,
    teacher: str,
    student: str,
    gate: float = DEFAULT_GATE_THRESHOLD,
    out_dir: Path | str | None = None,
    trainer: str = "auto",
    seed: int | None = None,
    limit: int | None = None,
    compare_split: str = DEFAULT_COMPARE_SPLIT,
    compare_limit: int | None = None,
    teacher_fn: TeacherFn | None = None,
    trainer_fn: TrainerFn | None = None,
    comparison_fn: ComparisonFn | None = None,
) -> DistillResult:
    """Distill accepted tuning-split teacher answers into a student adapter."""
    _validate_request(config, teacher=teacher, student=student, gate=gate)
    if comparison_fn is None and config.backend != "vllm":
        raise SystemExit(
            "[distill] distilled-vs-reference adapter comparison needs --backend vllm "
            "(direct LoRA serving), or an injected comparison_fn"
        )
    run_seed = config.seed if seed is None else seed
    root = Path(out_dir or _default_out_dir(config))
    root.mkdir(parents=True, exist_ok=True)

    tuning_items = _load_items(config.goldset_path, split=TUNING_SPLIT, limit=limit)
    compare_items = _load_items(config.goldset_path, split=compare_split, limit=compare_limit)
    teacher_config = config.with_overrides(model=teacher, adapter_path=None)
    responses = (teacher_fn or _default_teacher_fn)(teacher_config, tuning_items, root)
    records = _gate_responses(tuning_items, responses, gate=gate)
    _write_jsonl(root / TEACHER_OUTPUTS, [record.as_dict() for record in records])

    accepted = [record for record in records if record.accepted]
    if not accepted:
        raise SystemExit(
            f"[distill] no teacher answers met gate {gate:.3f}; no training dataset was written"
        )

    dataset_dir = root / DATASET_DIRNAME
    reference_dataset_dir = root / REFERENCE_DATASET_DIRNAME
    dataset_manifest = _write_training_dataset(
        accepted,
        out_dir=dataset_dir,
        teacher=teacher,
        student=student,
        gate=gate,
        target=TEACHER_TARGET,
    )
    reference_manifest = _write_training_dataset(
        accepted,
        out_dir=reference_dataset_dir,
        teacher=teacher,
        student=student,
        gate=gate,
        target=REFERENCE_TARGET,
    )

    active_trainer = trainer_fn or _default_trainer_fn(config, trainer)
    adapter_dir = root / ADAPTER_DIRNAME
    reference_adapter_dir = root / REFERENCE_ADAPTER_DIRNAME
    adapter_manifest = active_trainer(dataset_dir, student, adapter_dir, run_seed)
    reference_adapter_manifest = active_trainer(
        reference_dataset_dir, student, reference_adapter_dir, run_seed
    )

    comparison = (comparison_fn or _default_comparison_fn)(
        config.with_overrides(model=student),
        adapter_dir,
        reference_adapter_dir,
        compare_items,
        root / COMPARISON_DIRNAME,
    )
    registered = try_register_adapter(
        registry=registry_path(config.data_dir),
        adapter_dir=adapter_dir,
        goldset_path=config.goldset_path,
        corpus_root=config.corpus_root,
        source_run=root,
        eval_summary={
            "objective_score": comparison.distilled_objective,
            "reference_objective": comparison.reference_objective,
            "delta": comparison.delta,
            "compare_split": comparison.split,
            "n_compare_items": comparison.n_items,
            "teacher_model": teacher,
            "student_model": student,
            "gate": gate,
        },
    )
    registered_id = registered.adapter_id if registered is not None else None
    manifest_path = _write_manifest(
        root,
        teacher=teacher,
        student=student,
        gate=gate,
        dataset_manifest=dataset_manifest,
        reference_manifest=reference_manifest,
        adapter_manifest=adapter_manifest,
        reference_adapter_manifest=reference_adapter_manifest,
        records=records,
        comparison=comparison,
        registered_adapter_id=registered_id,
    )
    report_path = _write_report(
        root,
        teacher=teacher,
        student=student,
        gate=gate,
        records=records,
        dataset_dir=dataset_dir,
        adapter_dir=adapter_dir,
        reference_adapter_dir=reference_adapter_dir,
        comparison=comparison,
        registered_adapter_id=registered_id,
    )
    return DistillResult(
        out_dir=root,
        teacher_outputs_path=root / TEACHER_OUTPUTS,
        dataset_dir=dataset_dir,
        reference_dataset_dir=reference_dataset_dir,
        adapter_dir=adapter_dir,
        reference_adapter_dir=reference_adapter_dir,
        report_path=report_path,
        manifest_path=manifest_path,
        accepted=len(accepted),
        rejected=len(records) - len(accepted),
        comparison=comparison,
        registered_adapter_id=registered_id,
    )


def _validate_request(config: RunConfig, *, teacher: str, student: str, gate: float) -> None:
    if teacher == student:
        raise SystemExit("[distill] teacher and student must be different models")
    if config.judge_model is not None and config.judge_model == teacher:
        raise SystemExit("[distill] the configured judge model cannot be the distillation teacher")
    if not 0.0 <= gate <= 1.0:
        raise ValueError(f"gate must be between 0 and 1, got {gate}")
