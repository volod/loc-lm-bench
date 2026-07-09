"""Local text-level distillation from a stronger local teacher into a student LoRA adapter.

The lane is deliberately control-plane first: teacher generation, adapter training, and adapter
comparison are injectable, so CI uses fakes while a CUDA host uses the same backend, trainer, guard,
and registry seams as `run-eval` and `self-improve`.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

from llb.bench.common import new_run_timestamp
from llb.core.config import RunConfig
from llb.core.contracts import ChatMessage, ChunkRecord, EvalResult, JsonObject
from llb.core.fsutil import atomic_write_text
from llb.eval import common as eval_common
from llb.eval.graph import build_messages
from llb.finetune.dataset import (
    DATASET_MANIFEST,
    DPO_FILENAME,
    SFT_FILENAME,
    TUNING_SPLIT,
    dataset_digest,
)
from llb.finetune.registry import registry_path, try_register_adapter
from llb.finetune.trainer import train_adapter
from llb.goldset.schema import GoldItem, load_goldset
from llb.scoring.aggregate import bootstrap_mean_ci
from llb.scoring.correctness import answer_correctness

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


def _load_items(goldset_path: Path, *, split: str, limit: int | None) -> list[GoldItem]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    items = [item for item in load_goldset(goldset_path) if item.verified and item.split == split]
    items.sort(key=lambda item: item.id)
    if limit is not None:
        items = items[:limit]
    if not items:
        raise SystemExit(f"[distill] no verified {split!r} items in {goldset_path}")
    return items


def _gate_responses(
    items: list[GoldItem], responses: list[TeacherResponse], *, gate: float
) -> list[GatedTeacherRecord]:
    by_id = {response.item_id: response for response in responses}
    missing = [item.id for item in items if item.id not in by_id]
    if missing:
        raise ValueError(f"teacher did not return answers for item ids: {', '.join(missing)}")
    records: list[GatedTeacherRecord] = []
    for item in items:
        response = by_id[item.id]
        corr = answer_correctness(response.answer, item.reference_answer)
        gate_score = float(corr["score"])
        accepted = response.status == eval_common.OK and gate_score >= gate
        records.append(
            GatedTeacherRecord(
                item=item,
                answer=response.answer,
                status=response.status,
                gate_score=gate_score,
                token_f1=float(corr["token_f1"]),
                exact=float(corr["exact"]),
                contains=float(corr["contains"]),
                accepted=accepted,
                context=response.context,
                retrieved=response.retrieved,
                messages=response.messages,
            )
        )
    return records


def _write_training_dataset(
    records: list[GatedTeacherRecord],
    *,
    out_dir: Path,
    teacher: str,
    student: str,
    gate: float,
    target: str,
) -> JsonObject:
    sft_records: list[JsonObject] = []
    gate_scores: JsonObject = {}
    for record in sorted(records, key=lambda row: row.item.id):
        response = record.answer if target == TEACHER_TARGET else record.item.reference_answer
        gate_scores[record.item.id] = round(record.gate_score, 6)
        sft_records.append(
            {
                "item_id": record.item.id,
                "split": record.item.split,
                "weight": 1.0,
                "messages": _messages_for_record(record),
                "response": response,
                "reference_answer": record.item.reference_answer,
                "teacher_answer": record.answer,
                "teacher_model": teacher,
                "student_model": student,
                "gate_score": round(record.gate_score, 6),
                "distillation_target": target,
                "prompt_template": "eval.rag.chat",
            }
        )
    dpo_records: list[JsonObject] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / SFT_FILENAME, sft_records)
    _write_jsonl(out_dir / DPO_FILENAME, dpo_records)
    digest = dataset_digest(sft_records, dpo_records)
    manifest: JsonObject = {
        "kind": "llb.finetune.dataset",
        "dataset_digest": digest,
        "source_run": str(out_dir.parent),
        "source_run_id": out_dir.parent.name,
        "item_ids": [str(record["item_id"]) for record in sft_records],
        "split_counts": dict(Counter(str(record["split"]) for record in sft_records)),
        "n_sft": len(sft_records),
        "n_dpo": len(dpo_records),
        "prompt_template": "eval.rag.chat",
        "distillation": {
            "teacher_model": teacher,
            "student_model": student,
            "gate_threshold": gate,
            "target": target,
            "gate_scores": gate_scores,
        },
    }
    atomic_write_text(
        out_dir / DATASET_MANIFEST,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def _messages_for_record(record: GatedTeacherRecord) -> list[ChatMessage]:
    if record.messages:
        return [cast(ChatMessage, dict(message)) for message in record.messages]
    context = record.context or _context_from_record(record)
    return build_messages(record.item.question, context)


def _context_from_record(record: GatedTeacherRecord) -> str:
    chunks: list[ChunkRecord] = list(record.retrieved)
    if not chunks:
        chunks = [
            {
                "doc_id": span.doc_id,
                "char_start": span.char_start,
                "char_end": span.char_end,
                "text": span.text,
            }
            for span in record.item.source_spans
        ]
    return eval_common.format_context(chunks)


def _default_trainer_fn(config: RunConfig, trainer: str) -> TrainerFn:
    from llb.finetune.hparam_search import trainer_defaults

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
    from llb.executor.runner import _resolve_eval_runner

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


def _write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _default_out_dir(config: RunConfig) -> Path:
    _run_id, stamp = new_run_timestamp()
    return config.data_dir / DISTILL_METHOD / stamp
