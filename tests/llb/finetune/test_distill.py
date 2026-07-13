"""Local distillation lane control-plane tests."""

import json
from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.core.contracts import JsonObject
from llb.finetune.distill.model import DistillComparison, TeacherResponse
from llb.finetune.distill.run import run_distillation
from llb.finetune.guard import validate_adapter_for_eval
from llb.finetune.registry import load_registry, registry_path
from llb.finetune.trainer import fake_train_adapter
from llb.goldset.schema import GoldItem, dump_goldset, load_goldset

TEACHER = "teacher-model"
STUDENT = "student-model"


def _item(item_id: str, split: str, reference: str | None = None) -> GoldItem:
    return GoldItem(
        id=item_id,
        question=f"Question {item_id}?",
        reference_answer=reference or f"Answer {item_id}",
        source_doc_id=f"{item_id}.txt",
        source_spans=[
            {
                "doc_id": f"{item_id}.txt",
                "char_start": 0,
                "char_end": 5,
                "text": "alpha",
            }
        ],
        provenance="human-authored",
        verified=True,
        split=split,
    )


def _goldset(tmp_path: Path) -> Path:
    path = tmp_path / "goldset.jsonl"
    dump_goldset(
        [
            _item("tune-good", "tuning", "correct answer"),
            _item("tune-bad", "tuning", "reference only"),
            _item("final-1", "final", "final answer"),
        ],
        path,
    )
    return path


def _config(tmp_path: Path, goldset: Path, *, judge_model: str | None = None) -> RunConfig:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    return RunConfig(
        data_dir=tmp_path,
        corpus_root=corpus,
        goldset_path=goldset,
        model=STUDENT,
        backend="vllm",
        judge_model=judge_model,
    )


def _trainer(dataset_dir: Path, model: str, adapter_dir: Path, seed: int) -> JsonObject:
    return fake_train_adapter(dataset_dir=dataset_dir, model=model, out_dir=adapter_dir, seed=seed)


def _comparison(
    _config: RunConfig,
    _adapter_dir: Path,
    _reference_adapter_dir: Path,
    items: list[GoldItem],
    _out_dir: Path,
) -> DistillComparison:
    assert {item.split for item in items} == {"final"}
    return DistillComparison(
        split="final",
        n_items=len(items),
        distilled_objective=0.75,
        reference_objective=0.50,
        delta=0.25,
        distilled_ci=(0.7, 0.8),
        reference_ci=(0.45, 0.55),
    )


def _read_jsonl(path: Path) -> list[JsonObject]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_distill_gate_excludes_teacher_misses_from_training_set(tmp_path: Path):
    goldset = _goldset(tmp_path)
    seen_item_ids: list[str] = []

    def teacher_fn(config: RunConfig, items: list[GoldItem], _root: Path) -> list[TeacherResponse]:
        assert config.model == TEACHER
        seen_item_ids.extend(item.id for item in items)
        return [
            TeacherResponse(item_id="tune-good", answer="correct answer", context="ctx good"),
            TeacherResponse(item_id="tune-bad", answer="unrelated miss", context="ctx bad"),
        ]

    result = run_distillation(
        _config(tmp_path, goldset),
        teacher=TEACHER,
        student=STUDENT,
        gate=0.8,
        out_dir=tmp_path / "distill-run",
        trainer_fn=_trainer,
        teacher_fn=teacher_fn,
        comparison_fn=_comparison,
    )

    assert seen_item_ids == ["tune-bad", "tune-good"]
    assert result.accepted == 1 and result.rejected == 1

    rows = _read_jsonl(result.dataset_dir / "sft.jsonl")
    assert [row["item_id"] for row in rows] == ["tune-good"]
    assert rows[0]["response"] == "correct answer"
    assert "unrelated miss" not in json.dumps(rows, ensure_ascii=False)

    reference_rows = _read_jsonl(result.reference_dataset_dir / "sft.jsonl")
    assert reference_rows[0]["response"] == "correct answer"
    assert reference_rows[0]["distillation_target"] == "reference"

    outputs = _read_jsonl(result.teacher_outputs_path)
    by_id = {str(row["item_id"]): row for row in outputs}
    assert by_id["tune-good"]["accepted"] is True
    assert by_id["tune-bad"]["accepted"] is False

    manifest = json.loads((result.dataset_dir / "dataset_manifest.json").read_text("utf-8"))
    assert manifest["split_counts"] == {"tuning": 1}
    assert manifest["distillation"]["teacher_model"] == TEACHER
    assert manifest["distillation"]["gate_threshold"] == 0.8
    assert set(manifest["distillation"]["gate_scores"]) == {"tune-good"}

    registry = load_registry(registry_path(tmp_path))
    assert result.registered_adapter_id in registry
    entry = registry[str(result.registered_adapter_id)]
    assert entry.dataset_item_ids == ("tune-good",)
    assert entry.eval_summary["delta"] == 0.25

    final_item = [item for item in load_goldset(goldset) if item.split == "final"]
    validate_adapter_for_eval(
        adapter_path=result.adapter_dir,
        items=final_item,
        model=STUDENT,
        registry=registry_path(tmp_path),
    )


def test_distill_identity_and_judge_teacher_guards_refuse_before_generation(tmp_path: Path):
    goldset = _goldset(tmp_path)

    def must_not_generate(
        _config: RunConfig, _items: list[GoldItem], _root: Path
    ) -> list[TeacherResponse]:
        raise AssertionError("teacher should not run")

    with pytest.raises(SystemExit, match="teacher and student must be different"):
        run_distillation(
            _config(tmp_path, goldset),
            teacher=STUDENT,
            student=STUDENT,
            teacher_fn=must_not_generate,
            trainer_fn=_trainer,
            comparison_fn=_comparison,
        )

    with pytest.raises(SystemExit, match="judge model cannot be the distillation teacher"):
        run_distillation(
            _config(tmp_path, goldset, judge_model=TEACHER),
            teacher=TEACHER,
            student=STUDENT,
            teacher_fn=must_not_generate,
            trainer_fn=_trainer,
            comparison_fn=_comparison,
        )

    with pytest.raises(SystemExit, match="adapter comparison needs --backend vllm"):
        run_distillation(
            _config(tmp_path, goldset).with_overrides(backend="ollama"),
            teacher=TEACHER,
            student=STUDENT,
            teacher_fn=must_not_generate,
            trainer_fn=_trainer,
        )


def test_distill_report_records_paired_comparison_math(tmp_path: Path):
    goldset = _goldset(tmp_path)

    def teacher_fn(_config: RunConfig, items: list[GoldItem], _root: Path) -> list[TeacherResponse]:
        assert {item.split for item in items} == {"tuning"}
        return [
            TeacherResponse(item_id=item.id, answer=item.reference_answer, context=f"ctx {item.id}")
            for item in items
        ]

    result = run_distillation(
        _config(tmp_path, goldset),
        teacher=TEACHER,
        student=STUDENT,
        gate=0.1,
        out_dir=tmp_path / "distill-run",
        trainer_fn=_trainer,
        teacher_fn=teacher_fn,
        comparison_fn=_comparison,
    )

    report = result.report_path.read_text(encoding="utf-8")
    assert "reference-sft" in report
    assert "Delta distilled-minus-reference: `0.2500`" in report
    assert "Verdict: `distilled-better`" in report
    distill_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert distill_manifest["comparison"]["delta"] == 0.25
    assert distill_manifest["n_accepted"] == 2
