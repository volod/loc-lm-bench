"""Write the SFT/DPO training datasets (teacher and reference targets) and their manifests."""

import json
from collections import Counter
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.finetune.dataset import (
    DATASET_MANIFEST,
    DPO_FILENAME,
    SFT_FILENAME,
    dataset_digest,
)
from llb.finetune.distill.gate import _messages_for_record
from llb.finetune.distill.model import TEACHER_TARGET, GatedTeacherRecord


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


def _write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )
