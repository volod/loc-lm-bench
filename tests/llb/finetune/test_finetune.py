"""Local model self-improvement control-plane tests."""

import json
from pathlib import Path

import pytest

from llb.finetune.dataset import export_finetune_set
from llb.finetune.guard import validate_adapter_for_eval
from llb.finetune.adapter_manifest import load_adapter_manifest
from llb.finetune.trainer import fake_train_adapter
from llb.goldset.schema import GoldItem, dump_goldset


def _item(item_id: str, split: str) -> GoldItem:
    return GoldItem(
        id=item_id,
        question=f"Question {item_id}?",
        reference_answer=f"Answer {item_id}",
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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_bundle(
    root: Path, name: str, split: str, items: list[GoldItem], objective: float
) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "run_name": name,
                "split": split,
                "config": {
                    "model": "base-model",
                    "backend": "vllm",
                    "goldset_path": str(root / "goldset.jsonl"),
                },
                "metrics": {"objective_score": objective, "reliability": 1.0, "tokens_per_s": 1.0},
                "n_cases": len(items),
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        run / "scores.jsonl",
        [
            {
                "item_id": item.id,
                "split": item.split,
                "status": "ok",
                "objective_score": objective,
                "token_f1": objective,
                "exact": 0.0,
                "contains": 0.0,
                "retrieval_hit": True,
                "first_hit_rank": 1,
                "tokens_per_s": 1.0,
                "latency_s": 0.1,
                "completion_tokens": 1,
                "answer_preview": "wrong answer",
            }
            for item in items
        ],
    )
    _write_jsonl(
        run / "retrieval.jsonl",
        [
            {
                "item_id": item.id,
                "retrieved": [
                    {
                        "doc_id": item.source_doc_id,
                        "char_start": 0,
                        "char_end": 5,
                        "rank": 1,
                        "text_preview": "alpha",
                    }
                ],
                "gold_spans": [span.model_dump() for span in item.source_spans],
            }
            for item in items
        ],
    )
    return run


def test_export_finetune_set_uses_only_tuning_and_builds_dpo(tmp_path: Path):
    tuning = _item("tune-1", "tuning")
    final = _item("final-1", "final")
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset([tuning, final], goldset)
    run = _write_bundle(tmp_path, "run-tuning", "tuning", [tuning, final], 0.0)
    misses = tmp_path / "misses.jsonl"
    _write_jsonl(
        misses,
        [
            {
                "item_id": "tune-1",
                "miss_class": "generation_miss",
                "topic": "topic-a",
            }
        ],
    )

    manifest = export_finetune_set(
        run_dir=run,
        goldset_path=goldset,
        out_dir=tmp_path / "dataset",
        misses_path=misses,
    )

    assert manifest["item_ids"] == ["tune-1"]
    assert manifest["split_counts"] == {"tuning": 1}
    sft = [
        json.loads(line) for line in (tmp_path / "dataset" / "sft.jsonl").read_text().splitlines()
    ]
    dpo = [
        json.loads(line) for line in (tmp_path / "dataset" / "dpo.jsonl").read_text().splitlines()
    ]
    assert sft[0]["messages"][0]["role"] == "system"
    assert sft[0]["response"] == "Answer tune-1"
    assert dpo[0]["chosen"] == "Answer tune-1"
    assert dpo[0]["rejected"] == "wrong answer"


def test_fake_trainer_records_adapter_provenance(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_digest": "abc",
                "item_ids": ["tune-1"],
                "split_counts": {"tuning": 1},
            }
        ),
        encoding="utf-8",
    )

    manifest = fake_train_adapter(
        dataset_dir=dataset_dir,
        model="base-model",
        out_dir=tmp_path / "adapter",
        seed=7,
    )

    assert manifest["base_model"] == "base-model"
    assert manifest["dataset_digest"] == "abc"
    assert manifest["dataset_item_ids"] == ["tune-1"]
    assert (
        load_adapter_manifest(tmp_path / "adapter")["adapter_digest"] == manifest["adapter_digest"]
    )


def test_contamination_guard_refuses_protected_split_ids(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_manifest.json").write_text(
        json.dumps(
            {
                "dataset_item_ids": ["final-1"],
                "dataset_split_counts": {"final": 1},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="final-1"):
        validate_adapter_for_eval(
            adapter_path=adapter,
            items=[_item("final-1", "final")],
            model="base-model",
        )
