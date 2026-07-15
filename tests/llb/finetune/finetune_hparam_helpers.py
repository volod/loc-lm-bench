"""Budgeted LoRA hyperparameter search: split discipline, determinism, budget, and consumption."""

import json
from pathlib import Path


from llb.core.config import RunConfig
from llb.core.contracts.common import JsonObject
from llb.finetune.hparam_search.model import (
    HPARAMS_MANIFEST,
    HPARAMS_METHOD,
)
from llb.finetune.naming import model_slug
from llb.finetune.trainer import fake_train_adapter
from llb.goldset.schema import GoldItem, dump_goldset

MODEL = "base-model"
TUNING_IDS = [f"tune-{index}" for index in range(8)]


def _item(item_id: str, split: str) -> GoldItem:
    return GoldItem(
        id=item_id,
        question=f"Question {item_id}?",
        reference_answer=f"Answer {item_id}",
        source_doc_id=f"{item_id}.txt",
        source_spans=[
            {"doc_id": f"{item_id}.txt", "char_start": 0, "char_end": 5, "text": "alpha"}
        ],
        provenance="human-authored",
        verified=True,
        split=split,
    )


def _goldset(tmp_path: Path) -> Path:
    path = tmp_path / "goldset.jsonl"
    items = [_item(item_id, "tuning") for item_id in TUNING_IDS]
    items += [_item("cal-1", "calibration"), _item("final-1", "final")]
    dump_goldset(items, path)
    return path


def _dataset(tmp_path: Path, *, item_ids: list[str] | None = None, digest: str = "d0") -> Path:
    """A tuning-split export, as `export-finetune-set` would leave it."""
    ids = item_ids or TUNING_IDS
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "item_id": item_id,
            "split": "tuning",
            "weight": 1.0,
            "messages": [{"role": "user", "content": item_id}],
            "response": f"Answer {item_id}",
        }
        for item_id in ids
    ]
    (dataset_dir / "sft.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (dataset_dir / "dpo.jsonl").write_text("", encoding="utf-8")
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "kind": "llb.finetune.dataset",
                "dataset_digest": digest,
                "item_ids": ids,
                "split_counts": {"tuning": len(ids)},
                "n_sft": len(ids),
                "n_dpo": 0,
                "prompt_template": "eval.rag.chat",
            }
        ),
        encoding="utf-8",
    )
    return dataset_dir


def _config(tmp_path: Path, goldset: Path | None = None) -> RunConfig:
    return RunConfig(
        data_dir=tmp_path,
        model=MODEL,
        backend="vllm",
        goldset_path=goldset or (tmp_path / "missing.jsonl"),
    )


def _trainer_fn(dataset_dir: Path, model: str, adapter_dir: Path, seed: int, params: JsonObject):
    return fake_train_adapter(
        dataset_dir=dataset_dir, model=model, out_dir=adapter_dir, seed=seed, hyperparameters=params
    )


def _rank_objective(_adapter_dir: Path, params: JsonObject) -> float:
    """Deterministic synthetic objective peaking at rank 16."""
    return 1.0 - abs(int(params["lora_r"]) - 16) / 64.0


def _fake_clock(step_s: float):
    """A clock that advances `step_s` on every read, so a wall-clock budget is testable."""
    ticks = iter(index * step_s for index in range(10_000))
    return lambda: next(ticks)


def _write_hparams_manifest(tmp_path: Path, model: str, best: JsonObject) -> Path:
    out = tmp_path / HPARAMS_METHOD / model_slug(model) / "20260101-000000"
    out.mkdir(parents=True)
    manifest = out / HPARAMS_MANIFEST
    manifest.write_text(
        json.dumps({"kind": "llb.finetune.hparams", "model": model, "best_hyperparameters": best}),
        encoding="utf-8",
    )
    return manifest


# finetune-hparams-stratified-dev-slice: 12 committed scores, 3 answerable (tune-0/1/2).
BASE_SCORE_RUN = Path("samples/finetune/base-score-run")
STRATIFY_IDS = [f"tune-{index}" for index in range(12)]


# finetune-hparams-infeasible-point-prune: a fixture arch a 16 GB-class host can reason about.
FIXTURE_ARCH = {"hidden_size": 4096, "n_layers": 32}


def _write_bundle(
    root: Path, name: str, split: str, items: list[GoldItem], objective: float
) -> Path:
    run = root / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "split": split,
                "config": {"model": MODEL, "backend": "vllm", "goldset_path": str(root)},
                "metrics": {"objective_score": objective, "reliability": 1.0, "tokens_per_s": 1.0},
                "n_cases": len(items),
            }
        ),
        encoding="utf-8",
    )
    (run / "scores.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "item_id": item.id,
                    "split": item.split,
                    "status": "ok",
                    "objective_score": objective,
                    "answer_preview": "wrong answer",
                }
            )
            + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    (run / "retrieval.jsonl").write_text(
        "".join(
            json.dumps(
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
            )
            + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    return run
