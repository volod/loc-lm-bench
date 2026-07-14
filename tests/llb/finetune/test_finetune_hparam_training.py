"""Tests for finetune hparam training."""

from pathlib import Path
from llb.core.config import RunConfig
from llb.core.contracts import EvalResult
from llb.finetune.dataset import subset_dataset
from llb.finetune.hparam_search.manifest_io import (
    latest_hparams_manifest,
    trainer_defaults,
)
from llb.finetune.loop import run_self_improve
from llb.finetune.adapter_manifest import load_adapter_manifest
from llb.finetune.trainer import fake_train_adapter
from llb.goldset.schema import dump_goldset, load_goldset
from finetune_hparam_helpers import (
    MODEL,
    TUNING_IDS,
    _dataset,
    _item,
    _write_bundle,
    _write_hparams_manifest,
)


def test_subset_dataset_recomputes_the_digest(tmp_path: Path):
    dataset = _dataset(tmp_path, digest="parent-digest")

    manifest = subset_dataset(
        dataset_dir=dataset,
        out_dir=tmp_path / "subset",
        item_ids=TUNING_IDS[:3],
        role="train",
    )

    assert manifest["item_ids"] == TUNING_IDS[:3]
    assert manifest["split_counts"] == {"tuning": 3}
    assert manifest["parent_dataset_digest"] == "parent-digest"
    assert manifest["dataset_digest"] != "parent-digest", "a subset is not its parent"
    assert manifest["subset_role"] == "train"


def test_trainer_consumes_the_recorded_best_config(tmp_path: Path):
    best = {"method": "lora", "lora_r": 32, "lora_alpha": 64, "learning_rate": 0.0001}
    manifest_path = _write_hparams_manifest(tmp_path, MODEL, best)
    dataset = _dataset(tmp_path)

    assert latest_hparams_manifest(tmp_path, MODEL) == manifest_path
    defaults = trainer_defaults(tmp_path, MODEL)
    assert defaults["hyperparameters"] == best
    assert defaults["hparams_manifest"] == str(manifest_path)

    fake_train_adapter(dataset_dir=dataset, model=MODEL, out_dir=tmp_path / "adapter", **defaults)

    adapter = load_adapter_manifest(tmp_path / "adapter")
    assert adapter["hyperparameters"]["lora_r"] == 32
    assert adapter["hparams_manifest"] == str(manifest_path)


def test_trainer_defaults_are_empty_without_a_recorded_search(tmp_path: Path):
    assert trainer_defaults(tmp_path, MODEL) == {}
    _write_hparams_manifest(tmp_path, MODEL, {})
    assert trainer_defaults(tmp_path, MODEL) == {}, "a study with no best config sets no default"


def test_self_improve_round_records_the_searched_config_in_provenance(tmp_path: Path):
    best = {"method": "lora", "lora_r": 8, "lora_alpha": 32, "num_train_epochs": 2.0}
    manifest_path = _write_hparams_manifest(tmp_path, MODEL, best)
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset([_item("tune-1", "tuning"), _item("final-1", "final")], goldset)
    config = RunConfig(data_dir=tmp_path, goldset_path=goldset, model=MODEL, backend="vllm")

    def eval_fn(cfg: RunConfig, split: str, _round_dir: Path) -> EvalResult:
        items = [item for item in load_goldset(goldset) if item.split == split]
        objective = 0.8 if cfg.adapter_path is not None and split == "final" else 0.2
        run = _write_bundle(tmp_path, f"{split}-{objective}", split, items, objective)
        return {
            "rows": [],
            "metrics": {"objective_score": objective, "reliability": 1.0, "tokens_per_s": 1.0},
            "retrieval": {"n": len(items), "k": 1, "recall_at_k": 1.0, "mrr": 1.0},
            "paths": {"manifest": str(run / "manifest.json"), "scores": str(run / "scores.jsonl")},
            "table": "",
            "telemetry": None,
            "manifest": None,
            "run_timestamp": run.name,
        }

    result = run_self_improve(
        config, rounds=1, out_dir=tmp_path / "campaign", trainer="fake", eval_fn=eval_fn
    )

    adapter = load_adapter_manifest(result.rounds[0].adapter_dir)
    assert adapter["hyperparameters"]["lora_r"] == 8
    assert adapter["hyperparameters"]["num_train_epochs"] == 2.0
    assert adapter["hparams_manifest"] == str(manifest_path)
