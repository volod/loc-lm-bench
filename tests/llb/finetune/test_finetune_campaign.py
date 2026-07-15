"""Tests for finetune campaign."""

import json
from pathlib import Path
import pytest
from llb.core.config import RunConfig
from llb.core.contracts.runs import EvalResult
from llb.finetune.campaign.model import COMPLETE_VERDICT, SKIP_VERDICT
from llb.finetune.campaign.run import run_finetune_campaign
from llb.finetune.registry.io import load_registry, registry_path
from llb.finetune.trainer import fake_train_adapter
from llb.goldset.schema import dump_goldset, load_goldset
from test_finetune import _item, _write_bundle


@pytest.mark.slow
def test_finetune_campaign_skips_infeasible_and_ranks_tunability(tmp_path: Path):
    tuning = _item("tune-1", "tuning")
    final = _item("final-1", "final")
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset([tuning, final], goldset)
    cfg = RunConfig(data_dir=tmp_path, goldset_path=goldset, model="seed-model", backend="vllm")
    eval_calls: list[tuple[str, str, bool]] = []
    train_calls: list[str] = []

    gains = {"model-a": 0.3, "model-b": 0.1}

    def eval_fn(config: RunConfig, split: str, round_dir: Path) -> EvalResult:
        eval_calls.append((config.model, split, config.adapter_path is not None))
        items = [item for item in load_goldset(goldset) if item.split == split]
        base = 0.2
        objective = base + gains.get(config.model, 0.0) if config.adapter_path else base
        run = _write_bundle(
            tmp_path,
            f"{len(eval_calls)}-{config.model.replace('/', '-')}-{split}",
            split,
            items,
            objective,
        )
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        manifest["config"]["model"] = config.model
        manifest["telemetry"] = {"peak_vram_mb": 1000 + len(eval_calls)}
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
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

    def trainer_fn(dataset_dir: Path, model: str, out_dir: Path, seed: int):
        train_calls.append(model)
        return fake_train_adapter(dataset_dir=dataset_dir, model=model, out_dir=out_dir, seed=seed)

    def planner_fn(model: str, config: RunConfig):
        verdict = "no" if model == "too-big" else "gpu"
        return {
            "name": model,
            "backend": config.backend,
            "params_b": 1.0,
            "quant": "q4",
            "weights_mib": 100.0,
            "n_layers": 1,
            "ctx_gpu": 2048,
            "ctx_max": 2048,
            "gpu_layers": 1,
            "verdict": verdict,
            "note": "does not fit" if verdict == "no" else "plan @ ctx=2048",
        }

    result = run_finetune_campaign(
        cfg,
        models=["too-big,model-b,model-a"],
        rounds=1,
        out_dir=tmp_path / "ft-campaign",
        eval_fn=eval_fn,
        trainer_fn=trainer_fn,
        planner_fn=planner_fn,
        reclaim_fn=lambda: {"reclaimed": True, "polls": 1, "residual_mb": 0},
    )

    assert [entry.model for entry in result.entries] == ["too-big", "model-b", "model-a"]
    assert result.entries[0].status == SKIP_VERDICT
    completed = [entry for entry in result.entries if entry.status == COMPLETE_VERDICT]
    assert {entry.shared_dataset_digest for entry in completed} == {
        json.loads(
            (result.shared_dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )["dataset_digest"]
    }
    assert train_calls == ["model-b", "model-a"]
    report = (tmp_path / "ft-campaign" / "report.md").read_text(encoding="utf-8")
    assert "| 1 | model-a |" in report
    assert "| 2 | model-b |" in report
    assert "skipped: does not fit" in report

    registered = load_registry(registry_path(tmp_path))
    assert {entry.base_model for entry in registered.values()} == {"model-a", "model-b"}
    by_model = {entry.base_model: entry for entry in registered.values()}
    assert by_model["model-a"].eval_summary["delta"] == pytest.approx(0.3)
    assert by_model["model-b"].eval_summary["delta"] == pytest.approx(0.1)


def test_finetune_campaign_skips_compressed_checkpoint_before_training(tmp_path: Path):
    """compressed-qat-adapter-support: the compat probe skips BEFORE base eval or training."""
    tuning = _item("tune-1", "tuning")
    final = _item("final-1", "final")
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset([tuning, final], goldset)
    cfg = RunConfig(data_dir=tmp_path, goldset_path=goldset, model="seed-model", backend="vllm")
    eval_calls: list[str] = []
    train_calls: list[str] = []

    def eval_fn(config: RunConfig, split: str, round_dir: Path) -> EvalResult:
        eval_calls.append(config.model)
        items = [item for item in load_goldset(goldset) if item.split == split]
        run = _write_bundle(
            tmp_path,
            f"{len(eval_calls)}-{config.model.replace('/', '-')}-{split}",
            split,
            items,
            0.2,
        )
        return {
            "rows": [],
            "metrics": {"objective_score": 0.2, "reliability": 1.0, "tokens_per_s": 1.0},
            "retrieval": {"n": len(items), "k": 1, "recall_at_k": 1.0, "mrr": 1.0},
            "paths": {"manifest": str(run / "manifest.json"), "scores": str(run / "scores.jsonl")},
            "table": "",
            "telemetry": None,
            "manifest": None,
            "run_timestamp": run.name,
        }

    def trainer_fn(dataset_dir: Path, model: str, out_dir: Path, seed: int):
        train_calls.append(model)
        return fake_train_adapter(dataset_dir=dataset_dir, model=model, out_dir=out_dir, seed=seed)

    def planner_fn(model: str, config: RunConfig):
        return {"name": model, "backend": config.backend, "verdict": "gpu", "note": "fits"}

    def compat_fn(model: str):
        if model == "fake/qat-w4a16-ct":
            return {
                "verdict": "not-trainable",
                "blocker": "native quant_method 'compressed-tensors' has no PEFT LoRA dispatch",
            }
        return {"verdict": "trainable", "injection_strategy": "peft-lora", "blocker": None}

    result = run_finetune_campaign(
        cfg,
        models=["fake/qat-w4a16-ct,model-a"],
        rounds=1,
        out_dir=tmp_path / "ft-campaign",
        eval_fn=eval_fn,
        trainer_fn=trainer_fn,
        planner_fn=planner_fn,
        reclaim_fn=lambda: {"reclaimed": True, "polls": 1, "residual_mb": 0},
        compat_fn=compat_fn,
    )

    skipped = result.entries[0]
    assert skipped.status == SKIP_VERDICT
    assert "compressed-tensors" in (skipped.reason or "")
    assert skipped.compat["verdict"] == "not-trainable"
    # the compressed model never reached an eval or the trainer
    assert all(model != "fake/qat-w4a16-ct" for model in eval_calls)
    assert train_calls == ["model-a"]
    progress = (tmp_path / "ft-campaign" / "campaign.progress.jsonl").read_text(encoding="utf-8")
    assert "compressed-tensors" in progress
