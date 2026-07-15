"""Tests for finetune campaign resume."""

from pathlib import Path
import pytest
from llb.core.config import RunConfig
from llb.core.contracts.runs import EvalResult
from llb.finetune.campaign.run import run_finetune_campaign
from llb.finetune.trainer import fake_train_adapter
from llb.goldset.schema import dump_goldset, load_goldset
from test_finetune import _item, _write_bundle


@pytest.mark.slow
def test_finetune_campaign_resume_does_not_retrain_completed_entry(tmp_path: Path):
    tuning = _item("tune-1", "tuning")
    final = _item("final-1", "final")
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset([tuning, final], goldset)
    cfg = RunConfig(data_dir=tmp_path, goldset_path=goldset, model="seed-model", backend="vllm")
    train_calls: list[str] = []
    eval_count = 0

    def eval_fn(config: RunConfig, split: str, round_dir: Path) -> EvalResult:
        nonlocal eval_count
        eval_count += 1
        items = [item for item in load_goldset(goldset) if item.split == split]
        objective = 0.6 if config.adapter_path else 0.2
        run = _write_bundle(
            tmp_path, f"resume-{eval_count}-{config.model}-{split}", split, items, objective
        )
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
            "verdict": "gpu",
            "note": "plan @ ctx=2048",
        }

    out_dir = tmp_path / "resume-campaign"
    run_finetune_campaign(
        cfg,
        models=["model-a"],
        rounds=1,
        out_dir=out_dir,
        eval_fn=eval_fn,
        trainer_fn=trainer_fn,
        planner_fn=planner_fn,
        reclaim_fn=lambda: {"reclaimed": True},
    )
    run_finetune_campaign(
        cfg,
        models=["model-a,model-b"],
        rounds=1,
        resume=out_dir,
        eval_fn=eval_fn,
        trainer_fn=trainer_fn,
        planner_fn=planner_fn,
        reclaim_fn=lambda: {"reclaimed": True},
    )

    assert train_calls == ["model-a", "model-b"]
    progress = (out_dir / "campaign.progress.jsonl").read_text(encoding="utf-8")
    assert progress.count('"model": "model-a"') == 1
    assert progress.count('"model": "model-b"') == 1
