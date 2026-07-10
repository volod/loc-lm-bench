"""Local model self-improvement control-plane tests."""

import json
from pathlib import Path

import pytest

from llb.backends.vllm import build_vllm_command, served_lora_rank
from llb.core.config import RunConfig
from llb.core.contracts import EvalResult
from llb.finetune.campaign import (
    COMPLETE_VERDICT,
    SKIP_VERDICT,
    run_finetune_campaign,
)
from llb.finetune.dataset import export_finetune_set
from llb.finetune.guard import validate_adapter_for_eval
from llb.finetune.loop import run_self_improve
from llb.finetune.registry import load_registry, registry_path
from llb.finetune.trainer import adapter_lora_rank, fake_train_adapter, load_adapter_manifest
from llb.goldset.schema import GoldItem, dump_goldset, load_goldset


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


def test_vllm_command_enables_lora_module():
    cmd = build_vllm_command("base-model", adapter_path="/tmp/adapter", adapter_name="adapter")
    assert "--enable-lora" in cmd
    assert "--lora-modules" in cmd
    assert "adapter=/tmp/adapter" in cmd
    assert "--max-lora-rank" not in cmd, "an unknown rank leaves vLLM on its own default"


def test_vllm_command_sizes_max_lora_rank_to_the_adapter():
    """vLLM defaults `--max-lora-rank` to 16, so a rank-32 adapter fails `add_lora` without this."""
    cmd = build_vllm_command("base-model", adapter_path="/tmp/adapter", max_lora_rank=32)

    assert cmd[cmd.index("--max-lora-rank") + 1] == "32"


def test_max_lora_rank_rounds_up_to_a_value_vllm_accepts():
    assert served_lora_rank(4) == 8, "vLLM accepts 1, 8, 16, ... -- never 4"
    assert served_lora_rank(16) == 16
    assert served_lora_rank(17) == 32

    with pytest.raises(SystemExit, match="exceeds the largest servable rank"):
        served_lora_rank(1024)


def test_max_lora_rank_is_omitted_when_no_adapter_is_served():
    assert "--max-lora-rank" not in build_vllm_command("base-model", max_lora_rank=64)


def test_adapter_lora_rank_prefers_the_peft_config(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_manifest.json").write_text(
        json.dumps({"hyperparameters": {"lora_r": 8}}), encoding="utf-8"
    )
    assert adapter_lora_rank(adapter) == 8, "fall back to our manifest when PEFT wrote no config"

    (adapter / "adapter_config.json").write_text(json.dumps({"r": 64}), encoding="utf-8")

    assert adapter_lora_rank(adapter) == 64, "PEFT's config describes the weights actually on disk"
    assert adapter_lora_rank(None) is None
    assert adapter_lora_rank(tmp_path / "missing") is None


def test_self_improve_fake_loop_writes_round_state(tmp_path: Path):
    tuning = _item("tune-1", "tuning")
    final = _item("final-1", "final")
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset([tuning, final], goldset)
    cfg = RunConfig(data_dir=tmp_path, goldset_path=goldset, model="base-model", backend="vllm")
    calls: list[str] = []

    def eval_fn(config: RunConfig, split: str, round_dir: Path) -> EvalResult:
        calls.append(split)
        items = [item for item in load_goldset(goldset) if item.split == split]
        objective = 0.8 if config.adapter_path is not None and split == "final" else 0.2
        run = _write_bundle(tmp_path, f"{len(calls)}-{split}", split, items, objective)
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
        cfg,
        rounds=1,
        out_dir=tmp_path / "campaign",
        trainer="fake",
        eval_fn=eval_fn,
    )

    assert calls == ["final", "tuning", "final"]
    assert result.verdict == "accept"
    state = json.loads((tmp_path / "campaign" / "state.json").read_text(encoding="utf-8"))
    assert state["rounds"][0]["delta"] == pytest.approx(0.6)

    registered = list(load_registry(registry_path(tmp_path)).values())
    assert len(registered) == 1
    assert registered[0].base_model == "base-model"
    assert registered[0].eval_summary["delta"] == pytest.approx(0.6)
    assert registered[0].eval_summary["verdict"] == "accept"
    assert registered[0].goldset_digest is not None


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
