"""Tests for self improve loop."""

import json
from pathlib import Path
import pytest
from llb.core.config import RunConfig
from llb.core.contracts import EvalResult
from llb.finetune.loop import run_self_improve
from llb.finetune.registry.io import load_registry, registry_path
from llb.goldset.schema import dump_goldset, load_goldset
from test_finetune import _item, _write_bundle


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
