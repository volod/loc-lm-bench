"""Tests for recommend runs."""

import json
from llb.board.recommend.build import (
    load_run_summaries,
)


def test_load_run_summaries_keeps_best_top_k_per_model(tmp_path):
    # The RAG-grid use case: the same model swept at two top_k values. best-per-model keeps the
    # higher-objective cell, so the model is represented by its BEST retrieval config.
    def write_bundle(name, top_k, obj):
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(
            json.dumps(
                {
                    "split": "final",
                    "n_cases": 82,
                    "config": {"model": "modelA", "backend": "ollama", "top_k": top_k},
                    "metrics": {"objective_score": obj, "reliability": 1.0, "tokens_per_s": 10.0},
                    "telemetry": {"peak_vram_mb": 9000},
                    "retrieval": {"recall": 0.9, "mrr": 0.8},
                }
            ),
            encoding="utf-8",
        )

    write_bundle("k3", 3, 0.40)
    write_bundle("k8", 8, 0.55)

    kept = load_run_summaries(tmp_path)
    assert len(kept) == 1
    assert kept[0].record.config["top_k"] == 8 and kept[0].result.objective_score == 0.55
