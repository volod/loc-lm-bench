"""Recommendation bundle schema and evidence rendering."""

from pathlib import Path

import yaml

from llb.auto_rag.recommendation import write_recommendation


def test_recommendation_contains_every_user_visible_knob(tmp_path: Path) -> None:
    outputs = {
        "verification": {
            "policy": "local",
            "n_total": 3,
            "n_accepted": 3,
            "accept_rate": 1.0,
        },
        "retrieval": {
            "selected": {
                "strategy": "recursive",
                "chunk_size": 800,
                "chunk_overlap": 120,
                "retrieval_mode": "hybrid",
            },
            "k": 10,
            "metrics": {"recall_at_k": 1.0, "mrr": 0.75},
            "repaired": False,
        },
        "joint_search": {
            "run_dir": "joint",
            "scoreboard": "scoreboard.json",
            "recommended": {
                "model": "mamaylm-v2-12b",
                "source": "mamay:12b",
                "backend": "ollama",
                "quality": 0.8,
                "overrides": {
                    "top_k": 3,
                    "fusion_weight": 0.6,
                    "reranker": "reranker",
                    "query_prep": ["normalize"],
                    "context_budget": 4096,
                },
            },
        },
        "prompt_system": {
            "prompt_system_id": "tree-prompt",
            "package": "prompt-package",
            "knowledge_tree": {"depth": 2},
        },
        "final_eval": {
            "quality": 0.75,
            "n_cases": 1,
            "split": "final",
            "manifest": "manifest.json",
            "scores": "scores.jsonl",
            "retrieval": {},
        },
    }
    result = write_recommendation(tmp_path, outputs)
    payload = yaml.safe_load(Path(result["recommendation"]).read_text(encoding="utf-8"))
    assert payload["model"] == "mamay:12b"
    assert payload["retrieval"]["mode"] == "hybrid"
    assert payload["retrieval"]["fusion_weight"] == 0.6
    assert payload["retrieval"]["reranker"] == "reranker"
    assert payload["retrieval"]["query_prep"] == ["normalize"]
    assert payload["retrieval"]["context_budget"] == 4096
    assert payload["prompt_system"]["id"] == "tree-prompt"
    assert Path(result["report"]).is_file()
