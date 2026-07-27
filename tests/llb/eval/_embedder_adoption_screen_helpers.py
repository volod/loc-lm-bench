"""adoption-bar-per-model-screen -- what does deciding the reranker question for ONE model cost?

Pure and file-driven: per-item deltas come back from run bundles on disk and the resampling study
is pure Python, so the whole vertical is unit-tested with dict rows -- no backend, store, or GPU.
"""

import json
from pathlib import Path

from llb.eval.embedder_adoption.cross_model import READING_ANSWER
from llb.eval.embedder_adoption.models import ItemDeltas
from llb.rag.fusion_evidence.stats import bootstrap_index_sets

FOCUS = "k10+rerank"

BASELINE = "intfloat/multilingual-e5-base"

CANDIDATE = "BAAI/bge-m3"


def _deltas(objective: list[float], rr: list[float] | None = None) -> ItemDeltas:
    return ItemDeltas(
        item_ids=[f"q{i}" for i in range(len(objective))],
        objective=objective,
        reciprocal_rank=rr if rr is not None else [0.0] * len(objective),
    )


def _index_sets(n: int, resamples: int = 300):
    return bootstrap_index_sets(n, resamples, 13)


def _screen(model: str, min_size: int | None, n: int = 40):
    return {
        "model": model,
        "n": n,
        "full_reading": READING_ANSWER,
        "recorded_reading": READING_ANSWER,
        "reproduced": True,
        "sizes": [],
        "min_size": min_size,
    }


def _row(item_id: str, objective: float, rank: int | None) -> dict:
    return {
        "item_id": item_id,
        "split": "final",
        "status": "ok",
        "objective_score": objective,
        "token_f1": objective,
        "retrieval_hit": 1.0 if rank else 0.0,
        "first_hit_rank": rank,
    }


def _bundle(path: Path, rows: list[dict]) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return str(path)


def _paired(mean: float, lo: float, hi: float, *, wins: int, losses: int = 0) -> dict:
    """A finished paired block, ledger included -- the gate reads the ledger, not only the bounds."""
    return {
        "delta": {"mean": mean, "lo": lo, "hi": hi},
        "wins": wins,
        "losses": losses,
        "ties": 0,
        "sign_test_p": 0.0,
    }


def _sweep_report(tmp_path: Path, *, base: list[dict], cand: list[dict], model="m") -> dict:
    """A finished-sweep shape whose focus cell names real bundles on disk."""
    return {
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "item_ids": [r["item_id"] for r in base],
        "metrics": [],
        "resamples": 200,
        "confidence": 0.95,
        "seed": 13,
        "cells": [
            {
                "label": "k10",
                "top_k": 10,
                "reranker": None,
                "n": len(base),
                "lanes": {
                    BASELINE: {"run_dirs": [_bundle(tmp_path / "k10-b", base)], "metrics": {}},
                    CANDIDATE: {"run_dirs": [_bundle(tmp_path / "k10-c", cand)], "metrics": {}},
                },
                "paired": {},
            },
            {
                "label": FOCUS,
                "top_k": 10,
                "reranker": "x",
                "n": len(base),
                "lanes": {
                    BASELINE: {"run_dirs": [_bundle(tmp_path / "f-b", base)], "metrics": {}},
                    CANDIDATE: {"run_dirs": [_bundle(tmp_path / "f-c", cand)], "metrics": {}},
                },
                "paired": {
                    metric: _paired(0.5, 0.4, 0.6, wins=len(base))
                    for metric in ("objective_score", "reciprocal_rank")
                },
            },
        ],
        "verdict": {"decision": "extend_bar"},
        "metadata": {"model": model, "goldset": "gs.jsonl", "corpus": "c", "split": "final"},
    }


def _wide_sweep(tmp_path: Path, model: str, n: int = 24) -> dict:
    base = [_row(f"q{i}", 0.0, 4) for i in range(n)]
    cand = [_row(f"q{i}", 1.0, 1) for i in range(n)]
    return _sweep_report(tmp_path / model, base=base, cand=cand, model=model)
