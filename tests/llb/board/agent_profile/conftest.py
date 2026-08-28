"""Fixture bundles for the agent-profile composition tests.

Every lane is written as the real lane writes it (a run directory named with the bundle timestamp,
holding the one artifact the profile reads), so the composition is exercised over the same shapes
that exist on a host -- no GPU, no model, no network.
"""

import json
from pathlib import Path

import pytest

from llb.board.recommend.build import build_recommendation
from llb.board.recommend.model import HostInfo, RunSummary
from llb.board.runs import RunRecord
from llb.scoring.leaderboard import ModelResult

MODEL = "mamaylm-v2-12b"
CORPUS = "/corpora/ua"
STORE = {
    "strategy": "recursive",
    "chunk_size": 800,
    "chunk_overlap": 120,
    "embedding_model": "intfloat/multilingual-e5-base",
    "retrieval_mode": "flat",
}
# The store metadata keys the retrieval fingerprint is read back through (`RETRIEVAL_FINGERPRINT_KEYS`).
STORE_META = {
    "strategy": "recursive",
    "size": 800,
    "overlap": 120,
    "embedding_model": "intfloat/multilingual-e5-base",
    "mode": "flat",
}
RUN_DIR_NAME = "20260828T101010.000000Z-aaaabbbbcccc"


def run_summary(tmp_path: Path, *, model: str = MODEL, **config_overrides: object) -> RunSummary:
    """One final-split run bundle on disk plus the in-memory summary the board would build."""
    run_dir = tmp_path / "run-eval" / RUN_DIR_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **STORE,
        "model": model,
        "backend": "ollama",
        "top_k": 3,
        "context_budget": 8192,
        "corpus_root": CORPUS,
        **config_overrides,
    }
    (run_dir / "manifest.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    result = ModelResult(
        model=model,
        backend="ollama",
        objective_score=0.6,
        n_cases=82,
        reliability=1.0,
        tokens_per_s=14.0,
        peak_vram_mb=9000,
        case_objectives=[0.6] * 82,
    )
    record = RunRecord(
        result=result,
        config=config,
        run_dir=str(run_dir),
        created_at="2026-08-28T10:10:10+00:00",
        split="final",
    )
    return RunSummary(record, quality_per_watt=0.1, mean_power_w=100.0, recall_at_k=0.95, mrr=0.86)


@pytest.fixture
def recommendation(tmp_path):
    """A one-model recommendation whose host pick anchors the profile."""
    return build_recommendation([run_summary(tmp_path)], HostInfo(16, 16380, "RTX 4060 Ti", True))


@pytest.fixture
def index_dir(tmp_path):
    """A built store whose retrieval fingerprint matches the anchor run."""
    path = tmp_path / "llb" / "rag"
    path.mkdir(parents=True, exist_ok=True)
    (path / "store_meta.json").write_text(json.dumps(STORE_META), encoding="utf-8")
    return path


def write_bundle(
    tmp_path: Path, method: str, filename: str, payload: object, *, run: str = RUN_DIR_NAME
) -> Path:
    """Write one lane artifact under `<tmp>/<method>/<run>/<filename>` and return its path."""
    run_dir = tmp_path / method / run
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_reranker(
    tmp_path: Path, *, model: str = "BAAI/bge-reranker-v2-m3", corpus: str = CORPUS
) -> Path:
    return write_bundle(
        tmp_path,
        "compare-rerankers",
        "report.json",
        {
            "corpus_root": corpus,
            "embedding_model": STORE["embedding_model"],
            "verdict": {
                "model": model,
                "decision": "retain",
                "reason": "no candidate clears a bar",
            },
            "candidates": [
                {
                    "model": model,
                    "paired_vs_baseline": {
                        "metrics": {"mrr": {"stability": {"reading": "flat", "borderline": False}}}
                    },
                }
            ],
        },
    )


def write_probe(
    tmp_path: Path,
    *,
    model: str = MODEL,
    recommendation: str = "rank",
    fingerprint: dict | None = None,
) -> Path:
    return write_bundle(
        tmp_path,
        "context-position",
        "probe.json",
        {
            "model": model,
            "retrieval_fingerprint": dict(STORE) if fingerprint is None else fingerprint,
            "backend": "ollama",
            "k": 5,
            "positions": [{"position": "head", "n": 82, "mean_objective": 0.58, "ci": [0.5, 0.66]}],
            "recommendation": recommendation,
            "recommendation_note": "head 0.583 vs tail 0.561",
            "verdict": "flat",
        },
    )


def write_loop_policy(tmp_path: Path, *, model: str = MODEL, max_steps: int = 6) -> Path:
    return write_bundle(
        tmp_path,
        "agentic-loop-policy",
        "recommendation.json",
        {
            "model": model,
            "max_steps": max_steps,
            "malformed_call_policy": "answer",
            "repeated_call_policy": "allow",
            "repeat_feedback_variant": "current",
            "changes_shipped_defaults": False,
            "verdict": "flat",
            "reason": "no candidate has a positive paired completion delta",
            "paired_completion_vs_baseline": {
                "stability": {"reading": "flat", "borderline": False}
            },
        },
    )


def write_context_policy(
    tmp_path: Path, *, model: str = MODEL, policy: str = "observation_cap"
) -> Path:
    run_dir = tmp_path / "agentic-context" / RUN_DIR_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scores.jsonl").write_text(
        "\n".join(json.dumps({"success": 1.0}) for _ in range(24)) + "\n", encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-08-20T00:00:00+00:00",
                "config": {
                    "category": "agentic-context",
                    "model": model,
                    "backend": "ollama",
                    "policy": policy,
                    "paired_vs_full": {
                        "completion": {"stability": {"reading": "separated", "borderline": False}}
                    },
                },
                "metrics": {"objective_score": 0.875},
            }
        ),
        encoding="utf-8",
    )
    return run_dir / "manifest.json"
