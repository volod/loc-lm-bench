"""embedder-first-hit-rank-adoption-bar -- does an encoder's RANK gain reach the answer?

Pure and file-driven: the comparison consumes canonical per-case rows, and the orchestration takes
an injected lane runner, so the whole vertical runs in the lightweight CI install (no FAISS, no
backend, no GPU). Only the CLI layers real stores and `run-eval` on top.
"""

import json
from pathlib import Path

from llb.core.config import RunConfig
from llb.eval.embedder_adoption.models import (
    CellSpec,
    EmbedderLane,
)
from llb.eval.embedder_adoption.compare import compare_cells
from llb.goldset.schema import GoldItem

BASELINE = "intfloat/multilingual-e5-base"

CANDIDATE = "BAAI/bge-m3"


def _row(item_id: str, objective: float, rank: int | None = 1, hit: float = 1.0) -> dict:
    return {
        "item_id": item_id,
        "split": "final",
        "status": "ok",
        "objective_score": objective,
        "token_f1": objective,
        "exact": 0.0,
        "contains": 1.0 if objective > 0 else 0.0,
        "retrieval_hit": hit,
        "first_hit_rank": rank,
    }


def _sweep(cells: dict[str, tuple[list[dict], list[dict]]], resamples: int = 200):
    specs = {"k10": CellSpec(10, None), "k3": CellSpec(3, None), "k3+rerank": CellSpec(3, "x")}
    return compare_cells(
        [
            (specs[label], {BASELINE: base, CANDIDATE: cand})
            for label, (base, cand) in cells.items()
        ],
        {},
        baseline=BASELINE,
        candidate=CANDIDATE,
        resamples=resamples,
    )


def _ids(n: int) -> list[str]:
    return [f"q{i}" for i in range(n)]


def _gold_item(item_id: str, verified: bool = True) -> GoldItem:
    return GoldItem(
        id=item_id,
        lang="uk",
        question=f"питання {item_id}",
        reference_answer="відповідь",
        source_doc_id="doc",
        source_spans=[{"doc_id": "doc", "char_start": 0, "char_end": 9, "text": "відповідь"}],
        provenance="human-authored",
        verified=verified,
        split="final",
    )


def _write_goldset(goldset: Path, verified: bool = True) -> None:
    goldset.write_text(
        "".join(
            _gold_item(item_id, verified).model_dump_json(exclude_none=True) + "\n"
            for item_id in ("q1", "q2")
        ),
        encoding="utf-8",
    )


def _recording_lane(tmp_path: Path, seen: list[tuple[str, str, int, str | None]]):
    """A fake lane runner whose candidate encoder ranks first and answers better at a small k."""

    def fake_lane(config: RunConfig, items: list[GoldItem], split: str) -> Path:
        seen.append((config.embedding_model, str(config.data_dir), config.top_k, config.reranker))
        run_dir = tmp_path / "run-eval" / f"{config.run_name}-{split}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scores = run_dir / "scores.jsonl"
        is_candidate = config.embedding_model == CANDIDATE
        rank = 1 if is_candidate else 3
        objective = 1.0 if is_candidate or config.top_k >= rank else 0.0
        scores.write_text(
            "".join(json.dumps(_row(item.id, objective, rank)) + "\n" for item in items),
            encoding="utf-8",
        )
        return scores

    return fake_lane


def _lanes(tmp_path: Path) -> list[EmbedderLane]:
    return [
        EmbedderLane(BASELINE, tmp_path / "e5"),
        EmbedderLane(CANDIDATE, tmp_path / "bge"),
    ]
