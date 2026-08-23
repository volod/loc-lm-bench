"""multi-hop-answer-quality -- the end-to-end answer comparison of two retrieval lanes.

Pure and file-driven: the comparison consumes canonical per-case rows, and the orchestration takes
an injected lane runner, so the whole vertical runs in the lightweight CI install (no FAISS, no
backend, no GPU). The CLI wiring layers real stores and `run-eval` on top.
"""

import json
from pathlib import Path

from llb.core.config import RunConfig
from llb.eval.answer_quality import compare_answer_quality
from llb.goldset.schema import GoldItem

VECTOR = "vector"

FUSED = "fused/global_community@0.10/d10"


def _row(item_id: str, objective: float, hit: float = 1.0) -> dict:
    return {
        "item_id": item_id,
        "split": "final",
        "status": "ok",
        "objective_score": objective,
        "token_f1": objective,
        "exact": 0.0,
        "contains": 0.0,
        "retrieval_hit": hit,
    }


def _lanes(vector: list[dict], fused: list[dict]) -> dict[str, list[dict]]:
    return {VECTOR: vector, FUSED: fused}


def _types(*item_ids: str) -> dict[str, str]:
    return {item_id: "multi-hop" for item_id in item_ids}


def _report(vector: list[float], fused: list[float], hits=None, resamples: int = 200):
    ids = [f"q{i}" for i in range(len(vector))]
    vector_hits, fused_hits = hits or ([1.0] * len(vector), [1.0] * len(fused))
    return compare_answer_quality(
        _lanes(
            [_row(i, s, h) for i, s, h in zip(ids, vector, vector_hits)],
            [_row(i, s, h) for i, s, h in zip(ids, fused, fused_hits)],
        ),
        _types(*ids),
        baseline=VECTOR,
        resamples=resamples,
    )


def _retrieval_record(item_id: str, covered_hops: int) -> str:
    """A retrieval sidecar row for a two-hop item whose context carries `covered_hops` of them."""
    gold = [
        {"doc_id": "d1", "char_start": 0, "char_end": 10, "text": "a"},
        {"doc_id": "d2", "char_start": 0, "char_end": 10, "text": "b"},
    ]
    return (
        json.dumps(
            {
                "item_id": item_id,
                "retrieved": [dict(span, rank=i) for i, span in enumerate(gold[:covered_hops], 1)],
                "gold_spans": gold,
            }
        )
        + "\n"
    )


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


def _write_bundle(goldset: Path, verified: bool = True) -> None:
    """A two-item gold set whose question types live in the needle sidecar beside it."""
    items = [_gold_item("q1", verified), _gold_item("q2", verified)]
    goldset.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items),
        encoding="utf-8",
    )
    (goldset.parent / "needle_items.jsonl").write_text(
        '{"id": "q1", "question_type": "multi-hop"}\n{"id": "q2", "question_type": "factoid"}\n',
        encoding="utf-8",
    )


def _bundle_lane(tmp_path: Path, *, covered: int = 2):
    """A fake lane runner that persists a FULL run bundle: scores, retrieval sidecar, manifest.

    `_recording_lane` writes only the score rows, which is all the orchestration reads back. A
    re-render reads the manifest (to check the bundle still describes its lane) and the retrieval
    sidecar (to recompute the coverage columns), so the bundles it is exercised over have to be the
    shape `run-eval` really persists.
    """

    def fake_lane(config: RunConfig, items: list[GoldItem], split: str) -> Path:
        run_dir = tmp_path / "run-eval" / f"{config.run_name}-{split}"
        run_dir.mkdir(parents=True, exist_ok=True)
        objective = 1.0 if config.retrieval_backend == "fused" else 0.0
        hops = covered if config.retrieval_backend == "fused" else 1
        (run_dir / "scores.jsonl").write_text(
            "".join(json.dumps(_row(item.id, objective)) + "\n" for item in items),
            encoding="utf-8",
        )
        (run_dir / "retrieval.jsonl").write_text(
            "".join(_retrieval_record(item.id, hops) for item in items), encoding="utf-8"
        )
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": config.run_name,
                    "run_name": config.run_name,
                    "split": split,
                    "config": config.fingerprint(),
                }
            ),
            encoding="utf-8",
        )
        return run_dir / "scores.jsonl"

    return fake_lane


def _recording_lane(tmp_path: Path, seen: list[tuple[str, str, tuple[str, ...]]]):
    """A fake lane runner that persists a `scores.jsonl` the fused lane always answers better."""

    def fake_lane(config: RunConfig, items: list[GoldItem], split: str) -> Path:
        seen.append((config.run_name, config.retrieval_backend, tuple(i.id for i in items)))
        run_dir = tmp_path / "run-eval" / f"{config.run_name}-{split}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scores = run_dir / "scores.jsonl"
        objective = 1.0 if config.retrieval_backend == "fused" else 0.0
        scores.write_text(
            "".join(json.dumps(_row(item.id, objective)) + "\n" for item in items),
            encoding="utf-8",
        )
        return scores

    return fake_lane
