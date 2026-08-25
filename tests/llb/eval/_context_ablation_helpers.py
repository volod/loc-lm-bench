"""rag-vs-long-context-ablation -- the context lanes over one identical item set.

Pure and file-driven: the comparison consumes canonical per-case rows, the context sources are
plain state->state closures, and the orchestration takes an injected lane runner, so the whole
vertical runs in the lightweight CI install (no FAISS, no backend, no GPU). The CLI wiring layers
real stores and `run-eval` on top.
"""

import json
from pathlib import Path

from llb.core.config import RunConfig
from llb.eval.context_ablation.models import (
    LANE_CLOSED_BOOK,
    LANE_LONG_CONTEXT,
    LANE_RAG,
    LANE_RETRIEVED_DOCUMENT,
)
from llb.goldset.schema import GoldItem

FITS = "docs fit"

ALWAYS_FITS = lambda chars: True  # noqa: E731 -- a one-line test double reads better inline

NEVER_FITS = lambda chars: False  # noqa: E731


def _row(item_id: str, objective: float, hit: float = 1.0, **extra) -> dict:
    return {
        "item_id": item_id,
        "split": "final",
        "status": "ok",
        "objective_score": objective,
        "token_f1": objective,
        "exact": 0.0,
        "contains": 0.0,
        "retrieval_hit": hit,
        **extra,
    }


def _lanes(
    closed: list[dict],
    rag: list[dict],
    long_context: list[dict] | None = None,
    retrieved_document: list[dict] | None = None,
) -> dict:
    lanes = {LANE_CLOSED_BOOK: closed, LANE_RAG: rag}
    if retrieved_document is not None:
        lanes[LANE_RETRIEVED_DOCUMENT] = retrieved_document
    if long_context is not None:
        lanes[LANE_LONG_CONTEXT] = long_context
    return lanes


def _types(*item_ids: str) -> dict[str, str]:
    return {item_id: "factoid" for item_id in item_ids}


def _derived(report, label):
    return next(entry for entry in report["derived"] if entry["label"] == label)


def _gold_item(item_id: str, verified: bool = True) -> GoldItem:
    return GoldItem(
        id=item_id,
        lang="uk",
        question=f"питання {item_id}",
        reference_answer="відповідь",
        source_doc_id="doc.txt",
        source_spans=[{"doc_id": "doc.txt", "char_start": 0, "char_end": 9, "text": "відповідь"}],
        provenance="human-authored",
        verified=verified,
        split="final",
    )


def _write_bundle(goldset: Path, verified: bool = True) -> None:
    items = [_gold_item("q1", verified), _gold_item("q2", verified)]
    goldset.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items), encoding="utf-8"
    )
    (goldset.parent / "needle_items.jsonl").write_text(
        '{"id": "q1", "question_type": "multi-hop"}\n{"id": "q2", "question_type": "factoid"}\n',
        encoding="utf-8",
    )


def _repeating_lane(
    tmp_path: Path,
    objectives: dict[str, list[float]],
    seen: list[tuple[str, str, tuple[str, ...]]] | None = None,
):
    """A fake lane runner that answers DIFFERENTLY on each repeat of the same lane.

    `objectives[lane]` is that lane's per-repeat objective, cycled when the run asks for more
    repeats than it lists -- which is how a lane that reproduces exactly is written as one value.
    Each repeat writes its own bundle directory, exactly as `run-eval` does.
    """
    passes: dict[str, int] = {}

    def fake_lane(config: RunConfig, items: list[GoldItem], split: str) -> Path:
        lane = config.context_strategy
        if seen is not None:
            seen.append((config.run_name, lane, tuple(i.id for i in items)))
        values = objectives[lane]
        objective = values[passes.get(lane, 0) % len(values)]
        passes[lane] = passes.get(lane, 0) + 1
        run_dir = tmp_path / "run-eval" / f"{config.run_name}-{split}-{passes[lane]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scores = run_dir / "scores.jsonl"
        scores.write_text(
            "".join(
                json.dumps(_row(item.id, objective, answer_preview=f"{lane}-{objective}")) + "\n"
                for item in items
            ),
            encoding="utf-8",
        )
        return scores

    return fake_lane


def _recording_lane(tmp_path: Path, seen: list[tuple[str, str, tuple[str, ...]]]):
    """A fake lane runner whose objective rises with the amount of context each lane laid in."""
    objective = {
        LANE_CLOSED_BOOK: 0.0,
        LANE_RAG: 0.5,
        LANE_RETRIEVED_DOCUMENT: 0.75,
        LANE_LONG_CONTEXT: 1.0,
    }

    def fake_lane(config: RunConfig, items: list[GoldItem], split: str) -> Path:
        seen.append((config.run_name, config.context_strategy, tuple(i.id for i in items)))
        run_dir = tmp_path / "run-eval" / f"{config.run_name}-{split}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scores = run_dir / "scores.jsonl"
        scores.write_text(
            "".join(
                json.dumps(_row(item.id, objective[config.context_strategy])) + "\n"
                for item in items
            ),
            encoding="utf-8",
        )
        return scores

    return fake_lane
