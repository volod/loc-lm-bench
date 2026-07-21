"""Retrieval validation with a bounded repair search when the baseline misses its gate."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.executor.cases import spans_as_dicts
from llb.goldset.schema import load_goldset
from llb.rag.retrieval import evaluate_retrieval
from llb.rag.store import RagStore


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    strategy: str
    chunk_size: int
    chunk_overlap: int
    mode: str

    def payload(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_mode": self.mode,
        }


BASELINE = RetrievalCandidate("recursive", 800, 120, "hybrid")
REPAIR_CANDIDATES = (
    RetrievalCandidate("recursive", 400, 60, "hybrid"),
    RetrievalCandidate("markdown", 800, 120, "hybrid"),
    RetrievalCandidate("recursive", 800, 120, "flat"),
)


def validate_and_repair_retrieval(
    corpus: Path,
    goldset: Path,
    stage_dir: Path,
    *,
    k: int,
    recall_gate: float,
    embedder: Any = None,
) -> dict[str, Any]:
    """Build the baseline, search alternatives only on failure, and persist the best store."""
    items = load_goldset(goldset)
    attempts: list[tuple[RetrievalCandidate, RagStore, dict[str, float]]] = []
    candidates = [BASELINE]
    baseline = _evaluate(corpus, items, BASELINE, k, embedder)
    attempts.append((BASELINE, *baseline))
    if baseline[1]["recall_at_k"] < recall_gate:
        candidates.extend(REPAIR_CANDIDATES)
        for candidate in REPAIR_CANDIDATES:
            attempts.append((candidate, *_evaluate(corpus, items, candidate, k, embedder)))
    selected, store, metrics = max(
        attempts,
        key=lambda row: (row[2]["recall_at_k"], row[2]["mrr"], -candidates.index(row[0])),
    )
    if metrics["recall_at_k"] < recall_gate:
        raise ValueError(
            f"retrieval repair exhausted {len(attempts)} configurations: "
            f"best recall@{k}={metrics['recall_at_k']:.3f} < {recall_gate:.3f}"
        )
    index_dir = stage_dir / "index"
    store.save(index_dir)
    return {
        "index_dir": str(index_dir),
        "k": k,
        "gate": recall_gate,
        "selected": selected.payload(),
        "metrics": metrics,
        "repaired": selected != BASELINE,
        "attempts": [
            {"config": candidate.payload(), "metrics": score}
            for candidate, _candidate_store, score in attempts
        ],
    }


def _evaluate(
    corpus: Path,
    items: list[Any],
    candidate: RetrievalCandidate,
    k: int,
    embedder: Any,
) -> tuple[RagStore, Any]:
    store = RagStore.build(
        corpus,
        candidate.strategy,
        candidate.chunk_size,
        candidate.chunk_overlap,
        mode=candidate.mode,
        embedder=embedder,
    )
    pairs = [(store.retrieve(item.question, k), spans_as_dicts(item)) for item in items]
    return store, evaluate_retrieval(pairs, k)
