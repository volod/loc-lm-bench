"""Score one candidate embedder: the retrieval pass, the report row, and the ranking.

Everything here is pure over an already-built store, so the whole scoring path is exercised with
fake stores -- no GPU, no FAISS, no network. `llb.rag.embedding_bakeoff.run` drives these over a
roster; nothing in this module knows a roster exists.
"""

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any

from llb.core.contracts.rag import RetrievalPair
from llb.rag.encoders.card_parity import CardParityResult
from llb.rag.embedding_bakeoff.models import (
    BakeoffItem,
    BuiltStore,
    CandidateResult,
)
from llb.rag.embedding_bakeoff.uncertainty import MetricVectors, item_vectors
from llb.rag.encoders.tuned import resolved_convention
from llb.rag.encoders.precision import published_dtype
from llb.rag.retrieval import evaluate_retrieval


def retrieve_pairs(store: Any, items: list[BakeoffItem], k: int) -> list[RetrievalPair]:
    """One top-k retrieval pass over the shared items; the row AND its per-item vectors read it."""
    return [(store.retrieve(question, k), spans) for question, spans in items]


def score_candidate(
    model: str, built: BuiltStore, items: list[BakeoffItem], k: int
) -> CandidateResult:
    """Score one built store's top-k retrieval over the shared items (pure; fake-store testable)."""
    return score_pairs(model, built, retrieve_pairs(built.store, items, k), k)


def score_pairs(
    model: str,
    built: BuiltStore,
    pairs: list[RetrievalPair],
    k: int,
    parity: CardParityResult | None = None,
) -> CandidateResult:
    """Shape one candidate row from an already-retrieved pass (so the pass is never repeated)."""
    metrics = evaluate_retrieval(pairs, k)
    meta = getattr(built.store, "meta", {}) or {}
    convention = resolved_convention(model)
    result: CandidateResult = {
        "model": model,
        "kind": built.kind,
        "family": convention.family,
        "recall_at_k": metrics["recall_at_k"],
        "mrr": metrics["mrr"],
        "n": metrics["n"],
        "k": metrics["k"],
        "dim": int(meta.get("dim", 0)),
        "n_indexed": int(meta.get("n_indexed", 0)),
        "embed_seconds": round(built.embed_seconds, 3),
        "index_bytes": int(built.index_bytes),
    }
    if parity is not None:
        result["card_parity"] = parity
    if built.dtype is not None:
        result["dtype"] = built.dtype
    declared = published_dtype(model)
    if declared is not None:
        result["published_dtype"] = declared
    if convention.trust_remote_code:
        result["trust_remote_code"] = True
    if built.device is not None:
        result["device"] = built.device
    if built.cost_usd is not None:
        result["cost_usd"] = round(built.cost_usd, 6)
    if built.throughput_profile is not None:
        result["throughput_profile"] = built.throughput_profile
    return result


def best_recall(candidates: list[CandidateResult]) -> str | None:
    """Model with the highest recall@k; ties break by MRR, then faster embed, then model id."""
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda c: (-c["recall_at_k"], -c["mrr"], c["embed_seconds"], c["model"]),
    )
    return best["model"]


@dataclass(slots=True)
class ScoredCandidates:
    """What one bake-off accumulates per candidate: its row, its per-item vectors, its store.

    A record rather than three dicts closed over by a nested `score`: the noise floor is measured
    over the SAME stores after every candidate is scored, so the three have to stay aligned, and a
    caller reading this can see what "scored" means without reading a closure.
    """

    k: int
    items: list[BakeoffItem]
    rows: list[CandidateResult] = field(default_factory=list)
    stores: dict[str, Any] = field(default_factory=dict)
    vectors: dict[str, MetricVectors] = field(default_factory=dict)

    def score(self, model: str, built: BuiltStore, parity: CardParityResult | None = None) -> None:
        """Retrieve once for this candidate, keep its row, vectors, and store, then free weights."""
        pairs = retrieve_pairs(built.store, self.items, self.k)
        self.rows.append(score_pairs(model, built, pairs, self.k, parity))
        self.vectors[model] = item_vectors(pairs, self.k)
        self.stores[model] = built.store
        # Free encoder weights after the retrieval pass; noise-floor / later reads reload lazily.
        release = getattr(getattr(built.store, "embedder", None), "release", None)
        if callable(release):
            release()


def paired_item_ledger(
    vectors: dict[str, MetricVectors], count: int, item_ids: Sequence[str] | None
) -> list[dict[str, object]]:
    """The per-item ledger a paired reading is recomputable from (shared with the reranker lane)."""
    return [
        {
            "item_id": item_ids[index] if item_ids is not None else str(index),
            "models": {
                model: {metric: values[metric][index] for metric in values}
                for model, values in vectors.items()
            },
        }
        for index in range(count)
    ]
