"""Optional cross-encoder reranking between retrieval and generation (rerank-context-order).

The seam is a wrapper over ANY retrieval backend exposing the RAG-store contract
`.retrieve(question, k) -> list[ChunkRecord]` (flat / parent_child / hybrid FAISS stores and
the GraphRAG store alike): retrieve `rerank_candidates` from the wrapped store, score every
(question, chunk_text) pair with a cross-encoder, and keep the `top_k` best. Reranking is OFF
by default (`RunConfig.reranker is None`) and never mutates chunk text or offsets, so the
source-span retrieval metrics score the reranked ranking on unchanged rules.

The scorer is injectable (`RerankScorer`: (question, texts) -> scores), so candidate flow,
kept sets, and rank bookkeeping are fully unit-testable without a model; the real
`CrossEncoderReranker` lazily loads a sentence-transformers CrossEncoder (the `[rag]` extra,
default `BAAI/bge-reranker-v2-m3` -- multilingual, covers Ukrainian). Per-stage wall-clock
(retrieve vs rerank) is recorded on the wrapper after every call so run telemetry can weigh
the precision gain against the reranker's latency cost.
"""

import time
from typing import Any, Callable, Protocol, cast

from llb.core.config_validation import DEFAULT_RERANK_CANDIDATES
from llb.core.contracts.rag import ChunkRecord

# Pinned default cross-encoder (multilingual; scores Ukrainian question/passage pairs).
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"

# Per-call stage wall-clock seconds: {"retrieve_s": float, "rerank_s": float}.
StageLatency = dict[str, float]


class RerankScorer(Protocol):
    """One relevance score per candidate text for a question (higher is more relevant)."""

    def __call__(self, question: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Lazy sentence-transformers CrossEncoder behind the `RerankScorer` seam ([rag] extra)."""

    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str | None = None):
        self.model_name = model_name
        self._device = device
        self._model = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise SystemExit(
                    'ERROR: reranking needs the [rag] extra. Run: uv pip install -e ".[rag]"'
                ) from exc
            self._model = CrossEncoder(self.model_name, device=self._device)
        return self._model

    def __call__(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._load().predict([(question, text) for text in texts])
        return [float(s) for s in scores]


def rerank_chunks(
    question: str, candidates: list[ChunkRecord], keep_k: int, scorer: RerankScorer
) -> list[ChunkRecord]:
    """Score `candidates` against `question`, keep the `keep_k` best, and renumber ranks.

    Pure: each kept chunk is a copy carrying `rerank_score`, its original retrieval position as
    `pre_rerank_rank`, and a fresh contiguous 1-based `rank`. Ties keep the retrieval order
    (stable sort), so an all-equal scorer is an exact no-op on the ordering.
    """
    if keep_k < 1:
        raise ValueError("keep_k must be >= 1")
    scores = scorer(question, [str(c.get("text", "")) for c in candidates])
    if len(scores) != len(candidates):
        raise ValueError(f"reranker returned {len(scores)} scores for {len(candidates)} candidates")
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    kept: list[ChunkRecord] = []
    for new_rank, i in enumerate(order[:keep_k], 1):
        chunk = cast(ChunkRecord, dict(candidates[i]))
        chunk["pre_rerank_rank"] = int(candidates[i].get("rank", i + 1))
        chunk["rerank_score"] = float(scores[i])
        chunk["rank"] = new_rank
        kept.append(chunk)
    return kept


class RerankingRetriever:
    """Wrap any RAG-store-contract retriever with a cross-encoder rerank stage.

    `retrieve(question, k)` pulls `max(candidates, k)` from the wrapped store, reranks, and
    returns the top k. Unknown attributes delegate to the wrapped store (`embedder`, `meta`,
    ...), so the wrapper drops into every seam a bare store fits. `stage_latency` holds the
    last call's retrieve/rerank wall-clock; `last_candidates` holds the last call's pre-rerank
    pool (rank order) for pre/post-rerank metric comparisons.
    """

    def __init__(
        self,
        store: Any,
        scorer: RerankScorer,
        candidates: int = DEFAULT_RERANK_CANDIDATES,
        clock: Callable[[], float] = time.perf_counter,
    ):
        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        self.store = store
        self.scorer = scorer
        self.candidates = candidates
        self.stage_latency: StageLatency = {"retrieve_s": 0.0, "rerank_s": 0.0}
        self.total_latency: StageLatency = {"retrieve_s": 0.0, "rerank_s": 0.0}
        self.n_queries = 0
        self.last_candidates: list[ChunkRecord] = []
        self._clock = clock

    def retrieve(self, question: str, k: int, **kwargs: Any) -> list[ChunkRecord]:
        depth = max(self.candidates, k)
        started = self._clock()
        pool = self.store.retrieve(question, depth, **kwargs)
        retrieved_at = self._clock()
        kept = rerank_chunks(question, pool, k, self.scorer)
        self.stage_latency = {
            "retrieve_s": retrieved_at - started,
            "rerank_s": self._clock() - retrieved_at,
        }
        for stage, seconds in self.stage_latency.items():
            self.total_latency[stage] += seconds
        self.n_queries += 1
        self.last_candidates = pool
        return kept

    def mean_stage_latency(self) -> StageLatency:
        """Mean per-query retrieve/rerank wall-clock over every call so far (zeros when unused)."""
        if not self.n_queries:
            return {"retrieve_s": 0.0, "rerank_s": 0.0}
        return {stage: seconds / self.n_queries for stage, seconds in self.total_latency.items()}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)


def maybe_wrap_reranker(store: Any, config: Any) -> Any:
    """Wrap `store` per the RunConfig reranker knobs; identity when reranking is off."""
    if getattr(config, "reranker", None) is None:
        return store
    return RerankingRetriever(
        store, CrossEncoderReranker(config.reranker), candidates=config.rerank_candidates
    )
