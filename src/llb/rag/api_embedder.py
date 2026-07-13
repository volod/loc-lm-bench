"""Opt-in API embedder for the embedding bake-off (Cohere multilingual via litellm).

Embedding a corpus through a hosted API is FULL corpus egress, so this lane exists ONLY as
bake-off evidence (`llb compare-embeddings`); it is never usable as `RunConfig.embedding_model`
for a scored run (scored retrieval stays local, per the egress policy in
`docs/impl/current/scope-boundaries.md`). Cohere's asymmetric retrieval convention is expressed
through `input_type`: `search_query` for questions, `search_document` for corpus passages -- the
API equivalent of the local families' query/passage seam.

`litellm` is imported lazily (the `[prep]` extra) and the embed call is injectable, so the
input_type mapping, batching, L2 normalization, and cost/budget accounting are pure and
unit-tested without any network or key.
"""

from typing import Any, Callable

from llb.prep.frontier_telemetry import ProvenanceLog

# Cohere `input_type` values -- the API analogue of the local query/passage seam.
COHERE_QUERY_INPUT_TYPE = "search_query"
COHERE_PASSAGE_INPUT_TYPE = "search_document"

DEFAULT_API_EMBED_MODEL = "cohere/embed-multilingual-v3.0"
DEFAULT_API_BATCH_SIZE = 96  # Cohere caps a single embed request at 96 inputs.

# (texts, input_type) -> list of raw (un-normalized) embedding vectors. The default binds litellm;
# tests inject a deterministic fake so no key or network is touched.
ApiEmbedFn = Callable[[list[str], str], list[list[float]]]


class BudgetExceeded(RuntimeError):
    """Raised when cumulative API embedding cost would exceed the operator's `--max-usd` cap."""


def record_embed_cost(
    log: ProvenanceLog | None,
    model: str,
    prompt_tokens: int,
    cost_usd: float,
    max_usd: float | None,
) -> None:
    """Record one embed call's tokens/cost, then abort if the running total passes `--max-usd`.

    Pure (no litellm / numpy), so the budget arithmetic and cap enforcement are unit-tested in CI.
    """
    if log is None:
        return
    log.record(model, prompt_tokens, 0, cost_usd)
    total = log.summary()["total_cost_usd"]
    if max_usd is not None and total > max_usd:
        raise BudgetExceeded(
            f"API embedding budget exceeded: spent ${total:.4f} > --max-usd ${max_usd:.4f}"
        )


def litellm_embed(
    model: str,
    *,
    log: ProvenanceLog | None = None,
    max_usd: float | None = None,
) -> ApiEmbedFn:
    """Default embed callable via litellm. Records per-call tokens/cost into `log` and aborts
    (BudgetExceeded) as soon as accumulated cost would pass `max_usd` (None disables the cap)."""

    def embed(texts: list[str], input_type: str) -> list[list[float]]:
        from litellm import embedding

        resp = embedding(model=model, input=texts, input_type=input_type)
        try:
            from litellm import completion_cost

            cost = float(completion_cost(completion_response=resp))
        except Exception:  # cost unavailable for some providers -- record 0, keep going
            cost = 0.0
        usage = resp.get("usage", {}) or {}
        record_embed_cost(log, model, int(usage.get("prompt_tokens", 0)), cost, max_usd)
        return [list(row["embedding"]) for row in resp["data"]]

    return embed


class ApiEmbedder:
    """Duck-typed `Embedder` for the bake-off API row: same `encode_*` seam, L2-normalized.

    Exposes `model_name` so `RagStore.build` records the API model in the store meta; a store
    built this way can never be silently reused for a scored run because loading it back would try
    to instantiate a LOCAL SentenceTransformer under that id (and the store/embedder mismatch guard
    would refuse it besides)."""

    def __init__(
        self,
        model: str,
        embed_fn: ApiEmbedFn,
        *,
        batch_size: int = DEFAULT_API_BATCH_SIZE,
    ):
        self.model_name = model
        self._embed_fn = embed_fn
        self._batch_size = max(1, batch_size)

    def _encode(self, texts: list[str], input_type: str) -> Any:
        import numpy as np

        rows: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            rows.extend(self._embed_fn(texts[start : start + self._batch_size], input_type))
        arr = np.asarray(rows, dtype="float32")
        if arr.size == 0:
            return arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0.0, 1.0, norms)

    def encode_passages(self, texts: list[str]) -> Any:
        return self._encode(texts, COHERE_PASSAGE_INPUT_TYPE)

    def encode_queries(self, texts: list[str]) -> Any:
        return self._encode(texts, COHERE_QUERY_INPUT_TYPE)
