"""Embedding bake-off: rank candidate embedders for Ukrainian RAG on ONE gold set.

"Which embedder for Ukrainian?" is an EVIDENCE question, not an assumption: a paraphrase/STS model
(`lang-uk/ukr-paraphrase-multilingual-mpnet-base`) may lose to a retrieval-tuned encoder
(E5 / BGE-M3) exactly because its objective differs, so the ranking must be measured. This builds
one store per candidate over the SAME corpus + chunking (each under its own family convention from
`src/llb/rag/embedding.py`) and scores recall@k / MRR by the model-independent source-span metric
(`evaluate_retrieval`), plus embed throughput, index size, and device -- ending in a written
recommendation the operator applies via `RunConfig.embedding_model`.

For OPEN corpora an operator may additionally opt in one Cohere API row (`src/llb/rag/api_embedder.py`):
full corpus egress, so it is gated on explicit consent + `--max-usd` and refused for any non-open
corpus. The API row is bake-off EVIDENCE ONLY; scored retrieval stays local.

Pure + injectable: the store builder is a seam, so the scoring, ranking, report shaping, and the
consent gate are unit-tested with fake stores/embedders -- no GPU, no FAISS, no network.
"""

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.rag import SourceSpanRecord
from llb.rag.retrieval import evaluate_retrieval

if TYPE_CHECKING:  # imported lazily: the floor is opt-in and costs an extra retrieval pass
    from llb.rag.noise_floor import NoiseFloorReport

_LOG = logging.getLogger(__name__)

# (question, gold source spans) -- the per-item input shared across every candidate.
BakeoffItem = tuple[str, list[SourceSpanRecord]]

KIND_LOCAL = "local"
KIND_API = "api"

# Default LOCAL candidates for Ukrainian RAG. The current default first, then two retrieval-tuned
# alternatives, then the paraphrase/STS model whose objective differs (why the ranking is measured).
DEFAULT_LOCAL_CANDIDATES = [
    "intfloat/multilingual-e5-base",  # current RunConfig default
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "lang-uk/ukr-paraphrase-multilingual-mpnet-base",
]

_BYTES_PER_MB = 1024 * 1024


def slugify_model(model: str) -> str:
    """Filesystem-safe slug for a model id, for the per-candidate store directory."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._") or "model"


@dataclass
class BuiltStore:
    """One built candidate store plus the build measurements the report ranks on.

    `store` exposes `.retrieve(question, k) -> list[ChunkRecord]` and `.meta` (dim / n_indexed /
    embedding_model). `cost_usd` is set only for the API row.
    """

    store: Any
    embed_seconds: float
    index_bytes: int
    kind: str = KIND_LOCAL
    device: str | None = None
    cost_usd: float | None = None


# embedding_model -> BuiltStore. The CLI binds the heavy real builder; tests inject a fake.
StoreBuilder = Callable[[str], BuiltStore]


class CandidateResult(TypedDict):
    """One embedder's row: retrieval quality plus throughput / size / device fit."""

    model: str
    kind: str
    recall_at_k: float
    mrr: float
    n: int
    k: int
    dim: int
    n_indexed: int
    embed_seconds: float
    index_bytes: int
    device: NotRequired[str]
    cost_usd: NotRequired[float]


class BakeoffReport(TypedDict):
    k: int
    n: int
    corpus_root: str
    candidates: list[CandidateResult]
    best_recall: str | None
    # Measurement floor across the candidate stores, present only when it was asked for
    # (`compare-embeddings --noise-floor`). This lane is exactly where a sub-item recall delta
    # gets read as a recommendation, so the floor states whether the winner is separated from
    # the runner-up at all. See `llb.rag.noise_floor`.
    noise_floor: NotRequired["NoiseFloorReport"]


def score_candidate(
    model: str, built: BuiltStore, items: list[BakeoffItem], k: int
) -> CandidateResult:
    """Score one built store's top-k retrieval over the shared items (pure; fake-store testable)."""
    pairs = [(built.store.retrieve(question, k), spans) for question, spans in items]
    metrics = evaluate_retrieval(pairs, k)
    meta = getattr(built.store, "meta", {}) or {}
    result: CandidateResult = {
        "model": model,
        "kind": built.kind,
        "recall_at_k": metrics["recall_at_k"],
        "mrr": metrics["mrr"],
        "n": metrics["n"],
        "k": metrics["k"],
        "dim": int(meta.get("dim", 0)),
        "n_indexed": int(meta.get("n_indexed", 0)),
        "embed_seconds": round(built.embed_seconds, 3),
        "index_bytes": int(built.index_bytes),
    }
    if built.device is not None:
        result["device"] = built.device
    if built.cost_usd is not None:
        result["cost_usd"] = round(built.cost_usd, 6)
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


def api_lane_enabled(
    api_model: str | None,
    data_classification: str | None,
    consent: Callable[[], bool],
) -> bool:
    """Decide whether the API row runs. Refuse a non-open corpus outright; skip on declined consent.

    A truthy `api_model` over a corpus that is not explicitly `open` is a hard refusal (corpus
    egress policy). Over an open corpus the operator's `consent()` must return True; a decline
    skips the row (the local bake-off still reports) and never touches the network.
    """
    if not api_model:
        return False
    if data_classification != "open":
        raise SystemExit(
            "[compare-embeddings] --api-model embeds the whole corpus through a hosted API "
            "(full egress); it is refused unless --data-classification open is set explicitly."
        )
    if not consent():
        _LOG.warning(
            "[compare-embeddings] corpus egress declined; skipping the API row (%s)", api_model
        )
        return False
    return True


def run_bakeoff(
    items: list[BakeoffItem],
    k: int,
    *,
    corpus_root: str,
    local_models: list[str],
    build_local: StoreBuilder,
    api_model: str | None = None,
    build_api: StoreBuilder | None = None,
    data_classification: str | None = None,
    consent: Callable[[], bool] = lambda: False,
    noise_floor: bool = False,
    noise_floor_replicates: int | None = None,
) -> BakeoffReport:
    """Build + score each local candidate, then the gated API row, and rank by recall@k.

    `build_local` / `build_api` are the injectable store-builder seam (real FAISS builds in the CLI,
    fakes in tests). The API row is added only when `api_lane_enabled` clears the consent + open-data
    gate, so a declined or non-open run never calls `build_api`.

    With `noise_floor` the candidate stores are kept until the whole set is scored and their
    measurement floor is measured over the SAME items, so the recommendation is published beside
    the delta it has to clear rather than as a bare third decimal.
    """
    candidates: list[CandidateResult] = []
    stores: dict[str, Any] = {}
    for model in local_models:
        _LOG.info("[compare-embeddings] building candidate store: %s", model)
        built = build_local(model)
        candidates.append(score_candidate(model, built, items, k))
        stores[model] = built.store

    if api_lane_enabled(api_model, data_classification, consent):
        assert api_model is not None and build_api is not None  # narrowed by the gate
        _LOG.info("[compare-embeddings] building API candidate (CORPUS EGRESS): %s", api_model)
        built = build_api(api_model)
        candidates.append(score_candidate(api_model, built, items, k))
        stores[api_model] = built.store

    report: BakeoffReport = {
        "k": k,
        "n": len(items),
        "corpus_root": corpus_root,
        "candidates": candidates,
        "best_recall": best_recall(candidates),
    }
    if noise_floor:
        from llb.rag.noise_floor import DEFAULT_REPLICATES, measure_noise_floor

        report["noise_floor"] = measure_noise_floor(
            stores, list(items), k, replicates=noise_floor_replicates or DEFAULT_REPLICATES
        )
    return report
