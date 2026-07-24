"""Embedding bake-off: rank candidate embedders for Ukrainian RAG on ONE gold set.

"Which embedder for Ukrainian?" is an EVIDENCE question, not an assumption: a paraphrase/STS model
(`lang-uk/ukr-paraphrase-multilingual-mpnet-base`) may lose to a retrieval-tuned encoder
(E5 / BGE-M3) exactly because its objective differs, so the ranking must be measured. This builds
one store per candidate over the SAME corpus + chunking (each under its own family convention from
`src/llb/rag/embedding.py`) and scores recall@k / MRR by the model-independent source-span metric
(`evaluate_retrieval`), plus embed throughput, index size, and device.

The recommendation is NOT the point-estimate order: each candidate is also PAIRED against the
baseline embedder (`llb.rag.embedding_bakeoff_uncertainty`), and the run ends in an adopt-or-retain
verdict that a lead inside its own sampling interval cannot win.

For OPEN corpora an operator may additionally opt in one Cohere API row (`src/llb/rag/api_embedder.py`):
full corpus egress, so it is gated on explicit consent + `--max-usd` and refused for any non-open
corpus. The API row is bake-off EVIDENCE ONLY; scored retrieval stays local.

Pure + injectable: the store builder is a seam, so the scoring, ranking, report shaping, and the
consent gate are unit-tested with fake stores/embedders -- no GPU, no FAISS, no network.
"""

import logging
from typing import Any, Callable

from llb.core.contracts.rag import RetrievalPair
from llb.rag.embedding_bakeoff_models import (
    BakeoffItem,
    BakeoffReport,
    BuiltStore,
    CandidateResult,
    StoreBuilder,
)
from llb.rag.embedding_bakeoff_uncertainty import (
    DEFAULT_BASELINE_MODEL,
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    MetricVectors,
    decide_verdict,
    item_vectors,
    paired_rows,
)
from llb.rag.retrieval import evaluate_retrieval

_LOG = logging.getLogger(__name__)


def retrieve_pairs(store: Any, items: list[BakeoffItem], k: int) -> list[RetrievalPair]:
    """One top-k retrieval pass over the shared items; the row AND its per-item vectors read it."""
    return [(store.retrieve(question, k), spans) for question, spans in items]


def score_candidate(
    model: str, built: BuiltStore, items: list[BakeoffItem], k: int
) -> CandidateResult:
    """Score one built store's top-k retrieval over the shared items (pure; fake-store testable)."""
    return score_pairs(model, built, retrieve_pairs(built.store, items, k), k)


def score_pairs(
    model: str, built: BuiltStore, pairs: list[RetrievalPair], k: int
) -> CandidateResult:
    """Shape one candidate row from an already-retrieved pass (so the pass is never repeated)."""
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
    baseline: str | None = DEFAULT_BASELINE_MODEL,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> BakeoffReport:
    """Build + score each local candidate, then the gated API row, and rank by recall@k.

    `build_local` / `build_api` are the injectable store-builder seam (real FAISS builds in the CLI,
    fakes in tests). The API row is added only when `api_lane_enabled` clears the consent + open-data
    gate, so a declined or non-open run never calls `build_api`.

    Every candidate also keeps its per-item metric vectors, so the run ends with a PAIRED interval
    against `baseline` and an adopt-or-retain verdict: the point-estimate order alone cannot say
    whether a two-question lead survives a different draw of questions
    (`llb.rag.embedding_bakeoff_uncertainty`).

    With `noise_floor` the candidate stores are kept until the whole set is scored and their
    measurement floor is measured over the SAME items, so the recommendation is published beside
    the delta it has to clear rather than as a bare third decimal.
    """
    candidates: list[CandidateResult] = []
    stores: dict[str, Any] = {}
    vectors: dict[str, MetricVectors] = {}

    def score(model: str, built: BuiltStore) -> None:
        pairs = retrieve_pairs(built.store, items, k)
        candidates.append(score_pairs(model, built, pairs, k))
        vectors[model] = item_vectors(pairs, k)
        stores[model] = built.store

    for model in local_models:
        _LOG.info("[compare-embeddings] building candidate store: %s", model)
        score(model, build_local(model))

    if api_lane_enabled(api_model, data_classification, consent):
        assert api_model is not None and build_api is not None  # narrowed by the gate
        _LOG.info("[compare-embeddings] building API candidate (CORPUS EGRESS): %s", api_model)
        score(api_model, build_api(api_model))

    report: BakeoffReport = {
        "k": k,
        "n": len(items),
        "corpus_root": corpus_root,
        "candidates": candidates,
        "best_recall": best_recall(candidates),
    }
    _attach_uncertainty(
        report, vectors, baseline, resamples=resamples, confidence=confidence, seed=seed
    )
    if noise_floor:
        from llb.rag.noise_floor import DEFAULT_REPLICATES, measure_noise_floor

        report["noise_floor"] = measure_noise_floor(
            stores, list(items), k, replicates=noise_floor_replicates or DEFAULT_REPLICATES
        )
    return report


def _attach_uncertainty(
    report: BakeoffReport,
    vectors: dict[str, MetricVectors],
    baseline: str | None,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> None:
    """Hang the paired interval on each row and the adopt-or-retain verdict on the report.

    A baseline the run did not score leaves the rows bare and the verdict `undecided` rather than
    silently re-pointing the comparison at whichever candidate happened to rank first.
    """
    report["uncertainty"] = {
        "baseline": baseline,
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
    }
    paired = (
        paired_rows(vectors, baseline, resamples=resamples, confidence=confidence, seed=seed)
        if baseline is not None
        else {}
    )
    for row in report["candidates"]:
        if row["model"] in paired:
            row["paired_vs_baseline"] = paired[row["model"]]
    report["verdict"] = decide_verdict(paired, baseline)
