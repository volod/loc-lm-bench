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

from dataclasses import dataclass, field
import logging
from collections.abc import Sequence
from typing import Any, Callable

from llb.core.contracts.rag import RetrievalPair
from llb.rag.candidate_screen import SkippedCandidate
from llb.rag.embedding_bakeoff_models import (
    BakeoffItem,
    BakeoffReport,
    BuiltStore,
    CandidateResult,
    StoreBuilder,
)
from llb.rag.embedding_families import resolve_convention
from llb.rag.embedding_bakeoff_uncertainty import (
    DEFAULT_BARS,
    DEFAULT_BASELINE_MODEL,
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    MetricVectors,
    item_vectors,
    paired_rows,
)
from llb.rag.embedding_bakeoff_verdict import (
    decide_verdict,
)
from llb.rag.embedding_bakeoff_selection import adjust_bakeoff_selection
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
    convention = resolve_convention(model)
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


@dataclass(slots=True)
class _ScoredCandidates:
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

    def score(self, model: str, built: BuiltStore) -> None:
        """Retrieve once for this candidate, keep its row, vectors, and store, then free weights."""
        pairs = retrieve_pairs(built.store, self.items, self.k)
        self.rows.append(score_pairs(model, built, pairs, self.k))
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


def run_bakeoff(
    items: list[BakeoffItem],
    k: int,
    *,
    corpus_root: str,
    local_models: list[str],
    build_local: StoreBuilder,
    item_ids: Sequence[str] | None = None,
    skipped: Sequence[SkippedCandidate] = (),
    api_model: str | None = None,
    build_api: StoreBuilder | None = None,
    data_classification: str | None = None,
    consent: Callable[[], bool] = lambda: False,
    noise_floor: bool = False,
    noise_floor_replicates: int | None = None,
    baseline: str | None = DEFAULT_BASELINE_MODEL,
    bars: Sequence[str] = DEFAULT_BARS,
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

    `skipped` carries roster entries the caller screened out before any build
    (`llb.rag.embedding_bakeoff_roster`); they ride into the report so a run that ranks fewer
    candidates than the roster names states which are missing and why.
    """
    if item_ids is not None and len(item_ids) != len(items):
        raise ValueError("the embedder paired ledger needs one item id per scored item")
    scored = _ScoredCandidates(k=k, items=items)
    for model in local_models:
        _LOG.info("[compare-embeddings] building candidate store: %s", model)
        scored.score(model, build_local(model))
    if api_lane_enabled(api_model, data_classification, consent):
        assert api_model is not None and build_api is not None  # narrowed by the gate
        _LOG.info("[compare-embeddings] building API candidate (CORPUS EGRESS): %s", api_model)
        scored.score(api_model, build_api(api_model))

    report: BakeoffReport = {
        "k": k,
        "n": len(items),
        "corpus_root": corpus_root,
        "candidates": scored.rows,
        "best_recall": best_recall(scored.rows),
        "paired_items": paired_item_ledger(scored.vectors, len(items), item_ids),
    }
    if skipped:
        report["skipped"] = list(skipped)
    _attach_uncertainty(
        report,
        scored.vectors,
        baseline,
        bars=bars,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    if noise_floor:
        from llb.rag.noise_floor import DEFAULT_REPLICATES, measure_noise_floor

        report["noise_floor"] = measure_noise_floor(
            scored.stores,
            list(items),
            k,
            replicates=noise_floor_replicates or DEFAULT_REPLICATES,
        )
    return report


def _attach_uncertainty(
    report: BakeoffReport,
    vectors: dict[str, MetricVectors],
    baseline: str | None,
    *,
    bars: Sequence[str],
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
        "bars": list(bars),
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
    adjustment = adjust_bakeoff_selection(
        vectors,
        baseline,
        bars,
        resamples=resamples,
        seed=seed,
    )
    report["verdict"] = decide_verdict(
        paired,
        baseline,
        bars,
        confidence,
        adjustment=adjustment,
    )
