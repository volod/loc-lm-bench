"""Embedding bake-off: rank candidate embedders for Ukrainian RAG on ONE gold set.

"Which embedder for Ukrainian?" is an EVIDENCE question, not an assumption: a paraphrase/STS model
(`lang-uk/ukr-paraphrase-multilingual-mpnet-base`) may lose to a retrieval-tuned encoder
(E5 / BGE-M3) exactly because its objective differs, so the ranking must be measured. This builds
one store per candidate over the SAME corpus + chunking (each under its own family convention from
`src/llb/rag/encoders/embedder.py`) and scores recall@k / MRR by the model-independent source-span metric
(`evaluate_retrieval`), plus embed throughput, index size, and device.

The recommendation is NOT the point-estimate order: each candidate is also PAIRED against the
baseline embedder (`llb.rag.embedding_bakeoff.uncertainty`), and the run ends in an adopt-or-retain
verdict that a lead inside its own sampling interval cannot win.

For OPEN corpora an operator may additionally opt in one Cohere API row (`src/llb/rag/encoders/api.py`):
full corpus egress, so it is gated on explicit consent + `--max-usd` and refused for any non-open
corpus. The API row is bake-off EVIDENCE ONLY; scored retrieval stays local.

Pure + injectable: the store builder is a seam, so the scoring, ranking, report shaping, and the
consent gate are unit-tested with fake stores/embedders -- no GPU, no FAISS, no network.
"""

import logging
from collections.abc import Sequence
from typing import Callable

from llb.rag.encoders.candidate_screen import SkippedCandidate
from llb.rag.encoders.card_parity import (
    CardParityResult,
    blocks_scoring,
    parity_skip_row,
    unpublished_result,
)
from llb.rag.embedding_bakeoff.models import (
    BakeoffItem,
    BakeoffReport,
    StoreBuilder,
)
from llb.rag.embedding_bakeoff.scoring import (
    ScoredCandidates,
    best_recall,
    paired_item_ledger,
)
from llb.rag.encoders.tuned import resolved_convention
from llb.rag.embedding_bakeoff.uncertainty import (
    DEFAULT_BARS,
    DEFAULT_BASELINE_MODEL,
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    MetricVectors,
    paired_rows,
)
from llb.rag.embedding_bakeoff.verdict import (
    decide_verdict,
)
from llb.rag.embedding_bakeoff.selection import adjust_bakeoff_selection

_LOG = logging.getLogger(__name__)


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
    card_parity: Callable[[str], CardParityResult] | None = None,
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
    (`llb.rag.embedding_bakeoff.uncertainty`).

    With `noise_floor` the candidate stores are kept until the whole set is scored and their
    measurement floor is measured over the SAME items, so the recommendation is published beside
    the delta it has to clear rather than as a bare third decimal.

    `skipped` carries roster entries the caller screened out before any build
    (`llb.rag.embedding_bakeoff.roster`); they ride into the report so a run that ranks fewer
    candidates than the roster names states which are missing and why.

    `card_parity` is the gate a candidate has to clear before a store is built for it: loading is
    not evidence that a model can be ranked, reproducing its own card is
    (`llb.rag.encoders.cards`). A candidate that runs and does not reproduce its card joins
    `skipped` with the diagnosis instead of contributing a number nobody can read. Left unbound,
    every row records `no_reference_declared` -- which is what the fake-store tests want, and what
    an operator must be able to tell apart from "checked and reproduced".
    """
    if item_ids is not None and len(item_ids) != len(items):
        raise ValueError("the embedder paired ledger needs one item id per scored item")
    scored = ScoredCandidates(k=k, items=items)
    declined = list(skipped)
    for model in local_models:
        parity = card_parity(model) if card_parity is not None else unpublished_result(model)
        if blocks_scoring(parity):
            _LOG.warning("[compare-embeddings] %s failed card parity: %s", model, parity["detail"])
            declined.append(parity_skip_row(parity, resolved_convention(model).family))
            continue
        _LOG.info("[compare-embeddings] building candidate store: %s", model)
        scored.score(model, build_local(model), parity)
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
    if declined:
        report["skipped"] = declined
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
        from llb.rag.noise_floor.measure import DEFAULT_REPLICATES, measure_noise_floor

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
