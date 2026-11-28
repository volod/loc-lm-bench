"""Score and read the paired span and PDF-page ranks from one retrieval pass per lane."""

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean

from llb.core.contracts.rag import ChunkRecord
from llb.rag.book_comparison.inputs import BookQuestion, pages_at_span
from llb.rag.cited_corpus import CitedDocument
from llb.rag.comparison.models import Retriever
from llb.rag.duplicates.collapse import occurrence_spans
from llb.rag.fusion_evidence.evidence_gate import minimum_discordant_pairs
from llb.rag.fusion_evidence.paired import paired_comparison, regresses, separates
from llb.rag.fusion_evidence.power import (
    DEFAULT_TARGET_POWER,
    evidence_floor_n,
    required_sample_size,
    resolvable_mde,
    sample_sd,
)
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_index_sets,
)
from llb.rag.retrieval import first_hit_rank


@dataclass(frozen=True)
class ReadingSettings:
    k: int
    minimum_detectable_delta: float
    confidence: float = DEFAULT_CONFIDENCE
    resamples: int = DEFAULT_RESAMPLES
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class Ranks:
    span: int | None
    page: int | None


def first_page_rank(
    chunks: list[ChunkRecord], pages: frozenset[int], documents: dict[str, CitedDocument]
) -> int | None:
    for rank, chunk in enumerate(chunks, 1):
        for place in occurrence_spans(chunk):
            doc = documents[place["doc_id"]]
            if not pages.isdisjoint(pages_at_span(doc, place["char_start"], place["char_end"])):
                return rank
    return None


def retrieve_ranks(
    store: Retriever,
    items: list[BookQuestion],
    documents: dict[str, CitedDocument],
    *,
    k: int,
    baseline: bool,
) -> list[Ranks]:
    """Use the same top-k for both readings and retain the ranks for replay."""
    ranks = []
    for item in items:
        chunks = store.retrieve(item.question, k)[:k]
        spans = item.baseline_spans if baseline else item.projection_spans
        ranks.append(
            Ranks(first_hit_rank(chunks, spans), first_page_rank(chunks, item.pages, documents))
        )
    return ranks


@dataclass(frozen=True)
class PairedRank:
    item: BookQuestion
    projection: Ranks
    baseline: Ranks


def _metric_vectors(ranks: Iterable[Ranks]) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {
        "span_recall_at_k": [],
        "span_mrr": [],
        "page_recall_at_k": [],
        "page_mrr": [],
    }
    for rank in ranks:
        vectors["span_recall_at_k"].append(float(rank.span is not None))
        vectors["span_mrr"].append(1.0 / rank.span if rank.span else 0.0)
        vectors["page_recall_at_k"].append(float(rank.page is not None))
        vectors["page_mrr"].append(1.0 / rank.page if rank.page else 0.0)
    return vectors


def _sensitivity(deltas: list[float], settings: ReadingSettings) -> dict[str, object]:
    confidence = settings.confidence
    discordance_n = evidence_floor_n(deltas, confidence)
    alpha = round(1.0 - confidence, 12)
    sd = sample_sd(deltas) if len(deltas) >= 2 else None
    variance_n = (
        required_sample_size(
            sd, settings.minimum_detectable_delta, alpha=alpha, target_power=DEFAULT_TARGET_POWER
        )
        if sd is not None
        else None
    )
    required_n = (
        max(variance_n, discordance_n)
        if (variance_n is not None and discordance_n is not None)
        else None
    )
    return {
        "variance_required_n": variance_n,
        "discordance_required_n": discordance_n,
        "required_n": required_n,
        "target_reached": required_n is not None and len(deltas) >= required_n,
        "resolvable_mde": (
            resolvable_mde(sd, len(deltas), alpha=alpha, target_power=DEFAULT_TARGET_POWER)
            if sd is not None
            else None
        ),
    }


def paired_metric_reading(
    candidate: list[float],
    baseline: list[float],
    draw: list[list[int]],
    settings: ReadingSettings,
) -> dict[str, object]:
    paired = paired_comparison(candidate, baseline, draw, settings.confidence)
    verdict = "inconclusive"
    if paired["delta"]["lo"] > 0 and separates(paired, settings.confidence):
        verdict = "improved"
    elif regresses(paired, settings.confidence):
        verdict = "regressed"
    return {
        "paired": paired,
        "verdict": verdict,
        "sensitivity": _sensitivity(
            [left - right for left, right in zip(candidate, baseline, strict=True)], settings
        ),
    }


def _read_slice(rows: list[PairedRank], settings: ReadingSettings) -> dict[str, object]:
    if not rows:
        return {"n": 0, "verdict": "inconclusive", "metrics": {}}
    draw = bootstrap_index_sets(len(rows), settings.resamples, settings.seed)
    projection = _metric_vectors(row.projection for row in rows)
    baseline = _metric_vectors(row.baseline for row in rows)
    metrics = {
        name: {
            "projection": mean(values),
            "baseline": mean(baseline[name]),
            **paired_metric_reading(values, baseline[name], draw, settings),
        }
        for name, values in projection.items()
    }
    return {
        "n": len(rows),
        "verdict": metrics["span_recall_at_k"]["verdict"],
        "metrics": metrics,
    }


def read_slices(rows: list[PairedRank], settings: ReadingSettings) -> dict[str, dict[str, object]]:
    """Read all accepted items and both baseline-text slices, with shared draws within each."""
    return {
        "all": _read_slice(rows, settings),
        "baseline_text_present": _read_slice(
            [row for row in rows if row.item.baseline_text_present], settings
        ),
        "baseline_text_missing": _read_slice(
            [row for row in rows if not row.item.baseline_text_present], settings
        ),
    }


def declared_settings(settings: ReadingSettings) -> dict[str, object]:
    return {
        "k": settings.k,
        "confidence": settings.confidence,
        "resamples": settings.resamples,
        "seed": settings.seed,
        "minimum_detectable_delta": settings.minimum_detectable_delta,
        "target_power": DEFAULT_TARGET_POWER,
        "minimum_discordant_pairs": minimum_discordant_pairs(settings.confidence),
        "primary_metric": "span_recall_at_k",
    }
