"""Is the null component of the observed cosine distribution identifiable at all?

Every calibrated-threshold proposal so far assumes the observed same-corpus similarity population
can be decomposed into an unrelated component and a related component, and that the unrelated one
can be located precisely enough to name a tail. This lane tests that assumption directly: anchor the
related mass with NON-cosine evidence (lexical n-gram overlap, plus the planted relation closure),
anchor the null mass with the control bank, then enumerate every mixture that reproduces the
observed distribution within the resolution the corpus's independent units can actually support.

If materially different mixtures stay observationally equivalent, cosine-only calibration is not
identifiable, and no amount of extra control construction repairs it.
"""

from dataclasses import dataclass

import numpy as np

from llb.conflicts.constants import MAX_SHINGLE_DOC_FREQUENCY
from llb.conflicts.tiers.lexical import candidate_pairs, containment, jaccard, shingles
from llb.conflicts.null_research.geometry import CorpusGeometry
from llb.core.contracts.common import JsonObject

CHUNK_SHINGLE_DOC_FREQUENCY = 0.02
RELATED_JACCARD_THRESHOLD = 0.5
RELATED_CONTAINMENT_THRESHOLD = 0.8
KS_GRID_POINTS = 256
SHIFT_GRID = (-0.4, 0.6, 51)
MASS_GRID = (1e-6, 0.3, 24)
KS_CRITICAL_95 = 1.358
MAX_IDENTIFIED_ROW_RATIO = 2.0


@dataclass(frozen=True)
class RelatedAnchors:
    """Chunk pairs a non-cosine tier proves related, their store ordinals, and their cosines."""

    pairs: list[tuple[int, int]]
    scores: list[float]
    evidence: JsonObject


def lexical_related_anchors(corpus: CorpusGeometry) -> RelatedAnchors:
    """Cross-document chunk pairs whose word 5-gram overlap already proves a relation."""
    texts = [corpus.chunks[index]["text"] for index in corpus.allowed]
    chunk_shingles = [shingles(text) for text in texts]
    pairs: list[tuple[int, int]] = []
    scores: list[float] = []
    for left, right in sorted(candidate_pairs(chunk_shingles, CHUNK_SHINGLE_DOC_FREQUENCY)):
        left_ordinal, right_ordinal = corpus.allowed[left], corpus.allowed[right]
        if corpus.chunks[left_ordinal]["doc_id"] == corpus.chunks[right_ordinal]["doc_id"]:
            continue
        a, b = chunk_shingles[left], chunk_shingles[right]
        inner, outer = (a, b) if len(a) <= len(b) else (b, a)
        if (
            jaccard(a, b) >= RELATED_JACCARD_THRESHOLD
            or containment(inner, outer) >= RELATED_CONTAINMENT_THRESHOLD
        ):
            pairs.append((left_ordinal, right_ordinal))
            scores.append(corpus.vectors.similarity(left_ordinal, right_ordinal))
    return RelatedAnchors(
        pairs=pairs,
        scores=sorted(scores),
        evidence={
            "anchor": "word-5-gram jaccard/containment over comparable chunks",
            "jaccard_threshold": RELATED_JACCARD_THRESHOLD,
            "containment_threshold": RELATED_CONTAINMENT_THRESHOLD,
            "shingle_doc_frequency_cap": CHUNK_SHINGLE_DOC_FREQUENCY,
            "corpus_shingle_doc_frequency_cap": MAX_SHINGLE_DOC_FREQUENCY,
            "anchor_pairs": len(scores),
        },
    )


@dataclass(frozen=True)
class _WeightedEcdf:
    """A weighted empirical distribution sorted once and then queried by shifted location."""

    values: np.ndarray
    cumulative: np.ndarray

    @classmethod
    def build(cls, values: np.ndarray, weights: np.ndarray) -> "_WeightedEcdf":
        order = np.argsort(values, kind="stable")
        cumulative = np.concatenate(([0.0], np.cumsum(weights[order])))
        return cls(values=values[order], cumulative=cumulative / cumulative[-1])

    def cdf(self, points: np.ndarray) -> np.ndarray:
        return self.cumulative[np.searchsorted(self.values, points, side="right")]

    def quantile(self, probability: float) -> float:
        position = int(np.searchsorted(self.cumulative[1:], probability, side="left"))
        return float(self.values[min(position, len(self.values) - 1)])

    def tail_share(self, threshold: float) -> float:
        return float(1.0 - self.cdf(np.asarray([threshold]))[0])


def _accepted_payload(
    shift: float,
    mass: float,
    distance: float,
    null: _WeightedEcdf,
    observed: np.ndarray,
    *,
    null_quantile: float,
    budget_threshold: float,
) -> JsonObject:
    threshold = null_quantile + shift
    return {
        "null_shift": round(shift, 6),
        "related_mass": float(f"{mass:.6g}"),
        "ks_distance": round(distance, 9),
        "implied_threshold": round(threshold, 6),
        "implied_selected_rows": int((observed >= threshold).sum()),
        "implied_fpr_at_budget_threshold": float(
            f"{null.tail_share(budget_threshold - shift):.6g}"
        ),
    }


def _fit_grid(
    observed_grid: np.ndarray,
    observed_cdf: np.ndarray,
    null: _WeightedEcdf,
    related: np.ndarray,
) -> list[tuple[float, float, float]]:
    """Shifting the null by `d` moves its CDF, so evaluate the fixed null at `grid - d`."""
    shifts = np.linspace(*SHIFT_GRID)
    masses = (
        np.concatenate(([0.0], np.geomspace(MASS_GRID[0], MASS_GRID[1], int(MASS_GRID[2]))))
        if len(related)
        else np.zeros(1)
    )
    related_cdf = (
        _WeightedEcdf.build(related, np.ones(len(related))).cdf(observed_grid)
        if len(related)
        else np.ones_like(observed_grid)
    )
    fits: list[tuple[float, float, float]] = []
    for shift in shifts.tolist():
        null_cdf = null.cdf(observed_grid - shift)
        for mass in masses.tolist():
            mixed = (1.0 - mass) * null_cdf + mass * related_cdf
            fits.append((shift, mass, float(np.abs(mixed - observed_cdf).max())))
    return fits


def mixture_identifiability(
    corpus: CorpusGeometry,
    null_scores: np.ndarray,
    null_weights: np.ndarray,
    anchors: RelatedAnchors,
    *,
    operating_alpha: float,
    candidate_cap: int,
    effective_units: int,
) -> JsonObject:
    """Enumerate the mixtures a corpus's independent units cannot tell apart."""
    observed = np.asarray(corpus.observed_similarities, dtype="float64")
    grid = np.quantile(observed, np.linspace(0.0, 1.0, KS_GRID_POINTS))
    observed_cdf = np.searchsorted(observed, grid, side="right") / len(observed)
    related = np.asarray(anchors.scores, dtype="float64")
    budget_threshold = float(observed[-min(max(candidate_cap, 1), len(observed))])
    null = _WeightedEcdf.build(null_scores, null_weights)
    null_quantile = null.quantile(1.0 - operating_alpha)
    fits = _fit_grid(grid, observed_cdf, null, related)
    best = min(distance for _, _, distance in fits)
    critical = KS_CRITICAL_95 / max(1.0, float(effective_units)) ** 0.5
    limit = max(best, critical)
    accepted = [
        _accepted_payload(
            shift,
            mass,
            distance,
            null,
            observed,
            null_quantile=null_quantile,
            budget_threshold=budget_threshold,
        )
        for shift, mass, distance in fits
        if distance <= limit
    ]
    rows = [int(payload["implied_selected_rows"]) for payload in accepted]
    fprs = [float(payload["implied_fpr_at_budget_threshold"]) for payload in accepted]
    ratio = max(rows) / max(1, min(rows))
    identified = ratio <= MAX_IDENTIFIED_ROW_RATIO
    usable = max(rows) <= candidate_cap
    return {
        "dataset": corpus.name,
        "related_anchor": anchors.evidence,
        "related_anchored": bool(len(related)),
        "effective_independent_units": effective_units,
        "operating_tail_alpha": float(f"{operating_alpha:.6g}"),
        "budget_threshold": round(budget_threshold, 6),
        "ks_best": round(best, 9),
        "ks_critical_95": round(critical, 9),
        "observationally_equivalent_mixtures": len(accepted),
        "grid_mixtures": len(fits),
        "null_shift_range": [
            min(float(payload["null_shift"]) for payload in accepted),
            max(float(payload["null_shift"]) for payload in accepted),
        ],
        "related_mass_range": [
            min(float(payload["related_mass"]) for payload in accepted),
            max(float(payload["related_mass"]) for payload in accepted),
        ],
        "implied_selected_rows_range": [min(rows), max(rows)],
        "implied_selected_rows_ratio": round(ratio, 3),
        "implied_fpr_range_at_budget": [float(f"{min(fprs):.6g}"), float(f"{max(fprs):.6g}")],
        "expected_false_rows_at_budget_threshold": [
            round(min(fprs) * len(observed), 1),
            round(max(fprs) * len(observed), 1),
        ],
        "best_fit": min(accepted, key=lambda payload: float(payload["ks_distance"])),
        "identified": identified,
        "operating_point_usable": usable,
        "resolves_operating_point": identified and usable,
    }
