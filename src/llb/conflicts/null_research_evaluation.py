"""Metrics and threshold decisions for corpus-conflict null research."""

from bisect import bisect_left
import math

import numpy as np

from llb.conflicts.null_distribution import _quantile
from llb.conflicts.null_research_geometry import DocPair
from llb.core.contracts.common import JsonObject

DEFAULT_RESEARCH_FPR = 0.01
DEFAULT_RESEARCH_RANK_BUDGET = 12
DEFAULT_TRANSFER_THRESHOLD = 0.6
DEFAULT_MAX_GOODS_CANDIDATES = 12
MIN_TAIL_OBSERVATIONS = 20
WILSON_Z_95 = 1.959963984540054
COVERAGE_SIMULATIONS = 10_000
MIN_COVERAGE_PROBABILITY = 0.93

_EQUIVALENT_2021 = (
    "regulation-2021.md",
    "regulation-2021-copy.md",
    "regulation-2021-reformatted.md",
)


def _pair(left: str, right: str) -> DocPair:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


FIXTURE_POSITIVE_DOC_PAIRS: frozenset[DocPair] = frozenset(
    {
        *(
            _pair(_EQUIVALENT_2021[left], _EQUIVALENT_2021[right])
            for left in range(len(_EQUIVALENT_2021))
            for right in range(left + 1, len(_EQUIVALENT_2021))
        ),
        *(_pair(doc_id, "regulation-2024.md") for doc_id in _EQUIVALENT_2021),
        _pair("e-appeals-note.md", "regulation-2024.md"),
        _pair("deadline-note.md", "regulation-2024.md"),
    }
)
"""Document-pair closure of the planted fixture's claim-bearing relations.

The three 2021 editions are equivalent, so their internal closure and each edition's relation to
the 2024 revision are genuine positives. This prevents equivalent aliases from being counted as
false positives merely because the README names the baseline edition once.
"""


def threshold_for_fpr(null_scores: list[float], fpr: float) -> float:
    if not null_scores:
        raise ValueError("a null distribution cannot be empty")
    if not 0.0 < fpr < 1.0:
        raise ValueError("fpr must be between zero and one")
    return _quantile(null_scores, 1.0 - fpr)


def count_at_or_above(sorted_scores: list[float], threshold: float) -> int:
    return len(sorted_scores) - bisect_left(sorted_scores, threshold)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    z2 = WILSON_Z_95 * WILSON_Z_95
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        WILSON_Z_95
        * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def simulated_wilson_coverage(
    total: int,
    probability: float,
    *,
    seed: int,
    simulations: int = COVERAGE_SIMULATIONS,
) -> float:
    """Deterministically check Wilson coverage for the declared independent-unit count."""
    if total <= 0:
        return 0.0
    if not 0.0 < probability < 1.0:
        raise ValueError("coverage probability must be between zero and one")
    if simulations < 1:
        raise ValueError("coverage simulations must be positive")
    draws = np.random.default_rng(seed).binomial(total, probability, simulations)
    covered = 0
    for successes in draws.tolist():
        lower, upper = wilson_interval(int(successes), total)
        covered += lower <= probability <= upper
    return covered / simulations


def null_tail_payload(null_scores: list[float], threshold: float, nominal_fpr: float) -> JsonObject:
    exceedances = count_at_or_above(null_scores, threshold)
    lower, upper = wilson_interval(exceedances, len(null_scores))
    expected_tail = nominal_fpr * len(null_scores)
    return {
        "n": len(null_scores),
        "min": round(null_scores[0], 6),
        "mean": round(sum(null_scores) / len(null_scores), 6),
        "max": round(null_scores[-1], 6),
        "nominal_fpr": nominal_fpr,
        "tail_exceedances": exceedances,
        "empirical_tail_rate": round(exceedances / len(null_scores), 9),
        "tail_wilson_95": [round(lower, 9), round(upper, 9)],
        "expected_tail_observations": round(expected_tail, 3),
        "tail_resolved": expected_tail >= MIN_TAIL_OBSERVATIONS,
    }


def fixture_metrics(
    maxima: dict[DocPair, float], threshold: float, positives: frozenset[DocPair]
) -> JsonObject:
    available_positives = positives & maxima.keys()
    predicted = {pair for pair, score in maxima.items() if score >= threshold}
    tp = len(predicted & available_positives)
    fp = len(predicted - available_positives)
    fn = len(available_positives - predicted)
    negatives = maxima.keys() - available_positives
    tn = len(negatives - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(threshold, 6),
        "predicted_document_pairs": len(predicted),
        "labelled_positive_pairs": len(available_positives),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_positive_rate": round(fp / (fp + tn), 6) if fp + tn else 0.0,
    }


def rank_baseline(
    observed_scores: list[float],
    maxima: dict[DocPair, float],
    positives: frozenset[DocPair],
    budget: int,
) -> JsonObject:
    if not observed_scores:
        raise ValueError("the observed fixture distribution is empty")
    rank = min(max(1, budget), len(observed_scores))
    threshold = observed_scores[-rank]
    return {
        "budget": budget,
        "selected_chunk_pairs": count_at_or_above(observed_scores, threshold),
        **fixture_metrics(maxima, threshold, positives),
    }


def fit_labelled_threshold(
    maxima: dict[DocPair, float], positives: frozenset[DocPair]
) -> tuple[float, JsonObject]:
    """Choose F1 on the planted fixture, breaking ties toward precision then strictness."""
    if not maxima:
        raise ValueError("labelled calibration needs document-pair scores")
    candidates = sorted(set(maxima.values()))
    scored = [
        (threshold, fixture_metrics(maxima, threshold, positives)) for threshold in candidates
    ]
    threshold, metrics = max(
        scored,
        key=lambda item: (
            float(item[1]["f1"]),
            float(item[1]["precision"]),
            float(item[1]["recall"]),
            item[0],
        ),
    )
    return threshold, metrics


def transfer_payload(
    observed_scores: list[float], threshold: float, baseline_threshold: float
) -> JsonObject:
    selected = count_at_or_above(observed_scores, threshold)
    baseline = count_at_or_above(observed_scores, baseline_threshold)
    recovered = count_at_or_above(observed_scores, max(threshold, baseline_threshold))
    return {
        "threshold": round(threshold, 6),
        "selected_chunk_pairs": selected,
        "baseline_threshold": baseline_threshold,
        "baseline_chunk_pairs": baseline,
        "baseline_recovered": recovered,
        "baseline_recall": round(recovered / baseline, 6) if baseline else 1.0,
    }
