"""Rank agreement and cost-per-item math for the frontier-judge authorization decision.

Two questions decide whether a frontier judge may gate autonomously:

1. does it rank Ukrainian answers like the human rater does (rho vs `human_rating`), and
2. does it agree with the local judge already in the loop (rho vs `judge_rating`)?

Both reuse the calibration statistics (Spearman over average ranks + a bootstrap CI) so the
trust threshold means exactly what it means for the local judge. Scales differ -- humans rate
1..5, judges score 0..1 -- which a rank correlation is indifferent to by construction.
"""

import math
from dataclasses import dataclass
from typing import Any

from llb.core.contracts.judging import CalibrationResult, JudgeScore
from llb.judge.calibration_stats import DEFAULT_THRESHOLD, calibrate
from llb.scoring.judge.scorer import judge_value

MEAN_METRIC = "mean"
AGREEMENT_METRICS = ("faithfulness", "answer_relevancy", MEAN_METRIC)
"""Per-metric correlations; `mean` is the headline (it is the scalar `run-eval` records)."""

MIN_PAIRS = 2
"""Fewer paired ratings than this cannot produce a correlation at all."""

CAP_SAFETY_FACTOR = 2.0
"""Headroom over measured cost when recommending a default per-run cap.

A recommended cap is a guard rail, not a forecast: prompt length, retry behavior, and provider
price changes all push real spend above the measured mean, and a cap that trips mid-run costs
an operator a resume cycle. Double the measured cost is cheap insurance against that.
"""

CENT = 0.01


def metric_value(score: JudgeScore, metric: str) -> float:
    """One scalar per score for the named metric; `mean` matches the `run-eval` judge scalar."""
    if metric == MEAN_METRIC:
        return judge_value(score)
    return float(score[metric])  # type: ignore[literal-required]


def paired(reference: list[float | None], judged: list[float]) -> tuple[list[float], list[float]]:
    """Drop positions where the reference rating is missing; keep both lists aligned."""
    left: list[float] = []
    right: list[float] = []
    for ref, value in zip(reference, judged):
        if ref is None:
            continue
        left.append(float(ref))
        right.append(float(value))
    return left, right


def correlate(
    reference: list[float | None],
    judged: list[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> CalibrationResult | None:
    """Spearman rho + bootstrap CI + trust decision, or None when there are too few pairs."""
    left, right = paired(reference, judged)
    if len(left) < MIN_PAIRS:
        return None
    return calibrate(left, right, threshold)


def _by_metric(
    reference: list[float | None],
    scores: list[JudgeScore],
    threshold: float,
) -> dict[str, CalibrationResult | None]:
    return {
        metric: correlate(reference, [metric_value(s, metric) for s in scores], threshold)
        for metric in AGREEMENT_METRICS
    }


def cost_summary(ledger: dict[str, Any], n_items: int) -> dict[str, Any]:
    """Cost per item plus the cap a full run of `n_items` should be given.

    `cost_usd` of zero means litellm could not price the response (an unlisted model), not that
    the call was free; the recommended cap is left unset in that case rather than guessed at.
    """
    cost_usd = float(ledger.get("cost_usd") or 0.0)
    calls = int(ledger.get("calls") or 0)
    priced = cost_usd > 0.0 and calls > 0
    cost_per_item = cost_usd / calls if priced else None
    recommended_cap = None
    if cost_per_item is not None and n_items > 0:
        raw = cost_per_item * n_items * CAP_SAFETY_FACTOR
        recommended_cap = round(math.ceil(raw / CENT) * CENT, 2)
    return {
        "calls": calls,
        "cost_usd": round(cost_usd, 6),
        "cost_per_item_usd": None if cost_per_item is None else round(cost_per_item, 6),
        "priced": priced,
        "n_items": n_items,
        "cap_safety_factor": CAP_SAFETY_FACTOR,
        "recommended_cap_usd": recommended_cap,
        "max_usd": ledger.get("max_usd"),
        "max_calls": ledger.get("max_calls"),
    }


@dataclass(frozen=True)
class ProviderAgreement:
    """Everything the human needs to accept or reject one provider for autonomous gating."""

    model: str
    provider: str
    n_items: int
    vs_human: dict[str, CalibrationResult | None]
    vs_local: dict[str, CalibrationResult | None]
    cost: dict[str, Any]
    threshold: float

    @property
    def headline(self) -> CalibrationResult | None:
        """The rho that decides trust: agreement with the human on the mean judge scalar."""
        return self.vs_human[MEAN_METRIC]

    @property
    def recommendation(self) -> str:
        """Machine recommendation only; the recorded decision stays the human's to make."""
        result = self.headline
        if result is None:
            return "insufficient-data"
        return "trusted" if result["trusted"] else "not-trusted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "n_items": self.n_items,
            "threshold": self.threshold,
            "vs_human": self.vs_human,
            "vs_local": self.vs_local,
            "cost": self.cost,
            "recommendation": self.recommendation,
            "human_decision": "pending",
        }


def build_agreement(
    *,
    model: str,
    provider: str,
    scores: list[JudgeScore],
    human_ratings: list[float | None],
    local_ratings: list[float | None],
    ledger: dict[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
) -> ProviderAgreement:
    """Correlate one provider's scores against both references and price its run."""
    return ProviderAgreement(
        model=model,
        provider=provider,
        n_items=len(scores),
        vs_human=_by_metric(human_ratings, scores, threshold),
        vs_local=_by_metric(local_ratings, scores, threshold),
        cost=cost_summary(ledger, len(scores)),
        threshold=threshold,
    )
