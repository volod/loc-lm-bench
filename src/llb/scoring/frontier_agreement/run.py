"""Orchestrate the frontier-judge authorization run: score, correlate, price, report.

One run judges the same calibration worksheet with every requested provider, correlates each
provider against both the human rating and the local judge rating, prices the run per item,
and writes the evidence the operator signs off on. Providers are independent: one provider's
budget abort or transport failure does not discard the evidence already gathered for the
others, because each provider owns its own ledger directory.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llb.core.fsutil import atomic_write_text
from llb.core.paths import resolve_data_dir
from llb.judge.calibration_stats import DEFAULT_THRESHOLD
from llb.scoring.frontier_agreement.agreement import (
    AGREEMENT_METRICS,
    ProviderAgreement,
    build_agreement,
    metric_value,
)
from llb.scoring.frontier_agreement.items import AgreementItem, load_agreement_items
from llb.scoring.frontier_agreement.provider import (
    ProviderResult,
    provider_slug,
    score_with_provider,
)
from llb.scoring.frontier_agreement.report import render_report
from llb.scoring.policy.errors import BudgetExceeded, ScorerPolicyError

_LOG = logging.getLogger(__name__)

ARTIFACT_ROOT = "frontier-judge"
REPORT_FILENAME = "report.md"
AGREEMENT_FILENAME = "agreement.json"
SCORES_FILENAME = "scores.jsonl"

CompleteFactory = Callable[[str], Any]
"""Given a model id, return a `FrontierComplete`; injected by tests, else litellm's default."""


@dataclass(frozen=True)
class ProviderFailure:
    """A provider whose run did not complete; its partial ledger is preserved on disk."""

    model: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"model": self.model, "reason": self.reason}


def default_out_dir(data_dir: Path | str | None = None) -> Path:
    from llb.bench.common import new_run_timestamp

    _, stamped = new_run_timestamp()
    return resolve_data_dir(data_dir) / ARTIFACT_ROOT / stamped


def _write_scores(path: Path, items: list[AgreementItem], result: ProviderResult) -> None:
    lines = []
    for item, score in zip(items, result.scores):
        row: dict[str, Any] = {
            "item_id": item.item_id,
            "model": result.model,
            "human_rating": item.human_rating,
            "local_rating": item.local_rating,
        }
        row.update({metric: round(metric_value(score, metric), 6) for metric in AGREEMENT_METRICS})
        lines.append(json.dumps(row, ensure_ascii=True))
    atomic_write_text(path, "\n".join(lines) + "\n" if lines else "")


def _score_one(
    items: list[AgreementItem],
    model: str,
    out_dir: Path,
    *,
    max_usd: float | None,
    max_calls: int | None,
    complete_factory: CompleteFactory | None,
    threshold: float,
) -> tuple[ProviderAgreement | None, ProviderFailure | None]:
    run_dir = out_dir / provider_slug(model)
    complete = None if complete_factory is None else complete_factory(model)
    try:
        result = score_with_provider(
            items,
            model=model,
            run_dir=run_dir,
            max_usd=max_usd,
            max_calls=max_calls,
            complete=complete,
        )
    except (BudgetExceeded, ScorerPolicyError) as exc:
        _LOG.error("[frontier-judge] %s did not complete: %s", model, exc)
        return None, ProviderFailure(model=model, reason=str(exc))
    _write_scores(run_dir / SCORES_FILENAME, items, result)
    agreement = build_agreement(
        model=result.model,
        provider=result.provider,
        scores=result.scores,
        human_ratings=[item.human_rating for item in items],
        local_ratings=[item.local_rating for item in items],
        ledger=result.ledger,
        threshold=threshold,
    )
    return agreement, None


def run_frontier_agreement(
    worksheet: Path,
    models: list[str],
    *,
    goldset: Path | None = None,
    corpus_root: Path | None = None,
    out_dir: Path | None = None,
    data_dir: Path | str | None = None,
    max_usd: float | None = None,
    max_calls: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    limit: int | None = None,
    complete_factory: CompleteFactory | None = None,
) -> tuple[dict[str, Any], Path]:
    """Judge the worksheet with every provider and write the agreement + cost evidence."""
    items = load_agreement_items(
        Path(worksheet), goldset=goldset, corpus_root=corpus_root, limit=limit
    )
    if not items:
        raise ScorerPolicyError(f"no judgeable rows (with a model_answer) in {worksheet}")
    destination = Path(out_dir) if out_dir is not None else default_out_dir(data_dir)
    destination.mkdir(parents=True, exist_ok=True)

    agreements: list[ProviderAgreement] = []
    failures: list[ProviderFailure] = []
    for model in models:
        agreement, failure = _score_one(
            items,
            model,
            destination,
            max_usd=max_usd,
            max_calls=max_calls,
            complete_factory=complete_factory,
            threshold=threshold,
        )
        if agreement is not None:
            agreements.append(agreement)
        if failure is not None:
            failures.append(failure)

    payload: dict[str, Any] = {
        "run": destination.name,
        "worksheet": str(worksheet),
        "goldset": None if goldset is None else str(goldset),
        "n_items": len(items),
        "threshold": threshold,
        "providers": [agreement.to_dict() for agreement in agreements],
        "failures": [failure.to_dict() for failure in failures],
    }
    atomic_write_text(
        destination / AGREEMENT_FILENAME,
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
    )
    atomic_write_text(destination / REPORT_FILENAME, render_report(payload, agreements))
    return payload, destination
