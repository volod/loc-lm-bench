"""Score the agreement worksheet with one frontier provider under the existing budget cap.

This is a thin orchestration over the scorer-policy seam: consent, ledger, resume, and the
budget abort all stay in `llb.scoring.policy`, so a provider run here obeys exactly the same
guarantees a `run-eval --scorer-policy frontier` run does.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.core.contracts.judging import JudgeScore
from llb.scoring.frontier_agreement.items import AgreementItem
from llb.scoring.policy import ScorerPolicyRequest, resolve_scorer

_LOG = logging.getLogger(__name__)


def provider_slug(model: str) -> str:
    """Filesystem-safe directory name for a litellm model id (`openai/gpt-x` -> `openai_gpt-x`)."""
    slug = "".join(char if char.isalnum() or char in "-._" else "_" for char in model)
    return slug.strip("_") or "provider"


def provider_name(model: str) -> str:
    """The provider prefix of a litellm model id; the whole id when it carries no prefix."""
    return model.split("/", 1)[0] if "/" in model else model


@dataclass(frozen=True)
class ProviderResult:
    """Per-item frontier scores for one provider plus its ledger spend summary."""

    model: str
    provider: str
    scores: list[JudgeScore]
    ledger: dict[str, Any]
    run_dir: Path


def score_with_provider(
    items: list[AgreementItem],
    *,
    model: str,
    run_dir: Path,
    max_usd: float | None,
    max_calls: int | None,
    complete: Any | None = None,
    egress_consent: bool = True,
) -> ProviderResult:
    """Judge every item with `model`, recording consent + spend under `run_dir/scorer/`.

    Raises `BudgetExceeded` when the cap is hit; the ledger keeps the cases already scored, so
    re-running the same `run_dir` with a raised cap resumes instead of re-spending.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_scorer(
        ScorerPolicyRequest(
            lane="frontier",
            judge_model=model,
            egress_consent=egress_consent,
            max_usd=max_usd,
            max_calls=max_calls,
            run_dir=run_dir,
            frontier_complete=complete,
        )
    )
    _LOG.info("[frontier-judge] scoring %d items with %s", len(items), model)
    scores = resolved.scorer([item.judge_record() for item in items], model)
    ledger = resolved.ledger.summary() if resolved.ledger is not None else {}
    return ProviderResult(
        model=model,
        provider=provider_name(model),
        scores=scores,
        ledger=ledger,
        run_dir=run_dir,
    )
