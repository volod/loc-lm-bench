"""Resolve a ScorerPolicy lane into a concrete JudgeScorer + metadata."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.core.contracts.judging import JudgeInputRecord, JudgeScore
from llb.scoring.policy.consent import record_consent, require_consent
from llb.scoring.policy.errors import ScorerPolicyError
from llb.scoring.policy.frontier import frontier_scorer
from llb.scoring.policy.human import HUMAN_LANE_REASON, human_scorer
from llb.scoring.policy.lanes import DEFAULT_SCORER_LANE, SCORER_LANES, ScorerLane
from llb.scoring.policy.ledger import CostLedger

JudgeScorerFn = Callable[[list[JudgeInputRecord], str], list[JudgeScore]]


@dataclass(frozen=True)
class ResolvedScorer:
    """Concrete scorer plus the metadata recorded in the run manifest."""

    lane: ScorerLane
    scorer: JudgeScorerFn
    reason: str
    ledger: CostLedger | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScorerPolicyRequest:
    """Inputs needed to bind one scorer lane for a run."""

    lane: ScorerLane
    judge_model: str | None
    judge_base_url: str | None = None
    egress_consent: bool = False
    max_usd: float | None = None
    max_calls: int | None = None
    run_dir: Path | None = None
    local_scorer: JudgeScorerFn | None = None
    frontier_complete: Any | None = None


def parse_scorer_lane(value: str | None) -> ScorerLane:
    """Parse a CLI/config lane string; default is local."""
    if value is None or value == "":
        return DEFAULT_SCORER_LANE
    if value not in SCORER_LANES:
        raise ScorerPolicyError(f"scorer_policy must be one of {SCORER_LANES}, got {value!r}")
    return value


def resolve_scorer(request: ScorerPolicyRequest) -> ResolvedScorer:
    """Bind the selected lane; frontier requires consent, budget, and a run directory."""
    if request.lane == "human":
        return ResolvedScorer(
            lane="human",
            scorer=human_scorer(),
            reason=HUMAN_LANE_REASON,
            metadata={"scorer_policy": "human", "provider": "human"},
        )
    if request.lane == "local":
        return _resolve_local(request)
    if request.lane == "frontier":
        return _resolve_frontier(request)
    raise ScorerPolicyError(f"unknown scorer_policy {request.lane!r}")


def _resolve_local(request: ScorerPolicyRequest) -> ResolvedScorer:
    if request.judge_model is None:
        raise ScorerPolicyError("local scorer_policy requires judge_model")
    if request.local_scorer is not None:
        scorer = request.local_scorer
        provider = "injected"
    else:
        from llb.scoring.judge.scorer import deepeval_scorer

        base_url = request.judge_base_url

        def scorer(records: list[JudgeInputRecord], model: str) -> list[JudgeScore]:
            return deepeval_scorer(records, model, base_url=base_url)

        provider = "deepeval-geval"

    return ResolvedScorer(
        lane="local",
        scorer=scorer,
        reason="local calibrated judge",
        metadata={
            "scorer_policy": "local",
            "provider": provider,
            "model": request.judge_model,
            "base_url": request.judge_base_url,
        },
    )


def _resolve_frontier(request: ScorerPolicyRequest) -> ResolvedScorer:
    if request.judge_model is None:
        raise ScorerPolicyError("frontier scorer_policy requires judge_model")
    if request.run_dir is None:
        raise ScorerPolicyError("frontier scorer_policy requires a run directory for the ledger")
    if request.egress_consent:
        consent = record_consent(
            request.run_dir,
            model=request.judge_model,
            approved=True,
            max_usd=request.max_usd,
            max_calls=request.max_calls,
        )
    else:
        consent = require_consent(request.run_dir, model=request.judge_model)
    ledger = CostLedger.open(request.run_dir, max_usd=consent.max_usd, max_calls=consent.max_calls)
    return ResolvedScorer(
        lane="frontier",
        scorer=frontier_scorer(request.judge_model, ledger, complete=request.frontier_complete),
        reason="frontier judge under budget cap",
        ledger=ledger,
        metadata={
            "scorer_policy": "frontier",
            "provider": "litellm-frontier",
            "model": request.judge_model,
            "budget": ledger.summary(),
            "consent": consent.to_dict(),
        },
    )
