"""Pluggable scorer policy: human, local calibrated judge, or budget-capped frontier."""

from llb.scoring.policy.consent import (
    ConsentRecord,
    consent_path,
    load_consent,
    record_consent,
    require_consent,
    scorer_dir,
)
from llb.scoring.policy.errors import BudgetExceeded, ScorerPolicyError
from llb.scoring.policy.frontier import (
    build_frontier_judge_prompt,
    frontier_scorer,
    litellm_frontier_complete,
    parse_frontier_judge_response,
    wrap_llm_complete,
)
from llb.scoring.policy.human import HUMAN_LANE_REASON, human_scorer
from llb.scoring.policy.lanes import DEFAULT_SCORER_LANE, SCORER_LANES, ScorerLane
from llb.scoring.policy.ledger import CostLedger, LedgerEntry
from llb.scoring.policy.resolve import (
    ResolvedScorer,
    ScorerPolicyRequest,
    parse_scorer_lane,
    resolve_scorer,
)

__all__ = [
    "BudgetExceeded",
    "ConsentRecord",
    "CostLedger",
    "DEFAULT_SCORER_LANE",
    "HUMAN_LANE_REASON",
    "LedgerEntry",
    "ResolvedScorer",
    "SCORER_LANES",
    "ScorerLane",
    "ScorerPolicyError",
    "ScorerPolicyRequest",
    "build_frontier_judge_prompt",
    "consent_path",
    "frontier_scorer",
    "human_scorer",
    "litellm_frontier_complete",
    "load_consent",
    "parse_frontier_judge_response",
    "parse_scorer_lane",
    "record_consent",
    "require_consent",
    "resolve_scorer",
    "scorer_dir",
    "wrap_llm_complete",
]
