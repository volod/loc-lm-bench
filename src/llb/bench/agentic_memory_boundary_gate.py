"""Direction-aware evidence gate for cap-fitting compact-versus-cap cells.

Completion is higher-is-better and total model-input cost is LOWER-is-better, and the paired block
prints ONE positive-tail randomization p. Reading a cost row off that field silently asks "is
compact more expensive", which is the opposite of the operational question. Every cap-fitting
reading therefore goes through this module: it takes the compact-minus-cap delta as recorded and
cuts it with the two-sided exact sign test plus the reachability floor for the reporting level.
"""

from typing import cast

from llb.bench.agentic_compact_vs_cap_report import (
    VERDICT_PREFER_CAP,
    VERDICT_PREFER_COMPACT,
)
from llb.bench.agentic_context_report import (
    METRIC_COMPLETION,
    METRIC_TOTAL_MODEL_INPUT_TOKENS,
)
from llb.rag.fusion_evidence.evidence_gate import (
    READING_SEPARATED,
    minimum_discordant_pairs,
    reaches_reporting_level,
)

SIDE_COMPACT_CHEAPER = "compact_cheaper"
SIDE_CAP_CHEAPER = "cap_cheaper"
SIDE_TIED = "cost_tied"


def clears_tighter_completion(row: dict[str, object], confidence: float) -> bool:
    """Whether one cell's completion gain is separated AND reachable at `confidence`."""
    paired = cast(dict[str, dict[str, object]], row["paired"])[METRIC_COMPLETION]
    stability = cast(dict[str, object], paired.get("stability", {}))
    discordant = int(cast(int, paired["wins"])) + int(cast(int, paired["losses"]))
    return bool(
        row["verdict"] == VERDICT_PREFER_COMPACT
        and discordant >= minimum_discordant_pairs(confidence)
        and stability.get("tighter_reading") == READING_SEPARATED
    )


def boundary_cost_evidence(row: dict[str, object], confidence: float) -> dict[str, object]:
    """Normalize one cap-fitting cell onto lower-is-better cost evidence."""
    paired = cast(dict[str, dict[str, object]], row["paired"])
    completion = paired[METRIC_COMPLETION]
    cost = paired[METRIC_TOTAL_MODEL_INPUT_TOKENS]
    cost_delta = cast(dict[str, float], cost["delta"])
    discordant = int(cast(int, cost["wins"])) + int(cast(int, cost["losses"]))
    sign_test_p = float(cast(float, cost["sign_test_p"]))
    cost_clears = bool(
        sign_test_p <= 1.0 - confidence and reaches_reporting_level(discordant, confidence)
    )
    completion_tied = cast(dict[str, float], completion["delta"])["mean"] == 0.0
    return {
        "confidence": confidence,
        "completion_tied": completion_tied,
        "compact_minus_cap_total_model_input_tokens": cost_delta,
        "cost_discordant_pairs": discordant,
        "cost_two_sided_sign_test_p": sign_test_p,
        "cost_clears": cost_clears,
        "cost_side": cost_side(cost_delta["mean"], cost_clears),
        "compact_preference_clears": bool(
            row["verdict"] == VERDICT_PREFER_COMPACT
            and (
                clears_tighter_completion(row, confidence)
                or (completion_tied and cost_delta["mean"] < 0.0 and cost_clears)
            )
        ),
        "cap_preference_clears": bool(
            row["verdict"] == VERDICT_PREFER_CAP
            and completion_tied
            and cost_delta["mean"] > 0.0
            and cost_clears
        ),
    }


def cost_side(mean_delta: float, cost_clears: bool) -> str:
    """Which policy the measured cost favors, or `cost_tied` when the sign is not readable."""
    if not cost_clears or mean_delta == 0.0:
        return SIDE_TIED
    return SIDE_COMPACT_CHEAPER if mean_delta < 0.0 else SIDE_CAP_CHEAPER
