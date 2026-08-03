"""Prospective Gemma controller-authority feedback transfer."""

from llb.bench.agentic.loop_policy import REPEAT_FEEDBACK_GEMMA_AUTHORITY
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_loop_feedback_transfer import (
    FORBIDDEN_NOTICE_TERMS,
    FeedbackTransferRun,
    analyze_feedback_transfer,
    validate_neutral_feedback_design,
)

STUDY_KIND = "repeat_feedback_controller_authority_transfer"
EXPECTED_HYPOTHESIS = (
    "Controller-authority feedback will make Gemma advance after a suppressed repeat in at least "
    "three task families while preserving material completion and paired cost bounds."
)

FeedbackAuthorityRun = FeedbackTransferRun

__all__ = [
    "EXPECTED_HYPOTHESIS",
    "FORBIDDEN_NOTICE_TERMS",
    "FeedbackAuthorityRun",
    "analyze_feedback_authority",
    "validate_feedback_authority_design",
]


def validate_feedback_authority_design(design: dict[str, object], tasks: list[AgenticTask]) -> None:
    """Lock the authority notice, fresh ledger, seeds, and gates before inference."""
    validate_neutral_feedback_design(
        design,
        tasks,
        study_kind=STUDY_KIND,
        hypothesis=EXPECTED_HYPOTHESIS,
        candidate_variant=REPEAT_FEEDBACK_GEMMA_AUTHORITY,
    )


def analyze_feedback_authority(
    design: dict[str, object], runs: list[FeedbackAuthorityRun]
) -> dict[str, object]:
    """Apply the transfer gates and expose the authority-specific decision."""
    analysis = analyze_feedback_transfer(design, runs)
    supported = bool(analysis["supports_task_family_transfer"])
    analysis["supports_controller_authority_transfer"] = supported
    analysis["reason"] = (
        "the authority notice clears response, completion, and cost gates on both seeds"
        if supported
        else "the authority notice does not clear every predeclared gate on both seeds"
    )
    return analysis
