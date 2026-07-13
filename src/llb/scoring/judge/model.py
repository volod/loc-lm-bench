"""Judge calibration gate, outcome, and batch routing policy."""

from collections.abc import Callable
from dataclasses import dataclass

from llb.core.contracts import JudgeDiagnostics, JudgeInputRecord, JudgeScore
from llb.prompts import render_text

DEFAULT_THRESHOLD = 0.6
JUDGE_BIAS_NOTE = render_text("scoring.judge.bias_note")


def judge_is_trusted(calibration_rho: float | None, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Return whether calibration exists and clears the configured threshold."""
    return calibration_rho is not None and calibration_rho >= threshold


@dataclass
class JudgeOutcome:
    """Result of attempting to judge a batch of answers."""

    trusted: bool
    reason: str
    scores: list[JudgeScore] | None = None
    diagnostics: JudgeDiagnostics | None = None


def run_judge(
    records: list[JudgeInputRecord],
    judge_model: str | None,
    calibration_rho: float | None,
    threshold: float = DEFAULT_THRESHOLD,
    scorer: Callable[[list[JudgeInputRecord], str], list[JudgeScore]] | None = None,
) -> JudgeOutcome:
    """Route to the judge only after calibration establishes trust."""
    if judge_model is None:
        return JudgeOutcome(trusted=False, reason="no judge configured")
    if calibration_rho is None:
        return JudgeOutcome(trusted=False, reason="judge not calibrated")
    if not judge_is_trusted(calibration_rho, threshold):
        return JudgeOutcome(
            trusted=False,
            reason=f"calibration rho {calibration_rho:.3f} < threshold {threshold}",
        )
    if scorer is None:
        from llb.scoring.judge.scorer import deepeval_scorer

        scorer = deepeval_scorer
    return JudgeOutcome(trusted=True, reason="calibrated", scores=scorer(records, judge_model))
