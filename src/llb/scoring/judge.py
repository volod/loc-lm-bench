"""Gated LLM judge (Ragas), with the trust gate as the first-class contract.

Premise 2: the judge is a GATED dependency. It only contributes to the ranking when it
has been calibrated against human ratings and clears the Spearman-rho floor (default
0.6). Below the bar -- or when no judge is configured / calibrated -- it is DEMOTED to a
diagnostic and the objective reference-correctness score carries the ranking alone.

The gate (`judge_is_trusted`, `run_judge` routing) is pure and unit-testable; the actual
Ragas scoring is injected (`scorer=`) and defaults to a lazy `[rag]`-extra implementation,
so CI never imports ragas.
"""

from dataclasses import dataclass
from typing import Callable

from llb.contracts import JudgeInputRecord, JudgeScore

DEFAULT_THRESHOLD = 0.6


def judge_is_trusted(calibration_rho: float | None, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True only if a calibration rho exists and meets the threshold."""
    return calibration_rho is not None and calibration_rho >= threshold


@dataclass
class JudgeOutcome:
    """Result of attempting to judge a batch of answers."""

    trusted: bool
    reason: str
    scores: list[JudgeScore] | None = None


def run_judge(
    records: list[JudgeInputRecord],
    judge_model: str | None,
    calibration_rho: float | None,
    threshold: float = DEFAULT_THRESHOLD,
    scorer: Callable[[list[JudgeInputRecord], str], list[JudgeScore]] | None = None,
) -> JudgeOutcome:
    """Route to the judge only if gated-in; otherwise return a demoted outcome.

    `records` are dicts with question / answer / contexts / reference. `scorer` defaults
    to the lazy Ragas implementation.
    """
    if judge_model is None:
        return JudgeOutcome(trusted=False, reason="no judge configured")
    if calibration_rho is None:
        return JudgeOutcome(trusted=False, reason="judge not calibrated")
    if not judge_is_trusted(calibration_rho, threshold):
        return JudgeOutcome(
            trusted=False,
            reason=f"calibration rho {calibration_rho:.3f} < threshold {threshold}",
        )
    scorer = scorer or ragas_scorer
    return JudgeOutcome(trusted=True, reason="calibrated", scores=scorer(records, judge_model))


def ragas_scorer(records: list[JudgeInputRecord], judge_model: str) -> list[JudgeScore]:
    """Default scorer: Ragas faithfulness + answer relevancy. Needs the `[rag]` extra."""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            'ERROR: the Ragas judge needs the [rag] extra. Run: uv pip install -e ".[rag]"'
        ) from exc
    # Full Ragas wiring (UA-localized metric prompts, judge endpoint) lands with the
    # M0.5 calibration data in Milestone 3; M1 only needs the gate to be correct.
    raise NotImplementedError(
        "Ragas scoring is wired in Milestone 3 once calibration ratings exist."
    )
