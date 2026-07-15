"""Judge scoring for the runner: build judge records, run the GATED judge (Premise 2), and emit
the calibration worksheet.

`run_eval` calls `_judge_cases` (trusted per-case score) and `_build_judge_metadata`;
`_write_calibration_worksheet` backs the `worksheet=` path. `JudgeScorer` types the injectable
scoring seam.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from llb.core.config import RunConfig
from llb.core.contracts.judging import JudgeInputRecord, JudgeScore, JudgeStatus
from llb.executor.cases import CaseBatch
from llb.scoring.judge.model import judge_is_trusted, run_judge

JudgeScorer = Callable[[list[JudgeInputRecord], str], list[JudgeScore]]
_LOG = logging.getLogger(__name__)


def _judge_records(batch: CaseBatch) -> list[JudgeInputRecord]:
    """The (question, answer, retrieved-contexts) record per case the judge scores."""
    return [
        {
            "question": item.question,
            "answer": answer,
            "contexts": [str(chunk.get("text", "")) for chunk in retrieved],
        }
        for (item, answer), (retrieved, _spans) in zip(batch.answers, batch.retrieval_pairs)
    ]


def _judge_value(score: JudgeScore) -> float:
    """One scalar judge rating per case: the mean of faithfulness + answer-relevancy."""
    return (score["faithfulness"] + score["answer_relevancy"]) / 2.0


def _configured_judge_scorer(config: RunConfig, scorer: JudgeScorer | None) -> JudgeScorer:
    """Bind the configured endpoint while preserving the injectable scorer seam."""
    if scorer is not None:
        return scorer
    from llb.scoring.judge.scorer import deepeval_scorer

    def score(records: list[JudgeInputRecord], model: str) -> list[JudgeScore]:
        return deepeval_scorer(records, model, base_url=config.judge_base_url)

    return score


def _judge_cases(
    config: RunConfig,
    batch: CaseBatch,
    judge_rho: float | None,
    scorer: JudgeScorer | None,
) -> float | None:
    """Score answers with the GATED judge (Premise 2) and attach per-case judge scores.

    Returns the mean per-case judge score ONLY when the judge is configured AND trusted
    (calibration rho >= threshold); otherwise the judge stays a demoted diagnostic and objective
    correctness ranks alone. The per-case judge value is the mean of faithfulness + relevancy.
    """
    if config.judge_model is None:
        return None
    outcome = run_judge(
        _judge_records(batch),
        config.judge_model,
        judge_rho,
        config.judge_threshold,
        scorer=_configured_judge_scorer(config, scorer),
    )
    if not outcome.trusted or not outcome.scores:
        _LOG.info("[run-eval] judge demoted (%s); objective ranks alone", outcome.reason)
        return None
    per_case = [_judge_value(s) for s in outcome.scores]
    for row, value in zip(batch.rows, per_case):
        row["judge_score"] = round(value, 4)
    return sum(per_case) / len(per_case) if per_case else None


def _judge_ratings(
    config: RunConfig, batch: CaseBatch, scorer: JudgeScorer | None
) -> list[float] | None:
    """Run the judge UNGATED and return one rating per case (judge calibration gate calibration scaffolding).

    Calibration measures whether the judge AGREES with humans, so the judge runs regardless of
    its (not-yet-known) trust -- the gate is irrelevant here. Returns None when no judge is
    configured; raises if the judge backend is unavailable (so the worksheet path can warn).
    """
    if config.judge_model is None:
        return None
    score_fn = _configured_judge_scorer(config, scorer)
    scores = score_fn(_judge_records(batch), config.judge_model)
    return [_judge_value(s) for s in scores]


def _build_judge_metadata(config: RunConfig, judge_rho: float | None) -> JudgeStatus:
    judge_metadata: JudgeStatus = {
        "calibration_rho": judge_rho,
        "threshold": config.judge_threshold,
        "trusted": judge_is_trusted(judge_rho, config.judge_threshold),
    }
    if config.judge_model is None:
        return judge_metadata
    from llb.scoring.judge.endpoint import judge_experiment_metadata

    experiment_metadata = judge_experiment_metadata(config.judge_model, config.judge_base_url)
    judge_metadata["provider"] = experiment_metadata["provider"]
    judge_metadata["model"] = experiment_metadata["model"]
    judge_metadata["base_url"] = experiment_metadata["base_url"]
    judge_metadata["prompt_language"] = experiment_metadata["prompt_language"]
    judge_metadata["metrics"] = experiment_metadata["metrics"]
    return judge_metadata


def _write_calibration_worksheet(
    config: RunConfig,
    batch: CaseBatch,
    worksheet: Path | str,
    judge_scorer: JudgeScorer | None,
) -> int:
    from llb.judge.calibration_worksheet import write_filled_worksheet

    judge_ratings: list[float] | None = None
    if config.judge_model is not None:
        try:
            judge_ratings = _judge_ratings(config, batch, judge_scorer)
        except (Exception, SystemExit) as exc:
            _LOG.warning(
                "[run-eval] judge unavailable for the worksheet (%s); judge_rating left blank "
                "-- pick the judge (OQ2) and install its backend to calibrate.",
                exc,
            )
    return write_filled_worksheet(batch.answers, Path(worksheet), judge_ratings=judge_ratings)
