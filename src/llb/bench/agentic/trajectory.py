"""The opt-in, gated trajectory-quality judge -- a signal a deterministic env-state check cannot
cover: is the final answer grounded in what the tools actually returned, and does it address the
goal?

It is recorded ALONGSIDE objective completion-rate, never folded into the headline, and only when
the judge is configured AND trusted (`judge_rho >= threshold`). The `scorer` is injectable, so the
wiring is provable with a FAKE judge (no DeepEval / endpoint / GPU).
"""

import json
import logging

from llb.bench.agentic.model import (
    AgenticTask,
    Episode,
    _JudgeConfig,
    _TrajectoryQualityResult,
)
from llb.bench.common import mean, run_gated_judge
from llb.core.contracts.benchmarks import AgenticCaseRow
from llb.core.contracts.judging import JudgeInputRecord, JudgeScore
from llb.prompts.registry import render_text
from llb.scoring.leaderboard import bootstrap_mean_ci

_LOG = logging.getLogger(__name__)

# The judge "question" for trajectory quality: a fixed UA intent that frames the agent's job, so
# answer-relevancy scores whether the final answer addresses the goal while faithfulness scores
# whether it stays grounded in the tool observations fed back as the retrieval context.
_TRAJECTORY_INTENT = render_text("bench.agentic.trajectory_intent")


def _trajectory_records(
    tasks: list[AgenticTask], episodes: list[Episode]
) -> list[JudgeInputRecord]:
    """One (goal, final answer, [tool observations]) record per episode for the trajectory judge.

    The tool observations become the retrieval context, so faithfulness scores whether the final
    answer stays grounded in what the tools actually returned (a check the env-state assertions
    cannot make), while answer-relevancy scores whether it addresses the goal.
    """
    return [
        {
            "question": render_text(
                "bench.agentic.trajectory_question",
                {"intent": _TRAJECTORY_INTENT, "task_prompt": task.prompt},
            ),
            "answer": episode.answer,
            "contexts": [
                f"{name}({json.dumps(args, ensure_ascii=False)}) -> {obs}"
                for name, args, obs in episode.transcript
            ],
        }
        for task, episode in zip(tasks, episodes)
    ]


def trajectory_quality(score: JudgeScore) -> float:
    """Collapse the judge's two G-Eval signals into one trajectory-quality scalar: the answer is
    GROUNDED in the tool observations (faithfulness) AND addresses the goal (answer_relevancy)."""
    return (float(score["faithfulness"]) + float(score["answer_relevancy"])) / 2.0


def _attach_trajectory_quality(
    rows: list[AgenticCaseRow], scores: list[JudgeScore]
) -> tuple[float, tuple[float, float] | None]:
    per_case = [trajectory_quality(score) for score in scores]
    for row, value in zip(rows, per_case):
        row["trajectory_quality"] = round(value, 6)
    return round(mean(per_case), 6), bootstrap_mean_ci(per_case)


def _run_trajectory_judge(
    tasks: list[AgenticTask],
    episodes: list[Episode],
    rows: list[AgenticCaseRow],
    config: _JudgeConfig,
) -> _TrajectoryQualityResult:
    outcome = run_gated_judge(
        _trajectory_records(tasks, episodes),
        judge_model=config.model,
        judge_rho=config.rho,
        threshold=config.threshold,
        scorer=config.scorer,
        base_url=config.base_url,
    )
    if outcome.trusted and outcome.scores:
        value, ci = _attach_trajectory_quality(rows, outcome.scores)
        return _TrajectoryQualityResult(outcome=outcome, value=value, ci=ci)
    if config.model is not None:
        _LOG.info("[agentic] judge demoted (%s); objective completion ranks alone", outcome.reason)
    return _TrajectoryQualityResult(outcome=outcome, value=None, ci=None)
