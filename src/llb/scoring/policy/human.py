"""Human scorer lane: no automated judge; objective scores rank alone."""

from collections.abc import Callable

from llb.core.contracts.judging import JudgeInputRecord, JudgeScore

HUMAN_LANE_REASON = "human review required; automated judge skipped"


def human_scorer() -> Callable[[list[JudgeInputRecord], str], list[JudgeScore]]:
    """Return an empty-score scorer so the gate stays diagnostic until a human rates."""

    def score(records: list[JudgeInputRecord], judge_model: str) -> list[JudgeScore]:
        del judge_model
        return [{"faithfulness": 0.0, "answer_relevancy": 0.0} for _ in records]

    return score
