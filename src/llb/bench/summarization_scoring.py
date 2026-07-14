"""Focused summarization scoring implementation."""

import logging
import re
from dataclasses import dataclass
from llb.bench.common import (
    JudgeScorer,
    LLMComplete,
    mean,
    run_gated_judge,
)
from llb.core.contracts import (
    JudgeInputRecord,
    JudgeScore,
    SummarizationCaseRow,
)
from llb.eval.common import EMPTY, OK
from llb.prompts.registry import render_text
from llb.scoring import text_analysis as ta
from llb.scoring.leaderboard import bootstrap_mean_ci
from llb.scoring.judge.model import JudgeOutcome

_LOG = logging.getLogger(__name__)

_FAITHFULNESS_INTENT = render_text("bench.summarization.faithfulness_intent")

_SENTENCE_SPLIT = re.compile(r"[.!?\n]+")


@dataclass(frozen=True)
class SummarizationCase:
    id: str
    document: str
    reference: str

    @classmethod
    def from_record(cls, record: dict[str, object]) -> "SummarizationCase":
        return cls(
            id=str(record["id"]),
            document=str(record["document"]),
            reference=str(record["reference"]),
        )


@dataclass(slots=True)
class _ScoredSummarizationCases:
    summaries: list[str]
    coverages: list[float]
    rows: list[SummarizationCaseRow]
    reliability: float
    coverage_ci: tuple[float, float] | None


@dataclass(slots=True)
class _FaithfulnessResult:
    outcome: JudgeOutcome
    value: float | None
    ci: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class _JudgeConfig:
    model: str | None
    rho: float | None
    threshold: float
    scorer: JudgeScorer | None
    base_url: str | None


def split_sentences(text: str) -> list[str]:
    """Split into non-empty trimmed sentences on terminal punctuation / newlines (UA-safe)."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def reference_coverage(reference: str, candidate: str, similarity: ta.Similarity) -> float:
    """Mean over reference sentences of the best cosine to any candidate sentence (0 when either
    side is empty)."""
    ref_sents = split_sentences(reference)
    cand_sents = split_sentences(candidate)
    if not ref_sents or not cand_sents:
        return 0.0
    return mean([max(similarity(rs, cs) for cs in cand_sents) for rs in ref_sents])


def summarize_prompt(document: str) -> str:
    return render_text("bench.summarization.summarize", {"document": document})


def _faithfulness_records(
    cases: list[SummarizationCase], summaries: list[str]
) -> list[JudgeInputRecord]:
    """One (intent, summary, [source document]) record per case for the faithfulness judge."""
    return [
        {"question": _FAITHFULNESS_INTENT, "answer": summary, "contexts": [c.document]}
        for c, summary in zip(cases, summaries)
    ]


def _generate_summaries(cases: list[SummarizationCase], complete: LLMComplete) -> list[str]:
    return [complete(summarize_prompt(case.document)) for case in cases]


def _case_row(case: SummarizationCase, summary: str, coverage: float) -> SummarizationCaseRow:
    return {
        "item_id": case.id,
        "status": EMPTY if not summary.strip() else OK,
        "coverage": round(coverage, 6),
        "objective_score": round(coverage, 6),
        "answer_preview": (summary or "")[:280],
    }


def _score_summaries(
    cases: list[SummarizationCase],
    summaries: list[str],
    similarity: ta.Similarity,
) -> _ScoredSummarizationCases:
    coverages = [
        reference_coverage(case.reference, summary, similarity)
        for case, summary in zip(cases, summaries)
    ]
    rows = [
        _case_row(case, summary, coverage)
        for case, summary, coverage in zip(cases, summaries, coverages)
    ]
    reliability = sum(1 for row in rows if row["status"] == OK) / len(rows)
    return _ScoredSummarizationCases(
        summaries=summaries,
        coverages=coverages,
        rows=rows,
        reliability=reliability,
        coverage_ci=bootstrap_mean_ci(coverages),
    )


def _attach_faithfulness(
    rows: list[SummarizationCaseRow], scores: list[JudgeScore]
) -> tuple[float, tuple[float, float] | None]:
    per_case = [float(score["faithfulness"]) for score in scores]
    for row, value in zip(rows, per_case):
        row["faithfulness"] = round(value, 6)
    return round(mean(per_case), 6), bootstrap_mean_ci(per_case)


def _run_faithfulness_judge(
    cases: list[SummarizationCase],
    scored: _ScoredSummarizationCases,
    config: _JudgeConfig,
) -> _FaithfulnessResult:
    outcome = run_gated_judge(
        _faithfulness_records(cases, scored.summaries),
        judge_model=config.model,
        judge_rho=config.rho,
        threshold=config.threshold,
        scorer=config.scorer,
        base_url=config.base_url,
    )
    if outcome.trusted and outcome.scores:
        value, ci = _attach_faithfulness(scored.rows, outcome.scores)
        return _FaithfulnessResult(outcome=outcome, value=value, ci=ci)
    if config.model is not None:
        _LOG.info(
            "[summarization] judge demoted (%s); objective coverage ranks alone", outcome.reason
        )
    return _FaithfulnessResult(outcome=outcome, value=None, ci=None)
