"""category expansion summarization runner -- reference coverage via pinned-embedder cosine (not ROUGE).

Scores a candidate summary by REFERENCE COVERAGE: for each reference-summary sentence, the best
cosine to any candidate sentence (over the project's PINNED embedder -- the same basis as
retrieval + the text-analysis matcher), averaged. The cosine `similarity` is injected, so the
runner is unit-tested from a fake endpoint + a fake similarity, no embedder or GPU.

The objective coverage is the headline. An OPT-IN gated-judge FAITHFULNESS signal (does the summary
stay grounded in the source?) is recorded ALONGSIDE -- never folded into the headline -- and only
when the judge is configured AND trusted (calibration `judge_rho >= threshold`, the judge calibration gate gate; the
faithfulness signal is exactly what judge calibration gate calibrated). The judge `scorer` is injectable, so the
wiring is provable with a fake judge (no DeepEval / endpoint / GPU).
"""

from dataclasses import dataclass
from pathlib import Path

from llb.bench.common import (
    DEFAULT_THRESHOLD,
    JudgeScorer,
    LLMComplete,
    Mirror,
    category_result,
    persist_category_run,
    render_board,
    verified_data_config,
)
from llb.bench.common_backend import ThroughputMeter
from llb.core.contracts.results import BoardRow
from llb.core.contracts.judging import JudgeStatus
from llb.core.contracts.runs import RunMetrics, RunPaths
from llb.core.contracts.benchmarks import SummarizationCaseRow
from llb.scoring import text_analysis as ta
from llb.scoring.text_analysis_similarity import embedder_similarity
from llb.scoring.aggregate import TIER_SUMMARIZATION
from llb.scoring.leaderboard import ModelResult
from llb.scoring.judge.model import JudgeOutcome
from llb.bench.summarization_scoring import (
    SummarizationCase,
    _FaithfulnessResult,
    _JudgeConfig,
    _LOG,
    _ScoredSummarizationCases,
    _generate_summaries,
    _run_faithfulness_judge,
    _score_summaries,
)


# The judge "question" for faithfulness: DeepEval faithfulness compares the answer (summary)
# against the retrieval context (the source document); a fixed UA intent frames the task.

METHOD = "summarization"


@dataclass(slots=True)
class SummarizationRun:
    result: ModelResult
    rows: list[SummarizationCaseRow]
    board: list[BoardRow]
    table: str
    coverage_ci: tuple[float, float] | None
    paths: RunPaths | None
    faithfulness: float | None = None  # mean gated-judge faithfulness (None when not trusted/run)
    faithfulness_ci: tuple[float, float] | None = None
    judge_trusted: bool = False
    judge_reason: str = "no judge configured"


@dataclass(frozen=True, slots=True)
class _SummarizationPersistInput:
    data_dir: Path | str | None
    run_name: str
    model: str
    backend: str
    n_cases: int
    result: ModelResult
    scored: _ScoredSummarizationCases
    faithfulness: _FaithfulnessResult
    judge_config: _JudgeConfig
    verification_cfg: dict[str, object]
    tokens_per_s: float
    mirror: Mirror | None


def _summarization_metrics(
    result: ModelResult, reliability: float, tokens_per_s: float
) -> RunMetrics:
    return {
        "objective_score": result.objective_score,  # mean reference coverage
        "reliability": reliability,
        "tokens_per_s": tokens_per_s,
    }


def _summarization_config(request: _SummarizationPersistInput) -> dict[str, object]:
    return {
        "model": request.model,
        "backend": request.backend,
        "tier": TIER_SUMMARIZATION,
        "category": "summarization",
        "n_cases": request.n_cases,
        "reference_coverage": request.result.objective_score,
        "reference_coverage_ci": list(request.scored.coverage_ci)
        if request.scored.coverage_ci
        else None,
        "judge_trusted": request.faithfulness.outcome.trusted,
        "faithfulness": request.faithfulness.value,  # gated diagnostic, NOT the headline
        "faithfulness_ci": list(request.faithfulness.ci) if request.faithfulness.ci else None,
        "judge_diagnostics": request.faithfulness.outcome.diagnostics,
        **request.verification_cfg,
    }


def _summarization_judge_status(
    config: _JudgeConfig,
    outcome: JudgeOutcome,
) -> JudgeStatus | None:
    if config.model is None:
        return None
    return {
        "calibration_rho": config.rho,
        "threshold": config.threshold,
        "trusted": outcome.trusted,
        "model": config.model,
        "metrics": ["faithfulness"],
        "diagnostics": outcome.diagnostics,
    }


def _persist_summarization_run(request: _SummarizationPersistInput) -> RunPaths | None:
    if request.data_dir is None:
        return None
    paths = persist_category_run(
        method=METHOD,
        data_dir=request.data_dir,
        run_name=request.run_name,
        config=_summarization_config(request),
        metrics=_summarization_metrics(
            request.result, request.scored.reliability, request.tokens_per_s
        ),
        case_rows=request.scored.rows,
        judge=_summarization_judge_status(request.judge_config, request.faithfulness.outcome),
        mirror=request.mirror,
    )
    _LOG.info(
        "[summarization] %s reference-coverage=%.3f faithfulness=%s -> %s",
        request.model,
        request.result.objective_score,
        f"{request.faithfulness.value:.3f}" if request.faithfulness.value is not None else "n/a",
        paths["manifest"],
    )
    return paths


def run_summarization(
    cases: list[SummarizationCase],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    similarity: ta.Similarity | None = None,
    judge_model: str | None = None,
    judge_rho: float | None = None,
    judge_threshold: float = DEFAULT_THRESHOLD,
    judge_scorer: JudgeScorer | None = None,
    judge_base_url: str | None = None,
    data_dir: Path | str | None = None,
    run_name: str = "summarization",
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
    meter: ThroughputMeter | None = None,
) -> SummarizationRun:
    """Score one model's summaries by reference coverage under TIER_SUMMARIZATION.

    Objective reference coverage is the headline. When a judge is configured AND trusted
    (`judge_rho >= judge_threshold`), an opt-in faithfulness signal is recorded ALONGSIDE (per-case
    + mean + CI) but never folded into the headline; otherwise the judge is demoted and coverage
    ranks alone. `judge_scorer` is injectable for tests. A `meter` (populated by the endpoint
    `complete`) supplies the run's real generation tok/s.
    """
    if not cases:
        raise SystemExit("no summarization cases provided")
    verification_cfg = verified_data_config(
        data_verified=data_verified, verification_ref=verification_ref
    )
    similarity_fn = similarity if similarity is not None else embedder_similarity()
    scored = _score_summaries(cases, _generate_summaries(cases, complete), similarity_fn)
    judge_config = _JudgeConfig(
        model=judge_model,
        rho=judge_rho,
        threshold=judge_threshold,
        scorer=judge_scorer,
        base_url=judge_base_url,
    )
    faithfulness = _run_faithfulness_judge(cases, scored, judge_config)

    tokens_per_s = meter.tokens_per_s if meter is not None else 0.0
    result = category_result(
        model=model,
        backend=backend,
        tier=TIER_SUMMARIZATION,
        case_objectives=scored.coverages,
        reliability=scored.reliability,
        tokens_per_s=tokens_per_s,
    )
    board, table = render_board([result])
    paths = (
        _persist_summarization_run(
            _SummarizationPersistInput(
                data_dir=data_dir,
                run_name=run_name,
                model=model,
                backend=backend,
                n_cases=len(cases),
                result=result,
                scored=scored,
                faithfulness=faithfulness,
                judge_config=judge_config,
                verification_cfg=verification_cfg,
                tokens_per_s=tokens_per_s,
                mirror=mirror,
            )
        )
        if persist
        else None
    )
    return SummarizationRun(
        result=result,
        rows=scored.rows,
        board=board,
        table=table,
        coverage_ci=scored.coverage_ci,
        paths=paths,
        faithfulness=faithfulness.value,
        faithfulness_ci=faithfulness.ci,
        judge_trusted=faithfulness.outcome.trusted,
        judge_reason=faithfulness.outcome.reason,
    )
