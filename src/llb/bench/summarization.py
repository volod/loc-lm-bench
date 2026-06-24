"""M5.4 summarization runner -- reference coverage via pinned-embedder cosine (not ROUGE).

Scores a candidate summary by REFERENCE COVERAGE: for each reference-summary sentence, the best
cosine to any candidate sentence (over the project's PINNED embedder -- the same basis as
retrieval + the text-analysis matcher), averaged. The cosine `similarity` is injected, so the
runner is unit-tested from a fake endpoint + a fake similarity, no embedder or GPU.

The objective coverage is the headline. An OPT-IN gated-judge FAITHFULNESS signal (does the summary
stay grounded in the source?) is recorded ALONGSIDE -- never folded into the headline -- and only
when the judge is configured AND trusted (calibration `judge_rho >= threshold`, the M3.8 gate; the
faithfulness signal is exactly what M3.8 calibrated). The judge `scorer` is injectable, so the
wiring is provable with a fake judge (no DeepEval / endpoint / GPU).
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from llb.bench.common import (
    DEFAULT_THRESHOLD,
    JudgeScorer,
    LLMComplete,
    Mirror,
    category_result,
    mean,
    persist_category_run,
    render_board,
    run_gated_judge,
)
from llb.contracts import (
    BoardRow,
    JudgeInputRecord,
    JudgeStatus,
    RunMetrics,
    RunPaths,
    SummarizationCaseRow,
)
from llb.eval.common import EMPTY, OK
from llb.scoring import text_analysis as ta
from llb.scoring.aggregate import TIER_SUMMARIZATION, ModelResult, bootstrap_mean_ci

_LOG = logging.getLogger(__name__)

# The judge "question" for faithfulness: DeepEval faithfulness compares the answer (summary)
# against the retrieval context (the source document); a fixed UA intent frames the task.
_FAITHFULNESS_INTENT = "Підсумуй документ, не додаючи фактів, яких у ньому немає."

METHOD = "summarization"
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
    return (
        "Стисло підсумуй наведений документ українською мовою (2-4 речення), "
        "зберігаючи ключові факти.\n\n"
        f"Документ:\n{document}\n\nПідсумок:"
    )


def _faithfulness_records(
    cases: list[SummarizationCase], summaries: list[str]
) -> list[JudgeInputRecord]:
    """One (intent, summary, [source document]) record per case for the faithfulness judge."""
    return [
        {"question": _FAITHFULNESS_INTENT, "answer": summary, "contexts": [c.document]}
        for c, summary in zip(cases, summaries)
    ]


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
    run_name: str = "m5-summarization",
    persist: bool = True,
    mirror: Mirror | None = None,
) -> SummarizationRun:
    """Score one model's summaries by reference coverage under TIER_SUMMARIZATION.

    Objective reference coverage is the headline. When a judge is configured AND trusted
    (`judge_rho >= judge_threshold`), an opt-in faithfulness signal is recorded ALONGSIDE (per-case
    + mean + CI) but never folded into the headline; otherwise the judge is demoted and coverage
    ranks alone. `judge_scorer` is injectable for tests.
    """
    if not cases:
        raise SystemExit("no summarization cases provided")
    if similarity is None:
        similarity = ta.embedder_similarity()
    summaries = [complete(summarize_prompt(c.document)) for c in cases]
    coverages = [reference_coverage(c.reference, s, similarity) for c, s in zip(cases, summaries)]

    rows: list[SummarizationCaseRow] = [
        {
            "item_id": c.id,
            "status": EMPTY if not s.strip() else OK,
            "coverage": round(cov, 6),
            "answer_preview": (s or "")[:280],
        }
        for c, s, cov in zip(cases, summaries, coverages)
    ]

    # Opt-in, gated faithfulness signal (objective coverage stays the headline).
    outcome = run_gated_judge(
        _faithfulness_records(cases, summaries),
        judge_model=judge_model,
        judge_rho=judge_rho,
        threshold=judge_threshold,
        scorer=judge_scorer,
        base_url=judge_base_url,
    )
    faithfulness: float | None = None
    faithfulness_ci: tuple[float, float] | None = None
    if outcome.trusted and outcome.scores:
        per_case = [float(s["faithfulness"]) for s in outcome.scores]
        for row, value in zip(rows, per_case):
            row["faithfulness"] = round(value, 6)
        faithfulness = round(mean(per_case), 6)
        faithfulness_ci = bootstrap_mean_ci(per_case)
    elif judge_model is not None:
        _LOG.info(
            "[summarization] judge demoted (%s); objective coverage ranks alone", outcome.reason
        )

    reliability = sum(1 for r in rows if r["status"] == OK) / len(rows) if rows else 0.0
    result = category_result(
        model=model,
        backend=backend,
        tier=TIER_SUMMARIZATION,
        case_objectives=coverages,
        reliability=reliability,
    )
    coverage_ci = bootstrap_mean_ci(coverages)
    board, table = render_board([result])

    paths: RunPaths | None = None
    if persist and data_dir is not None:
        metrics: RunMetrics = {
            "objective_score": result.objective_score,  # mean reference coverage
            "reliability": reliability,
            "tokens_per_s": 0.0,
        }
        config = {
            "model": model,
            "backend": backend,
            "tier": TIER_SUMMARIZATION,
            "category": "summarization",
            "n_cases": len(cases),
            "reference_coverage": result.objective_score,
            "reference_coverage_ci": list(coverage_ci) if coverage_ci else None,
            "judge_trusted": outcome.trusted,
            "faithfulness": faithfulness,  # gated diagnostic, NOT the headline
            "faithfulness_ci": list(faithfulness_ci) if faithfulness_ci else None,
        }
        judge_status: JudgeStatus | None = None
        if judge_model is not None:
            judge_status = {
                "calibration_rho": judge_rho,
                "threshold": judge_threshold,
                "trusted": outcome.trusted,
                "model": judge_model,
                "metrics": ["faithfulness"],
            }
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=run_name,
            config=config,
            metrics=metrics,
            case_rows=rows,
            judge=judge_status,
            mirror=mirror,
        )
        _LOG.info(
            "[summarization] %s reference-coverage=%.3f faithfulness=%s -> %s",
            model,
            result.objective_score,
            f"{faithfulness:.3f}" if faithfulness is not None else "n/a",
            paths["manifest"],
        )
    return SummarizationRun(
        result=result,
        rows=rows,
        board=board,
        table=table,
        coverage_ci=coverage_ci,
        paths=paths,
        faithfulness=faithfulness,
        faithfulness_ci=faithfulness_ci,
        judge_trusted=outcome.trusted,
        judge_reason=outcome.reason,
    )


def load_cases_file(path: Path | str) -> list[SummarizationCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of summarization cases")
    return [SummarizationCase.from_record(r) for r in raw]
