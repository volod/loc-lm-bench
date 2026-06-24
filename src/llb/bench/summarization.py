"""M5.4 summarization runner -- reference coverage via pinned-embedder cosine (not ROUGE).

Scores a candidate summary by REFERENCE COVERAGE: for each reference-summary sentence, the best
cosine to any candidate sentence (over the project's PINNED embedder -- the same basis as
retrieval + the text-analysis matcher), averaged. The gated-judge faithfulness signal is opt-in
(documented residual). The cosine `similarity` is injected, so the runner is unit-tested from a
fake endpoint + a fake similarity, no embedder or GPU.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from llb.bench.common import (
    LLMComplete,
    Mirror,
    category_result,
    mean,
    persist_category_run,
    render_board,
)
from llb.contracts import BoardRow, RunMetrics, RunPaths, SummarizationCaseRow
from llb.eval.common import EMPTY, OK
from llb.scoring import text_analysis as ta
from llb.scoring.aggregate import TIER_SUMMARIZATION, ModelResult, bootstrap_mean_ci

_LOG = logging.getLogger(__name__)

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


def run_summarization(
    cases: list[SummarizationCase],
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    similarity: ta.Similarity | None = None,
    data_dir: Path | str | None = None,
    run_name: str = "m5-summarization",
    persist: bool = True,
    mirror: Mirror | None = None,
) -> SummarizationRun:
    """Score one model's summaries by reference coverage under TIER_SUMMARIZATION."""
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
        }
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=run_name,
            config=config,
            metrics=metrics,
            case_rows=rows,
            mirror=mirror,
        )
        _LOG.info(
            "[summarization] %s reference-coverage=%.3f -> %s",
            model,
            result.objective_score,
            paths["manifest"],
        )
    return SummarizationRun(
        result=result, rows=rows, board=board, table=table, coverage_ci=coverage_ci, paths=paths
    )


def load_cases_file(path: Path | str) -> list[SummarizationCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of summarization cases")
    return [SummarizationCase.from_record(r) for r in raw]
