"""End-to-end text-analysis benchmark orchestration."""

from pathlib import Path

from llb.bench.common import (
    DEFAULT_THRESHOLD,
    JudgeScorer,
    LLMComplete,
    Mirror,
    ThroughputMeter,
    category_result,
    render_board,
    verified_data_config,
)
from llb.bench.text_analysis.bundle import load_corpus_docs, load_planted_by_doc, matching_doc_ids
from llb.bench.text_analysis.model import (
    JudgeConfig,
    TextAnalysisPersistInput,
    TextAnalysisRun,
)
from llb.bench.text_analysis.persist import persist_text_analysis_run
from llb.bench.text_analysis.scoring import run_judged_quality, score_doc_batch
from llb.scoring import text_analysis as ta
from llb.scoring.aggregate import TIER_TEXT_ANALYSIS


def run_text_analysis(
    bundle: Path | str,
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
    run_name: str = "text-analysis",
    limit: int | None = None,
    synthetic: bool = True,
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
    meter: ThroughputMeter | None = None,
) -> TextAnalysisRun:
    """Score a model's objective planted-label recovery and gated free-form quality."""
    verification_cfg = verified_data_config(
        data_verified=data_verified, verification_ref=verification_ref
    )
    labels_by_doc = load_planted_by_doc(bundle)
    docs = load_corpus_docs(bundle)
    doc_ids = matching_doc_ids(bundle, labels_by_doc, docs, limit)
    similarity_fn = similarity if similarity is not None else ta.embedder_similarity()
    scored = score_doc_batch(doc_ids, labels_by_doc, docs, complete, similarity_fn)
    judge_config = JudgeConfig(
        judge_model, judge_rho, judge_threshold, judge_scorer, judge_base_url
    )
    judge_result = run_judged_quality(scored, judge_config)
    reliability = scored.n_ok / len(scored.doc_ids)
    tokens_per_s = meter.tokens_per_s if meter is not None else 0.0
    result = category_result(
        model=model,
        backend=backend,
        tier=TIER_TEXT_ANALYSIS,
        case_objectives=scored.case_objectives,
        reliability=reliability,
        tokens_per_s=tokens_per_s,
    )
    board, table = render_board([result])
    paths = (
        persist_text_analysis_run(
            TextAnalysisPersistInput(
                data_dir,
                run_name,
                model,
                backend,
                bundle,
                synthetic,
                len(scored.doc_ids),
                result,
                reliability,
                scored.rows,
                judge_result,
                judge_config,
                verification_cfg,
                tokens_per_s,
                mirror,
            )
        )
        if persist
        else None
    )
    return TextAnalysisRun(
        result,
        scored.rows,
        board,
        table,
        paths,
        judge_result.value,
        judge_result.ci,
        judge_result.outcome.trusted,
        judge_result.outcome.reason,
    )
