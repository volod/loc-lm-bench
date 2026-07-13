"""Text-analysis manifest, metrics, judge status, and artifact persistence."""

import logging

from llb.bench.common import persist_category_run
from llb.bench.text_analysis.constants import METHOD
from llb.bench.text_analysis.model import TextAnalysisPersistInput
from llb.core.contracts import JudgeStatus, RunMetrics, RunPaths
from llb.scoring.aggregate import TIER_TEXT_ANALYSIS
from llb.scoring.judge.model import JudgeOutcome

_LOG = logging.getLogger(__name__)


def persist_text_analysis_run(request: TextAnalysisPersistInput) -> RunPaths | None:
    if request.data_dir is None:
        return None
    paths = persist_category_run(
        method=METHOD,
        data_dir=request.data_dir,
        run_name=request.run_name,
        config=_config(request),
        metrics=_metrics(request),
        case_rows=request.rows,
        judge=_judge_status(request),
        mirror=request.mirror,
    )
    _LOG.info(
        "[text-analysis] %s scored %d docs (objective=%.3f, judged-quality=%s) -> %s",
        request.model,
        request.n_docs,
        request.result.objective_score,
        f"{request.judge_result.value:.3f}" if request.judge_result.value is not None else "n/a",
        paths["manifest"],
    )
    return paths


def _metrics(request: TextAnalysisPersistInput) -> RunMetrics:
    return {
        "objective_score": request.result.objective_score,
        "reliability": request.reliability,
        "tokens_per_s": request.tokens_per_s,
    }


def _config(request: TextAnalysisPersistInput) -> dict[str, object]:
    return {
        "model": request.model,
        "backend": request.backend,
        "tier": TIER_TEXT_ANALYSIS,
        "category": "text_analysis",
        "bundle": str(request.bundle),
        "synthetic": request.synthetic,
        "n_docs": request.n_docs,
        "judge_trusted": request.judge_result.outcome.trusted,
        "judged_quality": request.judge_result.value,
        "judged_quality_ci": list(request.judge_result.ci) if request.judge_result.ci else None,
        "judge_diagnostics": request.judge_result.outcome.diagnostics,
        **request.verification_cfg,
    }


def _judge_status(request: TextAnalysisPersistInput) -> JudgeStatus | None:
    config = request.judge_config
    if config.model is None:
        return None
    outcome: JudgeOutcome = request.judge_result.outcome
    return {
        "calibration_rho": config.rho,
        "threshold": config.threshold,
        "trusted": outcome.trusted,
        "model": config.model,
        "metrics": ["judged_quality"],
        "diagnostics": outcome.diagnostics,
    }
