"""Frontier-judge authorization evidence: rank agreement plus measured cost per provider."""

from llb.scoring.frontier_agreement.agreement import (
    AGREEMENT_METRICS,
    CAP_SAFETY_FACTOR,
    MEAN_METRIC,
    ProviderAgreement,
    build_agreement,
    correlate,
    cost_summary,
    metric_value,
)
from llb.scoring.frontier_agreement.items import (
    AgreementItem,
    load_agreement_items,
    resolve_corpus_root,
)
from llb.scoring.frontier_agreement.provider import (
    ProviderResult,
    provider_name,
    provider_slug,
    score_with_provider,
)
from llb.scoring.frontier_agreement.report import render_report
from llb.scoring.frontier_agreement.run import (
    AGREEMENT_FILENAME,
    ARTIFACT_ROOT,
    REPORT_FILENAME,
    SCORES_FILENAME,
    ProviderFailure,
    default_out_dir,
    run_frontier_agreement,
)

__all__ = [
    "AGREEMENT_FILENAME",
    "AGREEMENT_METRICS",
    "ARTIFACT_ROOT",
    "AgreementItem",
    "CAP_SAFETY_FACTOR",
    "MEAN_METRIC",
    "ProviderAgreement",
    "ProviderFailure",
    "ProviderResult",
    "REPORT_FILENAME",
    "SCORES_FILENAME",
    "build_agreement",
    "correlate",
    "cost_summary",
    "default_out_dir",
    "load_agreement_items",
    "metric_value",
    "provider_name",
    "provider_slug",
    "render_report",
    "resolve_corpus_root",
    "run_frontier_agreement",
    "score_with_provider",
]
