"""Map run-eval CLI option names onto the core configuration contract."""

from collections.abc import Mapping
from typing import Optional

_CONFIG_OPTIONS = {
    "corpus_root": "corpus_root",
    "model": "model",
    "backend": "backend",
    "goldset": "goldset_path",
    "max_model_len": "max_model_len",
    "gpu_memory_utilization": "gpu_memory_utilization",
    "gpu_layers": "n_gpu_layers",
    "judge_model": "judge_model",
    "judge_base_url": "judge_base_url",
    "scorer_policy": "scorer_policy",
    "scorer_egress_consent": "scorer_egress_consent",
    "frontier_max_usd": "frontier_max_usd",
    "frontier_max_calls": "frontier_max_calls",
    "retrieval_backend": "retrieval_backend",
    "retrieval_strategy": "retrieval_strategy",
    "retrieval_mode": "retrieval_mode",
    "acl": "acl_label",
    "fusion_weight": "fusion_weight",
    "fusion_candidates": "fusion_candidates",
    "graph_weight": "graph_weight",
    "graph_fusion_candidates": "graph_fusion_candidates",
    "graph_fusion_span_identity": "graph_fusion_span_identity",
    "graph_fusion_span_merge_ratio": "graph_fusion_span_merge_ratio",
    "graph_fusion_router": "graph_fusion_router",
    "reranker": "reranker",
    "rerank_candidates": "rerank_candidates",
    "context_order": "context_order",
    "context_strategy": "context_strategy",
    "query_glossary": "query_glossary_path",
    "query_prep_typo_guard": "query_prep_typo_guard",
    "score_semantic": "score_semantic",
    "cited_answers": "cited_answers",
    "score_groundedness": "score_groundedness",
    "insufficient_context_probes": "insufficient_context_probes",
    "telemetry": "measure_telemetry",
}
_FALSE_MEANS_UNSET = {"scorer_egress_consent", "query_prep_typo_guard"}


def parse_query_prep(steps: Optional[str]) -> Optional[list[str]]:
    if steps is None:
        return None
    return [step.strip() for step in steps.split(",") if step.strip()]


def config_overrides(options: Mapping[str, object]) -> dict[str, object]:
    """Select and rename only options owned by ``RunConfig``."""
    overrides = {
        config_name: options[option_name] for option_name, config_name in _CONFIG_OPTIONS.items()
    }
    for option_name in _FALSE_MEANS_UNSET:
        if overrides[option_name] is False:
            overrides[option_name] = None
    query_prep = options["query_prep"]
    overrides["query_prep"] = parse_query_prep(query_prep if isinstance(query_prep, str) else None)
    return overrides
