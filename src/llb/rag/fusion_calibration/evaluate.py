"""Tune deterministic routing thresholds once, then score only the frozen policy on final."""

from llb.rag.fusion import fuse_lane_hits, lane_depth
from llb.rag.fusion_calibration.models import (
    PolicyResult,
    PolicySpec,
    RouteError,
    RouteQuality,
    RoutingCalibrationReport,
)
from llb.rag.fusion_evidence.models import EvidenceItem, Retriever
from llb.rag.fusion_evidence.rows import LaneCache
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_index_sets,
    bootstrap_ratio,
    paired_comparison,
)
from llb.rag.fusion_routing import HeuristicPolicy, QuestionTypeRouter, ROUTE_GRAPH, ROUTE_VECTOR
from llb.rag.retrieval import recall_at_k, span_coverage_at_k

SELECTION_METRIC = "multi_span_coverage_delta"
DECISION_RECOMMEND = "recommend"
DECISION_NO_RECOMMENDATION = "no_recommendation"


def calibrate_routing(
    vector: Retriever,
    graph: Retriever,
    tuning_items: list[EvidenceItem],
    final_items: list[EvidenceItem],
    policies: tuple[HeuristicPolicy, ...],
    *,
    k: int,
    graph_strategy: str,
    graph_weight: float,
    candidates: int | None,
    span_identity: str,
    tuning_split: str,
    final_split: str,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> RoutingCalibrationReport:
    """Evaluate every policy on tuning, freeze one, and evaluate only it on final."""
    if not tuning_items or not final_items:
        raise ValueError("routing calibration needs non-empty tuning and final splits")
    if not policies:
        raise ValueError("routing calibration needs at least one heuristic policy")
    depth = lane_depth(candidates, k)
    tuning_questions = [item.question for item in tuning_items]
    tuning_vector = LaneCache(vector, tuning_questions, depth)
    tuning_graph = LaneCache(graph, tuning_questions, depth)
    tuning = {
        policy.label: _evaluate_policy(
            policy,
            tuning_items,
            tuning_vector,
            tuning_graph,
            k,
            depth,
            graph_weight,
            span_identity,
            resamples,
            confidence,
            seed,
        )
        for policy in policies
    }
    frozen = max(policies, key=lambda policy: _selection_key(tuning[policy.label]))
    final_questions = [item.question for item in final_items]
    final_vector = LaneCache(vector, final_questions, depth)
    final_graph = LaneCache(graph, final_questions, depth)
    final = _evaluate_policy(
        frozen,
        final_items,
        final_vector,
        final_graph,
        k,
        depth,
        graph_weight,
        span_identity,
        resamples,
        confidence,
        seed,
    )
    validated = tuning[frozen.label]["recommendation_gate"] and final["recommendation_gate"]
    recommended = frozen.label if validated else None
    decision = DECISION_RECOMMEND if recommended else DECISION_NO_RECOMMENDATION
    reason = _reason(tuning[frozen.label], final, recommended is not None)
    return {
        "k": k,
        "graph_strategy": graph_strategy,
        "graph_weight": graph_weight,
        "candidates": depth,
        "span_identity": span_identity,
        "tuning_split": tuning_split,
        "final_split": final_split,
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
        "sidecar_hidden": True,
        "selection_metric": SELECTION_METRIC,
        "tuning": tuning,
        "frozen_policy": frozen.label,
        "recommended_policy": recommended,
        "final": final,
        "decision": decision,
        "reason": reason,
    }


def _evaluate_policy(
    policy: HeuristicPolicy,
    items: list[EvidenceItem],
    vector: LaneCache,
    graph: LaneCache,
    k: int,
    depth: int,
    graph_weight: float,
    span_identity: str,
    resamples: int,
    confidence: float,
    seed: int,
) -> PolicyResult:
    router = QuestionTypeRouter(graph_weight, {}, policy)
    predicted: list[bool] = []
    actual: list[bool] = []
    vector_recall: list[float] = []
    routed_recall: list[float] = []
    vector_coverage: list[float] = []
    routed_coverage: list[float] = []
    route_errors: list[RouteError] = []
    for item in items:
        vector_hits = vector.retrieve(item.question, depth)
        graph_hits = graph.retrieve(item.question, depth)
        decision = router.decide(item.question)
        hits = fuse_lane_hits(
            vector_hits,
            graph_hits,
            decision.graph_weight,
            k,
            span_identity=span_identity,
        )
        predicts_multi = decision.route == ROUTE_GRAPH
        is_multi = len(item.spans) > 1
        predicted.append(predicts_multi)
        actual.append(is_multi)
        if predicts_multi != is_multi:
            route_errors.append(
                {
                    "item_id": item.item_id,
                    "predicted": decision.route,
                    "expected": ROUTE_GRAPH if is_multi else ROUTE_VECTOR,
                    "signals": decision.signals,
                }
            )
        vector_recall.append(recall_at_k(vector_hits, item.spans, k))
        routed_recall.append(recall_at_k(hits, item.spans, k))
        vector_coverage.append(span_coverage_at_k(vector_hits, item.spans, k))
        routed_coverage.append(span_coverage_at_k(hits, item.spans, k))
    route_sets = bootstrap_index_sets(len(items), resamples, seed)
    multi = [i for i, is_multi in enumerate(actual) if is_multi]
    single = [i for i, is_multi in enumerate(actual) if not is_multi]
    route = _route_quality(predicted, actual, route_sets, confidence)
    multi_comparison = paired_comparison(
        [routed_coverage[i] for i in multi],
        [vector_coverage[i] for i in multi],
        bootstrap_index_sets(len(multi), resamples, seed),
        confidence,
    )
    single_comparison = paired_comparison(
        [routed_recall[i] for i in single],
        [vector_recall[i] for i in single],
        bootstrap_index_sets(len(single), resamples, seed),
        confidence,
    )
    gate = multi_comparison["delta"]["lo"] > 0.0 and single_comparison["delta"]["lo"] >= 0.0
    return {
        "policy": _policy_spec(policy),
        "n": len(items),
        "multi_span_n": len(multi),
        "single_span_n": len(single),
        "graph_questions": sum(predicted),
        "vector_questions": len(predicted) - sum(predicted),
        "route": route,
        "route_errors": route_errors,
        "multi_span_coverage": multi_comparison,
        "single_span_recall": single_comparison,
        "recommendation_gate": gate,
    }


def _route_quality(
    predicted: list[bool],
    actual: list[bool],
    index_sets: list[list[int]],
    confidence: float,
) -> RouteQuality:
    true_positive = [guess and truth for guess, truth in zip(predicted, actual)]
    return {
        "true_positive": sum(true_positive),
        "false_positive": sum(guess and not truth for guess, truth in zip(predicted, actual)),
        "true_negative": sum(not guess and not truth for guess, truth in zip(predicted, actual)),
        "false_negative": sum(not guess and truth for guess, truth in zip(predicted, actual)),
        "precision": bootstrap_ratio(true_positive, predicted, index_sets, confidence),
        "recall": bootstrap_ratio(true_positive, actual, index_sets, confidence),
    }


def _policy_spec(policy: HeuristicPolicy) -> PolicySpec:
    return {
        "label": policy.label,
        "long_question_words": policy.long_question_words,
        "min_linked_entities": policy.min_linked_entities,
    }


def _selection_key(result: PolicyResult) -> tuple[float, float, float, float, float, int, int]:
    multi = result["multi_span_coverage"]["delta"]
    single = result["single_span_recall"]["delta"]
    route = result["route"]
    policy = result["policy"]
    return (
        float(result["recommendation_gate"]),
        multi["lo"],
        multi["mean"],
        single["lo"],
        route["precision"]["mean"],
        policy["long_question_words"],
        policy["min_linked_entities"],
    )


def _reason(tuning: PolicyResult, final: PolicyResult, recommended: bool) -> str:
    tuning_multi = tuning["multi_span_coverage"]["delta"]
    tuning_single = tuning["single_span_recall"]["delta"]
    final_multi = final["multi_span_coverage"]["delta"]
    final_single = final["single_span_recall"]["delta"]
    if recommended:
        status = "clears the tuning and final gates"
    elif tuning["recommendation_gate"]:
        status = "clears the tuning gate but not the final gate"
    else:
        status = "does not clear the tuning gate"
    return (
        f"{tuning['policy']['label']} {status}: tuning multi-span coverage "
        f"{tuning_multi['mean']:+.3f} [{tuning_multi['lo']:+.3f}, {tuning_multi['hi']:+.3f}], "
        f"single-span recall {tuning_single['mean']:+.3f} "
        f"[{tuning_single['lo']:+.3f}, {tuning_single['hi']:+.3f}]; final multi-span coverage "
        f"{final_multi['mean']:+.3f} [{final_multi['lo']:+.3f}, {final_multi['hi']:+.3f}], "
        f"single-span recall {final_single['mean']:+.3f} "
        f"[{final_single['lo']:+.3f}, {final_single['hi']:+.3f}]"
    )
