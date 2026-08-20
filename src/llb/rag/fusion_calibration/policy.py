"""Routing-policy summaries, ordering, and decision explanations."""

from llb.rag.fusion_calibration.models import PolicyResult, PolicySpec, RouteQuality
from llb.rag.fusion_evidence.stats import bootstrap_ratio
from llb.rag.fusion.routing import HeuristicPolicy


def route_quality(
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


def policy_spec(policy: HeuristicPolicy) -> PolicySpec:
    return {
        "label": policy.label,
        "long_question_words": policy.long_question_words,
        "min_linked_entities": policy.min_linked_entities,
    }


def selection_key(
    result: PolicyResult,
) -> tuple[float, float, float, float, float, int, int]:
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


def decision_reason(tuning: PolicyResult, final: PolicyResult, recommended: bool) -> str:
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
