"""Operator-facing reading of the optional claim cross-encoder ordering."""

from llb.core.contracts.common import JsonObject


def prefilter_section(prefilter: JsonObject) -> list[str]:
    """Render cost, recall observability, and the flat/non-monotone fallback."""
    if not prefilter:
        return []
    same = prefilter["same_conflicts"]
    lines = [
        "## Claim cross-encoder prefilter",
        "",
        f"- scorer: `{prefilter['model']}` on `{prefilter['device']}`",
        f"- candidates scored: {prefilter['candidate_rows']} in "
        f"{float(prefilter['scoring_seconds']):.2f} s",
        f"- adjudicated rows: {prefilter['adjudicated_rows']}; provisional unadjudicated rows: "
        f"{prefilter['unadjudicated_rows']}",
        f"- ordering: `{prefilter['ordering']}` ({prefilter['rows_moved']} rows changed rank)",
        f"- adjudication order: `{prefilter['adjudication_order']}`",
    ]
    if prefilter["flat_scores"]:
        lines.append("- flat-score fallback: cosine order was preserved exactly")
    calibration = prefilter["calibration"]
    if calibration.get("resolved"):
        lines.append(
            "- labelled score bins monotone: " + ("yes" if calibration.get("monotone") else "no")
        )
    else:
        lines.append(
            f"- labelled score bins unresolved: {calibration.get('labelled_rows')} rows is too few"
        )
    if same["evaluated"]:
        lines += [
            f"- actionable rows found: {same['actionable_rows']}; conflict rows lost: "
            f"{same['conflict_rows_lost']}",
            f"- rows needed for that same set: cosine {same['cosine_rows_needed']}, "
            f"cross-encoder {same['reranked_rows_needed']} "
            f"(licensed saving {same['adjudication_calls_saved']} calls)",
        ]
        if same.get("fallback"):
            lines.append(
                "- smaller-cap recommendation: none; the ordering did not clear the monotone "
                "cost-saving gate, so use the full candidate list"
            )
        else:
            lines.append(f"- measured smaller-cap budget: {same['recommended_claim_budget']}")
    else:
        lines.append(f"- conflict loss: not measurable from this capped bundle; {same['reason']}")
    return lines + [
        "",
        "Cross-encoder scores are used only for ordering. They are not probabilities, rates, "
        "confidence values, or conflict verdicts.",
        "",
    ]
