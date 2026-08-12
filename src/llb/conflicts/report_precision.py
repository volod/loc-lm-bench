"""Render the claim-tier precision block of an audit report.

The block answers the question a rank cutoff cannot: of the rows this run actually handed the
operator, what share survived claim adjudication -- with a bound that respects how few distinct
chunks those rows come from, and only when the adjudicator earned the right to be quoted.
"""

from llb.conflicts.models import AuditResult
from llb.core.contracts.common import JsonObject


def _calibration_line(calibration: JsonObject | None) -> str:
    if not calibration:
        return "- adjudicator calibration: not run"
    return (
        f"- adjudicator calibration: {calibration['agreements']}/{calibration['parsed_pairs']} "
        f"frozen probe pairs agree (accuracy {calibration['accuracy']}, Wilson 95% "
        f"{calibration['accuracy_wilson_95']}, gate {calibration['min_accuracy_lcb']}), "
        f"probe `{calibration['probe_id']}`"
    )


def precision_section(result: AuditResult) -> list[str]:
    """What share of the returned list survived adjudication -- with the bound that earns it.

    The bound is two-way clustered over the left and right chunks, because rows that reuse a chunk
    are not independent evidence, and it is printed only when the adjudicator that produced the
    verdicts agreed with the frozen calibration probe.
    """
    block = result.claim_precision
    if not block:
        return []
    lines = ["## Claim-tier precision", ""]
    if not block.get("reported"):
        return lines + [
            f"Not reported: {block['reason']}.",
            "",
            _calibration_line(block.get("adjudicator_calibration")),
            "",
        ]
    point = block["returned_budget"]
    lines += [
        f"- adjudicated rows at the returned candidate budget: {point['budget']}",
        f"- rows an operator must act on: {point['actionable_rows']}",
        f"- **precision {point['precision']}**, two-way clustered 95% lower bound "
        f"**{point['two_way_clustered_lcb']}**",
        f"- Wilson 95% (pair rows, ignores clustering): {point['wilson_95']}",
        f"- distinct chunks the rows rest on: {point['left_clusters']} left, "
        f"{point['right_clusters']} right",
        f"- unparsed verdicts: {block['unparsed_rows']}"
        + (
            " (counted as not actionable, so the printed precision is a lower bound)"
            if block["unparsed_rows"]
            else ""
        ),
        _calibration_line(block.get("adjudicator_calibration")),
        "",
        "### Precision against the candidate budget",
        "",
        "Candidates come out in rank order, so the rows at each budget are a prefix of the same "
        "adjudicated list -- this is a sweep, not a re-measurement. `actionable left/right` counts "
        "the distinct chunks the ACTIONABLE rows sit on, which is what decides whether the "
        "clustered bound can clear zero: a resampled draw that misses all of them returns nothing.",
        "",
        "| budget | actionable | precision | Wilson 95% | two-way clustered LCB "
        "| left | right | actionable left | actionable right |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in block["precision_curve"]:
        lines.append(
            f"| {row['budget']} | {row['actionable_rows']} | {row['precision']} "
            f"| {row['wilson_95']} | {row['two_way_clustered_lcb']} "
            f"| {row['left_clusters']} | {row['right_clusters']} "
            f"| {row['actionable_left_clusters']} | {row['actionable_right_clusters']} |"
        )
    resolution = block["budget_resolution"]
    lines += [
        "",
        "- budget that first buys a non-zero floor: "
        + (
            f"**{resolution['resolving_budget']}** (bound {resolution['resolving_lcb']})"
            if resolution["resolving_budget"] is not None
            else "**none of the measured budgets**"
        ),
        f"- {resolution['reading']}.",
        "",
        "This is the share of the RETURNED list that survived claim adjudication, not a "
        "false-positive rate over the corpus's pair space -- the semantic tier's cutoff is still "
        "a rank (see the data-prep known limitation).",
        "",
    ]
    return lines
