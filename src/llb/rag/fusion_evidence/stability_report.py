"""Markdown rendering for the shared paired-reading stability shape."""

from collections.abc import Sequence

from llb.rag.fusion_evidence.evidence_gate import (
    READING_INSUFFICIENT_EVIDENCE,
    level_label,
    minimum_discordant_pairs,
    reading_label,
    resolving_item_count,
)
from llb.rag.fusion_evidence.stability import (
    LOOSER_CONFIDENCE,
    TIGHTER_CONFIDENCE,
    ReadingStability,
    decision_probability,
)
from llb.rag.fusion_evidence.randomization import randomization_alpha


def _resolving_cell(stability: ReadingStability, confidence: float) -> str:
    discordant = stability.get("discordant")
    pairs = stability.get("pairs")
    if discordant is None or pairs is None:
        return "-"
    required = resolving_item_count(discordant, pairs, confidence)
    return "-" if required is None else str(required)


def boundary_table(
    rows: Sequence[tuple[str, ReadingStability]],
    *,
    title: str,
    key_header: str,
    subject: str,
    confidence: float,
    evidence_counts: bool,
    positive_event: str | None,
) -> list[str]:
    """Where each row sits on the continuous scale its binary reading cut."""
    if not rows:
        return []
    calibrated = any("randomization_p" in entry for _, entry in rows)
    explanation = (
        f"`randomization p` is the calibrated one-sided sign-flip probability for "
        f"{positive_event or f'{subject} being ahead'}; the reading separates when it is at most "
        f"{randomization_alpha(confidence):.4f}. `p_positive` remains the share of bootstrap "
        "resamples above zero."
        if calibrated
        else f"`p_positive` is the share of paired resamples in which "
        f"{positive_event or f'{subject} is ahead'}; the reading clears zero exactly when it "
        f"exceeds {decision_probability(confidence):.3f}."
    )
    lines = [
        f"### {title}",
        "",
        f"{explanation} A row is unsettled when either neighbouring convention would read it "
        "differently.",
    ]
    if evidence_counts:
        lines[-1] += _evidence_explanation(confidence)
    lines.extend(_header(key_header, confidence, evidence_counts, calibrated))
    for key, entry in rows:
        lines.append(_row(key, entry, confidence, evidence_counts, calibrated))
    lines.append("")
    return lines


def _evidence_explanation(confidence: float) -> str:
    return (
        " `d` is the number of items the two lanes differ on: below "
        f"{minimum_discordant_pairs(confidence)} of them the level is unreachable whatever the "
        f"interval says, and the reading is `{reading_label(READING_INSUFFICIENT_EVIDENCE)}` "
        f"(the neighbouring levels need {minimum_discordant_pairs(LOOSER_CONFIDENCE)} and "
        f"{minimum_discordant_pairs(TIGHTER_CONFIDENCE)}). `n to reach` prices ANY row short of "
        "that bound, whatever its interval says: the items its own discordance rate would need "
        "before the level becomes reachable at all -- a floor on the item set, not an effect size "
        "it could then resolve. A `flat` row carrying one is a reading that could not have shown a "
        "difference, which is not the same as one that looked and found none."
    )


def _header(
    key_header: str, confidence: float, evidence_counts: bool, calibrated: bool
) -> list[str]:
    common = (
        f"| {key_header} | at {level_label(LOOSER_CONFIDENCE)} | reading "
        f"({level_label(confidence)}) | at {level_label(TIGHTER_CONFIDENCE)} "
        + ("| randomization p | p_positive " if calibrated else "| p_positive ")
    )
    return [
        "",
        (common + "| d | n to reach | settled? |" if evidence_counts else common + "| settled? |"),
        (
            (
                "| --- | :-: | :-: | :-: | ---: | ---: | ---: | ---: | :-: |"
                if calibrated
                else "| --- | :-: | :-: | :-: | ---: | ---: | ---: | :-: |"
            )
            if evidence_counts
            else (
                "| --- | :-: | :-: | :-: | ---: | ---: | :-: |"
                if calibrated
                else "| --- | :-: | :-: | :-: | ---: | :-: |"
            )
        ),
    ]


def _row(
    key: str,
    entry: ReadingStability,
    confidence: float,
    evidence_counts: bool,
    calibrated: bool,
) -> str:
    settled = f"NO ({entry['side']})" if entry["borderline"] else "yes"
    core = (
        f"| {key} | {reading_label(entry['looser_reading'])} "
        f"| {reading_label(entry['reading'])} | {reading_label(entry['tighter_reading'])} "
        + (
            f"| {entry['randomization_p']:.4f} | {entry['p_positive']:.3f} "
            if calibrated
            else f"| {entry['p_positive']:.3f} "
        )
    )
    if not evidence_counts:
        return core + f"| {settled} |"
    discordant = entry.get("discordant")
    return (
        core
        + f"| {'-' if discordant is None else discordant} "
        + f"| {_resolving_cell(entry, confidence)} | {settled} |"
    )
