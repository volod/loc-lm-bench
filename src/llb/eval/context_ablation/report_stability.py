"""Decoding-stability section of the context-ablation report.

Placed directly after the boundary table, because the two answer adjacent questions about the same
rows: that one asks whether a reading would flip under a neighbouring confidence CONVENTION, this
one whether it would flip under a re-RUN of the identical configuration.
"""

from llb.eval.context_ablation.models import (
    ContextAblationReport,
    DecodingFloorMargin,
    DecodingStabilityReport,
)
from llb.rag.fusion_evidence.spread import format_band

# A decode band is a fourth-decimal quantity -- a lane that moves one answer in eighty rounds to
# `+/-0.000` at the three decimals the metric tables use, which reads as "did not move".
_BAND_DIGITS = 4

_LANE_HEADER = (
    "| lane | grounded | objective band | token F1 band | match-rate band | items moved "
    "| answers moved | repeat groups |"
)
_DELTA_HEADER = "| delta | n | value | decoding floor | clears floor | floor multiple |"


def _lane_table(stability: DecodingStabilityReport) -> list[str]:
    lines = [_LANE_HEADER, "| --- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |"]
    for label in sorted(stability["lanes"]):
        lane = stability["lanes"][label]
        lines.append(
            f"| `{label}` | {'yes' if lane['grounded'] else 'no'} "
            f"| {format_band(lane['objective'], _BAND_DIGITS)} "
            f"| {format_band(lane['token_f1'], _BAND_DIGITS)} "
            f"| {format_band(lane['match_rate'], _BAND_DIGITS)} "
            f"| {lane['divergent_items']}/{stability['n']} "
            f"| {lane['answer_divergent_items']}/{stability['n']} "
            f"| {'+'.join(str(size) for size in lane['outcome_groups'])} |"
        )
    lines.append("")
    return lines


def _clears(margin: DecodingFloorMargin) -> str:
    """A delta of zero against a floor of zero has nothing to clear, and did not fail to."""
    if margin["clears_floor"]:
        return "yes"
    return "n/a" if margin["floor"] == 0.0 else "NO"


def _delta_table(stability: DecodingStabilityReport) -> list[str]:
    if not stability["deltas"]:
        return []
    lines = [_DELTA_HEADER, "| --- | ---: | ---: | ---: | :-: | ---: |"]
    for margin in stability["deltas"]:
        multiple = margin["floor_multiple"]
        lines.append(
            f"| `{margin['label']}` | {margin['n']} | {margin['delta']:+.4f} "
            f"| +/-{margin['floor']:.4f} | {_clears(margin)} "
            f"| {f'{multiple:.1f}x' if multiple is not None else '-'} |"
        )
    lines.append("")
    return lines


def _settling_note(stability: DecodingStabilityReport) -> list[str]:
    """Name the lanes whose quoted repeat is the one outcome that never came back.

    The tables above print the FIRST repeat. When that repeat's outcome is a group of one and the
    rest agreed with each other, the artifact is quoting the transient rather than the steady
    state -- which is a different instruction to the reader than a lane that simply wobbles.
    """
    odd = [
        label
        for label, lane in sorted(stability["lanes"].items())
        if len(lane["outcome_groups"]) > 1 and lane["outcome_groups"][0] == 1
    ]
    if not odd:
        return []
    lanes = ", ".join(f"`{label}`" for label in odd)
    return [
        f"**The quoted repeat is the odd one out** in {lanes}: its first pass produced an outcome "
        "no later pass repeated, while the later passes agreed with each other. The numbers above "
        "are that first pass, so read the band's other end as the settled value and treat a "
        "difference this size against an earlier artifact as a warm-up transient before treating "
        "it as a model or corpus change.",
        "",
    ]


def stability_section(report: ContextAblationReport) -> list[str]:
    """What a re-run of the identical configuration does to every number above."""
    stability = report.get("decoding_stability")
    if stability is None:
        return []
    lines = [
        "### How far a re-run moves each number",
        "",
        f"Every lane scored {stability['repeats']} times on the identical configuration and the "
        f"identical {stability['n']} items; the tables above quote the FIRST repeat, and the bands "
        "below say how far the others moved it. This is a different uncertainty from the bootstrap "
        "intervals, which resample the item set and cannot see the decode "
        "(`src/llb/eval/context_ablation/decoding_stability.py`).",
        "",
        *_lane_table(stability),
        *_delta_table(stability),
        "`repeat groups` is how the repeats partitioned into identical per-item outcomes -- one "
        "group means the lane reproduced throughout, a leading `1` means it answered differently "
        "once and then settled, and all-ones means no two passes were alike. Three very different "
        "findings can print the same band.",
        "",
        f"**Reading:** {stability['reading']} -- {stability['reason']}.",
        "",
        *_settling_note(stability),
    ]
    inside = [margin["label"] for margin in stability["deltas"] if not margin["clears_floor"]]
    if inside:
        labels = ", ".join(f"`{label}`" for label in inside)
        lines.extend(
            [
                f"{labels} does not clear its decoding floor, so it was observed rather than "
                "measured: another pass of the same configuration could report it with the "
                "opposite sign.",
                "",
            ]
        )
    return lines


def stability_note(report: ContextAblationReport) -> str:
    """The `+/-` to append to the header's closed-book match rate, when it was measured."""
    stability = report.get("decoding_stability")
    if stability is None:
        return ""
    lane = stability["lanes"].get(stability["baseline"])
    if lane is None:
        return ""
    return (
        f" +/-{lane['match_rate']['half_width']:.1%} over {stability['repeats']} identical repeats"
    )


__all__ = ["stability_note", "stability_section"]
